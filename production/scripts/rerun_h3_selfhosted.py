#!/usr/bin/env python3
"""自托管 H3 批量重跑驱动（2026-10-10 provider 切换配套）。

对每个候选项目（succeeded、有冻结 H3 产物、各段冻结 prompt ≤12000 字符）：
1. 归档当前成片到 output/history/v{N}/ 并登记 generation_history；
2. 清 picsart 提交账本与分段 H3 输出（保留冻结 prompt/素材）；
3. status 置 failed 后调 resume-picsart 重跑 H3 及后续装配。

用法（绝对路径、生产 API 走 3211）：
  python3 scripts/rerun_h3_selfhosted.py --project 081e3436        # 单个（金丝雀）
  python3 scripts/rerun_h3_selfhosted.py --all --concurrency 5     # 批量
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
PROJECTS = ROOT / "data" / "projects"
API = "https://127.0.0.1:3211"
PROMPT_MAX_CHARS = 12000


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['ACCESS_TOKEN']}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return json.loads(r.read())


def candidates() -> list[tuple[str, str]]:
    """返回 [(project_id, 跳过原因或'')]。收可重跑的 succeeded 项目与
    本脚本已做过手术（failed + h3_selfhosted_rerun）待点火的项目。"""
    out = []
    for state_path in sorted(PROJECTS.glob("*/state.json")):
        proj = state_path.parent
        pid = proj.name
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        prepared = (state.get("error") or {}).get("code") == "h3_selfhosted_rerun"
        if state.get("status") != "succeeded" and not prepared:
            continue
        if not isinstance(state.get("h3_prepared_artifacts"), dict):
            out.append((pid, "无 h3_prepared_artifacts"))
            continue
        over = [
            p for p in proj.glob("work/segments/*/h3_prompt.txt")
            if len(p.read_text(encoding="utf-8")) > PROMPT_MAX_CHARS
        ]
        if over:
            out.append((pid, f"{len(over)} 段 prompt 超 {PROMPT_MAX_CHARS} 字符"))
            continue
        if not (proj / "output" / "generated.mp4").is_file():
            out.append((pid, "缺 output/generated.mp4"))
            continue
        out.append((pid, ""))
    return out


def prepare_project(pid: str) -> None:
    """归档旧成片 + 清账本与分段 H3 输出 + 移出 picsart 提交日志 + 置 failed。"""
    proj = PROJECTS / pid
    state_path = proj / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("generation_history") or []
    version = max((e.get("version", 0) for e in history), default=0) + 1
    archive = proj / "output" / "history" / f"v{version}"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(proj / "output" / "generated.mp4", archive / "generated.mp4")
    history.append({
        "version": version,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "reason": "h3-selfhosted-switch-20261010",
        "video": f"output/history/v{version}/generated.mp4",
    })
    state["generation_history"] = history
    state.pop("picsart_submission_receipts", None)
    for seg_dir in proj.glob("work/segments/*"):
        for stale in list(seg_dir.glob("generated*.mp4")) + list(seg_dir.glob("h3-provider*.mp4")):
            stale.unlink()
    for seg in state.get("segments") or []:
        if isinstance(seg, dict):
            seg.pop("task_id", None)
    state["status"] = "failed"
    state["error"] = {
        "code": "h3_selfhosted_rerun",
        "message": "切换自托管 H3 provider 重跑（旧成片已归档 history）",
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    move_picsart_logs(proj, archive)


def move_picsart_logs(proj: Path, archive: Path) -> None:
    """resume-picsart 门禁：无账本时存在 logs/picsart*-submit 即拒
    （'有提交证据却没有回执账本'）。移入归档目录保留证据。"""
    log_archive = archive / "logs"
    for path in sorted((proj / "logs").glob("picsart*")):
        log_archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(log_archive / path.name))


def run_project(pid: str) -> tuple[str, str]:
    """返回 (pid, ok|fail原因)。调用方负责并发控制。"""
    try:
        state = json.loads((PROJECTS / pid / "state.json").read_text(encoding="utf-8"))
        # 已手术（failed + h3_selfhosted_rerun，或 pipeline 中途失败但历史里
        # 有本次切换归档记录）的项目跳过归档，只补移 picsart 提交日志。
        already = (state.get("error") or {}).get("code") == "h3_selfhosted_rerun" or any(
            e.get("reason") == "h3-selfhosted-switch-20261010"
            for e in (state.get("generation_history") or [])
            if isinstance(e, dict)
        )
        if already:
            proj = PROJECTS / pid
            history = state.get("generation_history") or []
            latest = max((e.get("version", 0) for e in history), default=0)
            if latest:
                move_picsart_logs(proj, proj / "output" / "history" / f"v{latest}")
        else:
            prepare_project(pid)
        api("POST", f"/api/conversations/{pid}/resume-picsart", {"confirm": True})
    except Exception as exc:
        return pid, f"prepare/resume 失败: {exc!r}"
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        time.sleep(20)
        try:
            detail = api("GET", f"/api/conversations/{pid}")
        except Exception:
            continue
        status = detail.get("status")
        if status == "succeeded":
            return pid, "ok"
        if status in ("failed", "canceled"):
            err = detail.get("error") or {}
            return pid, f"重跑失败: {err.get('code')} {str(err.get('message'))[:200]}"
    return pid, "超时(3600s)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    cands = dict(candidates())
    if args.all:
        todo = [pid for pid, reason in cands.items() if not reason]
        skipped = {pid: r for pid, r in cands.items() if r}
        if skipped:
            print("跳过:", file=sys.stderr)
            for pid, r in sorted(skipped.items()):
                print(f"  {pid[:8]} {r}", file=sys.stderr)
    else:
        todo = []
        for short in args.project:
            pid = next((p for p in cands if p.startswith(short)), None)
            if pid is not None and cands[pid]:
                print(f"候选检查不通过 {short}: {cands[pid]}", file=sys.stderr)
                return 2
            if pid is None:
                # 兜底：手术痕迹（history 里有切换归档）但状态已被后续失败
                # 覆盖的项目，candidates() 收不到，按目录名直接匹配放行。
                direct = next(
                    (p.name for p in sorted(PROJECTS.glob(short + "*")) if p.is_dir()),
                    None,
                )
                if direct is None:
                    print(f"找不到候选项目 {short}", file=sys.stderr)
                    return 2
                pid = direct
            todo.append(pid)
    print(f"待重跑 {len(todo)} 个项目，并发 {args.concurrency}")
    t0 = time.monotonic()
    results = {}
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_project, pid): pid for pid in todo}
        for fut in futures.as_completed(futs):
            pid, outcome = fut.result()
            results[pid] = outcome
            print(f"[{time.strftime('%H:%M:%S')}] {pid[:8]} -> {outcome}", flush=True)
    ok = sum(1 for v in results.values() if v == "ok")
    print(f"完成 {ok}/{len(results)}，耗时 {round(time.monotonic()-t0)}s")
    report = ROOT / "data-verify" / "h3-selfhosted-rerun-20261010" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
