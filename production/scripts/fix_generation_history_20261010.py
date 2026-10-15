#!/usr/bin/env python3
"""修复 2026-10-10 自托管 H3 批量重跑写入的畸形 generation_history 条目。

rerun_h3_selfhosted.py 写的条目缺 segments 列表且 video 是裸字符串，
prompt_review_store.generation_history 对畸形条目 fail-closed 返回空历史，
导致网页历史版本按钮不显示。本脚本按正式结构重建：

- 13 个真正重跑的项目：归档每段旧版 prompt 到
  output/history/v1/prompts/segment-NNNN.txt（旧 prompt 优先从归档的
  picsart submit 日志 input.json 恢复；恢复不到（IR 重跑覆盖过的段也
  优先日志）退回当前 h3_prompt.txt），条目写 {version, completed_at,
  video: {path}, segments: [...]}。
- 未重跑只还原的项目（restored，object_replace v4/v5）：删除畸形条目，
  保留 history/v1 目录（里面是转移留存的 picsart 提交日志证据）。

用法：python3 scripts/fix_generation_history_20261010.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/home/xy/duet-ad1-h3-max-r2v")
PROJECTS = ROOT / "data" / "projects"

RERUN = [
    "081e3436", "65404818", "585f3207", "13a8aa59", "176eb166",
    "c0b5fb69", "d844fa49", "f926f9a9", "83775319", "9cadc633",
    "a564dbb9", "d7b26903", "d9ed6b38",
]
RESTORED = [
    "09cf0e28", "0fe5f51b", "1cc3adb4", "28d71a93", "4f858699",
    "5ec53ec7", "9e282d49", "9f7f90de", "a7157d8a", "afef5ad1",
    "b4b3584a", "b53a8f55", "cda91775", "e46bf03d", "f0997e14",
]


def find_full(pid8: str) -> Path:
    matches = sorted(PROJECTS.glob(pid8 + "*"))
    if len(matches) != 1:
        raise SystemExit(f"项目 {pid8} 匹配异常: {matches}")
    return matches[0]


def recover_old_prompt(proj: Path, seg_index: int) -> str | None:
    """从归档的 submit 日志恢复旧版 prompt（取该段最后一个 attempt）。"""
    logs = proj / "output" / "history" / "v1" / "logs"
    best: tuple[int, str] | None = None
    for input_json in logs.glob(f"*submit*/segment-{seg_index:04d}/attempt-*/input.json"):
        attempt = int(input_json.parent.name.split("-")[-1])
        try:
            data = json.loads(input_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            if best is None or attempt > best[0]:
                best = (attempt, prompt)
    return best[1] if best else None


def fix_rerun(pid8: str) -> str:
    proj = find_full(pid8)
    state_path = proj / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("generation_history") or []
    entry = next(
        (e for e in history if e.get("reason") == "h3-selfhosted-switch-20261010"),
        None,
    )
    if entry is None:
        return "无切换条目，跳过"
    version = entry["version"]
    archive = proj / "output" / "history" / f"v{version}"
    prompts_dir = archive / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    recovered = fallback = 0
    for seg in state.get("segments") or []:
        idx = seg["index"]
        text = recover_old_prompt(proj, idx)
        if text is None:
            current = proj / "work" / "segments" / f"{idx:04d}" / "h3_prompt.txt"
            text = current.read_text(encoding="utf-8")
            fallback += 1
        else:
            recovered += 1
        prompt_rel = f"output/history/v{version}/prompts/segment-{idx:04d}.txt"
        (proj / prompt_rel).write_text(text, encoding="utf-8")
        segments.append({
            "segment_index": idx,
            "prompt_path": prompt_rel,
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "start_s": seg.get("start_s"),
            "end_s": seg.get("end_s"),
        })
    entry["completed_at"] = entry.get("archived_at")
    entry["video"] = {"path": f"output/history/v{version}/generated.mp4"}
    entry["segments"] = segments
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return f"v{version} 段数={len(segments)} 日志恢复={recovered} 当前文件兜底={fallback}"


def clean_restored(pid8: str) -> str:
    proj = find_full(pid8)
    state_path = proj / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    history = state.get("generation_history") or []
    kept = [
        e for e in history
        if not (isinstance(e, dict) and e.get("reason") == "h3-selfhosted-switch-20261010")
    ]
    if len(kept) == len(history):
        return "无切换条目，跳过"
    if kept:
        state["generation_history"] = kept
    else:
        state.pop("generation_history", None)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return "已删畸形条目（history/v1 目录留存提交日志证据）"


def main() -> None:
    for pid8 in RERUN:
        print(pid8, fix_rerun(pid8))
    for pid8 in RESTORED:
        print(pid8, clean_restored(pid8))


if __name__ == "__main__":
    main()
