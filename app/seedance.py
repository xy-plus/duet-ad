"""预留的 Seedance 真实提交：开关 + confirm + dry-run 复核三重门控，默认 501。

密钥只存在于服务进程环境（ARK_API_KEY），子进程直接继承；不进日志、不进响应、
不进 meta.json。报错信息一律先过 _sanitize 脱敏。
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from app import storage
from app.config import Settings

log = logging.getLogger(__name__)

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills" / "seedance-cleaning-video-maker" / "scripts" / "seedance_task.py"
)
_SUBMIT_TIMEOUT_S = 1800
_DRYRUN_TIMEOUT_S = 120
_DETAIL_LIMIT = 300
_LEAK_RE = re.compile(r"key|authorization", re.IGNORECASE)
_COMPARE_FIELDS = ("model", "ratio", "duration", "resolution", "generate_audio", "watermark")


class SubmitError(Exception):
    """提交门控/执行失败（HTTP 层转成 status+detail，同 UploadError 模式）。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


async def submit(
    settings: Settings, cid: str, payload: dict, locks: dict[str, asyncio.Lock]
) -> dict:
    """按固定顺序过门控；每会话一把锁，锁内重查 has_video 防并发重复扣费。"""
    if not settings.enable_seedance_submit:
        raise SubmitError(501, "Seedance submission is disabled.")
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        raise SubmitError(404, "not found")
    if payload.get("confirm") is not True:
        raise SubmitError(409, "confirmation required")
    if meta.get("status") != "done":
        raise SubmitError(409, "artifacts not ready")
    if meta.get("has_video"):
        raise SubmitError(409, "already submitted")
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        meta = storage.load_meta(settings.data_dir, cid)
        if meta is None or meta.get("has_video"):
            raise SubmitError(409, "already submitted")
        cdir = settings.data_dir / cid
        reviewed = _load_reviewed(cdir)
        keyframes = _keyframes(cdir)
        await asyncio.to_thread(_recheck_payload, cdir, reviewed, keyframes)
        if not os.environ.get("ARK_API_KEY", "").strip():
            raise SubmitError(503, "ARK_API_KEY not configured")
        await asyncio.to_thread(_run_submit, cdir, reviewed, keyframes)
        storage.mark_submitted(settings.data_dir, cid, _read_task_id(cdir))
        return {"status": "succeeded", "video": "generated.mp4"}


def _load_reviewed(cdir: Path) -> dict:
    """评审时落盘的 work/api_request.json；缺失/损坏/缺字段等同 payload 已变。"""
    try:
        data = json.loads((cdir / "work" / "api_request.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SubmitError(409, "payload changed since review") from e
    if not isinstance(data, dict) or any(
        data.get(k) is None for k in ("model", "ratio", "duration", "resolution")
    ):
        raise SubmitError(409, "payload changed since review")
    return data


def _keyframes(cdir: Path) -> list[Path]:
    # 与 pipeline 的产物布局对齐：目录里还有 contact_sheet.jpg/manifest.json，只取关键帧 PNG
    kdir = cdir / "work" / "keyframes"
    files = sorted(
        p for p in kdir.iterdir()
        if p.is_file() and p.suffix == ".png" and "keyframe" in p.name
    ) if kdir.is_dir() else []
    if not files:
        raise SubmitError(409, "payload changed since review")
    return files


def _create_argv(reviewed: dict, keyframes: list[Path]) -> list[str]:
    """以评审 payload 的标量 + 当前 prompt/keyframes 重建 create argv（相对 cwd 路径）。"""
    return [
        sys.executable, str(_SCRIPT), "create",
        "--prompt-file", "work/seedance_prompt.txt",
        "--ref-images", *[f"work/keyframes/{p.name}" for p in keyframes],
        "--model", str(reviewed["model"]),
        "--ratio", str(reviewed["ratio"]),
        "--duration", str(reviewed["duration"]),
        "--resolution", str(reviewed["resolution"]),
        "--generate-audio" if reviewed.get("generate_audio") else "--no-generate-audio",
        "--watermark" if reviewed.get("watermark") else "--no-watermark",
    ]


def _recheck_payload(cdir: Path, reviewed: dict, keyframes: list[Path]) -> None:
    """重放 dry-run 重建 payload，与评审版本逐项比对；任何不一致即 409。"""
    out = "work/recheck_payload.json"
    argv = _create_argv(reviewed, keyframes) + ["--dry-run", "--payload-out", out]
    try:
        r = subprocess.run(
            argv, cwd=cdir, capture_output=True, text=True, timeout=_DRYRUN_TIMEOUT_S
        )
        rebuilt = (
            json.loads((cdir / out).read_text(encoding="utf-8")) if r.returncode == 0 else None
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.info("seedance dry-run recheck failed for %s: %s", cdir.name, _sanitize(str(e)))
        raise SubmitError(409, "payload changed since review") from e
    finally:
        (cdir / out).unlink(missing_ok=True)
    if not isinstance(rebuilt, dict) or not _same_payload(reviewed, rebuilt):
        log.info("seedance payload mismatch for %s", cdir.name)
        raise SubmitError(409, "payload changed since review")


def _same_payload(reviewed: dict, rebuilt: dict) -> bool:
    if any(reviewed.get(k) != rebuilt.get(k) for k in _COMPARE_FIELDS):
        return False
    rc, bc = reviewed.get("content"), rebuilt.get("content")
    if not isinstance(rc, list) or not isinstance(bc, list) or len(rc) != len(bc):
        return False
    for a, b in zip(rc, bc):
        if not isinstance(a, dict) or not isinstance(b, dict) or a.get("text") != b.get("text"):
            return False
    return True


def _run_submit(cdir: Path, reviewed: dict, keyframes: list[Path]) -> None:
    """真实提交：argv 列表、无 shell、env 缺省继承服务进程、1800s 超时。"""
    argv = _create_argv(reviewed, keyframes) + [
        "--confirm-submit", "--wait",
        "--state-file", "work/task.json",
        "--download", "generated.mp4",
    ]
    try:
        r = subprocess.run(
            argv, cwd=cdir, capture_output=True, text=True, timeout=_SUBMIT_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        raise SubmitError(502, "seedance task timed out") from e
    except OSError as e:
        raise SubmitError(502, "seedance runner unavailable") from e
    if r.returncode != 0:
        detail = _sanitize(f"{r.stdout or ''}\n{r.stderr or ''}") or "seedance task failed"
        detail = detail[:_DETAIL_LIMIT]
        log.warning("seedance submit failed for %s: %s", cdir.name, detail)
        raise SubmitError(502, detail)


def _sanitize(text: str) -> str:
    """剔除任何含 key/authorization 的行，并就地抹除密钥字面值。"""
    out = "\n".join(ln for ln in text.splitlines() if not _LEAK_RE.search(ln)).strip()
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        out = out.replace(key, "***")
    return out


def _read_task_id(cdir: Path) -> str | None:
    """task.json 由脚本自写；读不到不致命（视频已下载，照样标记防重复扣费）。"""
    try:
        task = json.loads((cdir / "work" / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(task, dict) and task.get("id"):
        return str(task["id"])
    return None
