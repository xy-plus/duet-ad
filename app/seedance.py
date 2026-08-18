"""预留的 Seedance 真实提交：开关 + confirm + dry-run 复核三重门控，默认 501。

密钥只存在于服务进程环境（ARK_API_KEY），子进程直接继承；不进日志、不进响应、
不进 meta.json。报错信息一律先过 _sanitize 脱敏。
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from app import storage
from app.config import Settings
from app.sanitize import sanitize as _sanitize

log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parent / "seedance_task.py"
_SUBMIT_TIMEOUT_S = 1800
_DRYRUN_TIMEOUT_S = 120
# 建模特固定（新契约无评审 payload，提交时由 work/prompt.txt + work/keyframes/*.png 现构建请求）
_MODEL = "doubao-seedance-2-0-260128"


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
        # data_dir 可能是相对路径：dry-run/提交子进程带 cwd 时相对路径会错位，统一转绝对（与 pipeline 同口径）
        cdir = (settings.data_dir / cid).resolve()
        _check_prompt(cdir)
        keyframes = _keyframes(cdir)
        await asyncio.to_thread(_dryrun_check, cdir, keyframes)
        if not os.environ.get("ARK_API_KEY", "").strip():
            raise SubmitError(503, "ARK_API_KEY not configured")
        await asyncio.to_thread(_run_submit, cdir, keyframes)
        storage.mark_submitted(settings.data_dir, cid, _read_task_id(cdir))
        return {"status": "succeeded", "video": "generated.mp4"}


def _check_prompt(cdir: Path) -> None:
    """work/prompt.txt 是评审确认的提示词；缺失/为空等同产物已变。"""
    try:
        prompt = (cdir / "work" / "prompt.txt").read_text(encoding="utf-8")
    except OSError as e:
        raise SubmitError(409, "payload changed since review") from e
    if not prompt.strip():
        raise SubmitError(409, "payload changed since review")


def _keyframes(cdir: Path) -> list[Path]:
    """work/keyframes/*.png 即全部参考图（新契约该目录只有选定帧）；空则等同产物已变。

    T5b：每张帧若存在 work/postprocessed/<同名> 优化图则优先用之（多段暂不涉及——
    seedance 提交仅支持单段 work/ 契约）。
    """
    kdir = cdir / "work" / "keyframes"
    files = (
        sorted(p for p in kdir.glob("*.png") if p.is_file()) if kdir.is_dir() else []
    )
    if not files:
        raise SubmitError(409, "payload changed since review")
    post = cdir / "work" / "postprocessed"
    return [post / p.name if (post / p.name).is_file() else p for p in files]


def _create_argv(cdir: Path, keyframes: list[Path]) -> list[str]:
    """提交时现构建 create argv（相对 cwd 路径 + 固定建模特）。"""
    return [
        sys.executable, str(_SCRIPT), "create",
        "--prompt-file", "work/prompt.txt",
        "--ref-images", *[str(p.relative_to(cdir)) for p in keyframes],
        "--model", _MODEL,
        "--ratio", "9:16",
        "--duration", "15",
        "--resolution", "720p",
        "--generate-audio",
        "--no-watermark",
    ]


def _dryrun_check(cdir: Path, keyframes: list[Path]) -> None:
    """提交前 dry-run 重建 payload 预检（无网络、无费用）；构建失败等同产物已变。"""
    out = "work/recheck_payload.json"
    argv = _create_argv(cdir, keyframes) + ["--dry-run", "--payload-out", out]
    try:
        r = subprocess.run(
            argv, cwd=cdir, capture_output=True, text=True, timeout=_DRYRUN_TIMEOUT_S
        )
        ok = r.returncode == 0 and (cdir / out).is_file()
    except (OSError, subprocess.TimeoutExpired) as e:
        log.info("seedance dry-run precheck failed for %s: %s", cdir.name, _sanitize(str(e)))
        raise SubmitError(409, "payload changed since review") from e
    finally:
        (cdir / out).unlink(missing_ok=True)
    if not ok:
        log.info("seedance dry-run precheck failed for %s", cdir.name)
        raise SubmitError(409, "payload changed since review")


def _run_submit(cdir: Path, keyframes: list[Path]) -> None:
    """真实提交：argv 列表、无 shell、env 缺省继承服务进程、1800s 超时。"""
    argv = _create_argv(cdir, keyframes) + [
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
        log.warning("seedance submit failed for %s: %s", cdir.name, detail)
        raise SubmitError(502, detail)


def _read_task_id(cdir: Path) -> str | None:
    """task.json 由脚本自写；读不到不致命（视频已下载，照样标记防重复扣费）。"""
    try:
        task = json.loads((cdir / "work" / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(task, dict) and task.get("id"):
        return str(task["id"])
    return None
