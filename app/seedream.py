"""预留的 Seedream 图像编辑提交：开关 + confirm + 并发锁三重门控，默认 501。

confirm 语义由调用方传入（同 seedance.submit 的 payload.confirm，必须严格 True）：
路由层在用户确认后传 True；脚本侧 --confirm-submit 机械门控始终显式传入。密钥只
存在于服务进程环境（ARK_API_KEY），子进程直接继承；不进日志、不进响应。报错信息
一律先过 _sanitize 脱敏。
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from app.config import Settings
from app.sanitize import sanitize as _sanitize

log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parent / "seedream_task.py"
# 脚本最坏耗时 120(提交)+600(轮询)+120(下载)+120(请求超时) = 960s；
# 外层超时须覆盖全部，否则会在写盘中途杀子进程
_SUBMIT_TIMEOUT_S = 960
_DRYRUN_TIMEOUT_S = 120


class SeedreamError(Exception):
    """编辑门控/执行失败（路由层转成 status+detail，同 SubmitError 模式）。"""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


async def edit_image(
    settings: Settings,
    cdir: Path,
    image: Path,
    prompt: str,
    out: Path,
    lock: asyncio.Lock,
    confirm: bool,
) -> Path:
    """按固定顺序过门控（与 seedance.submit 对齐）；同一把锁内重查 out 已存在，防并发重复扣费。"""
    if not settings.enable_seedream_edit:
        raise SeedreamError(501, "Seedream edit is disabled.")
    if confirm is not True:
        raise SeedreamError(409, "confirmation required")
    if out.exists():
        raise SeedreamError(409, "already edited")
    _check_input(image, prompt)
    async with lock:
        if out.exists():
            raise SeedreamError(409, "already edited")
        await asyncio.to_thread(_dryrun_check, cdir, image, prompt, out, settings.seedream_model)
        if not os.environ.get("ARK_API_KEY", "").strip():
            raise SeedreamError(503, "ARK_API_KEY not configured")
        await asyncio.to_thread(_run_edit, cdir, image, prompt, out, settings.seedream_model)
        return out


def _check_input(image: Path, prompt: str) -> None:
    """入口级校验：图缺失/指令为空等同无效请求（深校验交给 dry-run 脚本）。"""
    if not image.is_file() or not prompt.strip():
        raise SeedreamError(409, "invalid edit request")


def _dryrun_check(cdir: Path, image: Path, prompt: str, out: Path, model: str) -> None:
    """提交前 dry-run 重建请求预检（无网络、无费用，带 --model 保证即真实请求形态）；构建失败等同无效请求。"""
    argv = [sys.executable, str(_SCRIPT), "edit",
            "--image", str(image), "--prompt", prompt, "--out", str(out),
            "--model", model, "--dry-run"]
    try:
        r = subprocess.run(
            argv, cwd=cdir, capture_output=True, text=True, timeout=_DRYRUN_TIMEOUT_S
        )
        ok = r.returncode == 0 and '"dry_run": true' in (r.stdout or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        log.info("seedream dry-run precheck failed: %s", _sanitize(str(e)))
        raise SeedreamError(409, "invalid edit request") from e
    if not ok:
        log.info("seedream dry-run precheck failed (rc=%s)", r.returncode)
        raise SeedreamError(409, "invalid edit request")


def _run_edit(cdir: Path, image: Path, prompt: str, out: Path, model: str) -> None:
    """真实提交：argv 列表、无 shell、env 缺省继承服务进程、900s 超时。"""
    argv = [sys.executable, str(_SCRIPT), "edit",
            "--image", str(image), "--prompt", prompt, "--out", str(out),
            "--model", model, "--confirm-submit"]
    try:
        r = subprocess.run(
            argv, cwd=cdir, capture_output=True, text=True, timeout=_SUBMIT_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        raise SeedreamError(502, "seedream task timed out") from e
    except OSError as e:
        raise SeedreamError(502, "seedream runner unavailable") from e
    if r.returncode != 0:
        detail = _sanitize(f"{r.stdout or ''}\n{r.stderr or ''}") or "seedream task failed"
        log.warning("seedream edit failed: %s", detail)
        raise SeedreamError(502, detail)
