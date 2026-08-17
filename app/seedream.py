"""预留的 Seedream 图像编辑提交：开关 + dry-run 复核 + confirm 机械门控，默认 501。

confirm 语义简化为内部 flag：本模块只提供纯函数 edit_image，由未来的路由层在用户
确认后调用；脚本侧 --confirm-submit 机械门控始终显式传入。密钥只存在于服务进程
环境（ARK_API_KEY），子进程直接继承；不进日志、不进响应。报错信息一律先过
_sanitize 脱敏。
"""

import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from app.config import Settings

log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve().parent / "seedream_task.py"
# 脚本 poll 默认 600s + 提交/下载各 120s，900s 覆盖其最坏耗时并留余量
_SUBMIT_TIMEOUT_S = 900
_DRYRUN_TIMEOUT_S = 120
_DETAIL_LIMIT = 300
_LEAK_RE = re.compile(r"key|authorization", re.IGNORECASE)


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
) -> Path:
    """按固定顺序过门控；同一把锁内重查 out 已存在，防并发重复扣费。"""
    if not settings.enable_seedream_edit:
        raise SeedreamError(501, "Seedream edit is disabled.")
    if out.exists():
        raise SeedreamError(409, "already edited")
    _check_input(image, prompt)
    async with lock:
        if out.exists():
            raise SeedreamError(409, "already edited")
        await asyncio.to_thread(_dryrun_check, cdir, image, prompt, out)
        if not os.environ.get("ARK_API_KEY", "").strip():
            raise SeedreamError(503, "ARK_API_KEY not configured")
        await asyncio.to_thread(_run_edit, cdir, image, prompt, out, settings.seedream_model)
        return out


def _check_input(image: Path, prompt: str) -> None:
    """入口级校验：图缺失/指令为空等同无效请求（深校验交给 dry-run 脚本）。"""
    if not image.is_file() or not prompt.strip():
        raise SeedreamError(409, "invalid edit request")


def _dryrun_check(cdir: Path, image: Path, prompt: str, out: Path) -> None:
    """提交前 dry-run 重建请求预检（无网络、无费用）；构建失败等同无效请求。"""
    argv = [sys.executable, str(_SCRIPT), "edit",
            "--image", str(image), "--prompt", prompt, "--out", str(out), "--dry-run"]
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
        detail = detail[:_DETAIL_LIMIT]
        log.warning("seedream edit failed: %s", detail)
        raise SeedreamError(502, detail)


def _sanitize(text: str) -> str:
    """剔除任何含 key/authorization 的行，并就地抹除密钥字面值。"""
    out = "\n".join(ln for ln in text.splitlines() if not _LEAK_RE.search(ln)).strip()
    key = os.environ.get("ARK_API_KEY", "").strip()
    if key:
        out = out.replace(key, "***")
    return out
