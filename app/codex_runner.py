"""沙箱化 codex exec 调用：workspace-write、断网、环境清洗、硬超时、并发信号量。

安全约束（逐条对应任务 B 要求，均经 codex-cli 0.147.0 实证）：
- 永远 argv 列表调起，永不 shell=True，永不使用 --dangerously-bypass-*；
- agent shell 断网：sandbox_workspace_write.network_access=false（实证 curl 不通）；
- 环境清洗双保险：
  a) 宿主进程级：调起 codex 前剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的环境变量
     （实证：0.147.0 的 shell 命令经 code-mode-host 执行，shell_environment_policy
     的 inherit/exclude 不能阻止宿主秘密泄漏进 agent shell，必须在本进程侧清洗）；
    b) codex 配置级：inherit="core" + exclude 兜底；
- 视觉步骤使用 Codex 默认沙箱后端；voice 步骤额外由 bwrap 遮住 checkout、会话与其余 /tmp；
  宿主服务必须允许两层沙箱创建所需的 user namespace；
- 硬超时 settings.codex_timeout_s；并发信号量 settings.codex_concurrency；
- 超时/非零退出 → CodexError，stderr 先剔除环境变量行再截断 ≤500 字。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, TypeVar

# 调用级固定 medium：流水线看图/听写任务要速度不要 max 深度（实测 max 下段任务 30 分钟超时 vs medium 410s）；用户交互式 codex 的全局 effort 不受影响
_SANDBOX_CONFIGS = [
    "model_reasoning_effort=\"medium\"",
    "sandbox_workspace_write.network_access=false",
    'shell_environment_policy.inherit="core"',
    'shell_environment_policy.exclude=["*KEY*","*TOKEN*","*SECRET*","*PASSWORD*"]',
]

_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
_VOICE_OUTPUT_MAX_BYTES = 32 * 1024
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_CHECKOUT_BOUNDARY = (
    _SOURCE_ROOT.parent.parent if _SOURCE_ROOT.parent.name == ".worktree" else _SOURCE_ROOT
)
_T = TypeVar("_T")

# run_voice() 仍经由公开 run() 进入同一并发/超时/env 清洗路径；ContextVar 只把本次同步调用
# 标为音频 stage，使 build_argv() 加上外层 bwrap。线程间不共享，视觉 run() 不会误继承。
_ACTIVE_VOICE_STAGE: ContextVar[tuple[int, Path, Path] | None] = ContextVar(
    "active_voice_stage", default=None
)


class CodexError(RuntimeError):
    """codex 启动/运行失败（超时、非零退出、找不到二进制）。"""


class CodexOutputError(RuntimeError):
    """codex 成功退出，但音频隔离区没有可安全读取的唯一输出。"""


def clean_stderr(text: str | None, limit: int = 500) -> str:
    """剔除环境变量行（KEY=VALUE），截断到 limit 字。"""
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not _ENV_LINE_RE.match(l.strip())]
    return "\n".join(lines).strip()[-limit:]


def _scrubbed_env() -> dict[str, str]:
    """剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的变量；PATH/HOME/代理等保留。"""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV_RE.search(k)}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_bwrap() -> Path:
    executable = shutil.which("bwrap")
    if not executable:
        raise CodexError("bwrap executable not found on PATH; voice isolation unavailable")
    try:
        resolved = Path(executable).resolve(strict=True)
    except OSError:
        raise CodexError("bwrap executable path is invalid; voice isolation unavailable") from None
    if not resolved.is_file() or not resolved.is_absolute():
        raise CodexError("bwrap executable path is invalid; voice isolation unavailable")
    return resolved


def _voice_outer_argv(stage: Path, session_dir: Path, inner_argv: list[str]) -> list[str]:
    """构造音频专用外层挂载沙箱；任一路径异常都拒绝运行。"""
    try:
        tmp_root = Path("/tmp").resolve(strict=True)
        stage = stage.resolve(strict=True)
        session_dir = session_dir.resolve(strict=True)
        checkout = _CHECKOUT_BOUNDARY.resolve(strict=True)
    except OSError:
        raise CodexError("voice isolation path is missing or invalid") from None
    if stage.parent != tmp_root or not stage.is_dir():
        raise CodexError("voice isolation stage must be a direct child of /tmp")
    if not session_dir.is_dir() or session_dir in {Path("/"), tmp_root, checkout}:
        raise CodexError("voice isolation session path is unsafe")
    if _is_relative_to(stage, session_dir):
        raise CodexError("voice isolation stage must be outside the source session")
    if not inner_argv:
        raise CodexError("voice isolation inner command is empty")

    argv = [
        str(_resolve_bwrap()),
        "--bind", "/", "/",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", str(checkout),
    ]
    # 生产会话通常位于 checkout 下，/tmp 测试会话由下方整棵 /tmp 隐藏；只有独立数据根
    # 才需额外遮住会话叶目录。避免在 tmpfs 父挂载后再引用已经消失的子挂载点。
    if not _is_relative_to(session_dir, checkout) and not _is_relative_to(session_dir, tmp_root):
        argv += ["--tmpfs", str(session_dir)]
    argv += [
        "--tmpfs", str(tmp_root),
        "--bind", str(stage), str(stage),
        "--chdir", str(stage),
        *inner_argv,
    ]
    return argv


def _copy_voice_input(source: Path, destination: Path) -> None:
    """用 O_NOFOLLOW 复制普通音频文件，避免 symlink/TOCTOU 把额外文件带进 stage。"""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError:
        raise CodexError("work/voice.mp3 is missing, symlinked, or unreadable") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise CodexError("work/voice.mp3 must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        os.close(fd)


def _read_voice_output(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise CodexOutputError("isolated voice_lines.json is missing or unreadable") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _VOICE_OUTPUT_MAX_BYTES:
            raise CodexOutputError("isolated voice_lines.json is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(_VOICE_OUTPUT_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _VOICE_OUTPUT_MAX_BYTES:
        raise CodexOutputError("isolated voice_lines.json exceeds output limit")
    return raw


class CodexRunner:
    def __init__(self, timeout_s: int, concurrency: int) -> None:
        self._timeout_s = timeout_s
        self._sem = threading.Semaphore(concurrency)

    def build_argv(self, workdir: Path, prompt: str) -> list[str]:
        active = _ACTIVE_VOICE_STAGE.get()
        voice_stage: tuple[Path, Path] | None = None
        if active is not None and active[0] == id(self):
            voice_stage = (active[1], active[2])
            try:
                actual = Path(workdir).resolve(strict=True)
            except OSError:
                raise CodexError("voice isolation workdir is invalid") from None
            if actual != voice_stage[0]:
                raise CodexError("voice isolation workdir mismatch")

        argv = [
            "codex", "exec",
            "-C", str(workdir),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "-o",
            "/dev/null"
            if voice_stage is not None
            else str(workdir / "codex_last_message.txt"),
        ]
        for cfg in _SANDBOX_CONFIGS:
            argv += ["-c", cfg]
        argv.append(prompt)
        if voice_stage is None:
            return argv
        return _voice_outer_argv(voice_stage[0], voice_stage[1], argv)

    def run(self, workdir: Path, prompt: str) -> None:
        argv = self.build_argv(workdir, prompt)
        with self._sem:
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_s,
                    env=_scrubbed_env(),
                )
            except subprocess.TimeoutExpired:
                raise CodexError(f"codex timed out after {self._timeout_s}s") from None
            except FileNotFoundError:
                raise CodexError("codex executable not found on PATH") from None
        if proc.returncode != 0:
            raise CodexError(f"codex exit {proc.returncode}: {clean_stderr(proc.stderr)}")

    def run_voice(
        self,
        work: Path,
        prompt: str,
        *,
        duration_s: float,
        validate_output: Callable[[bytes], _T],
    ) -> _T:
        """仅供自动台词步骤：最小输入 staging + 外层 bwrap + 校验后返回内存值。

        不接受任意输入清单，也不把 stage 路径交给调用方；因此调用方无法顺手把 source、帧或
        视觉 prompt 加进 agent 工作区。每次调用新建 stage，天然清除重试前的旧输出。Codex
        超时/失败但已经写出完整白名单产物时仍收养；否则保留原 CodexError。
        """
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
            raise CodexError("voice isolation duration must be a positive finite number")
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise CodexError("voice isolation duration must be a positive finite number")
        if not callable(validate_output):
            raise CodexError("voice isolation output validator is required")
        _resolve_bwrap()  # 缺能力时在复制输入、调 agent 前 fail closed。

        requested_work = Path(work)
        try:
            resolved_work = requested_work.resolve(strict=True)
        except OSError:
            raise CodexError("voice isolation work directory is missing or invalid") from None
        if (
            not requested_work.is_absolute()
            or requested_work != resolved_work
            or resolved_work.name != "work"
            or not resolved_work.is_dir()
        ):
            raise CodexError("voice isolation requires a regular session work directory")
        session_dir = resolved_work.parent

        try:
            tmp_root = Path("/tmp").resolve(strict=True)
            with tempfile.TemporaryDirectory(prefix="duet-voice-", dir=tmp_root) as raw_stage:
                stage = Path(raw_stage).resolve(strict=True)
                if stage.parent != tmp_root:
                    raise CodexError("voice isolation stage path is invalid")
                stage_work = stage / "work"
                stage_work.mkdir(mode=0o700)
                _copy_voice_input(resolved_work / "voice.mp3", stage_work / "voice.mp3")
                (stage_work / "manifest.json").write_text(
                    json.dumps({"duration_seconds": duration_s}, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )

                token = _ACTIVE_VOICE_STAGE.set((id(self), stage, session_dir))
                run_error: CodexError | None = None
                try:
                    try:
                        self.run(stage, prompt)
                    except CodexError as error:
                        run_error = error
                finally:
                    _ACTIVE_VOICE_STAGE.reset(token)

                try:
                    result = validate_output(
                        _read_voice_output(stage_work / "voice_lines.json")
                    )
                except Exception:
                    if run_error is not None:
                        raise run_error from None
                    raise
                return result
        except CodexError:
            raise
        except OSError:
            raise CodexError("voice isolation staging failed") from None
