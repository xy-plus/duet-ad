"""沙箱化 codex exec 调用：workspace-write、断网、环境清洗、硬超时、并发信号量。

安全约束（逐条对应任务 B 要求，均经 codex-cli 0.147.0 实证）：
- 永远 argv 列表调起，永不 shell=True，永不使用 --dangerously-bypass-*；
- agent shell 断网：sandbox_workspace_write.network_access=false（实证 curl 不通）；
- 环境清洗双保险：
  a) 宿主进程级：调起 codex 前剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的环境变量
     （实证：0.147.0 的 shell 命令经 code-mode-host 执行，shell_environment_policy
     的 inherit/exclude 不能阻止宿主秘密泄漏进 agent shell，必须在本进程侧清洗）；
    b) codex 配置级：inherit="core" + exclude 兜底；
- 视觉步骤使用 Codex 默认沙箱后端；隔离 Skill 与 voice 步骤额外由 bwrap 遮住 checkout、会话与其余 /tmp；
  宿主服务必须允许两层沙箱创建所需的 user namespace；
- 硬超时 settings.codex_timeout_s；并发信号量 settings.codex_concurrency；
- 超时/非零退出 → CodexError，stderr 先剔除环境变量行再截断 ≤500 字。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, NoReturn, TypeVar

from app import error_trace


_LOGGER = logging.getLogger(__name__)

# 默认调用级固定 medium；少数明确恢复任务可在构造器中显式覆盖。流水线看图/听写任务仍取默认值，
# 用户交互式 codex 的全局 effort 不受影响。
_SANDBOX_CONFIGS = [
    "sandbox_workspace_write.network_access=false",
    'shell_environment_policy.inherit="core"',
    'shell_environment_policy.exclude=["*KEY*","*TOKEN*","*SECRET*","*PASSWORD*"]',
]

_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
_CODEX_TELEMETRY_RE = re.compile(
    r"[ \t\r\n]*tokens used\r?\n"
    r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)[ \t\r\n]*"
)
_VOICE_OUTPUT_MAX_BYTES = 32 * 1024
_FINAL_OUTPUT_EXCERPT_BYTES = 256
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_CHECKOUT_BOUNDARY = (
    _SOURCE_ROOT.parent.parent if _SOURCE_ROOT.parent.name == ".worktree" else _SOURCE_ROOT
)
_T = TypeVar("_T")

# 隔离任务仍经由公开 run() 进入同一并发/超时/env 清洗路径；ContextVar 只把本次同步调用
# 标为独立 stage，使 build_argv() 加上外层 bwrap。线程间不共享，普通视觉 run() 不会误继承。
_ACTIVE_ISOLATED_STAGE: ContextVar[
    tuple[int, Path, Path, tuple[Path, ...], Path | None] | None
] = ContextVar(
    "active_isolated_stage", default=None
)


class CodexError(RuntimeError):
    """codex 启动/运行失败（超时、非零退出、找不到二进制）。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class CodexOutputError(RuntimeError):
    """codex 成功退出，但音频隔离区没有可安全读取的唯一输出。"""


def clean_stderr(text: str | None, limit: int = 500) -> str:
    """剔除环境变量行（KEY=VALUE），截断到 limit 字。"""
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not _ENV_LINE_RE.match(l.strip())]
    return "\n".join(lines).strip()[-limit:]


def _output_candidates(raw: bytes) -> tuple[bytes, ...]:
    """Normalize only transport wrappers; semantic JSON stays model-owned."""
    stripped = raw.strip()
    candidates = [stripped]
    if stripped.startswith(b"```") and stripped.endswith(b"```"):
        first_newline = stripped.find(b"\n")
        if first_newline >= 0:
            fenced = stripped[first_newline + 1:-3].strip()
            if fenced and fenced not in candidates:
                candidates.append(fenced)
    return tuple(candidates)


def _extract_codex_json_output(raw: bytes) -> bytes:
    """Extract exactly one JSON value from Codex's final-answer transport.

    Plain JSON is returned byte-for-byte.  The only accepted suffix outside
    that value is Codex CLI's known two-line token counter.  A complete JSON
    Markdown fence remains supported as an explicit transport wrapper; prose,
    a second JSON value, and every unknown prefix or suffix fail closed.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Codex final output is not UTF-8 JSON") from None

    def reject_constant(_value: str) -> NoReturn:
        raise ValueError("Codex final output contains a non-JSON constant")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    leading = re.match(r"[ \t\r\n]*", text)
    assert leading is not None
    start = leading.end()
    fenced = False
    if text.startswith("```", start):
        header = re.match(r"```(?:json)?[ \t]*\r?\n", text[start:])
        if header is None:
            raise ValueError("Codex final output has an invalid JSON fence")
        fenced = True
        start += header.end()
        json_leading = re.match(r"[ \t\r\n]*", text[start:])
        assert json_leading is not None
        start += json_leading.end()

    try:
        _value, end = decoder.raw_decode(text, start)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("Codex final output does not start with one JSON value") from None

    suffix_start = end
    if fenced:
        fence_gap = re.match(r"[ \t\r\n]*", text[end:])
        assert fence_gap is not None
        closing = end + fence_gap.end()
        if not text.startswith("```", closing):
            raise ValueError("Codex final output has an incomplete JSON fence")
        suffix_start = closing + 3

    suffix = text[suffix_start:]
    if not suffix.strip():
        if not fenced:
            return raw
        return text[start:end].encode("utf-8")
    if _CODEX_TELEMETRY_RE.fullmatch(suffix) is None:
        raise ValueError("Codex final output has unknown text outside its JSON value")
    return text[start:end].encode("utf-8")


def _redacted_output_excerpt(raw: bytes) -> str:
    """Preserve transport shape without persisting model-owned text."""
    rendered: list[str] = []
    for byte in raw:
        if 65 <= byte <= 90 or 97 <= byte <= 122:
            rendered.append("x")
        elif 48 <= byte <= 57:
            rendered.append("0")
        elif byte == 10:
            rendered.append(r"\n")
        elif byte == 13:
            rendered.append(r"\r")
        elif byte == 9:
            rendered.append(r"\t")
        elif byte == 32:
            rendered.append(" ")
        elif byte in b'{}[]:,"`':
            rendered.append(chr(byte))
        else:
            rendered.append(".")
    return "".join(rendered)


def _known_telemetry_suffix_matched(raw_tail: bytes) -> bool:
    text = raw_tail.decode("utf-8", errors="replace")
    return any(
        _CODEX_TELEMETRY_RE.fullmatch(text[index:]) is not None
        for index in range(len(text) + 1)
    )


def _atomic_publish_output(parent: Path, destination: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-transport-", dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


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
        raise CodexError("bwrap executable not found on PATH; isolated execution unavailable")
    try:
        resolved = Path(executable).resolve(strict=True)
    except OSError:
        raise CodexError("bwrap executable path is invalid; isolated execution unavailable") from None
    if not resolved.is_file() or not resolved.is_absolute():
        raise CodexError("bwrap executable path is invalid; isolated execution unavailable")
    return resolved


def _isolated_writable_paths(
    stage: Path, writable_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    resolved_paths: list[Path] = []
    for raw in writable_paths:
        requested = Path(raw)
        try:
            resolved = requested.resolve(strict=True)
            resolved.relative_to(stage)
        except (OSError, ValueError):
            raise CodexError("isolated writable path is invalid") from None
        if (
            not requested.is_absolute()
            or requested != resolved
            or requested.is_symlink()
            or not resolved.is_file()
            or resolved in resolved_paths
        ):
            raise CodexError("isolated writable path is invalid")
        resolved_paths.append(resolved)
    return tuple(resolved_paths)


def _isolated_readonly_inputs(
    stage: Path, writable_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return every staged input file that must remain immutable in the sandbox."""
    writable = set(writable_paths)
    readonly: list[Path] = []
    for candidate in sorted(stage.rglob("*"), key=lambda path: str(path)):
        if candidate.is_symlink():
            raise CodexError("isolated stage contains an invalid input")
        if candidate.is_dir():
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(stage)
        except (OSError, ValueError):
            raise CodexError("isolated stage contains an invalid input") from None
        if not resolved.is_file():
            raise CodexError("isolated stage contains an invalid input")
        if resolved not in writable:
            readonly.append(resolved)
    return tuple(readonly)


def _isolated_outer_argv(
    stage: Path,
    session_dir: Path,
    inner_argv: list[str],
    *,
    writable_paths: tuple[Path, ...] = (),
) -> list[str]:
    """构造单次任务外层挂载沙箱；任一路径异常都拒绝运行。"""
    try:
        tmp_root = Path("/tmp").resolve(strict=True)
        stage = stage.resolve(strict=True)
        session_dir = session_dir.resolve(strict=True)
        checkout = _CHECKOUT_BOUNDARY.resolve(strict=True)
    except OSError:
        raise CodexError("isolated execution path is missing or invalid") from None
    if stage.parent != tmp_root or not stage.is_dir():
        raise CodexError("isolated stage must be a direct child of /tmp")
    if not session_dir.is_dir() or session_dir in {Path("/"), tmp_root, checkout}:
        raise CodexError("isolated session path is unsafe")
    if _is_relative_to(stage, session_dir):
        raise CodexError("isolated stage must be outside the source session")
    if not inner_argv:
        raise CodexError("isolated inner command is empty")
    writable = _isolated_writable_paths(stage, writable_paths)
    readonly_inputs = _isolated_readonly_inputs(stage, writable)

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
    argv += ["--tmpfs", str(tmp_root)]
    if writable:
        # Keep the stage directories writable so tools that publish atomically
        # (temporary file + rename) can replace the declared output.  Overlay
        # every staged input as a read-only file mount; those mount points
        # cannot be modified, removed, or replaced from inside the namespace.
        argv += ["--bind", str(stage), str(stage)]
        for path in readonly_inputs:
            argv += ["--ro-bind", str(path), str(path)]
    else:
        argv += ["--bind", str(stage), str(stage)]
    argv += ["--chdir", str(stage), *inner_argv]
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


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort termination scoped to one start_new_session invocation."""
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        proc.poll()  # Reap an exited leader so it does not keep the PGID observable.
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            proc.wait()
            return
        time.sleep(0.02)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


class CodexRunner:
    def __init__(
        self,
        timeout_s: int,
        concurrency: int,
        *,
        model: str | None = None,
        reasoning_effort: str = "medium",
    ) -> None:
        if model is not None and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", model) is None:
            raise ValueError("invalid Codex model")
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid Codex reasoning effort")
        self._timeout_s = timeout_s
        self._sem = threading.Semaphore(concurrency)
        self._model = model
        self._reasoning_effort = reasoning_effort

    def build_argv(self, workdir: Path, prompt: str) -> list[str]:
        active = _ACTIVE_ISOLATED_STAGE.get()
        isolated_stage: tuple[Path, Path, tuple[Path, ...]] | None = None
        if active is not None and active[0] == id(self):
            isolated_stage = (active[1], active[2], active[3], active[4])
            try:
                actual = Path(workdir).resolve(strict=True)
            except OSError:
                raise CodexError("isolated workdir is invalid") from None
            if actual != isolated_stage[0]:
                raise CodexError("isolated workdir mismatch")

        argv = [
            "codex", "exec",
            "-C", str(workdir),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "-o",
            str(isolated_stage[3])
            if isolated_stage is not None and isolated_stage[3] is not None
            else "/dev/null"
            if isolated_stage is not None
            else str(workdir / "codex_last_message.txt"),
        ]
        if self._model is not None:
            argv += ["-m", self._model]
        argv += ["-c", f'model_reasoning_effort="{self._reasoning_effort}"']
        for cfg in _SANDBOX_CONFIGS:
            argv += ["-c", cfg]
        argv.append(prompt)
        if isolated_stage is None:
            return argv
        return _isolated_outer_argv(
            isolated_stage[0],
            isolated_stage[1],
            argv,
            writable_paths=isolated_stage[2],
        )

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
                raise CodexError(
                    f"codex timed out after {self._timeout_s}s", retryable=True
                ) from None
            except FileNotFoundError:
                raise CodexError("codex executable not found on PATH") from None
        if proc.returncode != 0:
            raise CodexError(
                f"codex exit {proc.returncode}: {clean_stderr(proc.stderr)}",
                retryable=True,
            )

    def run_isolated(
        self,
        workdir: Path,
        prompt: str,
        *,
        session_dir: Path,
        writable_paths: tuple[Path, ...] = (),
    ) -> None:
        """Run Codex in a prebuilt, single-use /tmp stage hidden from the source session."""
        _resolve_bwrap()
        try:
            tmp_root = Path("/tmp").resolve(strict=True)
            stage = Path(workdir).resolve(strict=True)
            session = Path(session_dir).resolve(strict=True)
        except OSError:
            raise CodexError("isolated execution path is missing or invalid") from None
        if stage.parent != tmp_root or not stage.is_dir():
            raise CodexError("isolated stage must be a direct child of /tmp")
        if not session.is_dir():
            raise CodexError("isolated session path is invalid")
        writable = _isolated_writable_paths(stage, writable_paths)
        token = _ACTIVE_ISOLATED_STAGE.set((id(self), stage, session, writable, None))
        try:
            self.run(stage, prompt)
        finally:
            _ACTIVE_ISOLATED_STAGE.reset(token)

    def run_isolated_until_output(
        self,
        workdir: Path,
        prompt: str,
        *,
        session_dir: Path,
        output_path: Path,
        max_output_bytes: int,
        validate_output: Callable[[bytes], _T],
    ) -> _T:
        """Return one valid declared output without delegating publication safety.

        The API creates the empty regular output placeholder itself.  A
        replacement inode that is byte-stable across two observations remains
        an early completion signal.  A normal in-place write is accepted only
        after Codex exits successfully and its process group has been stopped;
        the backend atomically publishes validated bytes to the declared path.
        Validation is a protocol/shape check supplied by the caller, not a
        content-quality decision.  Unrelated Codex work is untouched.
        """
        stage_hint = Path(workdir)
        session_hint = Path(session_dir)
        requested_hint = Path(output_path)
        final_failure_diagnostic: dict[str, object] | None = None

        def record_failure(exc: BaseException) -> None:
            """Best-effort diagnostics must never replace the operational error."""
            call_path = [
                "pipeline",
                "codex",
                stage_hint.name or "pre-spawn",
                requested_hint.name or "output",
            ]
            try:
                diagnostic_session = session_hint.resolve(strict=True)
                if not diagnostic_session.is_dir():
                    raise OSError("isolated session path is invalid")
                error_trace.record(
                    diagnostic_session / "work" / "errors"
                    / f"{stage_hint.name or 'pre-spawn'}.json",
                    call_path=call_path,
                    error=exc,
                    details=(
                        {"codex_final_output": final_failure_diagnostic}
                        if final_failure_diagnostic is not None
                        else None
                    ),
                    logger=_LOGGER,
                )
            except BaseException as record_error:
                try:
                    _LOGGER.error(
                        "codex_error_record_failed call_path=%s original=%s record_error=%s",
                        json.dumps(call_path, ensure_ascii=False, separators=(",", ":")),
                        json.dumps(
                            error_trace.exception_tree(exc),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            error_trace.exception_tree(record_error),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                except BaseException:
                    pass

        def raise_recorded(
            exc: BaseException, *, cause: BaseException | None = None,
        ) -> NoReturn:
            try:
                if cause is not None:
                    raise exc from cause
                raise exc
            except BaseException as raised:
                record_failure(raised)
                raise

        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
            or not callable(validate_output)
        ):
            raise_recorded(CodexError("isolated output contract is invalid"))
        try:
            _resolve_bwrap()
        except BaseException as exc:
            record_failure(exc)
            raise
        parent_fd = -1
        placeholder_fd = -1
        final_output_fd = -1
        try:
            tmp_root = Path("/tmp").resolve(strict=True)
            stage = Path(workdir).resolve(strict=True)
            session = Path(session_dir).resolve(strict=True)
            requested = Path(output_path)
            parent = requested.parent.resolve(strict=True)
            parent.relative_to(stage)
        except (OSError, ValueError) as cause:
            raise_recorded(CodexError("isolated output path is invalid"), cause=cause)
        if stage.parent != tmp_root or not stage.is_dir() or not session.is_dir():
            raise_recorded(CodexError("isolated execution path is missing or invalid"))
        if (
            not requested.is_absolute()
            or requested.parent != parent
            or requested.name in {"", ".", ".."}
            or requested.exists()
            or requested.is_symlink()
        ):
            raise_recorded(CodexError("isolated output must not already exist"))
        try:
            parent_fd = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            placeholder_fd = os.open(
                requested.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            initial = os.fstat(placeholder_fd)
            declared = Path(requested).resolve(strict=True)
            final_output = parent / ".codex-final-output.json"
            final_output_fd = os.open(
                final_output.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            final_output = final_output.resolve(strict=True)
        except OSError as cause:
            if final_output_fd >= 0:
                os.close(final_output_fd)
            if placeholder_fd >= 0:
                os.close(placeholder_fd)
            if parent_fd >= 0:
                os.close(parent_fd)
            raise_recorded(
                CodexError("isolated output placeholder could not be created"),
                cause=cause,
            )

        try:
            writable = _isolated_writable_paths(stage, (declared, final_output))
            token = _ACTIVE_ISOLATED_STAGE.set(
                (id(self), stage, session, writable, final_output)
            )
            try:
                argv = self.build_argv(stage, prompt)
            finally:
                _ACTIVE_ISOLATED_STAGE.reset(token)
        except BaseException as exc:
            record_failure(exc)
            os.close(final_output_fd)
            os.close(placeholder_fd)
            os.close(parent_fd)
            raise

        missing = object()
        invalid = object()

        def read_once(
            name: str,
            initial_stat: os.stat_result,
            *,
            allow_initial_inode: bool,
        ) -> tuple[tuple[int, int, int, int], bytes] | None:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(name, flags, dir_fd=parent_fd)
            except OSError:
                return None
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or (
                        not allow_initial_inode
                        and (info.st_dev, info.st_ino)
                        == (initial_stat.st_dev, initial_stat.st_ino)
                    )
                    or info.st_size <= 0
                    or info.st_size > max_output_bytes
                ):
                    return None
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    raw = stream.read(max_output_bytes + 1)
                if len(raw) > max_output_bytes:
                    return None
                after = os.fstat(fd)
                identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                if identity != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
                ):
                    return None
                return identity, raw
            finally:
                os.close(fd)

        def read_published(
            name: str = requested.name,
            initial_stat: os.stat_result = initial,
            *,
            allow_initial_inode: bool = False,
        ) -> tuple[_T, bytes] | object:
            first = read_once(
                name, initial_stat, allow_initial_inode=allow_initial_inode,
            )
            if first is None:
                return missing
            time.sleep(0.1)
            second = read_once(
                name, initial_stat, allow_initial_inode=allow_initial_inode,
            )
            if second is None or first != second:
                return missing
            candidates = _output_candidates(first[1])
            for candidate in candidates:
                try:
                    value = validate_output(candidate)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if value is None:
                    raise CodexError("isolated output validator returned None")
                return value, candidate
            return missing

        def publish(adopted: tuple[_T, bytes]) -> _T:
            value, payload = adopted
            _atomic_publish_output(parent, requested, payload)
            return value

        def final_snapshot() -> tuple[
            tuple[int, int, int, int], bytes, dict[str, object]
        ] | tuple[None, None, dict[str, object]]:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(final_output.name, flags, dir_fd=parent_fd)
            except OSError:
                try:
                    info = os.stat(
                        final_output.name, dir_fd=parent_fd, follow_symlinks=False,
                    )
                except OSError:
                    return None, None, {
                        "reason": "missing",
                        "size_bytes": None,
                        "sha256": None,
                        "telemetry_suffix_matched": False,
                        "head_redacted": "",
                        "tail_redacted": "",
                    }
                return None, None, {
                    "reason": (
                        "not_regular" if not stat.S_ISREG(info.st_mode)
                        else "unreadable"
                    ),
                    "size_bytes": info.st_size,
                    "sha256": None,
                    "telemetry_suffix_matched": False,
                    "head_redacted": "",
                    "tail_redacted": "",
                }
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    return None, None, {
                        "reason": "not_regular",
                        "size_bytes": info.st_size,
                        "sha256": None,
                        "telemetry_suffix_matched": False,
                        "head_redacted": "",
                        "tail_redacted": "",
                    }
                digest = hashlib.sha256()
                head = bytearray()
                tail = bytearray()
                payload = bytearray()
                total = 0
                while True:
                    chunk = os.read(fd, 64 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                    if len(head) < _FINAL_OUTPUT_EXCERPT_BYTES:
                        head.extend(
                            chunk[:_FINAL_OUTPUT_EXCERPT_BYTES - len(head)]
                        )
                    tail.extend(chunk)
                    if len(tail) > _FINAL_OUTPUT_EXCERPT_BYTES:
                        del tail[:-_FINAL_OUTPUT_EXCERPT_BYTES]
                    if len(payload) <= max_output_bytes:
                        payload.extend(chunk)
                        if len(payload) > max_output_bytes:
                            payload.clear()
                after = os.fstat(fd)
            finally:
                os.close(fd)
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            diagnostic: dict[str, object] = {
                "reason": "",
                "size_bytes": total,
                "max_bytes": max_output_bytes,
                "sha256": digest.hexdigest(),
                "telemetry_suffix_matched": _known_telemetry_suffix_matched(
                    bytes(tail)
                ),
                "excerpt_limit_bytes": _FINAL_OUTPUT_EXCERPT_BYTES,
                "head_redacted": _redacted_output_excerpt(bytes(head)),
                "tail_redacted": _redacted_output_excerpt(bytes(tail)),
            }
            if identity != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ) or total != info.st_size:
                diagnostic["reason"] = "unstable"
                return None, None, diagnostic
            if total == 0:
                diagnostic["reason"] = "empty"
                return None, None, diagnostic
            if total > max_output_bytes:
                diagnostic["reason"] = "oversize"
                return None, None, diagnostic
            return identity, bytes(payload), diagnostic

        def read_authoritative_final() -> tuple[_T, bytes] | object:
            nonlocal final_failure_diagnostic
            observed_identity, _observed_raw, observed_diagnostic = final_snapshot()
            if observed_identity is None:
                final_failure_diagnostic = observed_diagnostic
                if observed_diagnostic["reason"] in {"empty", "missing"}:
                    return missing
                return invalid

            first_identity, first_raw, first_diagnostic = final_snapshot()
            if first_identity is None or first_raw is None:
                first_diagnostic["reason"] = "unstable"
                final_failure_diagnostic = first_diagnostic
                return invalid
            time.sleep(0.1)
            second_identity, second_raw, second_diagnostic = final_snapshot()
            if (
                second_identity is None
                or second_raw is None
                or first_identity != second_identity
                or first_raw != second_raw
            ):
                second_diagnostic["reason"] = "unstable"
                final_failure_diagnostic = second_diagnostic
                return invalid
            try:
                candidate = _extract_codex_json_output(first_raw)
            except ValueError:
                first_diagnostic["reason"] = "transport_invalid"
                final_failure_diagnostic = first_diagnostic
                return invalid
            try:
                value = validate_output(candidate)
            except (TypeError, ValueError, json.JSONDecodeError) as validation_error:
                first_diagnostic["reason"] = "schema_invalid"
                first_diagnostic["validator_error_type"] = (
                    type(validation_error).__name__
                )
                final_failure_diagnostic = first_diagnostic
                return invalid
            if value is None:
                raise CodexError("isolated output validator returned None")
            final_failure_diagnostic = None
            return value, candidate

        try:
            with self._sem, tempfile.TemporaryFile(mode="w+b") as stderr_file:
                try:
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_file,
                        env=_scrubbed_env(),
                        start_new_session=True,
                    )
                except FileNotFoundError:
                    raise CodexError("codex executable not found on PATH") from None
                deadline = time.monotonic() + self._timeout_s
                process_group_stopped = False
                try:
                    while True:
                        returncode = proc.poll()
                        if returncode is not None:
                            stderr_file.seek(0, os.SEEK_END)
                            size = stderr_file.tell()
                            stderr_file.seek(max(0, size - 8192))
                            stderr = stderr_file.read().decode("utf-8", errors="replace")
                            if returncode != 0:
                                raise CodexError(
                                    f"codex exit {returncode}: {clean_stderr(stderr)}",
                                    retryable=True,
                                )
                            # Codex often follows its general file-editing
                            # contract and writes the declared file in place.
                            # That is not an early completion signal because a
                            # live descendant could still be writing.  Once the
                            # leader exits cleanly, stop the invocation's whole
                            # process group, then let the backend validate and
                            # adopt the stable bytes.
                            _terminate_process_group(proc)
                            process_group_stopped = True
                            adopted = read_authoritative_final()
                            if final_failure_diagnostic is not None:
                                final_failure_diagnostic["returncode"] = proc.returncode
                            if adopted is invalid:
                                raise CodexError(
                                    "codex exited without publishing valid output: "
                                    f"{clean_stderr(stderr)}",
                                    retryable=True,
                                )
                            if adopted is not missing:
                                return publish(adopted)
                            adopted = read_published(allow_initial_inode=True)
                            if adopted is not missing:
                                return publish(adopted)
                            raise CodexError(
                                "codex exited without publishing valid output: "
                                f"{clean_stderr(stderr)}",
                                retryable=True,
                            )
                        if time.monotonic() >= deadline:
                            raise CodexError(
                                f"codex timed out after {self._timeout_s}s", retryable=True
                            )
                        adopted = read_published()
                        if adopted is not missing:
                            if proc.poll() is not None:
                                continue
                            # Stop this invocation before backend publication,
                            # then re-read both transports.  This closes the
                            # post-validation write race while preserving the
                            # declared file's early-completion contract.
                            _terminate_process_group(proc)
                            process_group_stopped = True
                            final_adopted = read_authoritative_final()
                            if final_failure_diagnostic is not None:
                                final_failure_diagnostic["returncode"] = proc.returncode
                            if final_adopted is invalid:
                                raise CodexError(
                                    "codex exited without publishing valid output",
                                    retryable=True,
                                )
                            if final_adopted is not missing:
                                return publish(final_adopted)
                            adopted = read_published(allow_initial_inode=True)
                            if adopted is not missing:
                                return publish(adopted)
                            raise CodexError(
                                "codex exited without publishing valid output",
                                retryable=True,
                            )
                        time.sleep(0.1)
                finally:
                    if not process_group_stopped:
                        _terminate_process_group(proc)
        except BaseException as exc:
            record_failure(exc)
            raise
        finally:
            os.close(final_output_fd)
            os.close(placeholder_fd)
            os.close(parent_fd)

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

                run_error: CodexError | None = None
                try:
                    self.run_isolated(stage, prompt, session_dir=session_dir)
                except CodexError as error:
                    run_error = error

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
