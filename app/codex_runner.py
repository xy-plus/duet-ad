"""沙箱化 codex exec 调用：workspace-write、断网、环境清洗、硬超时、并发信号量。

安全约束（逐条对应任务 B 要求，均经 codex-cli 0.147.0 实证）：
- 永远 argv 列表调起，永不 shell=True，永不使用 --dangerously-bypass-*；
- agent shell 断网：sandbox_workspace_write.network_access=false（实证 curl 不通）；
- 环境清洗双保险：
  a) 宿主进程级：调起 codex 前剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的环境变量
     （实证：0.147.0 的 shell 命令经 code-mode-host 执行，shell_environment_policy
     的 inherit/exclude 不能阻止宿主秘密泄漏进 agent shell，必须在本进程侧清洗）；
  b) codex 配置级：inherit="core" + exclude 兜底（inherit="none" 实证会让沙箱
     启动器找不到 bwrap，不可用）；
- 硬超时 settings.codex_timeout_s；并发信号量 settings.codex_concurrency；
- 超时/非零退出 → CodexError，stderr 先剔除环境变量行再截断 ≤500 字。
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path

_SANDBOX_CONFIGS = [
    "sandbox_workspace_write.network_access=false",
    'shell_environment_policy.inherit="core"',
    'shell_environment_policy.exclude=["*KEY*","*TOKEN*","*SECRET*","*PASSWORD*"]',
]

_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)


class CodexError(RuntimeError):
    """codex 启动/运行失败（超时、非零退出、找不到二进制）。"""


def clean_stderr(text: str | None, limit: int = 500) -> str:
    """剔除环境变量行（KEY=VALUE），截断到 limit 字。"""
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not _ENV_LINE_RE.match(l.strip())]
    return "\n".join(lines).strip()[-limit:]


def _scrubbed_env() -> dict[str, str]:
    """剔除名字含 KEY/TOKEN/SECRET/PASSWORD 的变量；PATH/HOME/代理等保留。"""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV_RE.search(k)}


class CodexRunner:
    def __init__(self, timeout_s: int, concurrency: int) -> None:
        self._timeout_s = timeout_s
        self._sem = threading.Semaphore(concurrency)

    def build_argv(self, workdir: Path, prompt: str) -> list[str]:
        argv = [
            "codex", "exec",
            "-C", str(workdir),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "-o", str(workdir / "codex_last_message.txt"),
        ]
        for cfg in _SANDBOX_CONFIGS:
            argv += ["-c", cfg]
        argv.append(prompt)
        return argv

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
