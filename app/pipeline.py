"""处理流水线：queued → processing → done|failed。

步骤：extract_keyframes --fps 4 抽帧 + 分页联系表 → codex 沙箱按 SKILL.md 选帧/写 prompt →
后端白名单校验（不信任 agent 输出）→ meta 落盘。不再生成 preview.mp4（生成位留空）。
codex 运行前把 skill 的 scripts/ 拷进会话目录（裁剪工具按 scripts/crop_image.py 相对引用）。
流水线复用 skills/video-maker 的脚本，不重造。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from app import storage
from app.codex_runner import clean_stderr
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "video-maker"
SCRIPTS_DIR = SKILL_DIR / "scripts"
EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_keyframes.py"
SKILL_MD = SKILL_DIR / "SKILL.md"

MAX_PROMPT_BYTES = 32 * 1024


class PipelineError(RuntimeError):
    """流水线单步失败（HTTP 层不感知，只进 meta.error）。"""


def _run_cmd(argv: list[str], *, timeout: int, step: str, cwd: Path | None = None) -> None:
    """argv 列表子进程；超时/找不到可执行/非零退出 → PipelineError（stderr 已清洗）。"""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise PipelineError(f"{step} timed out after {timeout}s") from None
    except FileNotFoundError as e:
        raise PipelineError(f"{step} executable not found: {e.filename}") from None
    if proc.returncode != 0:
        raise PipelineError(f"{step} exit {proc.returncode}: {clean_stderr(proc.stderr)}")


def validate_work_dir(work: Path) -> tuple[list[str], str]:
    """产物白名单校验；返回 (关键帧文件名列表, prompt 文本)。任一不过 → PipelineError。"""
    frames = (
        sorted(p.name for p in (work / "keyframes").glob("*.png"))
        if (work / "keyframes").is_dir()
        else []
    )
    if not 1 <= len(frames) <= 9:
        raise PipelineError(f"keyframe count {len(frames)} not in 1..9")

    prompt_path = work / "prompt.txt"
    if not prompt_path.is_file():
        raise PipelineError("prompt.txt missing")
    raw = prompt_path.read_bytes()
    if not raw.strip():
        raise PipelineError("prompt.txt empty")
    if len(raw) > MAX_PROMPT_BYTES:
        raise PipelineError(f"prompt.txt exceeds {MAX_PROMPT_BYTES} bytes")
    prompt = raw.decode("utf-8", errors="replace")
    return frames, prompt


def _codex_prompt(cdir: Path) -> str:
    return f"""按技能文档执行（只读）：{SKILL_MD}。输入在 work/。

硬性禁令：
- 运行 Python 脚本一律用 {sys.executable}（系统 python3 缺 cv2）。
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。
"""


def run(settings: Settings, cid: str, runner) -> None:
    """后台任务入口；任何步骤失败 → status=failed + error，不抛异常。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    try:
        if storage.update_meta(settings.data_dir, cid, status="processing", error=None) is None:
            return
        sources = sorted(cdir.glob("source.*"))
        if not sources:
            raise PipelineError("source video missing")
        source = sources[0]
        _run_cmd(
            [
                sys.executable, str(EXTRACT_SCRIPT), str(source),
                "--out-dir", str(work),
                "--fps", "4",
            ],
            timeout=120,
            step="extract",
        )
        # skill 的裁剪工具以 scripts/crop_image.py 相对工作目录引用，codex 的 cwd 是 cdir
        shutil.copytree(SCRIPTS_DIR, cdir / "scripts")
        runner.run(cdir, _codex_prompt(cdir))
        keyframes, prompt = validate_work_dir(work)
        storage.update_meta(
            settings.data_dir, cid, status="done", keyframes=keyframes, prompt=prompt
        )
    except Exception as e:
        storage.update_meta(settings.data_dir, cid, status="failed", error=str(e)[:500])
