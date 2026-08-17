"""处理流水线：queued → processing → done|failed。

步骤：extract_keyframes --fps 4 抽帧 + 分页联系表 → （voice_mode ≠ none 时）抽音轨 +
codex 听写台词（voice_lines.json 白名单校验，落 meta.voice_lines）→ codex 沙箱按 SKILL.md
选帧/写 prompt → 后端白名单校验（不信任 agent 输出）→ meta 落盘。不再生成 preview.mp4（生成位留空）。
codex 超时/非零退出时先校验已落盘产物，完整则收养，不完整才判失败。
codex 运行前把 skill 的 scripts/ 拷进会话目录（裁剪工具按 scripts/crop_image.py 相对引用）。
流水线复用 skills/video-maker 的脚本，不重造。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

from app import storage, voice
from app.codex_runner import CodexError, clean_stderr
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
    for name in frames:
        if cv2.imread(str(work / "keyframes" / name)) is None:
            raise PipelineError(f"keyframe undecodable: {name}")

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
    return f"""按技能文档执行：{SKILL_MD}（该文档只读，禁止修改；「只读」指文档本身，不是执行模式）。输入在 work/，产物（keyframes/ 与 prompt.txt）必须按文档写入 work/。

硬性禁令：
- 运行 Python 脚本一律用 {sys.executable}（系统 python3 缺 cv2）。
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。
"""


def _voice_prompt(cdir: Path, voice_mode: str, target_language: str, duration_s: float) -> str:
    """口播步 codex prompt：听写 + 模式处理 + 硬性禁令（同 _codex_prompt）。"""
    if voice_mode == "keep":
        rule = "原文保持：只修正错别字与标点，不改写措辞。"
    elif voice_mode == "rewrite":
        rule = "洗稿：把台词改写得更自然；句数不变、句序不变、每句时间边界不变。"
    else:  # translate
        rule = f"翻译成{target_language}：句对句对齐，每句时间边界不变。"
    return f"""听写并处理视频台词。输入：work/voice.mp3（源视频音轨，16kHz 单声道）与 work/manifest.json（源视频元信息，供参考）。音频时长约 {duration_s:.3f} 秒。

任务：
- 听写音频中的人声台词，按句切分；
- 每句标出起止时间（秒，从音频开头起算）；
- {rule}
- 输出 work/voice_lines.json（UTF-8）：JSON 数组 [{{"text": "...", "start_s": 0.5, "end_s": 2.1}}]，0 ≤ start_s < end_s ≤ 音频时长，按 start_s 升序，覆盖人声区间；不写其他文件。

硬性禁令：
- 运行 Python 脚本一律用 {sys.executable}（系统 python3 缺 cv2）。
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。
"""


def _load_voice_lines(work: Path, duration_s: float) -> list[dict]:
    """读并校验 work/voice_lines.json；缺失/非法 → PipelineError。"""
    try:
        raw = (work / "voice_lines.json").read_bytes()
    except OSError:
        raise PipelineError("voice_lines.json missing") from None
    return voice.validate_voice_lines(raw, duration_s)


def _voice_step(
    settings: Settings, cid: str, cdir: Path, work: Path, runner,
    voice_mode: str, target_language: str,
) -> None:
    """口播步（抽帧后）：抽音轨 → codex 听写 → 白名单校验 → voice_lines 落 meta。

    时长约束用源视频时长，取自抽帧步产出的 manifest.json。失败 → PipelineError 走现有
    meta failed 落盘链路。
    """
    if voice_mode not in ("keep", "rewrite", "translate"):
        raise PipelineError(f"unknown voice_mode: {voice_mode}")
    if voice_mode == "translate" and not target_language:
        raise PipelineError("voice_mode=translate requires target_language")
    if voice.extract_audio(cdir) is None:
        raise PipelineError("source video has no audio track")
    try:
        manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PipelineError("manifest.json missing or invalid") from None
    try:
        duration_s = float(manifest["duration_seconds"])
    except (KeyError, TypeError, ValueError):
        raise PipelineError("manifest.json missing or invalid") from None
    if duration_s <= 0:
        raise PipelineError(f"manifest.json invalid duration: {duration_s}")
    try:
        runner.run(cdir, _voice_prompt(cdir, voice_mode, target_language, duration_s))
    except CodexError as e:
        # 超时被杀时产物可能已完整落盘：校验通过则收养，否则报原始错误
        try:
            _load_voice_lines(work, duration_s)
        except PipelineError:
            raise e from None
    lines = _load_voice_lines(work, duration_s)
    storage.update_meta(settings.data_dir, cid, voice_lines=lines)


def run(settings: Settings, cid: str, runner) -> None:
    """后台任务入口；任何步骤失败 → status=failed + error，不抛异常。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    try:
        meta = storage.update_meta(settings.data_dir, cid, status="processing", error=None)
        if meta is None:
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
        # 口播步放在抽帧之后：ASR 的输入含 work/manifest.json（抽帧步产物）
        voice_mode = meta.get("voice_mode", "none")
        if voice_mode != "none":
            _voice_step(
                settings, cid, cdir, work, runner, voice_mode,
                meta.get("target_language") or "",
            )
        # skill 的裁剪工具以 scripts/crop_image.py 相对工作目录引用，codex 的 cwd 是 cdir
        shutil.copytree(SCRIPTS_DIR, cdir / "scripts")
        try:
            runner.run(cdir, _codex_prompt(cdir))
        except CodexError as e:
            # 超时被杀时产物可能已完整落盘：校验通过则收养，否则报原始错误
            try:
                validate_work_dir(work)
            except PipelineError:
                raise e from None
        keyframes, prompt = validate_work_dir(work)
        storage.update_meta(
            settings.data_dir, cid, status="done", keyframes=keyframes, prompt=prompt
        )
    except Exception as e:
        storage.update_meta(settings.data_dir, cid, status="failed", error=str(e)[:500])
