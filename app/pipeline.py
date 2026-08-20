"""处理流水线：queued → processing → done|failed。

步骤：extract_keyframes --fps 4 抽帧 + 分页联系表 → （voice_mode ≠ none 时）抽音轨 +
codex 听写台词（voice_lines.json 白名单校验，落 meta.voice_lines）→ scenes.py 场景检测
（work/scenes.json）→ 按 segments 决定模式：
- 单段模式（segments 空）：codex 沙箱按 SKILL.md 选帧/写 prompt → 后端白名单校验 →
  meta 落盘（work/keyframes + work/prompt.txt，保持视觉 prompt 原文）。
- 多段模式（segments 非空）：ffmpeg 按段边界切源视频（work/segments/N/source.mp4，
  N 从 1 起），每段独立走抽帧 → codex prompt（单段/多段共用） → 校验 → 后端机械操作在 prompt 开头加
  「不要生成背景音乐」行；段间
  ThreadPoolExecutor 并发（每段目录独立，CodexRunner 自带信号量兜底）；meta.voice_lines
  按 start_s 归段（[start_s, end_s) 口径，恰在边界归后段），
  每段写 work/segments/N/work/voice_lines.json；任一段失败 → 整体 failed（error 指明段号）；
  meta.segments 落各段产物，顶层 keyframes/prompt 保持空值。段 codex 的 cwd 即段目录
  （物理隔离，看不到段外内容）；段目录内嵌套 work/（帧/台词/产物落段 work/，SKILL.md
  的 work/ 路径逐字适用，段 prompt 与单段逐字相同）；scripts/ 逐段拷入段目录，
  scenes.json 不拷入（段 codex 不需要知道全片）。
scenes 检测失败或 scenes.json 非法（含拆段不变量违规）→ 回退单段模式（meta.scenes_note
留痕），不做拆段；越界台词不归段并计数落 meta.voice_lines_dropped（内部字段）；翻译模式
的目标语言由后端写进 prompt（codex 不从台词反推）。
codex 超时/非零退出时先校验已落盘产物，完整则收养，不完整才判失败。
codex 运行前把 skill 的 scripts/ 拷进 codex 工作目录（单段=会话目录、多段=段目录；裁剪工具按 scripts/crop_image.py 相对引用）。
流水线复用 skills/video-maker 的脚本，不重造。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from app import prepared_input, storage, vocal, voice
from app.codex_runner import CodexError, clean_stderr
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "video-maker"
SCRIPTS_DIR = SKILL_DIR / "scripts"
EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_keyframes.py"
SCENES_SCRIPT = ROOT / "app" / "scenes.py"
SKILL_MD = SKILL_DIR / "SKILL.md"

MAX_PROMPT_BYTES = 32 * 1024
SCENES_TIMEOUT_S = 300  # scenes.py 场景检测超时（长视频 PySceneDetect 较慢）
CUT_DURATION_TOLERANCE_S = 0.1  # 切段时长允许误差（秒）
NO_BGM_LINE = "不要生成背景音乐"  # 多段模式由后端机械加进 prompt 首行（不依赖 codex 写）
_SEG_TAIL_EPS_S = 0.01  # 台词 start_s 超出末段终点 ≤0.01s（与 voice 校验容差同口径）→ 归末段
# YAMNet 量化步长为 1/256；0.2 相邻下档 51/256 是线下真实 sung 样本。
# 该阈值只决定空听写是否重试，不改变逐句 classify_segment 规则。
EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN = 51 / 256
EMPTY_TRANSCRIPT_WARNING = (
    "voice_lines.json empty after one retry despite vocal evidence; "
    "continuing without dialogue"
)
H3_DEFAULT_RATIO = "9:16"
H3_DEFAULT_FIT_MODE = "none"
H3_ENGINE_WORKFLOW = "minimax_h3_lightx2v_v5"
# 拆段不变量（与 app/scenes.py 的算法级不变量同值；不 import scenes：scenedetect 缺依赖时
# scenes 模块会 SystemExit，流水线不能因此加载失败）
SEGMENT_MIN_S = 4.0
SEGMENT_MAX_S = 15.0


class PipelineError(RuntimeError):
    """流水线单步失败（HTTP 层不感知，只进 meta.error）。"""


def _codex_output_error(stage: str) -> PipelineError:
    """把 Codex 成功退出后的产物问题归到正确阶段，不暴露内部文件校验细节。"""
    details = {
        "voice": "required voice_lines artifact is missing or invalid",
        "visual": "required keyframes/prompt artifacts are missing or invalid",
    }
    detail = details[stage]
    return PipelineError(f"codex {stage} output invalid: {detail}")


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


def _hard_rules(cdir: Path) -> str:
    """codex prompt 硬性禁令四条（口播/选帧各步共用）。"""
    return f"""硬性禁令：
- 运行 Python 脚本一律用 {sys.executable}（系统 python3 缺 cv2）。
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。"""


def _language_note(target_language: str) -> str:
    """翻译模式：目标语言由后端注入 prompt（codex 不从台词反推语言）。"""
    return f"提示词与台词使用目标语言：{target_language}。" if target_language else ""


def _codex_prompt(
    cdir: Path, target_language: str = "", *, visual_only: bool = False
) -> str:
    parts = [
        f"按技能文档执行：{SKILL_MD}（该文档只读，禁止修改；「只读」指文档本身，不是执行模式）。输入在 work/，产物（keyframes/ 与 prompt.txt）必须按文档写入 work/。"
    ]
    if target_language:
        parts.append(_language_note(target_language))
    if visual_only:
        parts.append(
            "本次只生成视觉叙事 prompt：不要读取、生成、推断、改写或编排 voice_lines；"
            "画面文字、OCR、字幕和备注只能作为可见视觉元素，禁止写成角色发声。"
            "最终台词由后端从结构化来源机械合成。"
        )
    parts.append(_hard_rules(cdir))
    return "\n\n".join(parts) + "\n"


def _run_visual_codex(
    runner,
    cdir: Path,
    prompt: str,
    work: Path,
    *,
    isolate_dialogue: bool,
) -> None:
    """新 H3 契约下让视觉 Codex 看不到 voice_lines，并丢弃它越权创建的同名文件。"""
    voice_path = work / "voice_lines.json"
    frozen_voice = voice_path.read_bytes() if isolate_dialogue and voice_path.is_file() else None
    if isolate_dialogue:
        voice_path.unlink(missing_ok=True)
    try:
        runner.run(cdir, prompt)
    finally:
        if isolate_dialogue:
            voice_path.unlink(missing_ok=True)
            if frozen_voice is not None:
                voice_path.write_bytes(frozen_voice)


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

{_hard_rules(cdir)}
"""


def _load_voice_lines(work: Path, duration_s: float) -> list[dict]:
    """读并校验 work/voice_lines.json；缺失/非法 → PipelineError。"""
    try:
        raw = (work / "voice_lines.json").read_bytes()
    except OSError:
        raise PipelineError("voice_lines.json missing") from None
    return voice.validate_voice_lines(raw, duration_s)


def _vocal_filter_enabled() -> bool:
    """唯一环境开关解析点；未知值按启用处理，避免误拼写静默放宽过滤。"""
    return os.environ.get("VOCAL_FILTER", "on").strip().lower() not in {
        "0", "off", "false"
    }


def _has_retryable_vocal_evidence(analysis: vocal.VocalAnalysis) -> bool:
    """空听写重试边界：覆盖真实 51/256 sung，排除 0.059 纯 BGM。"""
    return any(
        window.spoken >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN
        or window.sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN
        for window in analysis.windows
    )


def _voice_step(
    settings: Settings, cid: str, cdir: Path, work: Path, runner,
    voice_mode: str, target_language: str, *, allow_no_audio: bool = False,
) -> list[dict]:
    """口播步（抽帧后）：抽音轨 → 声学预判 → codex 听写 → 白名单校验 → voice_lines 落 meta。

    台词时间戳在音频时间轴上（codex 听 voice.mp3），校验基准与提示词时长用音频实际时长
    （音频流可比容器长几十 ms，常态）；容器时长（manifest）仍供场景/拆段用。空台词数组
    且音轨有人声证据 → 重试一次 codex，再空则写 warning 后以无台词继续。新 auto 契约下
    无音轨同样是合法空台词；旧 voice_mode 可保留严格失败行为。返回白名单净化后的台词列表。
    """
    target_language = (target_language or "").strip()  # 纯空白串视为缺失，不生成「翻译成   」prompt
    if voice_mode not in ("keep", "rewrite", "translate"):
        raise PipelineError(f"unknown voice_mode: {voice_mode}")
    if voice_mode == "translate" and not target_language:
        raise PipelineError("voice_mode=translate requires target_language")
    filter_enabled = _vocal_filter_enabled()
    audio = voice.extract_audio(cdir)
    if audio is None:
        if not allow_no_audio:
            raise PipelineError("source video has no audio track")
        (work / "voice.mp3").unlink(missing_ok=True)
        (work / "voice_lines.json").write_text("[]\n", encoding="utf-8")
        storage.update_meta(
            settings.data_dir,
            cid,
            voice_lines=[],
            has_bgm=False,
            vocal_filter_enabled=filter_enabled,
            voice_line_provenance=[],
            voice_warnings=[],
            voice_has_retryable_vocal_evidence=False,
            voice_lines_vocal_dropped=0,
        )
        return []
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
    # 台词校验基准 = 音频实际时长；probe 失败回退容器时长（旧行为）
    audio_duration_s = voice.probe_audio_duration(audio) or duration_s
    # 声学分析提前到听写前：空台词数组时区分「音轨无口播（合法无台词）」与「codex 摆烂（重试兜底）」
    try:
        analysis = vocal.analyze(audio)
    except Exception as e:
        raise PipelineError(f"vocal classification unavailable: {e}") from None
    has_vocal = _has_retryable_vocal_evidence(analysis)
    try:
        runner.run(cdir, _voice_prompt(cdir, voice_mode, target_language, audio_duration_s))
    except CodexError as e:
        # 超时被杀时产物可能已完整落盘：校验通过则收养，否则报原始错误
        try:
            lines = _load_voice_lines(work, audio_duration_s)
        except PipelineError:
            raise e from None
    else:
        try:
            lines = _load_voice_lines(work, audio_duration_s)
        except PipelineError:
            raise _codex_output_error("voice") from None
    if not lines and has_vocal:
        # 音轨有人声但听写为空：只重试一次；再次为空按用户确认继续，但必须明确留痕。
        try:
            runner.run(cdir, _voice_prompt(cdir, voice_mode, target_language, audio_duration_s))
        except CodexError as e:
            try:
                lines = _load_voice_lines(work, audio_duration_s)
            except PipelineError:
                raise e from None
        else:
            try:
                lines = _load_voice_lines(work, audio_duration_s)
            except PipelineError:
                raise _codex_output_error("voice") from None
    warnings = [EMPTY_TRANSCRIPT_WARNING] if not lines and has_vocal else []
    # VOCAL_FILTER=off 只旁路 keep/drop，不旁路分类：receipt 必须解释每句为何被保留。
    filtered_lines = []
    decisions = []
    for line in lines:
        classification = vocal.classify_segment(
            int(line["start_s"] * 1000), int(line["end_s"] * 1000), analysis.windows
        )
        kept = not filter_enabled or classification in ("spoken", "sung")
        decisions.append(
            {
                **line,
                "classification": classification,
                "provenance": "asr",
                "kept": kept,
            }
        )
        if kept:
            filtered_lines.append(line)
    vocal_dropped = sum(not decision["kept"] for decision in decisions)
    # 后续视觉步骤看到的 voice_lines 也只能是最终有效集，不能重新收养已过滤行。
    (work / "voice_lines.json").write_text(
        json.dumps(filtered_lines, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changes = {
        "voice_lines": filtered_lines,
        "has_bgm": bool(analysis.has_bgm),
        "vocal_filter_enabled": filter_enabled,
        "voice_line_provenance": decisions,
        "voice_warnings": warnings,
        "voice_has_retryable_vocal_evidence": has_vocal,
        "voice_lines_vocal_dropped": vocal_dropped,
    }
    storage.update_meta(settings.data_dir, cid, **changes)
    return filtered_lines


def _load_scenes(work: Path) -> list[dict]:
    """读并校验 work/scenes.json；返回 segments（空 = 单段模式）。缺失/非法 → PipelineError。

    结构不变量（与 scenes.py 同口径）：每段 4~15s（1e-9 容差）、相邻无缝（1e-6 容差，
    隐含单调有序）、首段 0 起且覆盖 [0, duration_s]。
    """
    try:
        data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PipelineError("scenes.json missing or invalid") from None
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        raise PipelineError("scenes.json missing segments")
    out: list[dict] = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("index"), int):
            raise PipelineError(f"scenes.json segments[{i}] must be an object with int index")
        start_s, end_s = seg.get("start_s"), seg.get("end_s")
        for key, val in (("start_s", start_s), ("end_s", end_s)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise PipelineError(f"scenes.json segments[{i}].{key} must be a number")
        start_s, end_s = float(start_s), float(end_s)
        if not (math.isfinite(start_s) and math.isfinite(end_s) and start_s < end_s):
            raise PipelineError(f"scenes.json segments[{i}] invalid bounds: {start_s}..{end_s}")
        out.append({"index": seg["index"], "start_s": start_s, "end_s": end_s})
    if not out:
        return out  # 空 segments = 单段模式（合法），无需时长与覆盖校验
    duration = data.get("duration_s")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise PipelineError("scenes.json missing valid duration_s")
    prev_end = 0.0
    for i, seg in enumerate(out):
        length = seg["end_s"] - seg["start_s"]
        if length < SEGMENT_MIN_S - 1e-9 or length > SEGMENT_MAX_S + 1e-9:
            raise PipelineError(f"scenes.json segments[{i}] length {length:.3f}s not in 4..15s")
        if abs(seg["start_s"] - prev_end) > 1e-6:
            raise PipelineError(f"scenes.json segments[{i}] not contiguous with previous")
        prev_end = seg["end_s"]
    if abs(prev_end - float(duration)) > 1e-6:
        raise PipelineError("scenes.json segments do not cover [0, duration]")
    return out


def _detect_segments(settings: Settings, cid: str, source: Path, work: Path) -> list[dict]:
    """跑 app/scenes.py 检测场景并读拆段建议。

    检测失败或 scenes.json 非法（含拆段结构不变量违规）→ 回退空列表 = 单段模式，
    不判失败，meta.scenes_note 留痕；segments 空（≤20s）是合法单段结果，不留痕。
    """
    try:
        _run_cmd(
            [sys.executable, str(SCENES_SCRIPT), str(source), "--work-dir", str(work)],
            timeout=SCENES_TIMEOUT_S,
            step="scenes",
        )
    except PipelineError as e:
        print(f"scenes detection failed ({e}); falling back to single-segment mode")
        storage.update_meta(
            settings.data_dir, cid,
            scenes_note="scenes detection failed, single-segment fallback",
        )
        return []
    try:
        return _load_scenes(work)
    except PipelineError as e:
        print(f"scenes.json invalid ({e}); falling back to single-segment mode")
        storage.update_meta(
            settings.data_dir, cid,
            scenes_note="scenes.json invalid, single-segment fallback",
        )
        return []


def _probe_duration(path: Path) -> float:
    """ffprobe 探测视频时长（秒）；失败 → PipelineError。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise PipelineError("ffprobe timed out after 30s") from None
    except FileNotFoundError:
        raise PipelineError("ffprobe not found on PATH") from None
    if proc.returncode != 0:
        raise PipelineError(f"ffprobe exit {proc.returncode}: {clean_stderr(proc.stderr)}")
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError):
        raise PipelineError("cannot parse video duration") from None


def _cut_segment(source: Path, start_s: float, end_s: float, segdir: Path) -> None:
    """ffmpeg 按段边界切源视频（-ss 在 -i 前重编码）+ ffprobe 验证切出时长误差 <0.1s。"""
    segdir.mkdir(parents=True, exist_ok=True)
    out = segdir / "source.mp4"
    length = end_s - start_s
    _run_cmd(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(source),
         "-to", f"{length:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(out)],
        timeout=300,
        step="segment cut",
    )
    duration = _probe_duration(out)
    if abs(duration - length) > CUT_DURATION_TOLERANCE_S:
        raise PipelineError(
            f"segment cut duration {duration:.3f}s off target {length:.3f}s "
            f"beyond tolerance {CUT_DURATION_TOLERANCE_S}s"
        )


def attribute_lines(lines: list[dict], segments: list[dict]) -> dict[int, list[dict]]:
    """按台词 start_s 落入段 [start_s, end_s) 归段（恰在边界归后段）；返回 {index: [台词]}。

    start_s 超出末段终点但在 0.01s 浮点误差内（voice 校验同口径容差）→ 归末段；
    更远的越界台词不归任何段。
    """
    result: dict[int, list[dict]] = {seg["index"]: [] for seg in segments}
    if not segments:
        return result
    last_end = segments[-1]["end_s"]
    for line in lines:
        target: int | None = None
        for seg in segments:
            if seg["start_s"] <= line["start_s"] < seg["end_s"]:
                target = seg["index"]
                break
        if target is None and line["start_s"] <= last_end + _SEG_TAIL_EPS_S:
            target = segments[-1]["index"]
        if target is not None:
            result[target].append(line)
    return result


def _apply_no_bgm_prefix(prompt: str, prompt_path: Path, *, enabled: bool) -> str:
    """多段模式机械添加 BGM 禁令；单段保持原文。两者都复核最终长度。"""
    final_prompt = f"{NO_BGM_LINE}\n{prompt}" if enabled else prompt
    if len(final_prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise PipelineError(f"prompt.txt exceeds {MAX_PROMPT_BYTES} bytes after prefix")
    prompt_path.write_text(final_prompt, encoding="utf-8")
    return final_prompt


def _keyframes_require_fit(work: Path, names: list[str]) -> bool:
    """按 H3 实际接收的关键帧判断是否需要 9:16 适配。"""
    for name in names:
        image = cv2.imread(str(work / "keyframes" / name))
        if image is None:
            raise PipelineError(f"keyframe cannot be decoded: {name}")
        height, width = image.shape[:2]
        if width * 16 != height * 9:
            return True
    return False


def _process_segment(work: Path, source: Path, seg: dict, runner,
                     lines: list[dict] | None, target_language: str = "") -> dict:
    """单段完整流程：切段 → 抽帧 → 写该段台词 → codex（cwd=段目录）→ 校验 → 后端加前缀。

    段目录内嵌套 work/：帧/台词/产物都在 segdir/work/，SKILL.md 的 work/ 路径逐字适用，
    段 prompt 与单段逐字相同（_codex_prompt）；codex 的 cwd 即段目录（物理隔离，看不到
    段外内容）；scripts/ 拷入段目录（裁剪工具按相对路径引用），scenes.json 不拷入。
    任一失败包装为 PipelineError 并指明段号；返回 meta.segments 条目。
    """
    index = seg["index"]
    segdir = work / "segments" / str(index)
    segwork = segdir / "work"
    try:
        _cut_segment(source, seg["start_s"], seg["end_s"], segdir)
        segwork.mkdir(parents=True, exist_ok=True)
        _run_cmd(
            [sys.executable, str(EXTRACT_SCRIPT), str(segdir / "source.mp4"),
             "--out-dir", str(segwork), "--fps", "4"],
            timeout=120,
            step=f"segment {index} extract",
        )
        # 该段台词（白名单净化后；lines 为 None = 无口播，不写文件）
        if lines is not None:
            (segwork / "voice_lines.json").write_text(
                json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # 裁剪工具按 scripts/crop_image.py 相对 cwd 引用：scripts/ 拷入段目录
        shutil.copytree(SCRIPTS_DIR, segdir / "scripts")
        try:
            runner.run(segdir, _codex_prompt(segdir, target_language))
        except CodexError as e:
            # 超时被杀时产物可能已完整落盘：校验通过则收养，否则报原始错误
            try:
                keyframes, prompt = validate_work_dir(segwork)
            except PipelineError:
                raise e from None
        else:
            try:
                keyframes, prompt = validate_work_dir(segwork)
            except PipelineError:
                raise _codex_output_error("visual") from None
        prompt = _apply_no_bgm_prefix(prompt, segwork / "prompt.txt", enabled=True)
        return {
            "index": index,
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "keyframes": keyframes,
            "prompt": prompt,
            "lines": [line["text"] for line in (lines or [])],
        }
    except Exception as e:
        raise PipelineError(f"segment {index} failed: {e}") from None


def _dialogue_for_prepared_input(
    meta: dict,
    mode: str,
    voice_lines: list[dict],
) -> tuple[dict, ...]:
    """把 meta 中的当前有效台词转换为 prepared-input 的显式来源契约。"""
    duration_s, _engine_duration = _prepared_durations(meta)
    filter_enabled = meta.get("vocal_filter_enabled", _vocal_filter_enabled())
    if not isinstance(filter_enabled, bool):
        raise PipelineError("vocal_filter_enabled in meta must be bool")
    if mode == "auto":
        automatic = [
            {
                "text": item["text"],
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "classification": item["classification"],
                "provenance": "asr",
            }
            for item in meta.get("voice_line_provenance", [])
            if item.get("kept") is True
        ]
        if len(automatic) != len(voice_lines):
            raise PipelineError("automatic dialogue provenance does not match effective lines")
        return prepared_input.prepare_dialogue(
            "auto",
            duration_s,
            automatic_lines=automatic,
            vocal_filter_enabled=filter_enabled,
        )
    if mode in ("edit", "custom"):
        return prepared_input.prepare_dialogue(
            mode,
            duration_s,
            supplied_lines=voice_lines,
            vocal_filter_enabled=filter_enabled,
        )
    if mode == "none":
        return prepared_input.prepare_dialogue("none", duration_s)
    raise PipelineError(f"unknown dialogue_mode: {mode}")


def _prepared_durations(meta: dict) -> tuple[float, int]:
    """返回 receipt 的实际时长与 H3 整秒请求时长（向上取整，范围 1..15）。"""
    raw = meta.get("duration_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PipelineError("duration_s in meta must be a positive finite number")
    actual = float(raw)
    if not math.isfinite(actual) or not 0 < actual <= 15:
        raise PipelineError("duration_s in meta must be within (0, 15]")
    engine_duration = math.ceil(actual)
    if not 1 <= engine_duration <= 15:
        raise PipelineError("engine duration must be within 1..15")
    return actual, engine_duration


def _write_prepared_receipt(
    settings: Settings,
    cid: str,
    cdir: Path,
    work: Path,
    source: Path,
    keyframes: list[str],
    visual_prompt: str,
    dialogue_mode: str,
    voice_lines: list[dict],
) -> str:
    """冻结单段 H3 输入并把 receipt 位置写入 meta；不触发任何远程请求。"""
    current = storage.load_meta(settings.data_dir, cid)
    if current is None:
        raise PipelineError("conversation disappeared while preparing H3 input")
    dialogue = _dialogue_for_prepared_input(current, dialogue_mode, voice_lines)
    duration_s, engine_duration = _prepared_durations(current)
    visual_path = work / "visual_prompt.txt"
    visual_path.write_text(visual_prompt, encoding="utf-8")
    ratio = current.get("ratio") or H3_DEFAULT_RATIO
    fit_mode = current.get("fit_mode") or H3_DEFAULT_FIT_MODE
    filter_enabled = current.get("vocal_filter_enabled", _vocal_filter_enabled())
    if not isinstance(filter_enabled, bool):
        raise PipelineError("vocal_filter_enabled in meta must be bool")
    engine_request = {
        "context_ir": {
            "model": "MiniMax-H3",
            "duration": engine_duration,
            "ratio": ratio,
        },
        "h3": {
            "workflow": H3_ENGINE_WORKFLOW,
            "duration": engine_duration,
            "resolution": "768p竖",
        },
    }
    try:
        frozen = prepared_input.write_prepared_input(
            root=cdir,
            source=source,
            audio=(work / "voice.mp3") if (work / "voice.mp3").is_file() else None,
            keyframes=[work / "keyframes" / name for name in keyframes],
            visual=visual_path,
            final=work / "prompt.txt",
            dialogue_mode=dialogue_mode,
            dialogue=dialogue,
            vocal_filter_enabled=filter_enabled,
            duration_s=duration_s,
            ratio=ratio,
            fit_mode=fit_mode,
            engine_request=engine_request,
        )
    except prepared_input.PreparedInputError as exc:
        raise PipelineError(f"prepared input invalid: {exc}") from None
    final_prompt = frozen.final_prompt.data.decode("utf-8")
    storage.update_meta(
        settings.data_dir,
        cid,
        prompt=final_prompt,
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
    )
    return final_prompt


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
        # 新 H3 会话以 dialogue_mode 为唯一产品契约；voice_mode 仅供旧会话内部兼容。
        # duration_s 是上传探测后才写入的新输入契约完成标记；仅有默认字段但尚未探测，
        # 或历史会话没有实际时长时，仍走旧流水线且不伪造 prepared receipt。
        new_input_contract = "dialogue_mode" in meta and meta.get("duration_s") is not None
        dialogue_mode = meta.get("dialogue_mode") if new_input_contract else None
        voice_mode = meta.get("voice_mode", "none")
        voice_lines: list[dict] | None = None
        if new_input_contract:
            if dialogue_mode != "auto":
                raise PipelineError(f"unknown initial dialogue_mode: {dialogue_mode}")
            auto_voice_mode = meta.get("voice_mode")
            if auto_voice_mode in (None, ""):
                auto_voice_mode = "keep"
            if auto_voice_mode not in ("keep", "rewrite", "translate"):
                raise PipelineError(f"unknown voice_mode for auto dialogue: {auto_voice_mode}")
            voice_lines = _voice_step(
                settings,
                cid,
                cdir,
                work,
                runner,
                auto_voice_mode,
                meta.get("target_language") or "",
                allow_no_audio=True,
            )
        elif voice_mode != "none":
            voice_lines = _voice_step(
                settings, cid, cdir, work, runner, voice_mode,
                meta.get("target_language") or "",
            )
        translate_lang = ""
        if not new_input_contract and voice_mode == "translate":
            translate_lang = (meta.get("target_language") or "").strip()
        segments = _detect_segments(settings, cid, source, work)
        if new_input_contract and segments:
            raise PipelineError("prepared input currently requires a single segment")
        if not segments:
            # 单段模式：work/keyframes + work/prompt.txt，不注入 workaround 前缀；
            # 裁剪工具以 scripts/crop_image.py 相对工作目录引用，scripts/ 拷进会话目录
            shutil.copytree(SCRIPTS_DIR, cdir / "scripts")
            try:
                _run_visual_codex(
                    runner,
                    cdir,
                    _codex_prompt(
                        cdir, translate_lang, visual_only=new_input_contract
                    ),
                    work,
                    isolate_dialogue=new_input_contract,
                )
            except CodexError as e:
                # 超时被杀时产物可能已完整落盘：校验通过则收养，否则报原始错误
                try:
                    keyframes, prompt = validate_work_dir(work)
                except PipelineError:
                    raise e from None
            else:
                try:
                    keyframes, prompt = validate_work_dir(work)
                except PipelineError:
                    raise _codex_output_error("visual") from None
            prompt = _apply_no_bgm_prefix(prompt, work / "prompt.txt", enabled=False)
            fit_required = _keyframes_require_fit(work, keyframes)
            if new_input_contract:
                prompt = _write_prepared_receipt(
                    settings,
                    cid,
                    cdir,
                    work,
                    source,
                    keyframes,
                    prompt,
                    dialogue_mode,
                    voice_lines or [],
                )
            storage.update_meta(
                settings.data_dir,
                cid,
                status="done",
                keyframes=keyframes,
                prompt=prompt,
                fit_required=fit_required,
            )
        else:
            # 多段模式：各段目录独立、无共享状态，线程池并发；CodexRunner.run 自带信号量兜底
            # lines_by_seg 为 None = 无口播：段目录不写 voice_lines.json
            lines_by_seg = (
                attribute_lines(voice_lines, segments) if voice_lines is not None else None
            )
            if lines_by_seg is not None:
                # 越界台词不归段：计数留痕（meta 内部字段，不静默丢失）
                dropped = len(voice_lines) - sum(len(v) for v in lines_by_seg.values())
                if dropped:
                    storage.update_meta(settings.data_dir, cid, voice_lines_dropped=dropped)
            # 段并发上限 = codex_concurrency 的一半：一条长视频不得占满全部 codex 槽饿死其他会话
            workers = min(len(segments), max(1, settings.codex_concurrency // 2))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                seg_metas = list(
                    pool.map(
                        lambda seg: _process_segment(
                            work, source, seg, runner,
                            lines_by_seg.get(seg["index"]) if lines_by_seg is not None else None,
                            translate_lang,
                        ),
                        segments,
                    )
                )
            # 顶层 keyframes/prompt 保持空值（各段产物在 segments 列表里，不重复写）
            storage.update_meta(settings.data_dir, cid, status="done", segments=seg_metas)
    except Exception as e:
        storage.update_meta(settings.data_dir, cid, status="failed", error=str(e)[:500])
