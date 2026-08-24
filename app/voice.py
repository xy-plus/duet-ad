"""口播链路纯函数：extract_audio 抽音轨、validate_voice_lines 台词白名单校验。

听写由 app.asr 或隔离 Codex 调用方完成。PipelineError 归口 pipeline.py；
pipeline 顶层导入本模块，为避免循环导入，异常类在函数内延迟导入。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from app import storage
from app.codex_runner import clean_stderr

_FFMPEG_TIMEOUT_S = 120
_EPS_S = 0.01  # 台词时间边界允许的浮点误差（秒）
MAX_VOICE_LINES_BYTES = 32 * 1024  # voice_lines.json 大小上限（与 MAX_PROMPT_BYTES 同级）
MAX_VOICE_TEXT_CHARS = 500  # 每行台词长度上限（strip 后）
MAX_VOICE_LINES_ITEMS = 200  # 台词条数上限（300s 视频按每句 1.5s 计 200 句，留裕量）
_UNRECOGNIZED_ASR_RE = re.compile(
    r"(?:[\[\(\{（【]\s*)?(?:无法(?:辨识|识别|听清)|听不清|"
    r"inaudible|unintelligible|unrecognized(?:\s+speech)?|"
    r"no\s+(?:speech|audio)|blank[_\s-]?audio|silence|music)"
    r"\s*[.!。]?\s*(?:[\]\)\}）】]|$)",
    re.IGNORECASE,
)


def is_unrecognized_text(text: str) -> bool:
    """整句 ASR 失败哨兵不是源素材台词，不能进入后续 prompt。"""
    return isinstance(text, str) and _UNRECOGNIZED_ASR_RE.search(text.strip()) is not None


def probe_audio_duration(path: Path) -> float | None:
    """ffprobe 音频文件实际时长（秒）；非音频/损坏 → None（不抛）。

    音频流可比容器长几十 ms（音轨尾部余量，常态）——台词时间戳在音频时间轴上，
    校验基准必须用音频时长而非容器时长。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        d = float(json.loads(r.stdout or "{}").get("format", {}).get("duration", ""))
    except (OSError, TypeError, ValueError, subprocess.TimeoutExpired):
        return None
    return d if d > 0 else None


def extract_audio(cdir: Path) -> Path | None:
    """从 source.* 抽音轨为 work/voice.mp3；无音轨 → None，ffmpeg 缺失/失败 → PipelineError。"""
    from app.pipeline import PipelineError  # 循环导入：pipeline 顶层导入本模块

    sources = sorted(cdir.glob("source.*"))
    if not sources:
        raise PipelineError("source video missing")
    if not shutil.which("ffmpeg"):
        raise PipelineError("ffmpeg not found on PATH")
    source = sources[0]
    # 音轨探测复用 storage.probe_audio（上传校验同源）：无音轨 → None；
    # 探测失败（UploadError）不拦截，交给 ffmpeg 裁决（与 skill 侧 probe_audio 三态语义一致）。
    try:
        if storage.probe_audio(source) is False:
            return None
    except storage.UploadError:
        pass
    work = cdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "voice.mp3"
    try:
        visual_duration = storage.probe_video(source).duration_s
        offset = (
            storage.probe_stream_first_pts(source, "a:0")
            - storage.probe_stream_first_pts(source, "v:0")
        )
    except storage.UploadError as exc:
        raise PipelineError(f"audio timeline probe failed: {exc}") from None
    trim_start = max(0.0, -offset)
    delay_ms = max(0, round(max(0.0, offset) * 1000))
    audio_filter = (
        f"atrim=start={trim_start:.9f},asetpts=PTS-STARTPTS,"
        f"adelay=delays={delay_ms}:all=1,apad,"
        f"atrim=duration={visual_duration:.9f}"
    )
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-filter:a", audio_filter, "-b:a", "64k", "-y", str(out)],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise PipelineError(f"audio extract timed out after {_FFMPEG_TIMEOUT_S}s") from None
    except FileNotFoundError:
        raise PipelineError("ffmpeg not found on PATH") from None
    if proc.returncode != 0:
        raise PipelineError(f"audio extract exit {proc.returncode}: {clean_stderr(proc.stderr)}")
    return out


def validate_voice_lines(raw: bytes, duration_s: float) -> list[dict]:
    """voice_lines.json 白名单校验（不信任 agent 输出）；返回台词列表，任一不过 → PipelineError。

    校验：raw ≤ 32KB（MAX_VOICE_LINES_BYTES）、UTF-8 可解、JSON 解析为 list（空数组合法 = 无台词）、
    条目 ≤ 200（MAX_VOICE_LINES_ITEMS）、每项含非空 text（strip 后 ≤ 500 字，
    MAX_VOICE_TEXT_CHARS）与 number 型 start_s/end_s 且 0 ≤ start_s < end_s ≤ duration_s
    （边界允许 0.01s 浮点误差）、start_s 单调不减。错误信息指明第几项与原因；返回项只保留白名单三字段。
    """
    from app.pipeline import PipelineError  # 循环导入：pipeline 顶层导入本模块

    if len(raw) > MAX_VOICE_LINES_BYTES:
        raise PipelineError(f"voice_lines.json exceeds {MAX_VOICE_LINES_BYTES} bytes: {len(raw)}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise PipelineError("voice_lines.json not valid UTF-8") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"voice_lines.json not valid JSON: {e}") from None
    if not isinstance(data, list):
        raise PipelineError("voice_lines.json must be an array")
    if not data:
        return []  # 空数组 = 音轨无台词（合法；codex 摆烂由 pipeline 声学预判 + 重试兜底）
    if len(data) > MAX_VOICE_LINES_ITEMS:
        raise PipelineError(f"voice_lines.json exceeds {MAX_VOICE_LINES_ITEMS} items: {len(data)}")
    lines: list[dict] = []
    prev_start = -1.0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise PipelineError(f"voice_lines[{i}] must be an object")
        line_text = item.get("text")
        if not isinstance(line_text, str):
            raise PipelineError(f"voice_lines[{i}].text must be a non-empty string")
        line_text = line_text.strip()
        if not line_text:
            raise PipelineError(f"voice_lines[{i}].text must be a non-empty string")
        if len(line_text) > MAX_VOICE_TEXT_CHARS:
            raise PipelineError(
                f"voice_lines[{i}].text exceeds {MAX_VOICE_TEXT_CHARS} chars: {len(line_text)}"
            )
        start_s, end_s = item.get("start_s"), item.get("end_s")
        for key, val in (("start_s", start_s), ("end_s", end_s)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise PipelineError(f"voice_lines[{i}].{key} must be a number")
        start_s, end_s = float(start_s), float(end_s)
        if not (-_EPS_S <= start_s < end_s <= duration_s + _EPS_S):
            raise PipelineError(
                f"voice_lines[{i}] times invalid: need 0 <= start_s({start_s:.3f}) "
                f"< end_s({end_s:.3f}) <= duration({duration_s:.3f})"
            )
        if start_s < prev_start:
            raise PipelineError(
                f"voice_lines[{i}].start_s {start_s:.3f}s before previous {prev_start:.3f}s"
            )
        prev_start = start_s
        lines.append({"text": line_text, "start_s": start_s, "end_s": end_s})
    return lines
