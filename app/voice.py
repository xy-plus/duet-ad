"""口播链路纯函数：extract_audio 抽音轨、validate_voice_lines 台词白名单校验。

听写本身交给 codex 沙箱，本模块不装任何 ASR 库。PipelineError 归口 pipeline.py；
pipeline 顶层导入本模块，为避免循环导入，异常类在函数内延迟导入。
"""

from __future__ import annotations

import json
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
        proc = subprocess.run(
            ["ffmpeg", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "64k", "-y", str(out)],
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

    校验：raw ≤ 32KB（MAX_VOICE_LINES_BYTES）、UTF-8 可解、JSON 解析为非空 list、
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
    if not isinstance(data, list) or not data:
        raise PipelineError("voice_lines.json must be a non-empty array")
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
