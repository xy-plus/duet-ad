"""Deterministic media algorithms for the H3 video pipeline.

The module intentionally keeps only the media mechanics needed by the research
pipeline: exact source-frame/scene planning, scene-aware exact-nine sampling,
visual-axis audio, 4fps analysis frames, and deterministic 24fps assembly.
It has no provider, receipt, recovery, CAS, or quality-gate dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from app import vocal


KEYFRAMES_PER_SEGMENT = 9
DISPLAY_TIME_PRECISION = 6
SHORT_VIDEO_MAX_S = 10.0
LONG_VIDEO_MAX_S = 300.0
SEGMENT_PROVIDER_MIN_DURATION_S = 4
SEGMENT_PROVIDER_MAX_DURATION_S = 10
SEGMENT_SOURCE_MIN_S = float(SEGMENT_PROVIDER_MIN_DURATION_S)
BOUNDARY_PRECISION = 6
STITCH_FPS = 24
STITCH_FRAME_DURATION_S = 1 / STITCH_FPS
_EPS = 1e-6

ANALYSIS_IMAGE_MIN_EDGE = 256
ANALYSIS_IMAGE_MAX_EDGE = 5760

_UNRECOGNIZED_ASR_RE = re.compile(
    r"(?:[\[\(\{（【]\s*)?(?:无法(?:辨识|识别|听清)|听不清|"
    r"inaudible|unintelligible|unrecognized(?:\s+speech)?|"
    r"no\s+(?:speech|audio)|blank[_\s-]?audio|silence|music)"
    r"\s*[.!。]?\s*(?:[\]\)\}）】]|$)",
    re.IGNORECASE,
)
_SUBTITLE_CREDIT_RE = re.compile(
    r"^\s*[\[\(\{（【]?\s*(?:"
    r"(?:字幕(?:制作|製作|翻译|翻譯|校对|校對|时间轴|時間軸)?|"
    r"翻译|翻譯|校对|校對|时间轴|時間軸)\s*[:：]\s*\S.{0,80}|"
    r"(?:subtitles?|captions?|translated|translation|timing|typeset)\s+"
    r"(?:by|:)\s*\S.{0,80}"
    r")\s*[\]\)\}）】]?\s*$",
    re.IGNORECASE,
)
_PURE_FILLERS = frozenset({
    "ah", "ano", "eh", "er", "erm", "hmm", "hm", "mm", "mhm",
    "oh", "uh", "um", "uhm", "umm", "ya", "yah",
    "あの", "ええと", "えー", "うーん",
    "啊", "呃", "额", "嗯", "哦", "喔", "唔", "哎", "哎呀", "呀",
    "哈", "哈哈", "呵", "呵呵",
})
_LATIN_FILLER = re.compile(
    r"(?:a+h+|e+h+|e+r+m*|h+m+|m+h+m+|o+h+|u+h+|u+h*m+|u+m+)",
    re.IGNORECASE,
)
_WRAPPED_NON_SPEECH = (
    re.compile(r"^\*+\s*\S(?:[\s\S]*\S)?\s*\*+$"),
    re.compile(r"^\[[\s\S]+\]$"),
    re.compile(r"^\([\s\S]+\)$"),
    re.compile(r"^<[^<>]+>$"),
    re.compile(r"^[♪♫][\s\S]*[♪♫]$"),
)


class CommandError(RuntimeError):
    def __init__(
        self,
        argv: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(
            f"command failed ({returncode}): {' '.join(argv)}\n{stderr[-2000:]}"
        )
        self.argv = argv
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class MediaError(RuntimeError):
    """A deterministic media invariant could not be satisfied."""


def run_command(
    argv: list[str],
    *,
    timeout: float = 600,
    cwd: Path | None = None,
    text: bool = True,
) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=text,
        timeout=timeout,
    )
    stdout = (
        completed.stdout
        if isinstance(completed.stdout, str)
        else completed.stdout.decode("utf-8", errors="replace")
    )
    stderr = (
        completed.stderr
        if isinstance(completed.stderr, str)
        else completed.stderr.decode("utf-8", errors="replace")
    )
    result = {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if completed.returncode != 0:
        raise CommandError(
            argv,
            completed.returncode,
            stdout,
            stderr,
        )
    return result


def _parse_rate(value: object) -> float:
    rate = _positive_rate(value)
    if rate is None:
        raise ValueError("invalid frame rate")
    return rate


def _positive_rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        rate = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _packet_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probe_packets(path: Path, selector: str) -> list[dict[str, Any]]:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", selector,
            "-show_packets", "-show_entries",
            "packet=pts_time,dts_time,duration_time,side_data_list",
            "-of", "json", str(path),
        ],
        timeout=30,
    )
    try:
        packets = json.loads(result["stdout"]).get("packets")
    except (AttributeError, TypeError, json.JSONDecodeError):
        packets = None
    if not isinstance(packets, list):
        raise MediaError(f"cannot parse {selector} packet timeline")
    return packets


def _probe_packet_duration(path: Path, stream: Mapping[str, object]) -> float | None:
    starts: list[float] = []
    explicit_ends: list[float] = []
    starts_with_duration: set[float] = set()
    for packet in _probe_packets(path, "v:0"):
        if not isinstance(packet, dict):
            continue
        start = _packet_number(packet.get("pts_time"))
        if start is None:
            start = _packet_number(packet.get("dts_time"))
        if start is None:
            continue
        starts.append(start)
        packet_duration = _packet_number(packet.get("duration_time"))
        if packet_duration is not None and packet_duration > 0:
            explicit_ends.append(start + packet_duration)
            starts_with_duration.add(start)
    starts = sorted(set(starts))
    if not starts:
        return None
    inferred_step = None
    if len(starts) > 1:
        positive_steps = [
            right - left
            for left, right in zip(starts, starts[1:])
            if right > left
        ]
        if positive_steps:
            inferred_step = positive_steps[-1]
    if inferred_step is None:
        rate = _positive_rate(stream.get("avg_frame_rate")) or _positive_rate(
            stream.get("r_frame_rate")
        )
        inferred_step = 1 / rate if rate else None
    ends = explicit_ends
    if (
        starts[-1] not in starts_with_duration
        and inferred_step is not None
        and inferred_step > 0
    ):
        ends.append(starts[-1] + inferred_step)
    if not ends:
        return None
    candidate = max(ends) - starts[0]
    return candidate if math.isfinite(candidate) and candidate > 0 else None


def probe(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=index,codec_type,codec_name,pix_fmt,width,height,duration,"
            "duration_ts,start_time,time_base,avg_frame_rate,r_frame_rate,nb_frames",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    payload = json.loads(result["stdout"])
    video = next(
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    )
    duration: float | None = None
    raw_duration = video.get("duration")
    if not isinstance(raw_duration, bool):
        try:
            candidate = float(raw_duration)
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
        except (TypeError, ValueError):
            pass
    if duration is None:
        try:
            raw_duration_ts = video.get("duration_ts")
            if isinstance(raw_duration_ts, bool):
                raise TypeError
            ticks = float(raw_duration_ts)
            numerator, denominator = str(video.get("time_base")).split("/", 1)
            candidate = ticks * float(numerator) / float(denominator)
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if duration is None:
        duration = _probe_packet_duration(path, video)
    if duration is None:
        raise MediaError("cannot parse video duration")
    try:
        frame_rate = _parse_rate(video.get("avg_frame_rate"))
    except (TypeError, ValueError, ZeroDivisionError):
        frame_rate = 0.0
    try:
        frame_count = int(video.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = 0
    return {
        "duration_s": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": frame_rate,
        "frame_count": frame_count,
        "codec_name": str(video.get("codec_name") or ""),
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "has_audio": any(
            stream.get("codec_type") == "audio"
            for stream in payload.get("streams", [])
        ),
        "raw": payload,
    }


def _stream_start_time(path: Path, selector: str) -> float:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", selector,
            "-show_entries", "stream=start_time,sample_rate", "-of", "json",
            str(path),
        ],
        timeout=30,
    )
    try:
        streams = json.loads(result["stdout"]).get("streams")
    except (AttributeError, TypeError, json.JSONDecodeError):
        streams = None
    if (
        not isinstance(streams, list)
        or not streams
        or not isinstance(streams[0], dict)
    ):
        raise MediaError(f"cannot parse {selector} stream timeline")
    stream = streams[0]
    start = _packet_number(stream.get("start_time"))
    if start is not None:
        return start

    starts: list[tuple[float, dict[str, Any]]] = []
    for packet in _probe_packets(path, selector):
        if not isinstance(packet, dict):
            continue
        packet_start = _packet_number(packet.get("pts_time"))
        if packet_start is None:
            packet_start = _packet_number(packet.get("dts_time"))
        if packet_start is not None:
            starts.append((packet_start, packet))
    if not starts:
        raise MediaError(f"cannot parse {selector} stream packet timeline")
    start, first_packet = min(starts, key=lambda item: item[0])
    if selector.startswith("a:"):
        sample_rate = _packet_number(stream.get("sample_rate"))
        side_data = first_packet.get("side_data_list")
        if isinstance(side_data, list):
            for item in side_data:
                if (
                    not isinstance(item, dict)
                    or item.get("side_data_type") != "Skip Samples"
                ):
                    continue
                skip_samples = _packet_number(item.get("skip_samples"))
                if (
                    sample_rate is None
                    or sample_rate <= 0
                    or skip_samples is None
                    or skip_samples < 0
                ):
                    raise MediaError("cannot parse audio priming timeline")
                start += skip_samples / sample_rate
                break
    return start


def extract_audio(source: Path, destination: Path) -> dict[str, Any] | None:
    """Extract mono 16k audio on the source video's visual timeline."""
    info = probe(source)
    if not info["has_audio"]:
        destination.unlink(missing_ok=True)
        return None
    video_start = _stream_start_time(source, "v:0")
    audio_filter = (
        f"asetpts=PTS-({video_start:.9f})/TB,"
        "aresample=async=1:first_pts=0,apad,"
        f"atrim=duration={info['duration_s']:.9f}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    codec = (
        ["-c:a", "pcm_s16le"]
        if destination.suffix.lower() == ".wav"
        else ["-b:a", "64k"]
    )
    return run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
            "-ac", "1", "-ar", "16000", "-filter:a", audio_filter,
            *codec, str(destination),
        ],
        timeout=180,
    )


def is_unrecognized_text(text: str) -> bool:
    return (
        isinstance(text, str)
        and _UNRECOGNIZED_ASR_RE.search(text.strip()) is not None
    )


def is_subtitle_credit_text(text: str) -> bool:
    return (
        isinstance(text, str)
        and len(text.strip()) <= 120
        and _SUBTITLE_CREDIT_RE.fullmatch(text.strip()) is not None
    )


def _is_non_dialogue_text(text: str) -> bool:
    """Reject Master's wrapped sound events and standalone filler utterances."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        return True
    if any(pattern.fullmatch(normalized) for pattern in _WRAPPED_NON_SPEECH):
        return True
    tokens = [
        token
        for token in re.split(
            r"[^\w\u3040-\u30ff\u3400-\u9fff]+",
            normalized.casefold(),
        )
        if token
    ]
    return bool(tokens) and all(
        token in _PURE_FILLERS or _LATIN_FILLER.fullmatch(token)
        for token in tokens
    )


def _probe_audio_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        timeout=30,
    )
    try:
        duration = float(json.loads(result["stdout"])["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise MediaError("cannot parse ASR audio duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("cannot parse ASR audio duration")
    return duration


def transcribe(
    audio: Path,
    destination: Path,
    cli: Path,
    model: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run whisper.cpp with byte-safe diagnostics and normalize timed dialogue."""
    prefix = destination.with_suffix("")
    result = run_command(
        [
            str(cli), "-m", str(model), "-f", str(audio), "-l", "auto",
            "-ojf", "-of", str(prefix), "-ng", "-t", "4",
        ],
        timeout=900,
        text=False,
    )
    try:
        raw = prefix.with_suffix(".json").read_bytes()
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        raise MediaError("ASR output is invalid") from None
    transcription = payload.get("transcription") if isinstance(payload, dict) else None
    if not isinstance(transcription, list):
        raise MediaError("ASR output is invalid")
    duration_s = _probe_audio_duration(audio)
    lines: list[dict[str, Any]] = []
    for item in transcription:
        if not isinstance(item, dict):
            raise MediaError("ASR output is invalid")
        text = item.get("text")
        offsets = item.get("offsets")
        if not isinstance(text, str) or not isinstance(offsets, dict):
            raise MediaError("ASR output is invalid")
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, (int, float))
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, (int, float))
        ):
            raise MediaError("ASR output is invalid")
        text = text.replace("\ufffd", "").strip()
        start_s = max(0.0, float(start_ms) / 1000.0)
        end_s = min(duration_s, float(end_ms) / 1000.0)
        if (
            text
            and not _is_non_dialogue_text(text)
            and not is_unrecognized_text(text)
            and math.isfinite(start_s)
            and math.isfinite(end_s)
            and start_s < end_s
        ):
            lines.append(
                {
                    "text": text,
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )
    destination.write_text(
        json.dumps(lines, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return lines, {"command": result, "provider_raw": payload}


def yamnet_analyze(
    audio: Path,
    model_path: Path | None = None,
) -> dict[str, Any]:
    analysis = vocal.analyze(audio, model_path=model_path)
    return {
        "windows": [
            {
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
                "sung": window.sung,
                "spoken": window.spoken,
                "music": window.music,
            }
            for window in analysis.windows
        ],
        "has_bgm": analysis.has_bgm,
    }


def filter_spoken_lines(
    lines: Iterable[Mapping[str, object]],
    yamnet: Mapping[str, object] | vocal.VocalAnalysis,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify every ASR row; only Master's effective ``spoken`` rows pass."""
    source_lines = [dict(line) for line in lines]
    analysis = (
        yamnet
        if isinstance(yamnet, vocal.VocalAnalysis)
        else vocal.analysis_from_mapping(yamnet)
    )
    spoken: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for line in source_lines:
        text = line.get("text")
        if not isinstance(text, str) or not text.strip():
            raise MediaError("ASR line text is invalid")
        if is_unrecognized_text(text):
            decision = {
                **line,
                "classification": None,
                "provenance": "asr",
                "kept": False,
                "drop_reason": "unrecognized_sentinel",
            }
        elif is_subtitle_credit_text(text):
            decision = {
                **line,
                "classification": None,
                "provenance": "asr",
                "kept": False,
                "drop_reason": "subtitle_credit",
            }
        else:
            classification = vocal.classify_line(
                line,
                analysis,
                only_line=len(source_lines) == 1,
            )
            decision = {
                **line,
                "classification": classification,
                "provenance": "asr",
                "kept": classification in {"spoken", "sung"},
            }
            if classification is None:
                decision["drop_reason"] = "no_vocal_evidence"
        decisions.append(decision)
        if decision["kept"] and decision["classification"] == "spoken":
            spoken.append(
                {
                    "text": text.strip(),
                    "start_s": float(line["start_s"]),
                    "end_s": float(line["end_s"]),
                }
            )
    return spoken, decisions


def extract_frames_4fps(source: Path, destination: Path) -> dict[str, Any]:
    """Decode the visual timeline once at 4fps and write Master's manifest core."""
    destination.mkdir(parents=True, exist_ok=True)
    info = probe(source)
    source_fps = float(info["fps"] or 0)
    frame_count = int(info["frame_count"] or 0)
    if source_fps <= 0 or frame_count <= 0:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise MediaError(f"could not open video: {source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
    if source_fps <= 0 or frame_count <= 0:
        raise MediaError("video metadata is incomplete")
    duration = float(info["duration_s"])
    rate = 4.0
    step = 1.0 / rate
    last_safe_time = max(0.0, duration - 1.0 / source_fps)
    times = [index * step for index in range(int(duration * rate) + 1)]
    if not times or last_safe_time - times[-1] > step / 2:
        times.append(last_safe_time)
    else:
        times[-1] = last_safe_time

    with tempfile.TemporaryDirectory(prefix=".frames-", dir=destination) as raw_tmp:
        temporary = Path(raw_tmp)
        run_command(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(source),
                "-map", "0:v:0", "-vf",
                "setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={step:.9f},"
                "fps=fps=4:start_time=0",
                "-frames:v", str(len(times)), str(temporary / "%06d.png"),
            ],
            timeout=300,
        )
        digits = max(2, len(str(len(times))))
        records: list[dict[str, object]] = []
        for index, requested_time in enumerate(times, 1):
            extracted = temporary / f"{index:06d}.png"
            if not extracted.is_file():
                raise MediaError("could not decode all requested video frames")
            filename = f"{index:0{digits}d}_frame_{requested_time:07.3f}s.png"
            extracted.replace(destination / filename)
            records.append(
                {
                    "index": index,
                    "time_seconds": round(requested_time, 6),
                    "file": filename,
                }
            )

    width = int(info["width"])
    height = int(info["height"])
    manifest = {
        "source": str(source),
        "file_size_bytes": source.stat().st_size,
        "duration_seconds": round(duration, 6),
        "fps": round(source_fps, 6),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "orientation": (
            "portrait"
            if height > width
            else "landscape" if width > height else "square"
        ),
        "audio_present": bool(info["has_audio"]),
        "frames": records,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _pad_analysis_image_ratio(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = max(height, ANALYSIS_IMAGE_MIN_EDGE)
    target_width = max(width, ANALYSIS_IMAGE_MIN_EDGE)
    if 2 * target_width > 5 * target_height:
        target_height = (2 * target_width + 4) // 5
    elif 5 * target_width < 2 * target_height:
        target_width = (2 * target_height + 4) // 5
    if target_width == width and target_height == height:
        return image
    vertical = target_height - height
    horizontal = target_width - width
    return cv2.copyMakeBorder(
        image,
        vertical // 2,
        vertical - vertical // 2,
        horizontal // 2,
        horizontal - horizontal // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _normalize_analysis_image_bounds(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = height
    target_width = width
    shortest, longest = min(width, height), max(width, height)
    ratio_is_valid = 5 * width >= 2 * height and 2 * width <= 5 * height
    if ratio_is_valid:
        if shortest < ANALYSIS_IMAGE_MIN_EDGE:
            if width <= height:
                target_width = ANALYSIS_IMAGE_MIN_EDGE
                target_height = round(height * target_width / width)
            else:
                target_height = ANALYSIS_IMAGE_MIN_EDGE
                target_width = round(width * target_height / height)
        elif longest > ANALYSIS_IMAGE_MAX_EDGE:
            if width >= height:
                target_width = ANALYSIS_IMAGE_MAX_EDGE
                target_height = round(height * target_width / width)
            else:
                target_height = ANALYSIS_IMAGE_MAX_EDGE
                target_width = round(width * target_height / height)
    else:
        target_longest = min(
            ANALYSIS_IMAGE_MAX_EDGE,
            max(ANALYSIS_IMAGE_MIN_EDGE, longest),
        )
        if longest != target_longest:
            if width >= height:
                target_width = target_longest
                target_height = max(1, round(height * target_width / width))
            else:
                target_height = target_longest
                target_width = max(1, round(width * target_height / height))
    if target_width != width or target_height != height:
        interpolation = (
            cv2.INTER_AREA
            if target_width < width or target_height < height
            else cv2.INTER_CUBIC
        )
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )
    else:
        resized = image
    return _pad_analysis_image_ratio(resized)


def half_resolution_png(source: Path, destination: Path) -> Path:
    """Write Master's deterministic bounded half-resolution PNG proxy."""
    data = source.read_bytes()
    if not data:
        raise MediaError("analysis image source is invalid")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise MediaError("analysis image source is invalid")
    image = _normalize_analysis_image_bounds(image)
    height, width = image.shape[:2]
    half_size = ((width + 1) // 2, (height + 1) // 2)
    resized = (
        cv2.resize(image, half_size, interpolation=cv2.INTER_AREA)
        if min(half_size) >= ANALYSIS_IMAGE_MIN_EDGE
        else image
    )
    resized = _pad_analysis_image_ratio(resized)
    encoded, proxy = cv2.imencode(
        ".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not encoded:
        raise MediaError("analysis image proxy is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(proxy.tobytes())
    return destination


def provider_reference_png(source: Path, destination: Path) -> Path:
    """Derive the same bounded MiniMax PNG proxy used by duet-ad1."""

    return half_resolution_png(source, destination)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _frame_time(frame: Mapping[str, object]) -> Fraction:
    pts = _integer(frame.get("pts"), "frame pts", minimum=-(2**63))
    origin = _integer(
        frame.get("pts_origin", 0),
        "frame PTS origin",
        minimum=-(2**63),
    )
    numerator = _integer(
        frame.get("time_base_num"), "time base numerator", minimum=1
    )
    denominator = _integer(
        frame.get("time_base_den"), "time base denominator", minimum=1
    )
    return Fraction((pts - origin) * numerator, denominator)


def _display_time(value: Fraction) -> float:
    return round(float(value), DISPLAY_TIME_PRECISION)


def _exact_time_receipt(value: Fraction) -> dict[str, int]:
    return {
        "pts": value.numerator,
        "time_base_num": 1,
        "time_base_den": value.denominator,
    }


def _receipt_time(value: object, label: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an exact time receipt")
    return Fraction(
        _integer(value.get("pts"), f"{label} pts", minimum=-(2**63))
        * _integer(value.get("time_base_num"), f"{label} numerator", minimum=1),
        _integer(value.get("time_base_den"), f"{label} denominator", minimum=1),
    )


def probe_decoded_frames(video: Path) -> list[dict[str, Any]]:
    """Read the original v:0 decoded ordinal and rational PTS inventory."""
    result = run_command(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_entries",
            "stream=time_base:frame=best_effort_timestamp,pts,pkt_duration",
            "-of", "json", str(video),
        ],
        timeout=300,
    )
    try:
        payload = json.loads(result["stdout"] or "{}")
        streams = payload.get("streams")
        raw_frames = payload.get("frames")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("missing v:0 stream")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("missing decoded frames")
        numerator_text, denominator_text = str(
            streams[0].get("time_base")
        ).split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if numerator <= 0 or denominator <= 0:
            raise ValueError("invalid v:0 time base")
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError(f"invalid decoded-frame inventory: {exc}") from None

    frames: list[dict[str, Any]] = []
    previous_time: Fraction | None = None
    pts_origin: int | None = None
    for decode_index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise MediaError("decoded-frame inventory contains an invalid item")
        raw_pts = raw.get("best_effort_timestamp", raw.get("pts"))
        try:
            pts = int(raw_pts)
        except (TypeError, ValueError):
            raise MediaError("decoded frame is missing integer PTS") from None
        if pts_origin is None:
            pts_origin = pts
        exact_time = Fraction((pts - pts_origin) * numerator, denominator)
        if previous_time is not None and exact_time < previous_time:
            raise MediaError("decoded frame PTS is not monotonic")
        duration_pts = raw.get("pkt_duration")
        try:
            duration_ticks = (
                int(duration_pts) if duration_pts is not None else None
            )
        except (TypeError, ValueError):
            duration_ticks = None
        frames.append(
            {
                "decode_frame_index": decode_index,
                "pts": pts,
                "pts_origin": pts_origin,
                "duration_pts": (
                    duration_ticks
                    if duration_ticks is not None and duration_ticks > 0
                    else None
                ),
                "time_base_num": numerator,
                "time_base_den": denominator,
                "source_time_s": _display_time(exact_time),
            }
        )
        previous_time = exact_time
    return frames


def _exclusive_end_time(
    decoded_frames: list[dict[str, Any]], end_index: int
) -> Fraction:
    if end_index < len(decoded_frames):
        return _frame_time(decoded_frames[end_index])
    last = decoded_frames[-1]
    duration_ticks = last.get("duration_pts")
    if isinstance(duration_ticks, int) and duration_ticks > 0:
        return _frame_time(last) + Fraction(
            duration_ticks * last["time_base_num"],
            last["time_base_den"],
        )
    if len(decoded_frames) > 1:
        step = _frame_time(decoded_frames[-1]) - _frame_time(decoded_frames[-2])
        if step > 0:
            return _frame_time(last) + step
    raise MediaError("cannot determine the final frame's exclusive end PTS")


def detect_exact_scene_bounds(
    video: Path,
    threshold: float,
    decoded_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run ContentDetector once and bind cuts to decoded-frame ordinals."""
    diagnostics: list[str] = []
    try:
        scene_list = detect(
            str(video),
            ContentDetector(threshold=threshold),
            start_in_scene=True,
        )
    except Exception:
        scene_list = []
        diagnostics.append("scene_detector_error_normalized")
    if not scene_list:
        diagnostics.append("scene_detector_no_cut_normalized")
        start_exact = _frame_time(decoded_frames[0])
        end_exact = _exclusive_end_time(decoded_frames, len(decoded_frames))
        common_denominator = math.lcm(
            start_exact.denominator, end_exact.denominator
        )
        return [
            {
                "index": 1,
                "start_decode_frame_index": 0,
                "end_decode_frame_index": len(decoded_frames),
                "start_pts": start_exact.numerator
                * (common_denominator // start_exact.denominator),
                "end_pts": end_exact.numerator
                * (common_denominator // end_exact.denominator),
                "time_base_num": 1,
                "time_base_den": common_denominator,
                "start_s": _display_time(start_exact),
                "end_s": _display_time(end_exact),
                "diagnostics": diagnostics,
            }
        ]
    bounds: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(scene_list, 1):
        start_index = max(0, min(int(start.frame_num), len(decoded_frames)))
        end_index = max(
            start_index, min(int(end.frame_num), len(decoded_frames))
        )
        start_exact = (
            _frame_time(decoded_frames[start_index])
            if start_index < len(decoded_frames)
            else _exclusive_end_time(decoded_frames, start_index)
        )
        end_exact = _exclusive_end_time(decoded_frames, end_index)
        denominator = math.lcm(start_exact.denominator, end_exact.denominator)
        bounds.append(
            {
                "index": index,
                "start_decode_frame_index": start_index,
                "end_decode_frame_index": end_index,
                "start_pts": start_exact.numerator
                * (denominator // start_exact.denominator),
                "end_pts": end_exact.numerator
                * (denominator // end_exact.denominator),
                "time_base_num": 1,
                "time_base_den": denominator,
                "start_s": _display_time(start_exact),
                "end_s": _display_time(end_exact),
                "diagnostics": [],
            }
        )
    return bounds


def _scene_bound_time(
    bound: Mapping[str, object],
    edge: str,
    frames: list[dict[str, Any]],
) -> Fraction:
    pts_key = f"{edge}_pts"
    if pts_key in bound:
        pts = _integer(bound.get(pts_key), pts_key, minimum=-(2**63))
        numerator = _integer(
            bound.get("time_base_num"),
            "scene time base numerator",
            minimum=1,
        )
        denominator = _integer(
            bound.get("time_base_den"),
            "scene time base denominator",
            minimum=1,
        )
        return Fraction(pts * numerator, denominator)
    display = bound.get(f"{edge}_s")
    if isinstance(display, bool) or not isinstance(display, (int, float)):
        if frames:
            return _frame_time(frames[0] if edge == "start" else frames[-1])
        return Fraction(0)
    if not math.isfinite(float(display)):
        raise ValueError(f"scene {edge}_s must be finite")
    return Fraction(str(display))


def normalize_scene_inventory(
    bounds: Iterable[Mapping[str, object]],
    decoded_frames: Iterable[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Assign every decoded frame to exact half-open scene intervals."""
    frames: list[dict[str, Any]] = []
    previous_index: int | None = None
    previous_time: Fraction | None = None
    for position, raw in enumerate(decoded_frames, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"decoded frame {position} must be an object")
        item = dict(raw)
        decode_index = _integer(
            item.get("decode_frame_index"), "decode frame index"
        )
        exact_time = _frame_time(item)
        if previous_index is not None and decode_index <= previous_index:
            raise ValueError("decoded frame indices must be strictly increasing")
        if previous_time is not None and exact_time < previous_time:
            raise ValueError("decoded frame PTS values must be nondecreasing")
        item["source_time_s"] = _display_time(exact_time)
        frames.append(item)
        previous_index = decode_index
        previous_time = exact_time
    if not frames:
        raise ValueError("decoded frame inventory must not be empty")

    raw_scenes: list[dict[str, Any]] = []
    previous_end: int | None = None
    for position, raw in enumerate(bounds, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"scene bound {position} must be an object")
        source_index = _integer(
            raw.get("index", position), "scene index", minimum=1
        )
        if source_index != position:
            raise ValueError("scene indices must be contiguous and one-based")
        start = _integer(
            raw.get("start_decode_frame_index"),
            "scene start decode frame index",
        )
        end = _integer(
            raw.get("end_decode_frame_index"),
            "scene end decode frame index",
        )
        if start > end or (previous_end is not None and start != previous_end):
            raise ValueError("scene decode frame intervals must be contiguous")
        owned = [
            frame
            for frame in frames
            if start <= frame["decode_frame_index"] < end
        ]
        raw_scenes.append(
            {
                "index": source_index,
                "source_scene_indices": [source_index],
                "start_decode_frame_index": start,
                "end_decode_frame_index": end,
                "start_exact": _scene_bound_time(raw, "start", owned),
                "end_exact": _scene_bound_time(raw, "end", owned),
                "frames": owned,
            }
        )
        previous_end = end
    if not raw_scenes:
        raise ValueError("scene inventory must not be empty")
    assigned = [
        frame["decode_frame_index"]
        for scene in raw_scenes
        for frame in scene["frames"]
    ]
    if assigned != [frame["decode_frame_index"] for frame in frames]:
        raise ValueError("scene intervals do not cover every decoded frame exactly once")

    first_populated = next(
        (
            position
            for position, scene in enumerate(raw_scenes)
            if scene["frames"]
        ),
        None,
    )
    if first_populated is None:
        raise ValueError("scene inventory contains no populated scene")
    if first_populated:
        leading = raw_scenes[:first_populated]
        target = raw_scenes[first_populated]
        target["source_scene_indices"] = [
            source
            for scene in leading + [target]
            for source in scene["source_scene_indices"]
        ]
        target["start_decode_frame_index"] = leading[0][
            "start_decode_frame_index"
        ]
        target["start_exact"] = leading[0]["start_exact"]

    effective: list[dict[str, Any]] = []
    for scene in raw_scenes[first_populated:]:
        if scene["frames"]:
            effective.append(scene)
            continue
        previous = effective[-1]
        previous["source_scene_indices"].extend(scene["source_scene_indices"])
        previous["end_decode_frame_index"] = scene["end_decode_frame_index"]
        previous["end_exact"] = scene["end_exact"]

    return [
        {
            "index": index,
            "source_scene_indices": scene["source_scene_indices"],
            "start_decode_frame_index": scene["start_decode_frame_index"],
            "end_decode_frame_index": scene["end_decode_frame_index"],
            "start_s": _display_time(scene["start_exact"]),
            "end_s": _display_time(scene["end_exact"]),
            "start_time": _exact_time_receipt(scene["start_exact"]),
            "end_time": _exact_time_receipt(scene["end_exact"]),
            "frames": scene["frames"],
        }
        for index, scene in enumerate(effective, 1)
    ]


def detect_scenes(
    source: Path,
    duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return ContentDetector(27) scenes with the complete source inventory."""
    if duration_s is not None and (
        not math.isfinite(float(duration_s)) or float(duration_s) <= 0
    ):
        raise ValueError("duration_s must be finite and positive")
    frames = probe_decoded_frames(source)
    bounds = detect_exact_scene_bounds(source, 27.0, frames)
    inventory = normalize_scene_inventory(bounds, frames)
    return inventory


def segment_duration_s(start_s: object, end_s: object) -> float:
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise MediaError("long_video_invalid_segment_duration")
    start = round(float(start_s), BOUNDARY_PRECISION)
    end = round(float(end_s), BOUNDARY_PRECISION)
    duration = round(end - start, BOUNDARY_PRECISION)
    if not (
        math.isfinite(start)
        and math.isfinite(end)
        and duration > 0
    ):
        raise MediaError("long_video_invalid_segment_duration")
    return duration


def provider_duration_s(start_s: object, end_s: object) -> int:
    return max(
        SEGMENT_PROVIDER_MIN_DURATION_S,
        math.ceil(segment_duration_s(start_s, end_s)),
    )


def _finite_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaError("long_video_invalid_duration")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("long_video_invalid_duration")
    if duration > LONG_VIDEO_MAX_S:
        raise MediaError("long_video_duration_exceeded")
    return duration


def _bounds(
    scenes: Iterable[Mapping[str, object] | Sequence[float]],
    duration_s: float,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for raw in scenes:
        try:
            if isinstance(raw, Mapping):
                start, end = raw["start_s"], raw["end_s"]
            else:
                start, end = raw[0], raw[1]
            start, end = float(start), float(end)
        except (KeyError, IndexError, TypeError, ValueError):
            raise MediaError("long_video_invalid_scenes") from None
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise MediaError("long_video_invalid_scenes")
        result.append((start, end))
    if not result:
        return [(0.0, duration_s)]
    if abs(result[0][0]) > _EPS or abs(result[-1][1] - duration_s) > _EPS:
        raise MediaError("long_video_invalid_scenes")
    previous = 0.0
    for start, end in result:
        if abs(start - previous) > _EPS:
            raise MediaError("long_video_invalid_scenes")
        previous = end
    return result


def _dialogue_intervals(
    lines: Iterable[Mapping[str, object]],
    duration_s: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    previous_start = -1.0
    for line in lines:
        try:
            start, end = float(line["start_s"]), float(line["end_s"])
        except (KeyError, TypeError, ValueError):
            raise MediaError("long_video_invalid_dialogue") from None
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or start >= end
            or end > duration_s + _EPS
            or start < previous_start
        ):
            raise MediaError("long_video_invalid_dialogue")
        intervals.append((start, min(end, duration_s)))
        previous_start = start
    return intervals


def _base_plan_segments(
    duration_s: float,
    scenes: Iterable[Mapping[str, object] | Sequence[float]],
    dialogue: Iterable[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Master's 4-10s scene-first planner; dialogue is validation-only."""
    duration = _finite_duration(duration_s)
    if duration <= SHORT_VIDEO_MAX_S:
        _dialogue_intervals(dialogue, duration)
        return [
            {
                "index": 1,
                "start_s": 0.0,
                "end_s": round(duration, BOUNDARY_PRECISION),
                "chain_id": "chain-001",
                "join_mode": "hard_cut",
            }
        ]
    scene_bounds = _bounds(scenes, duration)
    _dialogue_intervals(dialogue, duration)
    hard_cuts = {end for _start, end in scene_bounds[:-1]}

    cuts = [0.0]
    while provider_duration_s(cuts[-1], duration) > SEGMENT_PROVIDER_MAX_DURATION_S:
        start = cuts[-1]
        target = min(
            round(start + SEGMENT_PROVIDER_MAX_DURATION_S, BOUNDARY_PRECISION),
            duration - SEGMENT_SOURCE_MIN_S,
        )
        minimum = start + SEGMENT_SOURCE_MIN_S
        eligible_hard_cuts = [
            cut
            for cut in hard_cuts
            if minimum <= cut <= target
            and segment_duration_s(cut, duration) >= SEGMENT_SOURCE_MIN_S
            and provider_duration_s(start, cut) <= SEGMENT_PROVIDER_MAX_DURATION_S
        ]
        boundary = max(eligible_hard_cuts) if eligible_hard_cuts else target
        if boundary <= start + _EPS:
            raise MediaError("long_video_no_safe_dialogue_boundary")
        cuts.append(round(boundary, BOUNDARY_PRECISION))
    cuts.append(duration)

    if segment_duration_s(cuts[-2], cuts[-1]) < SEGMENT_SOURCE_MIN_S:
        cuts.pop(-2)
        target = duration - SEGMENT_SOURCE_MIN_S
        if provider_duration_s(target, duration) > SEGMENT_PROVIDER_MAX_DURATION_S:
            raise MediaError("long_video_no_safe_dialogue_boundary")
        cuts.insert(-1, round(target, BOUNDARY_PRECISION))

    segments: list[dict[str, Any]] = []
    chain_number = 1
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), 1):
        if (
            segment_duration_s(start, end) < SEGMENT_SOURCE_MIN_S
            or provider_duration_s(start, end) > SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            raise MediaError("long_video_no_safe_dialogue_boundary")
        hard_cut = index == 1 or any(abs(start - cut) <= _EPS for cut in hard_cuts)
        if index > 1 and hard_cut:
            chain_number += 1
        segments.append(
            {
                "index": index,
                "start_s": round(start, BOUNDARY_PRECISION),
                "end_s": round(end, BOUNDARY_PRECISION),
                "chain_id": f"chain-{chain_number:03d}",
                "join_mode": "hard_cut" if hard_cut else "continue",
            }
        )
    return segments


def _scene_exact_bound(scene: Mapping[str, object], edge: str) -> Fraction:
    receipt = scene.get(f"{edge}_time")
    return (
        _receipt_time(receipt, f"scene {edge}")
        if receipt is not None
        else Fraction(str(scene[f"{edge}_s"]))
    )


def _segment_exact_bound(
    scenes: Iterable[Mapping[str, object]], value: object
) -> Fraction:
    canonical = Fraction(str(value))
    display = _display_time(canonical)
    matches = {
        exact
        for scene in scenes
        for exact in (
            _scene_exact_bound(scene, "start"),
            _scene_exact_bound(scene, "end"),
        )
        if _display_time(exact) == display
    }
    if len(matches) > 1:
        raise ValueError("segment boundary is ambiguous at canonical precision")
    return next(iter(matches), canonical)


def _scene_indices_for_interval(
    scenes: list[dict[str, Any]], start: Fraction, end: Fraction
) -> list[int]:
    return [
        int(scene["index"])
        for scene in scenes
        if any(start <= _frame_time(frame) < end for frame in scene["frames"])
    ]


def _source_cut_timeline(
    scenes: Iterable[Mapping[str, object]],
    start: Fraction,
    end: Fraction,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for scene in scenes:
        scene_start = _scene_exact_bound(scene, "start")
        scene_end = _scene_exact_bound(scene, "end")
        clipped_start = max(start, scene_start)
        clipped_end = min(end, scene_end)
        if clipped_end <= clipped_start:
            continue
        canonical_start = _display_time(clipped_start)
        canonical_end = _display_time(clipped_end)
        if canonical_end <= canonical_start:
            continue
        timeline.append(
            {
                "order": len(timeline) + 1,
                "source_scene_id": f"SCENE_{scene['index']:02d}",
                "start_s": canonical_start,
                "end_s": canonical_end,
            }
        )
    if not timeline:
        raise ValueError("segment source cut timeline is empty")
    return timeline


def plan_segments(
    scenes: Iterable[Mapping[str, object]],
    duration_s: float,
    dialogue: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, Any]]:
    """Return scene-first segments with full source cut provenance."""
    scene_inventory = [dict(scene) for scene in scenes]
    if not scene_inventory or any(
        not scene.get("frames") for scene in scene_inventory
    ):
        raise ValueError("effective scenes must be populated")
    base = _base_plan_segments(duration_s, scene_inventory, dialogue)
    planned: list[dict[str, Any]] = []
    for segment in base:
        start = _segment_exact_bound(scene_inventory, segment["start_s"])
        end = _segment_exact_bound(scene_inventory, segment["end_s"])
        scene_indices = _scene_indices_for_interval(
            scene_inventory, start, end
        )
        source_duration = segment_duration_s(
            segment["start_s"], segment["end_s"]
        )
        if (
            (source_duration < SEGMENT_SOURCE_MIN_S and duration_s >= SEGMENT_SOURCE_MIN_S)
            or provider_duration_s(segment["start_s"], segment["end_s"])
            > SEGMENT_PROVIDER_MAX_DURATION_S
            or not scene_indices
        ):
            raise ValueError("segment provider duration invariant violated")
        planned.append(
            {
                **segment,
                "duration_s": source_duration,
                "scene_indices": scene_indices,
                "source_cut_timeline": _source_cut_timeline(
                    scene_inventory, start, end
                ),
            }
        )
    return planned


def _hamilton_scene_counts(capacities: list[int], seats: int) -> list[int]:
    if not capacities or len(capacities) > seats or sum(capacities) < seats:
        raise ValueError("scene capacities cannot fill keyframe seats")
    counts = [1] * len(capacities)
    remaining = seats - len(capacities)
    spare = [capacity - 1 for capacity in capacities]
    total_spare = sum(spare)
    if not remaining:
        return counts
    quotas = [Fraction(remaining * value, total_spare) for value in spare]
    floors = [quota.numerator // quota.denominator for quota in quotas]
    counts = [count + extra for count, extra in zip(counts, floors)]
    leftovers = remaining - sum(floors)
    ranked = sorted(
        range(len(capacities)),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in ranked[:leftovers]:
        counts[index] += 1
    if any(
        count > capacity for count, capacity in zip(counts, capacities)
    ):
        raise ValueError("Hamilton allocation exceeded scene capacity")
    return counts


def _equidistant_ordinals(capacity: int, count: int) -> list[int]:
    if count < 1 or count > capacity:
        raise ValueError("invalid frame selection count")
    if count == 1:
        return [(capacity - 1) // 2]
    result = []
    for position in range(count):
        exact = Fraction(position * (capacity - 1), count - 1)
        result.append(
            (2 * exact.numerator + exact.denominator)
            // (2 * exact.denominator)
        )
    if len(set(result)) != count:
        raise ValueError("ordinal spacing did not select distinct frames")
    return result


def _nearest_frame(target: Fraction, frames: list[dict[str, Any]]) -> int:
    return min(
        range(len(frames)),
        key=lambda index: (
            abs(_frame_time(frames[index]) - target),
            frames[index]["decode_frame_index"],
        ),
    )


def _under_capacity_counts(
    frames: list[dict[str, Any]], seats: int
) -> list[int]:
    if not frames or len(frames) >= seats:
        raise ValueError("under-capacity repeat planner received invalid frame count")
    if len(frames) == 1:
        return [seats]
    first = _frame_time(frames[0])
    last = _frame_time(frames[-1])
    targets = [
        first + Fraction(position, seats - 1) * (last - first)
        for position in range(seats)
    ]
    assignments = [_nearest_frame(target, frames) for target in targets]
    counts = [assignments.count(index) for index in range(len(frames))]
    for missing in [index for index, count in enumerate(counts) if count == 0]:
        candidates = [
            (
                abs(_frame_time(frames[source]) - targets[target_index]),
                target_index,
                source,
            )
            for target_index, source in enumerate(assignments)
            if counts[source] > 1
        ]
        if not candidates:
            raise ValueError("cannot preserve every decoded frame")
        _distance, target_index, source = max(candidates)
        assignments[target_index] = missing
        counts[source] -= 1
        counts[missing] += 1
    return counts


def _compact_analysis_scene_groups(
    scenes: list[dict[str, Any]],
    start: Fraction,
    end: Fraction,
) -> list[list[dict[str, Any]]]:
    if len(scenes) <= KEYFRAMES_PER_SEGMENT:
        return [[scene] for scene in scenes]
    weights = [
        min(end, _scene_exact_bound(scene, "end"))
        - max(start, _scene_exact_bound(scene, "start"))
        for scene in scenes
    ]
    if any(weight <= 0 for weight in weights):
        raise ValueError("analysis scenes must overlap the segment")
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    for seat in range(KEYFRAMES_PER_SEGMENT - 1):
        remaining_seats = KEYFRAMES_PER_SEGMENT - seat
        last_allowed = len(scenes) - (remaining_seats - 1)
        target = sum(weights[cursor:]) / remaining_seats
        running = Fraction(0)
        candidates = []
        for stop in range(cursor + 1, last_allowed + 1):
            running += weights[stop - 1]
            candidates.append((abs(running - target), stop))
        stop = min(candidates)[1]
        groups.append(scenes[cursor:stop])
        cursor = stop
    groups.append(scenes[cursor:])
    if (
        len(groups) != KEYFRAMES_PER_SEGMENT
        or any(not group for group in groups)
        or [scene["index"] for group in groups for scene in group]
        != [scene["index"] for scene in scenes]
    ):
        raise AssertionError("analysis scene compaction invariant violated")
    return groups


def _analysis_group_representative(
    group: list[dict[str, Any]],
    start: Fraction,
    end: Fraction,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_start = max(start, _scene_exact_bound(group[0], "start"))
    group_end = min(end, _scene_exact_bound(group[-1], "end"))
    target = group_start + (group_end - group_start) / 2
    candidates = [
        (scene, frame)
        for scene in group
        for frame in scene["frames"]
        if start <= _frame_time(frame) < end
    ]
    if not candidates:
        raise ValueError("analysis slot contains no source frame")
    return min(
        candidates,
        key=lambda item: (
            abs(_frame_time(item[1]) - target),
            item[1]["decode_frame_index"],
        ),
    )


def select_segment_keyframes(
    effective_scenes: Iterable[Mapping[str, object]],
    segment: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Select exactly nine deterministic backend-owned source frames."""
    try:
        inventory = [dict(scene) for scene in effective_scenes]
        start = _segment_exact_bound(inventory, segment["start_s"])
        end = _segment_exact_bound(inventory, segment["end_s"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("segment bounds are invalid") from None
    if start < 0 or end <= start:
        raise ValueError("segment bounds are invalid")

    scenes: list[dict[str, Any]] = []
    for raw in inventory:
        scene = dict(raw)
        frames = [
            dict(frame)
            for frame in scene.get("frames", [])
            if start <= _frame_time(frame) < end
        ]
        if frames:
            scene["frames"] = frames
            scenes.append(scene)
    if not scenes:
        raise ValueError("segment must contain at least one effective scene")
    all_frames = [frame for scene in scenes for frame in scene["frames"]]
    dense_groups: list[list[dict[str, Any]]] | None = None
    if len(scenes) > KEYFRAMES_PER_SEGMENT:
        dense_groups = _compact_analysis_scene_groups(scenes, start, end)
        chosen = [
            (*_analysis_group_representative(group, start, end), False)
            for group in dense_groups
        ]
    elif len(all_frames) >= KEYFRAMES_PER_SEGMENT:
        counts = _hamilton_scene_counts(
            [len(scene["frames"]) for scene in scenes],
            KEYFRAMES_PER_SEGMENT,
        )
        chosen = [
            (scene, scene["frames"][ordinal], False)
            for scene, count in zip(scenes, counts)
            for ordinal in _equidistant_ordinals(len(scene["frames"]), count)
        ]
    else:
        repeat_counts = _under_capacity_counts(
            all_frames, KEYFRAMES_PER_SEGMENT
        )
        scene_by_decode_index = {
            frame["decode_frame_index"]: scene
            for scene in scenes
            for frame in scene["frames"]
        }
        chosen = []
        for frame, count in zip(all_frames, repeat_counts):
            scene = scene_by_decode_index[frame["decode_frame_index"]]
            chosen.extend(
                (scene, frame, repeat > 0) for repeat in range(count)
            )

    selected: list[dict[str, Any]] = []
    previous_scene: int | None = None
    for order, (scene, frame, repeated) in enumerate(chosen, 1):
        exact_time = _frame_time(frame)
        scene_id = f"SCENE_{scene['index']:02d}"
        if order == 1:
            transition = {"type": "start", "at_s": _display_time(exact_time)}
        elif scene["index"] != previous_scene:
            transition = {
                "type": "hard_cut",
                "at_s": round(
                    float(scene["start_s"]), DISPLAY_TIME_PRECISION
                ),
            }
        else:
            transition = {"type": "continuous", "at_s": None}
        item: dict[str, Any] = {
            "order": order,
            "decode_frame_index": frame["decode_frame_index"],
            "pts": frame["pts"],
            "pts_origin": frame.get("pts_origin", 0),
            "time_base_num": frame["time_base_num"],
            "time_base_den": frame["time_base_den"],
            "source_time_s": _display_time(exact_time),
            "source_scene_id": scene_id,
            "source_scene_start_s": round(
                float(scene["start_s"]), DISPLAY_TIME_PRECISION
            ),
            "transition": transition,
            "repeated": repeated,
            "repeat_of_decode_frame_index": (
                frame["decode_frame_index"] if repeated else None
            ),
        }
        if dense_groups is not None:
            group = dense_groups[order - 1]
            group_timeline = _source_cut_timeline(group, start, end)
            item["analysis_slot"] = {
                "index": order,
                "source_scene_ids": [
                    entry["source_scene_id"] for entry in group_timeline
                ],
                "source_cut_timeline": group_timeline,
            }
        selected.append(item)
        previous_scene = int(scene["index"])
    if len(selected) != KEYFRAMES_PER_SEGMENT:
        raise AssertionError("keyframe sampler did not produce exactly nine frames")
    return selected


def materialize_keyframes(
    source: Path,
    selection: list[dict[str, Any]],
    destination: Path,
) -> list[Path]:
    """Sequentially decode global source ordinals and materialize exact-nine PNGs."""
    if len(selection) != KEYFRAMES_PER_SEGMENT:
        raise ValueError("keyframe selection must contain exactly nine frames")
    indices: list[int] = []
    for order, item in enumerate(selection, 1):
        decode_index = item.get("decode_frame_index")
        if (
            item.get("order") != order
            or isinstance(decode_index, bool)
            or not isinstance(decode_index, int)
            or decode_index < 0
        ):
            raise ValueError("keyframe selection is invalid")
        indices.append(decode_index)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MediaError("source video cannot be decoded for keyframes")
    wanted = set(indices)
    encoded: dict[int, bytes] = {}
    decode_index = 0
    try:
        while wanted - set(encoded):
            ok, frame = capture.read()
            if not ok:
                break
            if decode_index in wanted:
                success, png = cv2.imencode(".png", frame)
                if not success:
                    raise MediaError("keyframe PNG encoding failed")
                encoded[decode_index] = png.tobytes()
            decode_index += 1
    finally:
        capture.release()
    missing = sorted(wanted - set(encoded))
    if missing:
        raise MediaError(
            f"source decode inventory changed before keyframe freeze: {missing}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for order, source_index in enumerate(indices, 1):
        path = destination / f"{order:02d}.png"
        path.write_bytes(encoded[source_index])
        paths.append(path)
    return paths


def localize_dialogue(
    lines: Iterable[Mapping[str, object]],
    segment: Mapping[str, object],
    *,
    segments: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Split text by overlap ratio and shift it to segment-local time."""
    try:
        current_index = int(segment["index"])
        normalized_segments = [
            (
                int(item["index"]),
                float(item["start_s"]),
                float(item["end_s"]),
            )
            for item in segments
        ]
    except (KeyError, TypeError, ValueError):
        raise MediaError("long_video_invalid_dialogue") from None
    if (
        not normalized_segments
        or any(
            not (math.isfinite(start) and math.isfinite(end) and start < end)
            for _index, start, end in normalized_segments
        )
        or [index for index, _start, _end in normalized_segments]
        != list(range(1, len(normalized_segments) + 1))
        or abs(normalized_segments[0][1]) > _EPS
        or any(
            abs(left[2] - right[1]) > _EPS
            for left, right in zip(
                normalized_segments, normalized_segments[1:]
            )
        )
    ):
        raise MediaError("long_video_invalid_dialogue")
    try:
        current_index, start, end = next(
            item for item in normalized_segments if item[0] == current_index
        )
    except StopIteration:
        raise MediaError("long_video_invalid_dialogue") from None

    local: list[dict[str, Any]] = []
    for raw in lines:
        try:
            line_start = float(raw["start_s"])
            line_end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            raise MediaError("long_video_invalid_dialogue") from None
        if not (
            math.isfinite(line_start)
            and math.isfinite(line_end)
            and 0 <= line_start < line_end
            and line_end <= normalized_segments[-1][2] + _EPS
            and isinstance(raw.get("text"), str)
            and raw["text"]
        ):
            raise MediaError("long_video_invalid_dialogue")
        intersections = [
            (
                item_index,
                max(line_start, item_start),
                min(line_end, item_end),
            )
            for item_index, item_start, item_end in normalized_segments
            if min(line_end, item_end) > max(line_start, item_start)
        ]
        if not intersections:
            raise MediaError("long_video_invalid_dialogue")
        total_overlap = sum(
            overlap_end - overlap_start
            for _index, overlap_start, overlap_end in intersections
        )
        text = str(raw["text"])
        codepoint_start = 0
        cumulative_overlap = 0.0
        for position, (owner_index, overlap_start, overlap_end) in enumerate(
            intersections, 1
        ):
            cumulative_overlap += overlap_end - overlap_start
            codepoint_end = (
                len(text)
                if position == len(intersections)
                else min(
                    len(text),
                    max(
                        codepoint_start,
                        int(
                            math.floor(
                                len(text)
                                * cumulative_overlap
                                / total_overlap
                                + 0.5
                            )
                        ),
                    ),
                )
            )
            if owner_index != current_index or codepoint_end == codepoint_start:
                codepoint_start = codepoint_end
                continue
            item = dict(raw)
            item["text"] = text[codepoint_start:codepoint_end]
            item["start_s"] = round(
                overlap_start - start, BOUNDARY_PRECISION
            )
            item["end_s"] = round(overlap_end - start, BOUNDARY_PRECISION)
            if not 0 <= item["start_s"] < item["end_s"] <= end - start + _EPS:
                raise MediaError("long_video_invalid_dialogue")
            local.append(item)
            codepoint_start = codepoint_end
    return local


def cut_segment(
    source: Path,
    start_s: float,
    duration_s: float,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.6f}",
            "-i", str(source), "-t", f"{duration_s:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-avoid_negative_ts", "make_zero", str(destination),
        ],
        timeout=300,
    )


def mute(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-map", "0:v:0", "-c:v", "copy", "-an", str(destination),
        ]
    )


def _dialogue_line_duration(path: Path) -> float:
    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or path.suffix.lower() != ".wav"
    ):
        raise MediaError("dialogue line audio is invalid")
    return _probe_audio_duration(path)


def assemble_dialogue_track(
    lines: Iterable[Mapping[str, object]],
    total_duration_s: float,
    destination: Path,
) -> dict[str, Any]:
    """Place each synthesized line on a silent base of the full timeline."""
    if (
        isinstance(total_duration_s, bool)
        or not isinstance(total_duration_s, (int, float))
        or not math.isfinite(float(total_duration_s))
        or float(total_duration_s) <= 0
    ):
        raise MediaError("dialogue track duration is invalid")
    total = float(total_duration_s)
    placements: list[dict[str, Any]] = []
    for raw in lines:
        if not isinstance(raw, Mapping):
            raise MediaError("dialogue track line is invalid")
        try:
            start_s = float(raw["start_s"])
            end_s = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            raise MediaError("dialogue track line is invalid") from None
        if not (
            math.isfinite(start_s)
            and math.isfinite(end_s)
            and 0 <= start_s < end_s
            and end_s <= total + _EPS
        ):
            raise MediaError("dialogue track line window is invalid")
        audio = raw.get("audio")
        audio_duration = _dialogue_line_duration(audio)
        placements.append(
            {
                "audio": audio,
                "start_s": start_s,
                "end_s": end_s,
                "audio_duration_s": audio_duration,
            }
        )
    if not placements:
        raise MediaError("dialogue track has no lines")

    filters: list[str] = []
    labels: list[str] = []
    for position, placement in enumerate(placements):
        window = placement["end_s"] - placement["start_s"]
        chain = f"[{position}:a]aresample=48000"
        if placement["audio_duration_s"] > window + _EPS:
            # Speech longer than its window is compressed into it exactly.
            tempo = placement["audio_duration_s"] / window
            while tempo > 100.0:
                chain += ",atempo=100.0"
                tempo /= 100.0
            chain += f",atempo={tempo:.6f}"
            chain += f",atrim=start=0:end={window:.9f}"
        delay_ms = round(placement["start_s"] * 1000, 3)
        chain += f",asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1[line{position}]"
        filters.append(chain)
        labels.append(f"[line{position}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:normalize=0:duration=longest,"
        f"apad=whole_dur={total:.9f},atrim=start=0:end={total:.9f},"
        "asetpts=PTS-STARTPTS[out]"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "ffmpeg", "-v", "error", "-y",
            *(
                argument
                for placement in placements
                for argument in ("-i", str(placement["audio"]))
            ),
            "-filter_complex", ";".join(filters),
            "-map", "[out]",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            str(destination),
        ],
        timeout=300,
    )
    return {
        **result,
        "duration_s": total,
        "lines": [
            {
                key: value
                for key, value in placement.items()
                if key != "audio"
            }
            for placement in placements
        ],
    }


def slice_dialogue_track(
    source: Path,
    start_s: float,
    duration_s: float,
    destination: Path,
) -> dict[str, Any]:
    """Cut one segment's dialogue speech slice out of the full track."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-ss", f"{start_s:.6f}", "-t", f"{duration_s:.6f}",
            "-c:a", "pcm_s16le", str(destination),
        ],
        timeout=120,
    )


def replace_audio_track(
    video: Path,
    audio: Path,
    destination: Path,
) -> dict[str, Any]:
    """Overwrite the video's audio with the assembled dialogue track."""
    duration_s = probe(video)["duration_s"]
    audio_filter = (
        "aresample=48000,"
        f"apad=whole_dur={duration_s:.9f},atrim=start=0:end={duration_s:.9f},"
        "asetpts=PTS-STARTPTS"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".replace-audio-",
        suffix=destination.suffix or ".mp4",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        result = run_command(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(video), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-af", audio_filter,
                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", str(temporary),
            ],
            timeout=300,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {**result, "duration_s": duration_s}


def _frame_budgets(durations: Sequence[float]) -> list[int]:
    budgets: list[int] = []
    prior_total = 0
    target_total = 0.0
    for index, duration in enumerate(durations, 1):
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError(f"segment {index} duration must be finite and positive")
        target_total += float(duration)
        cumulative_frames = round(target_total * STITCH_FPS)
        frames = cumulative_frames - prior_total
        if frames < 1:
            raise ValueError(
                f"segment {index} duration is too short for {STITCH_FPS}fps"
            )
        budgets.append(frames)
        prior_total = cumulative_frames
    return budgets


def _normalize_stitch_segment(
    source: Path,
    destination: Path,
    *,
    frames: int,
    width: int,
    height: int,
    ensure_audio: bool,
) -> dict[str, Any]:
    duration_s = frames / STITCH_FPS
    video_filter = (
        "trim=start_frame=0,setpts=PTS-STARTPTS,"
        f"fps={STITCH_FPS},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"tpad=stop_mode=clone:stop_duration={duration_s + 1:.9f},"
        f"trim=end_frame={frames},setpts=N/({STITCH_FPS}*TB),format=yuv420p"
    )
    info = probe(source)
    argv = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-vf", video_filter,
        "-frames:v", str(frames), "-r", str(STITCH_FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-video_track_timescale", str(STITCH_FPS),
    ]
    if ensure_audio and info["has_audio"]:
        video_start = _stream_start_time(source, "v:0")
        audio_start = _stream_start_time(source, "a:0")
        relative_audio_start = audio_start - video_start
        if relative_audio_start >= 0:
            align = (
                "asetpts=PTS-STARTPTS,"
                f"adelay={relative_audio_start * 1000:.6f}:all=1"
            )
        else:
            align = (
                f"atrim=start={-relative_audio_start:.9f},"
                "asetpts=PTS-STARTPTS"
            )
        audio_filter = (
            f"{align},atrim=start=0:end={duration_s:.9f},"
            f"apad=whole_dur={duration_s:.9f},"
            f"atrim=start=0:end={duration_s:.9f},asetpts=PTS-STARTPTS"
        )
        argv += [
            "-map", "0:a:0", "-af", audio_filter,
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_s:.9f}",
        ]
    elif ensure_audio:
        audio_filter = (
            "anullsrc=r=48000:cl=stereo,"
            f"atrim=start=0:end={duration_s:.9f},"
            "asetpts=PTS-STARTPTS[a]"
        )
        argv += [
            "-filter_complex", audio_filter, "-map", "[a]",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-t", f"{duration_s:.9f}",
        ]
    else:
        argv += ["-an"]
    argv += ["-movflags", "+faststart", str(destination)]
    return run_command(argv, timeout=300)


def trim_24fps(
    source: Path,
    duration_s: float,
    destination: Path,
) -> dict[str, Any]:
    """Normalize one provider output to a 24fps frame budget, padding short video."""
    frames = _frame_budgets([duration_s])[0]
    info = probe(source)
    width = max(2, int(info["width"]) - int(info["width"]) % 2)
    height = max(2, int(info["height"]) - int(info["height"]) % 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _normalize_stitch_segment(
        source,
        destination,
        frames=frames,
        width=width,
        height=height,
        ensure_audio=True,
    )
    return {
        **result,
        "output_frames": frames,
        "target_duration_s": frames / STITCH_FPS,
    }


def concat(
    paths: Sequence[Path],
    destination: Path,
    target_durations: Sequence[float] | None = None,
    join_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalize and join segments using one cumulative 24fps frame budget."""
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("paths must not be empty")
    if target_durations is None:
        durations = [float(probe(path)["duration_s"]) for path in sources]
    else:
        durations = [float(value) for value in target_durations]
        if len(durations) != len(sources):
            raise ValueError("target_durations must match paths")
    modes = list(join_modes or ["hard_cut"] * len(sources))
    if (
        len(modes) != len(sources)
        or modes[0] != "hard_cut"
        or any(mode not in {"hard_cut", "continue"} for mode in modes)
    ):
        raise ValueError("join_modes are invalid")
    budgets = _frame_budgets(durations)
    first = probe(sources[0])
    width = max(2, int(first["width"]) - int(first["width"]) % 2)
    height = max(2, int(first["height"]) - int(first["height"]) % 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stitch-", dir=destination.parent
    ) as raw_tmp:
        temporary = Path(raw_tmp)
        normalized: list[Path] = []
        normalization: list[dict[str, Any]] = []
        for index, (source, frames) in enumerate(zip(sources, budgets), 1):
            output = temporary / f"segment-{index:04d}.mp4"
            normalization.append(
                _normalize_stitch_segment(
                    source,
                    output,
                    frames=frames,
                    width=width,
                    height=height,
                    ensure_audio=True,
                )
            )
            normalized.append(output)
        manifest = temporary / "concat.txt"
        manifest.write_text(
            "".join(f"file '{path.name}'\n" for path in normalized),
            encoding="utf-8",
        )
        joined = temporary / "joined.mp4"
        result = run_command(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "concat",
                "-safe", "1", "-i", manifest.name,
                "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                "-movflags", "+faststart", joined.name,
            ],
            cwd=temporary,
            timeout=300,
        )
        os.replace(joined, destination)
    return {
        **result,
        "frame_budgets": budgets,
        "join_modes": modes,
        "duration_s": sum(budgets) / STITCH_FPS,
        "normalization": normalization,
    }
