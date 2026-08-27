"""Deterministic local assembly for ordered H3 segment outputs.

This module has no provider dependency.  It normalizes paid segment artifacts,
joins their jointly generated A/V timelines (or a historical source-audio mode),
validates the result, and only then atomically replaces the conversation output.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from app import storage


FPS = 24
FRAME_DURATION_S = 1 / FPS
MAX_SEGMENT_DURATION_S = 15.0
MAX_TOTAL_DURATION_S = 300.0
RECEIPT_FILENAME = "stitch-receipt.json"
STITCH_RECEIPT_VERSION = 2
STITCH_ALGORITHM = "project-av-edl-v2"
_TIMEOUT_S = 300

JoinMode = Literal["continue", "hard_cut"]
AudioMode = Literal["keep", "mute", "provider_generated"]
AnchorKind = Literal["dialogue", "action"]


class StitchError(RuntimeError):
    """A local probe, normalization, mux, or validation step failed."""


@dataclass(frozen=True)
class TimelineAnchor:
    kind: AnchorKind
    source_start_s: float
    source_end_s: float
    anchor_id: str


@dataclass(frozen=True)
class ProviderMediaEvidence:
    """Exact local provider receipt selected by the upstream coordinator."""

    source: Literal["h3"]
    attempt_id: str
    receipt_path: Path
    receipt_sha256: str
    media_sha256: str
    media_size: int
    media_timeline: Mapping[str, object]


@dataclass(frozen=True)
class StitchSegment:
    path: Path
    target_duration_s: float
    join_mode: JoinMode
    source_start_s: float | None = None
    source_end_s: float | None = None
    dialogue_anchors: tuple[TimelineAnchor, ...] = ()
    action_anchors: tuple[TimelineAnchor, ...] = ()
    provider_evidence: ProviderMediaEvidence | None = None


@dataclass(frozen=True)
class StitchResult:
    output: Path
    receipt_path: Path
    duration_s: float
    sha256: str
    size: int


@dataclass(frozen=True)
class _VideoInfo:
    duration_s: float
    width: int
    height: int
    codec_name: str
    pix_fmt: str
    frame_rate: float
    has_audio: bool


@dataclass(frozen=True)
class _BoundaryDecision:
    method: str
    previous_last_sha256: str | None
    current_first_sha256: str | None
    duplicate_proven: bool
    dropped_leading_frames: int


def _clean_error(stderr: str) -> str:
    text = " ".join(stderr.strip().split())
    return text[-600:] if text else "no diagnostic output"


def _run(argv: list[str], *, cwd: Path | None = None, step: str) -> None:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except FileNotFoundError:
        raise StitchError(f"{argv[0]} not found during {step}") from None
    except subprocess.TimeoutExpired:
        raise StitchError(f"{step} timed out after {_TIMEOUT_S}s") from None
    if result.returncode != 0:
        raise StitchError(
            f"ffmpeg failed during {step}: {_clean_error(result.stderr)}"
        )


def _run_capture(argv: list[str], *, step: str, text: bool = True):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=text, timeout=_TIMEOUT_S,
        )
    except FileNotFoundError:
        raise StitchError(f"{argv[0]} not found during {step}") from None
    except subprocess.TimeoutExpired:
        raise StitchError(f"{step} timed out after {_TIMEOUT_S}s") from None
    if result.returncode != 0:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise StitchError(f"{step} failed: {_clean_error(stderr)}")
    return result.stdout


def _parse_rate(value: object) -> float:
    if not isinstance(value, str) or "/" not in value:
        raise ValueError
    numerator, denominator = value.split("/", 1)
    rate = float(numerator) / float(denominator)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError
    return rate


def _probe(path: Path) -> _VideoInfo:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,duration:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise StitchError("ffprobe not found") from None
    except subprocess.TimeoutExpired:
        raise StitchError("ffprobe timed out after 30s") from None
    if result.returncode != 0:
        raise StitchError(f"ffprobe failed for {path}: {_clean_error(result.stderr)}")
    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        duration = float(video.get("duration") or payload["format"]["duration"])
        width = int(video["width"])
        height = int(video["height"])
        codec_name = str(video["codec_name"])
        pix_fmt = str(video["pix_fmt"])
        frame_rate = _parse_rate(video["avg_frame_rate"])
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError):
        raise StitchError(f"ffprobe returned invalid video metadata for {path}") from None
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise StitchError(f"ffprobe returned invalid video metadata for {path}")
    return _VideoInfo(duration, width, height, codec_name, pix_fmt, frame_rate, has_audio)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("provider media timeline is not canonical JSON") from None
    return hashlib.sha256(encoded).hexdigest()


def validate_provider_media_evidence(
    evidence: object,
    *,
    segment_index: int,
) -> ProviderMediaEvidence:
    """Detach and validate an exact H3 attempt receipt hand-off."""
    if not isinstance(evidence, ProviderMediaEvidence):
        raise ValueError(
            f"segment {segment_index} requires exact provider media evidence"
        )
    if (
        evidence.source != "h3"
        or not isinstance(evidence.attempt_id, str)
        or len(evidence.attempt_id) != 6
        or not evidence.attempt_id.isdigit()
        or not isinstance(evidence.receipt_sha256, str)
        or len(evidence.receipt_sha256) != 64
        or not isinstance(evidence.media_sha256, str)
        or len(evidence.media_sha256) != 64
        or isinstance(evidence.media_size, bool)
        or not isinstance(evidence.media_size, int)
        or evidence.media_size <= 0
    ):
        raise ValueError(f"segment {segment_index} provider media evidence is invalid")
    receipt_path = Path(evidence.receipt_path).resolve()
    if (
        receipt_path.name != "attempt.json"
        or receipt_path.parent.name != evidence.attempt_id
        or receipt_path.parent.parent.name != "attempts"
        or receipt_path.parent.parent.parent.name != ".h3"
        or not receipt_path.is_file()
        or _sha256(receipt_path) != evidence.receipt_sha256
    ):
        raise ValueError(
            f"segment {segment_index} provider receipt binding is invalid"
        )
    try:
        timeline = json.loads(
            json.dumps(
                evidence.media_timeline,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"segment {segment_index} provider media timeline is invalid"
        ) from None
    if (
        not isinstance(timeline, dict)
        or timeline.get("schema") != "duet.h3.media_timeline"
        or timeline.get("version") != 1
        or timeline.get("decode_complete") is not True
        or not isinstance(timeline.get("video"), dict)
        or (
            timeline.get("audio") is not None
            and not isinstance(timeline.get("audio"), dict)
        )
    ):
        raise ValueError(
            f"segment {segment_index} provider media timeline is invalid"
        )
    return ProviderMediaEvidence(
        source="h3",
        attempt_id=evidence.attempt_id,
        receipt_path=receipt_path,
        receipt_sha256=evidence.receipt_sha256,
        media_sha256=evidence.media_sha256,
        media_size=evidence.media_size,
        media_timeline=timeline,
    )


def _validate_anchors(
    anchors: object,
    *,
    expected_kind: AnchorKind,
    source_start_s: float,
    source_end_s: float,
    segment_index: int,
) -> tuple[TimelineAnchor, ...]:
    if not isinstance(anchors, tuple):
        raise ValueError(f"segment {segment_index} {expected_kind}_anchors must be a tuple")
    frozen: list[TimelineAnchor] = []
    for anchor in anchors:
        if not isinstance(anchor, TimelineAnchor) or anchor.kind != expected_kind:
            raise ValueError(
                f"segment {segment_index} {expected_kind}_anchors contain invalid values"
            )
        start_s, end_s = anchor.source_start_s, anchor.source_end_s
        if (
            isinstance(start_s, bool)
            or isinstance(end_s, bool)
            or not isinstance(start_s, (int, float))
            or not isinstance(end_s, (int, float))
            or not math.isfinite(float(start_s))
            or not math.isfinite(float(end_s))
            or float(start_s) < source_start_s - 1e-6
            or float(end_s) > source_end_s + 1e-6
            or float(end_s) <= float(start_s)
            or not isinstance(anchor.anchor_id, str)
            or not anchor.anchor_id.strip()
        ):
            raise ValueError(
                f"segment {segment_index} {expected_kind} anchor is outside its source range"
            )
        frozen.append(
            TimelineAnchor(
                expected_kind, float(start_s), float(end_s), anchor.anchor_id.strip()
            )
        )
    return tuple(frozen)


def _validate_request(
    segments: Sequence[StitchSegment], source_video: Path, output: Path, audio_mode: str,
) -> tuple[tuple[StitchSegment, ...], Path, Path]:
    if audio_mode not in {"keep", "mute", "provider_generated"}:
        raise ValueError(
            "audio_mode must be 'keep', 'mute', or 'provider_generated'"
        )
    frozen = tuple(segments)
    if not frozen:
        raise ValueError("segments must not be empty")
    normalized: list[StitchSegment] = []
    total_duration_s = 0.0
    previous_source_end_s = 0.0
    for index, segment in enumerate(frozen):
        if not isinstance(segment, StitchSegment):
            raise TypeError("segments must contain StitchSegment values")
        path = Path(segment.path).resolve()
        duration = segment.target_duration_s
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"segment {index + 1} target_duration_s must be finite and positive")
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"segment {index + 1} target_duration_s must be finite and positive")
        if duration > MAX_SEGMENT_DURATION_S:
            raise ValueError(
                f"segment {index + 1} target_duration_s must not exceed "
                f"{MAX_SEGMENT_DURATION_S:g}s"
            )
        total_duration_s += duration
        if segment.join_mode not in {"continue", "hard_cut"}:
            raise ValueError(f"segment {index + 1} join_mode is invalid")
        if index == 0 and segment.join_mode != "hard_cut":
            raise ValueError("first segment join_mode must be 'hard_cut'")
        if not path.is_file():
            raise ValueError(f"segment {index + 1} does not exist: {path}")
        raw_start, raw_end = segment.source_start_s, segment.source_end_s
        if (raw_start is None) != (raw_end is None):
            raise ValueError(
                f"segment {index + 1} source_start_s and source_end_s must be supplied together"
            )
        if raw_start is None:
            source_start_s = previous_source_end_s
            source_end_s = source_start_s + duration
        else:
            if (
                isinstance(raw_start, bool)
                or isinstance(raw_end, bool)
                or not isinstance(raw_start, (int, float))
                or not isinstance(raw_end, (int, float))
            ):
                raise ValueError(f"segment {index + 1} source range must be finite")
            source_start_s, source_end_s = float(raw_start), float(raw_end)
            if not math.isfinite(source_start_s) or not math.isfinite(source_end_s):
                raise ValueError(f"segment {index + 1} source range must be finite")
        if abs(source_start_s - previous_source_end_s) > 1e-6:
            raise ValueError("source timeline must be contiguous from zero")
        source_duration_s = source_end_s - source_start_s
        if source_duration_s <= 0:
            raise ValueError(f"segment {index + 1} source range must be positive")
        if abs(source_duration_s - duration) > FRAME_DURATION_S + 1e-6:
            raise ValueError(
                f"segment {index + 1} source range does not match target duration within one frame"
            )
        dialogue = _validate_anchors(
            segment.dialogue_anchors,
            expected_kind="dialogue",
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            segment_index=index + 1,
        )
        actions = _validate_anchors(
            segment.action_anchors,
            expected_kind="action",
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            segment_index=index + 1,
        )
        provider_evidence = segment.provider_evidence
        if audio_mode == "provider_generated":
            provider_evidence = validate_provider_media_evidence(
                provider_evidence,
                segment_index=index + 1,
            )
        elif provider_evidence is not None:
            raise ValueError(
                f"segment {index + 1} provider evidence requires provider_generated audio"
            )
        normalized.append(
            StitchSegment(
                path, duration, segment.join_mode, source_start_s, source_end_s,
                dialogue, actions, provider_evidence,
            )
        )
        previous_source_end_s = source_end_s
    if total_duration_s > MAX_TOTAL_DURATION_S:
        raise ValueError(
            f"total target duration must not exceed {MAX_TOTAL_DURATION_S:g}s"
        )
    if previous_source_end_s > MAX_TOTAL_DURATION_S + 1e-6:
        raise ValueError(
            f"source EDL duration must not exceed {MAX_TOTAL_DURATION_S:g}s"
        )
    if abs(previous_source_end_s - total_duration_s) > FRAME_DURATION_S + 1e-6:
        raise ValueError(
            "cumulative source EDL and target duration differ by more than one frame"
        )
    source = Path(source_video).resolve()
    if not source.is_file():
        raise ValueError(f"source_video does not exist: {source}")
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = {source, *(segment.path for segment in normalized)}
    inputs.update(
        segment.provider_evidence.receipt_path
        for segment in normalized
        if segment.provider_evidence is not None
    )
    if destination in inputs:
        raise ValueError("output must not overwrite an input file")
    return tuple(normalized), source, destination


def _frame_budgets(segments: Sequence[StitchSegment]) -> list[int]:
    budgets: list[int] = []
    prior_total = 0
    target_total = 0.0
    for index, segment in enumerate(segments):
        target_total = (
            float(segment.source_end_s)
            if segment.source_end_s is not None
            else target_total + segment.target_duration_s
        )
        cumulative_frames = round(target_total * FPS)
        frames = cumulative_frames - prior_total
        if frames < 1:
            raise ValueError(
                f"segment {index + 1} target duration is too short for {FPS}fps output"
            )
        budgets.append(frames)
        prior_total = cumulative_frames
    return budgets


def _edge_frame_sha256(path: Path, *, last: bool) -> str:
    info = _probe(path)
    filters = "format=rgb24"
    if last:
        filters += ",reverse"
    raw = _run_capture(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-vf", filters, "-frames:v", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ],
        step=f"decoding {'last' if last else 'first'} boundary frame",
        text=False,
    )
    if len(raw) != info.width * info.height * 3:
        raise StitchError(f"could not decode boundary frame for {path}")
    identity = f"rgb24:{info.width}x{info.height}:".encode("ascii") + raw
    return hashlib.sha256(identity).hexdigest()


def _boundary_decisions(
    segments: Sequence[StitchSegment], *, preserve_joint_av: bool = False,
) -> list[_BoundaryDecision]:
    decisions: list[_BoundaryDecision] = []
    for index, segment in enumerate(segments):
        if index == 0 or segment.join_mode == "hard_cut":
            decisions.append(_BoundaryDecision("none", None, None, False, 0))
            continue
        previous_sha = _edge_frame_sha256(segments[index - 1].path, last=True)
        current_sha = _edge_frame_sha256(segment.path, last=False)
        duplicate = previous_sha == current_sha
        if preserve_joint_av:
            decisions.append(
                _BoundaryDecision(
                    "joint-av-preserve-v1",
                    previous_sha,
                    current_sha,
                    duplicate,
                    0,
                )
            )
            continue
        decisions.append(
            _BoundaryDecision(
                    "decoded-rgb24-full-frame-exact-v2",
                previous_sha,
                current_sha,
                duplicate,
                1 if duplicate else 0,
            )
        )
    return decisions


def _ffmpeg_version() -> str:
    stdout = _run_capture(["ffmpeg", "-version"], step="reading FFmpeg version")
    first = stdout.splitlines()[0].strip() if stdout else ""
    if not first.startswith("ffmpeg version "):
        raise StitchError("ffmpeg returned an invalid version string")
    return first


def _optional_float(value: object) -> float | None:
    if value in {None, "N/A"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_monotonic(values: Sequence[float]) -> bool:
    return all(right + 1e-9 >= left for left, right in zip(values, values[1:]))


def _stream_timeline(path: Path, selector: str) -> dict | None:
    stdout = _run_capture(
        [
            "ffprobe", "-v", "error", "-select_streams", selector,
            "-show_streams", "-show_packets", "-show_frames", "-show_entries",
            "stream=codec_type,codec_name,start_time,time_base,duration,"
            "sample_rate,channels,avg_frame_rate,r_frame_rate:"
            "packet=pts_time,dts_time,duration_time:"
            "frame=best_effort_timestamp_time,pkt_dts_time,pkt_duration_time",
            "-of", "json", str(path),
        ],
        step=f"probing {selector} timeline for {path}",
    )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        raise StitchError(f"ffprobe returned invalid {selector} timeline for {path}") from None
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    combined = payload.get("packets_and_frames")
    if isinstance(combined, list):
        packets = [
            item for item in combined
            if isinstance(item, dict) and item.get("type") == "packet"
        ]
        frames = [
            item for item in combined
            if isinstance(item, dict) and item.get("type") == "frame"
        ]
    else:
        packets = payload.get("packets")
        frames = payload.get("frames")
    if (
        not isinstance(stream, dict)
        or not isinstance(packets, list)
        or not isinstance(frames, list)
    ):
        raise StitchError(f"ffprobe returned invalid {selector} timeline for {path}")
    frame_rows = [
        (
            _optional_float(frame.get("best_effort_timestamp_time")),
            _optional_float(frame.get("pkt_dts_time")),
            _optional_float(frame.get("pkt_duration_time")),
        )
        for frame in frames
        if isinstance(frame, dict)
    ]
    frame_pts = [item[0] for item in frame_rows if item[0] is not None]
    frame_dts = [item[1] for item in frame_rows if item[1] is not None]
    packet_rows = [
        (
            _optional_float(packet.get("pts_time")),
            _optional_float(packet.get("dts_time")),
            _optional_float(packet.get("duration_time")),
        )
        for packet in packets
        if isinstance(packet, dict)
    ]
    packet_pts = [item[0] for item in packet_rows if item[0] is not None]
    packet_dts = [item[1] for item in packet_rows if item[1] is not None]
    if not frame_pts or not packet_rows or not packet_dts:
        raise StitchError(f"ffprobe found no decoded {selector} timestamps for {path}")
    duration_s = _optional_float(stream.get("duration"))
    frame_end_s = max(
        pts + (duration or 0.0)
        for pts, _dts, duration in frame_rows
        if pts is not None
    )
    packet_end_s = max(
        pts + (duration or 0.0)
        for pts, _dts, duration in packet_rows
        if pts is not None
    ) if packet_pts else None
    if duration_s is None:
        duration_s = frame_end_s - frame_pts[0]
    return {
        "codec_name": stream.get("codec_name"),
        "start_s": _optional_float(stream.get("start_time")),
        "duration_s": duration_s,
        "time_base": stream.get("time_base"),
        "sample_rate": (
            int(stream["sample_rate"]) if stream.get("sample_rate") else None
        ),
        "channels": int(stream["channels"]) if stream.get("channels") else None,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "first_pts_s": frame_pts[0],
        "last_pts_s": frame_pts[-1],
        "frame_end_s": frame_end_s,
        "first_dts_s": packet_dts[0],
        "last_dts_s": packet_dts[-1],
        "pts_monotonic": _is_monotonic(frame_pts),
        "dts_monotonic": _is_monotonic(packet_dts),
        "decoded_units": len(frame_pts),
        "packet_count": len(packet_rows),
        "first_packet_pts_s": min(packet_pts) if packet_pts else None,
        "last_packet_pts_s": max(packet_pts) if packet_pts else None,
        "packet_end_s": packet_end_s,
        "packet_pts_monotonic": _is_monotonic(packet_pts),
        "packet_dts_monotonic": _is_monotonic(packet_dts),
        "decoded_frame_dts_monotonic": _is_monotonic(frame_dts),
    }


def _decoded_audio_sha256(path: Path, *, has_audio: bool) -> str | None:
    if not has_audio:
        return None
    stdout = _run_capture(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-vn", "-sn", "-dn", "-c:a", "pcm_s16le", "-f", "hash",
            "-hash", "sha256", "-",
        ],
        step=f"hashing decoded audio for {path}",
    ).strip()
    prefix = "SHA256="
    if not stdout.startswith(prefix) or len(stdout[len(prefix):]) != 64:
        raise StitchError(f"ffmpeg returned invalid audio hash for {path}")
    return stdout[len(prefix):]


def _audio_binding(path: Path, info: _VideoInfo) -> dict:
    timeline = _stream_timeline(path, "a:0") if info.has_audio else None
    return {
        "container_sha256": _sha256(path),
        "stream_sha256": _decoded_audio_sha256(path, has_audio=info.has_audio),
        "duration_s": timeline.get("duration_s") if timeline else None,
        "timeline": timeline,
    }


def _av_delta(video: dict | None, audio: dict | None) -> dict | None:
    if video is None or audio is None:
        return None
    video_start = float(video["first_pts_s"])
    audio_start = float(audio["first_pts_s"])
    video_end = float(video["frame_end_s"])
    audio_end = float(audio["frame_end_s"])
    return {"start": audio_start - video_start, "end": audio_end - video_end}


def _validate_provider_joint_av(path: Path, video: dict | None, audio: dict | None) -> None:
    if video is None or audio is None:
        raise StitchError("provider_generated_audio_missing")
    if (
        video.get("pts_monotonic") is not True
        or video.get("dts_monotonic") is not True
        or audio.get("pts_monotonic") is not True
        or audio.get("dts_monotonic") is not True
    ):
        raise StitchError(f"provider_generated_av_timeline_invalid: {path}")
    delta = _av_delta(video, audio)
    assert delta is not None
    if abs(delta["start"]) > 0.1 or abs(delta["end"]) > 0.1:
        raise StitchError(f"provider_generated_av_timeline_invalid: {path}")


def _close_timeline_value(left: object, right: object, tolerance_s: float) -> bool:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    return (
        left_value is not None
        and right_value is not None
        and abs(left_value - right_value) <= tolerance_s
    )


def _provider_evidence_binding(
    segment: StitchSegment,
    video: dict | None,
    audio: dict | None,
    decoded_audio_sha256: str | None,
) -> dict:
    evidence = segment.provider_evidence
    if evidence is None or video is None or audio is None:
        raise StitchError("provider_generated_receipt_missing")
    timeline = evidence.media_timeline
    upstream_video = timeline.get("video")
    upstream_audio = timeline.get("audio")
    if not isinstance(upstream_video, dict) or not isinstance(upstream_audio, dict):
        raise StitchError("provider_generated_receipt_invalid")
    if (
        upstream_video.get("presentation_monotonic") is not True
        or upstream_video.get("packet_dts_monotonic") is not True
        or upstream_audio.get("presentation_monotonic") is not True
        or upstream_audio.get("packet_dts_monotonic") is not True
        or upstream_video.get("time_base") != video.get("time_base")
        or upstream_audio.get("time_base") != audio.get("time_base")
        or upstream_video.get("frame_count") != video.get("decoded_units")
        or upstream_audio.get("frame_count") != audio.get("decoded_units")
        or upstream_video.get("packet_count") != video.get("packet_count")
        or upstream_audio.get("packet_count") != audio.get("packet_count")
        or upstream_audio.get("decoded_sha256") != decoded_audio_sha256
        or not _close_timeline_value(
            upstream_video.get("first_frame_pts_s"), video.get("first_pts_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_video.get("last_frame_pts_s"), video.get("last_pts_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_audio.get("first_frame_pts_s"), audio.get("first_pts_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_audio.get("last_frame_pts_s"), audio.get("last_pts_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_video.get("frame_end_s"), video.get("frame_end_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_audio.get("frame_end_s"), audio.get("frame_end_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_video.get("first_packet_pts_s"),
            video.get("first_packet_pts_s"),
            1e-6,
        )
        or not _close_timeline_value(
            upstream_video.get("last_packet_pts_s"),
            video.get("last_packet_pts_s"),
            1e-6,
        )
        or not _close_timeline_value(
            upstream_audio.get("first_packet_pts_s"),
            audio.get("first_packet_pts_s"),
            1e-6,
        )
        or not _close_timeline_value(
            upstream_audio.get("last_packet_pts_s"),
            audio.get("last_packet_pts_s"),
            1e-6,
        )
        or not _close_timeline_value(
            upstream_video.get("packet_end_s"), video.get("packet_end_s"), 1e-6
        )
        or not _close_timeline_value(
            upstream_audio.get("packet_end_s"), audio.get("packet_end_s"), 1e-6
        )
    ):
        raise StitchError("provider_generated_receipt_mismatch")
    if _sha256(evidence.receipt_path) != evidence.receipt_sha256:
        raise StitchError("provider_generated_receipt_mismatch")
    if (
        _sha256(segment.path) != evidence.media_sha256
        or segment.path.stat().st_size != evidence.media_size
    ):
        raise StitchError("provider_generated_receipt_mismatch")
    return {
        "source": evidence.source,
        "attempt_id": evidence.attempt_id,
        "path": str(evidence.receipt_path),
        "sha256": evidence.receipt_sha256,
        "media_sha256": evidence.media_sha256,
        "media_size": evidence.media_size,
        "media_timeline_sha256": _canonical_json_sha256(timeline),
        "media_timeline": timeline,
    }


def _validate_full_decode(path: Path) -> None:
    _run(
        [
            "ffmpeg", "-v", "error", "-xerror", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-",
        ],
        step=f"fully decoding media {path}",
    )


def receipt_is_v2(path: Path) -> bool:
    """Return true only for the current local-stitch receipt schema.

    Version 1 receipts remain readable JSON artifacts for audit, but they do
    not prove the EDL, toolchain, decoded timestamps, or audio-master binding
    required for reuse.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == {
            "schema", "version", "algorithm", "toolchain", "edl",
            "segments", "audio", "output",
        }
        and payload.get("schema") == "duet.stitch"
        and payload.get("version") == STITCH_RECEIPT_VERSION
        and payload.get("algorithm") == STITCH_ALGORITHM
    )


def stitched_output_is_reusable(
    *,
    segments: Sequence[StitchSegment],
    source_video: Path,
    output: Path,
    audio_mode: AudioMode,
    receipt_path: Path | None = None,
) -> bool:
    """Validate a v2 receipt and every local artifact without provider access."""
    try:
        normalized, source, destination = _validate_request(
            segments, Path(source_video), Path(output), audio_mode
        )
        receipt = Path(
            receipt_path or destination.with_name(RECEIPT_FILENAME)
        ).resolve()
        if not destination.is_file() or not receipt_is_v2(receipt):
            return False
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("toolchain") != {"ffmpeg_version": _ffmpeg_version()}:
            return False
        budgets = _frame_budgets(normalized)
        decisions = _boundary_decisions(
            normalized, preserve_joint_av=audio_mode == "provider_generated"
        )
        edl = payload.get("edl")
        if not isinstance(edl, dict):
            return False
        if set(edl) != {
            "master_clock", "fps", "total_frames", "total_duration_s",
            "source", "entries",
        }:
            return False
        source_info = _probe(source)
        _validate_full_decode(source)
        source_video_timeline = _stream_timeline(source, "v:0")
        expected_master = (
            "provider_generated_av"
            if audio_mode == "provider_generated"
            else "source_audio"
            if audio_mode == "keep" and source_info.has_audio
            else "source_video_timeline"
        )
        if (
            edl.get("master_clock") != expected_master
            or edl.get("fps") != FPS
            or edl.get("total_frames") != sum(budgets)
            or not _finite_number(edl.get("total_duration_s"))
            or abs(float(edl["total_duration_s"]) - sum(budgets) / FPS) > 1e-9
            or edl.get("source") != {
                "path": str(source),
                "sha256": _sha256(source),
                "video_timeline": source_video_timeline,
            }
        ):
            return False
        entries = edl.get("entries")
        bindings = payload.get("segments")
        if (
            not isinstance(entries, list)
            or len(entries) != len(normalized)
            or not isinstance(bindings, list)
            or len(bindings) != len(normalized)
        ):
            return False
        frame_cursor = 0
        for index, (segment, frames, decision, entry, binding) in enumerate(
            zip(normalized, budgets, decisions, entries, bindings), 1
        ):
            expected_boundary = {
                "method": decision.method,
                "previous_last_sha256": decision.previous_last_sha256,
                "current_first_sha256": decision.current_first_sha256,
                "duplicate_proven": decision.duplicate_proven,
                "dropped_leading_frames": decision.dropped_leading_frames,
            }
            expected_dialogue = [
                {
                    "kind": anchor.kind,
                    "source_start_s": anchor.source_start_s,
                    "source_end_s": anchor.source_end_s,
                    "anchor_id": anchor.anchor_id,
                }
                for anchor in segment.dialogue_anchors
            ]
            expected_actions = [
                {
                    "kind": anchor.kind,
                    "source_start_s": anchor.source_start_s,
                    "source_end_s": anchor.source_end_s,
                    "anchor_id": anchor.anchor_id,
                }
                for anchor in segment.action_anchors
            ]
            expected_entry = {
                "index": index,
                "source_range_s": {
                    "start": segment.source_start_s,
                    "end": segment.source_end_s,
                },
                "target_frame_range": {
                    "start": frame_cursor,
                    "end": frame_cursor + frames,
                },
                "target_duration_s": segment.target_duration_s,
                "join_mode": segment.join_mode,
                "boundary": expected_boundary,
                "dialogue_anchors": expected_dialogue,
                "action_anchors": expected_actions,
            }
            provider_info = _probe(segment.path)
            if audio_mode == "provider_generated" and not provider_info.has_audio:
                return False
            provider_video = _stream_timeline(segment.path, "v:0")
            provider_audio = _audio_binding(segment.path, provider_info)
            _validate_full_decode(segment.path)
            if audio_mode == "provider_generated":
                _validate_provider_joint_av(
                    segment.path, provider_video, provider_audio["timeline"]
                )
                upstream_receipt = _provider_evidence_binding(
                    segment,
                    provider_video,
                    provider_audio["timeline"],
                    provider_audio["stream_sha256"],
                )
            else:
                upstream_receipt = None
            expected_binding = {
                "index": index,
                "path": str(segment.path),
                "sha256": _sha256(segment.path),
                "target_duration_s": segment.target_duration_s,
                "output_frames": frames,
                "join_mode": segment.join_mode,
                "provider_media": {
                    "video": provider_video,
                    "audio": provider_audio,
                    "av_delta_s": _av_delta(
                        provider_video, provider_audio["timeline"]
                    ),
                },
                "upstream_receipt": upstream_receipt,
            }
            if entry != expected_entry or binding != expected_binding:
                return False
            frame_cursor += frames
        audio = payload.get("audio")
        if (
            not isinstance(audio, dict)
            or set(audio) != {
                "mode", "master", "source_role", "time_stretch_applied",
                "source_path", "source_has_audio", "source", "providers",
                "final",
            }
            or audio.get("mode") != audio_mode
            or audio.get("master")
            != (
                "provider_segments"
                if audio_mode == "provider_generated"
                else "source"
                if audio_mode == "keep"
                else "none"
            )
            or audio.get("source_role")
            != (
                "upstream_h3_material_only"
                if audio_mode == "provider_generated"
                else "final_audio_source"
                if audio_mode == "keep"
                else "provenance_only"
            )
            or audio.get("time_stretch_applied") is not False
            or audio.get("source_path") != str(source)
            or audio.get("source_has_audio") is not source_info.has_audio
            or audio.get("source") != _audio_binding(source, source_info)
        ):
            return False
        expected_providers = [
            {"index": item["index"], **item["provider_media"]["audio"]}
            for item in bindings
        ]
        if audio.get("providers") != expected_providers:
            return False
        requested_duration = sum(budgets) / FPS
        final_info = _validate_output(
            destination, requested_duration, audio_mode, source_info.has_audio
        )
        output_binding = payload.get("output")
        final_audio = _audio_binding(destination, final_info)
        if (
            not isinstance(output_binding, dict)
            or set(output_binding) != {
                "name", "sha256", "size", "duration_s", "fps",
                "video_timeline", "audio_timeline",
            }
            or output_binding.get("name") != destination.name
            or output_binding.get("sha256") != _sha256(destination)
            or output_binding.get("size") != destination.stat().st_size
            or output_binding.get("fps") != FPS
            or not _finite_number(output_binding.get("duration_s"))
            or abs(float(output_binding["duration_s"]) - final_info.duration_s) > 1e-6
            or output_binding.get("video_timeline")
            != _stream_timeline(destination, "v:0")
            or output_binding.get("audio_timeline")
            != (final_audio["timeline"] if final_info.has_audio else None)
            or audio.get("final") != (final_audio if final_info.has_audio else None)
        ):
            return False
        return True
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        StitchError,
    ):
        return False


def _normalize_segment(
    segment: StitchSegment,
    destination: Path,
    frames: int,
    width: int,
    height: int,
    index: int,
    drop_leading_frames: int,
    audio_mode: AudioMode,
) -> None:
    # Trim before fps so a receipt-proven duplicate removes exactly one decoded
    # supplier frame.  ``continue`` alone never authorizes frame deletion.
    video_filter = (
        f"trim=start_frame={drop_leading_frames},setpts=PTS-STARTPTS,"
        f"fps={FPS},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"tpad=stop_mode=clone:stop_duration={frames / FPS + 1:.9f},"
        f"trim=end_frame={frames},setpts=N/({FPS}*TB),format=yuv420p"
    )
    argv = [
        "ffmpeg", "-v", "error", "-y", "-i", str(segment.path),
        "-map", "0:v:0", "-vf", video_filter,
        "-frames:v", str(frames), "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-video_track_timescale", str(FPS),
    ]
    if audio_mode == "provider_generated":
        try:
            video_start = storage.probe_stream_start_time(segment.path, "v:0")
            audio_start = storage.probe_stream_start_time(segment.path, "a:0")
        except storage.UploadError as exc:
            raise StitchError(
                f"provider_generated_audio_missing: segment {index + 1}: {exc}"
            ) from None
        relative_audio_start = audio_start - video_start
        if relative_audio_start >= 0:
            timeline_filter = (
                "asetpts=PTS-STARTPTS,"
                f"adelay={relative_audio_start * 1000:.6f}:all=1"
            )
        else:
            timeline_filter = (
                f"atrim=start={-relative_audio_start:.9f},asetpts=PTS-STARTPTS"
            )
        target_duration_s = frames / FPS
        audio_filter = (
            f"{timeline_filter},atrim=start=0:end={target_duration_s:.9f},"
            f"apad=whole_dur={target_duration_s:.9f}"
        )
        argv += [
            "-map", "0:a:0", "-af", audio_filter,
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-t", f"{target_duration_s:.9f}",
        ]
    else:
        argv += ["-an"]
    argv.append(str(destination))
    _run(argv, step=f"normalizing segment {index + 1}")


def _validate_output(path: Path, expected_duration_s: float, audio_mode: str,
                     source_has_audio: bool) -> _VideoInfo:
    _validate_full_decode(path)
    info = _probe(path)
    if info.codec_name != "h264" or info.pix_fmt != "yuv420p":
        raise StitchError("final video is not H.264/yuv420p")
    if abs(info.frame_rate - FPS) > 0.001:
        raise StitchError(f"final video frame rate is {info.frame_rate}, expected {FPS}")
    if abs(info.duration_s - expected_duration_s) > FRAME_DURATION_S + 1e-6:
        raise StitchError(
            f"final video duration {info.duration_s:.6f}s differs from "
            f"target {expected_duration_s:.6f}s by more than one frame"
        )
    expected_audio = (
        audio_mode == "provider_generated"
        or (audio_mode == "keep" and source_has_audio)
    )
    if info.has_audio != expected_audio:
        raise StitchError("final audio streams do not match requested audio strategy")
    video_timeline = _stream_timeline(path, "v:0")
    if (
        video_timeline is None
        or video_timeline["pts_monotonic"] is not True
        or video_timeline["dts_monotonic"] is not True
        or video_timeline["decoded_units"] != round(expected_duration_s * FPS)
        or abs(float(video_timeline["first_pts_s"])) > FRAME_DURATION_S + 1e-6
    ):
        raise StitchError("final video PTS/DTS or decoded frame count is invalid")
    if expected_audio:
        audio_timeline = _stream_timeline(path, "a:0")
        if (
            audio_timeline is None
            or audio_timeline["pts_monotonic"] is not True
            or audio_timeline["dts_monotonic"] is not True
        ):
            raise StitchError("final audio PTS/DTS is invalid")
        audio_end_s = float(audio_timeline["frame_end_s"])
        if abs(audio_end_s - expected_duration_s) > FRAME_DURATION_S + 0.03:
            raise StitchError("final audio timeline differs from master clock")
        final_delta = _av_delta(video_timeline, audio_timeline)
        if final_delta is None or any(
            abs(float(final_delta[edge])) > FRAME_DURATION_S + 1e-6
            for edge in ("start", "end")
        ):
            raise StitchError("final A/V PTS differs by more than one frame")
    return info


def stitch_video(
    *,
    segments: Sequence[StitchSegment],
    source_video: Path,
    output: Path,
    audio_mode: AudioMode,
    receipt_path: Path | None = None,
) -> StitchResult:
    """Assemble local segment files and atomically publish a validated MP4.

    ``join_mode`` describes the boundary before each segment.  The first segment
    must therefore be ``hard_cut``.  A ``continue`` boundary only removes the
    latter segment's first decoded frame when local decoded-frame evidence proves
    that it duplicates the previous segment's last frame.
    """
    normalized, source, destination = _validate_request(
        segments, Path(source_video), Path(output), audio_mode
    )
    receipt = Path(receipt_path or destination.with_name(RECEIPT_FILENAME)).resolve()
    if receipt.parent != destination.parent:
        raise ValueError("receipt_path must be in the output directory")
    if receipt == destination:
        raise ValueError("receipt_path must differ from output")
    if receipt == source or any(
        receipt == segment.path
        or (
            segment.provider_evidence is not None
            and receipt == segment.provider_evidence.receipt_path
        )
        for segment in normalized
    ):
        raise ValueError("receipt_path must not overwrite an input file")
    if receipt.exists() and not receipt.is_file():
        raise ValueError("receipt_path must be a regular file or not exist")

    budgets = _frame_budgets(normalized)
    first_info = _probe(normalized[0].path)
    width = first_info.width - first_info.width % 2
    height = first_info.height - first_info.height % 2
    source_info = _probe(source)
    _validate_full_decode(source)
    source_video_timeline = _stream_timeline(source, "v:0")
    provider_infos = [_probe(segment.path) for segment in normalized]
    if (
        audio_mode == "provider_generated"
        and any(not info.has_audio for info in provider_infos)
    ):
        raise StitchError("provider_generated_audio_missing")
    source_end_s = normalized[-1].source_end_s
    assert source_end_s is not None
    if source_end_s > source_info.duration_s + FRAME_DURATION_S + 1e-6:
        raise StitchError("EDL exceeds the source master timeline")
    encoded_duration = sum(budgets) / FPS
    decisions = _boundary_decisions(
        normalized, preserve_joint_av=audio_mode == "provider_generated"
    )
    frame_cursor = 0
    edl_entries = []
    for index, (segment, frames, decision) in enumerate(
        zip(normalized, budgets, decisions), 1
    ):
        entry = {
            "index": index,
            "source_range_s": {
                "start": segment.source_start_s,
                "end": segment.source_end_s,
            },
            "target_frame_range": {"start": frame_cursor, "end": frame_cursor + frames},
            "target_duration_s": segment.target_duration_s,
            "join_mode": segment.join_mode,
            "boundary": {
                "method": decision.method,
                "previous_last_sha256": decision.previous_last_sha256,
                "current_first_sha256": decision.current_first_sha256,
                "duplicate_proven": decision.duplicate_proven,
                "dropped_leading_frames": decision.dropped_leading_frames,
            },
            "dialogue_anchors": [
                {
                    "kind": anchor.kind,
                    "source_start_s": anchor.source_start_s,
                    "source_end_s": anchor.source_end_s,
                    "anchor_id": anchor.anchor_id,
                }
                for anchor in segment.dialogue_anchors
            ],
            "action_anchors": [
                {
                    "kind": anchor.kind,
                    "source_start_s": anchor.source_start_s,
                    "source_end_s": anchor.source_end_s,
                    "anchor_id": anchor.anchor_id,
                }
                for anchor in segment.action_anchors
            ],
        }
        edl_entries.append(entry)
        frame_cursor += frames
    segment_bindings = []
    for index, (segment, provider_info) in enumerate(
        zip(normalized, provider_infos), 1
    ):
        provider_video = _stream_timeline(segment.path, "v:0")
        provider_audio = _audio_binding(segment.path, provider_info)
        _validate_full_decode(segment.path)
        if audio_mode == "provider_generated":
            _validate_provider_joint_av(
                segment.path, provider_video, provider_audio["timeline"]
            )
            upstream_receipt = _provider_evidence_binding(
                segment,
                provider_video,
                provider_audio["timeline"],
                provider_audio["stream_sha256"],
            )
        else:
            upstream_receipt = None
        segment_bindings.append({
            "index": index,
            "path": str(segment.path),
            "sha256": _sha256(segment.path),
            "target_duration_s": segment.target_duration_s,
            "output_frames": budgets[index - 1],
            "join_mode": segment.join_mode,
            "provider_media": {
                "video": provider_video,
                "audio": provider_audio,
                "av_delta_s": _av_delta(
                    provider_video, provider_audio["timeline"]
                ),
            },
            "upstream_receipt": upstream_receipt,
        })

    with tempfile.TemporaryDirectory(prefix=".stitch-", dir=destination.parent) as raw_tmp:
        tmp = Path(raw_tmp)
        normalized_paths: list[Path] = []
        for index, (segment, frames, decision) in enumerate(
            zip(normalized, budgets, decisions), 1
        ):
            segment_output = tmp / f"segment-{index:04d}.mp4"
            _normalize_segment(
                segment, segment_output, frames, width, height, index - 1,
                decision.dropped_leading_frames, audio_mode,
            )
            normalized_paths.append(segment_output)

        concat_file = tmp / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{path.name}'\n" for path in normalized_paths),
            encoding="utf-8",
        )
        joined = tmp / "joined.mp4"
        concat_argv = [
                "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1",
                "-i", concat_file.name, "-map", "0:v:0", "-c:v", "copy",
        ]
        if audio_mode == "provider_generated":
            concat_argv += ["-map", "0:a:0", "-c:a", "copy"]
        else:
            concat_argv += ["-an"]
        concat_argv += ["-movflags", "+faststart", joined.name]
        _run(
            concat_argv,
            cwd=tmp,
            step="concatenating normalized segments",
        )

        candidate = joined
        if audio_mode == "keep" and source_info.has_audio:
            candidate = tmp / "candidate.mp4"
            try:
                video_start = storage.probe_stream_start_time(source, "v:0")
                audio_start = storage.probe_stream_start_time(source, "a:0")
            except storage.UploadError as exc:
                raise StitchError(f"source timeline probe failed: {exc}") from None
            relative_audio_start = audio_start - video_start
            if relative_audio_start >= 0:
                audio_timeline_filter = (
                    "asetpts=PTS-STARTPTS,"
                    f"adelay={relative_audio_start * 1000:.6f}:all=1"
                )
            else:
                audio_timeline_filter = (
                    f"atrim=start={-relative_audio_start:.9f},asetpts=PTS-STARTPTS"
                )
            audio_filter = (
                f"[1:a:0]{audio_timeline_filter},"
                f"atrim=start=0:end={encoded_duration:.9f},"
                f"apad=whole_dur={encoded_duration:.9f}[a]"
            )
            _run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(joined),
                    "-i", str(source), "-filter_complex", audio_filter,
                    "-map", "0:v:0", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-t", f"{encoded_duration:.9f}",
                    "-movflags", "+faststart", str(candidate),
                ],
                step="restoring source audio",
            )

        final_info = _validate_output(
            candidate, encoded_duration, audio_mode, source_info.has_audio
        )
        output_sha = _sha256(candidate)
        output_size = candidate.stat().st_size
        output_video_timeline = _stream_timeline(candidate, "v:0")
        output_audio_binding = _audio_binding(candidate, final_info)
        source_audio_binding = _audio_binding(source, source_info)
        payload = {
            "schema": "duet.stitch",
            "version": STITCH_RECEIPT_VERSION,
            "algorithm": STITCH_ALGORITHM,
            "toolchain": {"ffmpeg_version": _ffmpeg_version()},
            "edl": {
                "master_clock": (
                    "provider_generated_av"
                    if audio_mode == "provider_generated"
                    else "source_audio"
                    if audio_mode == "keep" and source_info.has_audio
                    else "source_video_timeline"
                ),
                "fps": FPS,
                "total_frames": sum(budgets),
                "total_duration_s": encoded_duration,
                "source": {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "video_timeline": source_video_timeline,
                },
                "entries": edl_entries,
            },
            "segments": segment_bindings,
            "audio": {
                "mode": audio_mode,
                "master": (
                    "provider_segments"
                    if audio_mode == "provider_generated"
                    else "source"
                    if audio_mode == "keep"
                    else "none"
                ),
                "source_role": (
                    "upstream_h3_material_only"
                    if audio_mode == "provider_generated"
                    else "final_audio_source"
                    if audio_mode == "keep"
                    else "provenance_only"
                ),
                "time_stretch_applied": False,
                "source_path": str(source),
                "source_has_audio": source_info.has_audio,
                "source": source_audio_binding,
                "providers": [
                    {
                        "index": item["index"],
                        **item["provider_media"]["audio"],
                    }
                    for item in segment_bindings
                ],
                "final": output_audio_binding if final_info.has_audio else None,
            },
            "output": {
                "name": destination.name,
                "sha256": output_sha,
                "size": output_size,
                "duration_s": final_info.duration_s,
                "fps": FPS,
                "video_timeline": output_video_timeline,
                "audio_timeline": (
                    output_audio_binding["timeline"] if final_info.has_audio else None
                ),
            },
        }
        temporary_receipt = tmp / "receipt.json"
        temporary_receipt.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(candidate, destination)
        os.replace(temporary_receipt, receipt)

    return StitchResult(destination, receipt, final_info.duration_s, output_sha, output_size)
