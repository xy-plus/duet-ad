"""Deterministic local assembly for ordered H3 segment outputs.

This module has no provider dependency.  It normalizes paid segment artifacts,
joins them, optionally restores the original source audio, validates the result,
and only then atomically replaces the conversation-level output.
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

from app import h3, storage


FPS = 24
FRAME_DURATION_S = 1 / FPS
MAX_SEGMENT_DURATION_S = 15.0
MAX_TOTAL_DURATION_S = 300.0
RECEIPT_FILENAME = "stitch-receipt.json"
_TIMEOUT_S = 300

JoinMode = Literal["continue", "hard_cut"]
AudioMode = Literal["keep", "mute", "provider_generated"]


class StitchError(RuntimeError):
    """A local probe, normalization, mux, or validation step failed."""


@dataclass(frozen=True)
class StitchSegment:
    path: Path
    target_duration_s: float
    join_mode: JoinMode
    provider_attempt_id: str | None = None
    provider_media_timeline: Mapping[str, object] | None = None


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


def _canonical_sha256(value: object) -> str:
    try:
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("provider media timeline is invalid") from None
    return hashlib.sha256(data).hexdigest()


def _provider_binding(segment: StitchSegment, index: int) -> dict[str, object]:
    attempt_id = segment.provider_attempt_id
    timeline = segment.provider_media_timeline
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
        or not isinstance(timeline, Mapping)
        or timeline.get("schema") != "duet.h3.media_timeline"
        or timeline.get("version") != 1
        or timeline.get("decode_complete") is not True
        or not isinstance(timeline.get("video"), Mapping)
        or not isinstance(timeline.get("audio"), Mapping)
    ):
        raise ValueError(
            f"segment {index} requires exact H3 native-audio evidence"
        )
    return {
        "source": "h3",
        "attempt_id": attempt_id,
        "media_timeline_sha256": _canonical_sha256(timeline),
        "decoded_audio_sha256": timeline["audio"].get("decoded_sha256"),
    }


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
        if audio_mode == "provider_generated":
            _provider_binding(segment, index + 1)
        elif (
            segment.provider_attempt_id is not None
            or segment.provider_media_timeline is not None
        ):
            raise ValueError(
                f"segment {index + 1} provider evidence requires provider_generated audio"
            )
        normalized.append(
            StitchSegment(
                path,
                duration,
                segment.join_mode,
                segment.provider_attempt_id,
                segment.provider_media_timeline,
            )
        )
    if total_duration_s > MAX_TOTAL_DURATION_S:
        raise ValueError(
            f"total target duration must not exceed {MAX_TOTAL_DURATION_S:g}s"
        )
    source = Path(source_video).resolve()
    if not source.is_file():
        raise ValueError(f"source_video does not exist: {source}")
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs = {source, *(segment.path for segment in normalized)}
    if destination in inputs:
        raise ValueError("output must not overwrite an input file")
    return tuple(normalized), source, destination


def _frame_budgets(segments: Sequence[StitchSegment]) -> list[int]:
    budgets: list[int] = []
    prior_total = 0
    target_total = 0.0
    for index, segment in enumerate(segments):
        target_total += segment.target_duration_s
        cumulative_frames = round(target_total * FPS)
        frames = cumulative_frames - prior_total
        if frames < 1:
            raise ValueError(
                f"segment {index + 1} target duration is too short for {FPS}fps output"
            )
        budgets.append(frames)
        prior_total = cumulative_frames
    return budgets


def _normalize_segment(
    segment: StitchSegment,
    destination: Path,
    frames: int,
    width: int,
    height: int,
    index: int,
    audio_mode: AudioMode,
) -> None:
    # ``continue`` describes generation continuity.  It is not proof that the
    # first decoded supplier frame duplicates the previous segment, so the EDL
    # never deletes visual content merely because the boundary is continuous.
    video_filter = (
        "trim=start_frame=0,setpts=PTS-STARTPTS,"
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
            align = (
                "asetpts=PTS-STARTPTS,"
                f"adelay={relative_audio_start * 1000:.6f}:all=1"
            )
        else:
            align = (
                f"atrim=start={-relative_audio_start:.9f},"
                "asetpts=PTS-STARTPTS"
            )
        duration_s = frames / FPS
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
    else:
        argv += ["-an"]
    argv.append(str(destination))
    _run(argv, step=f"normalizing segment {index + 1}")


def _validate_output(path: Path, expected_duration_s: float, audio_mode: str,
                     source_has_audio: bool) -> _VideoInfo:
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
    if audio_mode == "provider_generated":
        try:
            timeline = h3._probe_media_timeline(
                path,
                30,
                max_duration_s=expected_duration_s + FRAME_DURATION_S + 1e-6,
            )
        except (h3.H3Error, h3._ProbeUnavailable) as exc:
            raise StitchError("final native-audio timeline invalid") from exc
        if timeline.get("audio") is None or timeline.get("av_delta_s") is None:
            raise StitchError("final native-audio timeline invalid")
    return info


def output_is_reusable(
    *,
    segments: Sequence[StitchSegment],
    source_video: Path,
    output: Path,
    audio_mode: AudioMode,
    receipt_path: Path | None = None,
) -> bool:
    """Validate an existing output against the exact deterministic stitch input.

    In provider-generated mode this rebinds every segment to its exact H3
    attempt and media-timeline digest.  A merely playable output is never enough.
    """
    try:
        normalized, source, destination = _validate_request(
            segments, Path(source_video), Path(output), audio_mode
        )
        receipt = Path(
            receipt_path or destination.with_name(RECEIPT_FILENAME)
        ).resolve()
        if receipt.parent != destination.parent or not receipt.is_file():
            return False
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "version", "segments", "audio", "output"}
            or payload.get("schema") != "duet.stitch"
            or payload.get("version")
            != (2 if audio_mode == "provider_generated" else 1)
        ):
            return False
        budgets = _frame_budgets(normalized)
        expected_segments = [
            {
                "index": index,
                "path": str(segment.path),
                "sha256": _sha256(segment.path),
                "target_duration_s": segment.target_duration_s,
                "output_frames": budgets[index - 1],
                "join_mode": segment.join_mode,
            }
            for index, segment in enumerate(normalized, 1)
        ]
        if payload.get("segments") != expected_segments:
            return False
        source_info = _probe(source)
        expected_audio: dict[str, object] = {
            "mode": audio_mode,
            "source": str(source),
            "source_sha256": _sha256(source),
            "source_has_audio": source_info.has_audio,
        }
        if audio_mode == "provider_generated":
            expected_audio["provider_segments"] = [
                _provider_binding(segment, index)
                for index, segment in enumerate(normalized, 1)
            ]
            expected_audio["edl"] = {
                "schema": "duet.av-edl",
                "version": 1,
                "fps": FPS,
                "interval": "integer-half-open",
            }
        if payload.get("audio") != expected_audio:
            return False
        bound_output = payload.get("output")
        stat = destination.stat()
        if (
            not destination.is_file()
            or stat.st_size <= 0
            or not isinstance(bound_output, dict)
            or set(bound_output)
            != {"name", "sha256", "size", "duration_s", "fps"}
            or bound_output.get("name") != destination.name
            or bound_output.get("sha256") != _sha256(destination)
            or bound_output.get("size") != stat.st_size
            or bound_output.get("fps") != FPS
        ):
            return False
        requested_duration = sum(
            segment.target_duration_s for segment in normalized
        )
        info = _validate_output(
            destination,
            requested_duration,
            audio_mode,
            source_info.has_audio,
        )
        receipt_duration = float(bound_output.get("duration_s"))
        return (
            math.isfinite(receipt_duration)
            and abs(receipt_duration - info.duration_s) <= FRAME_DURATION_S + 1e-6
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        StitchError,
    ):
        return False


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
    must therefore be ``hard_cut``.  At a ``continue`` boundary, exactly the
    latter segment's first decoded frame is removed before 24fps conversion.
    """
    normalized, source, destination = _validate_request(
        segments, Path(source_video), Path(output), audio_mode
    )
    receipt = Path(receipt_path or destination.with_name(RECEIPT_FILENAME)).resolve()
    if receipt.parent != destination.parent:
        raise ValueError("receipt_path must be in the output directory")
    if receipt == destination:
        raise ValueError("receipt_path must differ from output")
    if receipt == source or any(receipt == segment.path for segment in normalized):
        raise ValueError("receipt_path must not overwrite an input file")
    if receipt.exists() and not receipt.is_file():
        raise ValueError("receipt_path must be a regular file or not exist")

    budgets = _frame_budgets(normalized)
    first_info = _probe(normalized[0].path)
    width = first_info.width - first_info.width % 2
    height = first_info.height - first_info.height % 2
    source_info = _probe(source)
    encoded_duration = sum(budgets) / FPS
    requested_duration = sum(segment.target_duration_s for segment in normalized)
    segment_bindings = [
        {
            "index": index,
            "path": str(segment.path),
            "sha256": _sha256(segment.path),
            "target_duration_s": segment.target_duration_s,
            "output_frames": budgets[index - 1],
            "join_mode": segment.join_mode,
        }
        for index, segment in enumerate(normalized, 1)
    ]

    with tempfile.TemporaryDirectory(prefix=".stitch-", dir=destination.parent) as raw_tmp:
        tmp = Path(raw_tmp)
        normalized_paths: list[Path] = []
        for index, (segment, frames) in enumerate(zip(normalized, budgets), 1):
            segment_output = tmp / f"segment-{index:04d}.mp4"
            _normalize_segment(
                segment,
                segment_output,
                frames,
                width,
                height,
                index - 1,
                audio_mode,
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
            "-i", concat_file.name, "-map", "0:v:0",
        ]
        if audio_mode == "provider_generated":
            concat_argv += ["-map", "0:a:0"]
        concat_argv += ["-c", "copy"]
        if audio_mode != "provider_generated":
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
            except storage.UploadError as exc:
                raise StitchError(f"source timeline probe failed: {exc}") from None
            audio_filter = (
                f"[1:a:0]asetpts=PTS-({video_start:.9f})/TB,"
                "aresample=async=1:first_pts=0,apad,"
                f"atrim=duration={encoded_duration:.9f}[a]"
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
            candidate, requested_duration, audio_mode, source_info.has_audio
        )
        output_sha = _sha256(candidate)
        output_size = candidate.stat().st_size
        payload = {
            "schema": "duet.stitch",
            "version": 2 if audio_mode == "provider_generated" else 1,
            "segments": segment_bindings,
            "audio": {
                "mode": audio_mode,
                "source": str(source),
                "source_sha256": _sha256(source),
                "source_has_audio": source_info.has_audio,
            },
            "output": {
                "name": destination.name,
                "sha256": output_sha,
                "size": output_size,
                "duration_s": final_info.duration_s,
                "fps": FPS,
            },
        }
        if audio_mode == "provider_generated":
            payload["audio"]["provider_segments"] = [
                _provider_binding(segment, index)
                for index, segment in enumerate(normalized, 1)
            ]
            payload["audio"]["edl"] = {
                "schema": "duet.av-edl",
                "version": 1,
                "fps": FPS,
                "interval": "integer-half-open",
            }
        temporary_receipt = tmp / "receipt.json"
        temporary_receipt.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(candidate, destination)
        os.replace(temporary_receipt, receipt)

    return StitchResult(destination, receipt, final_info.duration_s, output_sha, output_size)
