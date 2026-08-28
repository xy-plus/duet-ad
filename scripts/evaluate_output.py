#!/usr/bin/env python3
"""Score a rendered video offline and write a non-blocking JSON sidecar.

This module is intentionally outside ``app``.  It reads media files, computes
continuous measurements, and never imports generation/provider modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import vocal  # noqa: E402


ANALYSIS_WIDTH = 216
GRID_SIZE = 4
CUT_MATCH_RADIUS_S = 0.35
CUT_EXCLUSION_RADIUS_S = 0.15
MIN_CUT_SEPARATION_S = 0.5
_FFPROBE_TIMEOUT_S = 120
_RESERVED_VERDICT_KEYS = {"pass", "fail", "status", "decision"}


class EvaluationError(RuntimeError):
    """The requested offline measurement could not be computed."""


@dataclass(frozen=True)
class FrameMetric:
    """One adjacent-frame measurement at ``time_s``."""

    time_s: float
    cut_score: float
    topology_residual: float
    motion_residual: float
    scale: float | None
    scale_support: float
    scale_inliers: int
    scale_tracks: int
    topology_grid_row: int
    topology_grid_column: int
    topology_grid_score: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _number(value)
    numerator, denominator = value.split("/", 1)
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom in (None, 0.0):
        return None
    return top / bottom


def probe_media(path: Path) -> dict[str, Any]:
    """Return stable ffprobe evidence needed to interpret the scores."""
    try:
        completed = subprocess.run(
            [
                "/usr/bin/ffprobe",
                "-v",
                "error",
                "-show_entries",
                (
                    "format=duration,start_time,size,bit_rate:"
                    "stream=index,codec_type,codec_name,pix_fmt,width,height,"
                    "r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels,"
                    "start_time,duration,bit_rate"
                ),
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvaluationError("ffprobe execution failed") from exc
    if completed.returncode != 0:
        raise EvaluationError("ffprobe rejected the media input")
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvaluationError("ffprobe returned invalid JSON") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise EvaluationError("ffprobe streams are missing")
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if not isinstance(video_stream, dict):
        raise EvaluationError("video stream is missing")

    def compact_video(stream: dict[str, Any]) -> dict[str, Any]:
        return {
            "codec": stream.get("codec_name"),
            "pixel_format": stream.get("pix_fmt"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "reported_fps": _rate(stream.get("r_frame_rate")),
            "average_fps": _rate(stream.get("avg_frame_rate")),
            "frame_count": int(stream["nb_frames"])
            if str(stream.get("nb_frames", "")).isdigit()
            else None,
            "start_s": _number(stream.get("start_time")),
            "duration_s": _number(stream.get("duration")),
            "bit_rate": int(stream["bit_rate"])
            if str(stream.get("bit_rate", "")).isdigit()
            else None,
        }

    def compact_audio(stream: dict[str, Any] | None) -> dict[str, Any] | None:
        if stream is None:
            return None
        return {
            "codec": stream.get("codec_name"),
            "sample_rate": int(stream["sample_rate"])
            if str(stream.get("sample_rate", "")).isdigit()
            else None,
            "channels": stream.get("channels"),
            "start_s": _number(stream.get("start_time")),
            "duration_s": _number(stream.get("duration")),
            "bit_rate": int(stream["bit_rate"])
            if str(stream.get("bit_rate", "")).isdigit()
            else None,
        }

    media_format = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    return {
        "video": compact_video(video_stream),
        "audio": compact_audio(audio_stream),
        "container": {
            "start_s": _number(media_format.get("start_time")),
            "duration_s": _number(media_format.get("duration")),
            "size_bytes": int(media_format["size"])
            if str(media_format.get("size", "")).isdigit()
            else None,
            "bit_rate": int(media_format["bit_rate"])
            if str(media_format.get("bit_rate", "")).isdigit()
            else None,
        },
    }


def _media_timing_score(probe: dict[str, Any]) -> dict[str, float | None]:
    video = probe.get("video")
    audio = probe.get("audio")
    if not isinstance(video, dict) or not isinstance(audio, dict):
        return {"av_start_offset_ms": None, "av_end_offset_ms": None}
    video_start = _number(video.get("start_s"))
    video_duration = _number(video.get("duration_s"))
    audio_start = _number(audio.get("start_s"))
    audio_duration = _number(audio.get("duration_s"))
    if None in (video_start, video_duration, audio_start, audio_duration):
        return {"av_start_offset_ms": None, "av_end_offset_ms": None}
    assert video_start is not None and video_duration is not None
    assert audio_start is not None and audio_duration is not None
    return {
        "av_start_offset_ms": round((audio_start - video_start) * 1000, 6),
        "av_end_offset_ms": round(
            ((audio_start + audio_duration) - (video_start + video_duration)) * 1000,
            6,
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def analyze_music(path: Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Aggregate existing ``app.vocal`` YAMNet music window scores."""
    empty_score = {
        "music_score_mean": None,
        "music_score_p95": None,
        "music_score_max": None,
        "music_window_ratio_at_calibrated_floor": None,
    }
    try:
        analysis = vocal.analyze(path)
    except vocal.VocalError as exc:
        return empty_score, {
            "analyzer": "app.vocal.YAMNet",
            "music_score_floor": vocal.MUSIC_SCORE_MIN,
            "window_count": 0,
            "windows": [],
            "error": str(exc),
        }

    values = [float(window.music) for window in analysis.windows]
    score = dict(empty_score)
    if values:
        score.update(
            {
                "music_score_mean": float(np.mean(values)),
                "music_score_p95": _percentile(values, 95),
                "music_score_max": max(values),
                "music_window_ratio_at_calibrated_floor": sum(
                    value >= vocal.MUSIC_SCORE_MIN for value in values
                )
                / len(values),
            }
        )
    return score, {
        "analyzer": "app.vocal.YAMNet",
        "music_score_floor": vocal.MUSIC_SCORE_MIN,
        "window_count": len(analysis.windows),
        "windows": [
            {
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
                "music_score": float(window.music),
            }
            for window in analysis.windows
        ],
    }


def _edges(gray: np.ndarray) -> np.ndarray:
    median = float(np.median(gray))
    low = max(0, int(0.66 * median))
    high = min(255, int(1.33 * median))
    if high <= low:
        high = min(255, low + 1)
    canny = cv2.Canny(gray, low, high)
    return cv2.dilate(canny, np.ones((3, 3), np.uint8))


def _scale_estimate(previous: np.ndarray, current: np.ndarray) -> tuple[float | None, float, int, int]:
    height, width = previous.shape
    mask = np.full((height, width), 255, np.uint8)
    mask[
        int(0.22 * height):int(0.78 * height),
        int(0.22 * width):int(0.78 * width),
    ] = 0
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 6:
        return None, 0.0, 0, 0
    moved, valid, _errors = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if moved is None or valid is None:
        return None, 0.0, 0, 0
    keep = valid.reshape(-1) == 1
    before = points.reshape(-1, 2)[keep]
    after = moved.reshape(-1, 2)[keep]
    tracks = len(before)
    if tracks < 6:
        return None, 0.0, 0, tracks
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        before,
        after,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=500,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None:
        return None, 0.0, 0, tracks
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    scale = float(math.hypot(transform[0, 0], transform[0, 1]))
    if not math.isfinite(scale) or scale <= 0:
        return None, inliers / max(1, tracks), inliers, tracks
    return scale, inliers / max(1, tracks), inliers, tracks


def _decode_frame_metrics(path: Path) -> list[FrameMetric]:
    cv2.setRNGSeed(0)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise EvaluationError("video decode open failed")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise EvaluationError("video frame rate is invalid")

    rows: list[FrameMetric] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    frame_index = 0
    while True:
        decoded, frame = capture.read()
        if not decoded:
            break
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            capture.release()
            raise EvaluationError("video frame geometry is invalid")
        target_height = max(2, round(height * ANALYSIS_WIDTH / width))
        frame = cv2.resize(
            frame,
            (ANALYSIS_WIDTH, target_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]
        )
        cv2.normalize(histogram, histogram)

        if previous_gray is not None and previous_hist is not None:
            time_s = frame_index / fps
            cut_score = float(
                cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_BHATTACHARYYA)
            )
            backward_flow = cv2.calcOpticalFlowFarneback(
                gray, previous_gray, None, 0.5, 3, 15, 2, 5, 1.2, 0
            )
            grid_y, grid_x = np.mgrid[0:target_height, 0:ANALYSIS_WIDTH].astype(
                np.float32
            )
            map_x = grid_x + backward_flow[..., 0]
            map_y = grid_y + backward_flow[..., 1]
            warped_gray = cv2.remap(
                previous_gray,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            motion_residual = float(
                np.mean(cv2.absdiff(gray, warped_gray)) / 255.0
            )

            current_edges = _edges(gray)
            warped_edges = cv2.remap(
                _edges(previous_gray),
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            current_binary = current_edges > 0
            previous_binary = warped_edges > 0
            union_count = np.count_nonzero(current_binary | previous_binary)
            topology_residual = float(
                np.count_nonzero(current_binary ^ previous_binary)
                / max(1, union_count)
            )
            grid_scores: list[tuple[float, int, int]] = []
            for row in range(GRID_SIZE):
                for column in range(GRID_SIZE):
                    y0 = row * target_height // GRID_SIZE
                    y1 = (row + 1) * target_height // GRID_SIZE
                    x0 = column * ANALYSIS_WIDTH // GRID_SIZE
                    x1 = (column + 1) * ANALYSIS_WIDTH // GRID_SIZE
                    current_cell = current_binary[y0:y1, x0:x1]
                    previous_cell = previous_binary[y0:y1, x0:x1]
                    cell_union = np.count_nonzero(current_cell | previous_cell)
                    cell_score = float(
                        np.count_nonzero(current_cell ^ previous_cell)
                        / max(1, cell_union)
                    )
                    grid_scores.append((cell_score, row, column))
            topology_grid_score, topology_grid_row, topology_grid_column = max(
                grid_scores
            )
            scale, support, inliers, tracks = _scale_estimate(
                previous_gray, gray
            )
            rows.append(
                FrameMetric(
                    time_s=time_s,
                    cut_score=cut_score,
                    topology_residual=topology_residual,
                    motion_residual=motion_residual,
                    scale=scale,
                    scale_support=support,
                    scale_inliers=inliers,
                    scale_tracks=tracks,
                    topology_grid_row=topology_grid_row,
                    topology_grid_column=topology_grid_column,
                    topology_grid_score=topology_grid_score,
                )
            )
        previous_gray = gray
        previous_hist = histogram
        frame_index += 1
    capture.release()
    if not rows:
        raise EvaluationError("video has too few decoded frames")
    return rows


def _reference_hard_cuts(rows: Sequence[FrameMetric]) -> tuple[list[FrameMetric], float]:
    scores = np.asarray([row.cut_score for row in rows], dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    adaptive_floor = median + max(0.05, 6.0 * 1.4826 * mad)
    local_peaks = [
        row
        for index, row in enumerate(rows)
        if row.cut_score >= adaptive_floor
        and (index == 0 or row.cut_score >= rows[index - 1].cut_score)
        and (index == len(rows) - 1 or row.cut_score >= rows[index + 1].cut_score)
    ]
    selected: list[FrameMetric] = []
    for row in sorted(local_peaks, key=lambda item: item.cut_score, reverse=True):
        if all(
            abs(row.time_s - existing.time_s) >= MIN_CUT_SEPARATION_S
            for existing in selected
        ):
            selected.append(row)
    return sorted(selected, key=lambda item: item.time_s), adaptive_floor


def _metric_peak(
    rows: Sequence[FrameMetric], attribute: str
) -> dict[str, Any] | None:
    if not rows:
        return None
    row = max(rows, key=lambda item: float(getattr(item, attribute)))
    evidence = {
        "time_s": round(row.time_s, 6),
        "value": round(float(getattr(row, attribute)), 9),
    }
    if attribute == "topology_residual":
        evidence.update(
            {
                "grid_4x4": [
                    row.topology_grid_row,
                    row.topology_grid_column,
                ],
                "grid_value": round(row.topology_grid_score, 9),
            }
        )
    return evidence


def summarize_visual(
    reference_rows: Sequence[FrameMetric],
    candidate_rows: Sequence[FrameMetric],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Reduce frame measurements into continuous scores and trace evidence."""
    hard_cuts, adaptive_floor = _reference_hard_cuts(reference_rows)
    matches: list[dict[str, Any]] = []
    matched_candidate_times: list[float] = []
    for reference_peak in hard_cuts:
        nearby = [
            row
            for row in candidate_rows
            if abs(row.time_s - reference_peak.time_s) <= CUT_MATCH_RADIUS_S
        ]
        if not nearby:
            continue
        candidate_peak = max(nearby, key=lambda row: row.cut_score)
        offset_ms = (candidate_peak.time_s - reference_peak.time_s) * 1000
        matched_candidate_times.append(candidate_peak.time_s)
        matches.append(
            {
                "reference_peak_time_s": round(reference_peak.time_s, 6),
                "candidate_peak_time_s": round(candidate_peak.time_s, 6),
                "offset_ms": round(offset_ms, 6),
                "reference_cut_score": round(reference_peak.cut_score, 9),
                "candidate_cut_score": round(candidate_peak.cut_score, 9),
            }
        )

    within_shot = [
        row
        for row in candidate_rows
        if all(
            abs(row.time_s - cut_time) > CUT_EXCLUSION_RADIUS_S
            for cut_time in matched_candidate_times
        )
    ]
    if not within_shot:
        within_shot = list(candidate_rows)
    offsets = [abs(float(match["offset_ms"])) for match in matches]
    topology_values = [row.topology_residual for row in within_shot]
    motion_values = [row.motion_residual for row in within_shot]
    scale_values = [
        abs(math.log(row.scale)) * 100
        for row in within_shot
        if row.scale is not None and row.scale > 0
    ]

    scale_rows = [
        row for row in within_shot if row.scale is not None and row.scale > 0
    ]
    scale_peak = (
        max(scale_rows, key=lambda row: abs(math.log(float(row.scale))))
        if scale_rows
        else None
    )
    score = {
        "hard_cut_offset_ms_mean": float(np.mean(offsets)) if offsets else None,
        "hard_cut_offset_ms_max": max(offsets) if offsets else None,
        "topology_residual_p95": _percentile(topology_values, 95),
        "motion_residual_p95": _percentile(motion_values, 95),
        "scale_step_abs_log_pct_p95": _percentile(scale_values, 95),
    }
    evidence = {
        "analysis_width": ANALYSIS_WIDTH,
        "topology_grid": [GRID_SIZE, GRID_SIZE],
        "hard_cut_detection": {
            "method": "reference-hsv-bhattacharyya-local-peaks-mad-v1",
            "adaptive_floor": round(adaptive_floor, 9),
            "match_radius_s": CUT_MATCH_RADIUS_S,
            "within_shot_exclusion_radius_s": CUT_EXCLUSION_RADIUS_S,
            "reference_peak_count": len(hard_cuts),
        },
        "hard_cut_matches": matches,
        "within_shot_frame_pair_count": len(within_shot),
        "topology_peak": _metric_peak(within_shot, "topology_residual"),
        "motion_peak": _metric_peak(within_shot, "motion_residual"),
        "scale_peak": None
        if scale_peak is None
        else {
            "time_s": round(scale_peak.time_s, 6),
            "signed_step_pct": round((float(scale_peak.scale) - 1) * 100, 9),
            "abs_log_step_pct": round(
                abs(math.log(float(scale_peak.scale))) * 100, 9
            ),
            "feature_support": round(scale_peak.scale_support, 9),
            "inliers": scale_peak.scale_inliers,
            "tracks": scale_peak.scale_tracks,
        },
    }
    return score, evidence


def analyze_visual(
    reference_video: Path, candidate_video: Path
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Decode both videos and compute visual continuity scores."""
    return summarize_visual(
        _decode_frame_metrics(reference_video),
        _decode_frame_metrics(candidate_video),
    )


def evaluate(reference_video: Path, candidate_video: Path) -> dict[str, Any]:
    """Compute all offline scores without reading project state."""
    visual_score, visual_evidence = analyze_visual(
        reference_video, candidate_video
    )
    music_score, music_evidence = analyze_music(candidate_video)
    reference_probe = probe_media(reference_video)
    candidate_probe = probe_media(candidate_video)
    return {
        "score": {
            "visual_continuity": visual_score,
            "audio_music": music_score,
            "media_timing": _media_timing_score(candidate_probe),
        },
        "evidence": {
            "schema": "duet.offline-output-evaluation.v1",
            "inputs": {
                "reference_video": {
                    "path": str(reference_video),
                    "sha256": _sha256(reference_video),
                },
                "candidate_video": {
                    "path": str(candidate_video),
                    "sha256": _sha256(candidate_video),
                },
            },
            "visual_continuity": visual_evidence,
            "audio_music": music_evidence,
            "media": {
                "probe": "/usr/bin/ffprobe",
                "reference": reference_probe,
                "candidate": candidate_probe,
            },
        },
    }


def write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one explicit ``*.evaluation.json`` sidecar."""
    if not path.is_absolute() or not path.name.endswith(".evaluation.json"):
        raise ValueError("sidecar path must be absolute and end with .evaluation.json")
    if set(payload) != {"score", "evidence"}:
        raise ValueError("sidecar payload must contain only score and evidence")
    pending: list[Any] = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if _RESERVED_VERDICT_KEYS.intersection(value):
                raise ValueError("sidecar payload cannot contain verdict keys")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if not path.parent.is_dir():
        raise ValueError("sidecar parent directory must already exist")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _existing_absolute_file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    if not path.is_file():
        raise argparse.ArgumentTypeError("file does not exist")
    return path


def _absolute_sidecar(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.name.endswith(".evaluation.json"):
        raise argparse.ArgumentTypeError(
            "sidecar must be absolute and end with .evaluation.json"
        )
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError("sidecar parent directory does not exist")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute non-blocking offline video scores."
    )
    parser.add_argument(
        "--reference-video",
        required=True,
        type=_existing_absolute_file,
    )
    parser.add_argument(
        "--candidate-video",
        required=True,
        type=_existing_absolute_file,
    )
    parser.add_argument("--sidecar", required=True, type=_absolute_sidecar)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = evaluate(arguments.reference_video, arguments.candidate_video)
    write_sidecar(arguments.sidecar, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
