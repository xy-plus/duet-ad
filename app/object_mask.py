"""Deterministic gray-block masking of replaceable objects in the H3 reference video.

Mirrors app/face_mask.py: the OR-flagged objects must disappear from the
reference video so H3 can only rebuild them from the supplied user images.
The fal SAM-3 detector (app/object_detect.py) locates each target at
sparse timestamps; the gaps between anchor points are filled locally by
linear interpolation, and a track that loses its object keeps painting the
last known box for OBJECT_MASK_HOLD_FRAMES frames before disappearing.
The block is a solid deep gray (64,64,64), deliberately distinct from the
face gray (128,128,128); the output goes through the same mp4v -> ffmpeg
H.264/yuv420p pipeline as the face mask.

Pipeline integration (app/pipeline.py): the ``mask-replaceable-objects``
node runs immediately after ``mask-reference-faces`` on the same reference
video stream.  It triggers only when the segment has OR-flagged replaced
objects (targets come from the OR output) AND FAL_KEY is configured; a
missing key means the node is skipped entirely, like a segment without OR
targets.  Persistent failure
within OBJECT_MASK_MAX_ATTEMPTS degrades to the unmasked video plus an
``object_mask_degraded`` log event instead of blocking the render (unlike
the face mask, whose missing model stays fatal).  No new gates are added.

Two-block convention surfaced to H3 via the fusion/H3 prompts: mid gray
(128,128,128) covers source faces, deep gray (64,64,64) covers replaced
objects; H3 must not redraw block contents and must fill them from the
supplied reference images.  The object part of the prompt appears only
when this node actually produced masking for the segment (frozen per
segment, so resumes stay consistent).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import media
from app.config import get_settings
from app.object_detect import ObjectDetection, select_provider


_LOGGER = logging.getLogger(__name__)

OBJECT_BLOCK_FILL = (64, 64, 64)  # solid deep gray (BGR) filling the object box
OBJECT_MASK_HOLD_FRAMES = 12  # frames a lost object keeps its last known box
OBJECT_MASK_MAX_ATTEMPTS = 3  # retry budget for the pipeline node


class ObjectMaskError(RuntimeError):
    """Object detection or masking could not be completed deterministically."""


def _target_slug(target: str) -> str:
    """检测产物文件名的 target 段：可读前缀 + 短哈希，确定且防碰撞。"""
    readable = re.sub(r"[^a-z0-9]+", "-", target.lower()).strip("-")[:40].strip("-")
    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()[:8]
    return f"{readable or 'target'}-{digest}"


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    return ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) / 4.0


def _frame_boxes(
    detections: list[ObjectDetection],
    fps: float,
    *,
    hold_frames: set[int] | None = None,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Expand sparse timestamped detections into a per-frame box list.

    Timestamps convert to frame indices via the video fps. Detections sharing
    a label are chained into tracks: each detection joins the track whose last
    box overlaps it most (IoU), ties and non-overlaps broken by nearest
    center; a detection that fits no open track starts a new one (a second
    simultaneous instance). Within a track, boxes between anchor frames are
    linearly interpolated; after the last anchor the final box is held for
    OBJECT_MASK_HOLD_FRAMES frames, then the object is treated as gone.
    ``hold_frames``（可选 out 参数）收集由续盖逻辑产生的帧号。
    """
    points = sorted(
        (
            (round(detection.timestamp_s * fps), detection.bbox_xyxy_normalized, detection.label)
            for detection in detections
        ),
        key=lambda point: point[0],
    )
    tracks: dict[str, list[list[tuple[int, tuple[float, float, float, float]]]]] = {}
    for frame_index, bbox, label in points:
        label_tracks = tracks.setdefault(label, [])
        open_tracks = [track for track in label_tracks if track[-1][0] < frame_index]
        if not open_tracks:
            label_tracks.append([(frame_index, bbox)])
            continue
        best = max(
            open_tracks,
            key=lambda track: (
                _iou(track[-1][1], bbox),
                -_center_distance(track[-1][1], bbox),
            ),
        )
        best.append((frame_index, bbox))
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for label_tracks in tracks.values():
        for track in label_tracks:
            for (f0, b0), (f1, b1) in zip(track, track[1:]):
                # The next interval re-owns its left anchor; ending each
                # interval at f1 - 1 keeps every anchor frame painted once.
                for frame_index in range(f0, f1):
                    ratio = (frame_index - f0) / (f1 - f0)
                    box = tuple(
                        b0[axis] + (b1[axis] - b0[axis]) * ratio for axis in range(4)
                    )
                    boxes_by_frame.setdefault(frame_index, []).append(box)  # type: ignore[arg-type]
            last_frame, last_box = track[-1]
            boxes_by_frame.setdefault(last_frame, []).append(last_box)
            for frame_index in range(
                last_frame + 1, last_frame + OBJECT_MASK_HOLD_FRAMES + 1
            ):
                boxes_by_frame.setdefault(frame_index, []).append(last_box)
                if hold_frames is not None:
                    hold_frames.add(frame_index)
    return boxes_by_frame


def _pixel_box(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Convert a 0-1 normalized box to pixel coordinates, clipped to the frame."""
    x0 = max(0, min(width, int(round(bbox[0] * width))))
    y0 = max(0, min(height, int(round(bbox[1] * height))))
    x1 = max(0, min(width, int(round(bbox[2] * width))))
    y1 = max(0, min(height, int(round(bbox[3] * height))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _apply_object_block(
    frame: np.ndarray, box: tuple[int, int, int, int]
) -> None:
    """Fill the box in place with a solid deep gray that erases the object."""
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = OBJECT_BLOCK_FILL


def _production_detector() -> Any:
    client = select_provider(get_settings())
    if client is None:
        raise ObjectMaskError(
            "no object detection provider is available; "
            "the object mask node must not run"
        )
    return client


def _detect_all(
    source: Path,
    targets: list[str],
    detector: Any,
    artifacts_dir: Path | None,
) -> list[ObjectDetection]:
    """fal 端点单 target：N 个 target = N 次独立 detect，每次一个独立产物
    文件 object-detect-response-<slug>.json，各自指纹复用；某 target 无检出
    记日志跳过，不影响其它 target。"""
    detections: list[ObjectDetection] = []
    for target in targets:
        client = detector
        if artifacts_dir is not None and hasattr(detector, "with_cache"):
            client = detector.with_cache(
                artifacts_dir
                / f"object-detect-response-{_target_slug(target)}.json"
            )
        found = client.detect(source, [target])
        if not found:
            _LOGGER.warning(
                "object mask: target %r got no detections; other targets "
                "continue",
                target,
            )
        detections.extend(found)
    return detections


def mask_objects(
    source: Path,
    destination: Path,
    targets: list[str],
    *,
    detector: Any | None = None,
    detect_artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Cover every detected replaceable object with a deep gray block,
    keeping fps/resolution. ``detector`` follows the app/object_detect.py
    contract (detect(video_path, targets) -> list[ObjectDetection]); when
    omitted, the production fal SAM-3 client is used (requires FAL_KEY).
    ``detect_artifacts_dir`` 是检测产物落盘目录（项目 work 目录内）：
    fal client 的 with_cache 钩子把原始响应按 target 写到
    object-detect-response-<slug>.json，命中指纹即零 HTTP；
    解析后的契约 detections 写 object-detections.json（见 _detect_all）。"""
    if not isinstance(source, Path) or not source.is_file():
        raise ObjectMaskError("object mask source video is invalid")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ObjectMaskError("object mask targets must be non-empty strings")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ObjectMaskError(f"could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if width < 1 or height < 1 or fps <= 0:
        capture.release()
        raise ObjectMaskError("object mask source metadata is incomplete")
    if detector is None and targets:
        detector = _production_detector()
    detections = (
        _detect_all(source, targets, detector, detect_artifacts_dir)
        if targets
        else []
    )
    if detect_artifacts_dir is not None and targets:
        detect_artifacts_dir.mkdir(parents=True, exist_ok=True)
        (detect_artifacts_dir / "object-detections.json").write_text(
            json.dumps(
                {
                    "targets": list(targets),
                    "detections": [
                        {
                            "label": detection.label,
                            "timestamp_s": detection.timestamp_s,
                            "bbox_xyxy_normalized": list(
                                detection.bbox_xyxy_normalized
                            ),
                        }
                        for detection in detections
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    hold_frame_set: set[int] = set()
    boxes_by_frame = _frame_boxes(detections, fps, hold_frames=hold_frame_set)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".objects-masked-",
        suffix=".mp4",
        dir=destination.parent,
        delete=False,
    ) as handle:
        raw = Path(handle.name)
    stats: dict[str, Any] = {"frames": 0, "frames_with_block": 0, "blocks_total": 0}
    try:
        writer = cv2.VideoWriter(
            str(raw),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ObjectMaskError("could not open the object mask writer")
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            boxes = [
                pixel_box
                for bbox in boxes_by_frame.get(frame_index, [])
                if (pixel_box := _pixel_box(bbox, width, height)) is not None
            ]
            stats["frames"] += 1
            stats["blocks_total"] += len(boxes)
            if boxes:
                stats["frames_with_block"] += 1
            for box in boxes:
                _apply_object_block(frame, box)
            writer.write(frame)
            frame_index += 1
        capture.release()
        writer.release()
        if stats["frames"] < 1:
            raise ObjectMaskError("object mask source has no frames")
        # Re-encode to H.264/yuv420p for provider compatibility.
        media.run_command(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(raw),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-an", "-movflags", "+faststart",
                str(destination),
            ],
            timeout=300,
        )
    finally:
        raw.unlink(missing_ok=True)
    painted = set(range(stats["frames"]))
    anchor_frame_set = {
        round(detection.timestamp_s * fps) for detection in detections
    }
    stats["anchor_frames"] = len(anchor_frame_set & painted)
    stats["hold_frames"] = len(hold_frame_set & painted)
    stats["interpolated_frames"] = (
        stats["frames_with_block"] - stats["anchor_frames"] - stats["hold_frames"]
    )
    stats["targets"] = {
        target: {
            "detections": sum(1 for d in detections if d.label == target),
            "detected_frames": len({
                round(d.timestamp_s * fps) for d in detections if d.label == target
            } & painted),
        }
        for target in targets
    }
    return stats
