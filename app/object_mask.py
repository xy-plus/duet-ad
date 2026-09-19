"""Deterministic SAM gray-block masking for H3 reference videos.

fal SAM-3 receives all requested objects and, when a person is replaced,
``face`` in one comma-separated prompt.  Its calibrated multi-target response
is a single unlabelled per-frame union, so every returned RLE mask receives the
same deep gray (64,64,64) block; sparse/legacy anchors without a mask use their
bbox rectangle. Sparse anchors first discard local area
glitches, then are filled by local linear interpolation; a track that loses its box is held for
``OBJECT_MASK_HOLD_FRAMES`` frames.  The output uses the usual mp4v -> ffmpeg
H.264/yuv420p pipeline.

Pipeline integration (app/pipeline.py): ``mask-replaceable-objects`` consumes
the muted source directly.  It runs only when there are targets and FAL_KEY is
configured.  A persistent failure degrades to the unmasked video plus an
``object_mask_degraded`` log event; no new gate is added.  H3 receives one
frozen instruction that the deep-gray blocks conceal faces and replaceable
objects, and may reconstruct them only from the descriptions/reference images.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import media
from app.config import get_settings
from app.object_detect import ObjectDetection, select_provider


_LOGGER = logging.getLogger(__name__)

OBJECT_BLOCK_FILL = (64, 64, 64)  # deep gray (BGR) mixed into the object box
OBJECT_MASK_HOLD_FRAMES = 12  # frames a lost object keeps its last known box
OBJECT_MASK_MAX_ATTEMPTS = 3  # retry budget for the pipeline node
OBJECT_MASK_GLITCH_WINDOW_FRAMES = 15
OBJECT_MASK_GLITCH_AREA_RATIO = 0.6
OBJECT_MASK_RLE_FEATHER_KERNEL = 7


class ObjectMaskError(RuntimeError):
    """Object detection or masking could not be completed deterministically."""


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


def _without_glitch_anchors(
    track: list[tuple[int, tuple[float, float, float, float]]],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Remove short-lived area collapses before interpolating a track."""
    retained = []
    for frame_index, bbox in track:
        nearby_areas = [
            (nearby[1][2] - nearby[1][0]) * (nearby[1][3] - nearby[1][1])
            for nearby in track
            if abs(nearby[0] - frame_index) <= OBJECT_MASK_GLITCH_WINDOW_FRAMES
        ]
        median_area = float(np.median(nearby_areas))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # A tracked object's projection cannot collapse below 60% of its
        # local median in one second and rebound; real recession is gradual,
        # so the local median follows it instead of treating it as a glitch.
        if area >= median_area * OBJECT_MASK_GLITCH_AREA_RATIO:
            retained.append((frame_index, bbox))
    return retained


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
            track = _without_glitch_anchors(track)
            if not track:
                continue
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
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    opacity: float,
    blurred_frame: np.ndarray | None = None,
) -> None:
    """Mix the deep-gray block into the box in place.

    ``opacity=1.0`` deliberately retains the direct assignment so the default
    behavior remains byte-for-byte identical to the original solid block.
    """
    x0, y0, x1, y1 = box
    region = frame[y0:y1, x0:x1]
    if opacity == 1.0:
        region[:] = OBJECT_BLOCK_FILL
        return
    if blurred_frame is None:
        raise ObjectMaskError("blurred frame is required for a translucent object block")
    fill = np.asarray(OBJECT_BLOCK_FILL, dtype=np.float64)
    region[:] = np.rint(
        fill * opacity + blurred_frame[y0:y1, x0:x1].astype(np.float64) * (1.0 - opacity)
    ).astype(np.uint8)


def _decode_rle_mask(rle: str, width: int, height: int) -> np.ndarray:
    """Decode fal's row-major ``(start, length)`` RLE into a bool mask.

    The calibrated fixture's first frame yields bbox IoU 0.9958 with metadata
    under C order, versus 0.2437 under Fortran order; fal is not COCO counts.
    """
    try:
        counts = [int(value) for value in rle.split()]
    except (AttributeError, ValueError) as exc:
        raise ObjectMaskError("object mask RLE counts are invalid") from exc
    if len(counts) % 2 or any(count < 0 for count in counts):
        raise ObjectMaskError("object mask RLE pairs are invalid")
    pairs = list(zip(counts[::2], counts[1::2]))
    pixel_count = width * height
    # Real fal cache sample: frame 104 ends at 589825 for a 576*1024 frame.
    # Accept that documented one-pixel overrun and clamp it explicitly.
    if any(length == 0 or start >= pixel_count + 1 for start, length in pairs):
        raise ObjectMaskError("object mask RLE dimensions do not match the frame")
    if any(next_start < start + length for (start, length), (next_start, _) in zip(pairs, pairs[1:])):
        raise ObjectMaskError("object mask RLE pairs overlap or are unordered")
    flat = np.zeros(width * height, dtype=np.uint8)
    for start, length in pairs:
        end = min(start + length, pixel_count)
        flat[start:end] = 1
    return flat.reshape((height, width)).astype(bool)


def _apply_object_rle_mask(frame: np.ndarray, mask: np.ndarray, *, opacity: float,
                           blurred_frame: np.ndarray | None = None) -> None:
    """Feather an RLE mask and apply the ordinary block blend in place."""
    alpha = cv2.GaussianBlur(mask.astype(np.float32),
                             (OBJECT_MASK_RLE_FEATHER_KERNEL,) * 2, 0)[..., None]
    if opacity == 1.0:
        replacement: np.ndarray | tuple[int, int, int] = OBJECT_BLOCK_FILL
    else:
        if blurred_frame is None:
            raise ObjectMaskError("blurred frame is required for a translucent object block")
        replacement = (np.asarray(OBJECT_BLOCK_FILL) * opacity
                       + blurred_frame.astype(np.float64) * (1.0 - opacity))
    frame[:] = np.rint(frame.astype(np.float64) * (1.0 - alpha)
                       + np.asarray(replacement, dtype=np.float64) * alpha).astype(np.uint8)


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
    """One SAM request covers all targets, avoiding a second paid invocation.

    The raw detection response cache is keyed by the complete comma-separated
    prompt.  The project-local fallback is therefore one prompt-level path
    rather than a per-target path.  SAM does not label boxes by target in a
    multi-target response, so an empty result is meaningful only for the whole
    target group.
    """
    client = detector
    if artifacts_dir is not None and hasattr(detector, "with_cache"):
        client = detector.with_cache(artifacts_dir / "object-detect-response.json")
    detections = client.detect(source, targets)
    if not detections:
        _LOGGER.warning(
            "object mask: target group %r got no detections",
            ", ".join(target.strip() for target in targets),
        )
    return detections


def mask_objects(
    source: Path,
    destination: Path,
    targets: list[str],
    *,
    detector: Any | None = None,
    detect_artifacts_dir: Path | None = None,
    block_opacity: float = 1.0,
    block_blur: int = 31,
) -> dict[str, Any]:
    """Cover every detected replaceable object with a deep gray block,
    keeping fps/resolution. ``detector`` follows the app/object_detect.py
    contract (detect(video_path, targets) -> list[ObjectDetection]); when
    omitted, the production SAM-3 client is used (wavespeed → sandbase →
    fal 链路顺序，见 object_detect.select_provider，需对应平台 key)。
    ``detect_artifacts_dir`` 是检测产物落盘目录（项目 work 目录内）：
    解析后的契约 detections 写 object-detections.json（见 _detect_all）；
    检测原始响应的指纹缓存写实例级共享目录
    <DATA_DIR>/shared/object-detect-cache/（跨项目命中即零 HTTP，见
    object_detect 的落盘纪律注释），with_cache 传入的本目录
    object-detect-response.json 仅作项目级缓存的回退读。多目标的 SAM 结果
    无目标归属；其 detection label 是完整 prompt，所有框仍统一遮盖。
    ``block_opacity`` controls the fill contribution and defaults to the
    original fully opaque behavior.  For translucent blocks, ``block_blur``
    controls the Gaussian kernel applied once per source frame before its
    blurred residual is mixed into every block on that frame. """
    if not isinstance(source, Path) or not source.is_file():
        raise ObjectMaskError("object mask source video is invalid")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ObjectMaskError("object mask targets must be non-empty strings")
    if (
        isinstance(block_opacity, bool)
        or not isinstance(block_opacity, (int, float))
        or not np.isfinite(float(block_opacity))
        or not 0.0 < float(block_opacity) <= 1.0
    ):
        raise ObjectMaskError("object mask block opacity must be a number in (0, 1]")
    block_opacity = float(block_opacity)
    if isinstance(block_blur, bool) or not isinstance(block_blur, int) or block_blur < 1 or block_blur % 2 != 1:
        raise ObjectMaskError("object mask block blur must be a positive odd integer")
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
                            "rle": detection.rle,
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
    rles_by_frame: dict[int, list[str]] = {}
    for detection in detections:
        if detection.rle is not None:
            rles_by_frame.setdefault(round(detection.timestamp_s * fps), []).append(detection.rle)
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
            masks = [_decode_rle_mask(rle, width, height)
                     for rle in rles_by_frame.get(frame_index, [])]
            if masks:
                boxes = []  # real anchors supersede bbox; bridge frames remain rectangles
            stats["frames"] += 1
            stats["blocks_total"] += len(boxes) + len(masks)
            if boxes or masks:
                stats["frames_with_block"] += 1
            blurred_frame = (
                cv2.GaussianBlur(frame, (block_blur, block_blur), 0)
                if (boxes or masks) and block_opacity < 1.0
                else None
            )
            for mask in masks:
                _apply_object_rle_mask(frame, mask, opacity=block_opacity,
                                       blurred_frame=blurred_frame)
            for box in boxes:
                _apply_object_block(
                    frame, box, opacity=block_opacity, blurred_frame=blurred_frame
                )
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
    labels = list(dict.fromkeys(detection.label for detection in detections))
    if not labels:
        labels = [", ".join(target.strip() for target in targets)] if targets else []
    stats["targets"] = {
        label: {
            "detections": sum(1 for detection in detections if detection.label == label),
            "detected_frames": len({
                round(detection.timestamp_s * fps)
                for detection in detections
                if detection.label == label
            } & painted),
        }
        for label in labels
    }
    return stats
