from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import object_mask
from app.object_detect import ObjectDetection
from app.object_mask import (
    OBJECT_BLOCK_FILL,
    OBJECT_MASK_HOLD_FRAMES,
    ObjectMaskError,
)


class _SparseDetector:
    """Test seam mirroring the production client contract: one detect() call
    per video returning sparse timestamped ObjectDetection entries."""

    def __init__(self, detections: list[ObjectDetection]) -> None:
        self.detections = detections
        self.calls: list[tuple[Path, list[str]]] = []

    def detect(self, video_path: Path, targets: list[str]) -> list[ObjectDetection]:
        self.calls.append((video_path, targets))
        return list(self.detections)


def _solid_video(
    path: Path,
    *,
    frames: int,
    fps: float = 1.0,
    size: tuple[int, int] = (48, 32),
    color: tuple[int, int, int] = (10, 20, 30),
) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    writer.release()


def _read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _block_columns(frame: np.ndarray) -> np.ndarray:
    """Columns containing pixels painted with the object block fill color.

    The source color (10,20,30) differs from the fill (64,64,64) by >= 34 per
    channel, so a tolerance of 24 cleanly separates the two even after the
    lossy mp4v -> H.264 round trip.
    """
    diff = np.abs(
        frame.astype(np.int32) - np.array(OBJECT_BLOCK_FILL, dtype=np.int32)
    )
    masked = np.all(diff <= 24, axis=-1)
    return np.where(masked.any(axis=0))[0]


def _detection(
    timestamp_s: float, bbox: tuple[float, float, float, float], label: str = "mug"
) -> ObjectDetection:
    return ObjectDetection(
        label=label, timestamp_s=timestamp_s, bbox_xyxy_normalized=bbox
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_covers_the_box_with_dark_gray_and_preserves_timing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=8, fps=2.0)
    detector = _SparseDetector(
        [
            _detection(0.0, (0.05, 0.25, 0.35, 0.75)),
            _detection(1.5, (0.55, 0.25, 0.85, 0.75)),
        ]
    )

    destination = tmp_path / "source-objects-masked.mp4"
    stats = object_mask.mask_objects(
        source, destination, ["mug"], detector=detector
    )

    assert stats == {"frames": 8, "frames_with_block": 8, "blocks_total": 8}
    assert detector.calls == [(source, ["mug"])]
    capture = cv2.VideoCapture(str(destination))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(2.0)
    capture.release()
    frames = _read_frames(destination)
    assert len(frames) == 8
    # Frame 0 box: x in [round(0.05*48), round(0.35*48)] = [2, 17],
    # y in [8, 24]. The interior is a flat deep-gray block.
    pixel = frames[0][16, 9].astype(np.int32)
    assert all(
        abs(int(pixel[channel]) - fill) <= 12
        for channel, fill in enumerate(OBJECT_BLOCK_FILL)
    )
    # Pixels the box never reaches stay untouched in every frame.
    for frame in frames:
        outside = frame[16, 46].astype(np.int32)
        assert all(
            abs(int(outside[channel]) - original) <= 12
            for channel, original in enumerate((10, 20, 30))
        )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type", "-of", "json", str(destination),
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    assert [stream["codec_type"] for stream in streams] == ["video"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_interpolates_block_positions_between_timestamps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=8, fps=2.0)
    detector = _SparseDetector(
        [
            _detection(0.0, (0.05, 0.25, 0.35, 0.75)),
            _detection(1.5, (0.55, 0.25, 0.85, 0.75)),
        ]
    )

    destination = tmp_path / "source-objects-masked.mp4"
    object_mask.mask_objects(source, destination, ["mug"], detector=detector)

    frames = _read_frames(destination)
    left_edges = []
    for frame in frames:
        columns = _block_columns(frame)
        assert len(columns) > 0
        left_edges.append(int(columns.min()))
    # The block slides right between the two anchors, then holds: the left
    # edge never jumps backward and moves monotonically from ~2 to ~26.
    assert left_edges[0] <= 4
    assert left_edges[-1] >= 24
    assert all(
        later >= earlier for earlier, later in zip(left_edges, left_edges[1:])
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_holds_the_last_position_then_disappears(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=20, fps=1.0)
    detector = _SparseDetector([_detection(0.0, (0.25, 0.25, 0.75, 0.75))])

    destination = tmp_path / "source-objects-masked.mp4"
    stats = object_mask.mask_objects(
        source, destination, ["mug"], detector=detector
    )

    assert stats == {
        "frames": 20,
        "frames_with_block": 1 + OBJECT_MASK_HOLD_FRAMES,
        "blocks_total": 1 + OBJECT_MASK_HOLD_FRAMES,
    }
    frames = _read_frames(destination)
    assert len(frames) == 20
    # The block is visible on the anchor frame, held for OBJECT_MASK_HOLD_FRAMES
    # more frames, and gone afterwards.
    assert len(_block_columns(frames[0])) > 0
    assert len(_block_columns(frames[OBJECT_MASK_HOLD_FRAMES])) > 0
    assert len(_block_columns(frames[OBJECT_MASK_HOLD_FRAMES + 1])) == 0
    assert len(_block_columns(frames[-1])) == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_matches_simultaneous_boxes_to_tracks_by_iou(
    tmp_path: Path,
) -> None:
    """Two same-label boxes per timestamp must keep their identities: box A
    drifts right on the left half, box B drifts left on the right half, and
    the two blocks never merge or swap."""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3, fps=1.0)
    detector = _SparseDetector(
        [
            _detection(0.0, (0.04, 0.25, 0.33, 0.75)),
            _detection(0.0, (0.58, 0.25, 0.87, 0.75)),
            _detection(2.0, (0.08, 0.25, 0.37, 0.75)),
            _detection(2.0, (0.54, 0.25, 0.83, 0.75)),
        ]
    )

    destination = tmp_path / "source-objects-masked.mp4"
    stats = object_mask.mask_objects(
        source, destination, ["mug"], detector=detector
    )

    assert stats == {"frames": 3, "frames_with_block": 3, "blocks_total": 6}
    frames = _read_frames(destination)
    for frame in frames:
        columns = _block_columns(frame)
        # Two disjoint clusters: one fully left of column 20, one fully right
        # of column 24; the gap column stays unpainted.
        left = columns[columns < 20]
        right = columns[columns > 24]
        assert len(left) > 0
        assert len(right) > 0
        assert not np.any((columns >= 20) & (columns <= 24))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_without_detections_leaves_frames_untouched(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3)

    destination = tmp_path / "source-objects-masked.mp4"
    stats = object_mask.mask_objects(
        source, destination, ["mug"], detector=_SparseDetector([])
    )

    assert stats == {"frames": 3, "frames_with_block": 0, "blocks_total": 0}
    frames = _read_frames(destination)
    assert len(frames) == 3
    assert len(_block_columns(frames[0])) == 0


def test_frame_boxes_interpolates_and_holds_then_disappears() -> None:
    detections = [
        _detection(0.0, (0.1, 0.1, 0.2, 0.2)),
        _detection(2.0, (0.5, 0.3, 0.6, 0.4)),
    ]

    boxes = object_mask._frame_boxes(detections, fps=1.0)

    assert boxes[0] == [(0.1, 0.1, 0.2, 0.2)]
    assert boxes[1][0] == pytest.approx((0.3, 0.2, 0.4, 0.3))
    assert boxes[2] == [(0.5, 0.3, 0.6, 0.4)]
    assert boxes[2 + OBJECT_MASK_HOLD_FRAMES] == [(0.5, 0.3, 0.6, 0.4)]
    assert 2 + OBJECT_MASK_HOLD_FRAMES + 1 not in boxes


def test_frame_boxes_counts_each_shared_anchor_frame_once() -> None:
    # Adjacent interpolation intervals share their boundary anchor frame;
    # the anchor must appear once per frame, not once per interval.
    detections = [
        _detection(0.0, (0.1, 0.1, 0.2, 0.2)),
        _detection(1.0, (0.3, 0.1, 0.4, 0.2)),
        _detection(2.0, (0.5, 0.1, 0.6, 0.2)),
    ]

    boxes = object_mask._frame_boxes(detections, fps=1.0)

    assert boxes[0] == [(0.1, 0.1, 0.2, 0.2)]
    assert boxes[1] == [(0.3, 0.1, 0.4, 0.2)]
    assert boxes[2] == [(0.5, 0.1, 0.6, 0.2)]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_objects_blocks_total_counts_anchor_frames_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3, fps=1.0)
    detector = _SparseDetector(
        [
            _detection(0.0, (0.05, 0.25, 0.35, 0.75)),
            _detection(1.0, (0.30, 0.25, 0.60, 0.75)),
            _detection(2.0, (0.55, 0.25, 0.85, 0.75)),
        ]
    )

    destination = tmp_path / "source-objects-masked.mp4"
    stats = object_mask.mask_objects(
        source, destination, ["mug"], detector=detector
    )

    assert stats == {"frames": 3, "frames_with_block": 3, "blocks_total": 3}


def test_mask_objects_rejects_an_unreadable_source(tmp_path: Path) -> None:
    with pytest.raises(ObjectMaskError, match="source video is invalid"):
        object_mask.mask_objects(
            tmp_path / "missing.mp4",
            tmp_path / "out.mp4",
            ["mug"],
            detector=_SparseDetector([]),
        )


def test_mask_objects_requires_a_provider_when_no_detector_is_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=2)
    monkeypatch.setattr(object_mask, "get_settings", lambda: object())
    monkeypatch.setattr(object_mask, "select_provider", lambda settings: None)
    destination = tmp_path / "out.mp4"

    with pytest.raises(ObjectMaskError, match="no object detection provider"):
        object_mask.mask_objects(source, destination, ["mug"])

    assert not destination.exists()
