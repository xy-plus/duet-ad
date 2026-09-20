from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import object_mask
from app.config import Settings
from app.object_detect import FalSam3GroundingClient, ObjectDetection
from app.object_mask import (
    OBJECT_BLOCK_FILL,
    OBJECT_MASK_HOLD_FRAMES,
    ObjectMaskError,
)


class _SparseDetector:
    """Test seam mirroring the production client: one group detect() call
    returning sparse timestamped ObjectDetection entries."""

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


def test_frame_boxes_discards_short_area_collapse_and_interpolates_over_it() -> None:
    large = (0.0, 0.0, 1.0, 1.0)
    strip = (0.0, 0.8, 1.0, 1.0)
    detections = [
        _detection(frame / 30, strip if 20 <= frame < 32 else large)
        for frame in range(52)
    ]

    boxes = object_mask._frame_boxes(detections, fps=30.0)

    for frame in range(20, 32):
        x0, y0, x1, y1 = boxes[frame][0]
        assert (x1 - x0) * (y1 - y0) == pytest.approx(1.0)


def test_frame_boxes_keeps_gradual_area_change_anchors() -> None:
    detections = [
        _detection(frame / 30, (0.0, 0.0, 0.95 ** (frame / 2), 1.0))
        for frame in range(50)
    ]

    boxes = object_mask._frame_boxes(detections, fps=30.0)

    for frame, detection in enumerate(detections):
        assert boxes[frame][0] == pytest.approx(detection.bbox_xyxy_normalized)


def test_apply_object_block_opacity_one_is_the_original_solid_fill() -> None:
    frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    object_mask._apply_object_block(frame, (0, 0, 2, 1), opacity=1.0)

    assert np.array_equal(
        frame,
        np.array([[OBJECT_BLOCK_FILL, OBJECT_BLOCK_FILL]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("opacity", "expected"),
    [
        (0.95, (61, 62, 62)),
        (0.90, (59, 60, 61)),
    ],
)
def test_apply_object_block_blends_the_blurred_source_by_opacity(
    opacity: float, expected: tuple[int, int, int],
) -> None:
    frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    blurred = np.array([[[20, 30, 40]]], dtype=np.uint8)

    object_mask._apply_object_block(
        frame, (0, 0, 1, 1), opacity=opacity, blurred_frame=blurred
    )

    assert tuple(frame[0, 0]) == tuple(
        np.rint(np.array(OBJECT_BLOCK_FILL) * opacity + blurred[0, 0] * (1 - opacity))
        .astype(np.uint8)
    )


def test_apply_object_block_uses_gaussian_blur_not_original_pixels() -> None:
    original = np.zeros((5, 5, 3), dtype=np.uint8)
    original[2, 2] = (200, 100, 50)
    blurred = cv2.GaussianBlur(original, (3, 3), 0)
    frame = original.copy()

    object_mask._apply_object_block(
        frame, (2, 2, 3, 3), opacity=0.5, blurred_frame=blurred
    )

    expected = np.rint(np.array(OBJECT_BLOCK_FILL) * 0.5 + blurred[2, 2] * 0.5).astype(np.uint8)
    original_mix = np.rint(np.array(OBJECT_BLOCK_FILL) * 0.5 + original[2, 2] * 0.5).astype(np.uint8)
    assert np.array_equal(frame[2, 2], expected)
    assert not np.array_equal(frame[2, 2], original_mix)


def test_decode_rle_mask_uses_start_length_row_major_and_validates_dimensions() -> None:
    mask = object_mask._decode_rle_mask("0 1 3 1", width=2, height=2)
    assert np.array_equal(mask, np.array([[True, False], [False, True]]))
    with pytest.raises(ObjectMaskError, match="dimensions do not match"):
        object_mask._decode_rle_mask("5 1", width=2, height=2)


def test_decode_rle_mask_fixture_bbox_matches_metadata() -> None:
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "fal_sam3_video_rle_response.json").read_text()
    )
    mask = object_mask._decode_rle_mask(payload["rle"][0], width=576, height=1024)
    ys, xs = np.where(mask)
    actual = (xs.min() / 576, ys.min() / 1024, (xs.max() + 1) / 576, (ys.max() + 1) / 1024)
    cx, cy, width, height = payload["metadata"][0]["box"]
    expected = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    ix0, iy0 = max(actual[0], expected[0]), max(actual[1], expected[1])
    ix1, iy1 = min(actual[2], expected[2]), min(actual[3], expected[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (actual[2] - actual[0]) * (actual[3] - actual[1]) + (expected[2] - expected[0]) * (expected[3] - expected[1]) - intersection
    assert mask.sum() > 0
    assert intersection / union > 0.9


def test_decode_rle_mask_clamps_real_fal_end_off_by_one() -> None:
    # Final pair from experiment shared-cache response frame 104: end=589825.
    mask = object_mask._decode_rle_mask("589563 262", width=576, height=1024)
    assert mask.sum() == 261


def test_apply_object_rle_mask_keeps_unmasked_pixels() -> None:
    frame = np.zeros((15, 15, 3), dtype=np.uint8)
    mask = np.zeros((15, 15), dtype=bool)
    mask[3:12, 3:12] = True
    object_mask._apply_object_rle_mask(frame, mask, opacity=1.0)
    assert tuple(frame[7, 7]) == OBJECT_BLOCK_FILL
    assert tuple(frame[0, 0]) == (0, 0, 0)


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

    assert stats == {
        "frames": 8,
        "frames_with_block": 8,
        "blocks_total": 8,
        "anchor_frames": 2,
        "hold_frames": 4,
        "interpolated_frames": 2,
        "targets": {"mug": {"detections": 2, "detected_frames": 2}},
        "flow_compensated_frames": 0,
    }
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
        "anchor_frames": 1,
        "hold_frames": OBJECT_MASK_HOLD_FRAMES,
        "interpolated_frames": 0,
        "targets": {"mug": {"detections": 1, "detected_frames": 1}},
        "flow_compensated_frames": 0,
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

    assert stats == {
        "frames": 3,
        "frames_with_block": 3,
        "blocks_total": 6,
        "anchor_frames": 2,
        "hold_frames": 0,
        "interpolated_frames": 1,
        "targets": {"mug": {"detections": 4, "detected_frames": 2}},
        "flow_compensated_frames": 0,
    }
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

    assert stats == {
        "frames": 3,
        "frames_with_block": 0,
        "blocks_total": 0,
        "anchor_frames": 0,
        "hold_frames": 0,
        "interpolated_frames": 0,
        "targets": {"mug": {"detections": 0, "detected_frames": 0}},
        "flow_compensated_frames": 0,
    }
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

    assert stats == {
        "frames": 3,
        "frames_with_block": 3,
        "blocks_total": 3,
        "anchor_frames": 3,
        "hold_frames": 0,
        "interpolated_frames": 0,
        "targets": {"mug": {"detections": 3, "detected_frames": 3}},
        "flow_compensated_frames": 0,
    }


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


class _GroupDetector:
    """Test seam mirroring fal_sam3: one multi-target detect + cache hook."""

    def __init__(self, detections: list[ObjectDetection]) -> None:
        self.detections = detections
        self.calls: list[tuple[Path, list[str]]] = []
        self.cache_paths: list[Path] = []

    def with_cache(self, cache_path: Path) -> "_GroupDetector":
        self.cache_paths.append(cache_path)
        return self

    def detect(self, video_path: Path, targets: list[str]) -> list[ObjectDetection]:
        self.calls.append((video_path, list(targets)))
        return list(self.detections)


def test_mask_objects_detects_multiple_targets_in_one_request(
    tmp_path: Path,
) -> None:
    """多 target 只调用一次 detect，项目级 fallback 也只保留一个 prompt 路径。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3)
    detector = _GroupDetector([
        _detection(0.0, (0.1, 0.1, 0.4, 0.4), label="mug, scarf"),
        _detection(1.0, (0.5, 0.5, 0.9, 0.9), label="mug, scarf"),
    ])
    artifacts = tmp_path / "segment"
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug", "scarf"],
        detector=detector, detect_artifacts_dir=artifacts,
    )
    assert detector.calls == [(source, ["mug", "scarf"])]
    assert detector.cache_paths == [
        artifacts / "object-detect-response.json",
    ]
    detections_file = artifacts / "object-detections.json"
    assert detections_file.is_file()
    payload = json.loads(detections_file.read_text(encoding="utf-8"))
    assert payload["targets"] == ["mug", "scarf"]
    assert [entry["label"] for entry in payload["detections"]] == [
        "mug, scarf", "mug, scarf"
    ]
    assert stats["targets"] == {
        "mug, scarf": {"detections": 2, "detected_frames": 2},
    }


def test_mask_objects_multi_target_empty_detection_is_one_group_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """无 per-target 标注，整体零检出只能作为一个 target group 记日志。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3)
    detector = _GroupDetector([])
    caplog.set_level(logging.WARNING, logger="app.object_mask")
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug", "ghost"],
        detector=detector, detect_artifacts_dir=tmp_path / "segment",
    )
    assert detector.calls == [(source, ["mug", "ghost"])]
    assert stats["blocks_total"] == 0
    assert stats["targets"] == {
        "mug, ghost": {"detections": 0, "detected_frames": 0}
    }
    assert any(
        "target group 'mug, ghost' got no detections" in message
        for message in caplog.messages
    )


def test_mask_objects_stats_breakdown(tmp_path: Path) -> None:
    """统计分解：锚点帧 / 插值帧 / 续盖帧 / 每 target 检出率。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=30)
    # 帧 0 与帧 20 两个锚点 → 0 锚点、1-19 插值、20 锚点、21-32 续盖
    # （视频 30 帧，续盖实际画 21-29）。
    detector = _SparseDetector([
        _detection(0.0, (0.1, 0.1, 0.4, 0.4)),
        _detection(20.0, (0.2, 0.2, 0.5, 0.5)),
    ])
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug"], detector=detector,
    )
    assert stats["frames"] == 30
    assert stats["anchor_frames"] == 2
    assert stats["interpolated_frames"] == 19
    assert stats["hold_frames"] == 9
    assert (
        stats["frames_with_block"]
        == stats["anchor_frames"]
        + stats["interpolated_frames"]
        + stats["hold_frames"]
    )


# --- fal_sam3 真实 client 集成：产物落盘与指纹复用（mock HTTP，零真实调用）---

_FAL_QUEUE_RESULT = {
    "rle": ["0 1536"],
    "boxes": [[0.5, 0.5, 0.4, 0.4]],
    "metadata": [{"index": 0, "score": None, "box": [0.5, 0.5, 0.4, 0.4]}],
}


def _fal_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        access_token="test-access",
        data_dir=tmp_path,
        public_base_url="https://detect.test",
        multiref_mode="direct",
        fal_sam3_api_key="fal-secret",
        fal_sam3_api_base="https://fal.test/fal-ai/sam-3-1",
        fal_sam3_model="video-rle",
        provider_poll_interval_s=0.01,
        request_timeout_s=1.0,
    )


def _fal_mock_handler(state: dict[str, int]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/storage/upload/initiate":
            state["uploads"] += 1
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://fal.test/upload/presigned",
                    "file_url": "https://fal.test/files/source-muted.mp4",
                },
            )
        if request.method == "PUT" and path == "/upload/presigned":
            return httpx.Response(200)
        if request.method == "POST" and path == "/fal-ai/sam-3-1/video-rle":
            state["submits"] += 1
            body = json.loads(request.content)
            state.setdefault("prompts", []).append(body["prompt"])
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "status_url": "https://fal.test/requests/req-1/status",
                    "response_url": "https://fal.test/requests/req-1",
                },
            )
        if request.method == "GET" and path == "/requests/req-1/status":
            return httpx.Response(200, json={"status": "COMPLETED"})
        if request.method == "GET" and path == "/requests/req-1":
            return httpx.Response(200, json=_FAL_QUEUE_RESULT)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


def test_mask_objects_fal_artifacts_and_fingerprint_reuse(tmp_path: Path) -> None:
    """真实 fal client（mock HTTP）：2 target 合成一个提交，原始响应缓存写
    实例级共享目录（不写项目目录）；另一个项目目录同指纹二次运行零 HTTP。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=2)
    artifacts = tmp_path / "segment"
    state: dict[str, int] = {"uploads": 0, "submits": 0}
    client = FalSam3GroundingClient(
        _fal_test_settings(tmp_path),
        transport=httpx.MockTransport(_fal_mock_handler(state)),
    )
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["handbag", "bottle"],
        detector=client, detect_artifacts_dir=artifacts,
    )
    assert state["submits"] == 1
    assert state["uploads"] == 1
    assert state["prompts"] == ["handbag, bottle"]
    # 项目目录只留解析后的 detections；fal 响应缓存只在实例级共享目录。
    assert sorted(artifacts.glob("object-detect-response-*.json")) == []
    shared_dir = tmp_path / "shared" / "object-detect-cache"
    response_files = sorted(shared_dir.glob("object-detect-response-*.json"))
    assert len(response_files) == 1
    for path in response_files:
        cached = json.loads(path.read_text(encoding="utf-8"))
        # 原始响应字段一个不丢（rle/boxes/metadata 原样）。
        assert cached["fal_response"] == _FAL_QUEUE_RESULT
        assert set(cached["fingerprint"]) == {
            "video_sha256", "prompt", "detection_threshold", "model",
        }
        assert cached["fingerprint"]["prompt"] == "handbag, bottle"
    assert stats["blocks_total"] == 2  # 1 union box × (帧 0 锚点 + 帧 1 续盖)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    reused = FalSam3GroundingClient(
        _fal_test_settings(tmp_path),
        transport=httpx.MockTransport(fail_handler),
    )
    stats2 = object_mask.mask_objects(
        source, tmp_path / "out2.mp4", ["handbag", "bottle"],
        detector=reused, detect_artifacts_dir=tmp_path / "other-project-segment",
    )
    assert stats2["blocks_total"] == stats["blocks_total"]


# ==== 短时空洞的双向光流补偿（_flow_compensated_masks / mask_objects 集成） ====
# 算法与取舍见 app/object_mask.py 的段头注释。测试用固定种子噪声背景 +
# 匀速移动的亮方块（光流可跟踪），检测产物用合成 fal 行主序 rle；空洞
# 通过打断 track（锚点 IoU=0 的两段）+ monkeypatch 续盖帧数为 0 构造。


def _rect_rle(
    x0: int, y0: int, x1: int, y1: int, width: int, height: int
) -> str:
    """矩形区域 → fal 行主序 (start, length) 对 counts 字符串。"""
    pairs = []
    for row in range(y0, y1):
        pairs.append(f"{row * width + x0} {x1 - x0}")
    return " ".join(pairs)


def _rle_detection(
    frame: int, x0: int, y0: int, x1: int, y1: int, label: str,
    width: int, height: int,
) -> ObjectDetection:
    bbox = (x0 / width, y0 / height, x1 / width, y1 / height)
    return ObjectDetection(
        label=label,
        timestamp_s=float(frame),  # 测试视频 fps=1.0，时间戳 = 帧号
        bbox_xyxy_normalized=bbox,
        rle=_rect_rle(x0, y0, x1, y1, width, height),
    )


_FLOW_SIZE = (160, 120)  # (width, height)
_FLOW_SQUARE = 24  # 亮方块边长


def _moving_square_frame(
    square_x: int, square_y: int, background: np.ndarray
) -> np.ndarray:
    frame = background.copy()
    frame[square_y:square_y + _FLOW_SQUARE, square_x:square_x + _FLOW_SQUARE] = 240
    return frame


def _moving_square_video(
    path: Path, frames: int, x0: int, y0: int, dx_per_frame: int
) -> np.ndarray:
    """固定种子噪声背景 + 匀速右移的亮方块；返回背景数组供断言复用。"""
    rng = np.random.default_rng(42)
    background = rng.integers(10, 40, (_FLOW_SIZE[1], _FLOW_SIZE[0], 3), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, _FLOW_SIZE
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    for frame_index in range(frames):
        writer.write(
            _moving_square_frame(x0 + dx_per_frame * frame_index, y0, background)
        )
    writer.release()
    return background


def _flow_dis() -> cv2.DISOpticalFlow:
    return cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def _gray_of(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _square_mask(x: int, y: int) -> np.ndarray:
    mask = np.zeros((_FLOW_SIZE[1], _FLOW_SIZE[0]), dtype=np.uint8)
    mask[y:y + _FLOW_SQUARE, x:x + _FLOW_SQUARE] = 1
    return mask


def test_flow_compensated_masks_lands_on_true_gap_position() -> None:
    """单元：前帧方块 x=10、后帧 x=18、空洞帧真实 x=14（匀速）；补偿 mask
    与真实位置的 IoU 应显著 > 0。"""
    background = np.random.default_rng(42).integers(
        10, 40, (_FLOW_SIZE[1], _FLOW_SIZE[0], 3), dtype=np.uint8
    )
    prev = _moving_square_frame(10, 40, background)
    gap = _moving_square_frame(14, 40, background)
    next_ = _moving_square_frame(18, 40, background)
    compensated = object_mask._flow_compensated_masks(
        _gray_of(prev), _square_mask(10, 40),
        _gray_of(next_), _square_mask(18, 40),
        [_gray_of(gap)],
        _flow_dis(),
    )
    truth = _square_mask(14, 40).astype(bool)
    got = compensated[0]
    intersection = int((got & truth).sum())
    union = int((got | truth).sum())
    assert intersection / union > 0.5
    assert int(got.sum()) > 200


def test_mask_objects_fills_single_frame_gap(tmp_path: Path, monkeypatch) -> None:
    """单帧空洞（续盖=0 + 两侧 label 不同强制断轨——生产空洞来自整帧目标
    丢失，bbox 插值同 track 会盖住空洞，这是构造真空洞的测试接缝）：
    补偿 1 帧且落在真实位置。"""
    monkeypatch.setattr(object_mask, "OBJECT_MASK_HOLD_FRAMES", 0)
    source = tmp_path / "src.mp4"
    _moving_square_video(source, frames=5, x0=6, y0=40, dx_per_frame=4)
    # 帧 0/1 方块 x=6/10（label A）；帧 3/4 x=18/22（label B）；帧 2 空洞，
    # 方块真实位置 x=14。
    detections = [
        _rle_detection(0, 6, 40, 30, 64, "obj-pre", *_FLOW_SIZE),
        _rle_detection(1, 10, 40, 34, 64, "obj-pre", *_FLOW_SIZE),
        _rle_detection(3, 18, 40, 42, 64, "obj-post", *_FLOW_SIZE),
        _rle_detection(4, 22, 40, 46, 64, "obj-post", *_FLOW_SIZE),
    ]
    out = tmp_path / "out.mp4"
    stats = object_mask.mask_objects(
        source, out, ["obj-pre", "obj-post"], detector=_SparseDetector(detections),
    )
    assert stats["flow_compensated_frames"] == 1
    assert stats["frames"] == 5
    assert stats["frames_with_block"] == 5  # 空洞被补上
    frame2 = _read_frames(out)[2]
    # 空洞帧真实方块位置 x=14：该区域应已被灰块覆盖
    region = frame2[40:40 + _FLOW_SQUARE, 12:12 + _FLOW_SQUARE]
    grayish = (np.abs(region.astype(int) - 64).max(axis=2) <= 12).mean()
    assert grayish > 0.5


def test_mask_objects_fills_two_frame_gap(tmp_path: Path, monkeypatch) -> None:
    """两帧空洞：逐帧位置权重混合，两帧都补偿。"""
    monkeypatch.setattr(object_mask, "OBJECT_MASK_HOLD_FRAMES", 0)
    source = tmp_path / "src.mp4"
    _moving_square_video(source, frames=6, x0=6, y0=40, dx_per_frame=4)
    detections = [
        _rle_detection(0, 6, 40, 30, 64, "obj-pre", *_FLOW_SIZE),
        _rle_detection(1, 10, 40, 34, 64, "obj-pre", *_FLOW_SIZE),
        _rle_detection(4, 22, 40, 46, 64, "obj-post", *_FLOW_SIZE),
        _rle_detection(5, 26, 40, 50, 64, "obj-post", *_FLOW_SIZE),
    ]
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["obj-pre", "obj-post"],
        detector=_SparseDetector(detections),
    )
    assert stats["flow_compensated_frames"] == 2
    assert stats["frames_with_block"] == 6


def test_mask_objects_gap_longer_than_max_not_filled(
    tmp_path: Path, monkeypatch
) -> None:
    """连续无块帧 > OBJECT_MASK_FLOW_GAP_MAX_FRAMES：不补偿，原样输出。"""
    monkeypatch.setattr(object_mask, "OBJECT_MASK_HOLD_FRAMES", 0)
    source = tmp_path / "src.mp4"
    _moving_square_video(source, frames=7, x0=6, y0=40, dx_per_frame=4)
    detections = [
        _rle_detection(0, 6, 40, 30, 64, "obj-pre", *_FLOW_SIZE),
        _rle_detection(5, 26, 40, 50, 64, "obj-post", *_FLOW_SIZE),
        _rle_detection(6, 30, 40, 54, 64, "obj-post", *_FLOW_SIZE),
    ]
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["obj-pre", "obj-post"],
        detector=_SparseDetector(detections),
    )
    assert stats["flow_compensated_frames"] == 0
    assert stats["frames_with_block"] == 3  # 帧 1-4 空洞保持裸露


def test_mask_objects_head_and_tail_gaps_not_extended(
    tmp_path: Path, monkeypatch
) -> None:
    """首检出帧之前/末检出帧之后的空洞不延伸补偿。"""
    monkeypatch.setattr(object_mask, "OBJECT_MASK_HOLD_FRAMES", 0)
    source = tmp_path / "src.mp4"
    _moving_square_video(source, frames=8, x0=6, y0=40, dx_per_frame=4)
    detections = [
        _rle_detection(3, 18, 40, 42, 64, "obj", *_FLOW_SIZE),
        _rle_detection(4, 22, 40, 46, 64, "obj", *_FLOW_SIZE),
    ]
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["obj"],
        detector=_SparseDetector(detections),
    )
    assert stats["flow_compensated_frames"] == 0
    assert stats["frames"] == 8
    assert stats["frames_with_block"] == 2  # 只有帧 3/4 有块


def test_mask_objects_flow_compensation_covers_multiple_targets(
    tmp_path: Path, monkeypatch
) -> None:
    """多目标：两个 label 各自的 mask 并集一起被光流补偿。"""
    monkeypatch.setattr(object_mask, "OBJECT_MASK_HOLD_FRAMES", 0)
    source = tmp_path / "src.mp4"
    frames = 5
    rng = np.random.default_rng(7)
    background = rng.integers(
        10, 40, (_FLOW_SIZE[1], _FLOW_SIZE[0], 3), dtype=np.uint8
    )
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, _FLOW_SIZE
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    for frame_index in range(frames):
        frame = background.copy()
        # 目标 A 右移、目标 B 左移，空洞帧（帧 2）两者都在中间位置
        frame[20:44, 6 + 4 * frame_index:30 + 4 * frame_index] = 240
        frame[70:94, 100 - 4 * frame_index:124 - 4 * frame_index] = 240
        writer.write(frame)
    writer.release()
    detections = [
        _rle_detection(0, 6, 20, 30, 44, "obj-a-pre", *_FLOW_SIZE),
        _rle_detection(1, 10, 20, 34, 44, "obj-a-pre", *_FLOW_SIZE),
        _rle_detection(3, 18, 20, 42, 44, "obj-a-post", *_FLOW_SIZE),
        _rle_detection(4, 22, 20, 46, 44, "obj-a-post", *_FLOW_SIZE),
        _rle_detection(0, 100, 70, 124, 94, "obj-b-pre", *_FLOW_SIZE),
        _rle_detection(1, 96, 70, 120, 94, "obj-b-pre", *_FLOW_SIZE),
        _rle_detection(3, 88, 70, 112, 94, "obj-b-post", *_FLOW_SIZE),
        _rle_detection(4, 84, 70, 108, 94, "obj-b-post", *_FLOW_SIZE),
    ]
    out = tmp_path / "out.mp4"
    stats = object_mask.mask_objects(
        source, out,
        ["obj-a-pre", "obj-a-post", "obj-b-pre", "obj-b-post"],
        detector=_SparseDetector(detections),
    )
    assert stats["flow_compensated_frames"] == 1
    frame2 = _read_frames(out)[2]
    for (y, x_mid) in ((20, 14), (70, 92)):  # 帧 2 两目标真实位置附近
        region = frame2[y:y + 24, x_mid - 4:x_mid + 20]
        grayish = (np.abs(region.astype(int) - 64).max(axis=2) <= 12).mean()
        assert grayish > 0.5, f"target near x={x_mid} not compensated"
