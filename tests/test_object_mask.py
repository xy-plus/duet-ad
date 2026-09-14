from __future__ import annotations

import json
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

    assert stats == {
        "frames": 8,
        "frames_with_block": 8,
        "blocks_total": 8,
        "anchor_frames": 2,
        "hold_frames": 4,
        "interpolated_frames": 2,
        "targets": {"mug": {"detections": 2, "detected_frames": 2}},
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


class _SingleTargetDetector:
    """Test seam mirroring fal_sam3: single_target_only + with_cache hook."""

    single_target_only = True

    def __init__(self, detections_by_target: dict[str, list[ObjectDetection]]) -> None:
        self.detections_by_target = detections_by_target
        self.calls: list[tuple[Path, list[str]]] = []
        self.cache_paths: list[Path] = []

    def with_cache(self, cache_path: Path) -> "_SingleTargetDetector":
        self.cache_paths.append(cache_path)
        return self

    def detect(self, video_path: Path, targets: list[str]) -> list[ObjectDetection]:
        assert len(targets) == 1, "single_target_only detector got multi targets"
        self.calls.append((video_path, list(targets)))
        return list(self.detections_by_target.get(targets[0], []))


def test_mask_objects_splits_single_target_only_detector_per_target(
    tmp_path: Path,
) -> None:
    """fal 型 provider：N target = N 次独立 detect + 各自产物文件，合并盖块。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3)
    detector = _SingleTargetDetector({
        "mug": [_detection(0.0, (0.1, 0.1, 0.4, 0.4))],
        "scarf": [_detection(1.0, (0.5, 0.5, 0.9, 0.9), label="scarf")],
    })
    artifacts = tmp_path / "segment"
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug", "scarf"],
        detector=detector, detect_artifacts_dir=artifacts,
    )
    assert detector.calls == [(source, ["mug"]), (source, ["scarf"])]
    assert detector.cache_paths == [
        artifacts / f"object-detect-response-{object_mask._target_slug('mug')}.json",
        artifacts / f"object-detect-response-{object_mask._target_slug('scarf')}.json",
    ]
    detections_file = artifacts / "object-detections.json"
    assert detections_file.is_file()
    payload = json.loads(detections_file.read_text(encoding="utf-8"))
    assert payload["targets"] == ["mug", "scarf"]
    assert [entry["label"] for entry in payload["detections"]] == ["mug", "scarf"]
    assert stats["targets"] == {
        "mug": {"detections": 1, "detected_frames": 1},
        "scarf": {"detections": 1, "detected_frames": 1},
    }


def test_mask_objects_single_target_empty_detection_does_not_block_others(
    tmp_path: Path,
) -> None:
    """某 target 无检出：记日志跳过，其它 target 照常盖块。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=3)
    detector = _SingleTargetDetector({
        "mug": [_detection(0.0, (0.1, 0.1, 0.4, 0.4))],
        "ghost": [],
    })
    stats = object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug", "ghost"],
        detector=detector, detect_artifacts_dir=tmp_path / "segment",
    )
    assert stats["blocks_total"] > 0
    assert stats["targets"]["ghost"] == {"detections": 0, "detected_frames": 0}
    assert stats["targets"]["mug"]["detections"] == 1


def test_mask_objects_multi_target_detector_called_once(
    tmp_path: Path,
) -> None:
    """VLM 型 provider（无 single_target_only）：一次调用全部 targets。"""
    source = tmp_path / "source-muted.mp4"
    _solid_video(source, frames=2)
    detector = _SparseDetector([_detection(0.0, (0.1, 0.1, 0.4, 0.4))])
    artifacts = tmp_path / "segment"
    object_mask.mask_objects(
        source, tmp_path / "out.mp4", ["mug", "scarf"],
        detector=detector, detect_artifacts_dir=artifacts,
    )
    assert detector.calls == [(source, ["mug", "scarf"])]
    # VLM 无产物复用钩子：只有解析后的 detections JSON，无响应产物文件。
    assert (artifacts / "object-detections.json").is_file()
    assert not list(artifacts.glob("object-detect-response*.json"))


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
    "rle": ["rle-frame-0"],
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
            assert "," not in body["prompt"], "fal 单目标限制被穿透"
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
    """真实 fal client（mock HTTP）：2 target = 2 提交 + 2 产物文件；
    二次运行命中指纹零 HTTP，且结果一致。"""
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
    assert state["submits"] == 2
    assert state["uploads"] == 2
    response_files = sorted(artifacts.glob("object-detect-response-*.json"))
    assert len(response_files) == 2
    for path in response_files:
        cached = json.loads(path.read_text(encoding="utf-8"))
        # 原始响应字段一个不丢（rle/boxes/metadata 原样）。
        assert cached["fal_response"] == _FAL_QUEUE_RESULT
        assert set(cached["fingerprint"]) == {
            "video_sha256", "prompt", "detection_threshold", "model",
        }
    assert stats["blocks_total"] == 4  # 2 target × (帧 0 锚点 + 帧 1 续盖)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    reused = FalSam3GroundingClient(
        _fal_test_settings(tmp_path),
        transport=httpx.MockTransport(fail_handler),
    )
    stats2 = object_mask.mask_objects(
        source, tmp_path / "out2.mp4", ["handbag", "bottle"],
        detector=reused, detect_artifacts_dir=artifacts,
    )
    assert stats2["blocks_total"] == stats["blocks_total"]
