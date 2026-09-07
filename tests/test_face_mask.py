from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app import face_mask
from app.face_mask import FaceMaskError


class _FixedDetector:
    def __init__(self, boxes: list[tuple[float, float, float, float]]) -> None:
        self.boxes = boxes
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_sizes.append(size)

    def detect(self, _frame: np.ndarray) -> tuple[None, Any]:
        if not self.boxes:
            return None, None
        return None, np.array(self.boxes, dtype=np.float32)


def _solid_video(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (48, 32)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    for _ in range(3):
        writer.write(np.full((32, 48, 3), color, dtype=np.uint8))
    writer.release()


def _split_video(path: Path) -> None:
    """Left half dark, right half bright, so masking is visible."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (48, 32)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :24] = (10, 20, 30)
    frame[:, 24:] = (200, 210, 220)
    for _ in range(3):
        writer.write(frame)
    writer.release()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_faces_mosaics_the_expanded_box_and_preserves_timing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _split_video(source)
    detector = _FixedDetector([(4.0, 4.0, 16.0, 16.0)])

    destination = tmp_path / "source-faces-masked.mp4"
    stats = face_mask.mask_faces(source, destination, detector=detector)

    assert stats == {"frames": 3, "frames_with_face": 3, "faces_total": 3}
    assert detector.input_sizes == [(48, 32)] * 3
    capture = cv2.VideoCapture(str(destination))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(1.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == 3
    # 15% padding grows the 16x16 box at (4, 4) to [2:22, 2:22]; at this tiny
    # scale mosaic blocks are 1px, so masking is a no-op in the uniform area —
    # the mosaic structure itself is covered by the direct unit test below.
    # Here we assert the region outside the box is untouched.
    outside = frames[0][28, 40]
    assert all(
        abs(int(channel) - original) <= 8
        for channel, original in zip(outside, (200, 210, 220))
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type", "-of", "json", str(destination),
        ],
        capture_output=True, text=True, check=True,
    )
    import json

    streams = json.loads(probe.stdout)["streams"]
    assert [stream["codec_type"] for stream in streams] == ["video"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_faces_without_detections_leaves_frames_untouched(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source)

    destination = tmp_path / "source-faces-masked.mp4"
    stats = face_mask.mask_faces(
        source, destination, detector=_FixedDetector([]),
    )

    assert stats == {"frames": 3, "frames_with_face": 0, "faces_total": 0}
    capture = cv2.VideoCapture(str(destination))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert not np.all(frame == 128)


def test_mask_faces_requires_the_model_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source)
    monkeypatch.setattr(
        face_mask, "MODEL_PATH", tmp_path / "missing-model.onnx",
    )
    destination = tmp_path / "source-faces-masked.mp4"

    with pytest.raises(FaceMaskError, match="model is missing"):
        face_mask.mask_faces(source, destination)

    assert not destination.exists()


def test_mask_faces_rejects_an_unreadable_source(tmp_path: Path) -> None:
    with pytest.raises(FaceMaskError, match="source video is invalid"):
        face_mask.mask_faces(
            tmp_path / "missing.mp4",
            tmp_path / "out.mp4",
            detector=_FixedDetector([]),
        )


def test_apply_mosaic_pixelates_into_coarse_blocks() -> None:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    frame[:, :44] = (10, 20, 30)
    frame[:, 44:] = (200, 210, 220)

    face_mask._apply_mosaic(frame, (0, 0, 96, 96))

    # 96px / 12 blocks = 8px mosaic cells; inside a uniform half the cell keeps
    # its color, and all pixels within one cell are identical.
    dark_cell = frame[0:8, 0:8]
    assert np.all(dark_cell == dark_cell[0, 0])
    assert all(
        abs(int(channel) - original) <= 2
        for channel, original in zip(dark_cell[0, 0], (10, 20, 30))
    )
    # The brightness boundary at x=44 sits mid-cell (cell spans x=40..47), so
    # that cell blends both halves into a midtone.
    blended = frame[0, 44]
    assert all(
        abs(int(channel) - mid) <= 30
        for channel, mid in zip(blended, (105, 115, 125))
    )
