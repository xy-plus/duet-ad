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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_mask_faces_covers_the_expanded_box_and_preserves_timing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-muted.mp4"
    _solid_video(source)
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
    # 30% padding grows the 16x16 box at (4, 4) to [0:25, 0:25].
    inside = frames[0][12, 12]
    assert all(abs(int(channel) - 128) <= 8 for channel in inside)
    outside = frames[0][28, 40]
    assert all(
        abs(int(channel) - original) <= 8
        for channel, original in zip(outside, (10, 20, 30))
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
