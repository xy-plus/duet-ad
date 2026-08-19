from pathlib import Path

import cv2
import numpy as np
import pytest

from app import frame_fit


def _write(path: Path, width: int, height: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (10, 20, 30)
    assert cv2.imwrite(str(path), image)


@pytest.mark.parametrize("mode", ["crop", "pad"])
def test_fit_frames_generates_exact_9_by_16_images(tmp_path, mode):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    outputs = frame_fit.fit_frames([source], tmp_path / "derived", mode)
    assert len(outputs) == 1
    assert outputs[0] != source
    image = cv2.imread(str(outputs[0]))
    assert image.shape[1] * 16 == image.shape[0] * 9


def test_pad_uses_black_borders_and_crop_keeps_center(tmp_path):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    padded = cv2.imread(str(frame_fit.fit_frames([source], tmp_path / "pad", "pad")[0]))
    cropped = cv2.imread(str(frame_fit.fit_frames([source], tmp_path / "crop", "crop")[0]))
    assert np.all(padded[0, 0] == 0)
    assert cropped.shape[0] == 80 and cropped.shape[1] == 45


def test_none_or_invalid_mode_cannot_silently_transform(tmp_path):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.fit_frames([source], tmp_path / "out", "none")
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.fit_frames([source], tmp_path / "out", "cover")
