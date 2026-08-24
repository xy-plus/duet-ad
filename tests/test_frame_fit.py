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


def test_frames_require_fit_is_false_when_every_h3_anchor_is_9_by_16(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    _write(first, 90, 160)
    _write(last, 180, 320)

    assert frame_fit.frames_require_fit([first, last]) is False


def test_frames_require_fit_is_true_when_any_h3_anchor_is_not_9_by_16(tmp_path):
    portrait = tmp_path / "portrait.png"
    landscape = tmp_path / "landscape.png"
    _write(portrait, 90, 160)
    _write(landscape, 160, 90)

    assert frame_fit.frames_require_fit([portrait, landscape]) is True


def test_frames_require_fit_rejects_empty_or_undecodable_anchor_sets(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")

    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frames_require_fit([])
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frames_require_fit([broken])


def test_frame_bytes_require_fit_uses_the_supplied_immutable_png_bytes(tmp_path):
    portrait = tmp_path / "portrait.png"
    landscape = tmp_path / "landscape.png"
    _write(portrait, 90, 160)
    _write(landscape, 160, 90)

    assert frame_fit.frame_bytes_require_fit([portrait.read_bytes()]) is False
    assert frame_fit.frame_bytes_require_fit(
        [portrait.read_bytes(), landscape.read_bytes()]
    ) is True


def test_frame_bytes_require_fit_rejects_invalid_bytes():
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frame_bytes_require_fit([])
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frame_bytes_require_fit([b"not-png"])
