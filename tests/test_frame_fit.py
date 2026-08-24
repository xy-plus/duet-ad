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
@pytest.mark.parametrize("aspect_ratio", ["16:9", "9:16"])
def test_fit_frames_generates_exact_selected_aspect(tmp_path, mode, aspect_ratio):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    outputs = frame_fit.fit_frames(
        [source], tmp_path / "derived", mode, aspect_ratio
    )
    assert len(outputs) == 1
    assert outputs[0] != source
    image = cv2.imread(str(outputs[0]))
    target_width, target_height = map(int, aspect_ratio.split(":"))
    assert image.shape[1] * target_height == image.shape[0] * target_width


def test_pad_uses_black_borders_and_crop_keeps_center(tmp_path):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    padded = cv2.imread(str(frame_fit.fit_frames(
        [source], tmp_path / "pad", "pad", "9:16"
    )[0]))
    cropped = cv2.imread(str(frame_fit.fit_frames(
        [source], tmp_path / "crop", "crop", "9:16"
    )[0]))
    assert np.all(padded[0, 0] == 0)
    assert cropped.shape[0] == 80 and cropped.shape[1] == 45


def test_none_or_invalid_mode_cannot_silently_transform(tmp_path):
    source = tmp_path / "input.png"
    _write(source, 160, 90)
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.fit_frames([source], tmp_path / "out", "none", "9:16")
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.fit_frames([source], tmp_path / "out", "cover", "9:16")
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.fit_frames([source], tmp_path / "out", "crop", "1:1")


def test_frames_require_fit_is_false_when_every_h3_anchor_is_9_by_16(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    _write(first, 90, 160)
    _write(last, 180, 320)

    assert frame_fit.frames_require_fit([first, last], "9:16") is False


def test_frames_require_fit_is_true_when_any_h3_anchor_is_not_9_by_16(tmp_path):
    portrait = tmp_path / "portrait.png"
    landscape = tmp_path / "landscape.png"
    _write(portrait, 90, 160)
    _write(landscape, 160, 90)

    assert frame_fit.frames_require_fit([portrait, landscape], "9:16") is True
    assert frame_fit.frames_require_fit([portrait, landscape], "16:9") is True


def test_frames_require_fit_rejects_empty_or_undecodable_anchor_sets(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")

    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frames_require_fit([], "9:16")
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frames_require_fit([broken], "9:16")


def test_frame_bytes_require_fit_uses_the_supplied_immutable_png_bytes(tmp_path):
    portrait = tmp_path / "portrait.png"
    landscape = tmp_path / "landscape.png"
    _write(portrait, 90, 160)
    _write(landscape, 160, 90)

    assert frame_fit.frame_bytes_require_fit(
        [portrait.read_bytes()], "9:16"
    ) is False
    assert frame_fit.frame_bytes_require_fit(
        [portrait.read_bytes(), landscape.read_bytes()], "9:16"
    ) is True


def test_frame_bytes_require_fit_rejects_invalid_bytes():
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frame_bytes_require_fit([], "9:16")
    with pytest.raises(frame_fit.FrameFitError):
        frame_fit.frame_bytes_require_fit([b"not-png"], "9:16")


def test_generation_defaults_use_total_h3_geometry_and_source_tiebreak(tmp_path):
    first = tmp_path / "landscape.png"
    second = tmp_path / "portrait.png"
    _write(first, 160, 90)
    _write(second, 90, 160)

    profiles, selected = frame_fit.generation_fit_profiles(
        [first, second], source_width=1920, source_height=1080
    )

    assert selected == "16:9"
    assert profiles == {
        "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        "9:16": {"fit_required": True, "default_fit_mode": "crop"},
    }


@pytest.mark.parametrize(
    ("short_edge", "expected"),
    [(479, "480p"), (480, "480p"), (624, "480p"), (625, "768p"),
     (626, "768p"), (768, "768p")],
)
def test_resolution_default_uses_nearest_source_short_edge(short_edge, expected):
    assert frame_fit.recommended_resolution(short_edge) == expected
