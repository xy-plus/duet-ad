"""H3 输入画幅推荐与显式适配；浏览器不得自行猜测媒体几何。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


class FrameFitError(RuntimeError):
    """关键帧无法按已确认的画幅策略派生。"""


TARGET_ASPECTS = {"16:9": (16, 9), "9:16": (9, 16)}
TARGET_RESOLUTIONS = {"480p": 480, "768p": 768}


def _target_ratio(aspect_ratio: str) -> tuple[int, int]:
    target = TARGET_ASPECTS.get(aspect_ratio)
    if target is None:
        raise FrameFitError("aspect_ratio must be 16:9 or 9:16")
    return target


def _decode_frame(data: bytes, label: str) -> np.ndarray:
    if not isinstance(data, bytes) or not data:
        raise FrameFitError(f"cannot decode keyframe: {label}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise FrameFitError(f"cannot decode keyframe: {label}")
    return image


def normalize_image_to_canvas_png(
    data: bytes,
    canvas_width: int,
    canvas_height: int,
    *,
    label: str = "image",
) -> bytes:
    """Decode any supported raster into a centered BGR PNG canvas.

    Content is scaled uniformly to fit inside the requested canvas.  A ratio
    mismatch is padded with black pixels; content is never stretched or
    cropped.
    """
    if (
        isinstance(canvas_width, bool)
        or not isinstance(canvas_width, int)
        or isinstance(canvas_height, bool)
        or not isinstance(canvas_height, int)
        or canvas_width <= 0
        or canvas_height <= 0
    ):
        raise FrameFitError("canvas dimensions must be positive integers")
    image = _decode_frame(data, label)
    source_height, source_width = image.shape[:2]
    if source_width * canvas_height > canvas_width * source_height:
        fitted_width = canvas_width
        fitted_height = max(1, source_height * canvas_width // source_width)
    else:
        fitted_height = canvas_height
        fitted_width = max(1, source_width * canvas_height // source_height)
    if (fitted_width, fitted_height) != (source_width, source_height):
        interpolation = (
            cv2.INTER_AREA
            if fitted_width < source_width or fitted_height < source_height
            else cv2.INTER_CUBIC
        )
        image = cv2.resize(
            image, (fitted_width, fitted_height), interpolation=interpolation,
        )
    horizontal = canvas_width - fitted_width
    vertical = canvas_height - fitted_height
    if horizontal or vertical:
        image = cv2.copyMakeBorder(
            image,
            vertical // 2,
            vertical - vertical // 2,
            horizontal // 2,
            horizontal - horizontal // 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise FrameFitError(f"cannot encode image: {label}")
    return encoded.tobytes()


def _byte_dimensions(frames: Sequence[bytes]) -> list[tuple[int, int]]:
    snapshots = list(frames)
    if not snapshots:
        raise FrameFitError("frame set must not be empty")
    result = []
    for position, data in enumerate(snapshots, 1):
        image = _decode_frame(data, f"frame-{position}")
        height, width = image.shape[:2]
        result.append((width, height))
    return result


def _dimensions(paths: Sequence[Path]) -> list[tuple[int, int]]:
    inputs = [Path(path) for path in paths]
    try:
        snapshots = [path.read_bytes() for path in inputs]
    except OSError as exc:
        raise FrameFitError(f"cannot read keyframe: {exc.filename}") from None
    return _byte_dimensions(snapshots)


def frame_bytes_require_fit(
    frames: Sequence[bytes], aspect_ratio: str,
) -> bool:
    """Judge exact immutable H3 frame bytes against an explicit target."""
    target_width, target_height = _target_ratio(aspect_ratio)
    return any(
        width * target_height != height * target_width
        for width, height in _byte_dimensions(frames)
    )


def frames_require_fit(paths: Sequence[Path], aspect_ratio: str) -> bool:
    """Return whether any actual H3 input frame differs from the target."""
    target_width, target_height = _target_ratio(aspect_ratio)
    return any(
        width * target_height != height * target_width
        for width, height in _dimensions(paths)
    )


def _generation_fit_profiles(
    dimensions: Sequence[tuple[int, int]], *,
    source_width: int, source_height: int,
) -> tuple[dict[str, dict[str, object]], str]:
    """Recommend one closed aspect by total symmetric ratio loss.

    Equal H3-input loss is resolved by the source-video orientation; a square
    source leaves the product-level 9:16 fallback as the final deterministic tie.
    """
    if (
        isinstance(source_width, bool)
        or isinstance(source_height, bool)
        or not isinstance(source_width, int)
        or not isinstance(source_height, int)
        or source_width <= 0
        or source_height <= 0
    ):
        raise FrameFitError("source dimensions must be positive integers")
    losses = {
        aspect: sum(
            abs(math.log((width / height) / (target_width / target_height)))
            for width, height in dimensions
        )
        for aspect, (target_width, target_height) in TARGET_ASPECTS.items()
    }
    minimum = min(losses.values())
    candidates = [
        aspect for aspect in TARGET_ASPECTS
        if math.isclose(losses[aspect], minimum, rel_tol=0, abs_tol=1e-12)
    ]
    if len(candidates) == 1:
        selected = candidates[0]
    elif source_width > source_height:
        selected = "16:9"
    elif source_height > source_width:
        selected = "9:16"
    else:
        selected = "9:16"
    fit_requirements = {
        aspect: any(
            width * target_height != height * target_width
            for width, height in dimensions
        )
        for aspect, (target_width, target_height) in TARGET_ASPECTS.items()
    }
    profiles = {
        aspect: {
            "fit_required": required,
            "default_fit_mode": "crop" if required else "none",
        }
        for aspect, required in fit_requirements.items()
    }
    return profiles, selected


def generation_fit_profiles(
    paths: Sequence[Path], *, source_width: int, source_height: int,
) -> tuple[dict[str, dict[str, object]], str]:
    return _generation_fit_profiles(
        _dimensions(paths), source_width=source_width, source_height=source_height
    )


def generation_fit_profiles_from_bytes(
    frames: Sequence[bytes], *, source_width: int, source_height: int,
) -> tuple[dict[str, dict[str, object]], str]:
    """Recommend from the exact immutable frame bytes frozen for H3."""
    return _generation_fit_profiles(
        _byte_dimensions(frames),
        source_width=source_width,
        source_height=source_height,
    )


def recommended_resolution(source_short_edge: int) -> str:
    if (
        isinstance(source_short_edge, bool)
        or not isinstance(source_short_edge, int)
        or source_short_edge <= 0
    ):
        raise FrameFitError("source short edge must be a positive integer")
    return min(
        TARGET_RESOLUTIONS,
        key=lambda value: (
            abs(source_short_edge - TARGET_RESOLUTIONS[value]),
            TARGET_RESOLUTIONS[value],
        ),
    )


def _target_size(
    width: int, height: int, mode: str, aspect_ratio: str,
) -> tuple[int, int]:
    ratio_width, ratio_height = _target_ratio(aspect_ratio)
    if mode == "crop":
        scale = min(width // ratio_width, height // ratio_height)
    elif mode == "pad":
        scale = max(
            math.ceil(width / ratio_width), math.ceil(height / ratio_height)
        )
    else:
        raise FrameFitError("fit_mode must be crop or pad")
    scale = max(1, scale)
    return ratio_width * scale, ratio_height * scale


def _fit(image: np.ndarray, mode: str, aspect_ratio: str) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = _target_size(
        width, height, mode, aspect_ratio
    )
    if mode == "crop":
        if width < target_width or height < target_height:
            scale = max(target_width / width, target_height / height)
            width = max(target_width, round(width * scale))
            height = max(target_height, round(height * scale))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        return image[top : top + target_height, left : left + target_width].copy()

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def fit_frame_bytes(
    data: bytes, mode: str, aspect_ratio: str, *, label: str = "frame",
) -> bytes:
    """Derive encoded PNG bytes from one immutable source-frame snapshot."""
    if mode not in {"crop", "pad"}:
        raise FrameFitError("fit_mode must be crop or pad")
    _target_ratio(aspect_ratio)
    fitted = _fit(_decode_frame(data, label), mode, aspect_ratio)
    ok, encoded = cv2.imencode(".png", fitted)
    if not ok:
        raise FrameFitError(f"cannot encode keyframe: {label}")
    return encoded.tobytes()


def fit_frames(
    paths: Sequence[Path], output_dir: Path, mode: str, aspect_ratio: str,
) -> tuple[Path, ...]:
    """从给定原始帧生成同名 PNG；绝不推断或默认选择适配模式。"""
    if mode not in {"crop", "pad"}:
        raise FrameFitError("fit_mode must be crop or pad")
    _target_ratio(aspect_ratio)
    inputs = [Path(path) for path in paths]
    if not inputs or len(inputs) > 9:
        raise FrameFitError("keyframe count must be in 1..9")
    names = [path.name for path in inputs]
    if len(names) != len(set(names)):
        raise FrameFitError("keyframe names must be unique")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in inputs:
        try:
            data = source.read_bytes()
        except OSError:
            raise FrameFitError(f"cannot decode keyframe: {source.name}")
        encoded = fit_frame_bytes(
            data, mode, aspect_ratio, label=source.name
        )
        output = output_dir / (source.stem + ".png")
        temporary = output.with_suffix(output.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded)
            temporary.replace(output)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise FrameFitError(f"cannot write keyframe: {source.name}") from None
        outputs.append(output)
    return tuple(outputs)
