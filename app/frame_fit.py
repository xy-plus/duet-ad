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


def _dimensions(paths: Sequence[Path]) -> list[tuple[int, int]]:
    inputs = [Path(path) for path in paths]
    if not inputs:
        raise FrameFitError("frame set must not be empty")
    result = []
    for source in inputs:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise FrameFitError(f"cannot decode keyframe: {source.name}")
        height, width = image.shape[:2]
        result.append((width, height))
    return result


def frames_require_fit(paths: Sequence[Path], aspect_ratio: str) -> bool:
    """Return whether any actual H3 input frame differs from the target."""
    target_width, target_height = _target_ratio(aspect_ratio)
    return any(
        width * target_height != height * target_width
        for width, height in _dimensions(paths)
    )


def generation_fit_profiles(
    paths: Sequence[Path], *, source_width: int, source_height: int,
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
    dimensions = _dimensions(paths)
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
    profiles = {
        aspect: {
            "fit_required": any(
                width * target_height != height * target_width
                for width, height in dimensions
            ),
            "default_fit_mode": (
                "crop"
                if any(
                    width * target_height != height * target_width
                    for width, height in dimensions
                )
                else "none"
            ),
        }
        for aspect, (target_width, target_height) in TARGET_ASPECTS.items()
    }
    return profiles, selected


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
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise FrameFitError(f"cannot decode keyframe: {source.name}")
        fitted = _fit(image, mode, aspect_ratio)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            raise FrameFitError(f"cannot encode keyframe: {source.name}")
        output = output_dir / (source.stem + ".png")
        temporary = output.with_suffix(output.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(output)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise FrameFitError(f"cannot write keyframe: {source.name}") from None
        outputs.append(output)
    return tuple(outputs)
