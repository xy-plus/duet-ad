"""显式把原始关键帧适配为 9:16；调用方必须先得到用户的 crop/pad 选择。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


class FrameFitError(RuntimeError):
    """关键帧无法按已确认的画幅策略派生。"""


def _decode_frame(data: bytes, label: str) -> np.ndarray:
    if not isinstance(data, bytes) or not data:
        raise FrameFitError(f"cannot decode keyframe: {label}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise FrameFitError(f"cannot decode keyframe: {label}")
    return image


def frame_bytes_require_fit(frames: Sequence[bytes]) -> bool:
    """Judge exact immutable H3 frame bytes, decoding every supplied frame."""
    snapshots = list(frames)
    if not snapshots:
        raise FrameFitError("frame set must not be empty")
    required = False
    for position, data in enumerate(snapshots, 1):
        image = _decode_frame(data, f"frame-{position}")
        height, width = image.shape[:2]
        if width * 16 != height * 9:
            required = True
    return required


def frames_require_fit(paths: Sequence[Path]) -> bool:
    """Read each H3 input path once, then judge the immutable snapshots."""
    inputs = [Path(path) for path in paths]
    try:
        snapshots = [path.read_bytes() for path in inputs]
    except OSError as exc:
        raise FrameFitError(f"cannot read keyframe: {exc.filename}") from None
    return frame_bytes_require_fit(snapshots)


def _target_size(width: int, height: int, mode: str) -> tuple[int, int]:
    if mode == "crop":
        scale = min(width // 9, height // 16)
    elif mode == "pad":
        scale = max(math.ceil(width / 9), math.ceil(height / 16))
    else:
        raise FrameFitError("fit_mode must be crop or pad")
    if scale < 1:
        raise FrameFitError("frame is too small for 9:16 fitting")
    return 9 * scale, 16 * scale


def _fit(image: np.ndarray, mode: str) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = _target_size(width, height, mode)
    if mode == "crop":
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        return image[top : top + target_height, left : left + target_width].copy()

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def fit_frame_bytes(data: bytes, mode: str, *, label: str = "frame") -> bytes:
    """Derive encoded PNG bytes from one immutable source-frame snapshot."""
    if mode not in {"crop", "pad"}:
        raise FrameFitError("fit_mode must be crop or pad")
    fitted = _fit(_decode_frame(data, label), mode)
    ok, encoded = cv2.imencode(".png", fitted)
    if not ok:
        raise FrameFitError(f"cannot encode keyframe: {label}")
    return encoded.tobytes()


def fit_frames(paths: Sequence[Path], output_dir: Path, mode: str) -> tuple[Path, ...]:
    """从给定原始帧生成同名 PNG；绝不推断或默认选择适配模式。"""
    if mode not in {"crop", "pad"}:
        raise FrameFitError("fit_mode must be crop or pad")
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
        encoded = fit_frame_bytes(data, mode, label=source.name)
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
