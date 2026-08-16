#!/usr/bin/env python3
"""Crop an image to a pixel box: --box left,top,right,bottom."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit(
        "OpenCV is required. Install opencv-python-headless in a task-local environment."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop an image to --box left,top,right,bottom (pixels)."
    )
    parser.add_argument("image", help="Input image file.")
    parser.add_argument("--box", required=True, help="left,top,right,bottom in pixels.")
    parser.add_argument("--out", help="Output path (default: overwrite in place).")
    return parser.parse_args()


def parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--box must contain comma-separated integers.") from exc
    if len(parts) != 4:
        raise SystemExit("--box must have exactly 4 integers: left,top,right,bottom.")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not decode image: {image_path}")

    height, width = image.shape[:2]
    left, top, right, bottom = parse_box(args.box)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise SystemExit(
            f"--box {left},{top},{right},{bottom} outside image bounds {width}x{height}."
        )

    cropped = image[top:bottom, left:right]
    dest = Path(args.out).expanduser().resolve() if args.out else image_path
    success, encoded = cv2.imencode(dest.suffix or ".png", cropped)
    if not success:
        raise SystemExit(f"Could not encode image: {dest}")
    encoded.tofile(str(dest))
    print(f"{dest} {cropped.shape[1]}x{cropped.shape[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
