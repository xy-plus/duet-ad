"""Deterministic YuNet gray-block face masking for the H3 reference video.

H3 keeps the source face when it can see it; covering every detected face
in the muted reference video forces H3 to take the identity from the sheet
tile. The detector runs on CPU via cv2.FaceDetectorYN and the local YuNet
ONNX model; the output stays a pure video stream at the source fps/resolution.
The mask is a solid gray block (not a mosaic): every facial feature must be
completely erased, with no texture or color leaking through.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import media


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models" / "face_detection_yunet_2023mar.onnx"
)
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
PAD_RATIO = 0.35  # expand top/sides of the face box by 35% of its width/height
CHIN_PAD_RATIO = 0.5  # expand the bottom further: YuNet boxes end above the chin
GRAY_FILL = (128, 128, 128)  # solid neutral gray (BGR) filling the face box


class FaceMaskError(RuntimeError):
    """Face detection or masking could not be completed deterministically."""


def _face_boxes(
    detector: Any, frame: np.ndarray
) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    boxes: list[tuple[int, int, int, int]] = []
    if faces is None:
        return boxes
    for face in faces:
        x, y, w, h = (float(v) for v in face[:4])
        pad_x, pad_top = w * PAD_RATIO, h * PAD_RATIO
        pad_bottom = h * CHIN_PAD_RATIO
        x0 = max(0, int(round(x - pad_x)))
        y0 = max(0, int(round(y - pad_top)))
        x1 = min(width, int(round(x + w + pad_x)))
        y1 = min(height, int(round(y + h + pad_bottom)))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def _apply_gray_block(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Fill the box in place with a solid gray that hides every feature."""
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = GRAY_FILL


def mask_faces(
    source: Path,
    destination: Path,
    *,
    detector: Any | None = None,
) -> dict[str, Any]:
    """Cover every detected face with a gray block, keeping fps/resolution."""
    if not isinstance(source, Path) or not source.is_file():
        raise FaceMaskError("face mask source video is invalid")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FaceMaskError(f"could not open video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if width < 1 or height < 1 or fps <= 0:
        capture.release()
        raise FaceMaskError("face mask source metadata is incomplete")
    if detector is None:
        if not MODEL_PATH.is_file():
            capture.release()
            raise FaceMaskError("face detection model is missing")
        try:
            detector = cv2.FaceDetectorYN.create(
                str(MODEL_PATH),
                "",
                (width, height),
                SCORE_THRESHOLD,
                NMS_THRESHOLD,
                5000,
            )
        except cv2.error as exc:
            capture.release()
            raise FaceMaskError(
                "face detector initialization failed"
            ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".faces-masked-",
        suffix=".mp4",
        dir=destination.parent,
        delete=False,
    ) as handle:
        raw = Path(handle.name)
    stats = {"frames": 0, "frames_with_face": 0, "faces_total": 0}
    try:
        writer = cv2.VideoWriter(
            str(raw),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise FaceMaskError("could not open the face mask writer")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            boxes = _face_boxes(detector, frame)
            stats["frames"] += 1
            stats["faces_total"] += len(boxes)
            if boxes:
                stats["frames_with_face"] += 1
            for x0, y0, x1, y1 in boxes:
                _apply_gray_block(frame, (x0, y0, x1, y1))
            writer.write(frame)
        capture.release()
        writer.release()
        if stats["frames"] < 1:
            raise FaceMaskError("face mask source has no frames")
        # Re-encode to H.264/yuv420p for provider compatibility.
        media.run_command(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(raw),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-an", "-movflags", "+faststart",
                str(destination),
            ],
            timeout=300,
        )
    finally:
        raw.unlink(missing_ok=True)
    return stats
