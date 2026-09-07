"""Face-mosaic probe: mask every face in project 8e124288 segment-1 source-muted.mp4.

Hypothesis under test: H3 (Picsart reference-to-video) keeps the source face
because it can see it. If every face in Video 1 is masked, H3 has no source
identity and must take the face from the Image 1 tile.

Produces two masked variants (plus detection-overlay check frames):
  - mosaic: heavy pixelation inside each (expanded) face box
  - gray:   solid neutral-gray block inside each (expanded) face box

Subcommands (env: source .env first so PICSART_API_KEY is set):
    face_mosaic_probe.py mask            # detect + mask, write both variants
    face_mosaic_probe.py submit          # submit both variants to H3
    face_mosaic_probe.py poll            # poll pending tasks, download results

H3 submission reuses the exact recorded params of segment-0001 attempt-1
(same prompt, resolution, duration, aspect ratio); only the reference URLs
change. The live backend's register_asset is in-memory and its previously
signed URLs expired with the service restart. Video/audio data URIs are
rejected by the workflow at processing time (audio loses its extension),
so video/audio are hosted on litterbox.catbox.moe (see hosted-urls.json)
and images are inlined as base64 data URIs (documented support).

Run with the project venv (has cv2, no onnxruntime needed):
    /home/xy/duet-ad1/.venv/bin/python scripts/face_mosaic_probe.py mask
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/face_detection_yunet_2023mar.onnx"
PROJECT_DIR = PROJECT_ROOT / "data/projects/8e124288db2d43beb2c555f6762ce50f"
SOURCE = PROJECT_DIR / "work/segments/0001/source-muted.mp4"
RECORDED_INPUT = (
    PROJECT_DIR / "logs/picsart-submit/segment-0001/attempt-1/input.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/face-mosaic-probe"

API_BASE = "https://api.picsart.com"
WORKFLOW = "minimax/h3-max/reference-to-video"
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 3600

# Local files behind the (now expired) recorded reference URLs, per
# state.json replacement_sheet.h3_reference_images / voice_references.
IMAGE_FILES = [
    PROJECT_DIR / "work/global-reference-sheet/reference-sheet.png",
    PROJECT_DIR / "work/global-reference-sheet/user-reference.png",
]
AUDIO_FILES = [PROJECT_DIR / "work/voice-references/audio-01.wav"]

SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
PAD_RATIO = 0.3  # expand each side of the face box by 30% of its width/height
GRAY_LEVEL = 128
MOSAIC_BLOCKS = 12  # pixel blocks across the masked box width


def detect_faces(
    detector: cv2.FaceDetectorYN, frame: np.ndarray
) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    boxes: list[tuple[int, int, int, int]] = []
    if faces is None:
        return boxes
    for face in faces:
        x, y, w, h = (float(v) for v in face[:4])
        pad_x, pad_y = w * PAD_RATIO, h * PAD_RATIO
        x0 = max(0, int(round(x - pad_x)))
        y0 = max(0, int(round(y - pad_y)))
        x1 = min(width, int(round(x + w + pad_x)))
        y1 = min(height, int(round(y + h + pad_y)))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def apply_mosaic(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    region = frame[y0:y1, x0:x1]
    block = max(1, (x1 - x0) // MOSAIC_BLOCKS)
    small_w = max(1, (x1 - x0) // block)
    small_h = max(1, (y1 - y0) // block)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(
        small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
    )


def apply_gray(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = GRAY_LEVEL


def mask_video(mode: str, destination: Path) -> dict:
    """mode: 'mosaic' or 'gray'. Returns per-run detection stats."""
    capture = cv2.VideoCapture(str(SOURCE))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {SOURCE}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, 5000
    )
    raw_path = destination.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    apply = apply_mosaic if mode == "mosaic" else apply_gray
    stats = {"frames": 0, "frames_with_face": 0, "faces_total": 0, "max_faces": 0}
    overlays: list[tuple[int, np.ndarray]] = []
    check_indices = {0, 124, 248}  # first / middle / last for a 249-clip
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        boxes = detect_faces(detector, frame)
        stats["frames"] += 1
        stats["faces_total"] += len(boxes)
        stats["max_faces"] = max(stats["max_faces"], len(boxes))
        if boxes:
            stats["frames_with_face"] += 1
        if index in check_indices:
            overlay = frame.copy()
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
            overlays.append((index, overlay))
        for box in boxes:
            apply(frame, box)
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    # Re-encode to H.264/yuv420p for provider compatibility.
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(raw_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart",
            str(destination),
        ],
        check=True,
    )
    raw_path.unlink()
    for frame_index, overlay in overlays:
        overlay_path = destination.with_name(
            f"{destination.stem}-overlay-f{frame_index:04d}.png"
        )
        cv2.imwrite(str(overlay_path), overlay)
    return stats


def main_mask() -> int:
    if not MODEL_PATH.is_file():
        print(f"missing model {MODEL_PATH}", file=sys.stderr)
        return 2
    if not SOURCE.is_file():
        print(f"missing source {SOURCE}", file=sys.stderr)
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mode in ("mosaic", "gray", "gray2"):
        destination = OUTPUT_DIR / f"seg1-face-{mode}.mp4"
        stats = mask_video(mode, destination)
        stats["output"] = str(destination)
        stats["size_bytes"] = destination.stat().st_size
        summary[mode] = stats
        print(f"{mode}: {json.dumps(stats)}", flush=True)
    (OUTPUT_DIR / "mask-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


# --- H3 submission (adapted from scripts/h3_seg1_trials.py) -----------------


def _data_uri(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _api_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(300.0),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )


def _build_params(mode: str) -> dict:
    params = json.loads(RECORDED_INPUT.read_text(encoding="utf-8"))
    hosted = json.loads((OUTPUT_DIR / "hosted-urls.json").read_text(encoding="utf-8"))
    # Video/audio data URIs are rejected by the h3-max workflow (audio loses
    # its extension and fails as ".bin"); use hosted URLs for those.
    # Images are documented to support base64 data URIs, so they stay inline.
    # "gray2" is a same-params rerun of "gray" to detect provider-side drift.
    params["reference_video_urls"] = [hosted["video"].get(mode, hosted["video"]["gray"])]
    params["reference_image_urls"] = [
        _data_uri(path, "image/png") for path in IMAGE_FILES
    ]
    params["reference_audio_urls"] = list(hosted["audio"])
    return params


def _task_file(mode: str) -> Path:
    return OUTPUT_DIR / f"h3-task-{mode}.json"


def main_submit() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    modes = [m for m in sys.argv[2:] if m in ("mosaic", "gray", "gray2")] or ["mosaic", "gray"]
    recorded = json.loads(RECORDED_INPUT.read_text(encoding="utf-8"))
    with _api_client(api_key) as client:
        for mode in modes:
            task_file = _task_file(mode)
            if task_file.is_file():
                print(f"{mode}: task file exists, skip submit", flush=True)
                continue
            params = _build_params(mode)
            # Persist everything except the bulky data URIs for review.
            slim = {
                key: (
                    ["<data uri>" if str(v).startswith("data:") else v for v in value]
                    if key.endswith("_urls")
                    else value
                )
                for key, value in params.items()
            }
            assert params["prompt"] == recorded["prompt"]
            assert params["duration"] == recorded["duration"]
            assert params["resolution"] == recorded["resolution"]
            assert params["aspect_ratio"] == recorded["aspect_ratio"]
            response = client.post(
                f"{API_BASE}/workflows/{WORKFLOW}/submit",
                json={"params": params},
            )
            if response.status_code not in (200, 201):
                print(
                    f"{mode}: submit rejected {response.status_code} "
                    f"{response.text[:500]}",
                    flush=True,
                )
                return 1
            task_id = response.json()["response"]["id"]
            task_file.write_text(
                json.dumps({"task_id": task_id, "mode": mode, "params": slim},
                           ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"{mode}: submitted task_id={task_id}", flush=True)
    return 0


def main_poll() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    failures: dict[str, str] = {}
    with _api_client(api_key) as client:
        for mode in ("mosaic", "gray", "gray2"):
            task_file = _task_file(mode)
            if not task_file.is_file():
                print(f"{mode}: no task file, run submit first", flush=True)
                failures[mode] = "not submitted"
                continue
            record = json.loads(task_file.read_text(encoding="utf-8"))
            if record.get("downloaded"):
                print(f"{mode}: already downloaded", flush=True)
                continue
            task_id = record["task_id"]
            try:
                video_url = _poll_task(client, task_id)
                destination = OUTPUT_DIR / f"h3-result-{mode}.mp4"
                _download(client, video_url, destination)
                record["video_url"] = video_url
                record["downloaded"] = str(destination)
                task_file.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # keep going with the other variant
                print(f"{mode}: FAILED: {exc}", flush=True)
                failures[mode] = str(exc)
    return 0 if not failures else 1


def _poll_task(client: httpx.Client, task_id: str) -> str:
    url = f"{API_BASE}/workflows/{WORKFLOW}/{task_id}/result"
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        response = client.get(url)
        response.raise_for_status()
        envelope = response.json()["response"]
        status = str(envelope.get("status") or "").upper()
        if status == "COMPLETED":
            video_url = envelope["result"]["video"]["url"]
            print(f"task {task_id} COMPLETED {video_url}", flush=True)
            return video_url
        if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"task {task_id} ended with status {status}: {envelope}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"task {task_id} poll timeout")
        print(f"task {task_id} status={status}", flush=True)
        time.sleep(POLL_INTERVAL_S)


def _download(client: httpx.Client, url: str, destination: Path) -> None:
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    print(f"downloaded {destination} ({destination.stat().st_size} bytes)", flush=True)


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "mask"
    if command == "mask":
        return main_mask()
    if command == "submit":
        return main_submit()
    if command == "poll":
        return main_poll()
    print(f"unknown command {command!r} (mask|submit|poll)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
