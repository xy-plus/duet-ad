"""Dog seg1 audio-form A/B: same masked video + sheet, voice wav + speak-lines prompt.

Tests whether the v1 speech-track form (TTS slice + lip-sync prompt) is what
flips H3 into conservative dubbing mode. This run uses the old package:
voice-library wav as reference audio + "Speak the following lines" dialogue
block; everything else identical to project 61b070ff segment-0001.

Video/audio are self-hosted via web/.tmp-ab on :3301 (publicly reachable);
images are inlined as base64 data URIs.

Usage (env needs PICSART_API_KEY):
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_dog_audioform.py submit
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_dog_audioform.py poll
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOG = (
    PROJECT_ROOT.parent
    / "duet-ad1-h3-max-r2v-release/data/projects/61b070ffd36a441fa97c7d53396fc829"
).resolve()
VIDEO = DOG / "work/segments/0001/source-faces-masked.mp4"
IMAGES = [
    DOG / "work/global-reference-sheet/reference-sheet.png",
    DOG / "work/global-reference-sheet/user-reference.png",
]
AUDIO = DOG / "work/voice-references/audio-01.wav"
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/h3-dog-audioform"

API_BASE = "https://api.picsart.com"
WORKFLOW = "minimax/h3-max/reference-to-video"
PUBLIC_BASE = "https://8.166.140.227:3301/.tmp-ab"
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 3600

OLD_DIALOGUE_HEADER = (
    "Dialogue requirements: Speak the following lines naturally in the "
    "target language. Do not speak the original-language dialogue. Do not "
    "render captions or subtitles. Finish every line within the frozen "
    "source-segment time window. Do not add dialogue in padded time.\n"
)


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _prompt() -> str:
    recorded = json.loads(
        (DOG / "logs/picsart-submit/segment-0001/attempt-1/input.json").read_text(
            encoding="utf-8"
        )
    )["prompt"]
    match = re.search(r"Dialogue requirements:.*", recorded, re.S)
    if match is None:
        raise ValueError("recorded prompt has no dialogue block")
    lines = match.group(0).split("\n", 1)[1]  # keep the "[time] text" lines
    return recorded[: match.start()] + OLD_DIALOGUE_HEADER + lines


def main() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_file = OUTPUT_DIR / "h3-task-voicewav.json"
    web_tmp = PROJECT_ROOT / "web/.tmp-ab"
    web_tmp.mkdir(parents=True, exist_ok=True)
    for src_path, name in ((VIDEO, "dog-seg1-masked.mp4"), (AUDIO, "dog-voice.wav")):
        shutil.copyfile(src_path, web_tmp / name)
    params = {
        "workflow": WORKFLOW,
        "prompt": _prompt(),
        "reference_video_urls": [f"{PUBLIC_BASE}/dog-seg1-masked.mp4"],
        "reference_image_urls": [_data_uri(p, "image/png") for p in IMAGES],
        "reference_audio_urls": [f"{PUBLIC_BASE}/dog-voice.wav"],
        "resolution": "768P",
        "duration": 10,
        "aspect_ratio": "9:16",
        "prompt_expansion_mode": "quality",
    }
    with httpx.Client(
        timeout=httpx.Timeout(300.0),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    ) as client:
        if command == "submit":
            if task_file.is_file():
                print("already submitted", flush=True)
                return 0
            response = client.post(
                f"{API_BASE}/workflows/{WORKFLOW}/submit", json={"params": params}
            )
            if response.status_code not in (200, 201):
                print(f"rejected {response.status_code} {response.text[:400]}")
                return 1
            task_id = response.json()["response"]["id"]
            slim = dict(params)
            slim["reference_image_urls"] = ["<data uri>", "<data uri>"]
            task_file.write_text(
                json.dumps({"task_id": task_id, "params": slim}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"submitted {task_id}", flush=True)
            return 0
        if command == "poll":
            record = json.loads(task_file.read_text(encoding="utf-8"))
            if record.get("downloaded"):
                print("already downloaded", flush=True)
                return 0
            url = f"{API_BASE}/workflows/{WORKFLOW}/{record['task_id']}/result"
            deadline = time.monotonic() + POLL_TIMEOUT_S
            while True:
                envelope = client.get(url).json()["response"]
                status = str(envelope.get("status") or "").upper()
                if status == "COMPLETED":
                    video_url = envelope["result"]["video"]["url"]
                    break
                if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                    print(f"ended {status}: {envelope}", flush=True)
                    return 1
                if time.monotonic() > deadline:
                    print("poll timeout", flush=True)
                    return 1
                print(f"status={status}", flush=True)
                time.sleep(POLL_INTERVAL_S)
            destination = OUTPUT_DIR / "h3-result-voicewav.mp4"
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            record["downloaded"] = str(destination)
            task_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"downloaded {destination}", flush=True)
            return 0
    print("usage: h3_dog_audioform.py submit|poll", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
