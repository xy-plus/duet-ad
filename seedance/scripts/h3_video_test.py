"""Masked-video isolation test: is v2's reference sheet what breaks scene replacement?

Known good (gray2, today): probe seg1 gray-masked video + OLD sheet + old audio
+ old Chinese prompt -> face swapped, scene changed, no subtitles.
A/B (today): v2 seg2 masked video + V2 sheet + v2 audio + either prompt ->
scene kept.

This test swaps exactly one variable against the known-good run: the reference
images become V2's sheet + V2's user reference, with a condensed Chinese prompt
written for V2's tile bindings. Video and audio stay on the probe's litterbox
URLs (identical to the successful gray/gray2 submissions).

  - scene changes -> v2's sheet is fine; the culprit is v2's masked video or
    segment content.
  - scene stays   -> v2's sheet is the culprit.

Usage (env needs PICSART_API_KEY):
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_sheet_test.py submit
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_sheet_test.py poll
"""

from __future__ import annotations

import base64
import json
import shutil
import os
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2 = PROJECT_ROOT / "data/projects/0dfa41c017194b93a3282a5191e62bb0"
IMAGES = [
    V2 / "work/global-reference-sheet/reference-sheet.png",
    V2 / "work/global-reference-sheet/user-reference.png",
]
PROBE_HOSTED = (
    PROJECT_ROOT / "data/smoke-runs/face-mosaic-probe/hosted-urls.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/h3-sheet-test"

API_BASE = "https://api.picsart.com"
WORKFLOW = "minimax/h3-max/reference-to-video"
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 3600

PROMPT = (
    "FACE REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER EVERYTHING ELSE. "
    "Every person in Video 1 must appear with the face from their tile in "
    "Image 1: face shape, eyes, nose, mouth, skin tone, and hairstyle all "
    "come from that tile, never from Video 1. Keeping the source face is a "
    "failure, even if everything else is correct.\n\n"
    "当前视频片段为第一段。画面为正面中景的明亮客厅：浅灰色沙发配蓝白花纹靠枕，"
    "白色窗框的窗户挂浅蓝色窗帘，柔和自然日光从左侧射入（即 Image 1 图版 TILE_07 的场景实例）。"
    "一位东南亚年轻男子坐在沙发上：黑色短直发、干净的棱角脸型、浅小麦色肤色，"
    "穿带白色小logo的黑色圆领T恤，戴银色细链圆吊坠项链、左手银色手表、右手食指银戒指"
    "（人物整体形象必须来自 TILE_01，不得使用 Video 1 的脸）。"
    "他右手持一支黑色电动牙刷（teal色刷头、银色电源键，TILE_02），面向镜头说话、"
    "大笑并快速挥动牙刷；随后双手各持一支止汗膏举至胸前展示："
    "右手橙盖明黄色瓶身（TILE_03），左手紫盖浅灰色瓶身（TILE_04）。\n\n"
    "替换要求：人物用 TILE_01；牙刷用 TILE_02；两支止汗膏分别用 TILE_03、TILE_04；"
    "场景用 TILE_07。新身份与新设计只从 Image 1 对应 tile 复制。"
    "保持人物的动作、姿态、位置、与手中物体的接触关系不变；"
    "场景仅保留房间类型、布局、色调与光照方向。"
    "\n\nOverlay removal: Video 1 may contain burned-in subtitles, captions, "
    "on-screen text, watermarks and logos. Remove all of them and reconstruct "
    "the occluded background naturally. Do not reproduce, translate, or "
    "re-render any source on-screen text, watermark, or logo."
    "\n\nDialogue requirements: Speak the following lines naturally in the "
    "target language. Do not speak the original-language dialogue. Do not "
    "render captions or subtitles. Finish every line within the frozen "
    "source-segment time window. Do not add dialogue in padded time.\n"
    "[0.000-2.440s] 你们每天都刷牙\n"
    "[2.440-4.400s] 但还是黄黄的，还开始有味道\n"
    "[4.400-5.920s] 还是搞不定，对吧？\n"
    "[5.920-8.000s] 就是因为它太脏了\n"
    "[8.000-8.300s] 你"
)


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_file = OUTPUT_DIR / "h3-task-v2video-seg1.json"
    hosted = json.loads(PROBE_HOSTED.read_text(encoding="utf-8"))
    PUBLIC_BASE = "https://8.166.140.227:3301/.tmp-ab"
    web_tmp = PROJECT_ROOT / "web/.tmp-ab"
    web_tmp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V2 / "work/segments/0001/source-faces-masked.mp4", web_tmp / "v2-seg1-masked.mp4")
    params = {
        "workflow": WORKFLOW,
        "prompt": PROMPT,
        "reference_video_urls": [f"{PUBLIC_BASE}/v2-seg1-masked.mp4"],
        "reference_image_urls": [_data_uri(p, "image/png") for p in IMAGES],
        "reference_audio_urls": list(hosted["audio"]),
        "resolution": "768P",
        "duration": 9,
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
                f"{API_BASE}/workflows/{WORKFLOW}/submit",
                json={"params": params},
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
                    print(f"task ended with {status}: {envelope}", flush=True)
                    return 1
                if time.monotonic() > deadline:
                    print("poll timeout", flush=True)
                    return 1
                print(f"status={status}", flush=True)
                time.sleep(POLL_INTERVAL_S)
            destination = OUTPUT_DIR / "h3-result-v2video-seg1.mp4"
            with client.stream("GET", video_url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            record["downloaded"] = str(destination)
            task_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"downloaded {destination}", flush=True)
            return 0
    print("usage: h3_sheet_test.py submit|poll", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
