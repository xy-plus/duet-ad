"""H3 prompt A/B probe on project 0dfa41c0 segment-0002 (the bad segment).

Same masked reference video, tiles, voice wav and submit params; only the
prompt differs:

  - v2prompt:   the exact 9231-char English prompt v2 submitted (seg2 kept the
                scene and reproduced the burned-in subtitle verbatim).
  - condensed:  a ~1k-char Chinese prompt in the old-project style covering the
                same replacements (person TILE_01, tablet TILE_05, Listerine
                TILE_06, scene TILE_07) and the same five dialogue lines.

Hosting: litterbox/0x0.st are down, so video/audio are served from our own
public static dir web/.tmp-ab/ on port 3301 (the mount is unauthenticated and
H3 has fetched from this host before). Images stay inline as base64 data URIs.

Usage (env needs PICSART_API_KEY; source .env first):
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_prompt_ab.py submit
    /home/xy/duet-ad1/.venv/bin/python scripts/h3_prompt_ab.py poll
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2 = PROJECT_ROOT / "data/projects/0dfa41c017194b93a3282a5191e62bb0"
SEGMENT = V2 / "work/segments/0002"
VIDEO = SEGMENT / "source-faces-masked.mp4"
IMAGES = [
    V2 / "work/global-reference-sheet/reference-sheet.png",
    V2 / "work/global-reference-sheet/user-reference.png",
]
AUDIO = V2 / "work/voice-references/audio-01.wav"
V2_PROMPT = V2 / "logs/picsart-submit/segment-0002/attempt-1/input.json"
OUTPUT_DIR = PROJECT_ROOT / "data/smoke-runs/h3-prompt-ab"
WEB_TMP = PROJECT_ROOT / "web/.tmp-ab"

API_BASE = "https://api.picsart.com"
WORKFLOW = "minimax/h3-max/reference-to-video"
PUBLIC_BASE = "https://8.166.140.227:3301/.tmp-ab"
POLL_INTERVAL_S = 15
POLL_TIMEOUT_S = 3600

FACE_BLOCK = (
    "FACE REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER EVERYTHING ELSE. "
    "Every person in Video 1 must appear with the face from their tile in "
    "Image 1: face shape, eyes, nose, mouth, skin tone, and hairstyle all "
    "come from that tile, never from Video 1. Keeping the source face is a "
    "failure, even if everything else is correct.\n\n"
)
OVERLAY_BLOCK = (
    "\n\nOverlay removal: Video 1 may contain burned-in subtitles, captions, "
    "on-screen text, watermarks and logos. Remove all of them and reconstruct "
    "the occluded background naturally. Do not reproduce, translate, or "
    "re-render any source on-screen text, watermark, or logo."
)
DIALOGUE_BLOCK = (
    "\n\nDialogue requirements: Speak the following lines naturally in the "
    "target language. Do not speak the original-language dialogue. Do not "
    "render captions or subtitles. Finish every line within the frozen "
    "source-segment time window. Do not add dialogue in padded time.\n"
    "[0.000-1.460s] 们只需要吃一颗\n"
    "[1.460-2.660s] 嚼一嚼就行\n"
    "[2.660-4.300s] 等10秒让它起泡\n"
    "[4.300-6.060s] 之后你们直接漱口\n"
    "[6.060-7.700s] 这背后其实是益生菌开始发挥作用"
)

CONDENSED_PROMPT = FACE_BLOCK + (
    "当前视频片段为第二段。正面中景固定机位。明亮客厅：浅灰色沙发配蓝白花纹靠枕，"
    "白色窗框的窗户挂浅蓝色窗帘，柔和自然日光（即 Image 1 图版 TILE_07 的场景实例）。"
    "一位东南亚年轻男子：黑色短直发、干净的棱角脸型、浅小麦色肤色，"
    "穿带白色小logo的黑色圆领T恤，戴银色细链圆吊坠项链、左手银表、右手食指银戒指"
    "（人物整体形象必须来自 TILE_01，不得使用 Video 1 的脸）。"
    "他先微笑着用右手拇指和食指捏起一颗酒红色椭圆形亮面药片（TILE_05）举到嘴边放入口中，"
    "低头吞咽后抬头露齿大笑；随后举起一瓶1L装 Listerine Cool Mint 漱口水"
    "（TILE_06：透明瓶身、蓝色罗纹瓶盖、白色标签）就口饮用，闭眼仰头。"
    "结尾画面叠化：人物与瓶子半透明双重曝光。\n\n"
    "替换要求：人物用 TILE_01；药片用 TILE_05；漱口水瓶用 TILE_06；场景用 TILE_07。"
    "新身份与新设计只从 Image 1 对应 tile 复制。"
    "保持人物动作、姿态、位置、与物体的接触关系、构图与光照方向不变。"
) + OVERLAY_BLOCK + DIALOGUE_BLOCK


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _host_assets(client: httpx.Client) -> dict:
    WEB_TMP.mkdir(parents=True, exist_ok=True)
    staged = {}
    for key, src in (("video", VIDEO), ("audio", AUDIO)):
        destination = WEB_TMP / src.name
        shutil.copyfile(src, destination)
        staged[key] = f"{PUBLIC_BASE}/{src.name}"
    for url in staged.values():
        response = client.get(url)
        response.raise_for_status()
        print(f"self-hosted {url} ({len(response.content)} bytes)", flush=True)
    return staged


def _params(prompt: str, hosted: dict) -> dict:
    recorded = json.loads(V2_PROMPT.read_text(encoding="utf-8"))
    return {
        "workflow": WORKFLOW,
        "prompt": prompt,
        "reference_video_urls": [hosted["video"]],
        "reference_image_urls": [_data_uri(p, "image/png") for p in IMAGES],
        "reference_audio_urls": [hosted["audio"]],
        "resolution": recorded["resolution"],
        "duration": recorded["duration"],
        "aspect_ratio": recorded["aspect_ratio"],
        "prompt_expansion_mode": recorded["prompt_expansion_mode"],
    }


def _poll_task(client: httpx.Client, task_id: str) -> str:
    url = f"{API_BASE}/workflows/{WORKFLOW}/{task_id}/result"
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while True:
        envelope = client.get(url).json()["response"]
        status = str(envelope.get("status") or "").upper()
        if status == "COMPLETED":
            return envelope["result"]["video"]["url"]
        if status in {"FAILED", "ERROR", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"task {task_id} ended with {status}: {envelope}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"task {task_id} poll timeout")
        print(f"task {task_id} status={status}", flush=True)
        time.sleep(POLL_INTERVAL_S)


def main() -> int:
    api_key = os.environ.get("PICSART_API_KEY", "").strip()
    if not api_key:
        print("PICSART_API_KEY is required", file=sys.stderr)
        return 2
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    variants = {
        "v2prompt": json.loads(V2_PROMPT.read_text(encoding="utf-8"))["prompt"],
        "condensed": CONDENSED_PROMPT,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        timeout=httpx.Timeout(300.0),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        verify=False,
    ) as client:
        if command == "submit":
            hosted = _host_assets(client)
            for name, prompt in variants.items():
                task_file = OUTPUT_DIR / f"h3-task-{name}.json"
                if task_file.is_file():
                    print(f"{name}: already submitted", flush=True)
                    continue
                params = _params(prompt, hosted)
                response = client.post(
                    f"{API_BASE}/workflows/{WORKFLOW}/submit",
                    json={"params": params},
                )
                if response.status_code not in (200, 201):
                    print(f"{name}: rejected {response.status_code} {response.text[:400]}")
                    return 1
                task_id = response.json()["response"]["id"]
                slim = dict(params)
                slim["reference_image_urls"] = ["<data uri>", "<data uri>"]
                task_file.write_text(
                    json.dumps({"task_id": task_id, "params": slim}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"{name}: submitted {task_id} (prompt {len(prompt)} chars)", flush=True)
            return 0
        if command == "poll":
            failures = {}
            for name in variants:
                task_file = OUTPUT_DIR / f"h3-task-{name}.json"
                record = json.loads(task_file.read_text(encoding="utf-8"))
                if record.get("downloaded"):
                    continue
                try:
                    video_url = _poll_task(client, record["task_id"])
                    destination = OUTPUT_DIR / f"h3-result-{name}.mp4"
                    with client.stream("GET", video_url) as response:
                        response.raise_for_status()
                        with destination.open("wb") as handle:
                            for chunk in response.iter_bytes():
                                handle.write(chunk)
                    record["downloaded"] = str(destination)
                    task_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(f"{name}: downloaded {destination}", flush=True)
                except Exception as exc:
                    print(f"{name}: FAILED {exc}", flush=True)
                    failures[name] = str(exc)
            return 0 if not failures else 1
    print("usage: h3_prompt_ab.py submit|poll", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
