"""E2E 音频链抽验：从成片抽一段音轨，用方舟音频理解转写，验证语种正确无串中文。
凭证从 ~/.config/duet-ad1/dev-3252.env 读 ARK_AUDIO_*，不打印不不落盘。

用法: python transcribe_final_audio.py <video.mp4> <out_json>
"""
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ENV_PATH = Path.home() / ".config/duet-ad1/dev-3252.env"

creds = {}
for line in ENV_PATH.read_text().splitlines():
    line = line.strip()
    if line.startswith("ARK_AUDIO_") and "=" in line:
        k, v = line.split("=", 1)
        creds[k.strip()] = v.strip().strip('"').strip("'")

video = Path(sys.argv[1])
out_json = Path(sys.argv[2])

with tempfile.TemporaryDirectory() as td:
    clip = Path(td) / "clip.wav"
    # 抽中间 10 秒，16k 单声道 wav
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    dur = float(probe.stdout.strip())
    start = max(0.0, dur / 2 - 5)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", "10", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", str(clip)],
        check=True,
    )
    encoded = base64.b64encode(clip.read_bytes()).decode("ascii")

request = {
    "model": creds["ARK_AUDIO_MODEL"],
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "audio_url": f"data:audio/wav;base64,{encoded}"},
                {"type": "input_text", "text": (
                    "请转写这段音频的全部语音内容，并判断语种。只输出 JSON："
                    '{"transcript": "...", "language": "...", "contains_chinese": true/false}'
                )},
            ],
        }
    ],
}
with httpx.Client(timeout=300.0) as client:
    resp = client.post(
        f"{creds['ARK_AUDIO_API_BASE'].rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {creds['ARK_AUDIO_API_KEY']}", "Content-Type": "application/json"},
        json=request,
    )
if resp.status_code != 200:
    raise SystemExit(f"ark HTTP {resp.status_code}: {resp.text[:300]}")
body = resp.json()
texts = []
for item in body.get("output", []):
    for c in item.get("content", []):
        if c.get("type") in ("output_text", "text"):
            texts.append(c["text"])
result = {"video": str(video), "clip_start": start, "raw_text": "\n".join(texts)}
out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1))
print("wrote", out_json)
print(result["raw_text"][:800])
