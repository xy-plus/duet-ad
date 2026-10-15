"""audio-poc 共享库：路径、env、TTS/DeepSeek/方舟客户端。

自包含 POC，不 import 生产 app/。密钥只从 env 文件读入内存，绝不打印/落盘。
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx

# ---- 路径常量（逐字绝对路径） ----
POC_ROOT = Path("/home/xy/duet-ad1-h3-max-r2v/data-verify/audio-poc-20260925")
MODELS_DIR = POC_ROOT / "models"
PROJECTS_OUT = POC_ROOT / "projects"
OUTPUTS_DIR = POC_ROOT / "outputs"
PROD_REF_DIR = POC_ROOT / "production-reference"
VOICE_CATALOG = Path("/home/xy/duet-ad1-h3-max-r2v/assets/doubao-voices/catalog.json")

ENV_FILES = [
    Path("/home/xy/.config/duet-ad1/service.env"),
    Path("/home/xy/.config/claude/deepseek.env"),
    Path("/home/xy/duet-ad1-h3-max-r2v/.env"),
]

PROJECTS = {
    "b6240a72": "/home/xy/duet-ad1-h3-max-r2v/data/projects/b6240a721a244447abc697c55f6f9e1f",
    "c57d9d6c": "/home/xy/duet-ad1-h3-max-r2v/data/projects/c57d9d6c4c924c1a90b7fff35bdc6d64",
    "1cc3adb4": "/home/xy/duet-ad1-h3-max-r2v/data/projects/1cc3adb4e58442ea926242bc3ecb3063",
    "e46bf03d": "/home/xy/duet-ad1-h3-max-r2v/data/projects/e46bf03dcc514d06a6d56846ddb382ab",
    "6ecc60ca": "/home/xy/duet-ad1-h3-max-r2v/data/projects/6ecc60ca953147978c6926fdf3887e38",
}

DOUBAO_TTS_API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"
DOUBAO_TTS_RESOURCE_ID = "seed-tts-2.0"
DOUBAO_TTS_SAMPLE_RATE = 24000
_DOUBAO_TTS_DONE_CODE = 20000000

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
ARK_AUDIO_MODEL = "doubao-seed-2-0-lite-260428"

# capabilities 仅作提示词参考：实测 API 对 QA 音色也接受 context_texts/emotion，不作门禁

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_secrets() -> dict[str, str]:
    """环境变量优先（命令行注入的临时 key 不落盘），否则从 env 文件解析。
    密钥只进内存，绝不打印/落盘。env 文件间后者不覆盖前者。"""
    wanted = {
        "VOLC_VOICE_API_KEY",
        "ARK_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_MODEL",
    }
    found: dict[str, str] = {}
    for key in wanted:
        value = os.environ.get(key, "").strip()
        if value:
            found[key] = value
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = _ENV_LINE.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key not in wanted or key in found:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            found[key] = value
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"env 缺少必要变量: {sorted(missing)}")
    return found


def load_state(short_id: str) -> dict[str, Any]:
    return json.loads((Path(PROJECTS[short_id]) / "state.json").read_text())


def load_voice_catalog() -> dict[str, dict[str, Any]]:
    catalog = json.loads(VOICE_CATALOG.read_text())
    return {v["voice_type"]: v for v in catalog["voices"]}


def project_lines(short_id: str) -> list[dict[str, Any]]:
    """translated_voice_lines + 逐行音色，1 行 = 1 个合成任务。"""
    state = load_state(short_id)
    lines = state.get("translated_voice_lines") or []
    voices = (state.get("enhanced_element_index") or {}).get("dialogue_voices") or []
    if len(voices) != len(lines):
        raise ValueError(f"{short_id}: dialogue_voices 与 translated_voice_lines 行数不一致")
    out = []
    for i, (line, voice) in enumerate(zip(lines, voices), start=1):
        out.append({
            "index": i,
            "start_s": float(line["start_s"]),
            "end_s": float(line["end_s"]),
            "text": line["text"],
            "voice_type": voice,
        })
    return out


def project_dir(short_id: str) -> Path:
    return PROJECTS_OUT / short_id


def line_clip_path(short_id: str, index: int) -> Path:
    return project_dir(short_id) / "line-clips" / f"line-{index:04d}.wav"


def read_wav_float(path: Path) -> tuple[int, Any]:
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: 只支持 16bit PCM wav")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return rate, samples


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


# ---- 豆包 TTS 2.0 ----

def doubao_synthesize(
    secrets: dict[str, str],
    *,
    text: str,
    voice_type: str,
    destination: Path,
    speech_rate: int | None = None,
    emotion: str | None = None,
    context_texts: str | None = None,
) -> dict[str, Any]:
    """返回脱敏 request record（不含 key）。失败抛异常，调用方记录并继续。"""
    audio_params: dict[str, Any] = {
        "format": "pcm",
        "sample_rate": DOUBAO_TTS_SAMPLE_RATE,
    }
    if speech_rate is not None:
        audio_params["speech_rate"] = int(speech_rate)
    if emotion:
        audio_params["emotion"] = emotion
    req_params: dict[str, Any] = {
        "text": text,
        "speaker": voice_type,
        "audio_params": audio_params,
    }
    if context_texts:
        # additions 是 JSON string，不是嵌套对象
        req_params["additions"] = json.dumps(
            {"context_texts": [context_texts]}, ensure_ascii=False
        )
    body = {"user": {"uid": "duet-audio-poc"}, "req_params": req_params}
    record: dict[str, Any] = {"request": {"url": DOUBAO_TTS_API_URL, "body": body}}
    headers = {
        "X-Api-Key": secrets["VOLC_VOICE_API_KEY"].strip(),
        "X-Api-Resource-Id": DOUBAO_TTS_RESOURCE_ID,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    chunks: list[bytes] = []
    finished = False
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", DOUBAO_TTS_API_URL, headers=headers, json=body) as resp:
            record["status_code"] = resp.status_code
            if resp.status_code != 200:
                resp.read()
                raise RuntimeError(f"TTS HTTP {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if not line or line.startswith("event:"):
                    continue
                event = json.loads(line)
                code = event.get("code")
                if code == 0:
                    data = event.get("data")
                    if isinstance(data, str):
                        try:
                            chunks.append(base64.b64decode(data, validate=True))
                        except (binascii.Error, ValueError):
                            pass
                elif code == _DOUBAO_TTS_DONE_CODE:
                    finished = True
                    if isinstance(event.get("usage"), dict):
                        record["usage"] = event["usage"]
                else:
                    raise RuntimeError(
                        f"TTS failed code={code} message={event.get('message')!r}"
                    )
    pcm = b"".join(chunks)
    if not finished:
        raise RuntimeError("TTS stream ended without finish event")
    if not pcm or len(pcm) % 2:
        raise RuntimeError("TTS returned empty PCM")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pcm_to_wav_bytes(pcm, DOUBAO_TTS_SAMPLE_RATE))
    record["pcm_bytes"] = len(pcm)
    record["outcome"] = "synthesized"
    return record


# ---- DeepSeek Responses API（结构化输出） ----

def deepseek_structured(
    secrets: dict[str, str],
    *,
    instruction: str,
    payload: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    base = secrets["DEEPSEEK_API_BASE"].rstrip("/")
    if base.endswith("/beta"):
        base = base[:-5]
    url = base + "/responses"
    request = {
        "model": secrets["DEEPSEEK_MODEL"],
        "instructions": instruction,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}
                ],
            }
        ],
        "text": {"format": {"type": "json_schema", "name": "audio_poc_plan", "schema": schema}},
    }
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {secrets['DEEPSEEK_API_KEY'].strip()}",
                "Content-Type": "application/json",
            },
            json=request,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:400]}")
    body = resp.json()
    if body.get("status") != "completed":
        raise RuntimeError(f"DeepSeek status={body.get('status')!r}")
    texts = [
        item["text"]
        for out in body.get("output") or []
        if isinstance(out, dict) and out.get("type") == "message"
        for item in out.get("content") or []
        if isinstance(item, dict) and item.get("type") == "output_text"
    ]
    if len(texts) != 1:
        raise RuntimeError("DeepSeek 结构化输出缺少 output_text")
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        return json.loads(texts[0], strict=False)


# ---- 方舟音频理解 ----

def ark_audio_understand(
    secrets: dict[str, str],
    *,
    audio_path: Path,
    prompt: str,
    model: str = ARK_AUDIO_MODEL,
) -> str:
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    request = {
        "model": model,
        "instructions": "你是音频理解专家，擅长分析广告配音音频的语速、语调、情绪、停顿与整体语气。",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio_url": f"data:audio/wav;base64,{encoded}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            ARK_BASE + "/responses",
            headers={
                "Authorization": f"Bearer {secrets['ARK_API_KEY'].strip()}",
                "Content-Type": "application/json",
            },
            json=request,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"方舟 HTTP {resp.status_code}: {resp.text[:400]}")
    body = resp.json()
    if body.get("status") not in (None, "completed"):
        raise RuntimeError(f"方舟 status={body.get('status')!r} detail={str(body)[:300]}")
    texts = [
        item["text"]
        for out in body.get("output") or []
        if isinstance(out, dict) and out.get("type") == "message"
        for item in out.get("content") or []
        if isinstance(item, dict) and item.get("type") == "output_text"
    ]
    if not texts:
        raise RuntimeError(f"方舟返回无 output_text: {str(body)[:300]}")
    return texts[0]
