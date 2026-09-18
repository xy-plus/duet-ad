"""Doubao Seed-TTS 2.0 client (Volcengine openspeech V3 HTTP unidirectional).

接口：`POST https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse`
（SSE 单向流式，文档 6561/1257584；同 key 对非 SSE 的 chunked 变体
会 403 not granted，SSE 变体才在授权范围内，2026-09-19 实测）。新版控制台鉴权只有一把
API Key（X-Api-Key），没有 appid/cluster 概念；X-Api-Resource-Id 固定为
seed-tts-2.0 常量，不是配置项。音频用 pcm 拉流（wav 在流式下会重复返回
wav header，官方建议 pcm），本地拼好 PCM 后包成 24kHz 单声道 16bit WAV
落盘。付费接口不在 client 内重试：歧义故障的重提由 pipeline `_node` 的
重试策略统一负责，避免重复计费。
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.providers import (
    ProviderFailure,
    ProviderResult,
    _binary_record,
    _request_record,
)


DOUBAO_TTS_API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"
DOUBAO_TTS_RESOURCE_ID = "seed-tts-2.0"
DOUBAO_TTS_SAMPLE_RATE = 24000
# 接口文本上限 1024 字节（UTF-8），官方建议 300 字符以内。
_DOUBAO_TTS_MAX_TEXT_BYTES = 1024
# code 0 = 音频/句子分片；20000000 = 合成正常结束；其余 code 均为失败。
_DOUBAO_TTS_DONE_CODE = 20000000

# 语言码 -> 性别 -> 豆包 2.0 voice_type（音色列表文档 6561/1257544，
# 每语种选通用/中性音色；标注"仅限单向流"的音色与本接口匹配）。
# th 只有女声、ms 只有男声、it 只有男声，缺性别时回退到该语种已有音色。
DOUBAO_TTS_VOICES: dict[str, dict[str, str]] = {
    "zh": {
        "female": "zh_female_vv_uranus_bigtts",
        "male": "zh_male_m191_uranus_bigtts",
    },
    "en": {
        "female": "en_female_jane_uranus_bigtts",
        "male": "en_male_alex_uranus_bigtts",
    },
    "th": {"female": "th_female_bv568_neutral_uranus_bigtts"},
    "ja": {
        "female": "ja_female_bv522_uranus_bigtts",
        "male": "ja_male_bv524_uranus_bigtts",
    },
    "ko": {
        "female": "ko_female_bv546_uranus_bigtts",
        "male": "ko_male_bv545_uranus_bigtts",
    },
    "es": {
        "female": "es_female_bv084_uranus_bigtts",
        "male": "es_male_dani_uranus_bigtts",
    },
    "fr": {
        "female": "fr_female_fr_bv078_uranus_bigtts",
        "male": "fr_male_fr_m29_uranus_bigtts",
    },
    "de": {
        "female": "de_female_bv081_uranus_bigtts",
        "male": "de_male_sven_uranus_bigtts",
    },
    "ru": {
        "female": "ru_female_af07_uranus_bigtts",
        "male": "ru_male_pavel_uranus_bigtts",
    },
    "pt": {
        "female": "pt_female_bv173_uranus_bigtts",
        "male": "pt_male_bv172_uranus_bigtts",
    },
    "id": {
        "female": "id_female_bv161_uranus_bigtts",
        "male": "id_male_bv160_uranus_bigtts",
    },
    "ms": {"male": "ms_male_ham_uranus_bigtts"},
    "vi": {
        "female": "vi_female_ling_uranus_bigtts",
        "male": "vi_male_wumg_uranus_bigtts",
    },
    "tl": {
        "female": "tl_female_annika_uranus_bigtts",
        "male": "tl_male_ed_uranus_bigtts",
    },
    "ar": {
        "female": "ar_female_dina_uranus_bigtts",
        "male": "ar_male_youssef_uranus_bigtts",
    },
    "it": {"male": "it_male_enzo_uranus_bigtts"},
}

_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "zh": ("中文", "汉语", "普通话", "chinese", "mandarin", "zh"),
    "en": ("英语", "英文", "english", "en"),
    "th": ("泰语", "泰文", "thai", "th"),
    "ja": ("日语", "日文", "japanese", "ja"),
    "ko": ("韩语", "朝鲜语", "韩文", "korean", "ko"),
    "es": ("西班牙语", "西语", "spanish", "es"),
    "fr": ("法语", "french", "fr"),
    "de": ("德语", "german", "de"),
    "ru": ("俄语", "russian", "ru"),
    "pt": ("葡萄牙语", "葡语", "portuguese", "pt"),
    "id": ("印度尼西亚语", "印尼语", "indonesian", "id"),
    "ms": ("马来语", "malay", "ms"),
    "vi": ("越南语", "vietnamese", "vi"),
    "tl": ("菲律宾语", "filipino", "tagalog", "tl"),
    "ar": ("阿拉伯语", "arabic", "ar"),
    "it": ("意大利语", "italian", "it"),
}

_KNOWN_VOICE_TYPES = frozenset(
    voice_type
    for voices in DOUBAO_TTS_VOICES.values()
    for voice_type in voices.values()
)


def doubao_tts_language_key(language_label: str) -> str:
    """归一化目标语言标签（如“泰语”/"Thai"/"th"）到映射表 key；未知回退 zh。"""
    text = str(language_label or "").strip().lower()
    if not text:
        return "zh"
    for key, aliases in _LANGUAGE_ALIASES.items():
        for alias in aliases:
            if text == alias or (len(alias) > 2 and alias in text):
                return key
    return "zh"


def doubao_tts_voice_type(language_label: str, gender: str | None = None) -> str:
    """按语言（+可选性别 male/female）选豆包 voice_type。"""
    voices = DOUBAO_TTS_VOICES[doubao_tts_language_key(language_label)]
    if gender in voices:
        return voices[gender]
    # 缺性别或该语种没有对应性别音色：优先女声，其次该语种唯一音色。
    return voices.get("female") or next(iter(voices.values()))


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """openspeech pcm 分片拼接结果是 16bit 小端单声道裸流，包一层 WAV 头。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class DoubaoTtsClient:
    """One-shot Doubao Seed-TTS 2.0 synthesis over HTTP Chunked JSON lines."""

    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        token = (
            self.settings.volc_voice_api_key.strip()
            if isinstance(self.settings.volc_voice_api_key, str)
            else ""
        )
        if not token:
            raise ValueError(
                "VOLC_VOICE_API_KEY must be set to a non-empty value "
                "(火山引擎控制台 API Key 管理 获取)"
            )
        return {
            "X-Api-Key": token,
            "X-Api-Resource-Id": DOUBAO_TTS_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }

    def synthesize(
        self,
        *,
        text: str,
        voice_type: str,
        destination: Path,
    ) -> ProviderResult[Path]:
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text.encode("utf-8")) > _DOUBAO_TTS_MAX_TEXT_BYTES
        ):
            raise ValueError("Doubao TTS text is invalid")
        if voice_type not in _KNOWN_VOICE_TYPES:
            raise ValueError("Doubao TTS voice_type is unknown")
        body = {
            "user": {"uid": "duet-dialogue-tts"},
            "req_params": {
                "text": text,
                "speaker": voice_type,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": DOUBAO_TTS_SAMPLE_RATE,
                },
            },
        }
        record: dict[str, Any] | None = None
        headers = self._headers()
        try:
            with httpx.Client(
                timeout=self.settings.request_timeout_s, transport=self.transport
            ) as client:
                request = client.build_request(
                    "POST", DOUBAO_TTS_API_URL, headers=headers, json=body
                )
                record = {"request": _request_record(request, body)}
                response = client.send(request, stream=True)
                try:
                    record["response"] = {
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                    }
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise ProviderFailure(record, [], exc) from exc
                    pcm = self._read_stream(response, record)
                finally:
                    response.close()
            wav = _pcm_to_wav(pcm, DOUBAO_TTS_SAMPLE_RATE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(wav)
            record["outcome"] = "synthesized"
            record["artifact"] = _binary_record(wav, "audio/wav")
            return ProviderResult(destination, record, [destination])
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record or {}, [], exc) from exc

    @staticmethod
    def _read_stream(response: httpx.Response, record: dict[str, Any]) -> bytes:
        chunks: list[bytes] = []
        chunk_count = 0
        finished = False
        for line in response.iter_lines():
            line = line.strip()
            if not line or line.startswith("event:"):
                continue
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Doubao TTS stream chunk is not valid JSON") from exc
            if not isinstance(event, dict):
                raise ValueError("Doubao TTS stream chunk must be an object")
            code = event.get("code")
            if code == 0:
                data = event.get("data")
                if data is None:
                    continue  # 句子时间戳分片，不携带音频
                if not isinstance(data, str):
                    raise ValueError("Doubao TTS audio chunk is invalid")
                try:
                    chunks.append(base64.b64decode(data, validate=True))
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("Doubao TTS audio chunk is invalid") from exc
                chunk_count += 1
                continue
            if code == _DOUBAO_TTS_DONE_CODE:
                finished = True
                usage = event.get("usage")
                if isinstance(usage, dict):
                    record["usage"] = usage
                continue
            raise RuntimeError(
                f"Doubao TTS synthesis failed: code={code} "
                f"message={event.get('message')!r}"
            )
        if not finished:
            raise ValueError("Doubao TTS stream ended without a finish event")
        pcm = b"".join(chunks)
        if not pcm or len(pcm) % 2:
            raise ValueError("Doubao TTS returned no valid PCM audio")
        record["response"]["audio_chunks"] = chunk_count
        record["response"]["pcm_bytes"] = len(pcm)
        return pcm
