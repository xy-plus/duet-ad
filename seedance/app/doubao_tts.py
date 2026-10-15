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
from app.doubao_voices import known_voice_types
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

# record.outcome 中间态标记：流已正常收完 finish、只剩本地落盘。带此标记
# 的 ProviderFailure（如落盘 ENOSPC）说明音频已入服务端 section 历史，
# pipeline 重试分类据此判死（fail-closed），禁止同 section 重发——重发会
# 重复扣费并污染服务端 section 历史。
STREAM_FINISHED_OUTCOME = "stream-finished"


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """openspeech pcm 分片拼接结果是 16bit 小端单声道裸流，包一层 WAV 头。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


class EmptyAudioError(ValueError):
    """流正常结束（含 finish 事件）但没有任何音频分片：同一文本+音色重发
    必然得到同样结果，属于确定性失败，禁止重试。"""


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
        speech_rate: int | None = None,
        emotion: str | None = None,
        context_texts: list[str] | None = None,
        section_id: str | None = None,
        explicit_language: str | None = None,
    ) -> ProviderResult[Path]:
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text.encode("utf-8")) > _DOUBAO_TTS_MAX_TEXT_BYTES
        ):
            raise ValueError("Doubao TTS text is invalid")
        if voice_type not in known_voice_types():
            raise ValueError("Doubao TTS voice_type is unknown")
        # speech_rate/emotion 进 audio_params；emotion="neutral" 是无意义参数
        # （接口默认即 neutral），不发送。
        if speech_rate is not None and (
            isinstance(speech_rate, bool)
            or not isinstance(speech_rate, int)
            or not -50 <= speech_rate <= 100
        ):
            raise ValueError("Doubao TTS speech_rate must be an integer in [-50, 100]")
        if emotion is not None and (
            not isinstance(emotion, str) or not emotion.strip()
        ):
            raise ValueError("Doubao TTS emotion must be a non-empty string")
        if context_texts is not None and (
            not isinstance(context_texts, list)
            or any(
                not isinstance(item, str) or not item.strip()
                for item in context_texts
            )
        ):
            raise ValueError("Doubao TTS context_texts must be non-empty strings")
        if section_id is not None and (
            not isinstance(section_id, str) or not section_id.strip()
        ):
            raise ValueError("Doubao TTS section_id must be a non-empty string")
        # explicit_language（additions 字段，火山单向流文档）：锁定多语种
        # 精品音色（*_uranus_bigtts）的朗读语种；全量锁后同语种合成也传入
        # （app.dialogue_delivery._explicit_language_for_voice 的决策）。
        if explicit_language is not None and (
            not isinstance(explicit_language, str) or not explicit_language.strip()
        ):
            raise ValueError("Doubao TTS explicit_language must be a non-empty string")
        audio_params: dict[str, Any] = {
            "format": "pcm",
            "sample_rate": DOUBAO_TTS_SAMPLE_RATE,
        }
        if speech_rate is not None:
            audio_params["speech_rate"] = speech_rate
        if emotion is not None and emotion != "neutral":
            audio_params["emotion"] = emotion
        # context_texts/section_id 进 additions：req_params.additions 是
        # 字符串化 JSON（火山 V3 文档结构）。
        additions: dict[str, Any] = {}
        if context_texts is not None:
            additions["context_texts"] = context_texts
        if section_id is not None:
            additions["section_id"] = section_id
        if explicit_language is not None:
            additions["explicit_language"] = explicit_language
        req_params: dict[str, Any] = {
            "text": text,
            "speaker": voice_type,
            "audio_params": audio_params,
        }
        if additions:
            req_params["additions"] = json.dumps(additions, ensure_ascii=False)
        body = {
            "user": {"uid": "duet-dialogue-tts"},
            "req_params": req_params,
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
            # 写盘前标记流已 finish：此后失败（ENOSPC 等）抛出的
            # ProviderFailure 带着 stream-finished 中间态，重试分类判死。
            record["outcome"] = STREAM_FINISHED_OUTCOME
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
            raise EmptyAudioError("Doubao TTS returned no valid PCM audio")
        record["response"]["audio_chunks"] = chunk_count
        record["response"]["pcm_bytes"] = len(pcm)
        return pcm
