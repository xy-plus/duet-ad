from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import signal
import shutil
import ssl
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar
from urllib.parse import urlsplit

import httpx

from app.config import Settings


T = TypeVar("T")
_MISSING = object()
_LOGGER = logging.getLogger(__name__)
# Bounded retry for ambiguous network failures: first attempt + 2 retries,
# backing off 2s then 5s.  Only transport errors and gateway statuses are
# retried; 4xx and other definite answers are final on the first attempt.
RETRY_BACKOFF_S = (2.0, 5.0)
# A TCP connection error proves that this attempt did not reach the provider,
# so it is safe to wait out a short network outage without risking a duplicate
# paid submission.  Do not use this schedule for errors after a connection has
# been established: those may have transmitted the request.
CONNECTION_RETRY_BACKOFF_S = (10.0, 30.0, 60.0, 120.0, 300.0)
RETRYABLE_STATUSES = frozenset({502, 503, 504})
# 429 限流单独处理（默认关闭，仅调用方按预算启用——检测类 provider 经
# rate_limit_budget_s 启用，其余 provider 的 429 语义不变）：限流是"等
# 窗口过去就能成"的暂时性失败，与 RETRYABLE_STATUSES 的秒级短重试不同，
# 用分钟级退避。优先尊重响应 Retry-After 头（秒数或 HTTP-date，钳到
# RATE_LIMIT_RETRY_AFTER_MAX_S；缺失/非法/非正值时退化为本指数序列）；
# 累计等待超过调用方预算即停止重试、响应原样返回，由调用方
# _ensure_success 维持 4xx 确定性致命（原降级语义），不新增门禁。
RATE_LIMIT_BACKOFF_S = (30.0, 60.0, 120.0, 240.0, 480.0)
RATE_LIMIT_RETRY_AFTER_MAX_S = 900.0
_CONNECTION_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
)
_RETRYABLE_TRANSPORT_ERRORS = (
    *_CONNECTION_TRANSPORT_ERRORS,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    # SSL EOF/handshake failures usually surface wrapped as the httpx classes
    # above; keep the raw class so an unwrapped one is still retried.
    ssl.SSLError,
)
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_CODEX_ACCOUNT_COMMAND = ("/home/xy/.local/bin/codex-acct", "acct2")
_CODEX_MODEL = "gpt-5.6-sol"
_CODEX_REASONING_EFFORT = "low"
_CODEX_OUTPUT_MAX_BYTES = 64 * 1024
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ProviderResult(Generic[T]):
    value: T
    record: dict[str, Any]
    evidence_media: list[Path] = field(default_factory=list)


class ProviderFailure(RuntimeError):
    """A provider error that keeps all evidence collected before the failure."""

    def __init__(
        self,
        record: dict[str, Any],
        evidence_media: list[Path],
        cause: Exception,
    ) -> None:
        self.record = record
        self.evidence_media = list(evidence_media)
        self.cause = cause
        super().__init__(str(cause))


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _content_type(headers: Mapping[str, str]) -> str:
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _binary_record(content: bytes, content_type: str) -> dict[str, Any]:
    return {
        "content_type": content_type or "application/octet-stream",
        "length": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _media_type(path: Path, content: bytes) -> str:
    """Prefer the actual image signature over a stale processing suffix."""

    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _request_record(request: httpx.Request, body: Any = _MISSING) -> dict[str, Any]:
    record: dict[str, Any] = {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
    }
    if body is not _MISSING:
        record["body"] = body
    elif request.content:
        content_type = _content_type(request.headers)
        if content_type == "application/json" or content_type.endswith("+json"):
            try:
                record["body"] = json.loads(request.content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                record["body"] = request.content.decode("utf-8", errors="replace")
        elif content_type.startswith("text/"):
            record["body"] = request.content.decode("utf-8", errors="replace")
        else:
            record["body"] = _binary_record(request.content, content_type)
    return record


def response_record(response: httpx.Response, *, request_body: Any = _MISSING) -> dict[str, Any]:
    """Record the actual HTTP exchange without decoding binary data as text."""

    content_type = _content_type(response.headers)
    if content_type == "application/json" or content_type.endswith("+json"):
        try:
            body: Any = response.json()
        except ValueError:
            body = response.content.decode("utf-8", errors="replace")
    elif content_type.startswith("text/") or content_type in {
        "application/xml",
        "application/x-www-form-urlencoded",
    }:
        body = response.content.decode(response.encoding or "utf-8", errors="replace")
    else:
        body = _binary_record(response.content, content_type)
    return {
        "request": _request_record(response.request, request_body),
        "response": {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        },
    }


def _transport_record(request: httpx.Request, request_body: Any = _MISSING) -> dict[str, Any]:
    return {"request": _request_record(request, request_body), "response": None}


def is_retryable_transport_error(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_TRANSPORT_ERRORS)


def is_connection_transport_error(exc: BaseException) -> bool:
    """Whether an HTTP attempt failed before a TCP connection was established."""

    return isinstance(exc, _CONNECTION_TRANSPORT_ERRORS)


def retry_sleep(seconds: float) -> None:
    """Backoff seam between retries; tests replace it to avoid real sleeps."""

    time.sleep(seconds)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Retry-After 头（秒数或 HTTP-date）→ 等待秒数，钳到
    RATE_LIMIT_RETRY_AFTER_MAX_S；缺失/非法/非正值返回 None（调用方退化为
    RATE_LIMIT_BACKOFF_S 序列）。"""
    header = response.headers.get("retry-after")
    if header is None:
        return None
    header = header.strip()
    if header.isdigit():
        seconds = float(int(header))
    else:
        try:
            parsed = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        seconds = (parsed - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        return None
    return min(seconds, RATE_LIMIT_RETRY_AFTER_MAX_S)


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    json_body: Any = _MISSING,
    content: bytes | None = None,
    files: Mapping[str, Any] | None = None,
    request_body: Any = _MISSING,
    rate_limit_budget_s: float | None = None,
) -> tuple[httpx.Response, dict[str, Any]]:
    kwargs: dict[str, Any] = {"headers": headers}
    if json_body is not _MISSING:
        kwargs["json"] = json_body
        if request_body is _MISSING:
            request_body = json_body
    if content is not None:
        kwargs["content"] = content
    if files is not None:
        # multipart 上传（如 sandbase /v1/upload）；记录侧由
        # _request_record 按非 JSON 二进制体落 sha256，不落明文。
        kwargs["files"] = files
    request = client.build_request(method, url, **kwargs)
    attempts: list[dict[str, Any]] = []
    attempt_number = 0
    connection_phase = True
    connection_retries = 0
    ambiguous_retries = 0
    rate_limit_retries = 0
    rate_limit_waited_s = 0.0
    while True:
        attempt_number += 1
        try:
            response = client.send(request)
        except _RETRYABLE_TRANSPORT_ERRORS as exc:
            exchange = _transport_record(request, request_body)
            exchange["attempt"] = attempt_number
            exchange["error"] = {"type": type(exc).__name__, "message": str(exc)}
            attempts.append(exchange)
            if is_connection_transport_error(exc) and connection_phase:
                backoff_s = CONNECTION_RETRY_BACKOFF_S
                retry_index = connection_retries
                connection_retries += 1
            else:
                # Once a request could have crossed the network, every later
                # retry stays on the short, duplicate-safe schedule.
                connection_phase = False
                backoff_s = RETRY_BACKOFF_S
                retry_index = ambiguous_retries
                ambiguous_retries += 1
            if retry_index < len(backoff_s):
                max_attempts = attempt_number + len(backoff_s) - retry_index
                _LOGGER.warning(
                    "provider %s %s attempt %d/%d failed with %s: %s; retrying in %gs",
                    method,
                    url,
                    attempt_number,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                    backoff_s[retry_index],
                )
                retry_sleep(backoff_s[retry_index])
                continue
            failure = _transport_record(request, request_body)
            failure["attempts"] = attempts
            raise ProviderFailure(failure, [], exc) from exc
        except httpx.HTTPError as exc:
            exchange = _transport_record(request, request_body)
            exchange["attempt"] = attempt_number
            exchange["error"] = {"type": type(exc).__name__, "message": str(exc)}
            attempts.append(exchange)
            failure = _transport_record(request, request_body)
            if len(attempts) > 1:
                failure["attempts"] = attempts
            raise ProviderFailure(failure, [], exc) from exc
        exchange = response_record(response, request_body=request_body)
        exchange["attempt"] = attempt_number
        attempts.append(exchange)
        if response.status_code == 429 and rate_limit_budget_s is not None:
            # 429 限流：长退避等窗口（序列/钳制见 RATE_LIMIT_BACKOFF_S 段头
            # 注释）。累计等待超预算即落下响应，由调用方维持 4xx 致命语义。
            wait_s = _retry_after_seconds(response)
            if wait_s is None:
                wait_s = RATE_LIMIT_BACKOFF_S[
                    min(rate_limit_retries, len(RATE_LIMIT_BACKOFF_S) - 1)
                ]
            if rate_limit_waited_s + wait_s <= rate_limit_budget_s:
                rate_limit_retries += 1
                rate_limit_waited_s += wait_s
                _LOGGER.warning(
                    "provider %s %s attempt %d returned HTTP 429; "
                    "retrying in %gs (rate-limit wait %gs/%gs)",
                    method,
                    url,
                    attempt_number,
                    wait_s,
                    rate_limit_waited_s,
                    rate_limit_budget_s,
                )
                retry_sleep(wait_s)
                continue
        if response.status_code in RETRYABLE_STATUSES:
            # A gateway response is an ambiguous, post-transport outcome;
            # retain its historical two short retries even if preceding
            # connection failures were harmless.
            connection_phase = False
            retry_index = ambiguous_retries
            ambiguous_retries += 1
            if retry_index < len(RETRY_BACKOFF_S):
                max_attempts = attempt_number + len(RETRY_BACKOFF_S) - retry_index
                _LOGGER.warning(
                    "provider %s %s attempt %d/%d returned HTTP %d; retrying in %gs",
                    method,
                    url,
                    attempt_number,
                    max_attempts,
                    response.status_code,
                    RETRY_BACKOFF_S[retry_index],
                )
                retry_sleep(RETRY_BACKOFF_S[retry_index])
                continue
        if len(attempts) > 1:
            return response, {**exchange, "attempts": attempts}
        return response, exchange


def _ensure_success(
    response: httpx.Response,
    record: dict[str, Any],
    evidence_media: list[Path] | None = None,
) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ProviderFailure(record, evidence_media or [], exc) from exc


def _json_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("provider response is not valid JSON") from exc


def _deepseek_base(base: str) -> str:
    canonical = base.rstrip("/")
    if canonical.endswith("/beta"):
        canonical = canonical[:-5]
    return canonical


def _deepseek_responses_url(base: str) -> str:
    return _deepseek_base(base) + "/responses"


def _structured_output_text(body: Any) -> str:
    if not isinstance(body, dict):
        raise TypeError("DeepSeek response must be an object")
    status = body.get("status")
    if status != "completed":
        raise ValueError(f"DeepSeek structured response status is {status!r}")
    output = body.get("output")
    if not isinstance(output, list):
        raise TypeError("DeepSeek structured response output must be a list")
    messages = [
        item for item in output
        if isinstance(item, dict) and item.get("type") == "message"
    ]
    if not messages:
        raise ValueError("DeepSeek structured response must contain a message")
    # 取最后一个 message 项而非假设恰好一个：若服务端工具（web_search
    # 等）未来生效，output 可能插入非 message 项或产生多个 message
    # （spec 2026-09-28 §4 审查建议 5；2026-09-28 冒烟未观察到真实多项
    # 样本，此为防御性加固，非供应商实证行为）。
    message = messages[-1]
    if message.get("status") != "completed" or message.get("role") != "assistant":
        raise ValueError("DeepSeek structured response message is incomplete")
    content = message.get("content")
    if not isinstance(content, list):
        raise TypeError("DeepSeek structured response content must be a list")
    text_parts = [
        item for item in content
        if isinstance(item, dict) and item.get("type") == "output_text"
    ]
    if len(text_parts) != 1 or not isinstance(text_parts[0].get("text"), str):
        raise ValueError("DeepSeek structured response must contain one output_text")
    return text_parts[0]["text"]


def _parse_structured_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Models occasionally emit raw control characters inside string
        # values even under a strict transport schema; accept them before
        # asking the model to repair its output.
        parsed = json.loads(text, strict=False)
    if not isinstance(parsed, dict):
        raise TypeError("DeepSeek structured output must decode to an object")
    return parsed


# A malformed or schema-invalid structured output is repaired in the same
# conversation: the bad assistant output and a correction instruction are
# appended, and the model re-answers.  After JSON_REPAIR_MAX_ATTEMPTS repairs
# the original error is raised so the caller's node-level retry still applies.
JSON_REPAIR_MAX_ATTEMPTS = 3


def _json_repair_instruction(error: Exception) -> str:
    return (
        f"Your previous output failed JSON validation: {error}. "
        "Output corrected JSON only, no markdown, no commentary."
    )


# web_search 降级结论（2026-09-28 实弹冒烟，生产同款凭证与模型
# deepseek-v4-flash-vision-exp）：不接服务端搜索，请求不携带 tools。
# 冒烟三项证据：① DeepSeek 官方 Responses API 指南的 Tools 兼容表把
# web_search/file_search 等内置工具列为"忽略"
#（https://api-docs.deepseek.com/zh-cn/guides/responses_api）[OFFICIAL]；
# 实测带 tools=[{"type":"web_search"}] 的请求 HTTP 200、status=completed，
# 但 output 无 web_search_call 项（实测 output_item_types=['reasoning',
# 'message']），模型只在 message 里输出 DSML 纯文本"尝试调用"，服务端
# 未执行任何搜索 [EMPIRICAL]。② tools 与 json_schema + reasoning.effort
# 共存：HTTP 层不被拒（200），但模型一旦尝试调用被忽略的工具，message
# 输出是 DSML 调用标记文本而非 schema JSON；去掉 tools 的对照组输出合法
# schema JSON [EMPIRICAL]。③ 服务端不产生非 message 项，多项解析加固无
# 真实样本可验证（见 _structured_output_text 注释）。若未来要重试接入，
# 必须重新实弹冒烟验证这三项，不得仅凭文档变化推断支持。
class DeepSeekClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport

    @staticmethod
    def _responses_content(payload: Any, images: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": json.dumps(payload, ensure_ascii=False),
            }
        ]
        for image in images:
            image_bytes = image.read_bytes()
            mime = _media_type(image, image_bytes)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{encoded}",
                }
            )
        return content

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        *,
        validator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        """Structured call with an in-conversation JSON repair loop.

        When the model's output text fails to parse or the ``validator``
        rejects it, the bad output and a correction instruction are appended
        to the same conversation and the request is resent, up to
        ``JSON_REPAIR_MAX_ATTEMPTS`` times.  Every repair exchange is kept in
        the provider record under ``json_repairs``.  A persistent failure
        raises the original error so node-level retry still applies.
        """
        media = list(images or [])
        url = _deepseek_responses_url(self.settings.deepseek_api_base)
        try:
            content = self._responses_content(payload, media)
        except Exception as exc:
            record = {
                "request": {
                    "method": "POST",
                    "url": url,
                    "headers": _headers(self.settings.deepseek_api_key),
                    "body": {"instruction": instruction, "payload": payload},
                },
                "response": None,
            }
            raise ProviderFailure(record, media, exc) from exc

        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": content}
        ]
        record: dict[str, Any] | None = None
        repairs: list[dict[str, Any]] = []
        try:
            with httpx.Client(
                timeout=self.settings.request_timeout_s,
                transport=self.transport,
            ) as client:
                for repair_round in range(JSON_REPAIR_MAX_ATTEMPTS + 1):
                    request: dict[str, Any] = {
                        "model": self.settings.deepseek_model,
                        "instructions": instruction,
                        "input": conversation,
                        "reasoning": {"effort": "high"},
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "duet_structured_result",
                                "schema": schema,
                            }
                        },
                    }
                    response, record = _send(
                        client,
                        "POST",
                        url,
                        headers=_headers(self.settings.deepseek_api_key),
                        json_body=request,
                    )
                    _ensure_success(response, record, media)
                    text = _structured_output_text(_json_body(response))
                    try:
                        value = _parse_structured_json(text)
                        if validator is not None:
                            validator(value)
                    except Exception as exc:
                        if repair_round >= JSON_REPAIR_MAX_ATTEMPTS:
                            raise
                        repairs.append({
                            "repair": repair_round + 1,
                            "exchange": record,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        })
                        conversation = [
                            *conversation,
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": text}
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": _json_repair_instruction(exc),
                                    }
                                ],
                            },
                        ]
                        continue
                    if repairs:
                        record = {**record, "json_repairs": repairs}
                    return ProviderResult(
                        value=value, record=record, evidence_media=media,
                    )
                raise AssertionError("unreachable")
        except ProviderFailure as exc:
            if repairs and "json_repairs" not in exc.record:
                exc.record = {**exc.record, "json_repairs": repairs}
            if exc.evidence_media:
                raise
            raise ProviderFailure(exc.record, media, exc.cause) from exc.cause
        except Exception as exc:
            fallback = record or {
                "request": {
                    "method": "POST",
                    "url": url,
                    "headers": _headers(self.settings.deepseek_api_key),
                    "body": conversation,
                },
                "response": None,
            }
            if repairs and "json_repairs" not in fallback:
                fallback = {**fallback, "json_repairs": repairs}
            raise ProviderFailure(fallback, media, exc) from exc


# 方舟音频理解的系统指令（POC scripts/audio_poc/common.py 定稿口径）：
# 模型按 input_text 里的提示词输出自由 JSON 文本，由调用方解析。
ARK_AUDIO_INSTRUCTIONS = (
    "你是音频理解专家，擅长分析广告配音音频的语速、语调、情绪、停顿与整体语气。"
)


def _ark_output_text(body: Any) -> str:
    if not isinstance(body, dict):
        raise TypeError("ark audio response must be an object")
    status = body.get("status")
    if status not in (None, "completed"):
        raise ValueError(f"ark audio response status is {status!r}")
    output = body.get("output")
    if not isinstance(output, list):
        raise TypeError("ark audio response output must be a list")
    texts = [
        item["text"]
        for out in output
        if isinstance(out, dict) and out.get("type") == "message"
        for item in out.get("content") or []
        if isinstance(item, dict)
        and item.get("type") == "output_text"
        and isinstance(item.get("text"), str)
    ]
    if not texts:
        raise ValueError("ark audio response has no output_text")
    return texts[0]


class ArkAudioClient:
    """方舟音频理解客户端（ark-audio-analysis 节点）。

    POST {ark_audio_api_base}/responses，input content 为 input_audio
    （data:audio/wav;base64,...）+ input_text 提示词；返回模型的自由 JSON
    文本（不用 json_schema 约束，POC 口径），解析归调用方。三项配置缺任一
    即构造时报错（部署错误在节点首次用到时暴露，不静默降级）。
    """

    def __init__(
        self, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> None:
        missing = [
            name
            for name, value in (
                ("ARK_AUDIO_API_KEY", settings.ark_audio_api_key),
                ("ARK_AUDIO_API_BASE", settings.ark_audio_api_base),
                ("ARK_AUDIO_MODEL", settings.ark_audio_model),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set for ark audio analysis"
            )
        self.settings = settings
        self.transport = transport

    def analyze(self, audio_path: Path, prompt: str) -> ProviderResult[str]:
        """Send one audio clip for understanding; returns the raw output text."""
        media = [audio_path]
        url = f"{self.settings.ark_audio_api_base}/responses"
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        request: dict[str, Any] = {
            "model": self.settings.ark_audio_model,
            "instructions": ARK_AUDIO_INSTRUCTIONS,
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
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(
                timeout=self.settings.request_timeout_s,
                transport=self.transport,
            ) as client:
                response, record = _send(
                    client,
                    "POST",
                    url,
                    headers=_headers(self.settings.ark_audio_api_key),
                    json_body=request,
                )
                _ensure_success(response, record, media)
                text = _ark_output_text(_json_body(response))
                return ProviderResult(
                    value=text, record=record, evidence_media=media,
                )
        except ProviderFailure as exc:
            if exc.evidence_media:
                raise
            raise ProviderFailure(exc.record, media, exc.cause) from exc.cause
        except Exception as exc:
            fallback = record or {
                "request": {
                    "method": "POST",
                    "url": url,
                    "headers": _headers(self.settings.ark_audio_api_key),
                    "body": request,
                },
                "response": None,
            }
            raise ProviderFailure(fallback, media, exc) from exc


class CodexProcessError(RuntimeError):
    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(message)


def _scrubbed_process_env() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not _SECRET_ENV_RE.search(key)
    }
    environment["PATH"] = "/home/xy/.local/bin:/usr/local/bin:/usr/bin:/bin"
    return environment


def _clean_process_text(value: str, limit: int = 8192) -> str:
    lines = [
        line for line in value.splitlines()
        if _ENV_LINE_RE.match(line.strip()) is None
    ]
    return "\n".join(lines).strip()[-limit:]


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _read_regular_file(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 1 or info.st_size > limit:
            raise ValueError("Codex structured output has an invalid size")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        if len(value) > limit:
            raise ValueError("Codex structured output exceeds its limit")
        return value
    finally:
        os.close(descriptor)


def _copy_regular_file(source: Path, destination: Path, limit: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Codex image input is not a bounded regular file")
        output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(output, remaining)
                    remaining = remaining[written:]
            os.fsync(output)
        finally:
            os.close(output)
    finally:
        os.close(descriptor)


def _image_suffix(source: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        header = os.read(descriptor, 16)
    finally:
        os.close(descriptor)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("Codex image input format is unsupported")


def _parse_codex_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Codex structured output is not UTF-8") from None

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant is forbidden: {value}")

    decoder = json.JSONDecoder(parse_constant=reject_constant)
    start = len(text) - len(text.lstrip())
    fenced = False
    if text.startswith("```", start):
        header = re.match(r"```(?:json)?[ \t]*\r?\n", text[start:])
        if header is None:
            raise ValueError("Codex structured output has an invalid JSON fence")
        fenced = True
        start += header.end()
        start += len(text[start:]) - len(text[start:].lstrip())
    try:
        value, end = decoder.raw_decode(text, start)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("Codex structured output is not one JSON value") from None
    suffix_start = end
    if fenced:
        closing = end + len(text[end:]) - len(text[end:].lstrip())
        if not text.startswith("```", closing):
            raise ValueError("Codex structured output has an incomplete JSON fence")
        suffix_start = closing + 3
    # Any trailing text (including the "tokens used" telemetry footer) is
    # ignored: the JSON value above is the complete structured output.
    if not isinstance(value, dict):
        raise TypeError("Codex structured output must be an object")
    return value


class CodexClient:
    """Run one schema-constrained fusion call through ``codex-acct acct2``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gate = threading.BoundedSemaphore(settings.codex_concurrency)

    @staticmethod
    def _command(stage: Path, output: Path, images: list[Path]) -> list[str]:
        inner = [
            *_CODEX_ACCOUNT_COMMAND,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "-C", str(stage),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "-o", str(output),
            "-m", _CODEX_MODEL,
            "--output-schema", str(stage / ".codex-output-schema.json"),
            "-c", f'model_reasoning_effort="{_CODEX_REASONING_EFFORT}"',
            "-c", "sandbox_workspace_write.network_access=false",
            "-c", 'shell_environment_policy.inherit="core"',
            "-c", 'shell_environment_policy.exclude=["*KEY*","*TOKEN*","*SECRET*","*PASSWORD*"]',
        ]
        for image in images:
            inner.extend(("--image", str(image)))
        inner.append("--")
        inner.append(
            "严格执行当前目录 SKILL.md；只读取 input.json 和随消息附带的图片；"
            "按注入的 JSON Schema 输出唯一完整结果，不执行其他任务。"
        )
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise FileNotFoundError("bwrap executable not found")
        bwrap_path = Path(bwrap).resolve(strict=True)
        checkout = _SOURCE_ROOT.resolve(strict=True)
        command = [
            str(bwrap_path),
            "--bind", "/", "/",
            "--dev-bind", "/dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", str(checkout),
            "--tmpfs", "/tmp",
            "--bind", str(stage), str(stage),
        ]
        for candidate in sorted(stage.iterdir(), key=lambda path: path.name):
            if candidate != output:
                command.extend(("--ro-bind", str(candidate), str(candidate)))
        command.extend(("--chdir", str(stage), *inner))
        return command

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
        *,
        validator: Callable[[dict[str, Any]], Any] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        # Codex runs as a one-shot subprocess, so there is no conversation to
        # repair in: the validator rejects the output here and the caller's
        # node-level retry reruns the whole process.
        media = list(images or [])
        request_record: dict[str, Any] = {
            "provider": "codex",
            "account": "acct2",
            "model": _CODEX_MODEL,
            "reasoning_effort": _CODEX_REASONING_EFFORT,
            "input": payload,
            "schema": schema,
        }
        response_record: dict[str, Any] | None = None
        try:
            if not isinstance(payload, dict) or not isinstance(schema, dict):
                raise TypeError("Codex structured input and schema must be objects")
            with tempfile.TemporaryDirectory(
                prefix="duet-h3-fusion-", dir="/tmp",
            ) as raw_stage:
                stage = Path(raw_stage).resolve(strict=True)
                skill_path = stage / "SKILL.md"
                input_path = stage / "input.json"
                schema_path = stage / ".codex-output-schema.json"
                output_path = stage / ".codex-final-output.json"
                skill_path.write_text(instruction, encoding="utf-8")
                input_path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                schema_path.write_text(
                    json.dumps(
                        schema,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                for path in (skill_path, input_path, schema_path):
                    os.chmod(path, 0o600)
                staged_images: list[Path] = []
                for index, source in enumerate(media, 1):
                    suffix = _image_suffix(source)
                    target = stage / f"image-{index}{suffix}"
                    _copy_regular_file(source, target, 64 * 1024 * 1024)
                    staged_images.append(target)
                descriptor = os.open(
                    output_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.close(descriptor)
                command = self._command(stage, output_path, staged_images)
                request_record["command"] = [
                    value if not value.startswith(str(stage)) else value.replace(str(stage), "<stage>", 1)
                    for value in command[command.index(str(_CODEX_ACCOUNT_COMMAND[0])):]
                ]
                with self._gate, tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(mode="w+b") as stderr_file:
                    process = subprocess.Popen(
                        command,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        env=_scrubbed_process_env(),
                        start_new_session=True,
                    )
                    try:
                        returncode = process.wait(timeout=self.settings.codex_timeout_s)
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(process)
                        stderr_file.seek(0)
                        stderr = _clean_process_text(
                            stderr_file.read().decode("utf-8", errors="replace")
                        )
                        raise CodexProcessError(
                            f"codex timed out after {self.settings.codex_timeout_s:g}s",
                            stderr=stderr,
                        ) from None
                    _terminate_process_group(process)
                    stdout_file.seek(0)
                    stderr_file.seek(0)
                    stdout = _clean_process_text(
                        stdout_file.read().decode("utf-8", errors="replace")
                    )
                    stderr = _clean_process_text(
                        stderr_file.read().decode("utf-8", errors="replace")
                    )
                response_record = {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                if returncode != 0:
                    raise CodexProcessError(
                        f"codex exit {returncode}: {stderr[-500:]}",
                        stdout=stdout,
                        stderr=stderr,
                    )
                value = _parse_codex_object(
                    _read_regular_file(output_path, _CODEX_OUTPUT_MAX_BYTES)
                )
                response_record["body"] = value
                if validator is not None:
                    validator(value)
                return ProviderResult(
                    value=value,
                    record={"request": request_record, "response": response_record},
                    evidence_media=media,
                )
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(
                {"request": request_record, "response": response_record},
                media,
                exc,
            ) from exc


_MEDIAKIT_TASK_PENDING_STATUSES = frozenset(
    {"PENDING", "QUEUED", "RUNNING", "PROCESSING", "CREATED"}
)


def _mediakit_task_result(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ValueError("MediaKit task query was not successful")
    result = body.get("result")
    if not isinstance(result, dict):
        raise TypeError("MediaKit task query result must be an object")
    return result


class MediaKitClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport

    def _mediakit_headers(self) -> dict[str, str]:
        token = self.settings.mediakit_api_key
        if not token:
            raise ValueError("VOLC_MEDIAKIT_API_KEY must be set to a non-empty value")
        return _headers(token)

    def submit_video_subtitle_erase(self, video_url: str) -> ProviderResult[str]:
        if not isinstance(video_url, str) or not video_url.strip():
            raise ValueError("MediaKit video URL is invalid")
        video_url = video_url.strip()
        mode = self.settings.mediakit_video_erase_mode
        endpoint = (
            "erase-video-subtitle-pro"
            if mode == "pro"
            else "erase-video-subtitle"
        )
        body: dict[str, Any] = {"video_url": video_url, "mode": "Subtitle"}
        if mode == "pro":
            # The pro tier pins v5: faster and optimized against subtitle
            # flicker on AIGC video after erasure.
            body["model_version"] = "v5"
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                response, record = _send(
                    client,
                    "POST",
                    f"{self.settings.mediakit_api_base}/tools/{endpoint}",
                    headers=self._mediakit_headers(),
                    json_body=body,
                )
            _ensure_success(response, record)
            # Submit returns a flat {"success": true, "task_id": ...}; only
            # the poll response wraps its payload in "result".
            submit_body = _json_body(response)
            if not isinstance(submit_body, dict) or submit_body.get("success") is not True:
                raise ValueError("MediaKit task submit was not successful")
            task_id = submit_body.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("MediaKit task submit response has no task_id")
            record["outcome"] = "accepted"
            record["task_id"] = task_id
            return ProviderResult(task_id, record)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record or {}, [], exc) from exc

    def poll_task(self, task_id: str) -> ProviderResult[str]:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("MediaKit task id is invalid")
        task_id = task_id.strip()
        record: dict[str, Any] = {"task_id": task_id, "polls": []}
        poll_url = f"{self.settings.mediakit_api_base}/tasks/{task_id}"
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                while True:
                    try:
                        response, exchange = _send(
                            client,
                            "GET",
                            poll_url,
                            headers=self._mediakit_headers(),
                        )
                    except ProviderFailure as exc:
                        record["polls"].append(exc.record)
                        raise ProviderFailure(record, [], exc.cause) from exc.cause
                    record["polls"].append(exchange)
                    _ensure_success(response, record)
                    # Task query is flat: {"success": true, "status": ...} with
                    # "result": {"video_url": ...} only present once completed.
                    task_body = _json_body(response)
                    if (
                        not isinstance(task_body, dict)
                        or task_body.get("success") is not True
                    ):
                        raise ValueError("MediaKit task query was not successful")
                    status = str(task_body.get("status") or "").upper()
                    if status == "COMPLETED":
                        result = _mediakit_task_result(task_body)
                        video_url = result.get("video_url")
                        if not isinstance(video_url, str) or not video_url:
                            raise ValueError(
                                "MediaKit completed task has no video_url"
                            )
                        return ProviderResult(video_url, record)
                    if status not in _MEDIAKIT_TASK_PENDING_STATUSES:
                        raise RuntimeError(
                            "MediaKit task failed: "
                            + json.dumps(task_body, ensure_ascii=False)
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"MediaKit task {task_id} timed out")
                    time.sleep(self.settings.provider_poll_interval_s)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    def download(self, url: str, destination: Path) -> ProviderResult[Path]:
        record: dict[str, Any] | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(
                timeout=self.settings.request_timeout_s,
                transport=self.transport,
                follow_redirects=True,
            ) as client:
                response, record = _send(client, "GET", url)
            if response.history:
                record["redirects"] = [response_record(item) for item in response.history]
            _ensure_success(response, record)
            destination.write_bytes(response.content)
            return ProviderResult(destination, record, [destination])
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record or {}, [], exc) from exc


class PicsartClient:
    WORKFLOW = "minimax/h3-max/reference-to-video"
    # These client errors establish that the request was not accepted. Timeout,
    # conflict/early-data and rate-limit responses remain submission-unknown.
    SAFE_REJECT_STATUS_CODES = frozenset(
        code for code in range(400, 500)
        if code not in {408, 409, 425, 429}
    )

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport
        # 402 failover 进程内粘性：主 token 一旦被 402（预付 credits 用尽或
        # 订阅上限未开透支，官方 error-codes 文档）判负，本进程后续 submit
        # 直接用 backup；重启后重新先试 primary（账户可能已充值）。
        self._primary_exhausted = False

    def _api_key_for_role(self, role: str) -> str:
        """任务账户隔离：task 由哪把 key 创建，poll/result/download 必须用
        同一把（回执持久化的 api_key_role 为唯一来源）。"""
        if role == "primary":
            return self.settings.picsart_api_key
        if role == "backup":
            backup = self.settings.picsart_api_key_backup
            if not backup:
                raise ValueError(
                    "PICSART_API_KEY_BACKUP 未配置，无法按回执 role=backup 调用"
                )
            return backup
        raise ValueError(f"unknown Picsart api key role: {role!r}")

    def _http_client_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Keep Picsart traffic isolated from ambient proxy environment."""
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
            "trust_env": False,
            **kwargs,
        }
        if self.settings.picsart_proxy is not None:
            client_kwargs["proxy"] = self.settings.picsart_proxy
        return client_kwargs

    def submit(
        self,
        *,
        prompt: str,
        video_url: str,
        image_urls: list[str] | None,
        audio_urls: list[str] | None = None,
        duration: int,
        resolution: str,
        aspect_ratio: str,
    ) -> ProviderResult[str]:
        if image_urls is not None:
            if not isinstance(image_urls, list):
                raise ValueError("Picsart image_urls must be a list or None")
            if len(image_urls) > 9:
                raise ValueError("Picsart image_urls supports at most 9 URLs")
            if any(
                not isinstance(url, str) or not url.strip()
                for url in image_urls
            ):
                raise ValueError(
                    "Picsart image_urls must contain non-empty strings"
                )
            image_urls = [url.strip() for url in image_urls]

        if audio_urls is not None:
            if not isinstance(audio_urls, list):
                raise ValueError("Picsart audio_urls must be a list or None")
            if len(audio_urls) > 3:
                raise ValueError("Picsart audio_urls supports at most 3 URLs")
            if any(
                not isinstance(url, str) or not url.strip()
                for url in audio_urls
            ):
                raise ValueError(
                    "Picsart audio_urls must contain non-empty strings"
                )
            audio_urls = [url.strip() for url in audio_urls]

        params: dict[str, Any] = {
            "prompt": prompt,
            "reference_video_urls": [video_url],
            "resolution": resolution.upper(),
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "prompt_expansion_mode": "quality",
        }
        if image_urls:
            params["reference_image_urls"] = image_urls
        if audio_urls:
            params["reference_audio_urls"] = audio_urls
        body = {"params": params}
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(**self._http_client_kwargs()) as client:
                submit_url = (
                    f"{self.settings.picsart_api_base}"
                    f"/workflows/{self.WORKFLOW}/submit"
                )
                use_backup_first = bool(
                    self._primary_exhausted and self.settings.picsart_api_key_backup
                )
                first_role = "backup" if use_backup_first else "primary"
                try:
                    response, record = _send(
                        client,
                        "POST",
                        submit_url,
                        headers=_headers(self._api_key_for_role(first_role)),
                        json_body=body,
                    )
                except ProviderFailure as exc:
                    exc.record["outcome"] = "unknown"
                    exc.record["api_key_role"] = first_role
                    raise
                record["api_key_role"] = first_role
                if (
                    not use_backup_first
                    and response.status_code == 402
                    and self.settings.picsart_api_key_backup
                ):
                    # 402 = 预付 credits 用尽或订阅上限未开透支（官方
                    # error-codes / quotas-and-limits 文档），且在
                    # SAFE_REJECT_STATUS_CODES 内 = 同步拒绝、未建任务、
                    # 未扣费：同一次 submit 调用内用 backup 重发一次，无
                    # 双扣风险；flag 置位后本进程后续直接走 backup（重启
                    # 后重新先试 primary，账户可能已充值）。401=key 无效
                    # 属配置错误，不 failover；非 402 语义一律不变。
                    failover = {"from": "primary", "status": 402}
                    self._primary_exhausted = True
                    try:
                        response, record = _send(
                            client,
                            "POST",
                            submit_url,
                            headers=_headers(self._api_key_for_role("backup")),
                            json_body=body,
                        )
                    except ProviderFailure as exc:
                        exc.record["outcome"] = "unknown"
                        exc.record["api_key_role"] = "backup"
                        exc.record["key_failover"] = failover
                        raise
                    record["api_key_role"] = "backup"
                    record["key_failover"] = failover
            if not response.is_success:
                status_code = response.status_code
                record["outcome"] = (
                    "rejected"
                    if status_code in self.SAFE_REJECT_STATUS_CODES
                    else "unknown"
                )
                _ensure_success(response, record)
            payload = _json_body(response)
            envelope = payload.get("response") if isinstance(payload, dict) else None
            task_id = envelope.get("id") if isinstance(envelope, dict) else None
            if not isinstance(task_id, str) or not task_id:
                record["outcome"] = "unknown"
                raise ValueError("Picsart submit response has no task id")
            record["outcome"] = "accepted"
            record["task_id"] = task_id
            return ProviderResult(task_id, record)
        except ProviderFailure:
            raise
        except Exception as exc:
            failure_record = record or {}
            failure_record.setdefault("outcome", "unknown")
            raise ProviderFailure(failure_record, [], exc) from exc

    def poll(self, task_id: str, *, api_key_role: str = "primary") -> ProviderResult[str]:
        record: dict[str, Any] = {"task_id": task_id, "polls": []}
        record["api_key_role"] = api_key_role
        poll_headers = _headers(self._api_key_for_role(api_key_role))
        poll_url = (
            f"{self.settings.picsart_api_base}/workflows/{self.WORKFLOW}/{task_id}/result"
        )
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        try:
            with httpx.Client(**self._http_client_kwargs()) as client:
                while True:
                    try:
                        response, exchange = _send(
                            client,
                            "GET",
                            poll_url,
                            headers=poll_headers,
                        )
                    except ProviderFailure as exc:
                        record["polls"].append(exc.record)
                        raise ProviderFailure(record, [], exc.cause) from exc.cause
                    record["polls"].append(exchange)
                    _ensure_success(response, record)
                    payload = _json_body(response)
                    envelope = payload.get("response") if isinstance(payload, dict) else None
                    if not isinstance(envelope, dict):
                        raise TypeError("Picsart poll response envelope must be an object")
                    status = str(envelope.get("status") or "").upper()
                    if status == "COMPLETED":
                        result = envelope.get("result")
                        if not isinstance(result, dict):
                            raise TypeError("Picsart completed result must be an object")
                        video = result.get("video")
                        if not isinstance(video, dict):
                            raise TypeError("Picsart completed result must contain a video object")
                        url = video.get("url")
                        if not isinstance(url, str) or not url:
                            raise ValueError("Picsart completed video has no URL")
                        return ProviderResult(url, record)
                    if status == "FAILED":
                        raise RuntimeError("Picsart task failed: " + json.dumps(envelope, ensure_ascii=False))
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Picsart task {task_id} timed out")
                    time.sleep(self.settings.provider_poll_interval_s)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    def download(self, url: str, destination: Path, *, api_key_role: str = "primary") -> ProviderResult[Path]:
        record: dict[str, Any] | None = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(
                **self._http_client_kwargs(follow_redirects=True)
            ) as client:
                response, record = _send(
                    client, "GET", url,
                    headers={"Authorization": f"Bearer {self._api_key_for_role(api_key_role)}"},
                )
            if response.history:
                record["redirects"] = [response_record(item) for item in response.history]
            record["api_key_role"] = api_key_role
            _ensure_success(response, record)
            destination.write_bytes(response.content)
            return ProviderResult(destination, record, [destination])
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record or {}, [], exc) from exc



class ContextIRTaskFailedError(RuntimeError):
    """Context IR 任务到达 vendor 终态 failed（失败不计费，允许换任务重提）。"""


class ContextIRModerationError(ValueError):
    """MiniMax 内容审核拒绝（供应商错误码 1026）：提示词或素材未过审。"""


class MiniMaxContextIRClient:
    """MiniMax 官方 H3 Context IR 提示词优化（h3-context-ir 节点专用）。

    契约以 duet-video-v2 生产实测文档（/tmp/dvet2/docs/agent/reference/
    minimax-context-ir.md，2026-08 实测）与其工作代码 contextIrClient.service.ts
    为准，与 MiniMax-AI/MiniMax-H3 仓库 case 脚本一致：

    - 素材必须走 `POST <base>/v1/files/upload`（multipart，
      purpose=video_generation_input，同 Bearer 认证）拿 file_id，content 里用
      `mm_file://{file_id}` 引用。公网 URL 两条路实测均被禁：MiniMax 阻断数据中
      心 IP（media url unreachable, blocking datacenter IPs），HMAC 签名链接
      被 400。file_id 不跨运行缓存（MiniMax 文件只保留 7 天）。
    - 图片上传前本地校验宽高都在 [256, 5760]。
    - 查询 `GET <base>/v2/query/video_generation?task_id=<id>` 返回账户最近任务
      **列表**且不按 task_id 过滤（2026-08-18 实测），客户端按 items[].id 自取；
      目标不在列表 = 仍在排队；绝不取 items[0]。可接受状态仅
      queued/running/succeeded/failed（error 归 failed），其他报错。
    - 计费：input $0.87/M、output $3.45/M token（实测单次 ~¥0.085），失败不计费。
    """

    MODEL = "MiniMax-H3"
    SUBMIT_PATH = "/v2/h3_context_ir"
    UPLOAD_PATH = "/v1/files/upload"
    QUERY_PATH = "/v2/query/video_generation"
    FILE_URL_PREFIX = "mm_file://"
    UPLOAD_PURPOSE = "video_generation_input"
    # 官方文档列出的 ratio 全集；pipeline 只传与 Picsart 提交一致的 16:9/9:16。
    RATIOS = frozenset({"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
    DURATION_MIN_S = 4
    DURATION_MAX_S = 15
    IMAGE_EDGE_MIN = 256
    IMAGE_EDGE_MAX = 5760
    MODERATION_CODE = "1026"
    # 实测契约：单次 HTTP 60s；轮询每 2s；总窗口 360s（长尾 >120s）。
    REQUEST_TIMEOUT_S = 60.0
    POLL_INTERVAL_S = 2.0
    POLL_TIMEOUT_S = 360.0
    # 与 PicsartClient 同一语义：这些 4xx 证明请求未被接受（不扣费），
    # 重发原样报文必然同样失败；408/409/425/429 保持 submission-unknown。
    SAFE_REJECT_STATUS_CODES = PicsartClient.SAFE_REJECT_STATUS_CODES
    _PENDING_STATUSES = frozenset({"queued", "running"})

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        # 实例级覆盖位：测试把 poll 间隔/窗口调小，生产保持实测契约值。
        self.request_timeout_s = self.REQUEST_TIMEOUT_S
        self.poll_interval_s = self.POLL_INTERVAL_S
        self.poll_timeout_s = self.POLL_TIMEOUT_S

    def _http_client_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """国内直连 api.minimaxi.com，与 MediaKit 一样隔离环境代理。"""
        return {
            "timeout": self.request_timeout_s,
            "transport": self.transport,
            "trust_env": False,
            **kwargs,
        }

    def _api_key(self) -> str:
        key = self.settings.minimax_context_ir_api_key
        if not key:
            raise ValueError(
                "MINIMAX_CONTEXT_IR_API_KEY 未配置，h3-context-ir 节点不可用"
            )
        return key

    @staticmethod
    def _error_code(payload: Any) -> str | None:
        """供应商错误码可能出现在 error.code / code / base_resp.status_code。"""
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        base_resp = payload.get("base_resp")
        for candidate in (
            error.get("code") if isinstance(error, dict) else None,
            payload.get("code"),
            payload.get("error_code"),
            base_resp.get("status_code") if isinstance(base_resp, dict) else None,
        ):
            if isinstance(candidate, (str, int)) and str(candidate):
                return str(candidate)
        return None

    def _check_moderation(self, payload: Any, record: dict[str, Any]) -> None:
        if self._error_code(payload) == self.MODERATION_CODE:
            record["moderation"] = True
            raise ContextIRModerationError(
                "提示词或素材未通过 MiniMax 内容审核（供应商错误码 1026）"
            )

    def upload(self, path: Path, *, kind: str) -> ProviderResult[str]:
        """上传单个素材，返回 mm_file:// 引用 URL。kind ∈ image|video|audio。

        图片本地校验宽高 [256, 5760]（供应商会在误用 purpose 时读到 0x0，
        尺寸越界在上传前本地判掉）。上传失败只是本次增强失败，不扣 generation
        费用；确定性 4xx 记 rejected。
        """
        if kind not in {"image", "video", "audio"}:
            raise ValueError("Context IR upload kind must be image, video or audio")
        if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
            raise ValueError("Context IR upload file is invalid")
        content = path.read_bytes()
        if not content:
            raise ValueError("Context IR upload file is empty")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if kind == "image":
            from PIL import Image
            import io

            try:
                with Image.open(io.BytesIO(content)) as image:
                    width, height = image.size
            except Exception as exc:
                raise ValueError(
                    "Context IR 图片尺寸无法识别，请重新上传有效图片"
                ) from exc
            if not (
                self.IMAGE_EDGE_MIN <= width <= self.IMAGE_EDGE_MAX
                and self.IMAGE_EDGE_MIN <= height <= self.IMAGE_EDGE_MAX
            ):
                raise ValueError(
                    "Context IR 图片尺寸必须在 256-5760 像素之间"
                )
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(**self._http_client_kwargs()) as client:
                try:
                    response, record = _send(
                        client,
                        "POST",
                        f"{self.settings.minimax_context_ir_api_base}{self.UPLOAD_PATH}",
                        headers={
                            "Authorization": f"Bearer {self._api_key()}",
                        },
                        files={
                            "purpose": (None, self.UPLOAD_PURPOSE),
                            "file": (path.name, content, media_type),
                        },
                        request_body={
                            "purpose": self.UPLOAD_PURPOSE,
                            "file": f"<{len(content)} bytes {media_type}>",
                        },
                    )
                except ProviderFailure as exc:
                    exc.record["outcome"] = "unknown"
                    raise
            record["outcome"] = (
                None if response.is_success
                else (
                    "rejected"
                    if response.status_code in self.SAFE_REJECT_STATUS_CODES
                    else "unknown"
                )
            )
            payload = _json_body(response)
            self._check_moderation(payload, record)
            _ensure_success(response, record)
            base_resp = payload.get("base_resp") if isinstance(payload, dict) else None
            if isinstance(base_resp, dict) and base_resp.get("status_code") not in (
                0, None,
            ):
                raise ProviderFailure(
                    record, [],
                    RuntimeError(
                        "Context IR 素材上传失败: "
                        + json.dumps(payload, ensure_ascii=False)[:1000]
                    ),
                )
            file_info = payload.get("file") if isinstance(payload, dict) else None
            file_id = file_info.get("file_id") if isinstance(file_info, dict) else None
            if not isinstance(file_id, (str, int)) or isinstance(file_id, bool) or not str(file_id):
                raise ValueError("Context IR 素材上传未返回 file_id")
            record["file_id"] = str(file_id)
            return ProviderResult(f"{self.FILE_URL_PREFIX}{file_id}", record)
        except ProviderFailure:
            raise
        except Exception as exc:
            failure_record = record or {}
            failure_record.setdefault("outcome", "unknown")
            raise ProviderFailure(failure_record, [], exc) from exc

    def submit(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        video_url: str,
        audio_url: str | None,
        duration: int,
        ratio: str,
    ) -> ProviderResult[str]:
        """提交 Context IR 任务，返回 task_id。素材 URL 必须是 mm_file:// 引用。

        outcome 分类与 PicsartClient.submit 一致：accepted / rejected
        （SAFE_REJECT 4xx，确定性不重试；含 402 余额不足、1026 审核与
        2013 限长）/ unknown（POST 后无 task_id，允许有限自动重试——失败
        不计费）。content 顺序即素材编号顺序：[text] + Image 1..N +
        Video 1 + Audio 1。

        官方限长实测（2026-09-29）：text 项 ≤7000 字符，超出同步 400
        `content[0].text too long: N > 7000 characters (2013)`，不建任务
        不扣费。客户端不在本地预检限长：让供应商 400 走 rejected → 节点
        fail-closed（语义与对账纪律一致，本地预检会改变错误归类）。
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Context IR prompt must be a non-empty string")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or not self.DURATION_MIN_S <= duration <= self.DURATION_MAX_S
        ):
            raise ValueError("Context IR duration must be an int in [4, 15]")
        if ratio not in self.RATIOS:
            raise ValueError("Context IR ratio is unsupported")
        for url in [*image_urls, video_url, *([audio_url] if audio_url else [])]:
            if (
                not isinstance(url, str)
                or not url.startswith(self.FILE_URL_PREFIX)
                or not url[len(self.FILE_URL_PREFIX):]
            ):
                raise ValueError(
                    "Context IR media must be mm_file:// uploads "
                    "（公网 URL 被 MiniMax 阻断数据中心 IP，实测不可用）"
                )
        if len(image_urls) + 1 + (1 if audio_url else 0) > 12:
            raise ValueError("Context IR accepts at most 12 media files")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url},
             "role": "reference_image"}
            for url in image_urls
        )
        content.append({
            "type": "video_url",
            "video_url": {"url": video_url},
            "role": "reference_video",
        })
        if audio_url is not None:
            content.append({
                "type": "audio_url",
                "audio_url": {"url": audio_url},
                "role": "reference_audio",
            })
        body = {
            "model": self.MODEL,
            "content": content,
            "duration": duration,
            "ratio": ratio,
        }
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(**self._http_client_kwargs()) as client:
                try:
                    response, record = _send(
                        client,
                        "POST",
                        f"{self.settings.minimax_context_ir_api_base}{self.SUBMIT_PATH}",
                        headers=_headers(self._api_key()),
                        json_body=body,
                    )
                except ProviderFailure as exc:
                    exc.record["outcome"] = "unknown"
                    raise
            if not response.is_success:
                record["outcome"] = (
                    "rejected"
                    if response.status_code in self.SAFE_REJECT_STATUS_CODES
                    else "unknown"
                )
            payload = _json_body(response)
            # 审核拒绝（1026）可能以 4xx 或 200 信封出现，都要明确报错。
            self._check_moderation(payload, record)
            if not response.is_success:
                _ensure_success(response, record)
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not isinstance(task_id, str) or not task_id:
                record["outcome"] = "unknown"
                raise ValueError("Context IR submit response has no task_id")
            record["outcome"] = "accepted"
            record["task_id"] = task_id
            return ProviderResult(task_id, record)
        except ProviderFailure:
            raise
        except Exception as exc:
            failure_record = record or {}
            failure_record.setdefault("outcome", "unknown")
            raise ProviderFailure(failure_record, [], exc) from exc

    def poll(self, task_id: str) -> ProviderResult[dict[str, Any]]:
        """轮询已建任务到终态，返回 {"prompt": 增强提示词, "usage": {...}}。

        查询端点不按 task_id 过滤（2026-08-18 实测），必须按 items[].id 自取；
        目标不在列表视为仍在排队，绝不取 items[0]。拿到 task_id 后只轮询该
        任务，绝不重新 POST（重复提交即重复计费）。vendor 终态 failed 抛
        ContextIRTaskFailedError（失败不计费，由节点侧计入有限重提上限后
        重新 submit）。
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Context IR task_id must be a non-empty string")
        poll_url = (
            f"{self.settings.minimax_context_ir_api_base}{self.QUERY_PATH}"
            f"?task_id={task_id}"
        )
        record: dict[str, Any] = {"task_id": task_id, "polls": []}
        deadline = time.monotonic() + self.poll_timeout_s
        try:
            with httpx.Client(**self._http_client_kwargs()) as client:
                while True:
                    try:
                        response, exchange = _send(
                            client,
                            "GET",
                            poll_url,
                            headers={
                                "Authorization": f"Bearer {self._api_key()}",
                            },
                        )
                    except ProviderFailure as exc:
                        record["polls"].append(exc.record)
                        raise ProviderFailure(record, [], exc.cause) from exc.cause
                    record["polls"].append(exchange)
                    payload = _json_body(response)
                    self._check_moderation(payload, record)
                    _ensure_success(response, record)
                    items = payload.get("items") if isinstance(payload, dict) else None
                    if not isinstance(items, list):
                        raise TypeError("Context IR 查询未返回任务列表")
                    item = next(
                        (
                            entry
                            for entry in items
                            if isinstance(entry, dict) and entry.get("id") == task_id
                        ),
                        None,
                    )
                    if item is None:
                        # 刚提交或列表延迟：仍在排队，继续轮询。
                        pass
                    else:
                        status = item.get("status")
                        if status == "error":
                            status = "failed"
                        if status == "succeeded":
                            content = item.get("content")
                            prompt = (
                                content.get("prompt")
                                if isinstance(content, dict) else None
                            )
                            if not isinstance(prompt, str) or not prompt.strip():
                                raise ValueError(
                                    "Context IR succeeded task has no content.prompt"
                                )
                            record["usage"] = self._usage_of(item, payload)
                            return ProviderResult(
                                {"prompt": prompt, "usage": record["usage"]}, record,
                            )
                        if status == "failed":
                            vendor_error = item.get("error")
                            raise ContextIRTaskFailedError(
                                "Context IR task failed: "
                                + json.dumps(
                                    vendor_error if vendor_error is not None else item,
                                    ensure_ascii=False,
                                )[:2000]
                            )
                        if status not in self._PENDING_STATUSES:
                            # 只认实测过的状态；未知值是确定性供应商答复，
                            # 不许猜成仍在运行。
                            raise ValueError(
                                f"Context IR 返回未知任务状态: {status!r}"
                            )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Context IR task {task_id} timed out")
                    time.sleep(self.poll_interval_s)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    @staticmethod
    def _usage_of(item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
        """usage 可能在 item.usage / 根 usage / item.content.usage；数值或
        {total_tokens} 都归一成 dict 落回执，拿不到就记 None（不阻断取结果）。"""
        content = item.get("content")
        content_usage = content.get("usage") if isinstance(content, dict) else None
        for usage in (item.get("usage"), payload.get("usage"), content_usage):
            if isinstance(usage, dict):
                return usage
            if isinstance(usage, (int, float)) and not isinstance(usage, bool):
                return {"total_tokens": usage}
        return None
