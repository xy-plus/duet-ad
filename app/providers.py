from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

import httpx

from app.config import Settings


T = TypeVar("T")
_MISSING = object()


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


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    json_body: Any = _MISSING,
    content: bytes | None = None,
    request_body: Any = _MISSING,
) -> tuple[httpx.Response, dict[str, Any]]:
    kwargs: dict[str, Any] = {"headers": headers}
    if json_body is not _MISSING:
        kwargs["json"] = json_body
        if request_body is _MISSING:
            request_body = json_body
    if content is not None:
        kwargs["content"] = content
    request = client.build_request(method, url, **kwargs)
    try:
        response = client.send(request)
    except httpx.HTTPError as exc:
        raise ProviderFailure(_transport_record(request, request_body), [], exc) from exc
    return response, response_record(response, request_body=request_body)


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


def _deepseek_url(base: str, *, structured: bool) -> str:
    canonical = base.rstrip("/")
    if canonical.endswith("/beta"):
        canonical = canonical[:-5]
    suffix = "/beta/chat/completions" if structured else "/chat/completions"
    return canonical + suffix


def _message_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise TypeError("DeepSeek response must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("DeepSeek response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("DeepSeek response has no message content")
    return message["content"]


def _structured_arguments(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise TypeError("DeepSeek response must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("DeepSeek response has no unique choice")
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        raise ValueError("DeepSeek response did not finish with a tool call")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("DeepSeek response has no message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError("DeepSeek response must contain exactly one tool call")
    function = calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != "submit_result":
        raise ValueError("DeepSeek response did not call submit_result")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise TypeError("DeepSeek submit_result arguments must be JSON text")
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise TypeError("DeepSeek submit_result arguments must decode to an object")
    return parsed


class DeepSeekClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport

    @staticmethod
    def _content(instruction: str, payload: Any, images: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": instruction + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False),
            }
        ]
        for image in images:
            image_bytes = image.read_bytes()
            mime = _media_type(image, image_bytes)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
            )
        return content

    def _request(
        self,
        instruction: str,
        payload: Any,
        images: list[Path] | None,
        *,
        structured: bool,
        schema: dict[str, Any] | None = None,
        json_mode: bool = False,
    ) -> ProviderResult[Any]:
        media = list(images or [])
        try:
            content = self._content(instruction, payload, media)
        except Exception as exc:
            record = {
                "request": {
                    "method": "POST",
                    "url": _deepseek_url(self.settings.deepseek_api_base, structured=structured),
                    "headers": _headers(self.settings.deepseek_api_key),
                    "body": {"instruction": instruction, "payload": payload},
                },
                "response": None,
            }
            raise ProviderFailure(record, media, exc) from exc

        request: dict[str, Any] = {
            "model": self.settings.deepseek_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        if structured:
            request.update(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return the complete result only through one submit_result "
                            "function call. Do not return prose or Markdown."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0,
                stream=False,
                thinking={"type": "disabled"},
            )
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_result",
                        "description": "Submit the complete structured result.",
                        "strict": True,
                        "parameters": schema,
                    },
                }
            ]
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": "submit_result"},
            }
        elif json_mode:
            request["response_format"] = {"type": "json_object"}

        record: dict[str, Any] | None = None
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                response, record = _send(
                    client,
                    "POST",
                    _deepseek_url(self.settings.deepseek_api_base, structured=structured),
                    headers=_headers(self.settings.deepseek_api_key),
                    json_body=request,
                )
            _ensure_success(response, record, media)
            body = _json_body(response)
            value = _structured_arguments(body) if structured else _message_content(body)
            return ProviderResult(value=value, record=record, evidence_media=media)
        except ProviderFailure as exc:
            if exc.evidence_media:
                raise
            raise ProviderFailure(exc.record, media, exc.cause) from exc.cause
        except Exception as exc:
            fallback = record or {
                "request": {
                    "method": "POST",
                    "url": _deepseek_url(self.settings.deepseek_api_base, structured=structured),
                    "headers": _headers(self.settings.deepseek_api_key),
                    "body": request,
                },
                "response": None,
            }
            raise ProviderFailure(fallback, media, exc) from exc

    def complete(
        self,
        instruction: str,
        payload: Any,
        images: list[Path] | None = None,
        *,
        json_mode: bool = False,
    ) -> ProviderResult[str]:
        return self._request(
            instruction,
            payload,
            images,
            structured=False,
            json_mode=json_mode,
        )

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        return self._request(
            instruction,
            payload,
            images,
            structured=True,
            schema=schema,
        )


def _upload_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            raise TypeError("MediaKit upload headers must contain strings")
        return dict(value)
    if isinstance(value, list):
        headers: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("MediaKit upload header entry must be an object")
            key = item.get("key") or item.get("Key")
            val = item.get("value") or item.get("Value")
            if not isinstance(key, str) or not isinstance(val, str):
                raise TypeError("MediaKit upload header entry must contain string key/value")
            headers[key] = val
        return headers
    raise TypeError("MediaKit upload_headers must be an object or list")


def _setdefault_header(headers: dict[str, str], name: str, value: str) -> None:
    if not any(key.lower() == name.lower() for key in headers):
        headers[name] = value


def _upload_spec(body: Any) -> tuple[str, str, str, dict[str, str]]:
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ValueError("MediaKit upload initialization was not successful")
    result = body.get("result")
    if not isinstance(result, dict):
        raise TypeError("MediaKit upload initialization result must be an object")
    nested = result.get("upload")
    if nested is not None and not isinstance(nested, dict):
        raise TypeError("MediaKit upload initialization upload must be an object")
    values = {**result, **(nested or {})}
    file_id = values.get("file_id")
    method = values.get("method", "PUT")
    upload_url = values.get("upload_url") or values.get("url")
    raw_headers = values.get("upload_headers")
    if raw_headers is None and "headers" in values:
        raw_headers = values.get("headers")
    if not isinstance(file_id, str) or not file_id:
        raise ValueError("MediaKit upload initialization has no file_id")
    if not isinstance(method, str) or not method:
        raise ValueError("MediaKit upload initialization has no method")
    if not isinstance(upload_url, str) or not upload_url:
        raise ValueError("MediaKit upload initialization has no upload_url")
    return file_id, method.upper(), upload_url, _upload_headers(raw_headers)


def _mediakit_result_url(body: Any) -> str:
    if not isinstance(body, dict) or body.get("success") is not True:
        raise ValueError("MediaKit erase was not successful")
    result = body.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("image_url"), str):
        raise ValueError("MediaKit erase response has no image_url")
    return result["image_url"]


class MediaKitClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport

    def erase(
        self,
        source: Path,
        destination: Path,
        scenes: list[str],
    ) -> ProviderResult[Path]:
        record: dict[str, Any] = {"stages": []}
        evidence_media: list[Path] = []
        current = source
        current_mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                for index, scene in enumerate(scenes, 1):
                    stage: dict[str, Any] = {"scene": scene}
                    record["stages"].append(stage)
                    phase = "initialize"
                    try:
                        init, init_record = _send(
                            client,
                            "POST",
                            f"{self.settings.mediakit_api_base}/tools-sync/request-media-upload-url",
                            headers=_headers(self.settings.mediakit_api_key),
                            json_body={},
                        )
                        stage["initialize"] = init_record
                        _ensure_success(init, init_record, evidence_media)
                        file_id, method, upload_url, upload_headers = _upload_spec(
                            _json_body(init)
                        )

                        phase = "upload"
                        _setdefault_header(upload_headers, "Content-Type", current_mime)
                        source_bytes = current.read_bytes()
                        upload_body = _binary_record(source_bytes, current_mime)
                        uploaded, upload_record = _send(
                            client,
                            method,
                            upload_url,
                            headers=upload_headers,
                            content=source_bytes,
                            request_body=upload_body,
                        )
                        upload_record["request"]["body_file"] = str(current)
                        stage["upload"] = upload_record
                        _ensure_success(uploaded, upload_record, evidence_media)

                        phase = "erase"
                        erase_body = {"image_url": file_id, "standard_scene": scene}
                        erased, erase_record = _send(
                            client,
                            "POST",
                            f"{self.settings.mediakit_api_base}/tools-sync/erase-image",
                            headers=_headers(self.settings.mediakit_api_key),
                            json_body=erase_body,
                        )
                        stage["erase"] = erase_record
                        _ensure_success(erased, erase_record, evidence_media)
                        output_url = _mediakit_result_url(_json_body(erased))

                        phase = "download"
                        downloaded, download_record = _send(client, "GET", output_url)
                        stage["download"] = download_record
                        _ensure_success(downloaded, download_record, evidence_media)
                        stage_output = (
                            destination
                            if index == len(scenes)
                            else destination.with_name(
                                f"{destination.stem}.mediakit-stage-{index:02d}{destination.suffix}"
                            )
                        )
                        stage_output.write_bytes(downloaded.content)
                        evidence_media.append(stage_output)
                        current = stage_output
                        current_mime = (
                            _content_type(downloaded.headers)
                            or mimetypes.guess_type(stage_output.name)[0]
                            or "application/octet-stream"
                        )
                    except ProviderFailure as exc:
                        stage[phase] = exc.record
                        raise ProviderFailure(record, evidence_media + exc.evidence_media, exc.cause) from exc.cause
            if not scenes:
                if source != destination:
                    shutil.copyfile(source, destination)
                evidence_media.append(destination)
            return ProviderResult(destination, record, evidence_media)
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, evidence_media, exc) from exc


class PicsartClient:
    MODEL = "minimax-h3-max-r2v"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport

    def submit(
        self,
        *,
        prompt: str,
        video_url: str,
        image_url: str | None,
        duration: int,
        resolution: str,
        aspect_ratio: str,
    ) -> ProviderResult[str]:
        inner: dict[str, Any] = {
            "prompt": prompt,
            "videoUrls": [video_url],
            "resolution": resolution,
            "duration": duration,
            "aspectRatio": aspect_ratio,
            "promptExpansionMode": "quality",
        }
        if image_url:
            inner["imageUrls"] = [image_url]
        body = {"params": {"model": self.MODEL, "params": inner}}
        record: dict[str, Any] | None = None
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                try:
                    response, record = _send(
                        client,
                        "POST",
                        f"{self.settings.picsart_api_base}/workflows/v1/models/submit",
                        headers=_headers(self.settings.picsart_api_key),
                        json_body=body,
                    )
                except ProviderFailure as exc:
                    exc.record["outcome"] = "unknown"
                    raise
            if not response.is_success:
                record["outcome"] = "rejected"
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

    def poll(self, task_id: str) -> ProviderResult[str]:
        record: dict[str, Any] = {"task_id": task_id, "polls": []}
        poll_url = (
            f"{self.settings.picsart_api_base}/workflows/v1/models/{task_id}/result"
        )
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                while True:
                    try:
                        response, exchange = _send(
                            client,
                            "GET",
                            poll_url,
                            headers=_headers(self.settings.picsart_api_key),
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
                        url = result.get("url")
                        if not isinstance(url, str) or not url:
                            items = result.get("items")
                            first = items[0] if isinstance(items, list) and items else None
                            url = first.get("url") if isinstance(first, dict) else None
                        if not isinstance(url, str) or not url:
                            raise ValueError("Picsart completed result has no URL")
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
