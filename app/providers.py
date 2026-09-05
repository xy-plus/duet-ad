from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

import httpx

from app.config import Settings


T = TypeVar("T")
_MISSING = object()
_SECRET_ENV_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
_ENV_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_CODEX_TELEMETRY_RE = re.compile(
    r"[ \t\r\n]*tokens used\r?\n"
    r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)[ \t\r\n]*"
)
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


def _deepseek_base(base: str) -> str:
    canonical = base.rstrip("/")
    if canonical.endswith("/beta"):
        canonical = canonical[:-5]
    return canonical


def _deepseek_chat_url(base: str) -> str:
    return _deepseek_base(base) + "/chat/completions"


def _deepseek_responses_url(base: str) -> str:
    return _deepseek_base(base) + "/responses"


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


def _structured_output(body: Any) -> dict[str, Any]:
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
    if len(messages) != 1:
        raise ValueError("DeepSeek structured response must contain one message")
    message = messages[0]
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
    parsed = json.loads(text_parts[0]["text"])
    if not isinstance(parsed, dict):
        raise TypeError("DeepSeek structured output must decode to an object")
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

    def _request(
        self,
        instruction: str,
        payload: Any,
        images: list[Path] | None,
        *,
        json_mode: bool = False,
    ) -> ProviderResult[str]:
        media = list(images or [])
        try:
            content = self._content(instruction, payload, media)
        except Exception as exc:
            record = {
                "request": {
                    "method": "POST",
                    "url": _deepseek_chat_url(self.settings.deepseek_api_base),
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
            "thinking": {"type": "enabled"},
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        record: dict[str, Any] | None = None
        try:
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                response, record = _send(
                    client,
                    "POST",
                    _deepseek_chat_url(self.settings.deepseek_api_base),
                    headers=_headers(self.settings.deepseek_api_key),
                    json_body=request,
                )
            _ensure_success(response, record, media)
            body = _json_body(response)
            value = _message_content(body)
            return ProviderResult(value=value, record=record, evidence_media=media)
        except ProviderFailure as exc:
            if exc.evidence_media:
                raise
            raise ProviderFailure(exc.record, media, exc.cause) from exc.cause
        except Exception as exc:
            fallback = record or {
                "request": {
                    "method": "POST",
                    "url": _deepseek_chat_url(self.settings.deepseek_api_base),
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
            json_mode=json_mode,
        )

    def complete_structured(
        self,
        instruction: str,
        payload: Any,
        schema: dict[str, Any],
        images: list[Path] | None = None,
    ) -> ProviderResult[dict[str, Any]]:
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

        request: dict[str, Any] = {
            "model": self.settings.deepseek_model,
            "instructions": instruction,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "duet_structured_result",
                    "schema": schema,
                }
            },
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
                    headers=_headers(self.settings.deepseek_api_key),
                    json_body=request,
                )
            _ensure_success(response, record, media)
            value = _structured_output(_json_body(response))
            return ProviderResult(value=value, record=record, evidence_media=media)
        except ProviderFailure as exc:
            if exc.evidence_media:
                raise
            raise ProviderFailure(exc.record, media, exc.cause) from exc.cause
        except Exception as exc:
            fallback = record or {
                "request": {
                    "method": "POST",
                    "url": url,
                    "headers": _headers(self.settings.deepseek_api_key),
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
    suffix = text[suffix_start:]
    if suffix.strip() and _CODEX_TELEMETRY_RE.fullmatch(suffix) is None:
        raise ValueError("Codex structured output contains trailing text")
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
    ) -> ProviderResult[dict[str, Any]]:
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
            with httpx.Client(
                timeout=self.settings.request_timeout_s,
                transport=self.transport,
                trust_env=False,
            ) as client:
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
                not isinstance(url, str)
                or not url
                or url != url.strip()
                for url in image_urls
            ):
                raise ValueError(
                    "Picsart image_urls must contain trimmed non-empty strings"
                )
            if len(image_urls) != len(set(image_urls)):
                raise ValueError("Picsart image_urls must not contain duplicates")

        if audio_urls is not None:
            if not isinstance(audio_urls, list):
                raise ValueError("Picsart audio_urls must be a list or None")
            if len(audio_urls) > 3:
                raise ValueError("Picsart audio_urls supports at most 3 URLs")
            if any(
                not isinstance(url, str)
                or not url
                or url != url.strip()
                for url in audio_urls
            ):
                raise ValueError(
                    "Picsart audio_urls must contain trimmed non-empty strings"
                )
            if len(audio_urls) != len(set(audio_urls)):
                raise ValueError("Picsart audio_urls must not contain duplicates")

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
            with httpx.Client(timeout=self.settings.request_timeout_s, transport=self.transport) as client:
                try:
                    response, record = _send(
                        client,
                        "POST",
                        f"{self.settings.picsart_api_base}/workflows/{self.WORKFLOW}/submit",
                        headers=_headers(self.settings.picsart_api_key),
                        json_body=body,
                    )
                except ProviderFailure as exc:
                    exc.record["outcome"] = "unknown"
                    raise
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

    def poll(self, task_id: str) -> ProviderResult[str]:
        record: dict[str, Any] = {"task_id": task_id, "polls": []}
        poll_url = (
            f"{self.settings.picsart_api_base}/workflows/{self.WORKFLOW}/{task_id}/result"
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
