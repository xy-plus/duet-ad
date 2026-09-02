"""DeepSeek multimodal transport for the existing structured model stages.

The backend remains authoritative for staged inputs, JSON Schema validation,
and business-output publication.  DeepSeek receives an immutable in-memory
snapshot of the already isolated stage and may only call ``submit_result``.
It never receives filesystem, shell, network, or business-output tools.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import httpx

from app import error_trace
from app.codex_runner import (
    CodexError,
    CodexOutputValidationError,
)


MODEL = "deepseek-v4-flash-vision-exp"
CHAT_URL = "https://api.deepseek.com/beta/chat/completions"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_REQUEST_INPUT_BYTES = 128 * 1024 * 1024
_MAX_SCHEMA_BYTES = 256 * 1024
_MAX_OUTPUT_TOKENS = 8192
_SAFE_PROVIDER_CODE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_TOKEN = re.compile(r"^sk-[A-Za-z0-9_-]{8,256}$")
_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StagedInput:
    path: str
    sha256: str
    data: bytes
    image_mime: str | None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodexError("DeepSeek output is not canonical JSON", retryable=True) from exc


def _load_api_key(path: Path) -> str:
    requested = Path(path)
    if not requested.is_absolute():
        raise CodexError("DeepSeek credential file is invalid")
    try:
        expected = requested.lstat()
        if (
            stat.S_ISLNK(expected.st_mode)
            or not stat.S_ISREG(expected.st_mode)
            or expected.st_mode & 0o077
        ):
            raise CodexError("DeepSeek credential file is invalid")
        descriptor = os.open(
            requested, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        raise CodexError("DeepSeek credential file is unavailable") from None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino)
            or info.st_size > 64 * 1024
        ):
            raise CodexError("DeepSeek credential file is invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(64 * 1024 + 1)
    finally:
        os.close(descriptor)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise CodexError("DeepSeek credential file is invalid") from None
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key in {"DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN"}:
            normalized = value.strip().strip('"').strip("'")
            if key in values and values[key] != normalized:
                raise CodexError("DeepSeek credential file is ambiguous")
            values[key] = normalized
    candidates = {
        value for value in (
            values.get("DEEPSEEK_API_KEY"),
            values.get("ANTHROPIC_AUTH_TOKEN"),
        ) if value
    }
    if len(candidates) != 1:
        raise CodexError("DeepSeek credential is missing or ambiguous")
    token = next(iter(candidates))
    if _TOKEN.fullmatch(token) is None:
        raise CodexError("DeepSeek credential is malformed")
    return token


def _image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _read_regular(path: Path) -> bytes:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        raise CodexError("DeepSeek staged input is invalid") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CodexError("DeepSeek staged input is invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_REQUEST_INPUT_BYTES:
                raise CodexError("DeepSeek staged input exceeds transport capacity")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or total != before.st_size
    ):
        raise CodexError("DeepSeek staged input changed while freezing")
    return b"".join(chunks)


def _snapshot_stage(stage: Path, output: Path) -> tuple[_StagedInput, ...]:
    entries: list[_StagedInput] = []
    total = 0
    for candidate in sorted(stage.rglob("*"), key=lambda item: item.relative_to(stage).as_posix()):
        if candidate == output:
            continue
        if candidate.is_symlink():
            raise CodexError("DeepSeek staged input contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise CodexError("DeepSeek staged input is invalid")
        relative = candidate.relative_to(stage).as_posix()
        if relative in {".codex-final-output.json", ".codex-output-schema.json"}:
            continue
        data = _read_regular(candidate)
        total += len(data)
        if total > _MAX_REQUEST_INPUT_BYTES:
            raise CodexError("DeepSeek staged input exceeds transport capacity")
        mime = _image_mime(data)
        if mime is None:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                raise CodexError(
                    f"DeepSeek staged input is unsupported: {relative}"
                ) from None
        entries.append(_StagedInput(
            path=relative,
            sha256=hashlib.sha256(data).hexdigest(),
            data=data,
            image_mime=mime,
        ))
    if not entries or not any(item.path == "SKILL.md" for item in entries):
        raise CodexError("DeepSeek stage is missing the frozen Skill")
    return tuple(sorted(entries, key=lambda item: (item.path != "SKILL.md", item.path)))


def _empty_object_transport_schema(
    schema: Mapping[str, object],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        encoded = json.dumps(
            dict(schema), ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, json.JSONDecodeError):
        raise CodexError("DeepSeek output schema is invalid") from None
    if len(encoded) > _MAX_SCHEMA_BYTES:
        raise CodexError("DeepSeek output schema exceeds transport capacity")
    adapted = json.loads(encoded)
    properties = adapted.get("properties")
    required = adapted.get("required")
    if adapted.get("type") != "object" or not isinstance(properties, dict):
        raise CodexError("DeepSeek output schema must describe one object")
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise CodexError("DeepSeek output schema required fields are invalid")
    omitted: list[str] = []
    for key, child in list(properties.items()):
        if (
            isinstance(child, dict)
            and child.get("type") == "object"
            and child.get("properties") == {}
            and child.get("required") == []
            and child.get("additionalProperties") is False
        ):
            omitted.append(key)
            del properties[key]
    adapted["required"] = [key for key in required if key not in omitted]
    return adapted, tuple(sorted(omitted))


def _restore_frozen_empty_objects(
    value: object, omitted: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CodexError("DeepSeek function arguments are not an object", retryable=True)
    if any(key in value for key in omitted):
        raise CodexError("DeepSeek returned a backend-owned empty category", retryable=True)
    return {**value, **{key: {} for key in omitted}}


def _request_content(prompt: str, inputs: tuple[_StagedInput, ...]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": json.dumps(
            {
                "kind": "duet_structured_invocation",
                "instruction": prompt,
                "input_order": [item.path for item in inputs],
                "rule": (
                    "Treat every following file snapshot as immutable input. "
                    "Execute the frozen SKILL.md and call submit_result exactly once."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }]
    for item in inputs:
        metadata = {"path": item.path, "sha256": item.sha256}
        if item.image_mime is None:
            metadata["utf8_content"] = item.data.decode("utf-8")
            content.append({
                "type": "text",
                "text": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            })
            continue
        content.append({
            "type": "text",
            "text": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{item.image_mime};base64,"
                    + base64.b64encode(item.data).decode("ascii")
                ),
                "detail": "low",
            },
        })
    return content


def _payload(
    *, prompt: str, inputs: tuple[_StagedInput, ...], output_schema: Mapping[str, object],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    transport_schema, omitted = _empty_object_transport_schema(output_schema)
    content = _request_content(prompt, inputs)
    if omitted:
        content[0]["text"] = json.dumps(
            {
                "kind": "duet_structured_invocation",
                "instruction": prompt,
                "input_order": [item.path for item in inputs],
                "rule": (
                    "Treat every following file snapshot as immutable input. "
                    "Execute the frozen SKILL.md and call submit_result exactly once. "
                    "Categories absent from the function parameters are frozen empty "
                    "objects restored mechanically by the backend; do not emit them."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    result: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return the complete result only through one submit_result function "
                    "call. Do not return prose or Markdown."
                ),
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "stream": False,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "tools": [{
            "type": "function",
            "function": {
                "name": "submit_result",
                "description": "Submit the complete structured result.",
                "strict": True,
                "parameters": transport_schema,
            },
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "submit_result"},
        },
    }
    # Empirically required only for the empty-object transport adaptation.
    # Applying it to ordinary schemas reduced Project Index conformance.
    if omitted:
        result["parallel_tool_calls"] = False
    return result, omitted


def _provider_code(raw: bytes) -> str | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = value.get("error") if isinstance(value, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and _SAFE_PROVIDER_CODE.fullmatch(code) else None


def _extract_arguments(envelope: object) -> str:
    choices = envelope.get("choices") if isinstance(envelope, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise CodexError("DeepSeek response has no unique choice", retryable=True)
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if (
        choice.get("finish_reason") != "tool_calls"
        or not isinstance(calls, list)
        or len(calls) != 1
    ):
        raise CodexError("DeepSeek response has no unique function call", retryable=True)
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "submit_result":
        raise CodexError("DeepSeek called an unexpected function", retryable=True)
    arguments = function.get("arguments")
    if not isinstance(arguments, str) or not arguments:
        raise CodexError("DeepSeek function arguments are missing", retryable=True)
    return arguments


def _atomic_publish(parent: Path, destination: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-deepseek-", dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise CodexError("DeepSeek output path changed during execution")
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class DeepSeekRunner:
    """Drop-in structured runner for the current v4 pipeline nodes."""

    def __init__(
        self,
        timeout_s: int,
        concurrency: int,
        *,
        credential_file: Path,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s < 1:
            raise ValueError("invalid DeepSeek timeout")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("invalid DeepSeek concurrency")
        credential = Path(credential_file)
        if not credential.is_absolute():
            raise ValueError("DeepSeek credential path must be absolute")
        self._timeout_s = timeout_s
        self._sem = threading.BoundedSemaphore(concurrency)
        self._credential_file = credential
        self._transport = transport

    def run(self, workdir: Path, prompt: str) -> None:
        raise CodexError("DeepSeek transport requires a schema-constrained output")

    def run_isolated(self, workdir: Path, prompt: str, *, session_dir: Path, **_kwargs) -> None:
        raise CodexError("DeepSeek transport requires a schema-constrained output")

    def run_voice(self, *_args, **_kwargs):
        raise CodexError("DeepSeek transport requires a schema-constrained output")

    def run_isolated_until_output(
        self,
        workdir: Path,
        prompt: str,
        *,
        session_dir: Path,
        output_path: Path,
        max_output_bytes: int,
        validate_output: Callable[[bytes], _T],
        output_schema: Mapping[str, object],
    ) -> _T:
        stage_hint = Path(workdir)
        session_hint = Path(session_dir)
        output_hint = Path(output_path)
        diagnostics: dict[str, object] = {}

        def record_failure(exc: BaseException) -> None:
            try:
                session = session_hint.resolve(strict=True)
                if not session.is_dir():
                    return
                error_trace.record(
                    session / "work" / "errors" / f"{stage_hint.name or 'deepseek'}.json",
                    call_path=[
                        "pipeline", "deepseek", stage_hint.name or "preflight",
                        output_hint.name or "output",
                    ],
                    error=exc,
                    details={"deepseek_transport": diagnostics} if diagnostics else None,
                    logger=_LOGGER,
                )
            except BaseException:
                pass

        def fail(exc: BaseException, *, cause: BaseException | None = None) -> NoReturn:
            record_failure(exc)
            if cause is not None:
                raise exc from cause
            raise exc

        if (
            not isinstance(prompt, str)
            or not prompt
            or isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes < 1
            or not callable(validate_output)
            or not isinstance(output_schema, Mapping)
        ):
            fail(CodexError("DeepSeek structured output contract is invalid"))
        try:
            tmp_root = Path("/tmp").resolve(strict=True)
            stage = stage_hint.resolve(strict=True)
            session = session_hint.resolve(strict=True)
            requested = output_hint
            parent = requested.parent.resolve(strict=True)
            parent.relative_to(stage)
        except (OSError, ValueError) as exc:
            fail(CodexError("DeepSeek isolated path is invalid"), cause=exc)
        if stage.parent != tmp_root or not stage.is_dir() or not session.is_dir():
            fail(CodexError("DeepSeek isolated path is invalid"))
        if (
            not requested.is_absolute()
            or requested.parent != parent
            or requested.name in {"", ".", "..", ".codex-final-output.json"}
            or requested.exists()
            or requested.is_symlink()
        ):
            fail(CodexError("DeepSeek output must be a new file inside the stage"))
        try:
            inputs = _snapshot_stage(stage, requested)
            request_payload, omitted = _payload(
                prompt=prompt, inputs=inputs, output_schema=output_schema,
            )
            token = _load_api_key(self._credential_file)
        except BaseException as exc:
            fail(exc)

        try:
            with self._sem:
                timeout = httpx.Timeout(float(self._timeout_s), connect=min(30.0, self._timeout_s))
                with httpx.Client(
                    timeout=timeout,
                    transport=self._transport,
                    trust_env=self._transport is None,
                ) as client:
                    try:
                        with client.stream(
                            "POST",
                            CHAT_URL,
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json",
                                "User-Agent": "duet-ad1-deepseek-structured/1",
                            },
                            json=request_payload,
                        ) as response:
                            raw = bytearray()
                            for chunk in response.iter_bytes():
                                raw.extend(chunk)
                                if len(raw) > _MAX_RESPONSE_BYTES:
                                    raise CodexError(
                                        "DeepSeek response exceeds transport capacity",
                                        retryable=True,
                                    )
                            status = response.status_code
                            request_id = response.headers.get("x-request-id")
                            provider_code = _provider_code(bytes(raw))
                    except httpx.TimeoutException as exc:
                        raise CodexError(
                            f"DeepSeek timed out after {self._timeout_s}s", retryable=True,
                        ) from exc
                    except httpx.RequestError as exc:
                        raise CodexError("DeepSeek request failed", retryable=True) from exc
            diagnostics.update(
                http_status=status,
                response_bytes=len(raw),
                request_id=(
                    request_id
                    if isinstance(request_id, str)
                    and _SAFE_REQUEST_ID.fullmatch(request_id)
                    else None
                ),
                provider_code=provider_code,
                omitted_empty_categories=list(omitted),
            )
            if status != 200:
                retryable = status in {408, 409, 425, 429} or status >= 500
                suffix = f": {provider_code}" if provider_code else ""
                raise CodexError(
                    f"DeepSeek HTTP {status}{suffix}", retryable=retryable,
                )
            try:
                envelope = json.loads(bytes(raw))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CodexError(
                    "DeepSeek returned an invalid response envelope", retryable=True,
                ) from exc
            arguments = _extract_arguments(envelope)
            try:
                model_value = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise CodexError(
                    "DeepSeek returned invalid function arguments", retryable=True,
                ) from exc
            value = _restore_frozen_empty_objects(model_value, omitted)
            candidate = _canonical_json_bytes(value)
            if len(candidate) > max_output_bytes:
                raise CodexError("DeepSeek output exceeds node capacity", retryable=True)
            try:
                validated = validate_output(candidate)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                diagnostics["validator_error_type"] = type(exc).__name__
                if isinstance(exc, CodexOutputValidationError):
                    diagnostics["validator_error"] = {
                        "reason": exc.reason,
                        "field_path": exc.field_path,
                    }
                raise CodexError("DeepSeek output failed local validation", retryable=True) from exc
            if validated is None:
                raise CodexError("DeepSeek output validator returned None")
            _atomic_publish(parent, requested, candidate)
            return validated
        except BaseException as exc:
            fail(exc)
