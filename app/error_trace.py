"""Durable, non-controlling diagnostics for existing pipeline calls."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_SECRET_KEY = re.compile(r"token|secret|password|authorization|api[_-]?key", re.I)
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+|(?:token|secret|password|api[_-]?key)\s*[=:]\s*)"
    r"[^\s,;\]\}\"]+"
)
_MAX_TEXT = 32 * 1024


def _environment_secrets() -> tuple[str, ...]:
    try:
        return tuple(
            value
            for key, value in os.environ.items()
            if _SECRET_KEY.search(key) and len(value) >= 4
        )[:100]
    except BaseException:
        return ()


def _type_name(value: object) -> str:
    try:
        return type(value).__name__
    except BaseException:
        return "unknown"


def _object_text(value: object) -> str:
    try:
        return str(value)
    except BaseException:
        return f"[unprintable {_type_name(value)}]"


def _secret_values(secrets: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    try:
        candidates = (*secrets, *_environment_secrets())
    except BaseException:
        candidates = _environment_secrets()
    for candidate in candidates:
        try:
            value = str(candidate)
        except BaseException:
            continue
        if value and value not in values:
            values.append(value)
        if len(values) >= 200:
            break
    return tuple(values)


def _safe_text_with_truncation(
    value: object, secrets: Iterable[str] = ()
) -> tuple[str, bool]:
    safe = _object_text(value)
    try:
        for secret in secrets:
            if secret:
                safe = safe.replace(secret, "[REDACTED]")
        safe = _SECRET_VALUE.sub(
            lambda match: match.group(1) + "[REDACTED]", safe
        )
    except BaseException:
        safe = f"[unprintable {_type_name(value)}]"
    return safe[:_MAX_TEXT], len(safe) > _MAX_TEXT


def _safe_text(value: object, secrets: Iterable[str] = ()) -> str:
    return _safe_text_with_truncation(value, secrets)[0]


def _bounded_with_truncation(
    value: object,
    secrets: Iterable[str] = (),
    *,
    seen: set[int] | None = None,
) -> tuple[object, bool]:
    active = seen if seen is not None else set()
    if isinstance(value, Mapping):
        if id(value) in active:
            return "[CYCLE]", False
        active.add(id(value))
        result: dict[str, object] = {}
        truncated = False
        try:
            items = list(value.items())
        except BaseException:
            active.discard(id(value))
            return _safe_text_with_truncation(value, secrets)
        if len(items) > 200:
            items = items[:200]
            truncated = True
        for key, item in items:
            safe_key = _object_text(key)
            if _SECRET_KEY.search(safe_key):
                result[safe_key] = "[REDACTED]"
                continue
            bounded, child_truncated = _bounded_with_truncation(
                item, secrets, seen=active
            )
            result[safe_key] = bounded
            truncated = truncated or child_truncated
        active.discard(id(value))
        return result, truncated
    if isinstance(value, list):
        if id(value) in active:
            return "[CYCLE]", False
        active.add(id(value))
        truncated = len(value) > 200
        result = []
        for item in value[:200]:
            bounded, child_truncated = _bounded_with_truncation(
                item, secrets, seen=active
            )
            result.append(bounded)
            truncated = truncated or child_truncated
        active.discard(id(value))
        return result, truncated
    if isinstance(value, str):
        return _safe_text_with_truncation(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    try:
        rendered = repr(value)
    except BaseException:
        rendered = f"[unprintable {_type_name(value)}]"
    return _safe_text_with_truncation(rendered, secrets)


def _bounded(value: object, secrets: Iterable[str] = ()) -> object:
    return _bounded_with_truncation(value, secrets)[0]


def _provider_body_secrets(value: object) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[int] = set()

    def collect_scalar(current: object) -> None:
        if isinstance(current, str) and current and current not in found:
            found.append(current)
        elif isinstance(current, Mapping):
            walk(current, sensitive=True)
        elif isinstance(current, list):
            for item in current[:200]:
                collect_scalar(item)

    def walk(current: object, *, sensitive: bool = False) -> None:
        if len(found) >= 200 or id(current) in seen:
            return
        if isinstance(current, Mapping):
            seen.add(id(current))
            try:
                items = list(current.items())[:200]
            except BaseException:
                return
            for key, item in items:
                key_is_secret = sensitive or bool(
                    _SECRET_KEY.search(_object_text(key))
                )
                if key_is_secret:
                    collect_scalar(item)
                else:
                    walk(item)
        elif isinstance(current, list):
            seen.add(id(current))
            for item in current[:200]:
                walk(item, sensitive=sensitive)

    try:
        walk(value)
    except BaseException:
        pass
    return tuple(found[:200])


def exception_tree(exc: BaseException, *, secrets: Iterable[str] = ()) -> dict[str, Any]:
    seen: set[int] = set()

    def build(current: BaseException) -> dict[str, Any]:
        if id(current) in seen:
            return {"type": _type_name(current), "cycle": True}
        seen.add(id(current))
        try:
            extracted = traceback.extract_tb(current.__traceback__)
        except BaseException:
            extracted = ()
        frames = []
        for frame in extracted:
            try:
                frames.append({
                    "file": _safe_text(frame.filename, secrets),
                    "line": frame.lineno,
                    "function": _safe_text(frame.name, secrets),
                    "source": _safe_text(frame.line or "", secrets),
                })
            except BaseException:
                continue
        result: dict[str, Any] = {
            "type": _type_name(current),
            "message": _safe_text(current, secrets),
            "traceback": frames,
        }
        try:
            members = current.exceptions if isinstance(current, BaseExceptionGroup) else ()
        except BaseException:
            members = ()
        if members:
            result["exceptions"] = [build(member) for member in members]
        try:
            cause = current.__cause__
        except BaseException:
            cause = None
        try:
            context = current.__context__
        except BaseException:
            context = None
        if cause is not None:
            result["cause"] = build(cause)
        if context is not None and context is not cause:
            result["context"] = build(context)
            try:
                if current.__suppress_context__:
                    result["context_suppressed"] = True
            except BaseException:
                pass
        return result

    try:
        return build(exc)
    except BaseException as tree_error:
        return {
            "type": _type_name(exc),
            "message": _safe_text(exc, secrets),
            "traceback": [],
            "diagnostic_error": _safe_text(tree_error, secrets),
        }


def provider_response(response: object, *, secrets: Iterable[str] = ()) -> dict[str, Any]:
    secret_values = _secret_values(secrets)
    try:
        status = getattr(response, "status_code", None)
    except BaseException:
        status = None
    try:
        header_items = list(getattr(response, "headers", {}).items())
    except BaseException:
        header_items = []
    safe_headers: dict[str, str] = {}
    for key, value in header_items:
        safe_key = _object_text(key)
        if safe_key.lower() not in {
            "content-type", "x-request-id", "request-id", "trace-id",
            "x-trace-id", "retry-after",
        }:
            continue
        safe_headers[safe_key] = _safe_text(value, secret_values)[:1024]
    try:
        raw_value = getattr(response, "text", "")
    except BaseException:
        raw_value = ""
    result: dict[str, Any] = {
        "http_status": status,
        "headers": safe_headers,
    }
    try:
        parsed = response.json()
        body_secret_values = _secret_values(
            (*secret_values, *_provider_body_secrets(parsed))
        )
        result["body"], structured_truncated = _bounded_with_truncation(
            parsed, body_secret_values
        )
    except BaseException:
        body_secret_values = secret_values
        structured_truncated = False
    raw_body, raw_truncated = _safe_text_with_truncation(
        raw_value, body_secret_values
    )
    if "body" not in result:
        result["body"] = raw_body
    result["body_raw"] = raw_body
    result["body_truncated"] = raw_truncated or structured_truncated
    return result


def record(
    path: Path,
    *,
    call_path: list[str],
    error: BaseException | None = None,
    reason: object | None = None,
    logger: logging.Logger | None = None,
    secrets: Iterable[str] = (),
) -> dict[str, Any]:
    secret_values = _secret_values(secrets)
    try:
        payload: dict[str, Any] = {
            "schema": "duet.error-call-tree",
            "version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "call_path": [_safe_text(part, secret_values) for part in call_path],
            "error": (
                exception_tree(error, secrets=secret_values)
                if error is not None
                else _bounded(reason, secret_values)
            ),
        }
    except BaseException as payload_error:
        payload = {
            "schema": "duet.error-call-tree",
            "version": 1,
            "recorded_at": "",
            "call_path": [],
            "error": {
                "type": _type_name(error) if error is not None else "diagnostic",
                "message": (
                    _safe_text(error, secret_values)
                    if error is not None
                    else "[diagnostic payload unavailable]"
                ),
                "traceback": [],
                "diagnostic_error": _safe_text(payload_error, secret_values),
            },
        }
    name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            name = None
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)
    except BaseException as persist_error:
        if logger is not None:
            try:
                logger.error(
                    "pipeline_error_record_persist_failed path=%s error=%s payload=%s",
                    _safe_text(str(path), secret_values),
                    _safe_text(str(persist_error), secret_values),
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )
            except BaseException:
                pass
    if logger is not None:
        try:
            logger.error(
                "pipeline_call_failed %s",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                exc_info=(type(error), error, error.__traceback__) if error else None,
            )
        except BaseException:
            pass
    return payload
