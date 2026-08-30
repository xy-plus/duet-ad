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
    return tuple(
        value
        for key, value in os.environ.items()
        if _SECRET_KEY.search(key) and len(value) >= 4
    )[:100]


def _safe_text(value: str, secrets: Iterable[str] = ()) -> str:
    safe = value
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, "[REDACTED]")
    return _SECRET_VALUE.sub(lambda match: match.group(1) + "[REDACTED]", safe)[
        :_MAX_TEXT
    ]


def _bounded(value: object, secrets: Iterable[str] = ()) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else _bounded(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_bounded(item, secrets) for item in value[:200]]
    if isinstance(value, str):
        return _safe_text(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_text(repr(value), secrets)


def exception_tree(exc: BaseException, *, secrets: Iterable[str] = ()) -> dict[str, Any]:
    seen: set[int] = set()

    def build(current: BaseException) -> dict[str, Any]:
        if id(current) in seen:
            return {"type": type(current).__name__, "cycle": True}
        seen.add(id(current))
        frames = [
            {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
                "source": _safe_text(frame.line or "", secrets),
            }
            for frame in traceback.extract_tb(current.__traceback__)
        ]
        result: dict[str, Any] = {
            "type": type(current).__name__,
            "message": _safe_text(str(current), secrets),
            "traceback": frames,
        }
        child = current.__cause__ or current.__context__
        if child is not None:
            result["cause"] = build(child)
        return result

    return build(exc)


def provider_response(response: object, *, secrets: Iterable[str] = ()) -> dict[str, Any]:
    status = getattr(response, "status_code", None)
    headers = getattr(response, "headers", {})
    result: dict[str, Any] = {
        "http_status": status,
        "headers": {
            str(key): str(value)[:1024]
            for key, value in dict(headers).items()
            if str(key).lower() in {
                "content-type", "x-request-id", "request-id", "trace-id",
                "x-trace-id", "retry-after",
            }
        },
    }
    try:
        result["body"] = _bounded(response.json(), secrets)
    except Exception:
        result["body"] = _safe_text(str(getattr(response, "text", "")), secrets)
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
    secret_values = tuple(dict.fromkeys(
        secret for secret in (*secrets, *_environment_secrets()) if secret
    ))
    payload: dict[str, Any] = {
        "schema": "duet.error-call-tree",
        "version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "call_path": [_safe_text(str(part), secret_values) for part in call_path],
        "error": (
            exception_tree(error, secrets=secret_values)
            if error is not None
            else _bounded(reason, secret_values)
        ),
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
    except Exception as persist_error:
        if logger is not None:
            try:
                logger.error(
                    "pipeline_error_record_persist_failed path=%s error=%s payload=%s",
                    _safe_text(str(path), secret_values),
                    _safe_text(str(persist_error), secret_values),
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )
            except Exception:
                pass
    if logger is not None:
        try:
            logger.error(
                "pipeline_call_failed %s",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                exc_info=(type(error), error, error.__traceback__) if error else None,
            )
        except Exception:
            pass
    return payload
