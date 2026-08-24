"""Recoverable direct H3 generation primitives.

The caller owns review/confirmation and supplies an immutable request.  This
module owns the paid-provider crash boundary: every provider submission is
claimed on disk before POST, provider task identifiers are persisted before
polling, and recovery never creates a new provider task.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import ipaddress
import json
import logging
import math
import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence, TypeVar
from urllib.parse import urlsplit

import httpx

from app.retry import RetryPolicy, run_with_retry
from app.sanitize import sanitize


SCHEMA_VERSION = 1
H3_WORKFLOW = "minimax_h3_lightx2v_v5"
H3_BOUNDARY_WORKFLOW = "minimax_h3_lightx2v"
H3_RESOLUTION = "768p竖"
H3_MAX_DURATION_S = 10
H3_BOUNDARY_MAX_DURATION_S = 15
AUTODL_BASE_URL = "https://autodl.art"
MAX_VIDEO_BYTES = 200 * 1024 * 1024

_SAFE_ERROR_CODES = {
    "h3_submit_rejected",
    "h3_query_failed",
    "h3_result_missing",
    "h3_provider_failed",
    "h3_timeout",
    "download_failed",
    "download_dns_failed",
    "download_peer_unverified",
    "download_url_rejected",
    "download_redirect_rejected",
    "download_too_large",
    "download_invalid_video",
    "output_probe_failed",
    "output_write_failed",
}

log = logging.getLogger(__name__)
_T = TypeVar("_T")

FrozenKeyframes = tuple[tuple[Path, bytes], ...]
FrozenFrame = tuple[Path, bytes]
FrozenVoiceTexts = tuple[str, ...]
H3Mode = Literal["reference", "boundary"]


class H3Error(RuntimeError):
    """A safe, stable error whose text never includes provider data."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class ReceiptError(H3Error):
    """Persisted state or caller-supplied frozen input does not match."""


class H3BusyError(H3Error):
    """Another process/thread currently owns this session."""


class _AutomaticRetryH3Error(H3Error):
    """A safe same-attempt failure that may be retried without another POST."""


class _DNSLookupFailed(Exception):
    pass


class _ProbeUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Timeouts:
    request_s: float = 30.0
    h3_poll_s: float = 1500.0
    download_s: float = 180.0
    poll_interval_s: float = 3.0
    probe_s: float = 30.0
    retry_count: int = 2
    retry_interval_s: float = 15.0

    def __post_init__(self) -> None:
        positive = (
            self.request_s,
            self.h3_poll_s,
            self.download_s,
            self.probe_s,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise H3Error("invalid_timeout")
        if isinstance(self.poll_interval_s, bool) or self.poll_interval_s < 0:
            raise H3Error("invalid_timeout")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise H3Error("invalid_timeout")
        if (
            isinstance(self.retry_interval_s, bool)
            or not isinstance(self.retry_interval_s, (int, float))
            or not math.isfinite(float(self.retry_interval_s))
            or self.retry_interval_s < 0
        ):
            raise H3Error("invalid_timeout")


@dataclass(frozen=True)
class H3Request:
    cid: str
    workdir: Path
    client_request_id: str
    prompt: str
    keyframes: FrozenKeyframes
    voice_texts: FrozenVoiceTexts
    voice_receipt: str
    duration: int
    autodl_token: str
    timeouts: Timeouts = Timeouts()
    mode: H3Mode = "reference"
    first_frame: FrozenFrame | None = None
    last_frame: FrozenFrame | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workdir", Path(self.workdir))
        if not isinstance(self.cid, str) or not self.cid.strip():
            raise H3Error("invalid_cid")
        if not isinstance(self.client_request_id, str) or not self.client_request_id.strip():
            raise H3Error("invalid_client_request_id")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise H3Error("invalid_prompt")
        if not isinstance(self.mode, str) or self.mode not in {"reference", "boundary"}:
            raise H3Error("invalid_mode")
        max_duration = (
            H3_MAX_DURATION_S
            if self.mode == "reference"
            else H3_BOUNDARY_MAX_DURATION_S
        )
        if (
            not isinstance(self.duration, int)
            or isinstance(self.duration, bool)
            or not 1 <= self.duration <= max_duration
        ):
            raise H3Error("invalid_duration")
        if not isinstance(self.autodl_token, str) or not self.autodl_token.strip():
            raise H3Error("missing_autodl_credential")
        if not isinstance(self.keyframes, tuple):
            raise H3Error("invalid_keyframes")
        if self.mode == "reference":
            if self.first_frame is not None or self.last_frame is not None:
                raise H3Error("mixed_h3_inputs")
            if not 1 <= len(self.keyframes) <= 9:
                raise H3Error("invalid_keyframes")
            names = [
                _validate_frame(item, "invalid_keyframes")[0].name
                for item in self.keyframes
            ]
            if len(names) != len(set(names)):
                raise H3Error("duplicate_keyframe_name")
            if self.seed is not None and (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, int)
                or not 1 <= self.seed <= 999_999_999_999_999
            ):
                raise H3Error("invalid_seed")
        else:
            if self.keyframes:
                raise H3Error("mixed_h3_inputs")
            if self.first_frame is None or self.last_frame is None:
                raise H3Error("invalid_boundary_frames")
            _validate_frame(self.first_frame, "invalid_boundary_frames")
            _validate_frame(self.last_frame, "invalid_boundary_frames")
            if self.seed is not None:
                raise H3Error("seed_not_supported")
        if not isinstance(self.voice_texts, tuple):
            raise H3Error("invalid_voice_texts")
        if any(not isinstance(text, str) or not text for text in self.voice_texts):
            raise H3Error("invalid_voice_texts")
        if self.voice_receipt != voice_texts_receipt(self.voice_texts):
            raise ReceiptError("voice_receipt_mismatch")


def _validate_frame(item: Any, code: str) -> FrozenFrame:
    if not isinstance(item, tuple) or len(item) != 2:
        raise H3Error(code)
    path, blob = item
    if not isinstance(path, Path) or not isinstance(blob, bytes) or not blob:
        raise H3Error(code)
    if not path.name or path.name != Path(path.name).name:
        raise H3Error(code)
    return path, blob


@dataclass(frozen=True)
class H3Result:
    status: str
    attempt_id: str | None
    output: Path | None = None
    retryable: bool = False
    error_code: str | None = None


def _retryable_http_status(status: int) -> bool:
    return status in {408, 425, 429} or 500 <= status <= 599


def _provider_error_detail(payload: Mapping[str, Any], *, secret: str) -> str:
    """Extract only provider-owned error fields; never log the full response."""

    values: list[str] = []

    def append(value: Any) -> None:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if text and text not in values:
                values.append(text)

    for key in ("code", "msg", "message"):
        append(payload.get(key))
    error = payload.get("error")
    if isinstance(error, Mapping):
        for key in ("code", "msg", "message"):
            append(error.get(key))
    else:
        append(error)
    return sanitize(" | ".join(values), secrets=(secret,))


def _provider_failure_diagnostic(
    payload: Mapping[str, Any], *, status: str, secret: str
) -> dict[str, str]:
    """Keep only bounded provider-owned fields needed to investigate a failure."""

    diagnostic = {
        "status": status,
        "detail": _safe_provider_field(
            _provider_error_detail(payload, secret=secret), limit=300, secret=secret
        ),
    }
    request_id = payload.get("request_id")
    if isinstance(request_id, (str, int)) and not isinstance(request_id, bool):
        safe_request_id = _safe_provider_field(request_id, limit=128, secret=secret)
        if safe_request_id:
            diagnostic["request_id"] = safe_request_id
    return diagnostic


def _safe_provider_field(value: Any, *, limit: int, secret: str) -> str:
    return " ".join(
        sanitize(str(value).strip(), limit=limit, secrets=(secret,)).splitlines()
    )


def _run_automatic_retry(
    timeouts: Timeouts,
    operation: Callable[[], _T],
    *,
    step: str,
    deadline: float | None = None,
) -> _T:
    policy = RetryPolicy(timeouts.retry_count, timeouts.retry_interval_s)

    def retryable(exc: Exception) -> bool:
        if not isinstance(exc, _AutomaticRetryH3Error):
            return False
        return deadline is None or time.monotonic() + policy.interval_s < deadline

    def report(retry_number: int, _exc: Exception) -> None:
        log.warning(
            "%s failed; retry %d/%d in %.1fs",
            step,
            retry_number,
            policy.retries,
            policy.interval_s,
        )

    return run_with_retry(
        operation,
        policy=policy,
        is_retryable=retryable,
        on_retry=report,
    )


def freeze_keyframes(paths: Sequence[Path]) -> FrozenKeyframes:
    """Read the ordered image batch once; all later stages reuse these bytes."""
    try:
        frozen = tuple((Path(path), Path(path).read_bytes()) for path in paths)
    except OSError:
        raise H3Error("keyframe_read_failed") from None
    return frozen


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def voice_texts_receipt(voice_texts: Sequence[str]) -> str:
    return canonical_json_sha256(list(voice_texts))


def start(request: H3Request, *, client: httpx.Client | None = None) -> H3Result:
    """Start or idempotently advance one client request.

    A repeated ``client_request_id`` resolves to its existing attempt. It may
    query an already persisted task, but never repeats a provider POST whose
    outcome is unknown.
    """
    with _session_lease(request):
        if _has_output(request):
            return _output_result(request, None)
        existing = _find_attempt(request, request.client_request_id)
        if existing is None:
            state = _create_attempt(request, request.client_request_id)
            is_new = True
        else:
            state = existing
            is_new = False
        with _client(client) as active_client:
            return _advance(
                request,
                state,
                active_client,
                allow_submit=True,
                new_attempt=is_new,
            )

def inspect(request: H3Request) -> H3Result:
    """Read the latest attempt for UI/startup decisions, without any writes."""
    root = _state_root(request)
    marker = root / "session.json"
    if not root.exists():
        if _has_output(request):
            return _output_result(request, None)
        return H3Result(status="not_started", attempt_id=None)

    if _has_output(request):
        return _output_result(request, None)
    latest = _latest_attempt(request)
    if marker.is_file():
        expected = {"schema_version": SCHEMA_VERSION, "cid": request.cid}
        if _read_json(marker) != expected:
            raise ReceiptError("session_cid_mismatch")
    elif latest is not None:
        raise ReceiptError("state_invalid")

    if latest is not None:
        _validate_state(request, latest, require_client_request_id=False)
    if latest is None:
        return H3Result(status="not_started", attempt_id=None)
    return _result(latest)


def resume(request: H3Request, *, client: httpx.Client | None = None) -> H3Result:
    """Recover one attempt using GET only; intended for startup scanners."""
    with _session_lease(request):
        state = _find_attempt(request, request.client_request_id)
        if _has_output(request):
            return _output_result(request, state)
        if state is None:
            return H3Result(status="not_started", attempt_id=None)
        with _client(client) as active_client:
            return _advance(
                request,
                state,
                active_client,
                allow_submit=False,
                new_attempt=False,
            )


def retry(
    request: H3Request,
    client_request_id: str,
    *,
    client: httpx.Client | None = None,
) -> H3Result:
    """Explicitly create a paid retry, keyed by a new idempotency key."""
    retried = replace(request, client_request_id=client_request_id)
    with _session_lease(retried):
        existing = _find_attempt(retried, client_request_id)
        if _has_output(retried):
            return _output_result(retried, existing)
        is_new = existing is None
        state = _create_attempt(retried, client_request_id) if is_new else existing
        with _client(client) as active_client:
            return _advance(
                retried,
                state,
                active_client,
                allow_submit=True,
                new_attempt=is_new,
            )


def _state_root(request: H3Request) -> Path:
    return request.workdir / ".h3"


def _attempt_path(request: H3Request, attempt_id: str) -> Path:
    return _state_root(request) / "attempts" / attempt_id / "attempt.json"


def _has_output(request: H3Request) -> bool:
    output = request.workdir / "generated.mp4"
    try:
        return output.is_file() and output.stat().st_size > 0
    except OSError:
        return False


def _output_result(request: H3Request, state: Mapping[str, Any] | None) -> H3Result:
    attempt_id = str(state["attempt_id"]) if state is not None else None
    return H3Result(
        status="succeeded",
        attempt_id=attempt_id,
        output=request.workdir / "generated.mp4",
    )


@contextmanager
def _client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    if client is not None:
        yield client
        return
    with httpx.Client(trust_env=False) as owned:
        yield owned


@contextmanager
def _session_lease(request: H3Request) -> Iterator[None]:
    root = _state_root(request)
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / "session.lock", os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        raise H3Error("state_unavailable") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise H3BusyError("session_busy") from None
        _ensure_session_marker(request)
        yield
    finally:
        os.close(fd)


def _ensure_session_marker(request: H3Request) -> None:
    marker = _state_root(request) / "session.json"
    payload = {"schema_version": SCHEMA_VERSION, "cid": request.cid}
    try:
        _atomic_create_json(marker, payload)
        return
    except FileExistsError:
        pass
    except OSError:
        raise H3Error("state_unavailable") from None
    existing = _read_json(marker)
    if existing != payload:
        raise ReceiptError("session_cid_mismatch")


def _input_manifest(request: H3Request) -> dict[str, Any]:
    if request.mode == "reference":
        provider_request = {
            "h3_workflow": H3_WORKFLOW,
            "duration": request.duration,
            "resolution": H3_RESOLUTION,
        }
        if request.seed is not None:
            provider_request["seed"] = request.seed
        return {
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "keyframes": [
                {"name": path.name, "sha256": hashlib.sha256(blob).hexdigest()}
                for path, blob in request.keyframes
            ],
            "voice_texts_sha256": request.voice_receipt,
            "request": provider_request,
        }
    provider_request = {
        "mode": request.mode,
        "h3_workflow": _workflow(request),
        "duration": request.duration,
        "resolution": H3_RESOLUTION,
    }
    return {
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "images": _image_manifest(request),
        "voice_texts_sha256": request.voice_receipt,
        "request": provider_request,
    }


def _workflow(request: H3Request) -> str:
    return H3_WORKFLOW if request.mode == "reference" else H3_BOUNDARY_WORKFLOW


def _image_inputs(request: H3Request) -> tuple[tuple[str, FrozenFrame], ...]:
    if request.mode == "reference":
        return tuple(
            (f"ref_image_{index}", frame)
            for index, frame in enumerate(request.keyframes)
        )
    assert request.first_frame is not None and request.last_frame is not None
    return (
        ("first_frame", request.first_frame),
        ("last_frame", request.last_frame),
    )


def _image_manifest(request: H3Request) -> list[dict[str, str]]:
    return [
        {
            "role": role,
            "name": path.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        for role, (path, blob) in _image_inputs(request)
    ]


def _new_state(request: H3Request, attempt_id: str, client_request_id: str) -> dict[str, Any]:
    manifest = _input_manifest(request)
    return {
        "schema_version": SCHEMA_VERSION,
        "cid": request.cid,
        "attempt_id": attempt_id,
        "client_request_id": client_request_id,
        "input": manifest,
        "input_receipt": canonical_json_sha256(manifest),
        "status": "h3_submitting",
        "retryable": False,
        "h3": {"status": "submitting"},
    }


def _create_attempt(request: H3Request, client_request_id: str) -> dict[str, Any]:
    attempts = _state_root(request) / "attempts"
    try:
        attempts.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(path.name)
            for path in attempts.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        attempt_id = f"{max(numbers, default=0) + 1:06d}"
        directory = attempts / attempt_id
        directory.mkdir()
        state = _new_state(request, attempt_id, client_request_id)
        _atomic_create_json(directory / "attempt.json", state)
        return state
    except (OSError, ValueError):
        raise H3Error("attempt_claim_failed") from None


def _find_attempt(request: H3Request, client_request_id: str) -> dict[str, Any] | None:
    attempts = _state_root(request) / "attempts"
    if not attempts.is_dir():
        return None
    try:
        paths = sorted(attempts.glob("*/attempt.json"), reverse=True)
    except OSError:
        raise H3Error("state_unavailable") from None
    for path in paths:
        raw = _read_json(path)
        if raw.get("client_request_id") == client_request_id:
            _validate_state(request, raw)
            return raw
    return None


def _latest_attempt(request: H3Request) -> dict[str, Any] | None:
    attempts = _state_root(request) / "attempts"
    if not attempts.is_dir():
        return None
    try:
        path = next(iter(sorted(attempts.glob("*/attempt.json"), reverse=True)), None)
    except OSError:
        raise H3Error("state_unavailable") from None
    return _read_json(path) if path is not None else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReceiptError("state_invalid") from None
    if not isinstance(value, dict):
        raise ReceiptError("state_invalid")
    return value


def _validate_state(
    request: H3Request,
    state: Mapping[str, Any],
    *,
    require_client_request_id: bool = True,
) -> None:
    manifest = _input_manifest(request)
    attempt_id = state.get("attempt_id")
    stored_request_id = state.get("client_request_id")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("cid") != request.cid
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
        or not isinstance(stored_request_id, str)
        or not stored_request_id.strip()
        or (
            require_client_request_id
            and stored_request_id != request.client_request_id
        )
        or state.get("input") != manifest
        or state.get("input_receipt") != canonical_json_sha256(manifest)
    ):
        raise ReceiptError("receipt_mismatch")
    h3_state = state.get("h3")
    if not isinstance(h3_state, dict):
        raise ReceiptError("state_invalid")
    error = state.get("error")
    if error is not None and (
        not isinstance(error, dict) or error.get("code") not in _SAFE_ERROR_CODES
    ):
        raise ReceiptError("state_invalid")
    if isinstance(error, dict) and "provider" in error:
        provider = error["provider"]
        if (
            error.get("code") != "h3_provider_failed"
            or set(error) != {"code", "provider"}
            or not isinstance(provider, dict)
            or not {"status", "detail"}.issubset(provider)
            or not set(provider).issubset({"status", "detail", "request_id"})
            or provider.get("status") not in {"FAILED", "ERROR", "FAIL"}
            or not isinstance(provider.get("detail"), str)
            or len(provider["detail"]) > 300
            or provider["detail"]
            != _safe_provider_field(
                provider["detail"], limit=300, secret=request.autodl_token
            )
            or (
                "request_id" in provider
                and (
                    not isinstance(provider["request_id"], str)
                    or not provider["request_id"]
                    or len(provider["request_id"]) > 128
                    or provider["request_id"]
                    != _safe_provider_field(
                        provider["request_id"],
                        limit=128,
                        secret=request.autodl_token,
                    )
                )
            )
        ):
            raise ReceiptError("state_invalid")
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    if h3_task_id is not None:
        if h3_state.get("receipt") != _h3_receipt(request, h3_task_id):
            raise ReceiptError("receipt_mismatch")
    if "result_url" in h3_state:
        raise ReceiptError("state_invalid")
    output_receipt = h3_state.get("output")
    if output_receipt is not None and (
        not isinstance(output_receipt, dict)
        or set(output_receipt) != {"name", "sha256", "size"}
        or output_receipt.get("name") != "generated.mp4"
        or not isinstance(output_receipt.get("sha256"), str)
        or len(output_receipt["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in output_receipt["sha256"])
        or not isinstance(output_receipt.get("size"), int)
        or isinstance(output_receipt.get("size"), bool)
        or output_receipt["size"] <= 0
    ):
        raise ReceiptError("state_invalid")


def _task_id(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReceiptError("state_invalid")
    normalized = str(value).strip()
    if not normalized:
        raise ReceiptError("state_invalid")
    return normalized


def _h3_receipt(request: H3Request, task_id: str) -> dict[str, Any]:
    if request.mode == "reference":
        provider_request = {
            "workflow": H3_WORKFLOW,
            "duration": request.duration,
            "resolution": H3_RESOLUTION,
        }
        if request.seed is not None:
            provider_request["seed"] = request.seed
        return {
            "task_id": task_id,
            "input_receipt": canonical_json_sha256(_input_manifest(request)),
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "keyframes": _input_manifest(request)["keyframes"],
            "request": provider_request,
        }
    provider_request = {
        "mode": request.mode,
        "workflow": _workflow(request),
        "duration": request.duration,
        "resolution": H3_RESOLUTION,
    }
    return {
        "task_id": task_id,
        "input_receipt": canonical_json_sha256(_input_manifest(request)),
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "images": _image_manifest(request),
        "request": provider_request,
    }


def _advance(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    *,
    allow_submit: bool,
    new_attempt: bool,
) -> H3Result:
    _validate_state(request, state)
    if _has_output(request):
        return _output_result(request, state)
    status = str(state.get("status") or "")
    if status == "succeeded":
        raise ReceiptError("output_missing")
    if status == "submission_unknown":
        return _result(state)
    if status == "failed":
        return _result(state)

    h3_state = state["h3"]
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    if h3_task_id is None:
        if h3_state.get("status") == "submitting":
            if new_attempt and allow_submit:
                h3_task_id = _submit_h3(request, state, client)
                return _poll_h3(request, state, client, h3_task_id)
            state["status"] = "submission_unknown"
            state["retryable"] = False
            h3_state["status"] = "submission_unknown"
            _save_state(request, state)
            return _result(state)
        raise ReceiptError("state_invalid")

    return _poll_h3(request, state, client, h3_task_id)


def _query_json_with_retry(
    request: H3Request,
    operation: Callable[[], httpx.Response],
    *,
    code: str,
    step: str,
    deadline: float,
) -> tuple[httpx.Response, dict[str, Any]]:
    """Retry only same-task GET failures; provider POSTs never enter here."""

    def attempt() -> tuple[httpx.Response, dict[str, Any]]:
        try:
            response = operation()
        except httpx.HTTPError:
            raise _AutomaticRetryH3Error(code, retryable=True) from None
        if response.status_code != 200:
            error_type = (
                _AutomaticRetryH3Error
                if _retryable_http_status(response.status_code)
                else H3Error
            )
            raise error_type(code, retryable=True)
        try:
            payload = _response_json(response)
        except (ValueError, TypeError):
            raise _AutomaticRetryH3Error(code, retryable=True) from None
        return response, payload

    return _run_automatic_retry(
        request.timeouts,
        attempt,
        step=step,
        deadline=deadline,
    )


def _submit_h3(request: H3Request, state: dict[str, Any], client: httpx.Client) -> str:
    state["h3"] = {"status": "submitting"}
    state["status"] = "h3_submitting"
    state["retryable"] = False
    _save_state(request, state)
    body: dict[str, Any] = {
        "prompt": request.prompt,
        "duration": request.duration,
        "resolution": H3_RESOLUTION,
    }
    if request.seed is not None:
        body["seed"] = request.seed
    for role, (_path, blob) in _image_inputs(request):
        body[role] = (
            "data:image/png;base64," + base64.b64encode(blob).decode("ascii")
        )
    try:
        response = client.post(
            f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/{_workflow(request)}",
            headers={"Authorization": request.autodl_token},
            json=body,
            timeout=request.timeouts.request_s,
        )
        payload = _response_json(response)
    except (httpx.HTTPError, ValueError, TypeError):
        _submission_unknown(request, state, "h3")
        raise H3Error("submission_unknown") from None
    data = payload.get("data")
    task_value = data.get("task_id") if isinstance(data, dict) else None
    if response.status_code != 200 or isinstance(task_value, bool) or not isinstance(
        task_value, (str, int)
    ) or not str(task_value).strip():
        detail = _provider_error_detail(payload, secret=request.autodl_token)
        log.warning(
            "H3 submission rejected cid=%s attempt=%s http_status=%d detail=%s",
            request.cid,
            state.get("attempt_id"),
            response.status_code,
            detail or "no_safe_detail",
        )
        _fail(request, state, "h3_submit_rejected", retryable=False)
        raise H3Error("h3_submit_rejected")
    task_id = str(task_value).strip()
    state["h3"] = {
        "status": "running",
        "task_id": task_id,
        "receipt": _h3_receipt(request, task_id),
    }
    state["status"] = "h3_running"
    state["retryable"] = False
    _save_state(request, state)
    return task_id


def _poll_h3(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    task_id: str,
) -> H3Result:
    if state["h3"].get("receipt") != _h3_receipt(request, task_id):
        raise ReceiptError("receipt_mismatch")
    deadline = time.monotonic() + request.timeouts.h3_poll_s
    headers = {"Authorization": request.autodl_token}
    while True:
        try:
            _, payload = _query_json_with_retry(
                request,
                lambda: client.get(
                    f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/result/{task_id}",
                    headers=headers,
                    timeout=request.timeouts.request_s,
                ),
                code="h3_query_failed",
                step="H3 result query",
                deadline=deadline,
            )
        except H3Error:
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True) from None
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        provider_status = str(data.get("status") or "").upper()
        if provider_status in {"SUCCESS", "COMPLETED"}:
            results = data.get("results")
            url = next(
                (
                    item.get("url")
                    for item in results
                    if isinstance(item, dict) and isinstance(item.get("url"), str)
                ),
                None,
            ) if isinstance(results, list) else None
            if not url:
                _fail(request, state, "h3_result_missing", retryable=False, keep_task=True)
                raise H3Error("h3_result_missing")
            output_receipt = _download(request, state, client, url)
            state["h3"]["status"] = "succeeded"
            state["h3"]["output"] = output_receipt
            state["status"] = "succeeded"
            state["retryable"] = False
            _save_state(request, state)
            return _result(state, output=request.workdir / "generated.mp4")
        if provider_status in {"FAILED", "ERROR", "FAIL"}:
            diagnostic = _provider_failure_diagnostic(
                payload, status=provider_status, secret=request.autodl_token
            )
            log.warning(
                "H3 provider failed cid=%s attempt=%s request_id=%s detail=%s",
                request.cid,
                state.get("attempt_id"),
                diagnostic.get("request_id") or "unavailable",
                diagnostic.get("detail") or "no_safe_detail",
            )
            state["h3"]["status"] = "failed"
            _fail(
                request,
                state,
                "h3_provider_failed",
                retryable=False,
                keep_task=True,
                provider_diagnostic=diagnostic,
            )
            return _result(state)
        if time.monotonic() >= deadline:
            _fail(request, state, "h3_timeout", retryable=True, keep_task=True)
            return _result(state)
        _pause(request.timeouts.poll_interval_s)


def _download(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    url: str,
) -> dict[str, Any]:
    try:
        receipt = _run_automatic_retry(
            request.timeouts,
            lambda: _download_once(request, client, url),
            step="H3 result download",
        )
    except H3Error as exc:
        _fail(request, state, exc.code, retryable=exc.retryable, keep_task=True)
        raise
    # Earlier failed attempts persist a recoverable state.  A later success in
    # the same process must remove that stale error before the caller commits
    # the final succeeded state.
    state.pop("error", None)
    state["status"] = "h3_running"
    state["retryable"] = False
    state["h3"]["status"] = "running"
    return receipt


def _download_once(
    request: H3Request,
    client: httpx.Client,
    url: str,
) -> dict[str, Any]:
    try:
        public_url = _is_public_https_url(url)
    except _DNSLookupFailed:
        _raise_download_error("download_dns_failed", retryable=True)
    if not public_url:
        _raise_download_error("download_url_rejected", retryable=False)

    destination = request.workdir / "generated.mp4"
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    size = 0
    digest = hashlib.sha256()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with client.stream(
                "GET",
                url,
                timeout=request.timeouts.download_s,
                follow_redirects=False,
            ) as response:
                public_peer = _response_has_public_peer(response)
                if public_peer is None:
                    _raise_download_error(
                        "download_peer_unverified",
                        retryable=True,
                    )
                if not public_peer:
                    _raise_download_error(
                        "download_url_rejected",
                        retryable=False,
                    )
                if 300 <= response.status_code < 400:
                    _raise_download_error(
                        "download_redirect_rejected",
                        retryable=False,
                    )
                if response.status_code != 200:
                    _raise_download_error(
                        "download_failed",
                        retryable=True,
                        automatic_retryable=_retryable_http_status(response.status_code),
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        _raise_download_error("download_failed", retryable=True)
                    if declared_size < 0:
                        _raise_download_error("download_failed", retryable=True)
                    if declared_size > MAX_VIDEO_BYTES:
                        _raise_download_error("download_too_large", retryable=False)
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        _raise_download_error("download_too_large", retryable=False)
                    _write_all(fd, chunk)
                    digest.update(chunk)
        except httpx.HTTPError:
            _raise_download_error("download_failed", retryable=True)
        if size <= 0:
            _raise_download_error("download_failed", retryable=True)
        os.fsync(fd)
        os.close(fd)
        fd = None

        def probe_attempt() -> bool:
            try:
                return _probe_video(temporary, request.timeouts.probe_s)
            except _ProbeUnavailable:
                raise _AutomaticRetryH3Error(
                    "output_probe_failed", retryable=True
                ) from None

        try:
            valid_video = _run_automatic_retry(
                request.timeouts,
                probe_attempt,
                step="downloaded video probe",
            )
        except _AutomaticRetryH3Error:
            _raise_download_error(
                "output_probe_failed",
                retryable=True,
                automatic_retryable=False,
            )
        if not valid_video:
            _raise_download_error("download_invalid_video", retryable=False)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except OSError:
        raise _AutomaticRetryH3Error("output_write_failed", retryable=True) from None
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return {"name": "generated.mp4", "sha256": digest.hexdigest(), "size": size}


def _raise_download_error(
    code: str,
    *,
    retryable: bool,
    automatic_retryable: bool | None = None,
) -> None:
    automatic = retryable if automatic_retryable is None else automatic_retryable
    error_type = _AutomaticRetryH3Error if automatic else H3Error
    raise error_type(code, retryable=retryable)


def _is_public_https_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        port = parsed.port or 443
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except UnicodeError:
            return False
        except OSError:
            raise _DNSLookupFailed from None
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except (IndexError, TypeError, ValueError):
                return False
    else:
        addresses = [literal]
    return bool(addresses) and all(
        address.is_global and not address.is_multicast for address in addresses
    )


def _response_has_public_peer(response: httpx.Response) -> bool | None:
    network_stream = response.extensions.get("network_stream")
    get_extra_info = getattr(network_stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return None
    try:
        peer = get_extra_info("server_addr")
    except Exception:
        return None
    host = peer[0] if isinstance(peer, (tuple, list)) and peer else peer
    if not isinstance(host, str):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    return address.is_global and not address.is_multicast


def _probe_video(path: Path, timeout_s: float) -> bool:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,duration,duration_ts,time_base",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise _ProbeUnavailable from None
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams")
    except (
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return False
    if not isinstance(streams, list) or not streams:
        return False
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        return False
    try:
        duration = float(stream.get("duration"))
        if math.isfinite(duration) and duration > 0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        ticks = float(stream.get("duration_ts"))
        numerator, denominator = str(stream.get("time_base")).split("/", 1)
        duration = ticks * float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return math.isfinite(duration) and duration > 0


def _submission_unknown(request: H3Request, state: dict[str, Any], stage: str) -> None:
    state["status"] = "submission_unknown"
    state["retryable"] = False
    state[stage]["status"] = "submission_unknown"
    _save_state(request, state)


def _fail(
    request: H3Request,
    state: dict[str, Any],
    code: str,
    *,
    retryable: bool,
    keep_task: bool = False,
    provider_diagnostic: Mapping[str, str] | None = None,
) -> None:
    state["status"] = "retryable_failure" if retryable else "failed"
    state["retryable"] = retryable
    state["error"] = {"code": code}
    if provider_diagnostic is not None:
        state["error"]["provider"] = dict(provider_diagnostic)
    if not keep_task:
        if state["h3"].get("status") == "submitting":
            state["h3"]["status"] = "failed"
    _save_state(request, state)


def _result(
    state: Mapping[str, Any], *, output: Path | None = None
) -> H3Result:
    error = state.get("error")
    error_code = error.get("code") if isinstance(error, dict) else None
    if not isinstance(error_code, str):
        error_code = None
    return H3Result(
        status=str(state.get("status") or "failed"),
        attempt_id=str(state.get("attempt_id")),
        output=output,
        retryable=bool(state.get("retryable")),
        error_code=error_code,
    )


def _response_json(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("non-object response")
    return payload


def _pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _save_state(request: H3Request, state: Mapping[str, Any]) -> None:
    try:
        _atomic_write_json(_attempt_path(request, str(state["attempt_id"])), state)
    except OSError:
        raise H3Error("state_persist_failed") from None


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_bytes(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
