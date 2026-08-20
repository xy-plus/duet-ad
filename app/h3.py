"""Recoverable Context IR -> H3 generation primitives.

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
import math
import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import httpx


SCHEMA_VERSION = 1
IR_MODEL = "MiniMax-H3"
H3_WORKFLOW = "minimax_h3_lightx2v_v5"
H3_RESOLUTION = "768p竖"
MINIMAX_BASE_URL = "https://api.minimaxi.com"
AUTODL_BASE_URL = "https://autodl.art"
MAX_VIDEO_BYTES = 200 * 1024 * 1024

_MAX_CONTEXT_IR_PROMPT_BYTES = 32 * 1024

_SAFE_ERROR_CODES = {
    "ir_upload_failed",
    "ir_upload_rejected",
    "ir_submit_rejected",
    "ir_query_failed",
    "ir_prompt_missing",
    "ir_dialogue_mismatch",
    "ir_provider_failed",
    "ir_timeout",
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
_CONTEXT_IR_STATUSES = frozenset(
    {"submitting", "running", "succeeded", "failed", "submission_unknown"}
)

FrozenKeyframes = tuple[tuple[Path, bytes], ...]
FrozenVoiceTexts = tuple[str, ...]


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


class _DNSLookupFailed(Exception):
    pass


class _ProbeUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Timeouts:
    request_s: float = 30.0
    upload_s: float = 60.0
    ir_poll_s: float = 900.0
    h3_poll_s: float = 1500.0
    download_s: float = 180.0
    poll_interval_s: float = 3.0
    probe_s: float = 30.0

    def __post_init__(self) -> None:
        positive = (
            self.request_s,
            self.upload_s,
            self.ir_poll_s,
            self.h3_poll_s,
            self.download_s,
            self.probe_s,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise H3Error("invalid_timeout")
        if isinstance(self.poll_interval_s, bool) or self.poll_interval_s < 0:
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
    ratio: str
    minimax_api_key: str
    autodl_token: str
    timeouts: Timeouts = Timeouts()

    def __post_init__(self) -> None:
        object.__setattr__(self, "workdir", Path(self.workdir))
        if not isinstance(self.cid, str) or not self.cid.strip():
            raise H3Error("invalid_cid")
        if not isinstance(self.client_request_id, str) or not self.client_request_id.strip():
            raise H3Error("invalid_client_request_id")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise H3Error("invalid_prompt")
        if (
            not isinstance(self.duration, int)
            or isinstance(self.duration, bool)
            or not 1 <= self.duration <= 15
        ):
            raise H3Error("invalid_duration")
        if not isinstance(self.ratio, str) or not self.ratio.strip():
            raise H3Error("invalid_ratio")
        if not isinstance(self.minimax_api_key, str) or not self.minimax_api_key.strip():
            raise H3Error("missing_minimax_credential")
        if not isinstance(self.autodl_token, str) or not self.autodl_token.strip():
            raise H3Error("missing_autodl_credential")
        if not isinstance(self.keyframes, tuple) or not 1 <= len(self.keyframes) <= 9:
            raise H3Error("invalid_keyframes")
        names: list[str] = []
        for item in self.keyframes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise H3Error("invalid_keyframes")
            path, blob = item
            if not isinstance(path, Path) or not isinstance(blob, bytes) or not blob:
                raise H3Error("invalid_keyframes")
            if not path.name or path.name != Path(path.name).name:
                raise H3Error("invalid_keyframes")
            names.append(path.name)
        if len(names) != len(set(names)):
            raise H3Error("duplicate_keyframe_name")
        if not isinstance(self.voice_texts, tuple):
            raise H3Error("invalid_voice_texts")
        if any(not isinstance(text, str) or not text for text in self.voice_texts):
            raise H3Error("invalid_voice_texts")
        if self.voice_receipt != voice_texts_receipt(self.voice_texts):
            raise ReceiptError("voice_receipt_mismatch")


@dataclass(frozen=True)
class H3Result:
    status: str
    attempt_id: str | None
    output: Path | None = None
    retryable: bool = False
    error_code: str | None = None


@dataclass(frozen=True)
class ContextIRSnapshot:
    """Validated local Context IR observation; never contains provider metadata."""

    status: str
    prompt: str | None
    sha256: str | None


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

    A repeated ``client_request_id`` resolves to its existing attempt.  It may
    query an already persisted task or submit H3 from ``ready_for_h3``; it never
    repeats a provider POST whose outcome is unknown.
    """
    with _session_lease(request):
        existing = _find_attempt(request, request.client_request_id)
        if _has_output(request):
            return _output_result(request, existing)
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


def prepare_context_ir(
    request: H3Request, *, client: httpx.Client | None = None
) -> H3Result:
    """Create/query Context IR and stop before the paid H3 video submission."""
    with _session_lease(request):
        existing = _find_attempt(request, request.client_request_id)
        if _has_output(request):
            return _output_result(request, existing)
        state = (
            _create_attempt(request, request.client_request_id)
            if existing is None
            else existing
        )
        with _client(client) as active_client:
            return _advance(
                request,
                state,
                active_client,
                allow_submit=True,
                new_attempt=existing is None,
                stop_after_ir=True,
            )


def inspect(request: H3Request) -> H3Result:
    """Read the latest attempt for UI/startup decisions, without any writes."""
    root = _state_root(request)
    marker = root / "session.json"
    if not root.exists():
        if _has_output(request):
            return _output_result(request, None)
        return H3Result(status="not_started", attempt_id=None)

    latest = _latest_attempt(request)
    if marker.is_file():
        expected = {"schema_version": SCHEMA_VERSION, "cid": request.cid}
        if _read_json(marker) != expected:
            raise ReceiptError("session_cid_mismatch")
    elif latest is not None:
        raise ReceiptError("state_invalid")

    if latest is not None:
        _validate_state(request, latest, require_client_request_id=False)
    if _has_output(request):
        return _output_result(request, latest)
    if latest is None:
        return H3Result(status="not_started", attempt_id=None)
    return _result(latest)


def inspect_context_ir(workdir: Path, cid: str) -> ContextIRSnapshot:
    """Read only the latest Context IR status and verified prompt from disk."""
    root = Path(workdir) / ".h3"
    if not root.exists():
        return ContextIRSnapshot("not_started", None, None)

    attempts = root / "attempts"
    try:
        latest_path = (
            next(iter(sorted(attempts.glob("*/attempt.json"), reverse=True)), None)
            if attempts.is_dir()
            else None
        )
    except OSError:
        raise ReceiptError("state_invalid") from None
    latest = _read_json(latest_path) if latest_path is not None else None

    marker = root / "session.json"
    if marker.is_file():
        expected = {"schema_version": SCHEMA_VERSION, "cid": cid}
        if _read_json(marker) != expected:
            raise ReceiptError("session_cid_mismatch")
    elif latest is not None:
        raise ReceiptError("state_invalid")
    if latest is None:
        return ContextIRSnapshot("not_started", None, None)

    attempt_id = latest.get("attempt_id")
    client_request_id = latest.get("client_request_id")
    manifest = latest.get("input")
    ir = latest.get("ir")
    if (
        latest.get("schema_version") != SCHEMA_VERSION
        or latest.get("cid") != cid
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
        or not isinstance(client_request_id, str)
        or not client_request_id.strip()
        or not isinstance(manifest, dict)
        or latest.get("input_receipt") != canonical_json_sha256(manifest)
        or not isinstance(ir, dict)
    ):
        raise ReceiptError("state_invalid")

    ir_status = ir.get("status")
    if ir_status not in _CONTEXT_IR_STATUSES:
        raise ReceiptError("state_invalid")
    prompt = ir.get("optimized_prompt")
    digest = ir.get("optimized_prompt_sha256")
    if ir_status == "succeeded":
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(digest, str)
            or digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ):
            raise ReceiptError("context_ir_mismatch")
        return ContextIRSnapshot("succeeded", prompt, digest)
    if prompt is not None or digest is not None:
        raise ReceiptError("context_ir_mismatch")
    public_status = (
        "failed"
        if latest.get("status") in {"failed", "retryable_failure"}
        else str(ir_status)
    )
    return ContextIRSnapshot(public_status, None, None)


def edit_context_ir(
    request: H3Request,
    expected_sha256: str,
    prompt: str,
) -> ContextIRSnapshot:
    """Atomically replace the reviewed IR prompt before H3 has been submitted."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise H3Error("invalid_context_ir_prompt")
    encoded = prompt.encode("utf-8")
    if len(encoded) > _MAX_CONTEXT_IR_PROMPT_BYTES:
        raise H3Error("invalid_context_ir_prompt")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise H3Error("context_ir_version_conflict")

    with _session_lease(request):
        state = _find_attempt(request, request.client_request_id)
        if state is None:
            raise H3Error("context_ir_not_ready")
        ir = state.get("ir")
        h3_state = state.get("h3")
        if (
            state.get("status") != "ready_for_h3"
            or not isinstance(ir, dict)
            or ir.get("status") != "succeeded"
            or not isinstance(h3_state, dict)
            or h3_state.get("status") not in {"ready", "not_started"}
            or _task_id(h3_state.get("task_id"), required=False) is not None
        ):
            raise H3Error("context_ir_not_editable")
        current = ir.get("optimized_prompt")
        current_sha = ir.get("optimized_prompt_sha256")
        if (
            not isinstance(current, str)
            or not isinstance(current_sha, str)
            or current_sha != hashlib.sha256(current.encode("utf-8")).hexdigest()
        ):
            raise ReceiptError("context_ir_mismatch")
        if current_sha != expected_sha256:
            raise H3Error("context_ir_version_conflict")
        digest = hashlib.sha256(encoded).hexdigest()
        ir["optimized_prompt"] = prompt
        ir["optimized_prompt_sha256"] = digest
        state["h3"] = {"status": "ready"}
        _save_state(request, state)
        return ContextIRSnapshot("succeeded", prompt, digest)


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
    return {
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "keyframes": [
            {"name": path.name, "sha256": hashlib.sha256(blob).hexdigest()}
            for path, blob in request.keyframes
        ],
        "voice_texts_sha256": request.voice_receipt,
        "request": {
            "ir_model": IR_MODEL,
            "h3_workflow": H3_WORKFLOW,
            "duration": request.duration,
            "ratio": request.ratio,
            "resolution": H3_RESOLUTION,
        },
    }


def _new_state(request: H3Request, attempt_id: str, client_request_id: str) -> dict[str, Any]:
    manifest = _input_manifest(request)
    return {
        "schema_version": SCHEMA_VERSION,
        "cid": request.cid,
        "attempt_id": attempt_id,
        "client_request_id": client_request_id,
        "input": manifest,
        "input_receipt": canonical_json_sha256(manifest),
        "status": "ir_submitting",
        "retryable": False,
        "ir": {"status": "submitting"},
        "h3": {"status": "not_started"},
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
    ir = state.get("ir")
    h3_state = state.get("h3")
    if not isinstance(ir, dict) or not isinstance(h3_state, dict):
        raise ReceiptError("state_invalid")
    error = state.get("error")
    if error is not None and (
        not isinstance(error, dict) or error.get("code") not in _SAFE_ERROR_CODES
    ):
        raise ReceiptError("state_invalid")
    ir_task_id = _task_id(ir.get("task_id"), required=False)
    if ir_task_id is not None and ir.get("receipt") != _ir_receipt(request, ir_task_id):
        raise ReceiptError("receipt_mismatch")
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    optimized = ir.get("optimized_prompt")
    if optimized is not None:
        if not isinstance(optimized, str):
            raise ReceiptError("receipt_mismatch")
        if ir.get("optimized_prompt_sha256") != hashlib.sha256(
            optimized.encode("utf-8")
        ).hexdigest():
            raise ReceiptError("receipt_mismatch")
    if h3_task_id is not None:
        if not isinstance(optimized, str):
            raise ReceiptError("receipt_mismatch")
        if h3_state.get("receipt") != _h3_receipt(request, h3_task_id, optimized):
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


def _ir_receipt(request: H3Request, task_id: str) -> dict[str, Any]:
    manifest = _input_manifest(request)
    return {
        "task_id": task_id,
        "input_receipt": canonical_json_sha256(manifest),
        "prompt_sha256": manifest["prompt_sha256"],
        "keyframes": manifest["keyframes"],
        "voice_texts_sha256": manifest["voice_texts_sha256"],
        "request": {
            "model": IR_MODEL,
            "duration": request.duration,
            "ratio": request.ratio,
        },
    }


def _h3_receipt(request: H3Request, task_id: str, prompt: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "input_receipt": canonical_json_sha256(_input_manifest(request)),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "keyframes": _input_manifest(request)["keyframes"],
        "request": {
            "workflow": H3_WORKFLOW,
            "duration": request.duration,
            "resolution": H3_RESOLUTION,
        },
    }


def _advance(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    *,
    allow_submit: bool,
    new_attempt: bool,
    stop_after_ir: bool = False,
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
        error = state.get("error")
        ir = state.get("ir")
        h3_state = state.get("h3")
        if (
            error == {"code": "ir_dialogue_mismatch"}
            and isinstance(ir, dict)
            and ir.get("status") == "running"
            and _task_id(ir.get("task_id"), required=False) is not None
            and isinstance(h3_state, dict)
            and h3_state.get("status") == "not_started"
        ):
            # A validator-only failure has no paid H3 side effect.  After a
            # validator repair, re-query the already bound IR task instead of
            # creating a second attempt or repeating the IR POST.
            state["status"] = "ir_running"
            state["retryable"] = False
            state.pop("error", None)
            _save_state(request, state)
        else:
            return _result(state)

    ir = state["ir"]
    ir_task_id = _task_id(ir.get("task_id"), required=False)
    if ir_task_id is None:
        if new_attempt and allow_submit and ir.get("status") == "submitting":
            submitted = _submit_ir(request, state, client)
            if isinstance(submitted, H3Result):
                return submitted
            ir_task_id = submitted
        elif ir.get("status") == "submitting":
            state["status"] = "submission_unknown"
            state["retryable"] = False
            ir["status"] = "submission_unknown"
            _save_state(request, state)
            return _result(state)
        else:
            return _result(state)

    if state["ir"].get("status") != "succeeded":
        result = _poll_ir(request, state, client, ir_task_id)
        if result is not None:
            return result

    if stop_after_ir:
        return _result(state)

    h3_state = state["h3"]
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    if h3_task_id is None:
        if h3_state.get("status") == "submitting":
            state["status"] = "submission_unknown"
            state["retryable"] = False
            h3_state["status"] = "submission_unknown"
            _save_state(request, state)
            return _result(state)
        if not allow_submit:
            state["status"] = "ready_for_h3"
            state["retryable"] = False
            h3_state["status"] = "ready"
            _save_state(request, state)
            return _result(state)
        h3_task_id = _submit_h3(request, state, client)

    return _poll_h3(request, state, client, h3_task_id)


def _submit_ir(
    request: H3Request, state: dict[str, Any], client: httpx.Client
) -> str | H3Result:
    headers = {"Authorization": f"Bearer {request.minimax_api_key}"}
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
    for index, (path, blob) in enumerate(request.keyframes):
        try:
            response = client.post(
                f"{MINIMAX_BASE_URL}/v1/files/upload",
                headers=headers,
                files={"file": (path.name, blob, "image/png")},
                data={"purpose": "video_generation_input"},
                timeout=request.timeouts.upload_s,
            )
            payload = _response_json(response)
        except (httpx.HTTPError, ValueError, TypeError):
            _fail(request, state, "ir_upload_failed", retryable=True)
            raise H3Error("ir_upload_failed", retryable=True) from None
        file_data = payload.get("file")
        file_id = file_data.get("file_id") if isinstance(file_data, dict) else None
        if response.status_code != 200 or not isinstance(file_id, (str, int)) or not str(file_id):
            _fail(request, state, "ir_upload_rejected", retryable=False)
            raise H3Error("ir_upload_rejected")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"mm_file://{file_id}"},
                "role": "reference_image",
            }
        )
    body = {
        "model": IR_MODEL,
        "content": content,
        "duration": request.duration,
        "ratio": request.ratio,
    }
    try:
        response = client.post(
            f"{MINIMAX_BASE_URL}/v2/h3_context_ir",
            headers=headers,
            json=body,
            timeout=request.timeouts.request_s,
        )
        payload = _response_json(response)
    except (httpx.HTTPError, ValueError, TypeError):
        _submission_unknown(request, state, "ir")
        raise H3Error("submission_unknown") from None
    task_value = payload.get("task_id")
    if response.status_code != 200 or isinstance(task_value, bool) or not isinstance(
        task_value, (str, int)
    ) or not str(task_value).strip():
        _fail(request, state, "ir_submit_rejected", retryable=False)
        raise H3Error("ir_submit_rejected")
    task_id = str(task_value).strip()
    state["ir"] = {
        "status": "running",
        "task_id": task_id,
        "receipt": _ir_receipt(request, task_id),
    }
    state["status"] = "ir_running"
    state["retryable"] = False
    _save_state(request, state)
    return task_id


def _poll_ir(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    task_id: str,
) -> H3Result | None:
    if state["ir"].get("receipt") != _ir_receipt(request, task_id):
        raise ReceiptError("receipt_mismatch")
    deadline = time.monotonic() + request.timeouts.ir_poll_s
    headers = {"Authorization": f"Bearer {request.minimax_api_key}"}
    while True:
        try:
            response = client.get(
                f"{MINIMAX_BASE_URL}/v2/query/video_generation",
                headers=headers,
                params={"task_id": task_id},
                timeout=request.timeouts.request_s,
            )
            payload = _response_json(response)
        except (httpx.HTTPError, ValueError, TypeError):
            _fail(request, state, "ir_query_failed", retryable=True, keep_task=True)
            raise H3Error("ir_query_failed", retryable=True) from None
        if response.status_code != 200:
            _fail(request, state, "ir_query_failed", retryable=True, keep_task=True)
            raise H3Error("ir_query_failed", retryable=True)
        items = payload.get("items")
        item = next(
            (
                candidate
                for candidate in items if isinstance(candidate, dict)
                and str(candidate.get("id")) == task_id
            ),
            None,
        ) if isinstance(items, list) else None
        if item is not None:
            provider_status = str(item.get("status") or "").lower()
            if provider_status == "succeeded":
                content = item.get("content")
                optimized = (
                    content.get("prompt")
                    if isinstance(content, dict)
                    else content if isinstance(content, str) else None
                )
                if not isinstance(optimized, str) or not optimized:
                    _fail(request, state, "ir_prompt_missing", retryable=False, keep_task=True)
                    raise H3Error("ir_prompt_missing")
                state["ir"]["status"] = "succeeded"
                state["ir"]["optimized_prompt"] = optimized
                state["ir"]["optimized_prompt_sha256"] = hashlib.sha256(
                    optimized.encode("utf-8")
                ).hexdigest()
                state["status"] = "ready_for_h3"
                state["retryable"] = False
                state["h3"] = {"status": "ready"}
                _save_state(request, state)
                return None
            if provider_status not in {"", "queued", "running"}:
                _fail(request, state, "ir_provider_failed", retryable=False, keep_task=True)
                return _result(state)
        if time.monotonic() >= deadline:
            _fail(request, state, "ir_timeout", retryable=True, keep_task=True)
            return _result(state)
        _pause(request.timeouts.poll_interval_s)


def _submit_h3(request: H3Request, state: dict[str, Any], client: httpx.Client) -> str:
    optimized = state["ir"].get("optimized_prompt")
    if not isinstance(optimized, str):
        raise ReceiptError("receipt_mismatch")
    state["h3"] = {"status": "submitting"}
    state["status"] = "h3_submitting"
    state["retryable"] = False
    _save_state(request, state)
    body: dict[str, Any] = {
        "prompt": optimized,
        "duration": request.duration,
        "resolution": H3_RESOLUTION,
    }
    for index, (_path, blob) in enumerate(request.keyframes):
        body[f"ref_image_{index}"] = (
            "data:image/png;base64," + base64.b64encode(blob).decode("ascii")
        )
    try:
        response = client.post(
            f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/{H3_WORKFLOW}",
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
        _fail(request, state, "h3_submit_rejected", retryable=False)
        raise H3Error("h3_submit_rejected")
    task_id = str(task_value).strip()
    state["h3"] = {
        "status": "running",
        "task_id": task_id,
        "receipt": _h3_receipt(request, task_id, optimized),
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
    optimized = state["ir"].get("optimized_prompt")
    if not isinstance(optimized, str) or state["h3"].get("receipt") != _h3_receipt(
        request, task_id, optimized
    ):
        raise ReceiptError("receipt_mismatch")
    deadline = time.monotonic() + request.timeouts.h3_poll_s
    headers = {"Authorization": request.autodl_token}
    while True:
        try:
            response = client.get(
                f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/result/{task_id}",
                headers=headers,
                timeout=request.timeouts.request_s,
            )
            payload = _response_json(response)
        except (httpx.HTTPError, ValueError, TypeError):
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True) from None
        if response.status_code != 200:
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True)
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
            _fail(request, state, "h3_provider_failed", retryable=False, keep_task=True)
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
        public_url = _is_public_https_url(url)
    except _DNSLookupFailed:
        _reject_download(request, state, "download_dns_failed", retryable=True)
    if not public_url:
        _reject_download(request, state, "download_url_rejected", retryable=False)

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
                    _reject_download(
                        request,
                        state,
                        "download_peer_unverified",
                        retryable=True,
                    )
                if not public_peer:
                    _reject_download(
                        request,
                        state,
                        "download_url_rejected",
                        retryable=False,
                    )
                if 300 <= response.status_code < 400:
                    _reject_download(
                        request,
                        state,
                        "download_redirect_rejected",
                        retryable=False,
                    )
                if response.status_code != 200:
                    _reject_download(request, state, "download_failed", retryable=True)
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        _reject_download(
                            request, state, "download_failed", retryable=True
                        )
                    if declared_size < 0:
                        _reject_download(
                            request, state, "download_failed", retryable=True
                        )
                    if declared_size > MAX_VIDEO_BYTES:
                        _reject_download(
                            request, state, "download_too_large", retryable=False
                        )
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        _reject_download(
                            request, state, "download_too_large", retryable=False
                        )
                    _write_all(fd, chunk)
                    digest.update(chunk)
        except httpx.HTTPError:
            _reject_download(request, state, "download_failed", retryable=True)
        if size <= 0:
            _reject_download(request, state, "download_failed", retryable=True)
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            valid_video = _probe_video(temporary, request.timeouts.probe_s)
        except _ProbeUnavailable:
            _reject_download(
                request, state, "output_probe_failed", retryable=True
            )
        if not valid_video:
            _reject_download(
                request, state, "download_invalid_video", retryable=False
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except OSError:
        _fail(request, state, "output_write_failed", retryable=True, keep_task=True)
        raise H3Error("output_write_failed", retryable=True) from None
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return {"name": "generated.mp4", "sha256": digest.hexdigest(), "size": size}


def _reject_download(
    request: H3Request,
    state: dict[str, Any],
    code: str,
    *,
    retryable: bool,
) -> None:
    _fail(request, state, code, retryable=retryable, keep_task=True)
    raise H3Error(code, retryable=retryable)


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
                "-show_entries",
                "stream=codec_type:format=duration",
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
        duration = float((payload.get("format") or {}).get("duration"))
        streams = payload.get("streams")
    except (
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return False
    return (
        math.isfinite(duration)
        and duration > 0
        and isinstance(streams, list)
        and any(
            isinstance(stream, dict) and stream.get("codec_type") == "video"
            for stream in streams
        )
    )


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
) -> None:
    state["status"] = "retryable_failure" if retryable else "failed"
    state["retryable"] = retryable
    state["error"] = {"code": code}
    if not keep_task:
        if state["h3"].get("status") == "submitting":
            state["h3"]["status"] = "failed"
        elif state["ir"].get("status") == "submitting":
            state["ir"]["status"] = "failed"
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
