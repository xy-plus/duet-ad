"""Async Seedream image editor with durable pre-POST attempt receipts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from app import error_trace, frame_fit
from app.config import SEEDREAM_PRO_MODEL, Settings

ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MAX_POST_ATTEMPTS = 1
_LOGGER = logging.getLogger(__name__)


class SeedreamError(RuntimeError):
    def __init__(
        self,
        code: str,
        detail: str = "Seedream image edit failed",
        *,
        provider_error_code: str | None = None,
    ):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.provider_error_code = provider_error_code


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".attempt-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _claim_path(receipt_path: Path) -> Path:
    """Return the permanent, request-bound paid-POST ownership record."""
    return receipt_path.with_name(f"{receipt_path.name}.post-claim")


def _exclusive_json(path: Path, payload: dict) -> bool:
    """Create one durable JSON file exactly once across threads/processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(path.parent)
    except BaseException:
        # The file is deliberately retained.  Once ownership may have been
        # observed, deleting it could authorize a second paid POST.
        raise
    return True


def _read_bound_claim(path: Path, request_sha256: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The O_EXCL winner becomes visible before its fsync completes.  A
        # concurrent reader (or a crashed writer) must conservatively treat
        # that observation as ambiguous, never as permission to POST.
        raise SeedreamError("submission_unknown") from None
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or value.get("request_sha256") != request_sha256
        or value.get("kind") != "seedream_paid_post"
    ):
        raise SeedreamError("attempt_receipt_invalid")
    return value


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _data_url(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise SeedreamError("invalid_input")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _safe_provider_error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    code = body.get("error", {}).get("code") if isinstance(body, dict) else None
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", code):
        return code
    return None


def _decode(payload: dict) -> bytes:
    try:
        item = payload["data"][0]
        raw = base64.b64decode(item["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError):
        raise SeedreamError("provider_protocol_error") from None
    return raw


def _write_exact_png(raw: bytes, out: Path, width: int, height: int) -> None:
    try:
        encoded = frame_fit.normalize_image_to_canvas_png(
            raw, width, height, label="Seedream provider output",
        )
    except frame_fit.FrameFitError:
        raise SeedreamError("provider_output_invalid")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".seedream-", suffix=".png", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, out)
        _fsync_dir(out.parent)
    finally:
        Path(name).unlink(missing_ok=True)


async def edit(settings: Settings, images: list[bytes], prompt: str, out: Path, *,
               receipt_path: Path, transport=None,
               max_post_attempts: int = MAX_POST_ATTEMPTS) -> Path:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise SeedreamError("seedream_not_configured")
    if not images:
        raise SeedreamError("invalid_input")
    if (
        isinstance(max_post_attempts, bool)
        or not isinstance(max_post_attempts, int)
        or not 1 <= max_post_attempts <= MAX_POST_ATTEMPTS
    ):
        raise SeedreamError("invalid_input")
    first = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SeedreamError("invalid_input")
    height, width = first.shape[:2]
    image_data_urls = [_data_url(item) for item in images]
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    input_shas = [hashlib.sha256(item).hexdigest() for item in images]
    request_binding = {
        "model": settings.seedream_model,
        "mode": settings.seedream_edit_mode,
        "prompt_sha256": prompt_sha,
        "input_sha256": input_shas,
    }
    request_sha = hashlib.sha256(
        json.dumps(
            request_binding, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    base_receipt = {
        "version": 1, "status": "submitting", "attempt": 0,
        "model": settings.seedream_model, "mode": settings.seedream_edit_mode,
        "prompt_sha256": prompt_sha, "input_sha256": input_shas, "attempts": [],
    }
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SeedreamError("attempt_receipt_invalid") from None
        bound = (
            existing.get("model") == settings.seedream_model
            and existing.get("mode") == settings.seedream_edit_mode
            and existing.get("prompt_sha256") == prompt_sha
            and existing.get("input_sha256") == input_shas
        )
        if not bound:
            raise SeedreamError("attempt_receipt_invalid")
        status = existing.get("status")
        if (
            status == "succeeded" and out.is_file()
            and hashlib.sha256(out.read_bytes()).hexdigest() == existing.get("output_sha256")
        ):
            return out
        if status in {"response_received", "succeeded"}:
            result = receipt_path.parent / existing.get("result_file", "")
            if not result.is_file() or hashlib.sha256(result.read_bytes()).hexdigest() != existing.get("result_sha256"):
                raise SeedreamError("attempt_receipt_invalid")
            _write_exact_png(result.read_bytes(), out, width, height)
            existing.update(
                status="succeeded",
                output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            )
            _atomic_json(receipt_path, existing)
            return out
        if status in {"submitting", "submission_unknown"}:
            raise SeedreamError("submission_unknown")
        if status == "failed":
            provider_error_code = existing.get("provider_error_code")
            raise SeedreamError(
                "provider_rejected",
                provider_error_code=(
                    provider_error_code if isinstance(provider_error_code, str) else None
                ),
            )
        if status == "quota_retryable":
            # This legacy state proves that the frozen request was already
            # submitted once.  Never turn recovery into another paid POST.
            existing.update(
                status="failed",
                http_status=429,
                provider_error_code="QuotaExceeded",
            )
            _atomic_json(receipt_path, existing)
            raise SeedreamError("provider_rejected")
        elif status not in {
            "succeeded", "response_received", "submitting", "submission_unknown", "failed",
        }:
            raise SeedreamError("attempt_receipt_invalid")
    claim_path = _claim_path(receipt_path)
    won_claim = _exclusive_json(claim_path, {
        "version": 1,
        "kind": "seedream_paid_post",
        "request_sha256": request_sha,
        "claimed_at_unix_ns": time.time_ns(),
    })
    if not won_claim:
        _read_bound_claim(claim_path, request_sha)
        # A missing receipt means the winner may have crashed after the
        # durable claim and before/during POST.  Never infer "not submitted".
        if not receipt_path.is_file():
            raise SeedreamError("submission_unknown")
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SeedreamError("attempt_receipt_invalid") from None
        status = existing.get("status") if isinstance(existing, dict) else None
        if status == "failed":
            provider_code = existing.get("provider_error_code")
            raise SeedreamError(
                "provider_rejected",
                provider_error_code=provider_code if isinstance(provider_code, str) else None,
            )
        if status in {"submitting", "submission_unknown"}:
            raise SeedreamError("submission_unknown")
        # A terminal local result can only have been missed if it appeared
        # after the first replay check.  Re-enter the read-only replay path.
        return await edit(
            settings, images, prompt, out,
            receipt_path=receipt_path,
            transport=transport,
            max_post_attempts=max_post_attempts,
        )
    payload = {
        "model": settings.seedream_model,
        "prompt": prompt,
        "image": image_data_urls,
        "response_format": "b64_json",
        "watermark": False,
    }
    if settings.seedream_model != SEEDREAM_PRO_MODEL:
        payload["sequential_image_generation"] = "disabled"
    timeout = httpx.Timeout(settings.seedream_timeout_s)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for attempt in range(1, max_post_attempts + 1):
            attempts = [{"number": attempt, "status": "submitting"}]
            current = {**base_receipt, "attempt": attempt, "attempts": attempts}
            _atomic_json(receipt_path, current)
            try:
                response = await client.post(
                    ENDPOINT, headers={"Authorization": f"Bearer {key}"}, json=payload
                )
            except asyncio.CancelledError:
                attempts[-1]["status"] = "submission_unknown"
                _atomic_json(receipt_path, {
                    **current, "status": "submission_unknown", "attempts": attempts,
                })
                raise
            except httpx.RequestError as exc:
                attempts[-1]["status"] = "submission_unknown"
                _atomic_json(receipt_path, {**current, "status": "submission_unknown", "attempts": attempts})
                error_trace.record(
                    receipt_path.with_suffix(".error.json"),
                    call_path=["postprocess", "seedream", receipt_path.stem, "POST"],
                    error=exc,
                    logger=_LOGGER,
                    secrets=(key,),
                )
                raise SeedreamError("submission_unknown") from None
            if response.status_code >= 400:
                attempts[-1]["status"] = "failed"
                failed_receipt = {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                }
                error_code = _safe_provider_error_code(response)
                if error_code:
                    failed_receipt["provider_error_code"] = error_code
                _atomic_json(receipt_path, failed_receipt)
                error_trace.record(
                    receipt_path.with_suffix(".error.json"),
                    call_path=["postprocess", "seedream", receipt_path.stem, "POST"],
                    reason={
                        "code": "provider_rejected",
                        "provider": error_trace.provider_response(response, secrets=(key,)),
                    },
                    logger=_LOGGER,
                    secrets=(key,),
                )
                raise SeedreamError(
                    "provider_rejected", provider_error_code=error_code,
                )
            try:
                response_payload = response.json()
            except ValueError as exc:
                attempts[-1]["status"] = "failed"
                _atomic_json(receipt_path, {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                })
                error_trace.record(
                    receipt_path.with_suffix(".error.json"),
                    call_path=["postprocess", "seedream", receipt_path.stem, "response_json"],
                    reason={
                        "code": "provider_protocol_error",
                        "cause": error_trace.exception_tree(exc, secrets=(key,)),
                        "provider": error_trace.provider_response(response, secrets=(key,)),
                    },
                    logger=_LOGGER,
                    secrets=(key,),
                )
                raise SeedreamError("provider_protocol_error") from None
            try:
                raw = _decode(response_payload)
            except SeedreamError as exc:
                attempts[-1]["status"] = "failed"
                _atomic_json(receipt_path, {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                })
                error_trace.record(
                    receipt_path.with_suffix(".error.json"),
                    call_path=["postprocess", "seedream", receipt_path.stem, "response_decode"],
                    reason={
                        "code": exc.code,
                        "cause": error_trace.exception_tree(exc, secrets=(key,)),
                        "provider": error_trace.provider_response(response, secrets=(key,)),
                    },
                    logger=_LOGGER,
                    secrets=(key,),
                )
                raise
            # Persist the received bytes before publishing the canonical output. Recovery may
            # replay this local result, but must never issue another POST for this attempt.
            result_path = receipt_path.with_suffix(".result")
            _atomic_bytes(result_path, raw)
            attempts[-1]["status"] = "response_received"
            _atomic_json(receipt_path, {
                **current, "status": "response_received",
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "result_file": result_path.name,
                "attempts": attempts,
            })
            try:
                _write_exact_png(raw, out, width, height)
            except SeedreamError as exc:
                attempts[-1]["status"] = "failed"
                _atomic_json(receipt_path, {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                })
                error_trace.record(
                    receipt_path.with_suffix(".error.json"),
                    call_path=["postprocess", "seedream", receipt_path.stem, "output_validation"],
                    reason={
                        "code": exc.code,
                        "cause": error_trace.exception_tree(exc, secrets=(key,)),
                        "provider": error_trace.provider_response(response, secrets=(key,)),
                    },
                    logger=_LOGGER,
                    secrets=(key,),
                )
                raise
            attempts[-1]["status"] = "succeeded"
            _atomic_json(receipt_path, {
                **current, "status": "succeeded",
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "result_file": result_path.name,
                "attempts": attempts,
            })
            return out
    raise SeedreamError("provider_rejected")
