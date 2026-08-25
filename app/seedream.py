"""Async Seedream image editor with durable pre-POST attempt receipts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.config import Settings

ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MAX_POST_ATTEMPTS = 3


class SeedreamError(RuntimeError):
    def __init__(self, code: str, detail: str = "Seedream image edit failed"):
        super().__init__(detail)
        self.code = code
        self.detail = detail


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
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _exact_quota(response: httpx.Response) -> bool:
    if response.status_code != 429:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return (
        isinstance(body, dict) and "data" not in body
        and isinstance(body.get("error"), dict)
        and body["error"].get("code") == "QuotaExceeded"
    )


def _decode(payload: dict) -> bytes:
    try:
        item = payload["data"][0]
        raw = base64.b64decode(item["b64_json"], validate=True)
    except (KeyError, IndexError, TypeError, ValueError):
        raise SeedreamError("provider_protocol_error") from None
    return raw


def _write_exact_png(raw: bytes, out: Path, width: int, height: int) -> None:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SeedreamError("provider_output_invalid")
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise SeedreamError("provider_output_invalid")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".seedream-", suffix=".png", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, out)
        _fsync_dir(out.parent)
    finally:
        Path(name).unlink(missing_ok=True)


async def edit(settings: Settings, images: list[bytes], prompt: str, out: Path, *,
               receipt_path: Path, transport=None) -> Path:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key:
        raise SeedreamError("seedream_not_configured")
    if not images:
        raise SeedreamError("invalid_input")
    first = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SeedreamError("invalid_input")
    height, width = first.shape[:2]
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    input_shas = [hashlib.sha256(item).hexdigest() for item in images]
    base_receipt = {
        "version": 1, "status": "submitting", "attempt": 0,
        "model": settings.seedream_model, "mode": settings.seedream_edit_mode,
        "prompt_sha256": prompt_sha, "input_sha256": input_shas, "attempts": [],
    }
    start_attempt = 1
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
            raise SeedreamError("provider_rejected")
        if status == "quota_retryable":
            base_receipt = existing
            previous_attempt = existing.get("attempt")
            if (
                isinstance(previous_attempt, bool)
                or not isinstance(previous_attempt, int)
                or previous_attempt < 1
            ):
                raise SeedreamError("attempt_receipt_invalid")
            start_attempt = previous_attempt + 1
        elif status not in {
            "succeeded", "response_received", "submitting", "submission_unknown", "failed",
        }:
            raise SeedreamError("attempt_receipt_invalid")
    payload = {
        "model": settings.seedream_model,
        "prompt": prompt,
        "image": [_data_url(item) for item in images],
        "sequential_image_generation": "disabled",
        "response_format": "b64_json",
        "watermark": False,
    }
    timeout = httpx.Timeout(settings.seedream_timeout_s)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for attempt in range(start_attempt, MAX_POST_ATTEMPTS + 1):
            attempts = list(base_receipt.get("attempts") or [])
            attempts.append({"number": attempt, "status": "submitting"})
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
            except httpx.RequestError:
                attempts[-1]["status"] = "submission_unknown"
                _atomic_json(receipt_path, {**current, "status": "submission_unknown", "attempts": attempts})
                raise SeedreamError("submission_unknown") from None
            if _exact_quota(response) and attempt < MAX_POST_ATTEMPTS:
                attempts[-1]["status"] = "quota_retryable"
                base_receipt = {**current, "status": "quota_retryable", "attempts": attempts}
                _atomic_json(receipt_path, base_receipt)
                if settings.retry_interval_s:
                    await asyncio.sleep(settings.retry_interval_s)
                continue
            if response.status_code >= 400:
                attempts[-1]["status"] = "failed"
                _atomic_json(receipt_path, {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                })
                raise SeedreamError("provider_rejected")
            try:
                raw = _decode(response.json())
            except (ValueError, SeedreamError):
                attempts[-1]["status"] = "failed"
                _atomic_json(receipt_path, {**current, "status": "failed", "attempts": attempts})
                raise SeedreamError("provider_protocol_error") from None
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
            _write_exact_png(raw, out, width, height)
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
