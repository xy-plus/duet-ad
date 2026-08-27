"""Async Seedream image editor with durable pre-POST attempt receipts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.config import Settings

ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
MAX_POST_ATTEMPTS = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_ROLE_RE = re.compile(
    r"^(?:current_frame|"
    r"identity:[A-Za-z0-9_.-]{1,128}(?::[A-Za-z0-9_.-]{1,64})?|"
    r"scene:[A-Za-z0-9_.-]{1,128}(?::[A-Za-z0-9_.-]{1,64})?|"
    r"source_negative:(?:person|scene):[A-Za-z0-9_.-]{1,128}:[1-9]\d{0,3}|"
    r"target_reference:(?:person|scene):[A-Za-z0-9_.-]{1,128}:primary|"
    r"layout:[A-Za-z0-9_.-]{1,128}|annotation)$"
)


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


def _execution_binding(binding: object, input_shas: list[str]) -> dict:
    """Canonicalize the paid v2 request binding written before the provider POST."""
    base_keys = {
        "plan_sha256", "profile", "revision", "input_roles",
    }
    frame_keys = base_keys | {
        "reference_pack_candidate_sha256", "mask_manifest_sha256",
    }
    poc_keys = base_keys | {"reference_pack_candidate_sha256"}
    binding_keys = frozenset(binding) if isinstance(binding, dict) else frozenset()
    if not isinstance(binding, dict) or binding_keys not in {
        frozenset(base_keys), frozenset(poc_keys), frozenset(frame_keys),
    }:
        raise SeedreamError("invalid_input")
    plan_sha = binding.get("plan_sha256")
    profile = binding.get("profile")
    revision = binding.get("revision")
    roles = binding.get("input_roles")
    if (
        not isinstance(plan_sha, str) or _SHA256_RE.fullmatch(plan_sha) is None
        or not isinstance(profile, dict) or set(profile) != {"id", "revision"}
        or not isinstance(profile.get("id"), str)
        or profile["id"] != profile["id"].strip()
        or not profile["id"] or len(profile["id"].encode("utf-8")) > 128
        or isinstance(profile.get("revision"), bool)
        or not isinstance(profile.get("revision"), int) or profile["revision"] < 1
        or isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        or not isinstance(roles, list) or len(roles) != len(input_shas)
        or not roles or roles[0] != "current_frame"
        or any(
            not isinstance(role, str) or _INPUT_ROLE_RE.fullmatch(role) is None
            for role in roles
        )
        or len(set(roles)) != len(roles)
        or (
            binding_keys in {frozenset(poc_keys), frozenset(frame_keys)}
            and (
                _SHA256_RE.fullmatch(
                    binding.get("reference_pack_candidate_sha256", "")
                ) is None
                or (
                    binding_keys == frozenset(frame_keys)
                    and _SHA256_RE.fullmatch(
                        binding.get("mask_manifest_sha256", "")
                    ) is None
                )
            )
        )
    ):
        raise SeedreamError("invalid_input")
    frozen = {
        "plan_sha256": plan_sha,
        "profile": {"id": profile["id"], "revision": profile["revision"]},
        "revision": revision,
        "input_order": [
            {"position": position, "role": role, "sha256": digest}
            for position, (role, digest) in enumerate(zip(roles, input_shas), 1)
        ],
        # Ark exposes ordered multi-reference inputs here, not hard mask/depth/pose
        # controls. Production needs a downstream quality barrier; the isolated
        # three-frame probe records soft control and never publishes production.
        "soft_control": {
            "mechanism": "ordered_multi_reference",
            "annotation_image": "annotation" in roles,
            "hard_mask": False,
            "hard_depth": False,
            "hard_pose": False,
        },
    }
    if binding_keys in {frozenset(poc_keys), frozenset(frame_keys)}:
        frozen["reference_pack_candidate_sha256"] = binding[
            "reference_pack_candidate_sha256"
        ]
    if binding_keys == frozenset(frame_keys):
        frozen["mask_manifest_sha256"] = binding["mask_manifest_sha256"]
    return frozen


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
               receipt_path: Path, transport=None,
               execution_binding: dict | None = None) -> Path:
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
        "version": 2 if execution_binding is not None else 1,
        "status": "submitting", "attempt": 0,
        "model": settings.seedream_model, "mode": settings.seedream_edit_mode,
        "prompt_sha256": prompt_sha, "input_sha256": input_shas, "attempts": [],
    }
    if execution_binding is not None:
        base_receipt.update(
            prompt=prompt,
            **_execution_binding(execution_binding, input_shas),
        )
    start_attempt = 1
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SeedreamError("attempt_receipt_invalid") from None
        bound = all(
            existing.get(key) == value
            for key, value in base_receipt.items()
            if key not in {"status", "attempt", "attempts"}
        ) and existing.get("version") == base_receipt["version"]
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
                failed_receipt = {
                    **current, "status": "failed", "http_status": response.status_code,
                    "attempts": attempts,
                }
                error_code = _safe_provider_error_code(response)
                if error_code:
                    failed_receipt["provider_error_code"] = error_code
                _atomic_json(receipt_path, failed_receipt)
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
