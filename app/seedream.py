"""Synchronous Ark Seedream client with a one-POST durable receipt boundary."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from app.config import Settings
from app.providers import (
    RETRY_BACKOFF_S,
    RETRYABLE_STATUSES,
    ProviderFailure,
    ProviderResult,
    is_retryable_transport_error,
    retry_sleep,
)


_LOGGER = logging.getLogger(__name__)


_SAFE_PROVIDER_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SAFE_CONTENT_TYPE_RE = re.compile(r"^[a-z0-9.+-]{1,64}/[a-z0-9.+-]{1,64}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEEDREAM_OUTPUT_SIZE = "2048x2048"
SEEDREAM_OUTPUT_WIDTH = 2048
SEEDREAM_OUTPUT_HEIGHT = 2048
_SAFE_REJECT_STATUS_CODES = frozenset(
    code for code in range(400, 500) if code not in {408, 409, 425, 429}
)
_TERMINAL_NO_REPOST = frozenset(
    {"submitting", "submission_unknown", "rejected", "protocol_error"}
)


class SeedreamError(RuntimeError):
    """Machine-readable cause wrapped by :class:`ProviderFailure`."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        provider_error_code: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.provider_error_code = provider_error_code
        super().__init__(detail)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> bool:
    """Publish a permanent ownership claim once, even across processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        # Once visible, the claim must remain: deleting it could authorize a
        # second paid POST after an ambiguous crash.
        raise
    return True


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _image_media_type(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("image bytes must be PNG, JPEG, or WebP")


def _validated_image_media_type(payload: bytes) -> str:
    media_type = _image_media_type(payload)
    if media_type == "image/png" and not payload.endswith(
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    ):
        raise ValueError("PNG result is incomplete")
    if media_type == "image/jpeg" and not payload.endswith(b"\xff\xd9"):
        raise ValueError("JPEG result is incomplete")
    if media_type == "image/webp" and (
        len(payload) < 12
        or int.from_bytes(payload[4:8], "little") + 8 != len(payload)
    ):
        raise ValueError("WebP result is incomplete")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size < 1:
        raise ValueError("image result cannot be fully decoded")
    return media_type


def _data_url(payload: bytes) -> tuple[str, str]:
    media_type = _image_media_type(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}", media_type


def _claim_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.name + ".post-claim")


def _response_marker_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.name + ".response.json")


def _result_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.name + ".result")


def reset_attempt(receipt_path: Path, destination: Path) -> list[str]:
    """Delete every durable artifact of a terminal Seedream attempt so a fresh
    POST may be issued under the same receipt path.  Returns removed names."""
    removed: list[str] = []
    candidates = (
        receipt_path,
        _claim_path(receipt_path),
        _response_marker_path(receipt_path),
        _result_path(receipt_path),
        Path(destination),
    )
    for path in candidates:
        if path.is_symlink():
            raise ValueError(f"Seedream attempt artifact is a symlink: {path.name}")
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path.name)
    return removed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"invalid durable JSON file: {path.name}") from None
    if not isinstance(value, dict):
        raise ValueError(f"durable JSON file is not an object: {path.name}")
    return value


def _safe_response_headers(response: httpx.Response) -> dict[str, str]:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if _SAFE_CONTENT_TYPE_RE.fullmatch(content_type):
        return {"content-type": content_type}
    return {}


def _provider_error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and _SAFE_PROVIDER_CODE_RE.fullmatch(code):
        return code
    return None


def _decode_response(response: httpx.Response) -> tuple[bytes, str]:
    try:
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError
        data = body.get("data")
        if (
            not isinstance(data, list)
            or not data
            or not isinstance(data[0], dict)
        ):
            raise TypeError
        encoded = data[0].get("b64_json")
        if not isinstance(encoded, str) or not encoded:
            raise TypeError
        raw = base64.b64decode(encoded, validate=True)
        if not raw:
            raise ValueError
        media_type = _validated_image_media_type(raw)
    except (binascii.Error, TypeError, ValueError):
        raise SeedreamError(
            "provider_protocol_error",
            "Seedream response must contain exactly one complete "
            f"{SEEDREAM_OUTPUT_SIZE} base64 image",
        ) from None
    return raw, media_type


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _failure(
    record: dict[str, Any],
    code: str,
    detail: str,
    *,
    evidence_media: list[Path] | None = None,
    provider_error_code: str | None = None,
) -> ProviderFailure:
    return ProviderFailure(
        record,
        evidence_media or [],
        SeedreamError(
            code,
            detail,
            provider_error_code=provider_error_code,
        ),
    )


class SeedreamClient:
    """Generate one image with at most one paid POST per durable receipt path."""

    SAFE_REJECT_STATUS_CODES = _SAFE_REJECT_STATUS_CODES

    def __init__(self, settings: Settings, transport: Any = None) -> None:
        self.settings = settings
        self.transport = transport

    def _record(
        self,
        request_summary: dict[str, Any],
        *,
        receipt_path: Path,
        status: str,
        response: dict[str, Any] | None = None,
        replayed: bool = False,
    ) -> dict[str, Any]:
        return {
            "provider": "ark_seedream",
            "operation": "image_generation",
            "outcome": status,
            "replayed": replayed,
            "request": request_summary,
            "response": response,
            "receipt": {
                "path": str(receipt_path),
                "status": status,
                "request_sha256": request_summary["request_sha256"],
            },
        }

    @staticmethod
    def _validate_binding(
        value: dict[str, Any],
        *,
        request_binding: dict[str, Any],
        request_sha256: str,
    ) -> None:
        if (
            type(value.get("version")) is not int
            or value.get("version") != 1
            or value.get("request_sha256") != request_sha256
            or value.get("model") != request_binding["model"]
            or value.get("prompt_sha256") != request_binding["prompt_sha256"]
            or value.get("input_sha256") != request_binding["input_sha256"]
            or value.get("size") != request_binding["size"]
        ):
            raise ValueError("Seedream durable receipt does not match the request")

    @staticmethod
    def _frozen_request(
        receipt: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        endpoint = receipt.get("endpoint")
        model = receipt.get("model")
        prompt_sha256 = receipt.get("prompt_sha256")
        input_sha256 = receipt.get("input_sha256")
        size = receipt.get("size")
        request_sha256 = receipt.get("request_sha256")
        if (
            type(receipt.get("version")) is not int
            or receipt.get("version") != 1
            or receipt.get("provider") != "ark_seedream"
            or not isinstance(endpoint, str)
            or not endpoint.startswith("https://")
            or not endpoint.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in endpoint)
            or not isinstance(model, str)
            or _MODEL_ID_RE.fullmatch(model) is None
            or not isinstance(prompt_sha256, str)
            or _SHA256_RE.fullmatch(prompt_sha256) is None
            or not isinstance(input_sha256, list)
            or any(
                not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                for value in input_sha256
            )
            or size != SEEDREAM_OUTPUT_SIZE
            or not isinstance(request_sha256, str)
            or _SHA256_RE.fullmatch(request_sha256) is None
        ):
            raise ValueError("Seedream receipt request binding is invalid")
        request_binding = {
            "version": 1,
            "endpoint": endpoint,
            "model": model,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": size,
            "response_format": "b64_json",
            "watermark": False,
        }
        if _canonical_sha256(request_binding) != request_sha256:
            raise ValueError("Seedream receipt request hash is invalid")
        request_summary = {
            "method": "POST",
            "url": endpoint,
            "headers": {"Authorization": "<redacted>"},
            "body": {
                "model": model,
                "prompt_sha256": prompt_sha256,
                "input_sha256": list(input_sha256),
                "size": size,
                "response_format": "b64_json",
                "watermark": False,
            },
            "request_sha256": request_sha256,
        }
        return request_sha256, request_summary

    @staticmethod
    def _validate_claim(claim_path: Path, request_sha256: str) -> None:
        claim = _read_json(claim_path)
        if (
            type(claim.get("version")) is not int
            or claim.get("version") != 1
            or claim.get("kind") != "seedream_paid_post"
            or claim.get("request_sha256") != request_sha256
        ):
            raise ValueError("Seedream POST claim does not match its receipt")

    @staticmethod
    def _read_result(
        *,
        marker_path: Path,
        result_path: Path,
        request_sha256: str,
    ) -> tuple[bytes, dict[str, Any]]:
        marker = _read_json(marker_path)
        if (
            type(marker.get("version")) is not int
            or marker.get("version") != 1
            or marker.get("kind") != "seedream_response"
            or marker.get("request_sha256") != request_sha256
            or marker.get("result_file") != result_path.name
            or not isinstance(marker.get("result_sha256"), str)
            or not isinstance(marker.get("result_bytes"), int)
            or marker.get("width") != SEEDREAM_OUTPUT_WIDTH
            or marker.get("height") != SEEDREAM_OUTPUT_HEIGHT
        ):
            raise ValueError("Seedream response marker is invalid")
        try:
            raw = result_path.read_bytes()
        except OSError:
            raise ValueError("Seedream raw result is missing") from None
        if (
            len(raw) != marker["result_bytes"]
            or _sha256(raw) != marker["result_sha256"]
        ):
            raise ValueError("Seedream raw result does not match its marker")
        try:
            media_type = _validated_image_media_type(raw)
        except ValueError:
            raise ValueError("Seedream raw result is not an image") from None
        if marker.get("media_type") != media_type:
            raise ValueError("Seedream raw result media type does not match its marker")
        return raw, marker

    def _replay(
        self,
        *,
        receipt: dict[str, Any],
        receipt_path: Path,
        marker_path: Path,
        result_path: Path,
        destination: Path,
        request_summary: dict[str, Any],
        request_sha256: str,
    ) -> ProviderResult[Path]:
        try:
            raw, marker = self._read_result(
                marker_path=marker_path,
                result_path=result_path,
                request_sha256=request_sha256,
            )
        except (OSError, ValueError) as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="protocol_error",
                replayed=True,
            )
            raise _failure(
                record,
                "attempt_receipt_invalid",
                str(exc),
                evidence_media=[result_path] if result_path.is_file() else [],
            ) from exc
        if receipt.get("status") == "succeeded" and (
            receipt.get("result_file") != result_path.name
            or receipt.get("result_sha256") != marker["result_sha256"]
            or receipt.get("result_bytes") != marker["result_bytes"]
            or receipt.get("response_marker_file") != marker_path.name
            or receipt.get("output_sha256") != _sha256(raw)
        ):
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="protocol_error",
                replayed=True,
            )
            raise _failure(
                record,
                "attempt_receipt_invalid",
                "Seedream succeeded receipt does not match its result",
                evidence_media=[result_path],
            )
        try:
            _atomic_bytes(destination, raw)
            if receipt.get("status") != "succeeded":
                completed = {
                    **receipt,
                    "status": "succeeded",
                    "result_file": result_path.name,
                    "result_sha256": marker["result_sha256"],
                    "result_bytes": marker["result_bytes"],
                    "response_marker_file": marker_path.name,
                    "output_sha256": _sha256(raw),
                    "completed_at_unix_ns": time.time_ns(),
                }
                _atomic_json(receipt_path, completed)
        except OSError as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="response_received",
                replayed=True,
            )
            raise _failure(
                record,
                "local_persistence_error",
                "Seedream local replay could not be durably published",
                evidence_media=[result_path],
            ) from exc
        response_summary = {
            "status_code": marker.get("http_status", 200),
            "headers": marker.get("response_headers", {}),
            "body": {
                "image": {
                    "bytes": marker["result_bytes"],
                    "sha256": marker["result_sha256"],
                    "media_type": marker["media_type"],
                    "width": marker["width"],
                    "height": marker["height"],
                }
            },
        }
        record = self._record(
            request_summary,
            receipt_path=receipt_path,
            status="succeeded",
            response=response_summary,
            replayed=True,
        )
        return ProviderResult(destination, record, [destination])

    def replay_received(
        self,
        *,
        destination: Path,
        receipt_path: Path,
    ) -> ProviderResult[Path]:
        """Recover a durably received image without consulting provider settings."""

        destination = Path(destination)
        receipt_path = Path(receipt_path)
        claim_path = _claim_path(receipt_path)
        marker_path = _response_marker_path(receipt_path)
        result_path = _result_path(receipt_path)
        if len({destination, receipt_path, claim_path, marker_path, result_path}) != 5:
            raise _failure(
                {
                    "provider": "ark_seedream",
                    "operation": "image_generation_replay",
                    "outcome": "invalid_input",
                    "replayed": True,
                },
                "invalid_input",
                "Seedream replay artifact paths must be distinct",
            )
        try:
            receipt = _read_json(receipt_path)
            request_sha256, request_summary = self._frozen_request(receipt)
            self._validate_claim(claim_path, request_sha256)
        except (OSError, ValueError) as exc:
            record = {
                "provider": "ark_seedream",
                "operation": "image_generation_replay",
                "outcome": "protocol_error",
                "replayed": True,
                "request": None,
                "response": None,
                "receipt": {
                    "path": str(receipt_path),
                    "status": "protocol_error",
                },
            }
            raise _failure(
                record,
                "attempt_receipt_invalid",
                str(exc),
                evidence_media=[result_path] if result_path.is_file() else [],
            ) from exc

        status = receipt.get("status")
        if status not in {"submitting", "response_received", "succeeded"}:
            code = {
                "rejected": "provider_rejected",
                "protocol_error": "provider_protocol_error",
            }.get(status, "submission_unknown")
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status=status if isinstance(status, str) else "protocol_error",
                replayed=True,
            )
            raise _failure(
                record,
                code,
                f"Seedream receipt status {status!r} has no recoverable response",
                evidence_media=[result_path] if result_path.is_file() else [],
                provider_error_code=(
                    receipt.get("provider_error_code")
                    if isinstance(receipt.get("provider_error_code"), str)
                    else None
                ),
            )

        if status == "submitting":
            if marker_path.exists():
                try:
                    self._read_result(
                        marker_path=marker_path,
                        result_path=result_path,
                        request_sha256=request_sha256,
                    )
                except (OSError, ValueError) as exc:
                    record = self._record(
                        request_summary,
                        receipt_path=receipt_path,
                        status="submission_unknown",
                        replayed=True,
                    )
                    raise _failure(
                        record,
                        "submission_unknown",
                        "Seedream submitting receipt has invalid local response evidence",
                        evidence_media=[result_path] if result_path.is_file() else [],
                    ) from exc
            else:
                try:
                    raw_result = result_path.read_bytes()
                    media_type = _validated_image_media_type(raw_result)
                    if not raw_result:
                        raise ValueError("Seedream raw result is empty")
                except (OSError, ValueError) as exc:
                    record = self._record(
                        request_summary,
                        receipt_path=receipt_path,
                        status="submission_unknown",
                        replayed=True,
                    )
                    raise _failure(
                        record,
                        "submission_unknown",
                        "Seedream submitting receipt has no valid local response",
                        evidence_media=[result_path] if result_path.is_file() else [],
                    ) from exc
                result_sha256 = _sha256(raw_result)
                marker = {
                    "version": 1,
                    "kind": "seedream_response",
                    "request_sha256": request_sha256,
                    "http_status": 200,
                    "response_headers": {},
                    "result_file": result_path.name,
                    "result_bytes": len(raw_result),
                    "result_sha256": result_sha256,
                    "media_type": media_type,
                    "width": SEEDREAM_OUTPUT_WIDTH,
                    "height": SEEDREAM_OUTPUT_HEIGHT,
                    "received_at_unix_ns": time.time_ns(),
                    "recovered_from": "durable_raw_result",
                }
                try:
                    _atomic_json(marker_path, marker)
                except OSError as exc:
                    record = self._record(
                        request_summary,
                        receipt_path=receipt_path,
                        status="response_received",
                        replayed=True,
                    )
                    raise _failure(
                        record,
                        "local_persistence_error",
                        "Seedream response marker could not be durably recovered",
                        evidence_media=[result_path],
                    ) from exc

        return self._replay(
            receipt=receipt,
            receipt_path=receipt_path,
            marker_path=marker_path,
            result_path=result_path,
            destination=destination,
            request_summary=request_summary,
            request_sha256=request_sha256,
        )

    def generate(
        self,
        *,
        prompt: str,
        images: list[Path],
        destination: Path,
        receipt_path: Path,
    ) -> ProviderResult[Path]:
        """Generate exactly one image; a request binding can issue at most one POST."""

        if not isinstance(prompt, str) or not prompt.strip():
            record = {
                "provider": "ark_seedream",
                "operation": "image_generation",
                "outcome": "invalid_input",
            }
            raise _failure(record, "invalid_input", "Seedream prompt must be non-empty")
        if not isinstance(images, list) or any(not isinstance(path, Path) for path in images):
            record = {
                "provider": "ark_seedream",
                "operation": "image_generation",
                "outcome": "invalid_input",
            }
            raise _failure(record, "invalid_input", "Seedream images must be a list of paths")
        destination = Path(destination)
        receipt_path = Path(receipt_path)
        marker_path = _response_marker_path(receipt_path)
        result_path = _result_path(receipt_path)
        claim_path = _claim_path(receipt_path)
        artifact_paths = {destination, receipt_path, marker_path, result_path, claim_path}
        if len(artifact_paths) != 5 or destination in images:
            record = {
                "provider": "ark_seedream",
                "operation": "image_generation",
                "outcome": "invalid_input",
            }
            raise _failure(
                record,
                "invalid_input",
                "Seedream input and durable artifact paths must be distinct",
            )

        image_payloads: list[str] = []
        input_receipts: list[dict[str, Any]] = []
        input_sha256: list[str] = []
        try:
            for index, path in enumerate(images, 1):
                raw = path.read_bytes()
                data_url, media_type = _data_url(raw)
                digest = _sha256(raw)
                image_payloads.append(data_url)
                input_sha256.append(digest)
                input_receipts.append(
                    {
                        "index": index,
                        "bytes": len(raw),
                        "sha256": digest,
                        "media_type": media_type,
                    }
                )
        except (OSError, ValueError) as exc:
            record = {
                "provider": "ark_seedream",
                "operation": "image_generation",
                "outcome": "invalid_input",
            }
            raise _failure(record, "invalid_input", str(exc)) from exc

        endpoint = self.settings.seedream_api_base + "/images/generations"
        prompt_sha256 = _sha256(prompt.encode("utf-8"))
        request_binding = {
            "version": 1,
            "endpoint": endpoint,
            "model": self.settings.seedream_model,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": SEEDREAM_OUTPUT_SIZE,
            "response_format": "b64_json",
            "watermark": False,
        }
        request_sha256 = _canonical_sha256(request_binding)
        request_summary = {
            "method": "POST",
            "url": endpoint,
            "headers": {
                "Authorization": "<redacted>",
                "Content-Type": "application/json",
            },
            "body": {
                "model": self.settings.seedream_model,
                "prompt_sha256": prompt_sha256,
                "input_receipts": input_receipts,
                "size": SEEDREAM_OUTPUT_SIZE,
                "response_format": "b64_json",
                "watermark": False,
            },
            "request_sha256": request_sha256,
        }
        base_receipt = {
            "version": 1,
            "provider": "ark_seedream",
            "request_sha256": request_sha256,
            "endpoint": endpoint,
            "model": self.settings.seedream_model,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": SEEDREAM_OUTPUT_SIZE,
        }
        payload: dict[str, Any] = {
            "model": self.settings.seedream_model,
            "prompt": prompt,
            "size": SEEDREAM_OUTPUT_SIZE,
            "response_format": "b64_json",
            "watermark": False,
        }
        if image_payloads:
            payload["image"] = image_payloads
        try:
            request_content = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="invalid_input",
            )
            raise _failure(
                record,
                "invalid_input",
                "Seedream request cannot be encoded as JSON",
                evidence_media=list(images),
            ) from exc

        if receipt_path.exists():
            try:
                receipt = _read_json(receipt_path)
                self._validate_binding(
                    receipt,
                    request_binding=request_binding,
                    request_sha256=request_sha256,
                )
            except ValueError as exc:
                record = self._record(
                    request_summary,
                    receipt_path=receipt_path,
                    status="protocol_error",
                )
                raise _failure(record, "attempt_receipt_invalid", str(exc)) from exc
            status = receipt.get("status")
            if status in {"response_received", "succeeded"} or (
                status == "submitting"
                and (marker_path.is_file() or result_path.is_file())
            ):
                return self.replay_received(
                    destination=destination,
                    receipt_path=receipt_path,
                )
            if status not in _TERMINAL_NO_REPOST:
                record = self._record(
                    request_summary,
                    receipt_path=receipt_path,
                    status="protocol_error",
                )
                raise _failure(
                    record,
                    "attempt_receipt_invalid",
                    "Seedream receipt has an unknown status",
                )
            code = {
                "rejected": "provider_rejected",
                "protocol_error": "provider_protocol_error",
            }.get(status, "submission_unknown")
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status=status,
            )
            raise _failure(
                record,
                code,
                f"Seedream request is already terminal with status {status}",
                provider_error_code=(
                    receipt.get("provider_error_code")
                    if isinstance(receipt.get("provider_error_code"), str)
                    else None
                ),
            )

        if marker_path.exists() or result_path.exists() or destination.exists():
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="submission_unknown",
            )
            raise _failure(
                record,
                "submission_unknown",
                "Seedream artifacts exist without a bound receipt",
                evidence_media=[result_path] if result_path.is_file() else [],
            )

        if not self.settings.seedream_api_key:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="not_configured",
            )
            raise _failure(
                record,
                "seedream_not_configured",
                "ARK_API_KEY is not configured",
                evidence_media=list(images),
            )
        try:
            # Finish all deterministic URL/header/body construction before
            # acquiring the permanent paid-POST claim.
            httpx.Request(
                "POST",
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.settings.seedream_api_key}",
                    "Content-Type": "application/json",
                },
                content=request_content,
            )
        except (TypeError, ValueError) as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="invalid_input",
            )
            raise _failure(
                record,
                "invalid_input",
                "Seedream HTTP request is invalid",
                evidence_media=list(images),
            ) from exc

        claim = {
            "version": 1,
            "kind": "seedream_paid_post",
            "request_sha256": request_sha256,
            "claimed_at_unix_ns": time.time_ns(),
        }
        try:
            won_claim = _exclusive_json(claim_path, claim)
        except OSError as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="submission_unknown",
            )
            raise _failure(
                record,
                "submission_unknown",
                "Seedream durable POST claim could not be committed",
            ) from exc
        if not won_claim:
            try:
                existing_claim = _read_json(claim_path)
                if (
                    type(existing_claim.get("version")) is not int
                    or existing_claim.get("version") != 1
                    or existing_claim.get("kind") != "seedream_paid_post"
                    or existing_claim.get("request_sha256") != request_sha256
                ):
                    raise ValueError("Seedream POST claim does not match the request")
            except ValueError as exc:
                record = self._record(
                    request_summary,
                    receipt_path=receipt_path,
                    status="protocol_error",
                )
                raise _failure(record, "attempt_receipt_invalid", str(exc)) from exc
            # Another process may have published the receipt after our first
            # check. Re-enter once through the read-only path; otherwise the
            # durable claim alone is sufficient evidence to prohibit POST.
            if receipt_path.exists():
                return self.generate(
                    prompt=prompt,
                    images=images,
                    destination=destination,
                    receipt_path=receipt_path,
                )
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="submission_unknown",
            )
            raise _failure(
                record,
                "submission_unknown",
                "Seedream POST was already claimed and may have been submitted",
            )

        submitting_receipt = {
            **base_receipt,
            "status": "submitting",
            "submitted_at_unix_ns": time.time_ns(),
        }
        try:
            _atomic_json(receipt_path, submitting_receipt)
        except OSError as exc:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="submission_unknown",
            )
            raise _failure(
                record,
                "submission_unknown",
                "Seedream pre-POST receipt could not be committed",
            ) from exc

        response: httpx.Response | None = None
        transport_error_type: str | None = None
        client: httpx.Client | None = None
        send_attempts: list[dict[str, Any]] = []
        max_send_attempts = len(RETRY_BACKOFF_S) + 1
        try:
            client = httpx.Client(
                timeout=httpx.Timeout(self.settings.seedream_timeout_s),
                transport=self.transport,
            )
            for attempt_number in range(1, max_send_attempts + 1):
                try:
                    response = client.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {self.settings.seedream_api_key}",
                            "Content-Type": "application/json",
                        },
                        content=request_content,
                    )
                except Exception as exc:
                    send_attempts.append(
                        {
                            "attempt": attempt_number,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )
                    if (
                        is_retryable_transport_error(exc)
                        and attempt_number < max_send_attempts
                    ):
                        _LOGGER.warning(
                            "Seedream POST attempt %d/%d failed with %s: %s; retrying in %gs",
                            attempt_number,
                            max_send_attempts,
                            type(exc).__name__,
                            exc,
                            RETRY_BACKOFF_S[attempt_number - 1],
                        )
                        retry_sleep(RETRY_BACKOFF_S[attempt_number - 1])
                        continue
                    unknown_receipt = {
                        **submitting_receipt,
                        "status": "submission_unknown",
                        "failed_at_unix_ns": time.time_ns(),
                    }
                    try:
                        _atomic_json(receipt_path, unknown_receipt)
                    except OSError:
                        pass
                    transport_error_type = type(exc).__name__
                else:
                    send_attempts.append(
                        {
                            "attempt": attempt_number,
                            "status_code": response.status_code,
                        }
                    )
                    if (
                        response.status_code in RETRYABLE_STATUSES
                        and attempt_number < max_send_attempts
                    ):
                        _LOGGER.warning(
                            "Seedream POST attempt %d/%d returned HTTP %d; retrying in %gs",
                            attempt_number,
                            max_send_attempts,
                            response.status_code,
                            RETRY_BACKOFF_S[attempt_number - 1],
                        )
                        retry_sleep(RETRY_BACKOFF_S[attempt_number - 1])
                        response = None
                        continue
                break
        except BaseException:
            try:
                _atomic_json(
                    receipt_path,
                    {
                        **submitting_receipt,
                        "status": "submission_unknown",
                        "failed_at_unix_ns": time.time_ns(),
                    },
                )
            except OSError:
                pass
            raise
        finally:
            if client is not None:
                try:
                    client.close()
                except BaseException:
                    # Once a complete HTTP response exists, connection cleanup
                    # cannot invalidate or hide the paid provider response.
                    if response is None and transport_error_type is None:
                        transport_error_type = "ClientCloseError"
        if transport_error_type is not None:
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="submission_unknown",
            )
            record["transport_error_type"] = transport_error_type
            if len(send_attempts) > 1:
                record["attempts"] = send_attempts
            raise _failure(
                record,
                "submission_unknown",
                "Seedream transport ended without a definite response",
                evidence_media=list(images),
            )

        assert response is not None

        response_headers = _safe_response_headers(response)
        if not 200 <= response.status_code < 300:
            provider_code = _provider_error_code(response)
            safe_rejection = response.status_code in self.SAFE_REJECT_STATUS_CODES
            terminal_status = "rejected" if safe_rejection else "submission_unknown"
            terminal_receipt = {
                **submitting_receipt,
                "status": terminal_status,
                "http_status": response.status_code,
                "failed_at_unix_ns": time.time_ns(),
            }
            if provider_code is not None:
                terminal_receipt["provider_error_code"] = provider_code
            try:
                _atomic_json(receipt_path, terminal_receipt)
            except OSError:
                pass
            response_summary = {
                "status_code": response.status_code,
                "headers": response_headers,
                "body": {"error_code": provider_code},
            }
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status=terminal_status,
                response=response_summary,
            )
            if len(send_attempts) > 1:
                record["attempts"] = send_attempts
            raise _failure(
                record,
                "provider_rejected" if safe_rejection else "submission_unknown",
                (
                    f"Seedream rejected the request with HTTP {response.status_code}"
                    if safe_rejection
                    else "Seedream returned an HTTP status with ambiguous submission outcome"
                ),
                evidence_media=list(images),
                provider_error_code=provider_code,
            )

        try:
            raw_result, media_type = _decode_response(response)
        except SeedreamError as exc:
            protocol_receipt = {
                **submitting_receipt,
                "status": "protocol_error",
                "http_status": response.status_code,
                "failed_at_unix_ns": time.time_ns(),
            }
            try:
                _atomic_json(receipt_path, protocol_receipt)
            except OSError:
                pass
            response_summary = {
                "status_code": response.status_code,
                "headers": response_headers,
                "body": {"error": "invalid_image_payload"},
            }
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status="protocol_error",
                response=response_summary,
            )
            raise ProviderFailure(record, list(images), exc) from exc

        result_sha256 = _sha256(raw_result)
        marker = {
            "version": 1,
            "kind": "seedream_response",
            "request_sha256": request_sha256,
            "http_status": response.status_code,
            "response_headers": response_headers,
            "result_file": result_path.name,
            "result_bytes": len(raw_result),
            "result_sha256": result_sha256,
            "media_type": media_type,
            "width": SEEDREAM_OUTPUT_WIDTH,
            "height": SEEDREAM_OUTPUT_HEIGHT,
            "received_at_unix_ns": time.time_ns(),
        }
        try:
            # The response bytes precede the marker. A marker is therefore a
            # durable proof that local recovery can finish without another POST.
            _atomic_bytes(result_path, raw_result)
            _atomic_json(marker_path, marker)
            response_received_receipt = {
                **submitting_receipt,
                "status": "response_received",
                "http_status": response.status_code,
                "response_marker_file": marker_path.name,
                "result_file": result_path.name,
                "result_bytes": len(raw_result),
                "result_sha256": result_sha256,
                "received_at_unix_ns": time.time_ns(),
            }
            _atomic_json(receipt_path, response_received_receipt)
            _atomic_bytes(destination, raw_result)
            succeeded_receipt = {
                **response_received_receipt,
                "status": "succeeded",
                "output_sha256": result_sha256,
                "completed_at_unix_ns": time.time_ns(),
            }
            _atomic_json(receipt_path, succeeded_receipt)
        except OSError as exc:
            evidence = [result_path] if result_path.is_file() else []
            record = self._record(
                request_summary,
                receipt_path=receipt_path,
                status=(
                    "response_received" if marker_path.is_file() else "submission_unknown"
                ),
                response={
                    "status_code": response.status_code,
                    "headers": response_headers,
                    "body": {
                        "image": {
                            "bytes": len(raw_result),
                            "sha256": result_sha256,
                            "media_type": media_type,
                            "width": SEEDREAM_OUTPUT_WIDTH,
                            "height": SEEDREAM_OUTPUT_HEIGHT,
                        }
                    },
                },
            )
            raise _failure(
                record,
                "local_persistence_error",
                "Seedream response could not be durably published",
                evidence_media=evidence,
            ) from exc

        response_summary = {
            "status_code": response.status_code,
            "headers": response_headers,
            "body": {
                "image": {
                    "bytes": len(raw_result),
                    "sha256": result_sha256,
                    "media_type": media_type,
                    "width": SEEDREAM_OUTPUT_WIDTH,
                    "height": SEEDREAM_OUTPUT_HEIGHT,
                }
            },
        }
        record = self._record(
            request_summary,
            receipt_path=receipt_path,
            status="succeeded",
            response=response_summary,
        )
        if len(send_attempts) > 1:
            record["attempts"] = send_attempts
        return ProviderResult(destination, record, [destination])
