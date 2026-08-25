"""Volcengine AI MediaKit image erasure with durable paid-attempt receipts.

Each source image is uploaded through MediaKit's signed local-upload flow, then
processed by the synchronous ``erase-image`` tool.  A receipt is fsynced before
every paid tool POST.  If the connection fails after submission starts, the
attempt remains ``submitting`` and is never blindly repeated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from app.config import Settings
from app.retry import RetryPolicy, run_with_retry
from app.sanitize import sanitize

API_BASE = "https://mediakit.cn-beijing.volces.com/api/v1"
UPLOAD_URL = f"{API_BASE}/tools-sync/request-media-upload-url"
ERASE_URL = f"{API_BASE}/tools-sync/erase-image"
TEXT_SCENE = "full_screen_text_erase"
ICON_SCENE = "full_screen_icon_erase"
SCENES = frozenset({TEXT_SCENE, ICON_SCENE})
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_BYTES = 20 * 1024 * 1024
RECEIPT_VERSION = 1


class MediaKitError(Exception):
    """Safe public provider error used by the postprocess orchestrator."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _RequestLimitExceeded(Exception):
    """Explicit unaccepted submission that may be retried safely."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


async def erase_image(
    settings: Settings,
    cdir: Path,
    image: Path,
    out: Path,
    confirm: bool,
    scenes: tuple[str, ...],
) -> Path:
    """Erase selected text/icon scenes and atomically produce a real PNG.

    Multiple scenes are applied in order.  Successful intermediate stages are
    stored locally, so a later explicit provider failure can resume without
    paying for already completed stages.
    """
    if not settings.enable_mediakit_erase:
        raise MediaKitError(501, "MediaKit erase is disabled.")
    if confirm is not True:
        raise MediaKitError(409, "confirmation required")
    if out.exists():
        raise MediaKitError(409, "already edited")
    if not settings.mediakit_api_key:
        raise MediaKitError(503, "VOLC_MEDIAKIT_API_KEY not configured")
    if not scenes or len(set(scenes)) != len(scenes) or any(scene not in SCENES for scene in scenes):
        raise MediaKitError(409, "invalid erase request")

    image = image.resolve()
    out = out.resolve()
    cdir = cdir.resolve()
    if not image.is_relative_to(cdir) or not out.is_relative_to(cdir):
        raise MediaKitError(409, "invalid erase request")
    source = _inspect_input(image)
    await asyncio.to_thread(
        _run,
        settings,
        image,
        out,
        scenes,
        source,
    )
    return out


def _run(
    settings: Settings,
    image: Path,
    out: Path,
    scenes: tuple[str, ...],
    source: dict[str, Any],
) -> None:
    receipt_path, artifact_dir = _receipt_paths(out)
    receipt = _load_or_create_receipt(receipt_path, source, scenes)
    current = image
    timeout = httpx.Timeout(settings.mediakit_timeout_s, connect=10.0)
    headers = {
        "Authorization": f"Bearer {settings.mediakit_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for index, scene in enumerate(scenes):
            artifact = artifact_dir / f"{out.stem}.stage-{index + 1}.png"
            stage = receipt["stages"][index]
            state = stage.get("state")
            if state == "succeeded" and artifact.is_file():
                current = artifact
                continue
            if state == "response_received":
                result_url = stage.get("result_url")
                if not isinstance(result_url, str) or not result_url.startswith("https://"):
                    raise MediaKitError(502, "MediaKit receipt is invalid")
                _download_png(client, result_url, artifact, source)
                stage["state"] = "succeeded"
                stage["artifact"] = artifact.name
                _finish_current_attempt(stage)
                _atomic_json(receipt_path, receipt)
                current = artifact
                continue
            if state == "submitting":
                raise MediaKitError(409, "previous MediaKit submission outcome unknown")

            file_id = _upload(client, headers, current)
            try:
                result_url = run_with_retry(
                    lambda: _submit_erase_once(
                        client, headers, file_id, scene, stage, receipt_path, receipt,
                    ),
                    policy=RetryPolicy(settings.retry_count, settings.retry_interval_s),
                    is_retryable=lambda error: isinstance(error, _RequestLimitExceeded),
                )
            except _RequestLimitExceeded as error:
                raise MediaKitError(429, error.detail) from None
            _download_png(client, result_url, artifact, source)
            stage["state"] = "succeeded"
            stage["artifact"] = artifact.name
            stage["finished_at"] = _now()
            _finish_current_attempt(stage)
            _atomic_json(receipt_path, receipt)
            current = artifact

    if not current.is_file():
        raise MediaKitError(502, "MediaKit output is missing")
    _atomic_bytes(out, current.read_bytes())
    receipt["state"] = "succeeded"
    receipt["output"] = out.name
    receipt["finished_at"] = _now()
    _atomic_json(receipt_path, receipt)


def _submit_erase_once(
    client: httpx.Client,
    headers: dict[str, str],
    file_id: str,
    scene: str,
    stage: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
) -> str:
    attempts = _attempts(stage)
    attempt = {
        "attempt_id": uuid.uuid4().hex,
        "state": "submitting",
        "file_id": file_id,
        "started_at": _now(),
    }
    attempts.append(attempt)
    for key in (
        "failed_at", "error_code", "request_id", "task_id", "expires_at",
        "result_url", "response_received_at", "finished_at", "artifact",
    ):
        stage.pop(key, None)
    stage.update({
        "scene": scene,
        "state": "submitting",
        "attempt_id": attempt["attempt_id"],
        "file_id": file_id,
        "started_at": attempt["started_at"],
    })
    _atomic_json(receipt_path, receipt)
    try:
        response = client.post(
            ERASE_URL,
            headers=headers,
            json={"image_url": file_id, "standard_scene": scene},
        )
    except httpx.RequestError as error:
        raise MediaKitError(502, "MediaKit submission outcome unknown") from error

    body = _json_body(response)
    if response.status_code >= 400 or body.get("success") is not True:
        provider_error = body.get("error") if isinstance(body.get("error"), dict) else {}
        if body.get("success") is False:
            error_code = sanitize(str(provider_error.get("code") or "provider_rejected"))
            message = sanitize(str(provider_error.get("message") or "MediaKit erase rejected"))
            failed = {
                "state": "failed",
                "failed_at": _now(),
                "error_code": error_code,
                "request_id": body.get("request_id"),
                "task_id": body.get("task_id"),
            }
            attempt.update(failed)
            stage.update(failed)
            _atomic_json(receipt_path, receipt)
            if (
                response.status_code == 429
                and error_code == "RequestLimitExceeded"
                and body.get("task_id") in (None, "")
            ):
                raise _RequestLimitExceeded(message)
            raise MediaKitError(502, message)
        raise MediaKitError(502, "MediaKit submission outcome unknown")

    result = body.get("result")
    result_url = result.get("image_url") if isinstance(result, dict) else None
    if not isinstance(result_url, str) or not result_url.startswith("https://"):
        raise MediaKitError(502, "MediaKit submission outcome unknown")
    received = {
        "state": "response_received",
        "request_id": body.get("request_id"),
        "task_id": body.get("task_id"),
        "expires_at": body.get("expires_at"),
        "response_received_at": _now(),
    }
    attempt.update(received)
    stage.update({**received, "result_url": result_url})
    _atomic_json(receipt_path, receipt)
    return result_url


def _attempts(stage: dict[str, Any]) -> list[dict[str, Any]]:
    value = stage.get("attempts")
    if value is None:
        value = []
        stage["attempts"] = value
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise MediaKitError(409, "MediaKit receipt is invalid")
    return value


def _finish_current_attempt(stage: dict[str, Any]) -> None:
    attempts = _attempts(stage)
    if attempts and attempts[-1].get("attempt_id") == stage.get("attempt_id"):
        attempts[-1]["state"] = "succeeded"
        attempts[-1]["finished_at"] = _now()


def _inspect_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MediaKitError(409, "invalid erase request")
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise MediaKitError(409, "image exceeds MediaKit input limit")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MediaKitError(409, "invalid image")
    height, width = image.shape[:2]
    short, long = sorted((width, height))
    if short < 10 or long < 10 or short > 1440 or long > 2560:
        raise MediaKitError(409, "image dimensions exceed MediaKit input limit")
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": size,
        "width": width,
        "height": height,
    }


def _upload(client: httpx.Client, headers: dict[str, str], path: Path) -> str:
    try:
        response = client.post(UPLOAD_URL, headers=headers, json={})
    except httpx.RequestError as error:
        raise MediaKitError(502, "MediaKit upload initialization failed") from error
    body = _json_body(response)
    result = body.get("result") if body.get("success") is True else None
    if response.status_code >= 400 or not isinstance(result, dict):
        raise MediaKitError(502, "MediaKit upload initialization failed")
    file_id = result.get("file_id")
    upload_url = result.get("upload_url")
    if not isinstance(file_id, str) or not file_id.startswith("mediakit://"):
        raise MediaKitError(502, "MediaKit upload response is invalid")
    if not isinstance(upload_url, str) or not upload_url.startswith("https://"):
        raise MediaKitError(502, "MediaKit upload response is invalid")
    upload_headers = _upload_headers(result.get("upload_headers"))
    upload_headers.setdefault("Content-Type", mimetypes.guess_type(path.name)[0] or "image/png")
    try:
        with path.open("rb") as stream:
            uploaded = client.put(upload_url, headers=upload_headers, content=stream)
        uploaded.raise_for_status()
    except (OSError, httpx.HTTPError) as error:
        raise MediaKitError(502, "MediaKit file upload failed") from error
    return file_id


def _upload_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        return dict(value)
    if isinstance(value, list):
        headers: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                raise MediaKitError(502, "MediaKit upload response is invalid")
            key = item.get("key") or item.get("Key")
            val = item.get("value") or item.get("Value")
            if not isinstance(key, str) or not isinstance(val, str):
                raise MediaKitError(502, "MediaKit upload response is invalid")
            headers[key] = val
        return headers
    raise MediaKitError(502, "MediaKit upload response is invalid")


def _download_png(
    client: httpx.Client,
    url: str,
    destination: Path,
    source: dict[str, Any],
) -> None:
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_OUTPUT_BYTES:
                    raise MediaKitError(502, "MediaKit output exceeds size limit")
                chunks.append(chunk)
    except MediaKitError:
        raise
    except httpx.HTTPError as error:
        raise MediaKitError(502, "MediaKit output download failed") from error
    encoded = b"".join(chunks)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise MediaKitError(502, "MediaKit output is not a valid image")
    height, width = decoded.shape[:2]
    if width != source["width"] or height != source["height"]:
        raise MediaKitError(502, "MediaKit output dimensions changed")
    ok, png = cv2.imencode(".png", decoded)
    if not ok:
        raise MediaKitError(502, "MediaKit output PNG conversion failed")
    _atomic_bytes(destination, png.tobytes())


def _json_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError) as error:
        raise MediaKitError(502, "MediaKit returned an invalid response") from error
    if not isinstance(body, dict):
        raise MediaKitError(502, "MediaKit returned an invalid response")
    return body


def _receipt_paths(out: Path) -> tuple[Path, Path]:
    receipt_dir = out.parent / ".mediakit"
    artifact_dir = receipt_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(receipt_dir, 0o700)
    os.chmod(artifact_dir, 0o700)
    return receipt_dir / f"{out.name}.json", artifact_dir


def _load_or_create_receipt(
    path: Path, source: dict[str, Any], scenes: tuple[str, ...]
) -> dict[str, Any]:
    if path.exists():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MediaKitError(409, "MediaKit receipt is invalid") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("version") != RECEIPT_VERSION
            or receipt.get("source") != source
            or receipt.get("scenes") != list(scenes)
            or not isinstance(receipt.get("stages"), list)
            or len(receipt["stages"]) != len(scenes)
        ):
            raise MediaKitError(409, "MediaKit receipt does not match request")
        return receipt
    receipt = {
        "version": RECEIPT_VERSION,
        "provider": "volc-mediakit",
        "operation": "erase-image",
        "state": "pending",
        "source": source,
        "scenes": list(scenes),
        "stages": [{"scene": scene, "state": "pending"} for scene in scenes],
        "created_at": _now(),
    }
    _atomic_json(path, receipt)
    return receipt


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
