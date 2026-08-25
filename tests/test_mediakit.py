"""AI MediaKit erase-image provider contract and paid-attempt safety."""

import asyncio
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import mediakit
from app.config import get_settings
from conftest import make_settings


def _png(path: Path, width=64, height=48, value=0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def _webp(width=64, height=48, value=80) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".webp", image)
    assert ok
    return encoded.tobytes()


def _settings(tmp_path, **overrides):
    return make_settings(
        tmp_path,
        enable_mediakit_erase=True,
        mediakit_api_key="test-mediakit-key",
        **overrides,
    )


def _install_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(**kwargs):
        return real_client(transport=transport, **kwargs)

    monkeypatch.setattr(mediakit.httpx, "Client", factory)


def _call(settings, cdir, source, out, scenes=(mediakit.TEXT_SCENE,)):
    return asyncio.run(mediakit.erase_image(settings, cdir, source, out, True, scenes))


def test_single_text_erase_uploads_local_file_downloads_and_converts_png(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    calls = []

    def handler(request: httpx.Request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("request-media-upload-url"):
            assert request.headers["authorization"] == "Bearer test-mediakit-key"
            return httpx.Response(200, json={
                "success": True,
                "request_id": "upload-rid",
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            assert request.headers["content-type"] == "image/png"
            assert request.content.startswith(b"\x89PNG\r\n\x1a\n")
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            body = json.loads(request.content)
            assert body == {
                "image_url": "mediakit://file-1",
                "standard_scene": mediakit.TEXT_SCENE,
            }
            return httpx.Response(200, json={
                "success": True,
                "request_id": "erase-rid",
                "task_id": "erase-task",
                "expires_at": 2000000000,
                "result": {"image_url": "https://result.example/out.webp"},
            })
        if request.url.host == "result.example":
            return httpx.Response(200, content=_webp())
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    assert _call(settings, cdir, source, out) == out.resolve()
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imread(str(out))
    assert decoded.shape[:2] == (48, 64)
    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    assert receipt["state"] == "succeeded"
    assert receipt["stages"][0]["state"] == "succeeded"
    assert receipt["stages"][0]["request_id"] == "erase-rid"
    assert calls == [
        ("POST", "/api/v1/tools-sync/request-media-upload-url"),
        ("PUT", "/file-1"),
        ("POST", "/api/v1/tools-sync/erase-image"),
        ("GET", "/out.webp"),
    ]


def test_two_options_are_two_ordered_receipted_stages(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    scenes = []
    uploads = 0

    def handler(request: httpx.Request):
        nonlocal uploads
        if request.url.path.endswith("request-media-upload-url"):
            uploads += 1
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": f"mediakit://file-{uploads}",
                    "upload_url": f"https://upload.example/file-{uploads}",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            scene = json.loads(request.content)["standard_scene"]
            scenes.append(scene)
            return httpx.Response(200, json={
                "success": True,
                "request_id": f"rid-{len(scenes)}",
                "task_id": f"task-{len(scenes)}",
                "result": {"image_url": f"https://result.example/{len(scenes)}.webp"},
            })
        if request.url.host == "result.example":
            return httpx.Response(200, content=_webp(value=40 * len(scenes)))
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    _call(settings, cdir, source, out, (mediakit.TEXT_SCENE, mediakit.ICON_SCENE))
    assert scenes == [mediakit.TEXT_SCENE, mediakit.ICON_SCENE]
    assert uploads == 2
    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    assert [stage["state"] for stage in receipt["stages"]] == ["succeeded", "succeeded"]


def test_unknown_paid_post_is_persisted_and_never_blindly_retried(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    erase_calls = 0

    def handler(request: httpx.Request):
        nonlocal erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            raise httpx.ReadTimeout("lost after submit", request=request)
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    with pytest.raises(mediakit.MediaKitError, match="submission outcome unknown"):
        _call(settings, cdir, source, out)
    receipt_path = out.parent / ".mediakit/01.png.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["stages"][0]["state"] == "submitting"

    with pytest.raises(mediakit.MediaKitError) as caught:
        _call(settings, cdir, source, out)
    assert caught.value.status == 409
    assert caught.value.detail == "previous MediaKit submission outcome unknown"
    assert erase_calls == 1


def test_response_received_recovers_with_get_only(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    erase_calls = 0
    fail_download = True

    def handler(request: httpx.Request):
        nonlocal erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            return httpx.Response(200, json={
                "success": True,
                "request_id": "rid",
                "task_id": "task",
                "result": {"image_url": "https://result.example/out.webp"},
            })
        if request.url.host == "result.example":
            if fail_download:
                return httpx.Response(503)
            return httpx.Response(200, content=_webp())
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    with pytest.raises(mediakit.MediaKitError, match="output download failed"):
        _call(settings, cdir, source, out)
    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    assert receipt["stages"][0]["state"] == "response_received"

    fail_download = False
    _call(settings, cdir, source, out)
    assert erase_calls == 1
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_explicit_provider_rejection_can_be_retried(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    erase_calls = 0

    def handler(request: httpx.Request):
        nonlocal erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": f"mediakit://file-{erase_calls + 1}",
                    "upload_url": f"https://upload.example/file-{erase_calls + 1}",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            if erase_calls == 1:
                return httpx.Response(400, json={
                    "success": False,
                    "error": {"code": "InvalidParameter", "message": "bad image"},
                })
            return httpx.Response(200, json={
                "success": True,
                "result": {"image_url": "https://result.example/out.webp"},
            })
        if request.url.host == "result.example":
            return httpx.Response(200, content=_webp())
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    with pytest.raises(mediakit.MediaKitError, match="bad image"):
        _call(settings, cdir, source, out)
    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    assert receipt["stages"][0]["state"] == "failed"
    _call(settings, cdir, source, out)
    assert erase_calls == 2


def test_exact_rate_limit_retries_with_one_upload_and_durable_attempts(tmp_path, monkeypatch):
    settings = _settings(tmp_path, retry_count=2, retry_interval_s=0)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    upload_requests = 0
    erase_calls = 0

    def handler(request: httpx.Request):
        nonlocal upload_requests, erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            upload_requests += 1
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            if erase_calls < 3:
                return httpx.Response(429, json={
                    "success": False,
                    "request_id": f"limit-{erase_calls}",
                    "error": {
                        "code": "RequestLimitExceeded",
                        "message": "slow down",
                    },
                })
            return httpx.Response(200, json={
                "success": True,
                "request_id": "erase-rid",
                "task_id": "erase-task",
                "result": {"image_url": "https://result.example/out.webp"},
            })
        if request.url.host == "result.example":
            return httpx.Response(200, content=_webp())
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    _call(settings, cdir, source, out)

    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    attempts = receipt["stages"][0]["attempts"]
    assert upload_requests == 1
    assert erase_calls == 3
    assert [attempt["state"] for attempt in attempts] == ["failed", "failed", "succeeded"]
    assert len({attempt["attempt_id"] for attempt in attempts}) == 3
    assert [attempt["request_id"] for attempt in attempts[:2]] == ["limit-1", "limit-2"]


def test_exact_rate_limit_exhaustion_preserves_every_attempt(tmp_path, monkeypatch):
    settings = _settings(tmp_path, retry_count=2, retry_interval_s=0)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    erase_calls = 0

    def handler(request: httpx.Request):
        nonlocal erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            return httpx.Response(429, json={
                "success": False,
                "request_id": f"limit-{erase_calls}",
                "error": {
                    "code": "RequestLimitExceeded",
                    "message": "slow down",
                },
            })
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    with pytest.raises(mediakit.MediaKitError, match="slow down") as caught:
        _call(settings, cdir, source, out)
    assert caught.value.status == 429
    receipt = json.loads((out.parent / ".mediakit/01.png.json").read_text())
    assert erase_calls == 3
    assert receipt["stages"][0]["state"] == "failed"
    assert [item["state"] for item in receipt["stages"][0]["attempts"]] == [
        "failed", "failed", "failed",
    ]


@pytest.mark.parametrize(
    ("status", "error_code", "task_id"),
    [
        (200, "RequestLimitExceeded", None),
        (429, "OtherLimit", None),
        (429, "RequestLimitExceeded", "possibly-accepted-task"),
    ],
)
def test_rate_limit_retry_requires_exact_http_and_error_shape(
    tmp_path, monkeypatch, status, error_code, task_id,
):
    settings = _settings(tmp_path, retry_count=2, retry_interval_s=0)
    cdir = settings.data_dir / "c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    erase_calls = 0

    def handler(request: httpx.Request):
        nonlocal erase_calls
        if request.url.path.endswith("request-media-upload-url"):
            return httpx.Response(200, json={
                "success": True,
                "result": {
                    "file_id": "mediakit://file-1",
                    "upload_url": "https://upload.example/file-1",
                    "upload_headers": [],
                },
            })
        if request.url.host == "upload.example":
            return httpx.Response(200)
        if request.url.path.endswith("erase-image"):
            erase_calls += 1
            return httpx.Response(status, json={
                "success": False,
                "task_id": task_id,
                "error": {"code": error_code, "message": "rejected"},
            })
        raise AssertionError(request.url)

    _install_client(monkeypatch, handler)
    with pytest.raises(mediakit.MediaKitError, match="rejected"):
        _call(settings, cdir, source, out)
    assert erase_calls == 1


def test_provider_gates_and_input_limits(tmp_path):
    cdir = tmp_path / "data/c1"
    source = _png(cdir / "work/keyframes/01.png")
    out = cdir / "work/postprocessed/01.png"
    disabled = make_settings(tmp_path)
    with pytest.raises(mediakit.MediaKitError) as caught:
        _call(disabled, cdir, source, out)
    assert caught.value.status == 501

    missing_key = make_settings(tmp_path, enable_mediakit_erase=True)
    with pytest.raises(mediakit.MediaKitError) as caught:
        _call(missing_key, cdir, source, out)
    assert caught.value.status == 503

    settings = _settings(tmp_path)
    with pytest.raises(mediakit.MediaKitError, match="invalid erase request"):
        _call(settings, cdir, source, out, ("unknown",))
    huge = _png(cdir / "work/keyframes/huge.png", width=2561, height=1440)
    with pytest.raises(mediakit.MediaKitError, match="dimensions exceed"):
        _call(settings, cdir, huge, cdir / "work/postprocessed/huge.png")


def test_settings_mediakit_env_and_secret_repr(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "token")
    monkeypatch.setenv("ENABLE_MEDIAKIT_ERASE", "true")
    monkeypatch.setenv("VOLC_MEDIAKIT_API_KEY", "secret-key")
    monkeypatch.setenv("MEDIAKIT_CONCURRENCY", "0")
    monkeypatch.setenv("MEDIAKIT_TIMEOUT_S", "77")
    settings = get_settings()
    assert settings.enable_mediakit_erase is True
    assert settings.mediakit_api_key == "secret-key"
    assert settings.mediakit_concurrency == 1
    assert settings.mediakit_timeout_s == 77
    assert "secret-key" not in repr(settings)
