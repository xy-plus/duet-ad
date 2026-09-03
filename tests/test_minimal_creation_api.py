from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import downloader, long_video, minimal_creation, pipeline, storage
from app.main import create_app
from conftest import AUTH, make_settings


def _generation_request(**overrides):
    request = {
        "version": 1,
        "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "target_language": "  日语  ",
        },
        "replacement_guidance": None,
    }
    request.update(overrides)
    return request


def _canonical_request(**overrides):
    request = _generation_request(**overrides)
    request["dialogue"]["target_language"] = request["dialogue"][
        "target_language"
    ].strip()
    guidance = request["replacement_guidance"]
    if guidance is not None:
        guidance["instruction"] = guidance["instruction"].strip()
    return request


def _request_json(**overrides):
    return json.dumps(
        _generation_request(**overrides), ensure_ascii=False
    )


def _create_data(client_request_id, **overrides):
    return {
        "client_request_id": client_request_id,
        "generation_request": _request_json(**overrides),
    }


def _post_file(client, video_bytes, client_request_id, **overrides):
    return client.post(
        "/api/conversations",
        headers=AUTH,
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        data=_create_data(client_request_id, **overrides),
    )


def _image_bytes():
    image = np.full((7, 9, 3), (20, 90, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def minimal_api(tmp_path, monkeypatch):
    # Contract tests isolate request handling from deployment readiness.  The
    # readiness matrix is exercised separately below with real local fixtures.
    monkeypatch.setattr("app.main._minimal_creation_ready", lambda _settings: True)
    settings = make_settings(
        tmp_path,
        enable_pipeline=False,
        enable_minimal_creation=True,
    )
    with TestClient(create_app(settings)) as client:
        yield client, settings


def _assert_no_projects(settings):
    assert storage.list_conversations(settings.data_dir) == []


def test_capabilities_publish_the_exact_minimal_creation_contract(minimal_api):
    client, _settings = minimal_api

    response = client.get("/api/capabilities", headers=AUTH)

    assert response.status_code == 200
    capability = response.json()["minimal_creation"]
    assert capability == minimal_creation.capability()
    assert capability["dialogue"] == {
        "mode": "auto_rewrite",
        "translation": True,
    }


def test_disabled_minimal_creation_is_not_advertised_or_accepted(tmp_path):
    settings = make_settings(
        tmp_path,
        enable_pipeline=False,
        enable_minimal_creation=False,
    )
    with TestClient(create_app(settings)) as client:
        capability = client.get("/api/capabilities", headers=AUTH)
        assert capability.status_code == 200
        assert "minimal_creation" not in capability.json()
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data={
                "client_request_id": "minimal-disabled-0001",
                "generation_request": _request_json(),
                "reference_url": "https://media.example.test/source.mp4",
            },
        )
    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "minimal_creation_unavailable",
            "message": "当前创建方式尚未启用",
        }
    }
    _assert_no_projects(settings)


def _ready_minimal_settings(tmp_path, monkeypatch, **overrides):
    asr_cli = tmp_path / "whisper-cli"
    asr_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    asr_cli.chmod(0o700)
    asr_model = tmp_path / "whisper-model.bin"
    asr_model.write_bytes(b"model")
    credential = tmp_path / "deepseek.env"
    credential.write_text(
        "DEEPSEEK_API_KEY=sk-minimal-readiness-test\n", encoding="utf-8"
    )
    credential.chmod(0o600)
    monkeypatch.setenv("ARK_API_KEY", "ark-minimal-readiness-test")
    configured = {
        "enable_minimal_creation": True,
        "enable_pipeline": True,
        "enable_h3_submit": True,
        "autodl_art_token": "autodl-minimal-readiness-test",
        "minimax_api_key": "minimax-minimal-readiness-test",
        "enable_mediakit_erase": True,
        "mediakit_api_key": "mediakit-minimal-readiness-test",
        "asr_cli": asr_cli,
        "asr_model": asr_model,
        "deepseek_credential_file": credential,
    }
    configured.update(overrides)
    return make_settings(tmp_path, **configured)


def test_ready_minimal_creation_is_advertised(tmp_path, monkeypatch):
    settings = _ready_minimal_settings(tmp_path, monkeypatch)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/capabilities", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["minimal_creation"] == minimal_creation.capability()


@pytest.mark.parametrize(
    "override",
    [
        {"enable_pipeline": False},
        {"enable_h3_submit": False},
        {"autodl_art_token": ""},
        {"minimax_api_key": ""},
        {"enable_mediakit_erase": False},
        {"mediakit_api_key": ""},
        {"asr_cli": None},
        {"asr_model": None},
    ],
)
def test_unready_minimal_creation_is_hidden_and_rejected_without_persistence(
    tmp_path, monkeypatch, override,
):
    settings = _ready_minimal_settings(tmp_path, monkeypatch, **override)
    with TestClient(create_app(settings)) as client:
        capability = client.get("/api/capabilities", headers=AUTH)
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data={
                "client_request_id": "minimal-unready-0001",
                "generation_request": _request_json(),
                "reference_url": "https://media.example.test/source.mp4",
            },
        )
    assert "minimal_creation" not in capability.json()
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "minimal_creation_unavailable"
    _assert_no_projects(settings)


@pytest.mark.parametrize("missing", ["deepseek", "ark"])
def test_missing_runtime_credentials_hide_and_reject_minimal_creation(
    tmp_path, monkeypatch, missing,
):
    settings = _ready_minimal_settings(tmp_path, monkeypatch)
    if missing == "deepseek":
        settings.deepseek_credential_file.write_text("", encoding="utf-8")
    else:
        monkeypatch.delenv("ARK_API_KEY")
    with TestClient(create_app(settings)) as client:
        capability = client.get("/api/capabilities", headers=AUTH)
        response = client.post(
            "/api/conversations",
            headers=AUTH,
            data={
                "client_request_id": "minimal-unready-0002",
                "generation_request": _request_json(),
                "reference_url": "https://media.example.test/source.mp4",
            },
        )
    assert "minimal_creation" not in capability.json()
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "minimal_creation_unavailable"
    _assert_no_projects(settings)


def test_v1_multipart_part_types_and_utf8_fail_with_structured_detail(
    minimal_api,
):
    client, settings = minimal_api
    wrong_type = client.post(
        "/api/conversations",
        headers=AUTH,
        data={"client_request_id": "minimal-part-type-0001"},
        files=[
            ("file", ("clip.mp4", b"source", "video/mp4")),
            (
                "generation_request",
                ("request.json", _request_json().encode(), "application/json"),
            ),
        ],
    )
    invalid_utf8 = client.post(
        "/api/conversations",
        headers=AUTH,
        data={"client_request_id": "minimal-part-utf8-0001"},
        files=[
            ("file", ("clip.mp4", b"source", "video/mp4")),
            ("generation_request", (None, b"\xff", "application/json")),
        ],
    )
    assert wrong_type.status_code == 422
    assert wrong_type.json()["detail"]["code"] == "invalid_create_request"
    assert invalid_utf8.status_code == 422
    assert invalid_utf8.json()["detail"]["code"] == "invalid_generation_request"
    _assert_no_projects(settings)


def test_file_creation_freezes_public_input_and_internal_pipeline_mapping(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    source = video_1s.read_bytes()
    effective_request = _canonical_request()

    response = _post_file(
        client, source, "minimal-file-create-0001"
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {
        "id",
        "title",
        "effective_request",
        "input_receipt",
        "creation_input",
        "project_progress",
        "has_video",
    }
    assert body["title"] == "clip"
    assert body["effective_request"] == effective_request
    assert body["project_progress"] == {
        "percent": 0,
        "status": "queued",
    }
    assert body["has_video"] is False

    canonical = json.dumps(
        effective_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert body["input_receipt"] == {
        "version": 1,
        "client_request_id": "minimal-file-create-0001",
        "generation_request_sha256": hashlib.sha256(canonical).hexdigest(),
        "source": {
            "sha256": hashlib.sha256(source).hexdigest(),
            "bytes": len(source),
        },
        "replacement_image": None,
    }
    assert body["creation_input"] == {
        "version": 1,
        "source": {
            "mode": "upload",
            "filename": "clip.mp4",
            "bytes": len(source),
        },
        "replacement_image": None,
    }

    cid = body["id"]
    meta_path = settings.data_dir / cid / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["effective_request"] == effective_request
    assert meta["input_receipt"] == body["input_receipt"]
    assert meta["creation_input"] == body["creation_input"]
    assert meta["generation_config"] == {
        "optimize_image": True,
        "remove_subtitle": True,
        "remove_watermark": True,
    }
    assert meta["dialogue_mode"] == "auto"
    assert meta["voice_mode"] == "translate"
    assert meta["target_language"] == "日语"
    assert meta["dialogue_review_policy"] == "auto_continue"
    assert meta["status"] == "queued"
    assert meta.get("dialogue_review", {}).get("status") != "waiting"

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["effective_request"] == effective_request
    assert detail.json()["input_receipt"] == body["input_receipt"]
    assert detail.json()["creation_input"] == body["creation_input"]
    assert detail.json()["project_progress"] == body["project_progress"]
    assert detail.json()["has_video"] is False
    assert detail.json()["navigation_status"] == "analysis_queued"
    preview = client.get(
        f"/api/conversations/{cid}/creation-input/replacement-image",
        headers=AUTH,
    )
    assert preview.status_code == 404
    assert storage.load_meta(settings.data_dir, cid)["status"] == "queued"

    before = meta_path.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        storage.update_meta(
            settings.data_dir,
            cid,
            effective_request={
                **effective_request,
                "output": {
                    **effective_request["output"],
                    "resolution": "480p",
                },
            },
        )
    assert meta_path.read_bytes() == before


def test_reference_url_creation_remains_supported_without_network(
    minimal_api, video_1s, monkeypatch,
):
    client, settings = minimal_api
    seen = []

    def fake_fetch(reference_url, staging, _settings):
        seen.append(reference_url)
        destination = staging / "source.mp4"
        shutil.copyfile(video_1s, destination)
        return destination

    monkeypatch.setattr("app.downloader.fetch_reference", fake_fetch)
    reference_url = (
        "https://media.example.test/source.mp4?quality=original&signature=test"
    )
    response = client.post(
        "/api/conversations",
        headers=AUTH,
        data={
            **_create_data("minimal-url-create-0001"),
            "reference_url": f"  {reference_url}  ",
        },
    )

    assert response.status_code == 201, response.text
    assert seen == [reference_url]
    receipt = response.json()["input_receipt"]
    source = video_1s.read_bytes()
    assert receipt["source"] == {
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
    }
    meta = storage.load_meta(settings.data_dir, response.json()["id"])
    assert meta is not None
    assert meta["effective_request"] == _canonical_request()
    assert response.json()["creation_input"] == {
        "version": 1,
        "source": {
            "mode": "link",
            "reference_url": reference_url,
        },
        "replacement_image": None,
    }
    assert meta["creation_input"] == response.json()["creation_input"]
    detail = client.get(
        f"/api/conversations/{response.json()['id']}", headers=AUTH
    )
    assert detail.json()["creation_input"] == response.json()["creation_input"]


def test_pre_snapshot_minimal_project_returns_null_and_still_updates(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    response = _post_file(
        client,
        video_1s.read_bytes(),
        "minimal-before-creation-input",
    )
    assert response.status_code == 201
    cid = response.json()["id"]
    meta_path = settings.data_dir / cid / "meta.json"
    historical = json.loads(meta_path.read_text(encoding="utf-8"))
    historical.pop("creation_input")
    meta_path.write_text(
        json.dumps(historical, ensure_ascii=False), encoding="utf-8"
    )

    updated = storage.update_meta(settings.data_dir, cid, status="processing")
    assert updated is not None
    assert "creation_input" not in updated
    assert updated["error"] is None
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["creation_input"] is None


def test_reference_url_size_limit_keeps_413_contract(
    minimal_api, monkeypatch,
):
    client, settings = minimal_api

    def reject_large(*_args, **_kwargs):
        raise downloader.DownloadError(
            "file exceeds limit", code="source_too_large"
        )

    monkeypatch.setattr("app.downloader.fetch_reference", reject_large)
    response = client.post(
        "/api/conversations",
        headers=AUTH,
        data={
            **_create_data("minimal-url-too-large-0001"),
            "reference_url": "https://media.example.test/large.mp4",
        },
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "source_too_large"
    _assert_no_projects(settings)


def test_unpublished_v1_generated_file_is_not_public(minimal_api, video_1s):
    client, settings = minimal_api
    response = _post_file(
        client,
        video_1s.read_bytes(),
        "minimal-unpublished-output-0001",
    )
    assert response.status_code == 201
    cid = response.json()["id"]
    (settings.data_dir / cid / "generated.mp4").write_bytes(
        video_1s.read_bytes()
    )

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["has_video"] is False
    assert detail.json()["project_progress"] != {
        "percent": 100,
        "status": "succeeded",
    }
    artifact = client.get(
        f"/api/conversations/{cid}/files/generated.mp4", headers=AUTH
    )
    assert artifact.status_code == 404


def test_duration_limit_error_keeps_a_string_message(
    minimal_api, video_1s, monkeypatch,
):
    client, settings = minimal_api
    monkeypatch.setattr(
        storage,
        "probe_video",
        lambda _path: storage.VideoProbe(
            long_video.LONG_VIDEO_MAX_S + 1,
            1280,
            720,
        ),
    )
    response = _post_file(
        client,
        video_1s.read_bytes(),
        "minimal-duration-limit-0001",
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "video_duration_exceeds_h3_limit"
    assert isinstance(detail["message"], str)
    _assert_no_projects(settings)


def test_replacement_image_and_guidance_are_frozen_together(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    image = _image_bytes()
    guidance = {
        "instruction": "  把画面中的白色水杯替换为参考图中的产品杯  ",
        "image_field": "replacement_image",
    }
    response = client.post(
        "/api/conversations",
        headers=AUTH,
        files={
            "file": ("clip.mp4", video_1s.read_bytes(), "video/mp4"),
            "replacement_image": ("product.png", image, "image/png"),
        },
        data=_create_data(
            "minimal-image-create-0001",
            replacement_guidance=guidance,
        ),
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["effective_request"]["replacement_guidance"] == {
        "instruction": "把画面中的白色水杯替换为参考图中的产品杯",
        "image_field": "replacement_image",
    }
    assert body["input_receipt"]["replacement_image"] == {
        "sha256": hashlib.sha256(image).hexdigest(),
        "bytes": len(image),
    }
    expected_preview = (
        f"/api/conversations/{body['id']}/creation-input/replacement-image"
    )
    assert body["creation_input"] == {
        "version": 1,
        "source": {
            "mode": "upload",
            "filename": "clip.mp4",
            "bytes": len(video_1s.read_bytes()),
        },
        "replacement_image": {
            "filename": "product.png",
            "bytes": len(image),
            "media_type": "image/png",
            "preview_url": expected_preview,
        },
    }
    detail = client.get(
        f"/api/conversations/{body['id']}", headers=AUTH
    )
    assert detail.status_code == 200
    assert detail.json()["creation_input"] == body["creation_input"]
    assert "_minimal_replacement_image_path" not in body
    assert "_minimal_replacement_image_path" not in detail.json()
    assert client.get(expected_preview).status_code == 401
    preview = client.get(expected_preview, headers=AUTH)
    assert preview.status_code == 200
    assert preview.content == image
    assert preview.headers["content-type"] == "image/png"
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert client.post(expected_preview, headers=AUTH).status_code == 405
    assert client.get(
        f"{expected_preview}/unexpected.png", headers=AUTH
    ).status_code == 404
    assert client.get(
        f"/api/conversations/{body['id']}/files/inputs/replacement_image.png",
        headers=AUTH,
    ).status_code == 404
    project = settings.data_dir / body["id"]
    meta = json.loads((project / "meta.json").read_text(encoding="utf-8"))
    assert meta["_minimal_replacement_image_path"] == (
        "inputs/replacement_image.png"
    )
    assert (project / meta["_minimal_replacement_image_path"]).read_bytes() == image


@pytest.mark.parametrize(
    ("change", "expected_status"),
    [
        ("private_path", 404),
        ("receipt_sha256", 200),
        ("snapshot_bytes", 200),
        ("media_type", 200),
        ("media_type_object", 200),
        ("file_bytes", 200),
        ("missing_file", 404),
        ("symlink", 404),
    ],
)
def test_replacement_preview_depends_only_on_an_existing_safe_persisted_path(
    minimal_api, video_1s, change, expected_status,
):
    client, settings = minimal_api
    image = _image_bytes()
    response = client.post(
        "/api/conversations",
        headers=AUTH,
        files={
            "file": ("clip.mp4", video_1s.read_bytes(), "video/mp4"),
            "replacement_image": ("product.png", image, "image/png"),
        },
        data=_create_data(
            f"minimal-preview-{change.replace('_', '-')}",
            replacement_guidance={
                "instruction": "替换产品",
                "image_field": "replacement_image",
            },
        ),
    )
    assert response.status_code == 201, response.text
    cid = response.json()["id"]
    project = settings.data_dir / cid
    meta_path = project / "meta.json"
    image_path = project / "inputs" / "replacement_image.png"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_bytes = image

    if change == "private_path":
        meta["_minimal_replacement_image_path"] = "../outside.png"
    elif change == "receipt_sha256":
        meta["input_receipt"]["replacement_image"]["sha256"] = "0" * 64
    elif change == "snapshot_bytes":
        meta["creation_input"]["replacement_image"]["bytes"] += 1
    elif change == "media_type":
        meta["creation_input"]["replacement_image"]["media_type"] = "image/jpeg"
    elif change == "media_type_object":
        meta["creation_input"]["replacement_image"]["media_type"] = ["image/png"]
    elif change == "file_bytes":
        expected_bytes = bytes([image[0] ^ 1]) + image[1:]
        image_path.write_bytes(expected_bytes)
    elif change == "missing_file":
        image_path.unlink()
    else:
        outside = settings.data_dir / "outside.png"
        outside.write_bytes(image)
        image_path.unlink()
        image_path.symlink_to(outside)
    if change not in {"file_bytes", "missing_file", "symlink"}:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    before_meta = meta_path.read_bytes()
    preview = client.get(
        f"/api/conversations/{cid}/creation-input/replacement-image",
        headers=AUTH,
    )

    assert preview.status_code == expected_status
    if expected_status == 200:
        assert preview.content == expected_bytes
    assert meta_path.read_bytes() == before_meta
    persisted = storage.load_meta(settings.data_dir, cid)
    assert persisted["status"] == "queued"
    assert persisted["error"] is None
    assert persisted["generation"] is None


@pytest.mark.parametrize(
    ("with_guidance", "with_image", "field"),
    [
        (
            True,
            False,
            "replacement_image",
        ),
        (
            False,
            True,
            "generation_request.replacement_guidance",
        ),
    ],
)
def test_replacement_image_and_guidance_must_be_a_pair(
    minimal_api,
    video_1s,
    with_guidance,
    with_image,
    field,
):
    client, settings = minimal_api
    guidance = (
        {
            "instruction": "替换产品",
            "image_field": "replacement_image",
        }
        if with_guidance
        else None
    )
    files = {"file": ("clip.mp4", video_1s.read_bytes(), "video/mp4")}
    if with_image:
        files["replacement_image"] = (
            "product.png",
            _image_bytes(),
            "image/png",
        )

    response = client.post(
        "/api/conversations",
        headers=AUTH,
        files=files,
        data=_create_data(
            "minimal-pair-image-0001" if with_guidance
            else "minimal-pair-guidance-0001",
            replacement_guidance=guidance,
        ),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "replacement_pair_required",
        "message": "参考图与替换说明需要一起提供",
        "field": field,
    }
    _assert_no_projects(settings)


@pytest.mark.parametrize(
    ("media_type", "max_bytes", "expected_status"),
    [
        ("image/gif", minimal_creation.REPLACEMENT_MAX_BYTES, 201),
        ("image/png", 1, 413),
    ],
)
def test_replacement_media_uses_actual_bytes_and_enforces_the_byte_limit(
    minimal_api,
    video_1s,
    monkeypatch,
    media_type,
    max_bytes,
    expected_status,
):
    client, settings = minimal_api
    monkeypatch.setattr(
        minimal_creation, "REPLACEMENT_MAX_BYTES", max_bytes
    )
    response = client.post(
        "/api/conversations",
        headers=AUTH,
        files={
            "file": ("clip.mp4", video_1s.read_bytes(), "video/mp4"),
            "replacement_image": (
                "product.png",
                _image_bytes(),
                media_type,
            ),
        },
        data=_create_data(
            f"minimal-image-error-{expected_status}",
            replacement_guidance={
                "instruction": "替换产品",
                "image_field": "replacement_image",
            },
        ),
    )
    assert response.status_code == expected_status
    if expected_status == 201:
        assert response.json()["creation_input"]["replacement_image"][
            "media_type"
        ] == "image/png"
        return
    assert response.json()["detail"]["code"] == "invalid_replacement_image"
    _assert_no_projects(settings)


@pytest.mark.parametrize(
    ("generation_request", "code", "field"),
    [
        (
            "not-json",
            "invalid_generation_request_json",
            "generation_request",
        ),
        (
            _request_json(
                processing={
                    "optimize_image": True,
                    "remove_subtitle": True,
                    "remove_logo": False,
                }
            ),
            "processing_must_be_enabled",
            "generation_request.processing",
        ),
    ],
)
def test_contract_failures_use_structured_public_errors_without_creation(
    minimal_api, video_1s, generation_request, code, field,
):
    client, settings = minimal_api

    response = client.post(
        "/api/conversations",
        headers=AUTH,
        files={
            "file": ("clip.mp4", video_1s.read_bytes(), "video/mp4")
        },
        data={
            "client_request_id": "minimal-invalid-json-0001"
            if generation_request == "not-json"
            else "minimal-invalid-processing-0001",
            "generation_request": generation_request,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_generation_request"
    assert response.json()["detail"]["field"] == field
    assert isinstance(response.json()["detail"]["message"], str)
    assert response.json()["detail"]["message"]
    _assert_no_projects(settings)


def test_minimal_creation_rate_limit_is_structured(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    payload = {
        "client_request_id": "minimal-rate-limit-0001",
        "generation_request": _request_json(),
    }
    for _ in range(10):
        response = client.post(
            "/api/conversations", headers=AUTH, data=payload
        )
        assert response.status_code == 400
    limited = client.post(
        "/api/conversations", headers=AUTH, data=payload
    )
    assert limited.status_code == 429
    assert limited.json() == {
        "detail": {
            "code": "rate_limited",
            "message": "创建请求过于频繁，请稍后重试",
        }
    }
    _assert_no_projects(settings)


def test_legacy_script_dialogue_is_rejected_without_creation(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    legacy_dialogue = {
        "mode": "auto_rewrite",
        "target_language": "日语",
        "script": "用户预写台词",
    }

    response = _post_file(
        client,
        video_1s.read_bytes(),
        "minimal-legacy-script-0001",
        dialogue=legacy_dialogue,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "invalid_generation_request",
        "message": "generation_request 结构不符合 v1 合同",
        "field": "generation_request.dialogue",
    }
    _assert_no_projects(settings)


@pytest.mark.parametrize("duplicate_field", ["generation_request", "client_request_id"])
def test_duplicate_multipart_fields_are_rejected_before_creation(
    minimal_api, video_1s, duplicate_field,
):
    client, settings = minimal_api
    generation_request = _request_json()
    fields = [
        ("file", ("clip.mp4", video_1s.read_bytes(), "video/mp4")),
        ("client_request_id", (None, "minimal-duplicate-0001")),
        ("generation_request", (None, generation_request)),
    ]
    duplicate_value = (
        generation_request
        if duplicate_field == "generation_request"
        else "minimal-duplicate-0002"
    )
    fields.append((duplicate_field, (None, duplicate_value)))

    response = client.post(
        "/api/conversations", headers=AUTH, files=fields
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "invalid_create_request",
        "message": "创建请求包含未知或重复字段",
    }
    _assert_no_projects(settings)


def test_client_request_id_is_idempotent_and_conflicts_on_different_input(
    minimal_api, video_1s,
):
    client, settings = minimal_api
    source = video_1s.read_bytes()
    client_request_id = "minimal-idempotency-0001"

    created = _post_file(client, source, client_request_id)
    repeated = client.post(
        "/api/conversations",
        headers=AUTH,
        files={"file": ("renamed.mp4", source, "video/mp4")},
        data=_create_data(client_request_id),
    )
    conflicting = _post_file(
        client,
        source,
        client_request_id,
        dialogue={
            "mode": "auto_rewrite",
            "target_language": "英语",
        },
    )

    assert created.status_code == 201, created.text
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == created.json()
    assert repeated.json()["creation_input"]["source"]["filename"] == "clip.mp4"
    assert conflicting.status_code == 409
    assert conflicting.json()["detail"] == {
        "code": "client_request_id_conflict",
        "message": "client_request_id 已绑定到不同的创建输入",
        "field": "client_request_id",
    }
    conversations = storage.list_conversations(settings.data_dir)
    assert [item["id"] for item in conversations] == [created.json()["id"]]
    assert not any(
        path.is_dir() and path.name.endswith(".staging")
        for path in settings.data_dir.iterdir()
    )


def test_concurrent_identical_client_request_id_publishes_once(
    minimal_api, video_1s, monkeypatch,
):
    client, settings = minimal_api
    source = video_1s.read_bytes()
    barrier = threading.Barrier(2)
    original_probe = storage.probe_video

    def synchronized_probe(path):
        video = original_probe(path)
        barrier.wait(timeout=5)
        return video

    monkeypatch.setattr(storage, "probe_video", synchronized_probe)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _post_file,
                client,
                source,
                "minimal-concurrent-same-0001",
            )
            for _ in range(2)
        ]
        responses = [future.result(timeout=10) for future in futures]

    assert sorted(item.status_code for item in responses) == [200, 201]
    assert responses[0].json() == responses[1].json()
    conversations = storage.list_conversations(settings.data_dir)
    assert [item["id"] for item in conversations] == [
        responses[0].json()["id"]
    ]
    assert not any(
        path.is_dir() and path.name.endswith(".staging")
        for path in settings.data_dir.iterdir()
    )


def test_concurrent_different_input_with_same_client_request_id_conflicts(
    minimal_api, video_1s, monkeypatch,
):
    client, settings = minimal_api
    source = video_1s.read_bytes()
    barrier = threading.Barrier(2)
    original_probe = storage.probe_video

    def synchronized_probe(path):
        video = original_probe(path)
        barrier.wait(timeout=5)
        return video

    monkeypatch.setattr(storage, "probe_video", synchronized_probe)
    languages = ("英语", "韩语")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _post_file,
                client,
                source,
                "minimal-concurrent-conflict-0001",
                dialogue={
                    "mode": "auto_rewrite",
                    "target_language": language,
                },
            )
            for language in languages
        ]
        responses = [future.result(timeout=10) for future in futures]

    assert sorted(item.status_code for item in responses) == [201, 409]
    created = next(item for item in responses if item.status_code == 201)
    conflict = next(item for item in responses if item.status_code == 409)
    assert conflict.json()["detail"] == {
        "code": "client_request_id_conflict",
        "message": "client_request_id 已绑定到不同的创建输入",
        "field": "client_request_id",
    }
    conversations = storage.list_conversations(settings.data_dir)
    assert [item["id"] for item in conversations] == [created.json()["id"]]
    assert conversations[0]["effective_request"] == created.json()[
        "effective_request"
    ]
    assert not any(
        path.is_dir() and path.name.endswith(".staging")
        for path in settings.data_dir.iterdir()
    )


def test_published_v1_queue_gap_is_claimed_by_startup_recovery(
    tmp_path, video_1s, monkeypatch,
):
    monkeypatch.setattr("app.main._minimal_creation_ready", lambda _settings: True)
    initial_settings = make_settings(
        tmp_path,
        enable_pipeline=False,
        enable_minimal_creation=True,
    )
    with TestClient(create_app(initial_settings)) as client:
        created = _post_file(
            client,
            video_1s.read_bytes(),
            "minimal-recovery-gap-0001",
        )
    assert created.status_code == 201, created.text
    cid = created.json()["id"]
    queued = storage.load_meta(initial_settings.data_dir, cid)
    assert queued is not None
    assert queued["status"] == "queued"
    assert queued.get("_input_owner") is None

    recovered = threading.Event()

    def fake_run(settings, recovered_cid, _runner, **kwargs):
        assert recovered_cid == cid
        owner = kwargs["claimed_owner"]
        claimed = storage.load_meta(settings.data_dir, recovered_cid)
        assert claimed is not None
        assert claimed["status"] == "processing"
        assert claimed["effective_request"] == created.json()[
            "effective_request"
        ]
        assert claimed["input_receipt"] == created.json()["input_receipt"]
        assert storage.finish_input_claim(
            settings.data_dir,
            recovered_cid,
            owner,
            status="failed",
            error="recovered-v1-queue-gap",
        )
        recovered.set()

    monkeypatch.setattr(pipeline, "run", fake_run)
    recovery_settings = make_settings(
        tmp_path,
        enable_pipeline=True,
        enable_h3_submit=False,
        enable_minimal_creation=True,
    )
    with TestClient(create_app(recovery_settings)):
        assert recovered.wait(timeout=2)

    final = storage.load_meta(recovery_settings.data_dir, cid)
    assert final is not None
    assert final["error"] == "recovered-v1-queue-gap"
