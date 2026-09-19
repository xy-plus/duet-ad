from __future__ import annotations

import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import public_api, public_api_auth, public_credits
from app.config import Settings
from app.main import create_app
from app.store import ProjectStore


TOKEN = "test-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_subtitles_form_freezes_boolean_and_binds_idempotency(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    try:
        with TestClient(app) as client:
            capability = client.get("/api/v1/video-generations/capabilities", headers=auth).json()
            assert capability["defaults"]["add_subtitles"] is False
            created = _create(client, auth, video_1s, key="subtitle01", add_subtitles="true")
            assert created.status_code == 201
            assert created.json()["parameters"]["add_subtitles"] is True
            pid = created.json()["id"][3:]
            state = client.get(f"/api/conversations/{pid}", headers=AUTH).json()
            assert state["effective_request"]["add_subtitles"] is True
            replay = _create(client, auth, video_1s, key="subtitle01", add_subtitles="1")
            assert replay.status_code == 200
            changed = _create(client, auth, video_1s, key="subtitle01", add_subtitles="false")
            assert changed.status_code == 409
            default = _create(client, auth, video_1s, key="subtitle02")
            assert default.status_code == 201
            assert default.json()["parameters"]["add_subtitles"] is False
            false_replay = _create(client, auth, video_1s, key="subtitle02", add_subtitles="false")
            assert false_replay.status_code == 200
            invalid = _create(client, auth, video_1s, key="subtitle03", add_subtitles="maybe")
            assert invalid.status_code == 422
            assert len(app.state.store.list()) == 2
    finally:
        app.state.pipeline.release.set()


def test_legacy_public_parameters_default_subtitles_off():
    assert public_api.Parameters(aspect_ratio="9:16", resolution="768p", target_language=None, replacement_image=False).add_subtitles is False


def test_pre_subtitle_persisted_job_replays_after_upgrade(tmp_path, video_1s, monkeypatch):
    import importlib
    main_module = importlib.import_module("app.main")
    old_parser = main_module._parse_generation_request
    old_hash = public_api.canonical_json_sha256
    def legacy_parser(*args):
        value = old_parser(*args)
        value.pop("add_subtitles", None)
        return value
    def legacy_hash(value):
        value = {**value, "request": {key: item for key, item in value["request"].items() if key != "add_subtitles"}}
        return old_hash(value)
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    try:
        with TestClient(app) as client:
            with monkeypatch.context() as old_code:
                old_code.setattr(main_module, "_parse_generation_request", legacy_parser)
                old_code.setattr(public_api, "canonical_json_sha256", legacy_hash)
                created = _create(client, auth, video_1s, key="legacy-subtitle")
            assert created.status_code == 201
            pid = created.json()["id"][3:]
            stored = app.state.store.read(pid)
            assert "add_subtitles" not in stored["effective_request"]
            receipt = dict(stored["_public_api"])
            receipt["parameters"].pop("add_subtitles", None)
            app.state.store.update(pid, _public_api=receipt)
            replay = _create(client, auth, video_1s, key="legacy-subtitle", add_subtitles="false")
            assert replay.status_code == 200
            assert replay.json()["id"] == created.json()["id"]
            assert replay.json()["parameters"]["add_subtitles"] is False
    finally:
        app.state.pipeline.release.set()


class ManualPipeline:
    """Leaves every project queued until the test releases it."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[str] = []

    def run(self, project_id: str) -> None:
        self.calls.append(project_id)
        self.release.wait(timeout=10)


def manual_factory(_settings, _store, _register_asset):
    return ManualPipeline()


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        access_token=TOKEN,
        multiref_mode="composed",
        data_dir=tmp_path / "data",
        public_base_url="https://studio.example",
        store_lock_enabled=False,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def video_1s(tmp_path):
    p = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-pix_fmt", "yuv420p", str(p),
        ],
        check=True, capture_output=True,
    )
    return p


def _public_setup(tmp_path, *, owner="partner_one", credits=5_000):
    registry = tmp_path / "public-api-clients.json"
    api_key = public_api_auth.create_client_key(registry, owner, key_id="key00001")
    settings = make_settings(
        tmp_path,
        public_api_enabled=True,
        public_api_clients_file=registry,
    )
    if credits:
        public_credits.adjust(
            settings.data_dir,
            owner,
            credits,
            reason="test grant",
            idempotency_key="grant0001",
        )
    return settings, {"Authorization": f"Bearer {api_key}"}, api_key


def _create(client, auth, video, *, key="request001", **data):
    payload = {
        "aspect_ratio": "9:16",
        "resolution": "768p",
        **data,
    }
    with video.open("rb") as stream:
        return client.post(
            "/api/v1/video-generations",
            headers={**auth, "Idempotency-Key": key},
            data=payload,
            files={"source_video": ("source.mp4", stream, "video/mp4")},
        )


def _create_upload(client, auth, video, *, key, filename, content_type):
    with video.open("rb") as stream_content:
        upload = (
            (filename, stream_content, content_type)
            if content_type is not None
            else (filename, stream_content)
        )
        return client.post(
            "/api/v1/video-generations",
            headers={**auth, "Idempotency-Key": key},
            data={"aspect_ratio": "9:16", "resolution": "768p"},
            files={"source_video": upload},
        )


def _store(app) -> ProjectStore:
    return app.state.store


def test_public_api_is_disabled_by_default(tmp_path):
    app = create_app(make_settings(tmp_path), pipeline_factory=manual_factory)
    with TestClient(app) as client:
        assert client.get("/api/v1/openapi.json").status_code == 404
        assert client.get("/api/v1/account/credits").status_code == 404


def test_public_auth_is_independent_and_owner_scoped(tmp_path, video_1s):
    settings, auth_one, _ = _public_setup(tmp_path)
    key_two = public_api_auth.create_client_key(
        settings.public_api_clients_file, "partner_two", key_id="key00002"
    )
    public_credits.adjust(
        settings.data_dir,
        "partner_two",
        5_000,
        reason="test grant",
        idempotency_key="grant0002",
    )
    auth_two = {"Authorization": f"Bearer {key_two}"}
    app = create_app(settings, pipeline_factory=manual_factory)
    app.state.pipeline.release.set()
    with TestClient(app) as client:
        created = _create(client, auth_one, video_1s)
        assert created.status_code == 201
        job_id = created.json()["id"]
        assert client.get(
            f"/api/v1/video-generations/{job_id}", headers=auth_two
        ).status_code == 404
        assert client.get(
            "/api/v1/account/credits",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ).status_code == 401
        assert client.get("/api/conversations", headers=auth_one).status_code == 401


def test_create_reserves_credits_and_is_rotation_safe_idempotent(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s)
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "queued"
        assert body["parameters"] == {
            "add_subtitles": False,
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "target_language": None,
            "replacement_image": False,
        }
        assert body["billing"] == {
            "currency": "CNY",
            "credits_per_cny": 100,
            "quoted_credits": 60,
            "quoted_amount_minor": 60,
            "price_version": "credits-per-second-60-v1",
            "settlement_status": "pending",
            "settled_credits": None,
            "settled_amount_minor": None,
        }
        assert created.headers["location"].endswith(body["id"])
        assert created.headers["retry-after"] == "5"
        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_940
        assert balance["reserved_credits"] == 60

        rotated = public_api_auth.create_client_key(
            settings.public_api_clients_file, "partner_one", key_id="key00003"
        )
        replay = _create(
            client,
            {"Authorization": f"Bearer {rotated}"},
            video_1s,
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == body["id"]
        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_940
        assert balance["reserved_credits"] == 60

        conflict = _create(
            client,
            auth,
            video_1s,
            aspect_ratio="16:9",
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"
    app.state.pipeline.release.set()


def test_url_idempotency_binds_url_not_mutable_download_bytes(
    tmp_path, video_1s, monkeypatch
):
    settings, auth, _ = _public_setup(tmp_path)
    calls = 0

    async def fake_download(url, destination, *, max_bytes, timeout_s, resolver, transport):
        nonlocal calls
        calls += 1
        shutil.copyfile(video_1s, destination)
        if calls > 1:
            with destination.open("ab") as stream:
                stream.write(b"changed remote bytes")
        return destination.stat().st_size

    monkeypatch.setattr("app.main.download_reference_url", fake_download)
    request = {
        "source_video_url": "https://EXAMPLE.com/video.mp4#ignored",
        "aspect_ratio": "9:16",
        "resolution": "768p",
    }
    headers = {**auth, "Idempotency-Key": "urlcase01"}
    app = create_app(settings, pipeline_factory=manual_factory)
    app.state.pipeline.release.set()
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/video-generations",
            headers=headers,
            files={key: (None, value) for key, value in request.items()},
        )
        second = client.post(
            "/api/v1/video-generations",
            headers=headers,
            files={key: (None, value) for key, value in request.items()},
        )
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert calls == 2


def test_insufficient_credits_rejects_before_publish(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path, credits=0)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = _create(client, auth, video_1s)
        assert response.status_code == 402
        assert response.json()["error"]["code"] == "insufficient_credits"
    assert _store(app).list() == []


def test_concurrent_creates_cannot_overdraw_or_duplicate(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path, credits=1_000)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            same = list(pool.map(
                lambda _: _create(client, auth, video_1s, key="same0001"),
                range(2),
            ))
        assert sorted(response.status_code for response in same) == [200, 201]
        assert len({response.json()["id"] for response in same}) == 1
        assert len(_store(app).list()) == 1
    app.state.pipeline.release.set()

    second_settings, second_auth, _ = _public_setup(
        tmp_path / "second", credits=100
    )
    second_app = create_app(second_settings, pipeline_factory=manual_factory)
    with TestClient(second_app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            distinct = list(pool.map(
                lambda key: _create(client, second_auth, video_1s, key=key),
                ("distinct01", "distinct02"),
            ))
        assert sorted(response.status_code for response in distinct) == [201, 402]
        assert len(_store(second_app).list()) == 1
        current = public_credits.balance(
            second_settings.data_dir, "partner_one"
        )
        assert current == {"available": 40, "reserved": 60, "spent": 0}
    second_app.state.pipeline.release.set()


def test_failure_releases_and_unknown_keeps_reservation(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        failed = _create(client, auth, video_1s, key="request010").json()
        failed_pid = failed["id"][3:]
        _store(app).update(
            failed_pid,
            status="failed",
            project_progress={"status": "failed", "percent": 40, "stage": "pipeline_failed"},
            generation={"status": "failed", "stage": "pipeline_failed", "error": "pipeline_failed"},
            error={"code": "pipeline_failed", "message": "流水线执行失败"},
        )
        failed_view = client.get(
            f"/api/v1/video-generations/{failed['id']}", headers=auth
        ).json()
        assert failed_view["status"] == "failed"
        assert failed_view["billing"]["settled_credits"] == 0

        unknown = _create(client, auth, video_1s, key="request011").json()
        unknown_pid = unknown["id"][3:]
        _store(app).update(
            unknown_pid,
            status="running",
            project_progress={"status": "running", "percent": 55, "stage": "picsart"},
            generation={"status": "running", "stage": "picsart", "error": None},
            picsart_submission_receipts={
                "version": 2,
                "segments": {
                    "1": [
                        {
                            "segment_index": 1,
                            "run_identity": "run-1",
                            "request_sha256": "0" * 64,
                            "preparation_sha256": "1" * 64,
                            "status": "unknown",
                            "task_id": None,
                        }
                    ]
                },
            },
        )
        unknown_view = client.get(
            f"/api/v1/video-generations/{unknown['id']}", headers=auth
        ).json()
        assert unknown_view["status"] == "submission_unknown"
        assert unknown_view["error"]["code"] == "submission_outcome_unknown"
        assert unknown_view["billing"]["settlement_status"] == "pending"
        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_940
        assert balance["reserved_credits"] == 60
    app.state.pipeline.release.set()


def test_pipeline_failure_is_terminal_and_releases_reservation(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s, key="pipelinefail01").json()
        _store(app).update(
            created["id"][3:],
            status="failed",
            project_progress={"status": "failed", "percent": 24, "stage": "pipeline_failed"},
            generation={"status": "failed", "stage": "pipeline_failed", "error": "prompt_fusion_output_invalid private detail"},
            error={"code": "pipeline_failed", "message": "prompt_fusion_output_invalid private detail"},
        )

        view = client.get(
            f"/api/v1/video-generations/{created['id']}", headers=auth
        )

    assert view.status_code == 200
    assert view.json()["status"] == "failed"
    assert view.json()["error"] == {
        "code": "generation_failed",
        "message": "视频生成失败",
        "retryable": False,
    }
    assert "private detail" not in str(view.json())
    assert public_credits.balance(settings.data_dir, "partner_one") == {
        "available": 5_000,
        "reserved": 0,
        "spent": 0,
    }
    app.state.pipeline.release.set()


@pytest.mark.parametrize(
    ("meta", "expected_code"),
    [
        (
            {"status": "failed", "error": {"code": "audio_required", "message": "x"}},
            "video_audio_required",
        ),
        (
            {
                "status": "failed",
                "error": {"code": "long_video_duration_exceeded", "message": "x"},
            },
            "video_duration_exceeds_h3_limit",
        ),
        (
            {
                "status": "failed",
                "error": {"code": "long_video_invalid_duration", "message": "x"},
            },
            "video_duration_invalid",
        ),
        (
            {
                "status": "failed",
                "error": {
                    "code": "long_video_no_safe_dialogue_boundary",
                    "message": "x",
                },
            },
            "video_duration_exceeds_h3_limit",
        ),
        (
            {
                "status": "failed",
                "error": {"code": "pipeline_failed", "message": "x"},
                "generation": {"status": "failed", "error": "picsart_submit_rejected"},
            },
            "video_rejected_by_provider",
        ),
        (
            {"status": "failed", "error": {"code": "private_node token=secret", "message": "secret"}},
            "generation_failed",
        ),
    ],
)
def test_failed_job_error_exposes_only_actionable_codes(meta, expected_code):
    error = public_api._failed_job_error(meta)
    assert error.code == expected_code
    assert "private_node" not in error.message
    assert "secret" not in error.message


def test_success_captures_credits_and_content_supports_head_and_range(
    tmp_path, video_1s
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s).json()
        pid = created["id"][3:]
        shutil.copyfile(
            video_1s, _store(app).path(pid) / "output" / "generated.mp4"
        )
        _store(app).update(
            pid,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
            generation={"status": "succeeded", "stage": "completed", "error": None},
        )
        view = client.get(
            f"/api/v1/video-generations/{created['id']}", headers=auth
        )
        assert view.status_code == 200
        body = view.json()
        assert body["status"] == "succeeded"
        assert body["billing"]["settled_credits"] == 60
        assert body["result"]["video"]["expires_at"] is None

        content_url = body["result"]["video"]["content_url"]
        head = client.head(content_url, headers=auth)
        assert head.status_code == 200
        assert head.headers["accept-ranges"] == "bytes"
        assert int(head.headers["content-length"]) == len(video_1s.read_bytes())
        ranged = client.get(content_url, headers={**auth, "Range": "bytes=0-9"})
        assert ranged.status_code == 206
        assert ranged.content == video_1s.read_bytes()[:10]
        assert ranged.headers["content-range"].startswith("bytes 0-9/")
        invalid = client.get(content_url, headers={**auth, "Range": "bytes=999999-"})
        assert invalid.status_code == 416
        assert invalid.json()["error"]["code"] == "invalid_range"

        not_ready = _create(client, auth, video_1s, key="request020").json()
        early = client.get(
            f"/api/v1/video-generations/{not_ready['id']}/content", headers=auth
        )
        assert early.status_code == 409
        assert early.json()["error"]["code"] == "result_not_ready"

        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_880
        assert balance["reserved_credits"] == 60
        assert balance["spent_credits"] == 60
    app.state.pipeline.release.set()


def test_public_openapi_contains_only_public_contract(tmp_path):
    settings, _auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        schema = client.get("/api/v1/openapi.json").json()
    assert schema["info"]["title"] == "Duet Video Generation API"
    assert "/api/v1/video-generations" in schema["paths"]
    assert "/api/v1/account/credits" in schema["paths"]
    assert not any(path.startswith("/api/conversations") for path in schema["paths"])
    assert "multipart/form-data" in schema["paths"]["/api/v1/video-generations"]["post"]["requestBody"]["content"]
    create_schema = schema["paths"]["/api/v1/video-generations"]["post"][
        "requestBody"
    ]["content"]["multipart/form-data"]["schema"]
    component_name = create_schema["$ref"].rsplit("/", 1)[-1]
    create_properties = schema["components"]["schemas"][component_name][
        "properties"
    ]
    assert create_properties["aspect_ratio"]["enum"] == ["9:16", "16:9"]
    assert create_properties["resolution"]["enum"] == ["768p", "480p"]
    assert "dialogue_mode" in create_properties
    assert "dialogue_brief" in create_properties
    (limit_parameter,) = schema["paths"][
        "/api/v1/account/credit-transactions"
    ]["get"]["parameters"]
    assert limit_parameter["name"] == "limit"
    assert limit_parameter["schema"]["type"] == "integer"


def test_create_rejects_whitespace_language_without_reserving(
    tmp_path, video_1s
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = _create(
            client,
            auth,
            video_1s,
            key="blanklang01",
            target_language=" \t ",
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_target_language"
    assert _store(app).list() == []
    assert public_credits.balance(settings.data_dir, "partner_one")[
        "reserved"
    ] == 0


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("aspect_ratio", "1:1", "invalid_aspect_ratio"),
        ("resolution", "720p", "invalid_resolution"),
    ],
)
def test_create_uses_specific_parameter_error_codes(
    tmp_path, video_1s, field, value, code
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = _create(
            client,
            auth,
            video_1s,
            key=f"invalid{field[:3]}01",
            **{field: value},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("source", "video/mp4"),
        ("clip.avi", "video/quicktime"),
        ("source", "video/webm"),
        ("clip.MP4", "application/octet-stream"),
    ],
)
def test_create_accepts_source_video_via_filename_or_content_type(
    tmp_path, video_1s, filename, content_type
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create_upload(
            client,
            auth,
            video_1s,
            key="uploadkind1",
            filename=filename,
            content_type=content_type,
        )

    assert created.status_code == 201


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("clip.avi", "video/x-msvideo"),
        ("source", "application/octet-stream"),
        ("source", None),
    ],
)
def test_create_rejects_source_video_when_filename_and_content_type_both_unknown(
    tmp_path, video_1s, filename, content_type
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = _create_upload(
            client,
            auth,
            video_1s,
            key="uploadkind1",
            filename=filename,
            content_type=content_type,
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_source_media_type"
    assert response.json()["error"]["field"] == "source_video"
    assert _store(app).list() == []


def test_create_requires_multipart_and_wraps_malformed_boundary(tmp_path):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    headers = {**auth, "Idempotency-Key": "contenttype01"}
    with TestClient(app) as client:
        wrong_type = client.post(
            "/api/v1/video-generations", headers=headers, json={}
        )
        malformed = client.post(
            "/api/v1/video-generations",
            headers={**headers, "Content-Type": "multipart/form-data"},
            content=b"not-multipart",
        )

    assert wrong_type.status_code == 415
    assert wrong_type.json()["error"]["code"] == "unsupported_content_type"
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_multipart"


def test_public_http_errors_use_the_v1_error_envelope(tmp_path):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.post("/api/v1/account/credits", headers=auth)

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"
    assert response.headers["allow"] == "GET"


def test_public_api_base_404_uses_the_v1_error_envelope(tmp_path):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.get("/api/v1", headers=auth)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "endpoint_not_found"


def test_private_source_url_is_rejected_before_download(tmp_path, monkeypatch):
    settings, auth, _ = _public_setup(tmp_path)

    async def unexpected_download(*_args, **_kwargs):
        raise AssertionError("private URL must not reach the downloader")

    monkeypatch.setattr("app.main.download_reference_url", unexpected_download)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video-generations",
            headers={**auth, "Idempotency-Key": "privateurl01"},
            files={"source_video_url": (None, "https://localhost/video.mp4")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_source_video_url"


def test_html_source_url_fails_creation_with_invalid_source_video_url(tmp_path, monkeypatch):
    from app.safe_download import ReferenceDownloadError

    settings, auth, _ = _public_setup(tmp_path)

    async def html_download(*_args, **_kwargs):
        raise ReferenceDownloadError(
            "reference_not_video",
            "链接解析到的不是视频（视频可能不存在、设为私密、需要登录或受地区限制），请检查链接或直接上传视频文件",
            status_code=422,
        )

    monkeypatch.setattr("app.main.download_reference_url", html_download)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video-generations",
            headers={**auth, "Idempotency-Key": "htmlurl0001"},
            files={"source_video_url": (None, "https://m.tiktok.com/v/7678258555106905365.html")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_source_video_url"
    assert "不是视频" in response.json()["error"]["message"]
    assert response.json()["error"]["field"] == "source_video_url"


@pytest.mark.parametrize("limit", ["", "0", "101", "-1", "+1", "01", "1.0", " 1 "])
def test_credit_transaction_limit_is_strict(tmp_path, limit):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/account/credit-transactions",
            headers=auth,
            params={"limit": limit},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_limit"


def test_credit_transactions_list_reserve_and_adjustment(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s, key="ledger001").json()
        response = client.get("/api/v1/account/credit-transactions", headers=auth)

    assert response.status_code == 200
    rows = response.json()["data"]
    assert rows[0]["id"] == f"job:{created['id']}:reserve"
    assert rows[0]["type"] == "reserve"
    assert rows[0]["credits"] == 60
    assert rows[0]["job_id"] == created["id"]
    assert any(
        row["type"] == "adjustment" and row["direction"] == "credit"
        for row in rows
    )
    app.state.pipeline.release.set()


def test_unknown_and_duplicate_query_parameters_are_rejected(tmp_path):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        unknown = client.get(
            "/api/v1/account/credits?extra=1", headers=auth
        )
        duplicate = client.get(
            "/api/v1/account/credit-transactions?limit=1&limit=2",
            headers=auth,
        )

    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "invalid_query_parameters"
    assert duplicate.status_code == 422
    assert duplicate.json()["error"]["code"] == "invalid_query_parameters"


def test_bearer_scheme_is_case_insensitive(tmp_path):
    settings, _auth, api_key = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/account/credits",
            headers={"Authorization": f"bearer {api_key}"},
        )

    assert response.status_code == 200


def test_download_leases_are_shared_and_bounded(tmp_path):
    first = public_api._acquire_download_lease(tmp_path, "partner_one")
    second = public_api._acquire_download_lease(tmp_path, "partner_one")
    assert first is not None
    assert second is not None
    try:
        assert public_api._acquire_download_lease(tmp_path, "partner_one") is None
    finally:
        first.release()
        second.release()
    replacement = public_api._acquire_download_lease(tmp_path, "partner_one")
    assert replacement is not None
    replacement.release()


def test_terminal_credit_history_survives_project_retention_cleanup(
    tmp_path, video_1s
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s, key="retentioncleanup01").json()
        pid = created["id"][3:]
        _store(app).update(
            pid,
            status="failed",
            project_progress={"status": "failed", "percent": 10, "stage": "pipeline_failed"},
            generation={"status": "failed", "stage": "pipeline_failed", "error": "pipeline_failed"},
            error={"code": "pipeline_failed", "message": "流水线执行失败"},
        )
        settled = client.get(
            f"/api/v1/video-generations/{created['id']}", headers=auth
        )
        assert settled.status_code == 200
        assert settled.json()["billing"]["settlement_status"] == "final"

    before = public_credits.balance(settings.data_dir, "partner_one")
    shutil.rmtree(settings.data_dir / "projects" / pid)

    assert public_credits.balance(settings.data_dir, "partner_one") == before
    assert before == {"available": 5_000, "reserved": 0, "spent": 0}
    app.state.pipeline.release.set()


def test_admin_adjustments_are_idempotent_and_never_overdraw(tmp_path):
    data_dir = tmp_path / "data"
    assert public_credits.adjust(
        data_dir,
        "partner_one",
        2_000,
        reason="manual grant",
        idempotency_key="adjust001",
    )
    assert not public_credits.adjust(
        data_dir,
        "partner_one",
        2_000,
        reason="manual grant",
        idempotency_key="adjust001",
    )
    assert public_credits.balance(data_dir, "partner_one")["available"] == 2_000
    try:
        public_credits.adjust(
            data_dir,
            "partner_one",
            -2_001,
            reason="manual debit",
            idempotency_key="adjust002",
        )
    except public_credits.CreditError as exc:
        assert exc.code == "insufficient_available_credits"
    else:
        raise AssertionError("overdraw must fail")


def test_capabilities_expose_dialogue_generate_mode(tmp_path):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        body = client.get(
            "/api/v1/video-generations/capabilities", headers=auth
        ).json()

    assert body["defaults"]["dialogue_mode"] == "auto_rewrite"
    assert body["dialogue"] == {
        "modes": ["auto_rewrite", "generate"],
        "default_mode": "auto_rewrite",
        "brief": {
            "required_for": "generate",
            "max_chars": 1000,
            "unit": "utf16_code_units",
        },
    }


def test_create_defaults_to_auto_rewrite_with_source_language(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(client, auth, video_1s, key="dialoguedef01")
        assert created.status_code == 201
        pid = created.json()["id"][3:]
        state = client.get(f"/api/conversations/{pid}", headers=AUTH).json()

    assert state["effective_request"]["dialogue"] == {
        "mode": "auto_rewrite",
        "target_language": "与原视频相同",
    }
    app.state.pipeline.release.set()


def test_create_generate_mode_maps_brief_into_generation_request(
    tmp_path, video_1s
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(
            client,
            auth,
            video_1s,
            key="dialoguegen01",
            target_language=" English ",
            dialogue_mode="generate",
            dialogue_brief=" 用轻松口吻介绍新产品 ",
        )
        assert created.status_code == 201
        body = created.json()
        assert body["parameters"]["target_language"] == "English"
        pid = body["id"][3:]
        state = client.get(f"/api/conversations/{pid}", headers=AUTH).json()

    assert state["effective_request"]["dialogue"] == {
        "mode": "generate",
        "target_language": "English",
        "brief": "用轻松口吻介绍新产品",
    }
    app.state.pipeline.release.set()


@pytest.mark.parametrize(
    ("data", "code", "field"),
    [
        ({"dialogue_mode": "generate"}, "invalid_dialogue_brief", "dialogue_brief"),
        (
            {"dialogue_mode": "generate", "dialogue_brief": "   "},
            "invalid_dialogue_brief",
            "dialogue_brief",
        ),
        (
            {"dialogue_brief": "介绍产品"},
            "invalid_dialogue_brief",
            "dialogue_brief",
        ),
        (
            {"dialogue_mode": "auto_rewrite", "dialogue_brief": "介绍产品"},
            "invalid_dialogue_brief",
            "dialogue_brief",
        ),
        (
            {"dialogue_mode": "manual", "dialogue_brief": "介绍产品"},
            "invalid_dialogue_mode",
            "dialogue_mode",
        ),
        (
            {"dialogue_mode": "generate", "dialogue_brief": "长" * 1001},
            "dialogue_brief_too_long",
            "dialogue_brief",
        ),
        (
            {"dialogue_mode": "generate", "dialogue_brief": "\U0001F600" * 501},
            "dialogue_brief_too_long",
            "dialogue_brief",
        ),
    ],
)
def test_dialogue_mode_validation_is_strict_and_reserve_free(
    tmp_path, video_1s, data, code, field
):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        response = _create(client, auth, video_1s, key="dialoguebad01", **data)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == code
    assert error["field"] == field
    assert _store(app).list() == []
    assert public_credits.balance(settings.data_dir, "partner_one")["reserved"] == 0


def test_idempotency_key_binds_dialogue_brief(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    app = create_app(settings, pipeline_factory=manual_factory)
    with TestClient(app) as client:
        created = _create(
            client,
            auth,
            video_1s,
            key="dialogueidem01",
            dialogue_mode="generate",
            dialogue_brief="介绍产品",
        )
        assert created.status_code == 201
        conflict = _create(
            client,
            auth,
            video_1s,
            key="dialogueidem01",
            dialogue_mode="generate",
            dialogue_brief="换一份台词",
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"
        replay = _create(
            client,
            auth,
            video_1s,
            key="dialogueidem01",
            dialogue_mode="generate",
            dialogue_brief="介绍产品",
        )
        assert replay.status_code == 200
        assert replay.json()["id"] == created.json()["id"]
    app.state.pipeline.release.set()
