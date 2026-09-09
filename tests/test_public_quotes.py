from __future__ import annotations

import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import public_api_auth, public_credits, public_quotes
from app.config import Settings
from app.main import create_app


BASE = "/api/v1/video-generations/quotes"


class RecordingPipeline:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, project_id):
        self.calls.append(project_id)
        self.started.set()
        self.release.wait(timeout=30)

    def validate_r2v_resume(self, project_id):
        raise ValueError("no frozen H3 inputs")

    def validate_pre_picsart_resume(self, project_id):
        raise ValueError("no reusable upstream inputs")


@pytest.fixture
def quote_video(tmp_path):
    path = tmp_path / "source.mp4"
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-f", "lavfi", "-i",
         "color=c=blue:s=64x64:d=1:r=10", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture
def quote_env(tmp_path):
    registry = tmp_path / "clients.json"
    key = public_api_auth.create_client_key(registry, "quote_owner", key_id="key00001")
    settings = Settings(
        access_token="private-test-secret", data_dir=tmp_path / "data",
        public_base_url="https://studio.example", store_lock_enabled=False,
        public_api_enabled=True, public_api_clients_file=registry,
    )
    public_credits.adjust(settings.data_dir, "quote_owner", 5000,
                          reason="test grant", idempotency_key="grant0001")
    app = create_app(settings, pipeline_factory=lambda *_: RecordingPipeline())
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            yield app, client, {"Authorization": f"Bearer {key}"}, settings
        finally:
            app.state.pipeline.release.set()


def quote(client, auth, video, *, key="quote0001", **data):
    with video.open("rb") as stream:
        return client.post(BASE, headers={**auth, "Idempotency-Key": key},
                           data={"aspect_ratio": "9:16", "resolution": "768p", **data},
                           files={"source_video": ("source.mp4", stream, "video/mp4")})


def confirm(client, auth, quote_id, **body):
    return client.post(f"{BASE}/{quote_id}/confirm", headers=auth,
                       json=body or {"confirm": True})


def record_path(settings, quote_id):
    paths = list(settings.data_dir.glob("projects/.public-quotes/*/quote.json"))
    matching = [p for p in paths if json.loads(p.read_text()).get("quote_id") == quote_id]
    assert len(matching) == 1, paths
    return matching[0]


def assert_idle(app, settings):
    assert app.state.store.list() == []
    assert app.state.pipeline.calls == []
    assert public_credits.balance(settings.data_dir, "quote_owner")["reserved"] == 0


def test_quote_probes_rounds_and_freezes_without_starting(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    monkeypatch.setattr("app.public_api.media.probe", lambda _: {"duration_s": 24.37})
    response = quote(client, auth, quote_video)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["duration_seconds"] == 24.37
    assert body["billable_seconds"] == 25
    assert body["estimated_credits"] == 1500
    assert body["credits_per_second"] == 60
    assert body["job_id"] is None
    assert body["status"] == "quoted"
    assert_idle(app, settings)
    fetched = client.get(f"{BASE}/{body['quote_id']}", headers=auth)
    assert fetched.status_code == 200
    assert fetched.json() == body
    duplicate = quote(client, auth, quote_video)
    assert duplicate.status_code == 200
    assert duplicate.json()["quote_id"] == body["quote_id"]
    changed = quote(client, auth, quote_video, aspect_ratio="16:9")
    assert changed.status_code == 409
    assert_idle(app, settings)


def test_confirm_uses_frozen_url_bytes_and_price(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    calls = []

    async def download(url, destination, **kwargs):
        calls.append(url)
        shutil.copyfile(quote_video, destination)
        return destination.stat().st_size

    monkeypatch.setattr("app.main.download_reference_url", download)
    response = client.post(BASE, headers={**auth, "Idempotency-Key": "urlquote01"},
                           files={"source_video_url": (None, "https://example.com/source.mp4")})
    assert response.status_code == 201, response.text
    body = response.json()
    assert_idle(app, settings)

    async def changed_remote(*args, **kwargs):
        pytest.fail("confirmation must not download the URL again")

    monkeypatch.setattr("app.main.download_reference_url", changed_remote)
    created = confirm(client, auth, body["quote_id"])
    assert created.status_code == 201, created.text
    assert created.json()["billing"]["quoted_credits"] == body["estimated_credits"] == 60
    project_id = created.json()["id"].removeprefix("vg_")
    assert (app.state.store.path(project_id) / "inputs/source.mp4").read_bytes() == quote_video.read_bytes()
    assert calls == ["https://example.com/source.mp4"]
    assert app.state.pipeline.started.wait(2)
    replay = confirm(client, auth, body["quote_id"])
    assert replay.status_code == 200
    assert replay.json()["id"] == created.json()["id"]
    assert app.state.pipeline.calls == [project_id]


def test_quote_owner_and_rotation_scope(quote_env, quote_video):
    app, client, auth, settings = quote_env
    body = quote(client, auth, quote_video).json()
    other = public_api_auth.create_client_key(settings.public_api_clients_file,
                                              "other_owner", key_id="key00002")
    other_auth = {"Authorization": f"Bearer {other}"}
    assert client.get(f"{BASE}/{body['quote_id']}", headers=other_auth).status_code == 404
    assert confirm(client, other_auth, body["quote_id"]).status_code == 404
    assert_idle(app, settings)
    rotated = public_api_auth.create_client_key(settings.public_api_clients_file,
                                                "quote_owner", key_id="key00003")
    assert confirm(client, {"Authorization": f"Bearer {rotated}"}, body["quote_id"]).status_code == 201


@pytest.mark.parametrize("body", [{}, {"confirm": False}, {"confirm": "true"},
                                    {"confirm": 1}, {"confirm": True, "estimated_credits": 1},
                                    {"confirm": True, "aspect_ratio": "16:9"}])
def test_confirmation_requires_explicit_true_without_mutations(quote_env, quote_video, body):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    response = client.post(f"{BASE}/{quote_id}/confirm", headers=auth, json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_confirm_request"
    assert_idle(app, settings)


def test_concurrent_confirm_creates_reserves_and_starts_once(quote_env, quote_video):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: confirm(client, auth, quote_id), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 201], [r.text for r in responses]
    assert len({r.json()["id"] for r in responses}) == 1
    assert len(app.state.store.list()) == 1
    assert app.state.pipeline.started.wait(2)
    assert len(app.state.pipeline.calls) == 1
    assert public_credits.balance(settings.data_dir, "quote_owner") == {
        "available": 4940, "reserved": 60, "spent": 0}


def test_insufficient_balance_does_not_consume_quote(quote_env, quote_video):
    app, client, auth, settings = quote_env
    public_credits.adjust(settings.data_dir, "quote_owner", -5000,
                          reason="use funds", idempotency_key="debit0001")
    response = quote(client, auth, quote_video)
    assert response.status_code == 201
    quote_id = response.json()["quote_id"]
    assert confirm(client, auth, quote_id).status_code == 402
    assert_idle(app, settings)
    public_credits.adjust(settings.data_dir, "quote_owner", 60,
                          reason="top up", idempotency_key="topup0001")
    assert confirm(client, auth, quote_id).status_code == 201


def test_expired_quote_rejected_but_confirmed_quote_remains_idempotent(quote_env, quote_video):
    app, client, auth, settings = quote_env
    expired_id = quote(client, auth, quote_video).json()["quote_id"]
    path = record_path(settings, expired_id)
    body = json.loads(path.read_text())
    body["expires_epoch"] = 1
    path.write_text(json.dumps(body))
    assert confirm(client, auth, expired_id).status_code == 410
    assert_idle(app, settings)
    live_id = quote(client, auth, quote_video, key="quote0002").json()["quote_id"]
    created = confirm(client, auth, live_id)
    assert created.status_code == 201
    path = record_path(settings, live_id)
    body = json.loads(path.read_text())
    body["expires_epoch"] = 1
    path.write_text(json.dumps(body))
    replay = confirm(client, auth, live_id)
    assert replay.status_code == 200
    assert replay.json()["id"] == created.json()["id"]


def test_failed_publish_cannot_reuse_orphan_reserve_to_overdraw(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    original = app.state.store.create_from_staging

    def crash(*args, **kwargs):
        raise OSError("simulated crash before publication")

    monkeypatch.setattr(app.state.store, "create_from_staging", crash)
    failed = confirm(client, auth, quote_id)
    assert failed.status_code >= 500
    assert_idle(app, settings)
    public_credits.adjust(settings.data_dir, "quote_owner", -5000,
                          reason="funds used after crash", idempotency_key="debit0001")
    monkeypatch.setattr(app.state.store, "create_from_staging", original)
    assert confirm(client, auth, quote_id).status_code == 402
    assert_idle(app, settings)
    public_credits.adjust(settings.data_dir, "quote_owner", 60,
                          reason="top up", idempotency_key="topup0001")
    assert confirm(client, auth, quote_id).status_code == 201


def test_frozen_media_tampering_fails_before_start(quote_env, quote_video):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    directory = record_path(settings, quote_id).parent
    candidates = [p for p in directory.rglob("*") if p.is_file() and p.read_bytes() == quote_video.read_bytes()]
    assert len(candidates) == 1
    candidates[0].write_bytes(b"not the quoted media")
    response = confirm(client, auth, quote_id)
    assert response.status_code >= 400
    assert_idle(app, settings)


def test_unconfirmed_quote_survives_application_restart(quote_env, quote_video):
    app, client, auth, settings = quote_env
    quoted = quote(client, auth, quote_video)
    assert quoted.status_code == 201
    quote_id = quoted.json()["quote_id"]
    assert_idle(app, settings)
    restarted = create_app(settings, pipeline_factory=lambda *_: RecordingPipeline())
    with TestClient(restarted) as fresh_client:
        try:
            fetched = fresh_client.get(f"{BASE}/{quote_id}", headers=auth)
            assert fetched.status_code == 200
            assert fetched.json()["estimated_credits"] == quoted.json()["estimated_credits"]
            created = confirm(fresh_client, auth, quote_id)
            assert created.status_code == 201, created.text
            assert restarted.state.pipeline.started.wait(2)
            assert len(restarted.state.pipeline.calls) == 1
        finally:
            restarted.state.pipeline.release.set()


def test_existing_quote_honors_frozen_price_after_rate_change(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    frozen = quote(client, auth, quote_video).json()
    monkeypatch.setattr(public_credits, "CREDITS_PER_SECOND", 120)
    monkeypatch.setattr(public_credits, "PRICE_VERSION", "test-new-rate-v2")
    refreshed = quote(client, auth, quote_video, key="newprice01")
    assert refreshed.status_code == 201
    assert refreshed.json()["estimated_credits"] == 120
    assert refreshed.json()["credits_per_second"] == 120
    assert refreshed.json()["price_version"] == "test-new-rate-v2"
    created = confirm(client, auth, frozen["quote_id"])
    assert created.status_code == 201, created.text
    billing = created.json()["billing"]
    assert billing["quoted_credits"] == frozen["estimated_credits"] == 60
    assert billing["price_version"] == frozen["price_version"]


@pytest.mark.parametrize("raw", [b'{"confirm":true,"confirm":true}',
                                 b'{"confirm":false,"confirm":true}',
                                 b'{"confirm":', b'null', b'[]'])
def test_confirmation_rejects_malformed_or_duplicate_json(quote_env, quote_video, raw):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    response = client.post(f"{BASE}/{quote_id}/confirm",
                           headers={**auth, "Content-Type": "application/json"}, content=raw)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_confirm_request"
    assert_idle(app, settings)


def test_confirmation_requires_json_content_type(quote_env, quote_video):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    response = client.post(f"{BASE}/{quote_id}/confirm", headers=auth, data={"confirm": "true"})
    assert response.status_code == 415, response.text
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert_idle(app, settings)


def test_confirmation_rejects_oversized_body_without_starting(quote_env, quote_video):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    response = client.post(f"{BASE}/{quote_id}/confirm",
                           headers={**auth, "Content-Type": "application/json"},
                           content=b'{"confirm":true}' + b" " * 1024)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_confirm_request"
    assert_idle(app, settings)


def test_cleanup_preserves_live_media_and_keeps_expired_receipt(quote_env, quote_video):
    app, client, auth, settings = quote_env
    expired_id = quote(client, auth, quote_video).json()["quote_id"]
    live_id = quote(client, auth, quote_video, key="livequote01").json()["quote_id"]
    expired_path = record_path(settings, expired_id)
    expired = json.loads(expired_path.read_text())
    expired["expires_epoch"] = 1
    expired_path.write_text(json.dumps(expired))
    store = public_quotes.QuoteStore(app.state.store.root)
    with store.lock():
        store.cleanup()
    assert expired_path.is_file()
    assert not (expired_path.parent / "media").exists()
    live_media = record_path(settings, live_id).parent / "media" / "source.mp4"
    assert live_media.read_bytes() == quote_video.read_bytes()
    assert confirm(client, auth, live_id).status_code == 201


def test_internal_web_creation_does_not_enter_quote_or_credit_gate(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env

    def forbidden(*args, **kwargs):
        pytest.fail("internal creation must not invoke public quote or billing")

    monkeypatch.setattr(public_quotes.QuoteStore, "create", forbidden)
    monkeypatch.setattr(public_credits, "reserve", forbidden)
    generation_request = {
        "version": 1,
        "output": {"aspect_ratio": "9:16", "resolution": "768p", "fit_mode": "auto"},
        "processing": {"optimize_image": True, "remove_subtitle": True, "remove_logo": True},
        "dialogue": {"mode": "auto_rewrite", "target_language": "与原视频相同"},
        "replacement_guidance": None,
    }
    with quote_video.open("rb") as stream:
        response = client.post("/api/conversations",
                               headers={"Authorization": "Bearer private-test-secret"},
                               data={"client_request_id": "internal0001",
                                     "generation_request": json.dumps(generation_request)},
                               files={"file": ("source.mp4", stream, "video/mp4")})
    assert response.status_code == 201, response.text
    assert app.state.pipeline.started.wait(2)
    assert len(app.state.pipeline.calls) == 1
    assert public_credits.balance(settings.data_dir, "quote_owner")["reserved"] == 0
    assert list(settings.data_dir.glob("projects/.public-quotes/*/quote.json")) == []


def test_retry_after_publication_failure_dispatches_original_job_once(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    original = app.state.store.create_from_staging

    def publish_then_crash(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("simulated response loss after publication before dispatch")

    monkeypatch.setattr(app.state.store, "create_from_staging", publish_then_crash)
    failed = confirm(client, auth, quote_id)
    assert failed.status_code >= 500
    assert len(app.state.store.list()) == 1
    assert app.state.pipeline.calls == []
    monkeypatch.setattr(app.state.store, "create_from_staging", original)
    retry = confirm(client, auth, quote_id)
    assert retry.status_code == 200, retry.text
    assert app.state.pipeline.started.wait(2)
    replay = confirm(client, auth, quote_id)
    assert replay.status_code == 200
    assert replay.json()["id"] == retry.json()["id"]
    assert len(app.state.pipeline.calls) == 1
    assert len(app.state.store.list()) == 1
    assert public_credits.balance(settings.data_dir, "quote_owner")["reserved"] == 60


@pytest.mark.parametrize("status", ["queued", "running"])
def test_real_lifespan_preserves_internal_restart_failure_gate(quote_env, status):
    app, _client, _auth, settings = quote_env
    project_id, _path, _state = app.state.store.create({}, f"internal-{status}")
    if status == "running":
        app.state.store.update(project_id, status="running")
    restarted = create_app(settings, pipeline_factory=lambda *_: RecordingPipeline())
    with TestClient(restarted):
        try:
            state = restarted.state.store.read(project_id)
            assert state["status"] == "failed"
            assert state["error"]["code"] == "process_restarted"
            assert restarted.state.pipeline.calls == []
        finally:
            restarted.state.pipeline.release.set()


def test_real_lifespan_recovers_published_quote_before_dispatch(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    quote_id = quote(client, auth, quote_video).json()["quote_id"]
    original = app.state.store.create_from_staging

    def publish_then_crash(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("simulated process death before dispatch")

    monkeypatch.setattr(app.state.store, "create_from_staging", publish_then_crash)
    assert confirm(client, auth, quote_id).status_code >= 500
    assert app.state.pipeline.calls == []
    restarted = create_app(settings, pipeline_factory=lambda *_: RecordingPipeline())
    with TestClient(restarted) as fresh_client:
        try:
            assert restarted.state.pipeline.started.wait(2)
            replay = confirm(fresh_client, auth, quote_id)
            assert replay.status_code == 200, replay.text
            assert restarted.state.pipeline.calls == [replay.json()["id"].removeprefix("vg_")]
            assert len(restarted.state.store.list()) == 1
            assert public_credits.balance(settings.data_dir, "quote_owner")["reserved"] == 60
        finally:
            restarted.state.pipeline.release.set()


def test_quote_rejects_video_longer_than_generation_limit(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    monkeypatch.setattr("app.public_api.media.probe", lambda _: {"duration_s": 300.01})
    response = quote(client, auth, quote_video)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "video_duration_invalid"
    assert_idle(app, settings)


def test_url_quote_replay_survives_expired_remote_url(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    calls = []

    async def download(url, destination, **kwargs):
        calls.append(url)
        shutil.copyfile(quote_video, destination)
        return destination.stat().st_size

    monkeypatch.setattr("app.main.download_reference_url", download)
    headers = {**auth, "Idempotency-Key": "urlretry01"}
    fields = {"source_video_url": (None, "https://example.com/temporary.mp4")}
    original = client.post(BASE, headers=headers, files=fields)
    assert original.status_code == 201, original.text

    async def expired_remote(*args, **kwargs):
        raise OSError("remote signed URL expired after first request")

    monkeypatch.setattr("app.main.download_reference_url", expired_remote)
    replay = client.post(BASE, headers=headers, files=fields)
    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    assert len(calls) == 1
    changed = client.post(BASE, headers=headers,
                          files={**fields, "aspect_ratio": (None, "16:9")})
    assert changed.status_code == 409, changed.text
    assert_idle(app, settings)


def test_url_quote_replay_binds_replacement_image_bytes(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env

    async def download(url, destination, **kwargs):
        shutil.copyfile(quote_video, destination)
        return destination.stat().st_size

    def image_bytes(color):
        return subprocess.run(
            ["/usr/bin/ffmpeg", "-f", "lavfi", "-i", f"color=c={color}:s=16x16",
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            check=True, capture_output=True,
        ).stdout

    monkeypatch.setattr("app.main.download_reference_url", download)
    headers = {**auth, "Idempotency-Key": "urlimage01"}
    fields = {"source_video_url": (None, "https://example.com/temporary.mp4"),
              "replacement_instruction": (None, "Replace the bag using the image"),
              "replacement_image": ("replacement.png", image_bytes("red"), "image/png")}
    original = client.post(BASE, headers=headers, files=fields)
    assert original.status_code == 201, original.text

    async def expired_remote(*args, **kwargs):
        raise OSError("original URL expired")

    monkeypatch.setattr("app.main.download_reference_url", expired_remote)
    replay = client.post(BASE, headers=headers, files=fields)
    assert replay.status_code == 200, replay.text
    assert replay.json() == original.json()
    changed = client.post(BASE, headers=headers, files={
        **fields, "replacement_image": ("replacement.png", image_bytes("blue"), "image/png")})
    assert changed.status_code == 409, changed.text
    assert_idle(app, settings)


def test_one_quote_recovery_write_failure_does_not_block_other_startup(quote_env, quote_video, monkeypatch):
    app, client, auth, settings = quote_env
    ids = [quote(client, auth, quote_video, key=key).json()["quote_id"]
           for key in ("recover001", "recover002")]
    original_create = app.state.store.create_from_staging
    published = []

    def publish_then_crash(*args, **kwargs):
        result = original_create(*args, **kwargs)
        published.append(result[0])
        raise OSError("process death before dispatch")

    monkeypatch.setattr(app.state.store, "create_from_staging", publish_then_crash)
    for quote_id in ids:
        assert confirm(client, auth, quote_id).status_code >= 500
    restarted = create_app(settings, pipeline_factory=lambda *_: RecordingPipeline())
    original_update = restarted.state.store.update

    def fail_one_recovery(project_id, **changes):
        if project_id == published[0] and changes.get("status") == "queued":
            raise OSError("simulated per-project recovery write failure")
        return original_update(project_id, **changes)

    monkeypatch.setattr(restarted.state.store, "update", fail_one_recovery)
    with TestClient(restarted) as fresh_client:
        try:
            assert fresh_client.get("/api/v1/account/credits", headers=auth).status_code == 200
            assert restarted.state.pipeline.started.wait(2)
            assert restarted.state.pipeline.calls == [published[1]]
            assert restarted.state.store.read(published[0])["status"] == "failed"
        finally:
            restarted.state.pipeline.release.set()
