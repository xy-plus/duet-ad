import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app import downloader, public_api_auth, public_credits, storage
from app.config import Settings
from app.main import create_app
from conftest import TOKEN, make_settings


def _public_setup(tmp_path, *, owner="partner_one", credits=5_000):
    registry = tmp_path / "public-api-clients.json"
    api_key = public_api_auth.create_client_key(registry, owner, key_id="key00001")
    settings = make_settings(
        tmp_path,
        public_api_enabled=True,
        public_api_clients_file=registry,
        enable_pipeline=False,
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


def test_public_api_is_disabled_by_default(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/v1/openapi.json").status_code == 404


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
    with TestClient(create_app(settings)) as client:
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
    with TestClient(create_app(settings)) as client:
        created = _create(client, auth, video_1s)
        assert created.status_code == 201
        body = created.json()
        assert body["status"] == "queued"
        assert body["parameters"] == {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "target_language": None,
            "replacement_image": False,
        }
        assert body["billing"] == {
            "currency": "CNY",
            "credits_per_cny": 100,
            "quoted_credits": 1000,
            "quoted_amount_minor": 1000,
            "price_version": "credits-fixed-1000-v1",
            "settlement_status": "pending",
            "settled_credits": None,
            "settled_amount_minor": None,
        }
        assert created.headers["location"].endswith(body["id"])
        assert created.headers["retry-after"] == "5"
        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_000
        assert balance["reserved_credits"] == 1_000

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
        assert balance["available_credits"] == 4_000
        assert balance["reserved_credits"] == 1_000

        conflict = _create(
            client,
            auth,
            video_1s,
            aspect_ratio="16:9",
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"


def test_url_idempotency_binds_url_not_mutable_download_bytes(
    tmp_path, video_1s, monkeypatch
):
    settings, auth, _ = _public_setup(tmp_path)
    calls = 0

    def fake_fetch(reference_url, staging, configured):
        nonlocal calls
        calls += 1
        destination = staging / "source.mp4"
        shutil.copyfile(video_1s, destination)
        if calls > 1:
            with destination.open("ab") as stream:
                stream.write(b"changed remote bytes")
        return destination

    monkeypatch.setattr(downloader, "fetch_reference", fake_fetch)
    request = {
        "source_video_url": "https://EXAMPLE.com/video.mp4#ignored",
        "aspect_ratio": "9:16",
        "resolution": "768p",
    }
    headers = {**auth, "Idempotency-Key": "urlcase01"}
    with TestClient(create_app(settings)) as client:
        first = client.post(
            "/api/v1/video-generations", headers=headers, data=request
        )
        second = client.post(
            "/api/v1/video-generations", headers=headers, data=request
        )
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert calls == 2


def test_insufficient_credits_rejects_before_publish(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path, credits=0)
    with TestClient(create_app(settings)) as client:
        response = _create(client, auth, video_1s)
        assert response.status_code == 402
        assert response.json()["error"]["code"] == "insufficient_credits"
    assert storage.list_conversations(settings.data_dir) == []


def test_concurrent_creates_cannot_overdraw_or_duplicate(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path, credits=1_000)
    with TestClient(create_app(settings)) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            same = list(pool.map(
                lambda _: _create(client, auth, video_1s, key="same0001"),
                range(2),
            ))
        assert sorted(response.status_code for response in same) == [200, 201]
        assert len({response.json()["id"] for response in same}) == 1
        assert len(storage.list_conversations(settings.data_dir)) == 1

    second_settings, second_auth, _ = _public_setup(
        tmp_path / "second", credits=1_000
    )
    with TestClient(create_app(second_settings)) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            distinct = list(pool.map(
                lambda key: _create(client, second_auth, video_1s, key=key),
                ("distinct01", "distinct02"),
            ))
        assert sorted(response.status_code for response in distinct) == [201, 402]
        assert len(storage.list_conversations(second_settings.data_dir)) == 1
        current = public_credits.balance(
            second_settings.data_dir, "partner_one"
        )
        assert current == {"available": 0, "reserved": 1_000, "spent": 0}


def test_failure_releases_and_unknown_keeps_reservation(tmp_path, video_1s):
    settings, auth, _ = _public_setup(tmp_path)
    with TestClient(create_app(settings)) as client:
        failed = _create(client, auth, video_1s, key="request010").json()
        failed_cid = failed["id"][3:]
        storage.update_meta(settings.data_dir, failed_cid, status="failed", error="pipeline_failed")
        failed_view = client.get(
            f"/api/v1/video-generations/{failed['id']}", headers=auth
        ).json()
        assert failed_view["status"] == "failed"
        assert failed_view["billing"]["settled_credits"] == 0

        unknown = _create(client, auth, video_1s, key="request011").json()
        unknown_cid = unknown["id"][3:]
        storage.update_meta(
            settings.data_dir,
            unknown_cid,
            status="done",
            generation={"status": "submission_unknown", "error": "submission_unknown"},
        )
        unknown_view = client.get(
            f"/api/v1/video-generations/{unknown['id']}", headers=auth
        ).json()
        assert unknown_view["status"] == "submission_unknown"
        assert unknown_view["billing"]["settlement_status"] == "pending"
        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_000
        assert balance["reserved_credits"] == 1_000


def test_success_captures_credits_and_content_supports_head_and_range(
    tmp_path, video_1s, monkeypatch
):
    settings, auth, _ = _public_setup(tmp_path)
    monkeypatch.setattr(
        "app.main._has_valid_generated_video",
        lambda configured, meta: (
            configured.data_dir / str(meta.get("id")) / "generated.mp4"
        ).is_file(),
    )
    with TestClient(create_app(settings)) as client:
        created = _create(client, auth, video_1s).json()
        cid = created["id"][3:]
        shutil.copyfile(video_1s, settings.data_dir / cid / "generated.mp4")
        storage.update_meta(
            settings.data_dir,
            cid,
            status="done",
            generation={"status": "succeeded"},
        )
        view = client.get(
            f"/api/v1/video-generations/{created['id']}", headers=auth
        )
        assert view.status_code == 200
        body = view.json()
        assert body["status"] == "succeeded"
        assert body["billing"]["settled_credits"] == 1_000
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

        balance = client.get("/api/v1/account/credits", headers=auth).json()
        assert balance["available_credits"] == 4_000
        assert balance["reserved_credits"] == 0
        assert balance["spent_credits"] == 1_000


def test_public_openapi_contains_only_public_contract(tmp_path):
    settings, _auth, _ = _public_setup(tmp_path)
    with TestClient(create_app(settings)) as client:
        schema = client.get("/api/v1/openapi.json").json()
    assert schema["info"]["title"] == "Duet Video Generation API"
    assert "/api/v1/video-generations" in schema["paths"]
    assert "/api/v1/account/credits" in schema["paths"]
    assert not any(path.startswith("/api/conversations") for path in schema["paths"])
    assert "multipart/form-data" in schema["paths"]["/api/v1/video-generations"]["post"]["requestBody"]["content"]


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
