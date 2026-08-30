import json
import threading

import pytest
from fastapi.testclient import TestClient

from app import generation_config, long_generation, main as main_module, postprocess, storage
from app.main import (
    _automatic_postprocess_request,
    _postprocess_matches_automatic_request,
    create_app,
)
from conftest import AUTH, make_settings


CUSTOM = {
    "optimize_image": False,
    "remove_subtitle": True,
    "remove_watermark": True,
}


def _post(client, video, *, request_id="preflight-config-1", config=None):
    data = {"client_request_id": request_id}
    if config is not None:
        data["generation_config"] = (
            config if isinstance(config, str) else json.dumps(config)
        )
    with open(video, "rb") as stream:
        return client.post(
            "/api/conversations",
            headers=AUTH,
            files={"file": ("clip.mp4", stream, "video/mp4")},
            data=data,
        )


def test_create_defaults_are_frozen_in_meta_and_work_receipt(
    client, settings, video_1s,
):
    response = _post(client, video_1s)
    assert response.status_code == 201
    cid = response.json()["id"]
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["generation_config"] == generation_config.DEFAULTS
    assert meta["generation_config_sha256"] == generation_config.sha256(
        generation_config.DEFAULTS
    )
    receipt = json.loads(
        (settings.data_dir / cid / "work" / "generation-config.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt == generation_config.receipt(generation_config.DEFAULTS)
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation_config"] == generation_config.DEFAULTS
    assert detail["generation_config_sha256"] == meta[
        "generation_config_sha256"
    ]


def test_create_custom_config_maps_watermark_to_internal_brand(
    client, settings, video_1s,
):
    response = _post(client, video_1s, config=CUSTOM)
    assert response.status_code == 201
    cid = response.json()["id"]
    meta = storage.load_meta(settings.data_dir, cid)
    request = _automatic_postprocess_request(settings.data_dir / cid, meta)
    assert request == {
        "confirm": True,
        "options": {
            "optimize_image": False,
            "remove_subtitle": True,
            "remove_brand": True,
        },
    }
    assert _postprocess_matches_automatic_request(
        {"status": "running", "options": request["options"]}, request
    )
    assert not _postprocess_matches_automatic_request(
        {
            "status": "running",
            "options": {**request["options"], "remove_brand": False},
        },
        request,
    )


def test_create_generation_config_is_strict_allowlist(
    client, settings, video_1s,
):
    invalid = [
        "not-json",
        {},
        {**generation_config.DEFAULTS, "remove_brand": False},
        {"optimize_image": True},
        {**generation_config.DEFAULTS, "remove_subtitle": 1},
    ]
    for index, value in enumerate(invalid):
        response = _post(
            client,
            video_1s,
            request_id=f"invalid-config-{index}",
            config=value,
        )
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid_generation_config"}
    assert not settings.data_dir.exists() or not list(settings.data_dir.iterdir())


def test_client_request_id_compares_frozen_generation_config(
    client, video_1s,
):
    first = _post(client, video_1s, config=CUSTOM)
    assert first.status_code == 201
    replay = _post(client, video_1s, config=CUSTOM)
    assert replay.status_code == 200
    assert replay.json()["id"] == first.json()["id"]
    conflict = _post(client, video_1s, config=generation_config.DEFAULTS)
    assert conflict.status_code == 409
    assert conflict.json() == {
        "detail": "client_request_id_generation_config_conflict"
    }


def test_restart_resolves_same_receipt_and_rejects_corruption(
    tmp_path, video_1s,
):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        cid = _post(client, video_1s, config=CUSTOM).json()["id"]
    with TestClient(create_app(settings)):
        meta = storage.load_meta(settings.data_dir, cid)
        assert generation_config.resolve(settings.data_dir / cid, meta) == CUSTOM
    receipt_path = settings.data_dir / cid / "work" / "generation-config.json"
    receipt_path.write_text("{}", encoding="utf-8")
    meta = storage.load_meta(settings.data_dir, cid)
    assert generation_config.resolve(settings.data_dir / cid, meta) is None
    storage.mutate_meta(
        settings.data_dir,
        cid,
        lambda current: (
            current.pop("generation_config", None),
            current.pop("generation_config_sha256", None),
        ),
    )
    meta = storage.load_meta(settings.data_dir, cid)
    assert generation_config.is_frozen(settings.data_dir / cid, meta)
    assert generation_config.resolve(settings.data_dir / cid, meta) is None


def test_all_disabled_legally_skips_postprocess_and_keeps_original_authority(
    settings,
):
    disabled = {
        "optimize_image": False,
        "remove_subtitle": False,
        "remove_watermark": False,
    }
    meta = storage.new_conversation(
        settings.data_dir,
        "",
        "clip.mp4",
        generation_config=disabled,
    )
    request = _automatic_postprocess_request(
        settings.data_dir / meta["id"], meta
    )
    assert request["options"] == {
        "optimize_image": False,
        "remove_subtitle": False,
        "remove_brand": False,
    }
    assert not any(request["options"].values())


def test_capability_get_declares_exact_create_contract(client):
    response = client.get("/api/capabilities", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {
        "dialogue_review": {
            "supported": True,
            "create_field": "dialogue_review_policy",
            "policies": ["auto_continue", "review_required"],
            "default": "auto_continue",
            "commit_path": "/api/conversations/{id}/dialogue-review/commit",
        },
        "generation_config": {
            "supported": True,
            "create_field": "generation_config",
            "encoding": "multipart_json",
            "fields": {
                "optimize_image": "boolean",
                "remove_subtitle": "boolean",
                "remove_watermark": "boolean",
            },
            "defaults": generation_config.DEFAULTS,
        }
    }


@pytest.mark.parametrize(
    ("config", "expected_events"),
    [
        (CUSTOM, ["start", "run", "generation"]),
        (
            {
                "optimize_image": False,
                "remove_subtitle": False,
                "remove_watermark": False,
            },
            ["generation"],
        ),
    ],
)
def test_startup_recovery_reuses_frozen_config_without_reordering(
    tmp_path, monkeypatch, config, expected_events,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "",
        "clip.mp4",
        generation_config=config,
    )
    cid = meta["id"]
    storage.update_meta(settings.data_dir, cid, status="done")
    events = []
    completed = threading.Event()

    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: object())

    async def start(_settings, called_cid, payload, _locks):
        events.append("start")
        assert called_cid == cid
        assert payload == _automatic_postprocess_request(
            settings.data_dir / cid,
            storage.load_meta(settings.data_dir, cid),
        )
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={
                "status": "running",
                "options": payload["options"],
            },
        )

    async def run(*_args, **_kwargs):
        events.append("run")
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "done"},
        )

    def generate(*_args):
        events.append("generation")
        completed.set()

    monkeypatch.setattr(postprocess, "start", start)
    monkeypatch.setattr(postprocess, "run_task", run)
    monkeypatch.setattr(
        postprocess,
        "image_acceptance_status",
        lambda *_args: {"required": False, "accepted": False},
    )
    monkeypatch.setattr(
        main_module, "_start_automatic_v4_generation", generate
    )
    with TestClient(create_app(settings)):
        assert completed.wait(2)
    assert events == expected_events


def test_successful_image_retry_reenters_the_same_operation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "",
        "clip.mp4",
        generation_config=generation_config.DEFAULTS,
    )
    cid = meta["id"]
    options = generation_config.postprocess_options(generation_config.DEFAULTS)
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        _postprocess_receipt={"version": 4, "options": options},
        postprocess={
            "status": "failed",
            "error": "provider_rejected",
            "options": options,
            "segments": [{
                "index": 1,
                "status": "failed",
                "error": "provider_rejected",
                "revision": 3,
            }],
        },
    )
    events = []

    async def retry(_settings, called_cid, index, payload, _locks):
        assert (called_cid, index) == (cid, 1)
        assert payload == {"confirm": True, "expected_revision": 3}
        events.append("retry")
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "running", "error": None},
        )

    async def run(_settings, called_cid, _media, _seedream, only, **_kwargs):
        assert (called_cid, only) == (cid, {1})
        events.append("run")
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "done", "error": None},
        )

    def generate(*_args):
        events.append("generation")

    monkeypatch.setattr(postprocess, "retry_segment", retry)
    monkeypatch.setattr(postprocess, "run_task", run)
    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: object())
    monkeypatch.setattr(
        postprocess,
        "image_acceptance_status",
        lambda *_args: {"required": False, "accepted": False},
    )
    monkeypatch.setattr(main_module, "_start_automatic_v4_generation", generate)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/postprocess/segments/1/retry",
            headers=AUTH,
            json={"confirm": True, "expected_revision": 3},
        )
    assert response.status_code == 202
    assert events == ["retry", "run", "generation"]


def test_failed_image_retry_does_not_advance_the_operation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "",
        "clip.mp4",
        generation_config=generation_config.DEFAULTS,
    )
    cid = meta["id"]
    options = generation_config.postprocess_options(generation_config.DEFAULTS)
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        _postprocess_receipt={"version": 4, "options": options},
        postprocess={
            "status": "failed",
            "error": "provider_rejected",
            "options": options,
            "segments": [{
                "index": 1,
                "status": "failed",
                "error": "provider_rejected",
                "revision": 1,
            }],
        },
    )

    async def retry(_settings, _cid, _index, _payload, _locks):
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "running", "error": None},
        )

    async def run(*_args, **_kwargs):
        current = storage.load_meta(settings.data_dir, cid)["postprocess"]
        storage.update_meta(
            settings.data_dir,
            cid,
            postprocess={**current, "status": "failed", "error": "provider_rejected"},
        )

    monkeypatch.setattr(postprocess, "retry_segment", retry)
    monkeypatch.setattr(postprocess, "run_task", run)
    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: object())
    monkeypatch.setattr(
        main_module,
        "_start_automatic_v4_generation",
        lambda *_args: pytest.fail("failed image retry must not advance"),
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/postprocess/segments/1/retry",
            headers=AUTH,
            json={"confirm": True, "expected_revision": 1},
        )
    assert response.status_code == 202
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["status"] == "failed"


def test_image_acceptance_reenters_the_same_operation(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "",
        "clip.mp4",
        generation_config=generation_config.DEFAULTS,
    )
    cid = meta["id"]
    options = generation_config.postprocess_options(generation_config.DEFAULTS)
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        _postprocess_receipt={"version": 4, "options": options},
        postprocess={"status": "done", "error": None, "options": options},
    )
    events = []

    def accept(_settings, called_cid, payload):
        assert called_cid == cid
        assert payload == {"confirm": True}
        events.append("accept")
        return {"required": True, "accepted": True}

    def acceptance_status(*_args):
        return {"required": True, "accepted": bool(events)}

    def generate(*_args):
        events.append("generation")

    monkeypatch.setattr(postprocess, "accept_images", accept)
    monkeypatch.setattr(postprocess, "image_acceptance_status", acceptance_status)
    monkeypatch.setattr(long_generation, "plan_receipt", lambda *_args: object())
    monkeypatch.setattr(main_module, "_start_automatic_v4_generation", generate)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/image-acceptance",
            headers=AUTH,
            json={"confirm": True},
        )
    assert response.status_code == 202
    assert events == ["accept", "generation"]
