import pytest
from fastapi.testclient import TestClient

from app import main as main_module, published_preview, storage
from app.main import create_app
from conftest import AUTH, make_settings


def _persisted_terminal_conversation(settings, *, cid_title="pure-read"):
    meta = storage.new_conversation(
        settings.data_dir,
        cid_title,
        "source.mp4",
    )
    cid = meta["id"]
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        generation={
            "status": "succeeded",
            "client_request_id": "request-pure-read",
            "attempt": 1,
            "stage": "stitch",
        },
    )
    (settings.data_dir / cid / "generated.mp4").write_bytes(b"published")
    return cid


def _forbid_read_time_validation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("read routes must not perform strict output validation")

    monkeypatch.setattr(
        main_module,
        "_validate_generated_video_uncached",
        forbidden,
    )
    monkeypatch.setattr(
        main_module,
        "_generated_video_validation_fingerprint",
        forbidden,
        raising=False,
    )
    monkeypatch.setattr(storage, "probe_video", forbidden)


def test_list_detail_and_file_are_pure_reads_of_terminal_state_and_file(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    _forbid_read_time_validation(monkeypatch)

    with TestClient(create_app(settings)) as client:
        cid = _persisted_terminal_conversation(settings)

        detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
        listed = client.get("/api/conversations", headers=AUTH)
        video = client.get(
            f"/api/conversations/{cid}/files/generated.mp4",
            headers=AUTH,
        )

    assert detail.status_code == 200
    assert detail.json()["has_video"] is True
    summary = next(item for item in listed.json() if item["id"] == cid)
    assert summary["has_video"] is True
    assert video.status_code == 200
    assert video.content == b"published"


def test_read_routes_fail_closed_without_terminal_state_or_output_file(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    _forbid_read_time_validation(monkeypatch)

    with TestClient(create_app(settings)) as client:
        cid = _persisted_terminal_conversation(settings, cid_title="read-state")
        meta = storage.load_meta(settings.data_dir, cid)
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={**meta["generation"], "status": "running"},
        )

        running_detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
        running_list = client.get("/api/conversations", headers=AUTH)
        running_file = client.get(
            f"/api/conversations/{cid}/files/generated.mp4",
            headers=AUTH,
        )

        meta = storage.load_meta(settings.data_dir, cid)
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={**meta["generation"], "status": "succeeded"},
        )
        (settings.data_dir / cid / "generated.mp4").unlink()

        missing_detail = client.get(f"/api/conversations/{cid}", headers=AUTH)
        missing_list = client.get("/api/conversations", headers=AUTH)
        missing_file = client.get(
            f"/api/conversations/{cid}/files/generated.mp4",
            headers=AUTH,
        )

    assert running_detail.json()["has_video"] is False
    running_summary = next(
        item for item in running_list.json() if item["id"] == cid
    )
    assert running_summary["has_video"] is False
    assert running_file.status_code == 404

    assert missing_detail.json()["has_video"] is False
    missing_summary = next(
        item for item in missing_list.json() if item["id"] == cid
    )
    assert missing_summary["has_video"] is False
    assert missing_file.status_code == 404


def test_published_preview_builder_keeps_strict_validation_boundary(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir,
        "publish-boundary",
        "source.mp4",
    )
    source_root = settings.data_dir / meta["id"]
    target_root = tmp_path / "published" / meta["id"]
    target_root.mkdir(parents=True)
    calls = []

    def reject_source(source_settings, source_meta):
        calls.append((source_settings.data_dir, source_meta["id"]))
        return False

    monkeypatch.setattr(
        main_module,
        "_validate_generated_video_uncached",
        reject_source,
    )

    with pytest.raises(
        published_preview.PublishedPreviewError,
        match="published_preview_source_invalid",
    ):
        main_module.build_published_preview_receipt(
            settings,
            source_root=source_root,
            target_root=target_root,
            target_meta=meta,
        )

    assert calls == [(settings.data_dir, meta["id"])]
