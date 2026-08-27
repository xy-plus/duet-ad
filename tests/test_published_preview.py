import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main, published_preview
from app.config import Settings


CID = "a" * 32


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _source_project(root: Path) -> dict:
    context_receipt = (
        root
        / "work/h3-native/.context-ir/attempts/000001/receipt.json"
    )
    files = {
        "source.mp4": b"source",
        "generated.mp4": b"published-video",
        "prepared_input.json": b"prepared",
        "work/multimodal_input.json": b"multimodal-input",
        "work/h3_multimodal_source.json": b"multimodal-source",
        "work/h3_prompt_plan.json": b"prompt-plan",
        "work/h3-native/.context-ir/attempts/000001/attempt.json": b"context-attempt",
        "work/h3-native/.context-ir/attempts/000001/receipt.json": b"context-receipt",
        "stitch-receipt.json": b"stitch",
        "work/keyframes/01.png": b"raw-1",
        "work/keyframes/02.png": b"raw-2",
        "work/postprocessed/01.png": b"optimized-1",
        "work/postprocessed/02.png": b"optimized-2",
    }
    for relative, data in files.items():
        _write(root / relative, data)
    h3_attempt = {
        "h3": {
            "output": {
                "media_timeline": {
                    "schema": "duet.h3.media_timeline",
                    "version": 1,
                    "decode_complete": True,
                    "video": {"duration_s": 4.0},
                    "audio": {"duration_s": 4.0},
                }
            }
        }
    }
    _write(
        root / "work/h3-native/.h3/attempts/000002/attempt.json",
        json.dumps(h3_attempt, sort_keys=True).encode(),
    )
    meta = {
        "schema_version": 2,
        "id": CID,
        "title": "preview",
        "note": "read only",
        "status": "done",
        "error": None,
        "created_at": "2026-08-27T00:00:00+00:00",
        "updated_at": "2026-08-27T00:00:00+00:00",
        "keyframes": ["01.png", "02.png"],
        "postprocess": {"status": "done", "frames": ["01.png", "02.png"]},
        "prepared_input_receipt": "prepared_input.json",
        "generation": {
            "status": "succeeded",
            "stage": "stitch",
            "h3_attempt_id": "000002",
            "context_ir": {
                "status": "succeeded",
                "attempt_id": "000001",
                "receipt_path": str(context_receipt),
            },
        },
    }
    _write(root / "meta.json", json.dumps(meta, sort_keys=True).encode())
    return meta


def _published_copy(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    source = tmp_path / "authoritative" / CID
    target = tmp_path / "published" / CID
    source_meta = _source_project(source)
    shutil.copytree(source, target)
    target_meta = {**source_meta, "title": "published preview", "note": "frozen"}
    receipt = published_preview._build(
        source_root=source,
        target_root=target,
        target_meta=target_meta,
    )
    published_preview.write(target, receipt)
    target_meta["published_preview_receipt"] = published_preview.RECEIPT_FILENAME
    _write(target / "meta.json", json.dumps(target_meta, sort_keys=True).encode())
    return source, target, target_meta, receipt


def test_builder_derives_complete_sorted_chain_and_public_artifacts(tmp_path):
    _source, _target, _meta, receipt = _published_copy(tmp_path)

    receipt_entries = receipt["binding"]["source"]["receipts"]
    assert [item["path"] for item in receipt_entries] == sorted(
        item["path"] for item in receipt_entries
    )
    assert {item["kind"] for item in receipt_entries} == {
        "prepared_input",
        "multimodal_input",
        "multimodal_source",
        "h3_prompt_plan",
        "context_ir_attempt",
        "context_ir_receipt",
        "h3_attempt",
        "stitch_receipt",
    }
    assert [item["path"] for item in receipt["binding"]["target"]["artifacts"]] == [
        "generated.mp4",
        "source.mp4",
        "work/keyframes/01.png",
        "work/keyframes/02.png",
        "work/postprocessed/01.png",
        "work/postprocessed/02.png",
    ]


def test_loader_rejects_artifact_tamper_and_symlink(tmp_path):
    _source, target, meta, _receipt = _published_copy(tmp_path)
    assert published_preview.load(target, meta)[0].name == CID

    (target / "generated.mp4").write_bytes(b"tampered")
    with pytest.raises(published_preview.PublishedPreviewError):
        published_preview.load(target, meta)

    (target / "generated.mp4").unlink()
    (target / "generated.mp4").symlink_to(target / "source.mp4")
    with pytest.raises(published_preview.PublishedPreviewError):
        published_preview.load(target, meta)


def test_main_validates_source_chain_before_build_and_on_every_load(tmp_path, monkeypatch):
    source, target, meta, receipt = _published_copy(tmp_path)
    settings = Settings(access_token="secret", data_dir=target.parent)
    calls = []

    def validate_source(source_settings, source_meta):
        calls.append((source_settings.data_dir, source_meta["id"]))
        return source_settings.data_dir == source.parent

    monkeypatch.setattr(main, "_validate_generated_video_uncached", validate_source)
    rebuilt = main.build_published_preview_receipt(
        settings,
        source_root=source,
        target_root=target,
        target_meta=meta,
    )
    assert rebuilt["binding"] == receipt["binding"]
    assert main._has_valid_generated_video(settings, meta) is True
    assert calls == [(source.parent, CID), (source.parent, CID)]

    source.joinpath("stitch-receipt.json").write_bytes(b"changed")
    assert main._has_valid_generated_video(settings, meta) is False

    monkeypatch.setattr(
        main, "_validate_generated_video_uncached", lambda *_args: False
    )
    with pytest.raises(published_preview.PublishedPreviewError):
        main.build_published_preview_receipt(
            settings,
            source_root=source,
            target_root=target,
            target_meta=meta,
        )


def test_published_preview_is_read_only_on_every_mutating_route(
    tmp_path, monkeypatch
):
    source, target, meta, _receipt = _published_copy(tmp_path)
    ordinary_cid = "b" * 32
    ordinary_meta = dict(meta)
    ordinary_meta["id"] = ordinary_cid
    ordinary_meta.pop("published_preview_receipt")
    _write(
        target.parent / ordinary_cid / "meta.json",
        json.dumps(ordinary_meta, sort_keys=True).encode(),
    )
    settings = Settings(
        access_token="secret",
        data_dir=target.parent,
        enable_h3_submit=False,
        enable_mediakit_erase=True,
    )
    monkeypatch.setattr(
        main,
        "_validate_generated_video_uncached",
        lambda source_settings, source_meta: source_settings.data_dir == source.parent,
    )
    headers = {"Authorization": "Bearer secret"}
    with TestClient(main.create_app(settings)) as client:
        detail = client.get(f"/api/conversations/{CID}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["read_only"] is True
        assert detail.json()["has_video"] is True
        assert detail.json()["navigation_status"] == "completed"

        requests = (
            client.post(
                f"/api/conversations/{CID}/submit",
                headers=headers,
                json={},
            ),
            client.patch(
                f"/api/conversations/{CID}/prompt",
                headers=headers,
                json={"confirm": True, "expected_sha256": "x", "prompt": "x"},
            ),
            client.patch(
                f"/api/conversations/{CID}/image-optimization-prompt",
                headers=headers,
                json={
                    "confirm": True,
                    "segment_index": 0,
                    "expected_sha256": "x",
                    "prompt": "x",
                },
            ),
            client.post(
                f"/api/conversations/{CID}/postprocess",
                headers=headers,
                json={},
            ),
            client.post(
                f"/api/conversations/{CID}/postprocess/segments/0/retry",
                headers=headers,
                json={},
            ),
        )
        ordinary_submit = client.post(
            f"/api/conversations/{ordinary_cid}/submit",
            headers=headers,
            json={},
        )
    assert [(response.status_code, response.json()) for response in requests] == [
        (409, {"detail": "read_only"}),
        (409, {"detail": "read_only"}),
        (409, {"detail": "read_only"}),
        (409, {"detail": "read_only"}),
        (409, {"detail": "read_only"}),
    ]
    assert ordinary_submit.status_code == 501
    assert ordinary_submit.json() == {"detail": "H3 submission is disabled."}
