import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app import h3, storage
from app.main import create_app
from conftest import AUTH, make_settings


CID = "a" * 32
REQUEST_ID = "legacy-request-1"


def _write_legacy_success(
    root,
    *,
    cid=CID,
    request_id=REQUEST_ID,
    attempt=1,
    requested_duration=13,
    video=b"legacy-published-video",
):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "generated.mp4"
    output.write_bytes(video)
    attempt_id = f"{attempt:06d}"
    path = root / ".h3" / "attempts" / attempt_id / "attempt.json"
    path.parent.mkdir(parents=True)
    state = {
        "schema_version": 1,
        "cid": cid,
        "attempt_id": attempt_id,
        "client_request_id": request_id,
        "input": {
            "prompt_sha256": "1" * 64,
            "keyframes": [],
            "voice_texts_sha256": "2" * 64,
            "request": {
                "duration": requested_duration,
                "h3_workflow": "minimax_h3_lightx2v_v5",
                "ir_model": "MiniMax-H3",
                "ratio": "9:16",
                "resolution": "768p竖",
            },
        },
        "input_receipt": "3" * 64,
        "status": "succeeded",
        "retryable": False,
        "ir": {"status": "succeeded"},
        "h3": {
            "status": "succeeded",
            "task_id": "legacy-paid-task",
            "receipt": {"legacy": True},
            "output": {
                "name": "generated.mp4",
                "sha256": hashlib.sha256(video).hexdigest(),
                "size": len(video),
            },
        },
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    return path, state


def _accept(root, **overrides):
    return h3.legacy_succeeded_output_is_valid(
        root,
        cid=overrides.pop("cid", CID),
        client_request_id=overrides.pop("client_request_id", REQUEST_ID),
        attempt=overrides.pop("attempt", 1),
        probe_timeout_s=1,
        **overrides,
    )


def test_legacy_success_accepts_exact_attempt_and_provider_clamped_duration(
    tmp_path, monkeypatch
):
    root = tmp_path / CID
    _write_legacy_success(root, requested_duration=13)
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.125)

    assert _accept(root) is True


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("cid", "b" * 32),
        ("client_request_id", "another-request"),
        ("attempt", 2),
    ],
)
def test_legacy_success_requires_exact_meta_attempt_binding(
    tmp_path, monkeypatch, override, value
):
    root = tmp_path / CID
    _write_legacy_success(root)
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.0)

    assert _accept(root, **{override: value}) is False


@pytest.mark.parametrize(
    "damage",
    [
        "missing_attempt",
        "corrupt_attempt",
        "wrong_state_cid",
        "state_failed",
        "h3_failed",
        "missing_receipt",
        "extra_receipt_field",
        "missing_video",
        "tampered_video",
    ],
)
def test_legacy_success_rejects_missing_corrupt_or_tampered_evidence(
    tmp_path, monkeypatch, damage
):
    root = tmp_path / CID
    path, state = _write_legacy_success(root)
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.0)
    if damage == "missing_attempt":
        path.unlink()
    elif damage == "corrupt_attempt":
        path.write_bytes(b"not-json")
    elif damage == "missing_video":
        (root / "generated.mp4").unlink()
    elif damage == "tampered_video":
        (root / "generated.mp4").write_bytes(b"tampered-published-video")
    else:
        if damage == "wrong_state_cid":
            state["cid"] = "b" * 32
        elif damage == "state_failed":
            state["status"] = "failed"
        elif damage == "h3_failed":
            state["h3"]["status"] = "failed"
        elif damage == "missing_receipt":
            state["h3"].pop("output")
        else:
            state["h3"]["output"]["duration"] = 10.0
        path.write_text(json.dumps(state), encoding="utf-8")

    assert _accept(root) is False


def test_legacy_success_rejects_invalid_or_transiently_unavailable_visual_probe(
    tmp_path, monkeypatch
):
    root = tmp_path / CID
    _write_legacy_success(root)
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: None)
    assert _accept(root) is False

    def unavailable(*_args):
        raise h3._ProbeUnavailable

    monkeypatch.setattr(h3, "_probe_video_duration", unavailable)
    assert _accept(root) is False


def test_current_attempt_cannot_use_legacy_compatibility(tmp_path, monkeypatch):
    root = tmp_path / CID
    path, state = _write_legacy_success(root)
    state.pop("ir")
    state["input"].pop("keyframes")
    state["input"]["images"] = []
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.0)

    assert _accept(root) is False


def test_startup_and_list_preserve_valid_legacy_success_without_provider(
    tmp_path, monkeypatch
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="unused-test-token",
    )
    meta = storage.new_conversation(
        settings.data_dir, note="legacy", orig_name="legacy.mp4"
    )
    cid = meta["id"]
    cdir = settings.data_dir / cid
    _write_legacy_success(cdir, cid=cid)
    (cdir / "prepared_input.json").write_text("{}", encoding="utf-8")
    storage.update_meta(
        settings.data_dir,
        cid,
        status="done",
        generation={
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
        prepared_input_receipt="prepared_input.json",
        prepared_dialogue=[],
        dialogue_mode="auto",
        fit_mode="none",
    )
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.0)
    monkeypatch.setattr(h3, "resume", lambda *_args: pytest.fail("no provider recovery"))

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/conversations", headers=AUTH).json()

    assert listed[0]["id"] == cid
    assert listed[0]["has_video"] is True
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"
