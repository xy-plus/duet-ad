import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings
from app import h3, prepared_input, storage
from app.main import _result_fields, _resume_generation, create_app


PROMPT = "镜头从整洁的房间缓慢推进。"
REQUEST_ID = "request-123456"


def _png(path: Path, width: int = 90, height: int = 160, value: int = 127) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded.tobytes()
    path.write_bytes(data)
    return data


def _make_conv(settings, *, fit_required=False, duration_s=9.2, status="done"):
    meta = storage.new_conversation(settings.data_dir, "n", "a.mp4")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    (cdir / "source.mp4").write_bytes(b"source-video")
    original = _png(
        cdir / "work" / "keyframes" / "01.png",
        160 if fit_required else 90,
        90 if fit_required else 160,
    )
    (cdir / "work" / "visual_prompt.txt").write_text(PROMPT, encoding="utf-8")
    (cdir / "work" / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    storage.update_meta(
        settings.data_dir,
        cid,
        status=status,
        duration_s=duration_s,
        source_width=160 if fit_required else 90,
        source_height=90 if fit_required else 160,
        fit_required=fit_required,
        keyframes=["01.png"],
        prompt=PROMPT,
        voice_lines=[],
        voice_line_provenance=[],
    )
    return cid, original


def _write_initial_receipt(settings, cid):
    cdir = settings.data_dir / cid
    frozen = prepared_input.write_prepared_input(
        root=cdir,
        source=cdir / "source.mp4",
        audio=None,
        keyframes=[cdir / "work" / "keyframes" / "01.png"],
        visual=cdir / "work" / "visual_prompt.txt",
        final=cdir / "work" / "prompt.txt",
        dialogue_mode="auto",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=9.2,
        ratio="9:16",
        fit_mode="none",
        engine_request={"fixture": "initial"},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        prompt=frozen.prompt_text,
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
    )
    return frozen


def _payload(request_id=REQUEST_ID, *, mode="none", fit="none", lines=None):
    payload = {
        "confirm": True,
        "client_request_id": request_id,
        "dialogue_mode": mode,
        "fit_mode": fit,
    }
    if lines is not None:
        payload["lines"] = lines
    return payload


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="art-test-secret",
    )
    with TestClient(create_app(settings)) as client:
        yield settings, client


def test_source_prompt_can_be_cas_edited_before_h3(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    initial = _write_initial_receipt(settings, cid)
    replacement = "镜头改为从宠物眼睛高度缓慢向前移动。"
    response = client.patch(
        f"/api/conversations/{cid}/prompt",
        headers=AUTH,
        json={
            "confirm": True,
            "expected_sha256": initial.visual_prompt.sha256,
            "prompt": replacement,
        },
    )
    assert response.status_code == 200
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    submitted = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()
    )
    assert submitted.status_code == 202
    assert seen[0].prompt.startswith(replacement)


def test_source_prompt_is_frozen_once_h3_session_exists(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    initial = _write_initial_receipt(settings, cid)
    root = settings.data_dir / cid / ".h3"
    root.mkdir()
    (root / "session.json").write_text("{}", encoding="utf-8")
    response = client.patch(
        f"/api/conversations/{cid}/prompt",
        headers=AUTH,
        json={"confirm": True, "expected_sha256": initial.visual_prompt.sha256, "prompt": "不得写入"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "prompt_frozen"}


def test_disabled_is_501_before_lookup(client):
    response = client.post(
        f"/api/conversations/{'0' * 32}/submit", headers=AUTH, json={"confirm": True}
    )
    assert response.status_code == 501


def test_submit_requires_auth(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    assert client.post(f"/api/conversations/{cid}/submit", json={}).status_code == 401


def test_submit_requires_only_autodl_credential(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="art")
    cid, _ = _make_conv(settings)
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert response.status_code == 202
    assert seen and seen[0].autodl_token == "art"


def test_missing_autodl_credential_is_503(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True, autodl_art_token="")
    cid, _ = _make_conv(settings)
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert response.status_code == 503
    assert response.json() == {"detail": "h3_credentials_missing"}


def test_context_ir_field_is_rejected_from_submit_contract(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={**_payload(), "context_ir_enabled": False},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_submit_request"}


def test_submit_freezes_direct_h3_receipt_and_source_prompt(enabled, monkeypatch):
    settings, client = enabled
    cid, original = _make_conv(settings, duration_s=12.4)
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert response.status_code == 202
    (request,) = seen
    assert request.prompt.startswith(PROMPT)
    assert "无台词" in request.prompt
    assert request.duration == 13
    assert request.keyframes[0][1] == original
    meta = storage.load_meta(settings.data_dir, cid)
    cdir = settings.data_dir / cid
    frozen = prepared_input.load_prepared_input(
        cdir, cdir / prepared_input.RECEIPT_FILENAME, expected_dialogue=meta["prepared_dialogue"]
    )
    assert set(frozen.engine_request) == {"h3"}
    assert frozen.engine_request["h3"] == {
        "workflow": h3.H3_WORKFLOW,
        "duration": 13,
        "resolution": h3.H3_RESOLUTION,
    }
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert "context_ir_enabled" not in detail["generation"]
    assert detail["generation"]["stage"] == "h3"


@pytest.mark.parametrize(
    "change,detail",
    [
        ({"confirm": False}, "confirmation required"),
        ({"client_request_id": "bad"}, "invalid_client_request_id"),
        ({"dialogue_mode": "unknown"}, "invalid_dialogue"),
        ({"fit_mode": "crop"}, "fit_mode_not_allowed"),
    ],
)
def test_submit_validates_confirmation_id_dialogue_and_fit(enabled, change, detail):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={**_payload(), **change},
    )
    assert response.status_code in {409, 422}
    assert response.json() == {"detail": detail}


def test_fit_required_forces_crop_or_pad(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings, fit_required=True)
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert response.status_code == 422
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload(fit="crop")
    )
    assert response.status_code == 202
    assert seen and seen[0].keyframes[0][0].parent.name == "crop"


def test_custom_dialogue_is_frozen_and_exposed(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    lines = [{"text": "自定义台词", "start_s": 0, "end_s": 1.5}]
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(mode="custom", lines=lines),
    )
    assert response.status_code == 202
    assert seen[0].voice_texts == ("自定义台词",)
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["dialogue"]["lines"] == lines


def test_failed_attempt_requires_new_id_and_uses_retry(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    calls = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda request, request_id: calls.append((request, request_id))
        or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()).status_code == 202
    same = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert same.status_code == 409
    newer = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload("request-654321"),
    )
    assert newer.status_code == 202
    assert calls and calls[0][1] == "request-654321"


def test_resume_required_reuses_same_id_and_receipt(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    seen = []
    def start(request):
        seen.append(request)
        if len(seen) == 1:
            return h3.H3Result(
                "retryable_failure", "000001", retryable=True, error_code="h3_query_failed"
            )
        return h3.H3Result("failed", "000001", error_code="h3_provider_failed")

    monkeypatch.setattr(h3, "start", start)
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()).status_code == 202
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["status"] == "resume_required"
    wrong = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload("request-654321")
    )
    assert wrong.status_code == 409
    resumed = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert resumed.status_code == 202
    assert len(seen) == 2
    assert all(request.client_request_id == REQUEST_ID for request in seen)


@pytest.mark.parametrize("status", ["queued", "running", "succeeded"])
def test_active_or_succeeded_generation_cannot_be_duplicated(enabled, status):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": status,
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
    )
    same = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert same.status_code in {202, 409}
    different = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload("request-654321")
    )
    assert different.status_code == 409


def test_result_fields_only_contains_direct_h3_states():
    assert _result_fields(h3.H3Result("succeeded", "000001")) == ("succeeded", None)
    assert _result_fields(h3.H3Result("h3_running", "000001")) == (
        "resume_required", "h3_running"
    )
    assert _result_fields(
        h3.H3Result("failed", "000001", error_code="h3_provider_failed")
    ) == ("failed", "h3_provider_failed")


def test_legacy_context_ir_attempt_is_exposed_as_direct_retry(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "resume_required",
            "error": "ready_for_h3",
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "context_ir",
            "context_ir_enabled": True,
        },
    )
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"] == {
        "status": "failed",
        "error": "generation_path_removed",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    seen = []
    monkeypatch.setattr(
        h3,
        "retry",
        lambda request, request_id: seen.append((request, request_id))
        or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload("request-654321"),
    )
    assert response.status_code == 202
    assert seen and seen[0][1] == "request-654321"


def test_startup_resume_uses_direct_h3_state(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art-test-secret"
    )
    cid, _ = _make_conv(settings)
    request = []
    monkeypatch.setattr(
        h3,
        "resume",
        lambda value: request.append(value) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    client_request_id = REQUEST_ID
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={"status": "running", "error": None, "attempt": 1, "client_request_id": client_request_id, "stage": "h3"},
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        fit_mode="none",
        dialogue_mode="none",
    )
    cdir = settings.data_dir / cid
    prepared_input.write_prepared_input(
        root=cdir,
        source=cdir / "source.mp4",
        audio=None,
        keyframes=[cdir / "work" / "keyframes" / "01.png"],
        visual=cdir / "work" / "visual_prompt.txt",
        final=cdir / "work" / "prompt.txt",
        dialogue_mode="none",
        dialogue=(),
        vocal_filter_enabled=True,
        duration_s=9.2,
        ratio="9:16",
        fit_mode="none",
        engine_request={"h3": {"workflow": h3.H3_WORKFLOW, "duration": 10, "resolution": h3.H3_RESOLUTION}},
    )
    _resume_generation(settings, cid)
    assert request and request[0].client_request_id == REQUEST_ID
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["generation"]["status"] == "failed"
