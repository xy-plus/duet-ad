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
    original = _png(cdir / "work" / "keyframes" / "01.png", 160 if fit_required else 90, 90 if fit_required else 160)
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


def test_source_h3_prompt_can_be_cas_edited_before_context_ir(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    initial = _write_initial_receipt(settings, cid)
    digest = initial.visual_prompt.sha256
    replacement = "镜头改为从宠物眼睛高度缓慢向前移动。"

    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    response = client.patch(
        f"/api/conversations/{cid}/prompt",
        headers=AUTH,
        json={"confirm": True, "expected_sha256": digest, "prompt": replacement},
    )

    assert detail["source_prompt"] == PROMPT
    assert detail["source_prompt_sha256"] == digest
    assert response.status_code == 200
    assert response.json()["prompt"] == replacement
    cdir = settings.data_dir / cid
    stored = storage.load_meta(settings.data_dir, cid)
    assert (cdir / "work" / "visual_prompt.txt").read_text() == replacement
    assert stored["prompt"].startswith(replacement)
    reloaded = prepared_input.load_prepared_input(
        cdir, cdir / prepared_input.RECEIPT_FILENAME, expected_dialogue=()
    )
    assert reloaded.visual_prompt.data.decode() == replacement
    assert reloaded.final_prompt.data.decode() == stored["prompt"]

    prepared = []
    monkeypatch.setattr(
        h3,
        "prepare_context_ir",
        lambda request: prepared.append(request)
        or h3.H3Result("ready_for_h3", "000001"),
    )
    started = client.post(
        f"/api/conversations/{cid}/context-ir",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert started.status_code == 202
    assert len(prepared) == 1
    assert prepared[0].prompt.startswith(replacement)


def test_source_h3_prompt_is_frozen_once_context_ir_exists(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    initial = _write_initial_receipt(settings, cid)
    h3_root = settings.data_dir / cid / ".h3"
    h3_root.mkdir()
    (h3_root / "session.json").write_text("{}", encoding="utf-8")

    response = client.patch(
        f"/api/conversations/{cid}/prompt",
        headers=AUTH,
        json={
            "confirm": True,
            "expected_sha256": initial.visual_prompt.sha256,
            "prompt": "不得写入",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "prompt_frozen"}
    assert (settings.data_dir / cid / "work" / "visual_prompt.txt").read_text() == PROMPT


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        minimax_api_key="mm-test-secret",
        autodl_art_token="art-test-secret",
    )
    with TestClient(create_app(settings)) as client:
        yield settings, client


def test_disabled_is_501_before_conversation_lookup(client):
    response = client.post(
        f"/api/conversations/{'0' * 32}/submit",
        headers=AUTH,
        json={"confirm": True},
    )
    assert response.status_code == 501
    assert response.json() == {"detail": "H3 submission is disabled."}


def test_submit_requires_auth(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    assert client.post(f"/api/conversations/{cid}/submit", json={}).status_code == 401


def test_context_ir_is_explicit_and_h3_submit_reuses_the_ready_attempt(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    prepared_requests = []
    h3_requests = []

    monkeypatch.setattr(
        h3,
        "prepare_context_ir",
        lambda request: prepared_requests.append(request)
        or h3.H3Result("ready_for_h3", "000001"),
    )
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: h3_requests.append(request)
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    payload = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }

    prepared = client.post(
        f"/api/conversations/{cid}/context-ir", headers=AUTH, json=payload
    )
    assert prepared.status_code == 202
    assert prepared_requests and h3_requests == []
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["status"] == "resume_required"
    assert detail["generation"]["error"] == "ready_for_h3"

    generated = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
    )
    assert generated.status_code == 202
    assert len(prepared_requests) == len(h3_requests) == 1
    assert prepared_requests[0].client_request_id == h3_requests[0].client_request_id


def test_context_ir_resume_stops_for_review_and_never_calls_h3(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    calls = []

    def prepare(request):
        calls.append(request)
        if len(calls) == 1:
            return h3.H3Result(
                "retryable_failure",
                "000001",
                retryable=True,
                error_code="ir_timeout",
            )
        return h3.H3Result("ready_for_h3", "000001")

    monkeypatch.setattr(h3, "prepare_context_ir", prepare)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not call H3"))
    payload = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }

    first = client.post(
        f"/api/conversations/{cid}/context-ir", headers=AUTH, json=payload
    )
    second = client.post(
        f"/api/conversations/{cid}/context-ir", headers=AUTH, json=payload
    )

    assert first.status_code == second.status_code == 202
    assert len(calls) == 2
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["status"] == "resume_required"
    assert detail["generation"]["error"] == "ready_for_h3"


def test_context_ir_patch_returns_reviewed_prompt(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    payload = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    monkeypatch.setattr(
        h3,
        "prepare_context_ir",
        lambda _request: h3.H3Result("ready_for_h3", "000001"),
    )
    assert client.post(
        f"/api/conversations/{cid}/context-ir", headers=AUTH, json=payload
    ).status_code == 202
    digest = "a" * 64
    reviewed = "reviewed prompt"
    monkeypatch.setattr(
        h3,
        "edit_context_ir",
        lambda _request, expected, prompt: h3.ContextIRSnapshot(
            "succeeded", prompt, digest
        ),
    )

    response = client.patch(
        f"/api/conversations/{cid}/context-ir",
        headers=AUTH,
        json={
            "confirm": True,
            "expected_sha256": "b" * 64,
            "prompt": reviewed,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "succeeded",
        "prompt": reviewed,
        "sha256": digest,
        "dialogue_valid": True,
    }


def test_submit_validates_confirmation_id_dialogue_and_fit(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not submit"))

    cases = [
        ({"client_request_id": REQUEST_ID, "dialogue_mode": "auto", "fit_mode": "none"}, 409),
        ({"confirm": True, "client_request_id": "short", "dialogue_mode": "auto", "fit_mode": "none"}, 422),
        ({"confirm": True, "client_request_id": REQUEST_ID, "dialogue_mode": "auto", "lines": [], "fit_mode": "none"}, 422),
        ({"confirm": True, "client_request_id": REQUEST_ID, "dialogue_mode": "custom", "lines": [], "fit_mode": "none"}, 422),
        ({"confirm": True, "client_request_id": REQUEST_ID, "dialogue_mode": "custom", "lines": [{"text": "x", "start_s": 0, "end_s": 1, "extra": True}], "fit_mode": "none"}, 422),
        ({"confirm": True, "client_request_id": REQUEST_ID, "dialogue_mode": "auto", "fit_mode": "crop"}, 422),
    ]
    for payload, status in cases:
        response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=payload)
        assert response.status_code == status, payload


def test_fit_required_forces_explicit_crop_or_pad(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings, fit_required=True)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not submit"))
    payload = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=payload)
    assert response.status_code == 422


def test_submit_uses_frozen_original_frames_never_postprocessed(enabled, monkeypatch):
    settings, client = enabled
    cid, original = _make_conv(settings, duration_s=9.2)
    cdir = settings.data_dir / cid
    _png(cdir / "work" / "postprocessed" / "01.png", value=240)
    seen = []

    def fake_start(request):
        seen.append(request)
        (request.workdir / "generated.mp4").write_bytes(b"generated")
        return h3.H3Result(status="succeeded", attempt_id="000001", output=request.workdir / "generated.mp4")

    monkeypatch.setattr(h3, "start", fake_start)
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert response.status_code == 202
    assert response.json() == {"status": "queued", "attempt": 1}
    (request,) = seen
    assert request.duration == 10
    assert request.ratio == "9:16"
    assert request.keyframes[0][1] == original
    assert request.minimax_api_key == "mm-test-secret"
    assert request.autodl_token == "art-test-secret"
    assert request.voice_receipt == h3.voice_texts_receipt(())

    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["generation"] == {
        "status": "succeeded",
        "error": None,
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    receipt = prepared_input.load_prepared_input(
        cdir,
        cdir / prepared_input.RECEIPT_FILENAME,
        expected_dialogue=(),
    )
    assert receipt.fit_mode == "none"
    assert receipt.keyframes[0].data == original


def test_fit_frames_are_derived_and_bound_to_receipt(enabled, monkeypatch):
    settings, client = enabled
    cid, original = _make_conv(settings, fit_required=True)
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "crop",
        },
    )
    assert response.status_code == 202
    derived = seen[0].keyframes[0][1]
    assert derived != original
    image = cv2.imdecode(np.frombuffer(derived, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[1] * 16 == image.shape[0] * 9
    receipt = json.loads((settings.data_dir / cid / prepared_input.RECEIPT_FILENAME).read_text())
    assert receipt["video"]["fit_mode"] == "crop"
    assert "postprocessed" not in json.dumps(receipt)


def test_custom_dialogue_is_validated_frozen_and_exposed(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings, duration_s=2.0)
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request) or h3.H3Result("failed", "000001", error_code="ir_provider_failed"),
    )
    lines = [{"text": "  hello  ", "start_s": 0, "end_s": 1.5}]
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "custom",
            "lines": lines,
            "fit_mode": "none",
        },
    )
    assert response.status_code == 202
    assert seen[0].voice_texts == ("hello",)
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["dialogue"] == {
        "mode": "custom",
        "lines": [{"text": "hello", "start_s": 0.0, "end_s": 1.5}],
        "auto_lines": [],
    }
    assert detail["generation"]["error"] == "ir_provider_failed"


def test_failed_attempt_requires_a_new_id_and_uses_retry(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    starts = []
    retries = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: starts.append(request) or h3.H3Result("failed", "000001", error_code="ir_provider_failed"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda request, request_id: retries.append((request, request_id)) or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 409
    body["client_request_id"] = "request-654321"
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    assert len(starts) == 1
    assert len(retries) == 1 and retries[0][1] == "request-654321"
    assert storage.load_meta(settings.data_dir, cid)["generation"]["attempt"] == 2


def test_submission_unknown_cannot_create_a_second_paid_attempt(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "submission_unknown",
            "error": "submission_unknown",
            "attempt": 1,
            "client_request_id": REQUEST_ID,
        },
    )
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not submit"))
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("must not retry"))

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": "request-different",
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "submission_outcome_unknown"}
    assert storage.load_meta(settings.data_dir, cid)["generation"]["attempt"] == 1


def test_state_persist_failure_with_raw_submitting_state_is_submission_unknown(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    inspected = []

    def fail_after_post(_request):
        raise h3.H3Error("state_persist_failed")

    def fake_inspect(request):
        inspected.append(request)
        return h3.H3Result("ir_submitting", "000001")

    monkeypatch.setattr(h3, "start", fail_after_post)
    monkeypatch.setattr(h3, "inspect", fake_inspect)
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation == {
        "status": "submission_unknown",
        "error": "submission_unknown",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    assert len(inspected) == 1

    body["client_request_id"] = "request-new-123"
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body)
    assert response.status_code == 409
    assert response.json() == {"detail": "submission_outcome_unknown"}


@pytest.mark.parametrize("error_code", ["state_persist_failed", "submission_unknown"])
def test_ambiguous_submit_error_stays_unknown_when_inspect_also_fails(
    enabled, monkeypatch, error_code
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    monkeypatch.setattr(
        h3, "start", lambda _request: (_ for _ in ()).throw(h3.H3Error(error_code))
    )
    monkeypatch.setattr(
        h3, "inspect", lambda _request: (_ for _ in ()).throw(h3.ReceiptError("state_invalid"))
    )
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert response.status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "submission_unknown"
    assert generation["error"] == "submission_unknown"
    assert generation["attempt"] == 1


@pytest.mark.parametrize(
    "error_code",
    [
        "ir_query_failed",
        "ir_timeout",
        "h3_query_failed",
        "h3_timeout",
        "download_failed",
        "download_dns_failed",
        "download_peer_unverified",
        "output_write_failed",
        "output_probe_failed",
    ],
)
def test_known_remote_task_failure_requires_same_attempt_resume(
    enabled, monkeypatch, error_code
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    monkeypatch.setattr(
        h3,
        "start",
        lambda _request: h3.H3Result(
            "retryable_failure", "000001", retryable=True, error_code=error_code
        ),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert response.status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "resume_required"
    assert generation["error"] == error_code
    assert generation["attempt"] == 1


@pytest.mark.parametrize(
    ("dialogue_mode", "lines", "expected_voice_texts"),
    [
        ("edit", [{"text": "修正台词", "start_s": 0, "end_s": 1}], ("修正台词",)),
        ("custom", [{"text": "自定义台词", "start_s": 0, "end_s": 1}], ("自定义台词",)),
        ("none", None, ()),
    ],
)
def test_ir_dialogue_mismatch_is_failed_and_new_id_can_rebuild_dialogue_attempt(
    enabled, monkeypatch, dialogue_mode, lines, expected_voice_texts
):
    settings, client = enabled
    cid, _ = _make_conv(settings, duration_s=2.0)
    starts = []
    retries = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: starts.append(request)
        or h3.H3Result("failed", "000001", error_code="ir_dialogue_mismatch"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda request, request_id: retries.append((request, request_id))
        or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    original = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "auto",
        "fit_mode": "none",
    }

    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=original
    ).status_code == 202
    failed_meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={**failed_meta["generation"], "status": "resume_required"},
    )
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"] == {
        "status": "failed",
        "error": "ir_dialogue_mismatch",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    same_id = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=original
    )
    assert same_id.status_code == 409
    assert same_id.json() == {"detail": "new client_request_id required"}

    unchanged_auto = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={**original, "client_request_id": "request-dialogue-auto-2"},
    )
    assert unchanged_auto.status_code == 409
    assert unchanged_auto.json() == {"detail": "ir_dialogue_correction_required"}
    assert retries == []

    changed = {
        **original,
        "client_request_id": "request-dialogue-2",
        "dialogue_mode": dialogue_mode,
    }
    if lines is not None:
        changed["lines"] = lines
    retried = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=changed
    )

    assert retried.status_code == 202
    assert len(starts) == 1
    assert len(retries) == 1
    assert retries[0][1] == "request-dialogue-2"
    assert retries[0][0].voice_texts == expected_voice_texts
    meta = storage.load_meta(settings.data_dir, cid)
    assert meta["generation"]["attempt"] == 2
    assert meta["dialogue_mode"] == dialogue_mode


@pytest.mark.parametrize("inspect_fails", [False, True])
def test_raised_known_task_query_error_cannot_open_a_paid_retry(
    enabled, monkeypatch, inspect_fails
):
    settings, client = enabled
    cid, _ = _make_conv(settings)

    def raise_query_error(_request):
        raise h3.H3Error("ir_query_failed", retryable=True)

    def inspect_state(_request):
        if inspect_fails:
            raise h3.ReceiptError("state_invalid")
        return h3.H3Result(
            "retryable_failure",
            "000001",
            retryable=True,
            error_code="ir_query_failed",
        )

    monkeypatch.setattr(h3, "start", raise_query_error)
    monkeypatch.setattr(h3, "inspect", inspect_state)
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "resume_required"
    assert generation["error"] == "ir_query_failed"
    assert generation["attempt"] == 1

    body["client_request_id"] = "request-new-456"
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body)
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_request_id_mismatch"}
    assert storage.load_meta(settings.data_dir, cid)["generation"]["attempt"] == 1


def test_unexpected_provider_exception_inspects_and_resumes_same_attempt(
    enabled, monkeypatch
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    calls = {"start": 0, "inspect": 0}

    def unexpected_then_continue(_request):
        calls["start"] += 1
        if calls["start"] == 1:
            raise RuntimeError("provider transport broke after POST")
        return h3.H3Result("h3_running", "000001")

    def inspect_running(_request):
        calls["inspect"] += 1
        return h3.H3Result("h3_running", "000001")

    monkeypatch.setattr(h3, "start", unexpected_then_continue)
    monkeypatch.setattr(h3, "inspect", inspect_running)
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("must not create a paid retry"))
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }

    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation == {
        "status": "resume_required",
        "error": "h3_running",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    assert calls == {"start": 1, "inspect": 1}

    changed_id = {**body, "client_request_id": "request-new-456"}
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=changed_id)
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_request_id_mismatch"}
    assert calls == {"start": 1, "inspect": 1}

    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body)
    assert response.status_code == 202
    assert response.json() == {"status": "queued", "attempt": 1}
    assert calls == {"start": 2, "inspect": 1}
    assert storage.load_meta(settings.data_dir, cid)["generation"]["attempt"] == 1


@pytest.mark.parametrize(
    ("inspect_result", "expected_status", "expected_error"),
    [
        (None, "submission_unknown", "submission_unknown"),
        (h3.H3Result("not_started", None), "submission_unknown", "submission_unknown"),
        (h3.H3Result("ir_submitting", "000001"), "submission_unknown", "submission_unknown"),
        (h3.H3Result("h3_submitting", "000001"), "submission_unknown", "submission_unknown"),
        (h3.H3Result("unexpected_state", "000001"), "submission_unknown", "submission_unknown"),
        (h3.H3Result("ir_running", "000001"), "resume_required", "ir_running"),
        (
            h3.H3Result(
                "retryable_failure",
                "000001",
                retryable=True,
                error_code="download_dns_failed",
            ),
            "resume_required",
            "download_dns_failed",
        ),
        (
            h3.H3Result("failed", "000001", error_code="download_invalid_video"),
            "failed",
            "download_invalid_video",
        ),
    ],
)
def test_unexpected_provider_exception_uses_only_inspected_state(
    enabled, monkeypatch, inspect_result, expected_status, expected_error
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    inspected = []

    monkeypatch.setattr(
        h3,
        "start",
        lambda _request: (_ for _ in ()).throw(RuntimeError("unexpected provider failure")),
    )

    def inspect_state(request):
        inspected.append(request)
        if inspect_result is None:
            raise RuntimeError("state unavailable")
        return inspect_result

    monkeypatch.setattr(h3, "inspect", inspect_state)
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert len(inspected) == 1
    assert generation["status"] == expected_status
    assert generation["error"] == expected_error
    assert generation["attempt"] == 1


@pytest.mark.parametrize("raw_status", ["ir_submitting", "h3_submitting"])
def test_raw_submitting_status_is_never_treated_as_safe_retry(raw_status):
    assert _result_fields(h3.H3Result(raw_status, "000001")) == (
        "submission_unknown",
        "submission_unknown",
    )


def test_resume_required_same_id_reuses_receipt_and_attempt(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    starts = []

    def ready(request):
        starts.append(request)
        return h3.H3Result("ready_for_h3", "000001")

    monkeypatch.setattr(h3, "start", ready)
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("resume must not retry"))
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "none",
        "fit_mode": "none",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    receipt_path = settings.data_dir / cid / prepared_input.RECEIPT_FILENAME
    receipt_before = receipt_path.read_bytes()
    meta_before = storage.load_meta(settings.data_dir, cid)
    assert meta_before["generation"]["status"] == "resume_required"

    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body)

    assert response.status_code == 202
    assert response.json() == {"status": "queued", "attempt": 1}
    assert len(starts) == 2
    meta_after = storage.load_meta(settings.data_dir, cid)
    assert meta_after["generation"]["status"] == "resume_required"
    assert meta_after["generation"]["attempt"] == 1
    assert receipt_path.read_bytes() == receipt_before
    assert meta_after["prepared_dialogue"] == meta_before["prepared_dialogue"]


def test_resume_required_rejects_new_id_and_frozen_parameter_drift(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings, fit_required=True)
    calls = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: calls.append(request) or h3.H3Result("ready_for_h3", "000001"),
    )
    body = {
        "confirm": True,
        "client_request_id": REQUEST_ID,
        "dialogue_mode": "custom",
        "lines": [{"text": "hello", "start_s": 0, "end_s": 1}],
        "fit_mode": "crop",
    }
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=body).status_code == 202
    receipt_path = settings.data_dir / cid / prepared_input.RECEIPT_FILENAME
    receipt_before = receipt_path.read_bytes()

    changed_id = {**body, "client_request_id": "request-other-123"}
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=changed_id)
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_request_id_mismatch"}

    changed_lines = {**body, "lines": [{"text": "changed", "start_s": 0, "end_s": 1}]}
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=changed_lines)
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}

    changed_fit = {**body, "fit_mode": "pad"}
    response = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=changed_fit)
    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}

    assert len(calls) == 1
    assert receipt_path.read_bytes() == receipt_before
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["fit_mode"] == "crop"
    assert detail["generation"]["attempt"] == 1


@pytest.mark.parametrize("status", ["queued", "running", "succeeded"])
def test_active_or_succeeded_generation_cannot_be_duplicated(enabled, monkeypatch, status):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={"status": status, "error": None, "attempt": 1, "client_request_id": REQUEST_ID},
    )
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not submit"))
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": "request-different",
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert response.status_code == 409


def test_old_session_is_read_only_for_submit(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    meta.pop("schema_version")
    (settings.data_dir / cid / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json={
            "confirm": True,
            "client_request_id": REQUEST_ID,
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "read_only"}


@pytest.mark.parametrize(
    ("resumed_result", "expected_error"),
    [
        (h3.H3Result("ready_for_h3", "000001"), "ready_for_h3"),
        (
            h3.H3Result(
                "retryable_failure",
                "000001",
                retryable=True,
                error_code="h3_timeout",
            ),
            "h3_timeout",
        ),
    ],
)
def test_startup_resumes_with_get_only_and_marks_confirmation_required(
    tmp_path, monkeypatch, resumed_result, expected_error
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        minimax_api_key="mm",
        autodl_art_token="art",
    )
    cid, _ = _make_conv(settings)
    cdir = settings.data_dir / cid
    frozen = prepared_input.write_prepared_input(
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
        engine_request={"duration": 10},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        dialogue_mode="none",
        fit_mode="none",
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        generation={"status": "running", "error": None, "attempt": 1, "client_request_id": REQUEST_ID},
    )
    resumed = []

    def fake_resume(request):
        resumed.append(request)
        return resumed_result

    monkeypatch.setattr(h3, "resume", fake_resume)
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("startup must not POST"))
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("startup must not retry"))
    with TestClient(create_app(settings)) as client:
        for thread in client.app.state.h3_resume_threads:
            thread.join(timeout=1)
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert len(resumed) == 1
    assert resumed[0].client_request_id == REQUEST_ID
    assert detail["generation"]["status"] == "resume_required"
    assert detail["generation"]["error"] == expected_error
    assert frozen.voice_texts == ()


def test_startup_unexpected_error_inspects_existing_attempt_before_retry_gate(
    tmp_path, monkeypatch
):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        minimax_api_key="mm",
        autodl_art_token="art",
    )
    cid, _ = _make_conv(settings)
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
        engine_request={"duration": 10},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        dialogue_mode="none",
        fit_mode="none",
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        generation={
            "status": "running",
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
        },
    )
    inspected = []

    monkeypatch.setattr(
        h3,
        "resume",
        lambda _request: (_ for _ in ()).throw(RuntimeError("unexpected provider failure")),
    )

    def inspect_running(request):
        inspected.append(request)
        return h3.H3Result("h3_running", "000001")

    monkeypatch.setattr(h3, "inspect", inspect_running)
    _resume_generation(settings, cid)

    assert len(inspected) == 1
    assert storage.load_meta(settings.data_dir, cid)["generation"] == {
        "status": "resume_required",
        "error": "h3_running",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
    }


def _make_running_generation(settings):
    cid, _ = _make_conv(settings)
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
        engine_request={"duration": 10},
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        dialogue_mode="none",
        fit_mode="none",
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        generation={
            "status": "running",
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
        },
    )
    return cid


def test_startup_missing_credentials_locks_existing_attempt(tmp_path):
    settings = make_settings(tmp_path, enable_h3_submit=True)
    cid = _make_running_generation(settings)

    _resume_generation(settings, cid)

    assert storage.load_meta(settings.data_dir, cid)["generation"] == {
        "status": "submission_unknown",
        "error": "submission_unknown",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
    }


def test_startup_invalid_receipt_locks_existing_attempt(tmp_path):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        minimax_api_key="mm",
        autodl_art_token="art",
    )
    cid = _make_running_generation(settings)
    (settings.data_dir / cid / prepared_input.RECEIPT_FILENAME).unlink()

    _resume_generation(settings, cid)

    assert storage.load_meta(settings.data_dir, cid)["generation"] == {
        "status": "submission_unknown",
        "error": "submission_unknown",
        "attempt": 1,
        "client_request_id": REQUEST_ID,
    }
