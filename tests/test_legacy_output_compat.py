import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import h3, main as main_module, storage
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
    skipped_ir=False,
):
    root.mkdir(parents=True, exist_ok=True)
    output = root / "generated.mp4"
    output.write_bytes(video)
    attempt_id = f"{attempt:06d}"
    path = root / ".h3" / "attempts" / attempt_id / "attempt.json"
    path.parent.mkdir(parents=True)
    source_prompt = "legacy source prompt"
    optimized_prompt = source_prompt if skipped_ir else "legacy optimized prompt"
    source_prompt_sha256 = hashlib.sha256(source_prompt.encode()).hexdigest()
    optimized_prompt_sha256 = hashlib.sha256(optimized_prompt.encode()).hexdigest()
    keyframes = [
        {
            "name": "legacy-frame.png",
            "sha256": hashlib.sha256(b"legacy-frame").hexdigest(),
        }
    ]
    request = {
        "duration": requested_duration,
        "h3_workflow": "minimax_h3_lightx2v_v5",
        "ir_model": "MiniMax-H3",
        "ratio": "9:16",
        "resolution": "768p竖",
    }
    if skipped_ir:
        request["context_ir_enabled"] = False
    frozen_input = {
        "prompt_sha256": source_prompt_sha256,
        "keyframes": keyframes,
        "voice_texts_sha256": "2" * 64,
        "request": request,
    }
    input_receipt = h3.canonical_json_sha256(frozen_input)
    ir_state = (
        {
            "mode": "skipped",
            "optimized_prompt": optimized_prompt,
            "optimized_prompt_sha256": optimized_prompt_sha256,
            "status": "succeeded",
        }
        if skipped_ir
        else {
            "optimized_prompt": optimized_prompt,
            "optimized_prompt_sha256": optimized_prompt_sha256,
            "receipt": {
                "input_receipt": input_receipt,
                "keyframes": keyframes,
                "prompt_sha256": source_prompt_sha256,
                "request": {
                    "duration": requested_duration,
                    "model": "MiniMax-H3",
                    "ratio": "9:16",
                },
                "task_id": "legacy-ir-task",
                "voice_texts_sha256": "2" * 64,
            },
            "status": "succeeded",
            "task_id": "legacy-ir-task",
        }
    )
    state = {
        "schema_version": 1,
        "cid": cid,
        "attempt_id": attempt_id,
        "client_request_id": request_id,
        "input": frozen_input,
        "input_receipt": input_receipt,
        "status": "succeeded",
        "retryable": False,
        "ir": ir_state,
        "h3": {
            "status": "succeeded",
            "task_id": "legacy-paid-task",
            "receipt": {
                "input_receipt": input_receipt,
                "keyframes": keyframes,
                "prompt_sha256": optimized_prompt_sha256,
                "request": {
                    "duration": min(requested_duration, h3.H3_MAX_DURATION_S),
                    "resolution": "768p竖",
                    "workflow": "minimax_h3_lightx2v_v5",
                },
                "task_id": "legacy-paid-task",
            },
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


@pytest.mark.parametrize("skipped_ir", [False, True])
def test_legacy_success_accepts_exact_historical_ir_shapes_and_clamped_duration(
    tmp_path, monkeypatch, skipped_ir
):
    root = tmp_path / CID
    _write_legacy_success(root, requested_duration=13, skipped_ir=skipped_ir)
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
        "tampered_input_receipt",
        "current_reference_request",
        "wrong_clamped_duration",
        "duration_too_large",
        "missing_keyframes",
        "duplicate_keyframe_name",
        "retryable_success",
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
        elif damage == "tampered_input_receipt":
            state["input_receipt"] = "0" * 64
        elif damage == "current_reference_request":
            state["input"]["request"] = {
                "duration": 10,
                "h3_workflow": "minimax_h3_lightx2v_v5",
                "resolution": "768p竖",
            }
            state["input_receipt"] = h3.canonical_json_sha256(state["input"])
        elif damage == "wrong_clamped_duration":
            state["h3"]["receipt"]["request"]["duration"] = 11
        elif damage == "duration_too_large":
            state["input"]["request"]["duration"] = 999
            state["input_receipt"] = h3.canonical_json_sha256(state["input"])
            state["ir"]["receipt"]["input_receipt"] = state["input_receipt"]
            state["ir"]["receipt"]["request"]["duration"] = 999
            state["h3"]["receipt"]["input_receipt"] = state["input_receipt"]
        elif damage == "missing_keyframes":
            state["input"]["keyframes"] = []
            state["input_receipt"] = h3.canonical_json_sha256(state["input"])
            state["ir"]["receipt"]["input_receipt"] = state["input_receipt"]
            state["ir"]["receipt"]["keyframes"] = []
            state["h3"]["receipt"]["input_receipt"] = state["input_receipt"]
            state["h3"]["receipt"]["keyframes"] = []
        elif damage == "duplicate_keyframe_name":
            state["input"]["keyframes"] *= 2
            state["input_receipt"] = h3.canonical_json_sha256(state["input"])
            state["ir"]["receipt"]["input_receipt"] = state["input_receipt"]
            state["ir"]["receipt"]["keyframes"] *= 2
            state["h3"]["receipt"]["input_receipt"] = state["input_receipt"]
            state["h3"]["receipt"]["keyframes"] *= 2
        elif damage == "retryable_success":
            state["retryable"] = True
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


def test_current_reference_attempt_cannot_masquerade_as_legacy(
    tmp_path, monkeypatch
):
    root = tmp_path / CID
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"current-reference-frame")
    request = h3.H3Request(
        cid=CID,
        workdir=root,
        client_request_id=REQUEST_ID,
        prompt="current prompt",
        keyframes=h3.freeze_keyframes((frame,)),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=10,
        autodl_token="unused-test-token",
    )
    state = h3._new_state(request, "000001", REQUEST_ID)
    state["status"] = "succeeded"
    state["input_receipt"] = "0" * 64
    state["ir"] = {}
    state["h3"] = {
        "status": "succeeded",
        "task_id": "current-paid-task",
        "receipt": {},
        "output": {
            "name": "generated.mp4",
            "sha256": hashlib.sha256(b"current-video").hexdigest(),
            "size": len(b"current-video"),
        },
    }
    path = root / ".h3" / "attempts" / "000001" / "attempt.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    (root / "generated.mp4").write_bytes(b"current-video")
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.0)

    assert _accept(root) is False


def test_short_legacy_evidence_is_checked_only_after_strict_returns_false(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="legacy", orig_name="legacy.mp4"
    )
    cid = meta["id"]
    _write_legacy_success(settings.data_dir / cid, cid=cid)
    generation = {
        "status": "succeeded",
        "error": None,
        "attempt": 1,
        "client_request_id": REQUEST_ID,
        "stage": "h3",
    }
    meta = {
        **meta,
        "status": "done",
        "duration_s": 10.0,
        "generation": generation,
    }
    calls = []
    monkeypatch.setattr(
        main_module, "_load_h3_request", lambda *_args: calls.append("load") or object()
    )
    monkeypatch.setattr(
        h3, "output_is_reusable", lambda *_args: calls.append("strict") or False
    )
    monkeypatch.setattr(
        h3,
        "legacy_succeeded_output_is_valid",
        lambda *_args, **_kwargs: calls.append("legacy") or True,
    )

    assert main_module._validate_generated_video_uncached(settings, meta) is True
    assert calls == ["load", "strict", "legacy"]


def test_strict_success_never_calls_legacy_fallback(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = storage.new_conversation(
        settings.data_dir, note="current", orig_name="current.mp4"
    )
    meta = {
        **meta,
        "status": "done",
        "duration_s": 10.0,
        "generation": {
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": "current-request",
            "stage": "h3",
        },
    }
    monkeypatch.setattr(main_module, "_load_h3_request", lambda *_args: object())
    monkeypatch.setattr(h3, "output_is_reusable", lambda *_args: True)
    monkeypatch.setattr(
        h3,
        "legacy_succeeded_output_is_valid",
        lambda *_args, **_kwargs: pytest.fail("legacy fallback must not run"),
    )

    assert main_module._validate_generated_video_uncached(settings, meta) is True


def test_long_strict_success_never_calls_legacy_fallback(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    meta = {
        "id": CID,
        "duration_s": 12.5,
        "frozen_plan_receipt": "a" * 64,
        "fit_mode": "none",
        "dialogue_mode": "auto",
        "segments": [{"index": 1}],
        "generation": {
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": "long-request",
            "stage": "stitch",
            "segments": [{"index": 1}],
        },
    }
    plan = SimpleNamespace(segments=(SimpleNamespace(index=1),))
    monkeypatch.setattr(
        main_module.long_generation,
        "generation_segments_are_valid",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        main_module.long_generation, "freeze_plan", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        main_module.long_generation,
        "bound_reusable_segment_indices",
        lambda *_args: frozenset({1}),
    )
    monkeypatch.setattr(
        main_module.long_generation,
        "stitched_output_is_reusable",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        h3,
        "legacy_succeeded_output_is_valid",
        lambda *_args, **_kwargs: pytest.fail("legacy fallback must not run"),
    )

    assert main_module._validate_generated_video_uncached(settings, meta) is True


def test_long_legacy_cache_invalidates_when_root_attempt_changes(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    cdir = settings.data_dir / CID
    attempt_path, _state = _write_legacy_success(
        cdir, requested_duration=13
    )
    meta = {
        "id": CID,
        "status": "done",
        "duration_s": 12.5,
        "generation": {
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
    }
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 10.125)

    assert main_module._has_valid_generated_video(settings, meta) is True
    attempt_path.write_bytes(b"tampered-attempt")
    assert main_module._has_valid_generated_video(settings, meta) is False


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
