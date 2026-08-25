import hashlib
import json
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, make_settings
from app import h3, pipeline, prepared_input, storage
from app.main import (
    _freeze_submission,
    _load_h3_request,
    _result_fields,
    _resume_generation,
    create_app,
)


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
        fit_profiles={
            "16:9": {
                "fit_required": not fit_required,
                "default_fit_mode": "crop" if not fit_required else "none",
            },
            "9:16": {
                "fit_required": fit_required,
                "default_fit_mode": "crop" if fit_required else "none",
            },
        },
        aspect_ratio="9:16",
        resolution="768p",
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
        engine_request={
            "h3": {
                "workflow": h3.H3_WORKFLOW,
                "duration": 10,
                "resolution": h3.H3_RESOLUTION,
            }
        },
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        prompt=frozen.prompt_text,
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
    )
    return frozen


def _write_startup_h3_attempt(settings, cid, *, generation_status="running"):
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
        engine_request={
            "h3": {
                "workflow": h3.H3_WORKFLOW,
                "duration": 10,
                "resolution": h3.H3_RESOLUTION,
            }
        },
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": generation_status,
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
        prepared_dialogue=[],
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
        fit_mode="none",
        dialogue_mode="none",
    )
    request = _load_h3_request(
        settings, cid, storage.load_meta(settings.data_dir, cid)
    )
    state_root = cdir / ".h3"
    attempt_path = state_root / "attempts" / "000001" / "attempt.json"
    attempt_path.parent.mkdir(parents=True)
    (state_root / "session.json").write_text(
        json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": cid}),
        encoding="utf-8",
    )
    state = h3._new_state(request, "000001", REQUEST_ID)
    task_id = "known-paid-task"
    state.update(status="h3_running", retryable=False)
    state["h3"] = {
        "status": "running",
        "task_id": task_id,
        "receipt": h3._h3_receipt(request, task_id),
    }
    attempt_path.write_text(json.dumps(state), encoding="utf-8")
    return request, state_root / "session.json", attempt_path


def _write_legacy_pre_h3_attempt(
    settings, cid, *, attempt=1, request_id=REQUEST_ID, h3_state=None,
):
    attempt_id = f"{attempt:06d}"
    keyframes = [{
        "name": "legacy-frame.png",
        "sha256": hashlib.sha256(b"legacy-frame").hexdigest(),
    }]
    source_prompt = "legacy source prompt"
    optimized_prompt = "legacy optimized prompt"
    legacy_input = {
        "prompt_sha256": hashlib.sha256(source_prompt.encode("utf-8")).hexdigest(),
        "keyframes": keyframes,
        "voice_texts_sha256": "2" * 64,
        "request": {
            "duration": 10,
            "h3_workflow": h3.H3_WORKFLOW,
            "ir_model": "MiniMax-H3",
            "ratio": h3.H3_DEFAULT_ASPECT_RATIO,
            "resolution": h3.H3_RESOLUTION,
        },
    }
    input_receipt = h3.canonical_json_sha256(legacy_input)
    ir_task_id = "legacy-ir-task"
    state = {
        "schema_version": h3.SCHEMA_VERSION,
        "cid": cid,
        "attempt_id": attempt_id,
        "client_request_id": request_id,
        "input": legacy_input,
        "input_receipt": input_receipt,
        "status": "ready_for_h3",
        "retryable": False,
        "ir": {
            "optimized_prompt": optimized_prompt,
            "optimized_prompt_sha256": hashlib.sha256(
                optimized_prompt.encode("utf-8")
            ).hexdigest(),
            "receipt": {
                "input_receipt": input_receipt,
                "keyframes": keyframes,
                "prompt_sha256": legacy_input["prompt_sha256"],
                "request": {
                    "duration": 10,
                    "model": "MiniMax-H3",
                    "ratio": h3.H3_DEFAULT_ASPECT_RATIO,
                },
                "task_id": ir_task_id,
                "voice_texts_sha256": legacy_input["voice_texts_sha256"],
            },
            "status": "succeeded",
            "task_id": ir_task_id,
        },
        "h3": h3_state or {"status": "ready"},
    }
    path = (
        settings.data_dir / cid / ".h3" / "attempts" / attempt_id / "attempt.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parents[2] / "session.json").write_text(
        json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": cid}),
        encoding="utf-8",
    )
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def _payload(
    request_id=REQUEST_ID, *, mode="none", fit="none", lines=None,
    aspect_ratio="9:16", resolution="768p",
):
    payload = {
        "confirm": True,
        "client_request_id": request_id,
        "dialogue_mode": mode,
        "fit_mode": fit,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    if lines is not None:
        payload["lines"] = lines
    return payload


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [("aspect_ratio", "1:1", "invalid_aspect_ratio"),
     ("resolution", "1080p", "invalid_resolution")],
)
def test_submit_rejects_generation_parameters_before_provider_post(
    enabled, monkeypatch, field, value, code,
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    called = []
    monkeypatch.setattr(h3, "start", lambda *_a, **_kw: called.append(True))
    payload = _payload()
    payload[field] = value

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=payload
    )

    assert response.status_code == 422
    assert response.json() == {"detail": code}
    assert called == []
    assert storage.load_meta(settings.data_dir, cid)["generation"] is None


@pytest.fixture
def enabled(tmp_path):
    settings = make_settings(
        tmp_path,
        enable_h3_submit=True,
        autodl_art_token="art-test-secret",
    )
    with TestClient(create_app(settings)) as client:
        yield settings, client


@pytest.fixture(scope="session")
def recovery_video_bytes(tmp_path_factory):
    root = tmp_path_factory.mktemp("short-recovery-videos")
    result = {}
    for name, duration in (("target", 10), ("wrong", 1)):
        path = root / f"{name}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=black:s=32x32:r=5:d={duration}",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-y", str(path),
            ],
            check=True,
            capture_output=True,
        )
        result[name] = path.read_bytes()
    return result


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


def test_source_prompt_cas_conflict_is_structured_and_does_not_write(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    _write_initial_receipt(settings, cid)
    cdir = settings.data_dir / cid
    tracked = [
        cdir / "meta.json",
        cdir / "work" / "visual_prompt.txt",
        cdir / "work" / "prompt.txt",
        cdir / prepared_input.RECEIPT_FILENAME,
    ]
    before = {path: path.read_bytes() for path in tracked}

    response = client.patch(
        f"/api/conversations/{cid}/prompt",
        headers=AUTH,
        json={
            "confirm": True,
            "expected_sha256": "0" * 64,
            "prompt": "不得覆盖现有提示词。",
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": {
        "code": "prompt_changed",
        "message": "提示词已更新，请刷新页面后重试。",
    }}
    assert {path: path.read_bytes() for path in tracked} == before


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


def test_submit_rejects_long_video_without_bound_plan_before_provider(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings, duration_s=15.001)
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json={**_payload(), "expected_plan_receipt": "0" * 64},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "long_video_plan_invalid"}
    assert not (settings.data_dir / cid / ".h3").exists()


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
    cid, original = _make_conv(settings, duration_s=9.2)
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
    assert request.duration == 10
    assert request.keyframes[0][1] == original
    meta = storage.load_meta(settings.data_dir, cid)
    cdir = settings.data_dir / cid
    frozen = prepared_input.load_prepared_input(
        cdir, cdir / prepared_input.RECEIPT_FILENAME, expected_dialogue=meta["prepared_dialogue"]
    )
    assert set(frozen.engine_request) == {"h3"}
    assert frozen.engine_request["h3"] == {
        "workflow": h3.H3_WORKFLOW,
        "duration": 10,
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "provider_resolution": "768p竖",
    }
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert "context_ir_enabled" not in detail["generation"]
    assert detail["generation"]["stage"] == "h3"


def test_submit_freezes_postprocessed_keyframe_bytes_when_optimization_done(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, original = _make_conv(settings, duration_s=9.2)
    cdir = settings.data_dir / cid
    optimized = _png(cdir / "work" / "postprocessed" / "01.png", value=231)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "done",
            "options": {"remove_subtitle": True, "remove_brand": False},
            "frames": ["01.png"],
            "error": None,
        },
    )
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request)
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()
    )

    assert response.status_code == 202
    assert seen[0].keyframes[0][1] == optimized
    assert seen[0].keyframes[0][1] != original
    frozen = prepared_input.load_prepared_input(
        cdir,
        cdir / prepared_input.RECEIPT_FILENAME,
        expected_dialogue=storage.load_meta(settings.data_dir, cid)["prepared_dialogue"],
    )
    assert frozen.keyframes[0].path == cdir / "work" / "postprocessed" / "01.png"


def test_submit_does_not_fall_back_while_postprocess_is_incomplete(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings, duration_s=9.2)
    storage.update_meta(
        settings.data_dir,
        cid,
        postprocess={
            "status": "running",
            "options": {"remove_subtitle": True, "remove_brand": False},
            "frames": [],
            "error": None,
        },
    )

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "postprocess_not_ready"}


def test_submit_freezes_horizontal_480p_across_meta_receipt_and_request(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, _ = _make_conv(settings, fit_required=True)
    seen = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: seen.append(request)
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(aspect_ratio="16:9", resolution="480p"),
    )

    assert response.status_code == 202
    request = seen[0]
    assert (request.aspect_ratio, request.resolution) == ("16:9", "480p")
    meta = storage.load_meta(settings.data_dir, cid)
    assert (meta["aspect_ratio"], meta["resolution"]) == ("16:9", "480p")
    receipt = json.loads(
        (settings.data_dir / cid / prepared_input.RECEIPT_FILENAME).read_text()
    )
    assert receipt["video"]["ratio"] == "16:9"
    assert receipt["engine_request"]["h3"] == {
        "workflow": h3.H3_WORKFLOW,
        "duration": 10,
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "provider_resolution": "480p横",
    }


def test_paid_retry_rejects_aspect_or_resolution_drift_before_provider(
    enabled, monkeypatch,
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    calls = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: calls.append(("start", request))
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda *args: calls.append(("retry", args))
        or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()
    ).status_code == 202

    changed = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload(
            "request-654321", aspect_ratio="9:16", resolution="480p"
        ),
    )

    assert changed.status_code == 409
    assert changed.json() == {"detail": "resume_parameters_changed"}
    assert [kind for kind, _value in calls] == ["start"]


@pytest.mark.parametrize(
    ("fit_required", "initial", "changed"),
    [
        (False, _payload(), _payload("request-654321", mode="auto")),
        (
            False,
            _payload(
                mode="custom",
                lines=[{"text": "台词 A", "start_s": 0, "end_s": 1.0}],
            ),
            _payload(
                "request-654321",
                mode="custom",
                lines=[{"text": "台词 B", "start_s": 0, "end_s": 1.0}],
            ),
        ),
        (
            True,
            _payload(fit="crop"),
            _payload("request-654321", fit="pad"),
        ),
        (
            False,
            _payload(),
            _payload("request-654321", fit="crop", aspect_ratio="16:9"),
        ),
        (
            False,
            _payload(),
            _payload("request-654321", resolution="480p"),
        ),
    ],
    ids=["dialogue-mode", "dialogue-lines", "fit", "aspect", "resolution"],
)
def test_paid_retry_rejects_every_frozen_input_drift_before_provider(
    enabled, monkeypatch, fit_required, initial, changed,
):
    settings, client = enabled
    cid, _ = _make_conv(settings, fit_required=fit_required)
    calls = []
    monkeypatch.setattr(
        h3,
        "start",
        lambda request: calls.append(("start", request))
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda *args: calls.append(("retry", args))
        or h3.H3Result("failed", "000002", error_code="h3_provider_failed"),
    )
    assert client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=initial
    ).status_code == 202

    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=changed
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "resume_parameters_changed"}
    assert [kind for kind, _value in calls] == ["start"]


def test_submit_claim_wins_atomically_before_pipeline(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    cdir = settings.data_dir / cid
    entered = threading.Event()
    release = threading.Event()
    blocked = False
    original_write = storage._write_meta

    def block_first_submit_claim(path, meta):
        nonlocal blocked
        owner = meta.get("_input_owner")
        if (
            isinstance(owner, dict)
            and owner.get("kind") == "submit"
            and owner.get("request_id") == REQUEST_ID
            and not blocked
        ):
            blocked = True
            entered.set()
            assert release.wait(timeout=5)
        original_write(path, meta)

    provider_calls = []
    pipeline_steps = []
    monkeypatch.setattr(storage, "_write_meta", block_first_submit_claim)
    monkeypatch.setattr(
        h3, "start",
        lambda request: provider_calls.append(request) or h3.H3Result(
            "failed", "000001", error_code="h3_provider_failed"
        ),
    )
    monkeypatch.setattr(
        pipeline, "_run_cmd",
        lambda *_a, **_kw: pipeline_steps.append(_kw.get("step")),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = pool.submit(
            client.post, f"/api/conversations/{cid}/submit",
            headers=AUTH, json=_payload(),
        )
        assert entered.wait(timeout=5)
        pipeline_future = pool.submit(pipeline.run, settings, cid, object())
        release.set()
        response = submitted.result()
        pipeline_future.result()

    assert response.status_code == 202
    assert len(provider_calls) == 1
    assert pipeline_steps == []
    receipt = (cdir / prepared_input.RECEIPT_FILENAME).read_bytes()
    meta_bytes = (cdir / "meta.json").read_bytes()
    pipeline.run(settings, cid, object())
    assert (cdir / prepared_input.RECEIPT_FILENAME).read_bytes() == receipt
    assert (cdir / "meta.json").read_bytes() == meta_bytes


def test_pipeline_claim_wins_atomically_before_submit(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    cdir = settings.data_dir / cid
    entered = threading.Event()
    release = threading.Event()
    blocked = False
    original_write = storage._write_meta

    def block_first_pipeline_claim(path, meta):
        nonlocal blocked
        owner = meta.get("_input_owner")
        if isinstance(owner, dict) and owner.get("kind") == "pipeline" and not blocked:
            blocked = True
            entered.set()
            assert release.wait(timeout=5)
        original_write(path, meta)

    provider_calls = []
    monkeypatch.setattr(storage, "_write_meta", block_first_pipeline_claim)
    monkeypatch.setattr(
        h3, "start", lambda request: provider_calls.append(request)
    )
    frozen_files = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        pipeline_future = pool.submit(pipeline.run, settings, cid, object())
        assert entered.wait(timeout=5)
        submitted = pool.submit(
            client.post, f"/api/conversations/{cid}/submit",
            headers=AUTH, json=_payload(),
        )
        release.set()
        response = submitted.result()
        pipeline_future.result()

    assert response.status_code == 409
    assert provider_calls == []
    assert not (cdir / prepared_input.RECEIPT_FILENAME).exists()
    assert {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    } == frozen_files


def test_startup_reconciles_half_frozen_short_submit_without_provider(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=False, enable_h3_submit=False,
        autodl_art_token="unused-test-token",
    )
    cid, _ = _make_conv(settings)
    _write_initial_receipt(settings, cid)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    claimed = storage.claim_submission_input(
        settings.data_dir, cid, "request-old-123"
    )
    assert claimed
    _freeze_submission(
        settings, cid, claimed, "request-old-123", "none", "none",
        "9:16", "768p", (),
    )
    cdir = settings.data_dir / cid
    before = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    provider_calls = []
    monkeypatch.setattr(h3, "start", lambda *_args, **_kwargs: provider_calls.append(1))

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, cid)
    after = {
        path.relative_to(cdir).as_posix(): path.read_bytes()
        for path in cdir.rglob("*") if path.is_file() and path.name != "meta.json"
    }
    assert recovered["status"] == "done"
    assert recovered["error"] == "submission_recovery_required"
    assert recovered["generation"] is None
    assert recovered["_input_owner"] is None
    assert provider_calls == []
    assert after == before
    assert storage.claim_submission_input(
        settings.data_dir, cid, "request-new-123"
    )


def test_startup_releases_non_utf8_half_frozen_submit_without_provider(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_pipeline=False, enable_h3_submit=False
    )
    cid, _ = _make_conv(settings)
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-old")
    assert storage.claim_submission_input(
        settings.data_dir, cid, "request-old-bad"
    )
    receipt = settings.data_dir / cid / prepared_input.RECEIPT_FILENAME
    receipt.write_bytes(b"\xff\xfeinvalid")
    before = receipt.read_bytes()
    monkeypatch.setattr(storage, "PROCESS_GENERATION", "boot-new")
    provider_calls = []
    monkeypatch.setattr(h3, "start", lambda *_args, **_kwargs: provider_calls.append(1))

    with TestClient(create_app(settings)):
        pass

    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["status"] == "done"
    assert recovered["error"] == "submission_recovery_required"
    assert recovered["generation"] is None
    assert recovered["_input_owner"] is None
    assert provider_calls == []
    assert receipt.read_bytes() == before


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
        lambda request: h3.H3Result("failed", "000001", error_code="download_invalid_video"),
    )
    monkeypatch.setattr(
        h3,
        "retry",
        lambda request, request_id: calls.append((request, request_id))
        or h3.H3Result("failed", "000002", error_code="download_invalid_video"),
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
        return h3.H3Result(
            "retryable_failure", "000001", retryable=True, error_code="h3_query_failed"
        )

    monkeypatch.setattr(h3, "start", start)
    resumed_requests = []
    monkeypatch.setattr(
        h3,
        "resume",
        lambda request: resumed_requests.append(request)
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )
    monkeypatch.setattr(h3, "retry", lambda *_a, **_kw: pytest.fail("resume must not retry"))
    assert client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()).status_code == 202
    detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    assert detail["generation"]["status"] == "resume_required"
    wrong = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload("request-654321")
    )
    assert wrong.status_code == 409
    resumed = client.post(f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload())
    assert resumed.status_code == 202
    assert len(seen) == 1
    assert len(resumed_requests) == 1
    assert resumed_requests[0].client_request_id == REQUEST_ID


def test_short_resume_required_missing_attempt_locks_unknown_without_post(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    _write_initial_receipt(settings, cid)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "resume_required",
            "error": "h3_query_failed",
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
        prepared_dialogue=[],
        fit_mode="none",
        dialogue_mode="auto",
    )
    calls = []
    monkeypatch.setattr(h3, "start", lambda request: calls.append("start"))
    monkeypatch.setattr(h3, "retry", lambda *args: calls.append("retry"))
    monkeypatch.setattr(
        h3, "resume",
        lambda request: calls.append("resume") or h3.H3Result("not_started", None),
    )
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(mode="auto"),
    )
    assert response.status_code == 202
    assert calls == ["resume"]
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "submission_unknown"


def test_short_succeeded_missing_output_redownloads_known_task_get_only(enabled, monkeypatch):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    _write_initial_receipt(settings, cid)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "succeeded",
            "error": None,
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
        prepared_dialogue=[],
        fit_mode="none",
        dialogue_mode="auto",
    )
    calls = []
    monkeypatch.setattr(h3, "start", lambda _request: pytest.fail("must not POST"))
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("must not retry"))

    def resume(request):
        calls.append(request)
        request.workdir.joinpath("generated.mp4").write_bytes(b"recovered")
        return h3.H3Result(
            "succeeded", "000001", output=request.workdir / "generated.mp4"
        )

    monkeypatch.setattr(h3, "resume", resume)
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(mode="auto"),
    )
    assert response.status_code == 202
    assert len(calls) == 1
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


def test_short_resume_corrupt_receipt_converges_to_unknown(enabled):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    _write_initial_receipt(settings, cid)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={
            "status": "resume_required",
            "error": "h3_query_failed",
            "attempt": 1,
            "client_request_id": REQUEST_ID,
            "stage": "h3",
        },
        prepared_dialogue=[],
        fit_mode="none",
        dialogue_mode="auto",
    )
    (settings.data_dir / cid / prepared_input.RECEIPT_FILENAME).write_text("broken")
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH,
        json=_payload(mode="auto"),
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "submission_outcome_unknown"}
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "submission_unknown"


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
    _write_legacy_pre_h3_attempt(settings, cid)
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


@pytest.mark.parametrize(
    "evidence",
    [
        "missing",
        "corrupt",
        "paid",
        "ambiguous",
        "top-level-extra",
        "contradictory-status",
        "retryable",
        "corrupt-input",
        "corrupt-ir",
        "marker-missing",
        "marker-corrupt",
        "marker-mismatched",
        "marker-extra",
    ],
)
def test_legacy_context_marker_cannot_hide_ambiguous_or_paid_h3_attempt(
    enabled, monkeypatch, evidence,
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    if evidence == "corrupt":
        path = _write_legacy_pre_h3_attempt(settings, cid)
        path.write_text("{", encoding="utf-8")
    elif evidence == "paid":
        _write_legacy_pre_h3_attempt(
            settings,
            cid,
            h3_state={"status": "running", "task_id": "already-paid-h3"},
        )
    elif evidence == "ambiguous":
        _write_legacy_pre_h3_attempt(settings, cid)
        _write_legacy_pre_h3_attempt(
            settings, cid, attempt=2, request_id="older-request-123"
        )
    elif evidence != "missing":
        path = _write_legacy_pre_h3_attempt(settings, cid)
        state = json.loads(path.read_text(encoding="utf-8"))
        if evidence == "top-level-extra":
            state["h3_task_id"] = "ambiguous-paid-evidence"
        elif evidence == "contradictory-status":
            state["status"] = "succeeded"
        elif evidence == "retryable":
            state["retryable"] = True
        elif evidence == "corrupt-input":
            state["input_receipt"] = "0" * 64
        elif evidence == "corrupt-ir":
            state["ir"]["receipt"]["input_receipt"] = "0" * 64
        path.write_text(json.dumps(state), encoding="utf-8")
        marker = path.parents[2] / "session.json"
        if evidence == "marker-missing":
            marker.unlink()
        elif evidence == "marker-corrupt":
            marker.write_text("{", encoding="utf-8")
        elif evidence == "marker-mismatched":
            marker.write_text(
                json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": "wrong"}),
                encoding="utf-8",
            )
        elif evidence == "marker-extra":
            marker.write_text(
                json.dumps({
                    "schema_version": h3.SCHEMA_VERSION,
                    "cid": cid,
                    "extra": True,
                }),
                encoding="utf-8",
            )
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
    monkeypatch.setattr(
        h3, "retry", lambda *_args, **_kwargs: pytest.fail("must not retry")
    )

    response = client.post(
        f"/api/conversations/{cid}/submit",
        headers=AUTH,
        json=_payload("request-654321", resolution="480p"),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "submission_outcome_unknown"}
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "submission_unknown"


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


def test_short_submit_background_converges_after_provider_auto_retry(
    enabled, monkeypatch, recovery_video_bytes,
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    original_start = h3.start
    posts = 0

    class PublicStream:
        def get_extra_info(self, name):
            return ("93.184.216.34", 443) if name == "server_addr" else None

    def provider(req: httpx.Request) -> httpx.Response:
        nonlocal posts
        if req.method == "POST":
            posts += 1
            return httpx.Response(200, json={"data": {"task_id": f"task-{posts}"}})
        if req.url.path.endswith("/result/task-1"):
            return httpx.Response(200, json={
                "request_id": "provider-failure",
                "data": {"status": "FAILED"},
            })
        if req.url.path.endswith("/result/task-2"):
            return httpx.Response(200, json={"data": {
                "status": "SUCCESS",
                "results": [{"url": "https://download.invalid/video.mp4"}],
            }})
        return httpx.Response(
            200,
            content=recovery_video_bytes["target"],
            extensions={"network_stream": PublicStream()},
        )

    monkeypatch.setattr(h3, "_pause", lambda _seconds: None)
    monkeypatch.setattr(
        h3.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def start(request):
        with httpx.Client(transport=httpx.MockTransport(provider)) as provider_client:
            return original_start(request, client=provider_client)

    monkeypatch.setattr(h3, "start", start)
    response = client.post(
        f"/api/conversations/{cid}/submit", headers=AUTH, json=_payload()
    )

    assert response.status_code == 202
    assert posts == 2
    assert storage.load_meta(settings.data_dir, cid)["generation"]["status"] == "succeeded"


def test_short_startup_scanner_claims_persisted_provider_failure(
    tmp_path, monkeypatch,
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art-test-secret"
    )
    cid, _ = _make_conv(settings)
    _request, _session, attempt_path = _write_startup_h3_attempt(
        settings, cid, generation_status="failed"
    )
    state = json.loads(attempt_path.read_text(encoding="utf-8"))
    state.update(status="failed", retryable=False)
    state["h3"]["status"] = "failed"
    state["error"] = {
        "code": "h3_provider_failed",
        "provider": {"status": "FAILED", "detail": "GPU OOM"},
    }
    attempt_path.write_text(json.dumps(state), encoding="utf-8")
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir,
        cid,
        generation={**meta["generation"], "status": "failed", "error": "h3_provider_failed"},
    )
    resumed = []
    monkeypatch.setattr(
        h3,
        "resume",
        lambda request: resumed.append(request)
        or h3.H3Result("failed", "000001", error_code="h3_provider_failed"),
    )

    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/conversations/{cid}", headers=AUTH).status_code == 200
        for thread in client.app.state.h3_resume_threads:
            thread.join(timeout=2)

    assert len(resumed) == 1
    assert resumed[0].client_request_id == REQUEST_ID


@pytest.mark.parametrize(
    ("damage", "generation_status"),
    [
        ("session_wrong_cid", "running"),
        ("session_unicode", "running"),
        ("attempt_missing", "running"),
        ("attempt_json", "running"),
        ("attempt_unicode", "running"),
        ("input_receipt", "running"),
        ("output_receipt", "succeeded"),
    ],
)
def test_short_startup_corrupt_paid_receipt_locks_unknown_without_provider_post(
    tmp_path, monkeypatch, damage, generation_status
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art-test-secret"
    )
    cid, _ = _make_conv(settings)
    _request, session_path, attempt_path = _write_startup_h3_attempt(
        settings, cid, generation_status=generation_status
    )
    if damage == "session_wrong_cid":
        session_path.write_text(
            json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": "wrong"}),
            encoding="utf-8",
        )
    elif damage == "session_unicode":
        session_path.write_bytes(b"\xff")
    elif damage == "attempt_missing":
        attempt_path.unlink()
    elif damage == "attempt_json":
        attempt_path.write_text("{", encoding="utf-8")
    elif damage == "attempt_unicode":
        attempt_path.write_bytes(b"\xff")
    else:
        state = json.loads(attempt_path.read_text(encoding="utf-8"))
        if damage == "input_receipt":
            state["input_receipt"] = "0" * 64
        else:
            state.update(status="succeeded", retryable=False)
            state["h3"].update(
                status="succeeded",
                output={"name": "generated.mp4", "sha256": "bad", "size": 1},
            )
        attempt_path.write_text(json.dumps(state), encoding="utf-8")

    prepared_path = settings.data_dir / cid / prepared_input.RECEIPT_FILENAME
    frozen_prepared = prepared_path.read_bytes()
    frozen_attempt = attempt_path.read_bytes() if attempt_path.exists() else None
    provider_calls = []
    original_resume = h3.resume

    def provider(request: httpx.Request) -> httpx.Response:
        provider_calls.append(request)
        assert request.method == "GET"
        return httpx.Response(503)

    def resume(request):
        with httpx.Client(transport=httpx.MockTransport(provider)) as client:
            return original_resume(request, client=client)

    monkeypatch.setattr(h3, "resume", resume)
    _resume_generation(settings, cid)

    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == "submission_unknown"
    assert generation["error"] == "submission_unknown"
    assert generation["attempt"] == 1
    assert provider_calls == []
    assert prepared_path.read_bytes() == frozen_prepared
    if frozen_attempt is None:
        assert not attempt_path.exists()
    else:
        assert attempt_path.read_bytes() == frozen_attempt


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("state_unavailable", "submission_unknown"),
        ("h3_provider_failed", "failed"),
    ],
)
def test_short_startup_only_locks_local_state_access_errors(
    tmp_path, monkeypatch, code, expected_status
):
    settings = make_settings(
        tmp_path, enable_h3_submit=True, autodl_art_token="art-test-secret"
    )
    cid, _ = _make_conv(settings)
    _write_startup_h3_attempt(settings, cid)
    monkeypatch.setattr(
        h3,
        "resume",
        lambda _request: (_ for _ in ()).throw(h3.H3Error(code)),
    )
    monkeypatch.setattr(h3, "retry", lambda *_args: pytest.fail("must not retry"))
    _resume_generation(settings, cid)
    generation = storage.load_meta(settings.data_dir, cid)["generation"]
    assert generation["status"] == expected_status
    assert generation["attempt"] == 1


@pytest.mark.parametrize("damage", ["zero", "tampered", "wrong_duration"])
def test_short_invalid_published_video_is_hidden_and_startup_redownloads_get_only(
    enabled, monkeypatch, recovery_video_bytes, damage
):
    settings, client = enabled
    cid, _ = _make_conv(settings)
    request, _session_path, attempt_path = _write_startup_h3_attempt(
        settings, cid, generation_status="succeeded"
    )
    output = request.workdir / "generated.mp4"
    target = recovery_video_bytes["target"]
    output.write_bytes(target)
    state = json.loads(attempt_path.read_text(encoding="utf-8"))
    state.update(status="succeeded", retryable=False)
    state["h3"].update(
        status="succeeded",
        output={
            "name": "generated.mp4",
            "sha256": hashlib.sha256(target).hexdigest(),
            "size": len(target),
        },
    )
    if damage == "zero":
        output.write_bytes(b"")
    elif damage == "tampered":
        output.write_bytes(target[:-32] + b"x" * 32)
    else:
        wrong = recovery_video_bytes["wrong"]
        output.write_bytes(wrong)
        state["h3"]["output"] = {
            "name": "generated.mp4",
            "sha256": hashlib.sha256(wrong).hexdigest(),
            "size": len(wrong),
        }
    attempt_path.write_text(json.dumps(state), encoding="utf-8")
    frozen_input = (state["input"], state["input_receipt"], state["attempt_id"])
    prepared_path = settings.data_dir / cid / prepared_input.RECEIPT_FILENAME
    frozen_prepared = prepared_path.read_bytes()

    assert client.get(f"/api/conversations/{cid}", headers=AUTH).json()["has_video"] is False
    listed = client.get("/api/conversations", headers=AUTH).json()
    assert next(item for item in listed if item["id"] == cid)["has_video"] is False
    calls = []

    class PublicStream:
        def get_extra_info(self, name):
            return ("93.184.216.34", 443) if name == "server_addr" else None

    def provider(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.method == "GET"
        if req.url.path.endswith("/result/known-paid-task"):
            return httpx.Response(
                200,
                json={"data": {"status": "SUCCESS", "results": [
                    {"url": "https://download.invalid/video.mp4"}
                ]}},
            )
        return httpx.Response(
            200,
            content=target,
            extensions={"network_stream": PublicStream()},
        )

    original_resume = h3.resume

    def resume(value):
        with httpx.Client(transport=httpx.MockTransport(provider)) as provider_client:
            return original_resume(value, client=provider_client)

    monkeypatch.setattr(h3, "resume", resume)
    monkeypatch.setattr(
        h3.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    _resume_generation(settings, cid)

    assert calls and all(call.method == "GET" for call in calls)
    assert output.read_bytes() == target
    recovered = storage.load_meta(settings.data_dir, cid)
    assert recovered["generation"]["status"] == "succeeded"
    assert recovered["generation"]["attempt"] == 1
    assert prepared_path.read_bytes() == frozen_prepared
    final_state = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert (
        final_state["input"],
        final_state["input_receipt"],
        final_state["attempt_id"],
    ) == frozen_input
    assert client.get(f"/api/conversations/{cid}", headers=AUTH).json()["has_video"] is True
    listed = client.get("/api/conversations", headers=AUTH).json()
    assert next(item for item in listed if item["id"] == cid)["has_video"] is True
