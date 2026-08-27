import asyncio
import base64
import hashlib
import json
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import image_optimization, mediakit, postprocess, seedream, storage
from app.config import Settings, get_settings
from app.main import create_app
from conftest import AUTH, make_settings


def _png(width=5, height=3, value=127):
    ok, encoded = cv2.imencode(".png", np.full((height, width, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def _done(settings, *, segments=False):
    meta = storage.new_conversation(settings.data_dir, "x", "x.mp4")
    cid = meta["id"]
    cdir = settings.data_dir / cid
    if segments:
        items = [
            {"index": 1, "prompt": "first", "keyframes": ["01.png"]},
            {"index": 2, "prompt": "second", "keyframes": ["01.png"]},
        ]
        for item in items:
            path = cdir / "work" / "segments" / str(item["index"]) / "work" / "keyframes"
            path.mkdir(parents=True)
            (path / "01.png").write_bytes(_png())
        storage.update_meta(settings.data_dir, cid, status="done", duration_s=20, segments=items)
    else:
        path = cdir / "work" / "keyframes"
        path.mkdir(parents=True)
        (path / "01.png").write_bytes(_png())
        storage.update_meta(
            settings.data_dir, cid, status="done", duration_s=5,
            prompt="base", keyframes=["01.png"],
        )
    return cid


def _codex_prompts(meta):
    segments = meta.get("segments")
    indices = [item["index"] for item in segments] if segments else [0]
    return {
        index: f"本段 {index} 的 Codex 图片二次编辑提示词，不得出现文字。"
        for index in indices
    }


def _freeze(settings, meta):
    return image_optimization.freeze_prompts(settings, meta, _codex_prompts(meta))


def test_seedream_settings_are_closed_and_secret_is_not_a_setting(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "super-secret")
    settings = Settings(access_token="x")
    assert settings.seedream_model == "doubao-seedream-5-0-pro-260628"
    assert settings.seedream_edit_mode == "independent_parallel"
    assert settings.seedream_timeout_s == 300.0
    assert "super-secret" not in repr(settings)
    assert not hasattr(settings, "ark_api_key")
    for field, value in (
        ("seedream_model", "unknown"),
        ("seedream_edit_mode", "all-at-once"),
        ("seedream_concurrency", 0),
        ("seedream_timeout_s", float("nan")),
    ):
        with pytest.raises(ValueError):
            Settings(access_token="x", **{field: value})


def test_seedream_timeout_environment_default_and_explicit_override(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("SEEDREAM_TIMEOUT_S", raising=False)
    assert get_settings().seedream_timeout_s == 300.0

    monkeypatch.setenv("SEEDREAM_TIMEOUT_S", "180")
    assert get_settings().seedream_timeout_s == 180.0


def test_seedream_edit_mode_environment_default_and_explicit_anchor(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("SEEDREAM_EDIT_MODE", raising=False)
    assert get_settings().seedream_edit_mode == "independent_parallel"

    monkeypatch.setenv("SEEDREAM_EDIT_MODE", "anchor_consistency")
    assert get_settings().seedream_edit_mode == "anchor_consistency"


@pytest.mark.parametrize(("model", "has_sequential"), [
    ("doubao-seedream-5-0-pro-260628", False),
    ("doubao-seedream-5-0-260128", True),
    ("doubao-seedream-4-5-251128", True),
    ("doubao-seedream-4-0-250828", True),
])
def test_seedream_payload_is_model_capability_driven(
    tmp_path, monkeypatch, model, has_sequential,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, seedream_model=model)
    requests = []
    output = base64.b64encode(_png()).decode()

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"b64_json": output}]})

    asyncio.run(seedream.edit(
        settings, [_png()], "prompt", tmp_path / f"{model}.png",
        receipt_path=tmp_path / f"{model}.json", transport=httpx.MockTransport(handler),
    ))

    assert len(requests) == 1
    payload = requests[0]
    assert payload["model"] == model
    assert payload["prompt"] == "prompt"
    assert len(payload["image"]) == 1 and payload["image"][0].startswith("data:image/png;base64,")
    assert payload["response_format"] == "b64_json"
    assert payload["watermark"] is False
    assert ("sequential_image_generation" in payload) is has_sequential
    if has_sequential:
        assert payload["sequential_image_generation"] == "disabled"


def test_prompt_freeze_and_projection_are_segment_scoped(tmp_path):
    settings = make_settings(tmp_path)
    cid = _done(settings, segments=True)
    meta = storage.load_meta(settings.data_dir, cid)
    changes = _freeze(settings, meta)
    frozen = changes["_image_optimization"]
    assert [item["segment_index"] for item in frozen["segments"]] == [1, 2]
    assert frozen["model"] == settings.seedream_model
    assert frozen["edit_mode"] == "independent_parallel"
    for item in frozen["segments"]:
        assert item["default"] == item["current"]
        assert item["sha256"] == hashlib.sha256(item["current"].encode()).hexdigest()
        assert item["current"] == _codex_prompts(meta)[item["segment_index"]]


@pytest.mark.parametrize("indices", ([0, 1], [1, 3], [True, 2]))
def test_long_prompt_segments_must_be_positive_and_contiguous(tmp_path, indices):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2, "status": "done",
        "segments": [{"index": index, "prompt": "p"} for index in indices],
    }
    with pytest.raises(ValueError):
        image_optimization.freeze_prompts(settings, meta, _codex_prompts(meta))


def test_receipt_rejects_short_long_mixed_or_gapped_indices(tmp_path):
    settings = make_settings(tmp_path)
    meta = {"schema_version": 2, "status": "done", "prompt": "p"}
    frozen = _freeze(settings, meta)["_image_optimization"]
    meta["_image_optimization"] = frozen
    assert image_optimization.receipt(meta, settings) is not None
    meta["segments"] = [{"index": 1, "prompt": "a"}, {"index": 2, "prompt": "b"}]
    assert image_optimization.receipt(meta, settings) is None


def test_postprocess_uses_project_frozen_provider_settings(tmp_path):
    analysis_settings = make_settings(
        tmp_path, enable_mediakit_erase=True,
        seedream_model="doubao-seedream-4-5-251128",
        seedream_edit_mode="independent_parallel",
    )
    cid = _done(analysis_settings)
    meta = storage.load_meta(analysis_settings.data_dir, cid)
    storage.update_meta(
        analysis_settings.data_dir, cid,
        **_freeze(analysis_settings, meta),
    )
    runtime = replace(
        analysis_settings, seedream_model="doubao-seedream-5-0-260128",
        seedream_edit_mode="anchor_consistency",
        seedream_concurrency=99,
    )
    asyncio.run(postprocess.start(
        runtime, cid,
        {"confirm": True, "options": {"remove_subtitle": True, "remove_brand": False}},
        {},
    ))
    private = storage.load_meta(runtime.data_dir, cid)["_postprocess_receipt"]
    assert (private["model"], private["edit_mode"]) == (
        "doubao-seedream-4-5-251128", "independent_parallel",
    )
    assert "concurrency" not in private


def test_public_options_tamper_cannot_skip_frozen_seedream_or_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    stored = storage.load_meta(settings.data_dir, cid)["postprocess"]
    stored["options"] = {
        "remove_subtitle": False, "remove_brand": False, "optimize_image": False,
    }
    storage.update_meta(settings.data_dir, cid, postprocess=stored)
    monkeypatch.setattr(
        postprocess.seedream, "edit", lambda *args, **kwargs: pytest.fail("must not submit")
    )

    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
    ))

    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest["status"] == "failed"
    assert latest["error"] == "postprocess_receipt_invalid"
    assert not (settings.data_dir / cid / "work" / "postprocessed").exists()


def test_private_receipt_must_match_project_frozen_optimization(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": False, "optimize_image": False,
        }},
        {},
    ))
    frozen = storage.load_meta(settings.data_dir, cid)["_image_optimization"]
    frozen["model"] = "doubao-seedream-4-5-251128"
    storage.update_meta(settings.data_dir, cid, _image_optimization=frozen)
    monkeypatch.setattr(
        postprocess.mediakit, "erase_image", lambda *args, **kwargs: pytest.fail("must not submit")
    )

    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
    ))
    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest["status"] == "failed"
    assert latest["error"] == "postprocess_receipt_invalid"


def test_all_false_private_options_fail_closed_in_run_retry_and_recovery(tmp_path):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": False, "optimize_image": False,
        }},
        {},
    ))
    current = storage.load_meta(settings.data_dir, cid)
    false_options = {key: False for key in postprocess.OPTION_KEYS}
    private = current["_postprocess_receipt"]
    private["options"] = false_options
    public = current["postprocess"]
    public["options"] = false_options
    storage.update_meta(
        settings.data_dir, cid, _postprocess_receipt=private, postprocess=public
    )

    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
    ))
    failed = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert failed["error"] == "postprocess_receipt_invalid"

    failed["segments"][0].update(status="failed", error="segment_failed")
    storage.update_meta(settings.data_dir, cid, postprocess=failed)
    with pytest.raises(postprocess.PostprocessError) as caught:
        asyncio.run(postprocess.retry_segment(
            settings, cid, 0, {"confirm": True, "expected_revision": 1}, {}
        ))
    assert caught.value.detail == "postprocess_receipt_invalid"

    failed.update(status="running", error=None)
    failed["segments"][0].update(status="running", error=None)
    storage.update_meta(settings.data_dir, cid, postprocess=failed)
    assert postprocess.recover_running(settings) == []
    recovered = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert recovered["status"] == "failed"
    assert recovered["error"] == "postprocess_receipt_invalid"


def test_non_dict_postprocess_fails_closed_without_attribute_error(tmp_path):
    settings = make_settings(tmp_path)
    cid = _done(settings)
    storage.update_meta(settings.data_dir, cid, postprocess="corrupt")
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
    ))
    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest == {"status": "failed", "error": "postprocess_receipt_invalid"}


def test_prompt_patch_is_strict_cas_and_freezes_on_postprocess(tmp_path):
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(settings.data_dir, cid, **_freeze(settings, meta))
    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
        prompt = detail["image_optimization_prompt"]
        assert set(prompt) == {"text", "default_text", "sha256"}
        response = client.patch(
            f"/api/conversations/{cid}/image-optimization-prompt", headers=AUTH,
            json={"confirm": True, "segment_index": 0,
                  "expected_sha256": prompt["sha256"], "prompt": " replacement "},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "replacement"
        stale = client.patch(
            f"/api/conversations/{cid}/image-optimization-prompt", headers=AUTH,
            json={"confirm": True, "segment_index": 0,
                  "expected_sha256": prompt["sha256"], "prompt": "again"},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["code"] == "image_optimization_prompt_changed"
        storage.update_meta(settings.data_dir, cid, postprocess={"status": "running"})
        frozen = client.patch(
            f"/api/conversations/{cid}/image-optimization-prompt", headers=AUTH,
            json={"confirm": True, "segment_index": 0,
                  "expected_sha256": response.json()["sha256"], "prompt": "again"},
        )
        assert frozen.status_code == 409
        assert frozen.json()["detail"]["code"] == "image_optimization_prompt_frozen"


def test_private_global_continuity_never_enters_detail(tmp_path):
    settings = make_settings(tmp_path)
    cid = _done(settings)
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_continuity={"private": "GLOBAL_CONTINUITY_SECRET"},
    )
    with TestClient(create_app(settings)) as client:
        detail = client.get(f"/api/conversations/{cid}", headers=AUTH).json()
    encoded = json.dumps(detail, ensure_ascii=False)
    assert "_image_continuity" not in encoded
    assert "GLOBAL_CONTINUITY_SECRET" not in encoded


def test_prompt_patch_cannot_cross_first_submit_claim_window(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    frozen = _freeze(settings, meta)["_image_optimization"]
    storage.update_meta(settings.data_dir, cid, _image_optimization=frozen)
    current_sha = frozen["segments"][0]["sha256"]
    owner_writing = threading.Event()
    release_owner = threading.Event()
    original_write = storage._write_meta

    def blocked_write(cdir, current):
        owner = current.get("_input_owner")
        if isinstance(owner, dict) and owner.get("kind") == "submit":
            owner_writing.set()
            assert release_owner.wait(2)
        return original_write(cdir, current)

    monkeypatch.setattr(storage, "_write_meta", blocked_write)
    claim_result = []
    patch_errors = []

    def claim():
        claim_result.append(storage.claim_submission_input(
            settings.data_dir, cid, "request-race-1234"
        ))

    def patch():
        try:
            def mutate(current):
                current["_image_optimization"] = image_optimization.replace(
                    current, settings, 0, current_sha, "must not win"
                )
            storage.mutate_meta(settings.data_dir, cid, mutate)
        except image_optimization.ImageOptimizationError as exc:
            patch_errors.append(exc)

    claim_thread = threading.Thread(target=claim)
    patch_thread = threading.Thread(target=patch)
    claim_thread.start()
    assert owner_writing.wait(2)
    patch_thread.start()
    time.sleep(0.03)
    assert patch_thread.is_alive()  # blocked behind the claim's storage transaction
    release_owner.set()
    claim_thread.join(2)
    patch_thread.join(2)

    assert claim_result[0]["_input_owner"]["kind"] == "submit"
    assert patch_errors[0].detail["code"] == "image_optimization_prompt_frozen"
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["_image_optimization"]["segments"][0]["current"] != "must not win"


def test_seedream_retries_only_exact_quota_and_restores_png_size(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    calls = []
    output = base64.b64encode(_png(2, 2, 33)).decode()

    async def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"code": "QuotaExceeded"}})
        return httpx.Response(200, json={"data": [{"b64_json": output}]})

    transport = httpx.MockTransport(handler)
    out = tmp_path / "out.png"
    asyncio.run(seedream.edit(
        settings, [_png(5, 3)], "safe prompt", out,
        receipt_path=tmp_path / "attempt.json", transport=transport,
    ))
    assert len(calls) == 3
    decoded = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert decoded.shape[:2] == (3, 5)
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    receipt = json.loads((tmp_path / "attempt.json").read_text())
    assert receipt["status"] == "succeeded" and receipt["attempt"] == 3
    assert [item["status"] for item in receipt["attempts"]] == [
        "quota_retryable", "quota_retryable", "succeeded",
    ]
    assert "secret" not in json.dumps(receipt)


def test_seedream_timeout_is_submission_unknown_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous", request=request)

    async def run():
        await seedream.edit(
            settings, [_png()], "p", tmp_path / "out.png",
            receipt_path=tmp_path / "attempt.json", transport=httpx.MockTransport(handler),
        )
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(run())
    assert caught.value.code == "submission_unknown"
    assert calls == 1
    assert json.loads((tmp_path / "attempt.json").read_text())["status"] == "submission_unknown"


def test_seedream_unknown_existing_receipt_status_never_posts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    image = _png()
    prompt = "p"
    receipt = tmp_path / "attempt.json"
    receipt.write_text(json.dumps({
        "version": 1, "status": "provider_mystery", "attempt": 1,
        "model": settings.seedream_model, "mode": settings.seedream_edit_mode,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_sha256": [hashlib.sha256(image).hexdigest()], "attempts": [],
    }))
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            settings, [image], prompt, tmp_path / "out.png", receipt_path=receipt,
            transport=httpx.MockTransport(handler),
        ))
    assert caught.value.code == "attempt_receipt_invalid"
    assert calls == 0


def test_seedream_deterministic_4xx_never_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"code": "InvalidParameter"}})

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            settings, [_png()], "p", tmp_path / "out.png",
            receipt_path=tmp_path / "attempt.json", transport=httpx.MockTransport(handler),
        ))
    assert caught.value.code == "provider_rejected"
    assert calls == 1


def test_seedream_all_request_errors_are_unknown_and_retry_budget_is_hard_capped(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(
        access_token="x", data_dir=tmp_path, retry_count=99, retry_interval_s=0,
    )
    request_errors = 0

    async def broken(request):
        nonlocal request_errors
        request_errors += 1
        raise httpx.RemoteProtocolError("peer reset", request=request)

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            settings, [_png()], "p", tmp_path / "unknown.png",
            receipt_path=tmp_path / "unknown.json",
            transport=httpx.MockTransport(broken),
        ))
    assert caught.value.code == "submission_unknown" and request_errors == 1

    quota_calls = 0

    async def quota(_request):
        nonlocal quota_calls
        quota_calls += 1
        return httpx.Response(429, json={"error": {"code": "QuotaExceeded"}})

    with pytest.raises(seedream.SeedreamError):
        asyncio.run(seedream.edit(
            settings, [_png()], "p", tmp_path / "quota.png",
            receipt_path=tmp_path / "quota.json",
            transport=httpx.MockTransport(quota),
        ))
    assert quota_calls == 3
    assert json.loads((tmp_path / "quota.json").read_text())["attempt"] == 3


def test_seedream_cancel_during_post_persists_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    entered = asyncio.Event()

    async def hanging(_request):
        entered.set()
        await asyncio.Event().wait()

    async def drive():
        task = asyncio.create_task(seedream.edit(
            settings, [_png()], "p", tmp_path / "cancel.png",
            receipt_path=tmp_path / "cancel.json",
            transport=httpx.MockTransport(hanging),
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert json.loads((tmp_path / "cancel.json").read_text())["status"] == "submission_unknown"


def test_seedream_atomic_files_fsync_parent_directories(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(seedream, "_fsync_dir", lambda path: calls.append(path))
    seedream._atomic_json(tmp_path / "receipt.json", {"status": "x"})
    seedream._atomic_bytes(tmp_path / "result.bin", b"x")
    seedream._write_exact_png(_png(), tmp_path / "out.png", 5, 3)
    assert calls == [tmp_path, tmp_path, tmp_path]


def test_cancelled_postprocess_projects_unknown_and_failed_recovery_does_not_resubmit(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    entered = asyncio.Event()
    calls = 0

    async def submitting(_settings, _images, _prompt, _output, *, receipt_path, transport=None):
        nonlocal calls
        calls += 1
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"status": "submitting"}))
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(postprocess.seedream, "edit", submitting)

    async def drive():
        task = asyncio.create_task(postprocess.run_task(
            settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
        ))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    stored = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert stored["status"] == "failed" and stored["error"] == "submission_unknown"
    assert stored["segments"][0]["error"] == "submission_unknown"
    assert postprocess.public_state(stored)["error"] == "submission_unknown"
    assert postprocess.recover_running(settings) == []
    assert calls == 1


def test_anchor_first_frame_real_timeout_projects_submission_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(
        tmp_path, retry_interval_s=0, seedream_edit_mode="anchor_consistency"
    )
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    real_edit = seedream.edit
    calls = 0

    async def timeout_handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("anchor timeout", request=request)

    mock_transport = httpx.MockTransport(timeout_handler)

    async def edit_with_timeout(settings_, images, prompt, output, *, receipt_path, transport=None):
        return await real_edit(
            settings_, images, prompt, output, receipt_path=receipt_path,
            transport=mock_transport,
        )

    monkeypatch.setattr(postprocess.seedream, "edit", edit_with_timeout)
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1)
    ))

    stored = storage.load_meta(settings.data_dir, cid)["postprocess"]
    public = postprocess.public_state(stored)
    assert calls == 1
    assert stored["segments"][0]["error"] == "submission_unknown"
    assert stored["error"] == "submission_unknown"
    assert public["segments"][0]["error"] == "submission_unknown"
    assert public["error"] == "submission_unknown"


def test_postprocess_has_strict_stage_barriers_and_anchor_single_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(
        tmp_path, enable_mediakit_erase=True, seedream_edit_mode="anchor_consistency"
    )
    cid = _done(settings)
    cdir = settings.data_dir / cid
    # Add two more ordered frames.
    for number in (2, 3):
        (cdir / "work" / "keyframes" / f"{number:02d}.png").write_bytes(_png(value=number))
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(settings.data_dir, cid, keyframes=["01.png", "02.png", "03.png"],
                        **_freeze(settings, meta))
    events = []

    async def erase(_settings, _cdir, source, output, confirm, scenes):
        assert confirm is True and len(scenes) == 1
        events.append(("media", scenes[0], source.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    async def edit(_settings, images, prompt, output, *, receipt_path, transport=None):
        events.append(("seedream", len(images), output.name))
        assert prompt
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"status": "succeeded"}))
        return output

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/postprocess", headers=AUTH,
            json={"confirm": True, "options": {
                "remove_subtitle": True, "remove_brand": True, "optimize_image": True,
            }},
        )
        assert response.status_code == 200
    stages = [item[1] for item in events if item[0] == "media"]
    assert stages == [mediakit.TEXT_SCENE] * 3 + [mediakit.ICON_SCENE] * 3
    seedream_calls = [item for item in events if item[0] == "seedream"]
    assert [item[1] for item in seedream_calls] == [3, 2, 2]
    state = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert state["status"] == "done"
    assert state["options"]["optimize_image"] is True
    assert state["segments"] == [{
        "index": 0, "status": "done", "stage": "done", "completed_frames": 3,
        "total_frames": 3, "revision": 1, "error": None,
    }]


def test_brand_only_never_calls_seedream_and_old_two_options_are_canonical(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(settings.data_dir, cid, **_freeze(settings, meta))

    async def erase(_settings, _cdir, source, output, _confirm, _scenes):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    monkeypatch.setattr(postprocess.seedream, "edit", lambda *a, **k: pytest.fail("must not call"))
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/postprocess", headers=AUTH,
            json={"confirm": True, "options": {
                "remove_subtitle": False, "remove_brand": True,
            }},
        )
        assert response.status_code == 200
    assert storage.load_meta(settings.data_dir, cid)["postprocess"]["options"] == {
        "remove_subtitle": False, "remove_brand": True, "optimize_image": False,
    }


def test_manual_segment_retry_preserves_unknown_attempt_while_recovery_never_posts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(tmp_path)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    prompt = _freeze(settings, storage.load_meta(settings.data_dir, cid))[
        "_image_optimization"
    ]
    attempts = cdir / "work" / ".postprocess-private" / "0" / "attempts"
    attempts.mkdir(parents=True)
    old = attempts / "0001-r1.json"
    old.write_text(json.dumps({"status": "submission_unknown"}))
    storage.update_meta(
        settings.data_dir, cid, _image_optimization=prompt,
        _postprocess_receipt={
            "version": 2,
            "options": {"remove_subtitle": False, "remove_brand": False, "optimize_image": True},
            "model": settings.seedream_model, "edit_mode": settings.seedream_edit_mode,
            "timeout_s": settings.seedream_timeout_s,
            "prompts": prompt["segments"],
        },
        postprocess={
            "status": "failed",
            "options": {"remove_subtitle": False, "remove_brand": False, "optimize_image": True},
            "frames": [], "error": "cancelled",
            "segments": [{"index": 0, "status": "failed", "stage": "seedream",
                          "completed_frames": 0, "total_frames": 1, "revision": 1,
                          "error": "cancelled"}],
        },
    )
    assert postprocess.recover_running(settings) == []
    failed = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert failed["segments"][0]["status"] == "failed"
    assert failed["segments"][0]["error"] == "submission_unknown"
    assert failed["error"] == "submission_unknown"
    assert old.is_file()

    async def manually_confirmed(_settings, images, _prompt, output, *, receipt_path, transport=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"status": "succeeded"}))
        return output

    monkeypatch.setattr(postprocess.seedream, "edit", manually_confirmed)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"/api/conversations/{cid}/postprocess/segments/0/retry", headers=AUTH,
            json={"confirm": True, "expected_revision": 1},
        )
        assert response.status_code == 200
    assert old.is_file()
    assert (attempts / "0001-r2.json").is_file()
    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest["status"] == "done" and latest["segments"][0]["revision"] == 2


def test_recovery_ignores_old_unknown_revision_on_done_segment(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cid = _done(settings, segments=True)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **_freeze(settings, meta)
    )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": False, "optimize_image": False,
        }},
        {},
    ))
    post = storage.load_meta(settings.data_dir, cid)["postprocess"]
    post["segments"][0].update(
        status="done", stage="done", completed_frames=1, revision=2
    )
    storage.update_meta(settings.data_dir, cid, postprocess=post)
    old = (
        settings.data_dir / cid / "work" / ".postprocess-private"
        / "1" / "attempts" / "0001-r1.json"
    )
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"status": "submission_unknown"}))
    done_output = (
        settings.data_dir / cid / "work" / "segments" / "1"
        / "work" / "postprocessed" / "01.png"
    )
    done_output.parent.mkdir(parents=True)
    done_output.write_bytes(_png())

    recovered = postprocess.recover_running(settings)
    assert recovered == [cid]
    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest["status"] == "running"
    assert latest["segments"][0]["status"] == "done"
    assert latest["segments"][1]["status"] == "running"

    calls = []

    async def erase(_settings, _cdir, source, output, _confirm, _scenes):
        calls.append(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1)
    ))
    latest = storage.load_meta(settings.data_dir, cid)["postprocess"]
    assert latest["status"] == "done"
    assert len(calls) == 1 and "/segments/2/" in calls[0].as_posix()
    assert old.is_file()


def _v3_frame_bound_plan() -> dict:
    return {
        "version": 3,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": True,
        "reason": None,
        "person_plans": [{
            "id": "PERSON_01",
            "source_identity": "源叙事主人物",
            "replacement_identity": "不同的新人物设计",
            "wardrobe_change": "保持用途并改变款式",
            "local_color_change": "人物局部固有色明显改变",
            "reference": {"segment_index": 0, "frame_index": 1},
            "observable_segments": [0],
        }],
        "scene_plans": [{
            "id": "SCENE_01",
            "source_scene": "源环境",
            "replacement_scene": "同用途的真实新环境",
            "semantic_change": "环境语义明显改变",
            "geometry_changes": ["主要形状与连接关系改变"],
            "depth_changes": ["前后纵深改变"],
            "layout_changes": ["功能区域布局改变"],
            "local_color_change": "场景局部固有色明显改变",
            "reference": {"segment_index": 0, "frame_index": 1},
            "segments": [0],
        }],
        "segments": [{
            "segment_index": 0,
            "persons": [{
                "id": "PERSON_01",
                "state": "replace",
                "observable_frames": [1, 2],
                "target_region": "完整可见主人物",
                "boundary": "人物可见轮廓",
            }],
            "scene": {
                "scene_id": "SCENE_01",
                "target_region": "人物以外完整场景",
                "boundary": "人物与前景实体轮廓",
                "layout_reference_frame_index": 1,
            },
            "protected_non_target_people": [],
            "protected_relations": ["可见物理关系保持"],
            "frame_constraints": [
                {
                    "frame_index": 1,
                    "visible_body_parts": "帧一身体可见部位数量保持",
                    "pose_skeleton": "帧一姿态骨架保持",
                    "contact_points": "帧一接触点保持",
                    "occlusion_order": "帧一遮挡顺序保持",
                    "out_of_frame_crop": "帧一画外裁切保持",
                    "non_person_entity_ledger": {
                        "entities": [{
                            "entity_id": "ENTITY_01",
                            "description": "帧一可见非人物实体",
                            "visibility": "full",
                        }],
                        "relations": [{
                            "subject_id": "ENTITY_01",
                            "predicate": "contacts",
                            "object_id": "PERSON_01",
                        }],
                    },
                    "dominant_palette_contract": {
                        "area_weighted_warm_cool_family": "balanced",
                        "saturation_style": "muted",
                    },
                },
                {
                    "frame_index": 2,
                    "visible_body_parts": "帧二身体可见部位数量保持",
                    "pose_skeleton": "帧二姿态骨架保持",
                    "contact_points": "帧二接触点保持",
                    "occlusion_order": "帧二遮挡顺序保持",
                    "out_of_frame_crop": "帧二画外裁切保持",
                    "non_person_entity_ledger": {
                        "entities": [{
                            "entity_id": "ENTITY_01",
                            "description": "帧二可见非人物实体",
                            "visibility": "edge_fragment",
                        }],
                        "relations": [{
                            "subject_id": "ENTITY_01",
                            "predicate": "contacts",
                            "object_id": "PERSON_01",
                        }],
                    },
                    "dominant_palette_contract": {
                        "area_weighted_warm_cool_family": "balanced",
                        "saturation_style": "muted",
                    },
                },
            ],
            "photometric_contract": {
                "light_direction": "全局光源方向保持",
                "light_quality": "全局光线软硬保持",
                "exposure_or_intensity": "全局曝光强度保持",
                "wb_cct": "白平衡色温保持",
                "global_contrast": "全局对比保持",
                "tone_curve": "全局 tone curve 保持",
            },
        }],
    }


def _v4_frame_bound_plan() -> dict:
    plan = _v3_frame_bound_plan()
    plan["version"] = 4
    plan["scene_plans"][0]["continuity_graph"] = {
        "components": [{
            "component_id": "COMPONENT_01",
            "target_spec": "target-spec-01",
        }],
        "topology": [],
        "views": [
            {
                "segment_index": 0,
                "frame_index": 1,
                "transition_from_previous": "start",
                "observations": [{
                    "component_id": "COMPONENT_01",
                    "visibility": "full",
                }],
                "view_relations": [],
            },
            {
                "segment_index": 0,
                "frame_index": 2,
                "transition_from_previous": "same_camera",
                "observations": [{
                    "component_id": "COMPONENT_01",
                    "visibility": "full",
                }],
                "view_relations": [],
            },
        ],
    }
    return plan


def test_v4_canonical_plan_accepts_sparse_observations_as_prompt_facts():
    plan = _v4_frame_bound_plan()
    plan["person_plans"] = []
    plan["segments"][0]["persons"] = []
    for frame in plan["segments"][0]["frame_constraints"]:
        frame["non_person_entity_ledger"] = {"entities": [], "relations": []}
    plan["scene_plans"][0]["continuity_graph"]["views"][1]["observations"][0][
        "visibility"
    ] = "out_of_view"

    canonical = image_optimization.canonical_plan_v4(
        plan, [0], frame_counts={0: 2},
    )
    assert canonical["person_plans"] == []
    assert canonical["segments"][0]["persons"] == []
    assert canonical["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ] == {"entities": [], "relations": []}
    prompts = image_optimization.compile_frame_prompts(
        canonical, "anchor_consistency",
    )
    assert "当前帧场景视图：" in prompts[0][2]
    assert "dominant_palette_contract=" in prompts[0][2]


@pytest.mark.parametrize("eligible", [False, True])
def test_v4_empty_or_refusal_plan_is_protocol_error_not_content_ineligibility(eligible):
    value = {
        "version": 4,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": eligible,
        "reason": None if eligible else "scene_components_ambiguous",
        "person_plans": [],
        "scene_plans": [],
        "segments": [],
    }
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization._canonical_project_output(
            value, [0], "anchor_consistency", {0: 1},
        )


def test_v4_backend_freezes_source_pixel_palette_into_provider_prompts(tmp_path):
    paths = []
    for index, value in enumerate((31, 62), 1):
        path = tmp_path / f"{index:02d}.png"
        path.write_bytes(_png(value=value))
        paths.append(path)
    plan = _v4_frame_bound_plan()
    for frame in plan["segments"][0]["frame_constraints"]:
        frame["dominant_palette_contract"] = {
            "area_weighted_warm_cool_family": "warm",
            "saturation_style": "vivid",
        }

    canonical, prompts = image_optimization._canonical_project_output(
        plan,
        [0],
        "anchor_consistency",
        {0: 2},
        source_frames={0: paths},
    )

    expected = {
        "area_weighted_warm_cool_family": "balanced",
        "saturation_style": "muted",
    }
    assert all(
        frame["dominant_palette_contract"] == expected
        for frame in canonical["segments"][0]["frame_constraints"]
    )
    assert '"area_weighted_warm_cool_family":"balanced"' in prompts[0][1]


def test_v4_frame_receipt_deeply_binds_graph_view_plan_and_source(tmp_path):
    settings = make_settings(tmp_path)
    plan = _v4_frame_bound_plan()
    inventory = [
        {
            "segment_index": 0,
            "frame_index": frame_index,
            "frame_name": f"{frame_index:02d}.png",
            "source_sha256": str(frame_index) * 64,
            "source_transition_from_previous": (
                "start" if frame_index == 1 else "same_camera"
            ),
            "source_transition_evidence_sha256": (
                "a" * 64 if frame_index == 1 else "b" * 64
            ),
        }
        for frame_index in (1, 2)
    ]
    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "generic-profile", "revision": 1},
        model=settings.seedream_model,
        frame_inventory=inventory,
    )
    prompts = image_optimization.compile_frame_prompts(
        plan, settings.seedream_edit_mode
    )
    frozen = image_optimization.freeze_frame_prompts(
        settings, execution, prompts, plan=plan
    )
    meta = {
        **image_optimization.freeze_continuity(
            plan, frame_counts={0: 2}
        ),
        **frozen,
    }
    assert image_optimization.receipt(meta, settings) == frozen[
        "_image_optimization"
    ]

    changed_view = deepcopy(execution)
    changed_view["frames"][1]["scene_continuity_view"][
        "transition_from_previous"
    ] = "start"
    with pytest.raises(ValueError, match="frame prompts"):
        image_optimization.freeze_frame_prompts(
            settings, changed_view, prompts, plan=plan
        )

    changed_target_plan = deepcopy(plan)
    changed_target_plan["scene_plans"][0]["continuity_graph"]["components"][0][
        "target_spec"
    ] = "target-spec-revision"
    changed_target_meta = {
        **image_optimization.freeze_continuity(
            changed_target_plan, frame_counts={0: 2}
        ),
        **frozen,
    }
    assert image_optimization.receipt(changed_target_meta, settings) is None

    changed_view_plan = deepcopy(plan)
    changed_view_plan["scene_plans"][0]["continuity_graph"]["views"][1][
        "observations"
    ][0]["visibility"] = "edge_fragment"
    changed_view_meta = {
        **image_optimization.freeze_continuity(
            changed_view_plan, frame_counts={0: 2}
        ),
        **frozen,
    }
    assert image_optimization.receipt(changed_view_meta, settings) is None


def test_v4_schedule_freezes_unique_typed_paid_dag_order(tmp_path):
    settings = make_settings(tmp_path)
    plan = _v4_frame_bound_plan()
    inventory = [
        {
            "segment_index": 0,
            "frame_index": frame_index,
            "frame_name": f"{frame_index:02d}.png",
            "source_sha256": str(frame_index) * 64,
            "source_transition_from_previous": "start" if frame_index == 1 else "same_camera",
            "source_transition_evidence_sha256": str(frame_index + 2) * 64,
        }
        for frame_index in (1, 2)
    ]
    execution = image_optimization.freeze_execution_inputs(
        plan, revision=1, profile={"id": "image-postprocess", "revision": 1},
        model=settings.seedream_model, frame_inventory=inventory,
    )
    nodes = image_optimization._scene_anchor_schedule(plan, execution)["nodes"]
    assert [node["anchor"]["order"] for node in nodes] == list(range(1, len(nodes) + 1))
    assert [(node["scene_id"], node["label"]) for node in nodes] == [
        ("SCENE_01", "global"),
        ("SCENE_01", "layout-0000"),
        ("SCENE_01", "fanout-0000-0002"),
    ]


def test_v4_plan_rejects_model_transition_that_differs_from_backend_skeleton(tmp_path):
    settings = make_settings(tmp_path)
    session = tmp_path / "session"
    frames = session / "work" / "keyframes"
    frames.mkdir(parents=True)
    for number in (1, 2):
        (frames / f"{number:02d}.png").write_bytes(_png(value=number))
    skeleton = [
        {
            "segment_index": 0,
            "frame_index": number,
            "frame_name": f"{number:02d}.png",
            "source_sha256": hashlib.sha256(
                (frames / f"{number:02d}.png").read_bytes()
            ).hexdigest(),
            "source_transition_from_previous": "start" if number == 1 else "same_camera",
            "source_transition_evidence_sha256": str(number + 7) * 64,
        }
        for number in (1, 2)
    ]

    class Runner:
        def run_isolated(self, workdir, _prompt, *, session_dir):
            request = json.loads((Path(workdir) / "work" / "request.json").read_text())
            assert request["segments"][0]["transition_skeleton"] == skeleton
            plan = _v4_frame_bound_plan()
            plan["scene_plans"][0]["continuity_graph"]["views"][1][
                "transition_from_previous"
            ] = "camera_motion"
            (Path(workdir) / "work" / "image_optimization.json").write_text(
                json.dumps(plan), encoding="utf-8",
            )

    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_project_prompts(
            Runner(),
            [{
                "index": 0, "chain_id": "short-000", "join_mode": "hard_cut",
                "keyframes_dir": frames, "transition_skeleton": skeleton,
            }],
            settings.seedream_edit_mode,
            session_dir=session,
        )


class _PlanAuditRunner:
    def __init__(self, status: str, verify_status: str = "pass") -> None:
        self.status = status
        self.verify_status = verify_status
        self.calls = 0

    def run_isolated(self, workdir, _prompt, *, session_dir) -> None:
        self.calls += 1
        root = Path(workdir)
        request = json.loads((root / "work" / "request.json").read_text(
            encoding="utf-8"
        ))
        if request["phase"] == "verify":
            frozen = json.loads((root / "work" / "frozen_plan.json").read_text(
                encoding="utf-8"
            ))
            plan = {key: value for key, value in frozen.items() if key != "sha256"}
            status = self.verify_status
            base_check = {
                "status": "pass" if status == "fail" else status,
                "evidence": "current-frame verification evidence",
            }
            palette_check = {
                "status": status,
                "evidence": "current-frame palette evidence",
            }
            reason = None if status == "pass" else (
                "verification_unknown" if status == "unknown"
                else "dominant_palette_preservation_failed"
            )
            segments = []
            for segment in plan["segments"]:
                frame_checks = [
                    {
                        "frame_index": frame["frame_index"],
                        **{
                            key: (
                                dict(palette_check)
                                if key == "dominant_palette_contract"
                                else dict(base_check)
                            )
                            for key in (
                                "visible_body_parts", "pose_skeleton",
                                "contact_points", "occlusion_order",
                                "out_of_frame_crop", "non_person_entity_ledger",
                                "dominant_palette_contract",
                                "photometric_contract",
                            )
                        },
                    }
                    for frame in segment["frame_constraints"]
                ]
                segments.append({
                    "segment_index": segment["segment_index"],
                    "passed": status == "pass",
                    "person_checks": [{
                        "person_id": person["id"],
                        "identity_changed": dict(base_check),
                        "source_identity_absent": dict(base_check),
                        "local_color_change": dict(base_check),
                    } for person in segment["persons"]],
                    "scene_checks": {
                        key: dict(base_check)
                        for key in (
                            "semantic_change", "geometry_change", "depth_change",
                            "layout_change", "local_color_change",
                        )
                    },
                    "invariants": {
                        key: dict(base_check)
                        for key in (
                            "lighting_preservation", "interaction_preservation",
                            "cross_frame_continuity",
                        )
                    },
                    "frame_checks": frame_checks,
                })
            (root / "work" / "image_verification.json").write_text(
                json.dumps(
                    {
                        "version": 3,
                        "phase": "verify",
                        "plan_sha256": image_optimization.plan_sha256(plan),
                        "segment_indices": plan["segment_indices"],
                        "passed": status == "pass",
                        "reason": reason,
                        "segments": segments,
                        "project_checks": {
                            key: dict(base_check)
                            for key in (
                                "narrative_person_completeness", "no_identity_swap",
                                "no_unplanned_person", "person_identity_continuity",
                                "scene_continuity",
                            )
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return
        receipt = json.loads((root / "work" / "audit_inputs.json").read_text(
            encoding="utf-8"
        ))
        reason = None if self.status == "pass" else (
            "plan_audit_unknown" if self.status == "unknown" else "plan_audit_failed"
        )
        (root / "work" / "plan_audit.json").write_text(
            json.dumps(
                {
                    "version": 3,
                    "phase": "plan_audit",
                    "plan_sha256": receipt["plan_sha256"],
                    "continuity_sha256": receipt["continuity_sha256"],
                    "audit_input_sha256": receipt["sha256"],
                    "passed": self.status == "pass",
                    "reason": reason,
                    "frame_checks": [
                        {
                            "segment_index": item["segment_index"],
                            "frame_index": item["frame_index"],
                            "source_sha256": item["source_sha256"],
                            **{
                                key: {
                                    "status": self.status,
                                    "evidence": "current-frame audit evidence",
                                }
                                for key in (
                                    "body_closure", "scene_closure",
                                    "entity_closure", "relation_closure",
                                )
                            },
                        }
                        for item in receipt["frames"]
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class _V4LifecycleRunner:
    def __init__(self, failed_phase: str | None = None) -> None:
        self.failed_phase = failed_phase
        self.phases = []
        self.verification_error = None

    @staticmethod
    def _check(status: str = "pass", evidence: str = "generic evidence") -> dict:
        return {"status": status, "evidence": evidence}

    def run_isolated(self, workdir, _prompt, *, session_dir) -> None:
        root = Path(workdir)
        request = json.loads((root / "work" / "request.json").read_text(
            encoding="utf-8"
        ))
        phase = request["phase"]
        self.phases.append(phase)
        if phase == "verify" and self.failed_phase == "unavailable":
            raise RuntimeError("acceptance unavailable")
        frozen = json.loads((root / "work" / "frozen_plan.json").read_text(
            encoding="utf-8"
        ))
        plan = {key: value for key, value in frozen.items() if key != "sha256"}
        failed = phase == self.failed_phase
        status = "fail" if failed else "pass"
        if phase == "plan_audit":
            receipt = json.loads((root / "work" / "audit_inputs.json").read_text(
                encoding="utf-8"
            ))
            output = {
                "version": 4,
                "phase": "plan_audit",
                "plan_sha256": receipt["plan_sha256"],
                "continuity_sha256": receipt["continuity_sha256"],
                "audit_input_sha256": receipt["sha256"],
                "passed": not failed,
                "reason": None if not failed else "plan_audit_failed",
                "frame_checks": [{
                    "segment_index": item["segment_index"],
                    "frame_index": item["frame_index"],
                    "source_sha256": item["source_sha256"],
                    **{
                        key: self._check(status)
                        for key in (
                            "body_closure", "scene_closure", "entity_closure",
                            "relation_closure", "scene_continuity_closure",
                        )
                    },
                } for item in receipt["frames"]],
            }
            target = root / "work" / "plan_audit.json"
        elif phase == "verify_pack":
            evidence = " ".join(
                token
                for scene in plan["scene_plans"]
                for token in image_optimization._scene_continuity_evidence_tokens(scene)
            )
            output = {
                "version": 4,
                "phase": "verify_pack",
                "plan_sha256": image_optimization.plan_sha256(plan),
                "passed": not failed,
                "reason": None if not failed else "scene_geometry_change_failed",
                "persons": [{
                    "person_id": item["id"],
                    "passed": not failed,
                    "checks": {
                        key: self._check(status)
                        for key in image_optimization._PACK_PERSON_CHECKS
                    },
                } for item in plan["person_plans"]],
                "scenes": [{
                    "scene_id": item["id"],
                    "passed": not failed,
                    "checks": {
                        key: self._check(status, evidence)
                        for key in image_optimization._PACK_SCENE_CHECKS
                    },
                } for item in plan["scene_plans"]],
                "project": {
                    key: self._check(status)
                    for key in image_optimization._PACK_PROJECT_CHECKS
                },
            }
            target = root / "work" / "reference_pack_verification.json"
        elif phase == "verify":
            base = self._check("pass")
            segments = []
            for segment in plan["segments"]:
                segments.append({
                    "segment_index": segment["segment_index"],
                    "passed": not failed,
                    "person_checks": [{
                        "person_id": person["id"],
                        "identity_changed": dict(base),
                        "source_identity_absent": dict(base),
                        "local_color_change": dict(base),
                    } for person in segment["persons"]],
                    "scene_checks": {
                        key: dict(base) for key in (
                            "semantic_change", "geometry_change", "depth_change",
                            "layout_change", "local_color_change",
                        )
                    },
                    "invariants": {
                        key: dict(base) for key in (
                            "lighting_preservation", "interaction_preservation",
                            "cross_frame_continuity",
                        )
                    },
                    "frame_checks": [{
                        "frame_index": frame["frame_index"],
                        **{key: dict(base) for key in (
                            "visible_body_parts", "pose_skeleton", "contact_points",
                            "occlusion_order", "out_of_frame_crop",
                            "non_person_entity_ledger",
                            "dominant_palette_contract", "photometric_contract",
                        )},
                        "scene_continuity_view": self._check(status),
                    } for frame in segment["frame_constraints"]],
                })
            output = {
                "version": 4,
                "phase": "verify",
                "plan_sha256": image_optimization.plan_sha256(plan),
                "segment_indices": plan["segment_indices"],
                "passed": not failed,
                "reason": None if not failed else "scene_continuity_failed",
                "segments": segments,
                "project_checks": {
                    key: (
                        self._check(
                            status,
                            " ".join(
                                token for scene in plan["scene_plans"]
                                for token in image_optimization._scene_continuity_evidence_tokens(scene)
                            ),
                        )
                        if key == "scene_continuity" else dict(base)
                    ) for key in (
                        "narrative_person_completeness", "no_identity_swap",
                        "no_unplanned_person", "person_identity_continuity",
                        "scene_continuity",
                    )
                },
            }
            try:
                image_optimization.canonical_verification(output, plan)
            except Exception as exc:
                self.verification_error = repr(exc)
            target = root / "work" / "image_verification.json"
        else:
            raise AssertionError(f"unexpected phase: {phase}")
        target.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


def _freeze_v3_image_optimization(settings, cid: str, plan: dict) -> None:
    cdir = settings.data_dir / cid
    frames = sorted((cdir / "work" / "keyframes").glob("*.png"))
    inventory = [
        {
            "segment_index": 0,
            "frame_index": index,
            "frame_name": frame.name,
            "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        }
        for index, frame in enumerate(frames, 1)
    ]
    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 3},
        model=settings.seedream_model,
        frame_inventory=inventory,
    )
    frozen = image_optimization.freeze_frame_prompts(
        settings,
        execution,
        image_optimization.compile_frame_prompts(plan, settings.seedream_edit_mode),
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        keyframes=[frame.name for frame in frames],
        **image_optimization.freeze_continuity(plan, frame_counts={0: len(frames)}),
        **frozen,
    )


def _freeze_v4_image_optimization(settings, cid: str, plan: dict) -> None:
    cdir = settings.data_dir / cid
    frames = sorted((cdir / "work" / "keyframes").glob("*.png"))
    inventory = [
        {
            "segment_index": 0,
            "frame_index": index,
            "frame_name": frame.name,
            "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            "source_transition_from_previous": (
                "start" if index == 1 else "same_camera"
            ),
            "source_transition_evidence_sha256": (
                "a" * 64 if index == 1 else "b" * 64
            ),
        }
        for index, frame in enumerate(frames, 1)
    ]
    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 4},
        model=settings.seedream_model,
        frame_inventory=inventory,
    )
    frozen = image_optimization.freeze_frame_prompts(
        settings,
        execution,
        image_optimization.compile_frame_prompts(plan, settings.seedream_edit_mode),
        plan=plan,
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        keyframes=[frame.name for frame in frames],
        **image_optimization.freeze_continuity(plan, frame_counts={0: len(frames)}),
        **frozen,
    )


def test_v4_plan_skill_is_never_called_after_generation_or_before_web_publish(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    plan = _v4_frame_bound_plan()
    for frame in plan["segments"][0]["frame_constraints"]:
        frame["dominant_palette_contract"] = {
            "area_weighted_warm_cool_family": "warm",
            "saturation_style": "natural",
        }
    _freeze_v4_image_optimization(settings, cid, plan)
    posts = []

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        posts.append(output.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    for name in (
        "generate_plan_audit_verdict",
        "generate_reference_pack_verdict",
        "generate_project_verdict",
    ):
        monkeypatch.setattr(
            postprocess.image_optimization,
            name,
            lambda *_args, **_kwargs: pytest.fail(
                "plan-only image Skill must not run during generation"
            ),
        )
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner("verify")
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    latest = storage.load_meta(settings.data_dir, cid)
    assert runner.phases == []
    assert len(posts) == 3
    assert latest["postprocess"]["status"] == "done"
    assert "_image_verification" not in latest
    with TestClient(create_app(settings)) as client:
        response = client.get(
            f"/api/conversations/{cid}/files/postprocessed/01.png", headers=AUTH,
        )
    assert response.status_code == 200
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
            settings=settings,
        )


def test_v4_single_frame_valid_input_reaches_anchor_generation_without_quality_preflight(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    source = cdir / "work" / "keyframes" / "01.png"
    plan, prompts = image_optimization.generic_project_prompts(
        [{
            "index": 0,
            "chain_id": "short-000",
            "join_mode": "hard_cut",
            "keyframes_dir": source.parent,
            "transition_skeleton": [{
                "segment_index": 0,
                "frame_index": 1,
                "frame_name": "01.png",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_transition_from_previous": "start",
                "source_transition_evidence_sha256": "a" * 64,
            }],
        }],
        settings.seedream_edit_mode,
        session_dir=cdir,
    )
    assert [item["id"] for item in plan["person_plans"]] == ["PERSON_01"]
    assert plan["segments"][0]["persons"][0]["observable_frames"] == [1]
    assert "仅在当前帧确有可见人物时替换" in prompts[0][1]
    assert "若无则不得新增或补造人物" in prompts[0][1]
    _freeze_v4_image_optimization(settings, cid, plan)
    posts = []

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        posts.append(output.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner()
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    latest = storage.load_meta(settings.data_dir, cid)
    assert runner.phases == []
    assert len(posts) == 2
    assert latest["postprocess"]["status"] == "done"
    assert "_image_verification" not in latest
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, [cdir / "work" / "keyframes" / "01.png"],
            settings=settings,
        )


def test_v3_failed_quality_acceptance_publishes_but_remains_blocked_from_h3(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v3_image_optimization(settings, cid, _v3_frame_bound_plan())
    posts = []

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        posts.append(output.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _PlanAuditRunner("fail", verify_status="fail")
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    latest = storage.load_meta(settings.data_dir, cid)
    assert len(posts) == 2
    assert latest["postprocess"]["status"] == "done"
    assert runner.calls == 0
    assert "_image_verification" not in latest
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
            settings=settings,
        )


def test_v4_generic_fallback_uses_detection_only_to_choose_person_reference(
    tmp_path, monkeypatch,
):
    segment_dirs = []
    for segment_index, frame_count in ((1, 2), (2, 1)):
        directory = tmp_path / f"segment-{segment_index}" / "keyframes"
        directory.mkdir(parents=True)
        for frame_index in range(1, frame_count + 1):
            (directory / f"{frame_index:02d}.png").write_bytes(_png())
        segment_dirs.append(directory)

    monkeypatch.setattr(
        image_optimization,
        "_source_has_observable_person",
        lambda path: path.parent.parent.name == "segment-1" and path.name == "02.png",
    )
    segments = []
    previous = None
    for segment_index, directory in enumerate(segment_dirs, 1):
        skeleton = []
        for frame_index, source in enumerate(sorted(directory.glob("*.png")), 1):
            transition = "start" if previous is None else (
                "hard_cut" if frame_index == 1 else "same_camera"
            )
            skeleton.append({
                "segment_index": segment_index,
                "frame_index": frame_index,
                "frame_name": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_transition_from_previous": transition,
                "source_transition_evidence_sha256": "a" * 64,
            })
            previous = source
        segments.append({
            "index": segment_index,
            "chain_id": f"chain-{segment_index}",
            "join_mode": "hard_cut",
            "keyframes_dir": directory,
            "transition_skeleton": skeleton,
        })

    plan, prompts = image_optimization.generic_project_prompts(
        segments, "anchor_consistency", session_dir=tmp_path,
    )

    assert [person["id"] for person in plan["person_plans"]] == ["PERSON_01"]
    assert plan["person_plans"][0]["reference"] == {
        "segment_index": 1, "frame_index": 2,
    }
    assert plan["person_plans"][0]["observable_segments"] == [1, 2]
    assert plan["segments"][0]["persons"] == [{
        "id": "PERSON_01",
        "state": "replace",
        "observable_frames": [1, 2],
        "target_region": "当前帧实际可见的完整人物区域（若无可见人物则为空）",
        "boundary": "若人物可见则保持姿态、轮廓、裁切与遮挡；若无则不得新增或补造人物",
    }]
    assert plan["segments"][1]["persons"] == [{
        "id": "PERSON_01",
        "state": "replace",
        "observable_frames": [1],
        "target_region": "当前帧实际可见的完整人物区域（若无可见人物则为空）",
        "boundary": "若人物可见则保持姿态、轮廓、裁切与遮挡；若无则不得新增或补造人物",
    }]
    assert all(segment["protected_non_target_people"] == [] for segment in plan["segments"])
    assert all(
        "仅在当前帧确有可见人物时替换" in prompt
        and "若无则不得新增或补造人物" in prompt
        for segment_prompts in prompts.values()
        for prompt in segment_prompts.values()
    )


@pytest.mark.parametrize("status", ["fail", "unknown"])
def test_v3_plan_audit_is_not_called_before_seedream(
    tmp_path, monkeypatch, status,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    (settings.data_dir / cid / "work" / "keyframes" / "02.png").write_bytes(
        _png(value=62)
    )
    _freeze_v3_image_optimization(settings, cid, _v3_frame_bound_plan())
    calls = []

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        calls.append(output.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))

    runner = _PlanAuditRunner(status)
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1), audit_runner=runner,
    ))

    # A plan-only Skill is never invoked as either audit or acceptance.
    assert runner.calls == 0
    assert calls == ["01.png", "02.png"]
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"


def test_v4_combined_mediakit_preprocesses_all_canvases_before_seedream(
    tmp_path, monkeypatch,
):
    """A v4 project may only enter its paid DAG after every canvas is frozen."""
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, enable_mediakit_erase=True, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())
    events = []

    async def erase(_settings, _cdir, source, output, confirm, scenes):
        assert confirm is True and len(scenes) == 1
        events.append(("mediakit", scenes[0], source.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        receipt = {
            "version": 1, "state": "succeeded",
            "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "scenes": [scenes[0]],
            "stages": [{"scene": scenes[0], "state": "succeeded"}],
            "output": output.name,
        }
        path = output.parent / ".mediakit" / f"{output.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return output

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        events.append(("seedream",))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": True, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner()
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1), audit_runner=runner,
    ))

    assert events[:4] == [
        ("mediakit", mediakit.TEXT_SCENE, "01.png"),
        ("mediakit", mediakit.TEXT_SCENE, "02.png"),
        ("mediakit", mediakit.ICON_SCENE, "01.png"),
        ("mediakit", mediakit.ICON_SCENE, "02.png"),
    ]
    assert events[4:] == [("seedream",)] * 3
    assert runner.phases == []
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"
    canvas = latest["_v4_canvas_execution"]
    assert [item["canvas_sha256"] for item in canvas["frames"]]


def test_v4_mediakit_only_keeps_legacy_publish_and_h3_selection_path(tmp_path, monkeypatch):
    """A v4 image plan does not make a non-optimization MediaKit run an anchor DAG."""
    settings = make_settings(tmp_path, enable_mediakit_erase=True, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())
    calls = []

    async def erase(_settings, _cdir, source, output, _confirm, scenes):
        calls.append((scenes[0], source.name))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        receipt_path = output.parent / ".mediakit" / f"{output.name}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "version": 1, "state": "succeeded", "output": output.name,
            "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "scenes": [scenes[0]],
            "stages": [{"scene": scenes[0], "state": "succeeded"}],
        }), encoding="utf-8")
        return output

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    monkeypatch.setattr(postprocess.seedream, "edit", lambda *_a, **_k: pytest.fail("no Seedream"))
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": False, "optimize_image": False,
        }},
        {},
    ))
    asyncio.run(postprocess.run_task(settings, cid, asyncio.Semaphore(1)))

    assert calls == [(mediakit.TEXT_SCENE, "01.png"), (mediakit.TEXT_SCENE, "02.png")]
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"
    assert postprocess.generation_keyframes(
        cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
    ) == sorted((cdir / "work" / "postprocessed").glob("*.png"))
    receipt = cdir / "work" / ".postprocess-private" / "0" / "text" / ".mediakit" / "01.png.json"
    original_receipt = receipt.read_text(encoding="utf-8")
    drifted = json.loads(original_receipt)
    drifted["source"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
        )
    receipt.write_text(original_receipt, encoding="utf-8")
    receipt.unlink()
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
        )


def test_mediakit_stage_never_reuses_an_output_without_its_provider_receipt(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path, enable_mediakit_erase=True)
    cdir = tmp_path / "project"
    source = cdir / "work" / "keyframes" / "01.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(_png())
    stale = cdir / "work" / ".postprocess-private" / "0" / "text" / "01.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(_png())
    calls = []

    async def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("an output-only hit is an unknown paid state")

    monkeypatch.setattr(postprocess.mediakit, "erase_image", forbidden)
    with pytest.raises(postprocess.PostprocessError, match="submission_unknown"):
        asyncio.run(postprocess._mediakit_stage(
            settings, cdir, 0, [source], "text", mediakit.TEXT_SCENE, asyncio.Semaphore(1),
        ))
    assert calls == []


def test_v4_staged_canonical_outputs_are_not_visible_before_project_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    staged = cdir / "work" / "postprocessed"
    staged.mkdir(parents=True)
    (staged / "01.png").write_bytes(_png())
    # The endpoint gate only depends on the project not being published.  Do
    # not let TestClient startup launch this deliberately incomplete v4 DAG
    # against its real Codex runner while checking a read-only files request.
    state = storage.load_meta(settings.data_dir, cid)["postprocess"]
    storage.update_meta(
        settings.data_dir, cid,
        postprocess={**state, "status": "failed", "error": "test_unpublished"},
    )

    with TestClient(create_app(settings)) as client:
        response = client.get(
            f"/api/conversations/{cid}/files/postprocessed/01.png", headers=AUTH,
        )
    assert response.status_code == 404


def test_v4_runtime_publishes_without_in_band_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    source_two = _png(value=62)
    (cdir / "work" / "keyframes" / "02.png").write_bytes(source_two)
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())
    calls = []

    async def staged_edit(_settings, images, _prompt, output, *, receipt_path):
        calls.append(images)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", staged_edit)
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner("verify")
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    assert runner.phases == []
    assert len(calls) == 3
    assert [len(images) for images in calls] == [1, 2, 2]
    private = cdir / "work" / ".postprocess-private" / "scene-anchors" / "SCENE_01"
    assert json.loads((private / "global.json").read_text())["input_roles"] == ["canvas"]
    assert not (private / "pack-alternate.json").exists()
    assert json.loads((private / "layout-0000.json").read_text())["input_roles"] == [
        "canvas", "global_scene_anchor",
    ]
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"
    assert "_image_verification" not in latest
    assert latest["postprocess"]["frames"] == ["01.png", "02.png"]
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
            settings=settings,
        )


def test_v4_combined_canvas_receipt_binds_generated_sources_without_acceptance(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, enable_mediakit_erase=True, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())

    async def erase(_settings, _cdir, source, output, _confirm, scenes):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_png(value=101))
        receipt = {
            "version": 1, "state": "succeeded",
            "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "scenes": [scenes[0]],
            "stages": [{"scene": scenes[0], "state": "succeeded"}],
            "output": output.name,
        }
        path = output.parent / ".mediakit" / f"{output.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return output

    posts = []

    async def edit(_settings, images, _prompt, output, *, receipt_path):
        posts.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(images[0])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.mediakit, "erase_image", erase)
    monkeypatch.setattr(postprocess.seedream, "edit", edit)
    asyncio.run(postprocess.start(
        settings, cid,
        {"confirm": True, "options": {
            "remove_subtitle": True, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner()
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    latest = storage.load_meta(settings.data_dir, cid)
    canvas = latest["_v4_canvas_execution"]
    assert latest["postprocess"]["status"] == "done"
    assert runner.phases == []
    assert "_image_verification" not in latest
    assert canvas["frames"][0]["canvas_sha256"] != hashlib.sha256(
        (cdir / "work" / "keyframes" / "01.png").read_bytes()
    ).hexdigest()
    assert len(posts) == 3
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir, latest, sorted((cdir / "work" / "keyframes").glob("*.png")),
            settings=settings,
        )


def test_v4_invalid_global_anchor_blocks_layout_fanout_and_acceptance(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v4_image_optimization(settings, cid, _v4_frame_bound_plan())
    calls = []

    async def invalid_anchor(_settings, images, _prompt, output, *, receipt_path):
        calls.append(images)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"not-an-image")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", invalid_anchor)
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _V4LifecycleRunner()
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner, verification_runner=runner,
    ))

    assert runner.phases == []
    assert len(calls) == 1 and len(calls[0]) == 1
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "failed"
    assert latest["postprocess"]["error"] == "scene_anchor_verification_failed"
    assert not (cdir / "work" / "postprocessed").exists()


def test_v3_frame_receipt_binds_each_seedream_http_body_to_its_source_frame(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    sources = [_png(value=31), _png(value=62)]
    (cdir / "work" / "keyframes" / "01.png").write_bytes(sources[0])
    (cdir / "work" / "keyframes" / "02.png").write_bytes(sources[1])
    plan = _v3_frame_bound_plan()
    inventory = [
        {
            "segment_index": 0,
            "frame_index": index,
            "frame_name": f"{index:02d}.png",
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }
        for index, source in enumerate(sources, 1)
    ]
    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 3},
        model=settings.seedream_model,
        frame_inventory=inventory,
    )
    assert execution["frames"][0]["frame_constraint"][
        "non_person_entity_ledger"
    ] == plan["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]
    assert execution["frames"][0]["frame_constraint"][
        "dominant_palette_contract"
    ] == plan["segments"][0]["frame_constraints"][0][
        "dominant_palette_contract"
    ]
    frozen = image_optimization.freeze_frame_prompts(
        settings,
        execution,
        image_optimization.compile_frame_prompts(plan, settings.seedream_edit_mode),
    )
    storage.update_meta(
        settings.data_dir,
        cid,
        keyframes=["01.png", "02.png"],
        **image_optimization.freeze_continuity(plan, frame_counts={0: 2}),
        **frozen,
    )
    created = storage.load_meta(settings.data_dir, cid)
    assert image_optimization.continuity_receipt(created) is not None
    assert image_optimization.dual_target_plan_receipt(created) is not None
    assert image_optimization.receipt(created, settings) is not None
    captured = []
    response_image = base64.b64encode(_png(value=99)).decode()

    async def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"b64_json": response_image}]})

    real_edit = seedream.edit

    async def capture_edit(settings_, images, prompt, output, *, receipt_path, transport=None):
        return await real_edit(
            settings_, images, prompt, output, receipt_path=receipt_path,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(postprocess.seedream, "edit", capture_edit)
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    audit_meta = storage.load_meta(settings.data_dir, cid)
    audit_plan, audit_inputs, audit_segments = postprocess._v3_plan_audit_inputs(
        audit_meta,
        postprocess._private_receipt(audit_meta),
        postprocess._group_targets(cdir, audit_meta),
    )
    assert audit_inputs["plan_sha256"] == image_optimization.plan_sha256(audit_plan)
    assert audit_segments == [{
        "index": 0, "source_keyframes_dir": cdir / "work" / "keyframes",
    }]
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=_PlanAuditRunner("pass"),
    ))

    assert len(captured) == 2, storage.load_meta(settings.data_dir, cid)["postprocess"]["error"]
    prompts = [body["prompt"] for body in captured]
    assert any("帧一身体可见部位数量保持" in prompt for prompt in prompts)
    assert any("帧二身体可见部位数量保持" in prompt for prompt in prompts)
    assert any("帧一可见非人物实体" in prompt for prompt in prompts)
    assert any("帧二可见非人物实体" in prompt for prompt in prompts)
    assert all(len(body["image"]) == 1 for body in captured)
    latest = storage.load_meta(settings.data_dir, cid)
    assert latest["postprocess"]["status"] == "done"
    receipt = image_optimization.receipt(latest, settings)
    assert receipt is not None and receipt["version"] == 3
    assert {
        (item["segment_index"], item["frame_name"], item["source_sha256"])
        for item in receipt["frames"]
    } == {
        (item["segment_index"], item["frame_name"], item["source_sha256"])
        for item in inventory
    }
    assert "_image_verification" not in latest
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir,
            latest,
            sorted((cdir / "work" / "keyframes").glob("*.png")),
            settings=settings,
        )

    tampered_continuity = deepcopy(latest)
    tampered_continuity["_image_continuity"]["segments"][0][
        "frame_constraints"
    ][0]["non_person_entity_ledger"]["entities"][0][
        "description"
    ] = "被篡改的当前帧实体描述"
    assert image_optimization.continuity_receipt(tampered_continuity) is None
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.dual_target_plan_receipt(tampered_continuity)

    changed_plan = deepcopy(plan)
    changed_plan["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]["entities"][0]["description"] = "另一份合法当前帧实体描述"
    changed_plan["segments"][0]["frame_constraints"][0][
        "dominant_palette_contract"
    ]["saturation_style"] = "vivid"
    mismatched_receipts = deepcopy(latest)
    mismatched_receipts.update(
        image_optimization.freeze_continuity(changed_plan, frame_counts={0: 2})
    )
    assert image_optimization.continuity_receipt(mismatched_receipts) is not None
    assert image_optimization.receipt(mismatched_receipts, settings) is None

    damaged_execution = deepcopy(execution)
    damaged_execution["frames"][0]["frame_constraint"][
        "non_person_entity_ledger"
    ]["relations"][0]["object_id"] = "ENTITY_99"
    with pytest.raises(ValueError):
        image_optimization.freeze_frame_prompts(settings, damaged_execution, prompts)

    malformed_observable_ids = deepcopy(execution)
    malformed_observable_ids["frames"][0]["observable_person_ids"] = [[]]
    with pytest.raises(ValueError, match="invalid image optimization frame prompts"):
        image_optimization.freeze_frame_prompts(
            settings, malformed_observable_ids, prompts
        )

    for mutate in (
        lambda value: value["frames"].pop(),
        lambda value: value["frames"].append(dict(value["frames"][0])),
        lambda value: value["frames"][0].update(source_sha256="0" * 64),
    ):
        damaged = deepcopy(latest)
        mutate(damaged["_image_optimization"])
        assert image_optimization.receipt(damaged, settings) is None


def test_v3_publish_does_not_create_in_band_acceptance(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = make_settings(tmp_path, retry_interval_s=0)
    cid = _done(settings)
    cdir = settings.data_dir / cid
    (cdir / "work" / "keyframes" / "02.png").write_bytes(_png(value=62))
    _freeze_v3_image_optimization(settings, cid, _v3_frame_bound_plan())

    async def staged_edit(_settings, _images, _prompt, output, *, receipt_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_png(value=99))
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(postprocess.seedream, "edit", staged_edit)
    asyncio.run(postprocess.start(
        settings,
        cid,
        {"confirm": True, "options": {
            "remove_subtitle": False, "remove_brand": False, "optimize_image": True,
        }},
        {},
    ))
    runner = _PlanAuditRunner("pass", verify_status="fail")
    asyncio.run(postprocess.run_task(
        settings, cid, asyncio.Semaphore(1), asyncio.Semaphore(1),
        audit_runner=runner,
    ))

    latest = storage.load_meta(settings.data_dir, cid)
    assert runner.calls == 0, latest["postprocess"]["segments"][0]["error"]
    assert latest["postprocess"]["status"] == "done"
    assert "_image_verification" not in latest
    assert (cdir / "work" / "postprocessed" / "01.png").is_file()
    assert (cdir / "work" / ".postprocess-private" / "0" / "seedream" / "01.png").is_file()
    with pytest.raises(postprocess.PostprocessError, match="artifacts_invalid"):
        postprocess.generation_keyframes(
            cdir,
            latest,
            sorted((cdir / "work" / "keyframes").glob("*.png")),
        )
