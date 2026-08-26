import asyncio
import base64
import hashlib
import json
import threading
import time
from dataclasses import replace

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


def test_seedream_settings_are_closed_and_secret_is_not_a_setting(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "super-secret")
    settings = Settings(access_token="x")
    assert settings.seedream_model == "doubao-seedream-5-0-pro-260628"
    assert settings.seedream_edit_mode == "independent_parallel"
    assert settings.seedream_prompt_template == "balanced"
    assert settings.seedream_timeout_s == 300.0
    assert "super-secret" not in repr(settings)
    assert not hasattr(settings, "ark_api_key")
    for field, value in (
        ("seedream_model", "unknown"),
        ("seedream_edit_mode", "all-at-once"),
        ("seedream_prompt_template", "extreme"),
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
    changes = image_optimization.freeze_prompts(settings, meta)
    frozen = changes["_image_optimization"]
    assert [item["segment_index"] for item in frozen["segments"]] == [1, 2]
    assert frozen["model"] == settings.seedream_model
    assert frozen["edit_mode"] == "independent_parallel"
    for item in frozen["segments"]:
        assert item["default"] == item["current"]
        assert item["sha256"] == hashlib.sha256(item["current"].encode()).hexdigest()
        assert "文字" in item["current"] and "同一套新设计" in item["current"]


@pytest.mark.parametrize("indices", ([0, 1], [1, 3], [True, 2]))
def test_long_prompt_segments_must_be_positive_and_contiguous(tmp_path, indices):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2, "status": "done",
        "segments": [{"index": index, "prompt": "p"} for index in indices],
    }
    with pytest.raises(ValueError):
        image_optimization.freeze_prompts(settings, meta)


def test_receipt_rejects_short_long_mixed_or_gapped_indices(tmp_path):
    settings = make_settings(tmp_path)
    meta = {"schema_version": 2, "status": "done", "prompt": "p"}
    frozen = image_optimization.freeze_prompts(settings, meta)["_image_optimization"]
    meta["_image_optimization"] = frozen
    assert image_optimization.receipt(meta, settings) is not None
    meta["segments"] = [{"index": 1, "prompt": "a"}, {"index": 2, "prompt": "b"}]
    assert image_optimization.receipt(meta, settings) is None


def test_postprocess_uses_project_frozen_provider_settings(tmp_path):
    analysis_settings = make_settings(
        tmp_path, enable_mediakit_erase=True,
        seedream_model="doubao-seedream-4-5-251128",
        seedream_edit_mode="independent_parallel", seedream_prompt_template="strong",
    )
    cid = _done(analysis_settings)
    meta = storage.load_meta(analysis_settings.data_dir, cid)
    storage.update_meta(
        analysis_settings.data_dir, cid,
        **image_optimization.freeze_prompts(analysis_settings, meta),
    )
    runtime = replace(
        analysis_settings, seedream_model="doubao-seedream-5-0-260128",
        seedream_edit_mode="anchor_consistency", seedream_prompt_template="light",
        seedream_concurrency=99,
    )
    asyncio.run(postprocess.start(
        runtime, cid,
        {"confirm": True, "options": {"remove_subtitle": True, "remove_brand": False}},
        {},
    ))
    private = storage.load_meta(runtime.data_dir, cid)["_postprocess_receipt"]
    assert (private["model"], private["edit_mode"], private["prompt_template"]) == (
        "doubao-seedream-4-5-251128", "independent_parallel", "strong",
    )
    assert "concurrency" not in private


def test_public_options_tamper_cannot_skip_frozen_seedream_or_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "secret")
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    storage.update_meta(
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
    storage.update_meta(settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta))
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


def test_prompt_patch_cannot_cross_first_submit_claim_window(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    cid = _done(settings)
    meta = storage.load_meta(settings.data_dir, cid)
    frozen = image_optimization.freeze_prompts(settings, meta)["_image_optimization"]
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
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
                        **image_optimization.freeze_prompts(settings, meta))
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
    prompt = image_optimization.freeze_prompts(settings, storage.load_meta(settings.data_dir, cid))[
        "_image_optimization"
    ]
    attempts = cdir / "work" / ".postprocess-private" / "0" / "attempts"
    attempts.mkdir(parents=True)
    old = attempts / "0001-r1.json"
    old.write_text(json.dumps({"status": "submission_unknown"}))
    storage.update_meta(
        settings.data_dir, cid, _image_optimization=prompt,
        _postprocess_receipt={
            "version": 1,
            "options": {"remove_subtitle": False, "remove_brand": False, "optimize_image": True},
            "model": settings.seedream_model, "edit_mode": settings.seedream_edit_mode,
            "prompt_template": settings.seedream_prompt_template,
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
        settings.data_dir, cid, **image_optimization.freeze_prompts(settings, meta)
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
