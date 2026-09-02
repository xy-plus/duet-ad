from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import multiprocessing
import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import seedream, seedream_recovery
from app.codex_runner import CodexError
from app.deepseek_runner import DeepSeekRunner
from app.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(access_token="x", data_dir=tmp_path, codex_timeout_s=17)


def _png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((4, 6, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    session = tmp_path / "conversation"
    attempts = session / "work" / ".postprocess-private" / "0" / "attempts"
    attempts.mkdir(parents=True)
    receipt = attempts / "0001-r1.json"
    receipt.write_text("{}", encoding="utf-8")
    return session, receipt, tmp_path / "out.png"


def _exclusive_claim_worker(path: str, payload: dict, start, results) -> None:
    start.wait()
    results.put(seedream._exclusive_json(Path(path), payload))


def _valid_result(prompt: str, neutralized: str, semantic_context=None) -> dict:
    contract = seedream_recovery.freeze_semantic_contract(prompt, semantic_context)
    return {
        "version": 1,
        "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "semantic_contract_sha256": contract["sha256"],
        "neutralized_free_text": neutralized,
    }


def test_only_explicit_content_rejection_triggers_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, receipt, out = _paths(tmp_path)
    codex_calls: list[str] = []

    async def rejected(*_args, **_kwargs):
        raise seedream.SeedreamError(
            "provider_rejected", provider_error_code="InvalidParameter"
        )

    monkeypatch.setattr(seedream, "edit", rejected)
    monkeypatch.setattr(
        seedream_recovery,
        "_run_codex",
        lambda *_args: codex_calls.append("called"),
    )
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], "prompt", out,
            receipt_path=receipt, session_dir=session,
        ))
    assert caught.value.code == "provider_rejected"
    assert codex_calls == []


@pytest.mark.parametrize(
    "provider_code",
    ["InputTextSensitiveContentDetected", "OutputImageSensitiveContentDetected"],
)
def test_content_rejection_neutralizes_once_and_resubmits_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_code: str,
) -> None:
    session, receipt, out = _paths(tmp_path)
    original = '人物关系保持，激烈对抗；"scene_id":"scene-001"'
    neutralized = '人物关系保持，克制互动；"scene_id":"scene-001"'
    provider_calls: list[tuple[str, int | None]] = []
    codex_calls: list[str] = []

    async def edit(_settings, _images, prompt, output, **kwargs):
        provider_calls.append((prompt, kwargs.get("max_post_attempts")))
        if prompt == original:
            raise seedream.SeedreamError(
                "provider_rejected", provider_error_code=provider_code,
            )
        output.write_bytes(b"png")
        return output

    def neutralize(_settings, _session, prompt, _contract):
        codex_calls.append(prompt)
        return _valid_result(prompt, neutralized)

    monkeypatch.setattr(seedream, "edit", edit)
    monkeypatch.setattr(seedream_recovery, "_run_codex", neutralize)
    result = asyncio.run(seedream_recovery.edit_with_content_recovery(
        _settings(tmp_path), [b"image"], original, out,
        receipt_path=receipt, session_dir=session,
    ))
    assert result == out
    assert provider_calls[0] == (original, None)
    assert provider_calls[1][1] == 1
    assert provider_calls[1][0].startswith(neutralized)
    assert "BACKEND_IMMUTABLE_SEMANTIC_CONTRACT" in provider_calls[1][0]
    assert codex_calls == [original]
    diagnostic = json.loads(
        receipt.with_name("0001-r1.neutralization.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "succeeded"
    assert diagnostic["original_prompt_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert diagnostic["neutralized_free_text"] == neutralized
    assert diagnostic["neutralized_prompt_sha256"] == hashlib.sha256(
        provider_calls[1][0].encode()
    ).hexdigest()
    assert diagnostic["codex_call"] == {
        "call_path": ["postprocess", "seedream", "0001-r1", "neutralize"],
        "provider": "deepseek",
        "model": "deepseek-v4-flash-vision-exp",
        "thinking": "disabled",
        "attempt": 1,
    }


def test_second_provider_rejection_closes_provider_rejected_without_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, receipt, out = _paths(tmp_path)
    original = "激烈对抗"
    neutralized = "克制互动"
    calls: list[Path] = []

    async def rejected(_settings, _images, prompt, _out, **kwargs):
        current_receipt = kwargs["receipt_path"]
        calls.append(current_receipt)
        current_receipt.write_text("{}", encoding="utf-8")
        code = (
            "InputTextSensitiveContentDetected"
            if prompt == original else "OutputImageSensitiveContentDetected"
        )
        raise seedream.SeedreamError("provider_rejected", provider_error_code=code)

    monkeypatch.setattr(seedream, "edit", rejected)
    monkeypatch.setattr(
        seedream_recovery, "_run_codex",
        lambda *_args: _valid_result(original, neutralized),
    )
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], original, out,
            receipt_path=receipt, session_dir=session,
        ))
    assert caught.value.code == "provider_rejected"
    assert calls == [receipt, receipt.with_name("0001-r1.neutralized.json")]
    diagnostic = json.loads(
        receipt.with_name("0001-r1.neutralization.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "provider_rejected"
    assert diagnostic["provider_error_codes"] == [
        "InputTextSensitiveContentDetected",
        "OutputImageSensitiveContentDetected",
    ]
    assert len(diagnostic["provider_error_traces"]) == 2


def test_codex_failure_keeps_original_provider_rejection_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, receipt, out = _paths(tmp_path)
    provider_calls = 0
    codex_calls = 0

    async def rejected(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise seedream.SeedreamError(
            "provider_rejected",
            provider_error_code="InputTextSensitiveContentDetected",
        )

    def codex_failed(*_args):
        nonlocal codex_calls
        codex_calls += 1
        raise RuntimeError("codex failed")

    monkeypatch.setattr(seedream, "edit", rejected)
    monkeypatch.setattr(seedream_recovery, "_run_codex", codex_failed)
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], "prompt", out,
            receipt_path=receipt, session_dir=session,
        ))
    assert caught.value.code == "provider_rejected"
    assert caught.value.provider_error_code == "InputTextSensitiveContentDetected"
    assert provider_calls == 1
    assert codex_calls == 1
    diagnostic = json.loads(
        receipt.with_name("0001-r1.neutralization.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "failed"
    assert diagnostic["stage"] == "codex"
    assert diagnostic["codex_error"]["type"] == "RuntimeError"
    with pytest.raises(seedream.SeedreamError):
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], "prompt", out,
            receipt_path=receipt, session_dir=session,
        ))
    assert provider_calls == 2  # The durable original receipt is inspected again.
    assert codex_calls == 1


def test_recovery_receipt_failure_never_masks_original_provider_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, receipt, out = _paths(tmp_path)

    async def rejected(*_args, **_kwargs):
        raise seedream.SeedreamError(
            "provider_rejected",
            provider_error_code="InputTextSensitiveContentDetected",
        )

    def cannot_write(*_args):
        raise OSError("disk failure")

    monkeypatch.setattr(seedream, "edit", rejected)
    monkeypatch.setattr(seedream, "_exclusive_json", cannot_write)
    monkeypatch.setattr(
        seedream_recovery, "_run_codex",
        lambda *_args: pytest.fail("must not invoke Codex without a durable budget receipt"),
    )
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], "prompt", out,
            receipt_path=receipt, session_dir=session,
        ))
    assert caught.value.code == "provider_rejected"
    assert caught.value.provider_error_code == "InputTextSensitiveContentDetected"


def test_schema_preserves_all_semantic_contract_fields_and_stable_literals() -> None:
    original = (
        '主体A对主体B激烈对抗，先靠近再停下；镜头固定。'
        '"scene_id":"scene-001","source_sha256":"' + "a" * 64 + '" @asset/hero'
    )
    neutralized = (
        '主体A对主体B克制互动，先靠近再停下；镜头固定。'
        '"scene_id":"scene-001","source_sha256":"' + "a" * 64 + '" @asset/hero'
    )
    semantic_context = {
        "entities": [
            {"stable_key": "主体A", "count": 1},
            {"stable_key": "主体B", "count": 1},
        ],
        "relations": [{
            "subject_key": "主体A", "predicate": "approaches",
            "object_key": "主体B", "order": 1,
        }],
        "camera": {"composition": "fixed"},
    }
    contract = seedream_recovery.freeze_semantic_contract(original, semantic_context)
    value = _valid_result(original, neutralized, semantic_context)
    parsed = seedream_recovery._validate_output(
        json.dumps(value, ensure_ascii=False).encode(),
        original_prompt=original, semantic_contract=contract,
    )
    assert parsed["semantic_contract_sha256"] == contract["sha256"]
    assert contract["protected_literals"] == ["@asset/hero", "a" * 64, "scene-001"]
    assert contract["structured_semantics"]["relations"][0] == {
        "object_key": "主体B", "order": 1, "predicate": "approaches",
        "subject_key": "主体A",
    }
    damaged = json.loads(json.dumps(value))
    damaged["semantic_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable input"):
        seedream_recovery._validate_output(
            json.dumps(damaged).encode(), original_prompt=original,
            semantic_contract=contract,
        )
    published = seedream_recovery._inject_semantic_contract(
        "仅保留克制互动，不写数量或关系", contract,
    )
    # Even a non-literal deletion cannot remove direction, quantity, bindings,
    # camera or environment: the backend reconstructs the authoritative block.
    assert '"count":1' in published
    assert '"predicate":"approaches"' in published
    assert '"subject_key":"主体A"' in published
    assert '"object_key":"主体B"' in published


def test_real_frozen_frame_schema_never_reinjects_rejected_free_text() -> None:
    rejected = "两件玩具激烈碰撞并产生危险冲突"
    original = f"请编辑画面：{rejected}"
    frame = {
        "segment_index": 2,
        "frame_index": 3,
        "frame_name": "03.png",
        "source_sha256": "b" * 64,
        "default": rejected,
        "current": rejected,
        "prompt": rejected,
        "frame_constraint": {
            "frame_index": 3,
            "visible_body_parts": rejected,
            "relation_occurrences": [{
                "relation_id": "REL_01",
                "subject_key": "toy-a",
                "predicate": "contacts",
                "object_key": "toy-b",
                "state": "active",
                "geometry": "left-to-right",
                "preserve": True,
                "count": 2,
                "order": 1,
                "time": 1.25,
            }],
        },
        "scene_continuity_view": {
            "scene_id": "SCENE_01",
            "camera": "fixed",
            "composition": "two-shot",
            "environment": "indoor-workbench",
            "transition_from_previous": "same_camera",
            "unknown_notes": {"text": rejected, "anything": rejected},
        },
        # An unknown parent must never grant blanket access to its descendants.
        "unknown_dict": {
            "default": rejected,
            "arbitrary": rejected,
            "nested": {"free_text": rejected},
        },
    }
    contract = seedream_recovery.freeze_semantic_contract(
        original, {"frozen_frame": frame},
    )
    retry_prompt = seedream_recovery._inject_semantic_contract(
        "两件玩具进行轻柔、克制的接触", contract,
    )

    assert rejected not in retry_prompt
    assert '"relation_id":"REL_01"' in retry_prompt
    assert '"subject_key":"toy-a"' in retry_prompt
    assert '"predicate":"contacts"' in retry_prompt
    assert '"object_key":"toy-b"' in retry_prompt
    assert '"count":2' in retry_prompt
    assert '"order":1' in retry_prompt
    assert '"time":1.25' in retry_prompt
    assert '"camera":"fixed"' in retry_prompt
    assert '"composition":"two-shot"' in retry_prompt
    assert '"environment":"indoor-workbench"' in retry_prompt
    assert '"source_sha256":"' + "b" * 64 + '"' in retry_prompt
    assert "unknown_dict" not in retry_prompt


def test_neutralizer_runner_explicitly_uses_deepseek_transport(tmp_path: Path) -> None:
    runner = DeepSeekRunner(
        timeout_s=1,
        concurrency=1,
        credential_file=tmp_path / "deepseek.env",
    )
    assert seedream_recovery.MODEL == "deepseek-v4-flash-vision-exp"
    assert seedream_recovery.REASONING_EFFORT == "disabled"
    with pytest.raises(CodexError, match="schema-constrained"):
        runner.run(tmp_path, "prompt")


def test_seedream_exposes_and_durably_recovers_policy_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "secret")
    receipt = tmp_path / "attempt.json"

    async def reject(_request):
        return httpx.Response(400, json={
            "error": {"code": "InputTextSensitiveContentDetected"},
        })

    kwargs = {
        "receipt_path": receipt,
        "transport": httpx.MockTransport(reject),
    }
    with pytest.raises(seedream.SeedreamError) as first:
        asyncio.run(seedream.edit(
            _settings(tmp_path), [_png()], "prompt", tmp_path / "out.png", **kwargs,
        ))
    assert first.value.code == "provider_rejected"
    assert first.value.provider_error_code == "InputTextSensitiveContentDetected"
    assert json.loads(receipt.read_text())["provider_error_code"] == (
        "InputTextSensitiveContentDetected"
    )
    with pytest.raises(seedream.SeedreamError) as replay:
        asyncio.run(seedream.edit(
            _settings(tmp_path), [_png()], "prompt", tmp_path / "out.png", **kwargs,
        ))
    assert replay.value.provider_error_code == "InputTextSensitiveContentDetected"


def test_neutralized_seedream_budget_hard_caps_provider_post_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "secret")
    calls = 0

    async def quota(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"code": "QuotaExceeded"}})

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            _settings(tmp_path), [_png()], "neutral prompt", tmp_path / "out.png",
            receipt_path=tmp_path / "attempt.json",
            transport=httpx.MockTransport(quota),
            max_post_attempts=1,
        ))
    assert caught.value.code == "provider_rejected"
    assert calls == 1


def test_seedream_paid_receipt_has_one_cross_thread_post_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "secret")
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(2)
    encoded = __import__("base64").b64encode(_png()).decode()

    async def handler(_request):
        nonlocal calls
        with calls_lock:
            calls += 1
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    transport = httpx.MockTransport(handler)
    receipt = tmp_path / "attempt.json"
    output = tmp_path / "output.png"

    def invoke():
        start.wait()
        try:
            return asyncio.run(seedream.edit(
                _settings(tmp_path), [_png()], "same frozen prompt", output,
                receipt_path=receipt, transport=transport,
            ))
        except seedream.SeedreamError as exc:
            return exc.code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _item: invoke(), range(2)))

    assert calls == 1
    assert output in outcomes
    assert set(item for item in outcomes if isinstance(item, str)) <= {"submission_unknown"}
    assert json.loads(receipt.read_text())["status"] == "succeeded"
    assert seedream._claim_path(receipt).is_file()

    # A network replay is strictly local and cannot issue another POST.
    assert asyncio.run(seedream.edit(
        _settings(tmp_path), [_png()], "same frozen prompt", output,
        receipt_path=receipt, transport=transport,
    )) == output
    assert calls == 1


def test_seedream_claim_has_one_multiprocess_winner_and_crash_is_get_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "secret")
    context = multiprocessing.get_context("fork")
    claim = tmp_path / "attempt.json.post-claim"
    prompt = "same frozen prompt"
    image = _png()
    settings = _settings(tmp_path)
    binding = {
        "model": settings.seedream_model,
        "mode": settings.seedream_edit_mode,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_sha256": [hashlib.sha256(image).hexdigest()],
    }
    request_sha = hashlib.sha256(json.dumps(
        binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    payload = {
        "version": 1,
        "kind": "seedream_paid_post",
        "request_sha256": request_sha,
        "claimed_at_unix_ns": 1,
    }
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_exclusive_claim_worker,
            args=(str(claim), payload, start, results),
        )
        for _index in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=1) for _process in processes]
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 3
    assert json.loads(claim.read_text()) == payload

    # The winner is now treated as if it crashed before writing the receipt.
    # The surviving process must not infer that no provider POST occurred.
    calls = 0

    async def forbidden_post(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            settings, [image], prompt, tmp_path / "out.png",
            receipt_path=tmp_path / "attempt.json",
            transport=httpx.MockTransport(forbidden_post),
        ))
    assert caught.value.code == "submission_unknown"
    assert calls == 0


def test_neutralization_claim_caps_codex_and_retry_post_across_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARK_API_KEY", "secret")
    session = tmp_path / "conversation"
    attempts = session / "work" / ".postprocess-private" / "0" / "attempts"
    attempts.mkdir(parents=True)
    receipt = attempts / "0001-r1.json"
    output = session / "output.png"
    original = "两件玩具激烈碰撞"
    neutral = "两件玩具轻柔接触"
    context = {
        "entities": [
            {"stable_key": "toy-a", "count": 1},
            {"stable_key": "toy-b", "count": 1},
        ],
        "relations": [{
            "subject_key": "toy-a", "predicate": "contacts",
            "object_key": "toy-b", "order": 1,
        }],
    }
    post_prompts: list[str] = []
    post_lock = threading.Lock()
    start = threading.Barrier(2)
    codex_calls = 0
    codex_lock = threading.Lock()
    encoded = __import__("base64").b64encode(_png()).decode()

    async def handler(request):
        prompt = json.loads(request.content)["prompt"]
        with post_lock:
            post_prompts.append(prompt)
        await asyncio.sleep(0.08)
        if prompt == original:
            return httpx.Response(400, json={
                "error": {"code": "InputTextSensitiveContentDetected"},
            })
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    def neutralize(_settings, _session, prompt, contract):
        nonlocal codex_calls
        with codex_lock:
            codex_calls += 1
        time.sleep(0.08)
        return {
            "version": 1,
            "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "semantic_contract_sha256": contract["sha256"],
            "neutralized_free_text": neutral,
        }

    monkeypatch.setattr(seedream_recovery, "_run_codex", neutralize)
    transport = httpx.MockTransport(handler)

    def invoke():
        start.wait()
        try:
            return asyncio.run(seedream_recovery.edit_with_content_recovery(
                _settings(tmp_path), [_png()], original, output,
                receipt_path=receipt, session_dir=session,
                semantic_context=context, transport=transport,
            ))
        except seedream.SeedreamError as exc:
            return exc.code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _item: invoke(), range(2)))

    assert codex_calls == 1
    assert len(post_prompts) == 2
    assert post_prompts.count(original) == 1
    neutralized_posts = [item for item in post_prompts if item != original]
    assert len(neutralized_posts) == 1
    assert '"count":1' in neutralized_posts[0]
    assert '"predicate":"contacts"' in neutralized_posts[0]
    assert output in outcomes

    # Replay after both receipts settle is entirely local.
    assert asyncio.run(seedream_recovery.edit_with_content_recovery(
        _settings(tmp_path), [_png()], original, output,
        receipt_path=receipt, session_dir=session,
        semantic_context=context, transport=transport,
    )) == output
    assert codex_calls == 1
    assert len(post_prompts) == 2

    # A crash after the neutralized provider receipt settles but before the
    # diagnostic terminal update is repaired from local bytes, not reposted.
    diagnostic = receipt.with_name("0001-r1.neutralization.json")
    crashed = json.loads(diagnostic.read_text())
    crashed.update(status="submission_unknown", stage="provider_retry")
    seedream._atomic_json(diagnostic, crashed)
    output.unlink()
    assert asyncio.run(seedream_recovery.edit_with_content_recovery(
        _settings(tmp_path), [_png()], original, output,
        receipt_path=receipt, session_dir=session,
        semantic_context=context, transport=transport,
    )) == output
    assert len(post_prompts) == 2
    assert json.loads(diagnostic.read_text())["status"] == "succeeded"


def test_crashed_codex_claim_stays_submission_unknown_without_new_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, receipt, out = _paths(tmp_path)
    prompt = "激烈对抗"
    contract = seedream_recovery.freeze_semantic_contract(prompt, None)
    diagnostic = receipt.with_name("0001-r1.neutralization.json")
    seedream._exclusive_json(diagnostic, {
        "version": 1,
        "status": "submission_unknown",
        "stage": "codex",
        "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "semantic_contract_sha256": contract["sha256"],
    })
    provider_calls = 0

    async def original_replay(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise seedream.SeedreamError(
            "provider_rejected",
            provider_error_code="InputTextSensitiveContentDetected",
        )

    monkeypatch.setattr(seedream, "edit", original_replay)
    monkeypatch.setattr(
        seedream_recovery, "_run_codex",
        lambda *_args: pytest.fail("crashed claim must never rerun Codex"),
    )
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream_recovery.edit_with_content_recovery(
            _settings(tmp_path), [b"image"], prompt, out,
            receipt_path=receipt, session_dir=session,
        ))
    assert caught.value.code == "submission_unknown"
    assert provider_calls == 1  # receipt inspection mock, not a new real POST
