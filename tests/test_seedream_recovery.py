from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import seedream, seedream_recovery
from app.codex_runner import CodexRunner
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


def _valid_result(prompt: str, neutralized: str) -> dict:
    return {
        "version": 1,
        "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "neutralized_prompt": neutralized,
        "protected_literals": seedream_recovery._protected_literals(prompt),
        "semantic_fidelity": {
            field: True for field in seedream_recovery._FIDELITY_FIELDS
        },
        "changes": [{"original": "激烈对抗", "neutralized": "克制互动"}],
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

    def neutralize(_settings, _session, prompt):
        codex_calls.append(prompt)
        return _valid_result(prompt, neutralized)

    monkeypatch.setattr(seedream, "edit", edit)
    monkeypatch.setattr(seedream_recovery, "_run_codex", neutralize)
    result = asyncio.run(seedream_recovery.edit_with_content_recovery(
        _settings(tmp_path), [b"image"], original, out,
        receipt_path=receipt, session_dir=session,
    ))
    assert result == out
    assert provider_calls == [(original, None), (neutralized, 1)]
    assert codex_calls == [original]
    diagnostic = json.loads(
        receipt.with_name("0001-r1.neutralization.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "succeeded"
    assert diagnostic["original_prompt_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert diagnostic["neutralized_prompt_sha256"] == hashlib.sha256(neutralized.encode()).hexdigest()
    assert diagnostic["codex_call"] == {
        "call_path": ["postprocess", "seedream", "0001-r1", "neutralize"],
        "model": "gpt-5.6-luna",
        "reasoning_effort": "max",
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
    assert diagnostic["status"] == "codex_failed"
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
    monkeypatch.setattr(seedream_recovery, "_write_recovery", cannot_write)
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
    value = _valid_result(original, neutralized)
    parsed = seedream_recovery._validate_output(
        json.dumps(value, ensure_ascii=False).encode(), original_prompt=original,
    )
    assert set(parsed["semantic_fidelity"]) == seedream_recovery._FIDELITY_FIELDS
    assert parsed["protected_literals"] == ["@asset/hero", "a" * 64, "scene-001"]
    damaged = json.loads(json.dumps(value))
    damaged["neutralized_prompt"] = neutralized.replace("scene-001", "scene-002")
    with pytest.raises(ValueError, match="stable binding"):
        seedream_recovery._validate_output(
            json.dumps(damaged).encode(), original_prompt=original,
        )


def test_neutralizer_runner_explicitly_uses_luna_max(tmp_path: Path) -> None:
    argv = CodexRunner(
        timeout_s=1,
        concurrency=1,
        model=seedream_recovery.MODEL,
        reasoning_effort=seedream_recovery.REASONING_EFFORT,
    ).build_argv(tmp_path, "prompt")
    assert argv[argv.index("-m") + 1] == "gpt-5.6-luna"
    configs = [argv[index + 1] for index, item in enumerate(argv) if item == "-c"]
    assert 'model_reasoning_effort="max"' in configs
    assert 'model_reasoning_effort="medium"' not in configs


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
