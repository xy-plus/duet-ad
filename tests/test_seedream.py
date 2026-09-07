from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import seedream
from app.config import SEEDREAM_API_BASE, SEEDREAM_PRO_MODEL, Settings, get_settings
from app.providers import ProviderFailure, ProviderResult
from app.seedream import SeedreamClient, SeedreamError


def _png(width: int, height: int) -> bytes:
    image = np.zeros((height, width), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    return encoded.tobytes()


_PNG = _png(2048, 2048)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://seedream-app.test",
        "seedream_api_key": "ark-secret",
        "seedream_api_base": "https://ark.test/api/v3",
        "seedream_model": "seedream-test-model",
        "seedream_timeout_s": 1.0,
        "store_lock_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _cause(exc: pytest.ExceptionInfo[ProviderFailure]) -> SeedreamError:
    assert isinstance(exc.value.cause, SeedreamError)
    return exc.value.cause


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seedream, "retry_sleep", lambda _seconds: None)


def _success_response() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Content-Type": "application/json", "X-Request-Id": "req-1"},
        json={"data": [{"b64_json": base64.b64encode(_PNG).decode("ascii")}]},
    )


def test_seedream_settings_have_closed_defaults_and_environment_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = Settings(
        access_token="test-access",
        public_base_url="https://seedream-app.test",
    )
    assert defaults.seedream_api_base == SEEDREAM_API_BASE
    assert defaults.seedream_model == SEEDREAM_PRO_MODEL
    assert defaults.seedream_timeout_s == 300.0
    assert defaults.seedream_api_key == ""

    monkeypatch.setenv("ACCESS_TOKEN", "test-access")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://seedream-app.test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("ARK_API_KEY", "  ark-key  ")
    monkeypatch.setenv("SEEDREAM_API_BASE", "https://ark.override/api/v3/")
    monkeypatch.setenv("SEEDREAM_MODEL", "seedream-override")
    monkeypatch.setenv("SEEDREAM_TIMEOUT_S", "42")
    configured = get_settings()
    assert configured.seedream_api_key == "ark-key"
    assert configured.seedream_api_base == "https://ark.override/api/v3"
    assert configured.seedream_model == "seedream-override"
    assert configured.seedream_timeout_s == 42.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seedream_api_base", "http://ark.test/api/v3", "SEEDREAM_API_BASE"),
        ("seedream_api_base", "https://user@ark.test/api/v3", "SEEDREAM_API_BASE"),
        ("seedream_api_base", "https://ark.test/api/\n/v3", "SEEDREAM_API_BASE"),
        ("seedream_api_key", "bad\nkey", "ARK_API_KEY"),
        ("seedream_api_key", "bad key", "ARK_API_KEY"),
        ("seedream_model", "bad model", "SEEDREAM_MODEL"),
        ("seedream_model", "", "SEEDREAM_MODEL"),
        ("seedream_timeout_s", 0, "SEEDREAM_TIMEOUT_S"),
        ("seedream_timeout_s", float("nan"), "SEEDREAM_TIMEOUT_S"),
        ("seedream_timeout_s", True, "SEEDREAM_TIMEOUT_S"),
    ],
)
def test_seedream_settings_reject_invalid_structure(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _settings(tmp_path, **{field: value})


def test_generate_posts_once_and_persists_only_redacted_receipts(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(_PNG)
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "seedream-receipt.json"
    calls = 0
    encoded = base64.b64encode(_PNG).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.method == "POST"
        assert request.url == httpx.URL("https://ark.test/api/v3/images/generations")
        assert request.headers["authorization"] == "Bearer ark-secret"
        body = json.loads(request.content)
        assert body["model"] == "seedream-test-model"
        assert body["prompt"] == "one reference sheet"
        assert body["size"] == "2048x2048"
        assert body["response_format"] == "b64_json"
        assert body["watermark"] is False
        assert len(body["image"]) == 1
        assert body["image"][0].startswith("data:image/png;base64,")
        return _success_response()

    result = SeedreamClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).generate(
        prompt="one reference sheet",
        images=[source],
        destination=destination,
        receipt_path=receipt_path,
    )

    assert isinstance(result, ProviderResult)
    assert result.value == destination
    assert result.evidence_media == [destination]
    assert destination.read_bytes() == _PNG
    assert calls == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    claim = json.loads(
        (tmp_path / "seedream-receipt.json.post-claim").read_text(encoding="utf-8")
    )
    marker = json.loads(
        (tmp_path / "seedream-receipt.json.response.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "succeeded"
    assert receipt["size"] == "2048x2048"
    assert receipt["request_sha256"] == claim["request_sha256"]
    assert receipt["request_sha256"] == marker["request_sha256"]
    assert marker["width"] == 2048
    assert marker["height"] == 2048
    assert (tmp_path / "seedream-receipt.json.result").read_bytes() == _PNG
    serialized = json.dumps(
        {
            "record": result.record,
            "receipt": receipt,
            "claim": claim,
            "marker": marker,
        },
        ensure_ascii=False,
    )
    assert "ark-secret" not in serialized
    assert encoded not in serialized
    assert result.record["request"]["headers"]["Authorization"] == "<redacted>"
    assert result.record["request"]["body"]["input_receipts"][0]["sha256"]


def test_text_to_image_omits_image_field(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "image" not in body
        return _success_response()

    result = SeedreamClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).generate(
        prompt="all replacement elements in one sheet",
        images=[],
        destination=tmp_path / "sheet.png",
        receipt_path=tmp_path / "receipt.json",
    )
    assert result.value.read_bytes() == _PNG
    assert result.record["request"]["body"]["input_receipts"] == []


@pytest.mark.parametrize("stored_status", ["response_received", "succeeded"])
def test_received_response_replays_locally_without_post(
    tmp_path: Path,
    stored_status: str,
) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "receipt.json"
    calls = 0

    def first_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response()

    settings = _settings(tmp_path)
    SeedreamClient(settings, httpx.MockTransport(first_handler)).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = stored_status
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    destination.unlink()

    def must_not_post(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"local replay must not POST: {request}")

    replayed = SeedreamClient(
        settings, httpx.MockTransport(must_not_post)
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    assert calls == 1
    assert replayed.record["replayed"] is True
    assert destination.read_bytes() == _PNG
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "succeeded"


def test_succeeded_result_replays_without_current_provider_key(tmp_path: Path) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "receipt.json"
    configured = _settings(tmp_path)
    SeedreamClient(
        configured, httpx.MockTransport(lambda _request: _success_response())
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    frozen_receipt = receipt_path.read_bytes()
    destination.unlink()

    replayed = SeedreamClient(
        _settings(tmp_path, seedream_api_key=""),
        httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    assert replayed.record["replayed"] is True
    assert destination.read_bytes() == _PNG
    assert receipt_path.read_bytes() == frozen_receipt


def test_complete_response_survives_transport_close_failure(tmp_path: Path) -> None:
    calls = 0

    class CloseFailingTransport(httpx.MockTransport):
        def close(self) -> None:
            raise RuntimeError("injected close failure after response")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response()

    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "receipt.json"
    result = SeedreamClient(
        _settings(tmp_path), CloseFailingTransport(handler),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )

    assert calls == 1
    assert result.value.read_bytes() == _PNG
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == (
        "succeeded"
    )


def test_submitting_with_durable_response_marker_replays_after_crash(tmp_path: Path) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "receipt.json"
    settings = _settings(tmp_path)
    SeedreamClient(settings, httpx.MockTransport(lambda _request: _success_response())).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "submitting"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    destination.unlink()

    replayed = SeedreamClient(
        settings,
        httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    assert replayed.record["replayed"] is True
    assert destination.read_bytes() == _PNG


def test_transport_ambiguity_is_terminal_and_never_reposts(tmp_path: Path) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def ambiguous(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous", request=request)

    client = SeedreamClient(_settings(tmp_path), httpx.MockTransport(ambiguous))
    with pytest.raises(ProviderFailure) as first:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(first).code == "submission_unknown"
    assert first.value.__cause__ is None
    assert first.value.__context__ is None
    # Bounded retry: first attempt + 2 retries before declaring ambiguity.
    assert calls == 3
    assert [
        entry["error"]["type"] for entry in first.value.record["attempts"]
    ] == ["ReadTimeout"] * 3
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "submission_unknown"

    with pytest.raises(ProviderFailure) as repeated:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(repeated).code == "submission_unknown"
    assert calls == 3


def test_concurrent_call_observes_durable_claim_and_cannot_post_twice(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    post_started = threading.Event()
    allow_response = threading.Event()
    calls = 0
    first_result: list[ProviderResult[Path]] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        post_started.set()
        assert allow_response.wait(timeout=2)
        return _success_response()

    settings = _settings(tmp_path)

    def first_call() -> None:
        first_result.append(
            SeedreamClient(settings, httpx.MockTransport(handler)).generate(
                prompt="frozen prompt",
                images=[],
                destination=destination,
                receipt_path=receipt_path,
            )
        )

    worker = threading.Thread(target=first_call)
    worker.start()
    assert post_started.wait(timeout=2)
    try:
        with pytest.raises(ProviderFailure) as competing:
            SeedreamClient(
                settings,
                httpx.MockTransport(
                    lambda request: pytest.fail(f"second call must not POST: {request}")
                ),
            ).generate(
                prompt="frozen prompt",
                images=[],
                destination=destination,
                receipt_path=receipt_path,
            )
        assert _cause(competing).code == "submission_unknown"
    finally:
        allow_response.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(first_result) == 1
    assert calls == 1
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "succeeded"


def test_local_replay_write_failure_is_not_misclassified_as_bad_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "receipt.json"
    settings = _settings(tmp_path)
    SeedreamClient(settings, httpx.MockTransport(lambda _request: _success_response())).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    destination.unlink()
    real_atomic_bytes = seedream._atomic_bytes

    def fail_output(path: Path, payload: bytes) -> None:
        if path == destination:
            raise OSError("disk is read-only")
        real_atomic_bytes(path, payload)

    monkeypatch.setattr(seedream, "_atomic_bytes", fail_output)
    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(
            settings,
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).generate(
            prompt="frozen prompt",
            images=[],
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "local_persistence_error"
    assert caught.value.record["outcome"] == "response_received"


def test_provider_rejection_is_terminal_and_redacted(tmp_path: Path) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def rejected(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad prompt"}},
        )

    client = SeedreamClient(_settings(tmp_path), httpx.MockTransport(rejected))
    with pytest.raises(ProviderFailure) as first:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(first).code == "provider_rejected"
    assert _cause(first).provider_error_code == "InvalidParameter"
    assert calls == 1
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "rejected"
    assert receipt["provider_error_code"] == "InvalidParameter"
    assert "ark-secret" not in json.dumps(first.value.record)

    with pytest.raises(ProviderFailure) as repeated:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(repeated).code == "provider_rejected"
    assert calls == 1


@pytest.mark.parametrize("status_code", [302, 408, 409, 425, 429, 500, 502, 503])
def test_ambiguous_http_status_is_unknown_and_never_reposts(
    tmp_path: Path,
    status_code: int,
) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def ambiguous(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": {"code": "Ambiguous"}})

    client = SeedreamClient(_settings(tmp_path), httpx.MockTransport(ambiguous))
    with pytest.raises(ProviderFailure) as first:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(first).code == "submission_unknown"
    assert first.value.record["outcome"] == "submission_unknown"
    # Gateway statuses are retried; every other ambiguous status is terminal
    # after a single attempt.
    expected_calls = 3 if status_code in {502, 503} else 1
    assert calls == expected_calls
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "submission_unknown"

    with pytest.raises(ProviderFailure) as repeated:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(repeated).code == "submission_unknown"
    assert calls == expected_calls


def test_untrusted_response_header_values_are_not_persisted(tmp_path: Path) -> None:
    leaked = "Bearer ark-secret data:image/png;base64,secret"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Request-Id": leaked,
                "X-Trace-Id": leaked,
            },
            json={"data": [{"b64_json": base64.b64encode(_PNG).decode("ascii")}]},
        )

    receipt_path = tmp_path / "receipt.json"
    result = SeedreamClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=tmp_path / "sheet.png",
        receipt_path=receipt_path,
    )
    marker = json.loads(
        (tmp_path / "receipt.json.response.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps({"record": result.record, "marker": marker})
    assert leaked not in serialized
    assert marker["response_headers"] == {"content-type": "application/json"}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"data": []}),
        httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(_PNG).decode("ascii")},
                    {"b64_json": base64.b64encode(_PNG).decode("ascii")},
                ],
            },
        ),
        httpx.Response(200, json={"data": [{"b64_json": "%%%"}]}),
        httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"not-image").decode("ascii")}]},
        ),
        httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(_png(1, 1)).decode("ascii")}
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "data": [
                    {
                        "b64_json": base64.b64encode(
                            _png(2048, 1024)
                        ).decode("ascii")
                    }
                ]
            },
        ),
    ],
    ids=[
        "non-json",
        "zero-images",
        "multiple-images",
        "invalid-base64",
        "not-an-image",
        "one-by-one-image",
        "wrong-dimensions-and-aspect-ratio",
    ],
)
def test_protocol_error_is_terminal_and_never_reposts(
    tmp_path: Path,
    response: httpx.Response,
) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def malformed(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    client = SeedreamClient(_settings(tmp_path), httpx.MockTransport(malformed))
    with pytest.raises(ProviderFailure) as first:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(first).code == "provider_protocol_error"
    assert calls == 1
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "protocol_error"

    with pytest.raises(ProviderFailure) as repeated:
        client.generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(repeated).code == "provider_protocol_error"
    assert calls == 1


def test_request_binding_mismatch_never_posts(tmp_path: Path) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success_response()

    client = SeedreamClient(_settings(tmp_path), httpx.MockTransport(handler))
    client.generate(
        prompt="first prompt",
        images=[],
        destination=tmp_path / "sheet.png",
        receipt_path=receipt_path,
    )
    with pytest.raises(ProviderFailure) as caught:
        client.generate(
            prompt="different prompt",
            images=[],
            destination=tmp_path / "different.png",
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "attempt_receipt_invalid"
    assert calls == 1


def test_claim_without_receipt_is_ambiguous_and_never_posts(tmp_path: Path) -> None:
    calls = 0
    receipt_path = tmp_path / "receipt.json"

    def ambiguous(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous", request=request)

    settings = _settings(tmp_path)
    with pytest.raises(ProviderFailure):
        SeedreamClient(settings, httpx.MockTransport(ambiguous)).generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    receipt_path.unlink()

    with pytest.raises(ProviderFailure) as repeated:
        SeedreamClient(
            settings,
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(repeated).code == "submission_unknown"
    # The first generate used its bounded retries; the durable claim alone
    # still forbids any further POST.
    assert calls == 3


def test_missing_key_and_invalid_input_do_not_create_paid_claim(tmp_path: Path) -> None:
    settings = _settings(tmp_path, seedream_api_key="")
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(settings).generate(
            prompt="frozen prompt",
            images=[],
            destination=tmp_path / "sheet.png",
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "seedream_not_configured"
    assert not receipt_path.exists()
    assert not (tmp_path / "receipt.json.post-claim").exists()


def test_replay_received_uses_only_frozen_receipts_across_configuration_drift(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    original = _settings(tmp_path)
    SeedreamClient(
        original, httpx.MockTransport(lambda _request: _success_response())
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    destination.unlink()
    network_calls = 0

    def must_not_call_network(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        pytest.fail(f"read-only recovery called the network: {request}")

    drifted = _settings(
        tmp_path,
        seedream_api_key="",
        seedream_api_base="https://different-ark.test/other-version",
        seedream_model="different-model",
    )
    replayed = SeedreamClient(
        drifted, httpx.MockTransport(must_not_call_network)
    ).replay_received(
        destination=destination,
        receipt_path=receipt_path,
    )

    assert network_calls == 0
    assert destination.read_bytes() == _PNG
    assert replayed.record["replayed"] is True
    assert replayed.record["request"]["url"] == (
        "https://ark.test/api/v3/images/generations"
    )
    assert replayed.record["request"]["body"]["model"] == "seedream-test-model"
    assert "ark-secret" not in json.dumps(replayed.record)


def test_result_fsync_before_marker_failure_is_recovered_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    marker_path = tmp_path / "receipt.json.response.json"
    result_path = tmp_path / "receipt.json.result"
    real_atomic_json = seedream._atomic_json
    provider_calls = 0

    def provider(_request: httpx.Request) -> httpx.Response:
        nonlocal provider_calls
        provider_calls += 1
        return _success_response()

    def fail_first_marker(path: Path, payload: dict[str, object]) -> None:
        if path == marker_path:
            raise OSError("injected crash before response marker fsync")
        real_atomic_json(path, payload)

    monkeypatch.setattr(seedream, "_atomic_json", fail_first_marker)
    with pytest.raises(ProviderFailure) as interrupted:
        SeedreamClient(
            _settings(tmp_path), httpx.MockTransport(provider)
        ).generate(
            prompt="frozen prompt",
            images=[],
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(interrupted).code == "local_persistence_error"
    assert provider_calls == 1
    assert result_path.read_bytes() == _PNG
    assert not marker_path.exists()
    assert not destination.exists()
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "submitting"

    monkeypatch.setattr(seedream, "_atomic_json", real_atomic_json)
    replayed = SeedreamClient(
        _settings(
            tmp_path,
            seedream_api_key="",
            seedream_api_base="https://drifted.test/api/v9",
            seedream_model="drifted-model",
        ),
        httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
    ).replay_received(
        destination=destination,
        receipt_path=receipt_path,
    )

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert replayed.value == destination
    assert destination.read_bytes() == _PNG
    assert marker["recovered_from"] == "durable_raw_result"
    assert marker["result_sha256"] == seedream._sha256(result_path.read_bytes())
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "succeeded"


@pytest.mark.parametrize(
    "damage", ["missing", "invalid", "truncated", "wrong_dimensions"]
)
def test_submitting_receipt_without_valid_result_stays_unknown_and_never_posts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    marker_path = tmp_path / "receipt.json.response.json"
    result_path = tmp_path / "receipt.json.result"
    real_atomic_json = seedream._atomic_json

    def fail_marker(path: Path, payload: dict[str, object]) -> None:
        if path == marker_path:
            raise OSError("injected marker failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(seedream, "_atomic_json", fail_marker)
    with pytest.raises(ProviderFailure):
        SeedreamClient(
            _settings(tmp_path),
            httpx.MockTransport(lambda _request: _success_response()),
        ).generate(
            prompt="frozen prompt",
            images=[],
            destination=destination,
            receipt_path=receipt_path,
        )
    monkeypatch.setattr(seedream, "_atomic_json", real_atomic_json)
    if damage == "missing":
        result_path.unlink()
    elif damage == "truncated":
        result_path.write_bytes(_PNG[:-8])
    elif damage == "wrong_dimensions":
        result_path.write_bytes(_png(1, 1))
    else:
        result_path.write_bytes(b"not-an-image")

    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(
            _settings(tmp_path, seedream_api_key=""),
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).replay_received(
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "submission_unknown"
    assert caught.value.record["outcome"] == "submission_unknown"
    assert not marker_path.exists()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        ("submission_unknown", "submission_unknown"),
        ("rejected", "provider_rejected"),
        ("protocol_error", "provider_protocol_error"),
    ],
)
def test_replay_received_rejects_nonrecoverable_status_without_network(
    tmp_path: Path,
    status: str,
    expected_code: str,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    SeedreamClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda _request: _success_response()),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = status
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    destination.unlink()

    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(
            _settings(tmp_path, seedream_api_key=""),
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).replay_received(
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == expected_code
    assert not destination.exists()


def test_replay_received_rejects_mismatched_claim_without_network(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    SeedreamClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda _request: _success_response()),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    claim_path = tmp_path / "receipt.json.post-claim"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["request_sha256"] = "0" * 64
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    destination.unlink()

    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(
            _settings(tmp_path, seedream_api_key=""),
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).replay_received(
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "attempt_receipt_invalid"
    assert not destination.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "tampered-model"),
        ("prompt_sha256", "0" * 64),
        ("input_sha256", ["0" * 64]),
        ("size", "1x1"),
        ("request_sha256", "0" * 64),
    ],
)
def test_replay_received_rejects_tampered_frozen_request_binding(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    destination = tmp_path / "sheet.png"
    SeedreamClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda _request: _success_response()),
    ).generate(
        prompt="frozen prompt",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    destination.unlink()

    with pytest.raises(ProviderFailure) as caught:
        SeedreamClient(
            _settings(
                tmp_path,
                seedream_api_key="",
                seedream_api_base="https://drifted.test/api/v9",
                seedream_model="drifted-model",
            ),
            httpx.MockTransport(lambda request: pytest.fail(f"must not POST: {request}")),
        ).replay_received(
            destination=destination,
            receipt_path=receipt_path,
        )
    assert _cause(caught).code == "attempt_receipt_invalid"
    assert not destination.exists()


def test_generate_retries_connect_error_then_succeeds(tmp_path: Path) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "seedream-receipt.json"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("ssl eof", request=request)
        return _success_response()

    result = SeedreamClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).generate(
        prompt="one reference sheet",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )

    assert result.value == destination
    assert destination.read_bytes() == _PNG
    assert calls == 2
    attempts = result.record["attempts"]
    assert [entry["attempt"] for entry in attempts] == [1, 2]
    assert attempts[0]["error"]["type"] == "ConnectError"
    assert attempts[1]["status_code"] == 200
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"


def test_generate_retries_503_then_succeeds_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "sheet.png"
    receipt_path = tmp_path / "seedream-receipt.json"
    calls = 0
    sleeps: list[float] = []
    monkeypatch.setattr(seedream, "retry_sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"code": "Unavailable"}})
        return _success_response()

    result = SeedreamClient(
        _settings(tmp_path), httpx.MockTransport(handler)
    ).generate(
        prompt="one reference sheet",
        images=[],
        destination=destination,
        receipt_path=receipt_path,
    )

    assert result.value == destination
    assert calls == 2
    assert sleeps == [2.0]
    assert [
        entry.get("status_code") for entry in result.record["attempts"]
    ] == [503, 200]
