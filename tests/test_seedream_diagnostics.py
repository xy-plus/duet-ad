import asyncio
import base64
import json

import httpx
import cv2
import numpy as np
import pytest

from app import seedream
from app.config import Settings


def _png():
    ok, encoded = cv2.imencode(".png", np.zeros((1, 1, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_seedream_payload_uses_each_input_magic_for_data_url_mime(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    inputs = []
    for extension in (".png", ".jpg", ".webp"):
        ok, encoded = cv2.imencode(
            extension, np.full((2, 3, 3), 127, dtype=np.uint8),
        )
        assert ok
        inputs.append(encoded.tobytes())
    payloads = []

    async def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "data": [{"b64_json": base64.b64encode(_png()).decode("ascii")}],
        })

    asyncio.run(seedream.edit(
        settings,
        inputs,
        "prompt",
        tmp_path / "output.png",
        receipt_path=tmp_path / "attempt.json",
        transport=httpx.MockTransport(handler),
    ))

    assert len(payloads) == 1
    urls = payloads[0]["image"]
    assert [url.split(";", 1)[0] for url in urls] == [
        "data:image/png", "data:image/jpeg", "data:image/webp",
    ]
    assert [base64.b64decode(url.split(",", 1)[1]) for url in urls] == inputs


def test_seedream_rejects_unknown_input_magic_before_paid_claim(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    receipt_path = tmp_path / "attempt.json"

    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(seedream.edit(
            settings,
            [_png(), b"GIF89a"],
            "prompt",
            tmp_path / "output.png",
            receipt_path=receipt_path,
            transport=httpx.MockTransport(lambda _request: pytest.fail("POST called")),
        ))

    assert caught.value.code == "invalid_input"
    assert not receipt_path.exists()
    assert not seedream._claim_path(receipt_path).exists()


@pytest.mark.parametrize(
    ("response", "expected_code", "expected_stage"),
    [
        (
            httpx.Response(
                200,
                content=b"not-json Bearer sk-live-secret",
                headers={"content-type": "text/plain"},
            ),
            "provider_protocol_error",
            "response_json",
        ),
        (
            httpx.Response(200, json={"data": [{}]}),
            "provider_protocol_error",
            "response_decode",
        ),
        (
            httpx.Response(200, json={"data": [{"b64_json": "not-base64"}]}),
            "provider_protocol_error",
            "response_decode",
        ),
        (
            httpx.Response(
                200,
                json={"data": [{"b64_json": base64.b64encode(b"not-image").decode()}]},
            ),
            "provider_output_invalid",
            "output_validation",
        ),
    ],
)
def test_seedream_2xx_protocol_failures_persist_redacted_diagnostics(
    tmp_path, monkeypatch, response, expected_code, expected_stage,
):
    monkeypatch.setenv("ARK_API_KEY", "sk-live-secret")
    settings = Settings(access_token="x", data_dir=tmp_path, retry_interval_s=0)
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return response

    receipt_path = tmp_path / "attempt.json"
    output_path = tmp_path / "output.png"
    with pytest.raises(seedream.SeedreamError) as caught:
        asyncio.run(
            seedream.edit(
                settings,
                [_png()],
                "prompt",
                output_path,
                receipt_path=receipt_path,
                transport=httpx.MockTransport(handler),
            )
        )

    assert caught.value.code == expected_code
    assert calls == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["http_status"] == 200
    assert receipt["attempts"] == [{"number": 1, "status": "failed"}]
    error_trace = json.loads(receipt_path.with_suffix(".error.json").read_text())
    assert error_trace["error"]["provider"]["http_status"] == 200
    assert error_trace["error"]["provider"]["body"] is not None
    assert error_trace["error"]["code"] == expected_code
    assert error_trace["error"]["cause"]["type"] in {
        "JSONDecodeError",
        "SeedreamError",
    }
    assert expected_stage in error_trace["call_path"]
    assert "sk-live-secret" not in json.dumps(receipt)
    assert not output_path.exists()
