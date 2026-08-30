"""Provider failure traces stay useful without persisting or logging credentials."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest

from app import context_ir_bridge, h3, seedream
from app.config import Settings


def _png() -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((3, 5, 3), 127, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _read_trace(path: Path, *, call_path: list[str], secret: str) -> dict:
    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    trace = json.loads(raw)
    assert trace["schema"] == "duet.error-call-tree"
    assert trace["version"] == 1
    assert trace["call_path"] == call_path
    return trace


def test_seedream_rejection_writes_sanitized_sidecar_and_log(
    tmp_path, monkeypatch, caplog
):
    secret = "seedream-private-token"
    monkeypatch.setenv("ARK_API_KEY", secret)
    settings = Settings(access_token="test", data_dir=tmp_path, retry_interval_s=0)
    receipt = tmp_path / "attempt.json"

    async def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": f"provider echoed {secret}",
                }
            },
        )

    caplog.set_level(logging.ERROR, logger="app.seedream")
    with pytest.raises(seedream.SeedreamError, match="Seedream image edit failed"):
        asyncio.run(
            seedream.edit(
                settings,
                [_png()],
                "prompt",
                tmp_path / "output.png",
                receipt_path=receipt,
                transport=httpx.MockTransport(reject),
            )
        )

    trace = _read_trace(
        receipt.with_suffix(".error.json"),
        call_path=["postprocess", "seedream", "attempt", "POST"],
        secret=secret,
    )
    assert trace["error"]["code"] == "provider_rejected"
    assert trace["error"]["provider"]["http_status"] == 400
    assert "InvalidParameter" in json.dumps(trace, ensure_ascii=False)
    assert secret not in caplog.text


def test_context_ir_rejection_writes_sanitized_sidecar_and_log(
    tmp_path, monkeypatch, caplog
):
    secret = "context-ir-private-token"
    attempt = tmp_path / ".context-ir" / "attempts" / "000001" / "attempt.json"
    state = {
        "status": "ready_to_submit",
        "error": None,
        "context_ir_request": {"prompt": "safe"},
    }
    request = SimpleNamespace(
        minimax_api_key=secret,
        timeouts=SimpleNamespace(request_s=0.1),
    )

    def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "InvalidParameter",
                "message": f"provider echoed {secret}",
            },
        )

    # This unit targets the provider boundary; receipt schema persistence is covered
    # separately and would obscure the sidecar/log contract under test here.
    monkeypatch.setattr(context_ir_bridge, "_persist", lambda _path, _state: None)
    caplog.set_level(logging.ERROR, logger="app.context_ir_bridge")
    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        context_ir_bridge._submit(request, state, attempt, client)

    trace = _read_trace(
        attempt.with_name("error.json"),
        call_path=["generation", "context_ir", "submit"],
        secret=secret,
    )
    assert state["status"] == "failed"
    assert state["provider_error_code"] == "InvalidParameter"
    assert trace["error"]["provider"]["http_status"] == 422
    assert "InvalidParameter" in json.dumps(trace, ensure_ascii=False)
    assert secret not in caplog.text


def _h3_request(tmp_path: Path, secret: str) -> h3.H3Request:
    first = tmp_path / "01.png"
    second = tmp_path / "02.png"
    first.write_bytes(b"first-frame")
    second.write_bytes(b"second-frame")
    voice_texts = ("第一句台词", "第二句台词")
    return h3.H3Request(
        cid="cid-provider-trace",
        workdir=tmp_path / "session",
        client_request_id="request-provider-trace",
        prompt="原始 prompt，包含第一句台词和第二句台词。",
        keyframes=h3.freeze_keyframes((first, second)),
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
        duration=10,
        autodl_token=secret,
        timeouts=h3.Timeouts(
            request_s=0.1,
            h3_poll_s=0.03,
            download_s=0.1,
            poll_interval_s=0,
            retry_interval_s=0,
        ),
    )


def test_h3_rejection_writes_sanitized_sidecar_and_log(
    tmp_path, monkeypatch, caplog
):
    secret = "h3-private-token"
    monkeypatch.setenv("AUTODL_ART_TOKEN", secret)
    request = _h3_request(tmp_path, secret)

    def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "InvalidParameter",
                "msg": f"provider echoed {secret}",
            },
        )

    caplog.set_level(logging.WARNING, logger="app.h3")
    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        with pytest.raises(h3.H3Error, match="h3_submit_rejected"):
            h3.start(request, client=client)

    trace = _read_trace(
        request.workdir / "errors" / "h3-submit.json",
        call_path=["generation", "h3", "submit"],
        secret=secret,
    )
    assert trace["error"]["code"] == "h3_submit_rejected"
    assert trace["error"]["provider"]["http_status"] == 422
    assert "InvalidParameter" in json.dumps(trace, ensure_ascii=False)
    assert secret not in caplog.text
