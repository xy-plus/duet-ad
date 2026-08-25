import base64
import hashlib
import json
import logging
import socket
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import h3


VOICE_TEXTS = ("第一句台词", "第二句台词")
NO_SUBTITLES_PROMPT_PREFIX = (
    "不要产出任何字幕和水印，如果参考图里有，请帮忙先消除掉字幕和水印后再根据提示词让图片动起来"
)


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution", "provider_value"),
    [
        ("16:9", "480p", "480p横"),
        ("9:16", "480p", "480p竖"),
        ("16:9", "768p", "768p横"),
        ("9:16", "768p", "768p竖"),
    ],
)
def test_provider_resolution_is_the_only_semantic_projection(
    aspect_ratio, resolution, provider_value,
):
    assert h3.provider_resolution(aspect_ratio, resolution) == provider_value


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution"),
    [("1:1", "768p"), ("9:16", "1080p"), (None, "480p")],
)
def test_h3_request_rejects_generation_values_before_provider_post(
    tmp_path, aspect_ratio, resolution,
):
    with pytest.raises(h3.H3Error, match="invalid_(aspect_ratio|resolution)"):
        replace(
            _request(tmp_path),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )


@pytest.mark.parametrize(
    "stream,expected",
    [
        ({"codec_type": "video", "duration": "1.25"}, True),
        ({"codec_type": "video", "duration_ts": "30", "time_base": "1/24"}, True),
        ({"codec_type": "video", "duration": "0"}, False),
        ({"codec_type": "video", "duration": True}, False),
        ({"codec_type": "video", "duration": "N/A", "duration_ts": True,
          "time_base": "1/24"}, False),
        ({"codec_type": "video"}, False),
        ({"codec_type": "audio", "duration": "2"}, False),
    ],
)
def test_download_probe_requires_positive_visual_stream_duration(
    tmp_path, monkeypatch, stream, expected,
):
    completed = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({
            "format": {"duration": "99"}, "streams": [stream],
        }), stderr="",
    )
    monkeypatch.setattr(h3.subprocess, "run", lambda *_a, **_kw: completed)
    assert h3._probe_video(tmp_path / "generated.mp4", 1) is expected


def _request(tmp_path: Path, *, request_id: str = "request-1") -> h3.H3Request:
    first = tmp_path / "01.png"
    second = tmp_path / "02.png"
    first.write_bytes(b"first-frame")
    second.write_bytes(b"second-frame")
    frozen = h3.freeze_keyframes((first, second))
    return h3.H3Request(
        cid="cid-1",
        workdir=tmp_path / "session",
        client_request_id=request_id,
        prompt='原始 prompt，包含"第一句台词"和"第二句台词"。',
        keyframes=frozen,
        voice_texts=VOICE_TEXTS,
        voice_receipt=h3.voice_texts_receipt(VOICE_TEXTS),
        duration=10,
        autodl_token="art-secret",
        timeouts=h3.Timeouts(
            request_s=0.1,
            h3_poll_s=0.03,
            download_s=0.1,
            poll_interval_s=0,
            retry_interval_s=0,
        ),
    )


def _boundary_request(tmp_path: Path, *, request_id: str = "boundary-1") -> h3.H3Request:
    first = (Path("first.png"), b"first-boundary")
    last = (Path("last.png"), b"last-boundary")
    return replace(
        _request(tmp_path, request_id=request_id),
        mode="boundary",
        keyframes=(),
        first_frame=first,
        last_frame=last,
        duration=15,
    )


@pytest.mark.parametrize("mode", ["reference", "boundary"])
def test_request_injects_no_subtitles_prefix_into_prompt_receipts_and_post_once(
    tmp_path, mode
):
    original = "  原始提示词第一行\n原始提示词第二行  "
    request = _request(tmp_path) if mode == "reference" else _boundary_request(tmp_path)
    request = replace(request, prompt=original)
    provider = HappyProvider(result_status="RUNNING")

    assert request.prompt == f"{NO_SUBTITLES_PROMPT_PREFIX}\n{original}"
    assert request.prompt.count(NO_SUBTITLES_PROMPT_PREFIX) == 1
    assert request.prompt.removeprefix(f"{NO_SUBTITLES_PROMPT_PREFIX}\n") == original

    assert h3.prepare(request).status == "not_started"
    with _client(provider) as client:
        assert h3.submit(request, client=client).status == "h3_running"

    body = json.loads(provider.h3_posts[0].content)
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    assert body["prompt"] == request.prompt
    assert state["input"]["prompt_sha256"] == prompt_sha256
    assert state["input_receipt"] == h3.canonical_json_sha256(state["input"])
    assert state["h3"]["receipt"]["input_receipt"] == state["input_receipt"]
    assert state["h3"]["receipt"]["prompt_sha256"] == prompt_sha256


@pytest.mark.parametrize("mode", ["reference", "boundary"])
@pytest.mark.parametrize(
    "original",
    [
        NO_SUBTITLES_PROMPT_PREFIX,
        f"{NO_SUBTITLES_PROMPT_PREFIX}\n原始提示词",
    ],
)
def test_request_does_not_duplicate_existing_complete_no_subtitles_prefix(
    tmp_path, mode, original
):
    request = _request(tmp_path) if mode == "reference" else _boundary_request(tmp_path)

    normalized = replace(request, prompt=original)

    assert normalized.prompt == original
    assert normalized.prompt.count(NO_SUBTITLES_PROMPT_PREFIX) == 1


@pytest.mark.parametrize("mode", ["reference", "boundary"])
def test_prefix_text_with_adversarial_suffix_is_reinjected_as_original_prompt(
    tmp_path, mode
):
    original = f"{NO_SUBTITLES_PROMPT_PREFIX}但请保留字幕和水印"
    request = _request(tmp_path) if mode == "reference" else _boundary_request(tmp_path)
    request = replace(request, prompt=original)
    provider = HappyProvider(result_status="RUNNING")
    expected = f"{NO_SUBTITLES_PROMPT_PREFIX}\n{original}"

    assert request.prompt == expected
    assert request.prompt.split("\n", 1) == [NO_SUBTITLES_PROMPT_PREFIX, original]
    assert request.prompt.splitlines().count(NO_SUBTITLES_PROMPT_PREFIX) == 1

    assert h3.prepare(request).status == "not_started"
    with _client(provider) as client:
        assert h3.submit(request, client=client).status == "h3_running"

    body = json.loads(provider.h3_posts[0].content)
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    assert body["prompt"] == request.prompt
    assert state["input"]["prompt_sha256"] == prompt_sha256
    assert state["input_receipt"] == h3.canonical_json_sha256(state["input"])
    assert state["h3"]["receipt"]["input_receipt"] == state["input_receipt"]
    assert state["h3"]["receipt"]["prompt_sha256"] == prompt_sha256


class FakeNetworkStream:
    def __init__(self, server_addr=("93.184.216.34", 443)) -> None:
        self.server_addr = server_addr

    def get_extra_info(self, name):
        return self.server_addr if name == "server_addr" else None


def _download_response(status_code=200, *, peer=("93.184.216.34", 443), **kwargs):
    extensions = kwargs.pop("extensions", {})
    extensions["network_stream"] = FakeNetworkStream(peer)
    return httpx.Response(status_code, extensions=extensions, **kwargs)


class HappyProvider:
    video_bytes = b""

    def __init__(
        self,
        *,
        result_status: str = "SUCCESS",
        result_url: str = "https://download.invalid/video.mp4",
    ) -> None:
        self.result_status = result_status
        self.result_url = result_url
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith(("/minimax_h3_lightx2v_v5", "/minimax_h3_lightx2v")):
            return httpx.Response(200, json={"data": {"task_id": "h3-task-local"}})
        if path.endswith("/result/h3-task-local"):
            payload = {"status": self.result_status}
            if self.result_status in {"SUCCESS", "COMPLETED"}:
                payload["results"] = [{"url": self.result_url}]
            return httpx.Response(200, json={"data": payload})
        if request.url.host == "download.invalid":
            return _download_response(200, content=self.video_bytes)
        raise AssertionError(f"unexpected request path: {path}")

    @property
    def h3_posts(self) -> list[httpx.Request]:
        return [
            request
            for request in self.requests
            if request.url.path.endswith(("/minimax_h3_lightx2v_v5", "/minimax_h3_lightx2v"))
        ]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(scope="session")
def valid_video_bytes(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("h3-video") / "valid.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=5:d=10",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path.read_bytes()


@pytest.fixture(autouse=True)
def public_download_host(monkeypatch, valid_video_bytes):
    HappyProvider.video_bytes = valid_video_bytes
    monkeypatch.setattr(
        h3.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


def _attempt_file(request: h3.H3Request, number: int = 1) -> Path:
    return request.workdir / ".h3" / "attempts" / f"{number:06d}" / "attempt.json"


def test_start_submits_source_prompt_directly_and_writes_state(tmp_path):
    request = _request(tmp_path)
    (tmp_path / "01.png").write_bytes(b"mutated-on-disk")
    provider = HappyProvider()

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert result.output.read_bytes() == provider.video_bytes
    assert len(provider.h3_posts) == 1
    body = json.loads(provider.h3_posts[0].content)
    assert body == {
        "prompt": request.prompt,
        "duration": 10,
        "resolution": "768p竖",
        "ref_image_0": "data:image/png;base64,"
        + base64.b64encode(b"first-frame").decode("ascii"),
        "ref_image_1": "data:image/png;base64,"
        + base64.b64encode(b"second-frame").decode("ascii"),
    }
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "succeeded"
    assert "ir" not in state
    assert state["h3"]["task_id"] == "h3-task-local"
    assert state["h3"]["output"] == {
        "name": "generated.mp4",
        "sha256": hashlib.sha256(provider.video_bytes).hexdigest(),
        "size": len(provider.video_bytes),
    }
    assert "download.invalid" not in json.dumps(state)
    assert "art-secret" not in json.dumps(state)


def test_prepare_then_submit_persists_receipt_before_post_and_does_not_poll(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider(result_status="RUNNING")

    prepared = h3.prepare(request)

    assert prepared.status == "not_started"
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "ready_to_submit"
    assert state["h3"] == {"status": "ready"}
    assert isinstance(state["input_receipt"], str)
    with _client(provider) as client:
        submitted = h3.submit(request, client=client)

    assert submitted.status == "h3_running"
    assert len(provider.h3_posts) == 1
    assert not any("/result/" in item.url.path for item in provider.requests)
    persisted = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert persisted["h3"]["task_id"] == "h3-task-local"


def test_prepared_attempt_is_get_only_on_resume_and_submit_is_idempotent(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider(result_status="RUNNING")
    h3.prepare(request)

    with _client(provider) as client:
        assert h3.resume(request, client=client).status == "not_started"
        assert h3.submit(request, client=client).status == "h3_running"
        assert h3.submit(request, client=client).status == "h3_running"

    assert len(provider.h3_posts) == 1


def test_boundary_mode_selects_fl2va_workflow_and_only_boundary_fields(tmp_path):
    request = _boundary_request(tmp_path)
    provider = HappyProvider()

    with _client(provider) as client:
        assert h3.start(request, client=client).status == "succeeded"

    post = provider.h3_posts[0]
    assert post.url.path.endswith("/minimax_h3_lightx2v")
    assert json.loads(post.content) == {
        "prompt": request.prompt,
        "duration": 15,
        "resolution": "768p竖",
        "first_frame": "data:image/png;base64,"
        + base64.b64encode(b"first-boundary").decode("ascii"),
        "last_frame": "data:image/png;base64,"
        + base64.b64encode(b"last-boundary").decode("ascii"),
    }
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["input"]["request"] == {
        "mode": "boundary",
        "h3_workflow": "minimax_h3_lightx2v",
        "duration": 15,
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "provider_resolution": "768p竖",
    }
    assert state["input"]["images"] == [
        {
            "role": "first_frame",
            "name": "first.png",
            "sha256": hashlib.sha256(b"first-boundary").hexdigest(),
        },
        {
            "role": "last_frame",
            "name": "last.png",
            "sha256": hashlib.sha256(b"last-boundary").hexdigest(),
        },
    ]


def test_explicit_reference_seed_is_posted_and_frozen_in_receipts(tmp_path):
    request = replace(_request(tmp_path), seed=123456)
    provider = HappyProvider(result_status="RUNNING")
    with _client(provider) as client:
        assert h3.start(request, client=client).status == "retryable_failure"

    assert json.loads(provider.h3_posts[0].content)["seed"] == 123456
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["input"]["request"]["seed"] == 123456
    assert state["h3"]["receipt"]["request"]["seed"] == 123456

    drifted = replace(request, seed=654321)
    with _client(lambda _request: pytest.fail("drift must fail before network")) as client:
        with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
            h3.resume(drifted, client=client)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"mode": "unknown"}, "invalid_mode"),
        ({"mode": "boundary", "keyframes": (), "first_frame": (Path("first.png"), b"a"), "last_frame": None}, "invalid_boundary_frames"),
        ({"mode": "boundary", "first_frame": (Path("first.png"), b"a"), "last_frame": (Path("last.png"), b"b")}, "mixed_h3_inputs"),
        ({"mode": "reference", "first_frame": (Path("first.png"), b"a")}, "mixed_h3_inputs"),
        ({"mode": "reference", "seed": 0}, "invalid_seed"),
        ({"mode": "reference", "seed": 1_000_000_000_000_000}, "invalid_seed"),
        ({"mode": "boundary", "keyframes": (), "first_frame": (Path("first.png"), b"a"), "last_frame": (Path("last.png"), b"b"), "seed": 1}, "seed_not_supported"),
    ],
)
def test_request_modes_reject_ambiguous_or_invalid_inputs(tmp_path, changes, error):
    with pytest.raises(h3.H3Error, match=error):
        replace(_request(tmp_path), **changes)


def test_mode_specific_duration_limits(tmp_path):
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(_request(tmp_path), duration=11)
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(_boundary_request(tmp_path), duration=16)


def test_boundary_receipt_drift_is_rejected_before_network(tmp_path):
    request = _boundary_request(tmp_path)
    with _client(HappyProvider(result_status="RUNNING")) as client:
        assert h3.start(request, client=client).status == "retryable_failure"

    drifted = replace(request, last_frame=(Path("last.png"), b"changed"))
    with _client(lambda _request: pytest.fail("drift must fail before network")) as client:
        with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
            h3.resume(drifted, client=client)


def test_h3_rejection_logs_sanitized_provider_reason_without_retry(tmp_path, caplog):
    request = _request(tmp_path)
    calls = []

    def rejected(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(
            422,
            json={"code": "InvalidParameter", "msg": "duration rejected art-secret"},
        )

    caplog.set_level(logging.WARNING, logger="app.h3")
    with _client(rejected) as client:
        with pytest.raises(h3.H3Error, match="h3_submit_rejected"):
            h3.start(request, client=client)

    assert len(calls) == 1
    assert "InvalidParameter" in caplog.text
    assert "art-secret" not in caplog.text
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["error"] == {"code": "h3_submit_rejected"}


def test_provider_failure_diagnostic_survives_successful_paid_retry(
    tmp_path, caplog
):
    request = _boundary_request(tmp_path)

    def failed(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/minimax_h3_lightx2v"):
            return httpx.Response(200, json={"data": {"task_id": "art-secret"}})
        if req.url.path.endswith("/result/art-secret"):
            return httpx.Response(
                200,
                json={
                    "code": "Success",
                    "msg": "GPU OOM art-secret",
                    "request_id": "provider-request-1",
                    "data": {"status": "FAILED", "results": []},
                },
            )
        raise AssertionError(f"unexpected request path: {req.url.path}")

    caplog.set_level(logging.WARNING, logger="app.h3")
    with _client(failed) as client:
        result = h3.start(request, client=client)

    assert result.status == "failed"
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["h3"]["status"] == "failed"
    assert state["error"] == {
        "code": "h3_provider_failed",
        "provider": {
            "status": "FAILED",
            "request_id": "provider-request-1",
            "detail": "Success | GPU OOM ***",
        },
    }
    assert "provider-request-1" in caplog.text
    assert "art-secret" not in caplog.text

    failed_attempt = _attempt_file(request).read_bytes()

    def recovered(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/minimax_h3_lightx2v"):
            return httpx.Response(200, json={"data": {"task_id": "recovered-task"}})
        if req.url.path.endswith("/result/recovered-task"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "status": "SUCCESS",
                        "results": [{"url": "https://download.invalid/video.mp4"}],
                    }
                },
            )
        if req.url.host == "download.invalid":
            return _download_response(200, content=HappyProvider.video_bytes)
        raise AssertionError(f"unexpected request path: {req.url.path}")

    with _client(recovered) as client:
        retried = h3.retry(request, "boundary-2", client=client)

    assert retried.status == "succeeded"
    assert _attempt_file(request).read_bytes() == failed_attempt
    recovered_state = json.loads(
        _attempt_file(request, 2).read_text(encoding="utf-8")
    )
    assert recovered_state["status"] == "succeeded"
    assert recovered_state["input_receipt"] == state["input_receipt"]


@pytest.mark.parametrize(
    "case", ("secret_detail", "multiline_detail", "multiline_request_id", "extra_key")
)
def test_tampered_provider_failure_diagnostic_is_rejected(tmp_path, case):
    request = _boundary_request(tmp_path)

    def failed(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/minimax_h3_lightx2v"):
            return httpx.Response(200, json={"data": {"task_id": "failed-task"}})
        return httpx.Response(
            200,
            json={
                "msg": "GPU OOM",
                "request_id": "provider-request-1",
                "data": {"status": "FAILED"},
            },
        )

    with _client(failed) as client:
        assert h3.start(request, client=client).status == "failed"

    path = _attempt_file(request)
    state = json.loads(path.read_text(encoding="utf-8"))
    if case == "secret_detail":
        state["error"]["provider"]["detail"] = "art-secret"
    elif case == "multiline_detail":
        state["error"]["provider"]["detail"] = "safe\nforged"
    elif case == "multiline_request_id":
        state["error"]["provider"]["request_id"] = "safe\nforged"
    else:
        state["error"]["unexpected"] = "value"
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(h3.ReceiptError, match="state_invalid"):
        h3.inspect(request)


def test_legacy_provider_failure_without_diagnostic_remains_idempotent(tmp_path):
    request = _boundary_request(tmp_path)

    def failed(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/minimax_h3_lightx2v"):
            return httpx.Response(200, json={"data": {"task_id": "legacy-task"}})
        return httpx.Response(200, json={"data": {"status": "FAILED"}})

    with _client(failed) as client:
        assert h3.start(request, client=client).status == "failed"

    path = _attempt_file(request)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["error"] = {"code": "h3_provider_failed"}
    state["h3"]["status"] = "running"
    path.write_text(json.dumps(state), encoding="utf-8")

    assert h3.inspect(request).status == "failed"
    with _client(lambda _req: pytest.fail("legacy terminal state must stay offline")) as client:
        assert h3.start(request, client=client).status == "failed"


def test_unbound_generated_video_is_not_treated_as_resumable_success(tmp_path):
    request = _request(tmp_path)
    request.workdir.mkdir(parents=True)
    output = request.workdir / "generated.mp4"
    output.write_bytes(b"already-done")

    with _client(lambda _request: pytest.fail("resume must not use network without task")) as client:
        result = h3.resume(request, client=client)

    assert result.status == "not_started"


def test_reusable_output_requires_bound_receipt_valid_video_duration_and_frozen_bytes(
    tmp_path
):
    request = _request(tmp_path)
    provider = HappyProvider()
    with _client(provider) as client:
        assert h3.start(request, client=client).status == "succeeded"
    output = request.workdir / "generated.mp4"
    actual_duration = h3._probe_video_duration(output, 1)
    assert actual_duration is not None
    assert h3.output_is_reusable(
        request, expected_duration_s=actual_duration
    ) is True
    assert h3.output_is_reusable(
        request, expected_duration_s=actual_duration + 2
    ) is False

    original = output.read_bytes()
    output.write_bytes(b"")
    assert h3.output_is_reusable(request) is False
    output.write_bytes(b"not-a-video")
    assert h3.output_is_reusable(request) is False
    output.write_bytes(original)
    drifted = replace(request, prompt=request.prompt + " changed")
    with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
        h3.output_is_reusable(drifted)


@pytest.mark.parametrize(
    ("request_duration", "source_target", "output_duration", "expected"),
    [
        (11, 10.84, 11.541667, True),
        (13, 12.0, 13.666667, True),
        (11, 10.84, 10.79, False),
        (11, 10.84, 12.000001, False),
    ],
)
def test_boundary_output_reuse_uses_source_floor_and_provider_request_ceiling(
    tmp_path, monkeypatch, request_duration, source_target, output_duration, expected,
):
    request = replace(_boundary_request(tmp_path), duration=request_duration)
    with _client(HappyProvider()) as client:
        assert h3.start(request, client=client).status == "succeeded"
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: output_duration)

    assert h3.output_is_reusable(
        request, expected_duration_s=source_target
    ) is expected


def test_reference_output_reuse_keeps_half_second_tolerance(tmp_path, monkeypatch):
    request = _request(tmp_path)
    with _client(HappyProvider()) as client:
        assert h3.start(request, client=client).status == "succeeded"

    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 9.5)
    assert h3.output_is_reusable(request, expected_duration_s=10.0) is True
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 9.499999)
    assert h3.output_is_reusable(request, expected_duration_s=10.0) is False


def test_succeeded_attempt_with_missing_output_redownloads_by_get_only(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()
    with _client(provider) as client:
        assert h3.start(request, client=client).status == "succeeded"
    (request.workdir / "generated.mp4").unlink()
    calls = []

    def recover(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.method == "GET"
        if req.url.path.endswith("/result/h3-task-local"):
            return httpx.Response(
                200,
                json={"data": {"status": "SUCCESS", "results": [
                    {"url": "https://download.invalid/video.mp4"}
                ]}},
            )
        return _download_response(200, content=HappyProvider.video_bytes)

    with _client(recover) as client:
        result = h3.resume(request, client=client)
    assert result.status == "succeeded"
    assert calls and all(call.method == "GET" for call in calls)
    assert h3.output_is_reusable(request) is True


def test_empty_voice_texts_are_valid_and_bound(tmp_path):
    request = replace(
        _request(tmp_path),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
    )
    with _client(HappyProvider()) as client:
        assert h3.start(request, client=client).status == "succeeded"
    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["input"]["voice_texts_sha256"] == h3.canonical_json_sha256([])


def test_inspect_is_read_only_before_start(tmp_path):
    request = _request(tmp_path)
    result = h3.inspect(request)
    assert result.status == "not_started"
    assert not (request.workdir / ".h3").exists()


def test_start_is_idempotent_for_same_request_id(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()
    with _client(provider) as client:
        first = h3.start(request, client=client)
        second = h3.start(request, client=client)
    assert first.status == second.status == "succeeded"
    assert len(provider.h3_posts) == 1


def test_resume_only_queries_existing_h3_task(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider(result_status="RUNNING")
    with _client(provider) as client:
        first = h3.start(request, client=client)
    assert first.status == "retryable_failure"
    assert len(provider.h3_posts) == 1

    calls = []

    def recovery(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.method == "GET"
        if req.url.path.endswith("/result/h3-task-local"):
            return httpx.Response(
                200,
                json={"data": {"status": "SUCCESS", "results": [{"url": "https://download.invalid/video.mp4"}]}},
            )
        return _download_response(200, content=HappyProvider.video_bytes)

    with _client(recovery) as client:
        recovered = h3.resume(request, client=client)
    assert recovered.status == "succeeded"
    assert calls and all(call.method == "GET" for call in calls)


def test_existing_v1_reference_attempt_remains_inspectable_and_get_only_resumable(tmp_path):
    request = _request(tmp_path)
    keyframes = [
        {"name": path.name, "sha256": hashlib.sha256(blob).hexdigest()}
        for path, blob in request.keyframes
    ]
    manifest = {
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "keyframes": keyframes,
        "voice_texts_sha256": request.voice_receipt,
        "request": {
            "h3_workflow": "minimax_h3_lightx2v_v5",
            "duration": 10,
            "resolution": "768p竖",
        },
    }
    receipt = {
        "task_id": "old-known-task",
        "input_receipt": h3.canonical_json_sha256(manifest),
        "prompt_sha256": manifest["prompt_sha256"],
        "keyframes": keyframes,
        "request": {
            "workflow": "minimax_h3_lightx2v_v5",
            "duration": 10,
            "resolution": "768p竖",
        },
    }
    attempt = {
        "schema_version": 1,
        "cid": request.cid,
        "attempt_id": "000001",
        "client_request_id": request.client_request_id,
        "input": manifest,
        "input_receipt": h3.canonical_json_sha256(manifest),
        "status": "h3_running",
        "retryable": False,
        "h3": {
            "status": "running",
            "task_id": "old-known-task",
            "receipt": receipt,
        },
    }
    path = _attempt_file(request)
    path.parent.mkdir(parents=True)
    (request.workdir / ".h3" / "session.json").write_text(
        json.dumps({"schema_version": 1, "cid": request.cid}),
        encoding="utf-8",
    )
    path.write_text(json.dumps(attempt), encoding="utf-8")

    assert h3.inspect(request).status == "h3_running"
    calls = []

    def recovery(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        assert req.method == "GET"
        if req.url.path.endswith("/result/old-known-task"):
            return httpx.Response(
                200,
                json={"data": {"status": "SUCCESS", "results": [{"url": "https://download.invalid/video.mp4"}]}},
            )
        return _download_response(200, content=HappyProvider.video_bytes)

    with _client(recovery) as client:
        assert h3.resume(request, client=client).status == "succeeded"
    assert calls and all(call.method == "GET" for call in calls)
    recovered = json.loads(path.read_text(encoding="utf-8"))
    assert recovered["input"] == manifest
    assert recovered["h3"]["receipt"] == receipt


def test_h3_query_retries_without_repeating_post(tmp_path):
    request = _request(tmp_path)
    provider = HappyProvider()
    queries = 0

    def flaky(req: httpx.Request) -> httpx.Response:
        nonlocal queries
        if req.url.path.endswith("/result/h3-task-local"):
            queries += 1
            if queries < 3:
                raise httpx.ReadTimeout("temporary", request=req)
        return provider(req)

    with _client(flaky) as client:
        result = h3.start(request, client=client)
    assert result.status == "succeeded"
    assert queries == 3
    assert len(provider.h3_posts) == 1


def test_receipt_tampering_blocks_recovery_before_network(tmp_path):
    request = _request(tmp_path)
    with _client(HappyProvider(result_status="RUNNING")) as client:
        assert h3.start(request, client=client).status == "retryable_failure"
    path = _attempt_file(request)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["h3"]["receipt"]["prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(state), encoding="utf-8")
    with _client(lambda _request: pytest.fail("network must not be used")) as client:
        with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
            h3.resume(request, client=client)


def test_post_crash_without_task_id_becomes_submission_unknown(tmp_path, monkeypatch):
    request = _request(tmp_path)
    provider = HappyProvider()
    original = h3._atomic_write_json
    crashed = False

    def crash_after_post(path, payload):
        nonlocal crashed
        if not crashed and payload.get("h3", {}).get("task_id"):
            crashed = True
            raise OSError("simulated power loss")
        return original(path, payload)

    monkeypatch.setattr(h3, "_atomic_write_json", crash_after_post)
    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="state_persist_failed"):
            h3.start(request, client=client)
    monkeypatch.setattr(h3, "_atomic_write_json", original)
    assert h3.inspect(request).status == "h3_submitting"
    with _client(lambda _request: pytest.fail("unknown POST must not repeat")) as client:
        assert h3.resume(request, client=client).status == "submission_unknown"
    assert len(provider.h3_posts) == 1


def test_manual_retry_uses_new_attempt(tmp_path):
    request = _request(tmp_path)
    submissions = 0

    def rejected(req: httpx.Request) -> httpx.Response:
        nonlocal submissions
        submissions += 1
        return httpx.Response(422, json={"code": "invalid"})

    with _client(rejected) as client:
        with pytest.raises(h3.H3Error, match="h3_submit_rejected"):
            h3.start(request, client=client)
        same = h3.retry(request, "request-1", client=client)
        assert same.attempt_id == "000001"
        with pytest.raises(h3.H3Error, match="h3_submit_rejected"):
            h3.retry(request, "request-2", client=client)
    assert submissions == 2
    assert _attempt_file(request, 2).is_file()


def test_nonblocking_session_lock_rejects_same_cid_concurrency(tmp_path):
    request = _request(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    provider = HappyProvider()

    def blocked(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/minimax_h3_lightx2v_v5") and not entered.is_set():
            entered.set()
            assert release.wait(2)
        return provider(req)

    errors = []

    def run_first():
        try:
            with _client(blocked) as client:
                h3.start(request, client=client)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert entered.wait(2)
    try:
        with _client(provider) as client:
            with pytest.raises(h3.H3BusyError, match="session_busy"):
                h3.start(request, client=client)
    finally:
        release.set()
        thread.join(2)
    assert errors == []


@pytest.mark.parametrize(
    "url",
    [
        "http://download.invalid/video.mp4",
        "https://user:password@download.invalid/video.mp4",
        "https://127.0.0.1/video.mp4",
        "https://localhost/video.mp4",
    ],
)
def test_download_rejects_unsafe_urls(tmp_path, url):
    request = _request(tmp_path)
    with _client(HappyProvider(result_url=url)) as client:
        with pytest.raises(h3.H3Error, match="download_url_rejected"):
            h3.start(request, client=client)


def test_voice_receipt_is_canonical_and_required(tmp_path):
    assert h3.voice_texts_receipt(("a", "b")) == h3.canonical_json_sha256(["a", "b"])
    with pytest.raises(h3.ReceiptError, match="voice_receipt_mismatch"):
        replace(_request(tmp_path), voice_receipt="0" * 64)


def test_duration_over_provider_limit_is_rejected(tmp_path):
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(_request(tmp_path), duration=h3.H3_MAX_DURATION_S + 1)


def test_zero_duration_is_rejected(tmp_path):
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(_request(tmp_path), duration=0)
