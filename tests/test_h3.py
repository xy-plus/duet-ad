import base64
import hashlib
import json
import logging
import shutil
import socket
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import h3


VOICE_TEXTS = ("第一句台词", "第二句台词")


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
        if path.endswith(("/minimax_h3_lightx2v_v5_15s", "/minimax_h3_lightx2v_v5", "/minimax_h3_lightx2v")):
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
            if request.url.path.endswith(("/minimax_h3_lightx2v_v5_15s", "/minimax_h3_lightx2v_v5", "/minimax_h3_lightx2v"))
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


@pytest.fixture(scope="session")
def valid_av_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h3-av") / "valid-av.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=24:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def valid_fragmented_av_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h3-fragmented-av") / "valid-fragmented.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=32x32:r=24:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _timeline_payload(*, audio_end_s: float = 1.0) -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "time_base": "1/24",
                "start_time": "0.000000",
                "duration": "1.000000",
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "time_base": "1/48000",
                "start_time": "0.000000",
                "duration": f"{audio_end_s:.6f}",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "packets_and_frames": [
            {
                "type": "packet",
                "stream_index": 0,
                "pts_time": "0.000000",
                "dts_time": "-0.041667",
                "duration_time": "0.041667",
            },
            {
                "type": "packet",
                "stream_index": 0,
                "pts_time": "0.958333",
                "dts_time": "0.916667",
                "duration_time": "0.041667",
            },
            {
                "type": "frame",
                "media_type": "video",
                "stream_index": 0,
                "best_effort_timestamp_time": "0.000000",
                "duration_time": "0.041667",
            },
            {
                "type": "frame",
                "media_type": "video",
                "stream_index": 0,
                "best_effort_timestamp_time": "0.958333",
                "duration_time": "0.041667",
            },
            {
                "type": "packet",
                "stream_index": 1,
                "pts_time": "0.000000",
                "dts_time": "0.000000",
                "duration_time": "0.020000",
            },
            {
                "type": "packet",
                "stream_index": 1,
                "pts_time": f"{audio_end_s - 0.02:.6f}",
                "dts_time": f"{audio_end_s - 0.02:.6f}",
                "duration_time": "0.020000",
            },
            {
                "type": "frame",
                "media_type": "audio",
                "stream_index": 1,
                "best_effort_timestamp_time": "0.000000",
                "duration_time": "0.020000",
            },
            {
                "type": "frame",
                "media_type": "audio",
                "stream_index": 1,
                "best_effort_timestamp_time": f"{audio_end_s - 0.02:.6f}",
                "duration_time": "0.020000",
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "start_time": "0.000000",
            "duration": "99.000000",
        },
    }


def test_media_timeline_probe_records_decoded_audio_and_presented_timeline(
    valid_av_path,
):
    receipt = h3._probe_media_timeline(valid_av_path, 5)

    assert receipt["schema"] == "duet.h3.media_timeline"
    assert receipt["version"] == 1
    assert receipt["decode_complete"] is True
    assert receipt["container"]["duration_s"] == 1.0
    assert receipt["video"]["time_base"] == "1/12288"
    assert receipt["video"]["avg_frame_rate"] == "24/1"
    assert receipt["video"]["packet_dts_monotonic"] is True
    assert receipt["video"]["presentation_monotonic"] is True
    assert receipt["video"]["frame_count"] == 24
    assert receipt["audio"]["sample_rate"] == 44100
    assert receipt["audio"]["channels"] == 1
    assert len(receipt["audio"]["decoded_sha256"]) == 64
    assert abs(receipt["av_delta_s"]["start"]) <= 0.1
    assert abs(receipt["av_delta_s"]["end"]) <= 0.1


def test_media_timeline_accepts_decodable_fragmented_mp4_with_missing_packet_duration(
    valid_fragmented_av_path,
):
    receipt = h3._probe_media_timeline(valid_fragmented_av_path, 5)

    assert receipt["video"]["packet_count"] > 0
    assert receipt["audio"]["packet_count"] > 0
    assert receipt["audio"]["decoded_sha256"]


def test_media_timeline_preflight_rejects_output_beyond_request_ceiling(valid_av_path):
    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._probe_media_timeline(valid_av_path, 5, max_duration_s=0.5)

    assert raised.value.retryable is False


def test_media_timeline_probe_maps_non_utf8_output_to_stable_error():
    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._decode_probe_json(b"\xff")

    assert raised.value.retryable is False


def test_media_timeline_rejects_event_count_above_resource_ceiling(monkeypatch):
    monkeypatch.setattr(h3, "MAX_MEDIA_TIMELINE_EVENTS", 2)

    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._media_events({"packets_and_frames": [{}, {}, {}]})

    assert raised.value.retryable is False


def test_media_timeline_does_not_use_container_duration_as_visual_duration():
    receipt = h3._parse_media_timeline(
        _timeline_payload(),
        decoded_audio_sha256="a" * 64,
    )

    assert receipt["container"]["duration_s"] == 99.0
    assert receipt["video"]["duration_s"] == 1.0
    assert receipt["video"]["frame_end_s"] == 1.0


def test_media_timeline_rejects_nonmonotonic_packet_dts():
    payload = _timeline_payload()
    payload["packets_and_frames"][1]["dts_time"] = "-0.050000"

    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._parse_media_timeline(payload, decoded_audio_sha256="a" * 64)

    assert raised.value.retryable is False


def test_media_timeline_rejects_nonmonotonic_frame_presentation():
    payload = _timeline_payload()
    payload["packets_and_frames"][3]["best_effort_timestamp_time"] = "-0.010000"

    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._parse_media_timeline(payload, decoded_audio_sha256="a" * 64)

    assert raised.value.retryable is False


def test_media_timeline_rejects_material_av_endpoint_skew():
    with pytest.raises(h3.H3Error, match="download_invalid_video") as raised:
        h3._parse_media_timeline(
            _timeline_payload(audio_end_s=0.75),
            decoded_audio_sha256="a" * 64,
        )

    assert raised.value.retryable is False


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
    assert {
        key: state["h3"]["output"][key]
        for key in ("name", "sha256", "size")
    } == {
        "name": "generated.mp4",
        "sha256": hashlib.sha256(provider.video_bytes).hexdigest(),
        "size": len(provider.video_bytes),
    }
    assert state["h3"]["output"]["media_timeline"]["audio"] is None
    assert "download.invalid" not in json.dumps(state)
    assert "art-secret" not in json.dumps(state)


def test_success_exposes_and_persists_versioned_media_timeline(
    tmp_path, valid_av_path,
):
    request = replace(_request(tmp_path), duration=1)
    provider = HappyProvider()
    provider.video_bytes = valid_av_path.read_bytes()

    with _client(provider) as client:
        result = h3.start(request, client=client)

    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    persisted = state["h3"]["output"]["media_timeline"]
    assert persisted["schema"] == "duet.h3.media_timeline"
    assert persisted["version"] == 1
    assert persisted["audio"]["decoded_sha256"]
    assert result.media_timeline == persisted
    assert h3.load_media_timeline_receipt(request, result.attempt_id) == persisted
    assert h3.output_is_reusable(request) is True


def test_timeline_failure_is_deterministic_and_does_not_repeat_download(
    tmp_path, monkeypatch,
):
    request = replace(_request(tmp_path), duration=1)
    provider = HappyProvider()
    download_calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal download_calls
        if req.url.path.endswith("/minimax_h3_lightx2v_v5_15s"):
            return httpx.Response(200, json={"data": {"task_id": "h3-task-local"}})
        if req.url.path.endswith("/result/h3-task-local"):
            return httpx.Response(200, json={"data": {
                "status": "SUCCESS",
                "results": [{"url": "https://download.invalid/video.mp4"}],
            }})
        download_calls += 1
        return _download_response(200, content=provider.video_bytes)

    def invalid_timeline(*_args, **_kwargs):
        raise h3.H3Error("download_invalid_video", retryable=False)

    monkeypatch.setattr(h3, "_probe_media_timeline", invalid_timeline)
    with _client(handler) as client:
        with pytest.raises(h3.H3Error, match="download_invalid_video"):
            h3.start(request, client=client)

    state = json.loads(_attempt_file(request).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["retryable"] is False
    assert download_calls == 1


def test_media_timeline_loader_rejects_tampered_receipt(tmp_path, valid_av_path):
    request = replace(_request(tmp_path), duration=1)
    provider = HappyProvider()
    provider.video_bytes = valid_av_path.read_bytes()
    with _client(provider) as client:
        result = h3.start(request, client=client)

    path = _attempt_file(request)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["h3"]["output"]["media_timeline"]["audio"]["decoded_sha256"] = "0" * 63
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(h3.ReceiptError, match="state_invalid"):
        h3.load_media_timeline_receipt(request, result.attempt_id)


@pytest.mark.parametrize("damage", ["session", "task_id", "task_receipt"])
def test_media_timeline_loader_requires_exact_session_and_provider_provenance(
    tmp_path, valid_av_path, damage,
):
    request = replace(_request(tmp_path), duration=1)
    provider = HappyProvider()
    provider.video_bytes = valid_av_path.read_bytes()
    with _client(provider) as client:
        result = h3.start(request, client=client)

    if damage == "session":
        marker = request.workdir / ".h3" / "session.json"
        marker.write_text(
            json.dumps({"schema_version": h3.SCHEMA_VERSION, "cid": "wrong"}),
            encoding="utf-8",
        )
    else:
        path = _attempt_file(request)
        state = json.loads(path.read_text(encoding="utf-8"))
        state["h3"].pop("task_id" if damage == "task_id" else "receipt")
        path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(h3.ReceiptError):
        h3.load_media_timeline_receipt(request, result.attempt_id)


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
    assert replace(_request(tmp_path), duration=15).duration == 15
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(_request(tmp_path), duration=16)
    with pytest.raises(h3.H3Error, match="invalid_duration"):
        replace(
            _request(tmp_path),
            duration=11,
            workflow=h3.H3_PREVIOUS_WORKFLOW,
        )
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
    request = replace(request, timeouts=replace(request.timeouts, retry_count=0))

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


def test_provider_failure_automatically_creates_same_request_attempt_and_succeeds(
    tmp_path,
):
    request = _boundary_request(tmp_path)
    posts = []

    def provider(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            posts.append(req)
            return httpx.Response(
                200, json={"data": {"task_id": f"task-{len(posts)}"}}
            )
        if req.url.path.endswith("/result/task-1"):
            return httpx.Response(200, json={
                "request_id": "provider-failure-1",
                "msg": "GPU OOM",
                "data": {"status": "FAILED"},
            })
        if req.url.path.endswith("/result/task-2"):
            return httpx.Response(200, json={"data": {
                "status": "SUCCESS",
                "results": [{"url": "https://download.invalid/video.mp4"}],
            }})
        if req.url.host == "download.invalid":
            return _download_response(200, content=HappyProvider.video_bytes)
        raise AssertionError(req.url.path)

    with _client(provider) as client:
        result = h3.start(request, client=client)

    assert result.status == "succeeded"
    assert len(posts) == 2
    first_bytes = _attempt_file(request, 1).read_bytes()
    first = json.loads(first_bytes)
    second = json.loads(_attempt_file(request, 2).read_text(encoding="utf-8"))
    assert first["client_request_id"] == second["client_request_id"] == "boundary-1"
    assert first["input_receipt"] == second["input_receipt"]
    assert first["h3"]["task_id"] == "task-1"
    assert first["error"]["provider"]["request_id"] == "provider-failure-1"
    assert _attempt_file(request, 1).read_bytes() == first_bytes


def test_provider_failure_auto_retry_budget_counts_created_attempts(tmp_path):
    request = replace(
        _boundary_request(tmp_path),
        timeouts=replace(_boundary_request(tmp_path).timeouts, retry_count=2),
    )
    posts = 0

    def provider(req: httpx.Request) -> httpx.Response:
        nonlocal posts
        if req.method == "POST":
            posts += 1
            return httpx.Response(200, json={"data": {"task_id": f"task-{posts}"}})
        return httpx.Response(200, json={
            "request_id": f"provider-failure-{posts}",
            "data": {"status": "ERROR"},
        })

    with _client(provider) as client:
        result = h3.start(request, client=client)
        assert h3.resume(request, client=client).status == "failed"

    assert result.status == "failed"
    assert posts == 1 + request.timeouts.retry_count
    assert not _attempt_file(request, posts + 1).exists()


@pytest.mark.parametrize("damage", ["first_attempt_json", "middle_attempt_dir"])
def test_provider_auto_retry_rejects_attempt_ledger_gaps_without_new_post(
    tmp_path, damage,
):
    request = _boundary_request(tmp_path)
    posts = 0

    def failed(req: httpx.Request) -> httpx.Response:
        nonlocal posts
        if req.method == "POST":
            posts += 1
            return httpx.Response(200, json={"data": {"task_id": f"task-{posts}"}})
        return httpx.Response(200, json={
            "request_id": f"provider-failure-{posts}",
            "data": {"status": "FAILED"},
        })

    with _client(failed) as client:
        assert h3.start(request, client=client).status == "failed"
    assert posts == 3
    if damage == "first_attempt_json":
        _attempt_file(request, 1).unlink()
    else:
        shutil.rmtree(_attempt_file(request, 2).parent)

    calls = []
    expanded = replace(
        request, timeouts=replace(request.timeouts, retry_count=3)
    )
    with _client(lambda req: calls.append(req) or failed(req)) as client:
        with pytest.raises(h3.ReceiptError, match="state_invalid"):
            h3.resume(expanded, client=client)

    assert calls == []
    assert not _attempt_file(request, 4).exists()


def test_provider_auto_retry_accepts_structurally_valid_other_request_attempt(
    tmp_path,
):
    request = _boundary_request(tmp_path)
    request = replace(request, timeouts=replace(request.timeouts, retry_count=1))
    assert h3.prepare(request).attempt_id == "000001"
    posts = 0

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
        return _download_response(200, content=HappyProvider.video_bytes)

    with _client(provider) as client:
        result = h3.retry(request, "boundary-2", client=client)

    assert result.status == "succeeded"
    assert posts == 2
    ids = [
        json.loads(_attempt_file(request, number).read_text())["client_request_id"]
        for number in (1, 2, 3)
    ]
    assert ids == ["boundary-1", "boundary-2", "boundary-2"]


def test_resume_automatically_retries_complete_provider_failure(tmp_path):
    request = _boundary_request(tmp_path)

    def failed(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(200, json={"data": {"task_id": "failed-task"}})
        return httpx.Response(200, json={
            "request_id": "provider-failure",
            "data": {"status": "FAIL"},
        })

    no_auto = replace(request, timeouts=replace(request.timeouts, retry_count=0))
    with _client(failed) as client:
        assert h3.start(no_auto, client=client).status == "failed"

    posts = []
    provider = HappyProvider()
    with _client(lambda req: posts.append(req) or provider(req)) as client:
        assert h3.resume(request, client=client).status == "succeeded"

    assert len([item for item in posts if item.method == "POST"]) == 1
    assert _attempt_file(request, 2).is_file()


def test_resume_submits_persisted_ready_auto_attempt_only_once(tmp_path, monkeypatch):
    request = _boundary_request(tmp_path)
    original_create = h3._create_attempt
    created = False

    def crash_after_create(*args, **kwargs):
        nonlocal created
        state = original_create(*args, **kwargs)
        if state["attempt_id"] == "000002" and not created:
            created = True
            raise RuntimeError("simulated crash after automatic receipt")
        return state

    provider_failed = HappyProvider(result_status="FAILED")
    monkeypatch.setattr(h3, "_create_attempt", crash_after_create)
    with _client(provider_failed) as client:
        with pytest.raises(RuntimeError, match="simulated crash"):
            h3.start(request, client=client)
    monkeypatch.setattr(h3, "_create_attempt", original_create)
    ready = json.loads(_attempt_file(request, 2).read_text(encoding="utf-8"))
    assert ready["status"] == "ready_to_submit"

    lowered = replace(request, timeouts=replace(request.timeouts, retry_count=0))
    provider = HappyProvider()
    with _client(provider) as client:
        assert h3.resume(lowered, client=client).status == "succeeded"
        assert h3.resume(lowered, client=client).status == "succeeded"
    assert len(provider.h3_posts) == 1


@pytest.mark.parametrize(
    "terminal",
    ["submission_unknown", "h3_submit_rejected", "h3_result_missing"],
)
def test_non_provider_terminal_states_never_create_automatic_attempt(
    tmp_path, terminal,
):
    request = _boundary_request(tmp_path)
    posts = 0

    def provider(req: httpx.Request) -> httpx.Response:
        nonlocal posts
        if req.method == "POST":
            posts += 1
            if terminal == "submission_unknown":
                raise httpx.ReadTimeout("unknown", request=req)
            if terminal == "h3_submit_rejected":
                return httpx.Response(422, json={"code": "invalid"})
            return httpx.Response(200, json={"data": {"task_id": "task-1"}})
        return httpx.Response(200, json={"data": {"status": "SUCCESS", "results": []}})

    with _client(provider) as client:
        if terminal == "h3_submit_rejected":
            with pytest.raises(h3.H3Error, match=terminal):
                h3.start(request, client=client)
        elif terminal == "submission_unknown":
            with pytest.raises(h3.H3Error, match=terminal):
                h3.start(request, client=client)
        else:
            with pytest.raises(h3.H3Error, match=terminal):
                h3.start(request, client=client)
        assert h3.resume(request, client=client).status != "succeeded"

    assert posts == 1
    assert not _attempt_file(request, 2).exists()


@pytest.mark.parametrize(
    "case", ("secret_detail", "multiline_detail", "multiline_request_id", "extra_key")
)
def test_tampered_provider_failure_diagnostic_is_rejected(tmp_path, case):
    request = _boundary_request(tmp_path)
    request = replace(request, timeouts=replace(request.timeouts, retry_count=0))

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
    request = replace(request, timeouts=replace(request.timeouts, retry_count=0))

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


def test_reference_long_segment_accepts_integer_provider_ceiling_for_stitch(
    tmp_path, monkeypatch,
):
    request = replace(_request(tmp_path), duration=2)
    with _client(HappyProvider()) as client:
        assert h3.start(request, client=client).status == "succeeded"
    monkeypatch.setattr(h3, "_probe_video_duration", lambda *_args: 2.0)

    assert h3.output_is_reusable(
        request,
        expected_duration_s=1.133333,
        allow_provider_duration_ceiling=True,
    ) is True
    assert h3.output_is_reusable(
        request,
        expected_duration_s=1.133333,
    ) is False


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
        if req.url.path.endswith("/minimax_h3_lightx2v_v5_15s") and not entered.is_set():
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
    provider = HappyProvider(result_url=url)
    with _client(provider) as client:
        with pytest.raises(h3.H3Error, match="download_url_rejected"):
            h3.start(request, client=client)
        assert h3.resume(request, client=client).status == "failed"
    assert len(provider.h3_posts) == 1
    assert not _attempt_file(request, 2).exists()


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


def _controlled_multimodal_request(
    tmp_path: Path, controlled_root: Path,
) -> h3.H3Request:
    base = _request(tmp_path)
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"frozen-voice-reference")
    audio = h3.FrozenReferenceAudio(
        path=audio_path,
        data=audio_path.read_bytes(),
        order=1,
        purpose="voice",
        format="wav",
        sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        duration_s=2.0,
    )
    return replace(
        base,
        duration=4,
        workflow=h3.H3_MULTIMODAL_WORKFLOW,
        reference_audios=(audio,),
        skill_plan_sha256="1" * 64,
        multimodal_compiler_version="duet.h3-multimodal.v2",
        upstream_dialogue_receipt_sha256="2" * 64,
        audio_required=True,
        context_ir_receipt_path=base.workdir / ".context-ir" / "attempts" / "000001" / "receipt.json",
        context_ir_receipt_sha256="3" * 64,
        gateway_storage_root=controlled_root,
    )


def test_controlled_gateway_projection_is_receipt_bound_and_idempotent(
    tmp_path, monkeypatch,
):
    controlled_root = tmp_path / "controlled"
    request = _controlled_multimodal_request(tmp_path, controlled_root)

    h3._materialize_gateway_inputs(request)
    manifest = h3._input_manifest(request)
    bound = manifest["multimodal"]["gateway_inputs"]

    assert [item["order"] for item in bound] == [1, 2, 1]
    assert [item["role"] for item in bound] == [
        "reference_image", "reference_image", "reference_audio",
    ]
    for item in bound:
        provider = Path(item["provider_path"])
        assert provider.resolve().is_relative_to(controlled_root.resolve())
        assert item["source_sha256"] == item["provider_sha256"]
        assert hashlib.sha256(provider.read_bytes()).hexdigest() == item["provider_sha256"]

    monkeypatch.setattr(
        h3,
        "_atomic_write_bytes",
        lambda *_args, **_kwargs: pytest.fail("exact projection must perform zero writes"),
    )
    h3._materialize_gateway_inputs(request)


def test_controlled_gateway_projection_rejects_source_symlink_and_drift(tmp_path):
    controlled_root = tmp_path / "controlled"
    request = _controlled_multimodal_request(tmp_path, controlled_root)
    source = request.keyframes[0][0]
    source.unlink()
    target = tmp_path / "real.png"
    target.write_bytes(request.keyframes[0][1])
    source.symlink_to(target)

    with pytest.raises(h3.ReceiptError, match="gateway_input_symlink"):
        h3._materialize_gateway_inputs(request)

    source.unlink()
    source.write_bytes(request.keyframes[0][1])
    h3._materialize_gateway_inputs(request)
    provider = Path(
        h3._input_manifest(request)["multimodal"]["gateway_inputs"][0]["provider_path"]
    )
    provider.write_bytes(b"drifted")
    with pytest.raises(h3.ReceiptError, match="receipt_mismatch"):
        h3._materialize_gateway_inputs(request)


def test_exact_legacy_storage_rejection_appends_same_client_attempt(
    tmp_path, monkeypatch,
):
    request = _controlled_multimodal_request(tmp_path, tmp_path / "controlled")
    request = replace(request, timeouts=replace(request.timeouts, retry_count=0))
    monkeypatch.setattr(h3, "_require_context_ir_receipt", lambda _request: None)
    old = h3._new_state(request, "000001", request.client_request_id)
    legacy = h3._pre_controlled_storage_input_manifest(request)
    old.update({
        "input": legacy,
        "input_receipt": h3.canonical_json_sha256(legacy),
        "status": "failed",
        "retryable": False,
        "h3": {"status": "failed"},
        "error": {"code": "h3_submit_rejected"},
    })
    h3._attempt_path(request, "000001").parent.mkdir(parents=True, exist_ok=True)
    h3._save_state(request, old)
    old_bytes = h3._attempt_path(request, "000001").read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    evidence_sha = hashlib.sha256(b"exact gateway 400 evidence").hexdigest()

    calls = []

    def gateway(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if req.method == "POST":
            return httpx.Response(201, json={"task_id": "new-task"})
        return httpx.Response(200, json={"status": "failed"})

    with _client(gateway) as client:
        result = h3.retry_controlled_storage_rejection(
            request,
            legacy_attempt_sha256=old_sha,
            legacy_evidence_sha256=evidence_sha,
            client=client,
        )

    assert result.attempt_id == "000002"
    assert h3._attempt_path(request, "000001").read_bytes() == old_bytes
    assert len([call for call in calls if call.method == "POST"]) == 1
    posted = json.loads(next(call for call in calls if call.method == "POST").content)
    assert all(
        Path(path).resolve().is_relative_to(request.gateway_storage_root.resolve())
        for path in posted["images"]
    )
    with pytest.raises(h3.H3Error, match="controlled_storage_retry_not_allowed"):
        h3.retry_controlled_storage_rejection(
            request,
            legacy_attempt_sha256=old_sha,
            legacy_evidence_sha256=evidence_sha,
            client=httpx.Client(transport=httpx.MockTransport(gateway)),
        )
