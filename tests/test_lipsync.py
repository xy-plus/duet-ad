import asyncio
import hashlib
import json
from dataclasses import replace

import pytest

from app import lipsync


def _write(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _project(tmp_path):
    _write(tmp_path / "work/silent.mp4", b"silent-video")
    _write(tmp_path / "work/audio/target.wav", b"unified-target-audio")
    _write(tmp_path / "work/audio/alice.wav", b"alice-audio")
    _write(tmp_path / "work/audio/bob.wav", b"bob-audio")
    _write(tmp_path / "work/faces/alice.png", b"alice-face")
    _write(tmp_path / "work/faces/bob.png", b"bob-face")
    _write(tmp_path / "work/upstream/h3-attempt.json", b'{"attempt":"visual-1"}\n')
    _write(tmp_path / "work/upstream/target-audio.json", b'{"audio":"target-1"}\n')
    return tmp_path


def _input(**changes):
    request = lipsync.LipSyncInput(
        video_path="work/silent.mp4",
        visual_receipt_path="work/upstream/h3-attempt.json",
        visual_attempt_id="visual-1",
        target_audio_path="work/audio/target.wav",
        target_audio_receipt_path="work/upstream/target-audio.json",
        target_audio_decoded_sha256="d" * 64,
        target_audio_sample_rate=48_000,
        target_audio_channels=2,
        speaker_to_face={"alice": "face-alice", "bob": "face-bob"},
        intervals=(
            lipsync.AudioInterval("alice", "work/audio/alice.wav", 0, 3500),
            lipsync.AudioInterval("bob", "work/audio/bob.wav", 4000, 7500),
        ),
        reference_frames={
            "face-alice": "work/faces/alice.png",
            "face-bob": "work/faces/bob.png",
        },
        pts_time_base_num=1,
        pts_time_base_den=1000,
        timeline_start_pts=0,
        timeline_end_pts=7500,
        provider=lipsync.TENCENT_MULTI_PERSON_PROVIDER,
        provider_params={
            "silent_mouth_mode": "ForceClosed",
            "face_match_mode": "Strict",
            "resolution": 0,
        },
        idempotency_key="lipsync-scene-0001",
        workflow="dual-target-av-v1",
    )
    return replace(request, **changes)


def _freeze(tmp_path, request=None):
    root = _project(tmp_path)
    result = lipsync.freeze_request(root, "work/lipsync/receipt.json", request or _input())
    return root, result


def _credentials(app_key="app-key", access_token="access-token"):
    return lipsync.TencentCredentials(app_key=app_key, access_token=access_token)


def _assets():
    return {
        "work/silent.mp4": "https://assets.example/silent.mp4",
        "work/audio/alice.wav": "https://assets.example/alice.wav",
        "work/audio/bob.wav": "https://assets.example/bob.wav",
        "work/faces/alice.png": "https://assets.example/alice.png",
        "work/faces/bob.png": "https://assets.example/bob.png",
    }


def _provider():
    return lipsync.TencentMultiPersonProvider(clock=lambda: 1_700_000_000)


def _run(coro):
    return asyncio.run(coro)


def test_freeze_receipt_binds_only_project_relative_files_and_hashes(tmp_path):
    root, result = _freeze(tmp_path)

    assert result == lipsync.LipSyncResult(status="prepared")
    receipt = json.loads((root / "work/lipsync/receipt.json").read_text())
    frozen = receipt["input"]
    assert receipt["schema"] == "duet.lipsync.request"
    assert receipt["status"] == "prepared"
    assert frozen["video"] == {
        "path": "work/silent.mp4",
        "sha256": hashlib.sha256(b"silent-video").hexdigest(),
        "size": len(b"silent-video"),
    }
    assert frozen["target_audio"] == {
        "path": "work/audio/target.wav",
        "sha256": hashlib.sha256(b"unified-target-audio").hexdigest(),
        "size": len(b"unified-target-audio"),
        "decoded_sha256": "d" * 64,
    }
    assert frozen["visual_receipt"]["path"] == "work/upstream/h3-attempt.json"
    assert frozen["target_audio_receipt"]["path"] == "work/upstream/target-audio.json"
    assert frozen["speaker_to_face"] == {
        "alice": "face-alice",
        "bob": "face-bob",
    }
    assert frozen["intervals"][0]["audio"] == {
        "path": "work/audio/alice.wav",
        "sha256": hashlib.sha256(b"alice-audio").hexdigest(),
        "size": len(b"alice-audio"),
    }
    assert frozen["reference_frames"]["face-bob"] == {
        "path": "work/faces/bob.png",
        "sha256": hashlib.sha256(b"bob-face").hexdigest(),
        "size": len(b"bob-face"),
    }
    assert frozen["pts_time_base"] == {"num": 1, "den": 1000}
    assert frozen["timeline"] == {"start_pts": 0, "end_pts": 7500}
    assert frozen["idempotency_key"] == "lipsync-scene-0001"
    assert receipt["input_receipt"] == lipsync.canonical_json_sha256(frozen)
    comparison = receipt["comparison"]
    assert comparison["schema"] == "duet.av-generation"
    assert comparison["version"] == 1
    assert comparison["route"] == "post_h3_lipsync"
    assert comparison["workflow"] == "dual-target-av-v1"
    assert comparison["visual_input"]["receipt_sha256"] == hashlib.sha256(
        b'{"attempt":"visual-1"}\n'
    ).hexdigest()
    assert comparison["visual_input"]["items"] == [
        {
            "order": 0,
            "name": "silent.mp4",
            "sha256": hashlib.sha256(b"silent-video").hexdigest(),
            "size": len(b"silent-video"),
        },
        {
            "order": 1,
            "name": "alice.png",
            "sha256": hashlib.sha256(b"alice-face").hexdigest(),
            "size": len(b"alice-face"),
        },
        {
            "order": 2,
            "name": "bob.png",
            "sha256": hashlib.sha256(b"bob-face").hexdigest(),
            "size": len(b"bob-face"),
        },
    ]
    assert comparison["target_audio_materials"]["receipt_sha256"] == hashlib.sha256(
        b'{"audio":"target-1"}\n'
    ).hexdigest()
    target_item = comparison["target_audio_materials"]["items"][0]
    assert target_item["role"] == "target_dialogue"
    assert target_item["sha256"] == hashlib.sha256(b"unified-target-audio").hexdigest()
    assert target_item["decoded_sha256"] == "d" * 64
    assert target_item["time_base"] == {"num": 1, "den": 1000}
    assert target_item["start_pts"] == 0 and target_item["end_pts"] == 7500
    assert comparison["upstream"] == {
        "receipt_sha256": comparison["visual_input"]["receipt_sha256"],
        "attempt_id": "visual-1",
    }
    assert comparison["output"] is None
    assert receipt["comparison_receipt"] == lipsync.canonical_json_sha256(comparison)
    assert str(root) not in json.dumps(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_path", "../outside.mp4"),
        ("video_path", "/tmp/outside.mp4"),
        ("audio_path", "../outside.wav"),
        ("target_audio_path", "../target.wav"),
        ("visual_receipt_path", "/tmp/visual.json"),
        ("target_audio_receipt_path", "work/../audio.json"),
        ("reference_frame", "work/../faces/outside.png"),
    ],
)
def test_freeze_rejects_absolute_or_traversing_input_paths(tmp_path, field, value):
    root = _project(tmp_path)
    request = _input()
    if field == "video_path":
        request = replace(request, video_path=value)
    elif field == "audio_path":
        intervals = list(request.intervals)
        intervals[0] = replace(intervals[0], audio_path=value)
        request = replace(request, intervals=tuple(intervals))
    elif field in {
        "target_audio_path", "visual_receipt_path", "target_audio_receipt_path"
    }:
        request = replace(request, **{field: value})
    else:
        request = replace(
            request,
            reference_frames={**request.reference_frames, "face-alice": value},
        )

    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(root, "work/lipsync/receipt.json", request)
    assert caught.value.code == "invalid_project_path"


def test_receipt_path_must_be_project_relative_and_cannot_alias_input(tmp_path):
    root = _project(tmp_path)
    for receipt_path in ("../receipt.json", "/tmp/receipt.json", "work/silent.mp4"):
        with pytest.raises(lipsync.LipSyncError) as caught:
            lipsync.freeze_request(root, receipt_path, _input())
        assert caught.value.code in {"invalid_project_path", "receipt_path_conflict"}


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (replace(_input(), speaker_to_face={"alice": "face-alice"}), "speaker_face_mapping_invalid"),
        (
            replace(_input(), reference_frames={"face-alice": "work/faces/alice.png"}),
            "speaker_face_mapping_invalid",
        ),
        (
            replace(
                _input(),
                speaker_to_face={"alice": "face-alice", "bob": "face-alice"},
                reference_frames={"face-alice": "work/faces/alice.png"},
            ),
            "speaker_face_mapping_invalid",
        ),
    ],
)
def test_freeze_fails_closed_on_missing_or_non_unique_speaker_face_mapping(
    tmp_path, candidate, code,
):
    root = _project(tmp_path)
    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(root, "work/lipsync/receipt.json", candidate)
    assert caught.value.code == code


def test_freeze_rejects_more_than_three_people(tmp_path):
    root = _project(tmp_path)
    _write(root / "work/audio/cara.wav", b"cara-audio")
    _write(root / "work/audio/dan.wav", b"dan-audio")
    _write(root / "work/faces/cara.png", b"cara-face")
    _write(root / "work/faces/dan.png", b"dan-face")
    request = replace(
        _input(),
        speaker_to_face={
            "alice": "face-alice",
            "bob": "face-bob",
            "cara": "face-cara",
            "dan": "face-dan",
        },
        intervals=_input().intervals
        + (
            lipsync.AudioInterval("cara", "work/audio/cara.wav", 8000, 9000),
            lipsync.AudioInterval("dan", "work/audio/dan.wav", 9000, 10000),
        ),
        reference_frames={
            **_input().reference_frames,
            "face-cara": "work/faces/cara.png",
            "face-dan": "work/faces/dan.png",
        },
    )
    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(root, "work/lipsync/receipt.json", request)
    assert caught.value.code == "too_many_speakers"


@pytest.mark.parametrize(
    "intervals",
    [
        (
            lipsync.AudioInterval("alice", "work/audio/alice.wav", 0, 3500),
            lipsync.AudioInterval("bob", "work/audio/bob.wav", 3499, 7500),
        ),
        (
            lipsync.AudioInterval("alice", "work/audio/alice.wav", True, 3500),
            lipsync.AudioInterval("bob", "work/audio/bob.wav", 4000, 7500),
        ),
        (
            lipsync.AudioInterval("alice", "work/audio/alice.wav", 3500, 3500),
            lipsync.AudioInterval("bob", "work/audio/bob.wav", 4000, 7500),
        ),
    ],
)
def test_freeze_rejects_overlapping_or_non_integer_pts_intervals(tmp_path, intervals):
    root = _project(tmp_path)
    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(
            root,
            "work/lipsync/receipt.json",
            replace(_input(), intervals=intervals),
        )
    assert caught.value.code == "intervals_invalid"


def test_freeze_is_idempotent_but_refuses_same_receipt_for_changed_input(tmp_path):
    root, first = _freeze(tmp_path)
    second = lipsync.freeze_request(root, "work/lipsync/receipt.json", _input())
    assert second == first

    changed = replace(_input(), idempotency_key="lipsync-scene-0002")
    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(root, "work/lipsync/receipt.json", changed)
    assert caught.value.code == "receipt_conflict"


def test_target_audio_requires_decoded_evidence_and_is_part_of_frozen_truth(tmp_path):
    root = _project(tmp_path)
    with pytest.raises(lipsync.LipSyncError) as caught:
        lipsync.freeze_request(
            root,
            "work/lipsync/receipt.json",
            replace(_input(), target_audio_decoded_sha256="not-a-hash"),
        )
    assert caught.value.code == "target_audio_evidence_invalid"

    lipsync.freeze_request(root, "work/lipsync/receipt.json", _input())
    (root / "work/audio/target.wav").write_bytes(b"different-target")
    calls = 0

    async def send(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("changed target audio must not submit")

    with pytest.raises(lipsync.LipSyncError) as changed:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=send,
            )
        )
    assert changed.value.code == "frozen_input_changed"
    assert calls == 0


def test_missing_credentials_fails_before_any_request(tmp_path):
    root, _ = _freeze(tmp_path)
    calls = 0

    async def send(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("must not send")

    with pytest.raises(lipsync.LipSyncError) as caught:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(app_key="", access_token=""),
                asset_urls=_assets(),
                send=send,
            )
        )
    assert caught.value.code == "lipsync_not_configured"
    assert calls == 0
    assert lipsync.load_status(root, "work/lipsync/receipt.json").status == "prepared"


def test_submit_persists_receipt_before_post_and_projects_tencent_multi_speaker(tmp_path):
    root, _ = _freeze(tmp_path)
    calls = []

    async def send(request):
        calls.append(request)
        durable = json.loads((root / "work/lipsync/receipt.json").read_text())
        assert durable["status"] == "submitting"
        assert durable["provider_request_sha256"] == lipsync.canonical_json_sha256(
            request.body
        )
        return lipsync.ProviderHttpResponse(
            200,
            {
                "Header": {"Code": 0, "Message": ""},
                "Payload": {"TaskId": "task-123"},
            },
        )

    result = _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=_assets(),
            send=send,
        )
    )
    assert result == lipsync.LipSyncResult(status="accepted", task_id="task-123")
    assert len(calls) == 1
    outbound = calls[0]
    assert outbound.operation == "submit"
    assert outbound.method == "POST"
    assert outbound.url.endswith("/videomakenotrain")
    assert outbound.query["appkey"] == "app-key"
    assert "access-token" not in json.dumps(outbound.query)
    payload = outbound.body["Payload"]
    assert payload["DriverType"] == "OriginalVoice"
    assert payload["InputAudioUrl"] == ""
    assert payload["InputSsml"] == ""
    assert payload["SpeechParam"] == {}
    assert payload["VideoParam"] == {
        "DisableIdDetect": 0,
        "MakeType": "Default",
        "StartTime": 0,
        "EndTime": 0,
        "Resolution": 0,
        "FaceMatchMode": "Strict",
        "RefPhotoUrl": "",
        "RefPhotoUrls": [],
    }
    assert payload["MultiSpeakerParam"]["Speakers"] == [
        {
            "IdPhotoUrl": "https://assets.example/alice.png",
            "AudioSegments": [
                {
                    "AudioUrl": "https://assets.example/alice.wav",
                    "StartTime": 0,
                    "EndTime": 3.5,
                }
            ],
        },
        {
            "IdPhotoUrl": "https://assets.example/bob.png",
            "AudioSegments": [
                {
                    "AudioUrl": "https://assets.example/bob.wav",
                    "StartTime": 4,
                    "EndTime": 7.5,
                }
            ],
        },
    ]
    receipt_text = (root / "work/lipsync/receipt.json").read_text()
    assert "app-key" not in receipt_text
    assert "access-token" not in receipt_text
    assert "assets.example" not in receipt_text


def test_post_timeout_without_task_id_is_unknown_and_never_resubmits(tmp_path):
    root, _ = _freeze(tmp_path)
    calls = 0

    async def timeout(_request):
        nonlocal calls
        calls += 1
        raise TimeoutError("provider may have accepted secret-request")

    with pytest.raises(lipsync.LipSyncError) as caught:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=timeout,
            )
        )
    assert caught.value.code == "submission_unknown"
    assert "secret-request" not in caught.value.detail
    assert calls == 1

    async def forbidden(_request):
        raise AssertionError("submission_unknown must not make any request")

    with pytest.raises(lipsync.LipSyncError) as second:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=forbidden,
            )
        )
    assert second.value.code == "submission_unknown"
    receipt = json.loads((root / "work/lipsync/receipt.json").read_text())
    assert receipt["status"] == "submission_unknown"
    assert "secret-request" not in json.dumps(receipt)


def test_crash_state_submitting_without_task_id_becomes_unknown_without_network(tmp_path):
    root, _ = _freeze(tmp_path)
    receipt_path = root / "work/lipsync/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "submitting"
    receipt["provider_request_sha256"] = "a" * 64
    receipt["asset_urls_sha256"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt))
    calls = 0

    async def send(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("crash recovery cannot resubmit")

    with pytest.raises(lipsync.LipSyncError) as caught:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=send,
            )
        )
    assert caught.value.code == "submission_unknown"
    assert calls == 0
    assert json.loads(receipt_path.read_text())["status"] == "submission_unknown"


def test_any_valid_task_id_switches_to_query_only_even_on_unusual_http_status(tmp_path):
    root, _ = _freeze(tmp_path)
    operations = []

    async def send(request):
        operations.append(request.operation)
        if request.operation == "submit":
            return lipsync.ProviderHttpResponse(
                503,
                {
                    "Header": {"Code": 999999, "Message": "raw"},
                    "Payload": {"TaskId": "accepted-despite-framing"},
                },
            )
        return lipsync.ProviderHttpResponse(
            200,
            {
                "Header": {"Code": 0},
                "Payload": {"Status": "MAKING", "Progress": 10},
            },
        )

    accepted = _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=_assets(),
            send=send,
        )
    )
    assert accepted.task_id == "accepted-despite-framing"
    assert _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=None,
            send=send,
        )
    ).status == "processing"
    assert operations == ["submit", "query"]


def test_task_id_recovery_can_only_query_and_does_not_need_asset_urls(tmp_path):
    root, _ = _freeze(tmp_path)
    operations = []

    async def send(request):
        operations.append(request.operation)
        if request.operation == "submit":
            return lipsync.ProviderHttpResponse(
                200, {"Header": {"Code": 0}, "Payload": {"TaskId": "task-123"}}
            )
        if len(operations) == 2:
            assert request.method == "POST"  # Tencent's official query endpoint is POST.
            assert request.body == {"Header": {}, "Payload": {"TaskId": "task-123"}}
            return lipsync.ProviderHttpResponse(
                200,
                {
                    "Header": {"Code": 0},
                    "Payload": {"Status": "MAKING", "Progress": 50},
                },
            )
        return lipsync.ProviderHttpResponse(
            200,
            {
                "Header": {"Code": 0},
                "Payload": {
                    "Status": "SUCCESS",
                    "Progress": 100,
                    "MediaUrl": "https://result.example/task-123.mp4",
                    "Duration": 7500,
                },
            },
        )

    accepted = _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=_assets(),
            send=send,
        )
    )
    processing = _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=None,
            send=send,
        )
    )
    succeeded = _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=None,
            send=send,
        )
    )
    assert accepted.status == "accepted"
    assert processing == lipsync.LipSyncResult(status="processing", task_id="task-123")
    assert succeeded == lipsync.LipSyncResult(
        status="succeeded",
        task_id="task-123",
        media_url="https://result.example/task-123.mp4",
        duration_ms=7500,
    )
    assert operations == ["submit", "query", "query"]


def test_query_error_is_safe_and_retryable_without_a_new_submission(tmp_path):
    root, _ = _freeze(tmp_path)

    async def accepted(_request):
        return lipsync.ProviderHttpResponse(
            200, {"Header": {"Code": 0}, "Payload": {"TaskId": "task-123"}}
        )

    _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=_assets(),
            send=accepted,
        )
    )
    operations = []

    async def transient(request):
        operations.append(request.operation)
        if len(operations) == 1:
            raise RuntimeError("raw-provider-secret")
        return lipsync.ProviderHttpResponse(
            200,
            {"Header": {"Code": 0}, "Payload": {"Status": "COMMIT", "Progress": 0}},
        )

    with pytest.raises(lipsync.LipSyncError) as caught:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=None,
                send=transient,
            )
        )
    assert caught.value.code == "provider_query_unavailable"
    assert "raw-provider-secret" not in caught.value.detail
    assert _run(
        lipsync.advance(
            root,
            "work/lipsync/receipt.json",
            provider=_provider(),
            credentials=_credentials(),
            asset_urls=None,
            send=transient,
        )
    ).status == "processing"
    assert operations == ["query", "query"]


def test_explicit_provider_error_is_redacted_from_receipt_and_public_error(tmp_path):
    root, _ = _freeze(tmp_path)

    async def rejected(_request):
        return lipsync.ProviderHttpResponse(
            400,
            {
                "Header": {
                    "Code": 100001,
                    "Message": "secret upstream diagnostic with customer data",
                }
            },
        )

    with pytest.raises(lipsync.LipSyncError) as caught:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=rejected,
            )
        )
    assert caught.value.code == "provider_rejected"
    assert caught.value.detail == "Lip-sync provider rejected the request"
    receipt_text = (root / "work/lipsync/receipt.json").read_text()
    assert "secret upstream" not in receipt_text
    assert json.loads(receipt_text)["provider_code"] == 100001


def test_input_drift_or_receipt_tampering_blocks_before_submission(tmp_path):
    root, _ = _freeze(tmp_path)
    (root / "work/audio/alice.wav").write_bytes(b"changed")
    calls = 0

    async def send(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("must not send")

    with pytest.raises(lipsync.LipSyncError) as drift:
        _run(
            lipsync.advance(
                root,
                "work/lipsync/receipt.json",
                provider=_provider(),
                credentials=_credentials(),
                asset_urls=_assets(),
                send=send,
            )
        )
    assert drift.value.code == "frozen_input_changed"
    assert calls == 0

    receipt_path = root / "work/lipsync/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["input"]["idempotency_key"] = "tampered"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(lipsync.LipSyncError) as tampered:
        lipsync.load_status(root, "work/lipsync/receipt.json")
    assert tampered.value.code == "receipt_invalid"


def test_atomic_receipt_write_fsyncs_parent_directory(tmp_path, monkeypatch):
    root = _project(tmp_path)
    fsyncs = []
    monkeypatch.setattr(lipsync, "_fsync_dir", lambda path: fsyncs.append(path))
    lipsync.freeze_request(root, "work/lipsync/receipt.json", _input())
    assert fsyncs == [root / "work/lipsync"]
