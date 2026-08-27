import copy
import hashlib
import json
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from app import dialogue_timing, h3, h3_multimodal


def _write_wav(path: Path, seconds: int = 2, sample_value: int = 0) -> bytes:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(sample_value.to_bytes(2, "little") * 8000 * seconds)
    return path.read_bytes()


def _visual(tmp_path: Path) -> h3_multimodal.FrozenVisualInput:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in (1, 2):
        path = tmp_path / f"{index:02d}.png"
        path.write_bytes(f"frame-{index}".encode())
        frames.append(path)
    return h3_multimodal.FrozenVisualInput(
        prompt="雨夜车站，人物面对镜头，镜头缓慢推进。",
        keyframes=h3.freeze_keyframes(tuple(frames)),
    )


def _audios(tmp_path: Path, purposes=("voice", "ambience")):
    sources = []
    for index, purpose in enumerate(purposes, 1):
        path = tmp_path / f"audio-{index}.wav"
        _write_wav(path, sample_value=index)
        sources.append((path, purpose))
    return h3.freeze_reference_audios(tuple(sources))


def _plan(visual_prompt: str) -> dict:
    return {
        "version": 2,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual_prompt,
        "dialogue_source_sha256": h3.canonical_json_sha256(
            list(_dialogue("我会准时回来。"))
        ),
        "subjects": [
            {"subject_id": "S1", "picture_refs": [1], "voice_ref": 1},
        ],
        "audio_refs": [
            {"audio_index": 1, "purpose": "voice", "subject_id": "S1"},
            {"audio_index": 2, "purpose": "ambience", "subject_id": None},
        ],
        "speech_bindings": [
            {
                "line_index": 1,
                "delivery": "on_screen",
                "subject_id": "S1",
                "language": "Chinese",
                "voice_ref": None,
            },
        ],
        "sound_design": {
            "ambience_refs": [
                {"audio_index": 2, "description": "远处雨声"},
            ],
            "effects": [],
        },
    }


def _dialogue(*texts: str) -> tuple[dict, ...]:
    return tuple(
        {
            "text": text,
            "start_s": float(index - 1) * 2,
            "end_s": float(index) * 2,
            "classification": "spoken",
            "provenance": "asr",
        }
        for index, text in enumerate(texts, 1)
    )


def _dialogue_args(*texts: str) -> dict:
    dialogue = _dialogue(*texts)
    return {
        "upstream_dialogue": dialogue,
        "upstream_dialogue_receipt_sha256": h3.canonical_json_sha256(
            list(dialogue)
        ),
        "upstream_dialogue_content_sha256": h3.canonical_json_sha256(
            list(dialogue)
        ),
        "speaker_timing": dialogue_timing.FrozenSpeakerTiming(
            sha256="c" * 64,
            source_sha256="a" * 64,
            duration=Fraction(8),
            windows={
                "S1": (
                    dialogue_timing.FrozenLipWindow(
                        Fraction(0), Fraction(8)
                    ),
                )
            },
        ),
    }


def _request(tmp_path: Path) -> h3.H3Request:
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    request = h3_multimodal.build_h3_request(
        skill_plan=plan,
        approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
        **_dialogue_args("我会准时回来。"),
        visual=visual,
        reference_audios=_audios(tmp_path),
        mode="multimodal",
        cid="cid-audio",
        workdir=tmp_path / "session",
        client_request_id="audio-request-1",
        duration=8,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="art-secret",
        timeouts=h3.Timeouts(
            request_s=0.1,
            h3_poll_s=0.02,
            download_s=0.1,
            poll_interval_s=0,
            retry_interval_s=0,
        ),
    )
    authority_root = tmp_path.resolve()
    legacy_receipt = authority_root / "legacy_h3_multimodal_source.json"
    legacy_receipt.write_text(json.dumps({
        "schema": "duet.h3-multimodal-source",
        "version": 2,
        "mode": "multimodal",
        "approved_skill_plan_sha256": h3.canonical_json_sha256(plan),
        "multimodal_input": {"path": "input.json", "sha256": "a" * 64},
        "skill_plan": {"path": "plan.json", "sha256": "b" * 64},
        "reference_audios": [],
    }), encoding="utf-8")
    return replace(
        request,
        gateway_storage_root=authority_root,
        speaker_timing_legacy_source_version=2,
        speaker_timing_legacy_receipt_path=legacy_receipt.name,
        speaker_timing_legacy_receipt_sha256=hashlib.sha256(
            legacy_receipt.read_bytes()
        ).hexdigest(),
        speaker_timing_authority_root=authority_root,
    )


def _attempt(request: h3.H3Request) -> dict:
    path = request.workdir / ".h3" / "attempts" / "000001" / "attempt.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_freeze_reference_audio_probes_exact_bytes_once(tmp_path):
    source = tmp_path / "voice.wav"
    original = _write_wav(source)

    frozen = h3.freeze_reference_audios(((source, "voice"),))
    source.write_bytes(b"changed-after-freeze")

    assert frozen[0].data == original
    assert frozen[0].sha256 == hashlib.sha256(original).hexdigest()
    assert frozen[0].duration_s == pytest.approx(2.0, abs=0.01)
    assert frozen[0].order == 1


def test_dialogue_content_hash_mismatch_fails_before_attempt(tmp_path):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    dialogue_args = _dialogue_args("我会准时回来。")
    dialogue_args["upstream_dialogue_content_sha256"] = "0" * 64

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="upstream_dialogue_receipt_mismatch",
    ):
        h3_multimodal.build_h3_request(
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            **dialogue_args,
            visual=visual,
            reference_audios=_audios(tmp_path),
            mode="multimodal",
            cid="mismatch",
            workdir=tmp_path / "mismatch",
            client_request_id="mismatch",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )

    assert not (tmp_path / "mismatch" / ".h3").exists()


def test_on_screen_dialogue_requires_explicit_verified_speaker_timing(tmp_path):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    args = _dialogue_args("我会准时回来。")
    args["speaker_timing"] = None

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="speaker_timing_evidence_missing",
    ):
        h3_multimodal.build_h3_request(
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            **args,
            visual=visual,
            reference_audios=_audios(tmp_path),
            mode="multimodal",
            cid="missing-speaker-timing",
            workdir=tmp_path / "missing-speaker-timing",
            client_request_id="missing-speaker-timing",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )

    assert not (tmp_path / "missing-speaker-timing" / ".h3").exists()


def test_on_screen_dialogue_cannot_start_before_verified_lip_window(tmp_path):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    args = _dialogue_args("我会准时回来。")
    args["speaker_timing"] = dialogue_timing.FrozenSpeakerTiming(
        sha256="c" * 64,
        source_sha256="a" * 64,
        duration=Fraction(8),
        windows={
            "S1": (
                dialogue_timing.FrozenLipWindow(
                    Fraction("0.75"), Fraction(8)
                ),
            )
        },
    )

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="dialogue_before_speaker_lip_window",
    ):
        h3_multimodal.build_h3_request(
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            **args,
            visual=visual,
            reference_audios=_audios(tmp_path),
            mode="multimodal",
            cid="early-dialogue",
            workdir=tmp_path / "early-dialogue",
            client_request_id="early-dialogue",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )

    assert not (tmp_path / "early-dialogue" / ".h3").exists()


def test_h3_paid_and_read_boundaries_revalidate_on_screen_dialogue_digest(
    tmp_path,
):
    request = _request(tmp_path)
    request.on_screen_dialogue[0]["start_s"] = 0.5

    with pytest.raises(
        h3.ReceiptError,
        match="on_screen_dialogue_receipt_mismatch",
    ):
        h3.inspect(request)

    assert not (request.workdir / ".h3").exists()


def test_factory_authority_validator_rechecks_window_not_only_timing_hash(
    tmp_path,
):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    dialogue_args = _dialogue_args("我会准时回来。")
    audios = _audios(tmp_path)
    request = h3_multimodal.build_h3_request(
        skill_plan=plan,
        approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
        **dialogue_args,
        visual=visual,
        reference_audios=audios,
        mode="multimodal",
        cid="factory-ignored-window",
        workdir=tmp_path / "factory-ignored-window",
        client_request_id="factory-ignored-window",
        duration=8,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="token",
    )
    restrictive = dialogue_timing.FrozenSpeakerTiming(
        sha256="d" * 64,
        source_sha256="a" * 64,
        duration=Fraction(8),
        windows={
            "S1": (
                dialogue_timing.FrozenLipWindow(
                    Fraction("0.75"), Fraction(8)
                ),
            )
        },
    )
    forged = replace(request, speaker_timing_sha256=restrictive.sha256)

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="dialogue_before_speaker_lip_window",
    ):
        h3_multimodal.validate_h3_request_authority(
            forged,
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            upstream_dialogue=dialogue_args["upstream_dialogue"],
            upstream_dialogue_receipt_sha256=(
                dialogue_args["upstream_dialogue_receipt_sha256"]
            ),
            visual=visual,
            reference_audios=audios,
            speaker_timing=restrictive,
        )


def test_on_screen_speaker_requires_subject_voice_reference(tmp_path):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    plan["subjects"][0]["voice_ref"] = None
    plan["audio_refs"] = [
        {"audio_index": 1, "purpose": "ambience", "subject_id": None},
    ]

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="dialogue_subject_invalid",
    ):
        h3_multimodal.build_h3_request(
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            **_dialogue_args("我会准时回来。"),
            visual=visual,
            reference_audios=_audios(tmp_path, purposes=("ambience",)),
            mode="multimodal",
            cid="missing-speaker-voice",
            workdir=tmp_path / "missing-speaker-voice",
            client_request_id="missing-speaker-voice",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )

    assert not (tmp_path / "missing-speaker-voice" / ".h3").exists()


@pytest.mark.parametrize("voice_ref", [2, 99])
def test_speaker_voice_reference_must_resolve_to_frozen_voice_audio(
    tmp_path, voice_ref,
):
    visual = _visual(tmp_path)
    plan = _plan(visual.prompt)
    plan["subjects"][0]["voice_ref"] = voice_ref
    plan["audio_refs"] = [
        {"audio_index": 1, "purpose": "ambience", "subject_id": None},
    ]
    plan["sound_design"]["ambience_refs"] = [
        {"audio_index": 1, "description": "远处雨声"},
    ]

    with pytest.raises(
        h3_multimodal.MultimodalContractError,
        match="voice_reference_unbound",
    ):
        h3_multimodal.build_h3_request(
            skill_plan=plan,
            approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
            **_dialogue_args("我会准时回来。"),
            visual=visual,
            reference_audios=_audios(tmp_path, purposes=("ambience",)),
            mode="multimodal",
            cid=f"invalid-voice-{voice_ref}",
            workdir=tmp_path / f"invalid-voice-{voice_ref}",
            client_request_id=f"invalid-voice-{voice_ref}",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )

    assert not (tmp_path / f"invalid-voice-{voice_ref}" / ".h3").exists()


@pytest.mark.parametrize("count", [0, 4])
def test_reference_audio_count_fails_before_attempt_claim(tmp_path, count):
    sources = []
    for index in range(count):
        path = tmp_path / f"{index}.wav"
        _write_wav(path)
        sources.append((path, "ambience"))
    with pytest.raises(h3.H3Error, match="invalid_reference_audio_count"):
        h3.freeze_reference_audios(tuple(sources))


def test_reference_audio_duration_and_format_fail_before_attempt_claim(tmp_path):
    short = tmp_path / "short.wav"
    _write_wav(short, seconds=1)
    with pytest.raises(h3.H3Error, match="invalid_reference_audio_duration"):
        h3.freeze_reference_audios(((short, "voice"),))

    wrong = tmp_path / "wrong.aac"
    wrong.write_bytes(b"not-a-reference-audio")
    with pytest.raises(h3.H3Error, match="invalid_reference_audio_format"):
        h3.freeze_reference_audios(((wrong, "voice"),))


def test_skill_consumer_compiles_exact_context_ir_and_rejects_old_blockers(tmp_path):
    visual = _visual(tmp_path)
    audios = _audios(tmp_path)
    plan = _plan(visual.prompt)
    request = _request(tmp_path)

    assert "<Picture 1>" in request.prompt
    assert "<Audio 1>" in request.prompt
    assert "<Subject 1>(S1)" in request.prompt
    assert "<d>[Chinese]我会准时回来。</d>" in request.prompt

    silent = copy.deepcopy(plan)
    silent["subjects"].append(
        {"subject_id": "S2", "picture_refs": [2], "voice_ref": None}
    )
    silent_request = h3_multimodal.build_h3_request(
        skill_plan=silent,
        approved_skill_plan_sha256=h3.canonical_json_sha256(silent),
        **_dialogue_args("我会准时回来。"),
        visual=visual,
        reference_audios=audios,
        mode="multimodal",
        cid="cid",
        workdir=tmp_path / "silent",
        client_request_id="silent",
        duration=8,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="token",
    )
    assert "this visible subject remains silent" in silent_request.prompt

    reused = copy.deepcopy(plan)
    reused["subjects"].append(
        {"subject_id": "S2", "picture_refs": [1], "voice_ref": 2}
    )
    reused["audio_refs"][1] = {
        "audio_index": 2,
        "purpose": "voice",
        "subject_id": "S2",
    }
    reused["speech_bindings"].append(
        {
            "line_index": 2,
            "delivery": "on_screen",
            "subject_id": "S2",
            "language": "English",
            "voice_ref": None,
        }
    )
    reused["dialogue_source_sha256"] = h3.canonical_json_sha256(
        list(_dialogue("我会准时回来。", "I am here."))
    )
    reused["sound_design"]["ambience_refs"] = []
    with pytest.raises(h3_multimodal.MultimodalContractError, match="picture_reused"):
        h3_multimodal.build_h3_request(
            skill_plan=reused,
            approved_skill_plan_sha256=h3.canonical_json_sha256(reused),
            **_dialogue_args("我会准时回来。", "I am here."),
            visual=visual,
            reference_audios=_audios(tmp_path, purposes=("voice", "voice")),
            mode="multimodal",
            cid="cid",
            workdir=tmp_path / "reused",
            client_request_id="reused",
            duration=8,
            resolution="768p",
            aspect_ratio="9:16",
            autodl_token="token",
        )


def test_same_confirmed_voice_ref_can_continue_as_offscreen_narration(tmp_path):
    visual = _visual(tmp_path)
    third = tmp_path / "03.png"
    third.write_bytes(b"frame-3")
    visual = h3_multimodal.FrozenVisualInput(
        visual.prompt,
        visual.keyframes + h3.freeze_keyframes((third,)),
    )
    audios = _audios(tmp_path, purposes=("voice",))
    plan = {
        "version": 2,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual.prompt,
        "dialogue_source_sha256": h3.canonical_json_sha256(
            list(_dialogue("我先在画内说话。", "随后仍用同一声线在画外说话。"))
        ),
        "subjects": [{"subject_id": "S1", "picture_refs": [1, 2, 3], "voice_ref": 1}],
        "audio_refs": [{"audio_index": 1, "purpose": "voice", "subject_id": "S1"}],
        "speech_bindings": [{
            "line_index": 1,
            "delivery": "on_screen",
            "subject_id": "S1",
            "language": "Chinese",
            "voice_ref": None,
        }, {
            "line_index": 2,
            "delivery": "off_screen_voiceover",
            "subject_id": None,
            "language": "Chinese",
            "voice_ref": 1,
        }],
        "sound_design": {
            "ambience_refs": [],
            "effects": [],
        },
    }

    request = h3_multimodal.build_h3_request(
        skill_plan=plan,
        approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
        **_dialogue_args("我先在画内说话。", "随后仍用同一声线在画外说话。"),
        visual=visual,
        reference_audios=audios,
        mode="multimodal",
        cid="av-a1",
        workdir=tmp_path / "session-a1",
        client_request_id="av-a1",
        duration=8,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="token",
    )

    assert request.voice_texts == (
        "我先在画内说话。",
        "随后仍用同一声线在画外说话。",
    )
    assert "off-screen narrator using <Audio 1>" in request.prompt


def test_real_submit_body_and_attempt_receipts_bind_all_multimodal_hashes(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(h3, "_require_context_ir_receipt", lambda _request: None)
    request = _request(tmp_path)
    posts = []
    for path, _blob in request.keyframes:
        path.write_bytes(b"changed-after-freeze")
    for audio in request.reference_audios:
        audio.path.write_bytes(b"changed-after-freeze")

    def handler(req: httpx.Request) -> httpx.Response:
        posts.append(req)
        return httpx.Response(201, json={"task_id": "audio-task"})

    assert h3.prepare(request).status == "not_started"
    with _client(handler) as client:
        assert h3.submit(request, client=client).status == "h3_running"

    assert len(posts) == 1
    assert posts[0].url == httpx.URL("http://127.0.0.1:31000/v1/videos")
    body = json.loads(posts[0].content)
    assert "audio_required" not in body
    assert set(body) == {
        "mode", "prompt", "duration_sec", "aspect_ratio", "resolution",
        "images", "audios",
    }
    assert body["mode"] == "multimodal"
    assert body["prompt"] == request.prompt
    assert body["duration_sec"] == 8
    assert body["aspect_ratio"] == "9:16"
    assert body["resolution"] == "768p"
    assert [item["kind"] for item in body["audios"]] == ["voice", "sound"]
    assert [Path(item["path"]).read_bytes() for item in body["audios"]] == [
        item.data for item in request.reference_audios
    ]
    assert [Path(path).read_bytes() for path in body["images"]] == [
        blob for _source, blob in request.keyframes
    ]

    state = _attempt(request)
    multimodal = state["input"]["multimodal"]
    assert multimodal["audio_required"] is True
    assert multimodal["skill_plan_sha256"] == request.skill_plan_sha256
    assert (
        multimodal["upstream_dialogue_receipt_sha256"]
        == request.upstream_dialogue_receipt_sha256
    )
    assert [item["sha256"] for item in multimodal["reference_audios"]] == [
        item.sha256 for item in request.reference_audios
    ]
    payload_sha = h3.canonical_json_sha256(body)
    assert state["input"]["request"]["payload_sha256"] == payload_sha
    assert state["h3"]["receipt"]["request"]["payload_sha256"] == payload_sha
    assert state["h3"]["receipt"]["input_receipt"] == state["input_receipt"]


def test_duplicate_submit_and_submission_unknown_never_repeat_audio_post(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(h3, "_require_context_ir_receipt", lambda _request: None)
    request = _request(tmp_path)
    calls = 0

    def accepted(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, json={"task_id": "audio-task"})

    h3.prepare(request)
    with _client(accepted) as client:
        assert h3.submit(request, client=client).status == "h3_running"
        assert h3.submit(request, client=client).status == "h3_running"
    assert calls == 1

    unknown = _request(tmp_path / "unknown")
    unknown_calls = 0

    def ambiguous(req: httpx.Request) -> httpx.Response:
        nonlocal unknown_calls
        unknown_calls += 1
        raise httpx.ReadTimeout("ambiguous", request=req)

    h3.prepare(unknown)
    with _client(ambiguous) as client:
        with pytest.raises(h3.H3Error, match="submission_unknown"):
            h3.submit(unknown, client=client)
    with _client(lambda _req: pytest.fail("resume must be GET-only")) as client:
        assert h3.resume(unknown, client=client).status == "submission_unknown"
    assert unknown_calls == 1


def test_audio_required_missing_output_audio_is_deterministic_and_get_only(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(h3, "_require_context_ir_receipt", lambda _request: None)
    request = _request(tmp_path)
    calls = []

    monkeypatch.setattr(h3, "_probe_media_timeline", lambda *_a, **_kw: {"audio": None})

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if req.method == "POST":
            return httpx.Response(201, json={"task_id": "audio-task"})
        if req.url.path == "/v1/videos/audio-task":
            return httpx.Response(200, json={"status": "succeeded"})
        return httpx.Response(
            200,
            content=b"video-without-audio",
        )

    with _client(handler) as client:
        with pytest.raises(h3.H3Error, match="output_audio_missing"):
            h3.start(request, client=client)

    state = _attempt(request)
    assert state["status"] == "failed"
    assert state["error"] == {"code": "output_audio_missing"}
    assert sum(call.method == "POST" for call in calls) == 1
    with _client(lambda _req: pytest.fail("deterministic output failure is GET-only")) as client:
        assert h3.resume(request, client=client).status == "failed"
