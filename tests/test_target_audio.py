"""Route-neutral target-audio plan and paid-safe materialization contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from app import target_audio


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeProbe:
    def __init__(self) -> None:
        self.values: dict[Path, target_audio.AudioProbe] = {}

    def add(
        self,
        path: Path,
        *,
        duration_pts: int,
        time_base: int = 1_000,
        decoded: str | None = None,
    ) -> None:
        self.values[path.resolve()] = target_audio.AudioProbe(
            format=path.suffix.removeprefix("."),
            time_base=time_base,
            duration_pts=duration_pts,
            sample_rate_hz=48_000,
            channels=1,
            decoded_sha256=decoded or _sha(b"pcm:" + path.name.encode()),
        )

    def __call__(self, path: Path) -> target_audio.AudioProbe:
        return self.values[path.resolve()]


class FakeMaterializer:
    def __init__(
        self,
        root: Path,
        probe: FakeProbe,
        *,
        submit_results: list[target_audio.MaterializationResult | Exception],
        get_results: list[target_audio.MaterializationResult | Exception] | None = None,
    ) -> None:
        self.root = root
        self.probe = probe
        self.submit_results = list(submit_results)
        self.get_results = list(get_results or [])
        self.submit_calls: list[target_audio.MaterializationRequest] = []
        self.get_calls: list[tuple[str, target_audio.MaterializationRequest]] = []
        self.receipts_at_submit: list[dict] = []
        self.receipt_path = root / target_audio.RECEIPT_FILENAME

    def submit(
        self, request: target_audio.MaterializationRequest
    ) -> target_audio.MaterializationResult:
        self.submit_calls.append(request)
        self.receipts_at_submit.append(
            json.loads(self.receipt_path.read_text())
        )
        result = self.submit_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get(
        self, task_id: str, request: target_audio.MaterializationRequest
    ) -> target_audio.MaterializationResult:
        self.get_calls.append((task_id, request))
        result = self.get_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _voice_strategy(reference_id: str = "voice-alice") -> target_audio.VoiceStrategy:
    return target_audio.VoiceStrategy(
        kind="voice_reference", voice_reference_id=reference_id
    )


def _request(root: Path, probe: FakeProbe) -> target_audio.TargetAudioRequest:
    source = root / "source.wav"
    voice_ref = root / "work/audio/voice-alice.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    voice_ref.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"optional-source-audio")
    voice_ref.write_bytes(b"clean-alice-reference")
    probe.add(source, duration_pts=20_000)
    probe.add(voice_ref, duration_pts=4_000)
    return target_audio.TargetAudioRequest(
        client_request_id="audio-request-001",
        mode="translate",
        source_audio=source,
        dialogue=(
            target_audio.DialogueLine(
                line_id="line-1",
                order=0,
                speaker_id="alice",
                language="en-US",
                time_base=1_000,
                start_pts=0,
                end_pts=2_000,
                text="Hello world.",
            ),
            target_audio.DialogueLine(
                line_id="line-2",
                order=1,
                speaker_id="alice",
                language="en-US",
                time_base=1_000,
                start_pts=2_000,
                end_pts=4_000,
                text="This is exact dialogue.",
            ),
        ),
        speaker_plan=(
            target_audio.SpeakerPlan(
                speaker_id="alice",
                language="en-US",
                voice_strategy=_voice_strategy(),
                kind="subject",
                subject_id="S1",
            ),
        ),
        audio_refs=(
            target_audio.AudioReference(
                reference_id="voice-alice",
                order=0,
                speaker_id="alice",
                path=voice_ref,
            ),
        ),
        effects=(
            target_audio.AudioCue(
                cue_id="room-tone",
                order=0,
                role="ambience",
                time_base=1_000,
                start_pts=0,
                end_pts=4_000,
                prompt="quiet office room tone",
            ),
        ),
        target_materials=(
            target_audio.TargetMaterialSpec(
                material_id="dialogue", order=0, role="dialogue",
                format="wav", time_base=1_000, duration_pts=4_000,
            ),
            target_audio.TargetMaterialSpec(
                material_id="ambience", order=1, role="ambience",
                format="mp3", time_base=1_000, duration_pts=4_000,
            ),
        ),
        speaker_mapping_verified=True,
        speaker_mapping_source="manual-review",
        materializer=target_audio.MaterializerSpec(
            provider="fake-tts-mixer",
            model="audio-model",
            version="2026-08-27",
            prompt="Translate exactly; preserve the approved timing.",
            cost=target_audio.CostEstimate(currency="USD", amount_micros=125_000),
        ),
    )


def _output(
    root: Path,
    probe: FakeProbe,
    material_id: str,
    filename: str,
    payload: bytes,
) -> target_audio.MaterializedOutput:
    path = root / "work/target_audio" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    probe.add(path, duration_pts=4_000)
    info = probe(path)
    return target_audio.MaterializedOutput(
        material_id=material_id,
        path=path,
        format=info.format,
        time_base=info.time_base,
        duration_pts=info.duration_pts,
        size_bytes=len(payload),
        sha256=_sha(payload),
        decoded_sha256=info.decoded_sha256,
    )


def test_freeze_is_route_neutral_and_source_is_not_a_target_material(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)

    plan = target_audio.freeze_target_audio_plan(root, request, probe=probe)

    assert plan.mode == "translate"
    assert plan.source_audio is not None
    assert plan.source_audio.data == b"optional-source-audio"
    assert [item.reference_id for item in plan.audio_refs] == ["voice-alice"]
    assert plan.audio_refs[0].data == b"clean-alice-reference"
    assert plan.target_materials == ()
    assert plan.materialization_status == "ready_to_submit"
    assert plan.dialogue[0].voice_strategy == _voice_strategy()
    assert plan.dialogue[0].end_pts == plan.dialogue[1].start_pts
    assert plan.speaker_plan[0].speaker_id == "alice"
    assert plan.speaker_plan[0].subject_id == "S1"

    receipt = json.loads(plan.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "duet.target-audio-plan"
    assert receipt["version"] == 1
    assert "visual" not in json.dumps(receipt)
    assert receipt["inputs"]["source_audio"]["path"] == "source.wav"
    assert receipt["inputs"]["source_audio"]["size_bytes"] == len(
        b"optional-source-audio"
    )
    assert receipt["inputs"]["audio_refs"][0]["order"] == 0
    assert receipt["inputs"]["audio_refs"][0]["purpose"] == "voice"
    assert receipt["speaker_map"]["speakers"][0]["subject_id"] == "S1"
    assert receipt["script"]["lines"][0]["voice_strategy"] == {
        "kind": "voice_reference",
        "voice_reference_id": "voice-alice",
        "target_voice": None,
    }
    assert receipt["materializer"]["model"] == "audio-model"
    assert receipt["materializer"]["version"] == "2026-08-27"
    assert receipt["materializer"]["prompt"] == (
        "Translate exactly; preserve the approved timing."
    )
    assert receipt["materializer"]["cost"] == {
        "amount_micros": 125_000,
        "currency": "USD",
    }
    assert len(receipt["plan_receipt"]) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing_speaker", "speaker"),
        ("overlap", "overlap"),
        ("empty_language", "language"),
        ("empty_text", "text"),
        ("unverified", "verified"),
        ("coarse_asr", "ASR"),
    ],
)
def test_invalid_dialogue_and_speaker_mapping_fail_closed(tmp_path, change, message):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    if change == "missing_speaker":
        request = replace(request, speaker_plan=())
    elif change == "overlap":
        lines = list(request.dialogue)
        lines[1] = replace(lines[1], start_pts=1_999)
        request = replace(request, dialogue=tuple(lines))
    elif change == "empty_language":
        lines = list(request.dialogue)
        lines[0] = replace(lines[0], language=" ")
        request = replace(request, dialogue=tuple(lines))
    elif change == "empty_text":
        lines = list(request.dialogue)
        lines[0] = replace(lines[0], text=" ")
        request = replace(request, dialogue=tuple(lines))
    elif change == "unverified":
        request = replace(request, speaker_mapping_verified=False)
    else:
        request = replace(request, speaker_mapping_source="asr_coarse")

    with pytest.raises(target_audio.TargetAudioError, match=message):
        target_audio.freeze_target_audio_plan(root, request, probe=probe)


def test_target_voice_strategy_needs_no_reference_but_is_explicit(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    strategy = target_audio.VoiceStrategy(
        kind="target_voice", target_voice="warm-neutral-v2"
    )
    request = replace(
        request,
        speaker_plan=(replace(request.speaker_plan[0], voice_strategy=strategy),),
        audio_refs=(),
        source_audio=None,
    )

    plan = target_audio.freeze_target_audio_plan(root, request, probe=probe)

    assert plan.source_audio is None
    assert plan.audio_refs == ()
    assert all(line.voice_strategy == strategy for line in plan.dialogue)


@pytest.mark.parametrize(
    ("durations", "message"),
    [
        ([1_999], "2..15"),
        ([15_001], "2..15"),
        ([8_000, 8_000], "total"),
    ],
)
def test_h3_voice_reference_duration_contract(tmp_path, durations, message):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    refs = []
    speakers = []
    lines = []
    for index, duration in enumerate(durations):
        speaker_id = f"speaker-{index}"
        reference_id = f"voice-{index}"
        path = root / f"work/audio/{reference_id}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(reference_id.encode())
        probe.add(path, duration_pts=duration)
        refs.append(target_audio.AudioReference(reference_id, index, speaker_id, path))
        speakers.append(
            target_audio.SpeakerPlan(
                speaker_id, "en", _voice_strategy(reference_id),
                "subject", f"S{index + 1}",
            )
        )
        lines.append(
            target_audio.DialogueLine(
                f"line-{index}", index, speaker_id, "en", 1_000,
                index * 2_000, (index + 1) * 2_000, "text",
            )
        )
    request = replace(
        request,
        audio_refs=tuple(refs),
        speaker_plan=tuple(speakers),
        dialogue=tuple(lines),
    )

    with pytest.raises(target_audio.TargetAudioError, match=message):
        target_audio.freeze_target_audio_plan(root, request, probe=probe)


def test_h3_accepts_at_most_three_ordered_voice_references(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    refs = tuple(
        target_audio.AudioReference("voice-alice", index, "alice", request.audio_refs[0].path)
        for index in range(4)
    )
    request = replace(request, audio_refs=refs)
    with pytest.raises(target_audio.TargetAudioError, match="at most 3"):
        target_audio.freeze_target_audio_plan(root, request, probe=probe)


def test_plan_can_project_narration_and_nonvoice_h3_references(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    ambience = root / "work/audio/office.wav"
    ambience.write_bytes(b"clean-office-ambience")
    probe.add(ambience, duration_pts=3_000)
    speakers = request.speaker_plan + (
        target_audio.SpeakerPlan(
            "narrator",
            "en-US",
            target_audio.VoiceStrategy(
                kind="target_voice", target_voice="neutral-narrator"
            ),
            kind="narrator",
            subject_id=None,
        ),
    )
    dialogue = request.dialogue + (
        target_audio.DialogueLine(
            "line-3", 2, "narrator", "en-US", 1_000, 4_000, 5_000,
            "Narration is not a visual subject.",
        ),
    )
    refs = request.audio_refs + (
        target_audio.AudioReference(
            "office-ambience",
            1,
            None,
            ambience,
            purpose="ambience",
            description="quiet office room tone",
        ),
    )

    plan = target_audio.freeze_target_audio_plan(
        root,
        replace(
            request,
            speaker_plan=speakers,
            dialogue=dialogue,
            audio_refs=refs,
            target_materials=tuple(
                replace(item, duration_pts=5_000)
                for item in request.target_materials
            ),
        ),
        probe=probe,
    )

    assert plan.speaker_plan[0].kind == "subject"
    assert plan.speaker_plan[0].subject_id == "S1"
    assert plan.speaker_plan[1].kind == "narrator"
    assert plan.speaker_plan[1].subject_id is None
    assert plan.audio_refs[1].purpose == "ambience"
    assert plan.audio_refs[1].speaker_id is None
    assert plan.audio_refs[1].description == "quiet office room tone"


def test_nonvoice_reference_cannot_impersonate_a_speaker(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    broken = replace(request.audio_refs[0], purpose="ambience")
    with pytest.raises(target_audio.TargetAudioError, match="must not bind a speaker"):
        target_audio.freeze_target_audio_plan(
            root, replace(request, audio_refs=(broken,)), probe=probe
        )


def test_materialization_success_freezes_order_bytes_hash_and_exact_duration(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    dialogue = _output(root, probe, "dialogue", "dialogue.wav", b"dialogue-track")
    ambience = _output(root, probe, "ambience", "ambience.mp3", b"ambience-track")
    materializer = FakeMaterializer(
        root,
        probe,
        submit_results=[
            target_audio.MaterializationResult(
                status="succeeded", task_id="paid-task-1",
                outputs=(dialogue, ambience),
            )
        ],
    )

    plan = target_audio.materialize_target_audio_plan(
        root, request, materializer=materializer, probe=probe
    )

    assert len(materializer.submit_calls) == 1
    assert materializer.receipts_at_submit[0]["materialization"] == {
        "status": "submitting",
        "task_id": None,
        "outputs": [],
        "error": None,
    }
    assert materializer.receipts_at_submit[0]["plan_receipt"] == plan.plan_receipt
    frozen_request = materializer.submit_calls[0]
    assert frozen_request.source_audio.data == b"optional-source-audio"
    assert frozen_request.audio_refs[0].data == b"clean-alice-reference"
    assert frozen_request.materializer.model == "audio-model"
    assert [item.material_id for item in plan.target_materials] == [
        "dialogue", "ambience"
    ]
    assert [item.order for item in plan.target_materials] == [0, 1]
    assert plan.target_materials[0].data == b"dialogue-track"
    assert plan.target_materials[0].sha256 == _sha(b"dialogue-track")
    assert plan.target_materials[0].probe.duration_pts == 4_000
    assert plan.materialization_status == "succeeded"
    assert plan.task_id == "paid-task-1"

    receipt = json.loads(plan.receipt_path.read_text(encoding="utf-8"))
    assert [
        (item["order"], item["role"], item["material_id"])
        for item in receipt["materialization"]["outputs"]
    ] == [(0, "dialogue", "dialogue"), (1, "ambience", "ambience")]
    assert receipt["materialization"]["outputs"][0]["probe"]["decoded_format"] == (
        "pcm_s16le"
    )

    loaded = target_audio.load_target_audio_plan(root, plan.receipt_path, probe=probe)
    assert loaded.target_materials == plan.target_materials
    assert loaded.dialogue == plan.dialogue


@pytest.mark.parametrize("damage", ["missing", "encoded_hash", "decoded_hash", "duration"])
def test_unmaterialized_or_mismatched_output_fails_closed(tmp_path, damage):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    output = _output(root, probe, "dialogue", "dialogue.wav", b"dialogue-track")
    if damage == "missing":
        outputs = ()
    elif damage == "encoded_hash":
        outputs = (replace(output, sha256="0" * 64),)
    elif damage == "decoded_hash":
        outputs = (replace(output, decoded_sha256="0" * 64),)
    else:
        outputs = (replace(output, duration_pts=3_999),)
    materializer = FakeMaterializer(
        root,
        probe,
        submit_results=[
            target_audio.MaterializationResult(
                status="succeeded", task_id="paid-task-1", outputs=outputs
            )
        ],
        get_results=[
            target_audio.MaterializationResult(
                status="succeeded", task_id="paid-task-1", outputs=outputs
            )
        ],
    )

    with pytest.raises(target_audio.TargetAudioError, match="material"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )

    receipt_path = root / target_audio.RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["materialization"]["status"] == "output_invalid"
    assert receipt["materialization"]["task_id"] == "paid-task-1"
    with pytest.raises(target_audio.TargetAudioError):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )
    assert len(materializer.submit_calls) == 1
    assert [task_id for task_id, _ in materializer.get_calls] == ["paid-task-1"]


def test_submission_unknown_is_persisted_and_never_reposted(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    materializer = FakeMaterializer(
        root, probe, submit_results=[TimeoutError("lost after POST")]
    )

    with pytest.raises(target_audio.TargetAudioError, match="submission_unknown"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )
    with pytest.raises(target_audio.TargetAudioError, match="submission_unknown"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )

    receipt = json.loads((root / target_audio.RECEIPT_FILENAME).read_text())
    assert receipt["materialization"]["status"] == "submission_unknown"
    assert receipt["materialization"]["task_id"] is None
    assert len(materializer.submit_calls) == 1


def test_provider_task_id_recovery_is_get_only(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    dialogue = _output(root, probe, "dialogue", "dialogue.wav", b"dialogue-track")
    ambience = _output(root, probe, "ambience", "ambience.mp3", b"ambience-track")
    materializer = FakeMaterializer(
        root,
        probe,
        submit_results=[
            target_audio.MaterializationResult(
                status="processing", task_id="paid-task-1"
            )
        ],
        get_results=[
            target_audio.MaterializationResult(
                status="succeeded", task_id="paid-task-1",
                outputs=(dialogue, ambience),
            )
        ],
    )

    with pytest.raises(target_audio.TargetAudioError, match="not_materialized"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )
    plan = target_audio.materialize_target_audio_plan(
        root, request, materializer=materializer, probe=probe
    )

    assert plan.materialization_status == "succeeded"
    assert len(materializer.submit_calls) == 1
    assert [task_id for task_id, _ in materializer.get_calls] == ["paid-task-1"]


def test_task_id_is_saved_before_invalid_provider_output_is_inspected(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    probe = FakeProbe()
    request = _request(root, probe)
    bad = target_audio.MaterializedOutput(
        material_id="dialogue",
        path=outside,
        format="wav",
        time_base=1_000,
        duration_pts=4_000,
        size_bytes=len(b"outside"),
        sha256=_sha(b"outside"),
        decoded_sha256=_sha(b"pcm-outside"),
    )
    materializer = FakeMaterializer(
        root,
        probe,
        submit_results=[
            target_audio.MaterializationResult(
                status="succeeded", task_id="paid-task-1", outputs=(bad,)
            )
        ],
        get_results=[
            target_audio.MaterializationResult(
                status="processing", task_id="paid-task-1"
            )
        ],
    )

    with pytest.raises(target_audio.TargetAudioError, match="material"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )
    receipt = json.loads((root / target_audio.RECEIPT_FILENAME).read_text())
    assert receipt["materialization"]["task_id"] == "paid-task-1"
    assert receipt["materialization"]["status"] == "output_invalid"

    with pytest.raises(target_audio.TargetAudioError, match="not_materialized"):
        target_audio.materialize_target_audio_plan(
            root, request, materializer=materializer, probe=probe
        )
    assert len(materializer.submit_calls) == 1
    assert [task_id for task_id, _ in materializer.get_calls] == ["paid-task-1"]


def test_nested_receipt_still_resolves_material_paths_from_project_root(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    dialogue = _output(root, probe, "dialogue", "dialogue.wav", b"dialogue-track")
    ambience = _output(root, probe, "ambience", "ambience.mp3", b"ambience-track")
    materializer = FakeMaterializer(
        root,
        probe,
        submit_results=[
            target_audio.MaterializationResult(
                "succeeded", "paid-task-1", (dialogue, ambience)
            )
        ],
    )
    receipt_path = root / "work/receipts/target_audio.json"
    materializer.receipt_path = receipt_path

    written = target_audio.materialize_target_audio_plan(
        root,
        request,
        materializer=materializer,
        probe=probe,
        receipt_path=receipt_path,
    )
    loaded = target_audio.load_target_audio_plan(root, receipt_path, probe=probe)

    assert written.receipt_path == receipt_path
    assert loaded.target_materials[0].relative_path == "work/target_audio/dialogue.wav"


def test_bound_input_and_receipt_tampering_fail_closed(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    plan = target_audio.freeze_target_audio_plan(root, request, probe=probe)
    request.audio_refs[0].path.write_bytes(b"changed")
    with pytest.raises(target_audio.TargetAudioError, match="sha256"):
        target_audio.load_target_audio_plan(root, plan.receipt_path, probe=probe)

    request.audio_refs[0].path.write_bytes(b"clean-alice-reference")
    receipt = json.loads(plan.receipt_path.read_text())
    receipt["script"]["lines"][0]["text"] = "tampered"
    plan.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(target_audio.TargetAudioError, match="plan receipt"):
        target_audio.load_target_audio_plan(root, plan.receipt_path, probe=probe)


def test_paths_are_project_relative_and_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"outside")
    probe = FakeProbe()
    request = _request(root, probe)
    link = root / "work/audio/linked.mp3"
    link.symlink_to(outside)
    probe.add(outside, duration_pts=4_000)
    request = replace(
        request,
        audio_refs=(replace(request.audio_refs[0], path=link),),
    )

    with pytest.raises(target_audio.TargetAudioError, match="symlink|escapes"):
        target_audio.freeze_target_audio_plan(root, request, probe=probe)


def test_existing_receipt_must_match_the_exact_frozen_request(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)
    target_audio.freeze_target_audio_plan(root, request, probe=probe)
    changed = replace(
        request,
        materializer=replace(request.materializer, prompt="a different prompt"),
    )

    with pytest.raises(target_audio.TargetAudioError, match="does not match"):
        target_audio.freeze_target_audio_plan(root, changed, probe=probe)


def test_script_and_cues_must_fit_unique_matching_target_tracks(tmp_path):
    root = tmp_path / "project"
    probe = FakeProbe()
    request = _request(root, probe)

    too_short = tuple(
        replace(item, duration_pts=3_999)
        if item.role == "dialogue" else item
        for item in request.target_materials
    )
    with pytest.raises(target_audio.TargetAudioError, match="dialogue PTS"):
        target_audio.freeze_target_audio_plan(
            root, replace(request, target_materials=too_short), probe=probe
        )

    duplicate_role = (
        request.target_materials[0],
        replace(
            request.target_materials[1],
            material_id="dialogue-2",
            role="dialogue",
        ),
    )
    with pytest.raises(target_audio.TargetAudioError, match="duplicate role"):
        target_audio.freeze_target_audio_plan(
            root, replace(request, target_materials=duplicate_role), probe=probe
        )
