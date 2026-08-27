import base64
import hashlib
from dataclasses import replace
import pytest

from app import h3_multimodal as multimodal


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame(order: int = 1, *, data: bytes | None = None):
    payload = data or b"\x89PNG\r\n\x1a\nframe-" + str(order).encode("ascii")
    return multimodal.FrozenKeyframe(
        name=f"frame-{order}.png",
        order=order,
        data=payload,
        sha256=_sha(payload),
        width=768,
        height=1344,
    )


def _audio(
    order: int = 1,
    *,
    kind: str = "voice",
    data: bytes | None = None,
    duration_pts: int = 3000,
    time_base: int = 1000,
):
    payload = data or b"ID3\x04\x00\x00reference-audio-" + str(order).encode("ascii")
    return multimodal.FrozenReferenceAudio(
        order=order,
        role="conditioning_reference",
        kind=kind,
        data=payload,
        format="mp3",
        sha256=_sha(payload),
        decoded_sha256=_sha(b"decoded-" + payload),
        time_base=time_base,
        duration_pts=duration_pts,
    )


def _visual(*, skill_receipt_sha256: str | None = None):
    prompt = "雨夜车站，电影感中景，人物面对镜头，镜头缓慢推进。"
    return multimodal.FrozenVisualInput(
        prompt=prompt,
        prompt_sha256=_sha(prompt.encode("utf-8")),
        keyframes=(_frame(),),
        legacy_plan_sha256=_sha(b"legacy-plan"),
        legacy_meta_sha256=_sha(b"legacy-meta"),
        skill_receipt_sha256=skill_receipt_sha256,
    )


def _dialogue(*, subject_id: str = "S1", text: str = "我会准时回来。"):
    return (
        multimodal.DialogueLine(
            order=1,
            subject_id=subject_id,
            language="Chinese",
            text=text,
        ),
    )


def _subjects(*, subject_id: str = "S1"):
    return (
        multimodal.SubjectBinding(
            subject_id=subject_id,
            picture_refs=(1,),
            voice_ref=1,
        ),
    )


def _preview(**overrides):
    values = {
        "visual": _visual(),
        "audio_references": (_audio(),),
        "dialogue": _dialogue(),
        "subjects": _subjects(),
        "sound_design": multimodal.SoundDesign(),
        "mode": "multimodal",
        "duration": 8,
        "resolution": "768p",
        "aspect_ratio": "9:16",
        "plan_id": "plan-1",
        "revision": 3,
    }
    values.update(overrides)
    return multimodal.preview_multimodal_plan(**values)


def _request(**overrides):
    preview = overrides.pop("preview", _preview())
    values = {
        "plan": preview,
        "approved_plan_sha256": preview.plan_sha256,
        "approved_by": "upstream-review-1",
        "idempotency_key": "generation-1:revision-3",
    }
    values.update(overrides)
    return multimodal.freeze_multimodal_request(**values)


def test_compiler_freezes_visual_source_and_effective_h3_prompt():
    plan = _preview()

    assert plan.compiler_version == multimodal.COMPILER_VERSION
    assert plan.visual.prompt in plan.effective_prompt
    assert plan.effective_prompt != plan.visual.prompt
    assert "<Picture 1>" in plan.effective_prompt
    assert "<Audio 1>" in plan.effective_prompt
    assert "<Subject 1>(S1)" in plan.effective_prompt
    assert "<d>[Chinese]我会准时回来。</d>" in plan.effective_prompt
    assert plan.effective_prompt_sha256 == _sha(plan.effective_prompt.encode("utf-8"))
    assert plan.visual.prompt_sha256 == _sha(plan.visual.prompt.encode("utf-8"))
    assert plan.audio_required is True
    assert plan.reference_audio_semantics == "conditioning_only"
    assert plan.output_audio_authority == "h3_generated"


def test_subject_binding_supports_multiple_one_based_picture_references():
    visual = replace(_visual(), keyframes=(_frame(1), _frame(2)))
    subject = replace(_subjects()[0], picture_refs=(1, 2))

    plan = _preview(visual=visual, subjects=(subject,))

    definition = next(
        line for line in plan.effective_prompt.splitlines() if "<Subject 1>(S1)" in line
    )
    assert "<Picture 1>" in definition
    assert "<Picture 2>" in definition


@pytest.mark.parametrize(
    ("mode", "resolution", "duration", "workflow", "rate", "credits"),
    [
        (
            "multimodal",
            "480p",
            15,
            "minimax_h3_image_audio_to_video_v2_15s",
            6,
            90,
        ),
        (
            "multimodal",
            "768p",
            8,
            "minimax_h3_image_audio_to_video_v2_15s",
            8,
            64,
        ),
        (
            "multimodal_hd",
            "1080p",
            10,
            "minimax_h3_image_audio_to_video_v2",
            12,
            120,
        ),
    ],
)
def test_mode_selects_one_workflow_and_quote(
    mode, resolution, duration, workflow, rate, credits,
):
    plan = _preview(mode=mode, resolution=resolution, duration=duration)

    assert plan.workflow == workflow
    assert plan.quote.credits_per_second == rate
    assert plan.quote.credits == credits
    assert plan.quote.duration == duration


def test_build_autodl_request_uses_only_frozen_bytes_and_preserves_order(monkeypatch):
    first_frame = _frame(1, data=b"\x89PNG\r\n\x1a\nfirst")
    second_frame = _frame(2, data=b"\x89PNG\r\n\x1a\nsecond")
    first_audio = _audio(1, data=b"ID3-first")
    second_audio = _audio(
        2,
        kind="ambience",
        data=b"ID3-second",
        duration_pts=2500,
    )
    visual = replace(_visual(), keyframes=(first_frame, second_frame))
    request = _request(
        preview=_preview(
            visual=visual,
            audio_references=(first_audio, second_audio),
            sound_design=multimodal.SoundDesign(
                ambience_refs=(
                    multimodal.SoundReference(
                        audio_index=2,
                        description="远处雨声",
                    ),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        multimodal.Path,
        "read_bytes",
        lambda *_args, **_kwargs: pytest.fail("payload builder must not read paths"),
    )

    built = multimodal.build_autodl_request(request)

    assert built.workflow == "minimax_h3_image_audio_to_video_v2_15s"
    assert built.idempotency_key == request.idempotency_key
    assert built.request_sha256 == request.request_sha256
    assert built.body["prompt"] == request.plan.effective_prompt
    assert built.body["duration"] == 8
    assert built.body["resolution"] == "768p竖"
    assert built.body["audio_required"] is True
    assert built.body["ref_image_0"] == "data:image/png;base64," + base64.b64encode(
        first_frame.data
    ).decode("ascii")
    assert built.body["ref_image_1"].endswith(
        base64.b64encode(second_frame.data).decode("ascii")
    )
    assert built.body["ref_audio_0"] == "data:audio/mpeg;base64," + base64.b64encode(
        first_audio.data
    ).decode("ascii")
    assert built.body["ref_audio_1"].endswith(
        base64.b64encode(second_audio.data).decode("ascii")
    )
    assert "source_audio" not in built.body
    assert "target_audio" not in built.body


def test_skill_prompt_plan_consumer_matches_frozen_one_based_schema():
    visual = replace(_visual(), keyframes=(_frame(1), _frame(2)))
    audios = (_audio(1), _audio(2, kind="ambience", duration_pts=2500))
    skill_plan = {
        "version": 1,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual.prompt,
        "subjects": [
            {"subject_id": "S1", "picture_refs": [1, 2], "voice_ref": 1},
        ],
        "audio_refs": [
            {"audio_index": 1, "purpose": "voice", "subject_id": "S1"},
            {"audio_index": 2, "purpose": "ambience", "subject_id": None},
        ],
        "dialogue": [
            {
                "order": 1,
                "subject_id": "S1",
                "language": "Chinese",
                "text": "我会准时回来。",
            },
        ],
        "sound_design": {
            "narration": [
                {
                    "order": 2,
                    "language": "Chinese",
                    "text": "列车驶入雨幕。",
                    "voice_ref": None,
                },
            ],
            "ambience_refs": [
                {"audio_index": 2, "description": "远处雨声"},
            ],
            "effects": [],
        },
    }

    semantic = multimodal.consume_skill_prompt_plan(
        skill_plan,
        visual=visual,
        audio_references=audios,
    )

    assert semantic.subjects[0].picture_refs == (1, 2)
    assert semantic.dialogue[0].order == 1
    assert semantic.sound_design.narration[0].order == 2
    assert semantic.sound_design.narration[0].voice_ref is None
    plan = _preview(
        visual=visual,
        audio_references=semantic.audio_references,
        dialogue=semantic.dialogue,
        subjects=semantic.subjects,
        sound_design=semantic.sound_design,
    )
    assert "<d>[Chinese]列车驶入雨幕。</d>" in plan.effective_prompt
    assert "off-screen narrator" in plan.effective_prompt


def test_skill_prompt_plan_is_optional_but_if_supplied_is_strict():
    plan = {
        "version": 1,
        "phase": "multimodal_audio",
        "eligible": False,
        "reason": "speaker_mapping_unverified",
        "visual_prompt": "",
        "subjects": [],
        "audio_refs": [],
        "dialogue": [],
        "sound_design": {"narration": [], "ambience_refs": [], "effects": []},
    }
    with pytest.raises(multimodal.H3MultimodalContractError, match="skill_plan_ineligible"):
        multimodal.consume_skill_prompt_plan(
            plan,
            visual=_visual(),
            audio_references=(_audio(),),
        )

    malformed = dict(plan, eligible=True, reason=None, visual_prompt=_visual().prompt)
    malformed["unexpected"] = True
    with pytest.raises(multimodal.H3MultimodalContractError, match="skill_plan_shape"):
        multimodal.consume_skill_prompt_plan(
            malformed,
            visual=_visual(),
            audio_references=(_audio(),),
        )


def test_skill_consumer_maps_effect_and_preserves_silent_subject():
    plan = {
        "version": 1,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": _visual().prompt,
        "subjects": [
            {"subject_id": "S1", "picture_refs": [1], "voice_ref": None},
        ],
        "audio_refs": [
            {"audio_index": 1, "purpose": "effect", "subject_id": None},
        ],
        "dialogue": [],
        "sound_design": {
            "narration": [],
            "ambience_refs": [],
            "effects": [{"audio_index": 1, "description": "木门轻响"}],
        },
    }

    semantic = multimodal.consume_skill_prompt_plan(
        plan,
        visual=_visual(),
        audio_references=(_audio(kind="sound_effect"),),
    )
    compiled = _preview(
        audio_references=semantic.audio_references,
        dialogue=semantic.dialogue,
        subjects=semantic.subjects,
        sound_design=semantic.sound_design,
    ).effective_prompt

    assert semantic.subjects[0].voice_ref is None
    assert "silent visual subject" in compiled
    assert "sound-effect conditioning reference" in compiled


@pytest.mark.parametrize("count", [0, 4])
def test_rejects_zero_or_more_than_three_reference_audios(count):
    audios = tuple(_audio(index) for index in range(1, count + 1))

    with pytest.raises(multimodal.H3MultimodalContractError, match="audio_count"):
        _preview(audio_references=audios, dialogue=(), subjects=())


@pytest.mark.parametrize(
    "audio",
    [
        pytest.param(
            lambda: replace(_audio(), format="aac"),
            id="unsupported-format",
        ),
        pytest.param(
            lambda: replace(
                _audio(),
                data=b"not-an-mp3",
                sha256=_sha(b"not-an-mp3"),
            ),
            id="invalid-container",
        ),
    ],
)
def test_rejects_non_mp3_wav_or_invalid_container(audio):
    with pytest.raises(multimodal.H3MultimodalContractError, match="audio_format"):
        audio()


@pytest.mark.parametrize(
    "audios",
    [
        pytest.param((_audio(2),), id="starts-out-of-order"),
        pytest.param((_audio(1), _audio(1)), id="duplicate-order"),
        pytest.param((_audio(1), _audio(2, data=_audio(1).data)), id="duplicate-bytes"),
    ],
)
def test_rejects_duplicate_or_out_of_order_audio(audios):
    with pytest.raises(
        multimodal.H3MultimodalContractError,
        match="audio_(order|duplicate)",
    ):
        _preview(audio_references=audios, dialogue=(), subjects=())


def test_rejects_out_of_order_or_duplicate_keyframes():
    with pytest.raises(multimodal.H3MultimodalContractError, match="keyframe_order"):
        replace(_visual(), keyframes=(_frame(2),))

    duplicate = _frame(1)
    with pytest.raises(multimodal.H3MultimodalContractError, match="keyframe_duplicate"):
        replace(_visual(), keyframes=(duplicate, replace(duplicate, order=2, name="two.png")))


@pytest.mark.parametrize("kind", ["keyframe", "audio", "decoded_audio"])
def test_rejects_declared_hash_drift(kind):
    if kind == "keyframe":
        with pytest.raises(multimodal.H3MultimodalContractError, match="keyframe_hash"):
            replace(_frame(), sha256="0" * 64)
    elif kind == "audio":
        with pytest.raises(multimodal.H3MultimodalContractError, match="audio_hash"):
            replace(_audio(), sha256="0" * 64)
    else:
        with pytest.raises(multimodal.H3MultimodalContractError, match="decoded_audio_hash"):
            replace(_audio(), decoded_sha256="not-a-sha256")


@pytest.mark.parametrize(
    ("mode", "resolution", "duration"),
    [
        ("multimodal", "768p", 3),
        ("multimodal", "768p", 16),
        ("multimodal", "1080p", 8),
        ("multimodal_hd", "1080p", 11),
        ("multimodal_hd", "768p", 8),
        ("unknown", "768p", 8),
    ],
)
def test_rejects_video_duration_resolution_or_mode_outside_workflow_contract(
    mode, resolution, duration,
):
    with pytest.raises(
        multimodal.H3MultimodalContractError,
        match="(mode|duration|resolution)",
    ):
        _preview(mode=mode, resolution=resolution, duration=duration)


@pytest.mark.parametrize(
    "audios",
    [
        (_audio(duration_pts=1999),),
        (_audio(duration_pts=15001),),
        (_audio(1, duration_pts=8000), _audio(2, kind="ambience", duration_pts=7001)),
    ],
)
def test_rejects_reference_audio_duration_outside_provider_bounds(audios):
    with pytest.raises(multimodal.H3MultimodalContractError, match="audio_duration"):
        _preview(audio_references=audios, dialogue=(), subjects=())


def test_rejects_ambiguous_unbound_or_non_linear_speaker_plans():
    second_dialogue = multimodal.DialogueLine(
        order=2,
        subject_id="S2",
        language="English",
        text="I am already here.",
    )
    second_subject = multimodal.SubjectBinding(
        subject_id="S2",
        picture_refs=(1,),
        voice_ref=1,
    )

    with pytest.raises(multimodal.H3MultimodalContractError, match="dialogue_order"):
        _preview(dialogue=_dialogue() + (replace(second_dialogue, order=1),))

    with pytest.raises(multimodal.H3MultimodalContractError, match="speaker_unbound"):
        _preview(dialogue=_dialogue(subject_id="S2"))

    with pytest.raises(multimodal.H3MultimodalContractError, match="speaker_ambiguous"):
        _preview(
            visual=replace(_visual(), keyframes=(_frame(1), _frame(2))),
            audio_references=(_audio(1), _audio(2)),
            dialogue=_dialogue() + (second_dialogue,),
            subjects=_subjects() + (second_subject,),
        )


def test_voice_reference_must_be_bound_but_sound_only_plan_may_have_no_dialogue():
    with pytest.raises(multimodal.H3MultimodalContractError, match="voice_reference_unbound"):
        _preview(dialogue=(), subjects=())

    sound = _audio(kind="ambience")
    plan = _preview(
        audio_references=(sound,),
        dialogue=(),
        subjects=(),
        sound_design=multimodal.SoundDesign(
            ambience_refs=(
                multimodal.SoundReference(audio_index=1, description="远处雨声"),
            ),
        ),
    )
    assert "overall_soundscape:" in plan.effective_prompt
    assert "远处雨声" in plan.effective_prompt


def test_sound_design_is_strict_and_cannot_leave_audio_unbound():
    ambience = _audio(kind="ambience")
    with pytest.raises(multimodal.H3MultimodalContractError, match="sound_reference_unbound"):
        _preview(audio_references=(ambience,), dialogue=(), subjects=())

    with pytest.raises(multimodal.H3MultimodalContractError, match="sound_reference_kind"):
        _preview(
            audio_references=(ambience,),
            dialogue=(),
            subjects=(),
            sound_design=multimodal.SoundDesign(
                effects=(
                    multimodal.SoundReference(
                        audio_index=1,
                        description="木门轻响",
                    ),
                ),
            ),
        )


def test_compiler_detects_prompt_or_script_tampering_before_payload_build():
    request = _request()
    changed_line = replace(request.plan.dialogue[0], text="另一句未批准台词。")
    changed_plan = replace(request.plan, dialogue=(changed_line,))
    changed_request = replace(request, plan=changed_plan)

    with pytest.raises(multimodal.H3MultimodalContractError, match="prompt_script_mismatch"):
        multimodal.build_autodl_request(changed_request)


def test_requires_exact_approval_revision_and_idempotency():
    plan = _preview()
    with pytest.raises(multimodal.H3MultimodalContractError, match="approval_mismatch"):
        multimodal.freeze_multimodal_request(
            plan=plan,
            approved_plan_sha256="0" * 64,
            approved_by="reviewer",
            idempotency_key="request-1",
        )
    with pytest.raises(multimodal.H3MultimodalContractError, match="invalid_revision"):
        _preview(revision=0)
    with pytest.raises(multimodal.H3MultimodalContractError, match="invalid_idempotency"):
        multimodal.freeze_multimodal_request(
            plan=plan,
            approved_plan_sha256=plan.plan_sha256,
            approved_by="reviewer",
            idempotency_key="",
        )


def test_skill_receipt_is_optional_provenance_not_an_eligibility_gate():
    without_skill = _preview(visual=_visual(skill_receipt_sha256=None))
    with_skill = _preview(visual=_visual(skill_receipt_sha256=_sha(b"skill")))

    assert without_skill.visual.skill_receipt_sha256 is None
    assert with_skill.visual.skill_receipt_sha256 == _sha(b"skill")
    assert without_skill.workflow == with_skill.workflow


def test_only_conditioning_reference_is_accepted_and_h3_audio_is_authoritative():
    with pytest.raises(multimodal.H3MultimodalContractError, match="audio_role"):
        replace(_audio(), role="target_dialogue")

    request = _request()
    built = multimodal.build_autodl_request(request)
    assert request.plan.reference_audio_semantics == "conditioning_only"
    assert request.plan.output_audio_authority == "h3_generated"
    assert built.output_audio_authority == "h3_generated"
