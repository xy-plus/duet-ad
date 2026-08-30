import copy
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from app import context_ir_bridge, dialogue_timing, h3, h3_multimodal


UPSTREAM_DIALOGUE_SHA256 = hashlib.sha256(
    b"verified-prepared-or-long-dialogue-receipt"
).hexdigest()


def _audio(path: Path, order: int, purpose: h3.ReferenceAudioPurpose) -> h3.FrozenReferenceAudio:
    data = f"audio-{order}".encode()
    return h3.FrozenReferenceAudio(
        path=path,
        data=data,
        order=order,
        purpose=purpose,
        format="wav",
        sha256=hashlib.sha256(data).hexdigest(),
        duration_s=2.0,
    )


def _plan(visual_prompt: str) -> dict:
    return {
        "version": 2,
        "phase": "multimodal_audio",
        "eligible": True,
        "reason": None,
        "visual_prompt": visual_prompt,
        "dialogue_source_sha256": UPSTREAM_DIALOGUE_SHA256,
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
            {
                "line_index": 2,
                "delivery": "off_screen_voiceover",
                "subject_id": None,
                "language": "Chinese",
                "voice_ref": 1,
            },
        ],
        "sound_design": {
            "ambience_refs": [
                {"audio_index": 2, "description": "远处雨声"},
            ],
            "effects": [],
        },
    }


def _source_request(tmp_path: Path) -> tuple[h3.H3Request, dict]:
    visual_prompt = "雨夜车站，人物面对镜头，镜头缓慢推进。"
    keyframes = (
        (tmp_path / "01.png", b"frame-one"),
        (tmp_path / "02.png", b"frame-two"),
    )
    visual = h3_multimodal.FrozenVisualInput(visual_prompt, keyframes)
    audios = (
        _audio(tmp_path / "voice.wav", 1, "voice"),
        _audio(tmp_path / "rain.wav", 2, "ambience"),
    )
    plan = _plan(visual_prompt)
    dialogue = (
        {
            "text": "我会准时回来。",
            "start_s": 0.2,
            "end_s": 3.0,
            "classification": "spoken",
            "provenance": "asr",
        },
        {
            "text": "随后转为画外旁白。",
            "start_s": 3.2,
            "end_s": 7.8,
            "classification": "spoken",
            "provenance": "asr",
        },
    )
    request = h3_multimodal.build_h3_request(
        skill_plan=plan,
        approved_skill_plan_sha256=h3.canonical_json_sha256(plan),
        upstream_dialogue=dialogue,
        upstream_dialogue_receipt_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_dialogue_content_sha256=h3.canonical_json_sha256(list(dialogue)),
        speaker_timing=dialogue_timing.FrozenSpeakerTiming(
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
        visual=visual,
        reference_audios=audios,
        mode="multimodal",
        cid="segment-1",
        workdir=tmp_path / "segment-1",
        client_request_id="h3-request-1",
        duration=8,
        resolution="768p",
        aspect_ratio="9:16",
        autodl_token="autodl-secret",
    )
    return request, plan


def _upstream_artifact(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "prepared_input.json"
    data = json.dumps(
        {"dialogue": {"sha256": UPSTREAM_DIALOGUE_SHA256}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _frozen(tmp_path: Path) -> context_ir_bridge.FrozenContextIrRequest:
    source, plan = _source_request(tmp_path)
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    return context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_artifact_path=artifact_path,
        upstream_artifact_sha256=artifact_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1,
            poll_total_s=0.2,
            poll_interval_s=0,
        ),
    )


def _no_audio_frozen(
    tmp_path: Path, *, prompt: str | None = None,
) -> context_ir_bridge.FrozenContextIrRequest:
    prompt = prompt or "final fusion visual prompt; no dialogue and no source audio overlay"
    source = h3.H3Request(
        cid="silent-segment-1",
        workdir=tmp_path / "silent-segment-1",
        client_request_id="silent-h3-request-1",
        prompt=prompt,
        keyframes=((Path("01.png"), b"silent-frame-one"),),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=8,
        autodl_token="autodl-secret",
        workflow=h3.H3_WORKFLOW,
        skill_plan_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        upstream_dialogue_receipt_sha256=UPSTREAM_DIALOGUE_SHA256,
        context_ir_required=True,
    )
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    return context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_artifact_path=artifact_path,
        upstream_artifact_sha256=artifact_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        source_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1,
            poll_total_s=0.2,
            poll_interval_s=0,
        ),
    )


def _reference_fusion_frozen(
    tmp_path: Path, *, dialogue: bool = True,
) -> context_ir_bridge.FrozenContextIrRequest:
    """Freeze the current nine-frame Ref2VA/reference prompt shape."""
    base = _no_audio_frozen(tmp_path)
    line = (
        "From 00:01.000 to 00:02.000, the off-screen narrator says: "
        "<d>[Undetermined]这是严格冻结的台词</d> while all visible lips "
        "remain closed."
        if dialogue else
        "No person speaks in this visual interval."
    )
    prompt = "\n".join([
        "subject_definitions:",
        *[
            f"<Picture {order}> is the storyboard anchor for ordered visual state."
            for order in range(1, 10)
        ],
        "summary:",
        "The target follows all ordered storyboard anchors.",
        "retention_analysis:",
        "All ordered visual anchors are retained.",
        "detailed_description:",
        "A restrained visual interval preserves the source composition.",
        line,
        "overall_soundscape:",
        "The specified sound layer is the only audible layer.",
        "non_diegetic_music:",
        "N/A",
    ])
    keyframes = tuple(
        (tmp_path / f"fusion-{order:02d}.png", f"fusion-frame-{order}".encode())
        for order in range(1, 10)
    )
    voice_texts = ("这是严格冻结的台词",) if dialogue else ()
    source = replace(
        base.source_h3_request,
        cid="fusion-reference-segment-1" if dialogue else "fusion-silent-segment-1",
        client_request_id=(
            "fusion-reference-request-1"
            if dialogue else "fusion-silent-request-1"
        ),
        prompt=prompt,
        keyframes=keyframes,
        voice_texts=voice_texts,
        voice_receipt=h3.voice_texts_receipt(voice_texts),
        skill_plan_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    return context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_artifact_path=artifact_path,
        upstream_artifact_sha256=artifact_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        source_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1,
            poll_total_s=0.2,
            poll_interval_s=0,
        ),
    )


def _timeline_frozen(tmp_path: Path) -> context_ir_bridge.FrozenContextIrRequest:
    base = _no_audio_frozen(tmp_path)
    timeline = []
    for order, segment_time_s in enumerate(
        [0.0, 1.0, 2.0, 2.5, 4.0, 6.0, 8.0, 11.0, 14.0], 1
    ):
        timeline.append({
            "order": order,
            "segment_time_s": segment_time_s,
            "source_scene_id": "SCENE_01" if order < 4 else "SCENE_02",
            "transition": (
                {"type": "start", "at_segment_s": 0.0}
                if order == 1 else
                {"type": "hard_cut", "at_segment_s": 2.267}
                if order == 4 else
                {"type": "continuous", "at_segment_s": None}
            ),
        })
    timeline_json = json.dumps(
        timeline,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    prompt = (
        "<VISUAL>\nsource visual\n</VISUAL>\n"
        + "<KEYFRAME_TIMELINE_JSON>"
        + timeline_json
        + "</KEYFRAME_TIMELINE_JSON>\n"
        + "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
        + "<MUSIC_POLICY>\nnon_diegetic_music: N/A\n</MUSIC_POLICY>"
    )
    source = replace(
        base.source_h3_request,
        prompt=prompt,
        skill_plan_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    return context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=base.upstream_dialogue_sha256,
        upstream_artifact_path=base.upstream_artifact_path,
        upstream_artifact_sha256=base.upstream_artifact_sha256,
        upstream_dialogue_sha256_path=base.upstream_dialogue_sha256_path,
        source_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=base.timeouts,
    )


def test_current_fusion_no_bgm_suffix_is_bound_and_must_survive_context(
    tmp_path: Path,
) -> None:
    suffix = (
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>\nnon_diegetic_music: N/A\n</MUSIC_POLICY>"
    )
    frozen = _no_audio_frozen(
        tmp_path, prompt=f"<VISUAL>source</VISUAL>\n{suffix}",
    )
    effective = f"<VISUAL>optimized</VISUAL>\n{suffix}"

    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.effective_prompt == context_ir_bridge._compile_effective_prompt(
        frozen, effective
    )


def test_reference_ref2va_uses_context_ir_and_binds_optimized_prompt(
    tmp_path: Path,
) -> None:
    frozen = _reference_fusion_frozen(tmp_path, dialogue=True)
    before = context_ir_bridge._dialogue_policy_score(
        frozen, frozen.source_prompt,
    )
    optimized = frozen.source_prompt.replace(
        "[Undetermined]", "[Chinese]", 1,
    )
    observed: list[httpx.Request] = []
    with _client(
        _success_handler(
            frozen, effective_prompt=optimized, requests=observed,
        )
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.provider_task_id == "context-task-1"
    assert sum(
        request.method == "POST" and request.url.path == "/v1/files/upload"
        for request in observed
    ) == 9
    assert sum(
        request.method == "POST" and request.url.path == "/v2/h3_context_ir"
        for request in observed
    ) == 1
    assert sum(
        request.method == "GET"
        and request.url.path.endswith("/context-task-1")
        for request in observed
    ) == 1
    submit = next(
        request for request in observed
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir"
    )
    body = json.loads(submit.content)
    assert body["model"] == "MiniMax-H3"
    assert body["content"][0]["text"] == context_ir_bridge._with_dialogue_policy(
        frozen.source_prompt
    )
    assert "<DUET_DIALOGUE_POLICY_V1>" in body["content"][0]["text"]

    assert before["overall"] < 1.0
    assert result.receipt_path is not None
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path,
    )
    raw_receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert raw_receipt["context_output_prompt"] == optimized
    assert receipt.effective_prompt == context_ir_bridge._compile_effective_prompt(
        frozen, optimized,
    )
    after = receipt.semantic_score["dialogue_policy"]
    assert isinstance(after, dict)
    assert after == {
        "language_explicit": 1.0,
        "exact_text": 1.0,
        "stop_after_line": 1.0,
        "no_repeat_or_improvise": 1.0,
        "no_extra_speech": 1.0,
        "overall": 1.0,
    }
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path,
    )
    assert "<d>[Chinese]这是严格冻结的台词</d>" in final_request.prompt
    assert "<d>[Undetermined]这是严格冻结的台词</d>" not in final_request.prompt
    assert final_request.prompt == receipt.effective_prompt
    assert hashlib.sha256(final_request.prompt.encode()).hexdigest() == (
        receipt.effective_prompt_sha256
    )
    assert h3._input_manifest(final_request)["prompt_sha256"] == (
        receipt.effective_prompt_sha256
    )
    h3._require_context_ir_receipt(final_request)


def test_reference_ref2va_restores_exact_dialogue_and_removes_extra_dialogue(
    tmp_path: Path,
) -> None:
    frozen = _reference_fusion_frozen(tmp_path, dialogue=True)
    optimized = frozen.source_prompt.replace(
        "<d>[Undetermined]这是严格冻结的台词</d>",
        "<d>[English]provider changed the line</d>",
    ) + "\n<d>[English]provider invented another line</d>"
    with _client(
        _success_handler(frozen, effective_prompt=optimized)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.receipt_path is not None
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path,
    )
    assert final_request.prompt.count("<d>") == 1
    assert "<d>[Chinese]这是严格冻结的台词</d>" in final_request.prompt
    assert "provider changed" not in final_request.prompt
    assert "provider invented" not in final_request.prompt
    assert final_request.voice_texts == ("这是严格冻结的台词",)


def test_reference_no_dialogue_prompt_explicitly_forbids_human_voice(
    tmp_path: Path,
) -> None:
    frozen = _reference_fusion_frozen(tmp_path, dialogue=False)
    before = context_ir_bridge._dialogue_policy_score(
        frozen, frozen.source_prompt,
    )
    optimized = frozen.source_prompt.replace(
        "restrained visual interval", "optimized visual interval", 1,
    )
    observed: list[httpx.Request] = []
    with _client(
        _success_handler(
            frozen, effective_prompt=optimized, requests=observed,
        )
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert sum(
        request.method == "POST" and request.url.path == "/v2/h3_context_ir"
        for request in observed
    ) == 1
    submit = next(
        request for request in observed
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir"
    )
    assert "no human voice" in json.loads(submit.content)["content"][0]["text"].lower()
    assert result.receipt_path is not None
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path,
    )
    after = receipt.semantic_score["dialogue_policy"]
    assert before["overall"] < after["overall"]
    assert isinstance(after, dict)
    assert after["language_explicit"] == 1.0
    assert after["exact_text"] == 1.0
    assert after["no_extra_speech"] == 1.0
    assert after["overall"] == 1.0
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path,
    )
    assert "no human voice" in final_request.prompt.lower()
    assert "closed lips" in final_request.prompt.lower()
    h3._require_context_ir_receipt(final_request)


def test_legacy_context_semantic_hash_is_unchanged_without_music_marker(
    tmp_path: Path,
) -> None:
    frozen = _no_audio_frozen(tmp_path)

    assert frozen.semantic_contract_sha256 == context_ir_bridge._canonical_sha256({
        "speech_markers": [],
        "dialogue_tokens": [],
    })


def test_legacy_audio_json_may_contain_music_marker_literal(
    tmp_path: Path,
) -> None:
    prompt = (
        'visual\n<AUDIO_CONTENT_JSON>[{"text":"literal '
        '<MUSIC_POLICY>forbid</MUSIC_POLICY>"}]'
        '</AUDIO_CONTENT_JSON>'
    )
    frozen = _no_audio_frozen(tmp_path, prompt=prompt)

    assert frozen.semantic_contract_sha256 == context_ir_bridge._canonical_sha256({
        "speech_markers": [],
        "dialogue_tokens": [],
    })


@pytest.mark.parametrize(
    "effective_suffix",
    [
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>",
        (
            "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
            "<MUSIC_POLICY>allow</MUSIC_POLICY>"
        ),
        (
            "<AUDIO_CONTENT_JSON>[ ]</AUDIO_CONTENT_JSON>\n"
            "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
        ),
    ],
    ids=("missing-policy", "changed-policy", "rewritten-audio"),
)
def test_context_scores_fusion_audio_or_music_policy_drift(
    tmp_path: Path, effective_suffix: str,
) -> None:
    source_suffix = (
        "<AUDIO_CONTENT_JSON>[]</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>\nnon_diegetic_music: N/A\n</MUSIC_POLICY>"
    )
    frozen = _no_audio_frozen(
        tmp_path, prompt=f"<VISUAL>source</VISUAL>\n{source_suffix}",
    )

    with _client(
        _success_handler(
            frozen,
            effective_prompt=f"<VISUAL>optimized</VISUAL>\n{effective_suffix}",
        )
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.semantic_score["music_policy"] == 0.0
    assert receipt.effective_prompt.endswith(source_suffix)
    assert not (frozen.workdir / ".h3").exists()


def _reference_audio_frozen(
    tmp_path: Path,
    *,
    purpose: h3.ReferenceAudioPurpose,
) -> context_ir_bridge.FrozenContextIrRequest:
    prompt = (
        "final fusion visual prompt"
        "\n<AUDIO_CONTENT_JSON>"
        '[{"order":1,"text":"first half second half","start_s":0.0,'
        '"end_s":8.0,"delivery":"off_screen","voice_ref":1}]'
        "</AUDIO_CONTENT_JSON>"
    )
    source = h3.H3Request(
        cid=f"{purpose}-segment-1",
        workdir=tmp_path / f"{purpose}-segment-1",
        client_request_id=f"{purpose}-h3-request-1",
        prompt=prompt,
        keyframes=((Path("01.png"), b"reference-frame-one"),),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=8,
        autodl_token="autodl-secret",
        workflow=h3.H3_MULTIMODAL_WORKFLOW,
        reference_audios=(
            _audio(tmp_path / f"{purpose}.wav", 1, purpose),
        ),
        skill_plan_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        multimodal_compiler_version="video-prompt-fusion-v1",
        upstream_dialogue_receipt_sha256=UPSTREAM_DIALOGUE_SHA256,
        audio_required=True,
        context_ir_required=True,
    )
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    return context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_artifact_path=artifact_path,
        upstream_artifact_sha256=artifact_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        source_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1,
            poll_total_s=0.2,
            poll_interval_s=0,
        ),
    )


def test_current_fusion_no_audio_effective_request_binds_context_and_drift_fails(
    tmp_path,
):
    frozen = _no_audio_frozen(tmp_path)
    with _client(_success_handler(frozen)) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert result.status == "succeeded"
    assert result.receipt_path is not None

    effective = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path,
    )
    assert effective.context_ir_required is True
    assert effective.context_ir_receipt_path == result.receipt_path
    assert effective.context_ir_receipt_sha256 == result.receipt_sha256
    h3._require_context_ir_receipt(effective)

    result.receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(h3.ReceiptError, match="context_ir_receipt"):
        h3._require_context_ir_receipt(effective)


def _response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _success_handler(
    frozen: context_ir_bridge.FrozenContextIrRequest,
    *,
    effective_prompt: str | None = None,
    requests: list[httpx.Request] | None = None,
):
    file_index = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal file_index
        if requests is not None:
            requests.append(request)
        if request.url.path == "/v1/files/upload":
            file_index += 1
            return _response(
                {
                    "file": {"file_id": str(427752006353317 + file_index)},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            )
        if request.url.path == "/v2/h3_context_ir":
            return _response({"task_id": "context-task-1"})
        if request.url.path == "/v2/query/video_generation/context-task-1":
            return _response(
                {
                    "task": {
                        "id": "context-task-1",
                        "task_type": "h3_context_ir",
                        "status": "succeeded",
                        "modality": "text",
                        "content": {
                            "prompt": effective_prompt
                            or frozen.source_prompt
                            + "\ncinematic lighting, restrained camera motion."
                        },
                    }
                }
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return handler


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_success_freezes_official_request_receipt_and_builds_h3_adapter(tmp_path):
    frozen = _frozen(tmp_path)
    observed: list[httpx.Request] = []
    with _client(_success_handler(frozen, requests=observed)) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.effective_prompt is not None
    assert result.receipt_path is not None and result.receipt_path.is_file()
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.source_prompt_sha256 == frozen.source_prompt_sha256
    assert receipt.effective_prompt_sha256 == hashlib.sha256(
        result.effective_prompt.encode()
    ).hexdigest()
    assert receipt.skill_plan_sha256 == frozen.skill_plan_sha256
    assert receipt.context_ir_request_sha256 == result.context_ir_request_sha256
    assert receipt.context_ir_task_sha256 == result.context_ir_task_sha256
    assert receipt.context_ir_attempt_sha256 == frozen.context_ir_attempt_sha256
    assert receipt.receipt_sha256 == result.receipt_sha256

    submits = [
        request for request in observed
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir"
    ]
    assert len(submits) == 1
    body = json.loads(submits[0].content)
    assert body == {
        "model": "MiniMax-H3",
        "content": [
            {
                "type": "text",
                "text": context_ir_bridge._with_dialogue_policy(
                    frozen.source_prompt
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": "mm_file://427752006353318"},
                "role": "reference_image",
            },
            {
                "type": "image_url",
                "image_url": {"url": "mm_file://427752006353319"},
                "role": "reference_image",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": "mm_file://427752006353320"},
                "role": "reference_audio",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": "mm_file://427752006353321"},
                "role": "reference_audio",
            },
        ],
        "duration": 8,
        "ratio": "9:16",
    }
    source_request, _plan_value = _source_request(tmp_path)
    final_request = context_ir_bridge.apply_effective_prompt(frozen, result.receipt_path)
    assert source_request.prompt == frozen.source_prompt
    assert final_request.prompt == receipt.effective_prompt
    assert final_request.voice_texts == source_request.voice_texts
    assert final_request.reference_audios == source_request.reference_audios
    assert final_request.context_ir_receipt_path == receipt.receipt_path
    assert final_request.context_ir_receipt_sha256 == receipt.receipt_sha256
    h3._require_context_ir_receipt(final_request)


def test_official_context_result_enters_h3_with_compiled_dialogue_policy(
    tmp_path,
):
    frozen = _frozen(tmp_path)
    provider_prompt = "\n".join(
        line for line in frozen.source_prompt.splitlines()
        if not line.lstrip().startswith("DUET_SPEECH_V1")
    )
    for token in frozen.dialogue_tokens:
        provider_prompt = provider_prompt.replace(token, token.replace("]", "] ", 1))
    assert "DUET_SPEECH_V1" not in provider_prompt
    assert all(token not in provider_prompt for token in frozen.dialogue_tokens)

    observed: list[httpx.Request] = []
    with _client(_success_handler(
        frozen,
        effective_prompt=provider_prompt,
        requests=observed,
    )) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.receipt_path is not None
    raw = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert raw["source_prompt_sha256"] == frozen.source_prompt_sha256
    compiled_prompt = context_ir_bridge._compile_effective_prompt(
        frozen, provider_prompt
    )
    assert raw["effective_prompt"] == compiled_prompt
    assert raw["effective_prompt_sha256"] == hashlib.sha256(
        compiled_prompt.encode()
    ).hexdigest()
    submits = [
        request for request in observed
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir"
    ]
    assert len(submits) == 1
    assert json.loads(submits[0].content)["content"][0]["text"] == (
        context_ir_bridge._with_dialogue_policy(frozen.source_prompt)
    )
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path
    )
    assert final_request.prompt.encode() == compiled_prompt.encode()
    assert final_request.prompt != frozen.source_prompt


@pytest.mark.parametrize(
    "mutate",
    [
        lambda prompt: prompt.replace("我会准时回来。", "我不会准时回来。"),
        lambda prompt: prompt + "\n<d>[Chinese]这是凭空新增的台词。</d>",
        lambda prompt: prompt.replace("mode=on_screen", "mode=off_screen", 1),
        lambda prompt: prompt.replace("voice_ref=1", "voice_ref=2", 1),
        lambda prompt: prompt.replace("[Chinese]我会准时回来。", "[English]我会准时回来。", 1),
        lambda prompt: "\n".join(
            reversed(prompt.splitlines())
        ),
    ],
    ids=[
        "dialogue_changed",
        "dialogue_invented",
        "screen_role_changed",
        "audio_mapping_changed",
        "language_changed",
        "speaking_order_changed",
    ],
)
def test_context_effective_prompt_is_not_blocked_or_rewritten_by_internal_syntax(
    tmp_path, mutate,
):
    frozen = _frozen(tmp_path)
    effective = mutate(frozen.source_prompt)
    with _client(_success_handler(frozen, effective_prompt=effective)) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.receipt_path is not None
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path
    )
    assert final_request.prompt.encode() == context_ir_bridge._with_dialogue_policy(
        effective
    ).encode()


def test_poll_transport_unknown_recovers_by_get_on_same_task_without_post(tmp_path):
    frozen = _frozen(tmp_path)
    first_requests: list[httpx.Request] = []
    base = _success_handler(frozen, requests=first_requests)

    def first_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            first_requests.append(request)
            raise httpx.ReadTimeout("ambiguous query", request=request)
        return base(request)

    with _client(first_handler) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "query_unknown"
    assert first.provider_task_id == "context-task-1"
    assert sum(
        request.method == "POST" and request.url.path == "/v2/h3_context_ir"
        for request in first_requests
    ) == 1

    resumed_requests: list[httpx.Request] = []
    with _client(_success_handler(frozen, requests=resumed_requests)) as client:
        resumed = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert resumed.status == "succeeded"
    assert resumed.provider_task_id == "context-task-1"
    assert [request.method for request in resumed_requests] == ["GET"]
    assert resumed_requests[0].url.path.endswith("/context-task-1")


def test_official_running_response_without_modality_keeps_polling_same_task(tmp_path):
    frozen = _frozen(tmp_path)
    calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=calls)
    query_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal query_count
        if request.url.path == "/v2/query/video_generation/context-task-1":
            calls.append(request)
            query_count += 1
            return _response({
                "task": {
                    "id": "context-task-1",
                    "task_type": "h3_context_ir",
                    "status": "running" if query_count == 1 else "succeeded",
                    "content": {
                        "prompt": "" if query_count == 1 else (
                            frozen.source_prompt
                            + "\ncinematic lighting, restrained camera motion."
                        ),
                    },
                },
            })
        return base(request)

    with _client(handler) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert query_count == 2
    assert sum(
        request.method == "POST" and request.url.path == "/v2/h3_context_ir"
        for request in calls
    ) == 1


def test_legacy_local_result_invalid_resumes_existing_task_with_get_only(tmp_path):
    frozen = _frozen(tmp_path)
    initial_calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=initial_calls)

    def query_unknown(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            initial_calls.append(request)
            raise httpx.ReadTimeout("ambiguous query", request=request)
        return base(request)

    with _client(query_unknown) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "query_unknown"
    attempt_path = (
        frozen.source_h3_request.workdir
        / ".context-ir" / "attempts" / "000001" / "attempt.json"
    )
    state = json.loads(attempt_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["error"] = "context_ir_result_invalid"
    attempt_path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    resumed_calls: list[httpx.Request] = []
    query_count = 0

    def resumed(request: httpx.Request) -> httpx.Response:
        nonlocal query_count
        resumed_calls.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v2/query/video_generation/context-task-1"
        query_count += 1
        return _response({
            "task": {
                "id": "context-task-1",
                "task_type": "h3_context_ir",
                "status": "running" if query_count == 1 else "succeeded",
                "content": {
                    "prompt": "" if query_count == 1 else (
                        frozen.source_prompt
                        + "\ncinematic lighting, restrained camera motion."
                    ),
                },
            },
        })

    with _client(resumed) as client:
        recovered = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert recovered.status == "succeeded"
    assert query_count == 2
    assert all(request.method == "GET" for request in resumed_calls)


def test_submission_unknown_never_reposts_without_task_id(tmp_path):
    frozen = _frozen(tmp_path)
    calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=calls)

    def ambiguous(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/h3_context_ir":
            calls.append(request)
            raise httpx.ReadTimeout("ambiguous submit", request=request)
        return base(request)

    with _client(ambiguous) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "submission_unknown"
    assert first.provider_task_id is None

    with _client(lambda request: pytest.fail(f"must not access network: {request}")) as client:
        repeated = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert repeated.status == "submission_unknown"
    assert sum(
        request.method == "POST" and request.url.path == "/v2/h3_context_ir"
        for request in calls
    ) == 1


def test_completed_duplicate_has_zero_network_and_exact_input_drift_is_rejected(tmp_path):
    frozen = _frozen(tmp_path)
    with _client(_success_handler(frozen)) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "succeeded"

    with _client(lambda request: pytest.fail(f"must not access network: {request}")) as client:
        repeated = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert repeated == first

    source, plan = _source_request(tmp_path)
    drifted = replace(source, duration=9)
    with pytest.raises(
        context_ir_bridge.ContextIrReceiptError,
        match="context_ir_request_receipt_mismatch",
    ):
        context_ir_bridge.freeze_context_ir_request(
            source_h3_request=drifted,
            upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
            upstream_artifact_path=frozen.upstream_artifact_path,
            upstream_artifact_sha256=frozen.upstream_artifact_sha256,
            upstream_dialogue_sha256_path=frozen.upstream_dialogue_sha256_path,
            source_prompt_sha256=hashlib.sha256(drifted.prompt.encode()).hexdigest(),
            minimax_api_key="minimax-secret",
        )


@pytest.mark.parametrize(
    "payload,expected_status,error_code",
    [
        (
            {"task": {"id": "context-task-1", "task_type": "h3_context_ir", "modality": "text", "status": "mystery", "content": {}}},
            "query_unknown",
            "context_ir_unknown_status",
        ),
        (
            {"task": {"id": "context-task-1", "task_type": "h3_context_ir", "modality": "text", "status": "succeeded", "content": {}}},
            "query_unknown",
            "context_ir_query_unknown",
        ),
        (
            {"task": {"id": "context-task-1", "task_type": "another_task", "status": "running", "content": {}}},
            "failed",
            "context_ir_result_type_invalid",
        ),
        (
            {"task": {"id": "context-task-1", "task_type": "h3_context_ir", "modality": "video", "status": "running", "content": {}}},
            "failed",
            "context_ir_result_type_invalid",
        ),
        (
            {"task": {"status": "succeeded", "content": {"prompt": "x"}}},
            "failed",
            "context_ir_task_mismatch",
        ),
    ],
)
def test_unknown_or_missing_query_fields_fail_closed(
    tmp_path, payload, expected_status, error_code
):
    frozen = _frozen(tmp_path)
    base = _success_handler(frozen)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            return _response(payload)
        return base(request)

    with _client(handler) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert result.status == expected_status
    assert result.error_code == error_code
    assert result.receipt_path is None


@pytest.mark.parametrize(
    ("status", "content", "error", "event"),
    [
        ("cancelled", {}, "context_ir_cancelled", "cancelled"),
        ("succeeded", {}, "context_ir_query_unknown", "result-missing"),
        (
            "succeeded",
            {"prompt": "x" * (context_ir_bridge.MAX_EFFECTIVE_PROMPT_BYTES + 1)},
            "context_ir_result_invalid",
            "result-invalid",
        ),
    ],
)
def test_terminal_context_ir_responses_preserve_attempt_scoped_diagnostic(
    tmp_path, status, content, error, event,
):
    frozen = _frozen(tmp_path)
    base = _success_handler(frozen)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            return _response({
                "task": {
                    "id": "context-task-1",
                    "task_type": "h3_context_ir",
                    "modality": "text",
                    "status": status,
                    "content": content,
                    "provider_marker": f"marker-{event}",
                },
            })
        return base(request)

    with _client(handler) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.error_code == error
    traces = sorted(
        (
            frozen.source_h3_request.workdir
            / ".context-ir" / "attempts" / "000001" / "errors"
        ).glob(f"context-ir-{event}-*.json")
    )
    assert len(traces) == 1
    raw = traces[0].read_text(encoding="utf-8")
    assert f"marker-{event}" in raw
    assert '"000001"' in raw or "attempt:000001" in raw


def test_context_ir_poll_timeout_preserves_last_running_response(tmp_path):
    frozen = replace(
        _frozen(tmp_path),
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1,
            poll_total_s=0.001,
            poll_interval_s=0,
        ),
    )
    base = _success_handler(frozen)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            return _response({
                "task": {
                    "id": "context-task-1",
                    "task_type": "h3_context_ir",
                    "modality": "text",
                    "status": "running",
                    "content": {},
                    "provider_marker": "last-running-response",
                },
            })
        return base(request)

    with _client(handler) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "query_unknown"
    assert result.error_code == "context_ir_poll_timeout"
    traces = sorted(
        (
            frozen.source_h3_request.workdir
            / ".context-ir" / "attempts" / "000001" / "errors"
        ).glob("context-ir-poll-timeout-*.json")
    )
    assert len(traces) == 1
    assert "last-running-response" in traces[0].read_text(encoding="utf-8")


def test_undocumented_video_list_query_shape_is_rejected(tmp_path):
    frozen = _frozen(tmp_path)
    base = _success_handler(frozen)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            return _response(
                {
                    "items": [
                        {
                            "id": "another-task",
                            "status": "succeeded",
                            "content": {"prompt": "别人的提示词"},
                        },
                        {
                            "id": "context-task-1",
                            "status": "succeeded",
                            "content": {"prompt": frozen.source_prompt},
                        },
                    ]
                }
            )
        return base(request)

    with _client(handler) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert result.status == "query_unknown"
    assert result.error_code == "context_ir_query_unknown"


def test_malformed_query_recovers_with_get_only_on_exact_task(tmp_path):
    frozen = _frozen(tmp_path)
    first_calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=first_calls)

    def malformed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            first_calls.append(request)
            return _response({"items": []})
        return base(request)

    with _client(malformed) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "query_unknown"

    resumed_calls: list[httpx.Request] = []
    with _client(_success_handler(frozen, requests=resumed_calls)) as client:
        resumed = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert resumed.status == "succeeded"
    assert [(call.method, call.url.path) for call in resumed_calls] == [
        ("GET", "/v2/query/video_generation/context-task-1")
    ]


def test_source_hash_and_compiler_markers_are_required_before_any_state_or_network(tmp_path):
    source, plan = _source_request(tmp_path)
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    assert "DUET_SPEECH_V1" in source.prompt
    with pytest.raises(
        context_ir_bridge.ContextIrContractError,
        match="source_prompt_sha256_mismatch",
    ):
        context_ir_bridge.freeze_context_ir_request(
            source_h3_request=source,
            upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
            upstream_artifact_path=artifact_path,
            upstream_artifact_sha256=artifact_sha256,
            upstream_dialogue_sha256_path=("dialogue", "sha256"),
            source_prompt_sha256="0" * 64,
            minimax_api_key="minimax-secret",
        )
    assert not (source.workdir / ".context-ir").exists()


def test_upstream_dialogue_receipt_is_required_and_frozen_into_final_receipt(tmp_path):
    source, _plan_value = _source_request(tmp_path)
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    with pytest.raises(
        context_ir_bridge.ContextIrContractError,
        match="source_speech_receipt_invalid",
    ):
        context_ir_bridge.freeze_context_ir_request(
            source_h3_request=source,
            upstream_dialogue_sha256="not-a-receipt",
            upstream_artifact_path=artifact_path,
            upstream_artifact_sha256=artifact_sha256,
            upstream_dialogue_sha256_path=("dialogue", "sha256"),
            source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest(),
            minimax_api_key="minimax-secret",
        )

    frozen = _frozen(tmp_path)
    with _client(_success_handler(frozen)) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert (
        receipt.upstream_dialogue_sha256
        == UPSTREAM_DIALOGUE_SHA256
    )


def test_dialogue_none_scores_context_ir_speech_without_blocking_output(tmp_path):
    frozen = _no_audio_frozen(tmp_path)
    with _client(
        _success_handler(
            frozen,
            effective_prompt=frozen.source_prompt + "\n<d>[Chinese]凭空说话。</d>",
        )
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert result.status == "succeeded"
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.semantic_score["speech_expected"] is False
    assert receipt.semantic_score["speech"] == 0.0
    assert not (frozen.workdir / ".h3").exists()


def test_context_ir_preserves_frozen_keyframe_timeline(tmp_path):
    frozen = _timeline_frozen(tmp_path)
    suffix = context_ir_bridge._fusion_policy_suffix(frozen.source_prompt)
    assert suffix is not None
    assert frozen.semantic_contract_sha256 == context_ir_bridge._canonical_sha256({
        "speech_markers": [],
        "dialogue_tokens": [],
        "keyframe_timeline_json": frozen.keyframe_timeline_json,
        "fusion_policy_suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
    })
    block_positions = tuple(
        frozen.source_prompt.index(marker)
        for marker in (
            "<VISUAL>",
            "<KEYFRAME_TIMELINE_JSON>",
            "<AUDIO_CONTENT_JSON>",
            "<MUSIC_POLICY>",
        )
    )
    assert block_positions == tuple(sorted(block_positions))
    effective = frozen.source_prompt.replace(
        "source visual", "rewritten visual prose", 1
    )
    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.effective_prompt == context_ir_bridge._compile_effective_prompt(
        frozen, effective
    )


def test_context_ir_scores_hard_cut_time_drift_without_blocking_h3(tmp_path):
    frozen = _timeline_frozen(tmp_path)
    effective = frozen.source_prompt.replace('"at_segment_s":2.267', '"at_segment_s":3.5')
    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.semantic_score["keyframe_timeline"] == 0.0
    assert receipt.effective_prompt.endswith(
        context_ir_bridge._fusion_policy_suffix(frozen.source_prompt)
    )
    assert not (frozen.workdir / ".h3").exists()


@pytest.mark.parametrize("mutation", ["reordered", "separated"])
def test_context_ir_scores_moved_fusion_contract_blocks_without_blocking_h3(
    tmp_path, mutation,
):
    frozen = _timeline_frozen(tmp_path)
    timeline_start = frozen.source_prompt.index("<KEYFRAME_TIMELINE_JSON>")
    audio_start = frozen.source_prompt.index("<AUDIO_CONTENT_JSON>")
    music_start = frozen.source_prompt.index("<MUSIC_POLICY>")
    visual = frozen.source_prompt[:timeline_start]
    timeline = frozen.source_prompt[timeline_start:audio_start].rstrip("\n")
    audio = frozen.source_prompt[audio_start:music_start].rstrip("\n")
    music = frozen.source_prompt[music_start:]
    effective = (
        f"{visual}{audio}\n{timeline}\n{music}"
        if mutation == "reordered"
        else f"{visual}{timeline}\nContext inserted prose.\n{audio}\n{music}"
    )
    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.semantic_score["music_policy"] == 0.0
    assert receipt.effective_prompt.endswith(
        context_ir_bridge._fusion_policy_suffix(frozen.source_prompt)
    )
    assert not (frozen.workdir / ".h3").exists()


def test_context_ir_rejects_non_nine_keyframe_timeline():
    timeline = [{
        "order": 1,
        "segment_time_s": 0.0,
        "source_scene_id": "SCENE_01",
        "transition": {"type": "start", "at_segment_s": 0.0},
    }]
    prompt = (
        "<KEYFRAME_TIMELINE_JSON>"
        + json.dumps(timeline, separators=(",", ":"))
        + "</KEYFRAME_TIMELINE_JSON>"
    )

    with pytest.raises(
        context_ir_bridge.ContextIrContractError,
        match="context_ir_semantic_mismatch",
    ):
        context_ir_bridge._keyframe_timeline_contract(prompt)


def test_receipt_bound_voice_allows_context_to_split_off_screen_dialogue(tmp_path):
    frozen = _reference_audio_frozen(tmp_path, purpose="voice")
    effective = (
        "final effective visual prompt; off-screen singer uses <Audio 1>.\n"
        "<d>[English]first half</d>\n"
        "the same off-screen singer continues from <Audio 1>.\n"
        "<d>[English]second half</d>"
    )
    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    assert result.effective_prompt == context_ir_bridge._with_dialogue_policy(
        effective
    )
    assert result.receipt_path is not None and result.receipt_path.is_file()


@pytest.mark.parametrize("purpose", ["ambience", "effect"])
def test_non_voice_audio_does_not_change_zero_speech_score(tmp_path, purpose):
    frozen = _reference_audio_frozen(tmp_path, purpose=purpose)
    effective = frozen.source_prompt + "\n<d>[English]invented speech</d>"
    with _client(
        _success_handler(frozen, effective_prompt=effective)
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "succeeded"
    receipt = context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    assert receipt.semantic_score["speech_expected"] is False
    assert receipt.semantic_score["speech"] == 0.0
    assert not (frozen.workdir / ".h3").exists()


def test_voice_object_without_audio_required_does_not_change_speech_score(
    tmp_path,
):
    frozen = _reference_audio_frozen(tmp_path, purpose="voice")
    source_without_audio_authority = copy.copy(frozen.source_h3_request)
    object.__setattr__(source_without_audio_authority, "audio_required", False)
    forged = replace(
        frozen,
        source_h3_request=source_without_audio_authority,
    )
    score = context_ir_bridge._semantic_score(
        forged,
        "<d>[English]invented speech</d>",
    )

    assert score["speech_expected"] is False
    assert score["speech"] == 0.0
    assert not (frozen.workdir / ".h3").exists()


@pytest.mark.parametrize("drift", ["purpose", "hash", "order"])
def test_receipt_bound_voice_authority_drift_fails_before_network(tmp_path, drift):
    frozen = _reference_audio_frozen(tmp_path, purpose="voice")
    if drift == "purpose":
        audio = frozen.source_h3_request.reference_audios[0]
        changed_source = replace(
            frozen.source_h3_request,
            reference_audios=(replace(audio, purpose="ambience"),),
        )
        changed = replace(frozen, source_h3_request=changed_source)
    elif drift == "hash":
        audio_index = next(
            index
            for index, reference in enumerate(frozen.references)
            if reference.role == "reference_audio"
        )
        changed_references = list(frozen.references)
        changed_references[audio_index] = replace(
            changed_references[audio_index], sha256="0" * 64,
        )
        changed = replace(frozen, references=tuple(changed_references))
    else:
        audio_index = next(
            index
            for index, reference in enumerate(frozen.references)
            if reference.role == "reference_audio"
        )
        changed_references = list(frozen.references)
        changed_references[audio_index] = replace(
            changed_references[audio_index], order=2,
        )
        changed = replace(frozen, references=tuple(changed_references))

    with _client(
        lambda request: pytest.fail(f"must not access network: {request}")
    ) as client:
        with pytest.raises((h3.H3Error, context_ir_bridge.ContextIrReceiptError)):
            context_ir_bridge.optimize_h3_prompt(changed, client=client)


def test_semantic_mismatch_voice_task_recovers_by_get_only_and_writes_receipt(
    tmp_path,
):
    frozen = _reference_audio_frozen(tmp_path, purpose="voice")
    initial_calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=initial_calls)

    def query_unknown(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/query/video_generation/context-task-1":
            initial_calls.append(request)
            raise httpx.ReadTimeout("ambiguous query", request=request)
        return base(request)

    with _client(query_unknown) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "query_unknown"
    assert first.provider_task_id == "context-task-1"

    attempt_path = (
        frozen.workdir
        / ".context-ir" / "attempts" / "000001" / "attempt.json"
    )
    state = json.loads(attempt_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["error"] = "context_ir_semantic_mismatch"
    attempt_path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    effective = (
        "final effective visual prompt; off-screen singer uses <Audio 1>.\n"
        "<d>[English]first half</d>\n"
        "<d>[English]second half</d>"
    )
    resumed_calls: list[httpx.Request] = []
    with _client(
        _success_handler(
            frozen,
            effective_prompt=effective,
            requests=resumed_calls,
        )
    ) as client:
        recovered = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert recovered.status == "succeeded"
    assert recovered.effective_prompt == context_ir_bridge._with_dialogue_policy(
        effective
    )
    assert recovered.receipt_path is not None
    assert recovered.receipt_path.is_file()
    assert [(call.method, call.url.path) for call in resumed_calls] == [
        ("GET", "/v2/query/video_generation/context-task-1")
    ]


def test_loaded_receipt_cannot_be_used_after_receipt_file_tampering(tmp_path):
    frozen = _frozen(tmp_path)
    with _client(_success_handler(frozen)) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    context_ir_bridge.load_effective_prompt_receipt(
        frozen, result.receipt_path
    )
    raw = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    raw["effective_prompt"] += "\ntampered"
    result.receipt_path.write_text(json.dumps(raw), encoding="utf-8")
    source, _plan_value = _source_request(tmp_path)
    with pytest.raises(
        context_ir_bridge.ContextIrReceiptError,
        match="context_ir_receipt_invalid",
    ):
        context_ir_bridge.apply_effective_prompt(frozen, result.receipt_path)


def test_success_response_without_task_id_is_unknown_and_never_reposted(tmp_path):
    frozen = _frozen(tmp_path)
    base = _success_handler(frozen)
    submit_count = 0

    def missing_task(request: httpx.Request) -> httpx.Response:
        nonlocal submit_count
        if request.url.path == "/v2/h3_context_ir":
            submit_count += 1
            return _response({})
        return base(request)

    with _client(missing_task) as client:
        first = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert first.status == "submission_unknown"
    with _client(lambda request: pytest.fail(f"must not access network: {request}")) as client:
        repeated = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert repeated.status == "submission_unknown"
    assert submit_count == 1


def test_upload_http_rejection_persists_private_provider_diagnostics(tmp_path):
    frozen = _frozen(tmp_path)
    secret_body = "provider-private-secret"

    def rejected(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/upload"
        return _response(
            {
                "base_resp": {
                    "status_code": 1004,
                    "status_msg": secret_body,
                },
            },
            status=422,
        )

    with _client(rejected) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == "failed"
    assert result.error_code == "context_ir_upload_rejected"
    attempt_path = (
        frozen.workdir
        / ".context-ir"
        / "attempts"
        / "000001"
        / "attempt.json"
    )
    raw = attempt_path.read_text(encoding="utf-8")
    state = json.loads(raw)
    assert state["http_status"] == 422
    assert state["provider_error_code"] == "1004"
    assert secret_body not in raw
    assert "provider_body" not in state


@pytest.mark.parametrize(
    ("status", "payload", "expected_status", "expected_error", "expected_code"),
    [
        (
            400,
            {"code": "InvalidParameter", "msg": "provider-private-secret"},
            "failed",
            "context_ir_submit_rejected",
            "InvalidParameter",
        ),
        (
            503,
            {
                "base_resp": {
                    "status_code": 2013,
                    "status_msg": "provider-private-secret",
                },
            },
            "submission_unknown",
            "context_ir_submission_unknown",
            "2013",
        ),
    ],
)
def test_submit_http_rejection_persists_private_provider_diagnostics(
    tmp_path,
    status,
    payload,
    expected_status,
    expected_error,
    expected_code,
):
    frozen = _frozen(tmp_path)
    calls: list[httpx.Request] = []
    base = _success_handler(frozen, requests=calls)

    def rejected(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v1/files/upload":
            return _response(
                {
                    "file": {
                        "file_id": str(427752006353317 + len(
                            [
                                call
                                for call in calls
                                if call.url.path == "/v1/files/upload"
                            ]
                        )),
                    },
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            )
        if request.url.path == "/v2/h3_context_ir":
            return _response(payload, status=status)
        return base(request)

    with _client(rejected) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)

    assert result.status == expected_status
    assert result.error_code == expected_error
    attempt_path = (
        frozen.workdir
        / ".context-ir"
        / "attempts"
        / "000001"
        / "attempt.json"
    )
    raw = attempt_path.read_text(encoding="utf-8")
    state = json.loads(raw)
    assert state["http_status"] == status
    assert state["provider_error_code"] == expected_code
    assert "provider-private-secret" not in raw
    assert "provider_body" not in state


def test_forged_frozen_request_and_changed_upstream_artifact_fail_before_network(
    tmp_path,
):
    frozen = _frozen(tmp_path)
    forged_source = replace(frozen.source_h3_request, duration=9)
    forged = replace(frozen, source_h3_request=forged_source, duration=9)
    with _client(lambda request: pytest.fail(f"must not access network: {request}")) as client:
        with pytest.raises(
            context_ir_bridge.ContextIrReceiptError,
            match="context_ir_request_receipt_mismatch",
        ):
            context_ir_bridge.optimize_h3_prompt(forged, client=client)

    frozen.upstream_artifact_path.write_text(
        json.dumps({"dialogue": {"sha256": "0" * 64}}), encoding="utf-8"
    )
    with _client(lambda request: pytest.fail(f"must not access network: {request}")) as client:
        with pytest.raises(
            context_ir_bridge.ContextIrContractError,
            match="upstream_artifact_sha256_mismatch",
        ):
            context_ir_bridge.optimize_h3_prompt(frozen, client=client)


def test_source_prompt_official_limit_fails_before_state_or_upload(tmp_path):
    source, _plan_value = _source_request(tmp_path)
    source = replace(source, prompt="x" * 7_001, workdir=tmp_path / "too-long")
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    with pytest.raises(
        context_ir_bridge.ContextIrContractError,
        match="source_prompt_too_long",
    ):
        context_ir_bridge.freeze_context_ir_request(
            source_h3_request=source,
            upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
            upstream_artifact_path=artifact_path,
            upstream_artifact_sha256=artifact_sha256,
            upstream_dialogue_sha256_path=("dialogue", "sha256"),
            source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest(),
            minimax_api_key="minimax-secret",
        )
    assert not (source.workdir / ".context-ir").exists()
