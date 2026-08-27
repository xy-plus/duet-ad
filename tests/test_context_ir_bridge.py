import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app import context_ir_bridge, h3, h3_multimodal


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
            {"type": "text", "text": frozen.source_prompt},
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


def test_official_context_result_enters_h3_byte_for_byte_without_speech_rewrite(
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
    assert raw["effective_prompt"] == provider_prompt
    assert raw["effective_prompt_sha256"] == hashlib.sha256(
        provider_prompt.encode()
    ).hexdigest()
    submits = [
        request for request in observed
        if request.method == "POST" and request.url.path == "/v2/h3_context_ir"
    ]
    assert len(submits) == 1
    assert json.loads(submits[0].content)["content"][0]["text"] == frozen.source_prompt
    final_request = context_ir_bridge.apply_effective_prompt(
        frozen, result.receipt_path
    )
    assert final_request.prompt.encode() == provider_prompt.encode()
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
    assert final_request.prompt.encode() == effective.encode()


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


def test_dialogue_none_rejects_any_context_ir_speech(tmp_path):
    source, _plan_value = _source_request(tmp_path)
    source = replace(
        source,
        prompt="references and visual semantics only; no speech events.",
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        workdir=tmp_path / "none-segment",
    )
    artifact_path, artifact_sha256 = _upstream_artifact(tmp_path)
    frozen = context_ir_bridge.freeze_context_ir_request(
        source_h3_request=source,
        upstream_dialogue_sha256=UPSTREAM_DIALOGUE_SHA256,
        upstream_artifact_path=artifact_path,
        upstream_artifact_sha256=artifact_sha256,
        upstream_dialogue_sha256_path=("dialogue", "sha256"),
        source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest(),
        minimax_api_key="minimax-secret",
        timeouts=context_ir_bridge.ContextIrTimeouts(
            request_s=0.1, poll_total_s=0.2, poll_interval_s=0
        ),
    )
    with _client(
        _success_handler(
            frozen,
            effective_prompt=frozen.source_prompt + "\n<d>[Chinese]凭空说话。</d>",
        )
    ) as client:
        result = context_ir_bridge.optimize_h3_prompt(frozen, client=client)
    assert result.status == "failed"
    assert result.error_code == "context_ir_semantic_mismatch"
    assert not (source.workdir / ".h3").exists()


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
