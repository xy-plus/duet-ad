"""Replay external (non-H3) pipeline nodes against their real providers.

DeepSeek nodes call the live API with the harvested payload (vision nodes
attach the harvested frame images) and assert the answer passes the same
validator/normalizer the pipeline applies.  ``seedream-reference-submit`` and
``dialogue-tts-line`` are paid calls and marked ``live_provider`` (disable
with ``NODE_REPLAY_LIVE=0``); see docs/node-replay-tests.md for costs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.fusion_schema import normalize_fusion_prompt
from app.object_replace_schema import (
    merge_object_replace_index,
    normalize_object_replace,
)
from app.pipeline import (
    DIALOGUE_GENERATION_SCHEMA,
    TRANSLATION_SCHEMA,
    VIDEO_SKILL,
    Pipeline,
    SAME_AS_SOURCE_LANGUAGE,
    _validate_dialogue_generation_shape,
    _validate_translation_shape,
)
from app.video_maker_schemas import (
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    normalize_visual_prompt,
    project_index_schema,
)

from node_replay_support import (
    Fixture,
    node_fixtures,
    sha256_file,
)


def _params(node: str) -> list[Fixture]:
    return node_fixtures(node)


def _ids(fixtures: list[Fixture]) -> list[str]:
    return [fixture.test_id for fixture in fixtures]


def _verify_frame_binding(fixture: Fixture) -> list[Path]:
    """The harvested images must hash to the payload's frame receipts."""
    images = fixture.media
    frames = [
        frame
        for segment in _segments_of(fixture.input)
        for frame in segment["frames"]
    ]
    assert len(images) == len(frames), (
        f"{fixture.node}/{fixture.scope}: {len(images)} media files "
        f"for {len(frames)} payload frames"
    )
    for image, frame in zip(images, frames):
        assert sha256_file(image) == frame["sha256"], (
            f"{fixture.node}/{fixture.scope}: frame {frame['frame_order']} "
            "no longer matches its harvested payload receipt"
        )
    return images


def _segments_of(payload: dict) -> list[dict]:
    if "segments" in payload:
        return payload["segments"]
    return [{"frames": payload["frames"]}]


# --------------------------------------------------------- translation


@pytest.mark.parametrize(
    "fixture",
    _params("deepseek-translation"),
    ids=_ids(_params("deepseek-translation")),
)
def test_deepseek_translation(
    fixture: Fixture, pipeline_instance: Pipeline,
) -> None:
    payload = fixture.input
    lines = payload["source_lines"]
    target_language = payload["target_language"]
    if not lines:
        pytest.skip("historical project had no spoken dialogue")
    if target_language == SAME_AS_SOURCE_LANGUAGE:
        # Deterministic skip branch: no API call in the real pipeline either.
        result = pipeline_instance._translation_request(
            lines, target_language, payload["visual_prompts"],
        )
        assert result.record["outcome"] == "skipped"
        translated = Pipeline._normalize_translation(result.value, lines)
    else:
        if not pipeline_instance.settings.deepseek_api_key:
            pytest.skip("DEEPSEEK_API_KEY is not set")
        result = pipeline_instance._translation_request(
            lines, target_language, payload["visual_prompts"],
        )
        # The node's own transport-shape gate.
        _validate_translation_shape(result.value)
        translated = Pipeline._normalize_translation(result.value, lines)
    # Downstream gate: dialogue-localize/TTS consume one text per source line.
    assert len(translated) == len(lines)
    assert all(line["text"].strip() for line in translated)


# ------------------------------------------------- dialogue generation


@pytest.mark.parametrize(
    "fixture",
    _params("deepseek-dialogue-generation"),
    ids=_ids(_params("deepseek-dialogue-generation")),
)
def test_deepseek_dialogue_generation(
    fixture: Fixture, pipeline_instance: Pipeline,
) -> None:
    payload = fixture.input
    if not pipeline_instance.settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY is not set")
    prepared = payload["segments"]
    result = pipeline_instance._dialogue_generation_request(
        payload["brief"], payload["target_language"], prepared,
    )
    _validate_dialogue_generation_shape(result.value)
    generated = Pipeline._normalize_generated_dialogue(result.value, prepared)
    assert generated, "dialogue generation produced no usable lines"
    duration = float(payload["duration_s"])
    for line in generated:
        assert 0 <= line["start_s"] < line["end_s"] <= duration + 0.01
        assert line["text"].strip()


# --------------------------------------------- video-maker visual analysis


@pytest.mark.parametrize(
    "fixture", _params("video-maker"), ids=_ids(_params("video-maker"))
)
def test_video_maker_segment_analysis(
    fixture: Fixture, deepseek_client,
) -> None:
    images = _verify_frame_binding(fixture)
    skill = VIDEO_SKILL.read_text(encoding="utf-8")
    result = deepseek_client.complete_structured(
        skill
        + "\n\n本次只执行逐段视觉分析；最终输出严格遵循本次注入的 JSON Schema。",
        fixture.input,
        VISUAL_PROMPT_SCHEMA,
        images,
        validator=normalize_visual_prompt,
    )
    # The node's transform is the downstream (fusion/index) gate.
    prompt = normalize_visual_prompt(result.value)
    assert isinstance(prompt, str) and len(prompt.strip()) >= 50


# ------------------------------------------------ video-maker project index


@pytest.mark.parametrize(
    "fixture",
    _params("video-maker-project-index"),
    ids=_ids(_params("video-maker-project-index")),
)
def test_video_maker_project_index(
    fixture: Fixture, deepseek_client,
) -> None:
    images = _verify_frame_binding(fixture)
    payload = fixture.input
    schema = project_index_schema([
        int(segment["segment_index"]) for segment in payload["segments"]
    ])
    skill = VIDEO_SKILL.read_text(encoding="utf-8")
    result = deepseek_client.complete_structured(
        skill
        + (
            "\n\n严格执行项目级索引阶段；像素帧是视觉事实的唯一依据；"
            "segments[].visual_prompt 只用于汇总逐段已发现对象并检查跨段覆盖，"
            "不得用它替代或推翻像素证据；最终填写本次注入的 Schema。"
        ),
        payload,
        schema,
        images,
        validator=normalize_project_index,
    )
    index = normalize_project_index(result.value)
    # Downstream gate: object-replace plans over these stable-key categories.
    assert {"people", "entities", "scenes", "relations"} <= set(index)
    assert any(index[category] for category in ("people", "entities", "scenes"))


# ------------------------------------------------------------ object-replace


@pytest.mark.parametrize(
    "fixture", _params("object-replace"), ids=_ids(_params("object-replace"))
)
def test_object_replace(
    fixture: Fixture, pipeline_instance: Pipeline, tmp_path: Path,
) -> None:
    if not pipeline_instance.settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY is not set")
    payload = fixture.input["payload"]
    element_index = payload["global_element_index"]
    translated = payload["translated_dialogue"]
    voice_ids = [
        voice["voice_id"] for voice in pipeline_instance.voice_library.voices
    ]
    user_reference = fixture.media[0] if payload["has_user_image"] else None
    result = pipeline_instance.deepseek.complete_structured(
        fixture.input["skill"],
        payload,
        fixture.input["output_schema"],
        [user_reference] if user_reference is not None else [],
        validator=lambda value: normalize_object_replace(
            value,
            input_sha256=fixture.input["input_sha256"],
            element_index=element_index,
            voice_ids=voice_ids,
            has_user_image=payload["has_user_image"],
            user_instruction=payload["user_replacement_instruction"],
            dialogue_line_count=len(translated),
        ),
    )
    patch = normalize_object_replace(
        result.value,
        input_sha256=fixture.input["input_sha256"],
        element_index=element_index,
        voice_ids=voice_ids,
        has_user_image=payload["has_user_image"],
        user_instruction=payload["user_replacement_instruction"],
        dialogue_line_count=len(translated),
    )
    # The transform's voice materialization, including the default-voice
    # fallback that keeps voiceless projects (e.g. dfe64795) alive.
    fallback_voice_id = (
        pipeline_instance.voice_library.default_voice(
            payload["target_language"]
        )["voice_id"]
        if translated
        else None
    )
    project = tmp_path / "project"
    (
        metadata_by_id, _references, paths, _fallback_used,
    ) = pipeline_instance._materialize_voice_references(
        project, patch, fallback_voice_id=fallback_voice_id,
    )
    enhanced = merge_object_replace_index(element_index, patch, metadata_by_id)
    assert {"people", "entities", "scenes", "dialogue_voices"} <= set(enhanced)
    if translated:
        # The TTS stage must find at least one materialized voice reference.
        assert paths, "dialogue requires TTS but no voice was materialized"
        for path in paths:
            assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------- h3-max-r2v-prompt


@pytest.mark.parametrize(
    "fixture",
    _params("h3-max-r2v-prompt"),
    ids=_ids(_params("h3-max-r2v-prompt")),
)
def test_h3_max_r2v_prompt(fixture: Fixture, deepseek_client) -> None:
    payload = fixture.input
    if payload["agent"] != "deepseek":
        pytest.skip(f"fusion agent {payload['agent']!r} is not replayed")
    input_sha256 = payload["input_sha256"]
    segment_index = int(payload["payload"]["segment_index"])
    constraints = payload["payload"]["authoritative_constraints"]

    def validate(value: dict) -> None:
        normalize_fusion_prompt(
            value,
            input_sha256=input_sha256,
            segment_index=segment_index,
            authoritative_constraints=constraints,
        )

    result = deepseek_client.complete_structured(
        payload["skill"], payload["payload"], payload["output_schema"], [],
        validator=validate,
    )
    # The node's transform is the downstream (Picsart submit) gate.
    prompt = normalize_fusion_prompt(
        result.value,
        input_sha256=input_sha256,
        segment_index=segment_index,
        authoritative_constraints=constraints,
    )
    assert isinstance(prompt, str) and len(prompt.strip()) >= 100


# -------------------------------------------------- seedream-reference-submit


@pytest.mark.live_provider
@pytest.mark.parametrize(
    "fixture",
    _params("seedream-reference-submit"),
    ids=_ids(_params("seedream-reference-submit")),
)
def test_seedream_reference_submit(
    fixture: Fixture, seedream_client, tmp_path: Path,
) -> None:
    payload = fixture.input
    prompt_path = fixture.directory / "extras" / "prompt.txt"
    if not prompt_path.is_file():
        pytest.skip("historical prompt.txt was not harvested")
    prompt = prompt_path.read_text(encoding="utf-8")
    assert (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        == payload["prompt_sha256"]
    )
    images = fixture.media
    receipts = payload["input_receipts"]
    assert len(images) == len(receipts)
    for image, receipt in zip(images, receipts):
        assert sha256_file(image) == receipt["sha256"]
    result = seedream_client.generate(
        prompt=prompt,
        images=images,
        destination=tmp_path / "provider-output.bin",
        receipt_path=tmp_path / "seedream-receipt.json",
    )
    output = Path(result.value)
    assert output.is_file() and output.stat().st_size > 0
    import cv2
    import numpy as np

    decoded = cv2.imdecode(
        np.frombuffer(output.read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None and decoded.size > 0


# ------------------------------------------------------------ dialogue-tts-line


@pytest.mark.live_provider
@pytest.mark.parametrize(
    "fixture",
    _params("dialogue-tts-line"),
    ids=_ids(_params("dialogue-tts-line")),
)
def test_dialogue_tts_line(
    fixture: Fixture, autodl_client, tmp_path: Path,
) -> None:
    from app import media

    payload = fixture.input
    voice = fixture.media[0]
    assert voice.suffix == ".wav"
    submitted = autodl_client.submit_tts(text=payload["text"], voice_path=voice)
    polled = autodl_client.poll(submitted.value)
    destination = tmp_path / "line.wav"
    autodl_client.download(polled.value, destination)
    assert destination.is_file() and destination.stat().st_size > 0
    probe = media.run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(destination),
        ],
        timeout=60,
    )
    assert float(json.loads(probe["stdout"])["format"]["duration"]) > 0.2
