from __future__ import annotations

import pytest

from app.fusion_schema import (
    AUTHORITY_BEGIN,
    AUTHORITY_END,
    FUSION_OUTPUT_NAME,
    fusion_prompt_schema,
    normalize_fusion_prompt,
)


def test_fusion_schema_binds_input_and_normalizes_prompt() -> None:
    payload = {"segment_index": 2, "visual_prompt": "source"}
    schema, digest = fusion_prompt_schema(payload)
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema", "version", "input_sha256", "segment_index", "prompt",
    ]
    assert schema["properties"]["input_sha256"]["enum"] == [digest]
    assert schema["properties"]["segment_index"]["enum"] == [2]
    assert normalize_fusion_prompt({
        "schema": FUSION_OUTPUT_NAME,
        "version": 1,
        "input_sha256": digest,
        "segment_index": 2,
        "prompt": "replace all people and scenes",
    }, input_sha256=digest, segment_index=2) == "replace all people and scenes"


def test_fusion_normalizer_rejects_unbound_output() -> None:
    payload = {"segment_index": 1}
    _schema, digest = fusion_prompt_schema(payload)
    base = {
        "schema": FUSION_OUTPUT_NAME,
        "version": 1,
        "input_sha256": digest,
        "segment_index": 1,
        "prompt": "valid",
    }
    with pytest.raises(ValueError):
        normalize_fusion_prompt(
            {**base, "segment_index": 2},
            input_sha256=digest,
            segment_index=1,
        )
    with pytest.raises(ValueError):
        normalize_fusion_prompt(
            {key: item for key, item in base.items() if key != "prompt"},
            input_sha256=digest,
            segment_index=1,
        )


def test_fusion_normalizer_tolerates_extra_top_level_keys() -> None:
    payload = {"segment_index": 1}
    _schema, digest = fusion_prompt_schema(payload)
    prompt = normalize_fusion_prompt(
        {
            "schema": FUSION_OUTPUT_NAME,
            "version": 1,
            "input_sha256": digest,
            "segment_index": 1,
            "prompt": "valid",
            "confidence": 0.9,
        },
        input_sha256=digest,
        segment_index=1,
    )
    assert prompt == "valid"


def test_fusion_normalizer_does_not_grade_prompt_semantics() -> None:
    _schema, digest = fusion_prompt_schema({"segment_index": 1})
    prompt = "A usable prompt that intentionally omits every expected mapping."
    value = {
        "schema": FUSION_OUTPUT_NAME,
        "version": 1,
        "input_sha256": digest,
        "segment_index": 1,
        "prompt": prompt,
    }

    assert normalize_fusion_prompt(
        value,
        input_sha256=digest,
        segment_index=1,
    ) == prompt


def test_fusion_normalizer_appends_backend_constraints_without_grading_them() -> None:
    _schema, digest = fusion_prompt_schema({"segment_index": 1})
    block = (
        f"{AUTHORITY_BEGIN}\n"
        "This block is the sole authority.\n"
        "- person-01 | voice: Audio 1 | reference: Image 1\n"
        f"{AUTHORITY_END}"
    )
    value = {
        "schema": FUSION_OUTPUT_NAME,
        "version": 1,
        "input_sha256": digest,
        "segment_index": 1,
        "prompt": " person-01 uses Audio 9 and ignores Image 1. ",
    }

    assert normalize_fusion_prompt(
        value,
        input_sha256=digest,
        segment_index=1,
        authoritative_constraints=block,
    ) == value["prompt"].strip() + "\n\n" + block


@pytest.mark.parametrize(
    "prompt",
    [None, 42, []],
)
def test_fusion_normalizer_rejects_non_string_prompt_values(prompt: object) -> None:
    _schema, digest = fusion_prompt_schema({"segment_index": 1})
    with pytest.raises(ValueError):
        normalize_fusion_prompt(
            {
                "schema": FUSION_OUTPUT_NAME,
                "version": 1,
                "input_sha256": digest,
                "segment_index": 1,
                "prompt": prompt,
            },
            input_sha256=digest,
            segment_index=1,
        )


def test_fusion_normalizer_allows_empty_model_prose_when_backend_has_constraints() -> None:
    _schema, digest = fusion_prompt_schema({"segment_index": 1})
    block = f"{AUTHORITY_BEGIN}\nkeep source continuity\n{AUTHORITY_END}"

    assert normalize_fusion_prompt(
        {
            "schema": FUSION_OUTPUT_NAME,
            "version": 1,
            "input_sha256": digest,
            "segment_index": 1,
            "prompt": "   ",
        },
        input_sha256=digest,
        segment_index=1,
        authoritative_constraints=block,
    ) == "\n\n" + block


def test_fusion_normalizer_does_not_apply_a_local_quality_length_gate() -> None:
    _schema, digest = fusion_prompt_schema({"segment_index": 1})
    prompt = "x" * 32769

    assert normalize_fusion_prompt(
        {
            "schema": FUSION_OUTPUT_NAME,
            "version": 1,
            "input_sha256": digest,
            "segment_index": 1,
            "prompt": prompt,
        },
        input_sha256=digest,
        segment_index=1,
    ) == prompt
