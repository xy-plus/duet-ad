from __future__ import annotations

import pytest

from app.fusion_schema import (
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


def test_fusion_normalizer_rejects_unbound_or_extra_output() -> None:
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
            {**base, "extra": True},
            input_sha256=digest,
            segment_index=1,
        )
