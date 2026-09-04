"""Provider-neutral structured contract for one H3 prompt-fusion segment."""

from __future__ import annotations

import hashlib
import json
from typing import Any


FUSION_OUTPUT_NAME = "duet.h3-max-r2v-prompt-output"
FUSION_OUTPUT_VERSION = 1
AUTHORITY_BEGIN = "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"
AUTHORITY_END = "END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"


def canonical_input(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fusion_prompt_schema(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    segment_index = payload.get("segment_index")
    if isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 1:
        raise ValueError("fusion input segment_index is invalid")
    input_sha256 = hashlib.sha256(canonical_input(payload)).hexdigest()
    properties = {
        "schema": {"type": "string", "enum": [FUSION_OUTPUT_NAME]},
        "version": {"type": "integer", "enum": [FUSION_OUTPUT_VERSION]},
        "input_sha256": {"type": "string", "enum": [input_sha256]},
        "segment_index": {"type": "integer", "enum": [segment_index]},
        "prompt": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }, input_sha256


def normalize_fusion_prompt(
    value: object,
    *,
    input_sha256: str,
    segment_index: int,
    authoritative_constraints: str = "",
) -> str:
    if not isinstance(value, dict) or set(value) != {
        "schema", "version", "input_sha256", "segment_index", "prompt",
    }:
        raise ValueError("fusion prompt output is invalid")
    if (
        value["schema"] != FUSION_OUTPUT_NAME
        or isinstance(value["version"], bool)
        or value["version"] != FUSION_OUTPUT_VERSION
        or value["input_sha256"] != input_sha256
        or isinstance(value["segment_index"], bool)
        or value["segment_index"] != segment_index
    ):
        raise ValueError("fusion prompt output is not bound to its input")
    prompt = value["prompt"]
    if not isinstance(prompt, str):
        raise ValueError("fusion prompt output is invalid")
    prompt = prompt.strip()
    if not isinstance(authoritative_constraints, str):
        raise ValueError("fusion authority block is invalid")
    authoritative_constraints = authoritative_constraints.strip()
    if authoritative_constraints:
        prompt = prompt + "\n\n" + authoritative_constraints
    return prompt
