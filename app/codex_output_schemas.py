"""Strict model-output DTOs for the three Codex Skill phases.

The schemas shape model transport only.  Backend validators still bind IDs,
counts, hashes, timelines, and publication.  Backend-known Global keys are
injected as exact object properties; only keys discovered by the model use
arrays with an explicit ``key`` field.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from app.codex_runner import CodexOutputValidationError


def _object(properties: Mapping[str, dict]) -> dict:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict, *, minimum: int = 0, maximum: int | None = None) -> dict:
    value: dict = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        value["maxItems"] = maximum
    return value


_TEXT = {"type": "string", "minLength": 1}
_INT0 = {"type": "integer", "minimum": 0}
_INT1 = {"type": "integer", "minimum": 1}
_BOOL = {"type": "boolean"}
_TEXTS = _array(_TEXT, maximum=20)


# Per-segment video-maker transport.  Segment identity and ordered keyframes
# are frozen by the backend invocation and deliberately absent here.
VISUAL_PROMPT_SCHEMA = _object({
    "prompt": {"type": "string", "minLength": 1, "maxLength": 32768},
})


def dialogue_lines_schema(*, line_count: int) -> dict:
    """Wrap complete model-owned lines without backend fields or text surgery."""
    if (
        isinstance(line_count, bool)
        or not isinstance(line_count, int)
        or not 1 <= line_count <= 200
    ):
        raise ValueError("dialogue line count is invalid")
    return _object({
        "lines": _array(
            _object({
                "content": {"type": "string", "minLength": 1, "maxLength": 500},
            }),
            minimum=line_count,
            maximum=line_count,
        ),
    })


def _reject(reason: str, field_path: str, message: str) -> None:
    raise CodexOutputValidationError(reason, field_path, message=message)


def normalize_visual_prompt(value: object) -> str:
    """Accept only model-owned visual prose; backend bindings stay implicit."""
    if not isinstance(value, dict) or set(value) != {"prompt"}:
        _reject(
            "visual_prompt_shape_invalid",
            "/prompt",
            "visual prompt output is invalid",
        )
    prompt = value["prompt"]
    if not isinstance(prompt, str):
        _reject(
            "visual_prompt_type_invalid",
            "/prompt",
            "visual prompt output is invalid",
        )
    if not prompt:
        _reject("visual_prompt_empty", "/prompt", "visual prompt output is invalid")
    if len(prompt.encode("utf-8")) > 32 * 1024:
        _reject("visual_prompt_oversize", "/prompt", "visual prompt output is invalid")
    return prompt


def normalize_dialogue_lines(
    value: object, *, source_lines: Iterable[Mapping[str, object]],
) -> list[dict]:
    """Positionally bind untouched model content to backend-frozen line data."""
    if isinstance(source_lines, (str, bytes)):
        _reject("dialogue_source_invalid", "/lines", "dialogue output is invalid")
    try:
        frozen = [dict(line) for line in source_lines]
    except (TypeError, ValueError):
        _reject("dialogue_source_invalid", "/lines", "dialogue output is invalid")
    if not 1 <= len(frozen) <= 200:
        _reject("dialogue_source_invalid", "/lines", "dialogue output is invalid")
    for index, line in enumerate(frozen):
        if (
            not isinstance(line.get("text"), str)
            or isinstance(line.get("start_s"), bool)
            or not isinstance(line.get("start_s"), (int, float))
            or isinstance(line.get("end_s"), bool)
            or not isinstance(line.get("end_s"), (int, float))
        ):
            _reject(
                "dialogue_source_invalid",
                f"/lines/{index}",
                "dialogue output is invalid",
            )
    if not isinstance(value, dict) or set(value) != {"lines"}:
        _reject("dialogue_shape_invalid", "/lines", "dialogue output is invalid")
    lines = value["lines"]
    if not isinstance(lines, list) or len(lines) != len(frozen):
        _reject("dialogue_count_invalid", "/lines", "dialogue output is invalid")
    normalized: list[dict] = []
    for index, (line, output_line) in enumerate(zip(frozen, lines, strict=True)):
        path = f"/lines/{index}/content"
        if not isinstance(output_line, dict) or set(output_line) != {"content"}:
            _reject("dialogue_line_shape_invalid", path, "dialogue output is invalid")
        content = output_line["content"]
        if not isinstance(content, str):
            _reject("dialogue_content_type_invalid", path, "dialogue output is invalid")
        if not content:
            _reject("dialogue_content_empty", path, "dialogue output is invalid")
        if len(content) > 500:
            _reject("dialogue_content_oversize", path, "dialogue output is invalid")
        normalized.append({**line, "text": content})
    return normalized


_ELEMENT_OCCURRENCE = _object({
    "segment_index": _INT0,
    "frame_orders": _array(_INT1, minimum=1, maximum=9),
})
_ELEMENT = _object({
    "key": _TEXT,
    "source_visual_description": _TEXT,
    "occurrences": _array(_ELEMENT_OCCURRENCE, maximum=100),
    "replaceable": _TEXTS,
    "preserve": _TEXTS,
})
_RELATION_FRAME = _object({
    "frame_order": _INT1,
    "state": _TEXT,
    "geometry": _TEXT,
})
_RELATION_OCCURRENCE = _object({
    "segment_index": _INT0,
    "frames": _array(_RELATION_FRAME, minimum=1, maximum=9),
})
_RELATION = _object({
    "key": _TEXT,
    "subject_key": _TEXT,
    "predicate": _TEXT,
    "object_key": _TEXT,
    "occurrences": _array(_RELATION_OCCURRENCE, maximum=100),
    "preserve": _TEXTS,
    "replace_together": _BOOL,
})

PROJECT_INDEX_SCHEMA = _object({
    "people": _array(_ELEMENT, maximum=100),
    "entities": _array(_ELEMENT, maximum=100),
    "scenes": _array(_ELEMENT, minimum=1, maximum=100),
    "relations": _array(_RELATION, maximum=200),
})


_GLOBAL_PERSON = _object({
    "source_identity": _TEXT,
    "replacement_identity": _TEXT,
    "wardrobe_change": _TEXT,
    "local_color_change": _TEXT,
})
_GLOBAL_ENTITY = _object({
    "description": _TEXT,
    "owner": _TEXT,
    "association": _TEXT,
    "persistence": _TEXT,
})
_GLOBAL_SCENE = _object({
    "source_scene": _TEXT,
    "replacement_scene": _TEXT,
    "semantic_change": _TEXT,
    "geometry_change": _TEXT,
    "depth_change": _TEXT,
    "layout_change": _TEXT,
    "local_color_change": _TEXT,
})
_GLOBAL_RELATION = _object({
    "replacement_system": _TEXT,
    "preserve": _TEXT,
})
_GLOBAL_VALUE_SCHEMAS = {
    "people": _GLOBAL_PERSON,
    "entities": _GLOBAL_ENTITY,
    "scenes": _GLOBAL_SCENE,
    "relations": _GLOBAL_RELATION,
}


def _global_stable_keys(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_GLOBAL_VALUE_SCHEMAS):
        raise ValueError("global plan stable keys are invalid")
    normalized: dict[str, tuple[str, ...]] = {}
    for category in _GLOBAL_VALUE_SCHEMAS:
        keys = value[category]
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Iterable):
            raise ValueError("global plan stable keys are invalid")
        materialized = tuple(keys)
        if (
            any(
                not isinstance(key, str) or not key.strip() or key != key.strip()
                for key in materialized
            )
            or len(materialized) != len(set(materialized))
        ):
            raise ValueError("global plan stable keys are invalid")
        normalized[category] = tuple(sorted(materialized))
    return normalized


def global_plan_schema(*, stable_keys: Mapping[str, Iterable[str]]) -> dict:
    """Bind every Global output property to one backend-frozen stable key."""
    keys = _global_stable_keys(stable_keys)
    return _object({
        category: _object({
            stable_key: _GLOBAL_VALUE_SCHEMAS[category]
            for stable_key in keys[category]
        })
        for category in _GLOBAL_VALUE_SCHEMAS
    })


_DERIVED_OBSERVATION = _object({
    "key": _TEXT,
    "mode": {
        "type": "string",
        "enum": ["optical_projection", "temporal_residual", "source-preserve"],
    },
    "source_carrier": _TEXT,
    "visible_region": _TEXT,
    "boundary": _TEXT,
    "relationship": _TEXT,
})
_FRAME_PERSON = _object({
    "key": _TEXT,
    "visible_region": _TEXT,
    "boundary": _TEXT,
    "body_and_pose": _TEXT,
    "derived_observations": _array(_DERIVED_OBSERVATION, maximum=30),
})
_FRAME_ENTITY = _object({
    "key": _TEXT,
    "visibility": {"type": "string", "enum": ["visible", "occluded"]},
    "relationship": _TEXT,
})
_FRAME = _object({
    "key": _TEXT,
    "people": _array(_FRAME_PERSON, maximum=30),
    "entities": _array(_FRAME_ENTITY, maximum=100),
    "relationships": _TEXT,
    "crop": _TEXT,
})
SEGMENT_FRAMES_SCHEMA = _object({
    "frames": _array(_FRAME, minimum=9, maximum=9),
})


def prompt_fusion_schema(*, input_sha256: str, segment_count: int) -> dict:
    if (
        isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or segment_count < 1
    ):
        raise ValueError("prompt fusion segment count is invalid")
    segment = _object({
        "index": _INT1,
        "visual": _array(_TEXT, minimum=9, maximum=9),
    })
    return _object({
        "schema": {
            "type": "string",
            "enum": ["duet.video-prompt-fusion-output"],
        },
        "version": {"type": "integer", "enum": [2]},
        "input_sha256": {"type": "string", "enum": [input_sha256]},
        "segments": _array(
            segment, minimum=segment_count, maximum=segment_count,
        ),
    })


def neutralization_schema(
    *, original_prompt_sha256: str, semantic_contract_sha256: str,
) -> dict:
    return _object({
        "version": {"type": "integer", "enum": [1]},
        "original_prompt_sha256": {
            "type": "string", "enum": [original_prompt_sha256],
        },
        "semantic_contract_sha256": {
            "type": "string", "enum": [semantic_contract_sha256],
        },
        "neutralized_free_text": _TEXT,
    })
def _index_records(value: object, *, fields: set[str]) -> dict[str, dict]:
    if not isinstance(value, list):
        raise ValueError("model output collection is invalid")
    indexed: dict[str, dict] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != fields | {"key"}:
            raise ValueError("model output record is invalid")
        key = item.get("key")
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError("model output key is invalid")
        if key in indexed:
            raise ValueError("model output key is duplicated")
        indexed[key] = {name: item[name] for name in fields}
    return indexed


def normalize_project_index(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "people", "entities", "scenes", "relations",
    }:
        raise ValueError("project index output is invalid")
    element_fields = {
        "source_visual_description", "occurrences", "replaceable", "preserve",
    }
    discovered = {
        category: _index_records(value[category], fields=element_fields)
        for category in ("people", "entities", "scenes")
    }
    discovered["relations"] = _index_records(value["relations"], fields={
        "subject_key", "predicate", "object_key", "occurrences", "preserve",
        "replace_together",
    })

    # Model keys are invocation-local references, not backend identities.  Drop
    # records with no visual evidence and assign stable IDs mechanically in the
    # observed order.  Ambiguous or unknown relation aliases remain unresolved
    # and are filtered by the backend relation binder.
    result: dict[str, dict[str, dict]] = {}
    alias_candidates: dict[str, list[str]] = {}
    for category, prefix in (
        ("people", "person"),
        ("entities", "entity"),
        ("scenes", "scene"),
    ):
        retained = [
            (local_key, record)
            for local_key, record in discovered[category].items()
            if record["occurrences"]
        ]
        result[category] = {}
        for index, (local_key, record) in enumerate(retained, 1):
            stable_key = f"{prefix}-{index:02d}"
            result[category][stable_key] = record
            alias_candidates.setdefault(local_key, []).append(stable_key)

    aliases = {
        local_key: candidates[0]
        for local_key, candidates in alias_candidates.items()
        if len(candidates) == 1
    }
    result["relations"] = {}
    retained_relations = [
        record
        for record in discovered["relations"].values()
        if record["occurrences"]
    ]
    for index, record in enumerate(retained_relations, 1):
        result["relations"][f"relation-{index:02d}"] = {
            **record,
            "subject_key": aliases.get(
                record["subject_key"], f"unresolved:{record['subject_key']}",
            ),
            "object_key": aliases.get(
                record["object_key"], f"unresolved:{record['object_key']}",
            ),
        }
    return result


def _reject_global_plan(reason: str, field_path: str) -> None:
    raise CodexOutputValidationError(
        reason, field_path, message="global plan output is invalid",
    )


def normalize_global_plan(
    value: object, *, stable_keys: Mapping[str, Iterable[str]],
) -> dict:
    expected = _global_stable_keys(stable_keys)
    if not isinstance(value, dict) or set(value) != {
        "people", "entities", "scenes", "relations",
    }:
        _reject_global_plan("global_plan_shape_invalid", "/global_plan")
    normalized: dict[str, dict[str, dict]] = {}
    for category, record_schema in _GLOBAL_VALUE_SCHEMAS.items():
        records = value[category]
        if not isinstance(records, dict) or set(records) != set(expected[category]):
            _reject_global_plan("global_plan_keys_invalid", f"/{category}")
        fields = set(record_schema["properties"])
        normalized[category] = {}
        for stable_key in expected[category]:
            record = records[stable_key]
            if not isinstance(record, dict) or set(record) != fields:
                _reject_global_plan("global_plan_record_invalid", f"/{category}")
            for field in fields:
                text = record[field]
                if not isinstance(text, str) or not text.strip():
                    _reject_global_plan(
                        "global_plan_text_invalid", f"/{category}/{field}",
                    )
            normalized[category][stable_key] = {
                field: record[field] for field in fields
            }
    return normalized


def normalize_segment_frames(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"frames"}:
        raise ValueError("segment frames output is invalid")
    frames = _index_records(value["frames"], fields={
        "people", "entities", "relationships", "crop",
    })
    for frame in frames.values():
        frame["people"] = _index_records(frame["people"], fields={
            "visible_region", "boundary", "body_and_pose", "derived_observations",
        })
        for person in frame["people"].values():
            person["derived_observations"] = _index_records(
                person["derived_observations"],
                fields={
                    "mode", "source_carrier", "visible_region", "boundary",
                    "relationship",
                },
            )
        frame["entities"] = _index_records(
            frame["entities"], fields={"visibility", "relationship"},
        )
    return {"frames": frames}
