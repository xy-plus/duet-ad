"""Strict model-output DTOs for the three Codex Skill phases.

The schemas shape model transport only.  Backend validators still bind IDs,
counts, hashes, timelines, and publication.  Maps whose keys are discovered by
the model use arrays with an explicit ``key`` field; the backend indexes them
after checking uniqueness.
"""

from __future__ import annotations

from typing import Iterable, Mapping


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


_ELEMENT_OCCURRENCE = _object({
    "segment_index": _INT0,
    "frame_orders": _array(_INT1, minimum=1, maximum=9),
})
_ELEMENT = _object({
    "key": _TEXT,
    "source_visual_description": _TEXT,
    "occurrences": _array(_ELEMENT_OCCURRENCE, minimum=1, maximum=100),
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
    "occurrences": _array(_RELATION_OCCURRENCE, minimum=1, maximum=100),
    "preserve": _TEXTS,
    "replace_together": _BOOL,
})

PROJECT_INDEX_SCHEMA = _object({
    "people": _array(_ELEMENT, maximum=100),
    "entities": _array(_ELEMENT, maximum=100),
    "scenes": _array(_ELEMENT, maximum=100),
    "relations": _array(_RELATION, maximum=200),
})


_GLOBAL_PERSON = _object({
    "key": _TEXT,
    "source_identity": _TEXT,
    "replacement_identity": _TEXT,
    "wardrobe_change": _TEXT,
    "local_color_change": _TEXT,
})
_GLOBAL_ENTITY = _object({
    "key": _TEXT,
    "description": _TEXT,
    "owner": _TEXT,
    "association": _TEXT,
    "persistence": _TEXT,
})
_GLOBAL_SCENE = _object({
    "key": _TEXT,
    "source_scene": _TEXT,
    "replacement_scene": _TEXT,
    "semantic_change": _TEXT,
    "geometry_change": _TEXT,
    "depth_change": _TEXT,
    "layout_change": _TEXT,
    "local_color_change": _TEXT,
})
_GLOBAL_RELATION = _object({
    "key": _TEXT,
    "replacement_system": _TEXT,
    "preserve": _TEXT,
})
GLOBAL_PLAN_SCHEMA = _object({
    "people": _array(_GLOBAL_PERSON, maximum=100),
    "entities": _array(_GLOBAL_ENTITY, maximum=100),
    "scenes": _array(_GLOBAL_SCENE, maximum=100),
    "relations": _array(_GLOBAL_RELATION, maximum=200),
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
    "frames": _array(_FRAME, minimum=1, maximum=9),
})


def prompt_fusion_schema(*, input_sha256: str, segment_count: int) -> dict:
    segment = _object({
        "index": _INT1,
        "visual": _array(_TEXT, minimum=1, maximum=9),
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
    result = {
        category: _index_records(value[category], fields=element_fields)
        for category in ("people", "entities", "scenes")
    }
    result["relations"] = _index_records(value["relations"], fields={
        "subject_key", "predicate", "object_key", "occurrences", "preserve",
        "replace_together",
    })
    return result


def normalize_global_plan(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "people", "entities", "scenes", "relations",
    }:
        raise ValueError("global plan output is invalid")
    return {
        "people": _index_records(value["people"], fields={
            "source_identity", "replacement_identity", "wardrobe_change",
            "local_color_change",
        }),
        "entities": _index_records(value["entities"], fields={
            "description", "owner", "association", "persistence",
        }),
        "scenes": _index_records(value["scenes"], fields={
            "source_scene", "replacement_scene", "semantic_change",
            "geometry_change", "depth_change", "layout_change",
            "local_color_change",
        }),
        "relations": _index_records(value["relations"], fields={
            "replacement_system", "preserve",
        }),
    }


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
