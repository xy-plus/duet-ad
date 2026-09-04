"""Frozen structured-output contracts for the two ``video-maker`` phases.

The shapes and normalization semantics are copied from
``github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf``.
They describe model transport only; provider records remain attempt-log data.
"""

from __future__ import annotations

from typing import Any, Mapping


def _object(properties: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(
    items: dict[str, Any], *, minimum: int = 0, maximum: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "array",
        "items": items,
        "minItems": minimum,
    }
    if maximum is not None:
        value["maxItems"] = maximum
    return value


_TEXT = {"type": "string"}
_INT0 = {"type": "integer", "minimum": 0}
_INT1 = {"type": "integer", "minimum": 1}
_BOOL = {"type": "boolean"}
_TEXTS = _array(_TEXT)


# Segment identity and ordered frames are backend-owned and absent here.
VISUAL_PROMPT_SCHEMA = _object({
    "prompt": {"type": "string"},
})


_ELEMENT_OCCURRENCE = _object({
    "segment_index": _INT0,
    "frame_orders": _array(_INT1, maximum=9),
})
_ELEMENT = _object({
    "key": _TEXT,
    "source_visual_description": _TEXT,
    "occurrences": _array(_ELEMENT_OCCURRENCE),
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
    "frames": _array(_RELATION_FRAME, maximum=9),
})
_RELATION = _object({
    "key": _TEXT,
    "subject_key": _TEXT,
    "predicate": _TEXT,
    "object_key": _TEXT,
    "occurrences": _array(_RELATION_OCCURRENCE),
    "preserve": _TEXTS,
    "replace_together": _BOOL,
})

PROJECT_INDEX_SCHEMA = _object({
    "people": _array(_ELEMENT),
    "entities": _array(_ELEMENT),
    "scenes": _array(_ELEMENT),
    "relations": _array(_RELATION),
})


def normalize_visual_prompt(value: object) -> str:
    """Return only model-owned prose from the structured transport object."""
    if not isinstance(value, dict) or set(value) != {"prompt"}:
        raise ValueError("visual prompt output is invalid")
    prompt = value["prompt"]
    if not isinstance(prompt, str):
        raise ValueError("visual prompt output is invalid")
    return prompt


def _index_records(
    value: object, *, fields: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("model output collection is invalid")
    indexed: list[tuple[str, dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != fields | {"key"}:
            raise ValueError("model output record is invalid")
        key = item.get("key")
        if not isinstance(key, str):
            raise ValueError("model output key is invalid")
        indexed.append((key, {name: item[name] for name in fields}))
    return indexed


def normalize_project_index(value: object) -> dict[str, dict[str, dict[str, Any]]]:
    """Mechanically replace invocation-local aliases with stable backend keys."""
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

    result: dict[str, dict[str, dict[str, Any]]] = {}
    alias_candidates: dict[str, list[str]] = {}
    for category, prefix in (
        ("people", "person"),
        ("entities", "entity"),
        ("scenes", "scene"),
    ):
        result[category] = {}
        for index, (local_key, record) in enumerate(discovered[category], 1):
            stable_key = f"{prefix}-{index:02d}"
            result[category][stable_key] = record
            alias_candidates.setdefault(local_key, []).append(stable_key)

    aliases = {
        local_key: candidates[0]
        for local_key, candidates in alias_candidates.items()
        if len(candidates) == 1
    }
    result["relations"] = {}
    for index, (_local_key, record) in enumerate(discovered["relations"], 1):
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
