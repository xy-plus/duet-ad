"""Frozen structured-output contracts for the two ``video-maker`` phases.

The shapes and normalization semantics are copied from
``github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf``.
They describe model transport only; provider records remain attempt-log data.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


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
_INT1 = {"type": "integer", "minimum": 1}
_BOOL = {"type": "boolean"}
_TEXTS = _array(_TEXT)


# Segment identity and ordered frames are backend-owned and absent here.
VISUAL_PROMPT_SCHEMA = _object({
    "prompt": {"type": "string"},
})


_RELATION_FRAME = _object({
    "frame_order": _INT1,
    "state": _TEXT,
    "geometry": _TEXT,
})


def _project_index_schema(
    segment_index_schema: dict[str, Any],
) -> dict[str, Any]:
    element_occurrence = _object({
        "segment_index": dict(segment_index_schema),
        "frame_orders": _array(_INT1, maximum=9),
    })
    element = _object({
        "key": _TEXT,
        "source_visual_description": _TEXT,
        "occurrences": _array(element_occurrence),
        "replaceable": _TEXTS,
        "preserve": _TEXTS,
    })
    relation_occurrence = _object({
        "segment_index": dict(segment_index_schema),
        "frames": _array(_RELATION_FRAME, maximum=9),
    })
    relation = _object({
        "key": _TEXT,
        "subject_key": _TEXT,
        "predicate": _TEXT,
        "object_key": _TEXT,
        "occurrences": _array(relation_occurrence),
        "preserve": _TEXTS,
        "replace_together": _BOOL,
    })
    return _object({
        "people": _array(element),
        "entities": _array(element),
        "scenes": _array(element),
        "relations": _array(relation),
    })


# Kept as the general exported contract for static inspection and callers that
# do not yet have a concrete request. Runtime calls use ``project_index_schema``
# so occurrences can only cite segment identities present in that request.
PROJECT_INDEX_SCHEMA = _project_index_schema(_INT1)


def project_index_schema(segment_indices: Sequence[int]) -> dict[str, Any]:
    """Bind project-index occurrences to the exact input segment identities."""
    indices = list(segment_indices)
    if (
        not indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 1
            for index in indices
        )
        or len(indices) != len(set(indices))
    ):
        raise ValueError("project-index segment identities are invalid")
    return _project_index_schema({"type": "integer", "enum": indices})


def normalize_visual_prompt(value: object) -> str:
    """Return only model-owned prose from the structured transport object.

    Extra keys (models sometimes emit stray fields such as
    ``is_project_index``) are tolerated and dropped instead of killing the
    output; only the ``prompt`` string is load-bearing.
    """
    if not isinstance(value, dict):
        raise ValueError("visual prompt output is invalid")
    prompt = value.get("prompt")
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


def normalize_project_index(value: object) -> dict[str, Any]:
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

    result: dict[str, Any] = {}
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
