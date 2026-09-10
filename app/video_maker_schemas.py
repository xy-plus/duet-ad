"""Frozen structured-output contracts for the two ``video-maker`` phases.

The shapes and normalization semantics are copied from
``github-xy-plus/Master@043166b13f21fdf715390b964e843be2c010ddcf``.
They describe model transport only; provider records remain attempt-log data.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _object(
    properties: Mapping[str, dict[str, Any]],
    *,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties) if required is None else list(required),
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


def _reference_image_ids(
    value: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError("project-index reference image IDs are invalid")
    values = tuple(value)
    if (
        any(not isinstance(reference_id, str) or not reference_id
               for reference_id in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("project-index reference image IDs are invalid")
    return values


def _project_index_schema(
    segment_index_schema: dict[str, Any],
    identity_evidence: Mapping[str, Any] | None = None,
    reference_image_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    element_occurrence = _object({
        "segment_index": dict(segment_index_schema),
        "frame_orders": _array(_INT1, maximum=9),
    })
    element_properties: dict[str, dict[str, Any]] = {
        "key": _TEXT,
        "source_visual_description": _TEXT,
        "occurrences": _array(element_occurrence),
        "replaceable": _TEXTS,
        "preserve": _TEXTS,
    }
    if reference_image_ids:
        element_properties["target_reference_image_ids"] = _array(
            {"type": "string", "enum": list(reference_image_ids)},
            minimum=1,
            maximum=len(reference_image_ids),
        )
        element_properties["target_reference_image_ids"]["uniqueItems"] = True
    element = _object(
        element_properties,
        required=(
            "key",
            "source_visual_description",
            "occurrences",
            "replaceable",
            "preserve",
        ),
    )
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
    person = element
    if identity_evidence is not None:
        observations = _identity_observations(identity_evidence)
        canonical_ids = sorted({obs["canonical_id"] for obs in observations.values()})
        person = _object({
            **element["properties"],
            "canonical_id": {"type": ["string", "null"], "enum": [*canonical_ids, None]},
            "identity_observation_ids": _array(
                {"type": "string", "enum": list(observations)} if observations else _TEXT,
                maximum=None if observations else 0,
            ),
        }, required=[
            *element["required"],
            "canonical_id",
            "identity_observation_ids",
        ])
    return _object({
        "people": _array(person),
        "entities": _array(element),
        "scenes": _array(element),
        "relations": _array(relation),
    })


# Kept as the general exported contract for static inspection and callers that
# do not yet have a concrete request. Runtime calls use ``project_index_schema``
# so occurrences can only cite segment identities present in that request.
PROJECT_INDEX_SCHEMA = _project_index_schema(_INT1)


def project_index_schema(
    segment_indices: Sequence[int],
    identity_evidence: Mapping[str, Any] | None = None,
    *,
    reference_image_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
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
    return _project_index_schema(
        {"type": "integer", "enum": indices},
        identity_evidence,
        _reference_image_ids(reference_image_ids),
    )


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
    value: object,
    *,
    fields: set[str],
    optional_fields: frozenset[str] = frozenset(),
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("model output collection is invalid")
    indexed: list[tuple[str, dict[str, Any]]] = []
    for item in value:
        if not isinstance(item, dict) or not (fields | {"key"}) <= set(item):
            raise ValueError("model output record is invalid")
        key = item.get("key")
        if not isinstance(key, str):
            raise ValueError("model output key is invalid")
        record = {name: item[name] for name in fields}
        record.update({
            name: item[name]
            for name in optional_fields
            if name in item
        })
        indexed.append((key, record))
    return indexed


def _identity_observations(evidence: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Only backend-confirmed observations of actual model input frames qualify."""
    if evidence.get("status") != "complete":
        return {}
    persons = {
        person["canonical_id"]: set(person.get("observation_ids", []))
        for person in evidence.get("persons", [])
        if isinstance(person, dict) and isinstance(person.get("canonical_id"), str)
    }
    result = {}
    duplicates = set()
    for obs in evidence.get("observations", []):
        if not isinstance(obs, dict) or obs.get("kind") != "keyframe":
            continue
        oid, cid = obs.get("observation_id"), obs.get("canonical_id")
        if not isinstance(oid, str) or not isinstance(cid, str):
            continue
        if oid in result:
            duplicates.add(oid)
        if oid not in persons.get(cid, set()):
            continue
        if any(type(obs.get(field)) is not int or obs[field] < 1
               for field in ("segment_index", "frame_order")):
            continue
        result[oid] = obs
    return {oid: obs for oid, obs in result.items() if oid not in duplicates}


def _person_identity(
    person: Mapping[str, Any], observations: Mapping[str, Mapping[str, Any]],
    occupied_frames: Mapping[tuple[int, int], int],
    observation_claimants: Mapping[str, int],
) -> tuple[str | None, list[str], str]:
    cid, ids = person.get("canonical_id"), person.get("identity_observation_ids")
    unclaimed = cid is None and not ids
    if not unclaimed and (not isinstance(cid, str) or not isinstance(ids, list) or not ids):
        return None, [], "invalid_claim"
    if not unclaimed and any(not isinstance(oid, str) or oid not in observations for oid in ids):
        return None, [], "unknown_observation"
    if not unclaimed and any(observations[oid]["canonical_id"] != cid for oid in ids):
        return None, [], "conflicting_identity"
    if not unclaimed and any(observation_claimants.get(oid, 0) > 1 for oid in ids):
        return None, [], "shared_observation_claim"
    claimed_frames = set()
    occurrences = person.get("occurrences")
    if not isinstance(occurrences, list):
        return None, [], "invalid_occurrences"
    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            return None, [], "invalid_occurrences"
        segment, frames = occurrence.get("segment_index"), occurrence.get("frame_orders")
        if type(segment) is not int or not isinstance(frames, list):
            return None, [], "invalid_occurrences"
        if any(type(frame) is not int for frame in frames):
            return None, [], "invalid_occurrences"
        claimed_frames.update((segment, frame) for frame in frames)
    candidates: dict[tuple[int, int], list[str]] = {}
    for oid, observation in observations.items():
        frame = (observation["segment_index"], observation["frame_order"])
        if frame in claimed_frames:
            candidates.setdefault(frame, []).append(oid)
    exclusive_ids = [oids[0] for frame, oids in candidates.items()
                     if len(oids) == 1 and occupied_frames.get(frame) == 1]
    exclusive_cids = {observations[oid]["canonical_id"] for oid in exclusive_ids}
    if unclaimed:
        # Canonical IDs already passed the CV prepass's multi-observation
        # confirmation; a single unambiguous model keyframe may bind one entry.
        if exclusive_ids and len(exclusive_cids) == 1:
            return next(iter(exclusive_cids)), exclusive_ids, "recovered_single_person_frames"
        return None, [], "unclaimed_conflict" if len(exclusive_cids) > 1 else "unclaimed"
    if exclusive_cids - {cid}:
        return None, [], "conflicting_occurrence_identity"
    observed_frames = {
        (observations[oid]["segment_index"], observations[oid]["frame_order"])
        for oid in ids
    }
    if not claimed_frames or not observed_frames <= claimed_frames:
        return None, [], "occurrence_coverage_mismatch"
    return cid, list(dict.fromkeys(ids)), "confirmed"


def _merge_person(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    descriptions = [target["source_visual_description"], source["source_visual_description"]]
    target["source_visual_description"] = "\n".join(dict.fromkeys(descriptions))
    for field in ("preserve", "replaceable", "identity_observation_ids"):
        target[field] = list(dict.fromkeys([*target[field], *source[field]]))
    frames: dict[int, set[int]] = {}
    for occurrence in [*target["occurrences"], *source["occurrences"]]:
        frames.setdefault(occurrence["segment_index"], set()).update(occurrence["frame_orders"])
    target["occurrences"] = [
        {"segment_index": segment, "frame_orders": sorted(orders)}
        for segment, orders in sorted(frames.items())
    ]


def normalize_project_index(
    value: object,
    identity_evidence: Mapping[str, Any] | None = None,
    *,
    reference_image_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Mechanically replace invocation-local aliases with stable backend keys."""
    if not isinstance(value, dict) or not {
        "people", "entities", "scenes", "relations",
    } <= set(value):
        raise ValueError("project index output is invalid")
    element_fields = {
        "source_visual_description", "occurrences", "replaceable", "preserve",
    }
    allowed_reference_image_ids = _reference_image_ids(reference_image_ids)
    optional_element_fields = (
        frozenset({"target_reference_image_ids"})
        if allowed_reference_image_ids else frozenset()
    )
    discovered = {
        category: _index_records(
            value[category],
            fields=element_fields,
            optional_fields=optional_element_fields,
        )
        for category in ("people", "entities", "scenes")
    }
    discovered["relations"] = _index_records(value["relations"], fields={
        "subject_key", "predicate", "object_key", "occurrences", "preserve",
        "replace_together",
    })
    if allowed_reference_image_ids:
        allowed = set(allowed_reference_image_ids)
        for category in ("people", "entities", "scenes"):
            for _key, record in discovered[category]:
                if "target_reference_image_ids" not in record:
                    continue
                target_reference_image_ids = _reference_image_ids(
                    record["target_reference_image_ids"],
                )
                if not set(target_reference_image_ids) <= allowed:
                    raise ValueError(
                        "project-index target reference image IDs are invalid",
                    )
                record["target_reference_image_ids"] = list(
                    target_reference_image_ids,
                )

    result: dict[str, Any] = {}
    alias_candidates: dict[str, list[str]] = {}
    observations = _identity_observations(identity_evidence) if identity_evidence is not None else {}
    canonical_keys: dict[str, str] = {}
    identity_audit = []
    occupied_frames: dict[tuple[int, int], int] = {}
    observation_claimants: dict[str, int] = {}
    if identity_evidence is not None:
        for person in value["people"]:
            claimed_ids = person.get("identity_observation_ids")
            if isinstance(claimed_ids, list):
                for oid in {item for item in claimed_ids if isinstance(item, str)}:
                    observation_claimants[oid] = observation_claimants.get(oid, 0) + 1
            frames = set()
            for occurrence in person.get("occurrences", []):
                if isinstance(occurrence, dict) and type(occurrence.get("segment_index")) is int:
                    frames.update((occurrence["segment_index"], frame)
                                  for frame in occurrence.get("frame_orders", []) if type(frame) is int)
            for frame in frames:
                occupied_frames[frame] = occupied_frames.get(frame, 0) + 1
    for category, prefix in (
        ("people", "person"),
        ("entities", "entity"),
        ("scenes", "scene"),
    ):
        result[category] = {}
        for index, (local_key, record) in enumerate(discovered[category], 1):
            stable_key = f"{prefix}-{index:02d}"
            canonical_id = None
            if category == "people" and identity_evidence is not None:
                canonical_id, observation_ids, reason = _person_identity(
                    value[category][index - 1], observations, occupied_frames, observation_claimants,
                )
                record = {**record, "canonical_id": canonical_id,
                          "identity_observation_ids": observation_ids}
                stable_key = canonical_keys.get(canonical_id, stable_key)
                identity_audit.append({"input_index": index, "input_key": local_key,
                                       "output_key": stable_key, "reason": reason})
            if canonical_id is not None and canonical_id in canonical_keys:
                _merge_person(result[category][stable_key], record)
            else:
                result[category][stable_key] = record
                if canonical_id is not None:
                    canonical_keys[canonical_id] = stable_key
            alias_candidates.setdefault(local_key, []).append(stable_key)

    aliases = {
        local_key: candidates[0]
        for local_key, candidates in alias_candidates.items()
        if len(set(candidates)) == 1
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
    if identity_evidence is not None:
        result["identity_resolution"] = identity_audit
    return result
