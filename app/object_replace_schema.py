"""Provider-neutral structured contract for project-wide object replacement."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from app.fusion_schema import canonical_input


OBJECT_REPLACE_OUTPUT_NAME = "duet.object-replace-output"
OBJECT_REPLACE_OUTPUT_VERSION = 1
MAX_REFERENCE_AUDIO = 3

_ELEMENT_CATEGORIES = ("people", "entities", "scenes")
_INDEX_KEYS = {*_ELEMENT_CATEGORIES, "relations"}
_PATCH_KEYS = {"matches_user_request", "replacement"}
_REPLACEMENT_KEYS = {"source", "target_visual_description"}
_REPLACEMENT_SOURCES = {"image_1", "generated_variant"}


def _closed_object(properties: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _element_ids(
    element_index: object,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(element_index, dict) or not _INDEX_KEYS <= set(element_index):
        raise ValueError("object-replace global element index is invalid")
    result: dict[str, tuple[str, ...]] = {}
    for category in _ELEMENT_CATEGORIES:
        records = element_index[category]
        if not isinstance(records, dict):
            raise ValueError("object-replace global element index is invalid")
        for stable_id, record in records.items():
            if (
                not isinstance(stable_id, str)
                or not stable_id
                or stable_id != stable_id.strip()
                or not isinstance(record, dict)
            ):
                raise ValueError("object-replace global element index is invalid")
        result[category] = tuple(records)
    if len({stable_id for ids in result.values() for stable_id in ids}) != sum(
        len(ids) for ids in result.values()
    ):
        raise ValueError("object-replace stable element id is ambiguous")
    relations = element_index["relations"]
    if not isinstance(relations, dict):
        raise ValueError("object-replace global element index is invalid")
    return result


def _catalog_ids(voice_ids: Collection[str]) -> tuple[str, ...]:
    if isinstance(voice_ids, (str, bytes)):
        raise ValueError("object-replace voice catalog is invalid")
    values = tuple(voice_ids)
    if (
        not values
        or any(
            not isinstance(voice_id, str)
            or not voice_id
            or voice_id != voice_id.strip()
            for voice_id in values
        )
        or len(values) != len(set(values))
    ):
        raise ValueError("object-replace voice catalog is invalid")
    return values


def _replacement_schema(
    *, voice_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {
        "source": {"type": "string", "enum": sorted(_REPLACEMENT_SOURCES)},
        "target_visual_description": {"type": "string"},
    }
    replacement = _closed_object(properties)
    patch_properties: dict[str, dict[str, Any]] = {
        "matches_user_request": {"type": "boolean"},
        "replacement": {"anyOf": [replacement, {"type": "null"}]},
    }
    if voice_ids is not None:
        patch_properties["voice_id"] = {
            "anyOf": [
                {"type": "string", "enum": list(voice_ids)},
                {"type": "null"},
            ],
        }
    return _closed_object(patch_properties)


def object_replace_schema(
    payload: dict[str, Any], voice_ids: list[str],
) -> tuple[dict[str, Any], str]:
    """Build a closed schema whose element keys are frozen to ``payload``."""
    if not isinstance(payload, dict):
        raise ValueError("object-replace input is invalid")
    element_ids = _element_ids(payload.get("global_element_index"))
    catalog_ids = _catalog_ids(voice_ids)
    input_sha256 = hashlib.sha256(canonical_input(payload)).hexdigest()

    category_schemas: dict[str, dict[str, Any]] = {}
    for category in _ELEMENT_CATEGORIES:
        properties: dict[str, dict[str, Any]] = {}
        for stable_id in element_ids[category]:
            patch_schema = _replacement_schema(
                voice_ids=catalog_ids if category == "people" else None,
            )
            properties[stable_id] = patch_schema
        category_schemas[category] = _closed_object(properties)

    properties = {
        "schema": {"type": "string", "enum": [OBJECT_REPLACE_OUTPUT_NAME]},
        "version": {"type": "integer", "enum": [OBJECT_REPLACE_OUTPUT_VERSION]},
        "input_sha256": {"type": "string", "enum": [input_sha256]},
        **category_schemas,
    }
    return _closed_object(properties), input_sha256


def _normalize_replacement(
    value: object,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _REPLACEMENT_KEYS:
        raise ValueError("object-replace replacement is invalid")
    source = value["source"]
    if not isinstance(source, str) or source not in _REPLACEMENT_SOURCES:
        raise ValueError("object-replace replacement source is invalid")
    description = value["target_visual_description"]
    if not isinstance(description, str):
        raise ValueError("object-replace target description is invalid")
    return {
        "source": source,
        "target_visual_description": description,
    }


def normalize_object_replace(
    value: object,
    *,
    input_sha256: str,
    element_index: dict[str, Any],
    voice_ids: Collection[str],
    has_user_image: bool,
    user_instruction: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalize the closed transport shape without judging model quality."""
    element_ids = _element_ids(element_index)
    catalog_ids = set(_catalog_ids(voice_ids))
    del has_user_image, user_instruction
    expected_top_level = {
        "schema", "version", "input_sha256", *_ELEMENT_CATEGORIES,
    }
    if not isinstance(value, dict) or set(value) != expected_top_level:
        raise ValueError("object-replace output is invalid")
    if (
        value["schema"] != OBJECT_REPLACE_OUTPUT_NAME
        or isinstance(value["version"], bool)
        or value["version"] != OBJECT_REPLACE_OUTPUT_VERSION
        or value["input_sha256"] != input_sha256
    ):
        raise ValueError("object-replace output is not bound to its input")

    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    selected_voice_ids: set[str] = set()
    for category in _ELEMENT_CATEGORIES:
        category_value = value[category]
        if (
            not isinstance(category_value, dict)
            or set(category_value) != set(element_ids[category])
        ):
            raise ValueError("object-replace output element keys are invalid")
        normalized[category] = {}
        expected_patch_keys = (
            _PATCH_KEYS | {"voice_id"} if category == "people" else _PATCH_KEYS
        )
        for stable_id in element_ids[category]:
            patch = category_value[stable_id]
            if not isinstance(patch, dict) or set(patch) != expected_patch_keys:
                raise ValueError("object-replace patch is invalid")
            matches_user_request = patch["matches_user_request"]
            if not isinstance(matches_user_request, bool):
                raise ValueError("object-replace match flag is invalid")
            replacement = _normalize_replacement(patch["replacement"])
            normalized_patch: dict[str, Any] = {
                "matches_user_request": matches_user_request,
                "replacement": replacement,
            }
            if category == "people":
                voice_id = patch["voice_id"]
                normalized_voice_id = (
                    voice_id
                    if isinstance(voice_id, str) and voice_id in catalog_ids
                    else None
                )
                if (
                    normalized_voice_id is not None
                    and normalized_voice_id not in selected_voice_ids
                    and len(selected_voice_ids) >= MAX_REFERENCE_AUDIO
                ):
                    normalized_voice_id = None
                if normalized_voice_id is not None:
                    selected_voice_ids.add(normalized_voice_id)
                normalized_patch["voice_id"] = normalized_voice_id
            normalized[category][stable_id] = normalized_patch
    return normalized


def merge_object_replace_index(
    original: dict[str, Any],
    patch: dict[str, dict[str, dict[str, Any]]],
    voice_metadata_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a normalized patch without exposing source facts to model edits."""
    element_ids = _element_ids(original)
    if not isinstance(patch, dict) or set(patch) != set(_ELEMENT_CATEGORIES):
        raise ValueError("object-replace patch root is invalid")
    if not isinstance(voice_metadata_by_id, Mapping):
        raise ValueError("object-replace voice metadata is invalid")

    merged = copy.deepcopy(original)
    selected_voice_ids: set[str] = set()
    for category in _ELEMENT_CATEGORIES:
        category_patch = patch[category]
        if (
            not isinstance(category_patch, dict)
            or set(category_patch) != set(element_ids[category])
        ):
            raise ValueError("object-replace patch element keys are invalid")
        expected_patch_keys = (
            _PATCH_KEYS | {"voice_id"} if category == "people" else _PATCH_KEYS
        )
        for stable_id in element_ids[category]:
            item = category_patch[stable_id]
            if not isinstance(item, dict) or set(item) != expected_patch_keys:
                raise ValueError("object-replace patch is invalid")
            matches_user_request = item["matches_user_request"]
            if not isinstance(matches_user_request, bool):
                raise ValueError("object-replace match flag is invalid")
            replacement = item["replacement"]
            if replacement is None:
                merged[category][stable_id]["replacement"] = None
            else:
                normalized_replacement = _normalize_replacement(replacement)
                assert normalized_replacement is not None
                merged[category][stable_id]["replacement"] = {
                    "reason": (
                        "user_requested"
                        if matches_user_request
                        else "automatic"
                    ),
                    **normalized_replacement,
                }
            if category == "people":
                voice_id = item["voice_id"]
                if (
                    not isinstance(voice_id, str)
                    or voice_id not in voice_metadata_by_id
                    or (
                        voice_id not in selected_voice_ids
                        and len(selected_voice_ids) >= MAX_REFERENCE_AUDIO
                    )
                ):
                    merged[category][stable_id]["voice"] = None
                    continue
                selected_voice_ids.add(voice_id)
                merged[category][stable_id]["voice"] = copy.deepcopy(
                    voice_metadata_by_id[voice_id],
                )
    return merged
