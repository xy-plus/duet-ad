"""Provider-neutral structured contract for project-wide object replacement."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from app.fusion_schema import canonical_input


OBJECT_REPLACE_OUTPUT_NAME = "duet.object-replace-output"
# v5：音色选择从本地音色库 voice_id 切换为豆包 voice_type。dialogue_voices
# 每项从人物 stable key 变为目录内 voice_type（或 null），people patch 不再
# 携带 voice_id，后端也不再物化本地 wav 参考。
# v6：人种一致性规则（skills/object-replace/SKILL.md 替换规划）。人物替换
# 默认保持源人物的性别、年龄段、人种和肤色；目标台词语言典型人种与源人物
# 不一致时以目标语言为准，dialogue_voices 音色同步匹配；人种/肤色不计入
# 「至少两个可见维度差异」的差异维度。schema 字段不变。
# 旧 v5/v4 冻结产物在版本门禁处明确报错，不做静默兼容。
OBJECT_REPLACE_OUTPUT_VERSION = 6

# entities replacement 的 detect_label：最多 3 个同义英文通用类别短语；每个
# 短语为 1–3 个全小写词，短语间只能用 ", " 分隔，总长最多 120。它是物体
# 检测节点定位源对象的 prompt；skills/object-replace/SKILL.md「检测标签」一节
# 是同一约束的语义面，两侧样例由测试锁定一致。
DETECT_LABEL_PATTERN = r"^(?=.{1,120}$)[a-z]+(?:[ -][a-z]+){0,2}(?:, [a-z]+(?:[ -][a-z]+){0,2}){0,2}$"
_DETECT_LABEL_RE = re.compile(DETECT_LABEL_PATTERN)

_ELEMENT_CATEGORIES = ("people", "entities", "scenes")
_INDEX_KEYS = {*_ELEMENT_CATEGORIES, "relations"}
_PATCH_KEYS = {"matches_user_request", "replacement"}
_REPLACEMENT_KEYS = {
    "source",
    "target_visual_description",
    "appearance_description",
}
_REPLACEMENT_FIELDS = (
    "source",
    "target_visual_description",
    "appearance_description",
)
# ``generated_variant`` is the automatic people identity replacement source;
# current entities/scenes cannot select it.
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
                or not isinstance(record, dict)
            ):
                raise ValueError("object-replace global element index is invalid")
        result[category] = tuple(records)
    relations = element_index["relations"]
    if not isinstance(relations, dict):
        raise ValueError("object-replace global element index is invalid")
    return result


def object_replace_planning_context(
    element_index: dict[str, Any],
) -> dict[str, Any]:
    """Build the model-facing index view and its deterministic replace set."""
    element_ids = _element_ids(element_index)
    planning_index = copy.deepcopy(element_index)
    for category in _ELEMENT_CATEGORIES:
        for stable_id in element_ids[category]:
            record = planning_index[category][stable_id]
            record.pop("preserve", None)
            record.pop("replaceable", None)
    return {
        "global_element_index": planning_index,
        "replacement_set": {
            category: sorted(element_ids[category])
            for category in _ELEMENT_CATEGORIES
        },
    }


def _voice_types(voice_types: Collection[str]) -> tuple[str, ...]:
    if isinstance(voice_types, (str, bytes)):
        raise ValueError("object-replace voice catalog is invalid")
    values = tuple(voice_types)
    if (
        not values
        or any(
            not isinstance(voice_type, str) or not voice_type
            for voice_type in values
        )
        or len(values) != len(set(values))
    ):
        raise ValueError("object-replace voice catalog is invalid")
    return values


def _reference_image_ids(
    value: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("object-replace reference image IDs are invalid")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("object-replace reference image IDs are invalid") from exc
    if (
        (not values and not allow_empty)
        or any(not isinstance(reference_id, str) or not reference_id
               for reference_id in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("object-replace reference image IDs are invalid")
    return values


def _payload_reference_image_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    references = payload.get("user_reference_images")
    if references is None:
        return ("image_1",) if payload.get("has_user_image") is True else ()
    if not isinstance(references, list):
        raise ValueError("object-replace reference images are invalid")
    return _reference_image_ids([
        reference.get("reference_id")
        if isinstance(reference, Mapping) else None
        for reference in references
    ], allow_empty=True)


def _replacement_object(
    *,
    sources: Collection[str],
    reference_image_ids: Sequence[str] | None = None,
    detect_label: bool = False,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {
        "source": {"type": "string", "enum": sorted(sources)},
        "target_visual_description": {"type": "string"},
        "appearance_description": {"type": "string"},
    }
    required = list(_REPLACEMENT_FIELDS)
    if detect_label:
        properties["detect_label"] = {
            "type": "string",
            "pattern": DETECT_LABEL_PATTERN,
        }
        required = [*required, "detect_label"]
    if reference_image_ids is not None:
        properties["reference_image_ids"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(reference_image_ids)},
        }
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _replacement_schema(
    *,
    allow_user_image_binding: bool,
    reference_image_ids: Sequence[str],
    detect_label: bool = False,
    allow_null: bool = False,
    force_null: bool = False,
    allow_generated_variant: bool = False,
) -> dict[str, Any]:
    if force_null:
        return _closed_object({
            "matches_user_request": {"type": "boolean", "enum": [False]},
            "replacement": {"type": "null"},
        })
    replacement_sources = {"image_1"} if allow_user_image_binding else set()
    if allow_generated_variant:
        replacement_sources.add("generated_variant")
    replacement = _replacement_object(
        sources=replacement_sources,
        reference_image_ids=(
            reference_image_ids if reference_image_ids else None
        ),
        detect_label=detect_label,
    )
    # The schema cannot classify an entity as an unrequested animal or a
    # transient physical effect. It therefore admits explicit KEEP only for
    # entities; Object Replace instructions own those semantic restrictions.
    replacement_value: dict[str, Any] = (
        {"anyOf": [replacement, {"type": "null"}]}
        if allow_null
        else replacement
    )
    patch_properties: dict[str, dict[str, Any]] = {
        "matches_user_request": (
            {"type": "boolean"}
            if allow_user_image_binding
            else {"type": "boolean", "enum": [False]}
        ),
        "replacement": replacement_value,
    }
    if not allow_user_image_binding:
        if allow_generated_variant:
            return _closed_object(patch_properties)
        return _closed_object({
            **patch_properties,
            "matches_user_request": {"type": "boolean", "enum": [False]},
            "replacement": {"type": "null"},
        })
    patch_schema = _closed_object(patch_properties)
    if allow_user_image_binding:
        image_patch_properties = copy.deepcopy(patch_properties)
        image_patch_properties["matches_user_request"] = {
            "type": "boolean",
            "enum": [True],
        }
        image_patch_properties["replacement"] = _replacement_object(
            sources={"image_1"},
            reference_image_ids=reference_image_ids,
            detect_label=detect_label,
        )

        patch_branches = [_closed_object(image_patch_properties)]
        if allow_generated_variant:
            generated_patch_properties = copy.deepcopy(patch_properties)
            generated_patch_properties["matches_user_request"] = {
                "type": "boolean",
                "enum": [False],
            }
            generated_patch_properties["replacement"] = _replacement_object(
                sources={"generated_variant"},
                reference_image_ids=reference_image_ids,
                detect_label=detect_label,
            )
            patch_branches.append(_closed_object(generated_patch_properties))
        if allow_null:
            keep_patch_properties = copy.deepcopy(patch_properties)
            keep_patch_properties["matches_user_request"] = {
                "type": "boolean",
                "enum": [False],
            }
            keep_patch_properties["replacement"] = {"type": "null"}
            patch_branches.append(_closed_object(keep_patch_properties))
        patch_schema["anyOf"] = patch_branches
    return patch_schema


def object_replace_schema(
    payload: dict[str, Any], voice_types: list[str],
) -> tuple[dict[str, Any], str]:
    """Build a closed schema whose element keys are frozen to ``payload``."""
    if not isinstance(payload, dict):
        raise ValueError("object-replace input is invalid")
    element_ids = _element_ids(payload.get("global_element_index"))
    translated_dialogue = payload.get("translated_dialogue")
    if not isinstance(translated_dialogue, list):
        raise ValueError("object-replace translated dialogue is invalid")
    catalog_voice_types = _voice_types(voice_types)
    input_sha256 = hashlib.sha256(canonical_input(payload)).hexdigest()
    user_instruction = payload.get("user_replacement_instruction")
    reference_image_ids = _payload_reference_image_ids(payload)
    allow_user_image_binding = (
        bool(reference_image_ids)
        and isinstance(user_instruction, str)
        and bool(user_instruction.strip())
    )

    category_schemas: dict[str, dict[str, Any]] = {}
    for category in _ELEMENT_CATEGORIES:
        properties: dict[str, dict[str, Any]] = {}
        for stable_id in element_ids[category]:
            patch_schema = _replacement_schema(
                allow_user_image_binding=allow_user_image_binding,
                reference_image_ids=reference_image_ids,
                # 只有 entities 会被派生成物体检测 target，detect_label 因此
                # 只约束 entities；people/scenes 的封闭 schema 直接拒绝该字段。
                detect_label=category == "entities",
                allow_null=category != "people",
                force_null=category == "scenes",
                allow_generated_variant=category == "people",
            )
            properties[stable_id] = patch_schema
        category_schemas[category] = _closed_object(properties)

    properties = {
        "schema": {"type": "string", "enum": [OBJECT_REPLACE_OUTPUT_NAME]},
        "version": {"type": "integer", "enum": [OBJECT_REPLACE_OUTPUT_VERSION]},
        "input_sha256": {"type": "string", "enum": [input_sha256]},
        **category_schemas,
        "dialogue_voices": {
            "type": "array",
            "minItems": len(translated_dialogue),
            "maxItems": len(translated_dialogue),
            "items": {
                "type": ["string", "null"],
                "enum": [*catalog_voice_types, None],
            },
        },
    }
    return _closed_object(properties), input_sha256


def _normalize_replacement(
    value: object,
    *,
    allowed_reference_image_ids: Collection[str] | None = None,
    require_detect_label: bool = False,
    allow_null: bool = False,
) -> dict[str, Any] | None:
    if value is None:
        if allow_null:
            return None
        raise ValueError("object-replace replacement is invalid")
    if not isinstance(value, dict) or not _REPLACEMENT_KEYS <= set(value):
        raise ValueError("object-replace replacement is invalid")
    source = value["source"]
    if not isinstance(source, str) or source not in _REPLACEMENT_SOURCES:
        raise ValueError("object-replace replacement source is invalid")
    description = value["target_visual_description"]
    if not isinstance(description, str):
        raise ValueError("object-replace target description is invalid")
    appearance_description = value["appearance_description"]
    if not isinstance(appearance_description, str):
        raise ValueError("object-replace appearance description is invalid")
    appearance_description = appearance_description.strip()
    if not appearance_description:
        raise ValueError("object-replace appearance description is empty")
    normalized: dict[str, Any] = {
        "source": source,
        "target_visual_description": description,
        "appearance_description": appearance_description,
    }
    if "detect_label" not in value:
        if require_detect_label:
            raise ValueError("object-replace detect label is missing")
    else:
        # 若存在必须可用：防 agent 写长句/中文，坏标签直接拒收重试。
        detect_label = value["detect_label"]
        if not isinstance(detect_label, str) or not _DETECT_LABEL_RE.fullmatch(
            detect_label
        ):
            raise ValueError("object-replace detect label is invalid")
        normalized["detect_label"] = detect_label
    if "reference_image_ids" not in value:
        return normalized
    # Normalize redundant model output only; configured IDs and frozen sheet
    # receipts keep their strict validators and original hashes.
    model_reference_ids = value["reference_image_ids"]
    if isinstance(model_reference_ids, list) and all(
        isinstance(item, str) for item in model_reference_ids
    ):
        model_reference_ids = list(dict.fromkeys(model_reference_ids))
    reference_image_ids = _reference_image_ids(model_reference_ids, allow_empty=True)
    if not reference_image_ids:
        return normalized
    if allowed_reference_image_ids is not None:
        allowed = set(_reference_image_ids(allowed_reference_image_ids, allow_empty=True))
        if not set(reference_image_ids) <= allowed:
            raise ValueError("object-replace replacement reference IDs are invalid")
    if source == "image_1":
        normalized["reference_image_ids"] = list(reference_image_ids)
    return normalized


def _normalize_dialogue_voices(
    value: object,
    *,
    dialogue_line_count: int,
    voice_types: tuple[str, ...],
) -> list[str | None]:
    if (
        isinstance(dialogue_line_count, bool)
        or not isinstance(dialogue_line_count, int)
        or dialogue_line_count < 0
        or not isinstance(value, list)
        or len(value) != dialogue_line_count
        or any(
            voice is not None
            and (not isinstance(voice, str) or voice not in voice_types)
            for voice in value
        )
    ):
        raise ValueError("object-replace dialogue voices are invalid")
    return list(value)


def normalize_object_replace(
    value: object,
    *,
    input_sha256: str,
    element_index: dict[str, Any],
    voice_types: Collection[str],
    has_user_image: bool,
    user_instruction: str,
    dialogue_line_count: int,
    reference_image_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Normalize the closed transport shape without judging model quality.

    ``reference_image_ids`` is supplied by every current Object Replace run,
    including runs without a user image.
    """
    element_ids = _element_ids(element_index)
    catalog_voice_types = _voice_types(voice_types)
    require_entity_detect_labels = reference_image_ids is not None
    allowed_reference_image_ids = (
        _reference_image_ids(reference_image_ids, allow_empty=True)
        if reference_image_ids is not None else None
    )
    expected_top_level = {
        "schema",
        "version",
        "input_sha256",
        *_ELEMENT_CATEGORIES,
        "dialogue_voices",
    }
    # Extra top-level keys are tolerated and dropped; only the frozen
    # contract keys below carry meaning.
    if not isinstance(value, dict) or not expected_top_level <= set(value):
        raise ValueError("object-replace output is invalid")
    if (
        value["schema"] != OBJECT_REPLACE_OUTPUT_NAME
        or type(value["version"]) is not int
        or value["version"] != OBJECT_REPLACE_OUTPUT_VERSION
        or value["input_sha256"] != input_sha256
    ):
        raise ValueError("object-replace output is not bound to its input")
    normalized: dict[str, Any] = {}
    image_1_locations: list[tuple[str, str]] = []
    for category in _ELEMENT_CATEGORIES:
        category_value = value[category]
        if (
            not isinstance(category_value, dict)
            or not set(element_ids[category]) <= set(category_value)
        ):
            raise ValueError("object-replace output element keys are invalid")
        normalized[category] = {}
        for stable_id in sorted(element_ids[category]):
            patch = category_value[stable_id]
            if not isinstance(patch, dict) or not _PATCH_KEYS <= set(patch):
                raise ValueError("object-replace patch is invalid")
            matches_user_request = patch["matches_user_request"]
            if not isinstance(matches_user_request, bool):
                raise ValueError("object-replace match flag is invalid")
            replacement = _normalize_replacement(
                patch["replacement"],
                allowed_reference_image_ids=allowed_reference_image_ids,
                require_detect_label=(
                    category == "entities" and require_entity_detect_labels
                ),
                allow_null=category != "people",
            )
            # Current requests always pass reference_image_ids, including an
            # empty list; omitting it marks a pre-detect_label replay where
            # entity labels and the KEEP-only policies are not enforceable.
            if reference_image_ids is not None and category == "scenes" and (
                replacement is not None or matches_user_request
            ):
                raise ValueError("object-replace scenes must be kept")
            if category == "entities" and reference_image_ids is not None:
                if not matches_user_request and replacement is not None:
                    raise ValueError("object-replace unmatched entity must be kept")
                if matches_user_request and (
                    replacement is None or replacement["source"] != "image_1"
                ):
                    raise ValueError(
                        "object-replace matched entity must use user image"
                    )
            if replacement is None:
                if matches_user_request:
                    raise ValueError(
                        "object-replace null replacement cannot match user request"
                    )
            elif replacement["source"] == "image_1":
                image_1_locations.append((category, stable_id))
            normalized_patch: dict[str, Any] = {
                "matches_user_request": matches_user_request,
                "replacement": replacement,
            }
            normalized[category][stable_id] = normalized_patch
    allows_image_1 = has_user_image and bool(user_instruction.strip())
    if not allows_image_1 and image_1_locations:
        raise ValueError("object-replace user-image binding count is invalid")
    normalized["dialogue_voices"] = _normalize_dialogue_voices(
        value["dialogue_voices"],
        dialogue_line_count=dialogue_line_count,
        voice_types=catalog_voice_types,
    )
    return normalized


def merge_object_replace_index(
    original: dict[str, Any],
    patch: dict[str, Any],
    voice_types: Collection[str],
) -> dict[str, Any]:
    """Merge a normalized patch without exposing source facts to model edits."""
    element_ids = _element_ids(original)
    catalog_voice_types = _voice_types(voice_types)

    merged = copy.deepcopy(original)
    for category in _ELEMENT_CATEGORIES:
        category_patch = patch[category]
        for stable_id in element_ids[category]:
            item = category_patch[stable_id]
            matches_user_request = item["matches_user_request"]
            normalized_replacement = _normalize_replacement(
                item["replacement"],
                allow_null=category != "people",
            )
            if normalized_replacement is None:
                if matches_user_request:
                    raise ValueError(
                        "object-replace null replacement cannot match user request"
                    )
                merged[category][stable_id]["replacement"] = None
            else:
                merged[category][stable_id]["replacement"] = {
                    "reason": (
                        "user_requested" if matches_user_request else "automatic"
                    ),
                    **normalized_replacement,
                }
    dialogue_voices = patch["dialogue_voices"]
    if not isinstance(dialogue_voices, list) or any(
        voice is not None
        and (not isinstance(voice, str) or voice not in catalog_voice_types)
        for voice in dialogue_voices
    ):
        raise ValueError("object-replace patch dialogue voices are invalid")
    merged["dialogue_voices"] = list(dialogue_voices)
    return merged
