from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from app.fusion_schema import canonical_input
from app.object_replace_schema import (
    MAX_REFERENCE_AUDIO,
    OBJECT_REPLACE_OUTPUT_NAME,
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_planning_context,
    object_replace_schema,
)


VOICE_IDS = ["voice-a", "voice-b", "voice-c", "voice-d"]


def _element(description: str) -> dict[str, Any]:
    return {
        "source_visual_description": description,
        "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
        "replaceable": ["appearance"],
        "preserve": ["geometry"],
    }


def _index(
    *,
    people: tuple[str, ...] = ("person-01", "person-02"),
    entities: tuple[str, ...] = ("entity-01", "entity-02", "entity-03"),
    scenes: tuple[str, ...] = ("scene-01",),
) -> dict[str, Any]:
    relations: dict[str, dict[str, Any]] = {}
    if people and entities:
        relations["relation-01"] = {
            "subject_key": people[0],
            "predicate": "holds",
            "object_key": entities[0],
            "occurrences": [{
                "segment_index": 1,
                "frames": [{
                    "frame_order": 1,
                    "state": "held",
                    "geometry": "right hand",
                }],
            }],
            "preserve": ["hand contact"],
            "replace_together": True,
        }
    return {
        "people": {stable_id: _element(stable_id) for stable_id in people},
        "entities": {stable_id: _element(stable_id) for stable_id in entities},
        "scenes": {stable_id: _element(stable_id) for stable_id in scenes},
        "relations": relations,
    }


def _replacement(source: str, description: str) -> dict[str, str]:
    return {
        "source": source,
        "target_visual_description": description,
    }


def _payload(element_index: dict[str, Any]) -> dict[str, Any]:
    return {
        **object_replace_planning_context(element_index),
        "segment_visual_prompts": [{"segment_index": 1, "visual_prompt": "source"}],
        "user_replacement_instruction": "replace the held product",
        "has_user_image": True,
        "user_replacement_image": {"name": "Image 1", "sha256": "a" * 64},
        "target_language": "English",
        "translated_dialogue": [],
        "voice_catalog_sha256": "b" * 64,
        "voice_catalog_lines": "catalog text is model context, not a local gate",
    }


def test_planning_context_freezes_all_keys_and_removes_only_item_hints() -> None:
    element_index = _index(
        people=("person-02", "person-01"),
        entities=("entity-03", "entity-01", "entity-02"),
        scenes=("scene-02", "scene-01"),
    )
    original = copy.deepcopy(element_index)

    context = object_replace_planning_context(element_index)

    assert context["replacement_set"] == {
        "people": ["person-01", "person-02"],
        "entities": ["entity-01", "entity-02", "entity-03"],
        "scenes": ["scene-01", "scene-02"],
    }
    planning_index = context["global_element_index"]
    for category in ("people", "entities", "scenes"):
        for stable_id, record in planning_index[category].items():
            assert "preserve" not in record
            assert "replaceable" not in record
            expected = copy.deepcopy(element_index[category][stable_id])
            expected.pop("preserve")
            expected.pop("replaceable")
            assert record == expected
    assert planning_index["relations"] == element_index["relations"]
    assert element_index == original


def test_skill_makes_the_deterministic_set_the_complete_replacement_plan() -> None:
    skill = (
        Path("/home/xy/duet-ad1-h3-max-r2v/skills/object-replace/SKILL.md")
        .read_text(encoding="utf-8")
    )

    for rule in (
        "replacement_set",
        "全部 stable key",
        "同类型但不同身份",
        "人脸或可见身体",
        "只有身体局部可见时",
        "服装保持颜色和风格但改变款式",
        "同类但不同实例",
        "所有可见实体都必须替换",
        "同一个 stable key 在不同图片、不同分段",
        "只把 Image 1 约束到文字要求唯一语义对应的一个既有 stable key",
        "不得把参考图背景或无关对象扩散到其他 stable key",
        "其他必换 key 仍各自使用 `generated_variant`",
        "不把这条语义职责变成质量门禁",
    ):
        assert rule in skill
    for obsolete_or_specific_rule in (
        "至少选择两个",
        "其余确实无需替换",
        "手、手臂",
        "虎斑猫",
        "梳毛工具",
    ):
        assert obsolete_or_specific_rule not in skill


def test_fusion_skill_does_not_restore_raw_appearance_for_replacements() -> None:
    skill = Path(
        "/home/xy/duet-ad1-h3-max-r2v/skills/h3-max-r2v-prompt/SKILL.md"
    ).read_text(encoding="utf-8")

    assert "只服从 `authoritative_constraints` 编译出的安全不变量" in skill
    assert "不得用 raw `preserve` 恢复源身份或外观" in skill
    assert "raw `preserve` 只直接用于 `replacement=null` 的 KEEP 项" in skill
    assert "同时遵守对象的 `preserve`" not in skill


def _output(
    element_index: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    return {
        "schema": OBJECT_REPLACE_OUTPUT_NAME,
        "version": OBJECT_REPLACE_OUTPUT_VERSION,
        "input_sha256": digest,
        "people": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new appearance for {stable_id}",
                ),
                "voice_id": VOICE_IDS[index % len(VOICE_IDS)],
            }
            for index, stable_id in enumerate(element_index["people"])
        },
        "entities": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new appearance for {stable_id}",
                ),
            }
            for stable_id in element_index["entities"]
        },
        "scenes": {
            stable_id: {
                "matches_user_request": False,
                "replacement": _replacement(
                    "generated_variant", f"new layout for {stable_id}",
                ),
            }
            for stable_id in element_index["scenes"]
        },
    }


def _normalize(
    value: object,
    *,
    digest: str,
    element_index: dict[str, Any],
    voice_ids: list[str] = VOICE_IDS,
    has_user_image: bool = True,
    instruction: str = "replace the held product",
) -> dict[str, dict[str, dict[str, Any]]]:
    return normalize_object_replace(
        value,
        input_sha256=digest,
        element_index=element_index,
        voice_ids=voice_ids,
        has_user_image=has_user_image,
        user_instruction=instruction,
    )


def test_schema_is_closed_bound_and_all_quality_choices_are_nullable() -> None:
    element_index = _index()
    payload = _payload(element_index)
    schema, digest = object_replace_schema(payload, VOICE_IDS)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema", "version", "input_sha256", "people", "entities", "scenes",
    ]
    assert schema["properties"]["input_sha256"]["enum"] == [digest]
    assert "relations" not in schema["properties"]

    for category in ("people", "entities", "scenes"):
        category_schema = schema["properties"][category]
        assert category_schema["additionalProperties"] is False
        assert list(category_schema["properties"]) == list(element_index[category])
        assert category_schema["required"] == list(element_index[category])
        for patch_schema in category_schema["properties"].values():
            assert patch_schema["additionalProperties"] is False
            replacement = patch_schema["properties"]["replacement"]
            assert replacement["anyOf"][1] == {"type": "null"}
            replacement_object = replacement["anyOf"][0]
            assert replacement_object["properties"]["source"]["enum"] == [
                "generated_variant", "image_1",
            ]
            assert replacement_object["properties"][
                "target_visual_description"
            ] == {"type": "string"}
            if category == "people":
                assert patch_schema["properties"]["voice_id"]["anyOf"] == [
                    {"type": "string", "enum": VOICE_IDS},
                    {"type": "null"},
                ]


@pytest.mark.parametrize(
    "replacement_set",
    [
        None,
        {"people": [], "entities": [], "scenes": []},
        {
            "people": ["person-01", "person-02"],
            "entities": ["entity-01", "entity-02", "entity-03"],
            "scenes": ["scene-01", "unknown-scene"],
        },
    ],
)
def test_schema_rejects_a_replacement_set_not_derived_from_the_index(
    replacement_set: object,
) -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["replacement_set"] = replacement_set

    with pytest.raises(ValueError, match="replacement set is invalid"):
        object_replace_schema(payload, VOICE_IDS)


def test_schema_does_not_revalidate_upstream_content_or_user_pairing() -> None:
    element_index = _index(scenes=())
    element_index["people"]["person-01"] = {
        "source_visual_description": "",
        "occurrences": "not locally reviewed",
        "unexpected": True,
    }
    element_index["relations"]["relation-01"]["object_key"] = "missing-object"
    element_index["extra_upstream_field"] = {"kept": True}
    payload = _payload(element_index)
    payload["has_user_image"] = False
    payload["user_replacement_image"] = None
    payload["segment_visual_prompts"] = []
    payload["extra_context"] = "forward this too"

    schema, digest = object_replace_schema(payload, VOICE_IDS)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["properties"]["scenes"]["properties"] == {}


def test_normalizer_allows_semantically_imperfect_but_consumable_output() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)

    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": None,
        "voice_id": None,
    }
    value["people"]["person-02"] = {
        "matches_user_request": False,
        "replacement": _replacement("image_1", "person-02"),
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("generated_variant", ""),
    }
    value["entities"]["entity-02"]["replacement"] = None
    value["entities"]["entity-03"]["replacement"] = None
    value["scenes"]["scene-01"] = {
        "matches_user_request": False,
        "replacement": None,
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
    )

    assert normalized == {
        category: value[category] for category in ("people", "entities", "scenes")
    }


def test_normalizer_does_not_limit_description_length_or_compare_source() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant", "person-01",
    )
    value["scenes"]["scene-01"]["replacement"] = _replacement(
        "image_1", "x" * 10_000,
    )

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert len(
        normalized["scenes"]["scene-01"]["replacement"][
            "target_visual_description"
        ]
    ) == 10_000


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"input_sha256": "0" * 64}),
        lambda value: value.update({"version": True}),
        lambda value: value.update({"extra": True}),
        lambda value: value["people"].pop("person-02"),
        lambda value: value["entities"].update({"entity-99": {}}),
        lambda value: value["scenes"]["scene-01"].update({"extra": True}),
        lambda value: value["people"]["person-01"].update(
            {"matches_user_request": None},
        ),
        lambda value: value["entities"]["entity-01"].update(
            {"replacement": {"source": "unknown", "target_visual_description": "x"}},
        ),
        lambda value: value["entities"]["entity-01"].update(
            {"replacement": {"source": "image_1", "target_visual_description": 1}},
        ),
    ],
)
def test_normalizer_rejects_only_unusable_transport_shapes(mutation: Any) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    mutation(value)

    with pytest.raises(ValueError):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_requires_unambiguous_stable_id_maps() -> None:
    malformed = _index()
    malformed["people"] = []
    with pytest.raises(ValueError, match="global element index"):
        object_replace_schema(_payload(malformed), VOICE_IDS)

    ambiguous = _index(entities=("person-01",))
    with pytest.raises(ValueError, match="stable element id is ambiguous"):
        object_replace_schema(_payload(ambiguous), VOICE_IDS)


def test_normalizer_degrades_null_or_unmapped_voice_to_no_voice() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["voice_id"] = None
    _normalize(value, digest=digest, element_index=element_index)

    value["people"]["person-01"]["voice_id"] = "unknown"
    normalized = _normalize(value, digest=digest, element_index=element_index)
    assert normalized["people"]["person-01"]["voice_id"] is None


def test_normalizer_degrades_fourth_unique_voice_in_stable_person_order() -> None:
    four_people = _index(
        people=("person-01", "person-02", "person-03", "person-04"),
        entities=(),
    )
    _schema, digest = object_replace_schema(_payload(four_people), VOICE_IDS)
    value = _output(four_people, digest)
    assert MAX_REFERENCE_AUDIO == 3
    normalized = _normalize(value, digest=digest, element_index=four_people)

    assert [
        normalized["people"][stable_id]["voice_id"]
        for stable_id in four_people["people"]
    ] == ["voice-a", "voice-b", "voice-c", None]


def test_merge_safely_copies_without_semantic_review() -> None:
    element_index = _index()
    element_index["relations"]["relation-01"]["object_key"] = "unresolved"
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": None,
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("image_1", ""),
    }
    normalized = _normalize(value, digest=digest, element_index=element_index)

    original_snapshot = copy.deepcopy(element_index)
    patch_snapshot = copy.deepcopy(normalized)
    voice_metadata = {
        "voice-b": {"file": "voices/b.wav", "features": {"gender": "male"}},
    }
    metadata_snapshot = copy.deepcopy(voice_metadata)
    merged = merge_object_replace_index(element_index, normalized, voice_metadata)

    assert element_index == original_snapshot
    assert normalized == patch_snapshot
    assert voice_metadata == metadata_snapshot
    assert merged is not element_index
    assert merged["relations"] == element_index["relations"]
    assert merged["people"]["person-01"]["replacement"] is None
    assert merged["people"]["person-01"]["voice"] is None
    assert merged["people"]["person-02"]["voice"] == voice_metadata["voice-b"]
    assert merged["people"]["person-02"]["voice"] is not voice_metadata["voice-b"]
    assert merged["entities"]["entity-01"]["replacement"] == {
        "reason": "automatic",
        "source": "image_1",
        "target_visual_description": "",
    }
    for category in ("people", "entities", "scenes"):
        for stable_id, merged_record in merged[category].items():
            facts = copy.deepcopy(merged_record)
            facts.pop("replacement")
            if category == "people":
                facts.pop("voice")
            assert facts == element_index[category][stable_id]


def test_merge_degrades_missing_backend_voice_metadata_to_no_voice() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    normalized = _normalize(
        _output(element_index, digest),
        digest=digest,
        element_index=element_index,
    )
    merged = merge_object_replace_index(element_index, normalized, {})
    assert all(person["voice"] is None for person in merged["people"].values())

    for person in normalized["people"].values():
        person["voice_id"] = None
    merged = merge_object_replace_index(element_index, normalized, {})
    assert all(person["voice"] is None for person in merged["people"].values())
