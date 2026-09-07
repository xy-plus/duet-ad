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


def _replacement(
    source: str,
    description: str,
    appearance: str = "isolated new appearance",
) -> dict[str, str]:
    return {
        "source": source,
        "target_visual_description": description,
        "appearance_description": appearance,
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
        "至少两个可见维度",
        "只有身体局部可见时",
        "服装保持颜色和风格但改变款式",
        "同类但不同实例",
        "所有可见实体都必须替换",
        "同一个 stable key 在不同图片、不同分段",
        "必须且只能把 Image 1 约束到文字要求唯一语义对应的一个既有 stable key",
        "这两个字段必须双向一致",
        "没有 Image 1 或用户替换要求为空时",
        "不得把参考图背景或无关对象扩散到其他 stable key",
        "其他必换 key 仍各自使用 `generated_variant`",
        "缺失或 `null` 属于结构不完整",
        "后端不对描述做审美判断",
        "stable key 是跨节点、跨图片、跨分段识别同一元素的唯一锚点",
        "`appearance_description` 只写当前 stable key 自身内在、可见的新外观",
        "动作和关系只能写在此字段",
        "只出现该 stable key 本身",
        "只描述该可见局部",
        "不得出现持有者、操作者、配套对象、被作用对象",
        "必须排除索引人物、动物、可移动实体、前景商品",
        "不得决定、声明或重排 TILE",
        "其他 stable key 的新形象不得继承 Image 1",
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
    assert "同时遵守对象的 `preserve`" not in skill


def _output(
    element_index: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    image_1_key = next(
        (
            stable_id
            for category in ("people", "entities", "scenes")
            for stable_id in element_index[category]
        ),
        None,
    )
    return {
        "schema": OBJECT_REPLACE_OUTPUT_NAME,
        "version": OBJECT_REPLACE_OUTPUT_VERSION,
        "input_sha256": digest,
        "dialogue_voices": [],
        "people": {
            stable_id: {
                "matches_user_request": stable_id == image_1_key,
                "replacement": _replacement(
                    (
                        "image_1"
                        if stable_id == image_1_key
                        else "generated_variant"
                    ),
                    f"new appearance for {stable_id}",
                ),
                "voice_id": VOICE_IDS[index % len(VOICE_IDS)],
            }
            for index, stable_id in enumerate(element_index["people"])
        },
        "entities": {
            stable_id: {
                "matches_user_request": stable_id == image_1_key,
                "replacement": _replacement(
                    (
                        "image_1"
                        if stable_id == image_1_key
                        else "generated_variant"
                    ),
                    f"new appearance for {stable_id}",
                ),
            }
            for stable_id in element_index["entities"]
        },
        "scenes": {
            stable_id: {
                "matches_user_request": stable_id == image_1_key,
                "replacement": _replacement(
                    (
                        "image_1"
                        if stable_id == image_1_key
                        else "generated_variant"
                    ),
                    f"new layout for {stable_id}",
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
    dialogue_line_count: int = 0,
) -> dict[str, Any]:
    return normalize_object_replace(
        value,
        input_sha256=digest,
        element_index=element_index,
        voice_ids=voice_ids,
        has_user_image=has_user_image,
        user_instruction=instruction,
        dialogue_line_count=dialogue_line_count,
    )


def test_schema_is_closed_bound_and_requires_every_replacement() -> None:
    element_index = _index()
    payload = _payload(element_index)
    schema, digest = object_replace_schema(payload, VOICE_IDS)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema",
        "version",
        "input_sha256",
        "people",
        "entities",
        "scenes",
        "dialogue_voices",
    ]
    assert OBJECT_REPLACE_OUTPUT_VERSION == 4
    assert schema["properties"]["input_sha256"]["enum"] == [digest]
    assert set(schema["properties"]) == {
        "schema",
        "version",
        "input_sha256",
        "people",
        "entities",
        "scenes",
        "dialogue_voices",
    }
    dialogue_voices_schema = schema["properties"]["dialogue_voices"]
    assert dialogue_voices_schema["type"] == "array"
    assert dialogue_voices_schema["minItems"] == 0
    assert dialogue_voices_schema["maxItems"] == 0
    assert dialogue_voices_schema["items"] == {
        "type": ["string", "null"],
        "enum": ["person-01", "person-02", None],
    }

    for category in ("people", "entities", "scenes"):
        category_schema = schema["properties"][category]
        assert category_schema["additionalProperties"] is False
        assert list(category_schema["properties"]) == list(element_index[category])
        assert category_schema["required"] == list(element_index[category])
        for patch_schema in category_schema["properties"].values():
            assert patch_schema["additionalProperties"] is False
            replacement_object = patch_schema["properties"]["replacement"]
            assert replacement_object["type"] == "object"
            assert replacement_object["additionalProperties"] is False
            assert replacement_object["properties"]["source"]["enum"] == [
                "generated_variant", "image_1",
            ]
            assert replacement_object["properties"][
                "target_visual_description"
            ] == {"type": "string"}
            assert replacement_object["properties"][
                "appearance_description"
            ] == {"type": "string"}
            assert replacement_object["required"] == [
                "source",
                "target_visual_description",
                "appearance_description",
            ]
            if category == "people":
                assert patch_schema["properties"]["voice_id"]["anyOf"] == [
                    {"type": "string", "enum": VOICE_IDS},
                    {"type": "null"},
                ]

            image_branch, generated_branch = patch_schema["anyOf"]
            assert image_branch["additionalProperties"] is False
            assert image_branch["properties"]["matches_user_request"][
                "enum"
            ] == [True]
            assert image_branch["properties"]["replacement"]["properties"][
                "source"
            ]["enum"] == ["image_1"]
            assert generated_branch["additionalProperties"] is False
            assert generated_branch["properties"]["matches_user_request"][
                "enum"
            ] == [False]
            assert generated_branch["properties"]["replacement"]["properties"][
                "source"
            ]["enum"] == ["generated_variant"]


@pytest.mark.parametrize(
    ("has_user_image", "instruction"),
    [(False, "replace the held product"), (True, "  \n\t  ")],
)
def test_schema_forbids_image_1_without_an_explicit_image_request(
    has_user_image: bool,
    instruction: str,
) -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["has_user_image"] = has_user_image
    payload["user_replacement_instruction"] = instruction
    if not has_user_image:
        payload["user_replacement_image"] = None

    schema, _digest = object_replace_schema(payload, VOICE_IDS)

    for category in ("people", "entities", "scenes"):
        for patch_schema in schema["properties"][category]["properties"].values():
            assert "anyOf" not in patch_schema
            assert patch_schema["properties"]["matches_user_request"][
                "enum"
            ] == [False]
            assert patch_schema["properties"]["replacement"]["properties"][
                "source"
            ]["enum"] == ["generated_variant"]


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
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", ""),
        "voice_id": None,
    }
    value["people"]["person-02"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-02"),
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", ""),
    }
    value["entities"]["entity-02"]["replacement"] = _replacement(
        "generated_variant",
        "entity-02",
    )
    value["entities"]["entity-03"]["replacement"] = _replacement(
        "generated_variant",
        "entity-03",
    )
    value["scenes"]["scene-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", ""),
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
    )

    assert normalized == {
        **{
            category: value[category]
            for category in ("people", "entities", "scenes")
        },
        "dialogue_voices": [],
    }


def test_normalizer_trims_a_renderable_appearance_without_a_quality_gate() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["entities"]["entity-01"]["replacement"] = _replacement(
        "generated_variant",
        "entity-01 performs its source action",
        "  x  ",
    )

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert normalized["entities"]["entity-01"]["replacement"][
        "appearance_description"
    ] == "x"


@pytest.mark.parametrize("appearance", ["", "   \n\t  "])
def test_normalizer_rejects_an_empty_trimmed_appearance(
    appearance: str,
) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["entities"]["entity-01"]["replacement"] = _replacement(
        "generated_variant",
        "entity-01",
        appearance,
    )

    with pytest.raises(ValueError, match="appearance description is empty"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_does_not_limit_description_length_or_compare_source() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "image_1", "person-01",
    )
    value["scenes"]["scene-01"]["replacement"] = _replacement(
        "generated_variant", "x" * 10_000,
    )

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert len(
        normalized["scenes"]["scene-01"]["replacement"][
            "target_visual_description"
        ]
    ) == 10_000


def test_normalizer_rejects_image_1_without_a_user_image() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)

    with pytest.raises(ValueError, match="binding count is invalid"):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            has_user_image=False,
        )


def test_normalizer_rejects_zero_image_1_bindings_for_a_nonblank_request() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant",
        "person-01",
    )
    value["people"]["person-01"]["matches_user_request"] = False

    with pytest.raises(ValueError, match="binding count is invalid"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_rejects_multiple_image_1_bindings() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("image_1", "entity-01"),
    }

    with pytest.raises(ValueError, match="binding count is invalid"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_rejects_image_1_when_the_match_flag_is_false() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["matches_user_request"] = False

    with pytest.raises(ValueError, match="match flag conflicts"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_rejects_true_match_without_image_1() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant",
        "x",
    )

    with pytest.raises(ValueError, match="match flag conflicts"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_rejects_true_match_without_a_user_image() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("generated_variant", "person-01"),
        "voice_id": None,
    }

    with pytest.raises(ValueError, match="match flag conflicts"):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            has_user_image=False,
        )


def test_normalizer_accepts_zero_bindings_without_a_user_image() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
        "voice_id": None,
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
    )

    assert normalized["people"]["person-01"]["matches_user_request"] is False


def test_normalizer_accepts_zero_bindings_for_a_blank_instruction() -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["user_replacement_instruction"] = "  \n\t  "
    _schema, digest = object_replace_schema(payload, VOICE_IDS)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
        "voice_id": None,
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        instruction="  \n\t  ",
    )

    assert normalized["people"]["person-01"]["matches_user_request"] is False


def test_normalizer_accepts_exactly_one_matching_image_1_binding() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    normalized = _normalize(value, digest=digest, element_index=element_index)

    replacements = [
        patch["replacement"]
        for category in ("people", "entities", "scenes")
        for patch in normalized[category].values()
    ]
    assert [
        replacement["source"] for replacement in replacements
    ].count("image_1") == 1
    assert all(
        replacement["source"] == "generated_variant"
        for replacement in replacements
        if replacement["source"] != "image_1"
    )
    assert normalized["people"]["person-01"]["matches_user_request"] is True


@pytest.mark.parametrize(
    ("category", "stable_id"),
    [
        ("people", "person-02"),
        ("entities", "entity-02"),
        ("scenes", "scene-01"),
    ],
)
def test_normalizer_rejects_a_null_replacement_for_any_stable_key(
    category: str,
    stable_id: str,
) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_IDS)
    value = _output(element_index, digest)
    value[category][stable_id]["replacement"] = None

    with pytest.raises(ValueError, match="replacement is invalid"):
        _normalize(value, digest=digest, element_index=element_index)


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
            {
                "replacement": {
                    "source": "unknown",
                    "target_visual_description": "x",
                    "appearance_description": "x",
                },
            },
        ),
        lambda value: value["entities"]["entity-01"].update(
            {
                "replacement": {
                    "source": "image_1",
                    "target_visual_description": 1,
                    "appearance_description": "x",
                },
            },
        ),
        lambda value: value["entities"]["entity-01"]["replacement"].pop(
            "appearance_description",
        ),
        lambda value: value["entities"]["entity-01"]["replacement"].update(
            {"appearance_description": 1},
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
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
        "voice_id": None,
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
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
    assert merged["dialogue_voices"] == []
    assert merged["people"]["person-01"]["replacement"] == {
        "reason": "automatic",
        "source": "generated_variant",
        "target_visual_description": "person-01",
        "appearance_description": "isolated new appearance",
    }
    assert merged["people"]["person-01"]["voice"] is None
    assert merged["people"]["person-02"]["voice"] == voice_metadata["voice-b"]
    assert merged["people"]["person-02"]["voice"] is not voice_metadata["voice-b"]
    assert merged["entities"]["entity-01"]["replacement"] == {
        "reason": "user_requested",
        "source": "image_1",
        "target_visual_description": "",
        "appearance_description": "isolated new appearance",
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


def _dialogue_payload(element_index: dict[str, Any]) -> dict[str, Any]:
    payload = _payload(element_index)
    payload["has_user_image"] = False
    payload["user_replacement_instruction"] = ""
    payload["user_replacement_image"] = None
    payload["translated_dialogue"] = [
        {"text": "第一句", "start_s": 0.0, "end_s": 1.0},
        {"text": "第二句", "start_s": 1.0, "end_s": 2.0},
    ]
    return payload


def _dialogue_output(
    element_index: dict[str, Any],
    digest: str,
    dialogue_voices: Any,
) -> dict[str, Any]:
    value = _output(element_index, digest)
    for category in ("people", "entities", "scenes"):
        for patch in value[category].values():
            patch["matches_user_request"] = False
            patch["replacement"]["source"] = "generated_variant"
    value["dialogue_voices"] = dialogue_voices
    return value


def test_schema_locks_dialogue_voices_to_the_translated_lines() -> None:
    element_index = _index()
    schema, _digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_IDS,
    )

    dialogue_voices_schema = schema["properties"]["dialogue_voices"]
    assert dialogue_voices_schema["minItems"] == 2
    assert dialogue_voices_schema["maxItems"] == 2
    assert dialogue_voices_schema["items"]["enum"] == [
        "person-01", "person-02", None,
    ]


def test_dialogue_voices_attribute_each_line_to_a_person_or_narration() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_IDS,
    )
    value = _dialogue_output(element_index, digest, ["person-02", None])

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
        dialogue_line_count=2,
    )
    merged = merge_object_replace_index(element_index, normalized, {})

    assert normalized["dialogue_voices"] == ["person-02", None]
    assert merged["dialogue_voices"] == ["person-02", None]


@pytest.mark.parametrize(
    "dialogue_voices",
    [
        ["person-02"],
        ["person-02", None, None],
        "person-02",
        ["person-02", "entity-01"],
        ["person-02", "unknown-person"],
        ["person-02", 1],
    ],
)
def test_normalizer_rejects_invalid_dialogue_voices(
    dialogue_voices: Any,
) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_IDS,
    )
    value = _dialogue_output(element_index, digest, dialogue_voices)

    with pytest.raises(ValueError, match="dialogue voices"):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            has_user_image=False,
            instruction="",
            dialogue_line_count=2,
        )


def test_merge_rejects_dialogue_voices_outside_the_people_index() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_IDS,
    )
    normalized = _normalize(
        _dialogue_output(element_index, digest, ["person-01", None]),
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
        dialogue_line_count=2,
    )
    normalized["dialogue_voices"] = ["person-01", "entity-01"]

    with pytest.raises(ValueError, match="dialogue voices"):
        merge_object_replace_index(element_index, normalized, {})
