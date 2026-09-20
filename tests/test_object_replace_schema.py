from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from app.fusion_schema import canonical_input
from app.object_replace_schema import (
    DETECT_LABEL_PATTERN,
    OBJECT_REPLACE_OUTPUT_NAME,
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_planning_context,
    object_replace_schema,
)


VOICE_TYPES = ["voice-a", "voice-b", "voice-c", "voice-d"]
ROOT = Path(__file__).resolve().parents[1]


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
) -> dict[str, Any]:
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
    skill = (ROOT / "skills" / "object-replace" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    for rule in (
        "replacement_set",
        "全部 stable key",
        "同类型但不同身份",
        "至少两个可见维度",
        "只有身体局部可见时",
        "服装保持颜色和风格但改变款式",
        "`scenes` 一律设 `replacement=null`",
        "未匹配实体设 `matches_user_request=false`、`replacement=null`",
        "没有用户图或用户替换要求为空时",
        "不得把绑定传播给无关对象、持有者、背景或关联组件",
        "仅用户点名替换的商品及其联动组件照常替换",
        "people 的每个 stable key 的 `replacement` 必须非空",
        "该 `null` key 仍必须出现在输出中（显式 `null`），不得省略",
        "不分配 `detect_label`，不进图版",
    ):
        assert rule in skill
    for obsolete_or_specific_rule in (
        "所有可见实体都必须替换",
        "缺失或 `null` 属于结构不完整",
        "至少选择两个",
        "其余确实无需替换",
        "手、手臂",
        "虎斑猫",
        "梳毛工具",
        "必须且只能把 Image 1",
    ):
        assert obsolete_or_specific_rule not in skill


def test_fusion_skill_does_not_restore_raw_appearance_for_replacements() -> None:
    skill = (ROOT / "skills" / "h3-max-r2v-prompt" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "形状、体积（含尺寸和比例）及功能只以该参考图为准" in skill
    assert "手指握持、遮挡和接触位置必须由视频模型按新物品几何自然重建" in skill
    assert "不得为了贴合源物品的旧几何而变形、缩放或改变新物品" in skill
    assert "只保持权威块指定的交互事件及其时序" in skill
    assert "只服从权威块的安全不变量" not in skill
    assert "用 raw `preserve` 恢复任何 REPLACE 项的源身份或外观" in skill
    assert "同时遵守对象的 `preserve`" not in skill
    assert "对 `replacement.source=image_1` 的 stable key，权威块会列出它唯一可用的用户图编号" in skill
    assert "不得扩散给未绑定 stable key" in skill
    assert "未被遮盖的瞬态/环境物理效果" in skill
    assert "运动、出现时机、强度和轨迹" in skill
    assert "不得重新编排其形态或出现时机" in skill
    assert "唯一绑定" not in skill
    assert "为每个绑定用户参考图的被替换商品补写一段“属性与用法”" in skill
    assert "不是一张贴图，而是有自身类别、结构、功能和使用方式的完整物品" in skill
    assert "必须符合该物品自身的属性与使用逻辑" in skill
    assert "以商品自身的真实使用方式为准" in skill
    assert "不属于变形或改变产品" in skill
    assert "仍以参考图为唯一基准" in skill
    for over_specific_rule in (
        "出料口",
        "喷头",
        "拿反",
        "功能端必须朝向",
    ):
        assert over_specific_rule not in skill


def test_fusion_skill_requires_faithful_video1_action_description() -> None:
    # 灰块整人+物品 100% 遮盖后动作信号丢失（化妆品项目挤膏动作消失），
    # 把 5fdb7d7 删掉的"描述原视频动作/运动/镜头"指导合并回来：纯描述，
    # 不改编不创作。2026-09-20 修订：禁止嵌入"以 Video 1 为准"类兜底声明
    # （实测会把权威域扩大到外观/身份，H3 向源人物回退）。
    skill = (ROOT / "skills" / "h3-max-r2v-prompt" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "主体开头先写一段对 Video 1 的忠实动作描述" in skill
    assert "动作、运动、姿态变化、画面位置与运动方向" in skill
    assert "镜头运动、机位、构图、速度、节奏和转场" in skill
    assert "描述对象始终是源视频 Video 1 本身，不是替换后的画面" in skill
    assert "不依赖被遮盖区域的可见内容" in skill
    assert "动作事实" in skill
    assert "不是创作或改编指令" in skill
    assert '不得写入任何"以 Video 1 为准"' in skill
    assert "以 Video 1 实际内容为准" not in skill
    assert "本段文字仅为辅助" not in skill
    assert "被替换对象不得写入源外观或源类别特征" in skill
    # 编辑任务框架与后续约束不回退
    assert "这是一次对 Video 1 的编辑任务" in skill
    assert "绝不改编或创作" in skill
    assert "不得新增、重新编排或补充叙事" in skill
    assert "整段动作描述统一由第 1 条承担" in skill


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
            }
            for stable_id in element_index["people"]
        },
        "entities": {
            stable_id: {
                "matches_user_request": stable_id == image_1_key,
                "replacement": {
                    **_replacement(
                        (
                            "image_1"
                            if stable_id == image_1_key
                            else "generated_variant"
                        ),
                        f"new appearance for {stable_id}",
                    ),
                    "detect_label": "object",
                },
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
    voice_types: list[str] = VOICE_TYPES,
    has_user_image: bool = True,
    instruction: str = "replace the held product",
    dialogue_line_count: int = 0,
    reference_image_ids: list[str] | None = None,
) -> dict[str, Any]:
    return normalize_object_replace(
        value,
        input_sha256=digest,
        element_index=element_index,
        voice_types=voice_types,
        has_user_image=has_user_image,
        user_instruction=instruction,
        dialogue_line_count=dialogue_line_count,
        reference_image_ids=reference_image_ids,
    )


def _keep_entities_and_scenes(value: dict[str, Any]) -> None:
    for category in ("entities", "scenes"):
        for patch in value[category].values():
            patch["matches_user_request"] = False
            patch["replacement"] = None


def test_schema_is_closed_bound_and_requires_every_replacement() -> None:
    element_index = _index()
    payload = _payload(element_index)
    schema, digest = object_replace_schema(payload, VOICE_TYPES)

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
    assert OBJECT_REPLACE_OUTPUT_VERSION == 5
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
        "enum": [*VOICE_TYPES, None],
    }

    for category in ("people", "entities", "scenes"):
        category_schema = schema["properties"][category]
        assert category_schema["additionalProperties"] is False
        assert list(category_schema["properties"]) == list(element_index[category])
        assert category_schema["required"] == list(element_index[category])

    for patch_schema in schema["properties"]["people"]["properties"].values():
        image_branch, generated_branch = patch_schema["anyOf"]
        assert image_branch["properties"]["matches_user_request"]["enum"] == [True]
        assert image_branch["properties"]["replacement"]["properties"]["source"]["enum"] == ["image_1"]
        assert generated_branch["properties"]["matches_user_request"]["enum"] == [False]
        assert generated_branch["properties"]["replacement"]["properties"]["source"]["enum"] == ["generated_variant"]
        assert set(patch_schema["properties"]) == {
            "matches_user_request", "replacement",
        }

    for patch_schema in schema["properties"]["entities"]["properties"].values():
        image_branch, keep_branch = patch_schema["anyOf"]
        replacement = image_branch["properties"]["replacement"]
        assert image_branch["properties"]["matches_user_request"]["enum"] == [True]
        assert replacement["properties"]["source"]["enum"] == ["image_1"]
        assert replacement["required"] == [
            "source", "target_visual_description", "appearance_description", "detect_label",
        ]
        assert replacement["properties"]["detect_label"] == {
            "type": "string", "pattern": DETECT_LABEL_PATTERN,
        }
        assert keep_branch["properties"] == {
            "matches_user_request": {"type": "boolean", "enum": [False]},
            "replacement": {"type": "null"},
        }

    for patch_schema in schema["properties"]["scenes"]["properties"].values():
        assert patch_schema["properties"] == {
            "matches_user_request": {"type": "boolean", "enum": [False]},
            "replacement": {"type": "null"},
        }


def test_schema_binds_optional_reference_ids_to_payload_images() -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["user_reference_images"] = [
        {"reference_id": reference_id, "sha256": "a" * 64}
        for reference_id in ("image_1", "image_2", "image_3", "image_4")
    ]

    schema, _digest = object_replace_schema(payload, VOICE_TYPES)

    expected_reference_ids = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["image_1", "image_2", "image_3", "image_4"],
        },
    }
    for category in ("people", "entities"):
        for patch_schema in schema["properties"][category]["properties"].values():
            image_branch = patch_schema["anyOf"][0]
            replacement = image_branch["properties"]["replacement"]
            assert replacement["properties"]["reference_image_ids"] == expected_reference_ids
            if category == "people":
                generated_branch = patch_schema["anyOf"][1]
                assert generated_branch["properties"]["replacement"]["properties"]["reference_image_ids"] == expected_reference_ids
            else:
                keep_branch = patch_schema["anyOf"][1]
                assert keep_branch["properties"]["replacement"] == {"type": "null"}

    for patch_schema in schema["properties"]["scenes"]["properties"].values():
        assert patch_schema["properties"]["replacement"] == {"type": "null"}


@pytest.mark.parametrize(
    "label",
    ["handbag", "camera", "strap", "camera strap", "camera-strap", "a", "handheld fan, portable fan, hand-carry fan"],
)
def test_normalizer_accepts_a_short_generic_detect_label(label: str) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-01"]["replacement"]["detect_label"] = label

    normalized = _normalize(value, digest=digest, element_index=element_index)

    replacement = normalized["entities"]["entity-01"]["replacement"]
    assert replacement["detect_label"] == label
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)
    assert (
        merged["entities"]["entity-01"]["replacement"]["detect_label"] == label
    )


@pytest.mark.parametrize(
    "label",
    [
        "Handbag",
        "the small black leather camera handbag",
        "黑色相机包",
        "handbag!",
        " handbag",
        "handbag ",
        "hand  bag",
        "",
        1,
        None,
        "handbag, bag, purse, clutch",
        "Handbag, bag",
        "handbag, , bag",
        "handbag, bag,",
        "a" * 121,
    ],
)
def test_normalizer_rejects_an_unusable_detect_label(label: Any) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-01"]["replacement"]["detect_label"] = label

    with pytest.raises(ValueError, match="detect label is invalid"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_tolerates_a_missing_detect_label_for_legacy_outputs() -> None:
    # 旧冻结产物（detect_label 引入前）没有该字段：normalize 必须继续放行，
    # 由检测 target 派生环节回退 source_visual_description。
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    for patch in value["entities"].values():
        patch["replacement"].pop("detect_label")

    normalized = _normalize(value, digest=digest, element_index=element_index)

    for patch in normalized["entities"].values():
        assert "detect_label" not in patch["replacement"]
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)
    for record in merged["entities"].values():
        assert "detect_label" not in record["replacement"]


def test_normalizer_and_merge_keep_an_explicit_null_entity_replacement() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-02"] = {
        "matches_user_request": False,
        "replacement": None,
    }

    normalized = _normalize(value, digest=digest, element_index=element_index)
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)

    assert set(normalized["entities"]["entity-02"]) == {
        "matches_user_request", "replacement",
    }
    assert normalized["entities"]["entity-02"]["replacement"] is None
    assert merged["entities"]["entity-02"]["replacement"] is None


def test_normalizer_rejects_a_null_replacement_for_people() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = None

    with pytest.raises(ValueError, match="replacement is invalid"):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_requires_the_explicit_null_entity_key_and_rejects_a_match() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"].pop("entity-02")

    with pytest.raises(ValueError, match="element keys are invalid"):
        _normalize(value, digest=digest, element_index=element_index)

    value = _output(element_index, digest)
    value["entities"]["entity-02"] = {
        "matches_user_request": True,
        "replacement": None,
    }
    with pytest.raises(ValueError, match="null replacement cannot match"):
        _normalize(value, digest=digest, element_index=element_index)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value["entities"]["entity-01"]["replacement"].pop(
                "detect_label"
            ),
            "detect label is missing",
        ),
        (
            lambda value: value["entities"]["entity-01"]["replacement"].update(
                {"detect_label": "a red leather handbag"}
            ),
            "detect label is invalid",
        ),
    ],
)
def test_normalizer_rejects_missing_or_invalid_detect_label_for_current_outputs(
    mutation: Any,
    error: str,
) -> None:
    """Validator ValueError is fed back to the current agent retry loop."""
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": {
            **_replacement("image_1", "entity-01"),
            "detect_label": "object",
        },
    }
    mutation(value)

    with pytest.raises(ValueError, match=error):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            # Current Object Replace always supplies this argument, including
            # the no-user-image case.  Legacy frozen v4 replay omits it.
            reference_image_ids=[],
        )


def test_normalizer_current_output_does_not_require_detect_label_for_people() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)

    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        reference_image_ids=[],
    )

    assert all(
        "detect_label" not in patch["replacement"]
        for patch in normalized["people"].values()
    )


def test_skill_defines_the_same_detect_label_contract_as_the_schema() -> None:
    skill = (
        Path(__file__).resolve().parents[1]
        / "skills" / "object-replace" / "SKILL.md"
    ).read_text(encoding="utf-8")

    for rule in (
        "`detect_label`",
        "源对象",
        "每个短语为 1–3 个",
        "1–3 个同义的英文通用类别短语",
        '用 ", " 分隔',
        "全小写",
        "单数",
        "通用类别短语",
        "每个有非空 `replacement` 的 entities stable key 都必须输出 `detect_label`",
        "瞬态/环境物理效果",
        "不分配 `detect_label`",
        "缺失即为无效输出",
        "开放词汇检测器",
        "必须能被开放词汇检测器识别",
        "handheld fan, portable fan, hand-carry fan",
        "不得含品牌、颜色、材质",
        "people 和 scenes 不得输出该字段",
    ):
        assert rule in skill
    # 代码侧正则与 SKILL 语义约束必须锁在同一批样例上。
    for label in ("handbag", "camera", "strap", "camera strap", "handheld fan, portable fan, hand-carry fan"):
        assert re.fullmatch(DETECT_LABEL_PATTERN, label)
    for label in ("a red leather handbag", "Handbag", "黑色相机包", "handbag, , bag", "handbag, bag,"):
        assert not re.fullmatch(DETECT_LABEL_PATTERN, label)


@pytest.mark.parametrize("source, references, expected", [
    ("image_1", [], None),
    ("image_1", ["image_2", "image_1", "image_2"], ["image_2", "image_1"]),
    ("generated_variant", [], None),
    ("generated_variant", ["image_1", "image_1"], None),
])
def test_model_reference_redundancy_is_normalized_before_sheet(source, references, expected):
    from app.replacement_sheet import build_replacement_sheet_spec, validate_replacement_sheet_spec

    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    for key in ("person-01",):
        value["people"][key]["replacement"].update(source=source, reference_image_ids=references)
        value["people"][key]["matches_user_request"] = source == "image_1"
    normalized = _normalize(value, digest=digest, element_index=element_index, reference_image_ids=["image_1", "image_2"])
    replacement = normalized["people"]["person-01"]["replacement"]
    assert replacement.get("reference_image_ids") == expected
    assert ("reference_image_ids" in replacement) == (expected is not None)
    assert value["people"]["person-01"]["replacement"]["reference_image_ids"] == references
    enhanced = merge_object_replace_index(element_index, normalized, VOICE_TYPES)
    spec = build_replacement_sheet_spec(enhanced)
    validate_replacement_sheet_spec(spec)
    tile = next(tile for tile in spec["tiles"] if tile["stable_key"] == "person-01")
    assert tile.get("reference_image_ids") == expected


def test_normalizer_preserves_multiple_user_reference_ids_per_stable_key() -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["user_reference_images"] = [
        {"reference_id": reference_id}
        for reference_id in ("image_1", "image_2", "image_3", "image_4")
    ]
    _schema, digest = object_replace_schema(payload, VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    value["people"]["person-01"]["replacement"]["reference_image_ids"] = [
        "image_1", "image_3",
    ]
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": {
            **_replacement("image_1", "entity-01"),
            "detect_label": "object",
            "reference_image_ids": ["image_2"],
        },
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        reference_image_ids=["image_1", "image_2", "image_3", "image_4"],
    )
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)

    assert normalized["people"]["person-01"]["replacement"][
        "reference_image_ids"
    ] == ["image_1", "image_3"]
    assert normalized["entities"]["entity-01"]["replacement"][
        "reference_image_ids"
    ] == ["image_2"]
    assert merged["people"]["person-01"]["replacement"][
        "reference_image_ids"
    ] == ["image_1", "image_3"]
    assert merged["entities"]["entity-01"]["replacement"][
        "reference_image_ids"
    ] == ["image_2"]


def test_normalizer_keeps_legacy_image_binding_without_reference_ids() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)

    normalized = _normalize(
        _output(element_index, digest),
        digest=digest,
        element_index=element_index,
    )

    assert "reference_image_ids" not in normalized["people"]["person-01"][
        "replacement"
    ]


@pytest.mark.parametrize(
    "replacement",
    [
        {
            **_replacement("generated_variant", "entity-01"),
            "detect_label": "object",
            "reference_image_ids": ["image_9"],
        },
        {
            **_replacement("image_1", "entity-01"),
            "detect_label": "object",
            "reference_image_ids": ["image_3"],
        },
        {
            **_replacement("image_1", "entity-01"),
            "detect_label": "object",
            "reference_image_ids": ["image_9"],
        },
    ],
)
def test_normalizer_rejects_unusable_reference_image_ids(
    replacement: dict[str, Any],
) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-01"]["replacement"] = replacement

    with pytest.raises(ValueError, match="reference"):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            reference_image_ids=["image_1", "image_2"],
        )


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

    schema, _digest = object_replace_schema(payload, VOICE_TYPES)

    for patch_schema in schema["properties"]["people"]["properties"].values():
        assert "anyOf" not in patch_schema
        assert patch_schema["properties"]["matches_user_request"]["enum"] == [False]
        assert patch_schema["properties"]["replacement"]["properties"]["source"]["enum"] == ["generated_variant"]
    for category in ("entities", "scenes"):
        for patch_schema in schema["properties"][category]["properties"].values():
            assert "anyOf" not in patch_schema
            assert patch_schema["properties"] == {
                "matches_user_request": {"type": "boolean", "enum": [False]},
                "replacement": {"type": "null"},
            }


def test_schema_accepts_an_empty_reference_list_without_a_user_image() -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["has_user_image"] = False
    payload["user_replacement_image"] = None
    payload["user_replacement_instruction"] = ""
    payload["user_reference_images"] = []

    schema, _digest = object_replace_schema(payload, VOICE_TYPES)

    for patch_schema in schema["properties"]["people"]["properties"].values():
        assert "reference_image_ids" not in patch_schema["properties"]["replacement"]["properties"]
    for category in ("entities", "scenes"):
        for patch_schema in schema["properties"][category]["properties"].values():
            assert patch_schema["properties"]["replacement"] == {"type": "null"}


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
def test_schema_ignores_a_replacement_set_not_derived_from_the_index(
    replacement_set: object,
) -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["replacement_set"] = replacement_set

    schema, digest = object_replace_schema(payload, VOICE_TYPES)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert set(schema["properties"]["people"]["properties"]) == {
        "person-01", "person-02",
    }


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

    schema, digest = object_replace_schema(payload, VOICE_TYPES)

    assert digest == hashlib.sha256(canonical_input(payload)).hexdigest()
    assert schema["properties"]["scenes"]["properties"] == {}


def test_normalizer_allows_semantically_imperfect_but_consumable_output() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)

    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", ""),
    }
    value["people"]["person-02"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-02"),
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)

    with pytest.raises(ValueError, match="binding count is invalid"):
        _normalize(
            value,
            digest=digest,
            element_index=element_index,
            has_user_image=False,
        )


def test_normalizer_accepts_zero_image_1_bindings_for_a_nonblank_request() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant",
        "person-01",
    )
    value["people"]["person-01"]["matches_user_request"] = False

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert all(
        patch["replacement"]["source"] == "generated_variant"
        for category in ("people", "entities", "scenes")
        for patch in normalized[category].values()
    )


def test_normalizer_accepts_all_generated_variants_for_multiple_images() -> None:
    element_index = _index()
    payload = _payload(element_index)
    payload["user_reference_images"] = [
        {"reference_id": reference_id}
        for reference_id in ("image_1", "image_2", "image_3", "image_4")
    ]
    _schema, digest = object_replace_schema(payload, VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        reference_image_ids=["image_1", "image_2", "image_3", "image_4"],
    )

    assert all(
        patch["replacement"]["source"] == "generated_variant"
        and "reference_image_ids" not in patch["replacement"]
        for patch in normalized["people"].values()
    )
    assert all(
        patch["replacement"] is None
        for category in ("entities", "scenes")
        for patch in normalized[category].values()
    )


def test_normalizer_accepts_multiple_image_1_bindings() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("image_1", "entity-01"),
    }

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert normalized["people"]["person-01"]["replacement"]["source"] == "image_1"
    assert normalized["entities"]["entity-01"]["replacement"]["source"] == "image_1"
    original = copy.deepcopy(element_index)
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)
    assert element_index == original
    assert merged["relations"] == original["relations"]
    for category in ("people", "entities", "scenes"):
        assert set(merged[category]) == set(original[category])
        for stable_id in original[category]:
            assert merged[category][stable_id]["occurrences"] == original[category][stable_id]["occurrences"]
            assert merged[category][stable_id]["replacement"]["source"] == normalized[category][stable_id]["replacement"]["source"]


def test_normalizer_accepts_image_1_even_when_the_match_flag_is_false() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"]["matches_user_request"] = False

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert normalized["people"]["person-01"]["matches_user_request"] is False
    assert (
        normalized["people"]["person-01"]["replacement"]["source"] == "image_1"
    )


def test_normalizer_accepts_true_match_without_image_1() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"]["replacement"] = _replacement(
        "generated_variant",
        "x",
    )

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert normalized["people"]["person-01"]["matches_user_request"] is True
    assert normalized["people"]["person-01"]["replacement"]["source"] == (
        "generated_variant"
    )


def test_normalizer_accepts_true_match_without_a_user_image() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("generated_variant", "person-01"),
    }

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
    )

    assert normalized["people"]["person-01"]["matches_user_request"] is True
    assert normalized["people"]["person-01"]["replacement"]["source"] == (
        "generated_variant"
    )


def test_normalizer_accepts_zero_bindings_without_a_user_image() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
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
    _schema, digest = object_replace_schema(payload, VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
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


def test_normalizer_rejects_a_nonnull_scene_for_current_outputs() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)
    value["scenes"]["scene-01"]["replacement"] = _replacement(
        "generated_variant", "scene-01",
    )

    with pytest.raises(ValueError, match="scenes must be kept"):
        _normalize(
            value, digest=digest, element_index=element_index,
            reference_image_ids=["image_1"],
        )


def test_current_normalizer_keeps_unmatched_entities_and_requires_user_images() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    _keep_entities_and_scenes(value)

    value["entities"]["entity-01"] = {
        "matches_user_request": False,
        "replacement": {
            **_replacement("generated_variant", "entity-01"),
            "detect_label": "object",
        },
    }
    with pytest.raises(ValueError, match="unmatched entity must be kept"):
        _normalize(
            value, digest=digest, element_index=element_index,
            reference_image_ids=["image_1"],
        )

    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": {
            **_replacement("generated_variant", "entity-01"),
            "detect_label": "object",
        },
    }
    with pytest.raises(ValueError, match="matched entity must use user image"):
        _normalize(
            value, digest=digest, element_index=element_index,
            reference_image_ids=["image_1"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"input_sha256": "0" * 64}),
        lambda value: value.update({"version": True}),
        lambda value: value["people"].pop("person-02"),
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
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    mutation(value)

    with pytest.raises(ValueError):
        _normalize(value, digest=digest, element_index=element_index)


def test_normalizer_tolerates_extra_top_level_keys() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["is_project_index"] = False
    value["notes"] = "stray model field"

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert "is_project_index" not in normalized
    assert "notes" not in normalized
    assert set(normalized) == {
        "people", "entities", "scenes", "dialogue_voices",
    }


def test_normalizer_tolerates_extra_element_and_patch_keys() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["entities"]["entity-99"] = {}
    value["scenes"]["scene-01"]["extra"] = True

    normalized = _normalize(value, digest=digest, element_index=element_index)

    assert set(normalized["entities"]) == {
        "entity-01", "entity-02", "entity-03",
    }
    assert "extra" not in normalized["scenes"]["scene-01"]


def test_normalizer_requires_dict_stable_id_maps() -> None:
    malformed = _index()
    malformed["people"] = []
    with pytest.raises(ValueError, match="global element index"):
        object_replace_schema(_payload(malformed), VOICE_TYPES)

    # A stable id repeated across categories is ambiguous for humans but
    # unambiguous for the per-category contract; it passes through.
    duplicated = _index(entities=("person-01",))
    schema, _digest = object_replace_schema(_payload(duplicated), VOICE_TYPES)
    assert set(schema["properties"]["entities"]["properties"]) == {"person-01"}


def test_merge_safely_copies_without_semantic_review() -> None:
    element_index = _index()
    element_index["relations"]["relation-01"]["object_key"] = "unresolved"
    _schema, digest = object_replace_schema(_payload(element_index), VOICE_TYPES)
    value = _output(element_index, digest)
    value["people"]["person-01"] = {
        "matches_user_request": False,
        "replacement": _replacement("generated_variant", "person-01"),
    }
    value["entities"]["entity-01"] = {
        "matches_user_request": True,
        "replacement": _replacement("image_1", ""),
    }
    normalized = _normalize(value, digest=digest, element_index=element_index)

    original_snapshot = copy.deepcopy(element_index)
    patch_snapshot = copy.deepcopy(normalized)
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)

    assert element_index == original_snapshot
    assert normalized == patch_snapshot
    assert merged is not element_index
    assert merged["relations"] == element_index["relations"]
    assert merged["dialogue_voices"] == []
    assert merged["people"]["person-01"]["replacement"] == {
        "reason": "automatic",
        "source": "generated_variant",
        "target_visual_description": "person-01",
        "appearance_description": "isolated new appearance",
    }
    assert all(
        "voice" not in record for record in merged["people"].values()
    )
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
            assert facts == element_index[category][stable_id]


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
        _dialogue_payload(element_index), VOICE_TYPES,
    )

    dialogue_voices_schema = schema["properties"]["dialogue_voices"]
    assert dialogue_voices_schema["minItems"] == 2
    assert dialogue_voices_schema["maxItems"] == 2
    assert dialogue_voices_schema["items"]["enum"] == [*VOICE_TYPES, None]


def test_dialogue_voices_bind_each_line_to_a_catalog_voice_type() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_TYPES,
    )
    value = _dialogue_output(element_index, digest, [VOICE_TYPES[1], None])

    normalized = _normalize(
        value,
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
        dialogue_line_count=2,
    )
    merged = merge_object_replace_index(element_index, normalized, VOICE_TYPES)

    assert normalized["dialogue_voices"] == [VOICE_TYPES[1], None]
    assert merged["dialogue_voices"] == [VOICE_TYPES[1], None]


@pytest.mark.parametrize(
    "dialogue_voices",
    [
        [VOICE_TYPES[1]],
        [VOICE_TYPES[1], None, None],
        VOICE_TYPES[1],
        [VOICE_TYPES[1], "zh_female_vv_uranus_bigtts"],
        [VOICE_TYPES[1], "unknown-voice"],
        [VOICE_TYPES[1], 1],
    ],
)
def test_normalizer_rejects_invalid_dialogue_voices(
    dialogue_voices: Any,
) -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_TYPES,
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


def test_merge_rejects_dialogue_voices_outside_the_catalog() -> None:
    element_index = _index()
    _schema, digest = object_replace_schema(
        _dialogue_payload(element_index), VOICE_TYPES,
    )
    normalized = _normalize(
        _dialogue_output(element_index, digest, [VOICE_TYPES[0], None]),
        digest=digest,
        element_index=element_index,
        has_user_image=False,
        instruction="",
        dialogue_line_count=2,
    )
    normalized["dialogue_voices"] = [VOICE_TYPES[0], "entity-01"]

    with pytest.raises(ValueError, match="dialogue voices"):
        merge_object_replace_index(element_index, normalized, VOICE_TYPES)
