import hashlib
import json
from types import SimpleNamespace

import pytest

from app import context_ir_bridge, h3, image_optimization, long_generation


def _indexed_item(description: str) -> dict:
    return {
        "source_visual_description": description,
        "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
        "replaceable": ["design"],
        "preserve": ["count", "role"],
    }


def test_image_compiler_freezes_relation_outside_truncated_descriptions() -> None:
    segment_specs = [{
        "index": 1,
        "chain_id": "chain-01",
        "join_mode": "hard_cut",
        "transition_skeleton": [{
            "frame_index": 1,
            "frame_name": "01.png",
            "source_transition_from_previous": "start",
        }],
    }]
    element_index = {
        "people": {},
        "entities": {"entity-01": _indexed_item("portable tool")},
        "scenes": {"scene-01": _indexed_item("work area")},
        "relations": {"relation-01": {
            "subject_key": "entity-01",
            "predicate": "安装于",
            "object_key": "scene-01",
            "occurrences": [{
                "segment_index": 1,
                "frames": [{
                    "frame_order": 1,
                    "state": "已安装并保持连接",
                    "geometry": "位于环境左侧固定接口",
                }],
            }],
            "preserve": ["有向安装角色", "数量一一对应"],
            "replace_together": True,
        }},
    }
    semantics = {
        "people": {},
        "entities": {"entity-01": {
            "description": "a distinct portable replacement tool",
            "owner": "project",
            "association": "belongs to the current project",
            "persistence": "one persistent instance",
        }},
        "relations": {"relation-01": {
            "replacement_system": "compatible replacement interface",
            "preserve": "keep the directed installation",
        }},
        "scenes": {"scene-01": {
            "source_scene": "source work area",
            "replacement_scene": "different work area",
            "semantic_change": "different purpose-equivalent environment",
            "geometry_change": "different visible geometry",
            "depth_change": "different depth layout",
            "layout_change": "different functional layout",
            "local_color_change": "different local material colors",
        }},
        "frames": {"frame-001": {
            "people": {},
            "entities": {"entity-01": {
                "visibility": "visible",
                "relationship": "mounted at the visible interface",
            }},
            "relationships": "directly visible mounting",
            "crop": "current source crop",
        }},
    }

    plan, _diagnostics = image_optimization.compile_semantic_plan(
        semantics, segment_specs, element_index=element_index,
    )
    frame = plan["segments"][0]["frame_constraints"][0]

    assert frame["relation_occurrences"] == [{
        "relation_id": "relation-01",
        "subject_key": "entity-01",
        "predicate": "安装于",
        "object_key": "scene-01",
        "state": "已安装并保持连接",
        "geometry": "位于环境左侧固定接口",
        "preserve": [
            "replacement_system=compatible replacement interface",
            "keep the directed installation",
            "有向安装角色", "数量一一对应",
        ],
        "replace_together": True,
    }]
    image_prompt = image_optimization.compile_frame_prompts(
        plan, "anchor_consistency",
    )[1][1]
    assert "replacement_system=compatible replacement interface" in image_prompt

    timeline = [{
        "order": order,
        "segment_time_s": float(order - 1),
        "source_scene_id": "scene-01",
        "transition": {
            "type": "start" if order == 1 else "continuous",
            "at_segment_s": 0.0 if order == 1 else None,
        },
    } for order in range(1, 10)]
    fusion_occurrence = [{
        **frame["relation_occurrences"][0],
        "frame": {
            "order": 1,
            "segment_time_s": 0.0,
            "source_scene_id": "scene-01",
        },
    }]
    h3_prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=["The visible replacement system keeps its current composition."],
        timeline=timeline,
        lines=[],
        music_policy="forbid",
        relation_occurrences=fusion_occurrence,
    )
    relation_marker = context_ir_bridge._relation_states_contract(h3_prompt)
    assert relation_marker is not None
    assert "replacement_system=compatible replacement interface" in relation_marker
    assert len(relation_marker) <= long_generation._MAX_RELATION_MARKER_CHARS
    assert {
        "subject_id": "ENTITY_01",
        "predicate": "installs",
        "object_id": "SCENE_01",
    } in frame["non_person_entity_ledger"]["relations"]
    description = frame["non_person_entity_ledger"]["entities"][0]["description"]
    assert "relation-01" not in description
    assert "relation_occurrences=" in image_optimization.compile_frame_prompts(
        plan, "anchor_consistency",
    )[1][1]


def test_image_compiler_preserves_person_to_person_contact_relation() -> None:
    segment_specs = [{
        "index": 1,
        "chain_id": "chain-01",
        "join_mode": "hard_cut",
        "transition_skeleton": [{
            "frame_index": 1,
            "frame_name": "01.png",
            "source_transition_from_previous": "start",
        }],
    }]
    element_index = {
        "people": {
            "person-01": _indexed_item("left person"),
            "person-02": _indexed_item("right person"),
        },
        "entities": {},
        "scenes": {"scene-01": _indexed_item("shared room")},
        "relations": {"relation-01": {
            "subject_key": "person-01",
            "predicate": "clasps_hands_with",
            "object_key": "person-02",
            "occurrences": [{
                "segment_index": 1,
                "frames": [{
                    "frame_order": 1,
                    "state": "两人双手接触",
                    "geometry": "两人相邻站立，手部位于画面中央",
                }],
            }],
            "preserve": ["人物身份与接触方向"],
            "replace_together": False,
        }},
    }
    semantics = {
        "people": {
            key: {
                "source_identity": key,
                "replacement_identity": f"replacement {key}",
                "wardrobe_change": "change wardrobe",
                "local_color_change": "change local colors",
            }
            for key in ("person-01", "person-02")
        },
        "entities": {},
        "relations": {"relation-01": {
            "replacement_system": "keep both people compatible",
            "preserve": "keep the visible hand contact",
        }},
        "scenes": {"scene-01": {
            "source_scene": "shared room",
            "replacement_scene": "new shared room",
            "semantic_change": "same function",
            "geometry_change": "new visible geometry",
            "depth_change": "new depth",
            "layout_change": "new layout",
            "local_color_change": "new colors",
        }},
        "frames": {"frame-001": {
            "people": {
                key: {
                    "visible_region": "full visible person",
                    "boundary": "visible body boundary",
                    "body_and_pose": "standing and reaching a hand",
                    "derived_observations": {},
                }
                for key in ("person-01", "person-02")
            },
            "entities": {},
            "relationships": "two visible people clasp hands",
            "crop": "source crop",
        }},
    }

    plan, _diagnostics = image_optimization.compile_semantic_plan(
        semantics, segment_specs, element_index=element_index,
    )
    frame = plan["segments"][0]["frame_constraints"][0]

    assert frame["relation_occurrences"][0]["subject_key"] == "person-01"
    assert frame["relation_occurrences"][0]["object_key"] == "person-02"
    assert {
        "subject_id": "PERSON_01",
        "predicate": "contacts",
        "object_id": "PERSON_02",
    } in frame["non_person_entity_ledger"]["relations"]


def _timeline() -> list[dict]:
    return [{
        "order": order,
        "segment_time_s": float(order - 1),
        "source_scene_id": "scene-a" if order < 5 else "scene-b",
        "transition": (
            {"type": "start", "at_segment_s": 0.0}
            if order == 1 else
            {"type": "hard_cut", "at_segment_s": 3.5}
            if order == 5 else
            {"type": "continuous", "at_segment_s": None}
        ),
    } for order in range(1, 10)]


def _occurrence(
    relation_id: str, subject: str, object_: str, frame_order: int, state: str,
) -> dict:
    frame = _timeline()[frame_order - 1]
    return {
        "relation_id": relation_id,
        "subject_key": subject,
        "predicate": "持有并操作",
        "object_key": object_,
        "state": state,
        "geometry": f"direct geometry {frame_order}",
        "preserve": ["role", "count"],
        "replace_together": True,
        "frame": {
            "order": frame_order,
            "segment_time_s": frame["segment_time_s"],
            "source_scene_id": frame["source_scene_id"],
        },
    }


def test_real_regression_shapes_do_not_mix_or_repair_upstream_relations() -> None:
    # c389-shaped evidence keeps two independently identified objects distinct.
    c389 = [
        _occurrence("relation-01", "person-01", "entity-01", 1, "held"),
        _occurrence("relation-02", "person-01", "entity-02", 2, "operated"),
        _occurrence("relation-01", "person-01", "entity-01", 5, "released"),
    ]
    frozen = long_generation._freeze_fusion_relation_occurrences(
        c389, _timeline(),
    )
    projected = long_generation._expected_fusion_relation_states(
        _timeline(), frozen,
    )
    first_relations = projected[0]["relations"]
    assert [(item["relation_id"], item["object_key"]) for item in first_relations] == [
        ("relation-01", "entity-01"),
        ("relation-02", "entity-02"),
    ]
    assert projected[1]["relations"][0]["states"] == [{
        "frame_order": 5,
        "state": "released",
        "geometry": "direct geometry 5",
    }]

    # A 6320-shaped wrong upstream edge is faithfully transported; Fusion does
    # not silently "correct" it into another object or add a second binding.
    wrong_upstream = [
        _occurrence("relation-01", "person-01", "entity-wrong", 1, "held"),
    ]
    wrong_projected = long_generation._expected_fusion_relation_states(
        _timeline(),
        long_generation._freeze_fusion_relation_occurrences(
            wrong_upstream, _timeline(),
        ),
    )
    assert wrong_projected[0]["relations"][0]["object_key"] == "entity-wrong"
    assert len(wrong_projected[0]["relations"]) == 1

    # A relation may describe two roles of the same indexed subject (for
    # example, a person touching their own hair).  Transport preserves that
    # evidence instead of applying a physical-meaning veto.
    self_relation = [
        _occurrence("relation-01", "person-01", "person-01", 1, "touching hair"),
    ]
    self_projected = long_generation._expected_fusion_relation_states(
        _timeline(),
        long_generation._freeze_fusion_relation_occurrences(
            self_relation, _timeline(),
        ),
    )
    assert self_projected[0]["relations"][0]["subject_key"] == "person-01"
    assert self_projected[0]["relations"][0]["object_key"] == "person-01"

    encoded = json.dumps(
        long_generation._compact_h3_relation_contract(projected),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    assert len(encoded) < 2_000
    assert hashlib.sha256(encoded.encode()).hexdigest()
    assert long_generation._expand_h3_relation_contract(
        json.loads(encoded)
    ) == projected


def test_c389_and_6320_exact_directed_relation_text_round_trips() -> None:
    timeline = _timeline()
    exact_cases = (
        {
            "relation_id": "relation-04",
            "subject_key": "entity-04",
            "predicate": "装配于并从其释放",
            "object_key": "entity-01",
            "preserve": [
                "接口中心对齐", "装配后可释放的功能关系",
                "从接合到分离的状态变化",
            ],
            "replace_together": True,
            "states": [
                (1, "陀螺放置在发射器顶部接口", "两者中心轴近似重合"),
                (2, "陀螺被向下按在黄色接口上", "陀螺底面贴近发射器顶缘"),
                (3, "陀螺保持装配", "陀螺位于发射器正上方"),
                (4, "装配体被移至地面上方", "两者仍同轴，整体竖直"),
                (5, "陀螺已与发射器分离并在地面运动", "两者之间出现明显水平和垂直间隔"),
            ],
        },
        {
            "relation_id": "relation-03",
            "subject_key": "entity-04",
            "predicate": "接合并脱离",
            "object_key": "entity-01",
            "preserve": [
                "接合阶段的同轴对准", "接口直径和尺度匹配",
                "脱离后两个实体的独立性",
            ],
            "replace_together": True,
            "states": [
                (1, "圆形旋转玩具被放置到顶部接口", "entity-04同轴位于entity-01黄色齿状接口上方"),
                (2, "圆形旋转玩具压入顶部", "两个部件保持竖直同轴并直接接触"),
                (3, "继续接合", "entity-04位于entity-01正上方且接口重合"),
                (5, "已经脱离", "entity-04在地板上旋转，entity-01仍在人物手中"),
            ],
        },
    )
    for case in exact_cases:
        occurrences = []
        for frame_order, state, geometry in case["states"]:
            frame = timeline[frame_order - 1]
            occurrences.append({
                key: case[key]
                for key in (
                    "relation_id", "subject_key", "predicate", "object_key",
                    "preserve", "replace_together",
                )
            } | {
                "state": state,
                "geometry": geometry,
                "frame": {
                    "order": frame_order,
                    "segment_time_s": frame["segment_time_s"],
                    "source_scene_id": frame["source_scene_id"],
                },
            })
        projected = long_generation._expected_fusion_relation_states(
            timeline,
            long_generation._freeze_fusion_relation_occurrences(
                occurrences, timeline,
            ),
        )
        contract = long_generation._compact_h3_relation_contract(projected)
        assert long_generation._expand_h3_relation_contract(contract) == projected
        definition = contract["d"][0]
        assert contract["e"][definition[0]] == case["subject_key"]
        assert contract["e"][definition[2]] == case["object_key"]
        assert contract["q"][definition[1]] == case["predicate"]


def test_sixty_relations_with_540_unique_verbose_states_fit_visual_budget() -> None:
    timeline = [{
        "order": order,
        "segment_time_s": float(order - 1),
        "source_scene_id": "scene-a",
        "transition": (
            {"type": "start", "at_segment_s": 0.0}
            if order == 1 else
            {"type": "continuous", "at_segment_s": None}
        ),
    } for order in range(1, 10)]
    occurrences = [{
        "relation_id": f"relation-{relation:02d}",
        "subject_key": f"entity-{relation:02d}",
        "predicate": "contacts",
        "object_key": "scene-a",
        "state": (
            f"unique-state-{frame}-{relation} released and fully separated "
            "after the visible action"
        ),
        "geometry": (
            f"unique-geometry-{frame}-{relation} with a visible gap and "
            "separation across the composition"
        ),
        "preserve": ["direction", "count"],
        "replace_together": relation % 2 == 0,
        "frame": {
            "order": frame,
            "segment_time_s": float(frame - 1),
            "source_scene_id": "scene-a",
        },
    } for frame in range(1, 10) for relation in range(1, 61)]

    frozen = long_generation._freeze_fusion_relation_occurrences(
        occurrences, timeline,
    )
    assert len(frozen) == 540
    projected = long_generation._expected_fusion_relation_states(
        timeline, frozen,
    )
    contract = long_generation._compact_h3_relation_contract(projected)

    assert contract["v"] == 3
    assert contract["m"][0] == 2
    assert len(contract["d"]) == 60
    assert "direction:S->O" in contract["l"]
    marker = (
        f"{long_generation.RELATION_STATES_OPEN}"
        f"{json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        f"{long_generation.RELATION_STATES_CLOSE}"
    )
    assert len(marker) <= long_generation._MAX_RELATION_MARKER_CHARS
    assert len(marker) / context_ir_bridge.MAX_SOURCE_PROMPT_CHARS <= 0.30
    assert context_ir_bridge.MAX_SOURCE_PROMPT_CHARS - len(marker) >= 4_200
    expanded = long_generation._expand_h3_relation_contract(contract)
    assert long_generation._compact_h3_relation_contract(expanded) == contract
    assert len(expanded[0]["relations"]) == 60
    assert all(
        relation["subject_key"].startswith("E")
        and relation["object_key"].startswith("E")
        and relation["predicate"] == "supported/contacting"
        and relation["states"] == [
            {
                "frame_order": frame,
                "state": (
                    "released/separated"
                    if frame in {1, 9} else "active/as-shown"
                ),
                "geometry": (
                    "separated" if frame in {1, 9} else "as-shown"
                ),
            }
            for frame in range(1, 10)
        ]
        for relation in expanded[0]["relations"]
    )
    for mutate in (
        lambda value: value["d"][0].__setitem__(0, 999),
        lambda value: value["i"][0].__setitem__(2, 777),
    ):
        malformed = json.loads(json.dumps(contract))
        mutate(malformed)
        with pytest.raises(
            long_generation.LongGenerationError,
            match="prompt_fusion_output_invalid",
        ):
            long_generation._expand_h3_relation_contract(malformed)
        encoded = json.dumps(
            malformed, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        with pytest.raises(
            context_ir_bridge.ContextIrContractError,
            match="source_relation_states_invalid",
        ):
            context_ir_bridge._relation_states_contract(
                f"{long_generation.RELATION_STATES_OPEN}{encoded}"
                f"{long_generation.RELATION_STATES_CLOSE}"
            )

    prompt = long_generation._compile_fusion_ref2va_prompt(
        visual=["V" * 3_200],
        timeline=timeline,
        lines=[],
        music_policy="forbid",
        relation_occurrences=occurrences,
    )
    assert len(prompt) <= long_generation._MAX_COMPILED_FUSION_CHARS
    assert (
        len(prompt) + len(context_ir_bridge._DIALOGUE_POLICY) + 1
        <= context_ir_bridge.MAX_SOURCE_PROMPT_CHARS
    )
    parsed = context_ir_bridge._relation_states_contract(prompt)
    assert parsed is not None
    parsed_contract = json.loads(
        parsed[len(long_generation.RELATION_STATES_OPEN):
               -len(long_generation.RELATION_STATES_CLOSE)]
    )
    assert long_generation._expand_h3_relation_contract(
        parsed_contract
    ) == expanded

    request = SimpleNamespace(
        source_prompt=prompt,
        dialogue_tokens=(),
        source_h3_request=SimpleNamespace(
            mode="reference",
            workflow=h3.H3_WORKFLOW,
            context_ir_required=True,
            keyframes=tuple(range(9)),
            reference_audios=(),
        ),
    )
    provider_output = "Context-expanded visual prose. " * 400
    effective = context_ir_bridge._compile_effective_prompt(
        request,
        provider_output,
    )
    assert effective == provider_output


def test_relation_runs_do_not_cross_missing_frames_or_hard_cuts() -> None:
    timeline = _timeline()
    occurrences = [
        _occurrence("relation-01", "person-01", "entity-01", 1, "held"),
        _occurrence("relation-01", "person-01", "entity-01", 3, "held"),
        _occurrence("relation-01", "person-01", "entity-01", 4, "held"),
        _occurrence("relation-01", "person-01", "entity-01", 5, "held"),
        _occurrence("relation-01", "person-01", "entity-01", 6, "held"),
    ]
    # Equalize geometry so only evidence adjacency and hard cuts control RLE.
    for occurrence in occurrences:
        occurrence["geometry"] = "same geometry"
    projected = long_generation._expected_fusion_relation_states(
        timeline,
        long_generation._freeze_fusion_relation_occurrences(
            occurrences, timeline,
        ),
    )
    contract = long_generation._compact_h3_relation_contract(projected)

    assert contract["i"][0][3] == 0
    assert contract["i"][1][3] == 1
    assert contract["i"][0][4][0][1] == [
        [1, 1, 0, 0], [3, 4, 0, 0],
    ]
    assert contract["i"][1][4][0][1] == [[5, 6, 0, 0]]
    assert long_generation._expand_h3_relation_contract(contract) == projected

    mutations = []
    overlap = json.loads(json.dumps(contract))
    overlap["i"][1][0] = overlap["i"][0][1]
    mutations.append(overlap)
    gap = json.loads(json.dumps(contract))
    gap["i"][1][0] = gap["i"][0][1] + 2
    mutations.append(gap)
    missing_head = json.loads(json.dumps(contract))
    missing_head["i"][0][0] = 2
    mutations.append(missing_head)
    missing_tail = json.loads(json.dumps(contract))
    missing_tail["i"][-1][1] = 8
    mutations.append(missing_tail)
    for malformed in mutations:
        with pytest.raises(
            long_generation.LongGenerationError,
            match="prompt_fusion_output_invalid",
        ):
            long_generation._expand_h3_relation_contract(malformed)
        encoded = json.dumps(
            malformed, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        with pytest.raises(
            context_ir_bridge.ContextIrContractError,
            match="source_relation_states_invalid",
        ):
            context_ir_bridge._relation_states_contract(
                f"{long_generation.RELATION_STATES_OPEN}{encoded}"
                f"{long_generation.RELATION_STATES_CLOSE}"
            )


def test_context_rejects_noncanonical_v3_aliases_indexes_and_runs() -> None:
    timeline = _timeline()
    occurrences = [
        _occurrence("relation-01", "person-01", "entity-01", 1, "held"),
        _occurrence("relation-01", "person-01", "entity-01", 2, "held"),
    ]
    for occurrence in occurrences:
        occurrence["geometry"] = "same geometry"
    projected = long_generation._expected_fusion_relation_states(
        timeline,
        long_generation._freeze_fusion_relation_occurrences(
            occurrences, timeline,
        ),
    )
    canonical = long_generation._compact_h3_relation_contract(projected)
    assert canonical["m"][0] == 0

    mutations = []
    duplicate_dictionary = json.loads(json.dumps(canonical))
    duplicate_dictionary["q"].append(duplicate_dictionary["q"][0])
    mutations.append(duplicate_dictionary)
    wrong_legend = json.loads(json.dumps(canonical))
    wrong_legend["l"] = "ambiguous"
    mutations.append(wrong_legend)
    bad_index = json.loads(json.dumps(canonical))
    bad_index["d"][0][0] = 999
    mutations.append(bad_index)
    split_run = json.loads(json.dumps(canonical))
    split_run["i"][0][4][0][1] = [[1, 1, 0, 0], [2, 2, 0, 0]]
    mutations.append(split_run)
    unused_predicate = json.loads(json.dumps(canonical))
    unused_predicate["q"] = sorted([*unused_predicate["q"], "zzz-unused"])
    mutations.append(unused_predicate)

    for malformed in mutations:
        encoded = json.dumps(
            malformed, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        prompt = (
            f"{long_generation.RELATION_STATES_OPEN}{encoded}"
            f"{long_generation.RELATION_STATES_CLOSE}"
        )
        with pytest.raises(
            context_ir_bridge.ContextIrContractError,
            match="source_relation_states_invalid",
        ):
            context_ir_bridge._relation_states_contract(prompt)


def test_state_aware_predicates_never_lower_release_or_separation_to_contact() -> None:
    assert image_optimization._lower_relation_predicate("释放", "已经离手") == "releases"
    assert image_optimization._lower_relation_predicate("分离", "彼此脱离") == "separate_from"
    assert image_optimization._lower_relation_predicate("安装", "固定到接口") == "installs"
    assert image_optimization._lower_relation_predicate("unknown", "释放完成") == "releases"
