import hashlib
import json

from app import image_optimization, long_generation


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
            "predicate": "generic contact",
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
            "relations": {},
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
        "preserve": ["有向安装角色", "数量一一对应"],
        "replace_together": True,
    }]
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

    encoded = json.dumps(
        long_generation._compact_h3_relation_contract(projected),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    assert len(encoded) < 2_000
    assert hashlib.sha256(encoded.encode()).hexdigest()


def test_state_aware_predicates_never_lower_release_or_separation_to_contact() -> None:
    assert image_optimization._lower_relation_predicate("释放", "已经离手") == "releases"
    assert image_optimization._lower_relation_predicate("分离", "彼此脱离") == "separate_from"
    assert image_optimization._lower_relation_predicate("安装", "固定到接口") == "installs"
    assert image_optimization._lower_relation_predicate("unknown", "释放完成") == "releases"
