from copy import deepcopy
from pathlib import Path

import pytest

from app import image_optimization


def _segments() -> list[dict]:
    return [
        {
            "index": 1,
            "chain_id": "continuity-chain",
            "join_mode": "hard_cut",
            "transition_skeleton": [
                {
                    "frame_index": 1,
                    "frame_name": "01.png",
                    "source_transition_from_previous": "start",
                },
                {
                    "frame_index": 2,
                    "frame_name": "02.png",
                    "source_transition_from_previous": "same_camera",
                },
            ],
        },
        {
            "index": 2,
            "chain_id": "continuity-chain",
            "join_mode": "hard_cut",
            "transition_skeleton": [
                {
                    "frame_index": 1,
                    "frame_name": "01.png",
                    "source_transition_from_previous": "hard_cut",
                },
                {
                    "frame_index": 2,
                    "frame_name": "02.png",
                    "source_transition_from_previous": "same_camera",
                },
            ],
        },
    ]


def _person_design() -> dict:
    return {
        "source_identity": "source narrative person",
        "replacement_identity": "distinct replacement narrative person",
        "wardrobe_change": "different practical wardrobe",
        "local_color_change": "different local wardrobe colors",
    }


def _person_observation() -> dict:
    return {
        "visible_region": "currently visible person region",
        "boundary": "current person and occlusion boundary",
        "body_and_pose": "currently visible body and pose",
    }


def _scene_design() -> dict:
    return {
        "source_scene": "source narrative environment",
        "replacement_scene": "different real narrative environment",
        "semantic_change": "same narrative purpose in a different environment",
        "geometry_change": "different visible geometry",
        "depth_change": "different visible depth organization",
        "layout_change": "different functional layout",
        "local_color_change": "different local material colors",
    }


def _frame(
    *,
    personal_visibility: str | None,
    project_visibility: str | None,
) -> dict:
    entities = {}
    if personal_visibility is not None:
        entities["personal-persistent-item"] = {
            "visibility": personal_visibility,
            "relationship": "keep its current relation to the narrative person",
        }
    if project_visibility is not None:
        entities["project-persistent-item"] = {
            "visibility": project_visibility,
            "relationship": "keep the current project-level spatial relation",
        }
    return {
        "people": {"narrative-person": _person_observation()},
        "relationships": "preserve current visible contacts and occlusions",
        "entities": entities,
        "crop": "preserve current crop and off-canvas extent",
    }


def _semantic_frames() -> dict:
    return {
        "frame-001": _frame(
            personal_visibility="visible",
            project_visibility="visible",
        ),
        "frame-002": _frame(
            personal_visibility="occluded",
            project_visibility="out_of_frame",
        ),
        "frame-003": _frame(
            personal_visibility="visible",
            project_visibility="visible",
        ),
        "frame-004": _frame(
            personal_visibility="visible",
            project_visibility="out_of_frame",
        ),
    }


def _semantic_entities() -> dict:
    return {
        "personal-persistent-item": {
            "description": "persistent visible item associated with one person",
            "owner": "narrative-person",
            "association": "kept by the same narrative person",
            "persistence": "keep one physical identity across the continuity chain",
        },
        "project-persistent-item": {
            "description": "persistent visible item associated with the project",
            "owner": "project",
            "association": "kept as one project-level persistent item",
            "persistence": "keep one physical identity across the continuity chain",
        },
    }


def test_semantic_compiler_keeps_entity_identity_owner_and_visibility_across_cut():
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": _semantic_frames(),
    }

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic, _segments(),
    )

    ledgers = [
        constraint["non_person_entity_ledger"]
        for segment in plan["segments"]
        for constraint in segment["frame_constraints"]
    ]
    assert [
        [entity["entity_id"] for entity in ledger["entities"]]
        for ledger in ledgers
    ] == [["ENTITY_01", "ENTITY_02"]] * 4
    assert [
        [entity["description"] for entity in ledger["entities"]]
        for ledger in ledgers
    ] == [[
        "persistent visible item associated with one person",
        "persistent visible item associated with the project",
    ]] * 4
    assert [
        [entity["visibility"] for entity in ledger["entities"]]
        for ledger in ledgers
    ] == [
        ["full", "full"],
        ["occluded", "out_of_frame"],
        ["full", "full"],
        ["full", "out_of_frame"],
    ]
    assert all(ledger["relations"] == [
        {
            "subject_id": "ENTITY_01",
            "predicate": "owned_by",
            "object_id": "PERSON_01",
        },
        {
            "subject_id": "ENTITY_02",
            "predicate": "owned_by",
            "object_id": "PROJECT",
        },
    ] for ledger in ledgers)
    assert plan["scene_plans"][0]["continuity_graph"]["views"][2][
        "transition_from_previous"
    ] == "hard_cut"
    assert diagnostics["entity_continuity"]["stable_entity_count"] == 2
    assert diagnostics["entity_continuity"]["source_preserve_defaults"] == []
    prompts = image_optimization.compile_frame_prompts(
        plan, "anchor_consistency",
    )
    assert '"predicate":"owned_by"' in prompts[1][1]
    assert '"visibility":"occluded"' in prompts[1][2]
    assert '"visibility":"out_of_frame"' in prompts[1][2]
    assert image_optimization.canonical_plan_v4(
        plan, [1, 2], frame_counts={1: 2, 2: 2},
    ) == plan


def test_semantic_compiler_defaults_missing_entity_state_to_source_preserve():
    frames = _semantic_frames()
    frames["frame-002"]["entities"].pop("personal-persistent-item")
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": frames,
    }

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic, _segments(),
    )

    defaulted = plan["segments"][0]["frame_constraints"][1][
        "non_person_entity_ledger"
    ]["entities"][0]
    assert defaulted == {
        "entity_id": "ENTITY_01",
        "description": "persistent visible item associated with one person",
        "visibility": "source_preserve",
    }
    assert diagnostics["entity_continuity"]["source_preserve_defaults"] == [
        "frames.frame-002.entities.personal-persistent-item"
    ]
    assert "blocking" not in diagnostics
    assert "retry" not in diagnostics
    assert "fallback" not in diagnostics
    assert plan["eligible"] is True


def test_semantic_compiler_keeps_running_when_entity_semantics_are_absent():
    semantic = {
        "people": {"narrative-person": _person_design()},
        "scenes": {"scene-001": _scene_design()},
        "frames": {
            key: {
                **value,
                "entities": "preserve source-visible non-person entities",
            }
            for key, value in _semantic_frames().items()
        },
    }

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic, _segments(),
    )

    assert all(
        constraint["non_person_entity_ledger"] == {
            "entities": [], "relations": [],
        }
        for segment in plan["segments"]
        for constraint in segment["frame_constraints"]
    )
    assert diagnostics["entity_continuity"] == {
        "stable_entity_count": 0,
        "source_preserve_defaults": ["entities"],
    }
    assert plan["eligible"] is True


def test_semantic_compiler_preserves_observed_keys_without_global_design():
    semantic = {
        "people": {"narrative-person": _person_design()},
        "scenes": {"scene-001": _scene_design()},
        "frames": _semantic_frames(),
    }

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic, _segments(),
    )

    first_ledger = plan["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]
    assert [item["entity_id"] for item in first_ledger["entities"]] == [
        "ENTITY_01", "ENTITY_02",
    ]
    assert all(
        item["description"].startswith("source-preserve/no-invention")
        for item in first_ledger["entities"]
    )
    assert first_ledger["relations"] == []
    assert "entities" in diagnostics["entity_continuity"][
        "source_preserve_defaults"
    ]
    assert plan["eligible"] is True


def test_semantic_compiler_does_not_propagate_entities_to_another_chain():
    segments = _segments()
    segments[1]["chain_id"] = "independent-chain"
    frames = _semantic_frames()
    frames["frame-003"]["entities"] = {}
    frames["frame-004"]["entities"] = {}
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {
            "scene-001": _scene_design(),
            "scene-002": _scene_design(),
        },
        "frames": frames,
    }

    plan, _ = image_optimization.compile_semantic_plan(semantic, segments)

    assert all(
        constraint["non_person_entity_ledger"] == {
            "entities": [], "relations": [],
        }
        for constraint in plan["segments"][1]["frame_constraints"]
    )


def test_entity_owner_graph_rejects_multiple_owners():
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": _semantic_frames(),
    }
    plan, _ = image_optimization.compile_semantic_plan(semantic, _segments())
    ambiguous = deepcopy(plan)
    ambiguous["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]["relations"].insert(1, {
        "subject_id": "ENTITY_01",
        "predicate": "owned_by",
        "object_id": "PROJECT",
    })

    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            ambiguous, [1, 2], frame_counts={1: 2, 2: 2},
        )


def test_provider_prompt_uses_only_the_current_frame_person_observation():
    frames = _semantic_frames()
    first = frames["frame-001"]["people"]["narrative-person"]
    first.update({
        "visible_region": "frame-one-exclusive-person-domain",
        "boundary": "frame-one-exclusive-person-boundary",
    })
    second = frames["frame-002"]["people"]["narrative-person"]
    second.update({
        "visible_region": "frame-two-exclusive-person-domain",
        "boundary": "frame-two-exclusive-person-boundary",
    })
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": frames,
    }

    plan, _ = image_optimization.compile_semantic_plan(semantic, _segments())
    prompts = image_optimization.compile_frame_prompts(
        plan, "anchor_consistency",
    )

    assert "frame-one-exclusive-person-domain" in prompts[1][1]
    assert "frame-one-exclusive-person-boundary" in prompts[1][1]
    assert "frame-two-exclusive-person-domain" not in prompts[1][1]
    assert "frame-two-exclusive-person-boundary" not in prompts[1][1]
    assert "frame-two-exclusive-person-domain" in prompts[1][2]
    assert "frame-two-exclusive-person-boundary" in prompts[1][2]
    assert "frame-one-exclusive-person-domain" not in prompts[1][2]
    assert "frame-one-exclusive-person-boundary" not in prompts[1][2]


@pytest.mark.parametrize("mode", ["optical_projection", "temporal_residual"])
def test_derived_person_observation_compiles_as_source_carrier_bound_nonphysical(
    mode,
):
    frames = _semantic_frames()
    frames["frame-001"]["people"]["narrative-person"][
        "derived_observations"
    ] = {
        "derived-view": {
            "mode": mode,
            "source_carrier": "source-carrier-alpha",
            "visible_region": "derived-observation-region-alpha",
            "boundary": "derived-observation-boundary-alpha",
            "relationship": "derived from the stable person through its source carrier",
        }
    }
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": frames,
    }

    plan, _ = image_optimization.compile_semantic_plan(semantic, _segments())
    constraint = plan["segments"][0]["frame_constraints"][0]
    compiled = "；".join(
        constraint[key]
        for key in image_optimization._FRAME_TEXT_CONSTRAINT_KEYS
    )
    prompt = image_optimization.compile_frame_prompts(
        plan, "anchor_consistency",
    )[1][1]

    for expected in (
        f'"mode":"{mode}"',
        '"physicality":"non_physical"',
        '"source_person":"PERSON_01"',
        '"source_carrier":"source-carrier-alpha"',
        '"instantiation":"source_carrier_bound"',
        "derived-observation-region-alpha",
        "derived-observation-boundary-alpha",
    ):
        assert expected in compiled
        assert expected in prompt


def test_missing_derived_person_semantics_are_source_preserve_diagnostics_only():
    frames = _semantic_frames()
    semantic = {
        "people": {"narrative-person": _person_design()},
        "entities": _semantic_entities(),
        "scenes": {"scene-001": _scene_design()},
        "frames": frames,
    }

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic, _segments(),
    )
    constraint = plan["segments"][0]["frame_constraints"][0]
    compiled = "；".join(
        constraint[key]
        for key in image_optimization._FRAME_TEXT_CONSTRAINT_KEYS
    )

    assert diagnostics["person_observation_continuity"][
        "source_preserve_defaults"
    ] == [
        "frames.frame-001.people.narrative-person.derived_observations",
        "frames.frame-002.people.narrative-person.derived_observations",
        "frames.frame-003.people.narrative-person.derived_observations",
        "frames.frame-004.people.narrative-person.derived_observations",
    ]
    assert "derived_observations=source-preserve/non-physical" in compiled
    assert plan["eligible"] is True
    assert "blocking" not in diagnostics
    assert "retry" not in diagnostics
    assert "fallback" not in diagnostics


def test_skill_assigns_semantics_only_and_requires_persistent_entity_states():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    for text in (
        "stable entity key",
        "项目级或人物归属",
        "visible/occluded/out_of_frame",
        "source-preserve",
        "hard_cut",
        "derived_observations",
        "optical_projection",
        "temporal_residual",
        "source-preserve/non-physical",
        "派生观测始终嵌套在来源人物下",
        "实体 ID、关系图和完整机械字段由后端构造",
    ):
        assert text in skill
    assert '顶层只含 `people/entities/scenes/frames`' in skill
    assert '"entities": {' in skill
    assert '"owner": "project 或 stable-person-key"' in skill
    assert '"association": "项目级持久关系，或由 stable-person-key 持有或穿戴"' in skill
    assert '"relationship": "当前帧直接可见或可判定的关系"' in skill
