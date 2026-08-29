import hashlib
import json
import re
import threading
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import image_optimization
from conftest import make_settings


def _png(value: int = 127) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((8, 6, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _plan(indices: list[int] | None = None) -> dict:
    indices = indices or [1, 2]
    first, last = indices[0], indices[-1]
    return {
        "version": 2,
        "phase": "plan",
        "segment_indices": indices,
        "eligible": True,
        "reason": None,
        "person_plans": [
            {
                "id": "PERSON_01",
                "source_identity": "反复出现的原叙事主人物",
                "replacement_identity": "脸部轮廓略有不同的同类新人物",
                "wardrobe_change": "保持服装用途与风格，改为不同款式",
                "local_color_change": "人物局部固有色产生可见变化",
                "reference": {"segment_index": first, "frame_index": 1},
                "observable_segments": indices,
            }
        ],
        "scene_plans": [
            {
                "id": "SCENE_01",
                "source_scene": "源叙事环境",
                "replacement_scene": "同用途但不同设计的真实新环境",
                "semantic_change": "替换为另一处真实叙事环境",
                "geometry_changes": ["改变主要结构的形状与连接关系"],
                "depth_changes": ["改变可见纵深与前后层级"],
                "layout_changes": ["改变功能区域与通行关系"],
                "local_color_change": "改变可见表面的局部固有色",
                "reference": {"segment_index": first, "frame_index": 1},
                "segments": indices,
            }
        ],
        "segments": [
            {
                "segment_index": index,
                "persons": [
                    {
                        "id": "PERSON_01",
                        "state": "replace",
                        "observable_frames": [1],
                        "target_region": "完整可见的叙事主人物",
                        "boundary": "人物可见轮廓与真实遮挡边界",
                    }
                ],
                "scene": {
                    "scene_id": "SCENE_01",
                    "target_region": "人物以外的完整可见场景",
                    "boundary": "场景边界止于人物和前景实体轮廓",
                    "layout_reference_frame_index": 1,
                },
                "protected_non_target_people": [],
                "protected_relations": ["人物与核心实体的可见接触关系"],
            }
            for index in indices
        ],
    }


def _ineligible(reason: str = "no_observable_narrative_person") -> dict:
    return {
        "version": 2,
        "phase": "plan",
        "segment_indices": [0],
        "eligible": False,
        "reason": reason,
        "person_plans": [],
        "scene_plans": [],
        "segments": [],
    }


def _segments(
    session: Path, indices: list[int] | None = None, *, frames: int = 1
) -> list[dict]:
    indices = indices or [1, 2]
    result = []
    for index in indices:
        path = session / "work" / "segments" / str(index) / "work" / "keyframes"
        path.mkdir(parents=True)
        for frame_index in range(1, frames + 1):
            (path / f"{frame_index:02d}.png").write_bytes(
                _png(index + frame_index)
            )
        result.append(
            {
                "index": index,
                "chain_id": "chain-001",
                "join_mode": "hard_cut" if index == indices[0] else "continue",
                "keyframes_dir": path,
            }
        )
    return result


def _transition_skeleton(segments: list[dict]) -> list[dict]:
    return [
        {
            **segment,
            "transition_skeleton": [
                {
                    "segment_index": segment["index"],
                    "frame_index": frame_index,
                    "frame_name": frame.name,
                    "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "source_transition_from_previous": (
                        "start" if segment["index"] == segments[0]["index"]
                        and frame_index == 1 else "same_camera"
                    ),
                    "source_transition_evidence_sha256": (
                        str(frame_index) * 64
                    ),
                }
                for frame_index, frame in enumerate(
                    sorted(Path(segment["keyframes_dir"]).glob("*.png")), 1
                )
            ],
        }
        for segment in segments
    ]


class _Runner:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict] = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        root = Path(workdir)
        request = json.loads(
            (root / "work" / "request.json").read_text(encoding="utf-8")
        )
        global_plan_path = root / "work" / "global_plan.json"
        self.calls.append(
            {
                "files": sorted(
                    str(path.relative_to(root))
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                "request": request,
                "global_plan": (
                    json.loads(global_plan_path.read_text(encoding="utf-8"))
                    if global_plan_path.is_file()
                    and global_plan_path.stat().st_size else None
                ),
                "prompt": prompt,
                "session_dir": Path(session_dir),
            }
        )
        name = {
            "global_plan": "global_plan.json",
            "segment_frames": "segment_frames.json",
            "plan_audit": "plan_audit.json",
            "verify": "image_verification.json",
            "verify_pack": "reference_pack_verification.json",
        }[self.calls[-1]["request"]["phase"]]
        output = self.output(request) if callable(self.output) else self.output
        if request["phase"] == "global_plan":
            output = {key: output.get(key, {}) for key in ("people", "entities", "scenes")}
        elif request["phase"] == "segment_frames":
            output = {"frames": output.get("frames", {})}
        (root / "work" / name).write_text(
            json.dumps(output, ensure_ascii=False), encoding="utf-8"
        )


class _ParallelRunner(_Runner):
    def __init__(self, output: object, segment_count: int) -> None:
        super().__init__(output)
        self.barrier = threading.Barrier(segment_count)
        self.lock = threading.Lock()
        self.active_segments = 0
        self.max_active_segments = 0
        self.entered_segments: list[int] = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        request = json.loads(
            (Path(workdir) / "work" / "request.json").read_text(
                encoding="utf-8"
            )
        )
        if request["phase"] != "segment_frames":
            return super().run_isolated(
                workdir, prompt, session_dir=session_dir
            )
        with self.lock:
            self.active_segments += 1
            self.max_active_segments = max(
                self.max_active_segments, self.active_segments
            )
            self.entered_segments.append(request["segment"]["index"])
        try:
            self.barrier.wait(timeout=5)
            return super().run_isolated(
                workdir, prompt, session_dir=session_dir
            )
        finally:
            with self.lock:
                self.active_segments -= 1


def _skill_contract() -> tuple[str, dict]:
    text = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    contracts = _json_contracts(text)
    example = (
        {**contracts[0], **contracts[1]}
        if len(contracts) == 2
        else contracts[0]
    )
    return text, example


def _json_contracts(text: str) -> list[dict]:
    return [
        json.loads(block)
        for block in re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    ]


def test_skill_exposes_closed_global_and_segment_json_contracts():
    skill, _example = _skill_contract()
    contracts = _json_contracts(skill)

    assert len(contracts) == 2
    global_plan, segment_frames = contracts
    assert set(global_plan) == {"people", "entities", "scenes"}
    assert set(segment_frames) == {"frames"}
    assert set(next(iter(global_plan["people"].values()))) == {
        "source_identity", "replacement_identity", "wardrobe_change",
        "local_color_change",
    }
    assert set(next(iter(global_plan["entities"].values()))) == {
        "description", "owner", "association", "persistence",
    }
    assert set(next(iter(global_plan["scenes"].values()))) == {
        "source_scene", "replacement_scene", "semantic_change",
        "geometry_change", "depth_change", "layout_change",
        "local_color_change",
    }
    frame = next(iter(segment_frames["frames"].values()))
    assert set(frame) == {"people", "relationships", "entities", "crop"}
    person = next(iter(frame["people"].values()))
    assert set(person) == {
        "visible_region", "boundary", "body_and_pose",
        "derived_observations",
    }
    observation = next(iter(person["derived_observations"].values()))
    assert set(observation) == {
        "mode", "source_carrier", "visible_region", "boundary",
        "relationship",
    }
    entity = next(iter(frame["entities"].values()))
    assert set(entity) == {"visibility", "relationship"}
    assert global_plan["entities"]["<stable-entity-key>"]["owner"] == "project"
    assert observation["mode"] in {
        "optical_projection", "temporal_residual", "source-preserve",
    }
    assert entity["visibility"] in {"visible", "occluded"}


def test_skill_frame_entities_are_direct_evidence_only_and_omit_out_of_frame():
    skill, _example = _skill_contract()
    segment_frames = _json_contracts(skill)[1]
    encoded = json.dumps(segment_frames, ensure_ascii=False)

    assert '"visibility": "out_of_frame"' not in encoded
    assert "当前帧有直接像素证据" in skill
    assert "完全出画、完全不可见或仅由邻帧推知时省略 key" in skill
    assert "不写 `out_of_frame` 占位" in skill


def test_skill_requires_slot_coverage_key_reuse_and_atomic_nonempty_completion():
    skill, _example = _skill_contract()

    for required in (
        "semantic_slots.scenes[].key",
        "semantic_slots.frames[].key",
        "全部 key",
        "逐字复用",
        "同目录临时文件",
        "自检",
        "原子替换",
        "输出非空",
        "不得结束或只给解释",
    ):
        assert required in skill
    assert "唯一输出" in skill
    assert "任何单一阶段不得直接输出四个字段" in skill


def test_skill_matches_runtime_semantic_enum_boundary_and_stays_short():
    skill, _example = _skill_contract()
    encoded = json.dumps(_json_contracts(skill), ensure_ascii=False)

    assert '"visibility": "visible/occluded/out_of_frame"' not in encoded
    assert "source-preserve 是 mode 的第三个值" in skill
    assert "不新增质量门禁" in skill
    assert "不新增 reject、retry 或 fallback" in skill
    assert len(skill.splitlines()) <= 70
    assert len(skill.encode("utf-8")) <= 7 * 1024


def test_skill_scopes_hard_cut_blur_and_edge_fragments_to_current_pixels():
    skill, _example = _skill_contract()

    for rule in (
        "transition_skeleton",
        "hard_cut",
        "相邻帧",
        "强运动模糊",
        "edge_fragment",
        "全局参考",
        "补头",
        "补人",
        "补衣服",
        "本帧直接可见像素",
    ):
        assert rule in skill
    assert len(skill.splitlines()) <= 52


def _element_index() -> dict:
    return {
        "people": {
            "subject": {
                "source_visual_description": "跨段可见的主人物",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replaceable": ["identity", "wardrobe", "local_color"],
                "preserve": ["pose", "action", "relationships"],
            }
        },
        "entities": {},
        "scenes": {
            "scene-001": {
                "source_visual_description": "跨段连续的源环境",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": 2, "frame_orders": [1]},
                ],
                "replaceable": ["environment_design"],
                "preserve": ["composition", "lighting", "tone"],
            }
        },
    }


def _semantic_output(request: dict, *, sparse: bool = False) -> dict:
    slots = request["semantic_slots"]
    scenes = {
        slot["key"]: (
            {"replacement_scene": "同用途且设计不同的真实新环境"}
            if sparse else {
                "source_scene": "当前可见源环境",
                "replacement_scene": "同用途且设计不同的真实新环境",
                "semantic_change": "环境语义明显变化",
                "geometry_change": "可见形状和空间结构明显变化",
                "depth_change": "前中后景纵深明显变化",
                "layout_change": "功能区域和实体布局明显变化",
                "local_color_change": "局部材质固有色明显变化",
            }
        )
        for slot in slots.get("scenes", [])
    }
    frames = {
        slot["key"]: (
            {"palette_description": "warm-neutral and natural-muted"}
            if sparse else {
                "people": {"subject": {
                    "visible_region": f"{slot['key']} 当前可见人物区域",
                    "boundary": f"{slot['key']} 当前可见边界",
                    "body_and_pose": f"{slot['key']} 当前可见身体与姿态",
                    "derived_observations": {},
                }},
                "relationships": f"{slot['key']} 当前可见接触与遮挡关系",
                "entities": f"{slot['key']} 当前可见非人物实体",
                "crop": f"{slot['key']} 当前画外裁切",
            }
        )
        for slot in slots.get("frames", [])
    }
    people = {} if sparse else {"subject": {
        "source_identity": "当前可见源人物",
        "replacement_identity": "明显不同且跨帧稳定的新人物",
        "wardrobe_change": "不同款式且保持用途的服装",
        "local_color_change": "人物局部固有色明显变化",
    }}
    return {"people": people, "scenes": scenes, "frames": frames}


def _compiled_semantic(tmp_path: Path, *, frames: int = 1, sparse: bool = False):
    segments = _transition_skeleton(
        _segments(tmp_path / "session", [0], frames=frames)
    )
    value = _semantic_output(
        {"semantic_slots": image_optimization.semantic_slot_manifest(segments)},
        sparse=sparse,
    )
    source_frames = {
        0: sorted(Path(segments[0]["keyframes_dir"]).glob("*.png"))
    }
    return image_optimization.compile_semantic_plan(
        value, segments, source_frames=source_frames,
    )


def _check(status: str = "pass", evidence: str = "证据充分") -> dict:
    return {"status": status, "evidence": evidence}


def _verdict(plan: dict, *, passed: bool = True) -> dict:
    reason = None if passed else "scene_semantic_change_failed"
    scene_status = "pass" if passed else "fail"
    return {
        "version": 2,
        "phase": "verify",
        "plan_sha256": image_optimization.plan_sha256(plan),
        "segment_indices": plan["segment_indices"],
        "passed": passed,
        "reason": reason,
        "segments": [
            {
                "segment_index": segment["segment_index"],
                "passed": passed,
                "person_checks": [
                    {
                        "person_id": person["id"],
                        "identity_changed": _check(
                            "pass" if person["state"] == "replace" else "not_applicable"
                        ),
                        "source_identity_absent": _check(
                            "pass" if person["state"] == "replace" else "not_applicable"
                        ),
                        "local_color_change": _check(
                            "pass" if person["state"] == "replace" else "not_applicable"
                        ),
                    }
                    for person in segment["persons"]
                ],
                "scene_checks": {
                    "semantic_change": _check(scene_status),
                    "geometry_change": _check(),
                    "depth_change": _check(),
                    "layout_change": _check(),
                    "local_color_change": _check(),
                },
                "invariants": {
                    "lighting_preservation": _check(),
                    "interaction_preservation": _check(),
                    "cross_frame_continuity": _check(),
                },
            }
            for segment in plan["segments"]
        ],
        "project_checks": {
            "narrative_person_completeness": _check(),
            "no_identity_swap": _check(),
            "no_unplanned_person": _check(),
            "person_identity_continuity": _check(),
            "scene_continuity": _check(),
        },
    }


def _pack_verdict(plan: dict, *, passed: bool = True) -> dict:
    person_status = "pass" if passed else "fail"
    reason = None if passed else "person_identity_change_failed"
    return {
        "version": plan["version"],
        "phase": "verify_pack",
        "plan_sha256": image_optimization.plan_sha256(plan),
        "passed": passed,
        "reason": reason,
        "persons": [
            {
                "person_id": person["id"],
                "passed": passed,
                "checks": {
                    "identity_changed": _check(person_status),
                    "source_identity_absent": _check(),
                    "multiview": _check(),
                    "local_color": _check(),
                },
            }
            for person in plan["person_plans"]
        ],
        "scenes": [
            {
                "scene_id": scene["id"],
                "passed": True,
                "checks": {
                    "semantic": _check(),
                    "geometry": _check(),
                    "depth": _check(),
                    "layout": _check(),
                    "local_color": _check(),
                },
            }
            for scene in plan["scene_plans"]
        ],
        "project": {
            "light_direction_preservation": _check(),
            "exposure_preservation": _check(),
            "wb_cct_preservation": _check(),
            "tone_curve_preservation": _check(),
        },
    }


def test_skill_uses_one_global_plan_and_parallel_segment_frame_contracts():
    skill, example = _skill_contract()

    assert "name: image-postprocess" in skill
    assert '`phase="global_plan"' in skill
    assert '`phase="segment_frames"' in skill
    assert set(example) == {"people", "entities", "scenes", "frames"}
    assert "只填写视觉语义" in skill
    assert "后端据此确定性编译" in skill
    assert "人物与场景同时替换" in skill
    assert "source-preserve/no-invention" in skill
    encoded = json.dumps(example, ensure_ascii=False)
    for backend_field in (
        "version", "segment_indices", "eligible", "reason", "person_plans",
        "scene_plans", "frame_constraints", "continuity_graph",
    ):
        assert backend_field not in encoded
    for retired in ("plan_audit", "verify_pack", "work/image_verification.json"):
        assert retired not in skill


def test_skill_additively_consumes_element_index_without_changing_output_schema(
    tmp_path,
):
    skill, example = _skill_contract()
    session = tmp_path / "session"
    segments = _transition_skeleton(_segments(session))
    element_index = _element_index()
    index_path = session / "work" / "element_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(element_index, ensure_ascii=False), encoding="utf-8"
    )
    runner = _Runner(_semantic_output)

    plan, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "anchor_consistency",
        session_dir=session,
        element_index_path=index_path,
    )

    assert runner.calls[0]["request"]["element_index"] == element_index
    assert runner.calls[0]["request"]["phase"] == "global_plan"
    assert {call["request"]["phase"] for call in runner.calls[1:]} == {
        "segment_frames"
    }
    assert [
        (tile["stable_key"], tile["tile_id"])
        for tile in image_optimization.composite_replacement_board_spec(plan)["tiles"]
    ] == [("subject", "TILE_01"), ("scene-001", "TILE_02")]
    assert "subject -> TILE_01" in prompts[1][1]
    assert "scene-001 -> TILE_02" in prompts[2][1]
    assert set(example) == {"people", "entities", "scenes", "frames"}
    for contract in (
        "element_index",
        "stable key",
        "逐字复用",
        "所有片段共享",
        "跨段一致",
    ):
        assert contract in skill
    assert "不要输出版本、段号、帧号" in skill


def test_skill_adds_no_content_gate_retry_or_fallback_for_element_index():
    skill, example = _skill_contract()
    encoded = json.dumps(example, ensure_ascii=False)

    for control_field in ('"eligible"', '"reason"', '"gate"', '"reject"'):
        assert control_field not in encoded
    assert "不新增质量门禁" in skill
    assert "不新增 reject、retry 或 fallback" in skill


def test_element_index_semantic_damage_is_non_blocking_normalization(tmp_path):
    session = tmp_path / "session"
    segments = _transition_skeleton(_segments(session))
    index_path = session / "work" / "element_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        "people": {
            "subject": {
                "source_visual_description": "跨段主人物",
                "occurrences": [
                    {"segment_index": 1, "frame_orders": [1]},
                    {"segment_index": "bad", "frame_orders": []},
                ],
                "replaceable": ["identity", None],
                "preserve": "bad",
            },
            "broken": "bad",
        },
        "entities": "bad",
    }, ensure_ascii=False), encoding="utf-8")
    runner = _Runner(_semantic_output)

    plan, _prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "anchor_consistency",
        session_dir=session,
        element_index_path=index_path,
    )

    assert runner.calls[0]["request"]["element_index"] == {
        "people": {
            "subject": {
                "source_visual_description": "跨段主人物",
                "occurrences": [{"segment_index": 1, "frame_orders": [1]}],
                "replaceable": ["identity"],
                "preserve": [],
            }
        },
        "entities": {},
        "scenes": {},
    }
    assert plan["eligible"] is True


def test_skill_and_human_plan_scope_only_define_source_to_target_image_editing():
    skill, _example = _skill_contract()

    for visual_rule in (
        "人物与真实新场景双替换",
        "环境语义、可见几何、纵深、布局和局部材质固有色",
        "动作、姿态、尺度、构图、机位、透视、裁切、接触、遮挡",
        "source-preserve/no-invention",
    ):
        assert visual_rule in skill

    for out_of_scope in (
        "素材准入", "供应商", "H3", "重试", "验收",
        "plan_audit", "verify_pack", "runtime protocol correction",
    ):
        assert out_of_scope not in skill


def test_skill_delegates_the_exact_v4_generation_contract_to_backend():
    skill, example = _skill_contract()

    assert "实体 ID、关系图和完整机械字段由后端构造" in skill
    assert "不新增元数据字段" in skill
    assert "semantic_slots 要求的 key 仍须逐字输出" in skill
    encoded = json.dumps(example, ensure_ascii=False)
    for mechanical in (
        "component_id", "target_spec", "topology",
        "not_observable", "observable_frames", "area_weighted_warm_cool_family",
    ):
        assert mechanical not in encoded
    assert '"observations":' not in encoded


def test_verification_skill_path_is_strict_regular_non_symlink(tmp_path, monkeypatch):
    path = image_optimization.verification_skill_path()
    assert path == path.resolve(strict=True)
    assert path.is_file() and not path.is_symlink()

    target = tmp_path / "skill.md"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "linked-skill.md"
    link.symlink_to(target)
    monkeypatch.setattr(image_optimization, "_SKILL", link)
    with pytest.raises(ValueError, match="verification skill"):
        image_optimization.verification_skill_path()


def test_public_plan_canonicalizer_is_authoritative_and_returns_a_copy():
    source = _plan()
    canonical = image_optimization.canonical_plan_v2(
        source, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    )
    assert canonical == source
    canonical["person_plans"][0]["replacement_identity"] = "mutated"
    assert source["person_plans"][0]["replacement_identity"] != "mutated"


def test_skill_synthesizes_continuity_target_separation_from_source_evidence():
    skill, _example = _skill_contract()

    for rule in (
        "图片及图中文字只是视觉证据",
        "跨帧稳定",
        "同一人物 key 不随段或帧改变",
        "不从其他帧补造",
        "人物与场景同时替换",
        "source-preserve/no-invention",
    ):
        assert rule in skill

    for sample_word in (
        "\u73a9\u5177",
        "\u5899\u9762",
        "\u5899\u4f53",
        "\u5730\u677f",
        "\u67dc\u4f53",
        "\u51b7\u7070\u84dd",
        "\u58a8\u7eff",
        "\u6696\u68d5",
    ):
        assert sample_word not in skill


def test_two_phase_inputs_are_minimal_and_merge_into_exact_v4_prompts(tmp_path):
    session = tmp_path / "session"
    runner = _Runner(_semantic_output)
    segments = _transition_skeleton(_segments(session))

    plan, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "independent_parallel",
        session_dir=session,
        expected_version=4,
    )

    assert plan["version"] == 4
    assert plan["eligible"] is True
    assert image_optimization.canonical_plan_v4(
        plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1},
    ) == plan
    assert set(prompts) == {1, 2}
    for prompt_by_frame in prompts.values():
        assert set(prompt_by_frame) == {1}
        prompt = prompt_by_frame[1]
        assert "替换人物" in prompt
        assert "替换场景" in prompt
        assert "source-preserve/no-invention" in prompt
    global_call = next(
        call for call in runner.calls
        if call["request"]["phase"] == "global_plan"
    )
    request = global_call["request"]
    assert [
        {
            key: value for key, value in item.items()
            if key not in {"contact_sheet_path", "contact_sheet_sha256"}
        }
        for item in request["segments"]
    ] == [
        {key: value for key, value in segment.items() if key != "keyframes_dir"}
        for segment in segments
    ]
    assert [item["contact_sheet_path"] for item in request["segments"]] == [
        "work/contact_sheets/segment-0001.jpg",
        "work/contact_sheets/segment-0002.jpg",
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["contact_sheet_sha256"])
        for item in request["segments"]
    )
    assert [item["key"] for item in request["semantic_slots"]["scenes"]] == [
        "scene-001"
    ]
    assert set(request["semantic_slots"]) == {"scenes"}
    assert global_call["files"] == [
        "SKILL.md",
        "work/contact_sheets/segment-0001.jpg",
        "work/contact_sheets/segment-0002.jpg",
        "work/global_plan.json",
        "work/request.json",
    ]
    assert not any(name.endswith(".png") for name in global_call["files"])

    segment_calls = sorted(
        (
            call for call in runner.calls
            if call["request"]["phase"] == "segment_frames"
        ),
        key=lambda call: call["request"]["segment"]["index"],
    )
    assert len(segment_calls) == 2
    assert segment_calls[0]["global_plan"] == segment_calls[1]["global_plan"]
    assert set(segment_calls[0]["global_plan"]) == {
        "people", "entities", "scenes"
    }
    for index, call in enumerate(segment_calls, 1):
        segment_request = call["request"]
        assert segment_request["segment"]["index"] == index
        assert segment_request["global_plan_path"] == "work/global_plan.json"
        assert set(segment_request["semantic_slots"]) == {"frames"}
        assert [
            item["key"] for item in segment_request["semantic_slots"]["frames"]
        ] == [f"frame-{index:03d}"]
        assert call["files"] == [
            "SKILL.md",
            "work/global_plan.json",
            "work/keyframes/01.png",
            "work/request.json",
            "work/segment_frames.json",
        ]


def test_short_video_semantics_compile_to_both_targets_in_two_phases(tmp_path):
    session = tmp_path / "session"
    runner = _Runner(_semantic_output)
    segments = _transition_skeleton(_segments(session, [0]))
    generated, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "anchor_consistency",
        session_dir=session,
        expected_version=4,
    )

    assert [call["request"]["phase"] for call in runner.calls] == [
        "global_plan", "segment_frames"
    ]
    assert generated["person_plans"] and generated["scene_plans"]
    assert generated["segments"][0]["persons"][0]["state"] == "replace"
    assert "替换人物" in prompts[0][1] and "替换场景" in prompts[0][1]


def test_segment_frame_skill_calls_overlap_and_mechanically_merge_to_v4(tmp_path):
    session = tmp_path / "session"
    segments = _transition_skeleton(_segments(session, [1, 2, 3]))
    runner = _ParallelRunner(_semantic_output, segment_count=3)

    plan, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "independent_parallel",
        session_dir=session,
        expected_version=4,
    )

    assert runner.max_active_segments == 3
    assert set(runner.entered_segments) == {1, 2, 3}
    assert plan["version"] == 4 and plan["eligible"] is True
    assert image_optimization.canonical_plan_v4(
        plan,
        segment_indices=[1, 2, 3],
        frame_counts={1: 1, 2: 1, 3: 1},
    ) == plan
    assert set(prompts) == {1, 2, 3}
    assert all(set(frames) == {1} for frames in prompts.values())


def test_plan_phase_v4_compiles_one_prompt_for_each_frozen_source_frame(tmp_path):
    session = tmp_path / "session"
    runner = _Runner(_semantic_output)
    segments = _transition_skeleton(_segments(session, [0], frames=2))

    generated, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "independent_parallel",
        session_dir=session,
        expected_version=4,
    )

    assert [call["request"]["phase"] for call in runner.calls] == [
        "global_plan", "segment_frames"
    ]
    assert generated["version"] == 4
    assert set(prompts) == {0} and set(prompts[0]) == {1, 2}
    assert prompts[0][1] != prompts[0][2]
    assert "frame-001 当前可见身体与姿态" in prompts[0][1]
    assert "frame-002 当前可见身体与姿态" in prompts[0][2]


def test_legacy_ineligible_plan_remains_runtime_compatibility_only():
    legacy = _ineligible()
    assert image_optimization.canonical_plan_v2(legacy) == legacy


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(version=1),
        lambda value: value["person_plans"].clear(),
        lambda value: value["person_plans"][0].update(observable_segments=[]),
        lambda value: value["segments"][0]["persons"].clear(),
        lambda value: value["segments"][0]["persons"][0].update(state="protected"),
        lambda value: value["scene_plans"][0].update(geometry_changes=[]),
        lambda value: value["scene_plans"][0].update(depth_changes=[]),
        lambda value: value["scene_plans"][0].update(layout_changes=[]),
        lambda value: value["scene_plans"][0].update(semantic_change=""),
        lambda value: value["scene_plans"][0].update(local_color_change=""),
        lambda value: value["scene_plans"][0].update(segments=[1]),
        lambda value: value["segments"][0]["scene"].update(scene_id="SCENE_99"),
    ],
)
def test_plan_rejects_missing_person_or_real_scene_invariants(mutate):
    value = _plan()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.freeze_continuity(value)


@pytest.mark.parametrize(
    "value",
    [
        {
            **_ineligible(),
            "reason": "unknown_reason",
        },
        {
            **_ineligible(),
            "person_plans": _plan([0])["person_plans"],
        },
        {
            **_ineligible(),
            "segments": _plan([0])["segments"],
        },
    ],
)
def test_ineligible_shape_is_exact_and_reason_is_closed(value):
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.freeze_continuity(value)


def test_multiple_people_are_explicit_and_frame_observability_is_derived():
    plan = _plan()
    second = {
        "id": "PERSON_02",
        "source_identity": "第二位叙事主人物",
        "replacement_identity": "保持可见人口属性但长相不同的第二位新人物",
        "wardrobe_change": "保持用途并更换为不同款式",
        "local_color_change": "人物局部固有色产生另一种可见变化",
        "reference": {"segment_index": 2, "frame_index": 1},
        "observable_segments": [2],
    }
    plan["person_plans"].append(second)
    plan["segments"][0]["persons"].append(
        {
            "id": "PERSON_02",
            "state": "not_observable",
            "observable_frames": [],
            "target_region": None,
            "boundary": None,
        }
    )
    plan["segments"][1]["persons"].append(
        {
            "id": "PERSON_02",
            "state": "replace",
            "observable_frames": [1],
            "target_region": "第二位主人物",
            "boundary": "第二位人物轮廓",
        }
    )

    frozen = image_optimization.freeze_continuity(plan)["_image_continuity"]
    assert [item["id"] for item in frozen["person_plans"]] == [
        "PERSON_01",
        "PERSON_02",
    ]
    context = image_optimization.semantic_context(plan)
    assert context["segments"][0]["observable_person_ids"] == ["PERSON_01"]
    assert context["segments"][1]["observable_person_ids"] == [
        "PERSON_01",
        "PERSON_02",
    ]


def test_reference_slots_and_execution_freeze_are_deterministic():
    plan = _plan()
    slots = image_optimization.reference_slots(plan)
    assert slots == {
        "identity": [
            {
                "role": "identity:PERSON_01",
                "person_id": "PERSON_01",
                "segment_index": 1,
                "frame_index": 1,
            }
        ],
        "scene": [
            {
                "role": "scene:SCENE_01",
                "scene_id": "SCENE_01",
                "segment_index": 1,
                "frame_index": 1,
            }
        ],
        "layout": [
            {
                "segment_index": 1,
                "role": "layout:SCENE_01",
                "scene_id": "SCENE_01",
                "frame_index": 1,
            },
            {
                "segment_index": 2,
                "role": "layout:SCENE_01",
                "scene_id": "SCENE_01",
                "frame_index": 1,
            },
        ],
    }
    inventory = [
        {
            "segment_index": index,
            "frame_index": 1,
            "frame_name": "01.png",
            "source_sha256": str(index) * 64,
        }
        for index in (1, 2)
    ]
    frozen = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 2},
        model="doubao-seedream-5-0-pro-260628",
        frame_inventory=inventory,
    )
    assert frozen["version"] == 2
    assert frozen["plan_sha256"] == image_optimization.plan_sha256(plan)
    assert frozen["identity_slots"][0]["source_sha256"] == "1" * 64
    assert frozen["scene_slots"][0]["source_sha256"] == "1" * 64
    assert frozen["frames"] == [
        {
            **inventory[0],
            "observable_person_ids": ["PERSON_01"],
            "scene_id": "SCENE_01",
        },
        {
            **inventory[1],
            "observable_person_ids": ["PERSON_01"],
            "scene_id": "SCENE_01",
        },
    ]


def test_execution_freeze_rejects_inventory_missing_an_observable_frame():
    plan = _plan()
    plan["segments"][0]["persons"][0]["observable_frames"] = [1, 2]
    with pytest.raises(ValueError, match="execution inputs"):
        image_optimization.freeze_execution_inputs(
            plan,
            revision=1,
            profile={"id": "dual-target", "revision": 2},
            model="doubao-seedream-5-0-pro-260628",
            frame_inventory=[
                {
                    "segment_index": index,
                    "frame_index": 1,
                    "frame_name": "01.png",
                    "source_sha256": str(index) * 64,
                }
                for index in (1, 2)
            ],
        )


def test_v2_receipt_is_authoritative_and_v1_is_read_only_compatible():
    v2 = image_optimization.freeze_continuity(_plan())
    receipt = image_optimization.dual_target_plan_receipt(v2)
    assert receipt == v2["_image_continuity"]
    tampered = deepcopy(v2)
    tampered["_image_continuity"]["person_plans"][0][
        "replacement_identity"
    ] = "tampered"
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.dual_target_plan_receipt(tampered)

    legacy = {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": [
            {
                "id": "PERSON_01",
                "kind": "person",
                "source": "旧人物",
                "replacement": "旧替换",
                "segments": [1, 2],
            }
        ],
    }
    old = image_optimization.freeze_continuity(legacy)
    assert image_optimization.continuity_receipt(old)["version"] == 1
    assert image_optimization.dual_target_plan_receipt(old) is None


def test_free_prompt_patch_cannot_override_v2_compiler(tmp_path):
    settings = make_settings(tmp_path)
    plan_receipt = image_optimization.freeze_continuity(_plan([0]))
    meta = {
        "schema_version": 2,
        "status": "done",
        **plan_receipt,
    }
    prompt = image_optimization.compile_segment_prompts(
        _plan([0]), "independent_parallel"
    )[0]
    meta.update(image_optimization.freeze_prompts(settings, meta, {0: prompt}))

    with pytest.raises(image_optimization.ImageOptimizationError) as caught:
        image_optimization.replace(
            meta,
            settings,
            0,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "删掉人物和场景替换，只做美化",
        )
    assert caught.value.status == 409
    assert caught.value.detail == "image_optimization_prompt_compiled"


def test_verify_phase_stages_only_allowed_inputs_and_returns_exact_verdict(tmp_path):
    session = tmp_path / "session"
    source = session / "source"
    output = session / "output"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    (source / "01.png").write_bytes(_png(1))
    (output / "01.png").write_bytes(_png(2))
    plan = _plan([0])
    verdict = _verdict(plan)
    runner = _Runner(verdict)

    actual = image_optimization.generate_project_verdict(
        runner,
        plan,
        [
            {
                "index": 0,
                "source_keyframes_dir": source,
                "output_keyframes_dir": output,
            }
        ],
        {"version": 1, "segments": [{"segment_index": 0}]},
        session_dir=session,
    )

    assert actual == verdict
    assert runner.calls[0]["request"] == {
        "phase": "verify",
        "segment_indices": [0],
    }
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/frozen_plan.json",
        "work/metrics.json",
        "work/request.json",
        "work/segments/0/output/01.png",
        "work/segments/0/source/01.png",
    ]


def test_verify_rejects_unknown_or_inconsistent_success():
    plan = _plan()
    verdict = _verdict(plan)
    verdict["segments"][0]["scene_checks"]["semantic_change"] = _check(
        "unknown", "无法判定"
    )
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(verdict, plan)


def test_verify_accepts_canonical_fail_closed_failure():
    plan = _plan()
    verdict = _verdict(plan, passed=False)
    assert image_optimization.canonical_verification(verdict, plan) == verdict


def test_verify_pack_stages_exact_semantic_inputs_and_returns_verdict(tmp_path):
    session = tmp_path / "session"
    source = session / "source.png"
    person_primary = session / "person-primary.png"
    person_alternate = session / "person-alternate.png"
    scene_primary = session / "scene-primary.png"
    scene_alternate = session / "scene-alternate.png"
    session.mkdir()
    for index, path in enumerate(
        (
            source,
            person_primary,
            person_alternate,
            scene_primary,
            scene_alternate,
        ),
        start=1,
    ):
        path.write_bytes(_png(index))
    plan = _plan([0])
    verdict = _pack_verdict(plan)
    runner = _Runner(verdict)

    actual = image_optimization.generate_reference_pack_verdict(
        runner,
        plan,
        [
            {
                "person_id": "PERSON_01",
                "source_path": source,
                "primary_path": person_primary,
                "alternate_path": person_alternate,
            }
        ],
        [
            {
                "scene_id": "SCENE_01",
                "source_path": source,
                "primary_path": scene_primary,
                "alternate_path": scene_alternate,
            }
        ],
        {"version": 1, "pack_metrics": []},
        session_dir=session,
    )

    assert actual == verdict
    assert runner.calls[0]["request"] == {
        "phase": "verify_pack",
        "plan_sha256": image_optimization.plan_sha256(plan),
        "person_ids": ["PERSON_01"],
        "scene_ids": ["SCENE_01"],
    }
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/frozen_plan.json",
        "work/metrics.json",
        "work/reference_packs/persons/PERSON_01/alternate.png",
        "work/reference_packs/persons/PERSON_01/primary.png",
        "work/reference_packs/persons/PERSON_01/source.png",
        "work/reference_packs/scenes/SCENE_01/alternate.png",
        "work/reference_packs/scenes/SCENE_01/primary.png",
        "work/reference_packs/scenes/SCENE_01/source.png",
        "work/request.json",
    ]
    assert not any(
        key.endswith("_path")
        for key in runner.calls[0]["request"]
    )


def test_verify_pack_accepts_canonical_fail_closed_failure():
    plan = _plan()
    verdict = _pack_verdict(plan, passed=False)
    assert (
        image_optimization.canonical_reference_pack_verdict(verdict, plan)
        == verdict
    )


def test_verify_pack_accepts_unknown_as_fail_closed_with_stable_reason():
    plan = _plan()
    verdict = _pack_verdict(plan)
    verdict["persons"][0]["checks"]["identity_changed"] = _check(
        "unknown", "可见证据不足"
    )
    verdict["persons"][0]["passed"] = False
    verdict["passed"] = False
    verdict["reason"] = "pack_verification_unknown"

    assert (
        image_optimization.canonical_reference_pack_verdict(verdict, plan)
        == verdict
    )


def test_skill_describes_per_frame_semantics_while_backend_builds_contracts(
    tmp_path,
):
    skill, example = _skill_contract()

    frame = next(iter(example["frames"].values()))
    assert set(frame) == {"people", "relationships", "entities", "crop"}
    person = next(iter(frame["people"].values()))
    assert set(person) == {
        "visible_region", "boundary", "body_and_pose", "derived_observations",
    }
    assert "每帧只描述当前帧直接可见" in skill
    plan, diagnostics = _compiled_semantic(tmp_path, frames=2)
    segment = plan["segments"][0]
    assert [item["frame_index"] for item in segment["frame_constraints"]] == [1, 2]
    assert set(segment["photometric_contract"]) == set(
        image_optimization._PHOTOMETRIC_CONTRACT_KEYS
    )
    assert diagnostics["score"] == 1.0

def test_skill_uses_current_frame_descriptions_not_backend_fragment_enums():
    skill, example = _skill_contract()
    encoded = json.dumps(example, ensure_ascii=False)

    for semantic in ("visible_region", "boundary", "body_and_pose", "crop"):
        assert semantic in encoded
    for mechanical in (
        "visible_body_parts", "non_person_entity_ledger", "partial",
        "cropped", "frame_constraints",
    ):
        assert mechanical not in encoded
    assert "不从其他帧补造" in skill


def test_skill_keeps_derived_person_observations_nested_and_semantic_only():
    skill, example = _skill_contract()
    person = next(iter(next(iter(example["frames"].values()))["people"].values()))
    observation = next(iter(person["derived_observations"].values()))

    assert set(observation) == {
        "mode", "source_carrier", "visible_region", "boundary", "relationship",
    }
    for mode in ("optical_projection", "temporal_residual", "source-preserve"):
        assert mode in skill
    for contract in (
        "不代表独立物理人物",
        "不新增顶层人物或实体",
        "不把该观测实例化到新场景",
        "source-preserve/non-physical",
        "不拒绝、不 retry、不 fallback",
    ):
        assert contract in skill
    for mechanical in (
        "observation_id", "physicality", "instantiation", "frame_constraints",
    ):
        assert mechanical not in json.dumps(example, ensure_ascii=False)


def test_skill_closes_each_frame_person_count_and_body_part_ownership():
    skill, example = _skill_contract()
    frame = next(iter(example["frames"].values()))
    person = next(iter(frame["people"].values()))
    observation = next(iter(person["derived_observations"].values()))

    assert set(example) == {"people", "entities", "scenes", "frames"}
    assert set(frame) == {"people", "relationships", "entities", "crop"}
    assert set(person) == {
        "visible_region", "boundary", "body_and_pose", "derived_observations",
    }
    assert set(observation) == {
        "mode", "source_carrier", "visible_region", "boundary", "relationship",
    }
    assert observation["mode"] in {
        "optical_projection", "temporal_residual", "source-preserve",
    }
    for rule in (
        "物理人物全集",
        "人物数量闭合",
        "头、躯干和手",
        "唯一归属",
        "反射",
        "残影",
        "边缘碎片",
        "遮挡碎片",
        "运动模糊",
        "不得升级为新物理人物",
        "source-preserve/no-invention",
    ):
        assert rule in skill
    encoded = json.dumps(example, ensure_ascii=False)
    for forbidden_field in (
        "physical_person_count", "body_part_ledger", "ambiguity_gate",
    ):
        assert forbidden_field not in encoded


def test_skill_keeps_entity_relationships_semantic_and_non_refusing():
    skill, example = _skill_contract()
    frame = next(iter(example["frames"].values()))

    assert "实体关系" in skill
    assert "entities" in frame and "relationships" in frame
    assert "source-preserve/no-invention" in skill
    assert "流程判断" in skill
    for refusal in ("eligible=false", "scene_components_ambiguous"):
        assert refusal not in skill


def test_legacy_ineligible_reason_remains_backend_only_not_skill_authority():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    human = Path(
        "docs/human/features/conversation-task/behaviors/postprocess.md"
    ).read_text(encoding="utf-8")
    reason = "scene_structure_replacement_unsafe"

    assert reason in image_optimization._INELIGIBLE_REASONS
    assert image_optimization.canonical_plan_v2(_ineligible(reason)) == _ineligible(reason)
    assert reason not in skill
    assert reason not in human
    assert "scene_replacement_unsafe" not in skill
    assert "scene_replacement_unsafe" not in human


def test_backend_not_skill_owns_exact_fields_and_current_frame_indices(tmp_path):
    skill, example = _skill_contract()
    plan, _diagnostics = _compiled_semantic(tmp_path, frames=2)

    assert "只填写视觉语义" in skill
    assert "frame_constraints" not in json.dumps(example, ensure_ascii=False)
    assert [
        frame["frame_index"]
        for frame in plan["segments"][0]["frame_constraints"]
    ] == [1, 2]
    assert plan["segments"][0]["scene"]["layout_reference_frame_index"] == 1


def test_missing_current_frame_semantics_are_diagnostics_not_admission_gate(
    tmp_path,
):
    skill, _example = _skill_contract()
    plan, diagnostics = _compiled_semantic(tmp_path, sparse=True)

    assert plan["version"] == 4 and plan["eligible"] is True
    assert plan["segments"][0]["persons"] == []
    assert diagnostics["score"] < 1.0
    assert diagnostics["issues"]
    assert "blocking" not in diagnostics
    assert "eligible" not in json.dumps(_skill_contract()[1])


def test_v4_skill_compiles_valid_frozen_inputs_and_preserves_ambiguous_regions():
    skill, example = _skill_contract()

    assert "source-preserve/no-invention" in skill
    assert "不可见或无法唯一判断" in skill
    assert "不从其他帧补造" in skill
    assert "eligible" not in json.dumps(example)
    assert "reason" not in json.dumps(example)
    assert "素材准入" not in skill


def test_skill_scopes_visible_relationships_without_content_failure_reasons():
    skill, example = _skill_contract()

    assert "当前帧直接可见" in skill
    assert "不从其他帧补造" in skill
    assert "source-preserve/no-invention" in skill
    assert "relationships" in next(iter(example["frames"].values()))

    for reason in (
        "person_replacement_unsafe",
        "scene_components_ambiguous",
        "scene_structure_replacement_unsafe",
    ):
        assert reason not in skill


def _plan_v3() -> dict:
    plan = _plan([0])
    plan["version"] = 3
    plan["segments"][0]["persons"][0]["observable_frames"] = [1, 2]
    plan["segments"][0]["frame_constraints"] = [
        {
            "frame_index": 1,
            "visible_body_parts": "可见部位数量与边界保持当前源帧",
            "pose_skeleton": "姿态骨架保持当前源帧",
            "contact_points": "接触点保持当前源帧",
            "occlusion_order": "遮挡前后顺序保持当前源帧",
            "out_of_frame_crop": "画外裁切保持当前源帧",
            "non_person_entity_ledger": {
                "entities": [{
                    "entity_id": "ENTITY_01",
                    "description": "当前帧可见操作实体",
                    "visibility": "full",
                }],
                "relations": [{
                    "subject_id": "ENTITY_01",
                    "predicate": "contacts",
                    "object_id": "PERSON_01",
                }],
            },
            "dominant_palette_contract": {
                "area_weighted_warm_cool_family": "warm",
                "saturation_style": "natural",
            },
        },
        {
            "frame_index": 2,
            "visible_body_parts": "第二帧可见部位数量与边界保持当前源帧",
            "pose_skeleton": "第二帧姿态骨架保持当前源帧",
            "contact_points": "第二帧接触点保持当前源帧",
            "occlusion_order": "第二帧遮挡前后顺序保持当前源帧",
            "out_of_frame_crop": "第二帧画外裁切保持当前源帧",
            "non_person_entity_ledger": {
                "entities": [{
                    "entity_id": "ENTITY_01",
                    "description": "第二帧可见操作实体",
                    "visibility": "edge_fragment",
                }],
                "relations": [{
                    "subject_id": "ENTITY_01",
                    "predicate": "contacts",
                    "object_id": "PERSON_01",
                }],
            },
            "dominant_palette_contract": {
                "area_weighted_warm_cool_family": "cool",
                "saturation_style": "muted",
            },
        },
    ]
    plan["segments"][0]["photometric_contract"] = {
        "light_direction": "全局光源方向保持当前源帧",
        "light_quality": "全局光线软硬保持当前源帧",
        "exposure_or_intensity": "全局曝光与强度保持当前源帧",
        "wb_cct": "白平衡与色温保持当前源帧",
        "global_contrast": "全局对比保持当前源帧",
        "tone_curve": "全局 tone curve 保持当前源帧",
    }
    return plan


def test_plan_audit_is_source_bound_and_fails_closed_per_frame():
    plan = _plan_v3()
    inventory = [
        {
            "segment_index": 0,
            "frame_index": index,
            "frame_name": f"{index:02d}.png",
            "source_sha256": str(index) * 64,
        }
        for index in (1, 2)
    ]
    receipt = image_optimization.freeze_plan_audit_inputs(
        plan, frame_inventory=inventory
    )
    assert set(receipt) == {
        "version", "plan_sha256", "continuity_sha256", "frames", "sha256"
    }
    assert receipt["plan_sha256"] == image_optimization.plan_sha256(plan)
    assert receipt["frames"] == inventory

    def verdict(status: str, reason: str | None) -> dict:
        return {
            "version": 3,
            "phase": "plan_audit",
            "plan_sha256": receipt["plan_sha256"],
            "continuity_sha256": receipt["continuity_sha256"],
            "audit_input_sha256": receipt["sha256"],
            "passed": status == "pass",
            "reason": reason,
            "frame_checks": [
                {
                    "segment_index": item["segment_index"],
                    "frame_index": item["frame_index"],
                    "source_sha256": item["source_sha256"],
                    "body_closure": _check(status),
                    "scene_closure": _check(status),
                    "entity_closure": _check(status),
                    "relation_closure": _check(status),
                }
                for item in receipt["frames"]
            ],
        }

    passed = verdict("pass", None)
    assert image_optimization.canonical_plan_audit_verdict(
        passed, plan, receipt
    ) == passed

    failed = verdict("fail", "plan_audit_failed")
    assert image_optimization.canonical_plan_audit_verdict(
        failed, plan, receipt
    )["passed"] is False

    unknown = verdict("unknown", "plan_audit_unknown")
    assert image_optimization.canonical_plan_audit_verdict(
        unknown, plan, receipt
    )["reason"] == "plan_audit_unknown"

    tampered = deepcopy(receipt)
    tampered["frames"][0]["source_sha256"] = "3" * 64
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_audit_verdict(passed, plan, tampered)


def test_plan_audit_stages_only_receipted_source_frames(tmp_path):
    session = tmp_path / "session"
    source = session / "source"
    source.mkdir(parents=True)
    frames = []
    for index in (1, 2):
        path = source / f"{index:02d}.png"
        path.write_bytes(_png(index))
        frames.append(path)
    plan = _plan_v3()
    inventory = [
        {
            "segment_index": 0,
            "frame_index": index,
            "frame_name": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for index, path in enumerate(frames, 1)
    ]
    receipt = image_optimization.freeze_plan_audit_inputs(
        plan, frame_inventory=inventory
    )
    output = {
        "version": 3,
        "phase": "plan_audit",
        "plan_sha256": receipt["plan_sha256"],
        "continuity_sha256": receipt["continuity_sha256"],
        "audit_input_sha256": receipt["sha256"],
        "passed": True,
        "reason": None,
        "frame_checks": [
            {
                "segment_index": item["segment_index"],
                "frame_index": item["frame_index"],
                "source_sha256": item["source_sha256"],
                "body_closure": _check(),
                "scene_closure": _check(),
                "entity_closure": _check(),
                "relation_closure": _check(),
            }
            for item in inventory
        ],
    }
    runner = _Runner(output)

    assert image_optimization.generate_plan_audit_verdict(
        runner,
        plan,
        receipt,
        [{"index": 0, "source_keyframes_dir": source}],
        session_dir=session,
    ) == output
    assert runner.calls[0]["request"] == {
        "phase": "plan_audit",
        "plan_sha256": receipt["plan_sha256"],
        "continuity_sha256": receipt["continuity_sha256"],
        "audit_input_sha256": receipt["sha256"],
        "segment_indices": [0],
    }
    assert "work/segments/0/source/01.png" in runner.calls[0]["files"]
    assert not any("output" in name for name in runner.calls[0]["files"])

    frames[0].write_bytes(_png(9))
    rejected = _Runner(output)
    with pytest.raises(ValueError):
        image_optimization.generate_plan_audit_verdict(
            rejected,
            plan,
            receipt,
            [{"index": 0, "source_keyframes_dir": source}],
            session_dir=session,
        )
    assert rejected.calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["persons"][0]["checks"]["identity_changed"].update(
            status="unknown"
        ),
        lambda value: value["scenes"][0]["checks"]["layout"].update(
            status="not_applicable"
        ),
        lambda value: value["persons"][0].update(person_id="PERSON_99"),
        lambda value: value["project"].pop("wb_cct_preservation"),
    ],
)
def test_verify_pack_rejects_unknown_or_inconsistent_success(mutate):
    plan = _plan()
    verdict = _pack_verdict(plan)
    mutate(verdict)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_reference_pack_verdict(verdict, plan)


def test_verify_pack_rejects_paths_outside_session_before_runner(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    inside = session / "inside.png"
    outside = tmp_path / "outside.png"
    inside.write_bytes(_png(1))
    outside.write_bytes(_png(2))
    runner = _Runner(_pack_verdict(_plan([0])))

    with pytest.raises(ValueError, match="reference pack verification input"):
        image_optimization.generate_reference_pack_verdict(
            runner,
            _plan([0]),
            [
                {
                    "person_id": "PERSON_01",
                    "source_path": outside,
                    "primary_path": inside,
                    "alternate_path": inside,
                }
            ],
            [
                {
                    "scene_id": "SCENE_01",
                    "source_path": inside,
                    "primary_path": inside,
                    "alternate_path": inside,
                }
            ],
            {},
            session_dir=session,
        )
    assert runner.calls == []


def test_verify_pack_accepts_v3_plan_without_dropping_frame_contracts():
    plan = _plan_v3()
    verdict = _pack_verdict(plan)

    assert (
        image_optimization.canonical_reference_pack_verdict(verdict, plan)
        == verdict
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["frame_constraints"].pop(),
        lambda value: value["segments"][0]["frame_constraints"].append(
            deepcopy(value["segments"][0]["frame_constraints"][0])
        ),
        lambda value: value["segments"][0]["frame_constraints"][1].update(
            frame_index=3
        ),
        lambda value: value["segments"][0]["photometric_contract"].pop(
            "tone_curve"
        ),
    ],
)
def test_v3_frame_contract_fails_closed_on_missing_duplicate_or_tampered_constraints(
    mutate,
):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["frame_constraints"][0].pop(
            "non_person_entity_ledger"
        ),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ].update(extra="forbidden"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["entities"][0].update(extra="forbidden"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(object_id="ENTITY_99"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["entities"][0].update(visibility=[]),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(predicate={}),
    ],
)
def test_v3_frame_entity_ledger_is_exact_and_fails_closed(mutate):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["frame_constraints"][0].update(
            non_person_entity_ledger={"entities": [], "relations": []}
        ),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ].update(relations=[]),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["entities"].append({
            "entity_id": "ENTITY_02",
            "description": "未参与关系的当前帧实体",
            "visibility": "full",
        }),
    ],
)
def test_v3_frame_entity_ledger_requires_visible_entities_and_relations(mutate):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["entities"][0].update(entity_id="ENTITY_02"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(extra="forbidden"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(object_id="PERSON_02"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(object_id="ENTITY_01"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"].append({
            "subject_id": "ENTITY_01",
            "predicate": "separate_from",
            "object_id": "PERSON_01",
        }),
    ],
)
def test_v3_frame_entity_ledger_rejects_unresolved_or_ambiguous_relations(mutate):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["scene"].pop(
            "layout_reference_frame_index"
        ),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"][0].update(
            subject_id="PERSON_01", object_id="ENTITY_01"
        ),
    ],
)
def test_v3_backend_exact_scene_and_symmetric_relation_order_fail_closed(mutate):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


def _set_entity_relation_cycle(value: dict, predicate: str) -> None:
    ledger = value["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]
    ledger["entities"] = [
        {
            "entity_id": f"ENTITY_{index:02d}",
            "description": f"当前帧可见实体{index}",
            "visibility": "full",
        }
        for index in range(1, 4)
    ]
    ledger["relations"] = [
        {
            "subject_id": f"ENTITY_{index:02d}",
            "predicate": predicate,
            "object_id": f"ENTITY_{index % 3 + 1:02d}",
        }
        for index in range(1, 4)
    ]


def _add_unsorted_entity_relation(value: dict) -> None:
    ledger = value["segments"][0]["frame_constraints"][0][
        "non_person_entity_ledger"
    ]
    ledger["entities"].append({
        "entity_id": "ENTITY_02",
        "description": "第二个当前帧可见实体",
        "visibility": "full",
    })
    ledger["relations"].insert(0, {
        "subject_id": "ENTITY_02",
        "predicate": "supports",
        "object_id": "ENTITY_01",
    })


@pytest.mark.parametrize(
    "mutate",
    [
        _add_unsorted_entity_relation,
        lambda value: value["segments"][0]["frame_constraints"][0][
            "non_person_entity_ledger"
        ]["relations"].append({
            "subject_id": "ENTITY_01",
            "predicate": "supports",
            "object_id": "PERSON_01",
        }),
        lambda value: _set_entity_relation_cycle(value, "supports"),
        lambda value: _set_entity_relation_cycle(value, "occludes"),
        lambda value: value["segments"][0]["persons"][0].update(
            observable_frames=[1]
        ),
    ],
)
def test_v3_frame_entity_ledger_rejects_ambiguous_graphs_and_nonobservable_people(
    mutate,
):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


def test_v3_frame_entity_ledger_roundtrips_without_cross_frame_identity():
    value = _plan_v3()
    canonical = image_optimization.canonical_plan_v3(
        value, segment_indices=[0], frame_counts={0: 2}
    )
    assert canonical == value
    ledgers = [
        item["non_person_entity_ledger"]
        for item in canonical["segments"][0]["frame_constraints"]
    ]
    assert [ledger["entities"][0]["entity_id"] for ledger in ledgers] == [
        "ENTITY_01", "ENTITY_01"
    ]
    assert ledgers[1]["entities"][0]["visibility"] == "edge_fragment"


def _generic_scene_continuity_plan() -> dict:
    return {
        "version": 4,
        "phase": "plan",
        "segment_indices": [1, 2],
        "eligible": True,
        "reason": None,
        "person_plans": [{
            "id": "PERSON_01",
            "source_identity": "source-identity-spec",
            "replacement_identity": "target-identity-spec",
            "wardrobe_change": "target-wardrobe-spec",
            "local_color_change": "target-local-appearance-spec",
            "reference": {"segment_index": 1, "frame_index": 1},
            "observable_segments": [1, 2],
        }],
        "scene_plans": [{
            "id": "SCENE_01",
            "source_scene": "source-scene-spec",
            "replacement_scene": "target-scene-spec",
            "semantic_change": "target-semantic-spec",
            "geometry_changes": ["target-geometry-spec"],
            "depth_changes": ["target-depth-spec"],
            "layout_changes": ["target-layout-spec"],
            "local_color_change": "target-local-surface-spec",
            "reference": {"segment_index": 1, "frame_index": 1},
            "segments": [1, 2],
            "continuity_graph": {
                "components": [
                    {
                        "component_id": "COMPONENT_01",
                        "target_spec": "target-spec-01",
                    },
                    {
                        "component_id": "COMPONENT_02",
                        "target_spec": "target-spec-02",
                    },
                ],
                "topology": [{
                    "subject_id": "COMPONENT_01",
                    "predicate": "supports",
                    "object_id": "COMPONENT_02",
                }],
                "views": [
                    {
                        "segment_index": 1,
                        "frame_index": 1,
                        "transition_from_previous": "start",
                        "observations": [
                            {"component_id": "COMPONENT_01", "visibility": "full"},
                            {"component_id": "COMPONENT_02", "visibility": "full"},
                        ],
                        "view_relations": [{
                            "subject_id": "COMPONENT_01",
                            "predicate": "in_front_of",
                            "object_id": "COMPONENT_02",
                        }],
                    },
                    {
                        "segment_index": 2,
                        "frame_index": 1,
                        "transition_from_previous": "same_camera",
                        "observations": [
                            {"component_id": "COMPONENT_01", "visibility": "full"},
                            {"component_id": "COMPONENT_02", "visibility": "full"},
                        ],
                        "view_relations": [{
                            "subject_id": "COMPONENT_01",
                            "predicate": "in_front_of",
                            "object_id": "COMPONENT_02",
                        }],
                    },
                ],
            },
        }],
        "segments": [
            {
                "segment_index": segment_index,
                "persons": [{
                    "id": "PERSON_01",
                    "state": "replace",
                    "observable_frames": [1],
                    "target_region": "target-region-spec",
                    "boundary": "target-boundary-spec",
                }],
                "scene": {
                    "scene_id": "SCENE_01",
                    "target_region": "scene-region-spec",
                    "boundary": "scene-boundary-spec",
                    "layout_reference_frame_index": 1,
                },
                "protected_non_target_people": [],
                "protected_relations": ["protected-relation-spec"],
                "frame_constraints": [{
                    "frame_index": 1,
                    "visible_body_parts": "frame-body-spec",
                    "pose_skeleton": "frame-pose-spec",
                    "contact_points": "frame-contact-spec",
                    "occlusion_order": "frame-occlusion-spec",
                    "out_of_frame_crop": "frame-crop-spec",
                    "non_person_entity_ledger": {
                        "entities": [{
                            "entity_id": "ENTITY_01",
                            "description": f"source-region-{segment_index:02d}",
                            "visibility": "full",
                        }],
                        "relations": [{
                            "subject_id": "ENTITY_01",
                            "predicate": "contacts",
                            "object_id": "PERSON_01",
                        }],
                    },
                    "dominant_palette_contract": {
                        "area_weighted_warm_cool_family": "balanced",
                        "saturation_style": "natural",
                    },
                }],
                "photometric_contract": {
                    "light_direction": "frame-light-direction-spec",
                    "light_quality": "frame-light-quality-spec",
                    "exposure_or_intensity": "frame-exposure-spec",
                    "wb_cct": "frame-wb-spec",
                    "global_contrast": "frame-contrast-spec",
                    "tone_curve": "frame-tone-spec",
                },
            }
            for segment_index in (1, 2)
        ],
    }


def test_v4_graph_closes_identity_gap_accepted_by_historical_v3():
    plan = _generic_scene_continuity_plan()
    historical = deepcopy(plan)
    historical["version"] = 3
    historical["scene_plans"][0].pop("continuity_graph")

    assert image_optimization.canonical_plan_v3(
        historical, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    ) == historical
    assert image_optimization.canonical_plan_v4(
        plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    ) == plan


def _add_redundant_view_pair_relation(value: dict) -> None:
    value["scene_plans"][0]["continuity_graph"]["views"][0][
        "view_relations"
    ].append({
        "subject_id": "COMPONENT_01",
        "predicate": "occludes",
        "object_id": "COMPONENT_02",
    })


def _make_view_relation_endpoint_out_of_view(value: dict) -> None:
    value["scene_plans"][0]["continuity_graph"]["views"][0][
        "observations"
    ][1]["visibility"] = "out_of_view"


def _make_occluding_subject_not_visible(value: dict) -> None:
    view = value["scene_plans"][0]["continuity_graph"]["views"][0]
    view["observations"][0]["visibility"] = "out_of_view"
    view["view_relations"][0]["predicate"] = "occludes"


def _add_redundant_topology_pair_relation(value: dict) -> None:
    value["scene_plans"][0]["continuity_graph"]["topology"].insert(0, {
        "subject_id": "COMPONENT_01",
        "predicate": "contacts",
        "object_id": "COMPONENT_02",
    })


@pytest.mark.parametrize(
    "mutate",
    [
        _add_redundant_view_pair_relation,
        _make_view_relation_endpoint_out_of_view,
        _make_occluding_subject_not_visible,
        _add_redundant_topology_pair_relation,
    ],
)
def test_v4_graph_rejects_redundant_pairs_and_invisible_endpoints(mutate):
    plan = _generic_scene_continuity_plan()
    mutate(plan)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
        )


def test_v4_transition_is_backend_authority_not_skill_content_gate(tmp_path):
    segments = _transition_skeleton(
        _segments(tmp_path / "session", [0], frames=2)
    )
    segments[0]["transition_skeleton"][1][
        "source_transition_from_previous"
    ] = "camera_motion"
    semantic = _semantic_output({
        "semantic_slots": image_optimization.semantic_slot_manifest(segments)
    })

    plan, diagnostics = image_optimization.compile_semantic_plan(
        semantic,
        segments,
        source_frames={
            0: sorted(Path(segments[0]["keyframes_dir"]).glob("*.png"))
        },
    )

    views = plan["scene_plans"][0]["continuity_graph"]["views"]
    assert [item["transition_from_previous"] for item in views] == [
        "start", "camera_motion"
    ]
    assert diagnostics["score"] == 1.0
    assert "transition_from_previous" not in json.dumps(_skill_contract()[1])


def _make_topology_cycle(value: dict) -> None:
    graph = value["scene_plans"][0]["continuity_graph"]
    graph["components"].append({
        "component_id": "COMPONENT_03",
        "target_spec": "target-spec-03",
    })
    graph["topology"] = [
        {
            "subject_id": "COMPONENT_01",
            "predicate": "supports",
            "object_id": "COMPONENT_02",
        },
        {
            "subject_id": "COMPONENT_02",
            "predicate": "supports",
            "object_id": "COMPONENT_03",
        },
        {
            "subject_id": "COMPONENT_03",
            "predicate": "supports",
            "object_id": "COMPONENT_01",
        },
    ]
    for view in graph["views"]:
        view["observations"].append({
            "component_id": "COMPONENT_03",
            "visibility": "full",
        })


def _make_component_never_visible(value: dict) -> None:
    for view in value["scene_plans"][0]["continuity_graph"]["views"]:
        view["observations"][1]["visibility"] = "occluded"
        view["view_relations"][0]["predicate"] = "occludes"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "components"
        ][1].update(component_id="COMPONENT_03"),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "components"
        ][1].update(target_spec="ENTITY_01"),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "topology"
        ][0].update(object_id="COMPONENT_99"),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "topology"
        ][0].update(predicate={}),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "topology"
        ][0].update(subject_id=7),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "views"
        ][0]["observations"].pop(),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "views"
        ][0]["view_relations"][0].update(object_id="COMPONENT_99"),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "views"
        ][0]["observations"][0].update(visibility=[]),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "views"
        ][1].update(transition_from_previous="start"),
        lambda value: value["scene_plans"][0]["continuity_graph"][
            "views"
        ][0]["observations"][1].update(visibility="occluded"),
        _make_topology_cycle,
        _make_component_never_visible,
    ],
)
def test_v4_graph_is_exact_closed_sorted_and_target_only(mutate):
    plan = _generic_scene_continuity_plan()
    mutate(plan)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
        )


def test_v4_allows_equal_target_specs_and_stable_isolated_components():
    plan = _generic_scene_continuity_plan()
    graph = plan["scene_plans"][0]["continuity_graph"]
    graph["components"][1]["target_spec"] = graph["components"][0]["target_spec"]
    graph["topology"] = []

    assert image_optimization.canonical_plan_v4(
        plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    ) == plan


@pytest.mark.parametrize("visibility", ["partial", "occluded"])
def test_v4_partial_and_occluded_require_incoming_visible_occlusion(visibility):
    plan = _generic_scene_continuity_plan()
    view = plan["scene_plans"][0]["continuity_graph"]["views"][0]
    view["observations"][1]["visibility"] = visibility
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
        )


def test_v4_incoming_occlusion_rejects_full_but_allows_edge_fragment_priority():
    plan = _generic_scene_continuity_plan()
    view = plan["scene_plans"][0]["continuity_graph"]["views"][0]
    view["view_relations"][0]["predicate"] = "occludes"
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
        )

    view["observations"][1]["visibility"] = "edge_fragment"
    assert image_optimization.canonical_plan_v4(
        plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    ) == plan


def _split_generic_scene_continuity_plan() -> dict:
    plan = _generic_scene_continuity_plan()
    first = plan["scene_plans"][0]
    second = deepcopy(first)
    first["segments"] = [1]
    first["continuity_graph"]["views"] = [
        first["continuity_graph"]["views"][0]
    ]
    second["id"] = "SCENE_02"
    second["reference"] = {"segment_index": 2, "frame_index": 1}
    second["segments"] = [2]
    second["continuity_graph"]["views"] = [
        second["continuity_graph"]["views"][1]
    ]
    second["continuity_graph"]["views"][0][
        "transition_from_previous"
    ] = "hard_cut"
    plan["scene_plans"].append(second)
    plan["segments"][1]["scene"]["scene_id"] = "SCENE_02"
    return plan


def test_v4_different_scene_starts_fresh_across_hard_cut():
    plan = _split_generic_scene_continuity_plan()
    assert image_optimization.canonical_plan_v4(
        plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
    ) == plan

    plan["scene_plans"][1]["continuity_graph"]["views"][0][
        "transition_from_previous"
    ] = "same_camera"
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v4(
            plan, segment_indices=[1, 2], frame_counts={1: 1, 2: 1}
        )


def _generic_frame_inventory(*, transition_evidence: bool = True) -> list[dict]:
    inventory = [
        {
            "segment_index": segment_index,
            "frame_index": 1,
            "frame_name": "01.png",
            "source_sha256": str(segment_index) * 64,
        }
        for segment_index in (1, 2)
    ]
    if transition_evidence:
        for item, transition, digest in zip(
            inventory, ("start", "same_camera"), ("a" * 64, "b" * 64)
        ):
            item["source_transition_from_previous"] = transition
            item["source_transition_evidence_sha256"] = digest
    return inventory


def test_v4_freeze_rejects_missing_authoritative_transition_evidence():
    with pytest.raises(ValueError, match="execution inputs"):
        image_optimization.freeze_execution_inputs(
            _generic_scene_continuity_plan(),
            revision=1,
            profile={"id": "generic-profile", "revision": 1},
            model="doubao-seedream-5-0-pro-260628",
            frame_inventory=_generic_frame_inventory(transition_evidence=False),
        )


def test_v4_compile_and_freeze_bind_one_shared_graph_and_exact_views(tmp_path):
    settings = make_settings(tmp_path)
    plan = _generic_scene_continuity_plan()
    prompts = image_optimization.compile_frame_prompts(
        plan, settings.seedream_edit_mode
    )
    assert set(prompts) == {1, 2}
    for prompt in (prompts[1][1], prompts[2][1]):
        assert '"component_id":"COMPONENT_01"' in prompt
        assert '"target_spec":"target-spec-02"' in prompt
        assert '"predicate":"supports"' in prompt
    assert '"segment_index":1' in prompts[1][1]
    assert '"segment_index":2' not in prompts[1][1]
    assert '"segment_index":2' in prompts[2][1]
    assert '"segment_index":1' not in prompts[2][1]

    execution = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "generic-profile", "revision": 1},
        model=settings.seedream_model,
        frame_inventory=_generic_frame_inventory(),
    )
    assert execution["version"] == 4
    assert all(
        "scene_continuity_graph" not in frame for frame in execution["frames"]
    )
    assert execution["continuity_sha256"]
    assert execution["sha256"]
    assert [
        frame["scene_continuity_view"]["segment_index"]
        for frame in execution["frames"]
    ] == [1, 2]
    frozen = image_optimization.freeze_frame_prompts(
        settings, execution, prompts, plan=plan
    )
    assert frozen["_image_optimization"]["version"] == 4

    continuity = image_optimization.freeze_continuity(
        plan, frame_counts={1: 1, 2: 1}
    )
    meta = {**continuity, **frozen}
    assert image_optimization.receipt(meta, settings) == frozen[
        "_image_optimization"
    ]
    changed = deepcopy(plan)
    changed["scene_plans"][0]["continuity_graph"]["components"][0][
        "target_spec"
    ] = "target-spec-revision"
    assert image_optimization.plan_sha256(changed) != image_optimization.plan_sha256(
        plan
    )
    assert image_optimization.freeze_continuity(
        changed, frame_counts={1: 1, 2: 1}
    )["_image_continuity"]["sha256"] != continuity["_image_continuity"][
        "sha256"
    ]

    changed_target = deepcopy(plan)
    changed_target["scene_plans"][0]["continuity_graph"]["components"][0][
        "target_spec"
    ] = "target-spec-execution-drift"
    changed_execution = image_optimization.freeze_execution_inputs(
        changed_target,
        revision=1,
        profile={"id": "generic-profile", "revision": 1},
        model=settings.seedream_model,
        frame_inventory=_generic_frame_inventory(),
    )
    with pytest.raises(ValueError, match="frame prompts"):
        image_optimization.freeze_frame_prompts(
            settings,
            changed_execution,
            image_optimization.compile_frame_prompts(
                changed_target, settings.seedream_edit_mode
            ),
            plan=plan,
        )

    changed_view = deepcopy(plan)
    changed_view["scene_plans"][0]["continuity_graph"]["views"][1][
        "transition_from_previous"
    ] = "camera_motion"
    changed_view_inventory = _generic_frame_inventory()
    changed_view_inventory[1]["source_transition_from_previous"] = "camera_motion"
    changed_view_execution = image_optimization.freeze_execution_inputs(
        changed_view,
        revision=1,
        profile={"id": "generic-profile", "revision": 1},
        model=settings.seedream_model,
        frame_inventory=changed_view_inventory,
    )
    with pytest.raises(ValueError, match="frame prompts"):
        image_optimization.freeze_frame_prompts(
            settings,
            changed_view_execution,
            image_optimization.compile_frame_prompts(
                changed_view, settings.seedream_edit_mode
            ),
            plan=plan,
        )


def test_two_phase_semantics_are_mechanically_canonicalized_per_frame(tmp_path):
    session = tmp_path / "session"
    runner = _Runner(_semantic_output)
    segments = _transition_skeleton(_segments(session, indices=[1, 2]))

    plan, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "independent_parallel",
        session_dir=session,
    )

    assert plan["version"] == 4
    assert image_optimization.canonical_plan_v4(
        plan,
        segment_indices=[1, 2],
        frame_counts={1: 1, 2: 1},
    ) == plan
    assert plan["person_plans"] and plan["scene_plans"]
    assert [
        frame["frame_index"]
        for segment in plan["segments"]
        for frame in segment["frame_constraints"]
    ] == [1, 1]
    assert set(prompts) == {1, 2}
    assert set(prompts[1]) == {1}
    assert set(prompts[2]) == {1}


def test_generate_project_prompts_rejects_v4_without_backend_transition_skeleton(tmp_path):
    session = tmp_path / "session"
    with pytest.raises(ValueError, match="image optimization segments"):
        image_optimization.generate_project_prompts(
            _Runner(_generic_scene_continuity_plan()),
            _segments(session, indices=[1, 2]),
            "independent_parallel",
            session_dir=session,
        )


def _generic_plan_audit_verdict(
    plan: dict, receipt: dict, status: str = "pass"
) -> dict:
    return {
        "version": plan["version"],
        "phase": "plan_audit",
        "plan_sha256": receipt["plan_sha256"],
        "continuity_sha256": receipt["continuity_sha256"],
        "audit_input_sha256": receipt["sha256"],
        "passed": status == "pass",
        "reason": None if status == "pass" else (
            "plan_audit_unknown" if status == "unknown" else "plan_audit_failed"
        ),
        "frame_checks": [
            {
                "segment_index": item["segment_index"],
                "frame_index": item["frame_index"],
                "source_sha256": item["source_sha256"],
                "body_closure": _check(status),
                "scene_closure": _check(status),
                "entity_closure": _check(status),
                "relation_closure": _check(status),
                "scene_continuity_closure": _check(status),
            }
            for item in receipt["frames"]
        ],
    }


def test_v4_plan_audit_adds_source_bound_scene_continuity_closure():
    plan = _generic_scene_continuity_plan()
    receipt = image_optimization.freeze_plan_audit_inputs(
        plan, frame_inventory=_generic_frame_inventory()
    )
    verdict = _generic_plan_audit_verdict(plan, receipt)
    assert image_optimization.canonical_plan_audit_verdict(
        verdict, plan, receipt
    ) == verdict

    missing = deepcopy(verdict)
    missing["frame_checks"][0].pop("scene_continuity_closure")
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_audit_verdict(missing, plan, receipt)

    changed = deepcopy(plan)
    changed["scene_plans"][0]["continuity_graph"]["views"][1][
        "transition_from_previous"
    ] = "camera_motion"
    with pytest.raises(ValueError, match="audit inputs"):
        image_optimization.freeze_plan_audit_inputs(
            changed, frame_inventory=_generic_frame_inventory()
        )
    changed_inventory = _generic_frame_inventory()
    changed_inventory[1]["source_transition_from_previous"] = "camera_motion"
    changed_receipt = image_optimization.freeze_plan_audit_inputs(
        changed, frame_inventory=changed_inventory
    )
    assert changed_receipt["plan_sha256"] != receipt["plan_sha256"]
    assert changed_receipt["continuity_sha256"] != receipt["continuity_sha256"]


def _generic_verdict_v4(plan: dict, status: str = "pass") -> dict:
    verdict = _verdict_v3(plan, status=status)
    verdict["version"] = 4
    verdict["plan_sha256"] = image_optimization.plan_sha256(plan)
    for segment in verdict["segments"]:
        for frame in segment["frame_checks"]:
            frame["scene_continuity_view"] = _check(status)
    verdict["project_checks"]["scene_continuity"]["evidence"] = (
        "SCENE_01/COMPONENT_01 SCENE_01/COMPONENT_02 "
        "SCENE_01/COMPONENT_01 supports SCENE_01/COMPONENT_02"
    )
    return verdict


def test_v4_verify_uses_project_graph_evidence_and_per_frame_view_only():
    plan = _generic_scene_continuity_plan()
    passed = _generic_verdict_v4(plan)
    assert image_optimization.canonical_verification(passed, plan) == passed

    missing = deepcopy(passed)
    missing["segments"][0]["frame_checks"][0].pop("scene_continuity_view")
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(missing, plan)

    failed = deepcopy(passed)
    failed["segments"][0]["frame_checks"][0]["scene_continuity_view"] = _check(
        "fail"
    )
    failed["segments"][0]["passed"] = False
    failed["passed"] = False
    failed["reason"] = "scene_continuity_failed"
    assert image_optimization.canonical_verification(failed, plan)[
        "reason"
    ] == "scene_continuity_failed"

    incomplete = deepcopy(passed)
    incomplete["project_checks"]["scene_continuity"]["evidence"] = (
        "SCENE_01/COMPONENT_01"
    )
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(incomplete, plan)


def test_v4_verify_pack_existing_checks_cover_graph_without_new_fields():
    plan = _generic_scene_continuity_plan()
    verdict = _pack_verdict(plan)
    evidence = (
        "SCENE_01/COMPONENT_01 SCENE_01/COMPONENT_02 "
        "SCENE_01/COMPONENT_01 supports SCENE_01/COMPONENT_02"
    )
    verdict["scenes"][0]["checks"]["geometry"]["evidence"] = evidence
    assert image_optimization.canonical_reference_pack_verdict(
        verdict, plan
    ) == verdict

    verdict["scenes"][0]["checks"]["geometry"]["evidence"] = (
        "SCENE_01/COMPONENT_01"
    )
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_reference_pack_verdict(verdict, plan)


def test_v4_skill_defines_semantic_target_authority_without_audit_role():
    skill, example = _skill_contract()
    encoded = json.dumps(example, ensure_ascii=False)

    assert set(example) == {"people", "entities", "scenes", "frames"}
    assert set(next(iter(example["scenes"].values()))) == {
        "source_scene", "replacement_scene", "semantic_change",
        "geometry_change", "depth_change", "layout_change",
        "local_color_change",
    }
    for mechanical in (
        "scene_plans", "continuity_graph", "COMPONENT_01",
        "transition_skeleton", "ENTITY_ID",
    ):
        assert mechanical not in encoded
    assert "plan_audit" not in skill
    assert "verify_pack" not in skill


def test_v3_compiles_distinct_current_frame_prompts_and_execution_binding():
    plan = _plan_v3()
    prompts = image_optimization.compile_frame_prompts(
        plan, "independent_parallel"
    )
    assert set(prompts) == {0} and set(prompts[0]) == {1, 2}
    assert prompts[0][1] != prompts[0][2]
    assert "可见部位数量与边界保持当前源帧" in prompts[0][1]
    assert "第二帧可见部位数量与边界保持当前源帧" in prompts[0][2]
    assert "白平衡与色温保持当前源帧" in prompts[0][1]
    assert '"description":"当前帧可见操作实体"' in prompts[0][1]
    assert '"description":"第二帧可见操作实体"' in prompts[0][2]
    assert '"description":"第二帧可见操作实体"' not in prompts[0][1]
    assert '"area_weighted_warm_cool_family":"warm"' in prompts[0][1]
    assert '"area_weighted_warm_cool_family":"cool"' in prompts[0][2]
    assert '"area_weighted_warm_cool_family":"cool"' not in prompts[0][1]
    frozen = image_optimization.freeze_execution_inputs(
        plan,
        revision=1,
        profile={"id": "dual-target", "revision": 3},
        model="doubao-seedream-5-0-pro-260628",
        frame_inventory=[
            {
                "segment_index": 0,
                "frame_index": index,
                "frame_name": f"{index:02d}.png",
                "source_sha256": str(index) * 64,
            }
            for index in (1, 2)
        ],
    )
    assert frozen["version"] == 3
    assert frozen["frames"][0]["frame_constraint"]["frame_index"] == 1
    assert frozen["frames"][1]["frame_constraint"]["frame_index"] == 2


def _verdict_v3(plan: dict, *, status: str = "pass") -> dict:
    base = _verdict(plan)
    base["version"] = 3
    base["plan_sha256"] = image_optimization.plan_sha256(plan)
    for segment in base["segments"]:
        segment["frame_checks"] = [
            {
                "frame_index": constraint["frame_index"],
                "visible_body_parts": _check(status),
                "pose_skeleton": _check(status),
                "contact_points": _check(status),
                "occlusion_order": _check(status),
                "out_of_frame_crop": _check(status),
                "non_person_entity_ledger": _check(status),
                "dominant_palette_contract": _check(status),
                "photometric_contract": _check(status),
            }
            for constraint in plan["segments"][0]["frame_constraints"]
        ]
        segment["passed"] = status == "pass"
    base["passed"] = status == "pass"
    base["reason"] = None if status == "pass" else "verification_unknown"
    return base


def test_v3_verify_requires_every_frame_constraint_and_fails_closed():
    plan = _plan_v3()
    passed = _verdict_v3(plan)
    assert image_optimization.canonical_verification(passed, plan) == passed

    missing = _verdict_v3(plan)
    missing["segments"][0]["frame_checks"].pop()
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(missing, plan)

    unknown = _verdict_v3(plan, status="unknown")
    assert image_optimization.canonical_verification(unknown, plan)["passed"] is False


def test_v3_verify_requires_and_evaluates_entity_ledger_check():
    plan = _plan_v3()
    missing = _verdict_v3(plan)
    missing["segments"][0]["frame_checks"][0].pop("non_person_entity_ledger")
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(missing, plan)

    unknown = _verdict_v3(plan)
    unknown["segments"][0]["frame_checks"][0]["non_person_entity_ledger"] = _check(
        "unknown"
    )
    unknown["segments"][0]["passed"] = False
    unknown["passed"] = False
    unknown["reason"] = "verification_unknown"
    assert image_optimization.canonical_verification(unknown, plan)["passed"] is False

    failed = _verdict_v3(plan)
    failed["segments"][0]["frame_checks"][0]["non_person_entity_ledger"] = _check(
        "fail"
    )
    failed["segments"][0]["passed"] = False
    failed["passed"] = False
    failed["reason"] = "interaction_preservation_failed"
    canonical = image_optimization.canonical_verification(failed, plan)
    assert canonical["passed"] is False
    assert canonical["reason"] == "interaction_preservation_failed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["segments"][0]["frame_constraints"][0].pop(
            "dominant_palette_contract"
        ),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "dominant_palette_contract"
        ].update(extra="forbidden"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "dominant_palette_contract"
        ].update(area_weighted_warm_cool_family="unbounded"),
        lambda value: value["segments"][0]["frame_constraints"][0][
            "dominant_palette_contract"
        ].update(saturation_style=[]),
    ],
)
def test_v3_dominant_palette_contract_is_exact_and_fails_closed(mutate):
    value = _plan_v3()
    mutate(value)
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_plan_v3(
            value, segment_indices=[0], frame_counts={0: 2}
        )


def test_v3_verify_requires_and_evaluates_dominant_palette_check():
    plan = _plan_v3()
    missing = _verdict_v3(plan)
    missing["segments"][0]["frame_checks"][0].pop("dominant_palette_contract")
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.canonical_verification(missing, plan)

    unknown = _verdict_v3(plan)
    unknown["segments"][0]["frame_checks"][0]["dominant_palette_contract"] = _check(
        "unknown"
    )
    unknown["segments"][0]["passed"] = False
    unknown["passed"] = False
    unknown["reason"] = "verification_unknown"
    assert image_optimization.canonical_verification(unknown, plan)["passed"] is False

    failed = _verdict_v3(plan)
    failed["segments"][0]["frame_checks"][0]["dominant_palette_contract"] = _check(
        "fail"
    )
    failed["segments"][0]["passed"] = False
    failed["passed"] = False
    failed["reason"] = "dominant_palette_preservation_failed"
    canonical = image_optimization.canonical_verification(failed, plan)
    assert canonical["passed"] is False
    assert canonical["reason"] == "dominant_palette_preservation_failed"


def test_skill_leaves_pixel_palette_contract_to_backend(tmp_path):
    skill, example = _skill_contract()
    plan, diagnostics = _compiled_semantic(tmp_path)
    palette = plan["segments"][0]["frame_constraints"][0][
        "dominant_palette_contract"
    ]

    assert "局部固有色" in skill
    assert "枚举 palette" in skill
    assert "palette" not in json.dumps(example, ensure_ascii=False).lower()
    assert set(palette) == {
        "area_weighted_warm_cool_family", "saturation_style",
    }
    assert diagnostics["score"] == 1.0
