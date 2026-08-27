import hashlib
import json
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
                "local_color_change": "服装主色改为墨绿色",
                "reference": {"segment_index": first, "frame_index": 1},
                "observable_segments": indices,
            }
        ],
        "scene_plans": [
            {
                "id": "SCENE_01",
                "source_scene": "原室内工作空间",
                "replacement_scene": "同用途但不同空间设计的新室内工作空间",
                "semantic_change": "从原房间替换为另一处真实房间",
                "geometry_changes": ["改变墙面转折与开口形状"],
                "depth_changes": ["改变背景纵深与前后层级"],
                "layout_changes": ["改变柜体与通道的空间布局"],
                "local_color_change": "主要墙面改为冷灰蓝色",
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
                    "target_region": "人物以外的完整可见室内背景",
                    "boundary": "背景边界止于人物和前景物体轮廓",
                    "layout_reference_frame_index": 1,
                },
                "protected_non_target_people": [],
                "protected_relations": ["人物手部与所持物体的接触关系"],
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


def _segments(session: Path, indices: list[int] | None = None) -> list[dict]:
    indices = indices or [1, 2]
    result = []
    for index in indices:
        path = session / "work" / "segments" / str(index) / "work" / "keyframes"
        path.mkdir(parents=True)
        (path / "01.png").write_bytes(_png(index))
        result.append(
            {
                "index": index,
                "chain_id": "chain-001",
                "join_mode": "hard_cut" if index == indices[0] else "continue",
                "keyframes_dir": path,
            }
        )
    return result


class _Runner:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict] = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        root = Path(workdir)
        self.calls.append(
            {
                "files": sorted(
                    str(path.relative_to(root))
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                "request": json.loads(
                    (root / "work" / "request.json").read_text(encoding="utf-8")
                ),
                "prompt": prompt,
                "session_dir": Path(session_dir),
            }
        )
        name = (
            "image_optimization.json"
            if self.calls[-1]["request"]["phase"] == "plan"
            else "image_verification.json"
        )
        (root / "work" / name).write_text(
            json.dumps(self.output, ensure_ascii=False), encoding="utf-8"
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


def test_skill_is_one_concise_plan_and_verify_skill():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "name: image-postprocess" in skill
    assert "`phase` 只能是 `plan` 或 `verify`" in skill
    assert "人物与真实新场景必须同时替换" in skill
    assert "`person_plans`" in skill and "`scene_plans`" in skill
    assert "只生成结构化设计，不直接编写 Seedream 提示词" in skill
    assert "确定性编译器" in skill
    assert "自由文本" in skill and "不得删除或覆盖硬约束" in skill
    assert "不得只改色相、材质或全局调色" in skill
    assert "光源方向、曝光、白平衡/CCT、tone curve" in skill
    assert "画幅、裁切、机位、镜头、透视、构图" in skill
    for invariant in ("接触", "持握", "遮挡", "数量", "动作目的", "叙事关系"):
        assert invariant in skill
    assert "不得出现素材特调" in skill
    assert "work/image_verification.json" in skill


def test_skill_states_the_adversarial_dual_target_contract_without_ambiguity():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "## plan 先决条件" in skill
    assert "## verify 清单" in skill
    assert "源身份与源场景只用于识别源内容，不得成为 target pack" in skill
    assert "短视频 `[0]` 也执行人物与场景双替换" in skill
    assert "每段 `persons` 按 ID 完整枚举全部主人物" in skill
    assert "不可见时才标 `not_observable`，且不得新增该人物" in skill
    assert "语义、几何、纵深、布局" in skill
    assert "局部固有色变化不等于全局调色" in skill
    assert "新几何只允许产生物理正确的局部阴影和反射" in skill
    assert "任何 `fail` 或 `unknown` 都令 `passed=false`" in skill
    assert "成功 no-op" in skill


def test_plan_phase_returns_v2_plan_and_compiled_dual_target_prompts(tmp_path):
    session = tmp_path / "session"
    runner = _Runner(_plan())

    plan, prompts = image_optimization.generate_project_prompts(
        runner,
        _segments(session),
        "independent_parallel",
        session_dir=session,
    )

    assert plan == _plan()
    assert set(prompts) == {1, 2}
    for prompt in prompts.values():
        assert "替换人物" in prompt
        assert "替换场景" in prompt
        assert "不同空间结构" in prompt
        assert "光源方向、曝光、白平衡、色温、全局色调曲线保持" in prompt
        assert "接触关系" in prompt
        assert "图1始终是唯一编辑画布" in prompt
        assert "只提供冻结人物身份、场景设计和本段布局" in prompt
    assert runner.calls[0]["request"] == {
        "phase": "plan",
        "edit_mode": "independent_parallel",
        "segments": [
            {"index": 1, "chain_id": "chain-001", "join_mode": "hard_cut"},
            {"index": 2, "chain_id": "chain-001", "join_mode": "continue"},
        ],
    }
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/request.json",
        "work/segments/1/keyframes/01.png",
        "work/segments/2/keyframes/01.png",
    ]


def test_short_video_can_express_both_targets_without_special_noop_contract(tmp_path):
    session = tmp_path / "session"
    plan = _plan([0])
    generated, prompts = image_optimization.generate_project_prompts(
        _Runner(plan),
        _segments(session, [0]),
        "anchor_consistency",
        session_dir=session,
    )

    assert generated["person_plans"] and generated["scene_plans"]
    assert generated["segments"][0]["persons"][0]["state"] == "replace"
    assert "替换人物" in prompts[0] and "替换场景" in prompts[0]
    assert "图1始终是唯一编辑画布" in prompts[0]


def test_ineligible_plan_is_a_stable_failure_not_successful_noop(tmp_path):
    session = tmp_path / "session"
    with pytest.raises(
        image_optimization.ImageOptimizationIneligibleError,
        match="no_observable_narrative_person",
    ):
        image_optimization.generate_project_prompts(
            _Runner(_ineligible()),
            _segments(session, [0]),
            "independent_parallel",
            session_dir=session,
        )


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
        "local_color_change": "服装主色改为暖棕色",
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
