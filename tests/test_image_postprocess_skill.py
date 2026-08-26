import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import codex_runner, image_optimization
from app.codex_runner import CodexRunner
from conftest import make_settings


def _png(value: int = 127) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((8, 6, 3), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _elements() -> list[dict]:
    return [
        {
            "id": "PERSON_01",
            "kind": "person",
            "source": "反复出现的深发女性",
            "replacement": "椭圆脸、自然直眉的新人物",
            "segments": [1, 2],
        },
    ]


def _multi_output() -> dict:
    return {
        "version": 1,
        "segment_indices": [1, 2],
        "global_elements": _elements(),
        "segment_prompts": [
            {"segment_index": 1, "prompt": "第一段真实 Seedream 提示词"},
            {"segment_index": 2, "prompt": "第二段真实 Seedream 提示词"},
        ],
    }


class _ProjectRunner:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls = []

    def run_isolated(self, workdir, prompt, *, session_dir):
        workdir = Path(workdir)
        self.calls.append({
            "files": sorted(
                str(path.relative_to(workdir))
                for path in workdir.rglob("*") if path.is_file()
            ),
            "request": json.loads(
                (workdir / "work" / "request.json").read_text(encoding="utf-8")
            ),
            "prompt": prompt,
            "session_dir": Path(session_dir),
        })
        (workdir / "work" / "image_optimization.json").write_text(
            json.dumps(self.output, ensure_ascii=False), encoding="utf-8"
        )


def _segments(session: Path) -> list[dict]:
    first = session / "work" / "segments" / "1" / "work" / "keyframes"
    second = session / "work" / "segments" / "2" / "work" / "keyframes"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "01.png").write_bytes(_png(10))
    (second / "01.png").write_bytes(_png(20))
    return [
        {
            "index": 1,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
            "keyframes_dir": first,
        },
        {
            "index": 2,
            "chain_id": "chain-001",
            "join_mode": "continue",
            "keyframes_dir": second,
        },
    ]


def test_skill_is_single_project_level_prompt_compiler():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    assert "name: image-postprocess" in skill
    assert "work/image_optimization.json" in skill
    assert "global_elements" in skill and "segment_prompts" in skill
    assert "真实提交给 Seedream" in skill
    assert not Path("skills/image-continuity/SKILL.md").exists()


def test_skill_keeps_hard_to_abuse_input_output_and_prompt_boundaries():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "本 Skill 只生成提示词，不编辑图片、不调用 Seedream" in skill
    assert "不代表图片二次编辑一定执行、成功或被视频生成采用" in skill
    assert "不得读取视频、视频生成提示词、台词、音频、项目目录、其他路径或未列出的文件" in skill

    assert "短视频的 `segment_indices` 是 `[0]`" in skill
    assert "`global_elements` 必须是空数组" in skill
    assert "除规定字段外不得增加字段" in skill
    assert "不得用 Markdown 代码围栏包裹 JSON" in skill

    assert "画幅、裁切、机位、镜头、透视、景别、构图、光线、焦点、景深" in skill
    assert "人物与动物数量、姿态、动作、视线" in skill
    assert "非目标元素的位置、比例和可见部分" in skill
    assert "禁止恢复或新增字幕、文字、Logo、水印、贴纸、界面元素、品牌标识或乱码" in skill

    assert "同一 `chain_id` 的 `continue` 段优先视为连续画面" in skill
    assert "`hard_cut` 不自动代表人物或场景变化" in skill

    assert "人物除脸外、服装、手、玩具、商品" in skill
    assert "每段所有关键帧共享一份可直接提交给 Seedream 的提示词" in skill
    assert "第一句只聚焦唯一替换目标及其原位条件" in skill
    assert "第二句写完整保护" in skill
    assert "保持简洁，但不得为缩短删除安全关系" in skill
    assert "禁止全景复述、画质美化或物体清单" in skill
    assert "保持叙事内核和关系不变，只改变表象" in skill
    assert "不超过 140 个 Unicode 字符" not in skill
    assert "生成前计数" not in skill

    assert "`independent_parallel` 无需写“当前图是唯一目标”" in skill
    assert "不得写参考人物、参考角色或其他图" in skill
    assert "`anchor_consistency` 必须限制其他图的角色" in skill
    assert "其他图只提供目标设计" in skill

    assert "任何组件只要背景路线和人物路线都不能证明安全" in skill
    assert "输出两句不改动提示词，不建立 `global_elements`" in skill
    assert "无人、人物不清晰且无稳定安全背景表面" in skill
    assert "高风险组件找不到合格表面时放弃去重" in skill
    assert "不得退到人物路线" in skill


def test_segment_prompt_rules_forbid_internal_mapping_labels():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    prompt_rules = skill.split("## 每段 Seedream 提示词", maxsplit=1)[1]

    assert '"id": "PERSON_01"' in skill
    assert "必须直接写自然语言目标与替换设计" in prompt_rules
    assert "绝不能包含内部 ID 或字段标签" in prompt_rules
    for label in (
        "PERSON_01", "SCENE_01", "source", "replacement", "global_elements"
    ):
        assert f"`{label}`" in prompt_rules


def test_skill_prompts_are_short_single_target_edits_with_locked_relations():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "第一优先级是最终视频连续性和现实合理性" in skill
    assert "扭曲、元素突变、物理或接触关系异常" in skill
    assert "一票否决" in skill
    assert "人物或背景替换仅是用于去重的第二优先级" in skill
    assert "只有通过第一层才评价" in skill

    assert "先检查全项目全部关键帧" in skill
    assert "去重目标只能是 `人物外观` 或 `背景`" in skill
    assert "功能道具、商品、玩具或人与物互动" in skill
    assert "新脸长相略有不同" in skill
    assert "优先选择人物外观" not in skill

    assert "核心商品、功能道具、玩具、手部" in skill
    assert "人与物/物与物的握持、接触、插入、对齐、连接、遮挡、前后顺序" in skill
    assert "数量、结构、方向" in skill
    assert "全部不可编辑" in skill
    assert "不得同类改款" in skill
    assert "画质、功能道具、玩具、商品、宠物不得作为差异化目标" in skill

    assert "恰好两句" in skill
    assert "文字或 Logo" in skill
    assert "不传递人物、构图、动作、物体或关系" in skill

    assert "本 Skill 的 `global_elements` 只允许 `PERSON`、`SCENE`" in skill
    assert "按完整 `id` 的字典序" in skill
    assert "不得输出 `OUTFIT`、`PROP`、`PRODUCT`、`SUBJECT`" in skill
    assert "陀螺" not in skill and "发射器" not in skill


def test_skill_freezes_balanced_component_strategy_and_exact_two_sentence_routes():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "全项目构建跨段相关组件" in skill
    assert "同一场景或同一稳定背景/固定建筑表面" in skill
    assert "每个组件只做一次策略选择并冻结" in skill
    assert "手—功能道具/玩具的高风险关系" in skill
    assert "整个组件统一选择背景路线" in skill
    assert "人物在所有相关段都稳定清晰" in skill

    assert "同功能、同表面类别但明显不同的材质或色调" in skill
    assert "墙地交界" in skill
    assert "接触几何" in skill
    assert "人物、服装、手、玩具、商品" in skill
    assert "握持、插接、对齐、遮挡关系" in skill
    assert "禁止增删文字或 Logo" in skill

    assert "只替换同一人物的新脸" in skill
    assert "头部位置、大小、朝向、裁切和遮挡" in skill
    assert "同色同风格的不同款服装" not in skill

    assert "其他图只提供目标设计" in skill

    assert "只收录实际跨段替换" in skill
    assert "完整 `id` 的字典序" in skill


def test_final_j_uses_one_large_stable_surface_and_face_only_fallback():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "只选一个能在所有相关段稳定对应" in skill
    assert "面积尽量大的安全背景表面类别" in skill
    assert "优先地面，其次墙面" in skill
    assert "允许个别关键帧因裁切不可见" in skill
    assert "只编辑每张图中可见的目标表面；目标表面不可见的图片不做任何改变" in skill
    assert "不得映射全部背景" in skill
    assert "为每类冻结" not in skill

    assert "原表面的边界、分块/拼缝、方向、透视和接触几何" in skill
    assert "保持光线方向" in skill
    assert "固定构件、家具、陈设不得新增、删除或移动" in skill
    assert "该帧不做其他编辑" in skill

    assert "找不到稳定安全背景表面" in skill
    assert "只换脸，不换服装" in skill
    assert "可见的性别呈现、肤色和整体风格、年龄和气质" in skill
    assert "跨段 `PERSON replacement`" in skill
    assert "不得输出 `OUTFIT`" in skill


def test_one_project_call_returns_global_map_and_all_real_prompts(tmp_path):
    session = tmp_path / "session"
    segments = _segments(session)
    (session / "work" / "prompt.txt").write_text("H3 SECRET", encoding="utf-8")
    runner = _ProjectRunner(_multi_output())

    continuity, prompts = image_optimization.generate_project_prompts(
        runner,
        segments,
        "anchor_consistency",
        session_dir=session,
    )

    assert len(runner.calls) == 1
    assert continuity == {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": _elements(),
    }
    assert prompts == {
        1: "第一段真实 Seedream 提示词",
        2: "第二段真实 Seedream 提示词",
    }
    assert runner.calls[0]["request"] == {
        "edit_mode": "anchor_consistency",
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
    assert "H3 SECRET" not in runner.calls[0]["prompt"]
    assert runner.calls[0]["session_dir"] == session.resolve()


def test_short_project_uses_same_skill_and_has_no_global_map(tmp_path):
    session = tmp_path / "session"
    keyframes = session / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    runner = _ProjectRunner({
        "version": 1,
        "segment_indices": [0],
        "global_elements": [],
        "segment_prompts": [
            {"segment_index": 0, "prompt": "短视频真实 Seedream 提示词"}
        ],
    })

    continuity, prompts = image_optimization.generate_project_prompts(
        runner,
        [{
            "index": 0,
            "chain_id": "short-000",
            "join_mode": "hard_cut",
            "keyframes_dir": keyframes,
        }],
        "independent_parallel",
        session_dir=session,
    )

    assert continuity is None
    assert prompts == {0: "短视频真实 Seedream 提示词"}
    assert runner.calls[0]["files"] == [
        "SKILL.md",
        "work/request.json",
        "work/segments/0/keyframes/01.png",
    ]


@pytest.mark.parametrize(
    "output",
    [
        {},
        {**_multi_output(), "extra": True},
        {**_multi_output(), "segment_indices": [2, 1]},
        {**_multi_output(), "segment_prompts": [{"segment_index": 1, "prompt": "x"}]},
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 2, "prompt": "x"},
                {"segment_index": 1, "prompt": "y"},
            ],
        },
        {
            **_multi_output(),
            "global_elements": [{
                "id": "SUBJECT_01",
                "kind": "person",
                "source": "猫",
                "replacement": "另一只猫",
                "segments": [1, 2],
            }],
        },
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 1, "prompt": " "},
                {"segment_index": 2, "prompt": "x"},
            ],
        },
        {
            **_multi_output(),
            "segment_prompts": [
                {"segment_index": 1, "prompt": "x" * (32 * 1024 + 1)},
                {"segment_index": 2, "prompt": "x"},
            ],
        },
    ],
)
def test_project_output_rejects_noncanonical_shapes(tmp_path, output):
    session = tmp_path / "session"
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_project_prompts(
            _ProjectRunner(output),
            _segments(session),
            "anchor_consistency",
            session_dir=session,
        )


def test_short_project_rejects_global_elements(tmp_path):
    keyframes = tmp_path / "work" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01.png").write_bytes(_png())
    output = {
        "version": 1,
        "segment_indices": [0],
        "global_elements": _elements(),
        "segment_prompts": [{"segment_index": 0, "prompt": "x"}],
    }
    with pytest.raises(image_optimization.ImageOptimizationOutputError):
        image_optimization.generate_project_prompts(
            _ProjectRunner(output),
            [{
                "index": 0,
                "chain_id": "short-000",
                "join_mode": "hard_cut",
                "keyframes_dir": keyframes,
            }],
            "anchor_consistency",
            session_dir=tmp_path,
        )


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_project_generation_rejects_unknown_mode(tmp_path, mode):
    with pytest.raises(ValueError, match="edit mode"):
        image_optimization.generate_project_prompts(
            _ProjectRunner({}), [], mode, session_dir=tmp_path
        )


def test_continuity_receipt_is_private_deterministic_and_exact():
    plan = {
        "version": 1,
        "segment_indices": [1, 2],
        "elements": _elements(),
    }
    frozen = image_optimization.freeze_continuity(plan)
    receipt = frozen["_image_continuity"]
    assert set(receipt) == {"version", "segment_indices", "elements", "sha256"}
    assert image_optimization.continuity_receipt(frozen) == receipt
    tampered = json.loads(json.dumps(frozen))
    tampered["_image_continuity"]["elements"][0]["replacement"] = "changed"
    assert image_optimization.continuity_receipt(tampered) is None


def test_generic_isolated_runner_wraps_stage_and_discards_last_message(tmp_path, monkeypatch):
    runner = CodexRunner(timeout_s=1, concurrency=1)
    captured = []
    monkeypatch.setattr(codex_runner, "_resolve_bwrap", lambda: Path("/bin/true"))

    def inspect(workdir, prompt):
        captured.append(runner.build_argv(workdir, prompt))

    monkeypatch.setattr(runner, "run", inspect)
    with tempfile.TemporaryDirectory(prefix="duet-isolated-test-", dir="/tmp") as raw:
        stage = Path(raw).resolve()
        runner.run_isolated(stage, "execute skill", session_dir=tmp_path)

    argv = captured[0]
    assert argv[0] == "/bin/true"
    assert "--tmpfs" in argv and str(codex_runner._CHECKOUT_BOUNDARY) in argv
    assert "/dev/null" in argv
    assert not any(
        str(tmp_path) in part and part.endswith("codex_last_message.txt")
        for part in argv
    )


def test_freeze_uses_only_codex_outputs_and_has_no_template(tmp_path):
    settings = make_settings(tmp_path, seedream_edit_mode="independent_parallel")
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [
            {"index": 1, "prompt": "H3 ONE"},
            {"index": 2, "prompt": "H3 TWO"},
        ],
    }
    frozen = image_optimization.freeze_prompts(
        settings, meta, {1: "Codex image one", 2: "Codex image two"}
    )["_image_optimization"]

    assert frozen["version"] == 2
    assert set(frozen) == {"version", "model", "edit_mode", "segments"}
    assert [item["current"] for item in frozen["segments"]] == [
        "Codex image one", "Codex image two",
    ]
    assert "H3 ONE" not in json.dumps(frozen)
    assert image_optimization.receipt(
        {**meta, "status": "done", "_image_optimization": frozen}
    ) == frozen


def test_freeze_requires_exact_prompt_for_every_segment(tmp_path):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [{"index": 1}, {"index": 2}],
    }
    with pytest.raises(ValueError, match="prompt segments"):
        image_optimization.freeze_prompts(settings, meta, {1: "only one"})


def test_freeze_rejects_boolean_segment_key(tmp_path):
    settings = make_settings(tmp_path)
    meta = {
        "schema_version": 2,
        "status": "processing",
        "segments": [{"index": 1}],
    }
    with pytest.raises(ValueError, match="prompt segments"):
        image_optimization.freeze_prompts(
            settings, meta, {True: "not segment one"}
        )
