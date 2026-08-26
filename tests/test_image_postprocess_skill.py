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
    assert "人物及其他实体的数量、姿态、动作、视线" in skill
    assert "非目标元素的位置、比例和可见部分" in skill
    assert "禁止恢复或新增字幕、文字、Logo、水印、贴纸、界面元素、品牌标识或乱码" in skill

    assert "同一 `chain_id` 的 `continue` 段优先视为连续画面" in skill
    assert "`hard_cut` 不自动代表人物或场景变化" in skill

    assert "人物除脸外、服装、核心实体、交互实体、非目标前景" in skill
    assert "每段所有关键帧共享一份可直接提交给 Seedream 的提示词" in skill
    assert "第一句只聚焦唯一替换目标、冻结的自然语言身份短语及其原位条件" in skill
    assert "第二句写完整保护" in skill
    assert "保持简洁，但不得为缩短删除安全关系" in skill
    assert "禁止全景复述、画质美化或实体清单" in skill
    assert "保持叙事内核和关系不变，只改变表象" in skill
    assert "不超过 140 个 Unicode 字符" not in skill
    assert "生成前计数" not in skill

    assert "`independent_parallel` 只依赖当前源图和已冻结的自然语言身份短语" in skill
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
    assert "识别核心实体、交互实体及实体间互动" in skill
    assert "新脸长相略有不同" in skill
    assert "优先选择人物外观" not in skill

    assert "核心实体、交互实体、非目标前景" in skill
    assert "实体间的空间与物理关系、数量、结构、方向" in skill
    assert "全部不可编辑" in skill
    assert "不得同类改款" in skill
    assert "画质、核心实体、交互实体、非目标前景不得作为差异化目标" in skill

    assert "恰好两句" in skill
    assert "文字或 Logo" in skill
    assert "不传递人物、构图、动作、实体或关系" in skill

    assert "本 Skill 的 `global_elements` 只允许 `PERSON`、`SCENE`" in skill
    assert "按完整 `id` 的字典序" in skill
    assert "不得输出 `OUTFIT`、`PROP`、`PRODUCT`、`SUBJECT`" in skill


def test_skill_defines_observable_entities_and_high_risk_relations():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "核心实体：删除或修改后会改变动作目的、剧情含义或画面识别的可见实体" in skill
    assert "交互实体：与人物或核心实体存在持握、接触、装配、插入、连接、对齐或承托等可见关系的实体" in skill
    assert "非目标前景：遮挡目标表面，或建立景深、接触边界的可见实体" in skill
    assert "高风险关系：上述任一可见关系存在" in skill
    assert "编辑对象变化可能破坏数量、结构、作用方向、接触点、遮挡或前后顺序" in skill


def test_skill_freezes_balanced_component_strategy_and_exact_two_sentence_routes():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "全项目构建跨段相关组件" in skill
    assert "同一场景或同一稳定背景表面" in skill
    assert "每个组件只做一次策略选择并冻结" in skill
    assert "高风险交互关系" in skill
    assert "整个组件统一选择背景路线" in skill
    assert "人物在所有相关段都稳定清晰" in skill

    assert "默认只改变一种低频外观属性，优先色相" in skill
    assert "固定构件和边界结构" in skill
    assert "接触几何" in skill
    assert "人物、服装、核心实体、交互实体、非目标前景" in skill
    assert "实体间空间与物理关系" in skill
    assert "禁止增删文字或 Logo" in skill

    assert "只替换同一人物的新脸" in skill
    assert "头部位置、大小、朝向、裁切和遮挡" in skill
    assert "同色同风格的不同款服装" not in skill

    assert "其他图只提供目标设计" in skill

    assert "只收录实际跨段替换" in skill
    assert "完整 `id` 的字典序" in skill


def test_uses_one_large_stable_surface_and_face_only_fallback():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "只选一个在相关帧中稳定对应且可见" in skill
    assert "面积尽量大的单一安全表面" in skill
    assert "允许个别关键帧因裁切不可见" in skill
    assert "只编辑可见的源表面，目标表面不可见的图片不做任何改变" in skill
    assert "不得映射全部背景" in skill
    assert "为每类冻结" not in skill

    assert "完整保留表面局部物理坐标中的原纹理图案、纹理相位、缺陷、边界、几何、方向、世界尺度和接触几何" in skill
    assert "透视仅服从当前源图投影" in skill
    assert "接触几何" in skill
    assert "保持光线方向" in skill
    assert "固定构件和边界结构不得新增、删除或移动" in skill
    assert "该帧不做其他编辑" in skill

    assert "找不到稳定安全背景表面" in skill
    assert "只换脸，不换服装" in skill
    assert "可见的性别呈现、肤色和整体风格、年龄和气质" in skill
    assert "跨段 `PERSON replacement`" in skill
    assert "不得输出 `OUTFIT`" in skill


def test_scene_membership_requires_each_segment_own_clear_surface_evidence():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "“同一场景”只用于构建相关组件，不能证明可编辑表面跨段为同一表面" in skill
    assert "每个拟列入 `SCENE segments` 的段必须用该段自身关键帧独立举证" in skill
    assert "不得用其他段、场景名称、语义类别相同或相似材质补证" in skill

    assert "目标表面的身份、全部目标成员、连续可编辑区域、完整边界和原纹理拓扑清晰可辨" in skill
    assert "足以按原位置保留" in skill
    assert "仅见模糊、小面积、遮挡到无法判界或不同朝向局部且无法确认同一表面" in skill
    assert "不得把该段列入 `SCENE segments`" in skill

    assert "同一物理表面，或同一场景内设计连续且可安全统一映射的同一表面" in skill
    assert "仅同类表面不够" in skill
    assert "`hard_cut` 不自动否定同一表面，但每个段必须独立举证" in skill

    assert "单个段内目标偶尔因裁切不可见时，该帧可以不改动" in skill
    assert "该段整体资格必须由该段自身其他清晰关键帧证明" in skill
    assert "最终少于两个合格段时不得建立背景 `SCENE`" in skill
    assert "继续按人物仅换脸或不改动决策" in skill


def test_scene_component_excludes_unqualified_segments_with_noop_prompts():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "背景 `SCENE` 规则中的“相关段”仅指列入该 `SCENE segments` 的合格段" in skill
    assert "`SCENE segments` 只包含通过逐段证据门禁的合格段" in skill
    assert "同一组件已有至少两个合格段并建立 `SCENE`，但另有一个或多个不合格段时" in skill
    assert "每个不合格整段必须输出两句不改动提示词" in skill
    assert "不得包含该 `SCENE` 的项目级替换短语" in skill
    assert "不得改走人物路线" in skill


def test_surface_edit_domain_uses_visible_local_stop_boundaries():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "目标表面的定位不得只依赖语义名称" in skill
    assert "存在相邻同类表面或连续平面时" in skill
    assert "拟编辑段必须用该段自身可见且明确的边界拓扑或停止地标定义编辑域" in skill
    assert "停止地标另一侧保持原样" in skill
    assert "因编辑域无法稳定定位而判定不合格的段，必须输出两句不改动提示词" in skill
    assert "不得在该段回退人物路线" in skill
    assert "全组件人物或不改动决策，不得覆盖该段已冻结的不改动结论" in skill

    assert "项目级替换短语在合格段间逐字一致" in skill
    assert "每段 `prompt` 可写该段不同的局部停止边界" in skill
    assert "局部停止边界不属于项目级替换短语" in skill
    assert "不要求整段 `prompt` 或整句逐字相同" in skill


def test_background_surface_edit_is_temporally_stable_and_mask_bounded():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    background_rules = skill.split("### 背景路线", maxsplit=1)[1].split(
        "### 人物路线", maxsplit=1
    )[0]

    assert "默认只改变一种低频外观属性，优先色相" in background_rules
    assert "只有色相不能安全获得明显差异时，才单独选择明暗、光泽或粗糙度之一" in background_rules
    assert "只改色相时，保持原亮度、光照、阴影、反射、纹理图案、纹理相位、缺陷、光泽和粗糙度" in background_rules
    assert "不得附加均匀、哑光或任何新表面处理" in background_rules
    assert "完整保留表面局部物理坐标中的原纹理图案、纹理相位、缺陷、边界、几何、方向、世界尺度和接触几何" in background_rules
    assert "透视仅服从当前源图投影" in background_rules
    assert "方向、透视、世界尺度" not in background_rules
    assert "不得新增、删除或移动纹理图案、接缝、重复单元或高频细节" in background_rules
    assert "表面翻新" not in background_rules
    assert "生成新材质" not in background_rules

    assert "差异必须明显可见" in background_rules
    assert "无法安全获得明显差异时允许不改动" in background_rules

    assert "只编辑可见的源表面" in background_rules
    assert "只有资格门通过后" in background_rules
    assert "在已证明的完整目标域内完整覆盖" in background_rules
    assert "边界不确定的像素保持不变" not in background_rules
    assert "非目标前景继续遮挡目标表面" in background_rules
    assert "接触点、接触阴影、反射和边界结构不得改变" in background_rules
    assert "禁止编辑外溢到目标表面之外" in background_rules

    assert "`SCENE replacement` 不直接提交给 Seedream" not in skill
    assert "`SCENE replacement` 本身就是项目级替换短语" in skill
    assert "目标表面描述 + 实际改变的一种低频属性" in skill
    assert "`SCENE replacement` 和项目级替换短语只写实际改变的一个属性" in skill
    assert "不得列出未改变属性或多个候选" in skill
    assert "每个相关段实际 `prompt` 的第一句必须逐字包含同一短语" in background_rules
    assert "其余关系保护可按本段可见关系写" in background_rules
    assert "不得改动项目级替换短语" in background_rules


def test_replacement_identity_is_frozen_across_frames_and_reappearances():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    person_rules = skill.split("### 人物路线", maxsplit=1)[1]

    assert "对每个实际替换的元素，项目级只冻结一个“替换身份”" in skill
    assert "通用替换身份只规定唯一性" in skill
    assert "`PERSON` 只冻结同一新脸身份特征" in skill
    assert "非脸部继承当前源图" in skill
    assert "`SCENE` 才冻结" in skill
    assert "表面局部物理坐标" in skill
    assert "材料、表面响应、纹理族与世界尺度、原有特征相位、结构单元、接缝或重复拓扑" in skill
    assert "纹理族" not in person_rules

    assert "视角、裁切、透视、连续受光、运动模糊和真实遮挡仅是投影变量" in skill
    assert "不能重生替换身份" in skill

    assert "同一物理实例在遮挡后、跨段或 `hard_cut` 后再次出现" in skill
    assert "映射回同一替换身份" in skill
    assert "不得独立重新设计" in skill


def test_replacement_domain_is_complete_hard_bounded_and_fail_closed():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    prompt_rules = skill.split("## 每段 Seedream 提示词", maxsplit=1)[1]

    assert "编辑资格门先于提示词" in skill
    assert "任一目标可见帧的全部目标成员或完整边界不能稳定定位" in skill
    assert "该段或组件必须输出两句不改动提示词" in skill
    assert "只有所有目标可见帧通过资格门" in skill
    assert "目标完全不可见的帧不做任何改变" in skill
    assert "在已证明的完整目标域内完整覆盖全部可见目标成员" in skill
    assert "不得留下旧外观孤岛" in skill
    assert "目标域外像素硬保留" in skill
    assert "保持轮廓、层级、可见性和运动模糊" in skill
    assert "边界不确定的像素保持不变" not in prompt_rules
    assert "边界置信不足时保留输入像素" not in skill


def test_two_sentence_prompts_execute_frozen_identity_without_false_references():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    prompt_rules = skill.split("## 每段 Seedream 提示词", maxsplit=1)[1]

    assert "段级提示词会复用于该段每次以一张当前源图为编辑目标的请求" in prompt_rules
    assert "每次单图请求" not in prompt_rules
    assert "只写对每次当前待编辑源图都成立的通用投影守恒" in prompt_rules
    assert "不得逐帧描述条件" in prompt_rules
    assert "不得声称同时看过多张输入图" in prompt_rules
    assert "各输入帧自身" not in prompt_rules
    assert "该帧的投影条件" not in prompt_rules
    assert "`independent_parallel` 只依赖当前源图和已冻结的自然语言身份短语" in prompt_rules
    assert "`anchor_consistency` 必须限制其他图的角色" in prompt_rules
    assert "其他图只提供目标设计" in prompt_rules


def test_output_contract_freezes_short_and_long_identity_carriers():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "segment 0 提示词中的自然语言项目身份短语" in skill
    assert "所有关键帧唯一的身份载体" in skill
    assert "不得声称存在 `PERSON` 或 `SCENE` 映射" in skill
    assert "短视频背景或人物路线的第一句必须逐字写入该短语" in skill

    assert "长视频每个 `global_elements` 元素的 `replacement`" in skill
    assert "本身必须是可直接复用的自然语言冻结短语" in skill
    assert "`PERSON` 和 `SCENE` 都必须把该短语逐字写入" in skill
    assert "其 `segments` 所列每段提示词的第一句" in skill
    assert "实际替换元素的 `segments` 必须两两不相交" in skill
    assert "任一长视频 `segment_index` 最多属于一个 `global_elements` 元素" in skill
    assert "同段有多个候选时，按既有安全门和已冻结策略只保留一个" in skill
    assert "其他候选不得进入该段映射或提示词" in skill
    assert "其他候选整段不改动" not in skill


def test_human_behavior_documents_generic_replacement_stability_contract():
    behavior = Path(
        "docs/human/features/conversation-task/behaviors/postprocess.md"
    ).read_text(encoding="utf-8")

    assert "`PERSON` 只冻结同一新脸身份" in behavior
    assert "`SCENE` 才冻结" in behavior and "局部物理坐标" in behavior
    assert "通用投影守恒" in behavior
    assert "每次以一张当前源图为编辑目标的请求" in behavior
    assert "每次单图请求" not in behavior
    assert "遮挡后、跨段或 hard cut 后再次出现" in behavior
    assert "任一目标可见帧" in behavior and "完整边界" in behavior
    assert "通过资格门" in behavior and "目标域外像素硬保留" in behavior
    assert "短视频" in behavior and "唯一身份载体" in behavior
    assert "长视频" in behavior and "replacement" in behavior
    assert "segments" in behavior and "两两不相交" in behavior


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
