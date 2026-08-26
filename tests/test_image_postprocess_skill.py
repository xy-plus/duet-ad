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
    assert "第一句只聚焦唯一替换目标及其原位条件" in skill
    assert "第二句写完整保护" in skill
    assert "保持简洁，但不得为缩短删除安全关系" in skill
    assert "禁止全景复述、画质美化或实体清单" in skill
    assert "保持叙事内核和关系不变，只改变表象" in skill
    assert "不超过 140 个 Unicode 字符" not in skill
    assert "生成前计数" not in skill

    assert "`independent_parallel` 无需写“当前图是唯一目标”" in skill
    assert "不得写参考人物、参考角色或其他图" in skill
    assert "`anchor_consistency` 必须限制其他图的角色" in skill
    assert "其他图只提供目标设计" in skill

    assert "任何组件只要背景路线和人物路线都不能证明安全" in skill
    assert "输出两句不改动提示词，不建立 `global_elements`" in skill
    assert "无人、人物不清晰且无稳定安全背景编辑域" in skill
    assert "高风险组件找不到合格背景编辑域时放弃去重" in skill
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
    assert "非目标前景：遮挡目标编辑域，或建立景深、接触边界的可见实体" in skill
    assert "高风险关系：上述任一可见关系存在" in skill
    assert "编辑对象变化可能破坏数量、结构、作用方向、接触点、遮挡或前后顺序" in skill


def test_interaction_support_regions_are_high_risk_and_never_background_targets():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "交互支持区域：任何物理表面若直接承托核心实体" in skill
    assert "接收核心实体或交互实体的落点" in skill
    assert "承载可见动作轨迹" in skill
    assert "被流体或颗粒接触、经过或扩散" in skill
    assert "承载可见功能过程" in skill
    assert "交互支持区域计入高风险关系" in skill
    assert "不得作为背景候选或差异化目标" in skill


def test_skill_freezes_balanced_component_strategy_and_exact_two_sentence_routes():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "全项目构建跨段相关组件" in skill
    assert "同一场景或同一稳定背景编辑域" in skill
    assert "每个组件只做一次策略选择并冻结" in skill
    assert "高风险交互关系" in skill
    assert "整个组件统一选择背景路线" in skill
    assert "同一人物在所有相关段稳定清晰且换脸安全" in skill

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


def test_uses_one_traceable_background_domain_and_face_only_fallback():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "只选一个可追踪的物理背景元素或稳定且完整闭合的环境背景域作为编辑域" in skill
    assert "物理元素编辑域可以包含同一元素的多个可见部分或不同朝向面" in skill
    assert "允许个别关键帧因裁切不可见" in skill
    assert "每张图完整覆盖全部可见成员，不得局部应用" in skill
    assert "不得把未被闭合边界纳入的其他背景并入编辑域" in skill
    assert "为每类冻结" not in skill

    assert "完整保留原纹理图案、纹理相位、缺陷、边界与几何" in skill
    assert "接触几何" in skill
    assert "保持光线方向" in skill
    assert "固定构件和边界结构不得新增、删除或移动" in skill
    assert "该帧不做其他编辑" in skill

    assert "找不到稳定安全的背景编辑域" in skill
    assert "只换脸，不换服装" in skill
    assert "可见的性别呈现、肤色和整体风格、年龄和气质" in skill
    assert "跨段 `PERSON replacement`" in skill
    assert "不得输出 `OUTFIT`" in skill


def test_closed_environment_background_domain_is_a_safe_background_candidate():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "可追踪的物理背景元素或稳定且完整闭合的环境背景域" in skill
    assert "环境背景域可以延伸到画外" in skill
    assert "每个目标可见帧内完整可见域的可见边界只能由画幅边与不可编辑前景或遮挡物的原外轮廓形成" in skill
    assert "身份与拓扑跨帧稳定" in skill
    assert "不得只凭颜色、空白或画面位置识别" in skill
    assert "边界和遮挡必须由每帧独立举证，不得跨帧补证" in skill

    assert "不得直接承托人物、核心实体或交互实体" in skill
    assert "不得与其存在可见物理接触" in skill
    assert "不得接收其落点" in skill
    assert "不得承载动作轨迹、流体或颗粒的接触、经过或扩散以及功能过程" in skill
    assert "任一部分属于交互支持区域，整个环境背景域即不合格" in skill

    assert "环境背景域只允许上述 hue-only 映射" in skill
    assert "逐帧继承原图的局部对比、亮度分布、渐变方向、渐变幅度和原有低频结构" in skill
    assert "禁止补全、拉直、平滑、扩张、收缩、改形或产生边缘光晕" in skill


def test_background_candidate_type_is_frozen_before_qualification():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "每个背景候选必须在证据评估前且仅一次分类为有限物理元素或闭合环境背景域" in skill
    assert "分类结果在该组件内冻结" in skill
    assert "具有独立物理身份与自身外轮廓的候选属于有限物理元素" in skill
    assert "有限物理元素永远只能通过物理元素门禁" in skill
    assert "不得借用环境背景域规则重新分类、缩小、扩大或救活" in skill
    assert "任一资格门失败即终止该候选" in skill
    assert "不得更换标签或调整成员边界后重试" in skill
    assert "闭合环境背景域只适用于身份本身就是画幅内闭合视觉场的区域" in skill
    assert "不得由有限物理元素或其周边区域改名构造" in skill


def test_background_candidate_type_requires_positive_visual_evidence():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "两种类型都必须由候选自身的可见正向证据证明" in skill
    assert "环境背景域不是物理元素门禁失败后的默认兜底" in skill
    assert "可触碰或承载的材质表面、结构平面、附着表面" in skill
    assert "属于物质表面的自身纹理拓扑" in skill
    assert "必须分类为有限物理元素" in skill
    assert "即使填满背景或视觉上由画幅边与遮挡闭合" in skill

    assert "无有限物质表面且无自身物理外轮廓的稳定环境视觉场" in skill
    assert "可见边界只能由画幅边与不可编辑前景或遮挡物的原外轮廓形成" in skill
    assert "表面折转、材质分界、颜色分界、照明分界或构造内部边界" in skill
    assert "不得用于构造闭合环境背景域" in skill
    assert "分类不确定时该候选失败，不得回退为环境背景域" in skill


def test_scene_membership_requires_each_segment_own_clear_surface_evidence():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "“同一场景”只用于构建相关组件，不能证明背景编辑域跨段为同一目标" in skill
    assert "每个拟列入 `SCENE segments` 的段必须用该段自身关键帧独立举证" in skill
    assert "不得用其他段、场景名称、语义类别相同或相似材质补证" in skill

    assert "目标表面的身份、连续可编辑区域、边界和原纹理拓扑清晰可辨" in skill
    assert "足以按原位置保留" in skill
    assert "仅见模糊、小面积、遮挡到无法判界或不同朝向局部且无法确认同一表面" in skill
    assert "不得把该段列入 `SCENE segments`" in skill

    assert "同一 `SCENE` 要求各合格段的可见证据证明它们属于同一物理背景元素" in skill
    assert "仅同类或同一局部平面不够" in skill
    assert "`hard_cut` 不自动否定同一物理元素或同一环境背景域，但每个段必须独立举证" in skill

    assert "单个段内目标偶尔因裁切完全不可见时，该帧可以不改动" in skill
    assert "该段整体资格必须由该段自身所有目标可见帧分别通过上述逐帧证据门禁" in skill
    assert "最终少于两个合格段时不得建立背景 `SCENE`" in skill
    assert "继续按人物仅换脸或不改动决策" in skill


def test_each_target_visible_frame_proves_its_complete_outer_contour():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "每个目标可见帧都必须独立提供证据" in skill
    assert "目标每个可见部分在图内的完整原外轮廓、外轮廓转折和遮挡边缘清晰可定位" in skill
    assert "不得用段内其他帧补证" in skill
    assert "任一目标可见帧不满足，该段的背景候选即不合格" in skill

    assert "严格沿本图中冻结编辑域每个可见部分的原外轮廓、外轮廓转折和遮挡边缘停止" in skill
    assert "保持轮廓几何" in skill
    assert "禁止补全、拉直、平滑、扩张、收缩、改形或产生边缘光晕" in skill
    assert "内部几何折转和同一物理元素内部边界仍不作为停止线，不得与外轮廓混淆" in skill


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
    assert "局部停止边界必须是冻结编辑域通向不同物理元素的外部边界" in skill
    assert "内部几何折转或同一物理元素内部边界不得作为停止地标" in skill
    assert "组件已经建立背景 `SCENE` 或因高风险交互冻结为背景路线后" in skill
    assert "个别因编辑域无法稳定定位而不合格的段必须输出两句不改动提示词" in skill
    assert "不得逐段改走人物路线" in skill

    assert "项目级替换短语在合格段间逐字一致" in skill
    assert "每段 `prompt` 可写该段不同的局部可见外部边界" in skill
    assert "局部停止边界不属于项目级替换短语" in skill
    assert "不要求整段 `prompt` 或整句逐字相同" in skill


def test_surface_domain_membership_is_frozen_across_target_visible_frames():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "冻结目标是一个可追踪的物理背景元素或稳定且完整闭合的环境背景域" in skill
    assert "在项目/组件级冻结编辑域成员关系" in skill
    assert "同一成员在所有目标可见帧中必须统一纳入或排除" in skill
    assert "不同视角下的局部可见边界只用于定位同一冻结编辑域，不能改变成员集合" in skill
    assert "物理元素编辑域可以包含同一元素的多个可见部分或不同朝向面" in skill

    assert "多平面或相邻连续部分只有在逐帧无歧义追踪同一成员集合时才合格" in skill
    assert "对于已因高风险交互冻结为背景路线的组件，成员关系不确定时必须在规划阶段令整个相关组件输出两句不改动提示词" in skill
    assert "不得逐帧漏改，也不得回退人物路线" in skill
    assert "成员关系或成员边界不确定时必须在规划阶段不建立该域" in skill
    assert "合格提示词必须完整覆盖已冻结成员" in skill

    assert "第一句要求完整覆盖已冻结编辑域，不得只修改其中局部" in skill
    assert "每张图完整覆盖全部可见成员，不得局部应用" in skill
    assert "冻结编辑域边界外全部保持不变" in skill
    assert "非目标前景及实体间空间与物理关系全不变" in skill


def test_domain_membership_gate_preserves_the_safe_face_only_route():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")

    assert "背景编辑域不合格不自动等于整个组件不改动" in skill
    assert "背景候选尚未建立或整体失败" in skill
    assert "组件无高风险交互关系、同一人物在所有相关段稳定清晰且换脸安全时" in skill
    assert "统一走人物路线且仅换脸" in skill
    assert "背景域收紧不得把该条件分支改为普遍不改动" in skill

    assert "只换脸，不换服装、背景或身体其他部分" in skill
    assert "人物除脸外、服装、核心实体、交互实体、非目标前景及实体间空间与物理关系全不变" in skill


def test_background_surface_edit_is_temporally_stable_and_mask_bounded():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    background_rules = skill.split("### 背景路线", maxsplit=1)[1].split(
        "### 人物路线", maxsplit=1
    )[0]

    assert "默认只改变一种低频外观属性，优先色相" in skill
    assert "只有色相不适合作为安全差异时，才单独选择明暗、光泽或粗糙度之一" in skill
    assert "选择色相时遵守下述 hue-only 契约" in skill
    assert "不得使用明亮、浅、深、鲜艳、柔和、均匀或哑光修饰目标色相" in background_rules
    assert "原光照、阴影、反射、纹理图案、纹理相位、缺陷、光泽和粗糙度" in background_rules
    assert "完整保留原边界与几何" in background_rules
    assert "不得新增、删除或移动纹理图案、接缝、重复单元或高频细节" in background_rules
    assert "表面翻新" not in background_rules
    assert "生成新材质" not in background_rules

    assert "无法安全取得合规差异时允许不改动" in background_rules

    assert "只编辑本图可见的冻结编辑域" in background_rules
    assert "成员关系或成员边界不确定时必须在规划阶段不建立该域" in background_rules
    assert "完整覆盖已冻结成员" in background_rules
    assert "非目标前景继续遮挡目标编辑域" in background_rules
    assert "接触点、接触阴影、反射和边界结构不得改变" in background_rules
    assert "禁止编辑外溢到目标编辑域之外" in background_rules

    assert "`SCENE replacement` 不直接提交给 Seedream" in skill
    assert "目标背景描述 + 实际改变的一种低频属性" in skill
    assert "`SCENE replacement` 和项目级替换短语只写实际改变的一个属性" in skill
    assert "不得列出未改变属性或多个候选" in skill
    assert "每个相关段逐字复用，不得缩写、改写或使用同义词" in background_rules
    assert "其余关系保护可按本段可见关系写" in background_rules
    assert "不得改动项目级替换短语" in background_rules


def test_background_prompt_prevents_global_color_and_lighting_drift():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    background_rules = skill.split("### 背景路线", maxsplit=1)[1].split(
        "### 人物路线", maxsplit=1
    )[0]

    assert "第二句要求仅目标域发生所选属性变化" in background_rules
    assert "禁止全局调色、白平衡、曝光、对比度或重新布光" in background_rules
    assert "目标域外保持原颜色与明暗" in background_rules


def test_hue_mapping_is_channel_only_and_leads_each_real_prompt():
    skill = Path("skills/image-postprocess/SKILL.md").read_text(encoding="utf-8")
    background_rules = skill.split("### 背景路线", maxsplit=1)[1].split(
        "### 人物路线", maxsplit=1
    )[0]

    assert "选择色相时，项目级替换短语和每段第一句必须明确写“只旋转色相”" in skill
    assert "第一句最前面先写 hue-only 及其通道不变量，再写编辑域边界" in background_rules
    assert "同一输入像素或同一物理区域的原明度或感知亮度与原饱和度保持不变" in background_rules
    assert "不得抬亮暗部或压暗高光" in background_rules

    assert "逐帧继承原图的局部对比、亮度分布、渐变方向、渐变幅度和原有低频结构" in background_rules
    assert "跨帧同一原始明度必须映射为同一输出明度" in background_rules
    assert "不得把目标域统一填充成固定亮度" in background_rules

    assert "目标色只使用不携带明度、感知亮度、饱和度或表面处理含义的色相类别" in background_rules
    assert "不得使用明亮、浅、深、鲜艳、柔和、均匀或哑光修饰目标色相" in background_rules
    assert "不得使用“明显”等程度修饰" in background_rules
    assert "环境背景域只允许上述 hue-only 映射" in background_rules
    assert "差异必须明显可见" not in background_rules


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
