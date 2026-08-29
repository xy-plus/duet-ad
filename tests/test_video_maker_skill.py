from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "video-maker" / "SKILL.md"
ARCHIVE = ROOT / "web" / "video-maker.zip"


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_has_one_original_video_analysis_responsibility():
    text = _skill_text()

    for required in (
        "关键词",
        "片段分析",
        "segments[N>=1]",
        "每段恰好 9 张",
        "原始关键帧",
        "旧视频提示词",
    ):
        assert required in text

    for forbidden in (
        "multimodal_audio",
        "multimodal_input.json",
        "h3_prompt_plan.json",
        "speaker_visibility",
        "reconcile_after_image_optimization",
        "unified_visual_ir.json",
        "phase_input_conflict",
        "音画阶段",
        "发声人物可见性阶段",
        "优化后语义协调阶段",
    ):
        assert forbidden not in text


def test_every_segment_gets_exactly_nine_unmodified_source_frames():
    text = _skill_text()

    for required in (
        "`work/NN_frame_*.png`",
        "`work/keyframes/01.png` 至 `09.png`",
        "恰好选择 9 张不同的原始帧",
        "按源时间升序",
        "逐字节复制",
        "不得裁剪、调色、修图、重绘或生成替代帧",
        "不得重复同一帧凑足 9 张",
    ):
        assert required in text

    for retired in (
        "最多选择 9 张",
        "不凑数",
        "用户指定数量时按用户",
        "scripts/crop_image.py",
    ):
        assert retired not in text


def test_old_video_prompt_preserves_only_observed_source_video_facts():
    text = _skill_text()

    for required in (
        "`work/prompt.txt`",
        "动作",
        "镜头",
        "构图",
        "节奏",
        "segment 时间轴",
        "与源片段时长一致",
        "不写具体秒数",
        "眼见为实",
        "不引入优化后的图片内容",
    ):
        assert required in text

    for forbidden in (
        "voice_lines.json",
        "dialogue",
        "声线",
        "环境声",
        "音效",
        "Context IR",
        "H3",
        "供应商",
    ):
        assert forbidden not in text


def test_skill_emits_no_extra_downstream_artifacts():
    text = _skill_text()

    assert "输出只有" in text
    assert "`work/keyframes/01.png` 至 `09.png`" in text
    assert "`work/prompt.txt`" in text
    for artifact in (
        "h3_prompt_plan.json",
        "speaker_visibility_output.json",
        "unified_visual_ir.json",
        "output receipt",
    ):
        assert artifact not in text


def test_project_index_phase_adds_one_optional_output_without_replacing_segment_work():
    text = _skill_text()

    for preserved in (
        "逐段视觉分析",
        "每段恰好 9 张",
        "`work/keyframes/01.png` 至 `09.png`",
        "`work/prompt.txt`",
        "旧视频提示词",
    ):
        assert preserved in text

    for added in (
        '`phase="project_index"`',
        "`work/project_index_request.json`",
        "`work/element_index.json`",
        "所有 segment 的冻结关键帧",
        "`people/entities/scenes`",
        "`source_visual_description`",
        "`occurrences`",
        "`segment_index`",
        "`frame_orders`",
        "`replaceable`",
        "`preserve`",
        "稳定语义 key",
    ):
        assert added in text

    assert "project_index 不读取 `work/prompt.txt`" in text
    assert "不改变逐 segment 阶段的任何既有输入、输出或规则" in text


def test_project_index_keys_are_neutral_ids_and_keep_instance_identity():
    text = _skill_text()

    for required in (
        "不可变索引 key",
        "stable key 是跨帧绑定用的 ID",
        "按 segment 顺序逐帧建立可见元素清单",
        "不得只输出首个 segment 或只输出画面主体",
        "person-01",
        "entity-01",
        "scene-01",
        "逐帧回查已建 ID",
        "换 scene 不会清除可见人物或实体的 occurrence",
        "硬切后先按新 scene 判断",
        "key 本身不得包含或暗示可替换的源属性",
        "逐字复用原 ID",
        "物理上独立、可分别移动或分别接触的同类元素分别分配 ID",
        "不用集合或群组 ID",
        "非人物实体优先限于前景、被持握或被动作直接作用的独立对象",
        "单帧出镜只有在对象清晰、完整且位于前景、被持握或被动作直接作用时才记录",
        "过渡扫到的家具/收纳物/陈设/杂物不升格为实体",
    ):
        assert required in text


def test_project_index_is_a_single_additive_phase_without_quality_control_flow():
    text = _skill_text()
    section = text.split("## 项目级可替换元素索引", 1)[1]

    assert "只执行一次" in section
    assert "只写 `work/element_index.json`" in section
    for forbidden in (
        "质量门禁",
        "重试",
        "fallback",
        "回退",
        "候选版本",
        "二次确认",
    ):
        assert forbidden not in section


def test_download_archive_matches_source_with_deterministic_metadata():
    source = SKILL.parent
    files = sorted(path for path in source.rglob("*") if path.is_file())
    expected = [f"video-maker/{path.relative_to(source).as_posix()}" for path in files]

    with ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == expected
        for path, info in zip(files, infos, strict=True):
            assert archive.read(info) == path.read_bytes()
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == ZIP_DEFLATED
            assert info.create_system == 3
            assert info.external_attr >> 16 == 0o100644
