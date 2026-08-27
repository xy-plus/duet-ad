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
