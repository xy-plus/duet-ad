from pathlib import Path
from zipfile import ZipFile


SKILL = Path(__file__).parents[1] / "skills" / "video-maker" / "SKILL.md"
ARCHIVE = Path(__file__).parents[1] / "web" / "video-maker.zip"
HUMAN_DOC = (
    Path(__file__).parents[1]
    / "docs"
    / "human"
    / "features"
    / "conversation-task"
    / "behaviors"
    / "upload-create.md"
)


def test_skill_keeps_visual_phase_independent_from_audio_planning():
    text = SKILL.read_text(encoding="utf-8")

    for retained in ("主体可见", "遮挡少"):
        assert retained in text
    assert "multimodal_input.json" in text
    assert "h3_prompt_plan.json" in text
    assert "视觉阶段" in text
    assert "音画阶段" in text
    for retired in (
        "捂脸",
        "1秒内快速把手放下",
        "face_hold",
        "Seedance",
    ):
        assert retired not in text


def test_multimodal_phase_emits_structured_plan_not_provider_prompt():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        '"phase": "multimodal_audio"',
        '"eligible": true',
        '"subjects"',
        '"audio_refs"',
        '"dialogue"',
        '"sound_design"',
        '"subject_id": "S1"',
        '"language"',
        '"text"',
        "后端确定性编译",
    ):
        assert required in text
    assert "不得直接写供应商最终提示词" in text


def test_multimodal_phase_binds_audio_semantics_and_fails_closed():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "1–3 段",
        "说话人与人物映射",
        "声线参考",
        "精确台词",
        "语言",
        "旁白",
        "环境声",
        "音效",
        "参考音频不是时间锁",
        '"eligible": false',
        '"reason"',
    ):
        assert required in text
    for forbidden in (
        "根据时间重叠猜测说话人",
        "参考音频会逐样本复用原音",
        "参考音频等于最终音轨",
    ):
        assert forbidden not in text


def test_multimodal_plan_uses_stable_one_based_references():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "图片与音频的外部编号都从 1 开始",
        "不得出现 0",
        "不得按数组位置重新编号",
        '"picture_refs": [1, 2]',
        "一个人物可引用多张图片",
    ):
        assert required in text


def test_multimodal_sound_items_have_a_strict_compiler_schema():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "`sound_design.narration[]` 每项严格只有 `order/language/text/voice_ref`",
        "`sound_design.ambience_refs[]` 每项严格只有 `audio_index/description`",
        "`sound_design.effects[]` 每项严格只有 `audio_index/description`",
        "共用一条从 1 开始且无缺号的全局发声顺序",
        '"voice_ref": null',
        '"description": "逐字保留输入的声音描述"',
    ):
        assert required in text


def test_multimodal_plan_does_not_claim_provider_level_audio_control():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "不是供应商的 speaker-face API",
        "不是精确 PTS",
        "不得使用 `fully_copy`、`partially_copy` 或 `audio reuse`",
        "不得向计划加入媒体路径、字节、哈希、格式、模式或供应商参数",
    ):
        assert required in text
    for forbidden in (
        '"speaker_id"',
        '"face_id"',
        '"start_pts"',
        '"end_pts"',
        "我会准时回来",
        "First batch of the morning",
    ):
        assert forbidden not in text


def test_human_doc_explains_multimodal_semantic_plan_boundary():
    text = HUMAN_DOC.read_text(encoding="utf-8")

    for required in (
        "H3 多模态语义计划",
        "1-based",
        "全局发声顺序",
        "speaker-face",
        "精确 PTS",
        "样本级音频复用",
        "字节、哈希、格式、模式",
    ):
        assert required in text


def test_audio_phase_preserves_visual_facts_and_does_not_reselect_frames():
    text = SKILL.read_text(encoding="utf-8")

    assert "不得改写视觉事实" in text
    assert "不得重新选关键帧" in text
    assert "不得补造未提供的人物、台词或声音事件" in text


def test_download_archive_matches_skill_source():
    source = SKILL.parent
    files = sorted(path for path in source.rglob("*") if path.is_file())

    with ZipFile(ARCHIVE) as archive:
        expected = sorted(f"video-maker/{path.relative_to(source).as_posix()}" for path in files)
        assert sorted(name for name in archive.namelist() if not name.endswith("/")) == expected
        for path, name in zip(files, expected, strict=True):
            assert archive.read(name) == path.read_bytes()


def test_prompt_contract_uses_source_duration_without_numeric_seconds():
    text = SKILL.read_text(encoding="utf-8")

    assert "与源片段时长一致" in text
    assert "[目标时长] 秒" not in text
    assert "不写具体秒数" in text
