from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills/video-maker/SKILL.md"
ARCHIVE = ROOT / "web/video-maker.zip"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_segment_contract_keeps_backend_authority_and_schema_output():
    text = _text()
    for required in (
        "segments[N>=1]", "work/keyframes/01.png", "09.png", "由后端负责",
        "JSON Schema", "禁止创建或修改", "动作因果", "segment 时间轴",
    ):
        assert required in text
    for forbidden in ("H3", "Context IR", "multimodal_input.json", "写 `work/prompt.txt`"):
        assert forbidden not in text


def test_render_switch_instruction_is_single_and_short():
    instruction = "开关为真时不描述字幕或标志。"
    assert _text().count(instruction) == 1
    assert len(instruction.removesuffix("。")) <= 15


def test_project_index_has_first_class_neutral_relations():
    text = _text()
    for required in (
        "people/entities/scenes/relations",
        "subject_key/predicate/object_key/occurrences/preserve/replace_together",
        "frame_order", "state", "geometry",
        "不可变中性 ID", "主客体互换", "当前状态和相对几何",
        "接口、尺度或功能配合", "不因常识补造功能",
    ):
        assert required in text


def test_skill_is_compact_and_sample_neutral():
    text = _text()
    assert len(text.encode("utf-8")) < 7_000
    for sample_term in ("陀螺", "发射器", "梳毛", "聚餐", "玩具"):
        assert sample_term not in text


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
