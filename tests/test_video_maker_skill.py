from pathlib import Path
from zipfile import ZipFile


SKILL = Path(__file__).parents[1] / "skills" / "video-maker" / "SKILL.md"
ARCHIVE = Path(__file__).parents[1] / "web" / "video-maker.zip"


def test_skill_keeps_normal_people_guidance_without_face_workaround():
    text = SKILL.read_text(encoding="utf-8")

    for retained in ("主体可见", "遮挡少", "嘴型与画面同步", "不要生成背景音乐"):
        assert retained in text
    for retired in ("捂脸", "1秒内快速把手放下", "face_hold", "Seedance"):
        assert retired not in text


def test_download_archive_matches_skill_source():
    source = SKILL.parent
    files = sorted(path for path in source.rglob("*") if path.is_file())

    with ZipFile(ARCHIVE) as archive:
        expected = sorted(f"video-maker/{path.relative_to(source).as_posix()}" for path in files)
        assert sorted(name for name in archive.namelist() if not name.endswith("/")) == expected
        for path, name in zip(files, expected, strict=True):
            assert archive.read(name) == path.read_bytes()
