from pathlib import Path


WEB_APP = Path(__file__).resolve().parents[1] / "web" / "app.js"


def test_legacy_web_renders_read_only_cid_skill_milestone_summary():
    source = WEB_APP.read_text(encoding="utf-8")
    view_start = source.index("function skillMilestoneView")
    view_end = source.index("function materialIndexView", view_start)
    view = source[view_start:view_end]
    start = source.index("function skillMilestoneSection")
    end = source.index("function renderResults", start)
    section = source[start:end]
    assert "detail && detail.skill_milestone" in view
    assert "milestone.id" in section
    assert "skill.sha256" in section
    assert "skill.short" in section
    assert "Skill v" in view
    assert "查看完整校验信息" in section
    assert "source_path" not in section
    assert "frozen_path" not in section
    assert "dataset.testid = \"skill-milestone\"" in section


def test_legacy_web_places_milestone_in_read_only_results_area():
    source = WEB_APP.read_text(encoding="utf-8")
    render = source[source.index("function renderResults"):source.index("function canOperate")]
    assert "skillMilestoneSection(detail)" in render
    assert "frag.appendChild(milestone)" in render
