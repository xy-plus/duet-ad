import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "app.js"
INDEX_HTML = ROOT / "web" / "index.html"
STYLES_CSS = ROOT / "web" / "styles.css"


def _run_contract(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    script = (
        "const contract=require(process.argv[1]);"
        f"const result=({expression});"
        "process.stdout.write(JSON.stringify(result));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_success_operation_has_one_truthful_nine_stage_timeline():
    result = _run_contract(
        "(()=>{const detail={id:'conversation-12345678',has_source:true,status:'done',"
        "created_at:'2026-08-30T00:00:00Z',updated_at:'2026-08-30T00:03:00Z',"
        "duration_s:20,has_video:true,segments:["
        "{index:1,start_s:0,end_s:10,keyframes:['01.png','02.png']},"
        "{index:2,start_s:10,end_s:20,keyframes:['01.png','02.png']}],"
        "postprocess:{status:'done',segments:["
        "{index:1,completed_frames:2,total_frames:2},{index:2,completed_frames:2,total_frames:2}]},"
        "prompt_fusion:{status:'done',segments:[{index:1,status:'done'},{index:2,status:'done'}]},"
        "generation:{status:'succeeded',stage:'stitch',segments:["
        "{index:1,status:'succeeded'},{index:2,status:'succeeded'}]}};"
        "const model=contract.operationTimeline(detail,Date.parse('2026-08-30T00:10:00Z'));"
        "return {keys:model.stages.map(x=>x.key),statuses:model.stages.map(x=>x.status),"
        "counts:model.stages.map(x=>x.count),current:model.current.key,elapsed:model.elapsed};})()"
    )
    assert result == {
        "keys": ["source", "analysis", "index", "image", "fusion", "context", "h3", "stitch", "output"],
        "statuses": ["done"] * 9,
        "counts": ["", "", "4 帧", "4/4 帧", "2/2 段", "", "2/2 段", "2 段", "可播放"],
        "current": "output",
        "elapsed": "已耗时 3分 00秒",
    }


def test_context_ir_uses_server_stage_and_never_fakes_segment_count():
    result = _run_contract(
        "(()=>{const model=contract.operationTimeline({id:'c1',has_source:true,status:'done',"
        "created_at:'2026-08-30T00:00:00Z',updated_at:'2026-08-30T00:01:00Z',"
        "segments:[{index:1,keyframes:['01.png']},{index:2,keyframes:['01.png']}],"
        "prompt_fusion:{status:'done',segments:[{index:1,status:'done'},{index:2,status:'done'}]},"
        "generation:{status:'running',stage:'context_ir',segments:[]}},"
        "Date.parse('2026-08-30T00:02:00Z'));"
        "const context=model.stages.find(x=>x.key==='context');"
        "const h3=model.stages.find(x=>x.key==='h3');"
        "return {current:model.current.key,context,h3,elapsed:model.elapsed};})()"
    )
    assert result["current"] == "context"
    assert result["context"]["status"] == "running"
    assert result["context"]["count"] == ""
    assert "未公开逐段计数" in result["context"]["detail"]
    assert result["h3"]["status"] == "waiting"
    assert result["elapsed"] == "已耗时 2分 00秒"


def test_submission_unknown_is_prominent_and_not_reinterpreted():
    result = _run_contract(
        "(()=>{const model=contract.operationTimeline({id:'c1',has_source:true,status:'done',"
        "created_at:'2026-08-30T00:00:00Z',updated_at:'2026-08-30T00:01:00Z',segments:[],"
        "generation:{status:'submission_unknown',stage:'h3',segments:[]}},"
        "Date.parse('2026-08-30T00:02:00Z'));return {current:model.current,elapsed:model.elapsed};})()"
    )
    assert result["current"]["key"] == "h3"
    assert result["current"]["status"] == "attention"
    assert "禁止重复提交" in result["current"]["detail"]
    assert result["elapsed"] == "已耗时 1分 00秒"


def test_output_b_is_done_only_when_server_reports_has_video():
    result = _run_contract(
        "[false,true].map(has_video=>{const stages=contract.operationTimeline({"
        "id:'c',has_source:true,status:'done',has_video,created_at:'2026-08-30T00:00:00Z',"
        "updated_at:'2026-08-30T00:01:00Z',generation:{status:'succeeded',stage:'stitch',segments:[]}}).stages;"
        "return stages.find(x=>x.key==='output')})"
    )
    assert result[0]["status"] == "failed"
    assert "缺失" in result[0]["detail"]
    assert result[1]["status"] == "done"
    assert result[1]["count"] == "可播放"


def test_material_index_and_skill_version_are_read_only_public_views():
    result = _run_contract(
        "(()=>{const sha='a'.repeat(64);return {"
        "missing:contract.materialIndexView({segments:[]}),"
        "index:contract.materialIndexView({element_index:{people:{p1:{description:'演员',occurrences:[1,2]}},"
        "entities:{cup:{source_visual_description:'红杯',occurrences:[1]}},scenes:{},"
        "relations:{r1:{subject:'p1',predicate:'拿着',object:'cup'}}}}),"
        "skill:contract.skillMilestoneView({skill_milestone:{id:'skill-'+sha,version:4,"
        "skills:[{name:'video-maker',sha256:sha}]}})}})()"
    )
    assert result["missing"] is None
    assert result["index"]["groups"][0]["entries"][0]["occurrences"] == 2
    assert result["index"]["relations"][0]["predicate"] == "拿着"
    assert result["skill"]["label"] == "Skill v4 · aaaaaaaa"
    assert result["skill"]["skills"][0]["short"] == "aaaaaaaa"
    assert result["skill"]["skills"][0]["sha256"] == "a" * 64


def test_error_summary_is_short_while_diagnostics_are_bounded():
    result = _run_contract(
        "(()=>{const raw={code:'submission_unknown',trace:'x'.repeat(2000)};return {"
        "summary:contract.safeErrorSummary(raw),diagnostic:contract.diagnosticText(raw)}})()"
    )
    assert result["summary"] == "提交结果未知，已禁止重复提交"
    assert len(result["diagnostic"]) < 1250
    assert result["diagnostic"].endswith("…诊断内容已截断")


def test_history_thumbnail_and_summary_use_authoritative_detail_fields():
    result = _run_contract(
        "(()=>{const detail={id:'conversation-12345678',status:'done',has_video:true,duration_s:12.5,"
        "updated_at:'2026-08-30T00:01:00Z',segment_count:1,segments:[{index:1,keyframes:['01.png'],"
        "keyframe_paths:['keyframes/01.png']}],"
        "generation:{status:'succeeded'},skill_milestone:null};const items=[{id:detail.id,title:'demo'}];"
        "return {changed:contract.syncConversationDetail(items,detail),summary:items[0],"
        "thumb:contract.conversationThumbnailPath(detail),short:contract.shortId(detail.id)}})()"
    )
    assert result["changed"] is True
    assert result["summary"]["has_video"] is True
    assert result["summary"]["duration_s"] == 12.5
    assert result["summary"]["thumbnail_path"] == "keyframes/01.png"
    assert result["short"] == "conversa"


def test_persistent_header_history_and_unambiguous_cost_copy_are_shipped():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    css = STYLES_CSS.read_text(encoding="utf-8")
    assert 'id="operation-status"' in html
    assert 'aria-live="polite"' in html
    assert ".operation-timeline" in css
    assert "repeat(9" in css
    for token in ("conv-thumb", "conv-id", "conv-output", "B 已提交", "B 未提交"):
        assert token in js
    for copy in (
        "开始生成成片 B（新增 ",
        "继续原任务（0 新增付费）",
        "仅重试拼接（0 新增付费）",
        "设置素材处理",
        "保持原素材",
    ):
        assert copy in js or copy in html


def test_history_hydration_is_get_only_and_bounded():
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("async function hydrateConversationSummaries")
    end = source.index("function renderList", start)
    hydration = source[start:end]
    assert 'apiJSON("/api/conversations/"' in hydration
    assert ".slice(0, 24)" in hydration
    assert "Promise.all([worker(), worker(), worker()])" in hydration
    for method in ('method: "POST"', 'method: "PATCH"', 'method: "DELETE"'):
        assert method not in hydration


def test_absent_element_index_is_disclosed_without_prompt_inference():
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index("function renderMaterialIndexSection")
    end = source.index("function renderResults", start)
    section = source[start:end]
    assert "人物与物体明细未公开" in section
    assert "页面不会从提示词猜测" in section
    assert "element_index" not in section  # parsing is isolated in the pure view helper
