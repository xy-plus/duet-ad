import json
import shutil
import subprocess

import pytest

from test_web_h3_contract import APP_JS, _run_contract


STYLES = APP_JS.parent / "styles.css"


def _run_jsdom_contract(expression: str):
    """Run optional DOM contracts without coupling them to a frontend project."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    availability = subprocess.run(
        [node, "-e", "require.resolve('jsdom')"],
        capture_output=True,
        text=True,
    )
    if availability.returncode != 0:
        pytest.skip("jsdom is unavailable")
    script = (
        "const contract=require(process.argv[1]);"
        "const {JSDOM}=require('jsdom');"
        "const dom=new JSDOM('<!doctype html><body></body>');"
        "global.document=dom.window.document;global.window=dom.window;"
        "global.URL.createObjectURL=(()=>{let n=0;return()=>`blob:test-${++n}`})();"
        "global.URL.revokeObjectURL=()=>{};"
        "global.fetch=async()=>({ok:true,blob:async()=>new Blob(['image'])});"
        f"Promise.resolve({expression}).then(result=>process.stdout.write(JSON.stringify(result)))"
        ".catch(error=>{console.error(error);process.exit(1)});"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_frame_entries_bind_authoritative_segment_paths_prompts_and_status():
    digest = "a" * 64
    result = _run_contract(
        "(()=>{const one={index:1,start_s:0,end_s:8,join_mode:'hard_cut',"
        "keyframes:['01.png','02.png'],keyframe_paths:['segments/1/work/keyframes/01.png',"
        "'segments/1/work/keyframes/02.png'],prompt:'segment-one',"
        f"image_optimization_prompts:[{{frame_name:'01.png',text:'image-one',default_text:'d1',sha256:'{digest}'}},"
        f"{{frame_name:'02.png',text:'image-two',default_text:'d2',sha256:'{digest}'}}]}};"
        "const two={index:2,start_s:8,end_s:16,join_mode:'continuous',keyframes:['01.png'],"
        "keyframe_paths:['segments/2/work/keyframes/01.png'],prompt:'segment-two'};"
        "const detail={id:'cid',segment_count:2,segments:[one,two],postprocess:{status:'running',"
        "options:{optimize_image:true},frames:['segments/1/work/postprocessed/01.png'],segments:["
        "{index:1,completed_frames:1,total_frames:2},{index:2,completed_frames:0,total_frames:1}]}};"
        "const entries=contract.frameViewerEntries(detail);"
        "return {entries,defaultId:contract.selectedFrameEntry(entries,null).id,"
        "persisted:contract.selectedFrameEntry(entries,entries[2].id).id}})()"
    )
    assert len(result["entries"]) == 3
    assert result["entries"][0] == {
        "id": "s1-f1-01.png",
        "segmentIndex": 1,
        "frameIndex": 1,
        "name": "01.png",
        "originalPath": "segments/1/work/keyframes/01.png",
        "optimizedPath": "segments/1/work/postprocessed/01.png",
        "status": "processed",
        "statusLabel": "已优化",
        "generationPrompt": "segment-one",
        "imagePrompt": "image-one",
        "relation": "第 1 段 · 硬切衔接 · 第 1 帧",
        "timeRange": "0.0–8.0 秒",
        "completedFrames": 1,
        "totalFrames": 2,
    }
    assert result["entries"][1]["status"] == "running"
    assert result["entries"][1]["imagePrompt"] == "image-two"
    assert result["defaultId"] == "s1-f2-02.png"
    assert result["persisted"] == "s2-f1-01.png"


def test_unpublished_or_untrusted_frames_are_explicit_and_never_guessed():
    result = _run_contract(
        "(()=>{const segment={index:1,start_s:0,end_s:8,keyframes:['01.png'],"
        "keyframe_paths:['keyframes/01.png'],prompt:'p'};"
        "const detail={id:'cid',segment_count:1,segments:[segment],postprocess:{status:'done',"
        "options:{optimize_image:true},frames:['unexpected.png']}};"
        "const entry=contract.frameViewerEntries(detail)[0];"
        "const invalid={...detail,segments:[{...segment,keyframe_paths:['keyframes/../meta.json']}]};"
        "return {entry,invalidCount:contract.frameViewerEntries(invalid).length}})()"
    )
    assert result["entry"]["optimizedPath"] is None
    assert result["entry"]["status"] == "unavailable"
    assert result["entry"]["statusLabel"] == "优化图未发布"
    assert result["invalidCount"] == 0


def test_frame_inspector_dom_renders_one_selection_and_persists_switch():
    result = _run_jsdom_contract(
        "(async()=>{const segment={index:1,start_s:0,end_s:8,join_mode:'hard_cut',"
        "keyframes:['01.png','02.png'],keyframe_paths:['keyframes/01.png','keyframes/02.png'],"
        "prompt:'generation'};const detail={id:'cid',segment_count:1,segments:[segment],"
        "postprocess:{status:'done',options:{optimize_image:true},frames:['01.png','02.png']}};"
        "const first=contract.frameInspector(detail,segment,{context:'dom',mode:'generation'});"
        "document.body.appendChild(first.node);await new Promise(resolve=>setTimeout(resolve,0));"
        "const before={cards:first.node.querySelectorAll('.frame-inspector-detail .frame-media-card').length,"
        "prompts:first.node.querySelectorAll('.frame-inspector-detail .prompt-card').length,"
        "options:first.node.querySelectorAll('[role=option]').length};"
        "const picker=first.node.querySelector('details');picker.open=true;"
        "picker.dispatchEvent(new window.Event('toggle'));await new Promise(resolve=>setTimeout(resolve,0));"
        "const options=[...first.node.querySelectorAll('[role=option]')];options[1].click();"
        "const after={label:first.node.querySelector('.frame-picker-selected').textContent,"
        "cards:first.node.querySelectorAll('.frame-inspector-detail .frame-media-card').length,"
        "prompts:first.node.querySelectorAll('.frame-inspector-detail .prompt-card').length,"
        "selected:options.map(x=>x.getAttribute('aria-selected')),connected:first.node.isConnected};"
        "first.dispose();first.node.remove();const second=contract.frameInspector(detail,segment,"
        "{context:'dom',mode:'generation'});document.body.appendChild(second.node);"
        "const restored=second.node.querySelector('.frame-picker-selected').textContent;second.dispose();"
        "return {before,optionCount:options.length,after,restored}})()"
    )
    assert result == {
        "before": {"cards": 2, "prompts": 1, "options": 0},
        "optionCount": 2,
        "after": {
            "label": "第 1 段 · 第 2 帧 · 02.png",
            "cards": 2,
            "prompts": 1,
            "selected": ["false", "true"],
            "connected": True,
        },
        "restored": "第 1 段 · 第 2 帧 · 02.png",
    }


def test_phase2_copy_hides_internal_b_terms_without_rewriting_api_fields():
    source = APP_JS.read_text(encoding="utf-8")
    for forbidden in (
        "成品 B", "成品B", "成片 B", "成片B", "B 已提交", "B已提交",
        "B 未提交", "B未提交", "B 输出", "B输出", "等待 B", "等待B",
    ):
        assert forbidden not in source

    results = source.split("function renderResults(detail)", 1)[1].split(
        "function canOperate", 1
    )[0]
    assert '"generated.mp4"' in results
    assert '"新视频"' in results
    for internal_mount in (
        "keyframesSection(",
        "promptWorkspace(",
        "renderSegments(",
        "segmentProductsDisclosure(",
        "ppFramesSection(",
    ):
        assert internal_mount not in results
    assert "has_video" in source


def test_phase2_viewer_is_single_selection_and_mobile_safe():
    source = APP_JS.read_text(encoding="utf-8")
    workspace = source.split("function promptWorkspace(detail", 1)[1].split(
        "function editablePromptCard", 1
    )[0]
    segments = source.split("function renderSegments", 1)[1].split(
        "function resetSegmentProductsDisclosure", 1
    )[0]
    optimized = source.split("function ppFramesSection", 1)[1].split(
        "function ppResultDisclosure", 1
    )[0]
    assert "frameInspector(detail, segment" in workspace
    assert "kfGrid(" not in workspace
    assert "kfGrid(" not in segments
    assert "kfGrid(" not in optimized

    css = STYLES.read_text(encoding="utf-8")
    mobile = css.split("@media (max-width: 768px)", 1)[1]
    assert ".frame-compare { grid-template-columns: 1fr; }" in mobile
    assert ".frame-picker-list" in mobile
    assert "position: fixed" in mobile.split(".frame-picker-list", 1)[1].split("}", 1)[0]
