import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "app.js"
INDEX_HTML = ROOT / "web" / "index.html"
STYLES_CSS = ROOT / "web" / "styles.css"


def _web_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (INDEX_HTML, APP_JS, STYLES_CSS)
    )


def _run_contract(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    script = (
        "const contract = require(process.argv[1]);"
        f"Promise.resolve(({expression})).then((result) => {{"
        "process.stdout.write(JSON.stringify(result));"
        "});"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _function_source(source, name):
    marker = f"function {name}("
    assert marker in source
    return marker + source.split(marker, 1)[1].split("\nfunction ", 1)[0]


def test_removed_legacy_endpoints_and_contracts_have_no_frontend_references():
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    for removed_endpoint in (
        '"/submit"',
        '"/prompt"',
        "/postprocess",
        "/dialogue-review/commit",
        "/image-optimization-prompt",
    ):
        assert removed_endpoint not in source
    for removed_contract in (
        "resume_required",
        "buildResumePayload",
        "buildSubmitPayload",
        "generationAction",
        "generationRetryContract",
        "canOperate",
        "longVideoContract",
        "fitProfile",
        "requestGenerationSubmit",
        "normalizeGenerationConfigCapability",
        "normalizeDialogueReviewCapability",
        "dialogueReviewPolicy",
        "loadGenerationConfigCapability",
        "resetGenerationConfigDisclosure",
        "guardDirtyPrompt",
        'dyn: ""',
    ):
        assert removed_contract not in source
    assert 'class="legacy-contract-controls"' not in html
    assert 'id="pp-dialog"' not in html
    assert 'id="draft-dialog"' not in html
    # 新续跑入口是详情页唯一的项目级恢复手段
    assert '"/resume-picsart"' in source
    assert '"/resume-pre-picsart"' in source


def test_resume_posts_confirm_true_json_to_seedance_first():
    result = _run_contract(
        "(async()=>{const calls=[];"
        "const request=async(path,options)=>{calls.push({path,method:options.method,"
        "contentType:options.headers['Content-Type'],body:JSON.parse(options.body)});"
        "return {status:'queued'};};"
        "const state=await contract.resumeFailedProject('proj 1',request);"
        "return {calls,state}})()"
    )
    assert result == {
        "calls": [{
            "path": "/api/conversations/proj%201/resume-picsart",
            "method": "POST",
            "contentType": "application/json",
            "body": {"confirm": True},
        }],
        "state": {"status": "queued"},
    }


def test_resume_falls_back_to_pre_seedance_only_when_seedance_not_ready():
    result = _run_contract(
        "(async()=>{const calls=[];const bodies=[];"
        "const request=async(path,options)=>{calls.push(path);bodies.push(JSON.parse(options.body));"
        "if(path.endsWith('/resume-picsart')){"
        "const e=new Error('项目缺少可复用的 Seedance 前置产物');e.code='seedance_resume_not_ready';throw e;}"
        "return {status:'queued'};};"
        "const state=await contract.resumeFailedProject('p1',request);"
        "return {calls,bodies,state}})()"
    )
    assert result["calls"] == [
        "/api/conversations/p1/resume-picsart",
        "/api/conversations/p1/resume-pre-picsart",
    ]
    assert result["bodies"] == [{"confirm": True}, {"confirm": True}]
    assert result["state"] == {"status": "queued"}


def test_resume_does_not_fall_back_when_project_is_not_failed():
    result = _run_contract(
        "(async()=>{const calls=[];"
        "const request=async(path)=>{calls.push(path);"
        "const e=new Error('只有失败项目可以续跑 Seedance');e.code='project_not_failed';throw e;};"
        "try{await contract.resumeFailedProject('p1',request);return {calls,error:null}}"
        "catch(e){return {calls,error:{code:e.code,message:e.message}}}})()"
    )
    assert result == {
        "calls": ["/api/conversations/p1/resume-picsart"],
        "error": {"code": "project_not_failed", "message": "只有失败项目可以续跑 Seedance"},
    }


def test_resume_surfaces_backend_detail_when_both_entry_points_are_not_ready():
    result = _run_contract(
        "(async()=>{const calls=[];"
        "const request=async(path)=>{calls.push(path);"
        "const e=path.endsWith('/resume-picsart')"
        "?Object.assign(new Error('项目缺少可复用的 Seedance 前置产物'),{code:'seedance_resume_not_ready'})"
        ":Object.assign(new Error('项目缺少可安全复用的前置产物，或已存在 Seedance 提交痕迹'),"
        "{code:'pre_submit_resume_not_ready'});throw e;};"
        "try{await contract.resumeFailedProject('p1',request);return {calls,error:null}}"
        "catch(e){return {calls,error:{code:e.code,message:e.message}}}})()"
    )
    assert result == {
        "calls": [
            "/api/conversations/p1/resume-picsart",
            "/api/conversations/p1/resume-pre-picsart",
        ],
        "error": {
            "code": "pre_submit_resume_not_ready",
            "message": "项目缺少可安全复用的前置产物，或已存在 Seedance 提交痕迹",
        },
    }


def test_failed_projects_get_a_confirmed_resume_action_in_the_fail_card():
    source = APP_JS.read_text(encoding="utf-8")
    render_fail = _function_source(source, "renderFail")
    assert 'detail.status === "failed"' in render_fail
    assert '"续跑生成"' in render_fail
    assert "window.confirm(" in render_fail
    assert "resumeFailedProject(detail.id)" in render_fail
    assert "loadDetail(detail.id, true)" in render_fail
    assert "showActionError(err, errorBox, [resume])" in render_fail
    assert "handleAuthError(err)" in render_fail
    # 续跑成功后不重发 POST，只刷新详情并交给现有轮询
    assert render_fail.count("resumeFailedProject(") == 1


def test_api_error_parser_reads_only_the_detail_envelope():
    result = _run_contract(
        "(()=>({"
        "detailObject:(()=>{const e=contract.apiErrorFromPayload("
        "{detail:{code:'queue_full',message:'当前队列已满'}},'fallback');"
        "return {code:e.code,message:e.message}})(),"
        "detailString:(()=>{const e=contract.apiErrorFromPayload({detail:'出错了'},'fallback');"
        "return {code:e.code,message:e.message}})(),"
        "legacyErrorKey:(()=>{const e=contract.apiErrorFromPayload("
        "{error:{code:'x',message:'y'}},'fallback');return {code:e.code,message:e.message}})(),"
        "legacyMessageKey:(()=>{const e=contract.apiErrorFromPayload({message:'y'},'fallback');"
        "return {code:e.code,message:e.message}})(),"
        "empty:(()=>{const e=contract.apiErrorFromPayload(null,'fallback');"
        "return {code:e.code,message:e.message}})()"
        "}))()"
    )
    assert result == {
        "detailObject": {"code": "queue_full", "message": "当前队列已满"},
        "detailString": {"code": "", "message": "出错了"},
        "legacyErrorKey": {"code": "", "message": "fallback"},
        "legacyMessageKey": {"code": "", "message": "fallback"},
        "empty": {"code": "", "message": "fallback"},
    }


def test_duration_limit_error_is_structured_and_shown_as_popup():
    source = APP_JS.read_text(encoding="utf-8")
    assert 'err.code === "video_duration_exceeds_h3_limit"' in source
    assert "window.alert(err.message)" in source
    result = _run_contract(
        "(() => {"
        "const error=contract.apiErrorFromPayload({detail:{code:'video_duration_exceeds_h3_limit',"
        "message:'最长 14 秒'}},'fallback');"
        "return {message:error.message,code:error.code};"
        "})()"
    )
    assert result == {
        "message": "最长 14 秒",
        "code": "video_duration_exceeds_h3_limit",
    }


def test_stale_page_error_shows_refresh_instruction():
    result = _run_contract(
        "(() => {"
        "const error=contract.apiErrorFromPayload({detail:{code:'client_refresh_required',"
        "message:'页面版本已更新，请刷新页面后重试。'}},'fallback');"
        "return {message:error.message,code:error.code};"
        "})()"
    )
    assert result == {
        "message": "页面版本已更新，请刷新页面后重试。",
        "code": "client_refresh_required",
    }
