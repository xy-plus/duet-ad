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
        f"const result = ({expression});"
        "process.stdout.write(JSON.stringify(result));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_creation_and_copy_use_h3_contract_only():
    source = _web_source()
    assert "voice-mode" in source
    assert "target_language" in source
    assert "H3" in source
    assert "最长 300 秒" in source
    assert "10 秒以上会拆分为最长 15 秒的 H3 子任务" in source
    assert "时长不限" not in source


def test_duration_limit_error_is_structured_and_shown_as_popup():
    source = APP_JS.read_text(encoding="utf-8")
    assert 'err.code === "video_duration_exceeds_h3_limit"' in source
    assert "window.alert(err.message)" in source
    result = _run_contract(
        "(() => {"
        "const error=contract.apiErrorFromPayload({detail:{code:'video_duration_exceeds_h3_limit',"
        "message:'最长 10 秒'}},'fallback');"
        "return {message:error.message,code:error.code};"
        "})()"
    )
    assert result == {
        "message": "最长 10 秒",
        "code": "video_duration_exceeds_h3_limit",
    }


def test_context_ir_is_absent_from_web_runtime():
    source = _web_source().lower()
    assert "context ir" not in source
    assert "context_ir" not in source
    assert "contextir" not in source
    assert "/context-ir" not in source


def test_submit_and_detail_state_are_part_of_static_contract():
    source = APP_JS.read_text(encoding="utf-8")
    for token in (
        '"/submit"',
        "client_request_id",
        "dialogue_mode",
        "fit_mode",
        "fit_required",
        "submission_unknown",
        "resume_required",
        "receipt_version",
        "read_only",
        "duration_s",
    ):
        assert token in source
    assert "提交结果未知，禁止重复提交" in source
    assert "H3 源提示词将直接提交生成" in source


def test_read_only_gate_fails_closed_for_legacy_details():
    result = _run_contract(
        "["
        "contract.canOperate({read_only:false, submit_enabled:true}),"
        "contract.canOperate({read_only:true, submit_enabled:true}),"
        "contract.canOperate({submit_enabled:true}),"
        "contract.canOperate({read_only:false, submit_enabled:false})"
        "]"
    )
    assert result == [True, False, False, False]


def test_detail_signature_tracks_direct_h3_render_fields():
    result = _run_contract(
        "(() => {"
        "const base={status:'done',read_only:false,submit_enabled:true,fit_required:false,duration_s:10,"
        "receipt_version:1,fit_mode:'none',dialogue:[],"
        "generation:{status:null,error:null,attempt:0,client_request_id:null,stage:'h3'},"
        "keyframes:[],segments:[]};"
        "const original=contract.detailSignature(base);"
        "const variants=["
        "{...base,read_only:true},{...base,dialogue:[{start_s:0,end_s:1,text:'x'}]},"
        "{...base,receipt_version:2},{...base,fit_mode:'pad'},"
        "{...base,generation:{status:'running',error:null,attempt:1,client_request_id:'request-a',stage:'h3'}},"
        "{...base,source_prompt:'edited',source_prompt_sha256:'b'.repeat(64)}];"
        "return variants.map(value => {const next=contract.detailSignature(value);"
        "return next.stable !== original.stable || next.generation !== original.generation;});"
        "})()"
    )
    assert result == [True, True, True, True, True, True]


def test_resume_builder_reuses_persisted_h3_attempt_exactly():
    payload = _run_contract(
        "contract.buildResumePayload({generation:{status:'resume_required',client_request_id:'request-old'},"
        "dialogue:{mode:'edit',lines:[{text:'x',start_s:0,end_s:1}]},fit_mode:'pad'})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-old",
        "dialogue_mode": "edit",
        "fit_mode": "pad",
        "lines": [{"text": "x", "start_s": 0, "end_s": 1}],
    }


def test_generation_actions_never_retry_unknown_or_active_statuses():
    result = _run_contract(
        "['failed','resume_required','submission_unknown','queued','running','succeeded',null]"
        ".map(contract.generationAction)"
    )
    assert result == ["retry", "resume", "none", "none", "none", "none", "new"]


def test_failed_stitch_has_distinct_non_paid_retry_action():
    result = _run_contract(
        "[contract.generationAction('failed','h3'),"
        "contract.generationAction('failed','stitch'),"
        "contract.generationAction('submission_unknown','stitch')]"
    )
    assert result == ["retry", "retry_stitch", "none"]


def test_resume_ui_is_locked_and_explicit_about_cost():
    source = APP_JS.read_text(encoding="utf-8")
    assert "继续既有 H3 任务" in source
    assert "继续原任务，不创建新的 H3 attempt" in source
    assert "buildResumePayload(detail)" in source


def test_submission_unknown_branch_has_no_action_button():
    source = APP_JS.read_text(encoding="utf-8")
    branch = source.split('if (generation.status === "submission_unknown")', 1)[1]
    branch = branch.split("if (!canOperate(detail))", 1)[0]
    assert "generation-submit" not in branch
    assert "禁止重复提交" in branch


def test_pure_submit_builder_omits_lines_for_auto_and_none():
    result = _run_contract(
        "["
        "contract.buildSubmitPayload({clientRequestId:'request-1',dialogueMode:'auto',fitRequired:false}),"
        "contract.buildSubmitPayload({clientRequestId:'request-2',dialogueMode:'none',fitRequired:false})"
        "]"
    )
    assert result == [
        {"confirm": True, "client_request_id": "request-1", "dialogue_mode": "auto", "fit_mode": "none"},
        {"confirm": True, "client_request_id": "request-2", "dialogue_mode": "none", "fit_mode": "none"},
    ]


def test_pure_submit_builder_parses_custom_lines():
    result = _run_contract(
        "contract.buildSubmitPayload({clientRequestId:'request-3',dialogueMode:'custom',"
        "linesText:'0 - 1.5 | first\\n1.5 - 3 | second',fitRequired:true,fitMode:'crop'})"
    )
    assert result == {
        "confirm": True,
        "client_request_id": "request-3",
        "dialogue_mode": "custom",
        "fit_mode": "crop",
        "lines": [
            {"start_s": 0, "end_s": 1.5, "text": "first"},
            {"start_s": 1.5, "end_s": 3, "text": "second"},
        ],
    }


@pytest.mark.parametrize(
    "expression",
    [
        "contract.buildSubmitPayload({clientRequestId:'request-1',dialogueMode:'custom',linesText:'',fitRequired:false})",
        "contract.buildSubmitPayload({clientRequestId:'request-2',dialogueMode:'auto',fitRequired:true,fitMode:'none'})",
    ],
)
def test_pure_submit_builder_rejects_incomplete_attempts(expression):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    script = (
        "const contract=require(process.argv[1]);"
        f"try{{{expression};process.exit(1)}}catch(error){{process.stdout.write(error.message)}}"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert completed.stdout
