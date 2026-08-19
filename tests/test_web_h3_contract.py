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
        pytest.skip("node is unavailable; static Web contract tests still run")
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
    assert "face" + "_hold" not in source
    assert "Seed" + "ance" not in source
    assert "H3" in source
    assert "最长 15 秒" in source
    assert "最长 " + "300 秒" not in source


def test_submit_and_detail_state_are_part_of_the_static_contract():
    source = APP_JS.read_text(encoding="utf-8")
    for token in (
        '"/submit"',
        "client_request_id",
        "dialogue_mode",
        "fit_mode",
        "fit_required",
        "target_language",
        "submission_unknown",
        "resume_required",
        "receipt_version",
        "read_only",
        "duration_s",
    ):
        assert token in source
    assert "提交结果未知，禁止重复提交" in source
    assert 'if (detail.has_video || generation.status === "succeeded")' not in source
    assert "if (detail.has_video)" in source


def test_creation_keeps_the_preparation_transform_without_none_fallback():
    source = APP_JS.read_text(encoding="utf-8")
    assert 'fd.append("voice_mode", mode || "keep")' in source
    assert 'if (mode === "translate" && targetLanguage)' in source


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


def test_detail_signature_tracks_h3_render_fields():
    result = _run_contract(
        "(() => {"
        "const base={status:'done',read_only:false,submit_enabled:true,fit_required:false,duration_s:10,"
        "receipt_version:1,fit_mode:'none',dialogue:[],"
        "generation:{status:null,error:null,attempt:0,client_request_id:null},keyframes:[],segments:[]};"
        "const original=contract.detailSignature(base).stable;"
        "const variants=["
        "{...base,read_only:true},{...base,dialogue:[{start_s:0,end_s:1,text:'x'}]},"
        "{...base,receipt_version:2},{...base,fit_mode:'pad'},"
        "{...base,generation:{status:'running',error:null,attempt:1,client_request_id:'request-a'}},"
        "{...base,generation:{status:'failed',error:'x',attempt:2,client_request_id:'request-b'}}];"
        "return variants.map(value => contract.detailSignature(value).stable !== original);"
        "})()"
    )
    assert result == [True, True, True, True, True, True]


def test_resume_builder_reuses_the_persisted_paid_attempt_exactly():
    result = _run_contract(
        "contract.buildResumePayload({"
        "generation:{status:'resume_required',client_request_id:'paid-attempt-1'},"
        "dialogue:{mode:'edit',lines:[{start_s:0,end_s:1.25,text:'原样台词'}]},fit_mode:'pad'})"
    )
    assert result == {
        "confirm": True,
        "client_request_id": "paid-attempt-1",
        "dialogue_mode": "edit",
        "lines": [{"start_s": 0, "end_s": 1.25, "text": "原样台词"}],
        "fit_mode": "pad",
    }


def test_generation_actions_never_retry_unknown_or_active_statuses():
    result = _run_contract(
        "[null,'failed','resume_required','submission_unknown','queued','running','succeeded']"
        ".map(contract.generationAction)"
    )
    assert result == ["new", "retry", "resume", "none", "none", "none", "none"]


def test_resume_ui_is_locked_and_explicit_about_cost():
    source = APP_JS.read_text(encoding="utf-8")
    assert "继续既有任务" in source
    assert "不会创建新付费 attempt" in source
    assert "buildResumePayload(detail)" in source
    resume_function = source.split("async function resumeGeneration", 1)[1].split("async function postGeneration", 1)[0]
    assert "newRequestId" not in resume_function
    new_or_retry_function = source.split("async function submitGeneration", 1)[1].split("async function resumeGeneration", 1)[0]
    assert "newRequestId()" in new_or_retry_function


def test_submission_unknown_branch_has_no_action_button():
    source = APP_JS.read_text(encoding="utf-8")
    branch = source.split('if (generation.status === "submission_unknown")', 1)[1].split("if (!canOperate(detail))", 1)[0]
    assert "button" not in branch


def test_pure_submit_builder_omits_lines_for_auto_and_none():
    result = _run_contract(
        "["
        "contract.buildSubmitPayload({clientRequestId:'request-123', dialogueMode:'auto', "
        "fitRequired:false, fitMode:''}),"
        "contract.buildSubmitPayload({clientRequestId:'request-456', dialogueMode:'none', "
        "fitRequired:false, fitMode:'crop'})"
        "]"
    )
    assert result == [
        {
            "confirm": True,
            "client_request_id": "request-123",
            "dialogue_mode": "auto",
            "fit_mode": "none",
        },
        {
            "confirm": True,
            "client_request_id": "request-456",
            "dialogue_mode": "none",
            "fit_mode": "none",
        },
    ]


def test_pure_submit_builder_parses_custom_lines():
    result = _run_contract(
        "contract.buildSubmitPayload({clientRequestId:'request-789', dialogueMode:'custom', "
        "linesText:'0 - 1.25 | Hello\\n1.25 - 2 | world', fitRequired:true, fitMode:'pad'})"
    )
    assert result == {
        "confirm": True,
        "client_request_id": "request-789",
        "dialogue_mode": "custom",
        "lines": [
            {"start_s": 0, "end_s": 1.25, "text": "Hello"},
            {"start_s": 1.25, "end_s": 2, "text": "world"},
        ],
        "fit_mode": "pad",
    }


@pytest.mark.parametrize(
    "expression",
    [
        "contract.buildSubmitPayload({clientRequestId:'request-1', dialogueMode:'custom', linesText:'', fitRequired:false})",
        "contract.buildSubmitPayload({clientRequestId:'request-2', dialogueMode:'auto', fitRequired:true, fitMode:'none'})",
    ],
)
def test_pure_submit_builder_rejects_incomplete_attempts(expression):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable; static Web contract tests still run")
    script = (
        "const contract = require(process.argv[1]);"
        f"try {{ {expression}; process.exit(2); }} catch (error) {{ process.stdout.write(error.message); }}"
    )
    completed = subprocess.run(
        [node, "-e", script, str(APP_JS)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip()
