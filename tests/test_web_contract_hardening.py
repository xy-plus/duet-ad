import json
import shutil
import subprocess

import pytest

from test_web_h3_contract import APP_JS


def _run_async_contract(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")
    script = (
        "const contract=require(process.argv[1]);"
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


def test_single_flight_polling_waits_for_rtt_recovers_dom_and_honors_switch():
    result = _run_async_contract(
        "(async()=>{let current=true,scheduled=0,release;const dom={textContent:'处理中'};"
        "const slow=contract.runSingleFlightPollCycle(()=>current,()=>new Promise(resolve=>{"
        "release=()=>{dom.textContent='已完成';resolve()}}),()=>{scheduled+=1});"
        "await Promise.resolve();const during={scheduled,text:dom.textContent};release();await slow;"
        "const afterSlow={scheduled,text:dom.textContent};"
        "dom.textContent='会话加载失败';await contract.runSingleFlightPollCycle(()=>current,async()=>{},"
        "()=>{scheduled+=1});const afterFailure={scheduled,text:dom.textContent};"
        "await contract.runSingleFlightPollCycle(()=>current,async()=>{dom.textContent='恢复并完成';current=false},"
        "()=>{scheduled+=1});const recovered={scheduled,text:dom.textContent};"
        "current=true;let switchRelease;const switched=contract.runSingleFlightPollCycle(()=>current,"
        "()=>new Promise(resolve=>{switchRelease=resolve}),()=>{scheduled+=1});await Promise.resolve();"
        "current=false;switchRelease();await switched;return {during,afterSlow,afterFailure,recovered,"
        "afterSwitch:scheduled}})()"
    )
    assert result == {
        "during": {"scheduled": 0, "text": "处理中"},
        "afterSlow": {"scheduled": 1, "text": "已完成"},
        "afterFailure": {"scheduled": 2, "text": "会话加载失败"},
        "recovered": {"scheduled": 2, "text": "恢复并完成"},
        "afterSwitch": 2,
    }


def test_prompt_changed_fetches_latest_page_and_resets_editor_sha():
    result = _run_async_contract(
        "(async()=>{const old={id:'c1',source_prompt:'page-a',source_prompt_sha256:'a'.repeat(64)};"
        "const output={textContent:'page-a'},textarea={value:'page-a edit'},error={textContent:'',hidden:true};"
        "let gets=0;const latest={id:'c1',source_prompt:'page-b',source_prompt_sha256:'b'.repeat(64),"
        "prompt:'page-b final'};const handled=await contract.recoverPromptChanged({code:'prompt_changed'},old,"
        "async()=>{gets+=1;return latest},{output,textarea,error});return {handled,gets,old,output,textarea,error}})()"
    )
    assert result == {
        "handled": True,
        "gets": 1,
        "old": {
            "id": "c1",
            "source_prompt": "page-b",
            "source_prompt_sha256": "b" * 64,
            "prompt": "page-b final",
        },
        "output": {"textContent": "page-b"},
        "textarea": {"value": "page-b"},
        "error": {
            "textContent": "提示词已在其他页面更新，已加载最新版本，请重新编辑",
            "hidden": False,
        },
    }


def test_postprocess_lock_fetches_latest_options_instead_of_captured_detail():
    result = _run_async_contract(
        "(async()=>{const inputs=[{value:'remove_subtitle',checked:true},"
        "{value:'remove_brand',checked:false}];const hint={hidden:true};"
        "const error={textContent:'',hidden:true};let gets=0;"
        "const latest={id:'c1',postprocess:{options:{remove_subtitle:false,remove_brand:true}}};"
        "const recovered=await contract.recoverLockedPostprocess({code:'postprocess_options_locked'},"
        "async()=>{gets+=1;return latest},inputs,hint,error);return {gets,recovered,inputs,hint,error}})()"
    )
    assert result == {
        "gets": 1,
        "recovered": {"id": "c1", "postprocess": {"options": {
            "remove_subtitle": False,
            "remove_brand": True,
        }}},
        "inputs": [
            {"value": "remove_subtitle", "checked": False},
            {"value": "remove_brand", "checked": True},
        ],
        "hint": {"hidden": False},
        "error": {
            "textContent": "选项已在其他页面锁定，已加载服务端选项，请直接确认",
            "hidden": False,
        },
    }


def test_client_refresh_required_restores_controls_with_stable_chinese_dom_copy():
    result = _run_async_contract(
        "(()=>{const error={textContent:'',hidden:true};const controls=[{disabled:true},{disabled:true}];"
        "contract.showActionError({code:'client_refresh_required',message:'raw'},error,controls);"
        "return {error,controls}})()"
    )
    assert result == {
        "error": {"textContent": "页面版本已更新，请刷新页面后重试。", "hidden": False},
        "controls": [{"disabled": False}, {"disabled": False}],
    }


def test_runtime_handlers_are_wired_to_structured_recovery_contracts():
    source = APP_JS.read_text(encoding="utf-8")
    polling = source.split("function startPolling(id)", 1)[1].split("function stopPolling()", 1)[0]
    assert "setInterval" not in polling
    assert "setTimeout(async () =>" in polling
    assert "await runSingleFlightPollCycle" in polling

    detail_failure = source.split("async function loadDetail", 1)[1].split(
        "function selectConversation", 1
    )[0]
    assert "state.detailSig = null" in detail_failure
    assert "if (silent && state.currentId === id) startPolling(id)" in detail_failure

    prompt = source.split("function renderSourcePromptCard", 1)[1].split(
        "function promptCard", 1
    )[0]
    assert "recoverPromptChanged" in prompt
    assert 'apiJSON("/api/conversations/" + encodeURIComponent(detail.id))' in prompt

    postprocess = source.split("async function submitPostprocess", 1)[1].split(
        "/* ===== 后处理聊天消息", 1
    )[0]
    assert "recoverLockedPostprocess" in postprocess
    assert "err.message.includes" not in postprocess
