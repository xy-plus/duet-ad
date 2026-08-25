from test_web_h3_contract import APP_JS, INDEX_HTML, STYLES_CSS, _run_contract
from test_web_contract_hardening import _run_async_contract


def _sources():
    return (
        APP_JS.read_text(encoding="utf-8"),
        INDEX_HTML.read_text(encoding="utf-8"),
        STYLES_CSS.read_text(encoding="utf-8"),
    )


def test_prompt_workspace_is_one_three_way_state_machine():
    js, _, css = _sources()
    assert "function promptWorkspace" in js
    assert '"展开生成提示词"' in js
    assert '"展开段台词"' in js
    assert '"展开图片优化"' in js
    assert "prompt-workspace-tabs" in js
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in css
    assert "white-space: normal" in css


def test_image_prompt_draft_restore_is_dirty_and_poll_merge_preserves_it():
    result = _run_contract(
        "(()=>{const saved={text:'saved',default_text:'default',sha256:'a'.repeat(64)};"
        "const draft=contract.createImagePromptDraft('c1',2,saved);"
        "contract.restoreImagePromptDefault(draft);"
        "contract.mergeImagePromptDraft(draft,{text:'server-new',default_text:'new-default',sha256:'b'.repeat(64)});"
        "return {text:draft.text,defaultText:draft.defaultText,sha:draft.sha256,dirty:draft.dirty}})()"
    )
    assert result == {
        "text": "default",
        "defaultText": "new-default",
        "sha": "a" * 64,
        "dirty": True,
    }


def test_image_prompt_patch_is_cas_and_segment_aware():
    result = _run_contract(
        "contract.buildImagePromptPatch({segmentIndex:3,sha256:'c'.repeat(64),text:'clean'})"
    )
    assert result == {
        "confirm": True,
        "segment_index": 3,
        "expected_sha256": "c" * 64,
        "prompt": "clean",
    }
    js = APP_JS.read_text(encoding="utf-8")
    assert '"/image-optimization-prompt"' in js

    short = _run_contract(
        "contract.buildImagePromptPatch(contract.createImagePromptDraft("
        "'c-short',null,{text:'short',default_text:'default',sha256:'d'.repeat(64)}))"
    )
    assert short["segment_index"] == 0
    invalid_rejected = _run_contract(
        "(()=>{try{contract.buildImagePromptPatch(contract.createImagePromptDraft("
        "'c-bad',-1,{text:'bad',default_text:'bad',sha256:'e'.repeat(64)}));return false}"
        "catch(error){return error.message==='图片优化提示词段号无效'}})()"
    )
    assert invalid_rejected is True


def test_image_prompt_edit_requires_capability_and_valid_segment_index():
    result = _run_contract(
        "(()=>{const base={read_only:false,submit_enabled:true,generation:null,postprocess:null};"
        "return {disabled:contract.imagePromptEditable({...base,postprocess_capabilities:{optimize_image:false}},0),"
        "enabled:contract.imagePromptEditable({...base,postprocess_capabilities:{optimize_image:true}},0),"
        "short:contract.promptSegmentIndex(null),long:contract.promptSegmentIndex({index:2}),"
        "longZero:contract.promptSegmentIndex({index:0}),invalid:contract.promptSegmentIndex({index:-1})}})()"
    )
    assert result == {
        "disabled": False,
        "enabled": True,
        "short": 0,
        "long": 2,
        "longZero": None,
        "invalid": None,
    }


def test_image_prompt_save_gate_never_requests_when_disabled_or_long_index_invalid():
    result = _run_async_contract(
        "(async()=>{let requests=0;const request=async()=>{requests+=1;return {}};"
        "const base={id:'c1',read_only:false,submit_enabled:true,generation:null,postprocess:null,"
        "postprocess_capabilities:{optimize_image:false}};"
        "const draft=contract.createImagePromptDraft('c1',0,{text:'x',default_text:'d',sha256:'a'.repeat(64)});"
        "let disabled=false,invalid=false,mismatch=false;try{await contract.saveImageOptimizationPrompt(base,0,draft,request)}"
        "catch(error){disabled=error.message==='当前会话未开放图片优化编辑'}"
        "try{await contract.saveImageOptimizationPrompt({...base,postprocess_capabilities:{optimize_image:true}},"
        "null,{...draft,segmentIndex:null},request)}catch(error){invalid=error.message==='图片优化提示词段号无效'}"
        "try{await contract.saveImageOptimizationPrompt({...base,postprocess_capabilities:{optimize_image:true}},"
        "1,{...draft,segmentIndex:2},request)}catch(error){mismatch=error.message==='图片优化提示词段号已变化'}"
        "return {requests,disabled,invalid,mismatch}})()"
    )
    assert result == {"requests": 0, "disabled": True, "invalid": True, "mismatch": True}


def test_discarded_singleton_draft_is_replaced_when_switching_segments():
    result = _run_contract(
        "(()=>{let draft=contract.createImagePromptDraft('c1',1,{text:'one',default_text:'base',sha256:'a'.repeat(64)});"
        "contract.restoreImagePromptDefault(draft);draft.text=draft.savedText;draft.dirty=false;"
        "if(draft.conversationId!=='c1'||draft.segmentIndex!==2){draft=contract.createImagePromptDraft("
        "'c1',2,{text:'two',default_text:'base-two',sha256:'b'.repeat(64)})}"
        "return {segmentIndex:draft.segmentIndex,text:draft.text,dirty:draft.dirty}})()"
    )
    assert result == {"segmentIndex": 2, "text": "two", "dirty": False}
    js = APP_JS.read_text(encoding="utf-8")
    assert "draft.conversationId !== detail.id || draft.segmentIndex !== segmentIndex" in js
    assert "state.promptDraft = draft" in js


def test_dirty_guard_covers_navigation_generation_and_postprocess():
    js = APP_JS.read_text(encoding="utf-8")
    assert 'window.addEventListener("beforeunload"' in js
    assert "guardDirtyPrompt" in js
    select = js.split("function selectConversation", 1)[1].split("function startPolling", 1)[0]
    assert "guardDirtyPrompt" in select
    generation = js.split("async function submitGeneration", 1)[1].split("function render", 1)[0]
    assert "guardDirtyPrompt" in generation
    modal = js.split("async function requestOpenPostprocessModal", 1)[1].split(
        "function closePostprocessModal", 1
    )[0]
    assert "guardDirtyPrompt" in modal
    assert 'decision === "cancel"' in js and 'decision === "discard"' in js
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "保存" in html and "丢弃" in html and "取消" in html


def test_dirty_prompt_dialog_escape_resolves_as_cancel():
    result = _run_async_contract(
        "(async()=>{const buttons={draftSave:{},draftDiscard:{},draftCancel:{}};"
        "let prevented=false,closed=0,shown=0;const dialog={oncancel:null,"
        "showModal(){shown+=1},close(){closed+=1}};"
        "global.document={getElementById:(id)=>({"
        "'draft-dialog':dialog,'draft-save':buttons.draftSave,"
        "'draft-discard':buttons.draftDiscard,'draft-cancel':buttons.draftCancel}[id])};"
        "const pending=contract.dirtyPromptDecision();"
        "dialog.oncancel({preventDefault(){prevented=true}});"
        "return {decision:await pending,prevented,closed,shown}})()"
    )
    assert result == {"decision": "cancel", "prevented": True, "closed": 1, "shown": 1}


def test_postprocess_modal_uses_capabilities_and_image_default_off():
    js, html, _ = _sources()
    assert 'value="optimize_image"' in html
    assert 'value="optimize_image" checked' not in html
    assert "postprocess_capabilities" in js
    assert 'c.value !== "optimize_image"' in js
    assert "c.disabled = !capabilities[c.value]" in js
    result = _run_contract(
        "({defaultChoice:contract.postprocessAskDefault(),modes:contract.promptWorkspaceModes()})"
    )
    assert result == {
        "defaultChoice": "no",
        "modes": [
            ["generation", "展开生成提示词"],
            ["dialogue", "展开段台词"],
            ["image", "展开图片优化"],
        ],
    }


def test_segment_progress_retry_and_unknown_warning_are_rendered():
    js = APP_JS.read_text(encoding="utf-8")
    assert "function renderPostprocessSegments" in js
    assert "completed_frames" in js
    assert "total_frames" in js
    assert '"重试本段"' in js
    assert '"/postprocess/segments/"' in js
    assert "expected_revision" in js
    assert "submission_unknown" in js
    assert "可能重复计费" in js
    assert 'segment.status === "failed"' in js
    assert "window.confirm" in js


def test_unknown_segment_retry_cancel_never_requests():
    result = _run_async_contract(
        "(async()=>{let requests=0,accepted=0,confirmed=0;"
        "const retried=await contract.retryPostprocessSegment({id:'c1',duration_s:20,segments:[{index:2}]},"
        "{index:2,status:'submission_unknown',revision:7},"
        "async()=>{requests+=1},()=>{confirmed+=1;return false},()=>{accepted+=1});"
        "return {retried,requests,accepted,confirmed}})()"
    )
    assert result == {"retried": False, "requests": 0, "accepted": 0, "confirmed": 1}


def test_segment_retry_enforces_short_zero_and_long_positive_indexes():
    result = _run_async_contract(
        "(async()=>{let requests=0;const request=async()=>{requests+=1};"
        "const retry=(detail,index)=>contract.retryPostprocessSegment(detail,"
        "{index,status:'failed',revision:1},request,()=>true,()=>{});"
        "const short={id:'short',duration_s:8,segments:[]};"
        "const long={id:'long',duration_s:20,segments:[{index:1}]};"
        "return {shortZero:await retry(short,0),shortPositive:await retry(short,1),"
        "longPositive:await retry(long,1),longZero:await retry(long,0),requests}})()"
    )
    assert result == {
        "shortZero": True,
        "shortPositive": False,
        "longPositive": True,
        "longZero": False,
        "requests": 2,
    }


def test_retry_request_rejection_never_exposes_raw_error():
    result = _run_async_contract(
        "(async()=>{try{await contract.retryPostprocessSegment("
        "{id:'short',duration_s:8,segments:[]},{index:0,status:'failed',revision:1},"
        "async()=>{throw new Error('provider raw stack token=secret')},()=>true,()=>{});return null}"
        "catch(error){return {message:error.message,leaked:/secret|raw stack/.test(error.message)}}})()"
    )
    assert result == {"message": "本段处理失败，请重试或联系管理员", "leaked": False}
    js = APP_JS.read_text(encoding="utf-8")
    retry_dom = js.split("function renderPostprocessSegments", 1)[1].split(
        "/* 助手消息", 1
    )[0]
    assert 'el("p", "form-error", safePostprocessError(err))' in retry_dom
    assert "String(err.message" not in retry_dom


def test_postprocess_segment_stage_and_error_are_allowlisted():
    result = _run_contract(
        "({knownStage:contract.safePostprocessStage('optimize_image'),"
        "unknownStage:contract.safePostprocessStage('model-template-secret'),"
        "knownError:contract.safePostprocessError('revision_conflict'),"
        "unknownError:contract.safePostprocessError('provider raw stack token=secret')})"
    )
    assert result == {
        "knownStage": "优化图片质量",
        "unknownStage": "处理中",
        "knownError": "分段状态已更新，请刷新后重试",
        "unknownError": "本段处理失败，请重试或联系管理员",
    }
    prototype_keys = _run_contract(
        "['toString','constructor','__proto__'].map(key=>({"
        "status:contract.postprocessSegmentStatus(key),stage:contract.safePostprocessStage(key),"
        "error:contract.safePostprocessError(key)}))"
    )
    assert prototype_keys == [
        {
            "status": "状态未知",
            "stage": "处理中",
            "error": "本段处理失败，请重试或联系管理员",
        }
    ] * 3


def test_removed_segment_disclosure_helpers_are_not_kept_for_tests():
    js = APP_JS.read_text(encoding="utf-8")
    for dead in ("function segmentDisclosure", "function segmentPromptDisclosure", "function segmentDialogueDisclosure"):
        assert dead not in js

    assert "function renderSourcePromptCard" not in js


def test_ui_never_discloses_backend_execution_identifiers():
    source = "\n".join(_sources())
    for forbidden in ("template_id", "model_id", "execution_mode"):
        assert forbidden not in source
