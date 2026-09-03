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
    postprocess_form = html.split('id="pp-form"', 1)[1].split("</form>", 1)[0]
    assert 'value="optimize_image"' in postprocess_form
    assert 'value="optimize_image" checked' not in postprocess_form
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
        "{index:2,status:'failed',stage:'seedream',error:'submission_unknown',revision:7},"
        "async()=>{requests+=1},()=>{confirmed+=1;return false},()=>{accepted+=1});"
        "return {retried,requests,accepted,confirmed}})()"
    )
    assert result == {"retried": False, "requests": 0, "accepted": 0, "confirmed": 1}


def test_submission_unknown_is_derived_only_from_segment_error():
    result = _run_contract(
        "["
        "contract.isPostprocessSubmissionUnknown({status:'failed',stage:'seedream',error:'submission_unknown'}),"
        "contract.isPostprocessSubmissionUnknown({status:'submission_unknown',stage:'seedream',error:'failed'}),"
        "contract.isPostprocessSubmissionUnknown({status:'failed',stage:'submission_unknown',error:'failed'})"
        "]"
    )
    assert result == [True, False, False]
    js = APP_JS.read_text(encoding="utf-8")
    postprocess = js.split("function postprocessSegmentStatus", 1)[1].split(
        "function renderPpChat", 1
    )[0]
    assert 'segment.status === "submission_unknown"' not in postprocess


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
        "({queued:contract.safePostprocessStage('queued'),text:contract.safePostprocessStage('text'),"
        "brand:contract.safePostprocessStage('brand'),knownStage:contract.safePostprocessStage('seedream'),"
        "publishing:contract.safePostprocessStage('publishing'),done:contract.safePostprocessStage('done'),"
        "unknownStage:contract.safePostprocessStage('model-template-secret'),"
        "knownError:contract.safePostprocessError('revision_conflict'),"
        "unknownError:contract.safePostprocessError('provider raw stack token=secret')})"
    )
    assert result == {
        "queued": "等待处理",
        "text": "移除文字/字幕",
        "brand": "移除常见 Logo/图标",
        "knownStage": "优化图片质量",
        "publishing": "正在发布结果",
        "done": "已完成",
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


def test_failed_postprocess_never_reopens_whole_project_post():
    base = (
        "{id:'c1',read_only:false,submit_enabled:true,postprocess_capabilities:{optimize_image:true},"
    )
    result = _run_contract(
        "["
        f"contract.shouldRenderPostprocessAsk({base}postprocess:null}}),"
        f"contract.shouldRenderPostprocessAsk({base}postprocess:{{status:'failed',segments:[{{index:1}}]}}}}),"
        f"contract.shouldRenderPostprocessAsk({base}postprocess:{{status:'failed',segments:[]}}}})"
        "]"
    )
    assert result == [True, False, False]
    js = APP_JS.read_text(encoding="utf-8")
    assert "分段状态不完整，请刷新页面后重试" in js
    ask = js.split("function renderPpAsk", 1)[1].split("function renderPpChat", 1)[0]
    assert "shouldRenderPostprocessAsk(detail)" in ask
    assert 'pp.status === "failed"' not in ask


def test_generation_waits_for_every_postprocess_segment_done():
    result = _run_contract(
        "(()=>{const base={id:'c1',duration_s:20,segment_count:2,plan_receipt:'a'.repeat(64),"
        "segments:[{index:1},{index:2}]};"
        "const seg=(index,status,completed=1,total=1,stage='done',error=null)=>({index,status,revision:1,"
        "completed_frames:completed,total_frames:total,stage,error});"
        "const pp=(status,segments)=>({...base,postprocess:{status,segments}});return {"
        "partial:contract.postprocessReadyForGeneration(pp('done',["
        "seg(1,'done'),seg(2,'running',0,1)])),"
        "failed:contract.postprocessReadyForGeneration(pp('failed',["
        "seg(1,'done'),seg(2,'failed',0,1)])),"
        "allDone:contract.postprocessReadyForGeneration(pp('done',["
        "seg(1,'done'),seg(2,'done')])),"
        "absent:contract.postprocessReadyForGeneration({...base,postprocess:null})}})()"
    )
    assert result == {"partial": False, "failed": False, "allDone": True, "absent": True}


def test_live_image_progress_uses_receipt_projected_segment_counts():
    result = _run_contract(
        "(()=>{const states=[];for(let completed=0;completed<=9;completed+=1){"
        "states.push(contract.ppCompletedFrames({postprocess:{frames:[],segments:["
        "{index:0,status:'running',stage:'seedream',completed_frames:completed,"
        "total_frames:9,revision:1,error:null}]}}));}return states})()"
    )
    assert result == list(range(10))


def test_detail_polling_uses_authoritative_project_progress_and_stops_terminal():
    result = _run_contract(
        "(()=>({queued:contract.shouldPollDetail({project_progress:{percent:0,status:'queued'}}),"
        "running:contract.shouldPollDetail({project_progress:{percent:42,status:'running'}}),"
        "done:contract.shouldPollDetail({project_progress:{percent:100,status:'succeeded'}}),"
        "failed:contract.shouldPollDetail({project_progress:{percent:42,status:'failed'}}),"
        "missing:contract.shouldPollDetail({status:'processing'})}))()"
    )
    assert result == {
        "queued": True,
        "running": True,
        "done": False,
        "failed": False,
        "missing": False,
    }


def test_blocked_postprocess_never_calls_generation_submit_request():
    result = _run_async_contract(
        "(async()=>{let requests=0;const request=async()=>{requests+=1;return {ok:true}};"
        "const base={id:'c1',duration_s:20,segment_count:2,plan_receipt:'a'.repeat(64),"
        "segments:[{index:1},{index:2}]};"
        "const blocked={...base,postprocess:{status:'done',segments:["
        "{index:1,status:'done',revision:1,completed_frames:1,total_frames:1},"
        "{index:2,status:'running',revision:1,completed_frames:0,total_frames:1,stage:'queued',error:null}]}};"
        "let rejected=false;try{await contract.requestGenerationSubmit(blocked,{confirm:true},request)}"
        "catch(error){rejected=error.message==='素材优化尚未全部完成，不能生成最终视频'}"
        "const allowed=await contract.requestGenerationSubmit({...base,postprocess:null},{confirm:true},request);"
        "return {requests,rejected,allowed}})()"
    )
    assert result == {"requests": 1, "rejected": True, "allowed": {"ok": True}}
    js = APP_JS.read_text(encoding="utf-8")
    submit = js.split("async function submitGeneration", 1)[1].split(
        "function generationStageText", 1
    )[0]
    assert "postprocessReadyForGeneration(detail)" in submit
    post = js.split("async function postGeneration", 1)[1].split(
        "/* 最终视频区", 1
    )[0]
    assert "requestGenerationSubmit(detail, body)" in post
    render = js.split("function renderFinalSection", 1)[1].split(
        "function kfGrid", 1
    )[0]
    assert "postprocessReadyForGeneration(detail)" in render


def test_generation_gate_rejects_non_contiguous_and_invalid_public_segment_fields():
    result = _run_contract(
        "(()=>{const seg=(index,revision=1,completed=1,total=1,stage='done',error=null)=>({index,status:'done',revision,"
        "completed_frames:completed,total_frames:total,stage,error});"
        "const ready=(detailSegments,ppSegments)=>contract.postprocessReadyForGeneration({"
        "id:'c1',duration_s:20,segment_count:detailSegments.length,plan_receipt:'a'.repeat(64),"
        "segments:detailSegments,postprocess:{status:'done',segments:ppSegments}});"
        "return {gap:ready([{index:1},{index:3}],[seg(1),seg(3)]),"
        "badRevision:ready([{index:1}],[seg(1,0)]),"
        "overflow:ready([{index:1}],[seg(1,1,2,1)]),"
        "negative:ready([{index:1}],[seg(1,1,-1,1)]),"
        "missing:ready([{index:1}],[{index:1,status:'done'}])}})()"
    )
    assert result == {
        "gap": False,
        "badRevision": False,
        "overflow": False,
        "negative": False,
        "missing": False,
    }


def test_generation_gate_requires_complete_done_terminal_fields_and_zero_requests():
    result = _run_async_contract(
        "(async()=>{let requests=0;const request=async()=>{requests+=1};"
        "const detail=(segment)=>({id:'short',duration_s:8,segments:[],"
        "postprocess:{status:'done',segments:[segment]}});"
        "const seg=(completed,total,stage='done',error=null)=>({index:0,status:'done',revision:1,"
        "completed_frames:completed,total_frames:total,stage,error});"
        "const cases=[seg(0,1),seg(0,0),seg(1,1,'publishing'),seg(1,1,'done','submission_unknown')];"
        "const ready=cases.map(value=>contract.postprocessReadyForGeneration(detail(value)));"
        "for(const value of cases){try{await contract.requestGenerationSubmit(detail(value),{confirm:true},request)}"
        "catch(_){}}return {ready,requests}})()"
    )
    assert result == {"ready": [False, False, False, False], "requests": 0}


def test_generation_gate_rejects_segment_count_mismatch_without_request():
    result = _run_async_contract(
        "(async()=>{let requests=0;const request=async()=>{requests+=1};"
        "const seg=index=>({index,status:'done',revision:1,completed_frames:1,total_frames:1,"
        "stage:'done',error:null});"
        "const detail={id:'long',duration_s:20,segment_count:3,plan_receipt:'a'.repeat(64),"
        "segments:[{index:1},{index:2}],postprocess:{status:'done',segments:[seg(1),seg(2)]}};"
        "const ready=contract.postprocessReadyForGeneration(detail);"
        "try{await contract.requestGenerationSubmit(detail,{confirm:true},request)}catch(_){}"
        "return {ready,requests}})()"
    )
    assert result == {"ready": False, "requests": 0}


def test_short_segment_zero_uses_current_video_copy():
    js = APP_JS.read_text(encoding="utf-8")
    segments = js.split("function renderPostprocessSegments", 1)[1].split(
        "/* 助手消息", 1
    )[0]
    assert 'segment.index === 0 ? "当前视频"' in segments


def test_removed_segment_disclosure_helpers_are_not_kept_for_tests():
    js = APP_JS.read_text(encoding="utf-8")
    for dead in ("function segmentDisclosure", "function segmentPromptDisclosure", "function segmentDialogueDisclosure"):
        assert dead not in js

    assert "function renderSourcePromptCard" not in js


def test_ui_never_discloses_backend_execution_identifiers():
    source = "\n".join(_sources())
    for forbidden in ("template_id", "model_id", "execution_mode"):
        assert forbidden not in source
