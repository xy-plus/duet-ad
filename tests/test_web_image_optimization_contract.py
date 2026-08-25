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
    ask = js.split("function renderPpAsk", 1)[1].split("function renderPpChat", 1)[0]
    assert '"btn btn-primary pp-ask-btn is-selected", "否"' in ask


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


def test_removed_segment_disclosure_helpers_are_not_kept_for_tests():
    js = APP_JS.read_text(encoding="utf-8")
    for dead in ("function segmentDisclosure", "function segmentPromptDisclosure", "function segmentDialogueDisclosure"):
        assert dead not in js

    assert "function renderSourcePromptCard" not in js


def test_ui_never_discloses_backend_execution_identifiers():
    source = "\n".join(_sources())
    for forbidden in ("template_id", "model_id", "execution_mode"):
        assert forbidden not in source
