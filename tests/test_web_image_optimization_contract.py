from test_web_h3_contract import APP_JS, INDEX_HTML, STYLES_CSS, _run_contract


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
    assert "保存" in js and "丢弃" in js and "取消" in js


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


def test_ui_never_discloses_backend_execution_identifiers():
    source = "\n".join(_sources())
    for forbidden in ("template_id", "model_id", "execution_mode"):
        assert forbidden not in source
