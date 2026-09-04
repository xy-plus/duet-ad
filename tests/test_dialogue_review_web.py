import json
import re

from test_legacy_phase2_frame_viewer import _run_jsdom_contract
from test_web_h3_contract import APP_JS, _run_contract


INDEX_HTML = APP_JS.parent / "index.html"
STYLES = APP_JS.parent / "styles.css"


def _capability(**overrides):
    value = {
        "supported": True,
        "create_field": "dialogue_review_policy",
        "policies": ["auto_continue", "review_required"],
        "default": "auto_continue",
        "commit_path": "/api/conversations/{id}/dialogue-review/commit",
    }
    value.update(overrides)
    return {"dialogue_review": value}


def _minimal_capability(**overrides):
    value = {
        "supported": True,
        "version": 1,
        "endpoint": "/api/conversations",
        "encoding": "multipart/form-data",
        "request_field": "generation_request",
        "replacement_image_field": "replacement_image",
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["480p", "768p"],
        "defaults": {
            "fit_mode": "auto",
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {"mode": "auto_rewrite", "translation": True},
        "replacement": {
            "supported": True,
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": 10 * 1024 * 1024,
            "max_instruction_chars": 1000,
        },
    }
    value.update(overrides)
    return value


def _review(status="waiting", **overrides):
    value = {
        "version": 1,
        "policy": "review_required",
        "status": status,
        "outcome": "recognized",
        "revision": 1,
        "machine_lines": [
            {"text": "机器稿", "start_s": 0.2, "end_s": 1.1}
        ],
        "machine_sha256": "a" * 64,
        "lines": [{"text": "机器稿", "start_s": 0.2, "end_s": 1.1}],
        "sha256": "b" * 64,
        "frozen_by": None if status == "waiting" else "user",
        "editable": status == "waiting",
    }
    value.update(overrides)
    return value


def test_capability_is_exact_and_unknown_contract_never_enables_post():
    good = _capability()
    result = _run_contract(
        "(()=>{const good=" + json.dumps(good) + ";return {"
        "good:contract.normalizeDialogueReviewCapability(good),"
        "unsupported:contract.normalizeDialogueReviewCapability({dialogue_review:{...good.dialogue_review,supported:false}}),"
        "wrongField:contract.normalizeDialogueReviewCapability({dialogue_review:{...good.dialogue_review,create_field:'other'}}),"
        "extraPolicy:contract.normalizeDialogueReviewCapability({dialogue_review:{...good.dialogue_review,policies:[...good.dialogue_review.policies,'other']}})}})()"
    )
    assert result["good"] == good["dialogue_review"]
    assert result["unsupported"] is None
    assert result["wrongField"] is None
    assert result["extraPolicy"] is None


def test_legacy_create_policy_helper_is_safe_but_hidden_from_auto_rewrite_creation():
    capability = _capability()["dialogue_review"]
    result = _run_contract(
        "(()=>{const capability=" + json.dumps(capability) + ";return {"
        "auto:contract.buildDialogueReviewCreateField(capability,'auto_continue','auto'),"
        "review:contract.buildDialogueReviewCreateField(capability,'review_required','auto'),"
        "none:contract.buildDialogueReviewCreateField(capability,'review_required','none'),"
        "unsupported:contract.buildDialogueReviewCreateField({...capability,supported:false},'review_required','auto')}})()"
    )
    assert result["auto"] == {
        "name": "dialogue_review_policy",
        "value": "auto_continue",
    }
    assert result["review"] == {
        "name": "dialogue_review_policy",
        "value": "review_required",
    }
    assert result["none"] is None
    assert result["unsupported"] is None

    html = INDEX_HTML.read_text(encoding="utf-8")
    visible, legacy = html.split(
        '<div class="legacy-contract-controls" hidden aria-hidden="true">', 1
    )
    auto = legacy.split('name="dialogue-review-policy" value="auto_continue"', 1)[1]
    assert "checked" in auto.split(">", 1)[0]
    assert 'id="dialogue-review-fields"' not in visible
    assert "中途校对" not in visible

    dialogue = visible.split('aria-labelledby="dialogue-title"', 1)[1].split(
        "</section>", 1
    )[0]
    assert 'id="script-input"' not in html
    assert 'id="translation-toggle"' not in html
    assert 'name="target-language-mode" value="same" checked' in dialogue
    assert 'name="target-language-mode" value="other"' in dialogue
    assert "与原视频相同" in dialogue
    assert "其他" in dialogue
    translation_fields = re.search(
        r'<div\b(?P<attrs>[^>]*)id="translation-fields"(?P<tail>[^>]*)>',
        dialogue,
        flags=re.DOTALL,
    )
    assert translation_fields is not None
    fields_attrs = translation_fields.group("attrs") + translation_fields.group("tail")
    assert "hidden" in fields_attrs
    language_input = re.search(
        r'<input\b(?P<attrs>[^>]*)id="lang-input"(?P<tail>[^>]*)>',
        dialogue,
        flags=re.DOTALL,
    )
    assert language_input is not None
    language_attrs = language_input.group("attrs") + language_input.group("tail")
    assert "disabled" in language_attrs
    assert "required" not in language_attrs

    minimal = _minimal_capability()
    language_contract = _run_contract(
        "(()=>{const capability="
        + json.dumps(minimal)
        + ";const base={aspectRatio:'9:16',resolution:'768p',"
        "targetLanguage:contract.resolveTargetLanguage('same',''),"
        "hasReplacementImage:false,replacementInstruction:''};"
        "const request=contract.buildMinimalGenerationRequest(base,capability);"
        "const other=contract.buildMinimalGenerationRequest({...base,"
        "targetLanguage:contract.resolveTargetLanguage('other',' 日语 ')},capability);"
        "const injected=contract.buildMinimalGenerationRequest({...base,script:'用户预写台词'},capability);"
        "const v2={...capability,version:2};let v2Error=null;"
        "try{contract.buildMinimalGenerationRequest(base,v2)}catch(error){v2Error=error.message}"
        "return {request:{version:request.version,dialogue:request.dialogue},"
        "other:{version:other.version,dialogue:other.dialogue},"
        "injected:{version:injected.version,dialogue:injected.dialogue},"
        "normalizedV2:contract.normalizeMinimalCreationCapability({minimal_creation:v2}),v2Error}})()"
    )
    assert language_contract == {
        "request": {
            "version": 1,
            "dialogue": {
                "mode": "auto_rewrite",
                "target_language": "与原视频相同",
            },
        },
        "other": {
            "version": 1,
            "dialogue": {"mode": "auto_rewrite", "target_language": "日语"},
        },
        "injected": {
            "version": 1,
            "dialogue": {
                "mode": "auto_rewrite",
                "target_language": "与原视频相同",
            },
        },
        "normalizedV2": None,
        "v2Error": "生成服务尚未支持当前创建方式",
    }


def test_commit_payload_is_exact_cas_and_does_not_invent_asr_metadata():
    review = _review()
    result = _run_contract(
        "(()=>{const review=" + json.dumps(review) + ";return {"
        "payload:contract.buildDialogueReviewCommitPayload(review,review.lines,'dialogue-review-0001'),"
        "view:contract.dialogueReviewView({dialogue_review:review})}})()"
    )
    assert result["payload"] == {
        "confirm": True,
        "client_request_id": "dialogue-review-0001",
        "expected_revision": 1,
        "expected_sha256": "b" * 64,
        "lines": [{"text": "机器稿", "start_s": 0.2, "end_s": 1.1}],
    }
    assert set(result["view"]["lines"][0]) == {"text", "start_s", "end_s"}
    assert not ({"language", "speaker", "confidence"} & set(result["view"]["lines"][0]))


def test_local_line_validation_covers_empty_out_of_range_and_order():
    result = _run_contract(
        "(()=>{const run=(lines)=>{try{return contract.validateDialogueReviewDraft({lines},6)}catch(error){return error.message}};return ["
        "run([]),"
        "run([{text:'',start_s:0,end_s:1}]),"
        "run([{text:'x',start_s:1,end_s:7}]),"
        "run([{text:'x',start_s:2,end_s:3},{text:'y',start_s:1,end_s:2}])]"
        "})()"
    )
    assert result == [
        [],
        "第 1 行台词不能为空",
        "第 1 行超过视频时长 6.00 秒",
        "第 2 行开始时间早于上一行",
    ]


def test_commit_conflicts_have_recoverable_user_copy():
    result = _run_contract(
        "['dialogue_review_conflict','dialogue_review_read_only','dialogue_review_not_waiting',"
        "'invalid_dialogue_review_lines'].map(message=>contract.dialogueReviewCommitErrorMessage({message}))"
    )
    assert result == [
        "服务端台词稿已更新，请刷新后重新校对。",
        "台词稿已冻结，当前任务已不能修改。",
        "当前任务已不再等待台词校对，请刷新查看最新状态。",
        "台词或时间码不符合要求，请逐行检查。",
    ]


def test_asr_operational_failure_is_distinct_from_valid_empty_outcomes():
    result = _run_contract(
        "({failure:contract.safeErrorSummary('codex voice output invalid: missing artifact'),"
        "noAudio:contract.dialogueReviewOutcomeText({outcome:'no_audio',lines:[]}),"
        "noVocal:contract.dialogueReviewOutcomeText({outcome:'no_vocal',lines:[]}),"
        "emptyVoice:contract.dialogueReviewOutcomeText({outcome:'vocal_unrecognized',lines:[]})})"
    )
    assert result == {
        "failure": "台词识别失败，请检查原视频音轨后重试",
        "noAudio": "未检测到音轨。你可以补充台词，或采用空稿按无台词继续。",
        "noVocal": "未检测到可信口播。你可以补充台词，或采用空稿继续。",
        "emptyVoice": "检测到人声，但未识别出可靠台词。请补充台词，或采用空稿继续。",
    }


def test_waiting_and_frozen_dom_have_one_clear_action_and_read_only_boundary():
    waiting = _review()
    frozen = _review(
        status="frozen",
        revision=2,
        sha256="c" * 64,
        frozen_by="user",
        editable=False,
    )
    result = _run_jsdom_contract(
        "(()=>{const waiting=" + json.dumps(waiting) + ";const frozen=" + json.dumps(frozen) + ";"
        "const editable=contract.renderDialogueReview({id:'cid-wait',duration_s:6,dialogue_review:waiting});"
        "const readonly=contract.renderDialogueReview({id:'cid-frozen',duration_s:6,dialogue_review:frozen});"
        "return {editableText:editable.textContent,rows:editable.querySelectorAll('.dialogue-review-line').length,"
        "submit:editable.querySelector('button[type=submit]').textContent,"
        "readonlyText:readonly.textContent,readonlyInputs:readonly.querySelectorAll('input').length}})()"
    )
    assert result["rows"] == 1
    assert result["submit"] == "采用此稿并继续"
    assert "等待你确认" in result["editableText"]
    assert "下游已开始后，本稿不可修改" in result["readonlyText"]
    assert result["readonlyInputs"] == 0


def test_legacy_timeline_helper_keeps_dialogue_wait_safety():
    review = _review()
    result = _run_contract(
        "(()=>{const model=contract.operationTimeline({id:'cid',status:'processing',has_source:true,"
        "created_at:'2026-08-30T00:00:00Z',updated_at:'2026-08-30T00:01:00Z',dialogue_review:"
        + json.dumps(review)
        + "},Date.parse('2026-08-30T00:02:00Z'));return {count:model.stages.length,current:model.current}})()"
    )
    assert result["count"] == 10
    assert result["current"]["key"] == "dialogue-review"
    assert result["current"]["status"] == "attention"
    assert "等待你校对" in result["current"]["detail"]

    css = STYLES.read_text(encoding="utf-8")
    assert ".dialogue-review-line" in css
    mobile = css.split("@media (max-width: 768px)", 1)[1]
    assert ".dialogue-review-policy { grid-template-columns: 1fr; }" in mobile


def test_composer_drawer_toggle_is_accessible_and_mobile_safe():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="composer-toggle"' in html
    assert 'aria-controls="composer-panel"' in html
    assert 'aria-expanded="true"' in html
    assert 'aria-label="收起创建抽屉"' in html

    source = APP_JS.read_text(encoding="utf-8")
    assert 'panel.hidden = !next' in source
    assert 'setComposerExpanded(true)' in source

    css = STYLES.read_text(encoding="utf-8")
    assert ".composer-dock.is-collapsed" in css
    assert "min-height: 32px" in css


def test_composer_drawer_dom_collapses_only_its_panel():

    result = _run_jsdom_contract(
        "(()=>{document.body.innerHTML='<div class=\"composer-dock\">' +"
        "'<button id=\"composer-toggle\" aria-controls=\"composer-panel\" aria-expanded=\"true\">' +"
        "'<span id=\"composer-toggle-label\">收起</span></button>' +"
        "'<div id=\"composer-panel\">内容</div></div>';"
        "const collapsed=contract.setComposerExpanded(false);"
        "const afterCollapse={collapsed,hidden:document.getElementById('composer-panel').hidden,"
        "className:document.querySelector('.composer-dock').className,"
        "expanded:document.getElementById('composer-toggle').getAttribute('aria-expanded'),"
        "label:document.getElementById('composer-toggle').getAttribute('aria-label'),"
        "text:document.getElementById('composer-toggle-label').textContent};"
        "const expanded=contract.setComposerExpanded(true);"
        "return {afterCollapse,afterExpand:{expanded,hidden:document.getElementById('composer-panel').hidden,"
        "className:document.querySelector('.composer-dock').className,"
        "aria:document.getElementById('composer-toggle').getAttribute('aria-expanded'),"
        "label:document.getElementById('composer-toggle').getAttribute('aria-label')}}})()"
    )
    assert result == {
        "afterCollapse": {
            "collapsed": False,
            "hidden": True,
            "className": "composer-dock is-collapsed",
            "expanded": "false",
            "label": "展开创建抽屉",
            "text": "展开",
        },
        "afterExpand": {
            "expanded": True,
            "hidden": False,
            "className": "composer-dock",
            "aria": "true",
            "label": "收起创建抽屉",
        },
    }
