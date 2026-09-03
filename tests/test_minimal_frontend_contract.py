import json
import re

from test_web_h3_contract import APP_JS, _run_contract


INDEX_HTML = APP_JS.parent / "index.html"
STYLES_CSS = APP_JS.parent / "styles.css"


def _capability(**overrides):
    capability = {
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
        "dialogue": {
            "mode": "auto_rewrite",
            "translation": True,
        },
        "replacement": {
            "supported": True,
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": 10 * 1024 * 1024,
            "max_instruction_chars": 1000,
        },
    }
    capability.update(overrides)
    return capability


def _function_source(source, name):
    marker = f"function {name}("
    assert marker in source
    return marker + source.split(marker, 1)[1].split("\nfunction ", 1)[0]


def _tag_attrs(source, tag, element_id):
    match = re.search(
        rf'<{tag}\b(?P<before>[^>]*)\bid="{re.escape(element_id)}"(?P<after>[^>]*)>',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("before") + match.group("after")


def _source_mode_attrs(source, value):
    match = re.search(
        rf'<input\b(?=[^>]*\bname="source-mode")(?=[^>]*\bvalue="{re.escape(value)}")(?P<attrs>[^>]*)>',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("attrs")


def _target_language_mode_attrs(source, value):
    match = re.search(
        rf'<input\b(?=[^>]*\bname="target-language-mode")(?=[^>]*\bvalue="{re.escape(value)}")(?P<attrs>[^>]*)>',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("attrs")


def _css_rule_bodies(source, selector):
    return re.findall(rf'{re.escape(selector)}\s*\{{(?P<body>[^}}]*)\}}', source)


def test_creation_form_defaults_to_link_and_same_source_language():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    composer = html.split('<form id="composer"', 1)[1].split("</form>", 1)[0]
    visible, legacy = composer.split(
        '<div class="legacy-contract-controls" hidden aria-hidden="true">', 1
    )

    source_section = visible.split(
        '<section class="setup-section source-section"', 1
    )[1].split("</section>", 1)[0]
    upload_attrs = _source_mode_attrs(source_section, "upload")
    link_attrs = _source_mode_attrs(source_section, "link")
    assert re.search(r"\bchecked\b", upload_attrs) is None
    assert re.search(r"\bchecked\b", link_attrs) is not None
    assert "hidden" in _tag_attrs(source_section, "button", "attach-btn")
    assert "hidden" not in _tag_attrs(source_section, "div", "url-row")
    assert "required" in _tag_attrs(source_section, "input", "url-input")

    assert '<h2 id="dialogue-title">改编台词</h2>' in visible
    dialogue = visible.split('aria-labelledby="dialogue-title"', 1)[1].split(
        "</section>", 1
    )[0]
    assert "自动识别并改编台词" in dialogue
    assert 'id="script-input"' not in html
    assert "script-input" not in source
    assert re.findall(r'<textarea\b[^>]*\bid="([^"]+)"', visible) == [
        "replacement-instruction"
    ]
    assert 'name="voice-mode"' not in visible
    assert 'name="dialogue-mode"' not in visible
    assert 'id="dialogue-review-fields"' not in visible
    assert 'id="generation-config"' not in visible
    assert "后处理" not in visible

    assert 'id="translation-toggle"' not in html
    assert '<fieldset class="language-fieldset">' in dialogue
    assert "最终视频语种" in dialogue
    assert "与原视频相同" in dialogue
    assert "其他" in dialogue
    same_attrs = _target_language_mode_attrs(dialogue, "same")
    other_attrs = _target_language_mode_attrs(dialogue, "other")
    assert re.search(r"\bchecked\b", same_attrs) is not None
    assert re.search(r"\bchecked\b", other_attrs) is None

    fields = re.search(
        r'<div\b(?P<attrs>[^>]*)id="translation-fields"(?P<tail>[^>]*)>',
        visible,
        flags=re.DOTALL,
    )
    assert fields is not None
    fields_attrs = fields.group("attrs") + fields.group("tail")
    assert 'class="translation-fields"' in fields_attrs
    assert "hidden" in fields_attrs
    language_attrs = _tag_attrs(visible, "input", "lang-input")
    assert 'maxlength="80"' in language_attrs
    assert "disabled" in language_attrs
    assert "required" not in language_attrs
    assert "translationEnabled" not in source
    assert "setTranslationEnabled" not in source

    target_language_mode = _function_source(source, "targetLanguageMode")
    assert 'input[name="target-language-mode"]:checked' in target_language_mode
    assert 'return checked ? checked.value : "same"' in target_language_mode
    sync_controls = _function_source(source, "syncTargetLanguageControls")
    assert 'targetLanguageMode() === "other"' in sync_controls
    assert '$("translation-fields").hidden = !isOther' in sync_controls
    assert '$("lang-input").required = isOther' in sync_controls
    assert (
        '$("lang-input").disabled = state.uploading || state.viewingSubmittedConfig || !isOther'
        in sync_controls
    )
    bind_events = _function_source(source, "bindEvents")
    assert 'input[name="target-language-mode"]' in bind_events
    assert "syncTargetLanguageControls();" in bind_events

    source_mode = _function_source(source, "sourceMode")
    assert 'return checked ? checked.value : "link"' in source_mode

    # Compatibility-only inputs may remain in the DOM, but never become creation UI.
    assert 'name="voice-mode"' in legacy
    assert 'name="dialogue-mode"' in legacy
    pp_dialog = re.search(r'<dialog\b(?P<attrs>[^>]*)id="pp-dialog"(?P<tail>[^>]*)>', html)
    assert pp_dialog is not None
    pp_attrs = pp_dialog.group("attrs") + pp_dialog.group("tail")
    assert "hidden" in pp_attrs
    assert 'aria-hidden="true"' in pp_attrs


def test_minimal_creation_capability_is_exact_and_fails_closed():
    good = _capability()
    result = _run_contract(
        "(()=>{const good="
        + json.dumps(good)
        + ";return {"
        "good:contract.normalizeMinimalCreationCapability({minimal_creation:good}),"
        "missing:contract.normalizeMinimalCreationCapability({}),"
        "unsupported:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,supported:false}}),"
        "wrongVersion:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,version:2}}),"
        "wrongField:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,request_field:'other'}}),"
        "defaultDrift:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,defaults:{...good.defaults,remove_logo:false}}}),"
        "extraDefault:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,defaults:{...good.defaults,extra:true}}}),"
        "dialogueDrift:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,dialogue:{...good.dialogue,translation:false}}}),"
        "dialogueExtra:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,dialogue:{...good.dialogue,max_target_language_chars:80}}}),"
        "replacementDrift:contract.normalizeMinimalCreationCapability({minimal_creation:{...good,replacement:{...good.replacement,accept:['image/png']}}})"
        "}})()"
    )

    assert result["good"] == {
        "endpoint": "/api/conversations",
        "request_field": "generation_request",
        "replacement_image_field": "replacement_image",
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["480p", "768p"],
        "dialogue": {
            "mode": "auto_rewrite",
            "translation": True,
        },
        "replacement": {
            "accept": ["image/jpeg", "image/png", "image/webp"],
            "max_bytes": 10 * 1024 * 1024,
            "max_instruction_chars": 1000,
        },
    }
    assert all(value is None for key, value in result.items() if key != "good")


def test_minimal_request_has_exact_auto_rewrite_and_replacement_shapes():
    capability = _capability()
    result = _run_contract(
        "(()=>{const capability="
        + json.dumps(capability)
        + ";const base={aspectRatio:'9:16',resolution:'768p',"
        "targetLanguage:' 日语 ',hasReplacementImage:false,replacementInstruction:''};"
        "return {same:contract.buildMinimalGenerationRequest({...base,"
        "targetLanguage:contract.resolveTargetLanguage('same','不会提交')},capability),"
        "resolved:{same:contract.resolveTargetLanguage('same','英语'),"
        "other:contract.resolveTargetLanguage('other',' 日语 '),"
        "unknown:contract.resolveTargetLanguage('unknown','日语')},"
        "request:contract.buildMinimalGenerationRequest(base,capability),"
        "injected:contract.buildMinimalGenerationRequest({...base,script:'用户注入台词'},capability),"
        "replacement:contract.buildMinimalGenerationRequest({...base,aspectRatio:'16:9',resolution:'480p',"
        "hasReplacementImage:true,replacementInstruction:' 替换成参考产品 '},capability)}})()"
    )

    request = {
        "version": 1,
        "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        },
        "processing": {
            "optimize_image": True,
            "remove_subtitle": True,
            "remove_logo": True,
        },
        "dialogue": {
            "mode": "auto_rewrite",
            "target_language": "日语",
        },
        "replacement_guidance": None,
    }
    same_request = json.loads(json.dumps(request))
    same_request["dialogue"]["target_language"] = "与原视频相同"
    assert result["same"] == same_request
    assert result["resolved"] == {
        "same": "与原视频相同",
        "other": "日语",
        "unknown": "",
    }
    assert result["request"] == request
    assert result["injected"] == request

    replacement = json.loads(json.dumps(request))
    replacement["output"] = {
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "fit_mode": "auto",
    }
    replacement["replacement_guidance"] = {
        "instruction": "替换成参考产品",
        "image_field": "replacement_image",
    }
    assert result["replacement"] == replacement
    assert "script" not in json.dumps(result)

    request_builder = _function_source(
        APP_JS.read_text(encoding="utf-8"), "buildMinimalGenerationRequest"
    )
    assert "script" not in request_builder.lower()


def test_minimal_request_rejects_missing_or_long_language_and_replacement_pair():
    capability = _capability()
    result = _run_contract(
        "(()=>{const capability="
        + json.dumps(capability)
        + ";const base={aspectRatio:'9:16',resolution:'768p',"
        "targetLanguage:'日语',hasReplacementImage:false,replacementInstruction:''};"
        "const capture=(value)=>{try{contract.buildMinimalGenerationRequest(value,capability);return null}"
        "catch(error){return error.message}};return {"
        "language:capture({...base,targetLanguage:'  '}),"
        "longLanguage:capture({...base,targetLanguage:'a'.repeat(81)}),"
        "imageOnly:capture({...base,hasReplacementImage:true}),"
        "instructionOnly:capture({...base,replacementInstruction:'替换产品'})}})()"
    )
    assert result == {
        "language": "请填写目标语言",
        "longLanguage": "目标语言超过长度限制",
        "imageOnly": "参考图与替换说明需要一起提供",
        "instructionOnly": "参考图与替换说明需要一起提供",
    }


def test_active_upload_appends_only_the_minimal_multipart_contract():
    source = APP_JS.read_text(encoding="utf-8")
    upload = _function_source(source, "uploadConversation")
    append_arguments = re.findall(r"fd\.append\((.+)\);", upload)

    assert append_arguments == [
        '"file", file, file.name',
        '"reference_url", url',
        '"client_request_id", requestId',
        "capability.request_field, JSON.stringify(generationRequest)",
        "capability.replacement_image_field, replacementImage, replacementImage.name",
    ]
    assert "if (file)" in upload
    assert "else if (url)" in upload
    assert "if (replacementImage)" in upload
    assert upload.count('xhr.open("POST", capability.endpoint)') == 1
    assert upload.count("xhr.send(fd)") == 1
    for legacy_field in (
        '"voice_mode"',
        '"dialogue_mode"',
        '"dialogue_review_policy"',
        '"generation_config"',
    ):
        assert legacy_field not in upload

    # Structured 429 errors use the same detail.code/detail.message parser as
    # every other v1 create failure; they are not stringified as [object Object].
    assert 'xhr.status === 429' not in upload
    parsed = _run_contract(
        "(()=>{const error=contract.apiErrorFromPayload("
        "{detail:{code:'queue_full',message:'当前队列已满'}},'fallback');"
        "return {code:error.code,message:error.message}})()"
    )
    assert parsed == {"code": "queue_full", "message": "当前队列已满"}


def test_creation_form_has_no_internal_vertical_scroll_or_height_cap():
    styles = (APP_JS.parent / "styles.css").read_text(encoding="utf-8")
    for selector in (".composer-dock", ".composer-inner", ".minimal-composer"):
        rule_bodies = _css_rule_bodies(styles, selector)
        assert rule_bodies, selector
        for body in rule_bodies:
            assert "max-height" not in body
            assert re.search(r"\boverflow(?:-y)?\s*:", body) is None


def test_project_progress_is_authoritative_for_polling_and_success_rendering():
    result = _run_contract(
        "(()=>({"
        "queued:contract.shouldPollDetail({status:'done',project_progress:{percent:0,status:'queued'}}),"
        "running:contract.shouldPollDetail({status:'failed',project_progress:{percent:47,status:'running'}}),"
        "succeeded:contract.shouldPollDetail({status:'processing',has_video:true,project_progress:{percent:100,status:'succeeded'}}),"
        "failed:contract.shouldPollDetail({status:'processing',project_progress:{percent:47,status:'failed'}}),"
        "missing:contract.shouldPollDetail({status:'processing'})"
        "}))()"
    )
    assert result == {
        "queued": True,
        "running": True,
        "succeeded": False,
        "failed": False,
        "missing": False,
    }

    render_stable = _function_source(
        APP_JS.read_text(encoding="utf-8"), "renderStable"
    )
    assert 'progress.state === "succeeded" && detail.has_video === true' in render_stable
    assert 'detail.status === "done"' not in render_stable


def test_source_video_stays_stable_while_progress_percent_changes():
    result = _run_contract(
        "(()=>{const base={id:'v1',title:'demo',status:'processing',has_source:true,has_video:false,"
        "project_progress:{percent:12,status:'running'},error:null};"
        "const original=contract.detailSignature(base).stable;return {"
        "progressPercent:contract.detailSignature({...base,project_progress:{percent:13,status:'running'}}).stable!==original,"
        "progressState:contract.detailSignature({...base,project_progress:{percent:12,status:'failed'}}).stable!==original,"
        "source:contract.detailSignature({...base,has_source:false}).stable!==original,"
        "error:contract.detailSignature({...base,error:{code:'project_failed',message:'生成失败'}}).stable!==original"
        "}})()"
    )
    assert result == {
        "progressPercent": False,
        "progressState": True,
        "source": True,
        "error": True,
    }


def test_project_progress_model_exposes_only_product_level_fields():
    result = _run_contract(
        "(()=>{const runningNow=new Date('2026-09-03T08:05:06').getTime();"
        "const terminalNow=new Date('2026-09-03T09:00:00').getTime();"
        "const authoritative=contract.projectProgressModel({title:'夏日项目',"
        "created_at:'2026-09-03T08:00:00',updated_at:'2026-09-03T08:04:00',"
        "project_progress:{percent:63,status:'running',stage:'h3',segments:[1,2]},"
        "generation:{stage:'h3',segments:[{status:'succeeded'}]}},runningNow);"
        "const complete=contract.projectProgressModel({has_video:true,status:'done',"
        "created_at:'2026-09-03T08:00:00',updated_at:'2026-09-03T08:12:34',"
        "project_progress:{percent:100,status:'succeeded'},"
        "generation:{stage:'stitch',segments:[{status:'succeeded'}]}},terminalNow);"
        "const missingV1=contract.projectProgressModel({title:'合同缺失',"
        "effective_request:{version:1},generation:{status:'running',stage:'h3'}},terminalNow);"
        "return {authoritative,complete,missingV1}})()"
    )
    assert result == {
        "authoritative": {
            "title": "夏日项目", "percent": 63, "state": "running",
            "startedAt": "2026-09-03 08:00:00",
            "elapsedLabel": "当前耗时", "elapsed": "5分 06秒",
            "phase": "AI 正在生成视频", "loading": True,
        },
        "complete": {
            "title": "未命名项目", "percent": 100, "state": "succeeded",
            "startedAt": "2026-09-03 08:00:00",
            "elapsedLabel": "总耗时", "elapsed": "12分 34秒",
            "phase": "视频生成完成", "loading": False,
        },
        "missingV1": {
            "title": "合同缺失", "percent": 0, "state": "failed",
            "startedAt": "未知", "elapsedLabel": "总耗时", "elapsed": "未知",
            "phase": "视频生成未完成", "loading": False,
        },
    }
    for model in result.values():
        assert set(model) == {
            "title", "percent", "state", "startedAt", "elapsedLabel", "elapsed",
            "phase", "loading",
        }
        assert "stage" not in model
        assert "segments" not in model


def test_progress_and_results_renderers_do_not_mount_technical_views():
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLES_CSS.read_text(encoding="utf-8")
    progress_renderer = _function_source(source, "renderOperationHeader")
    assert "projectProgressModel(detail)" in progress_renderer
    created_nodes = re.findall(
        r'el\("([^"]+)",\s*"([^"]+)"', progress_renderer
    )
    assert created_nodes == [
        ("div", "project-progress-header"),
        ("div", "project-progress-fields"),
        ("span", "project-progress-field project-progress-project"),
        ("span", "project-progress-label"),
        ("strong", "project-progress-title"),
        ("span", "project-progress-field"),
        ("span", "project-progress-label"),
        ("span", "project-progress-time"),
        ("span", "project-progress-field"),
        ("span", "project-progress-label"),
        ("span", "project-progress-time"),
        ("div", "project-phase"),
        ("span", "project-phase-spinner"),
        ("span", "project-phase-label"),
    ]
    for label in ('"项目"', '"发起时间"', "model.elapsedLabel", "model.elapsed"):
        assert label in progress_renderer
    assert progress_renderer.count('setAttribute("role", "status")') == 1
    assert '"project-progress-percent"' not in progress_renderer
    assert '"project-progress-track"' not in progress_renderer
    assert '"project-progress-fill"' not in progress_renderer
    assert "operation-timeline" not in progress_renderer
    assert "operationTimeline(" not in progress_renderer
    assert "model.stages" not in progress_renderer
    assert 'el("ol"' not in progress_renderer
    assert 'el("li"' not in progress_renderer
    assert "stage" not in progress_renderer.lower()
    assert "segment" not in progress_renderer.lower()
    for removed_progress_ui in (
        ".project-progress-percent", ".project-progress-track", ".project-progress-fill"
    ):
        assert removed_progress_ui not in css
    assert ".project-phase-spinner" in css
    assert "animation: project-phase-spin" in css

    user_bubble = _function_source(source, "renderUserBubble")
    assert "detail.has_source === true" in user_bubble
    assert 'videoPlayer(detail, "source.mp4")' in user_bubble
    assert 'preview.setAttribute("aria-label", "原视频")' in user_bubble

    results_renderer = _function_source(source, "renderResults")
    mounted_results = re.findall(
        r"frag\.appendChild\((.*?)\);", results_renderer, flags=re.DOTALL
    )
    assert mounted_results == [
        'videoSection(detail, "generated.mp4", "新视频", "已完成")'
    ]
    assert results_renderer.count("videoSection(") == 1

    result_renderers = "\n".join(
        _function_source(source, name)
        for name in (
            "renderResults",
            "renderStable",
            "renderPpDynamic",
            "renderGenerationDynamic",
        )
    )
    for technical_view in (
        "renderIntermediateStages(",
        "renderPpAsk(",
        "renderPpChat(",
    ):
        assert technical_view not in result_renderers


def test_three_public_progress_phases_follow_the_deepest_started_work():
    result = _run_contract(
        "(()=>({"
        "queued:contract.projectPhaseLabel({analysis_status:'queued',project_progress:{status:'queued'}}),"
        "analysis:contract.projectPhaseLabel({analysis_status:'processing'}),"
        "postprocess:contract.projectPhaseLabel({analysis_status:'done',postprocess:{status:'running'}}),"
        "fusion:contract.projectPhaseLabel({analysis_status:'done',prompt_fusion:{status:'running'}}),"
        "pendingOnly:contract.projectPhaseLabel({analysis_status:'processing',prompt_fusion:{status:'pending'}}),"
        "context:contract.projectPhaseLabel({generation:{status:'running',stage:'context_ir'}}),"
        "h3:contract.projectPhaseLabel({generation:{status:'running',stage:'h3'}}),"
        "stitch:contract.projectPhaseLabel({generation:{status:'running',stage:'stitch'}})"
        "}))()"
    )
    assert result == {
        "queued": "AI 正在理解视频",
        "analysis": "AI 正在理解视频",
        "postprocess": "AI 正在思考创意",
        "fusion": "AI 正在思考创意",
        "pendingOnly": "AI 正在理解视频",
        "context": "AI 正在思考创意",
        "h3": "AI 正在生成视频",
        "stitch": "AI 正在生成视频",
    }


def test_project_configuration_model_restores_server_values_and_local_inputs():
    result = _run_contract(
        "(()=>{const request={version:1,output:{aspect_ratio:'16:9',resolution:'480p',fit_mode:'auto'},"
        "processing:{optimize_image:true,remove_subtitle:true,remove_logo:true},"
        "dialogue:{mode:'auto_rewrite',target_language:'日语'},"
        "replacement_guidance:{instruction:'替换成参考产品',image_field:'replacement_image'}};"
        "const local={request,input:contract.creationInputSnapshot({url:'https://media.example/source.mp4',"
        "replacementImage:{name:'product.png',size:321,type:'image/png'}})};"
        "const restored=contract.projectComposerModel({effective_request:request,"
        "input_receipt:{replacement_image:{sha256:'a',bytes:321}}},local);"
        "const same=contract.projectComposerModel({effective_request:{...request,"
        "dialogue:{mode:'auto_rewrite',target_language:'与原视频相同'},replacement_guidance:null},"
        "creation_input:{version:1,source:{mode:'upload',filename:'clip.mp4',bytes:456},replacement_image:null},"
        "input_receipt:{replacement_image:null}},null);"
        "const historical=contract.projectComposerModel({effective_request:{...request,replacement_guidance:null},"
        "creation_input:null,input_receipt:{replacement_image:null}},local);"
        "const invalid=contract.projectComposerModel({effective_request:{version:1,output:{aspect_ratio:'1:1'}}},local);"
        "return {restored,same,historical,invalid}})()"
    )
    assert result["restored"] == {
        "aspectRatio": "16:9",
        "resolution": "480p",
        "languageMode": "other",
        "targetLanguage": "日语",
        "replacementInstruction": "替换成参考产品",
        "source": {"mode": "link", "reference_url": "https://media.example/source.mp4"},
        "replacementImage": {
            "filename": "product.png", "bytes": 321,
            "media_type": "image/png", "preview_url": "",
        },
    }
    assert result["same"] == {
        "aspectRatio": "16:9",
        "resolution": "480p",
        "languageMode": "same",
        "targetLanguage": "",
        "replacementInstruction": "",
        "source": {"mode": "upload", "filename": "clip.mp4", "bytes": 456},
        "replacementImage": None,
    }
    assert result["historical"] == {
        "aspectRatio": "16:9",
        "resolution": "480p",
        "languageMode": "other",
        "targetLanguage": "日语",
        "replacementInstruction": "",
        "source": None,
        "replacementImage": None,
    }
    assert result["invalid"] is None


def test_creation_input_server_presence_is_authoritative_and_link_null_is_preserved():
    result = _run_contract(
        "(()=>{const local=contract.creationInputSnapshot({url:'https://local.example/source.mp4'});"
        "const omitted=contract.resolveCreationInputSnapshot({},local);"
        "const explicitNull=contract.resolveCreationInputSnapshot({creation_input:null},local);"
        "const future=contract.resolveCreationInputSnapshot({creation_input:{version:2,source:{mode:'link',"
        "reference_url:'https://server.example/source.mp4'},replacement_image:null}},local);"
        "const malformed=contract.resolveCreationInputSnapshot({creation_input:{version:1,"
        "source:{mode:'upload',filename:'clip.mp4',bytes:12},replacement_image:{filename:'product.png',"
        "bytes:3,media_type:'image/png'}}},local);"
        "const redacted=contract.resolveCreationInputSnapshot({creation_input:{version:1,"
        "source:{mode:'link',reference_url:null},replacement_image:null}},local);"
        "return {omitted,explicitNull,future,malformed,redacted}})()"
    )

    assert result["omitted"]["source"] == {
        "mode": "link", "reference_url": "https://local.example/source.mp4"
    }
    assert result["explicitNull"] is None
    assert result["future"] is None
    assert result["malformed"] is None
    assert result["redacted"] == {
        "version": 1,
        "source": {"mode": "link", "reference_url": None},
        "replacement_image": None,
    }


def test_creation_input_preview_path_is_exact_and_same_origin():
    result = _run_contract(
        "(()=>{const id='project-1';const origin='https://studio.example';"
        "const path='/api/conversations/project-1/creation-input/replacement-image';"
        "return {relative:contract.creationInputReplacementPreviewPath(id,path,origin),"
        "absolute:contract.creationInputReplacementPreviewPath(id,origin+path,origin),"
        "external:contract.creationInputReplacementPreviewPath(id,'https://evil.example'+path,origin),"
        "otherProject:contract.creationInputReplacementPreviewPath(id,"
        "'/api/conversations/project-2/creation-input/replacement-image',origin),"
        "query:contract.creationInputReplacementPreviewPath(id,path+'?download=1',origin),"
        "suffix:contract.creationInputReplacementPreviewPath(id,path+'/extra',origin)}})()"
    )

    expected = "/api/conversations/project-1/creation-input/replacement-image"
    assert result == {
        "relative": expected,
        "absolute": expected,
        "external": None,
        "otherProject": None,
        "query": None,
        "suffix": None,
    }


def test_configuration_is_reapplied_on_refresh_and_project_switch_not_polling():
    source = APP_JS.read_text(encoding="utf-8")
    load_detail = _function_source(source, "loadDetail")
    submit = _function_source(source, "handleSend")
    restore = _function_source(source, "applyProjectComposer")

    assert "if (!silent) applyProjectComposer(detail);" in load_detail
    assert "rememberSubmittedComposer(" in submit
    assert submit.index("rememberSubmittedComposer(") < submit.index("selectConversation(created.id)")
    for mapping in (
        'input[name="aspect-ratio"]',
        'input[name="resolution"]',
        'input[name="target-language-mode"]',
        '$("lang-input").value = model.targetLanguage',
        '$("replacement-instruction").value = model.replacementInstruction',
        'model.source.mode === "upload"',
        "storedComposerDraft(detail.id)",
    ):
        assert mapping in restore

    select = _function_source(source, "selectConversation")
    loading = _function_source(source, "prepareComposerForProjectLoad")
    assert select.index("prepareComposerForProjectLoad()") < select.index("loadDetail(id, false)")
    assert "state.viewingSubmittedConfig = true" in loading
    assert 'showUnavailableSource("正在加载项目配置…")' in loading


def test_restored_project_configuration_is_view_only_and_releases_file_objects():
    source = APP_JS.read_text(encoding="utf-8")
    restore = _function_source(source, "applyProjectComposer")
    reset = _function_source(source, "resetComposerForNewProject")
    update_button = _function_source(source, "updateSendBtn")
    remember = _function_source(source, "rememberSubmittedComposer")
    cleanup = _function_source(source, "clearComposerSnapshots")
    expired = _function_source(source, "sessionExpired")

    assert "state.viewingSubmittedConfig = true" in restore
    assert "state.viewingSubmittedConfig = false" in reset
    assert "state.viewingSubmittedConfig" in update_button
    assert "const record = { request, input };" in remember
    assert "sourceFile" not in remember
    assert "replacementImage:" not in remember
    assert "sessionStorage.removeItem(COMPOSER_SNAPSHOTS_KEY)" in cleanup
    assert "URL.revokeObjectURL(state.replacementPreviewURL)" in cleanup
    assert "clearComposerSnapshots()" in expired

    enter = _function_source(source, "enterApp")
    assert "resetComposerForNewProject()" in enter

    preview = _function_source(source, "loadSubmittedReplacementPreview")
    assert "creationInputReplacementPreviewPath(conversationId, previewUrl)" in preview
    assert "api(safePath)" in preview
    assert "api(previewUrl)" not in preview

    assert 'showUnavailableSource("历史项目未保存来源信息")' in restore
    assert '$("url-input").placeholder = "链接来源已提交"' in restore
    assert "localPreviewAllowed" in restore


def test_success_response_freezes_configuration_before_followup_requests():
    source = APP_JS.read_text(encoding="utf-8")
    submit = _function_source(source, "handleSend")
    entry = submit.split("if (!await guardDirtyPrompt())", 1)[0]
    success = submit.split("const created = await uploadConversation", 1)[1].split(
        "} catch (err)", 1
    )[0]
    before_refresh = success.split("await refreshList(false)", 1)[0]

    assert "state.uploading || state.viewingSubmittedConfig" in entry
    for immediate_freeze in (
        "state.viewingSubmittedConfig = true",
        "state.file = null",
        "state.replacementImage = null",
        '$("file-input").value = ""',
        '$("replacement-image-input").value = ""',
    ):
        assert immediate_freeze in before_refresh
    assert before_refresh.index("rememberSubmittedComposer(") < before_refresh.index(
        "state.viewingSubmittedConfig = true"
    )
    assert before_refresh.index("state.viewingSubmittedConfig = true") < before_refresh.index(
        "setUploading(false)"
    )
    assert success.index("await refreshList(false)") < success.index(
        "selectConversation(created.id)"
    )


def test_view_only_configuration_blocks_hidden_file_and_drop_paths():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    set_uploading = _function_source(source, "setUploading")
    pick_video = _function_source(source, "pickFile")
    pick_replacement = _function_source(source, "pickReplacementImage")
    bind_events = _function_source(source, "bindEvents")

    for input_id in ("file-input", "replacement-image-input"):
        assert "hidden" in _tag_attrs(html, "input", input_id)
        assert f'$("{input_id}").disabled = controlsDisabled' in set_uploading
    assert "if (state.viewingSubmittedConfig) return;" in pick_video
    assert pick_video.index("state.viewingSubmittedConfig") < pick_video.index(
        "setComposerError(null)"
    )
    assert "if (state.viewingSubmittedConfig) return;" in pick_replacement
    assert pick_replacement.index("state.viewingSubmittedConfig") < pick_replacement.index(
        "setComposerError(null)"
    )
    assert bind_events.count("!state.viewingSubmittedConfig") >= 2
    assert (
        "state.uploading || state.viewingSubmittedConfig || sourceMode() !== \"upload\""
        in bind_events
    )


def test_reference_image_picker_rejects_unsupported_formats_before_submit():
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    file_check = _function_source(source, "isReferenceImageFile")
    picker = _function_source(source, "pickReplacementImage")
    capability_loader = _function_source(source, "loadMinimalCreationCapability")

    assert 'accept="image/jpeg,image/png,image/webp"' in _tag_attrs(
        html, "input", "replacement-image-input"
    )
    assert '$("replacement-image-input").accept = normalized.replacement.accept.join(",")' in capability_loader
    assert "REPLACEMENT_IMAGE_TYPES.includes(file.type)" in file_check
    assert "jpe?g|png|webp" in file_check
    assert "if (!isReferenceImageFile(file))" in picker
    assert "参考图仅支持 JPG、PNG 或 WebP" in picker
    assert "file.size > capability.replacement.max_bytes" in picker
    assert "state.replacementImage = file" in picker


def test_logout_and_401_scrub_composer_session_and_stream_urls():
    source = APP_JS.read_text(encoding="utf-8")
    auth_error = _function_source(source, "handleAuthError")
    expired = _function_source(source, "sessionExpired")
    cleanup = _function_source(source, "clearComposerSnapshots")
    reset = _function_source(source, "resetComposerForNewProject")
    clear_stream = _function_source(source, "clearStream")
    bind_events = _function_source(source, "bindEvents")
    logout = bind_events.split('$("logout-btn").addEventListener', 1)[1].split(
        '$("menu-btn").addEventListener', 1
    )[0]

    assert "sessionExpired()" in auth_error
    for action in (
        "state.authEpoch += 1",
        "state.token = null",
        "localStorage.removeItem(TOKEN_KEY)",
        "clearComposerSnapshots()",
        "resetComposerForNewProject()",
        "clearStream()",
    ):
        assert action in expired
        assert action in logout
    assert expired.index("clearComposerSnapshots()") < expired.index("showLogin(")
    assert logout.index("clearComposerSnapshots()") < logout.index("showLogin(")

    assert "sessionStorage.removeItem(COMPOSER_SNAPSHOTS_KEY)" in cleanup
    assert "URL.revokeObjectURL(state.replacementPreviewURL)" in cleanup
    for cleared_control in (
        '$("file-input").value = ""',
        '$("url-input").value = ""',
        "resetReplacementImageDisplay()",
        '$("replacement-instruction").value = ""',
        '$("lang-input").value = ""',
    ):
        assert cleared_control in reset
    assert "revokeURLs()" in clear_stream
    assert '$("stream").textContent = ""' in clear_stream


def test_inflight_upload_continuations_are_scoped_to_the_auth_epoch():
    source = APP_JS.read_text(encoding="utf-8")
    submit = _function_source(source, "handleSend")
    login = _function_source(source, "doLogin")
    before_upload = submit.split("const created = await uploadConversation", 1)[0]
    after_upload = submit.split("const created = await uploadConversation", 1)[1]
    success, failure = after_upload.split("} catch (err)", 1)

    assert "state.authEpoch += 1" in login
    assert "const submitEpoch = state.authEpoch" in before_upload
    assert before_upload.index("const submitEpoch = state.authEpoch") < before_upload.index(
        "const progress"
    )
    assert "if (submitEpoch !== state.authEpoch) return;" in success
    assert success.index("if (submitEpoch !== state.authEpoch) return;") < success.index(
        "rememberSubmittedComposer("
    )
    assert "if (submitEpoch !== state.authEpoch) return;" in failure
    assert failure.index("if (submitEpoch !== state.authEpoch) return;") < failure.index(
        "setUploading(false)"
    )


def test_success_keeps_submitted_configuration_in_the_collapsible_composer():
    source = APP_JS.read_text(encoding="utf-8")
    submit = _function_source(source, "handleSend")
    success = submit.split("const created = await uploadConversation", 1)[1].split(
        "} catch (err)", 1
    )[0]

    assert "state.clientRequestId = newRequestId()" in success
    assert "setUploading(false)" in success
    assert "selectConversation(created.id)" in success
    assert "created," in success
    assert success.index("state.viewingSubmittedConfig = true") < success.index("setUploading(false)")
    for destructive_reset in (
        "clearFile()",
        '$("url-input").value = ""',
        '$("lang-input").value = ""',
        '$("replacement-instruction").value = ""',
        "clearReplacementImage()",
    ):
        assert destructive_reset not in success


def test_sidebar_project_cards_restore_0903_information_density_and_manual_selection():
    source = APP_JS.read_text(encoding="utf-8")
    render_list = _function_source(source, "renderList")

    for visible_project_field in (
        'const badgeState = conversationBadge(c);',
        'el("span", "badge " + badgeState.className, badgeState.text)',
        'el("span", "conv-identity")',
        "formatDuration(c.duration_s)",
        'c.segment_count + " 段"',
        'el("span", "conv-footer")',
        'el("span", "conv-time", fmtTime(c.updated_at || c.created_at))',
        'el("span", "conv-output " + (c.has_video === true ? "is-ready" : "is-waiting")',
        'c.has_video === true ? "成片已提交" : "等待成片"',
    ):
        assert visible_project_field in render_list
    assert 'el("span", "conv-id", "#" + shortId(c.id))' not in render_list

    assert 'c.id === state.currentId ? " selected" : ""' in render_list
    click_handler = render_list.split('item.addEventListener("click", () => {', 1)[1]
    assert click_handler.index("selectConversation(c.id);") < click_handler.index(
        "closeDrawer();"
    )


def test_sidebar_refresh_uses_list_summaries_without_n_plus_one_detail_requests():
    source = APP_JS.read_text(encoding="utf-8")
    refresh = _function_source(source, "refreshList")
    hydrate = _function_source(source, "hydrateConversationSummaries")
    thumbnail = _function_source(source, "loadHistoryThumbnail")

    assert refresh.count('apiJSON("/api/conversations")') == 1
    assert 'apiJSON("/api/conversations/"' not in refresh
    assert "hydrateConversationSummaries()" in refresh
    assert 'apiJSON("/api/conversations/" + encodeURIComponent(summary.id))' not in source
    assert "apiJSON(" not in hydrate
    assert "loadDetail(" not in hydrate
    assert "loadHistoryThumbnail(summary)" in hydrate

    assert "summary.thumbnail_path" in thumbnail
    assert '"/files/" + encodedMediaPath(summary.thumbnail_path)' in thumbnail
    assert "apiJSON(" not in thumbnail
    assert "loadDetail(" not in thumbnail


def test_enter_app_stays_new_until_manual_or_created_project_selection():
    source = APP_JS.read_text(encoding="utf-8")
    enter = _function_source(source, "enterApp").split(
        "/* ===== 侧栏会话列表 ===== */", 1
    )[0]
    refresh = _function_source(source, "refreshList")
    select = _function_source(source, "selectConversation")
    render_list = _function_source(source, "renderList")
    bind = _function_source(source, "bindEvents")

    for caller in ("doLogin", "boot"):
        assert "enterApp();" in _function_source(source, caller)
    for new_project_action in (
        "state.currentId = null",
        "state.detail = null",
        "resetComposerForNewProject()",
        "renderEmptyHero()",
        "refreshList(false)",
    ):
        assert new_project_action in enter
    assert "SELECTED_PROJECT_KEY" not in source
    assert "storedSelectedProjectId" not in source
    assert "state.pendingRestoreId" not in source
    assert "rememberSelectedProjectId" not in source
    assert "clearSelectedProjectId" not in source
    assert "refreshList(true)" not in enter
    assert "selectConversation(" not in enter
    assert refresh.count('apiJSON("/api/conversations")') == 1
    assert "selectConversation(" not in refresh

    click_handler = render_list.split('item.addEventListener("click", () => {', 1)[1]
    assert "selectConversation(c.id);" in click_handler
    assert "loadDetail(id, false)" in select

    new_project = bind.split('$("new-chat-btn").addEventListener', 1)[1].split(
        '$("attach-btn").addEventListener', 1
    )[0]
    for reset_action in (
        "state.currentId = null",
        "state.detail = null",
        "resetComposerForNewProject()",
        "renderEmptyHero()",
    ):
        assert reset_action in new_project

    submit = _function_source(source, "handleSend")
    success = submit.split("const created = await uploadConversation", 1)[1].split(
        "} catch (err)", 1
    )[0]
    assert success.index("await refreshList(false)") < success.index(
        "selectConversation(created.id)"
    )


def test_initial_detail_failure_clears_selection_and_returns_to_new_project():
    source = APP_JS.read_text(encoding="utf-8")
    load_detail = _function_source(source, "loadDetail")
    failure = load_detail.split("} catch (err) {", 1)[1]
    silent_start = failure.index("if (silent) {")
    initial_start = failure.index("state.currentId = null", silent_start)
    silent_failure = failure[silent_start:initial_start]
    initial_failure = failure[initial_start:]

    assert "if (seq !== state.detailSeq || state.currentId !== id) return;" in failure
    assert "if (silent) {" in silent_failure
    assert 'renderStreamError("会话加载失败：" + err.message);' in silent_failure
    assert "startPolling(id);" in silent_failure
    for action in (
        "state.currentId = null",
        "state.detail = null",
        "resetComposerForNewProject()",
        "resetGenerationConfigDisclosure()",
        "renderList()",
        "renderEmptyHero()",
        'setComposerError("项目加载失败，已回到新建项目：" + err.message)',
    ):
        assert action in initial_failure

    ordered_actions = (
        "state.currentId = null",
        "state.detail = null",
        "resetComposerForNewProject()",
        "renderList()",
        "renderEmptyHero()",
    )
    positions = [initial_failure.index(action) for action in ordered_actions]
    assert positions == sorted(positions)


def test_terminal_detail_updates_sidebar_locally_without_another_list_get():
    source = APP_JS.read_text(encoding="utf-8")
    load_detail = _function_source(source, "loadDetail")
    poll_decision = load_detail.split("if (shouldPollDetail(detail)", 1)[1]

    assert "syncConversationDetail(state.conversations, detail)" in load_detail
    assert "renderList();" in load_detail
    assert "stopPolling();" in poll_decision
    assert "refreshList(" not in poll_decision
