from test_web_h3_contract import APP_JS, ROOT, _run_contract


def test_create_dialogue_fields_accept_all_modes_and_serialize_canonical_lines():
    result = _run_contract(
        "['auto','none'].map(mode=>contract.buildCreateDialogueFields(mode,''))"
        ".concat(['edit','custom'].map(mode=>contract.buildCreateDialogueFields("
        "mode,'0 - 1.5 | hello\\n1.5 - 3 | world')))"
    )
    assert result == [
        {"dialogue_mode": "auto"},
        {"dialogue_mode": "none"},
        {
            "dialogue_mode": "edit",
            "lines": (
                '[{"start_s":0,"end_s":1.5,"text":"hello"},'
                '{"start_s":1.5,"end_s":3,"text":"world"}]'
            ),
        },
        {
            "dialogue_mode": "custom",
            "lines": (
                '[{"start_s":0,"end_s":1.5,"text":"hello"},'
                '{"start_s":1.5,"end_s":3,"text":"world"}]'
            ),
        },
    ]


def test_create_dialogue_fields_reject_missing_manual_lines_locally():
    result = _run_contract(
        "['edit','custom'].map(mode=>{try{contract.buildCreateDialogueFields(mode,'  ');"
        "return null}catch(error){return error.message}})"
    )
    assert result == [
        "编辑台词模式请至少填写一行台词",
        "自定义台词模式请至少填写一行台词",
    ]


def test_create_dialogue_ui_and_formdata_are_bound_before_creation_only():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    source = APP_JS.read_text(encoding="utf-8")
    upload_source = source.split("function uploadConversation", 1)[1]
    upload_source = upload_source.split("async function handleSend", 1)[0]
    send_source = source.split("async function handleSend", 1)[1]
    send_source = send_source.split("/* ===== 事件绑定与启动 ===== */", 1)[0]

    assert html.count('name="dialogue-mode"') == 4
    for mode in ("auto", "none", "edit", "custom"):
        assert f'name="dialogue-mode" value="{mode}"' in html
    assert 'id="create-dialogue-lines"' in html
    assert 'fd.append("dialogue_mode", dialogue.dialogue_mode)' in upload_source
    assert 'fd.append("lines", dialogue.lines)' in upload_source
    assert "buildCreateDialogueFields" in send_source
    assert "buildSubmitPayload" not in send_source


def test_long_video_task_count_comes_from_frozen_plan():
    result = _run_contract(
        "["
        "contract.longVideoContract({duration_s:15,segment_count:1,plan_receipt:'a'.repeat(64)}),"
        "contract.longVideoContract({duration_s:30,segment_count:2,plan_receipt:'b'.repeat(64)})"
        "]"
    )
    assert result == [
        {"isLong": True, "ready": True, "segmentCount": 1, "planReceipt": "a" * 64},
        {"isLong": True, "ready": True, "segmentCount": 2, "planReceipt": "b" * 64},
    ]


def test_long_video_contract_fails_closed_when_plan_metadata_is_missing():
    result = _run_contract(
        "["
        "contract.longVideoContract({duration_s:15,segment_count:1}),"
        "contract.longVideoContract({duration_s:30,plan_receipt:'a'.repeat(64)}),"
        "contract.longVideoContract({duration_s:10})"
        "]"
    )
    assert result == [
        {"isLong": True, "ready": False, "segmentCount": 1, "planReceipt": None},
        {"isLong": True, "ready": False, "segmentCount": None, "planReceipt": "a" * 64},
        {"isLong": False, "ready": True, "segmentCount": None, "planReceipt": None},
    ]


def test_long_submit_is_restricted_and_binds_current_plan_receipt():
    payload = _run_contract(
        "contract.buildSubmitPayload({clientRequestId:'request-long',dialogueMode:'auto',"
        "fitRequired:false,isLong:true,fastMode:false,planReceipt:'c'.repeat(64),"
        "aspectRatio:'16:9',resolution:'480p'})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-long",
        "dialogue_mode": "auto",
        "fit_mode": "none",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "expected_plan_receipt": "c" * 64,
        "fast_mode": False,
    }


def test_long_submit_fast_mode_is_explicit_and_short_submit_is_unchanged():
    payloads = _run_contract(
        "[false,true].map(fastMode=>contract.buildSubmitPayload({"
        "clientRequestId:'request-long',dialogueMode:'auto',fitRequired:false,isLong:true,"
        "fastMode,planReceipt:'c'.repeat(64),aspectRatio:'16:9',resolution:'480p'}))"
        ".concat([contract.buildSubmitPayload({clientRequestId:'request-short',"
        "dialogueMode:'none',fitRequired:false,isLong:false,fastMode:true,"
        "aspectRatio:'9:16',resolution:'768p'})])"
    )
    assert payloads[0]["fast_mode"] is False
    assert payloads[1]["fast_mode"] is True
    assert "fast_mode" not in payloads[2]


def test_short_submit_payload_remains_unchanged():
    payload = _run_contract(
        "contract.buildSubmitPayload({clientRequestId:'request-short',dialogueMode:'custom',"
        "linesText:'0 - 1 | hello',fitRequired:false,isLong:false,planReceipt:'d'.repeat(64),"
        "aspectRatio:'9:16',resolution:'768p'})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-short",
        "dialogue_mode": "custom",
        "fit_mode": "none",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "lines": [{"start_s": 0, "end_s": 1, "text": "hello"}],
    }


def test_long_resume_reuses_attempt_and_current_plan_receipt():
    payload = _run_contract(
        "contract.buildResumePayload({duration_s:30,segment_count:2,plan_receipt:'f'.repeat(64),"
        "generation:{status:'resume_required',client_request_id:'request-old',fast_mode:true},"
        "dialogue:{mode:'none',lines:[]},fit_mode:'none',aspect_ratio:'16:9',resolution:'480p'})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-old",
        "dialogue_mode": "none",
        "fit_mode": "none",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "expected_plan_receipt": "f" * 64,
        "fast_mode": True,
    }


def test_long_retry_cost_uses_server_truth_and_stitch_is_free():
    result = _run_contract(
        "["
        "contract.generationRetryContract({duration_s:30,segment_count:3,"
        "generation:null}),"
        "contract.generationRetryContract({duration_s:30,segment_count:3,"
        "generation:{status:'failed',stage:'h3',retry_paid_segment_count:2,segments:["
        "{index:1,status:'succeeded'},{index:2,status:'succeeded'},"
        "{index:3,status:'succeeded'}]}}),"
        "contract.generationRetryContract({duration_s:30,segment_count:3,"
        "generation:{status:'failed',stage:'stitch',client_request_id:'request-old',"
        "retry_paid_segment_count:0,"
        "segments:[{index:1,status:'succeeded'},{index:2,status:'succeeded'},"
        "{index:3,status:'succeeded'}]}}),"
        "contract.generationRetryContract({duration_s:30,segment_count:3,"
        "generation:{status:'failed',stage:'h3',segments:["
        "{index:1,status:'failed'},{index:2,status:'failed'},"
        "{index:3,status:'failed'}]}}),"
        "contract.generationRetryContract({duration_s:30,segment_count:3,"
        "generation:{status:'submission_unknown',stage:'h3',segments:[]}})"
        "]"
    )
    assert result == [
        {"action": "new", "paidTaskCount": 3},
        {"action": "retry", "paidTaskCount": 2},
        {"action": "retry_stitch", "paidTaskCount": 0},
        {"action": "retry", "paidTaskCount": None},
        {"action": "none", "paidTaskCount": 0},
    ]


def test_stitch_retry_reuses_frozen_request_and_parameters():
    payload = _run_contract(
        "contract.buildStitchRetryPayload({duration_s:30,segment_count:2,"
        "plan_receipt:'a'.repeat(64),fit_mode:'pad',dialogue:{mode:'none',lines:[]},"
        "aspect_ratio:'9:16',resolution:'768p',"
        "generation:{status:'failed',stage:'stitch',client_request_id:'request-old',fast_mode:true}})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-old",
        "dialogue_mode": "none",
        "fit_mode": "pad",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "expected_plan_receipt": "a" * 64,
        "fast_mode": True,
    }


def test_failed_segment_retry_uses_new_request_with_frozen_parameters():
    payload = _run_contract(
        "contract.buildLongRetryPayload({duration_s:30,segment_count:2,"
        "plan_receipt:'b'.repeat(64),fit_required:true,fit_mode:'crop',"
        "aspect_ratio:'9:16',resolution:'768p',"
        "dialogue:{mode:'auto',lines:[]},generation:{status:'failed',stage:'h3',"
        "client_request_id:'request-old',fast_mode:true,segments:[{index:1,status:'succeeded'},"
        "{index:2,status:'failed'}]}},'request-new')"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-new",
        "dialogue_mode": "auto",
        "fit_mode": "crop",
        "aspect_ratio": "9:16",
        "resolution": "768p",
        "expected_plan_receipt": "b" * 64,
        "fast_mode": True,
    }


def test_historical_long_generation_freezes_fast_mode_off_for_retry_and_resume():
    payloads = _run_contract(
        "[contract.buildLongRetryPayload({duration_s:30,segment_count:2,"
        "plan_receipt:'a'.repeat(64),fit_required:false,fit_mode:'none',"
        "aspect_ratio:'16:9',resolution:'480p',dialogue:{mode:'auto',lines:[]},"
        "generation:{status:'failed',stage:'h3'}},'request-new'),"
        "contract.buildResumePayload({duration_s:30,segment_count:2,"
        "plan_receipt:'b'.repeat(64),fit_mode:'none',aspect_ratio:'16:9',resolution:'480p',"
        "dialogue:{mode:'none',lines:[]},generation:{status:'resume_required',"
        "client_request_id:'request-old'}})]"
    )
    assert [payload["fast_mode"] for payload in payloads] == [False, False]


def test_new_long_draft_defaults_fast_mode_on_and_server_frozen_values_win():
    result = _run_contract(
        "(()=>{const base={aspect_ratio:'16:9',resolution:'480p',fit_mode:'none',"
        "fit_profiles:{'16:9':{fit_required:false,default_fit_mode:'none'}},"
        "dialogue:{mode:'auto',lines:[]},receipt_version:1};"
        "const first=contract.generationDraft({...base,id:'cid-a',duration_s:30,generation:null});"
        "const polled=contract.generationDraft({...base,id:'cid-a',duration_s:30,generation:null});"
        "const other=contract.generationDraft({...base,id:'cid-b',duration_s:30,generation:null});"
        "const short=contract.generationDraft({...base,id:'cid-short',duration_s:10,generation:null});"
        "const frozenOn=contract.generationDraft({...base,id:'cid-c',duration_s:30,"
        "generation:{status:'failed',fast_mode:true}});"
        "frozenOn.fastMode=false;"
        "const resynced=contract.generationDraft({...base,id:'cid-c',duration_s:30,"
        "generation:{status:'failed',fast_mode:true}});"
        "const historical=contract.generationDraft({...base,id:'cid-d',duration_s:30,"
        "generation:{status:'succeeded'}});"
        "return {first:first.fastMode,polled:polled.fastMode,other:other.fastMode,"
        "short:short.fastMode,frozen:resynced.fastMode,"
        "historical:historical.fastMode}})()"
    )
    assert result == {
        "first": True,
        "polled": True,
        "other": True,
        "short": False,
        "frozen": True,
        "historical": False,
    }


def test_fast_mode_control_and_explanation_are_absent_from_web_source():
    source = APP_JS.read_text(encoding="utf-8")

    assert "function fastModeField" not in source
    assert "fastModeField," not in source
    assert "fastModeField(detail" not in source
    assert 'el("legend", null, "提交方式")' not in source
    assert "开启后会快速提交所有分段" not in source


def test_long_generation_parameter_summary_hides_fast_mode_label():
    source = APP_JS.read_text(encoding="utf-8")
    summary_source = source.split("function generationParameterSummary", 1)[1]
    summary_source = summary_source.split("function buildSubmitPayload", 1)[0]

    assert "快速模式" not in summary_source


def test_long_submit_rejects_edit_custom_and_missing_receipt():
    result = _run_contract(
        "['edit','custom'].map(mode=>{try{contract.buildSubmitPayload({clientRequestId:'request-x',"
        "dialogueMode:mode,linesText:'0 - 1 | x',fitRequired:false,isLong:true,"
        "planReceipt:'e'.repeat(64)});return null}catch(error){return error.message}})"
        ".concat([(()=>{try{contract.buildSubmitPayload({clientRequestId:'request-x',dialogueMode:'auto',"
        "fitRequired:false,isLong:true});return null}catch(error){return error.message}})()])"
    )
    assert result == [
        "长视频仅支持保留完整源音轨或静音",
        "长视频仅支持保留完整源音轨或静音",
        "长视频生成计划尚未就绪，请刷新后重试",
    ]


def test_generation_signature_tracks_plan_and_segment_progress_separately():
    result = _run_contract(
        "(()=>{const base={status:'done',duration_s:30,plan_receipt:'a'.repeat(64),segment_count:2,"
        "generation:{status:'running',stage:'h3',segments:[{index:1,status:'running'}]}};"
        "const original=contract.detailSignature(base);"
        "const progressed={...base,generation:{...base.generation,segments:[{index:1,status:'succeeded'}]}};"
        "const replanned={...base,plan_receipt:'b'.repeat(64)};"
        "return {progress:contract.detailSignature(progressed),replanned:contract.detailSignature(replanned),original};})()"
    )
    assert result["progress"]["stable"] == result["original"]["stable"]
    assert result["progress"]["generation"] != result["original"]["generation"]
    assert result["replanned"]["stable"] == result["original"]["stable"]
    assert result["replanned"]["generation"] != result["original"]["generation"]


def test_generation_segment_display_keeps_one_based_contract_index():
    result = _run_contract(
        "["
        "contract.generationSegmentLabel({index:1,status:'running'},0),"
        "contract.generationSegmentLabel({index:2,status:'succeeded'},1),"
        "contract.generationSegmentLabel({index:0,status:'failed'},2),"
        "contract.generationSegmentLabel({index:'bad',status:'pending'},3)"
        "]"
    )
    assert result == [
        "第 1 段 · 生成中",
        "第 2 段 · 已完成",
        "第 3 段 · 失败",
        "第 4 段 · 等待中",
    ]


def test_long_video_ui_copy_and_segment_progress_contract():
    source = APP_JS.read_text(encoding="utf-8")
    for text in (
        "本次新增 ",
        "个付费生成子任务",
        "重试生成",
        "重试拼接",
        "连续性为 best effort",
        "服务端按冻结模式复用成功段",
        "本次只提交上方所示新增付费分段",
        "保留完整源音轨",
        "静音",
        "完成 ",
        "generation-segments",
        "当前阶段",
        "chain",
        "join",
        "长视频生成计划尚未就绪，请刷新后重试",
    ):
        assert text in source
    assert 'if (stage === "stitch") return "视频拼接"' in source


def test_published_video_does_not_hide_stitch_recovery():
    source = APP_JS.read_text(encoding="utf-8")
    branch = source.split("function renderFinalSection(detail)", 1)[1]
    branch = branch.split("function kfGrid", 1)[0]
    assert "showPublishedVideo" in branch
    assert "showStitchRecovery" in branch
    assert "published.appendChild(videoSection" in branch
    assert "if (showPublishedVideo && !showStitchRecovery) return" in branch


def test_removed_frozen_prompt_display_copy_does_not_change_submit_or_cost_copy():
    removed_copy = "逐段冻结的模型提示词"
    display_assets = [APP_JS, ROOT / "web" / "index.html"]
    offenders = [
        str(path.relative_to(ROOT))
        for path in display_assets
        if removed_copy in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

    source = APP_JS.read_text(encoding="utf-8")
    assert "各段提示词将提交生成" in source
    assert "源提示词将直接提交生成" in source
    assert "个付费生成子任务" in source


def test_segment_disclosure_state_defaults_collapsed_and_toggles_both_ways():
    result = _run_contract(
        "(()=>{const trigger={attrs:{},textContent:'',setAttribute(k,v){this.attrs[k]=v}};"
        "const panel={hidden:false};const labels={expand:'展开第 1 段提示词',collapse:'收起第 1 段提示词',"
        "expandText:'展开提示词',collapseText:'收起提示词'};"
        "contract.setDisclosureState(trigger,panel,false,labels);"
        "const collapsed={attrs:{...trigger.attrs},text:trigger.textContent,hidden:panel.hidden};"
        "contract.setDisclosureState(trigger,panel,true,labels);"
        "const expanded={attrs:{...trigger.attrs},text:trigger.textContent,hidden:panel.hidden};"
        "contract.setDisclosureState(trigger,panel,false,labels);"
        "return {collapsed,expanded,collapsedAgain:{attrs:{...trigger.attrs},"
        "text:trigger.textContent,hidden:panel.hidden}}})()"
    )
    assert result == {
        "collapsed": {
            "attrs": {"aria-expanded": "false", "aria-label": "展开第 1 段提示词"},
            "text": "展开提示词",
            "hidden": True,
        },
        "expanded": {
            "attrs": {"aria-expanded": "true", "aria-label": "收起第 1 段提示词"},
            "text": "收起提示词",
            "hidden": False,
        },
        "collapsedAgain": {
            "attrs": {"aria-expanded": "false", "aria-label": "展开第 1 段提示词"},
            "text": "展开提示词",
            "hidden": True,
        },
    }


def test_long_segment_prompt_workspace_and_keyframes_are_accessible():
    source = APP_JS.read_text(encoding="utf-8")
    branch = source.split("function renderSegments(detail)", 1)[1]
    branch = branch.split("/* 关键帧放大查看", 1)[0]
    assert "promptWorkspace(detail, seg)" in branch
    assert "compact: true" in branch
    assert "expandable: true" in branch
    assert "authoritativeSegmentKeyframePaths(detail, seg)" in branch
    assert "onURL: (url) => mediaURLs.push(url)" in branch

    grid = source.split("function kfGrid", 1)[1]
    grid = grid.split("function sourcePromptEditable", 1)[0]
    assert 'el("button", "kf-expand-button")' in grid
    assert 'button.type = "button"' in grid
    assert "setDisclosureState(button, null, false" in grid
    assert "openLightbox(img.src, img.alt, button)" in grid
    assert 'trigger.setAttribute("aria-expanded", String(expanded))' in source


def test_long_segment_dialogue_reuses_the_three_way_prompt_workspace():
    source = APP_JS.read_text(encoding="utf-8")
    branch = source.split("function renderSegments(detail)", 1)[1]
    branch = branch.split("/* 关键帧放大查看", 1)[0]
    assert "promptWorkspace(detail, seg)" in branch

    generic = source.split("function createDisclosure", 1)[1]
    generic = generic.split("function dialogueText", 1)[0]
    assert 'button.type = "button"' in generic
    assert 'button.setAttribute("aria-controls", panel.id)' in generic
    assert "setDisclosureState(button, panel, initialExpanded, labels)" in generic

    workspace = source.split("function promptWorkspace", 1)[1]
    workspace = workspace.split("function editablePromptCard", 1)[0]
    assert '["generation", "展开生成提示词"]' in workspace
    assert '["dialogue", "展开段台词"]' in workspace
    assert '["image", "展开图片优化"]' in workspace


def test_lightbox_dom_lifecycle_hides_and_restores_focus_for_every_close_path():
    result = _run_contract(
        "(()=>{"
        "class FakeElement{constructor(tag,doc){this.tagName=tag.toUpperCase();this.doc=doc;"
        "this.children=[];this.attrs={};this.listeners={};this.hidden=false;this.className='';this.isConnected=false;"
        "this.classList={add:(v)=>{const s=new Set(this.className.split(/\\s+/).filter(Boolean));"
        "s.add(v);this.className=[...s].join(' ')},remove:(v)=>{this.className=this.className"
        ".split(/\\s+/).filter(x=>x&&x!==v).join(' ')},contains:(v)=>this.className.split(/\\s+/).includes(v)}}"
        "appendChild(child){this.children.push(child);child.parent=this;child.isConnected=this.isConnected;return child}"
        "setAttribute(k,v){this.attrs[k]=String(v)}getAttribute(k){return this.attrs[k]}"
        "removeAttribute(k){delete this.attrs[k]}"
        "addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)}"
        "dispatchEvent(e){e.target=e.target||this;e.currentTarget=this;"
        "e.stopPropagation=e.stopPropagation||(()=>{e.cancelBubble=true});"
        "for(const fn of this.listeners[e.type]||[])fn(e);"
        "if(e.bubbles!==false&&!e.cancelBubble&&this.parent)this.parent.dispatchEvent(e)}"
        "querySelector(sel){const match=(x)=>sel[0]==='.'?x.className.split(/\\s+/).includes(sel.slice(1)):"
        "x.tagName===sel.toUpperCase();for(const child of this.children){if(match(child))return child;"
        "const nested=child.querySelector(sel);if(nested)return nested}return null}"
        "contains(node){return this===node||this.children.some(x=>x.contains(node))}"
        "focus(){this.doc.activeElement=this}blur(){if(this.doc.activeElement===this)this.doc.activeElement=this.doc.body}}"
        "const doc={activeElement:null,listeners:{},createElement(tag){return new FakeElement(tag,this)},"
        "addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)},"
        "removeEventListener(k,fn){this.listeners[k]=(this.listeners[k]||[]).filter(x=>x!==fn)},"
        "dispatchEvent(e){for(const fn of this.listeners[e.type]||[])fn(e)}};"
        "doc.body=doc.createElement('body');doc.body.isConnected=true;"
        "doc.getElementById=(id)=>{const walk=(node)=>{if(node.attrs.id===id)return node;"
        "for(const child of node.children){const found=walk(child);if(found)return found}return null};return walk(doc.body)};"
        "global.document=doc;"
        "const shortOrigin=doc.createElement('button');doc.body.appendChild(shortOrigin);shortOrigin.focus();"
        "contract.openLightbox('blob:short','短视频关键帧');"
        "const box=doc.body.querySelector('.lightbox');const close=box.querySelector('.lightbox-close');"
        "const shortOpened={hidden:box.hidden,closeFocused:doc.activeElement===close};"
        "shortOrigin.focus();const tab={type:'keydown',key:'Tab',defaultPrevented:false,"
        "preventDefault(){this.defaultPrevented=true}};doc.dispatchEvent(tab);"
        "const tabTrapped={prevented:tab.defaultPrevented,closeFocused:doc.activeElement===close};"
        "shortOrigin.focus();const shiftTab={type:'keydown',key:'Tab',shiftKey:true,defaultPrevented:false,"
        "preventDefault(){this.defaultPrevented=true}};doc.dispatchEvent(shiftTab);"
        "const shiftTabTrapped={prevented:shiftTab.defaultPrevented,closeFocused:doc.activeElement===close};"
        "box.dispatchEvent({type:'click'});"
        "const backgroundClosed={hidden:box.hidden,focusRestored:doc.activeElement===shortOrigin};"
        "const sentinel=doc.createElement('button');doc.body.appendChild(sentinel);sentinel.focus();contract.closeLightbox();"
        "const backgroundCleared=doc.activeElement===sentinel;"
        "const otherOrigin=doc.createElement('button');doc.body.appendChild(otherOrigin);otherOrigin.focus();"
        "contract.openLightbox('blob:other','其他关键帧');let closeClickBubbled=0;"
        "box.addEventListener('click',()=>{closeClickBubbled+=1});close.dispatchEvent({type:'click'});"
        "const buttonClosed={hidden:box.hidden,focusRestored:doc.activeElement===otherOrigin,"
        "bubbled:closeClickBubbled};"
        "const buttonSentinel=doc.createElement('button');doc.body.appendChild(buttonSentinel);"
        "buttonSentinel.focus();contract.closeLightbox();const buttonCleared=doc.activeElement===buttonSentinel;"
        "const segment=doc.createElement('button');doc.body.appendChild(segment);segment.focus();"
        "contract.setDisclosureState(segment,null,false,{expand:'展开分段关键帧',collapse:'关闭分段关键帧'});"
        "contract.openLightbox('blob:segment','分段关键帧',segment);"
        "const segmentOpened={hidden:box.hidden,expanded:segment.getAttribute('aria-expanded'),"
        "closeFocused:doc.activeElement===close};"
        "doc.dispatchEvent({type:'keydown',key:'Escape'});"
        "const escapeClosed={hidden:box.hidden,expanded:segment.getAttribute('aria-expanded'),"
        "focusRestored:doc.activeElement===segment};"
        "const stream=doc.createElement('div');stream.setAttribute('id','stream');doc.body.appendChild(stream);"
        "const stale=doc.createElement('button');stream.appendChild(stale);stale.focus();"
        "contract.setDisclosureState(stale,null,false,{expand:'展开旧图',collapse:'关闭旧图'});"
        "contract.openLightbox('blob:stale','旧图',stale);stale.isConnected=false;contract.clearStream();"
        "const cleared={hidden:box.hidden,expanded:stale.getAttribute('aria-expanded'),"
        "src:box.querySelector('img').getAttribute('src')||null,"
        "alt:box.querySelector('img').getAttribute('alt')||null,focusLeftDialog:doc.activeElement===doc.body};"
        "return {shortOpened,tabTrapped,shiftTabTrapped,backgroundClosed,backgroundCleared,buttonClosed,buttonCleared,"
        "segmentOpened,escapeClosed,cleared}})()"
    )
    assert result == {
        "shortOpened": {"hidden": False, "closeFocused": True},
        "tabTrapped": {"prevented": True, "closeFocused": True},
        "shiftTabTrapped": {"prevented": True, "closeFocused": True},
        "backgroundClosed": {"hidden": True, "focusRestored": True},
        "backgroundCleared": True,
        "buttonClosed": {"hidden": True, "focusRestored": True, "bubbled": 0},
        "buttonCleared": True,
        "segmentOpened": {"hidden": False, "expanded": "true", "closeFocused": True},
        "escapeClosed": {"hidden": True, "expanded": "false", "focusRestored": True},
        "cleared": {
            "hidden": True,
            "expanded": "false",
            "src": None,
            "alt": None,
            "focusLeftDialog": True,
        },
    }
