from test_web_h3_contract import APP_JS, _run_contract


def test_segment_products_disclosure_is_lazy_native_and_toggleable():
    result = _run_contract(
        "(()=>{class Node{constructor(tag){this.tag=tag;this.children=[];this.attrs={};"
        "this.listeners={};this.hidden=false;this.textContent='';this.className='';this.id=''}"
        "appendChild(child){this.children.push(child);return child}"
        "setAttribute(k,v){this.attrs[k]=String(v)}getAttribute(k){return this.attrs[k]}"
        "addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)}"
        "click(){for(const fn of this.listeners.click||[])fn({type:'click'})}}"
        "global.document={createElement:(tag)=>new Node(tag)};let builds=0;"
        "const detail={id:'cid-products',segments:[{index:1}]};"
        "const first=contract.segmentProductsDisclosure(detail,()=>{builds+=1;return new Node('section')});"
        "const button=first.children[0],panel=first.children[1];"
        "const initial={tag:button.tag,builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded'),controls:button.getAttribute('aria-controls')};"
        "button.click();const expanded={builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded')};button.click();"
        "const collapsed={builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded')};button.click();"
        "return {initial,expanded,collapsed,reopened:{builds,hidden:panel.hidden}}})()"
    )
    assert result == {
        "initial": {
            "tag": "button",
            "builds": 0,
            "hidden": True,
            "text": "展开分段产物",
            "expanded": "false",
            "controls": "segment-products-1",
        },
        "expanded": {
            "builds": 1,
            "hidden": False,
            "text": "收起分段产物",
            "expanded": "true",
        },
        "collapsed": {
            "builds": 1,
            "hidden": True,
            "text": "展开分段产物",
            "expanded": "false",
        },
        "reopened": {"builds": 2, "hidden": False},
    }


def test_segment_products_state_survives_poll_render_and_resets_on_switch():
    result = _run_contract(
        "(()=>{class Node{constructor(){this.children=[];this.attrs={};this.listeners={};"
        "this.hidden=false;this.textContent='';this.className='';this.id=''}"
        "appendChild(child){this.children.push(child);return child}setAttribute(k,v){this.attrs[k]=String(v)}"
        "getAttribute(k){return this.attrs[k]}addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)}"
        "click(){for(const fn of this.listeners.click||[])fn({type:'click'})}}"
        "global.document={createElement:()=>new Node()};const detail={id:'cid-poll',segments:[{index:1}]};"
        "const build=()=>new Node();const first=contract.segmentProductsDisclosure(detail,build);"
        "first.children[0].click();const polled=contract.segmentProductsDisclosure(detail,build);"
        "const afterPoll={hidden:polled.children[1].hidden,expanded:polled.children[0].getAttribute('aria-expanded')};"
        "contract.resetSegmentProductsDisclosure(detail.id);"
        "const switched=contract.segmentProductsDisclosure(detail,build);"
        "return {afterPoll,afterSwitch:{hidden:switched.children[1].hidden,"
        "expanded:switched.children[0].getAttribute('aria-expanded')}}})()"
    )
    assert result == {
        "afterPoll": {"hidden": False, "expanded": "true"},
        "afterSwitch": {"hidden": True, "expanded": "false"},
    }


def test_segment_products_wiring_is_lazy_and_absent_without_segments():
    source = APP_JS.read_text(encoding="utf-8")
    results = source.split("function renderResults", 1)[1].split(
        "function renderUserBubble", 1
    )[0]
    assert "if (segments.length > 0)" in results
    assert "segmentProductsDisclosure(detail)" in results

    products = source.split("function segmentProductsDisclosure", 1)[1].split(
        "function openLightbox", 1
    )[0]
    assert 'expandText: "展开分段产物"' in products
    assert 'collapseText: "收起分段产物"' in products
    assert "() => renderSegments(detail" in products
    assert "closeLightbox({ restoreFocus: false })" in products
    assert "releaseTrackedURLs(mediaURLs)" in products

    select = source.split("function selectConversation(id)", 1)[1].split(
        "function startPolling", 1
    )[0]
    assert "resetSegmentProductsDisclosure(id)" in select


def test_segment_product_media_urls_are_released_once_on_dispose():
    result = _run_contract(
        "(()=>{const urls=['blob:one','blob:two'];const released=[];"
        "contract.releaseTrackedURLs(urls,(url)=>released.push(url));"
        "return {urls,released}})()"
    )
    assert result == {"urls": [], "released": ["blob:one", "blob:two"]}

    source = APP_JS.read_text(encoding="utf-8")
    grid = source.split("function kfGrid", 1)[1].split(
        "let disclosureSeq", 1
    )[0]
    assert "if (options.onURL) options.onURL(url)" in grid


def test_conversation_badge_uses_final_video_truth_and_generation_precedence():
    result = _run_contract(
        "(()=>{const cases=["
        "{status:'done',has_video:false,segments:[{index:1}]},"
        "{status:'done',has_video:true},"
        "{status:'done',has_video:true,generation:{status:'succeeded'}},"
        "{status:'done',has_video:false,generation:{status:'succeeded'}},"
        "{status:'done',has_video:true,generation:{status:'queued'}},"
        "{status:'done',has_video:true,generation:{status:'running'}},"
        "{status:'done',has_video:true,generation:{status:'failed'}},"
        "{status:'done',has_video:true,generation:{status:'submission_unknown'}},"
        "{status:'processing',has_video:false}];"
        "return cases.map(contract.conversationBadge)})()"
    )
    assert result == [
        {"className": "analyzed", "text": "分析完成"},
        {"className": "done", "text": "已完成"},
        {"className": "done", "text": "已完成"},
        {"className": "failed", "text": "最终视频缺失"},
        {"className": "processing", "text": "生成排队中"},
        {"className": "processing", "text": "生成中"},
        {"className": "failed", "text": "生成失败"},
        {"className": "failed", "text": "提交结果未知"},
        {"className": "processing", "text": "处理中"},
    ]


def test_authoritative_navigation_status_maps_without_local_inference():
    result = _run_contract(
        "(()=>['analysis_queued','analysis_processing','analysis_failed','analysis_unknown','analysis_complete',"
        "'generation_queued','generation_running','generation_failed','generation_submission_unknown',"
        "'generation_resume_required','generation_unknown','output_missing',"
        "'completed','postprocessing','postprocess_failed','postprocess_done','future']"
        ".map((navigation_status)=>contract.conversationBadge({navigation_status,status:'done',"
        "has_video:true,generation:{status:'succeeded'}})))()"
    )
    assert result == [
        {"className": "queued", "text": "分析排队中"},
        {"className": "processing", "text": "分析中"},
        {"className": "failed", "text": "分析失败"},
        {"className": "failed", "text": "分析状态未知"},
        {"className": "analyzed", "text": "分析完成"},
        {"className": "processing", "text": "生成排队中"},
        {"className": "processing", "text": "生成中"},
        {"className": "failed", "text": "生成失败"},
        {"className": "failed", "text": "提交结果未知"},
        {"className": "failed", "text": "等待继续"},
        {"className": "failed", "text": "生成状态未知"},
        {"className": "failed", "text": "最终视频缺失"},
        {"className": "done", "text": "已完成"},
        {"className": "processing", "text": "素材优化中"},
        {"className": "failed", "text": "素材优化失败"},
        {"className": "done", "text": "已完成"},
        {"className": "failed", "text": "状态异常"},
    ]


def test_present_but_invalid_navigation_status_never_uses_legacy_fallback():
    result = _run_contract(
        "(()=>[null,''].map((navigation_status)=>contract.conversationBadge({"
        "navigation_status,status:'done',has_video:true})))()"
    )
    assert result == [
        {"className": "failed", "text": "状态异常"},
        {"className": "failed", "text": "状态异常"},
    ]


def test_navigation_has_distinct_yellow_analysis_badge():
    css = (APP_JS.parent / "styles.css").read_text(encoding="utf-8")
    assert ".badge.analyzed" in css
    assert "var(--warning" in css.split(".badge.analyzed", 1)[1].split("}", 1)[0]

    source = APP_JS.read_text(encoding="utf-8")
    render_list = source.split("function renderList()", 1)[1].split(
        "function renderListError", 1
    )[0]
    assert "conversationBadge(c)" in render_list
    assert "STATUS_TEXT[c.status]" not in render_list


def test_list_refresh_preserves_generation_truth_learned_from_detail():
    result = _run_contract(
        "(()=>{const previous=[{id:'known',generation:{status:'failed'},navigation_status:'generation_failed',has_video:true},"
        "{id:'legacy',has_video:true}];const incoming=[{id:'known',status:'done',has_video:true},"
        "{id:'legacy',status:'done',has_video:true},{id:'new',status:'done',has_video:false,"
        "generation:{status:'running'}}];return contract.mergeConversationList(incoming,previous)})()"
    )
    assert result == [
        {
            "id": "known",
            "status": "done",
            "has_video": True,
            "generation": {"status": "failed"},
            "navigation_status": "generation_failed",
        },
        {"id": "legacy", "status": "done", "has_video": True},
        {
            "id": "new",
            "status": "done",
            "has_video": False,
            "generation": {"status": "running"},
        },
    ]


def test_detail_poll_updates_current_navigation_badge_without_another_request():
    result = _run_contract(
        "(()=>{const items=[{id:'current',status:'done',has_video:false}];"
        "const before=contract.conversationBadge(items[0]);"
        "contract.syncConversationDetail(items,{id:'current',status:'done',has_video:false,"
        "generation:{status:'running'},navigation_status:'generation_running'});"
        "const running=contract.conversationBadge(items[0]);"
        "contract.syncConversationDetail(items,{id:'current',status:'done',has_video:true,"
        "generation:{status:'succeeded'},navigation_status:'completed'});"
        "const succeeded=contract.conversationBadge(items[0]);"
        "return {before,running,succeeded}})()"
    )
    assert result == {
        "before": {"className": "analyzed", "text": "分析完成"},
        "running": {"className": "processing", "text": "生成中"},
        "succeeded": {"className": "done", "text": "已完成"},
    }
