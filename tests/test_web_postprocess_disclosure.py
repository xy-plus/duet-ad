from test_web_h3_contract import APP_JS, _run_contract


def test_disclosure_dom_is_lazy_toggleable_and_uses_unique_controls():
    result = _run_contract(
        "(()=>{class Node{constructor(tag){this.tag=tag;this.children=[];this.attrs={};"
        "this.listeners={};this.hidden=false;this.textContent='';this.className='';this.id=''}"
        "appendChild(child){this.children.push(child);return child}"
        "setAttribute(k,v){this.attrs[k]=String(v)}getAttribute(k){return this.attrs[k]}"
        "addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)}"
        "click(){for(const fn of this.listeners.click||[])fn({type:'click'})}}"
        "global.document={createElement:(tag)=>new Node(tag)};let builds=0;"
        "const labels={expand:'展开优化后素材',collapse:'收起优化后素材',"
        "expandText:'展开优化后素材',collapseText:'收起优化后素材'};"
        "const first=contract.createDisclosure(labels,()=>{builds+=1;return new Node('div')});"
        "const button=first.children[0],panel=first.children[1];"
        "const initial={builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded'),controls:button.getAttribute('aria-controls')};"
        "button.click();const expanded={builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded')};button.click();"
        "const collapsed={builds,hidden:panel.hidden,text:button.textContent,"
        "expanded:button.getAttribute('aria-expanded')};button.click();"
        "const expandedAgain={builds,hidden:panel.hidden};"
        "const second=contract.createDisclosure(labels,()=>new Node('div'));"
        "return {initial,expanded,collapsed,expandedAgain,unique:"
        "second.children[0].getAttribute('aria-controls')!==initial.controls}})()"
    )
    assert result == {
        "initial": {
            "builds": 0,
            "hidden": True,
            "text": "展开优化后素材",
            "expanded": "false",
            "controls": "disclosure-1",
        },
        "expanded": {
            "builds": 1,
            "hidden": False,
            "text": "收起优化后素材",
            "expanded": "true",
        },
        "collapsed": {
            "builds": 1,
            "hidden": True,
            "text": "展开优化后素材",
            "expanded": "false",
        },
        "expandedAgain": {"builds": 1, "hidden": False},
        "unique": True,
    }


def test_disclosure_state_survives_dynamic_rerender_but_can_be_reset_on_switch():
    result = _run_contract(
        "(()=>{class Node{constructor(){this.children=[];this.attrs={};this.listeners={};"
        "this.hidden=false;this.textContent='';this.className='';this.id=''}"
        "appendChild(child){this.children.push(child);return child}setAttribute(k,v){this.attrs[k]=String(v)}"
        "addEventListener(k,fn){(this.listeners[k]||(this.listeners[k]=[])).push(fn)}"
        "click(){for(const fn of this.listeners.click||[])fn({type:'click'})}}"
        "global.document={createElement:()=>new Node()};let expanded=false;"
        "const labels={expand:'展开优化后素材',collapse:'收起优化后素材',"
        "expandText:'展开优化后素材',collapseText:'收起优化后素材'};"
        "const render=()=>contract.createDisclosure(labels,()=>new Node(),"
        "{expanded,onChange:(value)=>{expanded=value}});"
        "const first=render();first.children[0].click();const afterUser=expanded;"
        "const polled=render();const afterPoll={expanded,hidden:polled.children[1].hidden};"
        "polled.children[0].click();const afterCollapse=expanded;expanded=false;"
        "const switched=render();return {afterUser,afterPoll,afterCollapse,"
        "afterSwitchHidden:switched.children[1].hidden}})()"
    )
    assert result == {
        "afterUser": True,
        "afterPoll": {"expanded": True, "hidden": False},
        "afterCollapse": False,
        "afterSwitchHidden": True,
    }


def test_postprocess_done_wires_lazy_results_only_when_frames_exist():
    source = APP_JS.read_text(encoding="utf-8")
    assistant = source.split("function renderPpAssistant", 1)[1].split(
        "/* 后处理入口消息", 1
    )[0]
    assert 'if (frames.length) {' in assistant
    assert "ppResultDisclosure(detail, frames)" in assistant
    assert "ppFramesSection(detail, frames)" not in assistant

    result = source.split("function ppResultDisclosure", 1)[1].split(
        "function ppTotalFrames", 1
    )[0]
    assert 'expandText: "展开优化后素材"' in result
    assert 'collapseText: "收起优化后素材"' in result
    assert "ppFramesSection(detail, frames)" in result
    assert "state.ppResultExpanded[detail.id]" in result

    select = source.split("function selectConversation(id)", 1)[1].split(
        "function startPolling", 1
    )[0]
    assert "delete state.ppResultExpanded[id]" in select


def test_detached_media_result_releases_tracked_blob_url():
    result = _run_contract(
        "(()=>{const urls=['blob:keep','blob:stale'];const revoked=[];"
        "contract.releaseTrackedURL('blob:stale',urls,(url)=>revoked.push(url));"
        "return {urls,revoked}})()"
    )
    assert result == {"urls": ["blob:keep"], "revoked": ["blob:stale"]}

    source = APP_JS.read_text(encoding="utf-8")
    grid = source.split("function kfGrid", 1)[1].split(
        "let disclosureSeq", 1
    )[0]
    assert "if (fig.isConnected === false)" in grid
    assert "releaseTrackedURL(url)" in grid
