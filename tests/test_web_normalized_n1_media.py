from test_web_h3_contract import APP_JS, _run_contract


def test_normalized_n1_uses_authoritative_root_keyframe_paths():
    result = _run_contract(
        "(()=>{const segment={index:1,keyframes:['01.png','02.png'],"
        "keyframe_paths:['keyframes/01.png','keyframes/02.png']};"
        "const detail={segment_count:1,segments:[segment]};"
        "return contract.authoritativeSegmentKeyframePaths(detail,segment)})()"
    )
    assert result == ["keyframes/01.png", "keyframes/02.png"]


def test_normalized_n1_maps_bare_postprocess_frames_to_root_paths():
    result = _run_contract(
        "(()=>{const segment={index:1,keyframes:['01.png','02.png'],"
        "keyframe_paths:['keyframes/01.png','keyframes/02.png']};"
        "const detail={segment_count:1,segments:[segment]};"
        "return contract.authoritativePostprocessFrameGroups(detail,['01.png','02.png'])})()"
    )
    assert result == [{
        "index": 1,
        "names": ["01.png", "02.png"],
        "paths": ["postprocessed/01.png", "postprocessed/02.png"],
    }]


def test_n_greater_than_one_never_falls_back_to_root_media():
    result = _run_contract(
        "(()=>{const one={index:1,keyframes:['01.png'],"
        "keyframe_paths:['segments/1/work/keyframes/01.png']};"
        "const two={index:2,keyframes:['01.png'],"
        "keyframe_paths:['segments/2/work/keyframes/01.png']};"
        "const detail={segment_count:2,segments:[one,two]};return {"
        "one:contract.authoritativeSegmentKeyframePaths(detail,one),"
        "segmented:contract.authoritativePostprocessFrameGroups(detail,["
        "'segments/1/work/postprocessed/01.png','segments/2/work/postprocessed/01.png']),"
        "bare:contract.authoritativePostprocessFrameGroups(detail,['01.png']),"
        "rootKeyframes:(()=>{const root={...one,keyframe_paths:['keyframes/01.png']};"
        "return contract.authoritativeSegmentKeyframePaths({segment_count:2,segments:[root,two]},root)})()"
        "}})()"
    )
    assert result == {
        "one": ["segments/1/work/keyframes/01.png"],
        "segmented": [
            {
                "index": 1,
                "names": ["01.png"],
                "paths": ["segments/1/work/postprocessed/01.png"],
            },
            {
                "index": 2,
                "names": ["01.png"],
                "paths": ["segments/2/work/postprocessed/01.png"],
            },
        ],
        "bare": [],
        "rootKeyframes": [],
    }


def test_authoritative_media_contract_rejects_traversal_and_partial_shapes():
    result = _run_contract(
        "(()=>{const segment={index:1,keyframes:['01.png'],"
        "keyframe_paths:['keyframes/01.png']};"
        "const detail={segment_count:1,segments:[segment]};"
        "const traversal={...segment,keyframe_paths:['keyframes/../meta.json']};return {"
        "traversal:contract.authoritativeSegmentKeyframePaths("
        "{segment_count:1,segments:[traversal]},traversal),"
        "bareTraversal:contract.authoritativePostprocessFrameGroups(detail,['../meta.json']),"
        "mismatch:contract.authoritativePostprocessFrameGroups(detail,['01.png','extra.png']),"
        "missingPaths:contract.authoritativeSegmentKeyframePaths(detail,{index:1,keyframes:['01.png']})"
        "}})()"
    )
    assert result == {
        "traversal": [],
        "bareTraversal": [],
        "mismatch": [],
        "missingPaths": [],
    }


def test_frame_prompts_render_all_nine_as_read_only_text_and_fail_closed():
    result = _run_contract(
        "(()=>{const prompts=Array.from({length:9},(_,i)=>({"
        "frame_name:String(i+1).padStart(2,'0')+'.png',text:'当前 '+(i+1),"
        "default_text:'默认 '+(i+1),sha256:'a'.repeat(64)}));return {"
        "text:contract.readOnlyImageFramePromptText(prompts),"
        "short:contract.readOnlyImageFramePromptText(prompts.slice(0,8)),"
        "traversal:contract.readOnlyImageFramePromptText(["
        "{...prompts[0],frame_name:'../01.png'},...prompts.slice(1)])}})()"
    )
    assert result["text"].startswith("01.png\n当前 1\n\n02.png\n当前 2")
    assert result["text"].endswith("09.png\n当前 9")
    assert result["short"] is None
    assert result["traversal"] is None


def test_legacy_media_fetch_keeps_bearer_api_and_does_not_guess_segment_paths():
    source = APP_JS.read_text(encoding="utf-8")
    api = source.split("async function api(path", 1)[1].split(
        "async function apiJSON", 1
    )[0]
    grid = source.split("function kfGrid", 1)[1].split(
        "let disclosureSeq", 1
    )[0]
    segments = source.split("function renderSegments(detail)", 1)[1].split(
        "function resetSegmentProductsDisclosure", 1
    )[0]
    assert 'headers["Authorization"] = "Bearer " + state.token' in api
    assert "apiBlobURL(" in grid
    assert "encodedMediaPath(path)" in grid
    viewer = source.split("function frameViewerEntries", 1)[1].split(
        "function selectedFrameEntry", 1
    )[0]
    assert "authoritativeSegmentKeyframePaths(detail, segment)" in viewer
    assert '"segments/" + n + "/work/keyframes"' not in segments
