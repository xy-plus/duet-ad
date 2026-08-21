from test_web_h3_contract import APP_JS, _run_contract


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
        "fitRequired:false,isLong:true,planReceipt:'c'.repeat(64)})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-long",
        "dialogue_mode": "auto",
        "fit_mode": "none",
        "expected_plan_receipt": "c" * 64,
    }


def test_short_submit_payload_remains_unchanged():
    payload = _run_contract(
        "contract.buildSubmitPayload({clientRequestId:'request-short',dialogueMode:'custom',"
        "linesText:'0 - 1 | hello',fitRequired:false,isLong:false,planReceipt:'d'.repeat(64)})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-short",
        "dialogue_mode": "custom",
        "fit_mode": "none",
        "lines": [{"start_s": 0, "end_s": 1, "text": "hello"}],
    }


def test_long_resume_reuses_attempt_and_current_plan_receipt():
    payload = _run_contract(
        "contract.buildResumePayload({duration_s:30,segment_count:2,plan_receipt:'f'.repeat(64),"
        "generation:{status:'resume_required',client_request_id:'request-old'},"
        "dialogue:{mode:'none',lines:[]},fit_mode:'none'})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-old",
        "dialogue_mode": "none",
        "fit_mode": "none",
        "expected_plan_receipt": "f" * 64,
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
        "generation:{status:'failed',stage:'stitch',client_request_id:'request-old'}})"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-old",
        "dialogue_mode": "none",
        "fit_mode": "pad",
        "expected_plan_receipt": "a" * 64,
    }


def test_failed_segment_retry_uses_new_request_with_frozen_parameters():
    payload = _run_contract(
        "contract.buildLongRetryPayload({duration_s:30,segment_count:2,"
        "plan_receipt:'b'.repeat(64),fit_required:true,fit_mode:'crop',"
        "dialogue:{mode:'auto',lines:[]},generation:{status:'failed',stage:'h3',"
        "client_request_id:'request-old',segments:[{index:1,status:'succeeded'},"
        "{index:2,status:'failed'}]}},'request-new')"
    )
    assert payload == {
        "confirm": True,
        "client_request_id": "request-new",
        "dialogue_mode": "auto",
        "fit_mode": "crop",
        "expected_plan_receipt": "b" * 64,
    }


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
        "个付费 H3 子任务",
        "重试生成",
        "重试拼接",
        "连续性为 best effort",
        "失败时只重做失败段",
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


def test_long_segment_prompt_copy_is_explicitly_frozen():
    source = APP_JS.read_text(encoding="utf-8")
    assert "逐段冻结的 H3 提示词将提交生成" in source
    assert "H3 源提示词将直接提交生成" in source
    assert (
        ': longContract.isLong\n'
        '      ? "逐段冻结的 H3 提示词将提交生成"\n'
        '      : "H3 源提示词将直接提交生成"'
    ) in source
