from pathlib import Path
from zipfile import ZipFile


SKILL = Path(__file__).parents[1] / "skills" / "video-maker" / "SKILL.md"
ARCHIVE = Path(__file__).parents[1] / "web" / "video-maker.zip"


def test_skill_keeps_visual_phase_independent_from_audio_planning():
    text = SKILL.read_text(encoding="utf-8")

    for retained in ("主体可见", "遮挡少"):
        assert retained in text
    assert "multimodal_input.json" in text
    assert "h3_prompt_plan.json" in text
    assert "视觉阶段" in text
    assert "音画阶段" in text
    assert "视觉阶段不得读取任何音频文件" in text
    for visual_contract in (
        "生成一支与源片段时长一致、采用源视频[比例]、[分辨率，默认 720p]",
        "无字幕、贴纸或水印等叠加元素",
        "因果：[先看到动作，再看到变化",
        "不从画面文字推断台词",
    ):
        assert visual_contract in text
    for retired in (
        "捂脸",
        "1秒内快速把手放下",
        "face_hold",
        "Seedance",
    ):
        assert retired not in text


def test_multimodal_phase_emits_structured_plan_not_provider_prompt():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        '"phase": "multimodal_audio"',
        '"eligible": true',
        '"subjects"',
        '"audio_refs"',
        '"dialogue_source_sha256"',
        '"speech_bindings"',
        '"sound_design"',
        "subjects: Array<{ subject_id: SubjectId; picture_refs: NonEmptyArray<Int1>; voice_ref: Int1 | null }>",
        'delivery: "on_screen" | "off_screen_voiceover"',
        "后端确定性编译",
    ):
        assert required in text
    assert "不得直接写供应商最终提示词" in text


def test_multimodal_phase_binds_audio_semantics_and_fails_closed():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "1–3 段",
        "说话人与人物映射",
        "声线参考",
        "现有 dialogue",
        "语言",
        "画外发声",
        "环境声",
        "音效",
        "参考音频不是时间锁",
        '"eligible": false',
        '"reason"',
        "任何必填事实缺失、未知、冲突或无法确认",
        "其余字段固定置空",
    ):
        assert required in text
    for forbidden in (
        "根据时间重叠猜测说话人",
        "参考音频会逐样本复用原音",
        "参考音频等于最终音轨",
    ):
        assert forbidden not in text


def test_audio_phase_preserves_visual_facts_and_does_not_reselect_frames():
    text = SKILL.read_text(encoding="utf-8")

    assert "不得改写视觉事实" in text
    assert "不得重新选关键帧" in text
    assert "不得复制或补造台词文本、时间窗、语言或声音事件" in text


def test_reconcile_is_a_third_isolated_phase_without_dialogue_or_frame_reselection():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "reconcile_after_image_optimization_input.json",
        '"phase": "reconcile_after_image_optimization"',
        "unified_visual_ir.json",
        "第三阶段",
        "existing dialogue 不得进入本阶段",
        "不得重新选择、增删或重排关键帧",
        "不得读取音频",
    ):
        assert required in text


def test_reconcile_has_explicit_dynamic_and_static_authorities():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "原始源帧像素与 PTS 是动作、机位和时序的事实权威",
        "旧视觉提示词仅是动作、机位和时序的低优先级检索证据",
        "冲突时以原始源帧像素与 PTS 为准",
        "canonical 图片计划是新人物和新场景的语义权威",
        "已选优化图及其输出 receipt 是该计划已实现结果的权威",
        "优化图不得反向改写动作、机位、时序或物理关系",
        "旧人物、旧服装、旧场景、旧材质和旧静态外观只作负证据",
        "不得复制 canonical 计划中的目标描述",
    ):
        assert required in text


def test_reconcile_emits_exact_unified_visual_ir_with_receipt_bindings():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "所有字段必填，禁止额外字段",
        "source_evidence_binding",
        "target_static_plan_binding",
        "frame_bindings",
        "preserved_beats",
        "conflicts",
        "old_visual_prompt_sha256",
        "image_plan_sha256",
        "image_verification_sha256",
        "output_receipt_file",
        "output_receipt_sha256",
        "action: { initial_state: NonEmpty; motion: NonEmpty; result_state: NonEmpty }",
        "camera: { shot_scale: NonEmpty; angle: NonEmpty; movement: NonEmpty; composition: NonEmpty; focus: NonEmpty }",
        "timing: { start_source_pts: Integer; end_source_pts: Integer; source_time_base: NonEmpty; pace: NonEmpty; transition: NonEmpty }",
        '"eligible": true',
        '"reason": null',
        '"conflicts": []',
    ):
        assert required in text


def test_reconcile_fails_closed_instead_of_legalizing_bad_optimized_images():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "frame_mapping_missing",
        "optimized_action_changed",
        "physical_support_unclosed",
        "source_static_semantics_leaked",
        "phase_input_conflict",
        "unexpected_dialogue_input",
        "receipt_binding_mismatch",
        "reconciliation_unknown",
        "ReconcileErrorCode =",
        "任一源帧、PTS、优化图或输出 receipt 映射缺失",
        "优化图改变姿态、动作状态、接触、遮挡或因果结果",
        "新场景不能闭合原动作依赖的支撑、接触或可达关系",
        "不得改写动态事实去迎合错误优化图",
        "eligible plan 的 conflicts 必须为空",
        "失败结构",
    ):
        assert required in text


def test_reconcile_ir_does_not_duplicate_static_target_or_final_prompt_text():
    text = SKILL.read_text(encoding="utf-8")
    reconcile = text.split("## 4. 优化后语义协调阶段", maxsplit=1)[1]

    for required in (
        "禁止二次自由文本真源",
        "人物、服装、场景、材质与光色只通过 target_static_plan_binding",
        "只描述动态事实",
    ):
        assert required in reconcile
    for forbidden in (
        "visual_prompt:",
        "replacement_identity:",
        "replacement_scene:",
        "final_prompt:",
        "dialogue:",
    ):
        assert forbidden not in reconcile


def test_reconcile_preserves_exact_source_pts_and_has_no_circular_output_receipt():
    text = SKILL.read_text(encoding="utf-8")
    reconcile = text.split("## 4. 优化后语义协调阶段", maxsplit=1)[1]

    for required in (
        "source_pts: Integer",
        "source_time_base: NonEmpty",
        "起止 PTS 必须逐字取自首尾 frame_refs",
        "image verification receipt 绑定完整且有序的 output receipt SHA 集合",
        "output receipt 只绑定 image plan、源帧和优化图",
    ):
        assert required in reconcile
    for lossy_or_circular in (
        "source_pts_ms",
        "start_pts_ms",
        "end_pts_ms",
        "output receipt 逐字绑定同一 `image_verification_sha256`",
    ):
        assert lossy_or_circular not in reconcile


def test_static_schema_matches_h3_multimodal_adapter_items():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "所有字段必填，禁止额外字段",
        'audio_refs: Array<{ audio_index: Int1; purpose: "voice" | "ambience" | "effect"; subject_id: SubjectId | null }>',
        "speech_bindings: Array<",
        "ambience_refs: Array<{ audio_index: Int1; description: NonEmpty }>",
        "effects: Array<{ audio_index: Int1; description: NonEmpty }>",
        "`line_index` 从 1 开始连续无缺号",
        "不得另写 `order/text/start_s/end_s/narration text`",
        "画外发声不得出现 `subject_id`",
    ):
        assert required in text
    for incompatible in (
        "ambience_refs: Array<{ order:",
        "effects: Array<{ order:",
        "effects: Array<{ audio_ref:",
    ):
        assert incompatible not in text


def test_static_schema_matches_adapter_reference_and_subject_guards():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "图片、音频外部编号从 1 开始",
        "不得按数组位置重编号",
        "`picture_refs` 是非空、升序、无重复的多值数组",
        "不同 `subject_id` 不得复用同一图片编号",
        "冻结输入必须已给出连续的 `S1…Sn`",
        "同画面的静默人物可以保留",
        "同一 `audio_index` 只能有一种 `purpose`",
    ):
        assert required in text


def test_audio_is_reference_only_and_backend_fields_stay_out_of_plan():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "只按 reference 语义使用",
        "不是供应商 PTS 硬锁",
        "不是未经确认的 speaker-face 推断",
        "不是最终音轨",
        "不进入语义计划",
    ):
        assert required in text
    for forbidden in (
        '"speaker_id"',
        '"face_id"',
        '"start_pts"',
        '"end_pts"',
        '"bytes"',
        '"hash"',
        '"format"',
        '"fully_copy"',
        '"partially_copy"',
    ):
        assert forbidden not in text


def test_skill_contains_no_reusable_audio_sample_words_or_word_count_kpi():
    text = SKILL.read_text(encoding="utf-8")

    for forbidden in (
        '"subject_id": "S1"',
        '"language": "Chinese"',
        "输入逐字原文",
        "逐字保留输入的声音描述",
        "我会准时回来",
        "First batch of the morning",
        "350-500",
        "350–500",
        "字数",
    ):
        assert forbidden not in text


def test_download_archive_matches_skill_source():
    source = SKILL.parent
    files = sorted(path for path in source.rglob("*") if path.is_file())

    with ZipFile(ARCHIVE) as archive:
        expected = sorted(f"video-maker/{path.relative_to(source).as_posix()}" for path in files)
        assert sorted(name for name in archive.namelist() if not name.endswith("/")) == expected
        for path, name in zip(files, expected, strict=True):
            assert archive.read(name) == path.read_bytes()


def test_prompt_contract_uses_source_duration_without_numeric_seconds():
    text = SKILL.read_text(encoding="utf-8")

    assert "与源片段时长一致" in text
    assert "[目标时长] 秒" not in text
    assert "不写具体秒数" in text
