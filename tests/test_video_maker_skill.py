from pathlib import Path
from zipfile import ZipFile


SKILL = Path(__file__).parents[1] / "skills" / "video-maker" / "SKILL.md"
ARCHIVE = Path(__file__).parents[1] / "web" / "video-maker.zip"
HUMAN = (
    Path(__file__).parents[1]
    / "docs"
    / "human"
    / "features"
    / "conversation-task"
    / "behaviors"
    / "postprocess.md"
)


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


def test_speaker_visibility_is_a_strict_priority_phase_with_isolated_reads():
    text = SKILL.read_text(encoding="utf-8")

    for required in (
        "speaker_visibility_input.json",
        "speaker_visibility_output.json",
        'phase: "speaker_visibility"',
        "存在时严格优先执行",
        "不得读取 `work/multimodal_input.json`",
        "不得读取 `work/reconcile_after_image_optimization_input.json`",
        "只读取描述符逐字绑定的采样帧、联系表和 PERSON identity refs",
    ):
        assert required in text


def test_speaker_visibility_input_matches_the_backend_exact_schema():
    text = SKILL.read_text(encoding="utf-8")
    visibility = text.split("## 5. 发声人物可见性阶段", maxsplit=1)[1]

    for required in (
        'schema: "duet.speaker-visibility-input"',
        "source: { sha256: Sha256; duration_pts: PositivePts; time_base: TimeBase }",
        'algorithm: "decoded_pts_nearest_v1"',
        "cadence_fps: 8",
        "max_unobserved_gap_pts: PositivePts",
        "endpoint_shrink_intervals: 1",
        "decoded_frame_pts: NonEmptyArray<Pts>",
        "cut_pts: Array<PositivePts>",
        'cut_source: { path: "scenes.json"; sha256: Sha256 }',
        "frames: NonEmptyArray<{ order: Int1; path: NonEmpty; sha256: Sha256; pts: Pts; cut_before: Boolean }>",
        "contact_sheets: NonEmptyArray<{ order: Int1; path: NonEmpty; sha256: Sha256; frame_orders: NonEmptyArray<Int1> }>",
        "persons: NonEmptyArray<{ person_id: PersonId; identity_refs: NonEmptyArray<{ path: NonEmpty; sha256: Sha256 }> }>",
        "on_screen_subjects: NonEmptyArray<SubjectId>",
        "所有字段必填，禁止额外字段",
        "禁止另猜切镜",
    ):
        assert required in visibility


def test_speaker_visibility_output_is_exact_frame_classification_not_timing():
    text = SKILL.read_text(encoding="utf-8")
    visibility = text.split("## 5. 发声人物可见性阶段", maxsplit=1)[1]

    for required in (
        'schema: "duet.speaker-visibility-output"',
        "input_sha256: Sha256",
        "subject_person_mapping: NonEmptyArray<{ subject_id: SubjectId; person_id: PersonId }>",
        "frames: NonEmptyArray<{ order: Int1; visible_person_ids: Array<PersonId>; lip_verifiable_person_ids: Array<PersonId> }>",
        "逐字节 SHA-256",
        "不得重排或重序列化 JSON 后再计算",
        "后端另行冻结本次使用的 Skill 原始字节及 SHA",
        "Skill 不生成时间窗、不合并区间、不收缩端点、不生成 timing 或 receipt",
    ):
        assert required in visibility
    output_schema = visibility.split("VisibilityOutput = {", maxsplit=1)[1].split(
        "```", maxsplit=1
    )[0]
    for forbidden in (
        "eligible:",
        "reason:",
        "start_pts:",
        "end_pts:",
        "evidence_keyframes:",
        "receipt:",
    ):
        assert forbidden not in output_schema


def test_speaker_visibility_exhausts_real_samples_without_interpolation():
    text = SKILL.read_text(encoding="utf-8")
    visibility = text.split("## 5. 发声人物可见性阶段", maxsplit=1)[1]

    for required in (
        "8 FPS 目标节拍选中的真实 decoded frame PTS",
        "每个输入 sample 恰有一个同 order 的输出项",
        "逐样本穷举",
        "lip_verifiable_person_ids 必须是 visible_person_ids 的子集",
        "无法唯一确认时写空数组",
        "联系表只用于导航",
        "只能以该 sample 自身图像字节作事实",
        "不得用相邻 sample 补证",
        "不得跨 `cut_before=true` 补证",
        "不得在未知空洞之间插值",
    ):
        assert required in visibility


def test_speaker_mapping_fails_closed_when_subject_identity_is_not_unique():
    text = SKILL.read_text(encoding="utf-8")
    visibility = text.split("## 5. 发声人物可见性阶段", maxsplit=1)[1]

    for required in (
        "subject_person_mapping 必须与 on_screen_subjects 等长且同序",
        "PERSON 映射必须一对一",
        "单一 on-screen subject 且 roster 只有一个 PERSON",
        "多人映射不能从冻结 identity refs 与采样证据唯一证明时",
        "不得写 `work/speaker_visibility_output.json`",
        "不得按 subject/PERSON 数组位置",
        "不得按嘴部运动",
        "不得按台词文本或台词时间窗反推",
    ):
        assert required in visibility


def test_speaker_visibility_none_and_offscreen_skip_before_skill():
    text = SKILL.read_text(encoding="utf-8")
    human = HUMAN.read_text(encoding="utf-8")

    for required in (
        "none 或全部 offscreen",
        "不得创建 `work/speaker_visibility_input.json`",
        "不得调用本阶段 Skill",
        "后端机械合并相邻 verified samples、按 cut/空洞断开并收缩窗端点",
    ):
        assert required in text
    for required in (
        "speaker_visibility",
        "8 FPS",
        "真实 decoded frame PTS",
        "不接收 dialogue 文本或时间窗",
        "none/offscreen 不调用 Skill",
        "后端机械合窗和收缩端点",
    ):
        assert required in human


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
        "TimeBase = { numerator: Int1; denominator: Int1 }",
        "source_time_base: TimeBase",
        "timing: { start_source_pts: Integer; end_source_pts: Integer; source_time_base: TimeBase; pace: NonEmpty; transition: NonEmpty }",
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
        "source_time_base: TimeBase",
        "起止 PTS 必须逐字取自首尾 frame_refs",
        "image verification receipt 绑定完整且有序的 output receipt SHA 集合",
        "output receipt 只绑定 image plan、源帧和优化图",
    ):
        assert required in reconcile
    for lossy_or_circular in (
        "source_pts_ms",
        "start_pts_ms",
        "end_pts_ms",
        "source_time_base: NonEmpty",
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
