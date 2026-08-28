from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "video-prompt-fusion" / "SKILL.md"
AGENT = ROOT / "skills" / "video-prompt-fusion" / "agents" / "openai.yaml"


def _skill() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_has_minimal_identity_and_invocation_metadata():
    text = _skill()
    agent = AGENT.read_text(encoding="utf-8")

    assert text.startswith("---\nname: video-prompt-fusion\n")
    assert "description: " in text.split("---", 2)[1]
    assert "TODO" not in text
    assert 'display_name: "Video Prompt Fusion"' in agent
    assert (
        'short_description: "Fuse confirmed frames and frozen audio into final prompts"'
        in agent
    )
    assert "$video-prompt-fusion" in agent
    assert sorted(
        path.relative_to(SKILL.parent).as_posix()
        for path in SKILL.parent.rglob("*")
        if path.is_file()
    ) == ["SKILL.md", "agents/openai.yaml"]


def test_input_contract_has_only_the_four_content_classes():
    text = _skill()

    for required in (
        "`work/multimodal_input.json`",
        'schema: "duet.video-prompt-fusion-input"',
        "version: 2",
        "segments: NonEmptyArray<SegmentInput>",
        "index: Int1",
        "new_keyframes: NineOrdered<KeyframeReceipt>",
        "old_video_prompt: FrozenText",
        "image_optimization_prompt: NineOrdered<FrozenFramePrompt>",
        "audio_content: FrozenAudioContent",
        'KeyframeReceipt = { order: Int1; path: NonEmpty; sha256: Sha256; source_time_s: Number; source_scene_id: NonEmpty; transition: { type: "start" | "continuous" | "hard_cut"; at_s: Number | null } }',
        "FrozenText = { text: NonEmpty; sha256: Sha256 }",
        "FrozenFramePrompt = { order: Int1; text: NonEmpty; sha256: Sha256 }",
        'FrozenAudioContent = { lines_json: NonEmptyJsonText; lines_sha256: Sha256; voice_references: Array<VoiceReference>; music_policy: "forbid" }',
        "AudioLine = { order: Int1; text: NonEmpty; start_s: Number; end_s: Number; delivery: NonEmpty; voice_ref: Int1 | null }",
        'VoiceReference = { voice_ref: Int1; path: NonEmpty; sha256: Sha256; purpose: "voice" }',
        "除结构字段 `schema/version/segments/index` 外，段内恰好只有上述四类输入",
        '`music_policy` 必须是 exact 字符串 `"forbid"`',
    ):
        assert required in text


def test_v1_is_read_only_and_v2_is_the_only_create_contract():
    text = _skill()

    for required in (
        "`version=1` 仅允许历史只读",
        "不得用 v1 创建或覆盖输出",
        "`version=2` 是唯一可创建合同",
        "不得迁移、补写或猜测 `music_policy`",
    ):
        assert required in text


def test_hash_order_and_segment_scope_are_closed_world():
    text = _skill()

    for required in (
        "segment `index` 从 1 连续升序",
        "每段恰好 9 张",
        "每条图片优化提示词与同 `order` 新关键帧一一对应",
        "不得选帧、删帧、补帧或重排",
        "按 `(segment index, keyframe order)` 全局严格递增",
        "只有项目第一张",
        '`type="start"` 且 `at_s=source_time_s`',
        "其余关键帧禁止 `start`",
        '`continuous` 的 `at_s` 必须为 `null`',
        '`continuous` 只表示没有 source hard cut',
        '`continuous` 不授权静态机位、构图或 camera movement',
        '`hard_cut` 的 `at_s` 必须是有限非负数',
        "前一张 `source_time_s < at_s <=` 当前张 `source_time_s`",
        "source scene 改变时必须是 `hard_cut`",
        "source scene 不变时必须是 `continuous`",
        "硬切前后 scene ids 必须逐值等于相邻关键帧的 `source_scene_id`",
        "硬切后的当前关键帧是新 anchor",
        "图片原始 bytes 的 SHA-256",
        "UTF-8 `text` bytes 的 SHA-256",
        "UTF-8 `audio_content.lines_json` exact bytes 的 SHA-256",
        "每个非 null `voice_ref` 必须唯一解析到同值 `voice_references[].voice_ref`",
        '`lines_json="[]"`',
        "不得跨 segment 借用",
        "任一 schema、数量、索引、顺序、路径或哈希不匹配时，不写输出文件",
    ):
        assert required in text

    assert '"same_camera"' not in text
    assert '"camera_motion"' not in text


def test_visual_authorities_are_explicit_and_non_overlapping():
    text = _skill()

    for required in (
        "新人物、新场景、新对象、服装、材质和空间结构只以新关键帧为准",
        "锚点取景、裁切、构图、景别、机位和静态镜头属性只以新关键帧为准",
        "人物、项链、背景、景别、裁切和静态机位等静态事实由新关键帧独占权威",
        "旧视频提示词只允许在同一 `hard_cut` 区间内贡献动作顺序、因果关系、camera movement type 和相对节奏",
        "不得贡献构图、取景、裁切、景别、机位或任何静态镜头属性",
        "泛化的“镜头”或“摄影”措辞不构成权限",
        "源硬切类型和绝对时点只以 `new_keyframes[].transition` 为准",
        "图片优化提示词只解释替换目标和保持约束",
        "删除旧视频提示词中与新关键帧冲突的旧人物、旧场景、旧对象、旧服装、旧材质和旧空间结构",
        "不得从静态关键帧反推或改写动作顺序、因果关系、camera movement type 或相对节奏",
        "不得引入四类输入均未支持的新事实",
        "内容冲突不构成拒绝",
        "技术有效的四类输入必须为每个 segment 产出一个最终提示词",
    ):
        assert required in text

    assert "如果新关键帧无法承载旧提示词" not in text


def test_old_static_content_requires_positive_new_evidence_and_mapping():
    text = _skill()

    for required in (
        "`old_video_prompt` 的闭世界白名单只有同一 hard-cut 区间内的动作顺序、因果关系、camera movement type 和相对节奏",
        "即使旧静态与新关键帧不显式冲突",
        "新关键帧中独立可见",
        "同 `order` 的图片优化提示词明确映射",
        "两个条件必须同时满足",
        "否则不得进入 `final_prompt` 的 `<VISUAL>`",
        "静态描述只能取自新关键帧",
    ):
        assert required in text

    for forbidden_authority in (
        "`old_video_prompt` 只允许贡献动作、镜头、构图",
        "动作、镜头、构图、相对节奏和非硬切时间关系只以旧视频提示词为准",
        "继续保留其他动作、镜头、构图、节奏和时间关系",
    ):
        assert forbidden_authority not in text


def test_old_dynamic_is_confined_to_each_hard_cut_interval():
    text = _skill()

    for required in (
        "不得跨越 `hard_cut` 传播动作、因果或 camera movement",
        "每个 hard-cut 区间独立融合",
        "只保留起止证据均落在同一 hard-cut 区间内的旧动态",
        "跨越硬切的连续 zoom、push、pan、tracking 或 morph",
        "必须整体删除",
        "不得截断、拆分或重分配到切点任一侧",
        "含多个 hard-cut 区间且缺少可定位边界的旧动态也删除",
        "无法唯一归属一个 hard-cut 区间",
        "不得把切前主体、构图或运动状态延续到切后",
        "`hard_cut.at_s` 不得改写为其他关键帧的 `source_time_s`",
        "硬切记录及切后当前 anchor",
    ):
        assert required in text

    assert "必须在切点终止，不得在切后自动续写" not in text


def test_audio_is_copied_exactly_inside_the_final_prompt():
    text = _skill()

    for required in (
        "音频文本、时间、delivery 和 voice_ref 逐值原样保持",
        "不得翻译、润色、纠错、合并、拆分、重排或补写音频行",
        "<AUDIO_CONTENT_JSON>",
        "</AUDIO_CONTENT_JSON>",
        "逐字复制 `audio_content.lines_json`",
        "音频块之外不得再复述或改写音频内容",
        "<MUSIC_POLICY>forbid</MUSIC_POLICY>",
        "恰好一次",
        "只允许改写 `<VISUAL>`",
        "两个合同块都逐 byte 保持",
    ):
        assert required in text

    assert (
        "<AUDIO_CONTENT_JSON>{lines_json}</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
    ) in text
    assert "opening tag 的下一 byte 必须是 `lines_json` 首 byte" in text
    assert "closing tag 紧随 `lines_json` 末 byte" in text
    assert (
        "<AUDIO_CONTENT_JSON>\naudio_content.lines_json\n</AUDIO_CONTENT_JSON>"
        not in text
    )


def test_keyframe_timeline_is_canonical_and_context_immutable():
    text = _skill()

    for required in (
        "<KEYFRAME_TIMELINE_JSON>{keyframe_timeline_json}</KEYFRAME_TIMELINE_JSON>",
        "逐项投影 `order/source_time_s/source_scene_id/transition`",
        "字段顺序固定为",
        "UTF-8 compact JSON",
        "不得加入 `path`、`sha256` 或其他字段",
        "timeline 块逐 byte 保持",
    ):
        assert required in text

    assert (
        "</VISUAL>\n"
        "<KEYFRAME_TIMELINE_JSON>{keyframe_timeline_json}</KEYFRAME_TIMELINE_JSON>\n"
        "<AUDIO_CONTENT_JSON>{lines_json}</AUDIO_CONTENT_JSON>\n"
        "<MUSIC_POLICY>forbid</MUSIC_POLICY>"
    ) in text


def test_clean_reference_proof_stays_out_of_the_skill_schema():
    text = _skill()

    assert "clean reference 的资格证明只由后端 frozen receipt 负责" in text
    for forbidden_field in (
        "clean_reference_proof:",
        "clean_voice_proof:",
        "reference_receipt:",
    ):
        assert forbidden_field not in text


def test_output_is_one_ordered_final_prompt_per_input_segment():
    text = _skill()

    for required in (
        "整个 ordered segments 数组只执行一次项目级调用",
        "同一次项目调用按序处理全部 segment",
        "`N=1` 与 `N>1` 使用同一合同",
        "`work/h3_prompt_plan.json`",
        'schema: "duet.video-prompt-fusion-output"',
        "version: 2",
        "input_sha256: Sha256",
        "segments: NonEmptyArray<{ index: Int1; final_prompt: NonEmpty }>",
        "输入描述符 exact bytes 的 SHA-256",
        "输出 segments 与输入 segments 一一对应且顺序相同",
        "不得增加、删除、合并或拆分 segment",
        "最终提示词",
    ):
        assert required in text


def test_skill_does_not_delegate_or_create_an_extra_stage():
    text = _skill().lower()

    for forbidden in (
        "binding skill",
        "audio skill",
        "speaker skill",
        "speaker-visibility",
        "reconcile_after_image_optimization",
        "调用另一个 skill",
        "provider post",
    ):
        assert forbidden not in text
