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
        'short_description: "Fuse confirmed frames into ordered visual prose"'
        in agent
    )
    assert "$video-prompt-fusion" in agent
    assert sorted(
        path.relative_to(SKILL.parent).as_posix()
        for path in SKILL.parent.rglob("*")
        if path.is_file()
    ) == ["SKILL.md", "agents/openai.yaml"]


def test_input_contract_keeps_the_existing_four_content_classes():
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
        'KeyframeReceipt = { order: Int1; path: NonEmpty; sha256: Sha256; segment_time_s: Number; source_scene_id: NonEmpty; transition: { type: "start" | "continuous" | "hard_cut"; at_segment_s: Number | null } }',
        "FrozenText = { text: NonEmpty; sha256: Sha256 }",
        "FrozenFramePrompt = { order: Int1; text: NonEmpty; sha256: Sha256 }",
        'FrozenAudioContent = { lines_json: NonEmptyJsonText; lines_sha256: Sha256; voice_references: []; music_policy: "forbid" }',
        "AudioLine = { order: Int1; text: NonEmpty; start_s: Number; end_s: Number; delivery: NonEmpty; voice_ref: null }",
        "除结构字段 `schema/version/segments/index` 外，段内恰好只有上述四类输入",
        '`music_policy` 必须是 exact 字符串 `"forbid"`',
    ):
        assert required in text

    for forbidden_new_field in (
        "global_element_index:",
        "global_replacement_mapping:",
        "composite_reference_image:",
        "reference_board:",
    ):
        assert forbidden_new_field not in text


def test_global_replacement_binding_is_carried_inside_existing_frame_prompts():
    text = _skill()

    for required in (
        "四类冻结输入和输入输出 schema 保持不变",
        "不得新增第五类输入",
        "`image_optimization_prompt[].text`",
        "全项目共享替换参考板绑定：{stable_key} -> TILE_XX -> {replacement_description}",
        "同一帧实际需要替换的所有人物、实体和场景",
        "`stable_key`、`tile_id` 与 `replacement_description`",
        "`new_keyframes` 已落实该绑定的视觉结果",
        "不得读取合并参考图或其他新增文件",
    ):
        assert required in text


def test_shared_identity_context_never_propagates_actions_or_cuts():
    text = _skill()

    for required in (
        "同一 `stable_key` 跨 segment 复用同一人物身份、对象设计或环境设计",
        "跨段共享只约束稳定身份、对象和环境设计",
        "不得跨 segment 传播动作、动作阶段、因果、camera movement、hard-cut 切点或无证据剧情",
        "每段动作和剧情仍只来自该段 `old_video_prompt`",
        "每段 hard-cut 仍只来自该段 `new_keyframes[].transition`",
    ):
        assert required in text


def test_fusion_explicitly_preserves_five_continuities_and_material_binding():
    text = _skill()

    for required in (
        "连续性、剧情连贯性、人物一致性、动作一致性、环境一致性",
        "`old_video_prompt` 的段内动态骨架",
        "`new_keyframes` 的冻结视觉事实",
        "`image_optimization_prompt` 的逐帧替换绑定",
        "`audio_content` 的冻结台词时间线",
        "按 segment、frame order 和 hard-cut 区间精确对齐",
        "输出 schema 和 `visual` 字段保持不变",
        "供 Context IR 优化",
        "逐段独立生成视频",
    ):
        assert required in text


def test_global_binding_adds_no_gate_retry_or_fallback():
    text = _skill()

    for required in (
        "全局绑定不增加新的技术校验条件",
        "不产生新的门禁、拒绝、重试、回退或 workflow 分支",
        "继续使用现有单次融合路径",
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
        "在该段按 keyframe `order` 严格递增",
        "每段 order 1",
        '`segment_time_s=0`、`type="start"`、`at_segment_s=0`',
        "输入和输出中禁止出现全局 `source_time_s` 或 `at_s`",
        '`continuous` 的 `at_segment_s` 必须为 `null`',
        '`continuous` 只表示没有 source hard cut',
        '`continuous` 不授权静态机位、构图或 camera movement',
        '`hard_cut` 的 `at_segment_s` 必须是有限非负数',
        "前一张 `segment_time_s < at_segment_s <=` 当前张 `segment_time_s`",
        "source scene 改变时必须是 `hard_cut`",
        "source scene 不变时必须是 `continuous`",
        "只在每个 segment 内比较相邻关键帧",
        "硬切后的当前关键帧是新 anchor",
        "图片原始 bytes 的 SHA-256",
        "UTF-8 `text` bytes 的 SHA-256",
        "UTF-8 `audio_content.lines_json` exact bytes 的 SHA-256",
        "`voice_ref` 必须逐行保持为 `null`",
        "`voice_references` 必须是 `[]`",
        "source audio 只属于上游 ASR/YAMNet 分析证据",
        "只有 `new_keyframes[].path` 是本 Skill 可读取的文件路径",
        "输入不存在可读音频路径",
        '`lines_json="[]"`',
        "不得跨 segment 借用",
        "任一 schema、数量、索引、顺序、路径或哈希不匹配时，不写输出文件",
    ):
        assert required in text

    assert '"same_camera"' not in text
    assert '"camera_motion"' not in text
    assert "路径必须解析到当前工作目录内已列出的普通文件" not in text


def test_visual_authorities_are_explicit_and_non_overlapping():
    text = _skill()

    for required in (
        "锚点取景、裁切、构图、景别、机位和静态镜头属性只以新关键帧为准",
        "人物身份与外观、人物附属物、服装与可穿戴物、手持物、关键商品、其他对象、场景、材质和空间结构只以新关键帧为准",
        "所有静态事实均由新关键帧独占权威",
        "旧视频提示词只允许在同一 `hard_cut` 区间内贡献动作顺序、因果关系、camera movement type 和相对节奏",
        "不得贡献构图、取景、裁切、景别、机位或任何静态镜头属性",
        "泛化的“镜头”或“摄影”措辞不构成权限",
        "源硬切类型和绝对时点只以 `new_keyframes[].transition` 为准",
        "图片优化提示词只解释替换目标和保持约束",
        "删除旧视频提示词中与新关键帧冲突的旧人物、旧场景、旧对象、旧服装、旧材质和旧空间结构",
        "不得从静态关键帧反推或改写动作顺序、因果关系、camera movement type 或相对节奏",
        "不得引入四类输入均未支持的新事实",
        "内容冲突和语义评分都不构成拒绝",
        "技术可读取的四类输入必须为每个 segment 的每个冻结 hard-cut 区间产出一条视觉文本",
        "不得据此拒绝、重试、改走另一 workflow 或不写输出",
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
        "否则不得进入 `visual`",
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
        "`hard_cut.at_segment_s` 及切后当前 anchor 必须逐值保持",
    ):
        assert required in text

    assert "必须在切点终止，不得在切后自动续写" not in text


def test_visual_text_uses_closed_inclusive_hard_cut_ranges():
    text = _skill()

    for required in (
        "先在内部按 `transition` 建立帧范围表；范围表不输出",
        "第一个 `hard_cut` 当前 `order-1`",
        "每个后续区间从该 `hard_cut` 当前 `order` 开始",
        "标有 `hard_cut` 的当前帧属于切后区间",
        "是该区间唯一的首 anchor",
        "`visual[N]` 只能引用第 N 个范围内",
        "区间外的帧不得作为该条 visual 的开头、结尾、对照、过渡或背景",
        "切后区间不得叙述切前主体、服装、构图、动作状态",
        "旧提示词若跨越硬切描述连续动作，只保留各自区间内有两端证据的部分",
        "区间只有一帧时，只写该帧可见 anchor",
        "标有 `hard_cut` 的帧仍带有切前主体或旧状态",
        "将其视为 outgoing residue",
        "不在切后 visual 中叙述",
        "优先以 transition 元数据和同一 `source_scene_id` 的首个清晰帧建立新 anchor",
        "没有清晰人物帧，只写该区间能确认的环境或物件",
    ):
        assert required in text


def test_dynamic_inheritance_requires_two_sided_new_frame_evidence():
    text = _skill()

    for required in (
        "旧动态的起点状态和终点状态都由该区间的新关键帧独立支持",
        "只支持一端",
        "不得外推、补全或保留该动态",
        "`continuous` 只维持 hard-cut 区间归属",
        "不得自由扩展运动",
    ):
        assert required in text


def test_interval_ledger_closes_current_bindings_and_audio_scope():
    text = _skill()

    for required in (
        "每个 hard-cut 区间先建立内部证据账本（不输出）",
        "逐帧记录当前帧直接可见的 `stable_key`/`TILE_XX` 及可见范围",
        "仅消费同一当前帧同时映射且可见的绑定",
        "不属于当前区间的前一帧、相邻区间或其他 segment 的人物、服装、环境、对象、动作补当前区间",
        "边缘、局部、遮挡、模糊或紧裁切只能写可见片段",
        "动态短语必须有本区间起止帧支持并停在最后支持状态",
        "`audio_content` 仅作为冻结台词事件的冲突边界，不生成声音、拟声、台词、说话动作或口型",
        "不生成声音、拟声、台词、说话动作或口型",
        "无证据短语删除，不拒绝、不重试、不回退、不改 schema",
    ):
        assert required in text


def test_people_and_entity_instance_counts_are_closed_by_new_frames():
    text = _skill()

    for required in (
        "人物和实体的实例集合及数量以该 hard-cut 区间的新关键帧闭合",
        "不得把反射、残影、边缘片段或旧提示词中的称谓计为独立人物或实体",
        "不得据此增加实例数量",
    ):
        assert required in text


def test_fusion_cannot_invent_transitions_or_motion_discontinuities():
    text = _skill()

    for required in (
        "不得增加任何切点",
        "不得输出 morph",
        "不得无证据反转镜头运动方向",
        "不得无证据重置镜头运动速度",
    ):
        assert required in text


def test_audio_and_provider_syntax_stay_out_of_skill_output():
    text = _skill()

    for required in (
        "`audio_content` 只用于避免视觉叙事与冻结 spoken timeline 冲突",
        "不得在输出中复述台词",
        "写 `<d>`",
        "后端会从冻结 audio/music 机械编译",
        "不要写 `[Shot N]`、时间戳、`<Picture N>`、`<Audio N>`",
        "这些 provider-facing 字段全部由后端",
    ):
        assert required in text

    assert "<AUDIO_CONTENT_JSON>{lines_json}" not in text
    assert "<MUSIC_POLICY>" not in text


def test_hard_cut_intervals_are_positional_visual_only():
    text = _skill()

    for required in (
        "每段只输出一个 `visual` 数组",
        "每个 `hard_cut` 开始下一个区间",
        "`continuous` 留在当前区间",
        "`visual` 长度必须等于区间数",
        "数组第 N 项只描述第 N 个区间",
        "provider-facing 字段全部由后端从冻结 timeline/audio/music 机械编译",
    ):
        assert required in text

    assert "<KEYFRAME_TIMELINE_JSON>" not in text


def test_clean_reference_proof_stays_out_of_the_skill_schema():
    text = _skill()

    assert "source audio 只属于上游 ASR/YAMNet 分析证据" in text
    assert "绝不作为当前 H3 reference" in text
    for forbidden_field in (
        "clean_reference_proof:",
        "clean_voice_proof:",
        "reference_receipt:",
    ):
        assert forbidden_field not in text


def test_output_is_one_ordered_visual_vector_per_input_segment():
    text = _skill()

    for required in (
        "整个 ordered segments 数组只执行一次项目级调用",
        "同一次项目调用按序处理全部 segment",
        "`N=1` 与 `N>1` 使用同一合同",
        "`work/h3_prompt_plan.json`",
        'schema: "duet.video-prompt-fusion-output"',
        "version: 2",
        "input_sha256: Sha256",
        "visual: NonEmptyArray<NonEmptyText>",
        "输入描述符 exact bytes 的 SHA-256",
        "输出 segments 与输入 segments 一一对应且顺序相同",
        "不得增加、删除、合并或拆分 segment",
        "每段恰有 `index/visual` 两个字段",
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
