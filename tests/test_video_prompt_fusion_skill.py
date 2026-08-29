from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills/video-prompt-fusion/SKILL.md"
AGENT = ROOT / "skills/video-prompt-fusion/agents/openai.yaml"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_metadata_and_four_input_contract_are_unchanged():
    text = _text()
    assert text.startswith("---\nname: video-prompt-fusion\n")
    assert AGENT.is_file()
    for required in (
        'schema: "duet.video-prompt-fusion-input"', "version: 2",
        "new_keyframes: NineOrdered<KeyframeReceipt>",
        "old_video_prompt: FrozenText",
        "image_optimization_prompt: NineOrdered<FrozenFramePrompt>",
        "audio_content: FrozenAudioContent", "不新增第五类输入",
    ):
        assert required in text


def test_relation_binding_and_state_lifecycle_are_consumed():
    text = _text()
    for required in (
        "全项目共享关系绑定：relation_key -> subject_key -> predicate -> object_key -> replacement_system",
        "主客体和功能角色全项目不变", "状态生命周期", "连接、装载、作用、释放、分离",
        "不能互换主客体", "末帧若仍在运动，只写可见状态",
    ):
        assert required in text


def test_authority_boundaries_and_output_remain_closed():
    text = _text()
    for required in (
        "新关键帧独占静态事实权威", "旧提示词只贡献同一区间内",
        "跨段只共享 stable element design 和 relation system",
        "质量评分不触发拒绝、重试、回退", "work/h3_prompt_plan.json",
        'schema: "duet.video-prompt-fusion-output"', "供 Context IR 优化",
    ):
        assert required in text
    assert "不输出时间戳、图片标记、stable key、tile、relation key" in text


def test_skill_is_compact_and_sample_neutral():
    text = _text()
    assert len(text.encode("utf-8")) < 9_000
    for sample_term in ("陀螺", "发射器", "梳毛", "聚餐", "玩具"):
        assert sample_term not in text
