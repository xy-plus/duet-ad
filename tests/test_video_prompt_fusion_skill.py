from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills/video-prompt-fusion/SKILL.md"
AGENT = ROOT / "skills/video-prompt-fusion/agents/openai.yaml"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_metadata_and_backend_bound_input_contract_are_explicit():
    text = _text()
    assert text.startswith("---\nname: video-prompt-fusion\n")
    assert AGENT.is_file()
    for required in (
        "multimodal_input.json", "version 2", "new_keyframes",
        "old_video_prompt", "image_optimization_prompt",
        "relation_occurrences", "audio_content", "一次处理全部 ordered segments",
    ):
        assert required in text


def test_relation_binding_and_state_lifecycle_are_consumed():
    text = _text()
    for required in (
        "`relation_occurrences` 是逐帧关系状态", "replacement system",
        "唯一结构化权威", "不得互换主客体", "末帧仅写可见状态",
    ):
        assert required in text


def test_authority_boundaries_and_output_remain_closed():
    text = _text()
    for required in (
        "新关键帧是人物", "旧提示词只可贡献同一区间",
        "跨段只保持同 stable element", "work/h3_prompt_plan.json",
        "注入的 JSON Schema", "编译 Context IR/H3",
        "每段按 hard-cut 区间写一条简洁英文 `visual` prose",
        "`continuous` 留在当前区间",
    ):
        assert required in text
    assert "不输出时间戳、图片标记、stable key、tile、relation key" in text


def test_skill_is_compact_and_sample_neutral():
    text = _text()
    assert len(text.encode("utf-8")) < 9_000
    for sample_term in ("陀螺", "发射器", "梳毛", "聚餐", "玩具"):
        assert sample_term not in text
