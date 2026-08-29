import json
import re
from pathlib import Path


SKILL = Path(
    "/home/xy/duet-ad1/.worktree/skill-milestone-r2/"
    "skills/image-postprocess/SKILL.md"
)


def test_r3_contract_is_explicit_without_growing_the_skill():
    text = SKILL.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 52
    assert len(text.encode("utf-8")) <= 6892
    for phrase in (
        "覆盖每个 stable key",
        "逐张读取全部 path（含首末帧）",
        "同一人物及服装跨段只复用",
        "不可见部分不继承上一帧",
        "element_index`/`global_plan`存在不等于本帧可见",
        "颜色/款式/材质/归属/关系不得漂移",
        "人物数量闭合",
        "不可见或无法唯一判断",
    ):
        assert phrase in text


def test_r3_examples_remain_two_closed_json_objects():
    contracts = [
        json.loads(block)
        for block in re.findall(r"```json\s*(.*?)\s*```", SKILL.read_text(encoding="utf-8"), re.DOTALL)
    ]
    assert len(contracts) == 2
    assert set(contracts[0]) == {"people", "entities", "scenes"}
    assert set(contracts[1]) == {"frames"}
