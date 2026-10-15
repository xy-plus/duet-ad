"""逐行音色绑定（画面人物优先）的纯函数测试。

fixture 结构取自生产项目 b6240a72（电热睫毛夹）真实的
enhanced_element_index：person-01 成年男性（seg1 开头 3 帧 + seg3 末 4 帧）、
person-02 成年女性（seg1 后 6 帧 + seg2 8 帧 + seg3 前 5 帧）、person-03
儿童女孩（seg2 单帧眼部特写）。验收锚点：line1→男声、line2/3→女声。
"""
from __future__ import annotations

import pytest

from app.dialogue_delivery import (
    ark_gender_age_vote,
    ark_gender_majority,
    person_description_gender,
    person_record_is_child,
    resolve_line_speakers,
    resolve_line_voices,
)


# ---- 真实结构 fixture（b6240a72 电热睫毛夹） ----

PEOPLE_EYELASH = {
    "person-01": {
        "source_visual_description": "浅棕色短发、额发向上梳起的成年男性，穿白底彩色横条纹短袖T恤。",
        "occurrences": [
            {"segment_index": 1, "frame_orders": [1, 2, 3]},
            {"segment_index": 3, "frame_orders": [6, 7, 8, 9]},
        ],
    },
    "person-02": {
        "source_visual_description": "金发女性，额发被发箍向后固定，戴金色圈形耳环与细金项链。",
        "occurrences": [
            {"segment_index": 1, "frame_orders": [4, 5, 6, 7, 8, 9]},
            {"segment_index": 2, "frame_orders": [1, 2, 3, 5, 6, 7, 8, 9]},
            {"segment_index": 3, "frame_orders": [1, 2, 3, 4, 5]},
        ],
    },
    "person-03": {
        "source_visual_description": "儿童（女孩），仅眼部极特写可见：深棕色大而圆的眼睛、上翘的长睫毛。",
        "occurrences": [
            {"segment_index": 2, "frame_orders": [4]},
        ],
    },
}

SEGMENTS_EYELASH = [
    {"index": 1, "start_s": 0.0, "end_s": 8.433334},
    {"index": 2, "start_s": 8.433334, "end_s": 18.4},
    {"index": 3, "start_s": 18.4, "end_s": 26.433333},
]

LINES_EYELASH = [
    {"text": "line1", "start_s": 0.0, "end_s": 8.433334},
    {"text": "line2", "start_s": 8.433334, "end_s": 18.4},
    {"text": "line3", "start_s": 18.4, "end_s": 26.433333},
]


class _StubCatalog:
    """DoubaoVoiceCatalog 最小接口替身：voice/has/voices_for_language。"""

    def __init__(self, voices: list[dict[str, object]]):
        self._voices = {voice["voice_type"]: voice for voice in voices}

    def voice(self, voice_type: str) -> dict[str, object]:
        return self._voices[voice_type]

    def has(self, voice_type: object) -> bool:
        return voice_type in self._voices

    def voices_for_language(self, language_label: str) -> tuple[dict[str, object], ...]:
        return tuple(self._voices.values())


CATALOG_MS = _StubCatalog(
    [
        {
            "voice_type": "ms_male_ham_uranus_bigtts",
            "gender": "male",
            "scenes": ["通用场景"],
        },
        {
            "voice_type": "zh_female_vv_uranus_bigtts",
            "gender": "female",
            "scenes": ["通用场景"],
        },
    ]
)


# ---- 人物描述性别/年龄关键词 ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("浅棕色短发的成年男性，穿条纹T恤", "male"),
        ("穿黑色短袖T恤与蓝色牛仔裤的男子", "male"),
        ("金发女性，戴金色圈形耳环", "female"),
        ("一名戴亮粉色头巾、穿碎花上衣的女子", "female"),
        ("仅手部入画的人物：指甲涂浅粉色甲油", None),
        ("男性与女性同时入画", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_person_description_gender(text, expected):
    assert person_description_gender(text) == expected


def test_person_record_is_child():
    assert person_record_is_child(PEOPLE_EYELASH["person-03"]) is True
    assert person_record_is_child(PEOPLE_EYELASH["person-01"]) is False
    assert person_record_is_child(PEOPLE_EYELASH["person-02"]) is False
    assert person_record_is_child({"source_visual_description": "男童坐在桌前"}) is True


def test_ark_gender_age_vote():
    assert ark_gender_age_vote("成年男性，30岁左右") == "male"
    assert ark_gender_age_vote("年轻女声") == "female"
    assert ark_gender_age_vote("难以判断") is None
    assert ark_gender_age_vote(None) is None


def test_ark_gender_majority():
    lines = [
        {"gender_age": "成年男性"},
        {"gender_age": "男声，低沉"},
        {"gender_age": "无法判断"},
    ]
    gender, evidence = ark_gender_majority(lines)
    assert gender == "male"
    assert "男" in evidence
    assert ark_gender_majority([]) == (None, "ark gender_age 全部缺失")


# ---- 逐行主口播人物解析 ----


def test_resolve_line_speakers_eyelash_curler():
    resolution = resolve_line_speakers(
        lines=LINES_EYELASH, segments=SEGMENTS_EYELASH, people=PEOPLE_EYELASH
    )
    assert [entry["gender"] for entry in resolution] == [
        "male",
        "female",
        "female",
    ]
    assert resolution[0]["segment_index"] == 1
    assert resolution[0]["person_id"] == "person-01"
    assert resolution[1]["segment_index"] == 2
    # line2 主人物必须是成年女性 person-02，不能是儿童 person-03。
    assert resolution[1]["person_id"] == "person-02"
    assert resolution[2]["segment_index"] == 3
    assert resolution[2]["person_id"] == "person-02"


def test_first_frame_wins_over_frame_count():
    """seg1 内 person-02 帧数（6）多于 person-01（3），但行起声时画面是
    person-01（首帧 1 < 4）：主口播人物取行窗起点的演示者。"""
    resolution = resolve_line_speakers(
        lines=[LINES_EYELASH[0]],
        segments=SEGMENTS_EYELASH,
        people=PEOPLE_EYELASH,
    )
    assert resolution[0]["person_id"] == "person-01"


def test_frame_count_breaks_first_frame_tie():
    """首帧并列时帧数多者胜（户外灯串 seg2 形态：男子 4 帧 vs 女性 9 帧，
    首帧同为 1）。"""
    people = {
        "person-01": {
            "source_visual_description": "穿黑色短袖T恤的男子",
            "occurrences": [{"segment_index": 1, "frame_orders": [1, 2, 3, 4]}],
        },
        "person-02": {
            "source_visual_description": "戴女巫帽的女性",
            "occurrences": [
                {"segment_index": 1, "frame_orders": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
            ],
        },
    }
    resolution = resolve_line_speakers(
        lines=[{"text": "x", "start_s": 0.0, "end_s": 5.0}],
        segments=[{"index": 1, "start_s": 0.0, "end_s": 5.0}],
        people=people,
    )
    assert resolution[0]["person_id"] == "person-02"
    assert resolution[0]["gender"] == "female"


def test_child_only_segment_falls_back_to_neighbor():
    """该段只有儿童出镜：儿童永不作口播依据，回退相邻段的成年人物。"""
    people = {
        "person-01": {
            "source_visual_description": "成年男性演示者",
            "occurrences": [{"segment_index": 1, "frame_orders": [1, 2, 3]}],
        },
        "person-02": {
            "source_visual_description": "儿童（女孩）眼部特写",
            "occurrences": [{"segment_index": 2, "frame_orders": [1, 2]}],
        },
    }
    segments = [
        {"index": 1, "start_s": 0.0, "end_s": 5.0},
        {"index": 2, "start_s": 5.0, "end_s": 10.0},
    ]
    resolution = resolve_line_speakers(
        lines=[{"text": "x", "start_s": 5.0, "end_s": 10.0}],
        segments=segments,
        people=people,
    )
    assert resolution[0]["segment_index"] == 2
    assert resolution[0]["person_id"] == "person-01"
    assert resolution[0]["gender"] == "male"
    assert resolution[0]["source"] == "neighbor_segment"


def test_no_people_returns_none():
    resolution = resolve_line_speakers(
        lines=[{"text": "x", "start_s": 0.0, "end_s": 5.0}],
        segments=[{"index": 1, "start_s": 0.0, "end_s": 5.0}],
        people={},
    )
    assert resolution[0]["person_id"] is None
    assert resolution[0]["gender"] is None
    assert resolution[0]["source"] == "none"


def test_line_assigned_to_segment_by_time_window():
    """行窗与段窗不一一对应：两行落在同一段，共享该段主人物。"""
    lines = [
        {"text": "a", "start_s": 0.0, "end_s": 4.0},
        {"text": "b", "start_s": 4.0, "end_s": 8.433334},
        {"text": "c", "start_s": 8.433334, "end_s": 18.4},
    ]
    resolution = resolve_line_speakers(
        lines=lines, segments=SEGMENTS_EYELASH, people=PEOPLE_EYELASH
    )
    assert [entry["segment_index"] for entry in resolution] == [1, 1, 2]
    assert [entry["gender"] for entry in resolution] == ["male", "male", "female"]


# ---- 逐行音色决策 ----


def test_resolve_line_voices_eyelash_curler():
    """b6240a72 验收：line1 保留马来男声；line2/3 改女声，马来语无女声音色
    → zh_female_vv。explicit_language 全量锁：uranus 精品音色同语种也锁
    （多语种混合训练模型数字/单位 token 概率性串语种，2026-09-27 实证）。"""
    resolution = resolve_line_voices(
        lines=LINES_EYELASH,
        segments=SEGMENTS_EYELASH,
        people=PEOPLE_EYELASH,
        inherited_voices=["ms_male_ham_uranus_bigtts"] * 3,
        catalog=CATALOG_MS,
        language_label="马来语",
        analysis_lines=[
            {"gender_age": "成年男性"},
            {"gender_age": "成年男性"},
            {"gender_age": "成年男性"},
        ],
    )
    assert [entry["voice_type"] for entry in resolution] == [
        "ms_male_ham_uranus_bigtts",
        "zh_female_vv_uranus_bigtts",
        "zh_female_vv_uranus_bigtts",
    ]
    assert [entry["explicit_language"] for entry in resolution] == [
        "ms",
        "ms",
        "ms",
    ]
    assert resolution[0]["reselected"] is False
    assert resolution[1]["reselected"] is True
    assert resolution[1]["previous_voice_type"] == "ms_male_ham_uranus_bigtts"


def test_resolve_line_voices_non_uranus_voice_not_locked():
    """非 *_uranus_bigtts 音色未验证 explicit_language 接受度，不传。"""
    catalog = _StubCatalog(
        [
            {
                "voice_type": "ms_male_ham_legacy",
                "gender": "male",
                "scenes": ["通用场景"],
            },
        ]
    )
    resolution = resolve_line_voices(
        lines=[LINES_EYELASH[0]],
        segments=SEGMENTS_EYELASH,
        people=PEOPLE_EYELASH,
        inherited_voices=["ms_male_ham_legacy"],
        catalog=catalog,
        language_label="马来语",
    )
    assert resolution[0]["voice_type"] == "ms_male_ham_legacy"
    assert resolution[0]["explicit_language"] is None


def test_resolve_line_voices_keeps_inherited_when_gender_matches():
    resolution = resolve_line_voices(
        lines=[LINES_EYELASH[0]],
        segments=SEGMENTS_EYELASH,
        people=PEOPLE_EYELASH,
        inherited_voices=["ms_male_ham_uranus_bigtts"],
        catalog=CATALOG_MS,
        language_label="马来语",
    )
    assert resolution[0]["voice_type"] == "ms_male_ham_uranus_bigtts"
    assert resolution[0]["explicit_language"] == "ms"
    assert resolution[0]["reselected"] is False


def test_resolve_line_voices_ark_fallback_when_no_visual_evidence():
    """画面无人物数据时方舟 gender_age 多数票接管（无画面证据时音频是唯一
    信号）；与旧绑定性别一致则保留。"""
    resolution = resolve_line_voices(
        lines=[{"text": "x", "start_s": 0.0, "end_s": 5.0}],
        segments=[{"index": 1, "start_s": 0.0, "end_s": 5.0}],
        people={},
        inherited_voices=["ms_male_ham_uranus_bigtts"],
        catalog=CATALOG_MS,
        language_label="马来语",
        analysis_lines=[{"gender_age": "成年男性"}, {"gender_age": "男声"}],
    )
    assert resolution[0]["voice_type"] == "ms_male_ham_uranus_bigtts"
    assert resolution[0]["reselected"] is False
    assert resolution[0]["gender_source"] == "ark_majority"


def test_resolve_line_voices_keeps_inherited_without_any_evidence():
    resolution = resolve_line_voices(
        lines=[{"text": "x", "start_s": 0.0, "end_s": 5.0}],
        segments=[{"index": 1, "start_s": 0.0, "end_s": 5.0}],
        people={},
        inherited_voices=["ms_male_ham_uranus_bigtts"],
        catalog=CATALOG_MS,
        language_label="马来语",
    )
    assert resolution[0]["voice_type"] == "ms_male_ham_uranus_bigtts"
    assert resolution[0]["gender"] is None
    assert resolution[0]["gender_source"] == "none"
    assert resolution[0]["reselected"] is False
