from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import pytest

from app.dialogue_delivery import (
    DIALOGUE_DELIVERY_OUTPUT_NAME,
    DIALOGUE_DELIVERY_OUTPUT_VERSION,
    VOICE_BASELINE_RATES,
    build_dialogue_delivery_payload,
    count_letters,
    dialogue_delivery_instruction,
    dialogue_delivery_schema,
    dialogue_tts_line_input_sha256,
    dialogue_tts_section_id,
    is_decodable_wav,
    normalize_dialogue_delivery,
    pick_voice_for_gender,
    suggested_speech_rate,
)
from app.doubao_voices import load_doubao_voice_catalog


def _wav_bytes(frames: int = 2400) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def _lines() -> list[dict[str, Any]]:
    return [
        {"text": "第一句", "start_s": 0.0, "end_s": 2.0},
        {"text": "第二句话更长", "start_s": 5.0, "end_s": 7.0},
    ]


def _payload(
    line_voices: list[str] | None = None,
    analysis_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_dialogue_delivery_payload(
        target_language="中文",
        translated_lines=_lines(),
        line_voices=line_voices or ["en_female_joanne_uranus_bigtts"] * 2,
        catalog=load_doubao_voice_catalog(),
        analysis_lines=analysis_lines,
    )


def test_baseline_rates_ship_the_poc_calibrated_values() -> None:
    # conftest autouse 只补全目录内其余音色（setitem，不改既有键），
    # POC 标定初值逐项原样保留。
    poc_values = {
        "ms_male_ham_uranus_bigtts": 13.65,
        "es_female_ht_mx_f6_uranus_bigtts": 12.39,
        "en_female_authoritative-informative_uranus_bigtts": 13.87,
        "en_female_joanne_uranus_bigtts": 10.86,
    }
    for voice_type, rate in poc_values.items():
        assert VOICE_BASELINE_RATES[voice_type] == rate
    assert all(
        isinstance(value, float) and value > 0
        for value in poc_values.values()
    )


def test_count_letters_ignores_digits_punctuation_and_whitespace() -> None:
    assert count_letters("第一句") == 3
    assert count_letters("Hello, world! 123") == 10
    assert count_letters("") == 0


def test_suggested_speech_rate_matches_window_budget_formula() -> None:
    # target = 1.2 * baseline -> +20%
    assert suggested_speech_rate(12.0, 10.0) == 20
    assert suggested_speech_rate(5.0, 10.0) == -50
    assert suggested_speech_rate(30.0, 10.0) == 100
    with pytest.raises(ValueError, match="baseline"):
        suggested_speech_rate(12.0, 0.0)


def test_build_payload_computes_window_budgets_and_suggested_rates() -> None:
    payload = _payload()
    assert payload["target_language"] == "中文"
    assert payload["voice_catalog_sha256"] == load_doubao_voice_catalog().digest
    lines = payload["lines"]
    assert [line["line_index"] for line in lines] == [1, 2]
    first = lines[0]
    assert first["text"] == "第一句"
    assert first["window_s"] == 2.0
    assert first["target_chars_per_window_s"] == 1.5
    assert first["voice_baseline_chars_per_s"] == 10.86
    assert first["suggested_speech_rate"] == suggested_speech_rate(1.5, 10.86)
    assert first["voice"]["voice_type"] == "en_female_joanne_uranus_bigtts"
    assert first["voice"]["gender"] == "female"
    # 未提供音频分析时 features 为空对象。
    assert first["features"] == {}


def test_build_payload_injects_optional_analysis_lines() -> None:
    payload = _payload(
        analysis_lines=[{"overall_tone": "兴奋"}, {"overall_tone": "平静"}],
    )
    assert payload["lines"][0]["features"] == {"overall_tone": "兴奋"}
    assert payload["lines"][1]["features"] == {"overall_tone": "平静"}


def test_build_payload_rejects_voices_without_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 覆盖整表（autouse 已补全全部目录音色）：只留一个无关注色，两行用到的
    # 音色均无标定值。
    monkeypatch.setattr(
        "app.doubao_voices.VOICE_BASELINE_RATES",
        {"ms_male_ham_uranus_bigtts": 13.65},
    )
    with pytest.raises(ValueError, match="baseline"):
        _payload(line_voices=["en_female_joanne_uranus_bigtts", "zh_female_vv_uranus_bigtts"])


# 生产标定表（VOICE_BASELINE_RATES 实际值）；conftest autouse 会把目录内
# 其余音色也补进表，选音色测试必须整体覆盖该表才能还原生产口径。
_PRODUCTION_BASELINE_RATES = {
    "ms_male_ham_uranus_bigtts": 13.65,
    "es_female_ht_mx_f6_uranus_bigtts": 12.39,
    "en_female_authoritative-informative_uranus_bigtts": 13.87,
    "en_female_joanne_uranus_bigtts": 10.86,
    "zh_female_vv_uranus_bigtts": 12.09,
    "en_male_adam-imitation_uranus_bigtts": 12.57,
}


@pytest.mark.parametrize(
    ("language_label", "gender", "expected"),
    [
        # 2026-09-28 防晒霜项目 c86c67ed 崩溃根因：英语女声目录序靠前的是
        # 未标定的 en_female_brittney，选音色必须只从标定音色中取。
        ("英语", "female", {
            "en_female_authoritative-informative_uranus_bigtts",
            "en_female_joanne_uranus_bigtts",
        }),
        ("英语", "male", {"en_male_adam-imitation_uranus_bigtts"}),
        ("马来语", "male", {"ms_male_ham_uranus_bigtts"}),
        ("西班牙语", "female", {"es_female_ht_mx_f6_uranus_bigtts"}),
        # 马来语目录无女声，走已标定的跨语种兜底 zh_female_vv。
        ("马来语", "female", {"zh_female_vv_uranus_bigtts"}),
    ],
)
def test_pick_voice_only_selects_calibrated_voices(
    monkeypatch: pytest.MonkeyPatch,
    language_label: str,
    gender: str,
    expected: set[str],
) -> None:
    monkeypatch.setattr(
        "app.doubao_voices.VOICE_BASELINE_RATES",
        dict(_PRODUCTION_BASELINE_RATES),
    )
    voice_type = pick_voice_for_gender(
        load_doubao_voice_catalog(), language_label, gender,
    )
    assert voice_type in expected
    assert voice_type in _PRODUCTION_BASELINE_RATES


def test_pick_voice_raises_when_no_calibrated_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 英语女声候选全部未标定，兜底 zh_female_vv 也未标定：明确报错，
    # 错误信息带语言/性别/可用标定音色清单。
    monkeypatch.setattr(
        "app.doubao_voices.VOICE_BASELINE_RATES",
        {"ms_male_ham_uranus_bigtts": 13.65},
    )
    with pytest.raises(
        RuntimeError, match="no calibrated female voice for 英语",
    ) as excinfo:
        pick_voice_for_gender(load_doubao_voice_catalog(), "英语", "female")
    assert "ms_male_ham_uranus_bigtts" in str(excinfo.value)


def test_schema_binds_input_sha256_and_line_count() -> None:
    payload = _payload()
    schema, input_sha256 = dialogue_delivery_schema(payload)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema"]["enum"] == [
        DIALOGUE_DELIVERY_OUTPUT_NAME
    ]
    assert schema["properties"]["version"]["enum"] == [
        DIALOGUE_DELIVERY_OUTPUT_VERSION
    ]
    assert schema["properties"]["input_sha256"]["enum"] == [input_sha256]
    lines_schema = schema["properties"]["lines"]
    assert lines_schema["minItems"] == lines_schema["maxItems"] == 2
    item = lines_schema["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["speech_rate"] == {
        "type": "integer",
        "minimum": -50,
        "maximum": 100,
    }
    # 输入相同则 sha256 稳定；输入变化则变化。
    assert dialogue_delivery_schema(payload)[1] == input_sha256
    changed = _payload(analysis_lines=[{"overall_tone": "兴奋"}])
    assert dialogue_delivery_schema(changed)[1] != input_sha256


def _model_output(payload: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    return {
        "schema": DIALOGUE_DELIVERY_OUTPUT_NAME,
        "version": DIALOGUE_DELIVERY_OUTPUT_VERSION,
        "input_sha256": input_sha256,
        "lines": [
            {
                "line_index": 2,
                "speech_rate": 7,
                "emotion": "happy",
                "context_texts": ["语气轻快"],
            },
            {
                "line_index": 1,
                "speech_rate": -3,
                "emotion": None,
                "context_texts": None,
            },
        ],
    }


def test_normalize_orders_lines_and_drops_line_index() -> None:
    payload = _payload()
    _schema, input_sha256 = dialogue_delivery_schema(payload)
    normalized = normalize_dialogue_delivery(
        _model_output(payload, input_sha256),
        input_sha256=input_sha256,
        line_count=2,
    )
    assert normalized == [
        {"speech_rate": -3, "emotion": None, "context_texts": None},
        {"speech_rate": 7, "emotion": "happy", "context_texts": ["语气轻快"]},
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda output: output.update(input_sha256="0" * 64),
        lambda output: output.update(version=DIALOGUE_DELIVERY_OUTPUT_VERSION + 1),
        lambda output: output.update(schema="duet.other-output"),
        lambda output: output["lines"].pop(),
        lambda output: output["lines"][0].update(line_index=1),
        lambda output: output["lines"][0].update(line_index=99),
        lambda output: output["lines"][0].update(speech_rate=101),
        lambda output: output["lines"][0].update(speech_rate=-51),
        lambda output: output["lines"][0].update(speech_rate=7.5),
        lambda output: output["lines"][0].update(emotion=""),
        lambda output: output["lines"][0].update(emotion=3),
        lambda output: output["lines"][0].update(context_texts="语气轻快"),
        lambda output: output["lines"][0].update(context_texts=[""]),
    ],
)
def test_normalize_rejects_unbound_or_malformed_output(mutation: Any) -> None:
    payload = _payload()
    _schema, input_sha256 = dialogue_delivery_schema(payload)
    output = _model_output(payload, input_sha256)
    mutation(output)
    with pytest.raises(ValueError, match="dialogue delivery"):
        normalize_dialogue_delivery(
            output, input_sha256=input_sha256, line_count=2,
        )


def test_instruction_carries_the_window_budget_formula() -> None:
    instruction = dialogue_delivery_instruction("中文")
    assert "suggested_speech_rate" in instruction
    assert "voice_baseline_chars_per_s" in instruction
    assert "-50" in instruction and "100" in instruction
    assert "中文" in instruction


def test_instruction_sets_ad_expressiveness_floor_for_flat_originals() -> None:
    # 念稿感修复：原音语气平淡（很多原片本身如此）不得照抄成 neutral 念稿，
    # 语气与情绪以替换后商品的广告口播语境为下限；语速/停顿仍以原音为基准。
    instruction = dialogue_delivery_instruction("中文")
    # 触发条件与禁止结果必须成对出现：平淡原音 → 禁止 neutral 念稿。
    assert "平淡" in instruction
    assert "不得整段 neutral" in instruction
    # 下限的可执行落点：积极情绪替代 + 带货口播腔的 context_texts。
    assert "happy" in instruction
    assert "亲切、有感染力的带货口播" in instruction
    # 旧措辞「拿不准填 neutral」与下限冲突，必须移除。
    assert "拿不准填 neutral" not in instruction


def test_instruction_restricts_context_language_to_zh_or_en() -> None:
    # 两轮实证（注释见 dialogue_delivery_instruction 上方）：小语种 context
    # 被 TTS 当正文续读（车载湿巾 line-0012 马来语 2/2 稳定复现）；中文
    # context 配异语言台词串语言（探针 2/5）；英文 context 探针 0/5 安全。
    # 规则：中文台词→中文 context，其余所有语言（含 ms/es/en）一律英文。
    instruction = dialogue_delivery_instruction("英语")
    assert "只允许中文或英文" in instruction
    assert "台词是中文就用中文写" in instruction
    assert "一律用英文写" in instruction
    # 小语种（马来语/西班牙语）被点名禁止，英文不在禁止之列。
    assert "禁止" in instruction and "小语种" in instruction
    # 旧的"与台词同语言"规则已被兜底规则取代，不得残留。
    assert "必须用与该行台词完全相同的语言撰写" not in instruction
    # 小语种标签的指令文本同样是这条规则（规则不随语言标签变化）。
    assert "一律用英文写" in dialogue_delivery_instruction("马来语")


def test_instruction_sha256_tracks_instruction_text() -> None:
    from app.dialogue_delivery import dialogue_delivery_instruction_sha256

    digest = dialogue_delivery_instruction_sha256("英语")
    assert isinstance(digest, str) and len(digest) == 64
    # 指令绑定语言标签：不同语言标签的指令文本不同，哈希必须不同。
    assert digest != dialogue_delivery_instruction_sha256("中文")


def test_section_id_is_uuid5_of_run_and_voice() -> None:
    first = dialogue_tts_section_id("run-1", "voice-a")
    assert first == dialogue_tts_section_id("run-1", "voice-a")
    assert first != dialogue_tts_section_id("run-2", "voice-a")
    assert first != dialogue_tts_section_id("run-1", "voice-b")


def test_is_decodable_wav_accepts_nonempty_pcm_wav(tmp_path: Path) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(_wav_bytes())
    assert is_decodable_wav(path) is True


@pytest.mark.parametrize(
    "payload_bytes",
    [b"", b"tts-audio", _wav_bytes(frames=0)],
)
def test_is_decodable_wav_rejects_garbage_and_empty_audio(
    tmp_path: Path, payload_bytes: bytes,
) -> None:
    path = tmp_path / "line.wav"
    path.write_bytes(payload_bytes)
    assert is_decodable_wav(path) is False


def test_is_decodable_wav_rejects_missing_file(tmp_path: Path) -> None:
    assert is_decodable_wav(tmp_path / "missing.wav") is False


def _line_input_hash(**changes: Any) -> str:
    values: dict[str, Any] = {
        "text": "第一句",
        "voice_type": "en_female_joanne_uranus_bigtts",
        "start_s": 0.0,
        "end_s": 2.0,
        "speech_rate": 0,
        "emotion": None,
        "context_texts": None,
        "explicit_language": "en",
    }
    values.update(changes)
    return dialogue_tts_line_input_sha256(**values)


def test_line_input_hash_binds_explicit_language() -> None:
    """explicit_language 全量锁（2026-09-27）后进逐行哈希：锁与不锁、锁不同
    语种的合成输入不同，旧 wav 不得复用（否则修复后旧串语种音频被错复用）。"""
    locked = _line_input_hash()
    assert locked == _line_input_hash()
    assert locked != _line_input_hash(explicit_language=None)
    assert locked != _line_input_hash(explicit_language="ms")
    assert locked != _line_input_hash(text="其他文本")
