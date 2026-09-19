"""豆包音色目录：加载校验、语言过滤子集、默认音色与快照再生成一致性。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.doubao_voices import (  # noqa: E402
    DOUBAO_VOICE_CATALOG_PATH,
    KNOWN_LANGUAGE_CODES,
    PROMPT_VOICE_LIMIT,
    doubao_voice_language_key,
    load_doubao_voice_catalog,
)

sys.path.insert(0, str(ROOT / "scripts"))
from build_doubao_voice_catalog import (  # noqa: E402
    SOURCE_PATH,
    _render,
    build_catalog,
)


def test_catalog_matches_the_source_snapshot() -> None:
    """catalog.json 必须是 build 脚本对 source-page.md 的确定性输出。"""
    rendered = _render(build_catalog(SOURCE_PATH.read_text(encoding="utf-8")))
    assert DOUBAO_VOICE_CATALOG_PATH.read_text(encoding="utf-8") == rendered


def test_catalog_loads_only_uranus_bigtts_voices_with_known_languages() -> None:
    catalog = load_doubao_voice_catalog()
    assert len(catalog.voices) > 200
    for voice in catalog.voices:
        assert voice["voice_type"].endswith("_uranus_bigtts")
        assert voice["gender"] in {"female", "male"}
        assert voice["voice_type"].split("_")[1] == voice["gender"]
        assert set(voice["languages"]) <= KNOWN_LANGUAGE_CODES
        assert voice["name"].strip()
    assert catalog.voice("zh_female_vv_uranus_bigtts")["name"] == "Vivi 2.0"
    # S2S / ICL / 1.0 音色不进目录。
    assert not catalog.has("zh_female_vv_jupiter_bigtts")
    assert not catalog.has("ICL_uranus_zh_female_kefuwanjun_tob")
    assert not catalog.has("zh_female_qinqienvsheng_moon_bigtts")


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("中文", "zh"),
        ("英语", "en"),
        ("Thai", "th"),
        ("th", "th"),
        ("日语", "ja"),
        ("马来语", "ms"),
        ("菲律宾语", "tl"),
        ("意大利语", "it"),
        ("克林贡语", "zh"),
        ("", "zh"),
    ],
)
def test_language_key_normalizes_labels(label: str, key: str) -> None:
    assert doubao_voice_language_key(label) == key


def test_language_subset_is_filtered_and_never_empty() -> None:
    catalog = load_doubao_voice_catalog()
    zh = catalog.voices_for_language("中文")
    assert zh
    assert all("zh" in voice["languages"] for voice in zh)
    assert all(
        not voice["voice_type"].startswith("en_") or "zh" in voice["languages"]
        for voice in zh
    )
    th = catalog.voices_for_language("泰语")
    assert th
    assert all("th" in voice["languages"] for voice in th)
    assert len(th) < len(zh) < len(catalog.voices)
    # 未知语言回退中文子集。
    assert catalog.voices_for_language("克林贡语") == zh


def test_default_voice_type_prefers_the_general_scene() -> None:
    catalog = load_doubao_voice_catalog()
    assert catalog.default_voice_type("中文") == "zh_female_vv_uranus_bigtts"
    default_th = catalog.default_voice_type("泰语")
    assert "th" in catalog.voice(default_th)["languages"]
    assert "通用场景" in catalog.voice(default_th)["scenes"]
    # 意大利语只有 Enzo（无通用场景标签时回退该语种第一条）。
    assert catalog.default_voice_type("意大利语") == "it_male_enzo_uranus_bigtts"
    assert catalog.default_voice_type("未知语言") == catalog.default_voice_type("中文")


def test_model_lines_are_compact_one_line_summaries() -> None:
    catalog = load_doubao_voice_catalog()
    lines = catalog.model_lines(catalog.voices_for_language("日语"))
    assert "{" not in lines
    assert all(" - " in line for line in lines.splitlines())
    assert "ja_female_bv522_uranus_bigtts" in lines


def test_prompt_subset_is_capped_and_keeps_default_and_genders() -> None:
    catalog = load_doubao_voice_catalog()
    zh = catalog.prompt_voices_for_language("中文")
    assert 0 < len(zh) <= PROMPT_VOICE_LIMIT
    assert len(zh) < len(catalog.voices_for_language("中文"))
    # 兜底音色必须在精选子集内，且两种性别都有覆盖（多说话人区分音色）。
    defaults = catalog.default_voice_type("中文")
    assert defaults in [voice["voice_type"] for voice in zh]
    assert {voice["gender"] for voice in zh} == {"female", "male"}
    # 小语种子集不足上限时原样返回。
    th = catalog.prompt_voices_for_language("泰语")
    assert th == catalog.voices_for_language("泰语")
    # 未知语言回退中文精选子集。
    assert catalog.prompt_voices_for_language("克林贡语") == zh


def test_prompt_lines_drop_language_and_notes_columns() -> None:
    catalog = load_doubao_voice_catalog()
    lines = catalog.prompt_lines(catalog.prompt_voices_for_language("中文"))
    assert "{" not in lines
    rows = lines.splitlines()
    assert len(rows) == len(catalog.prompt_voices_for_language("中文"))
    assert all(len(row.split(" - ")) == 4 for row in rows)
    assert "方言" not in lines


def test_loader_rejects_broken_catalogs(tmp_path: Path) -> None:
    raw = json.loads(DOUBAO_VOICE_CATALOG_PATH.read_text(encoding="utf-8"))

    def broken(mutate, message: str) -> None:
        value = json.loads(json.dumps(raw))
        mutate(value)
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_doubao_voice_catalog(path)

    broken(lambda value: value.update(schema="other"), "identity")
    broken(lambda value: value.update(version=999), "identity")
    broken(lambda value: value.update(voices=[]), "at least one")
    broken(
        lambda value: value["voices"].append(dict(value["voices"][0])),
        "duplicated",
    )
    broken(
        lambda value: value["voices"][0].update(languages=["xx"]),
        "languages",
    )
    broken(
        lambda value: value["voices"][0].update(gender="unknown"),
        "gender",
    )
    broken(
        lambda value: value["voices"][0].pop("notes"),
        "entry 1",
    )
