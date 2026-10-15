"""从火山引擎音色列表文档快照生成豆包 2.0 音色目录。

输入：assets/doubao-voices/source-page.md（文档 6561/1257544 的页面正文快照，
只含「豆包语音合成模型2.0」主表与外语音色表两节）。
输出：assets/doubao-voices/catalog.json（app/doubao_voices.py 加载）。

只收 voice_type 以 ``_uranus_bigtts`` 结尾的音色：主表内混排的 S2S
（jupiter/saturn）与 ICL（_tob）行、以及 1.0（mars/moon/wvae）音色一律排除。
gender 从 voice_type 的 ``{lang}_female_/ {lang}_male_`` 前缀提取；
语种列映射为语言码（墨西哥西语/巴西葡萄牙语等并入 es/pt 等大语种码）。
「X同款」营销标签不进目录；方言与「仅限单向流」限制记入 notes。

用法：

    /home/xy/duet-ad1/.venv/bin/python \
        /home/xy/duet-ad1-h3-max-r2v-object-mask/scripts/build_doubao_voice_catalog.py
    # 校验已落盘的 catalog.json 与快照再生成结果一致：
    .../build_doubao_voice_catalog.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "assets" / "doubao-voices" / "source-page.md"
CATALOG_PATH = ROOT / "assets" / "doubao-voices" / "catalog.json"

CATALOG_NAME = "duet.doubao-voices"
CATALOG_VERSION = 1
VOICE_TYPE_SUFFIX = "_uranus_bigtts"

_MAIN_SECTION = "豆包语音合成模型2.0、"
_FOREIGN_SECTION = '"豆包语音合成模型2.0" 外语音色列表'

_GENDER_RE = re.compile(r"^[a-z]{2}_(female|male)_")

# 文档语种列 -> 目录语言码。方言不映射（记入 notes）。
_LANGUAGE_CODES = {
    "中文": "zh",
    "美式英语": "en",
    "英式英语": "en",
    "澳洲英语": "en",
    "日语": "ja",
    "日文": "ja",
    "韩语": "ko",
    "泰语": "th",
    "阿拉伯语": "ar",
    "德语": "de",
    "西班牙语": "es",
    "墨西哥西语": "es",
    "墨西哥西班牙语": "es",
    "法语": "fr",
    "印尼语": "id",
    "印尼": "id",
    "马来语": "ms",
    "巴西葡萄牙语": "pt",
    "葡萄牙语": "pt",
    "俄语": "ru",
    "菲律宾语": "tl",
    "越南语": "vi",
    "意大利语": "it",
}

_UNIDIRECTIONAL_ONLY = "仅限单向流"


def _cells(row: str) -> list[str]:
    return [
        cell.strip().replace("\\_", "_").replace("\\|", "|")
        for cell in row.strip().strip("|").split("|")
    ]


def _section_rows(text: str, marker: str) -> list[list[str]]:
    """Extract table rows between the heading containing ``marker`` and the
    next ``##`` heading."""
    start = text.index(marker)
    end = text.find("\n## ", start + len(marker))
    section = text[start:] if end == -1 else text[start:end]
    rows: list[list[str]] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if not cells or all(set(cell) <= {"-", " "} for cell in cells):
            continue  # 分隔行
        if cells[0].lstrip("*") .startswith("场景"):
            continue  # 表头
        rows.append(cells)
    return rows


def _languages(cell: str) -> tuple[tuple[str, ...], str]:
    """Split the 语种/方言 cell into language codes and a dialect note."""
    language_part, _, dialect_part = cell.partition("<br>")
    language_part = language_part.removeprefix("语种：").strip()
    codes: list[str] = []
    for name in re.split(r"[、,，]", language_part):
        name = name.strip()
        if not name:
            continue
        code = _LANGUAGE_CODES.get(name)
        if code is None:
            raise ValueError(f"未知语种名：{name!r}（单元格 {cell!r}）")
        if code not in codes:
            codes.append(code)
    if not codes:
        raise ValueError(f"语种列为空：{cell!r}")
    dialect_note = ""
    dialect_part = dialect_part.removeprefix("方言：").strip()
    if dialect_part:
        dialect_note = f"方言：{dialect_part}"
    return tuple(codes), dialect_note


def _split_tags(cell: str) -> tuple[str, ...]:
    return tuple(
        part.strip()
        for part in re.split(r"[,，、]", cell)
        if part.strip()
    )


def _notes(*parts: str) -> str:
    notes = [part for part in (part.strip() for part in parts) if part]
    return "；".join(notes)


def build_catalog(text: str) -> dict:
    voices: list[dict] = []
    seen: set[str] = set()

    def add(cells: list[str], remark_index: int) -> None:
        scene_cell, name, voice_type, language_cell = cells[:4]
        capability_cell = cells[4] if len(cells) > 4 else ""
        remark_cell = cells[remark_index] if len(cells) > remark_index else ""
        if not voice_type.endswith(VOICE_TYPE_SUFFIX):
            return
        if voice_type in seen:
            raise ValueError(f"voice_type 重复：{voice_type}")
        seen.add(voice_type)
        gender_match = _GENDER_RE.match(voice_type)
        if gender_match is None:
            raise ValueError(f"voice_type 缺少性别前缀：{voice_type}")
        languages, dialect_note = _languages(language_cell)
        scenes = tuple(
            scene
            for scene in _split_tags(scene_cell)
            # 场景列混入的语种名（外语音色表的列噪音）与 S2S 标签不进 scenes。
            if scene not in _LANGUAGE_CODES and not scene.startswith("S2S")
        )
        capabilities = _split_tags(capability_cell)
        remark = remark_cell.strip()
        unidirectional = (
            _UNIDIRECTIONAL_ONLY if _UNIDIRECTIONAL_ONLY in remark else ""
        )
        voices.append({
            "voice_type": voice_type,
            "name": name,
            "languages": list(languages),
            "gender": gender_match.group(1),
            "scenes": list(scenes),
            "capabilities": list(capabilities),
            "notes": _notes(dialect_note, unidirectional),
        })

    for cells in _section_rows(text, _MAIN_SECTION):
        add(cells, remark_index=5)  # 特殊标签列
    for cells in _section_rows(text, _FOREIGN_SECTION):
        add(cells, remark_index=6)  # 备注列（推荐推理模式不占 notes）
    if not voices:
        raise ValueError("未从快照解析出任何 uranus_bigtts 音色")
    return {
        "schema": CATALOG_NAME,
        "version": CATALOG_VERSION,
        "source": {
            "url": "https://www.volcengine.com/docs/6561/1257544",
            "snapshot": "assets/doubao-voices/source-page.md",
            "voice_type_suffix": VOICE_TYPE_SUFFIX,
        },
        "voices": voices,
    }


def _render(catalog: dict) -> str:
    return json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验 catalog.json 与快照再生成结果一致，不写文件",
    )
    args = parser.parse_args()
    text = SOURCE_PATH.read_text(encoding="utf-8")
    catalog = build_catalog(text)
    rendered = _render(catalog)
    if args.check:
        current = (
            CATALOG_PATH.read_text(encoding="utf-8")
            if CATALOG_PATH.is_file()
            else ""
        )
        if current != rendered:
            print("catalog.json 与 source-page.md 不一致，请重跑生成脚本")
            return 1
        print("catalog.json 与 source-page.md 一致")
        return 0
    CATALOG_PATH.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"已写入 {CATALOG_PATH}：{len(catalog['voices'])} 条音色，sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
