"""豆包语音合成 2.0 音色目录：加载、校验、按语言过滤与默认音色选择。

目录文件 assets/doubao-voices/catalog.json 由
scripts/build_doubao_voice_catalog.py 从火山文档快照再生成（--check 校验一致），
不要手工编辑。每条音色含 voice_type、name、languages、gender、scenes、
capabilities、notes；目录只收 uranus_bigtts（2.0）音色。

语言标签（"泰语"/"Thai"/"th" 等）先经 doubao_voice_language_key 归一化为
语言码，未知标签回退 zh；每个语言码在目录中至少有一条音色，因此按语言过滤
与子集兜底都不会落空。

语速标定（VOICE_BASELINE_RATES）是音色可用性的总闸门：未标定音色没有
speech_rate 时长控制基准，不进 Object Replace 提示词精选子集、不做默认
音色、dialogue_delivery.pick_voice_for_gender 不选；消费侧把冻结产物里
的未标定绑定回落到默认音色。
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOUBAO_VOICE_CATALOG_PATH = ROOT / "assets" / "doubao-voices" / "catalog.json"
DOUBAO_VOICE_CATALOG_NAME = "duet.doubao-voices"
DOUBAO_VOICE_CATALOG_VERSION = 1

_LANGUAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "zh": ("中文", "汉语", "普通话", "chinese", "mandarin", "zh"),
    "en": ("英语", "英文", "english", "en"),
    "th": ("泰语", "泰文", "thai", "th"),
    "ja": ("日语", "日文", "japanese", "ja"),
    "ko": ("韩语", "朝鲜语", "韩文", "korean", "ko"),
    "es": ("西班牙语", "西语", "spanish", "es"),
    "fr": ("法语", "french", "fr"),
    "de": ("德语", "german", "de"),
    "ru": ("俄语", "russian", "ru"),
    "pt": ("葡萄牙语", "葡语", "portuguese", "pt"),
    "id": ("印度尼西亚语", "印尼语", "indonesian", "id"),
    "ms": ("马来语", "malay", "ms"),
    "vi": ("越南语", "vietnamese", "vi"),
    "tl": ("菲律宾语", "filipino", "tagalog", "tl"),
    "ar": ("阿拉伯语", "arabic", "ar"),
    "it": ("意大利语", "italian", "it"),
}
KNOWN_LANGUAGE_CODES = frozenset(_LANGUAGE_ALIASES)

# 音色 → speech_rate=0 实测合成语速（有效字符/秒）。初值来自 POC 标定
# （data-verify/audio-poc-20260925）。音色目录新增音色时必须先实测标定
# 再入表：未标定音色不进 Object Replace 提示词精选子集、不做默认音色、
# pick_voice_for_gender 不选；冻结产物里的未标定绑定在消费侧回落到默认
# 音色；仍漏到逐行合成的未标定音色由 dialogue-delivery 节点 fail-closed
# 报错兜底。
VOICE_BASELINE_RATES: dict[str, float] = {
    "ms_male_ham_uranus_bigtts": 13.65,
    "es_female_ht_mx_f6_uranus_bigtts": 12.39,
    "en_female_authoritative-informative_uranus_bigtts": 13.87,
    "en_female_joanne_uranus_bigtts": 10.86,
    # 2026-09-27 音频复刻音色性别修复补标定（脚本合成实测，产物在
    # data-verify/audio-regen-20260926/calibration/）：
    # zh_female_vv 以 explicit_language=ms 合成马来语台词实测（多语种
    # 识别音色，语速按马来语文本计）；en_male_adam-imitation 为英语男声。
    "zh_female_vv_uranus_bigtts": 12.09,
    "en_male_adam-imitation_uranus_bigtts": 12.57,
}


def is_calibrated_voice(voice_type: object) -> bool:
    """音色是否有实测基准语速标定（speech_rate 时长控制可用的前提）。"""
    return voice_type in VOICE_BASELINE_RATES


def calibrated_voice_types() -> list[str]:
    """已标定音色清单（字典序，用于错误信息与测试断言）。"""
    return sorted(VOICE_BASELINE_RATES)


def _no_calibrated_voice_error(language_label: str) -> RuntimeError:
    calibrated = ", ".join(calibrated_voice_types()) or "(空)"
    return RuntimeError(
        f"no calibrated voice for {language_label}; "
        f"calibrated voices: {calibrated}"
    )

# ISO 语言码（whisper 检测结果）→ 中文展示标签；目录不支持的语种返回 None，
# 调用方回退到文字推断。
LANGUAGE_CODE_LABELS: dict[str, str] = {
    "zh": "中文",
    "en": "英语",
    "th": "泰语",
    "ja": "日语",
    "ko": "韩语",
    "es": "西班牙语",
    "fr": "法语",
    "de": "德语",
    "ru": "俄语",
    "pt": "葡萄牙语",
    "id": "印度尼西亚语",
    "ms": "马来语",
    "vi": "越南语",
    "tl": "菲律宾语",
    "ar": "阿拉伯语",
    "it": "意大利语",
}


def doubao_voice_language_label(language_code: object) -> str | None:
    """whisper 检测到的 ISO 语言码 → 目录语种标签；未知/不支持返回 None。"""
    if not isinstance(language_code, str):
        return None
    return LANGUAGE_CODE_LABELS.get(language_code.strip().lower())

# 无人物/旁白台词的兜底音色：按语言选该语种第一条已标定的「通用场景」音色。
_DEFAULT_SCENE = "通用场景"

# 注入 Object Replace 提示词的音色精选规则：先按语言过滤、再限定已标定音色
# （未标定音色没有语速基准，看到即可绑定会漏到下游报错），子集仍可能超过
# 提示词预算。按场景相关度排序（靠前更贴合广告短视频配音），组内
# 男女交替保持性别覆盖，截取前 PROMPT_VOICE_LIMIT 条；输出枚举与提示词一致，
# 目录外、子集外的 voice_type 都不可输出。
_PROMPT_SCENE_PRIORITY = (
    "通用场景",
    "视频配音",
    "角色扮演",
    "有声阅读",
    "客服场景",
    "教学场景",
    "教育场景",
    "趣味口音",
    "外语音色",
)
PROMPT_VOICE_LIMIT = 20


def _scene_rank(voice: dict[str, Any]) -> int:
    ranks = [
        _PROMPT_SCENE_PRIORITY.index(scene)
        for scene in voice["scenes"]
        if scene in _PROMPT_SCENE_PRIORITY
    ]
    return min(ranks) if ranks else len(_PROMPT_SCENE_PRIORITY)


def doubao_voice_language_key(language_label: str) -> str:
    """归一化目标语言标签（如“泰语”/"Thai"/"th"）到目录语言码；未知回退 zh。"""
    text = str(language_label or "").strip().lower()
    if not text:
        return "zh"
    for key, aliases in _LANGUAGE_ALIASES.items():
        for alias in aliases:
            if text == alias or (len(alias) > 2 and alias in text):
                return key
    return "zh"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"doubao voice catalog {field} must be a string list")
    return list(value)


@dataclass(frozen=True)
class DoubaoVoiceCatalog:
    voices: tuple[dict[str, Any], ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_by_type",
            {voice["voice_type"]: voice for voice in self.voices},
        )

    def voice(self, voice_type: str) -> dict[str, Any]:
        return self._by_type[voice_type]  # type: ignore[attr-defined]

    def has(self, voice_type: object) -> bool:
        return voice_type in self._by_type  # type: ignore[attr-defined]

    def voice_types(self) -> tuple[str, ...]:
        return tuple(voice["voice_type"] for voice in self.voices)

    def voices_for_language(self, language_label: str) -> tuple[dict[str, Any], ...]:
        """目标语言的音色子集；标签无法解析时回退中文子集（永不为空）。"""
        key = doubao_voice_language_key(language_label)
        subset = tuple(
            voice for voice in self.voices if key in voice["languages"]
        )
        if not subset:
            raise ValueError(f"doubao voice catalog has no {key} voices")
        return subset

    def prompt_voices_for_language(
        self, language_label: str
    ) -> tuple[dict[str, Any], ...]:
        """注入 Object Replace 提示词的精选子集：只含已标定音色（LLM 看到
        即可绑定，未标定音色不可输出），场景相关度排序、组内男女交替、
        上限 PROMPT_VOICE_LIMIT 条；兜底音色始终在子集内。提示词行、输出
        枚举、下游校验必须使用同一子集，不得混用完整目录。语种子集无
        已标定音色时明确报错（标定缺口是部署错误）。"""
        subset = tuple(
            voice
            for voice in self.voices_for_language(language_label)
            if is_calibrated_voice(voice["voice_type"])
        )
        if not subset:
            raise _no_calibrated_voice_error(language_label)
        if len(subset) <= PROMPT_VOICE_LIMIT:
            return subset
        ranked: list[tuple[dict[str, Any], ...]] = []
        for rank in range(len(_PROMPT_SCENE_PRIORITY) + 1):
            group = [voice for voice in subset if _scene_rank(voice) == rank]
            female = [voice for voice in group if voice["gender"] == "female"]
            male = [voice for voice in group if voice["gender"] == "male"]
            interleaved: list[dict[str, Any]] = []
            for pair in itertools.zip_longest(female, male):
                interleaved.extend(voice for voice in pair if voice is not None)
            ranked.append(tuple(interleaved))
        curated = [
            voice for group in ranked for voice in group
        ][:PROMPT_VOICE_LIMIT]
        default = self.default_voice_type(language_label)
        if all(voice["voice_type"] != default for voice in curated):
            curated[-1] = self.voice(default)
        return tuple(curated)

    def default_voice_type(self, language_label: str) -> str:
        """该语种的兜底音色：第一条已标定的通用场景音色，否则该语种第一条
        已标定音色（含 zh_female_vv 这类跨语种已标定音色）；语种子集无
        已标定音色时明确报错——未标定音色没有时长控制基准，不可用。"""
        subset = [
            voice
            for voice in self.voices_for_language(language_label)
            if is_calibrated_voice(voice["voice_type"])
        ]
        for voice in subset:
            if _DEFAULT_SCENE in voice["scenes"]:
                return voice["voice_type"]
        if subset:
            return subset[0]["voice_type"]
        raise _no_calibrated_voice_error(language_label)

    @staticmethod
    def model_line(voice: dict[str, Any]) -> str:
        """一行模型可读摘要；永不输出 JSON。"""
        parts = [
            voice["voice_type"],
            voice["name"],
            ",".join(voice["languages"]),
            voice["gender"],
            ",".join(voice["scenes"]) or "unspecified",
        ]
        if voice["notes"]:
            parts.append(voice["notes"])
        return " - ".join(parts)

    def model_lines(self, voices: tuple[dict[str, Any], ...] | None = None) -> str:
        return "\n".join(self.model_line(voice) for voice in (voices or self.voices))

    @staticmethod
    def prompt_line(voice: dict[str, Any]) -> str:
        """提示词精选子集的极简行：语种已由过滤保证、备注冗长，均不下发。"""
        return " - ".join([
            voice["voice_type"],
            voice["name"],
            voice["gender"],
            ",".join(voice["scenes"]) or "unspecified",
        ])

    def prompt_lines(self, voices: tuple[dict[str, Any], ...]) -> str:
        return "\n".join(self.prompt_line(voice) for voice in voices)


def load_doubao_voice_catalog(
    path: Path = DOUBAO_VOICE_CATALOG_PATH,
) -> DoubaoVoiceCatalog:
    catalog_path = path.resolve(strict=True)
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not {"schema", "version", "voices"} <= set(raw):
        raise ValueError("doubao voice catalog root is invalid")
    if (
        raw["schema"] != DOUBAO_VOICE_CATALOG_NAME
        or type(raw["version"]) is not int
        or raw["version"] != DOUBAO_VOICE_CATALOG_VERSION
    ):
        raise ValueError("doubao voice catalog identity is invalid")
    voices = raw["voices"]
    if not isinstance(voices, list) or not voices:
        raise ValueError("doubao voice catalog must contain at least one voice")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, value in enumerate(voices, 1):
        if not isinstance(value, dict) or not {
            "voice_type", "name", "languages", "gender",
            "scenes", "capabilities", "notes",
        } <= set(value):
            raise ValueError(f"doubao voice catalog entry {index} is invalid")
        voice_type = value["voice_type"]
        if not isinstance(voice_type, str) or not voice_type:
            raise ValueError("doubao voice catalog voice_type is invalid")
        if voice_type in seen:
            raise ValueError("doubao voice catalog voice_type is duplicated")
        seen.add(voice_type)
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("doubao voice catalog name is invalid")
        languages = _string_list(value["languages"], "languages")
        if any(language not in KNOWN_LANGUAGE_CODES for language in languages):
            raise ValueError("doubao voice catalog languages are invalid")
        if value["gender"] not in {"female", "male"}:
            raise ValueError("doubao voice catalog gender is invalid")
        validated.append({
            "voice_type": voice_type,
            "name": value["name"],
            "languages": languages,
            "gender": value["gender"],
            "scenes": _string_list(value["scenes"], "scenes"),
            "capabilities": _string_list(value["capabilities"], "capabilities"),
            "notes": (
                value["notes"]
                if isinstance(value["notes"], str)
                else (_ for _ in ()).throw(
                    ValueError("doubao voice catalog notes are invalid")
                )
            ),
        })
    return DoubaoVoiceCatalog(
        voices=tuple(validated),
        digest=hashlib.sha256(_canonical(raw)).hexdigest(),
    )


@functools.lru_cache(maxsize=1)
def cached_doubao_voice_catalog() -> DoubaoVoiceCatalog:
    return load_doubao_voice_catalog()


def known_voice_types() -> frozenset[str]:
    return frozenset(cached_doubao_voice_catalog().voice_types())
