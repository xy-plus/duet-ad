from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from app import media
from app import artifact_retention
from app import object_detect
from app import object_mask
from app import subtitles
from app import person_identity
from app import product_views
from app import vocal_separation
from app.config import Settings
from app.fusion_schema import fusion_prompt_schema, normalize_fusion_prompt
from app.object_replace_schema import (
    OBJECT_REPLACE_OUTPUT_VERSION,
    merge_object_replace_index,
    normalize_object_replace,
    object_replace_planning_context,
    object_replace_schema,
)
from app.replacement_sheet import (
    build_replacement_sheet_spec,
    finalize_replacement_sheet,
    materialize_replacement_sheet_inputs,
    replacement_sheet_spec_sha256,
    validate_replacement_sheet_spec,
)
from app.providers import (
    ArkAudioClient,
    CodexClient,
    CodexProcessError,
    ContextIRTaskFailedError,
    DeepSeekClient,
    MediaKitClient,
    MiniMaxContextIRClient,
    PicsartClient,
    ProviderFailure,
    ProviderResult,
)
from app.doubao_tts import (
    STREAM_FINISHED_OUTCOME,
    DoubaoTtsClient,
    EmptyAudioError,
)
from app.dialogue_delivery import (
    DIALOGUE_DELIVERY_OUTPUT_VERSION,
    build_dialogue_delivery_payload,
    dialogue_delivery_instruction,
    dialogue_delivery_instruction_sha256,
    dialogue_delivery_schema,
    dialogue_tts_line_input_sha256,
    dialogue_tts_section_id,
    is_decodable_wav,
    normalize_dialogue_delivery,
    resolve_line_voices,
)
from app.doubao_voices import (
    DoubaoVoiceCatalog,
    doubao_voice_language_label,
    is_calibrated_voice,
    load_doubao_voice_catalog,
)
from app.seedream import (
    SEEDREAM_OUTPUT_HEIGHT,
    SEEDREAM_OUTPUT_SIZE,
    SEEDREAM_OUTPUT_WIDTH,
    SeedreamClient,
    _atomic_json as _persist_receipt_json,
    _validated_image_media_type,
    reset_attempt as _seedream_reset_attempt,
)
from app.research_log import ResearchLog, display_duration_seconds
from app import prompt_review
from app.prompt_review import PromptReviewError
from app.store import ProjectStore, replacement_input_paths
from app.video_maker_schemas import (
    VISUAL_PROMPT_SCHEMA,
    normalize_project_index,
    normalize_visual_prompt,
    project_index_schema,
)


VIDEO_SKILL = Path(__file__).resolve().parents[1] / "skills" / "video-maker" / "SKILL.md"
R2V_SKILL = Path(__file__).resolve().parents[1] / "skills" / "h3-max-r2v-prompt" / "SKILL.md"
OBJECT_REPLACE_SKILL = (
    Path(__file__).resolve().parents[1] / "skills" / "object-replace" / "SKILL.md"
)
DIALOGUE_WRITER_SKILL = (
    Path(__file__).resolve().parents[1] / "skills" / "dialogue-writer" / "SKILL.md"
)
SAME_AS_SOURCE_LANGUAGE = "与原视频相同"
MASTER_EMPTY_ASR_SIGNAL = 51 / 256
AUDIO_ANALYSIS_OUTPUT_VERSION = 1
# ark-audio-analysis 节点的逐行提示词（POC scripts/audio_poc/analyze_ark.py
# 定稿口径，transcript 字段帮助模型先转写再描述，落 state 时只取五字段）。
ARK_AUDIO_ANALYSIS_PROMPT = """这是一段广告视频的原声配音片段。请仔细听，然后只输出一个 JSON 对象（不要 markdown 代码围栏），字段如下：
{
  "transcript": "你听到的原文转写",
  "speech_rate_desc": "语速描述（慢/中等/偏快/快，并估计大致字符每秒）",
  "intonation_emotion": "语调与情绪描述（如热情、平稳、急促、上扬等）",
  "pause_pattern": "停顿模式描述（停顿位置、长短、节奏）",
  "overall_tone": "整体语气与风格（如带货激情、教学亲切等）",
  "gender_age": "说话人性别与年龄段猜测"
}"""
_ARK_AUDIO_ANALYSIS_FIELDS = (
    "speech_rate_desc",
    "intonation_emotion",
    "pause_pattern",
    "overall_tone",
    "gender_age",
)
_ARK_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# fusion payload 注入的 audio_analysis 字段（spec 2026-09-28 §2）。
# gender_age 不进：音色绑定已有确定性规则（resolve_line_voices 画面性别
# 校正链），再注入音频猜测会形成双事实源冲突。
_FUSION_AUDIO_ANALYSIS_FIELDS = (
    "intonation_emotion",
    "overall_tone",
    "speech_rate_desc",
    "pause_pattern",
)


def _numeric_window(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fusion_audio_analysis_rows(
    segment_start_s: Any, segment_end_s: Any, audio_analysis: Any,
) -> list[dict[str, Any]] | None:
    """fusion payload 的 audio_analysis 注入（spec 2026-09-28 §2）。

    归属判据：state.audio_analysis.lines[] 的全局行窗中点落入本段全局
    窗 [start_s, end_s) 即归本段——段窗是对时间轴的划分，中点必唯一
    归属，零歧义，不做 overlap 比例计算。不能按 line_index 对齐：全局
    translated_lines 无 line_index 字段，dialogue-localize 会拆行并把
    start_s/end_s 改写为段内局部时间（app/media.py localize_dialogue，
    2026-09-28 审查修正）。缺失降级省略整字段：无 audio_analysis（老
    项目/分析失败）、段窗非数值、段窗内无任何行中点或行四字段全不可
    用时返回 None，fusion 仅凭 visual_prompt 判断语气。
    """
    if not _numeric_window(segment_start_s) or not _numeric_window(
        segment_end_s
    ):
        return None
    if not segment_start_s < segment_end_s:
        return None
    if not isinstance(audio_analysis, dict):
        return None
    analysis_lines = audio_analysis.get("lines")
    if not isinstance(analysis_lines, list):
        return None
    rows: list[dict[str, Any]] = []
    for entry in analysis_lines:
        if not isinstance(entry, dict):
            continue
        start_s, end_s = entry.get("start_s"), entry.get("end_s")
        if not _numeric_window(start_s) or not _numeric_window(end_s):
            continue
        if not start_s < end_s:
            continue
        midpoint = (start_s + end_s) / 2.0
        if not segment_start_s <= midpoint < segment_end_s:
            continue
        row: dict[str, Any] = {}
        for field in _FUSION_AUDIO_ANALYSIS_FIELDS:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                row[field] = value
        if row:
            rows.append(row)
    return rows or None


def parse_ark_audio_analysis(text: str) -> dict[str, str]:
    """解析方舟音频理解的自由 JSON 输出（POC analyze_ark.py 口径：剥
    markdown 围栏、截取首个 { 到末个 }、strict=False 容错）。五字段缺任一
    或值不是非空字符串即 ValueError，节点按暂时性模型输出失败重试。"""
    cleaned = _ARK_JSON_FENCE_RE.sub("", text.strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(cleaned, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("ark audio analysis output is not a JSON object")
    description: dict[str, str] = {}
    for field in _ARK_AUDIO_ANALYSIS_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ark audio analysis field {field} is invalid")
        description[field] = value.strip()
    return description


_NO_REUSABLE_OUTPUT: Any = object()
# These labels describe transient or ambient physical phenomena, rather than
# objects with an identity that needs replacement.  Object Replace normally
# omits them; _replaced_object_targets checks every synonym phrase as
# defense in depth for imperfect or frozen planning output.
_TRANSIENT_PHYSICAL_EFFECT_DETECT_LABELS = frozenset(
    {
        "mist",
        "fog",
        "smoke",
        "steam",
        "vapor",
        "spray",
        "splash",
        "water",
        "fire",
        "flame",
        "spark",
        "ember",
        "dust",
        "particle",
        "foam",
        "bubble",
        "glare",
        "glow",
    }
)
TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}
DIALOGUE_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["segment_index", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _validate_translation_shape(value: dict[str, Any]) -> None:
    """Transport-shape gate for the JSON repair loop; content stays lenient."""
    lines = value.get("lines")
    if not isinstance(lines, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str)
        for item in lines
    ):
        raise ValueError("translation output is invalid")


def _validate_dialogue_generation_shape(value: dict[str, Any]) -> None:
    """Transport-shape gate for the JSON repair loop; content stays lenient."""
    lines = value.get("lines")
    if not isinstance(lines, list) or any(
        not isinstance(item, dict)
        or type(item.get("segment_index")) is not int
        or not isinstance(item.get("text"), str)
        for item in lines
    ):
        raise ValueError("dialogue generation output is invalid")


def _script_family(ch: str) -> str | None:
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return "han"
    if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "kana"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
        return "hangul"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "arabic"
    if "a" <= ch <= "z" or "A" <= ch <= "Z":
        return "latin"
    return None


_SCRIPT_LANGUAGE = {
    "han": "中文",
    "kana": "日语",
    "hangul": "韩语",
    "thai": "泰语",
    "cyrillic": "俄语",
    "arabic": "阿拉伯语",
    "latin": "英语",
}
_LANGUAGE_FAMILY = {language: family for family, language in _SCRIPT_LANGUAGE.items()}


def _dominant_script_language(text: str) -> str | None:
    """Classify text by script evidence; None when there is no letter evidence.

    whisper.cpp runs with ``-l auto`` and does detect a language, but
    media.transcribe only persists the transcript lines into state, so the
    transcript text itself is the strongest available language signal.
    Kana, hangul, thai, cyrillic and arabic are exclusive to one language
    each, so their presence is diagnostic (a decorative の in Chinese copy
    stays Chinese because its kana volume is dwarfed by han). Han vs latin
    is decided by majority. Latin-script languages all classify as 英语;
    that is the documented limit of this heuristic (digits, punctuation
    and emoji carry no signal)."""
    counts: dict[str, int] = {}
    for ch in text:
        family = _script_family(ch)
        if family is not None:
            counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    han = counts.get("han", 0)
    latin = counts.get("latin", 0)
    for family in ("kana", "hangul", "thai", "cyrillic", "arabic"):
        exclusive = counts.get(family, 0)
        if exclusive and exclusive * 2 >= han and exclusive >= latin:
            return _SCRIPT_LANGUAGE[family]
    if han and han >= latin:
        return "中文"
    if latin:
        return "英语"
    return None


def _resolve_dialogue_language(
    target_language: str,
    voice_lines: list[dict[str, Any]],
    brief: str,
    source_language_code: str | None = None,
) -> str:
    """Resolve the dialogue language deterministically for DeepSeek.

    A concrete target passes through unchanged. The same-as-source
    placeholder inherits the language whisper detected from the source
    audio track itself (``source_language_code``); only when that signal is
    absent (old frozen artifacts, unknown code) does the transcript's
    dominant script decide. When the source video has no dialogue there is
    nothing to inherit, so the brief's dominant script decides (a brief
    written in Chinese almost always wants Chinese narration). With no
    signal at all the ambiguity is raised instead of silently guessed."""
    if target_language != SAME_AS_SOURCE_LANGUAGE:
        return target_language
    label = doubao_voice_language_label(source_language_code)
    if label is not None:
        return label
    source_text = " ".join(
        str(line.get("text") or "")
        for line in voice_lines
        if isinstance(line, dict)
    )
    language = _dominant_script_language(source_text)
    if language is not None:
        return language
    language = _dominant_script_language(brief)
    if language is not None:
        return language
    raise ValueError(
        f"target_language 为“{SAME_AS_SOURCE_LANGUAGE}”，但源视频无台词且 "
        "brief 无法判定语种；请把 target_language 改为具体语种"
    )


def _dialogue_language_display(
    target_language: str,
    sample_lines: list[dict[str, Any]],
    source_language_code: str | None = None,
) -> str:
    """Concrete language label for model payloads.

    Models never see the source video, so the same-as-source placeholder is
    noise to them: it is always resolved to a concrete language here (the
    "与原视频相同" provenance stays in state.json's effective_request).
    whisper 的音轨语种检测优先；旧冻结产物没有该字段时回退台词文字推断。
    With no dialogue at all the language has no consumer, so the label is
    empty rather than an invented language."""
    if target_language != SAME_AS_SOURCE_LANGUAGE:
        return target_language
    label = doubao_voice_language_label(source_language_code)
    if label is not None:
        return label
    text = " ".join(
        str(line.get("text") or "")
        for line in sample_lines
        if isinstance(line, dict)
    )
    language = _dominant_script_language(text)
    if language is None:
        return ""
    return language


# Language-consistency gate for generated dialogue: fail-open by design.
# It only rejects a clear contradiction (the wrong script dominates the
# output), never borderline mixes, so normal output can never loop the
# JSON repair cycle forever.
_DIALOGUE_LANGUAGE_MIN_LETTERS = 8
_DIALOGUE_LANGUAGE_DOMINANCE = 0.7


def _validate_dialogue_language(
    value: dict[str, Any], expected_language: str,
) -> None:
    """Reject generated dialogue whose dominant script contradicts the target.

    Only fires on an unambiguous mismatch (wrong script family at
    >= _DIALOGUE_LANGUAGE_DOMINANCE of >= _DIALOGUE_LANGUAGE_MIN_LETTERS
    script letters); brand words, numbers and mixed text pass through, and
    target languages without a script mapping are not gated at all."""
    expected_family = next(
        (
            family
            for language, family in _LANGUAGE_FAMILY.items()
            if language in expected_language
        ),
        None,
    )
    if expected_family is None:
        return
    lines = value.get("lines")
    items = lines if isinstance(lines, list) else []
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        for ch in item["text"]:
            family = _script_family(ch)
            if family is not None:
                counts[family] = counts.get(family, 0) + 1
    total = sum(counts.values())
    if total < _DIALOGUE_LANGUAGE_MIN_LETTERS:
        return
    dominant_family = max(counts, key=counts.get)
    if dominant_family == expected_family:
        return
    if counts[dominant_family] / total >= _DIALOGUE_LANGUAGE_DOMINANCE:
        raise ValueError(
            f"台词语言应为{expected_language}，但输出以"
            f"{_SCRIPT_LANGUAGE[dominant_family]}为主；请用指定语言重写全部台词"
        )

PICSART_RECEIPT_VERSION = 2
H3_PREPARED_ARTIFACTS_VERSION = 7
REPLACEMENT_SHEET_VERSION = 4
PICSART_PROMPT_MAX_CHARS = 50_000
PICSART_REFERENCE_MIN_DURATION_S = 2
PICSART_REFERENCE_MAX_DURATION_S = 15
PICSART_GENERATION_MIN_DURATION_S = 5
PICSART_GENERATION_MAX_DURATION_S = 15
PICSART_RESOLUTIONS = frozenset({"480p", "768p"})
PICSART_ASPECT_RATIOS = frozenset({"16:9", "9:16"})
# h3-context-ir 节点（MiniMax 官方 Context IR 提示词优化）：每段总尝试上限
# 3 次——unknown 提交（POST 后无 task_id）与 vendor failed（失败不计费）
# 共用此上限（R5 风格，单次 ~$0.02）；传输类错误不值得为可选增强无限退避，
# 同样计入；确定性 4xx（含官方限长：实测 content text ≤7000 字符，超出同步
# 400 code 2013，2026-09-29 实证）一次即 fail-closed 不重试。
CONTEXT_IR_MAX_ATTEMPTS = 3
# 官方硬上限：content 只允许 1 个 text 项且 ≤7000 字符（两个 400 证据见
# data-verify/context-ir-20260929/receipts/）。我方冻结 prompt（8939-9840
# 字符）全超限，故发给 IR 的是蒸馏输入（_context_ir_input_text，目标
# ≤4000），不是 h3_prompt.txt 全文；IR 返回的官方结构化增强文本仍全量
# 覆写 h3_prompt.txt。
CONTEXT_IR_INPUT_MAX_CHARS = 7000


class ContextIRInputTooLongError(ValueError):
    """蒸馏输入超官方 7000 字符硬上限：与 vendor 400/2013 同语义，fail-closed。"""


class ContextIRSanitizeError(ValueError):
    """IR 输出清洗后 <d> 台词标签为空：结果不可用，节点走 fallback。"""


# 实测：IR 把跨硬切台词拆到相邻 Shot 时会在 <d> 内泄漏 <scenetrans> 残片
#（data-verify/context-ir-20260929 seg1/seg2 证据）。只在 <d>…</d> 内剥除，
# 台词词汇逐字不动。
_CONTEXT_IR_SCENETRANS_RE = re.compile(r"<scenetrans\s*/?>")
_CONTEXT_IR_DIALOGUE_RE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_CONTEXT_IR_LANGUAGE_TAG_RE = re.compile(r"^\s*\[[^\]]*\]")


def _context_ir_sanitize_output(text: str) -> tuple[str, int]:
    """剥除 <d>…</d> 内的 <scenetrans> 残片；返回 (清洗后文本, 剥除计数)。

    清洗后某条 <d> 为空（原内容只有残片/语言标签）时抛
    ContextIRSanitizeError——该段增强结果不可用，节点走 fallback。
    """
    removed = 0

    def _clean(match: re.Match[str]) -> str:
        nonlocal removed
        inner = match.group(1)
        cleaned, count = _CONTEXT_IR_SCENETRANS_RE.subn("", inner)
        removed += count
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        spoken = _CONTEXT_IR_LANGUAGE_TAG_RE.sub("", cleaned).strip()
        if not spoken:
            raise ContextIRSanitizeError(
                "Context IR 输出清洗后 <d> 台词为空，结果不可用"
            )
        return f"<d>{cleaned}</d>"

    return _CONTEXT_IR_DIALOGUE_RE.sub(_clean, text), removed


# IR 输出是官方固定结构：editing/face/dialogue 语义被吸收（官方改写），但
# 实测 mask 透出禁令与 overlay 擦除指令不被吸收（第五轮实证）。这两条由
# 节点在清洗后机械追加，不依赖 IR。措辞压缩自冻结 prompt 英文原文
#（f8280ad1 段1，原文摘抄见 data-verify/context-ir-20260929/report.md）。
_CONTEXT_IR_APPEND_MASK_THROUGH = (
    "The faint blurred source imagery showing through the semi-transparent "
    "white blocks serves ONLY as position, motion, and composition reference: "
    "never reconstruct, enhance, sharpen, or copy anything visible through "
    "the blocks."
)
_CONTEXT_IR_APPEND_OVERLAY_REMOVE = (
    "Remove all of them from the generated video and reconstruct the occluded "
    "background naturally. Do not reproduce, translate, or re-render any "
    "source on-screen text, watermark, or logo."
)
_CONTEXT_IR_APPEND_HEADER = (
    "# Backend-mandated constraints (not absorbed by Context IR):"
)
# 形态硬约束（方案 B，2026-10-03，根因项目 081e3436）：IR 蒸馏丢设备属性时
# H3 按训练先验错渲染产品形态（电动加热睫毛夹被画成机械钢丝款）。与既有
# 追加句同块机械追加，只在提取到【属性与用法】段落时追加。
_CONTEXT_IR_APPEND_FORM_AUTHORITY = (
    "绑定参考图（图版 tile 与用户参考图）是该物品形状、结构与功能的唯一权威；"
    "禁止把该物品渲染成与参考图不同的通用款或其他类型"
    "（例如把电动加热款画成机械钢丝款）。"
)

# 属性与用法段落（skill h3-max-r2v-prompt 第 5 条要求 DeepSeek 写进组装
# 后的冻结提示词正文，不在 visual_prompt.txt 里），历史上从未进入 IR
# 蒸馏输入——IR 自行补全的设备描述丢失区分性特征（081e3436：seg2 被
# 补全成机械钢丝夹特征）。提取函数供两处复用：方案 A 进蒸馏输入
# [属性与用法] 小节（每段截断 800 字符防极端长文），方案 B 在 IR 成功
# 输出尾部逐字机械追加（不截断）。
# 存在判据（2026-10-03 用户拍板）：DeepSeek 段头格式不可枚举（081e3436
# 两次运行就出了【属性与用法】/ 属性与用法（产品名）：/ 属性与用法：
# 三种），不做格式白名单——行首附近（≤8 字符前缀，容忍【/#/空格等装
# 饰）出现"属性与用法"且紧随其后是 】（（：: 之一的行即段落起点；紧随
# 字符约束挡住"他说属性与用法很重要"这类正文提及。
_USAGE_MARKER = "属性与用法"
_USAGE_MARKER_TAIL_CHARS = frozenset({"】", "（", "(", "：", ":"})
_USAGE_PARAGRAPH_DISTILL_MAX_CHARS = 800
# 段落终止边界之一：ASCII 字母开头的英文块头行（如
# "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN"）。
_ENGLISH_BLOCK_HEADER_RE = re.compile(r"^[A-Za-z]")


def _usage_marker_line_start(stripped: str) -> bool:
    """行首附近（≤8 字符前缀）含"属性与用法"且紧随其后是块装饰符。"""
    marker_at = stripped.find(_USAGE_MARKER)
    if marker_at < 0 or marker_at > 8:
        return False
    tail = stripped[marker_at + len(_USAGE_MARKER):]
    return tail[:1] in _USAGE_MARKER_TAIL_CHARS


def _extract_usage_paragraphs(origin_prompt: str) -> list[str]:
    """从冻结原文（IR 前的 h3_prompt 原文）提取所有属性与用法段落，逐字。

    段落边界：从标记行（_usage_marker_line_start）起，到空行、下一个【
    块头行、或英文块头行（ASCII 字母开头）为止；多行段落原样拼接。提取
    不到（非商品项目/skill 未写）返回空列表，不编造；只有标记没有内容
    的空段跳过。
    """
    if not isinstance(origin_prompt, str):
        raise ValueError("origin prompt must be a string")
    paragraphs: list[str] = []
    current: list[str] | None = None

    def _flush() -> None:
        nonlocal current
        if current is None:
            return
        text = "\n".join(current).strip()
        current = None
        if text:
            paragraphs.append(text)

    for line in origin_prompt.splitlines():
        stripped = line.strip()
        if _usage_marker_line_start(stripped):
            _flush()
            current = [stripped]
            continue
        if current is None:
            continue
        if (
            not stripped
            or stripped.startswith("【")
            or _ENGLISH_BLOCK_HEADER_RE.match(stripped)
        ):
            _flush()
            continue
        current.append(stripped)
    _flush()
    return paragraphs


def _require_fusion_usage_paragraphs(
    prompt: str,
    *,
    user_bound_product: bool,
) -> None:
    """fusion 正文的属性与用法硬门禁（2026-10-03 新增）。

    根因：081e3436 seg2 在 resume-h3-prep 重跑时 DeepSeek 漏写该段（skill
    第 5 条是软约束），下游提取/追加全部落空，IR 自行补全的设备描述丢
    区分性特征，H3 按训练先验把电动加热睫毛夹渲染成机械钢丝夹。段内有
    绑定用户参考图的被替换商品时，正文必须含"属性与用法"——存在判据
    就是包含该字面值（2026-10-03 用户拍板：段头格式不可枚举，包含即
    成功，不做格式白名单）；缺失即 ValueError，由 complete_structured
    的 JSON 修复环 + 节点重试让 DeepSeek 带反馈重写，而不是带缺陷放行。
    user_bound_product=False（无商品/纯换人项目）不门禁。
    """
    if user_bound_product and _USAGE_MARKER not in prompt:
        raise ValueError(
            "fusion prompt missing 属性与用法 paragraph for user-bound product"
        )


def _context_ir_append_backend_constraints(
    text: str,
    *,
    mask_profile: dict[str, Any] | None,
    processing: dict[str, Any] | None,
    usage_paragraphs: list[str],
) -> tuple[str, list[str]]:
    """清洗后追加 IR 实证不吸收的约束尾部块；返回 (最终文本, 追加的句子列表)。

    - mask 透出禁令：仅 masked 且物品半透块段（与蒸馏 [遮挡块] 半透分支同
      开关：objects_masked && object_blocked && object_semi_transparent）。
    - overlay 擦除：processing 开关驱动（remove_subtitle/remove_logo 任一），
      与台词无关——无台词段同样追加；我方没有"源片确定无烧录文字"的现成
      证据字段，开关开就无条件追加（宁可多一条指令）。
    - 【属性与用法】段落（方案 B）：逐字机械追加（不加 bullet、不截断），
      再补一句形态硬约束（_CONTEXT_IR_APPEND_FORM_AUTHORITY）；只在提取到
      段落时追加，空列表什么都不加。调用方保证只在 IR 成功覆写路径传入：
      fallback/禁用路径原文本身已含这些段落，不得重复追加。
    """
    if not isinstance(usage_paragraphs, list) or any(
        not isinstance(paragraph, str) or not paragraph.strip()
        for paragraph in usage_paragraphs
    ):
        raise ValueError("usage_paragraphs must be a list of non-empty strings")
    sentences: list[str] = []
    if (
        isinstance(mask_profile, dict)
        and mask_profile.get("objects_masked")
        and mask_profile.get("object_blocked")
        and mask_profile.get("object_semi_transparent")
    ):
        sentences.append(_CONTEXT_IR_APPEND_MASK_THROUGH)
    processing = processing if isinstance(processing, dict) else {}
    overlay_targets: list[str] = []
    if processing.get("remove_subtitle"):
        overlay_targets.append(
            "burned-in subtitles, captions, and on-screen text"
        )
    if processing.get("remove_logo"):
        overlay_targets.append("watermarks and logos")
    if overlay_targets:
        sentences.append(
            "Video 1 may contain "
            + " and ".join(overlay_targets)
            + ". "
            + _CONTEXT_IR_APPEND_OVERLAY_REMOVE
        )
    paragraphs = [paragraph.strip() for paragraph in usage_paragraphs]
    appended: list[str] = [*sentences, *paragraphs]
    if paragraphs:
        appended.append(_CONTEXT_IR_APPEND_FORM_AUTHORITY)
    if not appended:
        return text, []
    tail_lines = [_CONTEXT_IR_APPEND_HEADER]
    tail_lines.extend(f"- {sentence}" for sentence in sentences)
    tail_lines.extend(paragraphs)  # 逐字段落，不加 bullet
    if paragraphs:
        tail_lines.append(f"- {_CONTEXT_IR_APPEND_FORM_AUTHORITY}")
    return text.rstrip() + "\n\n" + "\n".join(tail_lines) + "\n", appended


def _context_ir_frozen_chain(segment_dir: Path, frozen_sha256: str) -> bool:
    """IR 后冻结校验链：origin==冻结 && 回执.origin==冻结 && 当前==optimized。

    冻结语义（2026-09-30 顺序调整）：`_freeze_h3_prepared_artifacts` 冻结的
    是 IR 前原文；IR 成功后 h3_prompt.txt 是增强文本。可验证链：
    h3_prompt_origin.txt 哈希 == 冻结哈希，且 context_ir.json 的
    origin_prompt_sha256 == 冻结哈希，且 h3_prompt.txt 哈希 ==
    optimized_prompt_sha256。旧顺序项目（冻结=IR 输出）h3_prompt.txt ==
    冻结哈希，走原文校验即过、不进本链，天然兼容。比较全部基于文件实际
    哈希，不信任任何状态记录。
    """
    origin_path = segment_dir / "h3_prompt_origin.txt"
    receipt_path = segment_dir / "context_ir.json"
    prompt_path = segment_dir / "h3_prompt.txt"
    if not (
        origin_path.is_file()
        and not origin_path.is_symlink()
        and receipt_path.is_file()
        and not receipt_path.is_symlink()
        and prompt_path.is_file()
        and not prompt_path.is_symlink()
    ):
        return False
    try:
        receipt_doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(receipt_doc, dict) or receipt_doc.get("status") != "succeeded":
        return False
    try:
        origin_digest = hashlib.sha256(origin_path.read_bytes()).hexdigest()
        current_digest = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    except OSError:
        return False
    return (
        origin_digest == frozen_sha256
        and receipt_doc.get("origin_prompt_sha256") == frozen_sha256
        and receipt_doc.get("optimized_prompt_sha256") == current_digest
    )


def _context_ir_identity_anchor(
    element_index: dict[str, Any], stable_key: str,
) -> str | None:
    """人物身份锚（性别/年龄段/人种）：取 object-replace 输出的
    replacement.appearance_description 首句前缀（"："之前，含性别/年龄/人种），
    不编造；取不到返回 None。"""
    people = element_index.get("people")
    entry = people.get(stable_key) if isinstance(people, dict) else None
    replacement = entry.get("replacement") if isinstance(entry, dict) else None
    description = (
        replacement.get("appearance_description")
        if isinstance(replacement, dict) else None
    )
    if not isinstance(description, str) or not description.strip():
        return None
    anchor = description.strip().split("：", 1)[0].split(":", 1)[0].strip()
    return anchor or None


def _context_ir_input_text(
    *,
    visual_prompt: str,
    dialogue_lines: list[dict[str, Any]],
    target_language: str | None,
    replacement_sheet: dict[str, Any],
    element_index: dict[str, Any],
    segment_index: int,
    mask_profile: dict[str, Any] | None = None,
    processing: dict[str, Any] | None = None,
    usage_paragraphs: list[str],
    voice_over: bool,
) -> str:
    """构建发给 Context IR 的蒸馏输入文本（可测纯函数）。

    冻结 prompt 的固定五块（editing/mask/face/overlay/dialogue）一块不丢、
    全部压缩进对应小节：[任务]=editing，[遮挡块]=mask（按 mask_profile
    与参考流真实内容对齐），[替换绑定] 身份锚+[保留与禁令] 首句=face
    显式优先级，[画面文字]=overlay（processing 开关驱动），[台词]=dialogue。
    另含：素材编号映射（Image 1..N 冻结顺序 / Video 1 / Audio 1）、
    [分镜归属]（occurrences 的 frame_orders 即视觉正文"第 N 时点"口径）、
    [保留道具]（本段出镜的 KEEP entities：时点+原样保留+不得归并，防止
    IR 把未登记道具张冠李戴给绑定 Subject）、[属性与用法]（方案 A：
    冻结原文里 skill 第 5 条写的设备属性段落逐字收录、每段截断 800
    字符；根因 081e3436——这些段落只存在于组装后正文、不在
    visual_prompt.txt，历史上从未发给 IR，IR 自行补全的设备描述丢失
    区分性特征，H3 按训练先验把电动加热睫毛夹渲染成机械钢丝款）。
    剔除：账本块冗长尾、重复描述。
    目标 ≤4500 字符；构建后断言 ≤ CONTEXT_IR_INPUT_MAX_CHARS，超限抛
    ContextIRInputTooLongError（fail-closed，不走 fallback）。
    """
    if not isinstance(visual_prompt, str) or not visual_prompt.strip():
        raise ValueError("Context IR 蒸馏需要非空 visual_prompt")
    if not isinstance(usage_paragraphs, list) or any(
        not isinstance(paragraph, str) or not paragraph.strip()
        for paragraph in usage_paragraphs
    ):
        raise ValueError("Context IR usage_paragraphs must be non-empty strings")

    references = replacement_sheet.get("h3_reference_images") or []
    user_images = [
        reference for reference in references
        if isinstance(reference, dict)
        and reference.get("role") == "user_replacement"
    ]
    # 用户图编号：image_i → 冻结序第 i 张用户图（其 image_reference 字段即
    # 全局编号，Image 1 恒为图版）。
    user_image_labels = {
        f"image_{order}": reference.get("image_reference")
        for order, reference in enumerate(user_images, 1)
    }

    lines: list[str] = [
        "[任务] 参照 Video 1 做替换编辑：逐镜动作、机位、构图、灯光、场景与"
        "风格严格保持源片，仅替换下方绑定的元素。",
    ]
    # [遮挡块]（mask 块全语义）：措辞与参考流真实内容对齐（mask_profile
    # 来自 mask 节点，口径同提示词 mask_block）。
    if isinstance(mask_profile, dict) and mask_profile.get("objects_masked"):
        mask_lines: list[str] = []
        if mask_profile.get("person_blocked"):
            mask_lines.append(
                "纯黑 (0,0,0) 块完全遮盖待替换人物——脸、头发、身体与服装全部包含"
            )
        if mask_profile.get("object_blocked"):
            if mask_profile.get("object_semi_transparent"):
                mask_lines.append(
                    "半透明白 (255,255,255) 块遮盖待替换物品，白块下透出的模糊"
                    "源画面仅作位置、动作与构图参考"
                )
            else:
                mask_lines.append("白色块遮盖待替换物品")
        mask_lines.append(
            "禁止复原、增强、锐化或照抄任何块下可见内容；禁止画出或还原任何"
            "块遮盖的内容：人物按 [替换绑定] 的描述自然重建，物品必须按绑定"
            "参考图重建——identity、形状、尺寸、比例与朝向一致，按其物理属性"
            "行为，以与其功能一致的方式被持有、佩戴或使用；禁止为了贴合源"
            "物品而改形、改尺寸或改用途"
        )
        lines.extend(["", "[遮挡块] " + "；".join(mask_lines) + "。"])
    lines.extend([
        "",
        "[视觉正文]",
        visual_prompt.strip(),
        "",
        "[素材]",
        "- Image 1：新形象图版（网格图，每个 tile 是一个被替换人物/物品的"
        "新形象权威）",
    ])
    for reference in user_images:
        label = reference.get("image_reference")
        if isinstance(label, str) and label:
            lines.append(f"- {label}：用户参考图")
    lines.append("- Video 1：源片段参考流（遮挡块区域为待替换元素占位）")
    if dialogue_lines and not voice_over:
        lines.append("- Audio 1：台词精确轨（对口型唯一依据）")

    tiles = (
        (replacement_sheet.get("spec") or {}).get("tiles")
        if isinstance(replacement_sheet.get("spec"), dict) else None
    ) or []
    binding_lines: list[str] = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        stable_key = tile.get("stable_key")
        tile_id = tile.get("tile_id")
        if not isinstance(stable_key, str) or not isinstance(tile_id, str):
            continue
        reference_ids = tile.get("reference_image_ids")
        bound_labels = [
            user_image_labels[reference_id]
            for reference_id in (reference_ids or [])
            if reference_id in user_image_labels
            and user_image_labels[reference_id]
        ]
        # 人物身份锚（性别/年龄段/人种）：IR 看不到灰块下的源人物性别，
        # 没有身份锚时会按绑定列表顺序向镜头分配（f8280ad1 段3 实测男女
        # 互换事故）。只取 object-replace 输出原文，不编造。
        anchor = (
            _context_ir_identity_anchor(element_index, stable_key)
            if tile.get("category") == "people"
            else None
        )
        prefix = f"- {stable_key}（{anchor}）" if anchor else f"- {stable_key}"
        if bound_labels:
            binding_lines.append(
                f"{prefix} → {' / '.join(bound_labels)}（用户参考图）"
            )
        else:
            binding_lines.append(
                f"{prefix} → Image 1 {tile_id}"
                f"（图版 r{tile.get('row')}c{tile.get('column')}）"
            )
    if binding_lines:
        lines.extend(["", "[替换绑定]", *binding_lines])

    # [属性与用法]（方案 A）：冻结原文提取的设备属性段落逐字收录，每段
    # 截断 800 字符（防极端长文挤爆 7000 硬上限）；空列表无此小节。
    if usage_paragraphs:
        lines.extend(["", "[属性与用法]"])
        for paragraph in usage_paragraphs:
            lines.append(
                paragraph.strip()[:_USAGE_PARAGRAPH_DISTILL_MAX_CHARS]
            )

    # 分镜归属：逐镜（时点）列出出镜的被替换 stable key。occurrences 的
    # frame_orders 与视觉正文"第 N 时点"同口径（段内相对编号）。
    order_to_keys: dict[int, list[str]] = {}
    for category in ("people", "entities"):
        entries = element_index.get(category)
        if not isinstance(entries, dict):
            continue
        for stable_key, entry in entries.items():
            if not isinstance(entry, dict) or not entry.get("replacement"):
                continue
            for occurrence in entry.get("occurrences") or []:
                if (
                    not isinstance(occurrence, dict)
                    or occurrence.get("segment_index") != segment_index
                ):
                    continue
                for order in occurrence.get("frame_orders") or []:
                    if isinstance(order, int) and not isinstance(order, bool):
                        order_to_keys.setdefault(order, []).append(stable_key)
    if order_to_keys:
        lines.extend(["", "[分镜归属]（时点编号与视觉正文一致）"])
        for order in sorted(order_to_keys):
            keys = "、".join(order_to_keys[order])
            lines.append(f"- 时点 {order}：{keys}")

    # [保留道具]：本段出镜的 KEEP entities（replacement 为 falsy）。
    # 根因证据：项目 7d4a3a9b 段3 的 KEEP 道具 entity-03（白色薄片状睫毛
    # 工具）在 IR 输入里没有任何结构化存在，被 IR 归并给主题商品（输出
    # "holds <Subject 3> by its thin handle portion"），H3 渲染成废片。
    keep_lines: list[str] = []
    entities = element_index.get("entities")
    if isinstance(entities, dict):
        for stable_key, entry in entities.items():
            if not isinstance(entry, dict) or entry.get("replacement"):
                continue
            orders = [
                order
                for occurrence in entry.get("occurrences") or []
                if isinstance(occurrence, dict)
                and occurrence.get("segment_index") == segment_index
                for order in occurrence.get("frame_orders") or []
                if isinstance(order, int) and not isinstance(order, bool)
            ]
            if not orders:
                continue
            # 简述与冻结提示词 KEEP 行同源（source_visual_description 首句，
            # 截 ~60 字符）；字段缺失不编造，只列 stable_key + 时点。
            summary = entry.get("source_visual_description")
            if isinstance(summary, str) and summary.strip():
                summary = (
                    summary.strip().split("：", 1)[0].split(":", 1)[0]
                    .split("。", 1)[0].strip()[:60]
                )
            else:
                summary = ""
            label = f"{stable_key}（{summary}）" if summary else stable_key
            keep_lines.append(
                f"- {label}：保留道具，时点 "
                f"{'、'.join(str(order) for order in sorted(set(orders)))}，"
                "原样保留，不得替换、改形、改色、改名或归并为任何绑定元素/Subject"
            )
    if keep_lines:
        lines.extend(["", "[保留道具]", *keep_lines])
        lines.append(
            "视觉正文中出现但未列入 [替换绑定] 的物体（含以上保留道具）"
            "一律原样保留。"
        )

    if dialogue_lines:
        language_note = f"，语种：{target_language}" if target_language else ""
        if voice_over:
            lines.extend(["", f"[台词]（幕后口播画外音{language_note}）"])
            for line in dialogue_lines:
                lines.append(
                    f"[{float(line['start_s']):.3f}-{float(line['end_s']):.3f}s] "
                    f"{line['text']}"
                )
            lines.append(
                "口播由单一音色旁白叙述，画面人物自然演示、不做与台词逐字"
                "同步的口型；不得增删台词，不得说原文台词，不生成字幕；"
                "台词逐字不得跨镜拆分。"
            )
        else:
            lines.extend(["", f"[台词]（Audio 1 逐字内容{language_note}）"])
            for line in dialogue_lines:
                lines.append(
                    f"[{float(line['start_s']):.3f}-{float(line['end_s']):.3f}s] "
                    f"{line['text']}"
                )
            lines.append(
                "对口型：所有说话人物的口型必须与 Audio 1 逐字精确同步；不得增删"
                "台词，不得说原文台词，不生成字幕；台词逐字不得跨镜拆分。"
            )
    else:
        lines.extend(["", "[台词] 本段无台词：不得添加任何语音、台词或字幕。"])

    # [画面文字]（overlay 块）：processing 开关驱动，是擦除指令不是描述。
    processing = processing if isinstance(processing, dict) else {}
    overlay_targets: list[str] = []
    if processing.get("remove_subtitle"):
        overlay_targets.append("烧录字幕与画面文字")
    if processing.get("remove_logo"):
        overlay_targets.append("水印与 logo")
    if overlay_targets:
        lines.extend([
            "",
            "[画面文字] 源片可能含" + "、".join(overlay_targets) + "：生成"
            "画面中全部擦除并自然补全背景；禁止复现、翻译或重绘任何源画面"
            "文字、水印或 logo。",
        ])

    # face 块显式优先级 + 保留规则压缩版。
    has_replaced_person = any(
        isinstance(record, dict)
        and isinstance(record.get("replacement"), dict)
        and any(
            isinstance(occurrence, dict)
            and occurrence.get("segment_index") == segment_index
            for occurrence in record.get("occurrences") or []
        )
        for record in (element_index.get("people") or {}).values()
    )
    preserve_lines: list[str] = []
    if has_replaced_person:
        preserve_lines.append(
            "人物替换是强制的且优先于源身份：脸形、眼、鼻、嘴、肤色、发型、"
            "体格与服装一律以绑定形象为唯一权威，源人物外观的任何保留都是"
            "失败，即使其他全部正确。"
        )
    preserve_lines.append(
        "机位、镜头运动、灯光、场景布局与整体风格严格保持源片；"
        "被替换元素的外观与身份以绑定素材为唯一权威，任何源身份残留（脸、"
        "发型、肤色、服装、物品外观）都是失败；遮挡块遮盖的内容一律不得重建"
        "或还原；不新增镜头、叙事或字幕。"
    )
    lines.extend(["", "[保留与禁令] " + "".join(preserve_lines)])
    text = "\n".join(lines)
    if len(text) > CONTEXT_IR_INPUT_MAX_CHARS:
        raise ContextIRInputTooLongError(
            f"Context IR 蒸馏输入 {len(text)} 字符超过官方硬上限 "
            f"{CONTEXT_IR_INPUT_MAX_CHARS}（vendor 语义 400/2013），fail-closed"
        )
    return text
# Provider-side fetch/validation failure markers classify why a terminal H3
# task failure happened; all terminal failures — explicit request-validation
# (content moderation) failures included — are resubmitted under a fresh
# run identity (a resubmit also means fresh asset URLs, sidestepping the
# 24h provider asset TTL), capped at PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS.
_PICSART_FETCH_FAILURE_MARKERS = (
    "failed to download",
    "unreachable",
    "validationerror",
)
# 每段终态失败/挂起的自动重提上限：每次重提都是一次新的付费提交，超限
# fail-closed 等人工介入（人工核实后可走 resume-picsart 续跑）。
PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS = 3
# poll 节点连续传输级失败（5xx 无终态语义 body、连接错误、超时等）的封顶：
# 达到上限按"供应商状态未知"终态化，fail-closed 报 clear error 等人工核实。
# 不自动重提的理由：无终态回执证据时，重提就是拿一笔新付费任务去赌供应商
# 侧真实状态（可能其实已成功/已扣费）；回执保持 accepted，人工核实后走
# resume-picsart 继续 poll 同一 task，不产生重复扣费。证据：2026-09-29 项目
# 0eafd5f5 三个任务 result 端点 502 file_copy_failed 被当可重试传输错误
# 无限退避（attempt-10、项目空挂 2 小时）。
PICSART_POLL_TRANSPORT_FAILURE_LIMIT = 10


class PicsartTaskHungError(RuntimeError):
    """A submitted Picsart task stayed non-terminal past the hang limit."""


class PicsartPollTransportFailureLimitError(RuntimeError):
    """poll 连续传输级失败达到 PICSART_POLL_TRANSPORT_FAILURE_LIMIT：
    供应商状态未知，fail-closed 等人工核实（不自动重提）。"""


class PicsartAutoResubmitExhaustedError(RuntimeError):
    """A segment's terminal/hung task was auto-resubmitted the maximum number
    of times and kept failing: fail-closed for manual intervention."""


# Seedream 提交终态 outcome（provider_rejected/submission_unknown/
# provider_protocol_error）的自动重置重提上限：每次重提都是一次新的付费
# 生成，超限 fail-closed（回执保留，人工走 resume 续跑时自动重置重提）。
SEEDREAM_REGENERATION_MAX_ATTEMPTS = 3


class SeedreamRegenerationExhaustedError(RuntimeError):
    """The frozen Seedream sheet request kept ending in a regeneratable
    terminal outcome past the re-POST cap: fail-closed for a manual resume."""


# MediaKit 字幕擦除节点整体重试上限：submit accepted 即已付费（回执落盘
# work/erase-source-subtitles-receipt.json），任何失败的重跑先读回执复用
# 已受理任务，只有回执任务终态失败才重新 submit；3 次仍失败 fail-closed。
MEDIAKIT_ERASE_NODE_MAX_ATTEMPTS = 3


class MediaKitEraseRetryExhaustedError(RuntimeError):
    """The subtitle-erase node failed its whole retry budget: fail-closed for
    manual intervention."""


# 节点级重跑预算（模型输出已付费但不可用类）：ValueError/MediaError/
# AssertionError 的每次重跑都意味着重新付费生成，无限自愈会变成无限
# 扣费；3 次仍不可用即 fail-closed 等人工介入。传输类错误（请求未到达
# 不扣费）不在此预算内，保持无限退避。
NODE_MODEL_OUTPUT_RETRY_MAX_ATTEMPTS = 3


class NodeModelOutputRetryExhaustedError(RuntimeError):
    """ValueError/MediaError/AssertionError 节点重跑达到预算上限：模型输出
    已连续不可用，fail-closed 等待人工介入。"""


@dataclass(frozen=True, slots=True)
class PreparedPicsartRequest:
    """Immutable boundary between local preparation and paid provider work."""

    project_id: str
    segment_index: int
    scope: str
    run_identity: str
    artifact_suffix: str
    final_prompt: str
    muted: Path
    provider_images: tuple[Path, ...]
    dialogue_track: Path | None
    video_url: str | None
    image_urls: tuple[str, ...]
    audio_urls: tuple[str, ...]
    provider_duration: int
    resolution: str
    aspect_ratio: str
    target_duration_s: float
    request_sha256: str
    preparation_sha256: str | None
    existing_task_id: str | None = None

    def submit_input(self) -> dict[str, Any]:
        if self.existing_task_id is not None:
            raise ValueError("an existing Picsart task has no submit input")
        if self.video_url is None:
            raise ValueError("prepared Picsart video URL is missing")
        return {
            "workflow": PicsartClient.WORKFLOW,
            "prompt": self.final_prompt,
            "reference_video_urls": [self.video_url],
            "reference_image_urls": list(self.image_urls),
            "reference_audio_urls": list(self.audio_urls),
            "resolution": self.resolution.upper(),
            "duration": self.provider_duration,
            "aspect_ratio": self.aspect_ratio,
            "prompt_expansion_mode": "quality",
        }


# 账本块同类 REPLACE 条目的固定尾部措辞：多人/多器物项目里逐条重复且逐字
# 相同，2026-09-29 起上提为块级全局行（每类一条，wrapper 行后注入），
# 条目只留 stable key/REPLACE/target/reference。措辞逐字保留原句强度；
# 改动任一常量即改变冻结 prompt 语义，须重做离线对账验证
# （data-verify/prompt-slim-20260929）。
_FUSION_PERSON_REPLACE_PRESERVE = (
    "source role, action, pose, position, and continuity; "
    "only explicitly stated gender, race, overall style, and "
    "primary clothing color"
)
_FUSION_PERSON_REPLACE_POLICY = (
    "replace with a clearly different identity; the new person must differ "
    "visibly from the source in at least two of: hair length, hair texture, "
    "hair color, face shape, facial hair or other prominent facial "
    "features; keep only explicitly stated gender and age group; when "
    "clothing is visible, change its design; do not infer unseen traits"
)
_FUSION_ENTITY_REPLACE_PRESERVE = (
    "source occurrence timing, position/orientation, action rhythm, "
    "camera language, and interaction events only; target shape, volume, "
    "dimensions, proportions, and function come solely from the bound "
    "reference; rebuild grip, occlusion, and contact placement naturally "
    "for the target geometry; never deform the target to fit source "
    "geometry"
)
_FUSION_ENTITY_REPLACE_POLICY = (
    "preserve only source occurrence timing, position/orientation, action "
    "rhythm, camera language, and interaction events; the bound reference "
    "is the sole authority for target shape, volume, dimensions, "
    "proportions, and function; rebuild grip, occlusion, and contact "
    "placement naturally for the target geometry and never deform the "
    "target to fit source geometry"
)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        store: ProjectStore,
        register_asset: Callable[[str, Path], str],
        *,
        codex: CodexClient | None = None,
        deepseek: DeepSeekClient | None = None,
        mediakit: MediaKitClient | None = None,
        seedream: SeedreamClient | None = None,
        picsart: PicsartClient | None = None,
        tts: DoubaoTtsClient | None = None,
        ark_audio: ArkAudioClient | None = None,
        context_ir: MiniMaxContextIRClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.register_asset = register_asset
        self.codex = codex or CodexClient(settings)
        self.deepseek = deepseek or DeepSeekClient(settings)
        self.mediakit = mediakit or MediaKitClient(settings)
        self.seedream = seedream or SeedreamClient(settings)
        if picsart is not None:
            self.picsart = picsart
        elif settings.h3_provider == "selfhosted":
            from app.h3_selfhosted import SelfhostedH3Client

            self.picsart = SelfhostedH3Client(settings)
        else:
            self.picsart = PicsartClient(settings)
        self.tts = tts or DoubaoTtsClient(settings)
        # ark_audio 不预构造：配置缺失要在节点首次用到时才报错（未启用
        # ark-audio-analysis 的路径不付出代价），节点内惰性
        # ArkAudioClient(self.settings)。
        self.ark_audio = ark_audio
        # context_ir 不预构造：配置缺失要在节点首次用到时才报错（未启用
        # h3-context-ir 的路径不付出代价），节点内惰性
        # MiniMaxContextIRClient(self.settings)。
        self.context_ir = context_ir
        self.voice_catalog: DoubaoVoiceCatalog = load_doubao_voice_catalog()
        # ProjectStore.update() is atomic, but a receipt ledger change is a
        # read-modify-write operation assembled here.  Serialize that whole
        # operation so parallel segment transitions cannot overwrite each
        # other before the paid POST boundary.
        self._picsart_receipt_lock = threading.Lock()

    def _progress(
        self,
        project_id: str,
        percent: int,
        stage: str,
        *,
        segments: list[dict[str, Any]] | None = None,
    ) -> None:
        changes: dict[str, Any] = {
            "status": "running",
            "project_progress": {
                "status": "running",
                "percent": percent,
                "stage": stage,
            },
            "generation": {"status": "running", "stage": stage, "error": None},
        }
        if segments is not None:
            changes["segments"] = segments
        self.store.update(project_id, **changes)

    def _node(
        self,
        log: ResearchLog,
        node: str,
        scope: str,
        inputs: Any,
        input_media: list[Path],
        operation: Callable[[], Any],
        output_media: Callable[[Any], list[Path]] | None = None,
        *,
        transform: Callable[[Any], Any] | None = None,
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> Any:
        """Run one node, retrying retryable failures until they succeed.

        Attempts 2 and 3 rerun immediately; from attempt 4 on the wait backs
        off exponentially from one minute (capped at thirty minutes). Nodes
        with an explicit ``retry_if`` keep their own gate (paid submissions
        with receipt semantics). Transport evidence is kept per attempt.
        Resumed nodes continue from the highest existing attempt number so
        the failed attempts' evidence is never overwritten.  Failures that
        burn paid model output (ValueError/MediaError/AssertionError) stop
        after ``NODE_MODEL_OUTPUT_RETRY_MAX_ATTEMPTS`` attempts and raise
        ``NodeModelOutputRetryExhaustedError``; transport-class failures
        (the request never arrived, so nothing was billed) stay uncapped.
        """
        last_error: Exception | None = None
        attempt = log.next_attempt(node, scope) - 1
        moderation_fingerprint: str | None = None
        moderation_streak = 0
        paid_output_attempts = 0
        while True:
            attempt += 1
            directory = log.attempt_dir(node, scope, attempt)
            log.write_json(directory / "input.json", inputs)
            log.copy_media(directory / "input-media", input_media)
            log.event("node_started", node=node, scope=scope, attempt=attempt)
            provider_record: dict[str, Any] | None = None
            provider_media: list[Path] = []
            try:
                raw = operation()
                if isinstance(raw, ProviderResult):
                    result = raw.value
                    provider_record = raw.record
                    provider_media = list(raw.evidence_media)
                else:
                    result = raw
                if transform is not None:
                    result = transform(result)
                logged_output: Any = result
                if provider_record is not None:
                    logged_output = {
                        "value": result,
                        "provider_record": provider_record,
                    }
                log.write_json(directory / "output.json", logged_output)
                media_paths = list(provider_media)
                if output_media is not None:
                    try:
                        media_paths.extend(output_media(result))
                    except Exception as exc:
                        log.event(
                            "research_log_media_selection_failed",
                            node=node,
                            scope=scope,
                            attempt=attempt,
                            error=str(exc),
                        )
                log.copy_media(directory / "output-media", media_paths)
                event = (
                    "node_skipped"
                    if provider_record is not None
                    and provider_record.get("outcome") == "skipped"
                    else "node_succeeded"
                )
                log.event(event, node=node, scope=scope, attempt=attempt)
                return result
            except ProviderFailure as exc:
                last_error = exc
                log.write_json(
                    directory / "output.json",
                    {"provider_record": exc.record},
                )
                log.copy_media(directory / "output-media", exc.evidence_media)
                log.error(directory / "error.json", exc)
                self._write_process_evidence(log, directory, exc)
                log.event(
                    "node_failed",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    error=str(exc),
                )
            except Exception as exc:
                last_error = exc
                if provider_record is not None:
                    log.write_json(
                        directory / "output.json",
                        {"provider_record": provider_record},
                    )
                    log.copy_media(directory / "output-media", provider_media)
                log.error(directory / "error.json", exc)
                self._write_process_evidence(log, directory, exc)
                log.event(
                    "node_failed",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    error=str(exc),
                )
            should_retry = (
                retry_if(last_error)
                if retry_if is not None
                else self._retryable_node_error(last_error)
            )
            if not should_retry:
                break
            if self._burns_paid_model_output(last_error):
                paid_output_attempts += 1
                if paid_output_attempts >= NODE_MODEL_OUTPUT_RETRY_MAX_ATTEMPTS:
                    log.event(
                        "node_model_output_retry_exhausted",
                        node=node,
                        scope=scope,
                        attempt=attempt,
                        attempts=paid_output_attempts,
                        error=str(last_error),
                    )
                    raise NodeModelOutputRetryExhaustedError(
                        f"node {node} (scope {scope}) 模型输出类错误重跑 "
                        f"{NODE_MODEL_OUTPUT_RETRY_MAX_ATTEMPTS} 次仍失败，"
                        f"fail-closed 需人工介入；末次错误: {last_error}"
                    ) from last_error
            if isinstance(last_error, ProviderFailure):
                moderation_text = (
                    json.dumps(
                        last_error.record, ensure_ascii=False, default=str,
                    )
                    + str(last_error)
                ).lower()
                if any(
                    marker in moderation_text
                    for marker in self._NODE_FATAL_MARKERS
                ):
                    fingerprint = hashlib.sha256(
                        moderation_text.encode("utf-8")
                    ).hexdigest()[:16]
                    moderation_streak = (
                        moderation_streak + 1
                        if fingerprint == moderation_fingerprint
                        else 1
                    )
                    moderation_fingerprint = fingerprint
                    if moderation_streak >= 5:
                        log.event(
                            "node_moderation_persistent",
                            node=node,
                            scope=scope,
                            attempt=attempt,
                            fingerprint=fingerprint,
                            consecutive=moderation_streak,
                        )
            delay = self._node_retry_delay(attempt)
            if delay:
                log.event(
                    "node_retry_wait",
                    node=node,
                    scope=scope,
                    attempt=attempt,
                    delay_s=delay,
                )
                self._retry_sleep(delay)
        assert last_error is not None
        raise last_error

    _NODE_FATAL_MARKERS = (
        "image_rejected",
        "content policy",
        "moderation",
        "审核",
    )

    @staticmethod
    def _is_enospc(exc: Exception) -> bool:
        return isinstance(exc, OSError) and exc.errno == errno.ENOSPC

    @classmethod
    def _has_deterministic_client_error(cls, value: Any) -> bool:
        """True when any recorded exchange ended in a deterministic 4xx.

        Deterministic means every client error except 408/429: re-issuing the
        identical request cannot succeed, so retrying it forever only loops.
        """
        if isinstance(value, dict):
            response = value.get("response")
            if isinstance(response, dict):
                status = response.get("status_code")
                if (
                    isinstance(status, int)
                    and not isinstance(status, bool)
                    and 400 <= status < 500
                    and status not in {408, 429}
                ):
                    return True
            return any(
                cls._has_deterministic_client_error(item)
                for item in value.values()
            )
        if isinstance(value, list):
            return any(cls._has_deterministic_client_error(item) for item in value)
        return False

    @classmethod
    def _retryable_node_error(cls, exc: Exception) -> bool:
        """Provider, network, subprocess and transient model-output failures
        retry; input and business validation errors don't.  Moderation
        rejections also retry with backoff instead of failing the project."""
        if isinstance(exc, ProviderFailure):
            # A deterministic failure can never succeed on retry: fail fast
            # instead of looping until the disk or quota dies.  Two kinds:
            # a recorded deterministic 4xx (e.g. a provider rejecting a
            # corrupt upload with HTTP 400), and a cause explicitly typed as
            # deterministic (object_detect.UncalibratedResponseError: the
            # provider returned an uncalibrated response shape, so re-issuing
            # the identical request returns the same shape).
            if isinstance(exc.cause, object_detect.UncalibratedResponseError):
                return False
            # 豆包 TTS 流正常结束但零音频：同一文本+音色重发必然同样失败。
            if isinstance(exc.cause, EmptyAudioError):
                return False
            # 豆包 TTS 流已收 finish、本地落盘失败（如 ENOSPC）：音频已入
            # 服务端 section 历史，同 section 重发会重复扣费并污染历史，
            # 判死让项目 failed（续跑走新 run_id 派生的新 section，天然
            # 安全）。流中途失败无此标记，保持可重试不变。
            if (
                isinstance(exc.record, dict)
                and exc.record.get("outcome") == STREAM_FINISHED_OUTCOME
            ):
                return False
            return not cls._has_deterministic_client_error(exc.record)
        # A full disk (ENOSPC) is transient: the operator frees space and the
        # node must rerun, never kill the project.  Research-log writes are
        # best-effort, so the retry path itself survives the full disk.
        if cls._is_enospc(exc):
            return True
        if isinstance(exc, (
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
            media.CommandError,
            CodexProcessError,
        )):
            return True
        # Any ValueError/MediaError from node execution (model-output or
        # media-consumption validation) is treated as transient: retrying
        # lets nondeterministic generation self-heal.  AssertionError covers
        # defensive leftovers.  TypeError/KeyError/AttributeError stay fatal
        # (programmer-bug semantics).
        if isinstance(exc, (ValueError, media.MediaError, AssertionError)):
            # ffprobe 对本地源文件报"无法识别容器"是永久性结论：字节根本不
            # 是视频（如下载节点存下的 HTML 登录墙），重试一万次也一样，
            # 直接判死让项目带清晰 error 进入 failed，不进退避重试。
            if isinstance(exc, media.UnrecognizedMediaError):
                return False
            return True
        return False

    @classmethod
    def _burns_paid_model_output(cls, exc: Exception) -> bool:
        """True when rerunning the node pays for model output again.

        ValueError/MediaError/AssertionError from node execution mean the
        model output was already paid for but unusable, so each retry buys
        it again — these are budgeted (NODE_MODEL_OUTPUT_RETRY_MAX_ATTEMPTS).
        Transport-class failures never reached the paid call: JSONDecodeError
        is a ValueError subclass but means the response bytes never arrived,
        so it stays uncapped.  UnrecognizedMediaError is a MediaError the
        fatal gate already rejects; exclude it defensively.
        """
        if isinstance(exc, (json.JSONDecodeError, media.UnrecognizedMediaError)):
            return False
        return isinstance(exc, (ValueError, media.MediaError, AssertionError))

    @staticmethod
    def _node_retry_delay(failed_attempt: int) -> float:
        """Attempts 2-3 rerun immediately; from attempt 4 on, back off
        exponentially from one minute, capped at thirty minutes."""
        if failed_attempt < 3:
            return 0.0
        return min(60.0 * 2 ** (failed_attempt - 3), 1800.0)

    def _retry_sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    def _write_process_evidence(
        log: ResearchLog, directory: Path, exc: Exception,
    ) -> None:
        cause = exc.cause if isinstance(exc, ProviderFailure) else exc
        stdout = getattr(cause, "stdout", None)
        stderr = getattr(cause, "stderr", None)
        if isinstance(stdout, str):
            log.write_text(directory / "stdout.log", stdout)
        if isinstance(stderr, str):
            log.write_text(directory / "stderr.log", stderr)

    @staticmethod
    def _stored_replacement_paths(project: Path, state: dict[str, Any]) -> list[Path]:
        single = state.get("replacement_image_path")
        paths = state.get("replacement_image_paths")
        if paths is None:
            paths = [single] if single is not None else []
        expected = replacement_input_paths(len(paths)) if isinstance(paths, list) and paths else []
        if not isinstance(paths, list) or len(paths) > 4 or paths != expected or single != (paths[0] if paths else None):
            raise ValueError("stored replacement images must preserve upload order")
        root = project.resolve(strict=True)
        result = []
        for name in paths:
            path = project / name
            resolved = path.resolve(strict=True)
            if path.is_symlink() or root not in resolved.parents or not resolved.is_file():
                raise ValueError("stored replacement image is outside the project")
            result.append(path)
        frozen = product_views.frozen_product_views(project)
        if frozen is not None and [item["path"] for item in frozen["sources"]] != paths:
            raise ValueError("frozen product view upload order changed")
        return result

    @staticmethod
    def _stored_input_paths(
        project: Path,
        state: dict[str, Any],
    ) -> tuple[Path, Path | None]:
        """Resolve only the two canonical project-owned input locations."""
        if state.get("source_path") != "inputs/source.mp4":
            raise ValueError("stored source path must be inputs/source.mp4")
        replacement_path = state.get("replacement_image_path")
        if replacement_path not in {None, "inputs/replacement-image"}:
            raise ValueError(
                "stored replacement path must be inputs/replacement-image"
            )
        project_root = project.resolve(strict=True)
        source_path = project / "inputs" / "source.mp4"
        source = source_path.resolve(strict=True)
        if (
            project_root not in source.parents
            or source_path.is_symlink()
            or not source.is_file()
        ):
            raise ValueError("stored source is outside the project")
        replacement: Path | None = None
        if replacement_path is not None:
            candidate = project / "inputs" / "replacement-image"
            replacement = candidate.resolve(strict=True)
            if (
                project_root not in replacement.parents
                or candidate.is_symlink()
                or not replacement.is_file()
            ):
                raise ValueError("stored replacement image is outside the project")
        Pipeline._stored_replacement_paths(project, state)
        return source, replacement

    @staticmethod
    def _mediakit_task_is_terminal_failure(exc: BaseException) -> bool:
        """True when the provider marked the accepted task itself FAILED (as
        opposed to a transport failure while querying it)."""
        cause = exc.cause if isinstance(exc, ProviderFailure) else exc
        return isinstance(cause, RuntimeError) and str(cause).startswith(
            "MediaKit task failed"
        )

    @staticmethod
    def _read_mediakit_erase_receipt(receipt_path: Path) -> dict[str, Any] | None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(receipt, dict)
            or receipt.get("version") != 1
            or receipt.get("kind") != "mediakit_subtitle_erase"
            or not isinstance(receipt.get("task_id"), str)
            or not receipt["task_id"].strip()
            or not isinstance(receipt.get("video_url"), str)
            or receipt.get("status") not in {"accepted", "provider_failed", "downloaded"}
        ):
            return None
        return receipt

    def _erase_source_subtitles(
        self,
        project_id: str,
        project: Path,
        source: Path,
        request: dict[str, Any],
        log: ResearchLog,
        *,
        resume: bool = False,
    ) -> Path:
        """Erase burned-in subtitles from the source before any downstream use.

        The MediaKit submit is the paid boundary: once accepted, its receipt
        (task id, video URL, submit record) is atomically persisted to
        ``work/erase-source-subtitles-receipt.json``.  A node rerun reads the
        receipt first and reuses the accepted task (poll/download only — the
        paid result is never thrown away); only a receipt whose task ended in
        a terminal provider failure allows a fresh paid submit.  A downloaded
        receipt with the output still on disk skips everything.  The whole
        node is capped at ``MEDIAKIT_ERASE_NODE_MAX_ATTEMPTS`` attempts and
        then fails closed."""
        processing = request.get("processing") if isinstance(request, dict) else None
        if not (
            isinstance(processing, dict)
            and processing.get("remove_subtitle") is True
        ):
            return source
        erased = project / "work" / "source-nosubs.mp4"
        if resume and erased.is_file():
            reused = self._reusable_node_output(
                project, "erase-source-subtitles", "project", [erased],
            )
            if reused is not _NO_REUSABLE_OUTPUT:
                log.event(
                    "node_output_reused",
                    node="erase-source-subtitles",
                    scope="project",
                )
                return erased

        receipt_path = project / "work" / "erase-source-subtitles-receipt.json"
        attempts = 0

        def operation() -> ProviderResult[Path]:
            nonlocal attempts
            attempts += 1
            if attempts > MEDIAKIT_ERASE_NODE_MAX_ATTEMPTS:
                log.event(
                    "mediakit_erase_retry_exhausted",
                    node="erase-source-subtitles",
                    attempts=MEDIAKIT_ERASE_NODE_MAX_ATTEMPTS,
                )
                raise MediaKitEraseRetryExhaustedError(
                    f"erase-source-subtitles 节点重试 "
                    f"{MEDIAKIT_ERASE_NODE_MAX_ATTEMPTS} 次仍失败，"
                    f"fail-closed 需人工介入"
                )
            receipt = self._read_mediakit_erase_receipt(receipt_path)
            if (
                receipt is not None
                and receipt["status"] == "downloaded"
                and erased.is_file()
            ):
                return ProviderResult(
                    erased,
                    {
                        "submit": receipt["submit"],
                        "video_url": receipt["video_url"],
                        "receipt_reused": "downloaded",
                    },
                    [erased],
                )
            if receipt is not None and receipt["status"] == "accepted":
                task_id = receipt["task_id"]
                video_url = receipt["video_url"]
                log.event(
                    "mediakit_subtitle_erase_receipt_reused",
                    task_id=task_id,
                )
                record: dict[str, Any] = {
                    "submit": receipt["submit"],
                    "video_url": video_url,
                    "receipt_reused": "accepted",
                }
            else:
                video_url = self.register_asset(project_id, source)
                submitted = self.mediakit.submit_video_subtitle_erase(video_url)
                task_id = submitted.value
                receipt = {
                    "version": 1,
                    "kind": "mediakit_subtitle_erase",
                    "status": "accepted",
                    "task_id": task_id,
                    "video_url": video_url,
                    "submit": submitted.record,
                }
                _persist_receipt_json(receipt_path, receipt)
                record = {"submit": submitted.record, "video_url": video_url}
            try:
                polled = self.mediakit.poll_task(task_id)
            except Exception as exc:
                if self._mediakit_task_is_terminal_failure(exc):
                    stored = self._read_mediakit_erase_receipt(receipt_path)
                    if (
                        stored is not None
                        and stored["task_id"] == task_id
                        and stored["status"] == "accepted"
                    ):
                        _persist_receipt_json(
                            receipt_path, {**stored, "status": "provider_failed"},
                        )
                        log.event(
                            "mediakit_subtitle_erase_receipt_reset",
                            task_id=task_id,
                            error=str(exc),
                        )
                raise
            record["poll"] = polled.record
            downloaded = self.mediakit.download(polled.value, erased)
            record["download"] = downloaded.record
            _persist_receipt_json(
                receipt_path,
                {
                    "version": 1,
                    "kind": "mediakit_subtitle_erase",
                    "status": "downloaded",
                    "task_id": task_id,
                    "video_url": video_url,
                    "submit": record["submit"],
                },
            )
            return ProviderResult(
                erased, record, list(downloaded.evidence_media),
            )

        self._progress(project_id, 3, "erase-source-subtitles")
        self._node(
            log,
            "erase-source-subtitles",
            "project",
            {
                "source": source,
                "erase_mode": self.settings.mediakit_video_erase_mode,
                "output": erased,
            },
            [source],
            operation,
            lambda value: [value],
        )
        return erased

    def run(self, project_id: str) -> None:
        project = self.store.path(project_id)
        log = ResearchLog(project)
        state = self.store.read(project_id)
        source, replacement = self._stored_input_paths(project, state)
        request = self._validated_effective_request(
            state.get("effective_request")
        )
        try:
            source, segments, selections, spoken_lines = (
                self._run_pre_segmentation(
                    project_id, project, source, request, log,
                )
            )
            self._progress(project_id, 24, "video-maker", segments=segments)
            prepared = self._prepare_project_segments(
                project_id, project, source, segments, selections, request, log,
            )
            self._continue_after_prepared(
                project_id,
                project,
                prepared,
                spoken_lines,
                replacement,
                request,
                log,
            )
        except Exception as exc:
            log.event("pipeline_failed", error=str(exc))
            code, message = self._failure_code(exc)
            self._persist_failure(
                project_id,
                log,
                stage="pipeline_failed",
                code=code,
                message=message,
                min_percent=1,
            )

    def _persist_failure(
        self,
        project_id: str,
        log: ResearchLog,
        *,
        stage: str,
        code: str,
        message: str,
        min_percent: int,
        extra_changes: dict[str, Any] | None = None,
    ) -> None:
        """Persist the failed terminal state, retrying while the disk is full.

        ENOSPC also breaks the state.json write itself; without a retry the
        worker dies leaving status "running" (fake-alive).  Keep retrying on
        the same backoff schedule as node retries so the terminal state lands
        once the operator frees space.  Research-log writes are best-effort
        and never raise, so this loop cannot be killed by its own logging."""
        current = self.store.read(project_id)
        percent = max(
            min_percent,
            int((current.get("project_progress") or {}).get("percent") or min_percent),
        )
        changes: dict[str, Any] = {
            "status": "failed",
            "project_progress": {
                "status": "failed",
                "percent": min(percent, 99),
                "stage": stage,
            },
            "generation": {
                "status": "failed",
                "stage": stage,
                "error": code,
            },
            "error": {
                "code": code,
                "message": message,
            },
        }
        if extra_changes:
            changes.update(extra_changes)
        attempt = 0
        while True:
            try:
                self.store.update(project_id, **changes)
                return
            except OSError as exc:
                if not self._is_enospc(exc):
                    raise
                attempt += 1
                log.event(
                    "failure_persist_enospc_retry",
                    stage=stage,
                    attempt=attempt,
                    error=str(exc),
                )
                self._retry_sleep(self._node_retry_delay(attempt))

    def _reusable_node_output(
        self,
        project: Path,
        node: str,
        scope: str,
        artifacts: Iterable[Path] = (),
    ) -> Any:
        """Last completed node output, only while its on-disk artifacts live.

        A 0-byte artifact (e.g. an ENOSPC-truncated write) counts as missing
        so the node regenerates instead of passing empty bytes downstream."""
        for artifact in artifacts:
            if not artifact.exists():
                return _NO_REUSABLE_OUTPUT
            try:
                if artifact.is_file() and artifact.stat().st_size == 0:
                    return _NO_REUSABLE_OUTPUT
            except OSError:
                return _NO_REUSABLE_OUTPUT
        try:
            return self._completed_node_value(project, node, scope)
        except (FileNotFoundError, OSError, ValueError):
            return _NO_REUSABLE_OUTPUT

    def _run_pre_segmentation(
        self,
        project_id: str,
        project: Path,
        source: Path,
        request: dict[str, Any],
        log: ResearchLog,
        *,
        resume: bool = False,
    ) -> tuple[
        Path,
        list[dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        list[dict[str, Any]],
    ]:
        """Run probe through keyframe selection.

        With ``resume`` set, a node whose completed output and artifacts are
        already on disk (source-nosubs.mp4, voice_lines.json, ...) is reused
        instead of re-executed, so a project that died before segment
        preparation continues where it stopped."""

        def run_node(
            node: str,
            scope: str,
            inputs: Any,
            input_media: list[Path],
            operation: Callable[[], Any],
            output_media: Callable[[Any], list[Path]] | None = None,
            *,
            artifacts: Iterable[Path] = (),
        ) -> Any:
            if resume:
                reused = self._reusable_node_output(project, node, scope, artifacts)
                if reused is not _NO_REUSABLE_OUTPUT:
                    log.event("node_output_reused", node=node, scope=scope)
                    return reused
            return self._node(
                log, node, scope, inputs, input_media, operation, output_media,
            )

        self._progress(project_id, 2, "probe")
        info = run_node(
            "probe",
            "project",
            {"source": source},
            [source],
            lambda: media.probe(source),
            artifacts=[source],
        )
        duration_s = float(info["duration_s"])
        self.store.update(project_id, duration_s=duration_s, has_source=True)

        source = self._erase_source_subtitles(
            project_id, project, source, request, log, resume=resume,
        )

        self._progress(project_id, 5, "source-4fps")
        source_frames = project / "work" / "source-4fps"
        source_manifest = run_node(
            "source-frames-4fps",
            "project",
            {"source": source, "fps": 4, "output": source_frames},
            [source],
            lambda: media.extract_frames_4fps(source, source_frames),
            lambda value: self._manifest_media(source_frames, value),
            artifacts=[source_frames],
        )
        duration_s = float(source_manifest.get("duration_seconds") or duration_s)
        self.store.update(project_id, duration_s=duration_s)

        yamnet, asr_lines, spoken_lines, voice_decisions, source_language_code = (
            self._analyze_audio_dialogue(
                project_id,
                project,
                source,
                has_audio=bool(info.get("has_audio")),
                log=log,
                resume=resume,
            )
        )
        self.store.update(
            project_id,
            voice_lines=spoken_lines,
            asr_voice_lines=asr_lines,
            voice_decisions=voice_decisions,
            vocal_analysis=yamnet,
            source_language_code=source_language_code,
        )

        self._progress(project_id, 18, "scene-detection")
        scenes = run_node(
            "scene-detection",
            "project",
            {"source": source, "threshold": 27.0},
            [source],
            lambda: media.detect_scenes(source, duration_s),
            artifacts=[source],
        )
        segments = run_node(
            "segment-planning",
            "project",
            {
                "scenes": scenes,
                "duration_s": duration_s,
                "dialogue": spoken_lines,
                "provider_window_s": [4, 10],
            },
            [],
            lambda: media.plan_segments(scenes, duration_s, spoken_lines),
        )
        self._validated_join_modes(segments)
        selections: dict[int, list[dict[str, Any]]] = {}
        for segment in segments:
            index = int(segment["index"])
            scope = self._segment_scope(index)
            selections[index] = run_node(
                "keyframe-selection-exact-nine",
                scope,
                {"segment": segment, "scenes": scenes, "count": 9},
                [],
                lambda segment=segment: media.select_segment_keyframes(
                    scenes, segment,
                ),
            )
        self.store.update(
            project_id,
            segment_count=len(segments),
            scenes=scenes,
            segments=segments,
        )
        return source, segments, selections, spoken_lines

    @staticmethod
    def _failure_code(exc: Exception) -> tuple[str, str]:
        """Deterministic input/business failures surface an actionable code
        instead of the generic pipeline_failed."""
        if isinstance(exc, media.MediaError):
            return str(exc), "输入视频不满足处理要求"
        if isinstance(exc, NodeModelOutputRetryExhaustedError):
            return str(exc), "模型输出连续不可用，需人工介入"
        if isinstance(exc, PicsartAutoResubmitExhaustedError):
            return str(exc), "H3 自动重提多次仍失败，需人工介入"
        if isinstance(exc, PicsartPollTransportFailureLimitError):
            return str(exc), "H3 任务状态未知（poll 连续传输失败），需人工介入"
        if isinstance(exc, SeedreamRegenerationExhaustedError):
            return str(exc), "图版生成自动重提多次仍失败，需人工介入"
        if isinstance(exc, MediaKitEraseRetryExhaustedError):
            return str(exc), "字幕擦除多次失败，需人工介入"
        cause = exc.cause if isinstance(exc, ProviderFailure) else exc
        if getattr(cause, "code", None) == "provider_rejected":
            return "provider_rejected", "参考图未通过供应商审核"
        if (
            isinstance(exc, ProviderFailure)
            and exc.record.get("outcome") == "rejected"
        ):
            return "picsart_submit_rejected", "视频提交未通过供应商审核"
        return "pipeline_failed", "流水线执行失败"

    @staticmethod
    def _validated_effective_request(value: object) -> dict[str, Any]:
        # 中间层不做形态校验：是我们自己写下的状态，读到啥用啥。
        if not isinstance(value, dict):
            raise ValueError("stored effective request is invalid")
        return value

    @staticmethod
    def _validated_join_modes(items: list[dict[str, Any]]) -> list[str]:
        if not items:
            raise ValueError("segment list is empty")
        modes: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("segment metadata is invalid")
            mode = item.get("join_mode")
            if mode not in {"hard_cut", "continue"}:
                raise ValueError("segment join mode is invalid")
            modes.append(mode)
        return modes

    def _fusion_client_for_request(
        self, request: dict[str, Any],
    ) -> tuple[str, CodexClient | DeepSeekClient]:
        request = self._validated_effective_request(request)
        configured_fusion = request["fusion"]
        if not isinstance(configured_fusion, dict):
            raise ValueError("frozen fusion configuration is invalid")
        agent = configured_fusion.get("agent")
        if agent == "codex":
            return agent, self.codex
        if agent == "deepseek":
            return agent, self.deepseek
        raise ValueError("frozen fusion agent is invalid")

    @staticmethod
    def _stored_voice_references(
        project: Path, state: dict[str, Any],
    ) -> list[Path]:
        """Validate the retired voice-reference slot.

        豆包 TTS 直接按 voice_type 合成，不再物化本地 wav 参考；v5 起该字段
        恒为空列表，仅保留字段名以维持 h3_prepared_artifacts 绑定结构
        （voice_references_sha256）。旧项目的非空引用由 Object Replace 版本
        门禁先行拒绝，走到这里仍非空即为损坏状态。"""
        references = state.get("voice_references")
        if not isinstance(references, list) or references:
            raise ValueError("stored voice references are invalid")
        return []

    @staticmethod
    def _replacement_image_paths(project, state=None, replacement=None):
        if state is None:
            state_path = project / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        values = state.get("replacement_image_paths")
        if not values:
            single = state.get("replacement_image_path")
            return [project / single] if single else ([replacement] if replacement else [])
        expected = replacement_input_paths(len(values))
        if not isinstance(values, list) or not 1 <= len(values) <= 4 or values != expected:
            raise ValueError("stored replacement image paths are invalid")
        paths = [project / value for value in values]
        for path in paths:
            if path.is_symlink() or project.resolve() not in path.resolve().parents:
                raise ValueError("stored replacement image is outside the project")
        return paths

    @staticmethod
    def _canonical_user_reference_path(project, order):
        name = "user-reference.png" if order == 1 else f"user-reference-{order}.png"
        return project / "work" / "global-reference-sheet" / name

    @staticmethod
    def _user_reference_metadata(paths, *, image_offset=0):
        return [{
            "reference_id": f"image_{order}",
            "image_reference": f"Image {order + image_offset}",
            "role": "user_target_reference_not_source_video",
            "sha256": Pipeline._sha256(path),
        } for order, path in enumerate(paths, 1)]

    def _materialize_canonical_user_reference(
        self,
        project: Path,
        replacement: Path,
        log: ResearchLog,
        order: int = 1,
        *,
        replacement_paths: list[Path] | None = None,
    ) -> Path:
        # 分叉点（参考图构造与冻结）：composed 模式多图合成为单一 provider
        # 参考图；direct 模式与两模式各自的 1 图情形走逐图规范化（order 定位）。
        if (
            replacement_paths is not None
            and len(replacement_paths) > 1
            and self.settings.multiref_mode == "composed"
        ):
            return self._node(
                log, "replacement-sheet-user-reference", "project",
                {"sources": replacement_paths, "interpretation": "one_product_multiple_views"},
                replacement_paths,
                lambda: product_views.compose_product_views(project, replacement_paths),
                output_media=lambda value: [value],
            )
        destination = self._canonical_user_reference_path(project, order)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        self._node(
            log,
            "replacement-sheet-user-reference",
            "project" if order == 1 else f"user-image-{order}",
            {"source": replacement, "output": destination},
            [replacement],
            lambda: media.user_reference_png(replacement, destination),
            output_media=lambda _value: [destination],
        )
        payload = destination.read_bytes()
        if (
            not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or len(payload) < 9
        ):
            raise ValueError("prepared provider reference is not PNG")
        self._project_artifact_receipt(
            project,
            destination,
            destination.relative_to(project).as_posix(),
        )
        return destination

    def _run_object_replace(
        self,
        project: Path,
        prepared: list[dict[str, Any]],
        translated_lines: list[dict[str, Any]],
        element_index: dict[str, Any],
        user_reference: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = self._validated_effective_request(request)
        guidance = request["replacement_guidance"]
        user_instruction = (
            guidance["instruction"] if guidance is not None else ""
        )
        target_language = request["dialogue"]["target_language"]
        reference_paths = list(prepared[0].get("user_reference_paths", [])) if prepared else []
        if not reference_paths and user_reference is not None:
            if self.settings.multiref_mode == "composed":
                # composed 模式：多图已合成为单一 provider 参考图，OR 只见合图。
                reference_paths = [user_reference]
            else:
                originals = self._replacement_image_paths(project)
                reference_paths = [user_reference, *[
                    self._materialize_canonical_user_reference(project, path, log, order)
                    for order, path in enumerate(originals[1:], 2)
                ]]
        reference_ids = [f"image_{order}" for order in range(1, len(reference_paths) + 1)]
        # 音色目录先按目标语言过滤、再限定已标定音色（LLM 看到即可绑定，
        # 未标定音色没有语速基准、不可输出），按场景相关度精选（上限
        # PROMPT_VOICE_LIMIT 条），避免全量目录淹没提示词；“与原视频相同”
        # 按台词文字系统解析，解析不出回退中文子集。提示词行与输出枚举
        # 使用同一精选子集。
        language_label = _dialogue_language_display(
            target_language,
            translated_lines,
            self.store.read(project.name).get("source_language_code"),
        )
        voice_subset = self.voice_catalog.prompt_voices_for_language(language_label)
        voice_types = [voice["voice_type"] for voice in voice_subset]
        planning_context = object_replace_planning_context(element_index)
        payload = {
            **planning_context,
            "segment_visual_prompts": [
                {
                    "segment_index": int(item["index"]),
                    "visual_prompt": item["visual_prompt"],
                }
                for item in sorted(prepared, key=lambda value: int(value["index"]))
            ],
            "target_language": language_label,
            "translated_dialogue": translated_lines,
            "user_replacement_instruction": user_instruction,
            "has_user_image": user_reference is not None,
            "user_reference_images": self._user_reference_metadata(reference_paths),
            "user_replacement_image": (
                {"name": "Image 1", "sha256": self._sha256(user_reference)}
                if user_reference is not None else None
            ),
            "voice_catalog_sha256": self.voice_catalog.digest,
            "voice_catalog_lines": self.voice_catalog.prompt_lines(voice_subset),
        }
        views = product_views.frozen_product_views(project)
        if views is not None and user_reference is not None:
            payload["user_replacement_image"].update(
                view_count=views["view_count"], interpretation=views["interpretation"],
                views=views["layout"]["views"],
                instruction=product_views.view_instruction(views["view_count"], "Image 1"),
            )
        output_schema, input_sha256 = object_replace_schema(payload, voice_types)

        def normalize_and_merge(value: object) -> dict[str, Any]:
            patch = normalize_object_replace(
                value,
                input_sha256=input_sha256,
                element_index=element_index,
                voice_types=voice_types,
                has_user_image=user_reference is not None,
                user_instruction=user_instruction,
                dialogue_line_count=len(translated_lines),
                reference_image_ids=reference_ids,
            )
            return merge_object_replace_index(
                element_index, patch, voice_types,
            )

        skill = OBJECT_REPLACE_SKILL.read_text(encoding="utf-8")
        if views is not None and user_reference is not None:
            skill += "\n" + product_views.view_instruction(views["view_count"], "Image 1")
        enhanced_index = self._node(
            log,
            "object-replace",
            "project",
            {
                "agent": "deepseek",
                "skill": skill,
                "input_sha256": input_sha256,
                "payload": payload,
                "output_schema": output_schema,
            },
            reference_paths,
            lambda: self.deepseek.complete_structured(
                skill,
                payload,
                output_schema,
                reference_paths,
                validator=lambda value: normalize_object_replace(
                    value,
                    input_sha256=input_sha256,
                    element_index=element_index,
                    voice_types=voice_types,
                    has_user_image=user_reference is not None,
                    user_instruction=user_instruction,
                    dialogue_line_count=len(translated_lines),
                    reference_image_ids=reference_ids,
                ),
            ),
            transform=normalize_and_merge,
            output_media=lambda _value: [],
        )
        metadata = {
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": input_sha256,
            "voice_catalog_sha256": self.voice_catalog.digest,
        }
        return enhanced_index, metadata

    def _ensure_ark_audio_analysis(
        self,
        project_id: str,
        project: Path,
        translated_lines: list[dict[str, Any]],
        log: ResearchLog,
    ) -> dict[str, Any]:
        """ark-audio-analysis 节点：按台词窗口从 work/source.wav 切逐行片段，
        逐行调方舟音频理解，五字段描述落 state.audio_analysis（version 1）。
        紧随翻译节点执行（输入就绪即跑，不等 object-replace）。无台词或无
        音轨（无 source.wav）跳过并落空结果块，续跑不重算。复用按行判断
        （对齐 dialogue-tts confirmed_lines 惯例）：行号 + 台词窗口一致且
        model 与当前配置一致的行直接复用，其余行重新分析；每行确认后整个
        块原子落盘，崩溃续跑不重付已完成行。方舟配置缺失在进 _node 重试
        循环前由客户端构造明确报错（部署错误重试修不好）。"""
        state = self.store.read(project_id)
        model = self.settings.ark_audio_model
        existing = state.get("audio_analysis")
        carried: dict[int, dict[str, Any]] = {}
        if (
            isinstance(existing, dict)
            and existing.get("version") == AUDIO_ANALYSIS_OUTPUT_VERSION
            and existing.get("model") == model
            and isinstance(existing.get("lines"), list)
        ):
            for item in existing["lines"]:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("line_index"), int)
                    and not isinstance(item.get("line_index"), bool)
                    and isinstance(item.get("start_s"), (int, float))
                    and not isinstance(item.get("start_s"), bool)
                    and isinstance(item.get("end_s"), (int, float))
                    and not isinstance(item.get("end_s"), bool)
                    and all(
                        isinstance(item.get(field), str)
                        for field in _ARK_AUDIO_ANALYSIS_FIELDS
                    )
                    and isinstance(item.get("clip_sha256"), str)
                ):
                    carried[item["line_index"]] = item
        audio = project / "work" / "source.wav"
        if not translated_lines or not audio.is_file():
            block: dict[str, Any] = {
                "version": AUDIO_ANALYSIS_OUTPUT_VERSION,
                "lines": [],
            }
            if block != existing:
                self.store.update(project_id, audio_analysis=block)
            return block
        if all(
            carried.get(line_index, {}).get("start_s") == float(line["start_s"])
            and carried.get(line_index, {}).get("end_s") == float(line["end_s"])
            for line_index, line in enumerate(translated_lines, 1)
        ) and len(carried) == len(translated_lines):
            # 全部行窗口与 model 一致：整块复用，不重写 state。
            return existing
        client = self.ark_audio or ArkAudioClient(self.settings)
        directory = project / "work" / "audio-analysis"
        lines: list[dict[str, Any]] = []

        def persist_audio_analysis() -> None:
            self.store.update(project_id, audio_analysis={
                "version": AUDIO_ANALYSIS_OUTPUT_VERSION,
                "model": model,
                "lines": list(lines),
            })

        for line_index, line in enumerate(translated_lines, 1):
            start_s = float(line["start_s"])
            end_s = float(line["end_s"])
            carried_line = carried.get(line_index)
            if (
                carried_line is not None
                and carried_line["start_s"] == start_s
                and carried_line["end_s"] == end_s
            ):
                lines.append(carried_line)
                continue
            clip = directory / f"line-{line_index:04d}.wav"

            def operation(
                clip: Path = clip, start_s: float = start_s, end_s: float = end_s,
            ) -> ProviderResult[dict[str, Any]]:
                media.slice_dialogue_track(audio, start_s, end_s - start_s, clip)
                result = client.analyze(clip, ARK_AUDIO_ANALYSIS_PROMPT)
                return ProviderResult(
                    value={
                        "text": result.value,
                        "clip_sha256": self._sha256(clip),
                    },
                    record=result.record,
                    evidence_media=result.evidence_media,
                )

            def normalize(
                value: dict[str, Any],
                line_index: int = line_index,
                start_s: float = start_s,
                end_s: float = end_s,
            ) -> dict[str, Any]:
                description = parse_ark_audio_analysis(value["text"])
                return {
                    "line_index": line_index,
                    "start_s": start_s,
                    "end_s": end_s,
                    **description,
                    "clip_sha256": value["clip_sha256"],
                }

            lines.append(
                self._node(
                    log,
                    "ark-audio-analysis",
                    "project",
                    {
                        "line_index": line_index,
                        "start_s": start_s,
                        "end_s": end_s,
                        "source": audio,
                        "clip": clip,
                        "model": model,
                    },
                    [audio],
                    operation,
                    lambda _value, clip=clip: [clip],
                    transform=normalize,
                )
            )
            persist_audio_analysis()
        block = {
            "version": AUDIO_ANALYSIS_OUTPUT_VERSION,
            "model": model,
            "lines": lines,
        }
        if block != existing:
            self.store.update(project_id, audio_analysis=block)
        return block

    def _ensure_dialogue_delivery(
        self,
        project_id: str,
        translated_lines: list[dict[str, Any]],
        line_voices: list[str],
        language_label: str,
        log: ResearchLog,
    ) -> dict[str, Any]:
        """dialogue-delivery-plan 节点：DeepSeek 逐行规划
        {speech_rate, emotion, context_texts}，落 state.dialogue_delivery
        （version 1）。独立节点、不并入 object-replace：它是合成期派生产物，
        不进冻结索引，避免污染图版哈希链。复用判据 = version + input_sha256
        + instruction_sha256：
        输入未变且指令文本未变时复用既有块；台词、音色绑定、音频分析或
        系统指令本身变化则自动重算，resume-pre-picsart 路径因此自然重跑
        该节点。audio_analysis 是可选输入（B2 产物），存在则注入提示词，
        缺失仅凭文本/窗口规划。"""
        state = self.store.read(project_id)
        analysis_lines: list[Any] | None = None
        audio_analysis = state.get("audio_analysis")
        if isinstance(audio_analysis, dict) and isinstance(
            audio_analysis.get("lines"), list,
        ):
            analysis_lines = audio_analysis["lines"]
        # 音色标定缺口在 payload 构造时明确报错：不进 _node 重试循环，
        # 重试修不好缺失的标定值（部署错误应当暴露）。
        payload = build_dialogue_delivery_payload(
            target_language=language_label,
            translated_lines=translated_lines,
            line_voices=line_voices,
            catalog=self.voice_catalog,
            analysis_lines=analysis_lines,
        )
        output_schema, input_sha256 = dialogue_delivery_schema(payload)
        instruction_sha256 = dialogue_delivery_instruction_sha256(language_label)
        existing = state.get("dialogue_delivery")
        if (
            isinstance(existing, dict)
            and existing.get("version") == DIALOGUE_DELIVERY_OUTPUT_VERSION
            and existing.get("input_sha256") == input_sha256
            # 无 instruction_sha256 的旧块直接判失效重算（不做缺省兼容）。
            and existing.get("instruction_sha256") == instruction_sha256
            and isinstance(existing.get("lines"), list)
            and len(existing["lines"]) == len(translated_lines)
        ):
            return existing

        def normalize(value: object) -> list[dict[str, Any]]:
            return normalize_dialogue_delivery(
                value,
                input_sha256=input_sha256,
                line_count=len(translated_lines),
            )

        lines = self._node(
            log,
            "dialogue-delivery-plan",
            "project",
            {
                "agent": "deepseek",
                "input_sha256": input_sha256,
                "instruction_sha256": instruction_sha256,
                "analysis_used": bool(analysis_lines),
                "payload": payload,
                "output_schema": output_schema,
            },
            [],
            lambda: self.deepseek.complete_structured(
                dialogue_delivery_instruction(language_label),
                payload,
                output_schema,
                [],
                validator=normalize,
            ),
            transform=normalize,
            output_media=lambda _value: [],
        )
        block = {
            "version": DIALOGUE_DELIVERY_OUTPUT_VERSION,
            "input_sha256": input_sha256,
            "instruction_sha256": instruction_sha256,
            # 空结果块（lines=[]，跳过产物）不含可用描述，记 false。
            "analysis_used": bool(analysis_lines),
            "lines": lines,
        }
        self.store.update(project_id, dialogue_delivery=block)
        return block

    def _run_dialogue_audio(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        target_language: str,
        log: ResearchLog,
    ) -> Path | None:
        """Synthesize every dialogue line and assemble the full speech track."""
        state = self.store.read(project_id)
        translated_lines = state.get("translated_voice_lines")
        if not isinstance(translated_lines, list) or any(
            not isinstance(line, dict) for line in translated_lines
        ):
            raise ValueError("stored translated dialogue is invalid")
        if not translated_lines:
            # 无台词：dialogue-delivery-plan 跳过并落空结果块（续跑不重算）。
            existing = state.get("dialogue_delivery")
            if not (
                isinstance(existing, dict)
                and existing.get("version") == DIALOGUE_DELIVERY_OUTPUT_VERSION
                and existing.get("lines") == []
            ):
                self.store.update(
                    project_id,
                    dialogue_delivery={
                        "version": DIALOGUE_DELIVERY_OUTPUT_VERSION,
                        "lines": [],
                    },
                )
            return None
        # Most lines pass through localize_dialogue, but resumed/frozen state
        # can also originate from an older path.  ASR follows the audio stream,
        # which may outlast the visual timeline by a few microseconds; keep
        # that harmless drift from reaching TTS as an out-of-range window.
        timeline_end = max(float(item["end_s"]) for item in prepared)
        dialogue_voices = enhanced_element_index.get("dialogue_voices")
        # 不做形态校验：音色绑定缺失/不齐时按无绑定处理，落到默认音色。
        if not isinstance(dialogue_voices, list):
            dialogue_voices = []
        if len(dialogue_voices) < len(translated_lines):
            dialogue_voices = dialogue_voices + [None] * (
                len(translated_lines) - len(dialogue_voices)
            )
        # 豆包 TTS 按语言选 voice_type；“与原视频相同”在这里解析成具体语言
        # 标签（台词文本的文字证据），解析不出时回落中文音色。
        language_label = _dialogue_language_display(
            target_language,
            translated_lines,
            self.store.read(project_id).get("source_language_code"),
        )
        default_voice_type = self.voice_catalog.default_voice_type(language_label)
        # dialogue_voices 逐项是 Object Replace 分配的豆包 voice_type；无绑定
        # （null）、目录外或未标定的值回落到该语种的已标定默认音色——老项目
        # 冻结的 enhanced_element_index 可能带着标定前分配的音色（如
        # en_female_brittney），续跑时在这里自动降级，无需手术改 state。
        # delivery 规划与逐行合成共用这份解析结果，speech_rate 基准才与
        # 实际合成音色一致。
        line_voices = [
            (
                assigned_voice_type
                if self.voice_catalog.has(assigned_voice_type)
                and is_calibrated_voice(assigned_voice_type)
                else default_voice_type
            )
            for assigned_voice_type in dialogue_voices[: len(translated_lines)]
        ]
        clamped_lines: list[dict[str, Any]] = []
        for line in translated_lines:
            text = line.get("text")
            try:
                start_s = float(line["start_s"])
                end_s = float(line["end_s"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "stored translated dialogue is invalid"
                ) from None
            if (
                not isinstance(text, str)
                or not text.strip()
                or not math.isfinite(start_s)
                or not math.isfinite(end_s)
                or not 0 <= start_s < end_s
            ):
                raise ValueError("stored translated dialogue is invalid")
            clamped_lines.append({
                "text": text,
                "start_s": min(start_s, timeline_end),
                "end_s": min(end_s, timeline_end),
            })
        # 逐行音色校正（2026-09-27）：dialogue_voices 是 Object Replace LLM
        # 的绑定，会错配性别（原片旁白性别与画面演示者可以不一致，如
        # b6240a72 全程男声旁白但 seg2/3 是女性演示）。以画面主口播人物
        # 性别为准逐行校正：一致保留 LLM 绑定（不重折腾），不一致按语言 ×
        # 性别重选；无画面证据时方舟 gender_age 多数票兜底。规则与音频复刻
        # 脚本（scripts/audio_regen_e2e.py）共用 resolve_line_voices。
        voice_analysis = state.get("audio_analysis")
        voice_analysis_lines = (
            voice_analysis["lines"]
            if isinstance(voice_analysis, dict)
            and isinstance(voice_analysis.get("lines"), list)
            else None
        )
        people = enhanced_element_index.get("people")
        voice_resolution = resolve_line_voices(
            lines=clamped_lines,
            segments=[
                {
                    "index": int(item["index"]),
                    "start_s": float(item["start_s"]),
                    "end_s": float(item["end_s"]),
                }
                for item in prepared
            ],
            people=people if isinstance(people, dict) else {},
            inherited_voices=line_voices,
            catalog=self.voice_catalog,
            language_label=language_label,
            analysis_lines=voice_analysis_lines,
        )
        line_voices = [entry["voice_type"] for entry in voice_resolution]
        line_explicit_languages = [
            entry["explicit_language"] for entry in voice_resolution
        ]
        log.event("dialogue_voice_resolution", lines=voice_resolution)
        # voice-over（幕后口播）判定（2026-10-03 用户拍板）：逐行音色解析出
        # 两种及以上音色时，统一为 default_voice_type（该语种已标定默认音
        # 色，确定性、有语速基准）的单一音色画外音，不做口型。判定证据
        # （解析出的逐行音色）随模式一起落 state：H3 冻结提示词 dialogue
        # 块、H3 提交 reference_audio_urls、Context IR 蒸馏输入/素材与
        # 冻结清单的 dialogue_track 回执一律消费持久化值，resume 逐字重放
        # 旧提示词时不重新判定。
        resolved_line_voices = list(line_voices)
        voice_over = len(set(resolved_line_voices)) >= 2
        if voice_over:
            line_voices = [default_voice_type] * len(line_voices)
            log.event(
                "dialogue_voice_over",
                resolved_voices=sorted(set(resolved_line_voices)),
                unified_voice_type=default_voice_type,
            )
        self.store.update(
            project_id,
            dialogue_voice_over={
                "enabled": voice_over,
                "resolved_voices": resolved_line_voices,
                "voice_type": default_voice_type if voice_over else None,
                "language_label": language_label,
            },
        )
        delivery = self._ensure_dialogue_delivery(
            project_id, clamped_lines, line_voices, language_label, log,
        )
        plan_lines = delivery.get("lines")
        if not isinstance(plan_lines, list) or len(plan_lines) != len(
            clamped_lines
        ):
            # delivery 块缺失/空结果：合成参数全部 None，行为与现状一致。
            plan_lines = []
        # dialogue_tts 状态块（version 1）：run_id 每次执行（含续跑）新生成，
        # section_id = uuid5(run_id:voice_type)，每次运行 × 每个说话人一个
        # section，同 section 内严格按行序串行发送（下方逐行串行循环）。
        # confirmed_lines 逐条记录
        # {line_index, input_sha256}：上次运行确认合成成功并落盘的行，绑定
        # 当次合成输入（文本+音色+窗口+delivery 参数）的哈希。续跑行复用
        # 判据 = 行号已确认 + 当次输入哈希一致 + 行 wav 非空可解码，三者
        # 缺一即重新合成（付费行为缩减：只有输入真正变化的行才重新扣费）：续跑重跑翻译
        # （输出非确定）产出行数相同、音色绑定相同、文本不同的译文时，
        # 旧文本 wav 不得复用。旧格式（纯行号，2026-09-26 前）无哈希可
        # 比对，一律重新合成——复用收紧是安全方向。
        confirmed_carry: dict[str, list[dict[str, Any]]] = {}
        previous_tts = state.get("dialogue_tts")
        if isinstance(previous_tts, dict) and isinstance(
            previous_tts.get("sections"), list,
        ):
            for previous_section in previous_tts["sections"]:
                if (
                    isinstance(previous_section, dict)
                    and isinstance(previous_section.get("voice_type"), str)
                    and isinstance(
                        previous_section.get("confirmed_lines"), list,
                    )
                ):
                    bucket = confirmed_carry.setdefault(
                        previous_section["voice_type"], [],
                    )
                    for item in previous_section["confirmed_lines"]:
                        if (
                            isinstance(item, dict)
                            and isinstance(item.get("line_index"), int)
                            and not isinstance(item.get("line_index"), bool)
                            and isinstance(item.get("input_sha256"), str)
                            and all(
                                entry["line_index"] != item["line_index"]
                                for entry in bucket
                            )
                        ):
                            bucket.append({
                                "line_index": item["line_index"],
                                "input_sha256": item["input_sha256"],
                            })
        run_id = str(uuid.uuid4())
        sections: dict[str, dict[str, Any]] = {}

        def persist_dialogue_tts() -> None:
            self.store.update(project_id, dialogue_tts={
                "version": 1,
                "run_id": run_id,
                "sections": list(sections.values()),
            })

        directory = project / "work" / "dialogue-audio"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        line_audios: list[dict[str, Any]] = []
        for offset, (line, voice_type) in enumerate(
            zip(clamped_lines, line_voices), 1
        ):
            text = line["text"]
            start_s = line["start_s"]
            end_s = line["end_s"]
            if start_s >= end_s:
                # A row wholly beyond the video has no audible placement.
                continue
            destination = directory / f"line-{offset:04d}.wav"
            section = sections.setdefault(voice_type, {
                "section_id": dialogue_tts_section_id(run_id, voice_type),
                "voice_type": voice_type,
                "line_indexes": [],
                "confirmed_lines": list(confirmed_carry.get(voice_type, [])),
            })
            section["line_indexes"].append(offset)
            plan = plan_lines[offset - 1] if plan_lines else {}
            speech_rate = plan.get("speech_rate")
            emotion = plan.get("emotion")
            context_texts = plan.get("context_texts")
            line_input_sha256 = dialogue_tts_line_input_sha256(
                text=text,
                voice_type=voice_type,
                start_s=start_s,
                end_s=end_s,
                speech_rate=speech_rate,
                emotion=emotion,
                context_texts=context_texts,
                explicit_language=line_explicit_languages[offset - 1],
            )
            confirmed_inputs = {
                item["line_index"]: item["input_sha256"]
                for item in section["confirmed_lines"]
            }
            if (
                confirmed_inputs.get(offset) == line_input_sha256
                and is_decodable_wav(destination)
            ):
                log.event(
                    "dialogue_tts_line_reused",
                    line_index=offset,
                    voice_type=voice_type,
                )
                line_audios.append(
                    {"audio": destination, "start_s": start_s, "end_s": end_s}
                )
                continue

            def operation(
                text: str = text,
                voice_type: str = voice_type,
                destination: Path = destination,
                speech_rate: int | None = speech_rate,
                emotion: str | None = emotion,
                context_texts: list[str] | None = context_texts,
                section_id: str = section["section_id"],
                explicit_language: str | None = line_explicit_languages[offset - 1],
            ) -> ProviderResult[Path]:
                return self.tts.synthesize(
                    text=text,
                    voice_type=voice_type,
                    destination=destination,
                    speech_rate=speech_rate,
                    emotion=emotion,
                    context_texts=context_texts,
                    section_id=section_id,
                    explicit_language=explicit_language,
                )

            self._node(
                log,
                "dialogue-tts-line",
                "project",
                {
                    "line_index": offset,
                    "text": text,
                    "start_s": start_s,
                    "end_s": end_s,
                    "language": language_label,
                    "voice_type": voice_type,
                    "explicit_language": line_explicit_languages[offset - 1],
                    "speech_rate": speech_rate,
                    "emotion": emotion,
                    "context_texts": context_texts,
                    "section_id": section["section_id"],
                    "output": destination,
                },
                [],
                operation,
                lambda value: [value],
            )
            # 重试二分门禁：synthesize 抛出的可重试错误（网络/5xx/超时）说明
            # 流未正常结束，_node 通用重试同 section 重发是安全的；synthesize
            # 成功返回后（流已读完、完整 PCM 一次性 write_bytes 落盘，无半
            # 文件窗口）这里只做行号确认与 state 原子落盘——此处的失败直接
            # 让节点失败（fail-closed），禁止重发，避免重复进入服务端
            # section 历史。
            section["confirmed_lines"] = [
                item
                for item in section["confirmed_lines"]
                if item["line_index"] != offset
            ]
            section["confirmed_lines"].append({
                "line_index": offset,
                "input_sha256": line_input_sha256,
            })
            persist_dialogue_tts()
            line_audios.append(
                {"audio": destination, "start_s": start_s, "end_s": end_s}
            )
        if not line_audios:
            return None
        full_track = directory / "full-track.wav"
        slices = [
            (
                float(item["start_s"]),
                float(item["duration_s"]),
                project
                / "work" / "segments" / f"{int(item['index']):04d}"
                / "dialogue-track.wav",
            )
            for item in prepared
            if item.get("dialogue")
        ]
        total_duration = sum(float(item["duration_s"]) for item in prepared)

        def assemble() -> dict[str, Any]:
            result = media.assemble_dialogue_track(
                line_audios, total_duration, full_track,
            )
            for start_s, duration_s, slice_path in slices:
                media.slice_dialogue_track(
                    full_track, start_s, duration_s, slice_path,
                )
            return result

        assembled = self._node(
            log,
            "assemble-dialogue-track",
            "project",
            {
                "lines": [
                    {**entry, "audio": entry["audio"].name}
                    for entry in line_audios
                ],
                "total_duration_s": total_duration,
                "output": full_track,
                "slices": [slice_path for _start, _duration, slice_path in slices],
            },
            [entry["audio"] for entry in line_audios],
            assemble,
            output_media=lambda _value: [
                full_track,
                *(slice_path for _s, _d, slice_path in slices),
            ],
        )
        # 超长减速封顶（atempo 1.15x）仍放不下的尾差被淡出截断：逐行记
        # warning（_node 成功后记一次，重试不重复记）。
        for line in assembled.get("lines", []):
            truncated_s = line.get("tail_truncated_s", 0.0)
            if (
                isinstance(truncated_s, (int, float))
                and not isinstance(truncated_s, bool)
                and truncated_s > 0
            ):
                log.event(
                    "dialogue_line_tail_truncated",
                    start_s=line.get("start_s"),
                    end_s=line.get("end_s"),
                    audio_duration_s=line.get("audio_duration_s"),
                    tail_truncated_s=truncated_s,
                )
        return full_track

    @staticmethod
    def _enhanced_index_uses_user_image(
        enhanced_element_index: dict[str, Any],
    ) -> bool:
        """Derive the effective user-image binding from the frozen index."""

        for category in ("people", "entities", "scenes"):
            records = enhanced_element_index.get(category)
            if not isinstance(records, dict):
                continue
            for record in records.values():
                replacement = (
                    record.get("replacement")
                    if isinstance(record, dict)
                    else None
                )
                if (
                    isinstance(replacement, dict)
                    and replacement.get("source") == "image_1"
                ):
                    return True
        return False

    @staticmethod
    def _enhanced_index_has_replacement(
        enhanced_element_index: dict[str, Any],
    ) -> bool:
        """Return whether any globally indexed visual element is replaced."""

        for category in ("people", "entities", "scenes"):
            records = enhanced_element_index.get(category)
            if not isinstance(records, dict):
                continue
            for record in records.values():
                if isinstance(record, dict) and record.get("replacement"):
                    return True
        return False

    def _run_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        object_replace: dict[str, Any],
        replacement: Path | None,
        log: ResearchLog,
        *,
        user_reference: Path | None = None,
    ) -> tuple[dict[str, Any], Path | None]:
        """Render the one project-wide appearance sheet from the frozen index."""

        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                "replacement sheet requires the current Object Replace output"
            )
        spec = build_replacement_sheet_spec(enhanced_element_index)
        replacement_images = self._replacement_image_paths(project, replacement=replacement)
        user_image_present = bool(replacement_images)
        existing = self.store.read(project_id).get("replacement_sheet")
        if (
            isinstance(existing, dict)
            and existing.get("version") != REPLACEMENT_SHEET_VERSION
        ):
            raise ValueError("stored replacement sheet version is unsupported")
        if isinstance(existing, dict) and existing.get("status") in {
            "not_needed", "succeeded",
        }:
            if existing.get("spec_sha256") != spec["spec_sha256"]:
                raise ValueError(
                    "frozen replacement sheet does not match the current global index"
                )
            existing_path = self._validated_stored_replacement_sheet(
                project, self.store.read(project_id),
            )
            return existing, existing_path
        if isinstance(existing, dict) and existing.get("status") == "prepared":
            if existing.get("spec_sha256") != spec["spec_sha256"]:
                raise ValueError(
                    "prepared replacement sheet does not match the current global index"
                )
            raise ValueError(
                "prepared replacement sheet requires receipt-safe replay"
            )
        if user_image_present and user_reference is None:
            assert replacement is not None
            user_reference = self._materialize_canonical_user_reference(
                project, replacement, log,
                replacement_paths=self._stored_replacement_paths(project, self.store.read(project_id)),
            )
        # The replacement sheet is always H3 Image 1 so the identity anchor
        # carries the primary reference weight; the user image, when present,
        # follows as Image 2.
        image_reference = "Image 1"
        h3_reference_images: list[dict[str, Any]] = []
        state = self.store.read(project_id)
        voice_references = state.get("voice_references")
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        if not isinstance(voice_references, list):
            raise ValueError("stored voice references are invalid")
        base_bundle: dict[str, Any] = {
            "version": REPLACEMENT_SHEET_VERSION,
            "status": "not_needed" if not spec["tiles"] else "prepared",
            "spec": spec,
            "spec_sha256": spec["spec_sha256"],
            "user_image_present": user_image_present,
            "h3_image_reference": image_reference if spec["tiles"] else None,
            "h3_reference_images": h3_reference_images,
            "upstream_bindings": self._replacement_sheet_upstream_bindings(
                project,
                prepared,
                enhanced_element_index,
                object_replace,
                voice_references,
                effective_request,
                replacement,
            ),
            "seedream": None,
            "output": None,
        }
        if not spec["tiles"]:
            views = product_views.frozen_product_views(project)
            if views is not None:
                # composed 模式：多图已合成为单一 provider 参考图。
                h3_reference_images.append({
                    "image_reference": "Image 1",
                    "role": "user_replacement",
                    **self._project_artifact_receipt(
                        project, project / product_views.REFERENCE,
                        product_views.REFERENCE,
                    ),
                })
            else:
                for order, original in enumerate(replacement_images, 1):
                    canonical = self._canonical_user_reference_path(project, order)
                    if not canonical.is_file():
                        canonical = self._materialize_canonical_user_reference(project, original, log, order)
                    h3_reference_images.append({
                        "image_reference": f"Image {order}",
                        "role": "user_replacement",
                        **self._project_artifact_receipt(project, canonical, canonical.relative_to(project).as_posix()),
                    })
            self.store.update(project_id, replacement_sheet=base_bundle)
            return base_bundle, None

        directory = project / "work" / "global-reference-sheet"
        views = product_views.frozen_product_views(project)
        inputs = self._node(
            log,
            "replacement-sheet-inputs",
            "project",
            {
                "spec": spec,
            },
            [],
            lambda: materialize_replacement_sheet_inputs(
                directory,
                spec,
                user_image_present=user_image_present,
                # composed 模式多图已合成单一 provider 参考图（1 图 + N 视角）；
                # direct 模式逐图直传，数量即上传图数。
                user_image_count=1 if views is not None else len(replacement_images),
                **({"user_view_count": views["view_count"]} if views is not None else {}),
            ),
            output_media=lambda value: [value["canvas_path"]],
        )
        canvas = inputs["canvas_path"]
        prompt_path = directory / "prompt.txt"
        spec_path = directory / "spec.json"
        canonical_replacements: list[Path] = []
        if views is not None:
            # composed 模式：多图已冻结合成为单一 provider 参考图。
            canonical_replacements.append(project / product_views.REFERENCE)
        else:
            for order, original in enumerate(replacement_images, 1):
                canonical_replacement = (
                    user_reference if order == 1 and user_reference is not None
                    else self._canonical_user_reference_path(project, order)
                )
                if not canonical_replacement.is_file():
                    canonical_replacement = self._materialize_canonical_user_reference(project, original, log, order)
                canonical_replacements.append(canonical_replacement)
        for order, canonical_replacement in enumerate(canonical_replacements, 1):
            canonical_bytes = canonical_replacement.read_bytes()
            if (
                not canonical_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or len(canonical_bytes) < 9
            ):
                raise ValueError("prepared provider reference is not PNG")
            canonical_receipt = self._project_artifact_receipt(
                project,
                canonical_replacement,
                canonical_replacement.relative_to(project).as_posix(),
            )
            h3_reference_images.append({
                "image_reference": f"Image {order + 1}",
                "role": "user_replacement",
                **canonical_receipt,
            })
        seedream_inputs = [
            canvas,
            *canonical_replacements,
        ]
        input_receipts = [
            self._project_artifact_receipt(
                project,
                path,
                path.relative_to(project).as_posix(),
            )
            for path in seedream_inputs
        ]
        base_bundle["h3_reference_images"] = h3_reference_images
        seedream_state = {
            "model": self.settings.seedream_model,
            "endpoint": (
                self.settings.seedream_api_base + "/images/generations"
            ),
            "prompt": self._project_artifact_receipt(
                project, prompt_path, "work/global-reference-sheet/prompt.txt",
            ),
            "spec": self._project_artifact_receipt(
                project, spec_path, "work/global-reference-sheet/spec.json",
            ),
            "inputs": input_receipts,
            "request_sha256": self._json_sha256({
                "version": 1,
                "endpoint": (
                    self.settings.seedream_api_base + "/images/generations"
                ),
                "model": self.settings.seedream_model,
                "prompt_sha256": self._sha256(prompt_path),
                "input_sha256": [
                    receipt["sha256"] for receipt in input_receipts
                ],
                "size": SEEDREAM_OUTPUT_SIZE,
                "response_format": "b64_json",
                "watermark": False,
            }),
            "receipt": None,
        }
        prepared_bundle = {**base_bundle, "seedream": seedream_state}
        # Persist the exact global-index-derived plan before the paid image POST.
        self.store.update(project_id, replacement_sheet=prepared_bundle)

        provider_output = directory / "provider-output.bin"
        receipt_path = directory / "seedream-receipt.json"
        self._seedream_generate_never_fail(
            project_id,
            project,
            log,
            "seedream-reference-submit",
            {
                "model": self.settings.seedream_model,
                "spec_sha256": spec["spec_sha256"],
                "prompt_sha256": self._sha256(prompt_path),
                "input_receipts": input_receipts,
            },
            seedream_inputs,
            prompt_path=prompt_path,
            provider_output=provider_output,
            receipt_path=receipt_path,
            seedream_state=seedream_state,
            sheet_bundle=prepared_bundle,
        )

        return self._finish_replacement_sheet(
            project_id,
            project,
            prepared_bundle,
            provider_output,
            receipt_path,
            log,
            node="replacement-sheet-finalize",
        )

    def _finish_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared_bundle: dict[str, Any],
        provider_output: Path,
        receipt_path: Path,
        log: ResearchLog,
        *,
        node: str,
    ) -> tuple[dict[str, Any], Path]:
        spec = prepared_bundle["spec"]
        final_path = project / "work" / "global-reference-sheet" / "reference-sheet.png"
        self._node(
            log,
            node,
            "project",
            {
                "provider_output": provider_output,
                "spec_sha256": spec["spec_sha256"],
                "output": final_path,
            },
            [provider_output],
            lambda: finalize_replacement_sheet(
                provider_output,
                spec,
                final_path,
            ),
            output_media=lambda _value: [final_path],
        )
        seedream_state = prepared_bundle["seedream"]
        completed_bundle = {
            **prepared_bundle,
            "status": "succeeded",
            "seedream": {
                **seedream_state,
                "receipt": self._project_artifact_receipt(
                    project,
                    receipt_path,
                    "work/global-reference-sheet/seedream-receipt.json",
                ),
            },
            "output": self._project_artifact_receipt(
                project,
                final_path,
                "work/global-reference-sheet/reference-sheet.png",
            ),
        }
        completed_bundle["h3_reference_images"] = [
            {
                "image_reference": prepared_bundle["h3_image_reference"],
                "role": "global_replacement_sheet",
                **completed_bundle["output"],
                "spec_sha256": spec["spec_sha256"],
            },
            *prepared_bundle["h3_reference_images"],
        ]
        self.store.update(project_id, replacement_sheet=completed_bundle)
        return completed_bundle, final_path

    _SEEDREAM_REGENERATABLE_CODES = frozenset({
        "provider_rejected", "submission_unknown", "provider_protocol_error",
    })

    def _seedream_generate_never_fail(
        self,
        project_id: str,
        project: Path,
        log: ResearchLog,
        node: str,
        inputs_summary: dict[str, Any],
        images: list[Path],
        *,
        prompt_path: Path,
        provider_output: Path,
        receipt_path: Path,
        seedream_state: dict[str, Any],
        sheet_bundle: dict[str, Any],
    ) -> None:
        """Submit the frozen Seedream sheet request, auto re-POSTing terminal
        submission outcomes within a cap: submission-unknown/protocol outcomes
        reset the attempt artifacts and re-POST; content rejections
        additionally perturb the prompt and re-freeze its receipt before
        re-POSTing.  Re-POSTs are capped at
        ``SEEDREAM_REGENERATION_MAX_ATTEMPTS`` (every re-POST is a new paid
        generation); past the cap the attempt evidence is kept and the
        project fails closed for a manual resume (which resets the receipt
        and re-POSTs)."""
        base_prompt = prompt_path.read_text(encoding="utf-8")
        regenerations = 0
        while True:
            prompt = prompt_path.read_text(encoding="utf-8")
            try:
                self._node(
                    log,
                    node,
                    "project",
                    {**inputs_summary, "regeneration": regenerations},
                    images,
                    lambda prompt=prompt: self.seedream.generate(
                        prompt=prompt,
                        images=images,
                        destination=provider_output,
                        receipt_path=receipt_path,
                    ),
                    output_media=lambda path: [path],
                    retry_if=lambda _exc: False,
                )
                return
            except ProviderFailure as exc:
                code = getattr(exc.cause, "code", None)
                if code not in self._SEEDREAM_REGENERATABLE_CODES:
                    raise
                regenerations += 1
                if regenerations > SEEDREAM_REGENERATION_MAX_ATTEMPTS:
                    log.event(
                        "seedream_regeneration_exhausted",
                        node=node,
                        reason=code,
                        regenerations=SEEDREAM_REGENERATION_MAX_ATTEMPTS,
                        error=str(exc),
                    )
                    raise SeedreamRegenerationExhaustedError(
                        f"Seedream 自动重提 "
                        f"{SEEDREAM_REGENERATION_MAX_ATTEMPTS} 次仍失败"
                        f"（末次 outcome {code}），fail-closed 需人工介入；"
                        f"人工可走 resume 续跑（续跑自动重置回执重提）: {exc}"
                    ) from exc
                removed = _seedream_reset_attempt(receipt_path, provider_output)
                if code == "provider_rejected":
                    prompt_path.write_text(
                        base_prompt
                        + f"\n\nRegeneration {regenerations}: produce a fresh "
                        "variation of the same specification.",
                        encoding="utf-8",
                    )
                    seedream_state["prompt"] = self._project_artifact_receipt(
                        project,
                        prompt_path,
                        "work/global-reference-sheet/prompt.txt",
                    )
                    seedream_state["request_sha256"] = self._json_sha256({
                        "version": 1,
                        "endpoint": seedream_state["endpoint"],
                        "model": seedream_state["model"],
                        "prompt_sha256": seedream_state["prompt"]["sha256"],
                        "input_sha256": [
                            receipt["sha256"]
                            for receipt in seedream_state["inputs"]
                        ],
                        "size": SEEDREAM_OUTPUT_SIZE,
                        "response_format": "b64_json",
                        "watermark": False,
                    })
                    self.store.update(
                        project_id, replacement_sheet=sheet_bundle,
                    )
                log.event(
                    "seedream_submission_regenerated",
                    node=node,
                    reason=code,
                    regeneration=regenerations,
                    removed_artifacts=removed,
                    error=str(exc),
                )
                self._retry_sleep(min(60.0 * regenerations, 1800.0))

    def _resume_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        frozen_sheet: dict[str, Any],
        recovery_mode: str,
        log: ResearchLog,
    ) -> tuple[dict[str, Any], Path | None]:
        """Resume only the frozen replacement-sheet checkpoint."""

        current = self.store.read(project_id)
        if current.get("replacement_sheet") != frozen_sheet:
            raise ValueError("stored replacement sheet changed during resume")
        validated_path = self._validated_stored_replacement_sheet(
            project,
            current,
            allow_prepared=True,
        )
        status = frozen_sheet.get("status")
        if status in {"not_needed", "succeeded"}:
            if recovery_mode != "reuse":
                raise ValueError("stored replacement sheet recovery mode is invalid")
            return frozen_sheet, validated_path
        if status != "prepared" or recovery_mode not in {
            "generate", "replay", "regenerate",
        }:
            raise ValueError("stored replacement sheet recovery mode is invalid")
        if self._validated_prepared_seedream_attempt(
            project, frozen_sheet,
        ) != recovery_mode:
            raise ValueError("Seedream attempt evidence changed during resume")

        seedream = frozen_sheet["seedream"]
        prompt_path = project / seedream["prompt"]["path"]
        images = [
            project / receipt["path"]
            for receipt in seedream["inputs"]
        ]
        directory = project / "work" / "global-reference-sheet"
        provider_output = directory / "provider-output.bin"
        receipt_path = directory / "seedream-receipt.json"
        if recovery_mode == "replay":
            self._node(
                log,
                "seedream-reference-replay",
                "project",
                {
                    "model": seedream["model"],
                    "endpoint": seedream["endpoint"],
                    "request_sha256": seedream["request_sha256"],
                    "spec_sha256": frozen_sheet["spec_sha256"],
                    "input_receipts": seedream["inputs"],
                },
                images,
                lambda: self.seedream.replay_received(
                    destination=provider_output,
                    receipt_path=receipt_path,
                ),
                output_media=lambda path: [path],
                retry_if=lambda _exc: False,
            )
        else:
            if recovery_mode == "regenerate":
                # A rejected/unknown terminal receipt cannot be replayed;
                # downgrade it: drop the attempt artifacts and re-POST.
                try:
                    prior = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    prior = None
                removed = _seedream_reset_attempt(receipt_path, provider_output)
                log.event(
                    "seedream_receipt_reset",
                    prior_status=(
                        prior.get("status") if isinstance(prior, dict) else None
                    ),
                    removed_artifacts=removed,
                )
            self._seedream_generate_never_fail(
                project_id,
                project,
                log,
                "seedream-reference-submit-resume",
                {
                    "model": seedream["model"],
                    "endpoint": seedream["endpoint"],
                    "request_sha256": seedream["request_sha256"],
                    "spec_sha256": frozen_sheet["spec_sha256"],
                    "input_receipts": seedream["inputs"],
                },
                images,
                prompt_path=prompt_path,
                provider_output=provider_output,
                receipt_path=receipt_path,
                seedream_state=seedream,
                sheet_bundle=frozen_sheet,
            )
        return self._finish_replacement_sheet(
            project_id,
            project,
            frozen_sheet,
            provider_output,
            receipt_path,
            log,
            node="replacement-sheet-finalize-resume",
        )

    @classmethod
    def _replacement_sheet_upstream_bindings(
        cls,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        object_replace: dict[str, Any],
        voice_references: list[dict[str, Any]],
        effective_request: dict[str, Any],
        replacement: Path | None,
    ) -> dict[str, Any]:
        effective_request = cls._validated_effective_request(
            effective_request
        )
        segment_bindings: list[dict[str, Any]] = []
        source_receipts: list[dict[str, Any]] = []
        sorted_prepared = sorted(
            prepared, key=lambda value: int(value["index"])
        )
        cls._validated_join_modes(sorted_prepared)
        for item in sorted_prepared:
            index = int(item["index"])
            source = item.get("source")
            if not isinstance(source, Path):
                raise ValueError(
                    "replacement sheet segment source is not frozen"
                )
            segment_bindings.append({
                "index": index,
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
                "duration_s": item.get("duration_s"),
                "chain_id": item.get("chain_id"),
                "join_mode": item["join_mode"],
                "visual_prompt": item.get("visual_prompt"),
                "dialogue": item.get("dialogue", []),
            })
            source_receipts.append(cls._project_artifact_receipt(
                project,
                source,
                f"work/segments/{index:04d}/source.mp4",
            ))
        replacement_paths = cls._replacement_image_paths(
            project, replacement=replacement,
        )
        replacement_receipts = [
            cls._project_artifact_receipt(project, path, name)
            for path, name in zip(
                replacement_paths,
                replacement_input_paths(len(replacement_paths)),
            )
        ]
        bindings = {
            "segments_sha256": cls._json_sha256(segment_bindings),
            "segment_sources": source_receipts,
            "enhanced_element_index_sha256": cls._json_sha256(
                enhanced_element_index
            ),
            "object_replace_sha256": cls._json_sha256(object_replace),
            "voice_references_sha256": cls._json_sha256(voice_references),
            "effective_request_sha256": cls._json_sha256(effective_request),
            "replacement_image": (
                replacement_receipts[0] if replacement_receipts else None
            ),
        }
        # composed 模式：冻结合图清单进 bindings。
        views = product_views.frozen_product_views(project)
        if views is not None:
            bindings["replacement_views"] = views
        # Keep old single-image checkpoint bytes stable.  Multi-image projects
        # bind every ordered original independently, including Image 1.
        if len(replacement_receipts) > 1:
            bindings["replacement_images"] = replacement_receipts
        return bindings

    @staticmethod
    def _muted_artifact_name(muted: Path) -> str:
        return muted.name

    @staticmethod
    def _project_artifact_receipt(
        project: Path,
        path: Path,
        relative_path: str,
    ) -> dict[str, Any]:
        expected = project / relative_path
        project_root = project.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if (
            resolved != expected.resolve(strict=True)
            or project_root not in resolved.parents
            or path.is_symlink()
            or not resolved.is_file()
        ):
            raise ValueError("prepared H3 artifact path is invalid")
        size_bytes = resolved.stat().st_size
        if size_bytes < 1:
            raise ValueError("prepared H3 artifact is empty")
        return {
            "path": relative_path,
            "size_bytes": size_bytes,
            "sha256": Pipeline._sha256(resolved),
        }

    @staticmethod
    def _validated_dialogue_voice_over(state: dict[str, Any]) -> bool:
        """voice-over（幕后口播）模式判定的持久化读取与一致性校验。

        state.dialogue_voice_over 由 _run_dialogue_audio 在逐行音色解析后
        原子写入：{enabled, resolved_voices, voice_type, language_label}。
        消费点（H3 提示词 dialogue 块、H3 提交台词轨、Context IR 蒸馏/
        素材、冻结清单回执）一律读这里的持久化值，不重新判定。字段缺失
        （2026-10-03 前冻结的旧项目）返回 False（既有对口型行为）；形态
        非法或模式与音色集矛盾一律 fail-closed，不静默切换。
        """
        raw = state.get("dialogue_voice_over")
        if raw is None:
            return False
        enabled = raw.get("enabled") if isinstance(raw, dict) else None
        resolved = raw.get("resolved_voices") if isinstance(raw, dict) else None
        voice_type = raw.get("voice_type") if isinstance(raw, dict) else None
        language_label = (
            raw.get("language_label") if isinstance(raw, dict) else None
        )
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {"enabled", "resolved_voices", "voice_type", "language_label"}
            or not isinstance(enabled, bool)
            or not isinstance(resolved, list)
            or any(
                not isinstance(voice, str) or not voice for voice in resolved
            )
            or (len(set(resolved)) >= 2) != enabled
            or (enabled and (not isinstance(voice_type, str) or not voice_type))
            or (not enabled and voice_type is not None)
            or not isinstance(language_label, str)
            or not language_label
        ):
            raise ValueError("stored dialogue voice-over state is invalid")
        return enabled

    def _dialogue_voice_over_enabled(self, project_id: str) -> bool:
        return self._validated_dialogue_voice_over(
            self.store.read(project_id)
        )

    def _freeze_h3_prepared_artifacts(
        self,
        project_id: str,
        project: Path,
        h3_inputs: list[tuple[dict[str, Any], str, Path, str]],
        *,
        replacement: Path | None = None,
        replacement_sheet: dict[str, Any] | None = None,
        request: dict[str, Any] | None = None,
    ) -> None:
        segments: dict[str, dict[str, Any]] = {}
        sorted_h3_inputs = sorted(
            h3_inputs, key=lambda value: int(value[0]["index"]),
        )
        self._validated_join_modes([value[0] for value in sorted_h3_inputs])
        voice_over = self._dialogue_voice_over_enabled(project_id)
        for item, final_prompt, muted, fusion_input_sha256 in sorted_h3_inputs:
            index = int(item["index"])
            base = f"work/segments/{index:04d}"
            # voice-over 段不传 Audio 1：回执与无台词段同形为 None。
            dialogue_track_receipt = (
                self._project_artifact_receipt(
                    project,
                    muted.parent / "dialogue-track.wav",
                    f"{base}/dialogue-track.wav",
                )
                if item.get("dialogue") and not voice_over
                else None
            )
            segments[str(index)] = {
                "segment_index": index,
                "source_video": self._project_artifact_receipt(
                    project, item["source"], f"{base}/source.mp4",
                ),
                "muted_video": self._project_artifact_receipt(
                    project, muted,
                    f"{base}/{self._muted_artifact_name(muted)}",
                ),
                "prompt": self._project_artifact_receipt(
                    project, muted.parent / "h3_prompt.txt", f"{base}/h3_prompt.txt",
                ),
                "dialogue_track": dialogue_track_receipt,
                "fusion_input_sha256": fusion_input_sha256,
                "segment_metadata_sha256": self._json_sha256({
                    "index": item.get("index"),
                    "start_s": item.get("start_s"),
                    "end_s": item.get("end_s"),
                    "duration_s": item.get("duration_s"),
                    "join_mode": item["join_mode"],
                }),
            }
        state = self.store.read(project_id)
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        frozen_replacement_sheet = (
            replacement_sheet
            if isinstance(replacement_sheet, dict)
            else state.get("replacement_sheet")
        )
        if (
            not isinstance(enhanced_index, dict)
            or not isinstance(voice_references, list)
            or not isinstance(object_replace, dict)
            or not isinstance(frozen_replacement_sheet, dict)
            or not isinstance(object_replace.get("input_sha256"), str)
            or len(object_replace["input_sha256"]) != 64
        ):
            raise ValueError("H3 preparation binding state is invalid")
        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires "
                f"Object Replace v{OBJECT_REPLACE_OUTPUT_VERSION}"
            )
        if frozen_replacement_sheet.get("version") != REPLACEMENT_SHEET_VERSION:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires "
                f"replacement sheet v{REPLACEMENT_SHEET_VERSION}"
            )
        if state.get("replacement_sheet") != frozen_replacement_sheet:
            raise ValueError(
                "H3 preparation replacement sheet is not the stored sheet"
            )
        self._validated_stored_replacement_sheet(project, state)
        replacement_paths = self._replacement_image_paths(
            project, state, replacement,
        )
        replacement_receipts = [
            self._project_artifact_receipt(project, path, name)
            for path, name in zip(
                replacement_paths,
                replacement_input_paths(len(replacement_paths)),
            )
        ]
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        if request is not None and request != effective_request:
            raise ValueError(
                "H3 preparation request is not the stored effective request"
            )
        output_request = effective_request["output"]
        bindings = {
            "enhanced_element_index_sha256": self._json_sha256(
                enhanced_index,
            ),
            "voice_references_sha256": self._json_sha256(
                voice_references,
            ),
            "object_replace_input_sha256": object_replace["input_sha256"],
            "replacement_image": (
                replacement_receipts[0] if replacement_receipts else None
            ),
            "replacement_sheet_sha256": self._json_sha256(
                frozen_replacement_sheet,
            ),
            "output_request_sha256": self._json_sha256(output_request),
        }
        if len(replacement_receipts) > 1:
            bindings["replacement_images"] = replacement_receipts
        self.store.update(
            project_id,
            h3_prepared_artifacts={
                "version": H3_PREPARED_ARTIFACTS_VERSION,
                "bindings": bindings,
                "segments": segments,
            },
        )

    @staticmethod
    def _write_prompt_review(
        project: Path,
        h3_inputs: list[tuple[dict[str, Any], str, Path, str]],
    ) -> None:
        """落盘每段提示词的四段结构（直通项目也落，供完成后只读展示）。

        body 在权威替换计划块（BEGIN AUTHORITATIVE ... 整行）处拆成
        editable + hidden：hidden 不展示、不可编辑，confirm 时逐字取自本
        文件。替换掉 _prepare_h3_segment 的测试替身不产出 prompt_parts，
        此时跳过落盘；生产路径的真实 _prepare_h3_segment 永远带齐三段。
        """
        segments: list[dict[str, Any]] = []
        for item, _final_prompt, _muted, _fusion_input_sha256 in sorted(
            h3_inputs, key=lambda value: int(value[0]["index"])
        ):
            parts = item.get("prompt_parts")
            if not isinstance(parts, dict):
                return
            editable, hidden = prompt_review.split_body(parts["body"])
            segments.append({
                "segment_index": int(item["index"]),
                "prefix": parts["prefix"],
                "editable": editable,
                "hidden": hidden,
                "suffix": parts["suffix"],
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
            })
        prompt_review.write_review(project, segments)

    def _run_h3_segments(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        replacement: Path | None,
        replacement_sheet: dict[str, Any],
        replacement_sheet_path: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> list[Path] | None:
        """Cross both free-work barriers before scheduling any paid submit.

        Returns None when the prompt-review switch paused the run after the
        freeze: callers must stop before stitching; the confirm endpoint
        continues via the resume-picsart equivalent path."""
        # fusion payload 的 audio_analysis 来自 state（B2 产物，可无）；
        # 一次性读出，各段共享同一块。
        audio_analysis = self.store.read(project_id).get("audio_analysis")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(prepared)),
        ) as pool:
            prepare_futures = [
                pool.submit(
                    self._prepare_h3_segment,
                    project,
                    item,
                    enhanced_element_index,
                    replacement_sheet,
                    request,
                    log,
                    audio_analysis,
                )
                for item in prepared
            ]
            h3_inputs = [future.result() for future in prepare_futures]
            # 冻结的是 IR 前原始提示词（2026-09-30 顺序调整，用户拍板）：
            # prepare → 冻结 →（prompt_review 暂停）→ Context IR（提交前
            # 优化层）→ 提交 H3。IR 覆写 h3_prompt.txt 不破坏冻结可验证性，
            # 校验链见 _validated_h3_prepared_artifacts 的 prompt 分支。
            self._freeze_h3_prepared_artifacts(
                project_id,
                project,
                h3_inputs,
                replacement=replacement,
                replacement_sheet=replacement_sheet,
                request=request,
            )
            self._write_prompt_review(project, h3_inputs)
            if (
                self._validated_effective_request(request).get("prompt_review")
                is True
            ):
                # 提示词审核开关：冻结后即暂停（此刻 IR 未跑，审核展示的是
                # 我方原文）。 confirm 走 resume-picsart 等价入口，IR 步骤
                # 在 resume_r2v 里幂等经过后逐字重放增强文本。
                log.event(
                    "prompt_review_awaiting",
                    segments=[
                        int(item["index"]) for item, *_rest in h3_inputs
                    ],
                )
                self.store.update(
                    project_id,
                    status="awaiting_prompt_review",
                    project_progress={
                        "status": "awaiting_prompt_review",
                        "percent": 57,
                        "stage": "awaiting_prompt_review",
                    },
                    generation={
                        "status": "awaiting_prompt_review",
                        "stage": "awaiting_prompt_review",
                        "error": None,
                    },
                )
                return None
            h3_inputs = self._enhance_h3_prompts_with_context_ir(
                project_id,
                project,
                h3_inputs,
                replacement_sheet,
                enhanced_element_index,
                request,
                log,
                pool,
            )
            preflight_futures = [
                pool.submit(
                    self._prepare_picsart_segment,
                    project_id,
                    project,
                    item,
                    replacement,
                    replacement_sheet_path,
                    request,
                    log,
                    final_prompt=final_prompt,
                    muted=muted,
                )
                for item, final_prompt, muted, _fusion_input_sha256 in h3_inputs
            ]
            picsart_requests = [
                future.result() for future in preflight_futures
            ]
            self._begin_submission_batch(picsart_requests)
            submit_futures = [
                pool.submit(
                    self._generate_picsart_segment,
                    picsart_request,
                    item,
                    log,
                )
                for (item, _final_prompt, _muted, _fusion_hash), picsart_request
                in zip(h3_inputs, picsart_requests)
            ]
            return [future.result() for future in submit_futures]

    def _enhance_h3_prompts_with_context_ir(
        self,
        project_id: str,
        project: Path,
        h3_inputs: list[tuple[dict[str, Any], str, Path, str]],
        replacement_sheet: dict[str, Any],
        element_index: dict[str, Any],
        request: dict[str, Any],
        log: ResearchLog,
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> list[tuple[dict[str, Any], str, Path, str]]:
        """h3-context-ir 节点：冻结后、提交前用 MiniMax 官方 Context IR 增强各段
        提示词（2026-09-30 顺序调整：冻结的是 IR 前原文，本阶段是付费提交前
        的幂等优化层，六条提交路径共用——初始 run/resume-pre-picsart/
        resume-h3-prep 经 _run_h3_segments，prompt-review confirm/
        resume-picsart/regenerate 经 resume_r2v）。

        段级并发与调用方共用同一个 pool。IR 返回的官方结构化增强文本全量
        覆写 h3_prompt.txt（原文备份 h3_prompt_origin.txt，已存在不覆盖；
        冻结校验链 = origin==冻结 && 回执.origin==冻结 && 当前==optimized）。
        prompt-review 审核记录保持 IR 前原文（本阶段不再改 prompt_parts）；
        下游 Picsart 链路消费的 final_prompt 换新为增强文本。enabled=false
        或已增强（context_ir.json 幂等键命中：origin 哈希==当前原文哈希 且
        optimized==当前 h3_prompt.txt 哈希）时逐段 node_skipped；原文变了
        （regenerate/resume-h3-prep 重建提示词后）幂等键自动失效重跑。
        重试耗尽 fail-open 保留原提示词，链路继续；确定性 4xx 与蒸馏超限
        都 fail-closed。
        """
        futures = [
            pool.submit(
                self._context_ir_enhance_segment,
                project_id,
                project,
                item,
                muted,
                replacement_sheet,
                element_index,
                request,
                log,
            )
            for item, _final_prompt, muted, _fusion_input_sha256 in h3_inputs
        ]
        outcomes = [future.result() for future in futures]
        enhanced_inputs: list[tuple[dict[str, Any], str, Path, str]] = []
        for (item, final_prompt, muted, fusion_input_sha256), outcome in zip(
            h3_inputs, outcomes,
        ):
            if outcome.get("status") == "succeeded":
                enhanced = outcome["prompt"]
                # 只换 final_prompt（提交文本）；prompt_parts 保持 IR 前原文，
                # prompt-review 审核/确认的四段重建公式始终面向我方原文。
                enhanced_inputs.append((item, enhanced, muted, fusion_input_sha256))
            else:
                enhanced_inputs.append((item, final_prompt, muted, fusion_input_sha256))
        return enhanced_inputs

    def _context_ir_mask_profile(
        self,
        item: dict[str, Any],
        muted: Path,
        element_index: dict[str, Any],
    ) -> dict[str, Any]:
        """IR 蒸馏/追加用的遮挡画像：prepare 内存产物优先；resume 路径 item
        来自 state（h3_mask_profile 不落 state），按 prepare 同口径推导——
        muted 文件名 + 人物/物品替换组 + opacity。推导版拿不到帧级
        mask_stats，退回目标组级判定（与提示词 mask_block 的 stats 版差异
        如实记录：stats 判定某组零帧遮挡时推导版仍按目标组计，措辞偏保守、
        多一条约束，不会漏约束）。
        """
        existing = item.get("h3_mask_profile")
        if isinstance(existing, dict):
            return existing
        objects_masked = muted.name == "source-objects-masked.mp4"
        return {
            "objects_masked": objects_masked,
            "person_blocked": (
                objects_masked
                and self._segment_has_replaced_person(
                    element_index, int(item["index"]),
                )
            ),
            "object_blocked": (
                objects_masked and bool(self._replaced_object_targets(element_index))
            ),
            "object_semi_transparent": (
                self.settings.object_mask_block_opacity < 1.0
            ),
        }

    def _context_ir_enhance_segment(
        self,
        project_id: str,
        project: Path,
        item: dict[str, Any],
        muted: Path,
        replacement_sheet: dict[str, Any],
        element_index: dict[str, Any],
        request: dict[str, Any],
        log: ResearchLog,
    ) -> dict[str, Any]:
        """单段 Context IR 增强；返回 {"status": ..., "prompt": ...} 结果包。"""
        index = int(item["index"])
        scope = self._segment_scope(index)
        segment_dir = muted.parent
        prompt_path = segment_dir / "h3_prompt.txt"
        origin_path = segment_dir / "h3_prompt_origin.txt"
        receipt_path = segment_dir / "context_ir.json"
        task_holder: dict[str, str | None] = {"task_id": None}
        attempts = 0

        def _is_idempotent() -> bool:
            """幂等键（2026-09-30 改）：回执 succeeded 且 origin 哈希==当前
            原文哈希 且 optimized==当前 h3_prompt.txt 哈希。原文= origin 备份
            文件内容（无备份即当前文件，对应 IR 未跑过）。全部基于文件实际
            哈希：旧顺序项目（origin/回执/当前三者自洽）天然命中不重跑；
            regenerate/resume-h3-prep 重建提示词后原文变化，键自动失效重跑。
            """
            if not receipt_path.is_file() or receipt_path.is_symlink():
                return False
            try:
                stored = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if not isinstance(stored, dict) or stored.get("status") != "succeeded":
                return False
            original_text = (
                origin_path.read_text(encoding="utf-8")
                if origin_path.is_file()
                else prompt_path.read_text(encoding="utf-8")
            )
            return (
                stored.get("origin_prompt_sha256")
                == self._text_sha256(original_text)
                and stored.get("optimized_prompt_sha256")
                == self._text_sha256(prompt_path.read_text(encoding="utf-8"))
            )

        # 蒸馏输入在节点外预构建：disabled / 幂等命中时不构建（避免无谓的
        # 超限 fail-closed）；超官方 7000 硬上限抛 ContextIRInputTooLongError，
        # 直接外抛 fail-closed（与 vendor 400/2013 同语义），不进 fallback。
        mask_profile = self._context_ir_mask_profile(item, muted, element_index)
        distilled_prompt: str | None = None
        # 冻结原文的【属性与用法】段落：方案 A 进蒸馏输入、方案 B 在 IR
        # 成功输出尾部逐字追加。fallback/禁用路径不追加（原文本身已含）。
        usage_paragraphs: list[str] = []
        voice_over = self._dialogue_voice_over_enabled(project_id)
        if self.settings.minimax_context_ir_enabled and not _is_idempotent():
            dialogue_config = (
                self._validated_effective_request(request).get("dialogue") or {}
            )
            processing_config = (
                self._validated_effective_request(request).get("processing") or {}
            )
            origin_text = (
                origin_path.read_text(encoding="utf-8")
                if origin_path.is_file()
                else prompt_path.read_text(encoding="utf-8")
            )
            usage_paragraphs = _extract_usage_paragraphs(origin_text)
            distilled_prompt = _context_ir_input_text(
                visual_prompt=item.get("visual_prompt"),
                dialogue_lines=item.get("dialogue") or [],
                target_language=(
                    dialogue_config.get("target_language")
                    if isinstance(dialogue_config, dict) else None
                ),
                replacement_sheet=replacement_sheet,
                element_index=element_index,
                segment_index=index,
                mask_profile=mask_profile,
                processing=(
                    processing_config if isinstance(processing_config, dict) else None
                ),
                usage_paragraphs=usage_paragraphs,
                voice_over=voice_over,
            )

        def _retry(exc: Exception) -> bool:
            nonlocal attempts
            attempts += 1
            if isinstance(exc, ProviderFailure):
                # 确定性 4xx（含官方限长）：同文重发必然同样失败，fail-closed。
                if self._has_deterministic_client_error(exc.record):
                    return False
                if isinstance(exc.record, dict) and (
                    exc.record.get("outcome") == "rejected"
                ):
                    return False
                # vendor 终态 failed 不计费：放掉旧 task_id，下次尝试重新 POST。
                if isinstance(exc.cause, ContextIRTaskFailedError):
                    task_holder["task_id"] = None
            # 网络/超时/5xx/unknown/vendor-failed 共用总上限；增强是可选
            # 优化（~$0.02/次），不享受传输类无限退避。
            if isinstance(exc, json.JSONDecodeError):
                return attempts < CONTEXT_IR_MAX_ATTEMPTS
            if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
                # 本地校验错误（prompt 为空、时长/比例非法等）重发必然同样
                # 失败；vendor succeeded 但缺 content.prompt 同理（终态任务
                # 重poll 不会变）。直接抛出由兜底 fail-open 接管。
                return False
            return attempts < CONTEXT_IR_MAX_ATTEMPTS

        def _operation() -> ProviderResult[dict[str, Any]]:
            if not self.settings.minimax_context_ir_enabled:
                return self._skipped_result(
                    {"status": "disabled"}, "MINIMAX_CONTEXT_IR_ENABLED=false",
                )
            if distilled_prompt is None:
                # 幂等：增强文本已在 h3_prompt.txt，不重复付费。
                return self._skipped_result(
                    {"status": "skipped", "reason": "idempotent"},
                    "context_ir receipt matches current prompt",
                )
            source_prompt = (
                origin_path.read_text(encoding="utf-8")
                if origin_path.is_file()
                else prompt_path.read_text(encoding="utf-8")
            )
            if not source_prompt.strip():
                raise ValueError("Context IR source prompt is empty")
            duration = item.get("duration_s")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) <= 0
            ):
                raise ValueError("Context IR segment metadata is invalid")
            # duration/ratio 与 Picsart 提交同值（4-15 ⊃ 我方 5-15 取整）。
            provider_duration = max(
                PICSART_GENERATION_MIN_DURATION_S, math.ceil(float(duration)),
            )
            if provider_duration > PICSART_GENERATION_MAX_DURATION_S:
                raise ValueError("Context IR generation duration is unsupported")
            aspect_ratio = self._validated_effective_request(request)["output"][
                "aspect_ratio"
            ]
            if aspect_ratio not in PICSART_ASPECT_RATIOS:
                raise ValueError("Context IR aspect ratio is unsupported")
            # 素材顺序 = 发给 Picsart 的顺序：Image 1..N 按冻结的
            # h3_reference_images，Video 1 = 选中参考流，Audio 1 = 台词轨
            # （仅该段有台词时）。
            image_paths = [
                (project / reference["path"]).resolve(strict=True)
                for reference in replacement_sheet["h3_reference_images"]
            ]
            if len(image_paths) != len(set(image_paths)):
                raise ValueError("Context IR image files must be unique")
            audio_path: Path | None = None
            # voice-over 段 IR 提交不附 Audio 1 素材（与无台词段同）。
            if item.get("dialogue") and not voice_over:
                project_root = project.resolve(strict=True)
                candidate = (segment_dir / "dialogue-track.wav").resolve(strict=True)
                if (
                    project_root not in candidate.parents
                    or candidate.is_symlink()
                    or not candidate.is_file()
                    or candidate.stat().st_size < 1
                ):
                    raise ValueError("Context IR dialogue track is invalid")
                audio_path = candidate
            client = self.context_ir or MiniMaxContextIRClient(self.settings)
            record: dict[str, Any] = {}
            if task_holder["task_id"] is None:
                # 素材必须走 /v1/files/upload → mm_file:// 引用（公网 URL
                # 被 MiniMax 阻断数据中心 IP，实测不可用；HMAC 签名链也被
                # 400）。每次尝试重传、不缓存 file_id（MiniMax 文件只保留
                # 7 天）。串行保持 Image 1..N 编号顺序。
                image_urls = []
                uploads_record = []
                for path in image_paths:
                    uploaded = client.upload(path, kind="image")
                    uploads_record.append(uploaded.record)
                    image_urls.append(uploaded.value)
                uploaded_video = client.upload(muted, kind="video")
                uploads_record.append(uploaded_video.record)
                audio_url = None
                if audio_path is not None:
                    uploaded_audio = client.upload(audio_path, kind="audio")
                    uploads_record.append(uploaded_audio.record)
                    audio_url = uploaded_audio.value
                record["uploads"] = uploads_record
                submit_result = client.submit(
                    prompt=distilled_prompt,
                    image_urls=image_urls,
                    video_url=uploaded_video.value,
                    audio_url=audio_url,
                    duration=provider_duration,
                    ratio=aspect_ratio,
                )
                task_holder["task_id"] = submit_result.value
                record["submit"] = submit_result.record
            else:
                # 已建任务只轮询不重 POST（重复提交即重复计费），也不重传素材。
                record["reused_task_id"] = task_holder["task_id"]
            poll_result = client.poll(task_holder["task_id"])
            record["poll"] = poll_result.record
            raw_enhanced = poll_result.value["prompt"]
            usage = poll_result.value.get("usage")
            # 机械清洗：剥除 <d> 内 <scenetrans> 残片（台词词汇逐字不动）；
            # 清洗后 <d> 为空抛 ContextIRSanitizeError → 不重试、走 fallback。
            enhanced, removed_fragments = _context_ir_sanitize_output(raw_enhanced)
            # IR 不吸收的约束（mask 透出禁令 / overlay 擦除，第五轮实证丢失项；
            # 用户决策"五块不能直接丢弃"）与【属性与用法】段落+形态硬约束
            #（方案 B，根因 081e3436）由节点机械追加，不依赖 IR。
            enhanced, appended_constraints = (
                _context_ir_append_backend_constraints(
                    enhanced,
                    mask_profile=mask_profile,
                    processing=(
                        self._validated_effective_request(request).get("processing")
                        or {}
                    ),
                    usage_paragraphs=usage_paragraphs,
                )
            )
            if not origin_path.exists():
                origin_path.write_text(source_prompt, encoding="utf-8")
            prompt_path.write_text(enhanced, encoding="utf-8")
            receipt_doc = {
                "task_id": task_holder["task_id"],
                "origin_prompt_sha256": self._text_sha256(source_prompt),
                "optimized_prompt_sha256": self._text_sha256(enhanced),
                "distilled_input_sha256": self._text_sha256(distilled_prompt),
                "distilled_input_chars": len(distilled_prompt),
                # 清洗前后文本都进回执审计（清洗后 = 落盘 h3_prompt.txt 那份）。
                "raw_enhanced_prompt": raw_enhanced,
                "sanitized_scenetrans_removed": removed_fragments,
                "appended_constraints": appended_constraints,
                "usage": usage,
                "status": "succeeded",
            }
            receipt_path.write_text(
                json.dumps(receipt_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return ProviderResult(
                {
                    "status": "succeeded",
                    "prompt": enhanced,
                    "prompt_chars": len(enhanced),
                    "task_id": task_holder["task_id"],
                    "usage": usage,
                },
                record,
                [
                    path
                    for path in (prompt_path, origin_path, receipt_path)
                    if path.is_file()
                ],
            )

        node_inputs = {
            "segment_index": index,
            "prompt_source": (
                "h3_prompt_origin.txt" if origin_path.is_file() else "h3_prompt.txt"
            ),
            "duration_s": item.get("duration_s"),
            "muted_video": str(muted),
            # 蒸馏输入全文落 input.json 供审计（每 attempt 重复记录）。
            "distilled_prompt": distilled_prompt,
        }
        try:
            return self._node(
                log,
                "h3-context-ir",
                scope,
                node_inputs,
                [muted],
                _operation,
                output_media=lambda value: [
                    path
                    for path in (prompt_path, origin_path, receipt_path)
                    if path.is_file()
                ],
                retry_if=_retry,
            )
        except ProviderFailure as exc:
            if self._has_deterministic_client_error(exc.record) or (
                isinstance(exc.record, dict)
                and exc.record.get("outcome") == "rejected"
            ):
                raise
            return self._context_ir_fallback(log, scope, prompt_path, origin_path, receipt_path, exc)
        except (TypeError, KeyError, AttributeError):
            raise
        except Exception as exc:
            return self._context_ir_fallback(log, scope, prompt_path, origin_path, receipt_path, exc)

    @staticmethod
    def _text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _context_ir_fallback(
        self,
        log: ResearchLog,
        scope: str,
        prompt_path: Path,
        origin_path: Path,
        receipt_path: Path,
        exc: Exception,
    ) -> dict[str, Any]:
        """重试耗尽 fail-open：保留/恢复原始提示词，链路继续。

        $0.02 的可选增强失败不得打死已烧了几美元的项目；覆写已发生则从
        origin 备份逐字节恢复。
        """
        reason = str(exc)[:500]
        if origin_path.is_file():
            prompt_path.write_text(origin_path.read_text(encoding="utf-8"), encoding="utf-8")
        receipt_doc = {"status": "fallback_original", "reason": reason}
        try:
            receipt_path.write_text(
                json.dumps(receipt_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        log.event("context_ir_fallback", scope=scope, error=reason)
        return {"status": "fallback_original"}

    def _continue_after_prepared(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        spoken_lines: list[dict[str, Any]],
        replacement: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        request = self._validated_effective_request(request)
        source_language_code = self.store.read(project_id).get(
            "source_language_code"
        )
        visual_prompts = "\n\n".join(
            f"Segment {item['index']}: {item['visual_prompt']}"
            for item in prepared
        )
        target_language = request["dialogue"]["target_language"]
        if request["dialogue"]["mode"] == "generate":
            brief = request["dialogue"]["brief"]
            resolved_language = _resolve_dialogue_language(
                target_language, spoken_lines, brief, source_language_code,
            )
            self._progress(project_id, 43, "dialogue-generation")
            translated_lines = self._node(
                log,
                "deepseek-dialogue-generation",
                "project",
                self._dialogue_generation_payload(
                    brief, prepared, resolved_language,
                ),
                [],
                lambda: self._dialogue_generation_request(
                    brief, target_language, prepared, resolved_language,
                ),
                transform=lambda value: self._normalize_generated_dialogue(
                    value, prepared,
                ),
            )
        else:
            self._progress(project_id, 43, "translation")
            translated_lines = self._node(
                log,
                "deepseek-translation",
                "project",
                {
                    "target_language": target_language,
                    "source_lines": spoken_lines,
                    "visual_prompts": visual_prompts,
                },
                [],
                lambda: self._translation_request(
                    spoken_lines, target_language, visual_prompts,
                ),
                transform=lambda value: self._normalize_translation(
                    value, spoken_lines,
                ),
            )
        # 项目级台词行不钉行号：audio_analysis 行的对齐改由 fusion 侧按
        # 全局行窗中点落入段窗判定（spec 2026-09-28 §2 审查修正）。
        # ark-audio-analysis：翻译完成即跑（只依赖台词窗口 + source.wav），
        # 不排在 object-replace 之后空等视觉分析链路。
        self._ensure_ark_audio_analysis(
            project_id, project, translated_lines, log,
        )
        for item in prepared:
            item["dialogue"] = self._node(
                log,
                "dialogue-localize",
                self._segment_scope(int(item["index"])),
                {"translated_lines": translated_lines, "segment": item},
                [],
                lambda item=item: media.localize_dialogue(
                    translated_lines, item, segments=prepared,
                ),
                retry_if=lambda _exc: False,
            )

        index_request, index_images = self._project_index_request(
            project,
            prepared,
            self._generation_config(request),
            user_replacement_instruction=(request.get("replacement_guidance") or {}).get("instruction", ""),
        )
        index_request_path = project / "work" / "project_index_request.json"
        index_request_path.write_text(
            json.dumps(index_request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        identity_evidence = index_request.get("person_identity")
        reference_ids = [entry["reference_id"] for entry in index_request.get("user_reference_images", [])]
        normalize_index = lambda value: normalize_project_index(value, identity_evidence, reference_image_ids=reference_ids)
        index_schema = project_index_schema([
            segment["segment_index"] for segment in index_request["segments"]
        ], identity_evidence, reference_image_ids=reference_ids)
        self._progress(project_id, 50, "video-maker-project-index")
        skill = VIDEO_SKILL.read_text(encoding="utf-8")
        element_index = self._node(
            log,
            "video-maker-project-index",
            "project",
            index_request,
            index_images,
            lambda: self.deepseek.complete_structured(
                skill
                + (
                    "\n\n严格执行项目级索引阶段；像素帧是视觉事实的唯一依据；"
                    "segments[].visual_prompt 只用于汇总逐段已发现对象并检查跨段覆盖，"
                    "不得用它替代或推翻像素证据；最终填写本次注入的 Schema。"
                    "\n若有 person_identity：这是固定算法输出的人物身份辅助证据，"
                    "请按 observations 的原图 bbox、segment_index、frame_order 对应像素中的人物，"
                    "填写 canonical_id 和 identity_observation_ids；已确认同一 canonical_id"
                    "不得拆成不同身份。canonical_id 为空只表示不确定，不表示另一人。"
                    "不同 canonical_id 也不证明一定是不同人，仍可按视觉证据合并。"
                    "source_time_s 是原视频时间；intervals 是采样支持区间，不保证区间内每帧可见。"
                ),
                index_request,
                index_schema,
                index_images,
                validator=normalize_index,
            ),
            transform=normalize_index,
        )
        element_path = project / "work" / "element_index.json"
        element_path.write_text(
            json.dumps(element_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        public_segments = [self._public_segment(item) for item in prepared]
        self.store.update(
            project_id,
            segments=public_segments,
            element_index=element_index,
            translated_voice_lines=translated_lines,
        )

        prepared_references = prepared[0].get("user_reference_paths", []) if prepared else []
        user_reference = prepared_references[0] if prepared_references else (
            self._materialize_canonical_user_reference(
                project, replacement, log,
                replacement_paths=self._stored_replacement_paths(project, self.store.read(project_id)),
            )
            if replacement is not None
            else None
        )
        self._progress(
            project_id, 53, "object-replace", segments=public_segments,
        )
        (
            enhanced_element_index,
            object_replace,
        ) = self._run_object_replace(
            project,
            prepared,
            translated_lines,
            element_index,
            user_reference,
            request,
            log,
        )
        enhanced_path = project / "work" / "enhanced_element_index.json"
        enhanced_temporary = project / "work" / ".enhanced_element_index.json.tmp"
        try:
            enhanced_temporary.write_text(
                json.dumps(
                    enhanced_element_index,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            enhanced_temporary.chmod(0o600)
            os.replace(enhanced_temporary, enhanced_path)
        finally:
            try:
                enhanced_temporary.unlink()
            except FileNotFoundError:
                pass
        self.store.update(
            project_id,
            enhanced_element_index=enhanced_element_index,
            object_replace=object_replace,
            voice_references=[],
        )

        self._progress(
            project_id, 55, "seedream-reference-sheet", segments=public_segments,
        )
        replacement_sheet, replacement_sheet_path = self._run_replacement_sheet(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            object_replace,
            replacement,
            log,
            user_reference=user_reference,
        )

        self._continue_after_replacement_sheet(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            replacement,
            replacement_sheet,
            replacement_sheet_path,
            request,
            log,
        )

    def _continue_after_replacement_sheet(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        enhanced_element_index: dict[str, Any],
        replacement: Path | None,
        replacement_sheet: dict[str, Any],
        replacement_sheet_path: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        """Run only Fusion, H3, and local stitching after the sheet checkpoint."""

        public_segments = [self._public_segment(item) for item in prepared]

        # This is the public-input barrier.  No paid segment branch starts
        # before every visual prompt, translation, localization, the single
        # project index and project-wide replacement/voice plan have completed.
        self._progress(project_id, 56, "dialogue-tts", segments=public_segments)
        self._run_dialogue_audio(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            request["dialogue"]["target_language"],
            log,
        )
        self._progress(
            project_id, 57, "h3-max-r2v", segments=public_segments,
        )
        outputs = self._run_h3_segments(
            project_id,
            project,
            prepared,
            enhanced_element_index,
            replacement,
            replacement_sheet,
            replacement_sheet_path,
            request,
            log,
        )
        if outputs is None:
            # 提示词审核暂停：产物已冻结，confirm 后走 resume-picsart 继续。
            return

        final = project / "output" / "generated.mp4"
        join_modes = self._validated_join_modes(prepared)
        self._progress(
            project_id,
            95,
            "stitch",
            segments=[self._public_segment(item) for item in prepared],
        )
        self._node(
            log,
            "stitch",
            "project",
            {
                "segments": outputs,
                "target_durations": [item["duration_s"] for item in prepared],
                "join_modes": join_modes,
                "output": final,
            },
            outputs,
            lambda: media.concat(
                outputs,
                final,
                target_durations=[item["duration_s"] for item in prepared],
                join_modes=join_modes,
            ),
            lambda _value: [final],
        )
        self._replace_dialogue_audio_track(project, prepared, final, log)
        self._add_subtitles(project_id, project, final, request, log)
        # 终态总耗时（前置节点 + 最后一次 H3 生成）随成功落盘，列表接口直接读，
        # 避免每次列表刷新解析全部项目 trace；详情接口仍按 trace 实时推导。
        display_duration = display_duration_seconds(project)
        self.store.update(
            project_id,
            status="succeeded",
            has_video=True,
            output_path="output/generated.mp4",
            **(
                {"display_duration_s": round(display_duration, 3)}
                if display_duration is not None
                else {}
            ),
            project_progress={
                "status": "succeeded",
                "percent": 100,
                "stage": "completed",
            },
            generation={
                "status": "succeeded",
                "stage": "completed",
                "error": None,
            },
            segments=[
                {**self._public_segment(item), "status": "succeeded"}
                for item in prepared
            ],
            error=None,
        )
        log.event("pipeline_succeeded", output=final)
        self._prune_artifacts_after_success(project, log)

    def _prune_artifacts_after_success(
        self,
        project: Path,
        log: ResearchLog,
    ) -> None:
        """Lean-retention cleanup after the terminal success state is durable.

        Runs inline in the pipeline worker thread, which is already
        asynchronous to the API request; pruning failures are logged and can
        never fail the project (app/artifact_retention.py).
        """
        settings = getattr(self, "settings", None)
        if settings is None or settings.artifact_retention != "lean":
            return
        try:
            state = self.store.read(project.name)
        except Exception as exc:
            log.event("artifact_prune_failed", error=str(exc))
            return
        artifact_retention.prune_after_success(project, state=state, log=log)

    def _replace_dialogue_audio_track(
        self,
        project: Path,
        prepared: list[dict[str, Any]],
        final: Path,
        log: ResearchLog,
    ) -> None:
        """Overwrite generated audio with dialogue, or silence when absent.

        有 work/accompaniment.wav（vocal-separation 产物）时：有台词先把
        台词轨与伴奏 ducking 混音再覆盖，无台词直接用伴奏做成片音轨（保留
        原音效/BGM/环境音）；accompaniment 缺失（无音轨源视频、存量续跑
        项目）完全沿用原降级分支。
        """
        accompaniment = project / "work" / "accompaniment.wav"
        has_accompaniment = (
            accompaniment.is_file() and accompaniment.stat().st_size > 0
        )
        if not any(item.get("dialogue") for item in prepared):
            if has_accompaniment:
                full_track = accompaniment
                log.event(
                    "audio_replaced_with_accompaniment_no_dialogue",
                    output=final,
                    audio=full_track,
                )
            else:
                full_track = (
                    project / "work" / "dialogue-audio" / "no-dialogue-silence.wav"
                )
                duration_s = float(media.probe(final)["duration_s"])
                self._node(
                    log,
                    "create-silent-audio-track",
                    "project",
                    {"video": final, "duration_s": duration_s, "output": full_track},
                    [final],
                    lambda: media.silent_audio_track(duration_s, full_track),
                    lambda _value: [full_track],
                    retry_if=lambda _exc: False,
                )
                log.event(
                    "audio_muted_no_dialogue",
                    output=final,
                    audio=full_track,
                    duration_s=duration_s,
                )
        else:
            full_track = project / "work" / "dialogue-audio" / "full-track.wav"
            if full_track.is_file() and has_accompaniment:
                mixed_track = (
                    project / "work" / "dialogue-audio" / "mixed-track.wav"
                )
                self._node(
                    log,
                    "mix-final-audio",
                    "project",
                    {
                        "dialogue_track": full_track,
                        "accompaniment": accompaniment,
                        "output": mixed_track,
                    },
                    [full_track, accompaniment],
                    # ducking 终值显式传入（2026-09-26 POC 三参数组标定 +
                    # 2026-10-03 用户听感复标，改参须重新标定），不依赖
                    # media 层缺省。
                    lambda: media.mix_dialogue_with_accompaniment(
                        full_track, accompaniment, mixed_track,
                        duck_threshold=0.01,
                        duck_ratio=6.0,
                        duck_attack_ms=10,
                        duck_release_ms=300,
                        accompaniment_gain_db=-2.0,
                    ),
                    lambda _value: [mixed_track],
                    retry_if=lambda _exc: False,
                )
                full_track = mixed_track
        if not full_track.is_file():
            # Degrade: keep the stitched video with its generated audio
            # when dialogue was expected but its assembled TTS track vanished.
            log.event(
                "dialogue_track_missing_degraded",
                output=final,
                impact=(
                    "assembled TTS dialogue track is missing; the stitched "
                    "video keeps its generated audio"
                ),
            )
            return
        self._node(
            log,
            "replace-audio-track",
            "project",
            {"video": final, "audio": full_track, "output": final},
            [final, full_track],
            # 整体响度终值显式传入（2026-10-03 实测 -27.5 LUFS→目标
            # ≈-17.5 LUFS，+10dB≈人类听觉一个量级/响度翻倍，alimiter
            # 防爆音；改参须重新实测），不依赖 media 层缺省。
            lambda: media.replace_audio_track(
                final, full_track, final, gain_db=10.0,
            ),
            lambda _value: [final],
            retry_if=lambda _exc: False,
        )

    def _add_subtitles(
        self,
        project_id: str,
        project: Path,
        final: Path,
        request: dict[str, Any],
        log: ResearchLog,
    ) -> None:
        """Burn the frozen TTS dialogue only after the final audio replacement."""
        if not request.get("add_subtitles", False):
            return
        self._progress(project_id, 98, "subtitles")
        lines = self.store.read(project_id).get("translated_voice_lines")
        directory = project / "work" / "dialogue-audio"
        clean = final.with_name("generated-clean.mp4")
        rendered = final.with_name("generated-subtitled.mp4")

        def render() -> dict[str, Any]:
            if not isinstance(lines, list) or any(
                not isinstance(line, dict) for line in lines
            ):
                raise ValueError("字幕需要有效的已冻结台词 translated_voice_lines")
            if not lines:
                log.event("subtitles_skipped", reason="no_dialogue")
                return {"status": "skipped", "reason": "no_dialogue"}
            duration_s = float(media.probe(final)["duration_s"])
            cue_lines: list[dict[str, Any]] = []
            tracks: list[Path] = []
            for index, line in enumerate(lines, 1):
                try:
                    start_s = float(line["start_s"])
                    end_s = float(line["end_s"])
                except (KeyError, TypeError, ValueError):
                    # Keep schema failures on the existing validation path.
                    start_s = end_s = float("nan")
                if (
                    math.isfinite(start_s)
                    and math.isfinite(end_s)
                    and 0 <= start_s < end_s
                    and start_s >= duration_s
                ):
                    # ASR can emit a final row just past the video timeline;
                    # it has no visible or audible window, so omit it together
                    # with the TTS file intentionally not produced upstream.
                    continue
                cue_lines.append(line)
                tracks.append(directory / f"line-{index:04d}.wav")
            if not cue_lines:
                log.event("subtitles_skipped", reason="outside_video_timeline")
                return {"status": "skipped", "reason": "outside_video_timeline"}
            for audio in [directory / "full-track.wav", *tracks]:
                if not audio.is_file() or audio.stat().st_size == 0:
                    raise ValueError(
                        f"字幕无法按实际配音对齐：缺少 TTS 音频 {audio.name}；"
                        "请恢复原配音产物后续跑"
                    )
            cues = subtitles.build_subtitle_cues(
                cue_lines, tracks, duration_s,
            )
            # Every resume re-stitches provider segments before entering here;
            # retain the clean, dubbed result and atomically publish only success.
            shutil.copyfile(final, clean)
            result = subtitles.burn_subtitles(clean, cues, rendered)
            os.replace(rendered, final)
            return {**result, "cues": cues}

        self._node(
            log, "subtitles", "project",
            {"voice_lines": lines, "video": final, "output": final},
            [], render, lambda _value: [final],
            retry_if=lambda _exc: False,
        )

    @staticmethod
    def _validated_stored_segments(
        state: dict[str, Any],
        *,
        missing_message: str,
    ) -> list[dict[str, Any]]:
        raw_segments = state.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(missing_message)
        segments: list[dict[str, Any]] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                raise TypeError("stored segment must be an object")
            index = raw.get("index")
            duration = raw.get("duration_s")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) * 24 < 1
            ):
                raise ValueError("stored segment metadata is invalid")
            join_mode = raw.get("join_mode")
            if join_mode not in {"hard_cut", "continue"}:
                raise ValueError("stored segment join mode is invalid")
            segments.append(dict(raw))
        segments.sort(key=lambda item: int(item["index"]))
        return segments

    def _load_pre_picsart_resume(
        self,
        project_id: str,
    ) -> tuple[
        Path,
        Path,
        Path | None,
        dict[str, Any],
        list[dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        list[dict[str, Any]],
        dict[str, Any] | None,
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        status = state.get("status")
        stage = (state.get("project_progress") or {}).get("stage")
        if status != "failed" and not (
            status == "queued" and stage == "pre_picsart_resume_queued"
        ):
            raise ValueError("only failed projects can resume preparation")
        logs = project / "logs"
        if (
            state.get("picsart_submission_receipts") is not None
            or any(path.is_dir() for path in logs.glob("picsart*-submit"))
        ):
            raise ValueError("project has Picsart submission evidence")
        if state.get("h3_prepared_artifacts") is not None:
            raise ValueError(
                "project already has frozen H3 artifacts; use Picsart resume"
            )
        source, replacement = self._stored_input_paths(project, state)
        if state.get("replacement_sheet") is None and (
            not isinstance(state.get("segments"), list) or not state["segments"]
        ):
            # Died before segment planning finished (e.g. transient ENOSPC in
            # whisper/mediakit): resume the pre-segmentation phase, reusing
            # every node output already on disk, instead of blocking.
            return (
                project,
                source,
                replacement,
                effective_request,
                [],
                {},
                [],
                {"early_resume": True},
            )
        raw_segments = self._validated_stored_segments(
            state, missing_message="stored segments are missing",
        )
        frozen_sheet = state.get("replacement_sheet")
        if frozen_sheet is not None:
            if not isinstance(frozen_sheet, dict):
                raise ValueError("stored replacement sheet state is invalid")
            sheet_path = self._validated_stored_replacement_sheet(
                project,
                state,
                allow_prepared=True,
            )
            sheet_status = frozen_sheet.get("status")
            recovery_mode = (
                "reuse"
                if sheet_status in {"not_needed", "succeeded"}
                else self._validated_prepared_seedream_attempt(
                    project, frozen_sheet,
                )
            )
            self._stored_voice_references(project, state)
            prepared_segments: list[dict[str, Any]] = []
            for raw in raw_segments:
                visual_prompt = raw.get("visual_prompt")
                dialogue = raw.get("dialogue")
                if not isinstance(visual_prompt, str) or not isinstance(
                    dialogue, list,
                ) or any(not isinstance(line, dict) for line in dialogue):
                    raise ValueError(
                        "stored replacement sheet segment content is invalid"
                    )
                index = int(raw["index"])
                prepared_segments.append({
                    **raw,
                    "source": (
                        project / "work" / "segments"
                        / f"{index:04d}" / "source.mp4"
                    ),
                    "status": "running",
                })
            enhanced_element_index = state.get("enhanced_element_index")
            if not isinstance(enhanced_element_index, dict):
                raise ValueError("stored enhanced element index is invalid")
            return (
                project,
                source,
                replacement,
                effective_request,
                prepared_segments,
                {},
                [],
                {
                    "replacement_sheet": frozen_sheet,
                    "replacement_sheet_path": sheet_path,
                    "recovery_mode": recovery_mode,
                    "enhanced_element_index": enhanced_element_index,
                },
            )
        segments: list[dict[str, Any]] = []
        selections: dict[int, list[dict[str, Any]]] = {}
        for raw in raw_segments:
            segment = dict(raw)
            index = segment["index"]
            scope = self._segment_scope(index)
            directory = project / "work" / "segments" / f"{index:04d}"
            selection_path = directory / "keyframe_selection.json"
            if not (
                selection_path.is_file() and selection_path.stat().st_size > 0
            ):
                raise FileNotFoundError("reusable keyframe selection is missing")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if not isinstance(selection, list):
                raise ValueError("reusable keyframe selection is invalid")
            segment_source_path = directory / "source.mp4"
            if not (
                segment_source_path.is_file()
                and segment_source_path.stat().st_size > 0
            ):
                raise FileNotFoundError("reusable segment source is missing")
            if not isinstance(
                self._completed_node_value(
                    project, "segment-frames-4fps", scope,
                ),
                dict,
            ):
                raise TypeError("reusable segment manifest is invalid")
            segments.append(segment)
            selections[index] = selection
        spoken_lines = state.get("voice_lines") or []
        if not isinstance(spoken_lines, list) or not all(
            isinstance(line, dict) for line in spoken_lines
        ):
            raise TypeError("stored spoken dialogue is invalid")
        return (
            project,
            source,
            replacement,
            effective_request,
            segments,
            selections,
            [dict(line) for line in spoken_lines],
            None,
        )

    def validate_pre_picsart_resume(self, project_id: str) -> None:
        self._load_pre_picsart_resume(project_id)

    def resume_pre_picsart(self, project_id: str) -> None:
        (
            project,
            source,
            replacement,
            request,
            segments,
            selections,
            spoken_lines,
            checkpoint,
        ) = self._load_pre_picsart_resume(project_id)
        log = ResearchLog(project)
        log.event("pre_picsart_resume_started")
        early_resume = checkpoint is not None and checkpoint.get("early_resume")
        try:
            if checkpoint is not None and not early_resume:
                self._progress(
                    project_id,
                    55,
                    "seedream-reference-sheet-resume",
                    segments=[self._public_segment(item) for item in segments],
                )
                replacement_sheet, replacement_sheet_path = (
                    self._resume_replacement_sheet(
                        project_id,
                        project,
                        checkpoint["replacement_sheet"],
                        checkpoint["recovery_mode"],
                        log,
                    )
                )
                self._continue_after_replacement_sheet(
                    project_id,
                    project,
                    segments,
                    checkpoint["enhanced_element_index"],
                    replacement,
                    replacement_sheet,
                    replacement_sheet_path,
                    request,
                    log,
                )
                log.event("pre_picsart_resume_succeeded")
                return
            if early_resume:
                source, segments, selections, spoken_lines = (
                    self._run_pre_segmentation(
                        project_id, project, source, request, log, resume=True,
                    )
                )
            self._progress(
                project_id, 24, "video-maker-resume", segments=segments,
            )
            prepared = self._prepare_project_segments(
                project_id, project, source, segments, selections, request, log,
                resume=not early_resume,
            )
            self._continue_after_prepared(
                project_id,
                project,
                prepared,
                spoken_lines,
                replacement,
                request,
                log,
            )
            log.event("pre_picsart_resume_succeeded")
        except Exception as exc:
            log.event("pre_picsart_resume_failed", error=str(exc))
            self._persist_failure(
                project_id,
                log,
                stage="pre_picsart_resume_failed",
                code="pre_picsart_resume_failed",
                message="前置产物续跑失败",
                min_percent=24,
            )

    @staticmethod
    def _resume_prefix_key(prefix: str) -> int:
        if prefix == "picsart-resume":
            return 1
        marker = "picsart-resume-"
        if prefix.startswith(marker):
            try:
                return int(prefix[len(marker):])
            except ValueError:
                return 0
        return 0

    def _resume_prefixes(self, project: Path) -> list[str]:
        prefixes: set[str] = set()
        for suffix in ("-submit", "-poll"):
            prefixes.update(
                path.name.removesuffix(suffix)
                for path in (project / "logs").glob(
                    f"picsart-resume*{suffix}"
                )
                if path.is_dir()
            )
        return sorted(prefixes, key=self._resume_prefix_key, reverse=True)

    def _next_resume_identity(
        self,
        project: Path,
        state: dict[str, Any],
    ) -> tuple[str, str]:
        prefixes = self._resume_prefixes(project)
        receipts = self._validated_submission_receipts(state)
        if receipts is not None:
            prefixes.extend(
                receipt["run_identity"]
                for records in receipts["segments"].values()
                for receipt in records
                if receipt["run_identity"].startswith("picsart-resume")
            )
        number = max((self._resume_prefix_key(value) for value in prefixes), default=0) + 1
        if number == 1:
            return "picsart-resume", "-resume"
        return f"picsart-resume-{number:02d}", f"-resume-{number:02d}"

    @classmethod
    def _validated_submission_receipts(
        cls,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw = state.get("picsart_submission_receipts")
        if raw is None:
            return None
        version = raw.get("version") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "segments"}
            or isinstance(version, bool)
            or version != PICSART_RECEIPT_VERSION
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored Picsart receipt ledger is invalid")
        prepared_manifest = state.get("h3_prepared_artifacts")
        prepared_version = (
            prepared_manifest.get("version")
            if isinstance(prepared_manifest, dict)
            else None
        )
        if prepared_version != H3_PREPARED_ARTIFACTS_VERSION:
            raise ValueError(
                "Picsart receipt v2 requires the current H3 preparation manifest"
            )
        preparation_sha256 = cls._json_sha256(prepared_manifest)
        normalized: dict[str, list[dict[str, Any]]] = {}
        allowed_statuses = {
            "submission_started", "accepted", "unknown", "rejected",
            "provider_failed",
        }
        for segment_key, records in raw["segments"].items():
            if (
                not isinstance(segment_key, str)
                or not segment_key.isdecimal()
                or str(int(segment_key)) != segment_key
                or int(segment_key) < 1
                or not isinstance(records, list)
                or not records
            ):
                raise ValueError("stored Picsart segment receipts are invalid")
            seen_runs: set[str] = set()
            copied: list[dict[str, Any]] = []
            for offset, receipt in enumerate(records):
                expected_receipt_keys = {
                    "segment_index", "run_identity", "request_sha256",
                    "preparation_sha256", "status", "task_id",
                }
                if (
                    not isinstance(receipt, dict)
                    or (
                        set(receipt) != expected_receipt_keys
                        and set(receipt)
                        != expected_receipt_keys | {"api_key_role"}
                    )
                    or (
                        receipt.get("api_key_role") is not None
                        and receipt.get("api_key_role")
                        not in {"primary", "backup"}
                    )
                ):
                    raise ValueError("stored Picsart receipt is invalid")
                run_identity = receipt["run_identity"]
                digest = receipt["request_sha256"]
                status = receipt["status"]
                task_id = receipt["task_id"]
                if (
                    receipt["segment_index"] != int(segment_key)
                    or isinstance(receipt["segment_index"], bool)
                    or not isinstance(run_identity, str)
                    or not run_identity
                    or run_identity != run_identity.strip()
                    or run_identity in seen_runs
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or status not in allowed_statuses
                    or (
                        status in {"accepted", "provider_failed"}
                        and (not isinstance(task_id, str) or not task_id.strip())
                    )
                    or (
                        status not in {"accepted", "provider_failed"}
                        and task_id is not None
                    )
                    or (
                        offset < len(records) - 1
                        and status not in {"rejected", "provider_failed"}
                    )
                    or receipt.get("preparation_sha256")
                    != preparation_sha256
                ):
                    raise ValueError("stored Picsart receipt fields are invalid")
                seen_runs.add(run_identity)
                copied.append(dict(receipt))
            normalized[segment_key] = copied
        return {"version": version, "segments": normalized}

    def _begin_submission_batch(
        self,
        requests: list[PreparedPicsartRequest],
    ) -> None:
        pending = [item for item in requests if item.existing_task_id is None]
        if not pending:
            return
        project_ids = {item.project_id for item in pending}
        segment_indices = [item.segment_index for item in pending]
        if len(project_ids) != 1 or len(segment_indices) != len(set(segment_indices)):
            raise ValueError("prepared Picsart submission batch is invalid")
        project_id = pending[0].project_id
        with self._picsart_receipt_lock:
            state = self.store.read(project_id)
            manifest = state.get("h3_prepared_artifacts")
            manifest_version = (
                manifest.get("version") if isinstance(manifest, dict) else None
            )
            if manifest_version != H3_PREPARED_ARTIFACTS_VERSION:
                raise ValueError(
                    "Picsart submission requires the current H3 preparation manifest"
                )
            ledger_version = PICSART_RECEIPT_VERSION
            preparation_sha256 = self._json_sha256(manifest)
            if any(
                prepared.preparation_sha256 != preparation_sha256
                for prepared in pending
            ):
                raise ValueError(
                    "prepared Picsart requests do not match H3 preparation"
                )
            ledger = self._validated_submission_receipts(state) or {
                "version": ledger_version,
                "segments": {},
            }
            if ledger["version"] != ledger_version:
                raise ValueError(
                    "Picsart receipt ledger does not match H3 preparation"
                )
            for prepared in sorted(pending, key=lambda item: item.segment_index):
                key = str(prepared.segment_index)
                records = ledger["segments"].setdefault(key, [])
                if any(
                    receipt["run_identity"] == prepared.run_identity
                    for receipt in records
                ):
                    raise ValueError("Picsart run identity was already used")
                if records and records[-1]["status"] not in {
                    "rejected", "provider_failed",
                }:
                    raise ValueError(
                        "Picsart submission is not proven safe to retry"
                    )
                receipt = {
                    "segment_index": prepared.segment_index,
                    "run_identity": prepared.run_identity,
                    "request_sha256": prepared.request_sha256,
                    "status": "submission_started",
                    "task_id": None,
                }
                receipt["preparation_sha256"] = preparation_sha256
                records.append(receipt)
            self.store.update(
                project_id,
                picsart_submission_receipts=ledger,
            )

    def _transition_submission_receipt(
        self,
        prepared: PreparedPicsartRequest,
        *,
        status: str,
        task_id: str | None = None,
        api_key_role: str | None = None,
    ) -> None:
        if status not in {"accepted", "unknown", "rejected"}:
            raise ValueError("invalid Picsart submission receipt transition")
        if status == "accepted":
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("accepted Picsart receipt requires a task id")
        elif task_id is not None:
            raise ValueError("non-accepted Picsart receipt cannot have a task id")
        with self._picsart_receipt_lock:
            state = self.store.read(prepared.project_id)
            ledger = self._validated_submission_receipts(state)
            if ledger is None:
                raise ValueError("Picsart submission receipt ledger is missing")
            records = ledger["segments"].get(str(prepared.segment_index)) or []
            matches = [
                receipt for receipt in records
                if receipt["run_identity"] == prepared.run_identity
            ]
            if len(matches) != 1:
                raise ValueError("Picsart submission receipt identity is missing")
            receipt = matches[0]
            if (
                receipt["request_sha256"] != prepared.request_sha256
                or (
                    ledger["version"] == PICSART_RECEIPT_VERSION
                    and receipt.get("preparation_sha256")
                    != prepared.preparation_sha256
                )
                or receipt["status"] != "submission_started"
            ):
                raise ValueError("Picsart submission receipt transition is unsafe")
            receipt["status"] = status
            receipt["task_id"] = task_id
            if status == "accepted":
                # 任务账户隔离：task 由哪把 key 创建（402 failover 证据见
                # PicsartClient.submit 的 api_key_role），poll/result/
                # download 都按回执 role 选 key；旧回执无此字段按 primary。
                receipt["api_key_role"] = (
                    api_key_role
                    if api_key_role in {"primary", "backup"}
                    else "primary"
                )
            self.store.update(
                prepared.project_id,
                picsart_submission_receipts=ledger,
            )

    def _mark_submission_task_failed(
        self,
        project_id: str,
        segment_index: int,
        task_id: str,
    ) -> None:
        with self._picsart_receipt_lock:
            state = self.store.read(project_id)
            ledger = self._validated_submission_receipts(state)
            if ledger is None:
                raise ValueError("Picsart submission receipt ledger is missing")
            records = ledger["segments"].get(str(segment_index)) or []
            if not records:
                raise ValueError("accepted Picsart task receipt is missing")
            matches = [
                receipt for receipt in records
                if receipt["status"] == "accepted"
                and receipt["task_id"] == task_id
            ]
            if len(matches) != 1:
                raise ValueError("accepted Picsart task receipt is missing")
            matches[0]["status"] = "provider_failed"
            self.store.update(
                project_id,
                picsart_submission_receipts=ledger,
            )

    @classmethod
    def _state_submission_task(
        cls,
        state: dict[str, Any],
        segment_index: int,
    ) -> tuple[bool, str | None]:
        ledger = cls._validated_submission_receipts(state)
        if ledger is None:
            return False, None
        records = ledger["segments"].get(str(segment_index)) or []
        if not records:
            raise ValueError("Picsart segment receipt is missing")
        receipt = records[-1]
        if receipt["status"] in {"submission_started", "unknown"}:
            raise ValueError(
                "Picsart submission outcome is unknown; reconcile it before resume"
            )
        if receipt["status"] == "accepted":
            return True, receipt["task_id"]
        return True, None

    @classmethod
    def _state_submission_api_key_role(
        cls,
        state: dict[str, Any],
        segment_index: int,
    ) -> str:
        """该段最新回执的 api_key_role（402 failover 的账户隔离依据）；
        无账本/无记录/无字段（旧回执）一律按 primary。"""
        ledger = cls._validated_submission_receipts(state)
        if ledger is None:
            return "primary"
        records = ledger["segments"].get(str(segment_index)) or []
        if not records:
            return "primary"
        role = records[-1].get("api_key_role")
        return role if role in {"primary", "backup"} else "primary"

    def _picsart_receipt_prefixes(self, project: Path) -> list[str]:
        prefixes = self._resume_prefixes(project)
        if (project / "logs" / "picsart-submit").is_dir():
            prefixes.append("picsart")
        return prefixes

    @staticmethod
    def _poll_record_is_terminal_failure(record: dict[str, Any]) -> bool:
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict):
                continue
            response = exchange.get("response")
            if not isinstance(response, dict):
                continue
            status_code = response.get("status_code")
            body = response.get("body")
            if status_code == 200:
                envelope = body.get("response") if isinstance(body, dict) else None
                if (
                    isinstance(envelope, dict)
                    and str(envelope.get("status") or "").upper() == "FAILED"
                ):
                    return True
                continue
            # The workflow result endpoint reports completed task failure as
            # this exact 422 or 503 envelope.  In particular, downstream
            # MiniMax unavailability is observed as 503 with this reason;
            # a bare 503 remains a transient failure, not a terminal task.
            if (
                status_code in (422, 503)
                and isinstance(body, dict)
                and str(body.get("status") or "").upper() == "ERROR"
                and body.get("reason") == f"{PicsartClient.WORKFLOW}_failed"
            ):
                return True
            # 错误体优先于状态码：5xx 也可能携带任务终态失败语义
            # （2026-09-29 项目 0eafd5f5 事故：result 端点以 502/504 返回
            # {"status":"error","reason":"file_copy_failed",...}，被误判为
            # 可重试传输错误无限退避）。带 status:"error" 信封的 5xx 是终态
            # 失败，进 R4 自动重提分类；无此信封的裸 5xx 仍是传输错误，
            # 走 PICSART_POLL_TRANSPORT_FAILURE_LIMIT 封顶。
            # 503 例外保留上方既有语义：它是典型的网关暂态状态，只有
            # workflow_failed 信封才算终态，其他 error reason 仍为传输错误。
            if (
                isinstance(status_code, int)
                and not isinstance(status_code, bool)
                and status_code >= 500
                and status_code != 503
                and isinstance(body, dict)
                and str(body.get("status") or "").upper() == "ERROR"
            ):
                return True
        return False

    @staticmethod
    def _poll_failure_provider_reason(record: dict[str, Any]) -> str | None:
        """终态 5xx 错误体里的供应商 reason（如 file_copy_failed），供 R4
        重提事件记录；取不到返回 None。"""
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict):
                continue
            response = exchange.get("response")
            if not isinstance(response, dict):
                continue
            body = response.get("body")
            if (
                isinstance(body, dict)
                and str(body.get("status") or "").upper() == "ERROR"
            ):
                reason = body.get("reason")
                if isinstance(reason, str) and reason.strip():
                    return reason.strip()[:64]
        return None

    @staticmethod
    def _poll_failure_is_request_validation(record: dict[str, Any]) -> bool:
        """Recognize explicit terminal validation of a frozen request field.

        A generic ValidationError can also mean transient asset download
        failure. Only the provider's body-property validation envelope is
        classified as request validation (content moderation of the frozen
        input). The moderation verdict is probabilistic — reposting the
        unchanged inputs can pass (projects f0997e14/d7eb0406, see the
        resubmit path in _generate_picsart_segment) — so this classification
        feeds the capped auto-resubmit path, not fail-closed.
        """
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict) or not Pipeline._poll_record_is_terminal_failure({"polls": [exchange]}):
                continue
            body = exchange["response"]["body"]
            envelope = body.get("response")
            details = envelope if isinstance(envelope, dict) else body
            error = details.get("error")
            values = [details.get("message"), error.get("message") if isinstance(error, dict) else error]
            for value in values:
                if not isinstance(value, str):
                    continue
                message = value.lower()
                if (
                    "validationerror:" in message
                    and "[property:" in message
                    and ("'body." in message or '"body.' in message)
                    and "message:" in message
                    and "failed to download" not in message
                    and "unreachable" not in message
                ):
                    return True
        return False

    @staticmethod
    def _poll_failure_is_asset_fetch(record: dict[str, Any]) -> bool:
        """Classify asset-fetch failures after excluding request validation."""

        texts: list[str] = []
        for exchange in record.get("polls") or []:
            if not isinstance(exchange, dict):
                continue
            response = exchange.get("response")
            if not isinstance(response, dict):
                continue
            body = response.get("body")
            if not isinstance(body, dict):
                continue
            envelope = body.get("response")
            candidates = envelope if isinstance(envelope, dict) else body
            error = candidates.get("error")
            if isinstance(error, dict):
                error = error.get("message")
            for value in (error, candidates.get("message"), candidates.get("reason")):
                if isinstance(value, str):
                    texts.append(value)
        haystack = "\n".join(texts).lower()
        return any(
            marker in haystack for marker in _PICSART_FETCH_FAILURE_MARKERS
        )

    def _terminal_resume_task_ids(
        self, project: Path, scope: str,
    ) -> set[str]:
        terminal_task_ids: set[str] = set()
        for prefix in self._picsart_receipt_prefixes(project):
            poll_root = project / "logs" / f"{prefix}-poll" / scope
            for poll_output in poll_root.glob("attempt-*/output.json"):
                try:
                    poll_logged = json.loads(
                        poll_output.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                poll_record = (
                    poll_logged.get("provider_record")
                    if isinstance(poll_logged, dict)
                    else None
                )
                task_id = (
                    poll_record.get("task_id")
                    if isinstance(poll_record, dict)
                    else None
                )
                if (
                    isinstance(task_id, str)
                    and task_id
                    and self._poll_record_is_terminal_failure(poll_record)
                ):
                    terminal_task_ids.add(task_id)
        return terminal_task_ids

    def _completed_resume_output(
        self, project: Path, segment_dir: Path, scope: str,
    ) -> Path | None:
        for prefix in self._picsart_receipt_prefixes(project):
            number = self._resume_prefix_key(prefix)
            suffix = (
                "" if prefix == "picsart"
                else "-resume" if number == 1
                else f"-resume-{number:02d}"
            )
            output = segment_dir / f"generated{suffix}.mp4"
            trim_root = (
                project / "logs"
                / (
                    "trim-provider-output-24fps"
                    if prefix == "picsart"
                    else f"{prefix}-trim-provider-output-24fps"
                )
                / scope
            )
            if output.is_file() and any(trim_root.glob("attempt-*/output.json")):
                if output.stat().st_size < 1:
                    raise ValueError("reusable generated segment is empty")
                info = media.probe(output)
                duration = info.get("duration_s") if isinstance(info, dict) else None
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0
                ):
                    raise ValueError("reusable generated segment is invalid")
                return output
        return None

    @staticmethod
    def _validate_artifact_receipt(
        project: Path,
        receipt: object,
        expected_path: str,
    ) -> Path:
        if not isinstance(receipt, dict) or set(receipt) != {
            "path", "size_bytes", "sha256",
        }:
            raise ValueError("stored H3 artifact receipt is invalid")
        size_bytes = receipt.get("size_bytes")
        digest = receipt.get("sha256")
        if (
            receipt.get("path") != expected_path
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("stored H3 artifact receipt fields are invalid")
        candidate = project / expected_path
        resolved = candidate.resolve(strict=True)
        project_root = project.resolve(strict=True)
        if (
            project_root not in resolved.parents
            or candidate.is_symlink()
            or not resolved.is_file()
            or resolved.stat().st_size != size_bytes
            or Pipeline._sha256(resolved) != digest
        ):
            raise ValueError("stored H3 artifact receipt does not match")
        return resolved

    @classmethod
    def _validated_h3_prepared_artifacts(
        cls,
        project: Path,
        state: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]] | None:
        raw = state.get("h3_prepared_artifacts")
        if raw is None:
            return None
        version = raw.get("version") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "bindings", "segments"}
            or isinstance(version, bool)
            or version != H3_PREPARED_ARTIFACTS_VERSION
            or not isinstance(raw.get("bindings"), dict)
            or not isinstance(raw.get("segments"), dict)
        ):
            raise ValueError("stored H3 preparation manifest is invalid")
        bindings = raw["bindings"]
        replacement_paths = cls._replacement_image_paths(project, state)
        expected_binding_keys = {
            "enhanced_element_index_sha256", "voice_references_sha256",
            "object_replace_input_sha256", "replacement_image",
            "output_request_sha256",
        }
        expected_binding_keys.add("replacement_sheet_sha256")
        if len(replacement_paths) > 1:
            expected_binding_keys.add("replacement_images")
        if set(bindings) != expected_binding_keys:
            raise ValueError("stored H3 preparation bindings are invalid")
        enhanced_index = state.get("enhanced_element_index")
        voice_references = state.get("voice_references")
        object_replace = state.get("object_replace")
        effective_request = cls._validated_effective_request(
            state.get("effective_request")
        )
        output_request = effective_request["output"]
        if (
            not isinstance(enhanced_index, dict)
            or not isinstance(voice_references, list)
            or not isinstance(object_replace, dict)
            or bindings.get("enhanced_element_index_sha256")
            != cls._json_sha256(enhanced_index)
            or bindings.get("voice_references_sha256")
            != cls._json_sha256(voice_references)
            or bindings.get("object_replace_input_sha256")
            != object_replace.get("input_sha256")
            or bindings.get("output_request_sha256")
            != cls._json_sha256(output_request)
        ):
            raise ValueError("stored H3 preparation bindings do not match")
        if object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION:
            raise ValueError(
                "stored H3 preparation Object Replace version does not match"
            )
        replacement_receipt = bindings.get("replacement_image")
        has_replacement = bool(replacement_paths)
        if has_replacement != (replacement_receipt is not None):
            raise ValueError("stored H3 preparation replacement binding does not match")
        if replacement_receipt is not None:
            cls._validate_artifact_receipt(
                project, replacement_receipt, "inputs/replacement-image",
            )
        if len(replacement_paths) > 1:
            replacement_receipts = bindings.get("replacement_images")
            expected_replacement_receipts = [
                cls._project_artifact_receipt(project, path, name)
                for path, name in zip(
                    replacement_paths,
                    replacement_input_paths(len(replacement_paths)),
                )
            ]
            if (
                replacement_receipts != expected_replacement_receipts
                or replacement_receipt != expected_replacement_receipts[0]
            ):
                raise ValueError(
                    "stored H3 preparation replacement bindings do not match"
                )
        replacement_sheet = state.get("replacement_sheet")
        if (
            not isinstance(replacement_sheet, dict)
            or bindings.get("replacement_sheet_sha256")
            != cls._json_sha256(replacement_sheet)
        ):
            raise ValueError(
                "stored H3 preparation replacement sheet binding does not match"
            )
        if replacement_sheet.get("version") != REPLACEMENT_SHEET_VERSION:
            raise ValueError(
                "stored H3 preparation replacement sheet version does not match"
            )
        cls._validated_stored_replacement_sheet(project, state)
        expected_keys = {str(item["index"]) for item in segments}
        if set(raw["segments"]) != expected_keys:
            raise ValueError("stored H3 preparation segments do not match")
        normalized: dict[str, dict[str, Any]] = {}
        for item in segments:
            index = item["index"]
            entry = raw["segments"].get(str(index))
            if not isinstance(entry, dict) or set(entry) != {
                "segment_index", "source_video", "muted_video", "prompt",
                "dialogue_track",
                "fusion_input_sha256", "segment_metadata_sha256",
            }:
                raise ValueError("stored H3 preparation segment is invalid")
            fusion_digest = entry.get("fusion_input_sha256")
            metadata_digest = entry.get("segment_metadata_sha256")
            expected_metadata_digest = cls._json_sha256({
                "index": item.get("index"),
                "start_s": item.get("start_s"),
                "end_s": item.get("end_s"),
                "duration_s": item.get("duration_s"),
                "join_mode": item["join_mode"],
            })
            if (
                entry.get("segment_index") != index
                or isinstance(entry.get("segment_index"), bool)
                or not isinstance(fusion_digest, str)
                or len(fusion_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in fusion_digest
                )
                or metadata_digest != expected_metadata_digest
            ):
                raise ValueError("stored H3 preparation segment fields are invalid")
            base = f"work/segments/{index:04d}"
            cls._validate_artifact_receipt(
                project, entry["source_video"], f"{base}/source.mp4",
            )
            muted_receipt = entry["muted_video"]
            muted_path = (
                muted_receipt.get("path")
                if isinstance(muted_receipt, dict)
                else None
            )
            if not isinstance(muted_path, str):
                raise ValueError("stored H3 artifact receipt fields are invalid")
            cls._validate_artifact_receipt(project, muted_receipt, muted_path)
            try:
                cls._validate_artifact_receipt(
                    project, entry["prompt"], f"{base}/h3_prompt.txt",
                )
            except ValueError:
                # 冻结的是 IR 前原文（2026-09-30 顺序调整）：IR 成功后
                # h3_prompt.txt 是增强文本，走 IR 链校验（origin==冻结 &&
                # 回执.origin==冻结 && 当前==optimized）。IR 未跑/禁用/
                # fallback 与旧顺序项目走上方原文校验即过。
                prompt_digest = (
                    entry["prompt"].get("sha256")
                    if isinstance(entry["prompt"], dict) else None
                )
                if not (
                    isinstance(prompt_digest, str)
                    and len(prompt_digest) == 64
                    and _context_ir_frozen_chain(
                        project / base, prompt_digest,
                    )
                ):
                    raise
            dialogue_track = entry["dialogue_track"]
            expects_dialogue_track = bool(
                item.get("dialogue")
            ) and not cls._validated_dialogue_voice_over(state)
            if (dialogue_track is not None) != expects_dialogue_track:
                raise ValueError(
                    "stored H3 preparation dialogue track does not match"
                )
            if dialogue_track is not None:
                cls._validate_artifact_receipt(
                    project, dialogue_track, f"{base}/dialogue-track.wav",
                )
            normalized[str(index)] = dict(entry)
        return normalized

    @classmethod
    def _validated_stored_replacement_sheet(
        cls,
        project: Path,
        state: dict[str, Any],
        *,
        allow_prepared: bool = False,
    ) -> Path | None:
        """Validate the frozen project-wide Image N binding and its checkpoint."""

        raw = state.get("replacement_sheet")
        if raw is None:
            return None
        expected_keys = {
            "version", "status", "spec", "spec_sha256",
            "user_image_present", "h3_image_reference",
            "h3_reference_images", "upstream_bindings", "seedream", "output",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_keys
            or raw.get("version") != REPLACEMENT_SHEET_VERSION
            or isinstance(raw.get("version"), bool)
            or not isinstance(raw.get("user_image_present"), bool)
            or (raw["user_image_present"] and not cls._replacement_image_paths(project, state))
            or not isinstance(raw.get("spec"), dict)
            or not isinstance(state.get("enhanced_element_index"), dict)
            or not isinstance(raw.get("h3_reference_images"), list)
        ):
            raise ValueError("stored replacement sheet state is invalid")
        try:
            validate_replacement_sheet_spec(raw["spec"], allow_empty=True)
        except ValueError as exc:
            raise ValueError("stored replacement sheet state is invalid") from exc
        if (
            raw.get("spec_sha256") != raw["spec"].get("spec_sha256")
            or raw.get("spec_sha256")
            != replacement_sheet_spec_sha256(raw["spec"])
            or raw["spec"].get("enhanced_element_index_sha256")
            != cls._json_sha256(state["enhanced_element_index"])
        ):
            raise ValueError("stored replacement sheet state is invalid")

        stored_segments = cls._validated_stored_segments(
            state,
            missing_message="stored replacement sheet segments are missing",
        )
        prepared_for_binding = [
            {
                **segment,
                "source": (
                    project / "work" / "segments"
                    / f"{int(segment['index']):04d}" / "source.mp4"
                ),
            }
            for segment in stored_segments
        ]
        object_replace = state.get("object_replace")
        voice_references = state.get("voice_references")
        effective_request = cls._validated_effective_request(
            state.get("effective_request")
        )
        if (
            not isinstance(object_replace, dict)
            or object_replace.get("version") != OBJECT_REPLACE_OUTPUT_VERSION
            or not isinstance(voice_references, list)
        ):
            raise ValueError("stored replacement sheet upstream state is invalid")
        replacement_paths = cls._replacement_image_paths(project, state)
        replacement = replacement_paths[0] if replacement_paths else None
        expected_upstream_bindings = cls._replacement_sheet_upstream_bindings(
            project,
            prepared_for_binding,
            state["enhanced_element_index"],
            object_replace,
            voice_references,
            effective_request,
            replacement,
        )
        if raw.get("upstream_bindings") != expected_upstream_bindings:
            raise ValueError(
                "stored replacement sheet upstream bindings do not match"
            )

        expected_spec = build_replacement_sheet_spec(
            state["enhanced_element_index"],
        )
        if raw["spec"] != expected_spec:
            raise ValueError(
                "stored replacement sheet tiles do not match the global index"
            )

        status = raw.get("status")
        tiles = raw["spec"].get("tiles")
        if not isinstance(tiles, list):
            raise ValueError("stored replacement sheet tiles are invalid")
        if status == "not_needed":
            frozen_views = product_views.frozen_product_views(project)
            if raw["user_image_present"] and frozen_views is not None:
                # composed 模式：多图冻结合成为单一 provider 参考图。
                expected_references = [{
                    "image_reference": "Image 1",
                    "role": "user_replacement",
                    **cls._project_artifact_receipt(
                        project, project / product_views.REFERENCE,
                        product_views.REFERENCE,
                    ),
                }]
            else:
                expected_references = [{
                    "image_reference": f"Image {order}",
                    "role": "user_replacement",
                    **cls._project_artifact_receipt(
                        project, cls._canonical_user_reference_path(project, order),
                        cls._canonical_user_reference_path(project, order).relative_to(project).as_posix(),
                    ),
                } for order, _path in enumerate(replacement_paths, 1)] if raw["user_image_present"] else []
            if (
                tiles
                or raw.get("h3_image_reference") is not None
                or raw.get("seedream") is not None
                or raw.get("output") is not None
                or raw["h3_reference_images"] != expected_references
            ):
                raise ValueError("stored replacement sheet skip state is invalid")
            return None
        if status not in {"prepared", "succeeded"} or not tiles:
            raise ValueError("stored replacement sheet is not reusable")

        expected_references: list[dict[str, Any]] = []
        if raw["user_image_present"]:
            # composed 模式多图冻结合成为单一 provider 参考图（order=1 即合图
            # 路径）；direct 模式逐图直传，provider 图数 = 上传图数。
            frozen_views = product_views.frozen_product_views(project)
            provider_image_count = (
                1 if frozen_views is not None else len(replacement_paths)
            )
            for order in range(1, provider_image_count + 1):
                canonical_user = cls._canonical_user_reference_path(project, order)
                relative_path = canonical_user.relative_to(project).as_posix()
                canonical_receipt = cls._project_artifact_receipt(
                    project, canonical_user, relative_path,
                )
                canonical_bytes = canonical_user.read_bytes()
                if (
                    not canonical_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                    or len(canonical_bytes) < 9
                ):
                    raise ValueError("stored user reference is not PNG")
                expected_references.append({
                    "image_reference": f"Image {order + 1}",
                    "role": "user_replacement",
                    **canonical_receipt,
                })

        # The sheet is always H3 Image 1; the user image follows as Image 2.
        expected_image_reference = "Image 1"
        if raw.get("h3_image_reference") != expected_image_reference:
            raise ValueError("stored H3 image reference order is invalid")

        seedream = raw.get("seedream")
        if not isinstance(seedream, dict) or set(seedream) != {
            "model", "endpoint", "prompt", "spec", "inputs",
            "request_sha256", "receipt",
        }:
            raise ValueError("stored Seedream state is invalid")
        model = seedream.get("model")
        endpoint = seedream.get("endpoint")
        model_characters = frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        if (
            not isinstance(model, str)
            or not 1 <= len(model) <= 128
            or not model[0].isalnum()
            or not model.isascii()
            or any(character not in model_characters for character in model)
            or not isinstance(endpoint, str)
            or not endpoint.startswith("https://")
            or not endpoint.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in endpoint)
            or not isinstance(seedream.get("inputs"), list)
        ):
            raise ValueError("stored Seedream fields are invalid")
        prompt_path = cls._validate_artifact_receipt(
            project,
            seedream["prompt"],
            "work/global-reference-sheet/prompt.txt",
        )
        spec_path = cls._validate_artifact_receipt(
            project,
            seedream["spec"],
            "work/global-reference-sheet/spec.json",
        )
        try:
            persisted_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored replacement sheet spec is invalid") from exc
        if persisted_spec != raw["spec"]:
            raise ValueError("stored replacement sheet spec does not match")
        allowed_inputs = [
            "work/global-reference-sheet/canvas.png",
            *(
                [
                    cls._canonical_user_reference_path(
                        project, order,
                    ).relative_to(project).as_posix()
                    for order in range(1, provider_image_count + 1)
                ]
                if raw["user_image_present"]
                else []
            ),
        ]
        if len(seedream["inputs"]) != len(allowed_inputs):
            raise ValueError("stored Seedream input order is invalid")
        for receipt, relative_path in zip(seedream["inputs"], allowed_inputs):
            cls._validate_artifact_receipt(project, receipt, relative_path)
        prompt_sha256 = seedream["prompt"].get("sha256")
        input_sha256 = [
            receipt.get("sha256")
            if isinstance(receipt, dict)
            else None
            for receipt in seedream["inputs"]
        ]
        request_binding = {
            "version": 1,
            "endpoint": seedream["endpoint"],
            "model": seedream["model"],
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "size": SEEDREAM_OUTPUT_SIZE,
            "response_format": "b64_json",
            "watermark": False,
        }
        request_sha256 = seedream.get("request_sha256")
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("stored Seedream prompt is invalid") from exc
        if (
            not prompt_text.strip()
            or not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in request_sha256
            )
            or cls._json_sha256(request_binding) != request_sha256
        ):
            raise ValueError("stored Seedream request binding is invalid")

        if status == "prepared":
            if (
                not allow_prepared
                or raw.get("output") is not None
                or seedream.get("receipt") is not None
                or raw["h3_reference_images"] != expected_references
            ):
                raise ValueError("stored replacement sheet is not reusable")
            return None

        output = raw.get("output")
        output_path = cls._validate_artifact_receipt(
            project,
            output,
            "work/global-reference-sheet/reference-sheet.png",
        )
        output_bytes = output_path.read_bytes()
        try:
            output_media_type = _validated_image_media_type(output_bytes)
        except ValueError as exc:
            raise ValueError(
                "stored replacement sheet is not a valid 2048x2048 image"
            ) from exc
        if output_media_type != "image/png":
            raise ValueError("stored replacement sheet is not PNG")
        expected_references.insert(0, {
            "image_reference": expected_image_reference,
            "role": "global_replacement_sheet",
            **output,
            "spec_sha256": raw["spec_sha256"],
        })
        if raw["h3_reference_images"] != expected_references:
            raise ValueError("stored H3 image reference order is invalid")
        receipt_path = cls._validate_artifact_receipt(
            project,
            seedream["receipt"],
            "work/global-reference-sheet/seedream-receipt.json",
        )
        try:
            seedream_receipt = json.loads(
                receipt_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream receipt is invalid") from exc
        endpoint = (
            seedream_receipt.get("endpoint")
            if isinstance(seedream_receipt, dict)
            else None
        )
        receipt_request_sha256 = (
            seedream_receipt.get("request_sha256")
            if isinstance(seedream_receipt, dict)
            else None
        )
        if (
            not isinstance(seedream_receipt, dict)
            or seedream_receipt.get("version") != 1
            or seedream_receipt.get("provider") != "ark_seedream"
            or seedream_receipt.get("status") != "succeeded"
            or endpoint != seedream["endpoint"]
            or seedream_receipt.get("model") != seedream["model"]
            or seedream_receipt.get("prompt_sha256") != prompt_sha256
            or seedream_receipt.get("input_sha256") != input_sha256
            or seedream_receipt.get("size") != SEEDREAM_OUTPUT_SIZE
            or receipt_request_sha256 != request_sha256
        ):
            raise ValueError(
                "stored Seedream receipt does not match its frozen request"
            )
        return output_path

    def _validated_prepared_seedream_attempt(
        self,
        project: Path,
        replacement_sheet: dict[str, Any],
    ) -> str:
        """Return generate/replay/regenerate after a read-only paid-attempt audit."""

        seedream = replacement_sheet.get("seedream")
        if not isinstance(seedream, dict):
            raise ValueError("stored Seedream state is invalid")
        directory = project / "work" / "global-reference-sheet"
        receipt_path = directory / "seedream-receipt.json"
        claim_path = directory / "seedream-receipt.json.post-claim"
        marker_path = directory / "seedream-receipt.json.response.json"
        result_path = directory / "seedream-receipt.json.result"
        provider_output = directory / "provider-output.bin"
        final_output = directory / "reference-sheet.png"

        if not receipt_path.exists():
            if any(
                path.exists()
                for path in (
                    claim_path,
                    marker_path,
                    result_path,
                    provider_output,
                    final_output,
                )
            ):
                raise ValueError(
                    "Seedream attempt evidence exists without its receipt"
                )
            expected_endpoint = (
                self.settings.seedream_api_base + "/images/generations"
            )
            if (
                seedream.get("endpoint") != expected_endpoint
                or seedream.get("model") != self.settings.seedream_model
                or not self.settings.seedream_api_key
            ):
                raise ValueError(
                    "current Seedream configuration does not match the frozen request"
                )
            return "generate"

        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("stored Seedream receipt path is invalid")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream receipt is invalid") from exc
        input_sha256 = [
            value.get("sha256") if isinstance(value, dict) else None
            for value in seedream.get("inputs", [])
        ]
        if (
            not isinstance(receipt, dict)
            or receipt.get("version") != 1
            or receipt.get("provider") != "ark_seedream"
            or receipt.get("endpoint") != seedream.get("endpoint")
            or receipt.get("model") != seedream.get("model")
            or receipt.get("prompt_sha256")
            != seedream.get("prompt", {}).get("sha256")
            or receipt.get("input_sha256") != input_sha256
            or receipt.get("size") != SEEDREAM_OUTPUT_SIZE
            or receipt.get("request_sha256")
            != seedream.get("request_sha256")
            or receipt.get("status")
            not in {
                "submitting", "response_received", "succeeded",
                "rejected", "submission_unknown",
            }
        ):
            raise ValueError(
                "stored Seedream receipt does not match its frozen request"
            )

        if receipt.get("status") in {"rejected", "submission_unknown"}:
            # A terminally rejected/unknown attempt has no recoverable
            # response; the caller resets its artifacts and re-POSTs the
            # frozen request instead of blocking the resume paths.
            return "regenerate"

        if claim_path.is_symlink() or not claim_path.is_file():
            raise ValueError("stored Seedream POST claim is missing")
        try:
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stored Seedream POST claim is invalid") from exc
        if (
            not isinstance(claim, dict)
            or claim.get("version") != 1
            or claim.get("kind") != "seedream_paid_post"
            or claim.get("request_sha256") != seedream.get("request_sha256")
        ):
            raise ValueError("stored Seedream POST claim is invalid")

        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("stored Seedream response is not locally recoverable")
        try:
            result = result_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                "stored Seedream response is not locally recoverable"
            ) from exc
        try:
            media_type = _validated_image_media_type(result)
        except ValueError as exc:
            raise ValueError(
                "stored Seedream result is not a complete image"
            ) from exc
        result_sha256 = hashlib.sha256(result).hexdigest()

        if marker_path.exists():
            if marker_path.is_symlink() or not marker_path.is_file():
                raise ValueError("stored Seedream response marker is invalid")
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("stored Seedream response marker is invalid") from exc
            if (
                not isinstance(marker, dict)
                or marker.get("version") != 1
                or marker.get("kind") != "seedream_response"
                or marker.get("request_sha256")
                != seedream.get("request_sha256")
                or marker.get("result_file") != result_path.name
                or marker.get("result_bytes") != len(result)
                or marker.get("result_sha256") != result_sha256
                or marker.get("media_type") != media_type
                or marker.get("width") != SEEDREAM_OUTPUT_WIDTH
                or marker.get("height") != SEEDREAM_OUTPUT_HEIGHT
            ):
                raise ValueError("stored Seedream response marker is invalid")
        elif receipt.get("status") != "submitting":
            raise ValueError("stored Seedream response marker is missing")

        if (
            receipt.get("result_bytes") not in {None, len(result)}
            or receipt.get("result_sha256") not in {None, result_sha256}
        ):
            raise ValueError("stored Seedream receipt result binding is invalid")
        if receipt.get("status") == "succeeded" and (
            receipt.get("result_file") != result_path.name
            or receipt.get("result_bytes") != len(result)
            or receipt.get("result_sha256") != result_sha256
            or receipt.get("response_marker_file") != marker_path.name
            or receipt.get("output_sha256") != result_sha256
        ):
            raise ValueError("stored Seedream succeeded receipt is invalid")
        return "replay"

    def _load_r2v_resume(
        self, project_id: str,
    ) -> tuple[
        Path, list[dict[str, Any]], Path | None, Path | None,
        dict[str, Any], list[Path],
    ]:
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        effective_request = self._validated_effective_request(
            state.get("effective_request")
        )
        status = state.get("status")
        stage = (state.get("project_progress") or {}).get("stage")
        if status != "failed" and not (
            status == "queued" and stage == "picsart_resume_queued"
        ):
            raise ValueError("only a failed or explicitly queued project can resume Picsart")
        _source_input, replacement = self._stored_input_paths(project, state)
        raw_segments = self._validated_stored_segments(
            state, missing_message="project has no reusable segments",
        )
        manifest = self._validated_h3_prepared_artifacts(
            project, state, raw_segments,
        )
        if manifest is None:
            raise ValueError(
                "project is missing the current H3 preparation manifest"
            )
        replacement_sheet = self._validated_stored_replacement_sheet(
            project, state,
        )
        voice_references = self._stored_voice_references(project, state)
        state_receipts = self._validated_submission_receipts(state)
        manifest_version = (
            (state.get("h3_prepared_artifacts") or {}).get("version")
            if isinstance(state.get("h3_prepared_artifacts"), dict)
            else None
        )
        if manifest_version != H3_PREPARED_ARTIFACTS_VERSION:
            raise ValueError("stored H3 preparation manifest is invalid")
        current_first_submit_resume = False
        expected_receipt_segments = {
            str(item["index"]) for item in raw_segments
        }
        if state_receipts is None:
            has_submit_directory = any(
                path.is_dir()
                for path in (project / "logs").glob("picsart*-submit")
            )
            if has_submit_directory:
                raise ValueError(
                    f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} has "
                    "Picsart submission evidence without its receipt ledger"
                )
            # The current manifest is frozen before global Picsart preflight.
            # A preflight failure therefore legitimately leaves neither a
            # ledger nor a submit directory. This is the only state in which
            # Picsart resume may perform the first paid submission.
            current_first_submit_resume = True
        elif set(state_receipts["segments"]) != expected_receipt_segments:
            raise ValueError(
                f"H3 preparation v{H3_PREPARED_ARTIFACTS_VERSION} requires a "
                "complete Picsart receipt ledger; reconcile it before any paid submit"
            )
        if state_receipts is not None and not set(
            state_receipts["segments"]
        ).issubset({str(item["index"]) for item in raw_segments}):
            raise ValueError("stored Picsart receipts reference unknown segments")

        prepared: list[dict[str, Any]] = []
        for raw in raw_segments:
            item = dict(raw)
            index = item["index"]
            segment_dir = project / "work" / "segments" / f"{index:04d}"
            source = segment_dir / "source.mp4"
            muted = project / manifest[str(index)]["muted_video"]["path"]
            prompt_path = segment_dir / "h3_prompt.txt"
            required_artifacts = [source, muted, prompt_path]
            dialogue_track_receipt = manifest[str(index)]["dialogue_track"]
            if dialogue_track_receipt is not None:
                required_artifacts.append(
                    project / dialogue_track_receipt["path"]
                )
            for required in required_artifacts:
                if not required.is_file():
                    raise FileNotFoundError(
                        f"reusable segment artifact is missing: {required.name}"
                    )
                if required.stat().st_size < 1:
                    raise ValueError(
                        f"reusable segment artifact is empty: {required.name}"
                    )
            self._validate_reference_video(muted)
            final_prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not final_prompt:
                raise ValueError("reusable H3 prompt is empty")
            resume_output = self._completed_resume_output(
                project, segment_dir, self._segment_scope(index),
            )
            scope = self._segment_scope(index)
            has_state_receipt, resume_task_id = self._state_submission_task(
                state, index,
            )
            if (
                has_state_receipt
                and isinstance(resume_task_id, str)
                and resume_task_id
                and resume_task_id
                in self._terminal_resume_task_ids(project, scope)
            ):
                # A previous runtime may have persisted the accepted receipt
                # before it learned this provider-specific terminal envelope.
                # Reconcile only the same task id under an explicitly
                # confirmed resume request, then allow this segment alone to
                # obtain a new submission receipt.
                self._mark_submission_task_failed(
                    project_id, index, resume_task_id,
                )
                resume_task_id = None
            elif not has_state_receipt and not current_first_submit_resume:
                raise ValueError("Picsart segment receipt is missing")
            item.update({
                "source": source,
                "resume_muted": muted,
                "resume_prompt": final_prompt,
                "resume_task_id": resume_task_id,
                "resume_output": resume_output,
                "status": "running",
            })
            prepared.append(item)
        prepared.sort(key=lambda item: int(item["index"]))
        return (
            project,
            prepared,
            replacement,
            replacement_sheet,
            effective_request,
            voice_references,
        )

    def _audit_h3_prep_submission_reset(
        self,
        project: Path,
        state: dict[str, Any],
        segments: list[dict[str, Any]],
    ) -> None:
        """Prove that replacing the H3 preparation cannot hide a paid POST.

        A new H3 preparation intentionally gets a new Picsart ledger, since
        its request can differ from the old frozen one.  That is safe only
        after the old ledger has been fully validated and none of its submits
        is left in the fail-closed, unknown-outcome state.
        """
        raw_ledger = state.get("picsart_submission_receipts")
        if raw_ledger is None:
            if any(
                path.is_dir()
                for path in (project / "logs").glob("picsart*-submit")
            ):
                raise ValueError(
                    "project has Picsart submission evidence without its receipt ledger"
                )
            return

        # Receipt v2 binds to the old manifest.  Validate that binding before
        # clearing either state field, so this explicit rerun never erases
        # evidence of an unaccounted provider submission.
        self._validated_h3_prepared_artifacts(project, state, segments)
        ledger = self._validated_submission_receipts(state)
        if ledger is None:  # Defensive: raw_ledger above was not None.
            raise ValueError("Picsart submission receipt ledger is missing")
        expected_segments = {str(item["index"]) for item in segments}
        if set(ledger["segments"]) != expected_segments:
            raise ValueError(
                "H3 preparation requires a complete Picsart receipt ledger; "
                "reconcile it before preparing again"
            )
        for item in segments:
            # This retains the exact submission_started/unknown fail-closed
            # semantics used by Picsart resume. Accepted and terminally
            # failed runs are auditable and may be superseded by the caller's
            # explicit changed-preparation request.
            self._state_submission_task(state, int(item["index"]))

    def _load_h3_prep_resume(
        self,
        project_id: str,
    ) -> tuple[
        Path,
        list[dict[str, Any]],
        Path | None,
        dict[str, Any],
        Path | None,
        dict[str, Any],
        str,
    ]:
        """Load trusted pre-H3 state without trusting prior H3 artifacts."""
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        request = self._validated_effective_request(
            state.get("effective_request")
        )
        status = state.get("status")
        stage = (state.get("project_progress") or {}).get("stage")
        if status not in {"failed", "succeeded"} and not (
            status == "queued" and stage == "h3_prep_resume_queued"
        ):
            raise ValueError(
                "only a failed, succeeded, or explicitly queued project can resume H3 preparation"
            )
        _source_input, replacement = self._stored_input_paths(project, state)
        raw_segments = self._validated_stored_segments(
            state, missing_message="project has no reusable segments",
        )
        enhanced_element_index = state.get("enhanced_element_index")
        if not isinstance(enhanced_element_index, dict):
            raise ValueError("stored enhanced element index is invalid")

        # This checks the frozen replacement plan, its source-video receipts,
        # object-replace bindings, and the completed Seedream artifact.  A
        # prepared (not yet completed) sheet remains recoverable through the
        # same Seedream receipt-reset path as resume-pre-picsart.
        frozen_sheet = state.get("replacement_sheet")
        if not isinstance(frozen_sheet, dict):
            raise ValueError("stored replacement sheet state is invalid")
        replacement_sheet_path = self._validated_stored_replacement_sheet(
            project, state, allow_prepared=True,
        )
        sheet_status = frozen_sheet.get("status")
        recovery_mode = (
            "reuse"
            if sheet_status in {"not_needed", "succeeded"}
            else self._validated_prepared_seedream_attempt(project, frozen_sheet)
        )
        self._stored_voice_references(project, state)

        prepared: list[dict[str, Any]] = []
        for raw in raw_segments:
            visual_prompt = raw.get("visual_prompt")
            dialogue = raw.get("dialogue")
            if (
                not isinstance(visual_prompt, str)
                or not isinstance(dialogue, list)
                or any(not isinstance(line, dict) for line in dialogue)
            ):
                raise ValueError("stored H3 preparation segment content is invalid")
            index = int(raw["index"])
            source = project / "work" / "segments" / f"{index:04d}" / "source.mp4"
            if not source.is_file() or source.is_symlink() or source.stat().st_size < 1:
                raise FileNotFoundError("reusable segment source is missing")
            if dialogue:
                track = source.parent / "dialogue-track.wav"
                if (
                    not track.is_file()
                    or track.is_symlink()
                    or track.stat().st_size < 1
                ):
                    raise FileNotFoundError("reusable dialogue track is missing")
            prepared.append({
                **raw,
                "source": source,
                "status": "running",
            })

        self._audit_h3_prep_submission_reset(project, state, raw_segments)
        return (
            project,
            prepared,
            replacement,
            frozen_sheet,
            replacement_sheet_path,
            enhanced_element_index,
            recovery_mode,
        )

    def _invalidate_h3_prepared_artifacts(
        self,
        project_id: str,
        project: Path,
        prepared: list[dict[str, Any]],
        log: ResearchLog,
    ) -> None:
        """Drop only products made after the trusted pre-H3 barrier.

        The local detection JSON files are deliberately removed so the mask
        node can render with its current options; object_detect's shared
        fingerprint cache lives outside a project directory and is untouched.
        """
        self.store.update(
            project_id,
            h3_prepared_artifacts=None,
            picsart_submission_receipts=None,
        )
        removed: list[str] = []
        for item in prepared:
            segment_dir = item["source"].parent
            candidates = [
                segment_dir / name
                for name in (
                    "source-muted.mp4",
                    "source-objects-masked.mp4",
                    "h3_prompt.txt",
                    # h3-context-ir 节点产物：原文备份与回执随重备一并失效，
                    # 否则幂等指纹会让旧增强文本挡住新 prompt。
                    "h3_prompt_origin.txt",
                    "context_ir.json",
                )
            ]
            # output_media records every local object-detect*.json artifact;
            # include legacy target-named fallbacks as well.  Do not recurse,
            # unlink directories, or follow a malformed symlink.
            candidates.extend(sorted(segment_dir.glob("object-detect*.json")))
            seen: set[Path] = set()
            for path in candidates:
                if path in seen:
                    continue
                seen.add(path)
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed.append(path.relative_to(project).as_posix())
        log.event(
            "h3_prep_artifacts_invalidated",
            removed_artifacts=removed,
            shared_object_detect_cache_preserved=True,
        )

    def validate_h3_prep_resume(self, project_id: str) -> None:
        self._load_h3_prep_resume(project_id)

    def resume_h3_prep(self, project_id: str) -> None:
        (
            project,
            prepared,
            replacement,
            frozen_sheet,
            _replacement_sheet_path,
            enhanced_element_index,
            recovery_mode,
        ) = self._load_h3_prep_resume(project_id)
        request = self._validated_effective_request(
            self.store.read(project_id).get("effective_request")
        )
        log = ResearchLog(project)
        log.event("h3_prep_resume_started")
        try:
            replacement_sheet, replacement_sheet_path = self._resume_replacement_sheet(
                project_id,
                project,
                frozen_sheet,
                recovery_mode,
                log,
            )
            self._invalidate_h3_prepared_artifacts(
                project_id, project, prepared, log,
            )
            self._progress(
                project_id,
                57,
                "h3-prep-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            outputs = self._run_h3_segments(
                project_id,
                project,
                prepared,
                enhanced_element_index,
                replacement,
                replacement_sheet,
                replacement_sheet_path,
                request,
                log,
            )
            if outputs is None:
                # 提示词审核暂停：重新准备的产物已冻结，confirm 后继续。
                log.event("h3_prep_resume_paused_for_prompt_review")
                return

            final = project / "output" / "generated.mp4"
            join_modes = self._validated_join_modes(prepared)
            self._progress(
                project_id,
                95,
                "stitch-h3-prep-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            self._node(
                log,
                "stitch-h3-prep-resume",
                "project",
                {
                    "segments": outputs,
                    "target_durations": [item["duration_s"] for item in prepared],
                    "join_modes": join_modes,
                    "output": final,
                },
                outputs,
                lambda: media.concat(
                    outputs,
                    final,
                    target_durations=[item["duration_s"] for item in prepared],
                    join_modes=join_modes,
                ),
                lambda _value: [final],
            )
            self._replace_dialogue_audio_track(project, prepared, final, log)
            self._add_subtitles(project_id, project, final, request, log)
            # 终态总耗时随成功落盘（供列表直接读），与主路径口径一致。
            display_duration = display_duration_seconds(project)
            self.store.update(
                project_id,
                status="succeeded",
                has_video=True,
                output_path="output/generated.mp4",
                **(
                    {"display_duration_s": round(display_duration, 3)}
                    if display_duration is not None
                    else {}
                ),
                project_progress={
                    "status": "succeeded",
                    "percent": 100,
                    "stage": "completed",
                },
                generation={
                    "status": "succeeded",
                    "stage": "completed",
                    "error": None,
                },
                segments=[
                    {**self._public_segment(item), "status": "succeeded"}
                    for item in prepared
                ],
                error=None,
            )
            log.event("h3_prep_resume_succeeded", output=final)
            self._prune_artifacts_after_success(project, log)
        except Exception as exc:
            log.event("h3_prep_resume_failed", error=str(exc))
            self._persist_failure(
                project_id,
                log,
                stage="h3_prep_resume_failed",
                code="h3_prep_resume_failed",
                message="H3 准备段续跑失败",
                min_percent=57,
                extra_changes={
                    "segments": [self._public_segment(item) for item in prepared],
                    "project_progress": {
                        "status": "failed",
                        "percent": 57,
                        "stage": "h3_prep_resume_failed",
                    },
                },
            )

    def confirm_prompt_review(
        self, project_id: str, edits: list[dict[str, Any]],
    ) -> None:
        """确认审核后的可编辑主体：重建 h3_prompt.txt 并重算冻结清单回执。

        重建公式：final = prefix + edited + hidden + suffix——hidden（权威
        替换计划块）与 prefix/suffix 逐字取自冻结的审核记录，用户传入的
        任何内容（包括伪造的 BEGIN 文本）都无法影响它们。只校验、不写状
        态；状态迁移（queued + picsart_resume_queued）与续跑调度由调用方
        在确认成功后完成（resume-picsart 等价入口逐字重放 h3_prompt.txt）。
        先全量校验再落盘，任何一段非法都不动文件。
        """
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        if state.get("status") != "awaiting_prompt_review":
            raise PromptReviewError(
                "prompt_review_not_pending", "项目不在等待提示词审核状态"
            )
        frozen = self._prompt_review_frozen_segments(project)
        normalized = self._validated_prompt_review_edits(frozen, edits)
        manifest = self._prompt_review_manifest(state, frozen)
        rebuilt = self._rebuild_frozen_prompts(project, frozen, normalized)
        self._apply_rebuilt_prompts(
            project, manifest, frozen, rebuilt, normalized,
        )
        self.store.update(project_id, h3_prepared_artifacts=manifest)
        ResearchLog(project).event(
            "prompt_review_confirmed",
            segments=sorted(rebuilt),
            edited=[
                index
                for index in sorted(rebuilt)
                if normalized[index] != frozen[index]["editable"]
            ],
        )

    def regenerate_with_edited_prompts(
        self, project_id: str, edits: list[dict[str, Any]],
    ) -> None:
        """组合入口：prepare + execute，不含状态翻转。

        端点用拆分形式：prepare（只读校验）→ 原子翻转 queued+has_video=False
        → execute（落盘），保证归档窗口内任何轮询都看不到"成功+旧视频"。
        """
        prepared = self.prepare_regenerate_prompts(project_id, edits)
        self.execute_regenerate(project_id, prepared)

    def prepare_regenerate_prompts(
        self, project_id: str, edits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """regenerate 只读阶段：门禁/编辑校验/stale 重建/账本审计/归档规划。

        succeeded 项目编辑冻结提示词重提 H3。与 confirm 共用同一份编辑校验
        与四段重建公式；Picsart 旧账本先过 _audit_h3_prep_submission_reset
        的 fail-closed 审计（execute 阶段才清零），使 resume-picsart 等价入
        口以全新提交重放新提示词。本阶段不改动任何文件与状态。
        """
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        if state.get("status") != "succeeded":
            raise PromptReviewError(
                "regenerate_not_succeeded",
                "只有已成功项目可以编辑提示词重新生成",
            )
        frozen = self._prompt_review_frozen_segments(project)
        normalized = self._validated_prompt_review_edits(frozen, edits)
        manifest = self._prompt_review_manifest(state, frozen)
        rebuilt = self._rebuild_frozen_prompts(project, frozen, normalized)
        raw_segments = self._validated_stored_segments(
            state, missing_message="project has no reusable segments",
        )
        # 审计旧账本（绑定清单且无 fail-closed 回执）后才允许清零重提。
        self._audit_h3_prep_submission_reset(project, state, raw_segments)
        archive = self._plan_current_generation_archive(project, state, frozen)
        return {
            "frozen": frozen,
            "normalized": normalized,
            "manifest": manifest,
            "rebuilt": rebuilt,
            "archive": archive,
        }

    def execute_regenerate(
        self, project_id: str, prepared: dict[str, Any],
    ) -> None:
        """regenerate 落盘阶段：归档当前版本（失败整体回滚）→ 重建提示词。

        调用方必须在执行前已把项目原子翻转成 queued + has_video=False，并
        在本方法抛错时把项目标 failed（不得停在"无视频但标成功"的假态）。
        """
        project = self.store.path(project_id)
        state = self.store.read(project_id)
        frozen = prepared["frozen"]
        normalized = prepared["normalized"]
        rebuilt = prepared["rebuilt"]
        history_entry = self._apply_archive_plan(project, prepared["archive"])
        self._apply_rebuilt_prompts(
            project, prepared["manifest"], frozen, rebuilt, normalized,
        )
        # 归档已留副本后清掉原件：重建后的 h3_prompt.txt 是用户审定文本，
        # 旧 Context IR 回执/原文备份不应留在当前版本目录（幂等指纹校验
        # 本来也挡得住误用，此处保持目录语义干净）。删除失败仅记事件，
        # 不影响 regenerate 主流程。
        for artifact in prepared["archive"]["context_ir_artifacts"]:
            try:
                artifact["source"].unlink()
            except OSError as exc:
                ResearchLog(project).event(
                    "context_ir_archive_cleanup_failed",
                    path=str(artifact["source"]),
                    error=str(exc),
                )
        history = state.get("generation_history") or []
        self.store.update(
            project_id,
            h3_prepared_artifacts=prepared["manifest"],
            picsart_submission_receipts=None,
            generation_history=[*history, history_entry],
        )
        ResearchLog(project).event(
            "prompt_regenerate_prepared",
            archived_version=history_entry["version"],
            segments=sorted(rebuilt),
            edited=[
                index
                for index in sorted(rebuilt)
                if normalized[index] != frozen[index]["editable"]
            ],
        )

    def _prompt_review_frozen_segments(
        self, project: Path,
    ) -> dict[int, dict[str, Any]]:
        review = prompt_review.read_review(project)
        if review is None:
            raise PromptReviewError(
                "prompt_review_unavailable", "冻结的提示词审核记录不可用"
            )
        return {
            segment["segment_index"]: segment for segment in review["segments"]
        }

    @staticmethod
    def _prompt_review_manifest(
        state: dict[str, Any], frozen: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        manifest = state.get("h3_prepared_artifacts")
        manifest_segments = (
            manifest.get("segments") if isinstance(manifest, dict) else None
        )
        if (
            not isinstance(manifest_segments, dict)
            or set(manifest_segments) != {str(index) for index in frozen}
            or not all(
                isinstance(value, dict) for value in manifest_segments.values()
            )
        ):
            raise PromptReviewError(
                "prompt_review_unavailable", "冻结的 H3 产物清单不可用"
            )
        return manifest

    @staticmethod
    def _validated_prompt_review_edits(
        frozen: dict[int, dict[str, Any]], edits: list[dict[str, Any]],
    ) -> dict[int, str]:
        """编辑请求的形态/段集合/长度校验（confirm 与 regenerate 共用）。"""
        if not isinstance(edits, list) or not edits:
            raise PromptReviewError(
                "invalid_prompt_review_edit", "segments 必须是非空数组"
            )
        normalized: dict[int, str] = {}
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"segment_index", "prompt"}:
                raise PromptReviewError(
                    "invalid_prompt_review_edit",
                    "每段只能包含 segment_index 与 prompt",
                )
            index = edit["segment_index"]
            body = edit["prompt"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise PromptReviewError(
                    "invalid_prompt_review_edit", "segment_index 必须是整数"
                )
            if not isinstance(body, str) or (
                not body.strip() and not frozen.get(index, {}).get("hidden")
            ):
                # 有 hidden 块的段允许把可编辑前半清空；没有隐藏块的段保持
                # 原有的非空约束（与拆分前行为一致）。
                raise PromptReviewError(
                    "invalid_prompt_review_edit", "提示词主体不能为空"
                )
            if index in normalized:
                raise PromptReviewError(
                    "invalid_prompt_review_edit", "segment_index 不能重复"
                )
            normalized[index] = body
        if set(normalized) != set(frozen):
            raise PromptReviewError(
                "invalid_prompt_review_edit", "段集合必须与冻结的审核记录一致"
            )
        for index, body in normalized.items():
            segment = frozen[index]
            final = (
                segment["prefix"] + body + segment["hidden"] + segment["suffix"]
            )
            stripped = final.strip()
            if not stripped or len(stripped) > PICSART_PROMPT_MAX_CHARS:
                raise PromptReviewError(
                    "invalid_prompt_review_edit",
                    f"分段 {index} 重建后的提示词为空或超过长度限制",
                )
        return normalized

    @staticmethod
    def _rebuild_frozen_prompts(
        project: Path,
        frozen: dict[int, dict[str, Any]],
        normalized: dict[int, str],
    ) -> dict[int, str]:
        """stale 校验 + 四段重建（confirm 与 regenerate 共用，不落盘）。"""
        rebuilt: dict[int, str] = {}
        for index in sorted(frozen):
            segment = frozen[index]
            prompt_path = (
                project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
            )
            try:
                current = prompt_path.read_text(encoding="utf-8")
            except OSError:
                raise PromptReviewError(
                    "prompt_review_unavailable", "冻结的 H3 提示词文件不可用"
                ) from None
            prefix, editable, hidden, suffix = (
                segment["prefix"], segment["editable"],
                segment["hidden"], segment["suffix"],
            )
            if current != prefix + editable + hidden + suffix:
                # 冻结提示词与审核记录不一致：若是 IR 增强后的文本（提交前
                # 优化层已跑），走 IR 链校验——origin==审核四段原文 且
                # 回执.origin==其哈希 且 当前==optimized，链成立即视为
                # 未变化（confirm/regenerate 重建会覆盖并失效旧 IR 产物）；
                # 链不成立才是真变化（如并发重备），拒绝覆盖。
                if not _context_ir_frozen_chain(
                    prompt_path.parent,
                    hashlib.sha256(
                        (prefix + editable + hidden + suffix).encode("utf-8")
                    ).hexdigest(),
                ):
                    raise PromptReviewError(
                        "prompt_review_stale", "冻结提示词已变化，请重新审核"
                    )
            rebuilt[index] = prefix + normalized[index] + hidden + suffix
        return rebuilt

    def _apply_rebuilt_prompts(
        self,
        project: Path,
        manifest: dict[str, Any],
        frozen: dict[int, dict[str, Any]],
        rebuilt: dict[int, str],
        normalized: dict[int, str],
    ) -> None:
        """写回各段 h3_prompt.txt、重算清单 prompt 回执、更新审核记录。"""
        manifest_segments = manifest["segments"]
        for index in sorted(rebuilt):
            prompt_path = (
                project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt"
            )
            prompt_path.write_text(rebuilt[index], encoding="utf-8")
            manifest_segments[str(index)]["prompt"] = (
                self._project_artifact_receipt(
                    project,
                    prompt_path,
                    f"work/segments/{index:04d}/h3_prompt.txt",
                )
            )
        prompt_review.write_review(
            project,
            [
                {
                    "segment_index": index,
                    "prefix": frozen[index]["prefix"],
                    "editable": normalized[index],
                    "hidden": frozen[index]["hidden"],
                    "suffix": frozen[index]["suffix"],
                    "start_s": frozen[index]["start_s"],
                    "end_s": frozen[index]["end_s"],
                }
                for index in sorted(frozen)
            ],
        )

    def _plan_current_generation_archive(
        self,
        project: Path,
        state: dict[str, Any],
        frozen: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """归档规划（只读）：全部 fail-closed 校验在本阶段完成。

        分段成片位置、冻结提示词内容、归档目录名冲突、成片缺失都在这里
        判掉；_apply_archive_plan 只剩纯文件操作（带整体回滚）。
        """
        raw_history = state.get("generation_history")
        if raw_history is not None and (
            not isinstance(raw_history, list)
            or any(
                not isinstance(entry, dict)
                or isinstance(entry.get("version"), bool)
                or not isinstance(entry.get("version"), int)
                or entry["version"] < 1
                for entry in raw_history
            )
        ):
            raise PromptReviewError(
                "regenerate_unavailable", "生成历史状态损坏，拒绝归档"
            )
        history = raw_history or []
        version = (
            max(entry["version"] for entry in history) + 1 if history else 1
        )
        archive_root = project / "output" / "history" / f"v{version}"
        if archive_root.exists():
            raise PromptReviewError(
                "regenerate_unavailable", "历史归档目录已存在，拒绝覆盖"
            )
        final_video = project / "output" / "generated.mp4"
        if (
            not final_video.is_file()
            or final_video.is_symlink()
            or final_video.stat().st_size < 1
        ):
            raise PromptReviewError(
                "regenerate_unavailable", "当前成片缺失，无法归档"
            )
        archive_segments: list[dict[str, Any]] = []
        segment_moves: list[tuple[Path, Path]] = []
        for index in sorted(frozen):
            segment_dir = project / "work" / "segments" / f"{index:04d}"
            prompt_path = segment_dir / "h3_prompt.txt"
            try:
                prompt_text = prompt_path.read_text(encoding="utf-8")
            except OSError:
                raise PromptReviewError(
                    "regenerate_unavailable", "冻结提示词缺失，无法归档"
                ) from None
            segment_output = self._completed_resume_output(
                project, segment_dir, self._segment_scope(index),
            )
            if segment_output is None:
                raise PromptReviewError(
                    "regenerate_unavailable",
                    f"分段 {index} 的成片缺失，无法归档",
                )
            archive_segments.append({
                "segment_index": index,
                "prompt_sha256": hashlib.sha256(
                    prompt_text.encode("utf-8")
                ).hexdigest(),
                "prompt_path": (
                    f"output/history/v{version}/prompts/segment-{index:04d}.txt"
                ),
                "start_s": frozen[index]["start_s"],
                "end_s": frozen[index]["end_s"],
            })
            segment_moves.append((
                segment_output,
                archive_root / "segments" / f"segment-{index:04d}.mp4",
            ))
        submit_logs = [
            path
            for path in sorted((project / "logs").glob("picsart*-submit"))
            if path.is_dir()
        ]
        # h3-context-ir 产物（存在才归档）：旧版本用过官方增强时，回执与
        # 原文备份随版本留存；归档成功后由 execute_regenerate 清掉原件，
        # 避免新版本目录残留指向旧提示词的幂等指纹。
        context_ir_artifacts: list[dict[str, Any]] = []
        for index in sorted(frozen):
            segment_dir = project / "work" / "segments" / f"{index:04d}"
            for name, suffix in (
                ("context_ir.json", "context-ir.json"),
                ("h3_prompt_origin.txt", "origin.txt"),
            ):
                candidate = segment_dir / name
                if candidate.is_file() and not candidate.is_symlink():
                    context_ir_artifacts.append({
                        "source": candidate,
                        "archive_name": f"segment-{index:04d}.{suffix}",
                    })
        return {
            "version": version,
            "completed_at": state.get("updated_at"),
            "archive_root": archive_root,
            "final_video": final_video,
            "segments": archive_segments,
            "segment_moves": segment_moves,
            "submit_logs": submit_logs,
            "context_ir_artifacts": context_ir_artifacts,
        }

    def _apply_archive_plan(
        self, project: Path, plan: dict[str, Any],
    ) -> dict[str, Any]:
        """执行归档落盘并返回历史条目；任何一步失败整体回滚。

        分段成片与旧 submit 证据日志是移动（不是复制）：resume 路径发现旧
        成片会直接复用而不重提，而清零账本后门禁要求不得遗留 submit 痕迹；
        证据不删除、随版本归档。回滚把已移动的搬回原位并删除半成品目录，
        不留残骸挡住重试。
        """
        archive_root = plan["archive_root"]
        version = plan["version"]
        (archive_root / "prompts").mkdir(parents=True, mode=0o700)
        (archive_root / "segments").mkdir(mode=0o700)
        moved: list[tuple[Path, Path]] = []
        try:
            shutil.copy2(plan["final_video"], archive_root / "generated.mp4")
            for segment in plan["segments"]:
                index = segment["segment_index"]
                shutil.copy2(
                    project / "work" / "segments" / f"{index:04d}" / "h3_prompt.txt",
                    project / segment["prompt_path"],
                )
            for artifact in plan["context_ir_artifacts"]:
                shutil.copy2(
                    artifact["source"],
                    archive_root / "prompts" / artifact["archive_name"],
                )
            for source, destination in plan["segment_moves"]:
                os.replace(source, destination)
                moved.append((destination, source))
            if plan["submit_logs"]:
                logs_archive = archive_root / "submit-logs"
                logs_archive.mkdir(mode=0o700)
                for directory in plan["submit_logs"]:
                    target = logs_archive / directory.name
                    os.rename(directory, target)
                    moved.append((target, directory))
        except Exception:
            for destination, source in reversed(moved):
                try:
                    os.replace(destination, source)
                except OSError:
                    pass
            shutil.rmtree(archive_root, ignore_errors=True)
            try:
                archive_root.parent.rmdir()  # 空父目录一并清掉，不挡重试
            except OSError:
                pass
            raise
        return {
            "version": version,
            "completed_at": plan["completed_at"],
            "video": self._project_artifact_receipt(
                project,
                archive_root / "generated.mp4",
                f"output/history/v{version}/generated.mp4",
            ),
            "segments": plan["segments"],
        }

    def validate_r2v_resume(self, project_id: str) -> None:
        self._load_r2v_resume(project_id)

    def resume_r2v(self, project_id: str) -> None:
        (
            project,
            prepared,
            replacement,
            replacement_sheet,
            request,
            _voice_references,
        ) = (
            self._load_r2v_resume(project_id)
        )
        log = ResearchLog(project)
        # Context IR 提交前优化层（六条提交路径共用，2026-09-30 顺序调整）：
        # 幂等键命中即复用旧 IR 输出（resume-picsart 的逐字重放纪律靠文件
        # 实际哈希保证，不是跳过 IR）；原文变化自动重跑；disabled/未跑过
        # 则原文直过。已完成分段（resume_output 可复用、不再提交）不参与。
        ir_candidates = [
            item for item in prepared
            if not isinstance(item.get("resume_output"), Path)
        ]
        if ir_candidates:
            resume_state = self.store.read(project_id)
            element_index = resume_state.get(
                "enhanced_element_index"
            )
            # resume 路径的 replacement_sheet 是 Path（图版文件），IR 蒸馏输入
            # 需要的是 state 里的 sheet 字典（h3_reference_images 清单）。
            stored_sheet = resume_state.get("replacement_sheet")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(ir_candidates)),
            ) as ir_pool:
                ir_inputs = self._enhance_h3_prompts_with_context_ir(
                    project_id,
                    project,
                    [
                        (item, item["resume_prompt"], item["resume_muted"], "")
                        for item in ir_candidates
                    ],
                    stored_sheet if isinstance(stored_sheet, dict) else {},
                    element_index if isinstance(element_index, dict) else {},
                    request,
                    log,
                    ir_pool,
                )
            for item, (_ir_item, ir_prompt, _ir_muted, _ir_hash) in zip(
                ir_candidates, ir_inputs,
            ):
                item["resume_prompt"] = ir_prompt
        node_prefix, artifact_suffix = self._next_resume_identity(
            project, self.store.read(project_id),
        )
        log.event("picsart_resume_started")
        try:
            self._progress(
                project_id,
                57,
                "picsart-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(prepared)),
            ) as pool:
                preflight_futures: list[
                    concurrent.futures.Future[PreparedPicsartRequest] | None
                ] = []
                for item in prepared:
                    reusable = item.get("resume_output")
                    if isinstance(reusable, Path):
                        item["final_prompt"] = item["resume_prompt"]
                        item["task_id"] = item.get("resume_task_id")
                        item["status"] = "succeeded"
                        preflight_futures.append(None)
                        continue
                    preflight_futures.append(pool.submit(
                        self._prepare_picsart_segment,
                        project_id,
                        project,
                        item,
                        replacement,
                        replacement_sheet,
                        request,
                        log,
                        final_prompt=item["resume_prompt"],
                        muted=item["resume_muted"],
                        node_prefix=node_prefix,
                        artifact_suffix=artifact_suffix,
                        existing_task_id=item.get("resume_task_id"),
                    ))
                picsart_requests = [
                    None if future is None else future.result()
                    for future in preflight_futures
                ]
                self._begin_submission_batch([
                    prepared_request for prepared_request in picsart_requests
                    if prepared_request is not None
                ])
                futures: list[concurrent.futures.Future[Path] | None] = []
                for item, prepared_request in zip(prepared, picsart_requests):
                    if prepared_request is None:
                        futures.append(None)
                        continue
                    futures.append(pool.submit(
                        self._generate_picsart_segment,
                        prepared_request,
                        item,
                        log,
                    ))
                outputs = [
                    item["resume_output"] if future is None else future.result()
                    for item, future in zip(prepared, futures)
                ]

            final = project / "output" / "generated.mp4"
            join_modes = self._validated_join_modes(prepared)
            self._progress(
                project_id,
                95,
                "stitch-resume",
                segments=[self._public_segment(item) for item in prepared],
            )
            self._node(
                log,
                "stitch-resume",
                "project",
                {
                    "segments": outputs,
                    "target_durations": [item["duration_s"] for item in prepared],
                    "join_modes": join_modes,
                    "output": final,
                },
                outputs,
                lambda: media.concat(
                    outputs,
                    final,
                    target_durations=[item["duration_s"] for item in prepared],
                    join_modes=join_modes,
                ),
                lambda _value: [final],
            )
            self._replace_dialogue_audio_track(project, prepared, final, log)
            self._add_subtitles(project_id, project, final, request, log)
            # 终态总耗时随成功落盘（供列表直接读），与主路径口径一致。
            display_duration = display_duration_seconds(project)
            self.store.update(
                project_id,
                status="succeeded",
                has_video=True,
                output_path="output/generated.mp4",
                **(
                    {"display_duration_s": round(display_duration, 3)}
                    if display_duration is not None
                    else {}
                ),
                project_progress={
                    "status": "succeeded",
                    "percent": 100,
                    "stage": "completed",
                },
                generation={
                    "status": "succeeded",
                    "stage": "completed",
                    "error": None,
                },
                segments=[
                    {**self._public_segment(item), "status": "succeeded"}
                    for item in prepared
                ],
                error=None,
            )
            log.event("picsart_resume_succeeded", output=final)
            self._prune_artifacts_after_success(project, log)
        except Exception as exc:
            log.event("picsart_resume_failed", error=str(exc))
            self._persist_failure(
                project_id,
                log,
                stage="picsart_resume_failed",
                code="picsart_resume_failed",
                message="Picsart 后续链路执行失败",
                min_percent=57,
                extra_changes={
                    "segments": [self._public_segment(item) for item in prepared],
                    "project_progress": {
                        "status": "failed",
                        "percent": 57,
                        "stage": "picsart_resume_failed",
                    },
                },
            )

    def _analyze_audio_dialogue(
        self,
        project_id: str,
        project: Path,
        source: Path,
        *,
        has_audio: bool,
        log: ResearchLog,
        resume: bool = False,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        str | None,
    ]:
        audio = project / "work" / "source.wav"
        asr_output = project / "work" / "voice_lines.json"

        def run_node(
            node: str,
            inputs: Any,
            input_media: list[Path],
            operation: Callable[[], Any],
            output_media: Callable[[Any], list[Path]] | None = None,
            *,
            artifacts: Iterable[Path] = (),
        ) -> Any:
            if resume:
                reused = self._reusable_node_output(
                    project, node, "project", artifacts,
                )
                if reused is not _NO_REUSABLE_OUTPUT:
                    log.event("node_output_reused", node=node, scope="project")
                    return reused
            return self._node(
                log, node, "project", inputs, input_media, operation, output_media,
            )

        if not has_audio:
            reason = "source_has_no_audio"
            self._progress(project_id, 8, "audio")
            run_node(
                "extract-audio",
                {
                    "source": source,
                    "output": audio,
                    "timeline": "video-v0",
                    "has_audio": False,
                },
                [source],
                lambda: self._skipped_result(None, reason),
            )
            run_node(
                "vocal-separation",
                {"audio": None, "reason": reason},
                [],
                lambda: self._skipped_result(None, reason),
            )
            self._progress(project_id, 11, "yamnet")
            yamnet = run_node(
                "yamnet",
                {"audio": None, "reason": reason},
                [],
                lambda: self._skipped_result(
                    {"windows": [], "has_bgm": False}, reason,
                ),
            )
            self._progress(project_id, 14, "whisper-asr")
            asr_lines = run_node(
                "whisper-asr",
                {"audio": None, "reason": reason},
                [],
                lambda: self._skipped_result([], reason),
            )
            spoken_lines, voice_decisions = run_node(
                "spoken-dialogue-filter",
                {"asr_lines": [], "yamnet": yamnet, "reason": reason},
                [],
                lambda: self._skipped_result(([], []), reason),
            )
            return yamnet, asr_lines, spoken_lines, voice_decisions, None

        self._progress(project_id, 8, "audio")
        run_node(
            "extract-audio",
            {"source": source, "output": audio, "timeline": "video-v0"},
            [source],
            lambda: media.extract_audio(source, audio),
            lambda _value: [audio],
            artifacts=[audio],
        )

        # 人声分离：非人声轨（BGM/音效/环境音）进混音，人声轨仅调试。
        # CLI/模型未配置即节点明确报错（部署错误应当暴露，不静默降级为无
        # 音效复刻）。
        accompaniment = project / "work" / "accompaniment.wav"
        separated_vocals = project / "work" / "vocals.wav"
        self._progress(project_id, 9, "vocal-separation")
        run_node(
            "vocal-separation",
            {
                "audio": audio,
                "cli": self.settings.vocal_separation_cli,
                "model": self.settings.vocal_separation_model_path,
                "accompaniment": accompaniment,
                "vocals": separated_vocals,
            },
            [audio],
            lambda: vocal_separation.separate_vocals(
                audio,
                accompaniment,
                separated_vocals,
                cli_path=self.settings.vocal_separation_cli,
                model_path=self.settings.vocal_separation_model_path,
            ),
            lambda _value: [accompaniment, separated_vocals],
            artifacts=[accompaniment, separated_vocals],
        )

        self._progress(project_id, 11, "yamnet")
        yamnet = run_node(
            "yamnet",
            {"audio": audio, "model": self.settings.yamnet_model},
            [audio, self.settings.yamnet_model],
            lambda: media.yamnet_analyze(audio, self.settings.yamnet_model),
            artifacts=[audio],
        )

        vocal_expected = self._yamnet_has_vocal_signal(yamnet)

        # whisper 的音轨语种检测随 ASR 一起产出；节点输出形状不变（仍为台词
        # 列表），语种经闭包带出并单独落盘，续跑复用旧 ASR 产物时为 None，
        # 由下游回退到台词文字推断。
        detected_language: dict[str, str | None] = {"code": None}

        def transcribe_once() -> ProviderResult[list[dict[str, Any]]]:
            lines, record, language_code = media.transcribe(
                audio,
                asr_output,
                self.settings.asr_cli,
                self.settings.asr_model,
            )
            detected_language["code"] = language_code
            return ProviderResult(
                value=lines,
                record=record,
                evidence_media=[asr_output],
            )

        self._progress(project_id, 14, "whisper-asr")
        asr_lines = run_node(
            "whisper-asr",
            {
                "audio": audio,
                "cli": self.settings.asr_cli,
                "model": self.settings.asr_model,
                "vocal_signal_observed": vocal_expected,
            },
            [audio, self.settings.asr_model],
            transcribe_once,
            artifacts=[audio, asr_output],
        )
        if vocal_expected and not asr_lines:
            log.event(
                "empty_asr_accepted",
                node="whisper-asr",
                scope="project",
                attempts=1,
            )
        spoken_lines, voice_decisions = run_node(
            "spoken-dialogue-filter",
            {"asr_lines": asr_lines, "yamnet": yamnet},
            [],
            lambda: media.filter_spoken_lines(asr_lines, yamnet),
        )
        return (
            yamnet, asr_lines, spoken_lines, voice_decisions,
            detected_language["code"],
        )

    @staticmethod
    def _completed_node_value(project: Path, node: str, scope: str) -> Any:
        attempts = project / "logs" / node / scope
        numbered = [
            directory
            for directory in attempts.glob("attempt-*")
            if directory.name.rsplit("-", 1)[-1].isdigit()
        ]
        numbered.sort(
            key=lambda directory: int(directory.name.rsplit("-", 1)[-1]),
            reverse=True,
        )
        for directory in numbered:
            output = directory / "output.json"
            if not output.is_file() or (directory / "error.json").exists():
                continue
            value = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "value" in value:
                return value["value"]
            return value
        raise FileNotFoundError(f"completed node output is missing: {node}/{scope}")

    def _prepare_segment(self, project_id, project, source, segment, selection,
                         request, log, *, resume=False):
        """Single-segment compatibility helper; project runs use the barrier below."""
        material = self._prepare_segment_materials(
            project_id, project, source, segment, selection, request, log,
            resume=resume,
        )
        return self._analyze_prepared_segment(
            project, material, request, log, resume=resume,
        )

    def _prepare_project_segments(
        self, project_id, project, source, segments, selections, request, log,
        *, resume=False,
    ):
        # All original frames must exist before any vision-agent request starts.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(segments)),
        ) as pool:
            futures = [pool.submit(
                self._prepare_segment_materials,
                project_id, project, source, segment,
                selections[int(segment["index"])], request, log, resume=resume,
            ) for segment in segments]
            materials = [future.result() for future in futures]
        identity = self._person_identity_prepass(project, source, materials, log)
        replacement_originals = self._replacement_image_paths(project)
        if (
            self.settings.multiref_mode == "composed"
            and len(replacement_originals) > 1
        ):
            # composed 模式：多图在参考图准备环节合成为单一 provider 参考图。
            user_references = [self._materialize_canonical_user_reference(
                project, replacement_originals[0], log,
                replacement_paths=replacement_originals,
            )]
        else:
            user_references = [self._materialize_canonical_user_reference(
                project, path, log, order,
            ) for order, path in enumerate(replacement_originals, 1)]
        for item in materials:
            if user_references:
                item["user_reference_paths"] = user_references
            item["person_identity"] = {
                **identity,
                "observations": [
                    observation for observation in identity.get("observations", [])
                    if observation.get("segment_index") == int(item["index"])
                ],
            }
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(materials)),
        ) as pool:
            futures = [pool.submit(
                self._analyze_prepared_segment, project, item, request, log,
                resume=resume,
            ) for item in materials]
            prepared = [future.result() for future in futures]
        return sorted(prepared, key=lambda item: int(item["index"]))

    def _person_identity_prepass(self, project, source, materials, log):
        source_dir = project / "work" / "source-4fps"
        manifest_path = source_dir / "manifest.json"
        frames = []
        scenes = self.store.read(project.name).get("scenes", [])
        def source_scene(time_s):
            for scene in scenes:
                if float(scene["start_s"]) <= time_s < float(scene["end_s"]):
                    return f"SCENE_{int(scene['index']):02d}"
            return "SCENE_UNKNOWN"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frames.extend({
                "path": str((source_dir / entry["file"]).resolve()),
                "source_time_s": entry["time_seconds"],
                "scene_id": source_scene(float(entry["time_seconds"])),
                "segment_index": None,
                "frame_order": entry.get("index"),
                "kind": "source",
            } for entry in manifest.get("frames", []))
        for item in sorted(materials, key=lambda item: int(item["index"])):
            for order, (path, selection) in enumerate(zip(
                item["raw_keyframe_paths"], item["keyframe_selection"], strict=True,
            ), 1):
                frames.append({
                    **selection,
                    "path": str(path.resolve()),
                    "source_time_s": selection.get("source_time_s", 0),
                    "scene_id": selection.get("source_scene_id"),
                    "segment_index": int(item["index"]),
                    "frame_order": order,
                    "kind": "keyframe",
                })
        binding = {
            "source_sha256": self._sha256(source),
            "frames_sha256": self._json_sha256([
                {**entry, "sha256": self._sha256(Path(entry["path"]))}
                for entry in frames
            ]),
            "configuration": person_identity.identity_configuration_fingerprint(),
        }
        sidecar = project / "work" / "person_identity.json"
        if sidecar.is_file():
            try:
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                cached = {}
            if (cached.get("input_binding") == binding
                    and (cached.get("status") == "complete"
                         or (cached.get("status") == "skipped"
                             and cached.get("reason") ==
                             "person_identity_model_unavailable_or_invalid"))):
                log.event("person_identity_prepass_reused", status=cached["status"])
                return cached
        result = self._node(
            log, "person-identity-prepass", "project", binding,
            [Path(entry["path"]) for entry in frames],
            lambda: person_identity.analyze_person_identity(frames, source=binding),
            retry_if=lambda _exc: False,
        )
        result = {**result, "input_binding": binding}
        sidecar.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log.event("person_identity_prepass_completed", status=result["status"],
                  reason=result.get("reason"))
        return result

    def _prepare_segment_materials(
        self,
        project_id: str,
        project: Path,
        source: Path,
        segment: dict[str, Any],
        selection: list[dict[str, Any]],
        request: dict[str, Any],
        log: ResearchLog,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        del project_id
        index = int(segment["index"])
        scope = self._segment_scope(index)
        directory = project / "work" / "segments" / f"{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        segment_source = directory / "source.mp4"
        segment_frames = directory / "source-4fps"
        raw_dir = directory / "keyframes"
        if resume:
            if not (
                segment_source.is_file() and segment_source.stat().st_size > 0
            ):
                raise FileNotFoundError("reusable segment source is missing")
            segment_manifest = self._completed_node_value(
                project, "segment-frames-4fps", scope,
            )
            if not isinstance(segment_manifest, dict):
                raise TypeError("reusable segment manifest is invalid")
            frames = [
                path
                for path in sorted(raw_dir.glob("*.png"))
                if media.is_decodable_image(path)
            ]
            if len(frames) != len(selection):
                # Damaged/0-byte keyframes (e.g. ENOSPC mid-write) are not
                # evidence of completion: regenerate them from the source.
                log.event(
                    "keyframe_artifacts_regenerated",
                    scope=scope,
                    reusable=len(frames),
                )
                frames = self._node(
                    log,
                    "keyframes-exact-nine",
                    scope,
                    {"source": source, "selection": selection, "count": 9},
                    [source],
                    lambda: media.materialize_keyframes(source, selection, raw_dir),
                    lambda paths: list(paths),
                )
            else:
                log.event("segment_preparation_reused", scope=scope)
        else:
            self._node(
                log,
                "segment-cut",
                scope,
                {"source": source, **segment},
                [source],
                lambda: media.cut_segment(
                    source,
                    float(segment["start_s"]),
                    float(segment["duration_s"]),
                    segment_source,
                ),
                lambda _value: [segment_source],
            )
            segment_manifest = self._node(
                log,
                "segment-frames-4fps",
                scope,
                {"source": segment_source, "fps": 4, "output": segment_frames},
                [segment_source],
                lambda: media.extract_frames_4fps(segment_source, segment_frames),
                lambda value: self._manifest_media(segment_frames, value),
            )
            frames = self._node(
                log,
                "keyframes-exact-nine",
                scope,
                {"source": source, "selection": selection, "count": 9},
                [source],
                lambda: media.materialize_keyframes(source, selection, raw_dir),
                lambda paths: list(paths),
            )
            (directory / "keyframe_selection.json").write_text(
                json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        authoritative_frames = list(frames)

        proxy_dir = directory / "analysis-keyframes"
        proxy_dir.mkdir(exist_ok=True)

        def make_proxies() -> list[Path]:
            proxies: list[Path] = []
            for frame in authoritative_frames:
                target = proxy_dir / frame.name
                proxies.append(media.half_resolution_png(frame, target))
            return proxies

        reusable_proxies = (
            [
                path
                for path in sorted(proxy_dir.glob("*.png"))
                if media.is_decodable_image(path)
            ]
            if resume
            else []
        )
        if len(reusable_proxies) == 9:
            analysis_frames = reusable_proxies
            log.event("analysis_proxies_reused", scope=scope)
        else:
            analysis_frames = self._node(
                log,
                "video-maker-analysis-proxies-resume"
                if resume else "video-maker-analysis-proxies",
                scope,
                {"frames": authoritative_frames, "scale": "half-resolution"},
                authoritative_frames,
                make_proxies,
                lambda paths: list(paths),
            )
        return {
            **segment,
            "source": segment_source,
            "keyframe_paths": authoritative_frames,
            "raw_keyframe_paths": frames,
            "keyframe_selection": selection,
            "analysis_paths": analysis_frames,
            "segment_manifest": segment_manifest,
        }

    def _analyze_prepared_segment(
        self, project: Path, segment: dict[str, Any], request: dict[str, Any],
        log: ResearchLog, *, resume: bool = False,
    ) -> dict[str, Any]:
        index = int(segment["index"])
        scope = self._segment_scope(index)
        directory = project / "work" / "segments" / f"{index:04d}"
        analysis_frames = segment["analysis_paths"]
        user_references = segment.get("user_reference_paths", [])
        vision_images = [*user_references, *analysis_frames]
        selection = segment.get("keyframe_selection", [])
        visual_payload = {
            "generation_config": self._generation_config(request),
            "segment": {
                key: value
                for key, value in segment.items()
                if key not in {"visual_prompt", "dialogue", "source",
                               "keyframe_paths", "raw_keyframe_paths",
                               "analysis_paths", "segment_manifest",
                               "person_identity", "keyframe_selection", "user_reference_paths"}
            },
            "manifest": segment["segment_manifest"],
            "frames": [
                {
                    "frame_order": order,
                    "image_reference": f"Image {len(user_references) + order}",
                    "path": path.relative_to(project).as_posix(),
                    "sha256": self._sha256(path),
                    **(selection[order - 1] if order <= len(selection) else {}),
                }
                for order, path in enumerate(analysis_frames, 1)
            ],
        }
        if "person_identity" in segment:
            visual_payload["person_identity"] = segment["person_identity"]
        visual_payload["user_replacement_instruction"] = (request.get("replacement_guidance") or {}).get("instruction", "")
        visual_payload["user_reference_images"] = self._user_reference_metadata(user_references)
        visual_path = directory / "visual_prompt.txt"
        visual_prompt = visual_path.read_text(encoding="utf-8").strip() if resume and visual_path.is_file() else ""
        if visual_prompt:
            log.event("video_maker_artifact_reused", scope=scope)
        else:
            skill = VIDEO_SKILL.read_text(encoding="utf-8")
            visual_prompt = self._node(
                log,
                "video-maker-resume" if resume else "video-maker",
                scope,
                visual_payload,
                vision_images,
                lambda: self.deepseek.complete_structured(
                    skill
                    + "\n\n本次只执行逐段视觉分析；最终输出严格遵循本次注入的 JSON Schema。"
                    + ("\n如果提供 person_identity，请按原视频 source_time_s、frame_order 和"
                       "原图 bbox 将确认的 canonical_id 对应到人物；同一 ID 是同一人，"
                       "衣帽变化不能将其拆开。未确认不表示不同人；区间仅是采样支持。"
                       "不同 ID 也不是不同人的证明。"
                       if "person_identity" in segment else ""),
                    visual_payload,
                    VISUAL_PROMPT_SCHEMA,
                    vision_images,
                    validator=normalize_visual_prompt,
                ),
                transform=normalize_visual_prompt,
            )
            visual_path.write_text(visual_prompt, encoding="utf-8")
        return {
            **segment,
            "visual_prompt": visual_prompt,
            "status": "running",
        }

    def _translation_request(
        self,
        lines: list[dict[str, Any]],
        target_language: str,
        visual_prompts: str,
    ) -> ProviderResult[Any]:
        if not lines:
            return ProviderResult(
                value={"lines": []},
                record={"outcome": "skipped", "reason": "no_spoken_dialogue"},
                evidence_media=[],
            )
        if target_language == SAME_AS_SOURCE_LANGUAGE:
            return ProviderResult(
                value={
                    "lines": [{"text": str(line["text"])} for line in lines],
                },
                record={"outcome": "skipped", "reason": "same_language"},
                evidence_media=[],
            )
        instruction = (
            f"把台词翻译并自然改写为{target_language}。保持输入行顺序；"
            "输出行数必须与输入完全一致，lines 每项只含非空 text。"
            "最终输出严格遵循本次注入的 JSON Schema。"
            "视频描述仅用于语境。"
        )
        return self.deepseek.complete_structured(
            instruction,
            {"lines": lines, "video_prompts": visual_prompts},
            TRANSLATION_SCHEMA,
            validator=_validate_translation_shape,
        )

    @staticmethod
    def _normalize_translation(
        value: Any, source_lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Structured output owns the ideal shape. Do not turn translation
        # quality into a second local gate: unusable/missing entries fall back
        # to their source text, and surplus entries are ignored.
        parsed = value.get("lines") if isinstance(value, dict) else None
        candidates = parsed if isinstance(parsed, list) else []
        translated: list[dict[str, Any]] = []
        for index, source in enumerate(source_lines):
            candidate = candidates[index] if index < len(candidates) else None
            text = candidate.get("text") if isinstance(candidate, dict) else None
            if not isinstance(text, str) or not text.strip():
                text = str(source.get("text") or "")
            translated.append({
                **source,
                "text": text.strip(),
            })
        return translated

    @staticmethod
    def _dialogue_generation_payload(
        brief: str,
        prepared: list[dict[str, Any]],
        resolved_language: str,
    ) -> dict[str, Any]:
        segments = [
            {
                "index": int(item["index"]),
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
                "visual_prompt": str(item["visual_prompt"]),
            }
            for item in prepared
        ]
        duration_s = segments[-1]["end_s"] if segments else 0.0
        # The model never sees the source video, so only the concrete
        # resolved language goes into the payload; the same-as-source
        # provenance stays in state.json's effective_request.
        return {
            "brief": brief,
            "target_language": resolved_language,
            "duration_s": duration_s,
            "segments": segments,
        }

    def _dialogue_generation_request(
        self,
        brief: str,
        target_language: str,
        prepared: list[dict[str, Any]],
        resolved_language: str,
    ) -> ProviderResult[Any]:
        skill = DIALOGUE_WRITER_SKILL.read_text(encoding="utf-8")
        if target_language == SAME_AS_SOURCE_LANGUAGE:
            language_instruction = (
                f"台词语言为{resolved_language}；"
                "brief 与画面描述的语言不得影响台词语言。"
            )
        else:
            language_instruction = f"台词语言为{resolved_language}。"
        instruction = (
            skill
            + "\n\n本次执行口播台词生成；"
            + language_instruction
            + "最终输出严格遵循本次注入的 JSON Schema。"
        )

        def _validate(value: dict[str, Any]) -> None:
            _validate_dialogue_generation_shape(value)
            _validate_dialogue_language(value, resolved_language)

        return self.deepseek.complete_structured(
            instruction,
            self._dialogue_generation_payload(brief, prepared, resolved_language),
            DIALOGUE_GENERATION_SCHEMA,
            validator=_validate,
        )

    @staticmethod
    def _normalize_generated_dialogue(
        value: Any, prepared: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Same contract as translated_lines: [{start_s, end_s, text}] in
        # absolute source time. Unusable entries are dropped instead of
        # blocking the project; multiple lines inside one segment split that
        # segment's window by text length.
        parsed = value.get("lines") if isinstance(value, dict) else None
        candidates = parsed if isinstance(parsed, list) else []
        by_segment: dict[int, list[str]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            segment_index = candidate.get("segment_index")
            text = candidate.get("text")
            if (
                type(segment_index) is not int
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue
            by_segment.setdefault(segment_index, []).append(text.strip())
        generated: list[dict[str, Any]] = []
        for item in prepared:
            texts = by_segment.get(int(item["index"]))
            if not texts:
                continue
            start = float(item["start_s"])
            end = float(item["end_s"])
            total_chars = sum(len(text) for text in texts)
            for position, text in enumerate(texts):
                line_start = (
                    start
                    if position == 0
                    else generated[-1]["end_s"]
                )
                line_end = (
                    end
                    if position == len(texts) - 1
                    else start
                    + (end - start)
                    * sum(len(text) for text in texts[: position + 1])
                    / total_chars
                )
                generated.append({
                    "start_s": line_start,
                    "end_s": line_end,
                    "text": text,
                })
        return generated

    @staticmethod
    def _project_index_request(
        project: Path,
        prepared: list[dict[str, Any]],
        generation_config: dict[str, bool],
        *, user_replacement_instruction: str = "",
    ) -> tuple[dict[str, Any], list[Path]]:
        segments: list[dict[str, Any]] = []
        user_references = list(prepared[0].get("user_reference_paths", [])) if prepared else []
        images: list[Path] = list(user_references)
        for item in sorted(prepared, key=lambda value: int(value["index"])):
            frames = []
            for order, path in enumerate(item["analysis_paths"], 1):
                frames.append({
                    **(item.get("keyframe_selection", [])[order - 1]
                       if order <= len(item.get("keyframe_selection", [])) else {}),
                    "frame_order": order,
                    "image_reference": f"Image {len(images) + 1}",
                    "path": path.relative_to(project).as_posix(),
                    "sha256": Pipeline._sha256(path),
                })
                images.append(path)
            segments.append({
                "segment_index": int(item["index"]),
                "visual_prompt": item["visual_prompt"],
                "frames": frames,
            })
        payload = {
            "phase": "project_index",
            "generation_config": generation_config,
            "segments": segments,
            "user_replacement_instruction": user_replacement_instruction,
            "user_reference_images": Pipeline._user_reference_metadata(user_references),
        }
        evidence = [item.get("person_identity") for item in prepared
                    if isinstance(item.get("person_identity"), dict)]
        if evidence:
            payload["person_identity"] = {
                **evidence[0],
                "observations": [observation for item in evidence
                                 for observation in item.get("observations", [])],
            }
        return payload, images

    @staticmethod
    def _one_line(value: object, fallback: str = "unspecified") -> str:
        if not isinstance(value, str):
            return fallback
        normalized = " ".join(value.split())
        return normalized or fallback

    @staticmethod
    def _appears_in_segment(record: dict[str, Any], segment_index: int) -> bool:
        occurrences = record.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            # If occurrence metadata is unusable, including the record is less
            # destructive than blocking the project or silently losing it.
            return True
        return any(
            isinstance(occurrence, dict)
            and occurrence.get("segment_index") == segment_index
            for occurrence in occurrences
        )

    @classmethod
    def _segment_has_replaced_person(
        cls, element_index: dict[str, Any], segment_index: int
    ) -> bool:
        people = element_index.get("people")
        if not isinstance(people, dict):
            return False
        return any(
            isinstance(record, dict)
            and isinstance(record.get("replacement"), dict)
            and cls._appears_in_segment(record, segment_index)
            for record in people.values()
        )

    @classmethod
    def _replaced_object_targets(
        cls, element_index: dict[str, Any]
    ) -> list[str]:
        """Merged, deduplicated detection prompts of every OR-flagged entity.

        The frozen element index is project-wide and finalized before any
        segment processing starts, so the merged list is the same for every
        segment: per-segment occurrence filtering is deliberately NOT applied
        (2026-09-20 用户拍板"每段都带被替换物品的提示词，不怕多只怕少"）。
        occurrence 规划来自关键帧抽样，会漏掉边缘帧的短暂出镜（化妆品段2
        管子边缘露出但未规划进段2），漏遮会让 H3 看到源物品并抄回源外观；
        多带的代价只是该段检测 prompt 里多几个词，fal 调用次数不变（每段
        仍各一次，检测必须跑在各自的分段视频上）。
        Detection runs on the SOURCE video, so each target must name the
        source object with a short generic English label: long Chinese
        descriptions detect poorly (measured: fal SAM 3 found nothing in
        three segments with them, while the short label "handbag" hit 84%
        of frames on the same video). Prefer the replacement's
        ``detect_label``, which object-replace writes for exactly this
        purpose. Targets are deduplicated in insertion order. The detection
        artifact fingerprint contains the prompt, so label changes (including
        deduplication) naturally miss the cache and trigger a fresh detection
        call — that is intended, not a cache bug.
        Transient physical-effect labels are excluded as defense in depth:
        they carry no replaceable identity and must remain visible as Video 1
        motion reference. Normal Object Replace planning already omits them.
        Frozen indexes from before ``detect_label`` fall back to
        ``source_visual_description`` so they stay resumable; the fallback
        keeps the poor detection quality above and must never appear in
        new outputs (the schema requires ``detect_label`` for entities).
        Records with neither cannot be detected and are left out.
        """
        entities = element_index.get("entities")
        if not isinstance(entities, dict):
            return []
        targets: list[str] = []
        for record in entities.values():
            if not isinstance(record, dict) or not isinstance(
                record.get("replacement"), dict
            ):
                continue
            detect_label = record["replacement"].get("detect_label")
            if isinstance(detect_label, str) and detect_label.strip():
                normalized_label = detect_label.strip()
                if not any(
                    phrase.casefold() in _TRANSIENT_PHYSICAL_EFFECT_DETECT_LABELS
                    for phrase in normalized_label.split(", ")
                ):
                    targets.append(normalized_label)
                continue
            description = record.get("source_visual_description")
            if isinstance(description, str) and description.strip():
                normalized_description = description.strip()
                if (
                    normalized_description.casefold()
                    not in _TRANSIENT_PHYSICAL_EFFECT_DETECT_LABELS
                ):
                    targets.append(normalized_description)
        return list(dict.fromkeys(targets))

    @classmethod
    def _has_replaced_object(cls, element_index: dict[str, Any]) -> bool:
        return bool(cls._replaced_object_targets(element_index))

    @classmethod
    def _fusion_authority_block(
        cls,
        element_index: dict[str, Any],
        segment_index: int,
        replacement_sheet: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            "BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN",
            (
                "This block is the sole authority. Ignore any conflicting "
                "replacement or voice statement elsewhere in the prompt."
            ),
        ]
        sheet_reference: str | None = None
        sheet_tiles: dict[str, dict[str, Any]] = {}
        user_image_mapping = {
            f"image_{order}": reference["image_reference"]
            for order, reference in enumerate([
                reference for reference in (replacement_sheet or {}).get("h3_reference_images", [])
                if reference.get("role") == "user_replacement"
            ], 1)
        }
        if isinstance(replacement_sheet, dict):
            candidate_reference = replacement_sheet.get("h3_image_reference")
            spec = replacement_sheet.get("spec")
            tiles = spec.get("tiles") if isinstance(spec, dict) else None
            if isinstance(candidate_reference, str) and candidate_reference.strip():
                sheet_reference = cls._one_line(candidate_reference)
            if isinstance(tiles, list):
                sheet_tiles = {
                    tile["stable_key"]: tile
                    for tile in tiles
                    if isinstance(tile, dict)
                    and isinstance(tile.get("stable_key"), str)
                }
        if sheet_reference is not None and sheet_tiles:
            lines.append(
                f"The single global replacement appearance sheet is {sheet_reference}. "
                "For generated-variant REPLACE items, copy the new identity or design only "
                "from their own bound tile. For user-bound items, the explicitly mapped "
                "original user images are appearance authority and the tile is a locator. "
                "Never borrow unrelated tiles, user images, or source Video 1 appearance."
            )
            if replacement_sheet.get("user_image_present") is True:
                views = replacement_sheet.get("upstream_bindings", {}).get("replacement_views")
                if isinstance(views, dict):
                    # composed 模式：多图合成为 Image 2 一张合图。
                    lines.append(product_views.view_instruction(views["view_count"], "Image 2"))
                else:
                    # direct 模式：逐图直传，编号以冻结的 h3_reference_images 为准。
                    lines.append(
                        "User reference mapping: " + ", ".join(
                            f"{reference_id} -> {image_label}"
                            for reference_id, image_label in user_image_mapping.items()
                        ) + ". "
                        "Only the original references explicitly bound to each stable key "
                        "control that key; multiple references need not show the same object."
                    )
        replaced_ids: set[str] = set()
        has_person_replace = False
        has_entity_replace = False
        for category in ("people", "entities", "scenes"):
            records = element_index.get(category)
            if not isinstance(records, dict):
                continue
            for stable_id, record in records.items():
                if not isinstance(stable_id, str) or not isinstance(record, dict):
                    continue
                if isinstance(record.get("replacement"), dict):
                    replaced_ids.add(stable_id)
                if not cls._appears_in_segment(record, segment_index):
                    continue
                source = cls._one_line(record.get("source_visual_description"))
                replacement = record.get("replacement")
                if isinstance(replacement, dict):
                    target = cls._one_line(
                        replacement.get("target_visual_description")
                    )
                    tile = sheet_tiles.get(stable_id)
                    if sheet_reference is not None and isinstance(tile, dict):
                        # row/column 只有 sheet 渲染消费（网格坐标合成与图上
                        # ASCII TILE 标签，见 app/replacement_sheet.py），
                        # H3 模型按图上标签定位，prompt 文本不带冗余行列号。
                        reference_text = (
                            f"{sheet_reference}, "
                            f"{cls._one_line(tile.get('tile_id'))}"
                        )
                        if (
                            replacement.get("source") == "image_1"
                            and replacement_sheet.get("user_image_present") is True
                        ):
                            reference_ids = replacement.get("reference_image_ids") or ["image_1"]
                            unmapped = [
                                reference_id
                                for reference_id in reference_ids
                                if reference_id not in user_image_mapping
                            ]
                            if unmapped:
                                # 冻结的 h3_reference_images 不含绑定的 id，
                                # 猜标签只会把错误指引进 prompt。
                                raise ValueError(
                                    "replacement reference_image_ids not in the "
                                    "frozen user reference mapping: "
                                    + ", ".join(unmapped)
                                )
                            image_labels = ", ".join(
                                user_image_mapping[reference_id]
                                for reference_id in reference_ids
                            )
                            reference_text += (
                                f"; original user references {image_labels} are appearance authority; "
                                "use this tile only to locate this stable key; interpret the "
                                "reference set according to user_replacement_instruction"
                            )
                    else:
                        reference_text = (
                            "Image 1"
                            if replacement.get("source") == "image_1"
                            else "generated variant"
                        )
                    line = (
                        f"- {stable_id} | REPLACE | target: "
                        f"{target} | reference: {reference_text}"
                    )
                    if category == "people":
                        # 固定 preserve/person policy 尾部上提为块级全局行
                        # （循环后统一注入），条目不再逐条重复。
                        has_person_replace = True
                    elif category == "entities":
                        has_entity_replace = True
                    else:
                        # scene REPLACE 目前全项目 0 次出现，逐条尾挂保持现状。
                        line += (
                            " | preserve: source scene type, color tone, "
                            "composition, camera position and angle, shot "
                            "movement, focal length, and lighting"
                            " | scene policy: same scene type, use, and layout "
                            "but a clearly different instance and visible "
                            "design details; preserve color tone, composition, "
                            "camera, shot movement, focal length, and lighting"
                        )
                else:
                    preserve = record.get("preserve")
                    preserve_items = (
                        [
                            item for item in preserve
                            if isinstance(item, str) and item.strip()
                        ]
                        if isinstance(preserve, list)
                        else []
                    )
                    preserve_text = "; ".join(
                        cls._one_line(item) for item in preserve_items
                    ) or "source action, geometry, scale, and continuity"
                    line = (
                        f"- {stable_id} | KEEP | source: {source} | preserve: "
                        f"{preserve_text}"
                    )
                lines.append(line)

        # 同类 REPLACE 条目的固定尾部各上提为一条全局行（wrapper 行后、逐条
        # 条目前），措辞与原逐条尾部逐字一致；纯 KEEP 段不注入。
        global_rules: list[str] = []
        if has_person_replace:
            global_rules.append(
                "All replaced people share these rules | preserve: "
                f"{_FUSION_PERSON_REPLACE_PRESERVE} | person policy: "
                f"{_FUSION_PERSON_REPLACE_POLICY}"
            )
        if has_entity_replace:
            global_rules.append(
                "All replaced entities share these rules | preserve: "
                f"{_FUSION_ENTITY_REPLACE_PRESERVE} | entity policy: "
                f"{_FUSION_ENTITY_REPLACE_POLICY}"
            )
        lines[2:2] = global_rules

        relations = element_index.get("relations")
        relation_items = relations.items() if isinstance(relations, dict) else ()
        for stable_id, relation in relation_items:
            if not isinstance(stable_id, str) or not isinstance(relation, dict):
                continue
            if not cls._appears_in_segment(relation, segment_index):
                continue
            subject = cls._one_line(relation.get("subject_key"))
            predicate = cls._one_line(relation.get("predicate"))
            object_key = cls._one_line(relation.get("object_key"))
            preserve = relation.get("preserve")
            preserve_items = (
                [item for item in preserve if isinstance(item, str) and item.strip()]
                if isinstance(preserve, list)
                else []
            )
            preserve_text = "; ".join(
                cls._one_line(item) for item in preserve_items
            ) or "source relation state and geometry"
            replace_together = relation.get("replace_together")
            replace_together_text = (
                str(replace_together).lower()
                if isinstance(replace_together, bool)
                else "unspecified"
            )
            if subject in replaced_ids or object_key in replaced_ids:
                lines.append(
                    f"- {stable_id} | RELATION | endpoints: {subject} -> "
                    f"{object_key} | preserve: interaction event, relative scene "
                    "placement, and motion continuity; rebuild contact and occlusion "
                    "for replacement geometry | replace together: "
                    f"{replace_together_text}"
                )
            else:
                lines.append(
                    f"- {stable_id} | RELATION | {subject} {predicate} {object_key} "
                    f"| preserve: {preserve_text} | replace together: "
                    f"{replace_together_text}"
                )
        lines.append("END AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN")
        return "\n".join(lines)

    def _prepare_h3_segment(
        self,
        project: Path,
        item: dict[str, Any],
        element_index: dict[str, Any],
        replacement_sheet: dict[str, Any],
        request: dict[str, Any],
        log: ResearchLog,
        audio_analysis: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, Path, str]:
        index = int(item["index"])
        scope = self._segment_scope(index)
        try:
            skill = R2V_SKILL.read_text(encoding="utf-8")
            fusion_payload = {
                "segment_index": index,
                "user_replacement_instruction": (request.get("replacement_guidance") or {}).get("instruction", ""),
                "visual_prompt": item["visual_prompt"],
                "global_element_index": element_index,
                "translated_dialogue": item["dialogue"],
                "h3_reference_images": replacement_sheet.get(
                    "h3_reference_images", []
                ),
                "user_reference_images": [{
                    "reference_id": f"image_{order}",
                    "image_reference": reference["image_reference"],
                    "sha256": reference["sha256"],
                } for order, reference in enumerate([
                    reference for reference in replacement_sheet.get("h3_reference_images", [])
                    if reference.get("role") == "user_replacement"
                ], 1)],
                "replacement_sheet": {
                    "spec_sha256": replacement_sheet.get("spec_sha256"),
                    "h3_image_reference": replacement_sheet.get(
                        "h3_image_reference"
                    ),
                    "tiles": (
                        (replacement_sheet.get("spec") or {}).get("tiles", [])
                        if isinstance(replacement_sheet.get("spec"), dict)
                        else []
                    ),
                },
                "authoritative_constraints": self._fusion_authority_block(
                    element_index, index, replacement_sheet,
                ),
            }
            # audio_analysis 注入 fusion payload（B2 产物逐行语气）：当前正式
            # skill（V2 产品突出版，2026-09-28 多版本对照定稿）未消费此字段，
            # 保留供表演类版本迭代；若长期不用应删除。
            audio_analysis_rows = _fusion_audio_analysis_rows(
                item.get("start_s"), item.get("end_s"), audio_analysis,
            )
            if audio_analysis_rows is not None:
                fusion_payload["audio_analysis"] = audio_analysis_rows
            fusion_schema, fusion_input_sha256 = fusion_prompt_schema(
                fusion_payload,
            )
            fusion_agent, fusion_client = self._fusion_client_for_request(request)
            fusion_images = [project / reference["path"] for reference in replacement_sheet.get("h3_reference_images", [])]
            # 段内有绑定用户参考图的被替换商品时，fusion 正文必须带【属性与
            # 用法】段（硬门禁，见 _require_fusion_usage_paragraphs  docstring
            # 根因记录）；validator 与 transform 同一标准，缓存复用不豁免。
            user_bound_product = any(
                isinstance(entry, dict)
                and entry.get("replacement")
                and entry.get("target_reference_image_ids")
                and any(
                    isinstance(occurrence, dict)
                    and occurrence.get("segment_index") == index
                    for occurrence in entry.get("occurrences") or []
                )
                for entry in (element_index.get("entities") or {}).values()
            )

            def _validate_fusion_prompt(value: Any) -> str:
                normalized = normalize_fusion_prompt(
                    value,
                    input_sha256=fusion_input_sha256,
                    segment_index=index,
                    authoritative_constraints=fusion_payload[
                        "authoritative_constraints"
                    ],
                )
                _require_fusion_usage_paragraphs(
                    normalized, user_bound_product=user_bound_product,
                )
                return normalized

            prompt = self._node(
                log,
                "h3-max-r2v-prompt",
                scope,
                {
                    "agent": fusion_agent,
                    "skill": skill,
                    "input_sha256": fusion_input_sha256,
                    "payload": fusion_payload,
                    "output_schema": fusion_schema,
                },
                fusion_images,
                lambda: fusion_client.complete_structured(
                    skill,
                    fusion_payload,
                    fusion_schema,
                    fusion_images,
                    validator=_validate_fusion_prompt,
                ),
                transform=_validate_fusion_prompt,
            )
            segment_dir = item["source"].parent
            muted = segment_dir / "source-muted.mp4"
            self._node(
                log,
                "mute-reference-video",
                scope,
                {"source": item["source"], "output": muted},
                [item["source"]],
                lambda: media.mute(item["source"], muted),
                lambda _value: [muted],
            )
            objects_masked = False
            mask_stats: dict[str, Any] | None = None
            object_targets = self._replaced_object_targets(element_index)
            # 人物组与物品组分开检测、分类渲染：fal/wavespeed 的逗号多目标
            # 返回逐帧合并 mask，无法归因到具体词，而人物块必须 100% 不透明
            # （2% 残影即身份泄漏实测）物品块 99%+ 模糊（保动作锚点），只能
            # 分组调用（每段最多 2 次检测，2026-09-20 用户拍板接受）。
            # 人物措辞经真实校准："person, hair" 可检出；"human"/"human
            # face"/"human hair"/"human head" 全零检出
            #（data-verify/wavespeed-calib-20260919 round3）。
            person_targets: list[str] = []
            if self._segment_has_replaced_person(element_index, index):
                person_targets = ["person", "hair"]
            if (object_targets or person_targets) and not self.settings.object_mask_enabled:
                # OBJECT_MASK_ENABLED=false（检测后端不可用期间临时整体关闭：
                # 2026-09-19 fal 账号被锁、sandbase worker 连续超时，wavespeed
                # 已接入并当日完成输出标定、尚未在真实项目启用）：
                # 不发起任何检测调用，产物形状与 object_mask_degraded 降级
                # 路径完全一致——参考视频保持 source-muted、objects_masked 保持
                # False（提示词按未遮盖处理），仅事件名区分"配置关闭"与"失败"。
                log.event(
                    "object_mask_disabled",
                    scope=scope,
                    targets=object_targets + person_targets,
                    reason="OBJECT_MASK_ENABLED=false",
                    impact=(
                        "replaceable faces and objects stay visible in "
                        "the reference video; H3 may copy their source "
                        "appearance"
                    ),
                )
            elif (
                object_targets or person_targets
            ) and object_detect.available_providers(self.settings):
                # This unified SAM node degrades on ANY failure: after the
                # node's own retries the render continues on the unmasked
                # stream. No configured provider means it never runs (the
                # missing-key case); a misconfigured explicitly enabled
                # provider is rejected earlier by Settings validation.
                object_masked = segment_dir / "source-objects-masked.mp4"
                # 检测产物落盘目录 = 段 work 目录（跨节点重试与
                # resume-pre-picsart 续跑存活）：解析后的契约 detections 收
                # object-detections.json，计入下面 output-media 的节点产物
                # 清单。检测原始响应的指纹缓存写实例级共享目录
                # <DATA_DIR>/shared/object-detect-cache/（跨项目同指纹零
                # HTTP，见 object_detect 的落盘纪律注释），段目录下的
                # object-detect-response.json 仅作项目级缓存的回退读。
                detect_artifacts_dir = segment_dir
                object_mask_failures = 0

                def _object_mask_retry(exc: Exception) -> bool:
                    nonlocal object_mask_failures
                    object_mask_failures += 1
                    # 确定性错误（fal 未校准响应形态：重提只会得到同样结果）
                    # 一次尝试即降级，不退避重试——与
                    # _retryable_node_error 对同一 cause 的分类一致。
                    if isinstance(exc, ProviderFailure) and isinstance(
                        exc.cause, object_detect.UncalibratedResponseError
                    ):
                        return False
                    return object_mask_failures < object_mask.OBJECT_MASK_MAX_ATTEMPTS

                try:
                    mask_stats = self._node(
                        log,
                        "mask-replaceable-objects",
                        scope,
                        {
                            "source": muted,
                            "output": object_masked,
                            "targets": object_targets,
                            "person_targets": person_targets,
                            "block_opacity": self.settings.object_mask_block_opacity,
                            "block_blur": self.settings.object_mask_block_blur,
                            # 分类参数进节点输入指纹：改动即触发重渲。
                            "person_block_opacity": (
                                self.settings.object_mask_person_block_opacity
                            ),
                        },
                        [muted],
                        lambda: object_mask.mask_objects(
                            muted, object_masked, object_targets,
                            person_targets=person_targets,
                            detect_artifacts_dir=detect_artifacts_dir,
                            block_opacity=self.settings.object_mask_block_opacity,
                            block_blur=self.settings.object_mask_block_blur,
                            person_block_opacity=(
                                self.settings.object_mask_person_block_opacity
                            ),
                        ),
                        lambda _value: [
                            path
                            for path in (
                                object_masked,
                                *sorted(segment_dir.glob("object-detect*.json")),
                            )
                            if path.is_file()
                        ],
                        retry_if=_object_mask_retry,
                    )
                except Exception as exc:
                    log.event(
                        "object_mask_degraded",
                        scope=scope,
                        attempts=object_mask_failures,
                        error=str(exc),
                        impact=(
                            "replaceable faces and objects stay visible in "
                            "the reference video; H3 may copy their source "
                            "appearance"
                        ),
                    )
                else:
                    if mask_stats["blocks_total"] > 0:
                        muted = object_masked
                        objects_masked = True
                    else:
                        # Every target exhausted its detection retries and was
                        # skipped: the re-encoded video carries no block, so
                        # keep the previous stream and prompt as unmasked.
                        log.event(
                            "object_mask_no_blocks",
                            scope=scope,
                            targets=object_targets + person_targets,
                            stats=mask_stats,
                            impact=(
                                "no target could be located, so no mask "
                                "block exists in the reference video"
                            ),
                        )
            # h3-context-ir 蒸馏输入的 [遮挡块] 小节数据画像：与下方提示词
            # mask_block 同一套 person/object 判定，保证蒸馏语义与参考流一致。
            item["h3_mask_profile"] = {
                "objects_masked": objects_masked,
                "person_blocked": (
                    bool(person_targets) if objects_masked else False
                ),
                "object_blocked": (
                    bool(object_targets) if objects_masked else False
                ),
                "object_semi_transparent": (
                    self.settings.object_mask_block_opacity < 1.0
                ),
            }
            if objects_masked and mask_stats is not None and (
                "person_frames_with_block" in mask_stats
            ):
                item["h3_mask_profile"]["person_blocked"] = (
                    mask_stats["person_frames_with_block"] > 0
                )
                item["h3_mask_profile"]["object_blocked"] = (
                    mask_stats["object_frames_with_block"] > 0
                )
            # The prompt is assembled only after the mask node so its unified
            # mask-block clause matches what the reference stream actually
            # contains: it appears only when SAM painted at least one block.
            # The frozen h3_prompt.txt is replayed verbatim on resume,
            # keeping the decision stable.
            processing = self._validated_effective_request(request)["processing"]
            has_replaced_content = (
                self._segment_has_replaced_person(element_index, index)
                or self._has_replaced_object(element_index)
            )
            editing_block = (
                "This is an editing task on Video 1, not a generation task: keep "
                "every shot, camera move, action, timing, transition and all "
                "unreplaced content exactly as in Video 1; only the elements "
                "described below are replaced.\n\n"
                if has_replaced_content
                else ""
            )
            face_block = ""
            sheet_reference = replacement_sheet.get("h3_image_reference")
            masks_faces = (
                isinstance(sheet_reference, str)
                and bool(sheet_reference.strip())
                and self._segment_has_replaced_person(element_index, index)
            )
            if masks_faces:
                # user-bound/generated_variant 的图片来源表述由账本块的 sheet
                # 引用行承载（更完整，含 tile locator 语义），此处只保留人物
                # 外观权威与失败判级，不重复映射关系（masks_faces 蕴含 sheet
                # 存在：有人物块 ⇒ 有被替换人物 ⇒ spec 有 tile ⇒
                # h3_image_reference 非空）。
                face_block = (
                    "PERSON REPLACEMENT IS MANDATORY AND TAKES PRIORITY OVER "
                    "SOURCE IDENTITY. Face shape, eyes, nose, mouth, skin "
                    "tone, hairstyle, body build, and clothing follow the "
                    "per-key appearance authority bound in the plan below, "
                    "never Video 1. Keeping any part of the source person's "
                    "appearance is a failure, even if everything else is "
                    "correct.\n\n"
                )
            mask_block = ""
            if objects_masked:
                # 文案必须匹配参考流的真实内容：人物块 100% 不透明纯黑（防身份
                # 泄漏），物品块在 opacity<1.0 时是半透明模糊白块（残影仅作
                # 位置/运动/构图锚点）。两组覆盖率取 mask_stats 的分组计数；
                # 无分组计数时按请求的组兜底。
                person_blocked = bool(person_targets)
                object_blocked = bool(object_targets)
                if mask_stats is not None and "person_frames_with_block" in mask_stats:
                    person_blocked = (
                        mask_stats["person_frames_with_block"] > 0
                    )
                    object_blocked = (
                        mask_stats["object_frames_with_block"] > 0
                    )
                if (
                    person_blocked
                    and object_blocked
                    and self.settings.object_mask_block_opacity < 1.0
                ):
                    # 混合模式：人物块纯黑不透明 + 物品块半透明白。残影意图
                    # 说死，否则模型会把淡残留当成暗光真实画面去增强重建。
                    # 两句禁令语义不同不合并（对账见
                    # data-verify/prompt-slim-20260929）：
                    # "never reconstruct...visible through the blocks" 针对
                    # 半透物品块的透出内容（25e90ae 事故加硬）；"never draw or
                    # restore what any block hides" 针对不透明人物块的隐藏
                    # 内容（09541f0/27b8a57 防身份泄漏加硬）。人物外观清单由
                    # face_block 完整承载，此处只留短句。
                    mask_block = (
                        "Solid black (0,0,0) blocks fully conceal the people "
                        "to replace — face, hair, body and clothing included; "
                        "semi-transparent white (255,255,255) blocks mask "
                        "replaceable objects. The faint blurred source imagery "
                        "showing through the object blocks serves ONLY as "
                        "position, motion, and composition reference: never "
                        "reconstruct, enhance, sharpen, or copy anything "
                        "visible through the blocks; never draw or restore "
                        "what any block hides: rebuild people naturally from "
                        "their descriptions and rebuild objects only from "
                        "the provided reference images. Each rebuilt object "
                        "must match its reference image in identity, shape, "
                        "size, proportions and orientation, behave according "
                        "to its physical nature, and be held, worn or used in "
                        "a way consistent with its function; never reshape, "
                        "rescale or repurpose it to fit the source object it "
                        "replaces.\n\n"
                    )
                elif (
                    object_blocked
                    and not person_blocked
                    and self.settings.object_mask_block_opacity < 1.0
                ):
                    # 仅物品组半透明：文案不提人物块（参考流里没有）。此分支
                    # 只有半透物品块，"never draw or restore what they hide"
                    # 与透出禁令指向同一批块、语义重复，合并保留更强表述。
                    mask_block = (
                        "Semi-transparent white (255,255,255) blocks mask "
                        "replaceable objects. "
                        "The faint blurred source imagery showing through the blocks "
                        "serves ONLY as position, motion, and composition "
                        "reference: never reconstruct, enhance, sharpen, or "
                        "copy anything visible through the blocks; rebuild "
                        "objects only from "
                        "the provided reference images. Each rebuilt object "
                        "must match its reference image in identity, shape, "
                        "size, proportions and orientation, behave according "
                        "to its physical nature, and be held, worn or used in "
                        "a way consistent with its function; never reshape, "
                        "rescale or repurpose it to fit the source object it "
                        "replaces.\n\n"
                    )
                else:
                    # Keep the compact mask instruction first after the fixed
                    # editing framework, and leave the following PERSON
                    # REPLACEMENT block to carry identity-reference binding
                    # details. 人物外观清单由 face_block 完整承载，此处只留
                    # 短句（有人物块 ⇒ face_block 存在，见 masks_faces 注释）。
                    mask_block = (
                        "Solid black (0,0,0) blocks fully conceal the people "
                        "to replace — face, hair, body and clothing included; "
                        "white blocks mask replaceable objects; never draw or "
                        "restore what they hide: rebuild people naturally from "
                        "their descriptions and rebuild objects only from "
                        "the provided reference images. Each rebuilt object "
                        "must match its reference image in identity, shape, "
                        "size, proportions and orientation, behave according "
                        "to its physical nature, and be held, worn or used in "
                        "a way consistent with its function; never reshape, "
                        "rescale or repurpose it to fit the source object it "
                        "replaces.\n\n"
                    )
            overlay_targets = []
            if processing["remove_subtitle"]:
                overlay_targets.append(
                    "burned-in subtitles, captions, and on-screen text"
                )
            if processing["remove_logo"]:
                overlay_targets.append("watermarks and logos")
            overlay_block = ""
            if overlay_targets:
                overlay_block = (
                    "\n\nOverlay removal: Video 1 may contain "
                    + " and ".join(overlay_targets)
                    + ". Remove all of them from the generated video and "
                    "reconstruct the occluded background naturally. Do not "
                    "reproduce, translate, or re-render any source on-screen "
                    "text, watermark, or logo."
                )
            dialogue = "\n".join(
                f"[{line['start_s']:.3f}-{line['end_s']:.3f}s] {line['text']}"
                for line in item["dialogue"]
            )
            # voice-over（幕后口播）模式：两种及以上音色时台词是单一音色旁白
            # 画外音，不做口型、H3 不挂 Audio 1；单音色保持对口型语义。
            voice_over = dialogue and self._dialogue_voice_over_enabled(
                project.name
            )
            dialogue_suffix = (
                (
                    "\n\nDialogue requirements: The spoken lines are a "
                    "single-voice narrator voice-over in the target "
                    "language, not the speech of any on-screen person. No "
                    "person's mouth movements or timing follow the "
                    "narration: people act and demonstrate naturally "
                    "without mouthing its words. Do not generate any "
                    "speech beyond the narration, do not speak the "
                    "original-language dialogue, and do not render "
                    "captions or subtitles. Do not add dialogue in "
                    "padded time. The narrator's lines are:\n" + dialogue
                )
                if voice_over
                else (
                    "\n\nDialogue requirements: Audio 1 is this segment's "
                    "pre-recorded dialogue speech track in the target language. "
                    "Every speaking person must lip-sync precisely to Audio 1: "
                    "match its timing, rhythm, and mouth movements exactly, and do "
                    "not change its words. Do not generate any speech beyond "
                    "Audio 1, do not speak the original-language dialogue, and do "
                    "not render captions or subtitles. Do not add dialogue in "
                    "padded time. The spoken lines in Audio 1 are:\n" + dialogue
                    if dialogue
                    else "\n\nDo not add captions, subtitles, or new dialogue."
                )
            )
            # 三段结构：prefix/suffix 是后端机械注入（用户不可见不可编辑），
            # body 是 DeepSeek fusion 主体。body 里从 BEGIN AUTHORITATIVE
            # OBJECT REPLACEMENT AND VOICE PLAN 整行起到末尾的权威计划块同样
            # 不可见不可编辑——审核落盘时由 _write_prompt_review 再拆成
            # editable + hidden。final_prompt 由三段拼接而成，与历史组装逐字
            # 节一致。
            prompt_prefix = editing_block + mask_block + face_block
            prompt_suffix = overlay_block + dialogue_suffix
            final_prompt = prompt_prefix + prompt + prompt_suffix
            item["prompt_parts"] = {
                "prefix": prompt_prefix,
                "body": prompt,
                "suffix": prompt_suffix,
            }
            (segment_dir / "h3_prompt.txt").write_text(
                final_prompt, encoding="utf-8",
            )
            return item, final_prompt, muted, fusion_input_sha256
        except Exception:
            item["status"] = "failed"
            raise

    @staticmethod
    def _validated_provider_asset_url(value: object) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("prepared provider asset URL is invalid")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
            raise ValueError("prepared provider asset URL must be public HTTPS")
        return value

    @staticmethod
    def _validate_reference_video(path: Path) -> None:
        info = media.probe(path)
        duration = info.get("duration_s") if isinstance(info, dict) else None
        width = info.get("width") if isinstance(info, dict) else None
        height = info.get("height") if isinstance(info, dict) else None
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not PICSART_REFERENCE_MIN_DURATION_S
            <= float(duration)
            <= PICSART_REFERENCE_MAX_DURATION_S
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
        ):
            raise ValueError("prepared Picsart reference video is invalid")

    def _prepare_picsart_segment(
        self,
        project_id: str,
        project: Path,
        item: dict[str, Any],
        replacement: Path | None,
        replacement_sheet: Path | None,
        request: dict[str, Any],
        log: ResearchLog,
        *,
        final_prompt: str,
        muted: Path,
        node_prefix: str = "picsart",
        artifact_suffix: str = "",
        existing_task_id: str | None = None,
    ) -> PreparedPicsartRequest:
        index = item.get("index")
        duration = item.get("duration_s")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 1
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("Picsart segment metadata is invalid")
        scope = self._segment_scope(index)
        segment_dir = project / "work" / "segments" / f"{index:04d}"
        muted_receipt = self._project_artifact_receipt(
            project,
            muted,
            f"work/segments/{index:04d}/{self._muted_artifact_name(muted)}",
        )
        self._validate_reference_video(muted)
        if not isinstance(final_prompt, str):
            raise ValueError("prepared Picsart prompt is invalid")
        final_prompt = final_prompt.strip()
        if not final_prompt or len(final_prompt) > PICSART_PROMPT_MAX_CHARS:
            raise ValueError("prepared Picsart prompt is invalid")
        project_root = project.resolve(strict=True)
        dialogue_track: Path | None = None
        # voice-over 段不传 reference_audio_urls（与无台词段同为空列表）。
        # 例外：自托管 H3 服务端强校验 prompt 媒体引用必须存在于 references，
        # voice-over 段冻结 prompt 仍引用 <Audio 1>，selfhosted 模式下始终
        # 附带台词轨（Picsart 路径行为不变）。
        if item.get("dialogue") and (
            self.settings.h3_provider == "selfhosted"
            or not self._dialogue_voice_over_enabled(project_id)
        ):
            candidate = segment_dir / "dialogue-track.wav"
            resolved_track = candidate.resolve(strict=True)
            if (
                project_root not in resolved_track.parents
                or resolved_track
                != (
                    project / "work" / "segments" / f"{index:04d}"
                    / "dialogue-track.wav"
                ).resolve()
                or candidate.is_symlink()
                or not resolved_track.is_file()
                or resolved_track.stat().st_size < 1
            ):
                raise ValueError("prepared H3 dialogue track is invalid")
            dialogue_track = resolved_track
        if existing_task_id is not None and (
            not isinstance(existing_task_id, str)
            or not existing_task_id
            or existing_task_id != existing_task_id.strip()
        ):
            raise ValueError("stored Picsart task id is invalid")
        request = self._validated_effective_request(request)
        output_request = request["output"]
        resolution = output_request["resolution"]
        aspect_ratio = output_request["aspect_ratio"]
        if not isinstance(resolution, str) or resolution not in PICSART_RESOLUTIONS:
            raise ValueError("Picsart resolution is unsupported")
        if (
            not isinstance(aspect_ratio, str)
            or aspect_ratio not in PICSART_ASPECT_RATIOS
        ):
            raise ValueError("Picsart aspect ratio is unsupported")
        provider_duration = max(
            PICSART_GENERATION_MIN_DURATION_S,
            math.ceil(float(duration)),
        )
        if not (
            PICSART_GENERATION_MIN_DURATION_S
            <= provider_duration
            <= PICSART_GENERATION_MAX_DURATION_S
        ):
            raise ValueError("Picsart generation duration is unsupported")

        provider_images: list[Path] = []
        video_url: str | None = None
        image_urls: list[str] = []
        audio_urls: list[str] = []
        binding_state = self.store.read(project_id)
        prepared_manifest = binding_state.get("h3_prepared_artifacts")
        prepared_manifest_version = (
            prepared_manifest.get("version")
            if isinstance(prepared_manifest, dict)
            else None
        )
        if prepared_manifest_version == H3_PREPARED_ARTIFACTS_VERSION:
            preparation_sha256 = self._json_sha256(prepared_manifest)
        else:
            raise ValueError("stored H3 preparation manifest is invalid")
        if existing_task_id is None:
            stored_state = binding_state
            stored_sheet = stored_state.get("replacement_sheet")
            if not isinstance(stored_sheet, dict):
                raise ValueError("stored replacement sheet state is invalid")
            validated_sheet = self._validated_stored_replacement_sheet(
                project, stored_state,
            )
            supplied_sheet = (
                replacement_sheet.resolve(strict=True)
                if replacement_sheet is not None
                else None
            )
            if supplied_sheet != validated_sheet:
                raise ValueError(
                    "prepared Picsart image binding does not match"
                )
            provider_images.extend(
                (project / reference["path"]).resolve(strict=True)
                for reference in stored_sheet["h3_reference_images"]
            )
            if len(provider_images) != len(set(provider_images)):
                raise ValueError("prepared Picsart image files must be unique")
            video_url = self._validated_provider_asset_url(
                self.register_asset(project_id, muted)
            )
            image_urls = [
                self._validated_provider_asset_url(
                    self.register_asset(project_id, provider_image)
                )
                for provider_image in provider_images
            ]
            if len(image_urls) != len(set(image_urls)):
                raise ValueError("prepared Picsart image URLs must be unique")
            audio_urls = (
                [
                    self._validated_provider_asset_url(
                        self.register_asset(project_id, dialogue_track)
                    )
                ]
                if dialogue_track is not None
                else []
            )
        request_payload = (
            {
                "existing_task_id": existing_task_id,
                "segment_index": index,
                "muted_video": muted_receipt,
            }
            if existing_task_id is not None
            else {
                "workflow": PicsartClient.WORKFLOW,
                "prompt": final_prompt,
                "reference_video_urls": [video_url],
                "reference_image_urls": image_urls,
                "reference_audio_urls": audio_urls,
                "resolution": resolution.upper(),
                "duration": provider_duration,
                "aspect_ratio": aspect_ratio,
                "prompt_expansion_mode": "quality",
            }
        )
        prepared = PreparedPicsartRequest(
            project_id=project_id,
            segment_index=index,
            scope=scope,
            run_identity=node_prefix,
            artifact_suffix=artifact_suffix,
            final_prompt=final_prompt,
            muted=muted.resolve(strict=True),
            provider_images=tuple(provider_images),
            dialogue_track=dialogue_track,
            video_url=video_url,
            image_urls=tuple(image_urls),
            audio_urls=tuple(audio_urls),
            provider_duration=provider_duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            target_duration_s=float(duration),
            request_sha256=self._json_sha256(request_payload),
            preparation_sha256=preparation_sha256,
            existing_task_id=existing_task_id,
        )
        log.event(
            "picsart_preflight_succeeded",
            segment_index=index,
            run_identity=node_prefix,
            request_sha256=prepared.request_sha256,
            reuses_task=existing_task_id is not None,
        )
        return prepared

    def _submit_prepared_picsart(
        self,
        prepared: PreparedPicsartRequest,
    ) -> ProviderResult[str]:
        if prepared.existing_task_id is not None or prepared.video_url is None:
            raise ValueError("prepared Picsart request is not submittable")
        try:
            result = self.picsart.submit(
                prompt=prepared.final_prompt,
                video_url=prepared.video_url,
                image_urls=list(prepared.image_urls),
                audio_urls=list(prepared.audio_urls),
                duration=prepared.provider_duration,
                resolution=prepared.resolution,
                aspect_ratio=prepared.aspect_ratio,
            )
            if (
                not isinstance(result, ProviderResult)
                or not isinstance(result.value, str)
                or not result.value.strip()
            ):
                raise ValueError("Picsart submit returned no task id")
            self._transition_submission_receipt(
                prepared, status="accepted", task_id=result.value,
                api_key_role=result.record.get("api_key_role"),
            )
            return result
        except ProviderFailure as exc:
            outcome = exc.record.get("outcome")
            self._transition_submission_receipt(
                prepared,
                status="rejected" if outcome == "rejected" else "unknown",
            )
            raise
        except BaseException:
            try:
                self._transition_submission_receipt(
                    prepared, status="unknown",
                )
            except BaseException:
                # The pre-POST submission_started receipt remains fail-closed.
                pass
            raise

    def _generate_picsart_segment(
        self,
        prepared: PreparedPicsartRequest,
        item: dict[str, Any],
        log: ResearchLog,
    ) -> Path:
        index = prepared.segment_index
        scope = prepared.scope
        segment_dir = prepared.muted.parent
        node_prefix = prepared.run_identity
        artifact_suffix = prepared.artifact_suffix
        try:
            current = prepared
            task_id = prepared.existing_task_id
            # 任务账户隔离：task 由哪把 key 创建（含 402 failover），poll/
            # result/download 都用同一把；role 唯一来源是落盘的回执。
            task_key_role = [
                (
                    self._state_submission_api_key_role(
                        self.store.read(prepared.project_id), index,
                    )
                    if task_id is not None
                    else "primary"
                )
            ]
            resubmits = 0
            while True:
                # Submit/poll node names carry the run identity, so every
                # (re)submission keeps its own complete receipt/log chain.
                submit_prefix = current.run_identity
                task_started = time.monotonic()
                # 每个任务独立的连续传输失败计数（封顶即 fail-closed 不重提）。
                transport_failures = [0]
                if task_id is None:
                    task_id = self._node(
                        log,
                        f"{submit_prefix}-submit",
                        scope,
                        current.submit_input(),
                        [current.muted]
                        + list(current.provider_images)
                        + ([current.dialogue_track] if current.dialogue_track is not None else []),
                        lambda current=current: self._submit_prepared_picsart(current),
                        retry_if=lambda _exc: False,
                    )
                    task_key_role[0] = self._state_submission_api_key_role(
                        self.store.read(prepared.project_id), index,
                    )
                else:
                    log.event(
                        "picsart_task_reused",
                        segment_index=index,
                        task_id=task_id,
                    )

                def poll_once(
                    task_id: str = task_id,
                    current: PreparedPicsartRequest = current,
                    task_started: float = task_started,
                    transport_failures: list[int] = transport_failures,
                    task_key_role: list[str] = task_key_role,
                ) -> ProviderResult[str]:
                    if (
                        time.monotonic() - task_started
                        > self.settings.picsart_task_hang_limit_s
                    ):
                        raise PicsartTaskHungError(
                            f"Picsart task {task_id} stayed non-terminal past "
                            f"{self.settings.picsart_task_hang_limit_s}s"
                        )
                    try:
                        return self.picsart.poll(
                            task_id, api_key_role=task_key_role[0],
                        )
                    except ProviderFailure as exc:
                        if self._poll_record_is_terminal_failure(exc.record):
                            self._mark_submission_task_failed(
                                current.project_id, index, task_id,
                            )
                            raise
                        # 连续传输级失败封顶：无终态回执证据时不拿新付费
                        # 任务去赌供应商状态，fail-closed 等人工核实。
                        transport_failures[0] += 1
                        if (
                            transport_failures[0]
                            >= PICSART_POLL_TRANSPORT_FAILURE_LIMIT
                        ):
                            log.event(
                                "picsart_poll_transport_failure_limit",
                                segment_index=index,
                                task_id=task_id,
                                consecutive_failures=transport_failures[0],
                                error=str(exc)[:300],
                            )
                            raise PicsartPollTransportFailureLimitError(
                                f"Picsart task {task_id} poll 连续 "
                                f"{transport_failures[0]} 次传输级失败"
                                "（无终态语义），供应商状态未知，"
                                "fail-closed 需人工核实；回执保持 accepted，"
                                "核实后可走 resume-picsart 继续 poll 该 task"
                            ) from exc
                        raise

                try:
                    generated_url = self._node(
                        log,
                        f"{submit_prefix}-poll",
                        scope,
                        {"task_id": task_id},
                        [],
                        poll_once,
                        retry_if=lambda exc: not (
                            isinstance(exc, (
                                PicsartTaskHungError,
                                PicsartPollTransportFailureLimitError,
                            ))
                            or (
                                isinstance(exc, ProviderFailure)
                                and self._poll_record_is_terminal_failure(
                                    exc.record
                                )
                            )
                        ),
                    )
                except (ProviderFailure, PicsartTaskHungError) as exc:
                    hung = isinstance(exc, PicsartTaskHungError)
                    if not hung and not self._poll_record_is_terminal_failure(
                        exc.record
                    ):
                        raise
                    if hung:
                        self._mark_submission_task_failed(
                            current.project_id, index, task_id,
                        )
                        reason = "task_hung"
                        error_text = str(exc)
                    elif self._poll_failure_is_request_validation(exc.record):
                        # 内容审核 422（request validation）纳入同一有限重提
                        # （2026-09-29 决策）：审核器是概率性判定，同文重提
                        # 可能通过——实证：项目 f0997e14 段 3 同一 prompt
                        # 逐字节相同，2026-09-14 连续 3 次 422 后第 4 次原样
                        # 重提 COMPLETED；项目 d7eb0406 段 5 2026-09-28 422
                        # 后 resume 原样重提一次通过。审核不过供应商不扣费，
                        # 重试零成本；上限 PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS
                        # 兜底真正确定性的失败场景。每次失败仍记
                        # picsart_request_validation_failed 事件；accepted
                        # 任务的终态回执已由 poll_once 落账。
                        log.event(
                            "picsart_request_validation_failed",
                            segment_index=index,
                            failed_task_id=task_id,
                        )
                        reason = "request_validation"
                        error_text = str(exc)
                    else:
                        # 终态 5xx 错误体的供应商 reason（如 file_copy_failed）
                        # 优先落事件；取不到再按 asset_fetch/通用终态归类。
                        provider_reason = self._poll_failure_provider_reason(
                            exc.record
                        )
                        reason = (
                            "asset_fetch"
                            if self._poll_failure_is_asset_fetch(exc.record)
                            else (provider_reason or "terminal_failure")
                        )
                        error_text = str(exc)
                    # Terminal task failures (request validation included) or a
                    # task hung past the hang limit resubmit the same frozen
                    # segment under a fresh run identity — fresh asset URLs
                    # included — capped at PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS
                    # per segment.
                    resubmits += 1
                    if resubmits > PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS:
                        log.event(
                            "picsart_auto_resubmit_exhausted",
                            segment_index=index,
                            resubmits=PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS,
                            failed_task_id=task_id,
                            reason=reason,
                            error=error_text,
                        )
                        raise PicsartAutoResubmitExhaustedError(
                            f"segment {index} 自动重提 "
                            f"{PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS} 次仍失败"
                            f"（末次原因 {reason}，task {task_id}），"
                            f"fail-closed 需人工介入: {error_text}"
                        ) from exc
                    log.event(
                        "picsart_terminal_failure_resubmit",
                        segment_index=index,
                        resubmit=resubmits,
                        failed_task_id=task_id,
                        reason=reason,
                        error=error_text,
                    )
                    current = replace(
                        prepared,
                        run_identity=f"{node_prefix}-resubmit-{resubmits:02d}",
                        existing_task_id=None,
                    )
                    self._begin_submission_batch([current])
                    task_id = None
                    continue
                break
            downloaded = segment_dir / f"h3-provider{artifact_suffix}.mp4"
            self._node(
                log,
                f"{node_prefix}-download",
                scope,
                {"url": generated_url, "task_id": task_id},
                [],
                lambda: self.picsart.download(
                    generated_url, downloaded, api_key_role=task_key_role[0],
                ),
                lambda path: [path],
            )
            trimmed = segment_dir / f"generated{artifact_suffix}.mp4"
            trim_node = (
                "trim-provider-output-24fps"
                if node_prefix == "picsart"
                else f"{node_prefix}-trim-provider-output-24fps"
            )
            self._node(
                log,
                trim_node,
                scope,
                {
                    "source": downloaded,
                    "target_duration_s": prepared.target_duration_s,
                },
                [downloaded],
                lambda: media.trim_24fps(
                    downloaded, prepared.target_duration_s, trimmed,
                ),
                lambda _value: [trimmed],
            )
            item["final_prompt"] = prepared.final_prompt
            item["task_id"] = task_id
            item["status"] = "succeeded"
            return trimmed
        except Exception:
            item["status"] = "failed"
            raise

    def _generation_config(self, request: dict[str, Any]) -> dict[str, bool]:
        request = self._validated_effective_request(request)
        processing = request["processing"]
        enabled = self.settings.enable_mediakit and bool(
            processing["optimize_image"]
        )
        return {
            "remove_subtitle": enabled and bool(
                processing["remove_subtitle"]
            ),
            "remove_watermark": enabled and bool(
                processing["remove_logo"]
            ),
        }

    @staticmethod
    def _skipped_result(value: Any, reason: str) -> ProviderResult[Any]:
        return ProviderResult(
            value=value,
            record={"outcome": "skipped", "reason": reason},
            evidence_media=[],
        )

    @staticmethod
    def _yamnet_has_vocal_signal(analysis: dict[str, Any]) -> bool:
        return any(
            max(float(window.get("spoken") or 0), float(window.get("sung") or 0))
            >= MASTER_EMPTY_ASR_SIGNAL
            for window in analysis.get("windows") or []
        )

    @staticmethod
    def _manifest_media(directory: Path, manifest: dict[str, Any]) -> list[Path]:
        paths: list[Path] = []
        for frame in manifest.get("frames") or []:
            name = frame.get("file") if isinstance(frame, dict) else None
            if isinstance(name, str):
                paths.append(directory / name)
        return paths

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _json_sha256(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _segment_scope(index: int) -> str:
        return f"segment-{index:04d}"

    @staticmethod
    def _public_segment(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": item["index"],
            "start_s": item["start_s"],
            "end_s": item["end_s"],
            "duration_s": item["duration_s"],
            "chain_id": item.get("chain_id"),
            "join_mode": item["join_mode"],
            "status": item.get("status", "running"),
            "visual_prompt": item.get("visual_prompt"),
            "final_prompt": item.get("final_prompt"),
            "dialogue": item.get("dialogue", []),
        }
