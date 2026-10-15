"""Dialogue delivery 规划（tts20-ark 音频链 B1）：逐行 TTS 参数计划的
输入构造、closed schema 绑定与归一化，以及 dialogue-tts 的 section 派生
与断点行复用探测。

语速口径（POC 修复后，见 scripts/audio_poc/plan_deepseek.py 头注）：第一
目标恰好填满窗口，speech_rate=(目标语速/基准语速-1)*100 夹 [-50,100]；
基准语速是该音色 speech_rate=0 的实测标定值（VOICE_BASELINE_RATES，
表本体在 app/doubao_voices.py）。pick_voice_for_gender 只从已标定音色中
选择（缺省基准会让新音色的时长控制静默失效）；沿用的旧绑定音色仍可能
无标定值，build_dialogue_delivery_payload 对此保持 fail-closed 报错兜底。
"""

from __future__ import annotations

import hashlib
import re
import uuid
import wave
from pathlib import Path
from typing import Any

from app import doubao_voices
from app.doubao_voices import (
    DoubaoVoiceCatalog,
    calibrated_voice_types,
    doubao_voice_language_key,
    is_calibrated_voice,
)
from app.fusion_schema import canonical_input

# 兼容既有 import 的再导出；表本体与访问口径（is_calibrated_voice 等）
# 在 app/doubao_voices.py。
VOICE_BASELINE_RATES = doubao_voices.VOICE_BASELINE_RATES


DIALOGUE_DELIVERY_OUTPUT_NAME = "duet.dialogue-delivery-output"
DIALOGUE_DELIVERY_OUTPUT_VERSION = 1

# section_id 派生命名空间：随机生成一次后固定为常量。section_id =
# uuid5(TTS_SECTION_NAMESPACE, f"{run_id}:{voice_type}")——每次
# _run_dialogue_audio 执行（含续跑）一个新 run_id × 每个说话人一个 section。
TTS_SECTION_NAMESPACE = uuid.UUID("f3b8c2a1-5d6e-4a7b-9c0d-1e2f3a4b5c6d")

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# 语速换算除零保护：窗口小于 10ms 的行没有实际配音空间。
_MIN_WINDOW_S = 0.01


def count_letters(text: str) -> int:
    """有效字符数：字母/汉字等可发音字符，忽略数字、标点与空白。"""
    return len(_LETTER.findall(text or ""))


def suggested_speech_rate(target_cps: float, baseline_cps: float) -> int:
    """窗口预算换算：speech_rate = round((目标/基准 - 1) * 100) 夹 [-50,100]。"""
    if not baseline_cps > 0:
        raise ValueError("voice baseline rate must be positive")
    return max(-50, min(100, round((target_cps / baseline_cps - 1.0) * 100)))


def build_dialogue_delivery_payload(
    *,
    target_language: str,
    translated_lines: list[dict[str, Any]],
    line_voices: list[str],
    catalog: DoubaoVoiceCatalog,
    analysis_lines: list[Any] | None = None,
) -> dict[str, Any]:
    """构造 DeepSeek 输入：逐行窗口、音色、基准语速、目标语速与预算好的
    suggested_speech_rate；analysis_lines 是可选的音频理解行列表（B2 产物，
    按行位对齐注入 features，缺失则空对象，仅凭文本/窗口规划）。

    用到的音色缺标定值时明确报错（不用缺省基准语速）。
    """
    if len(line_voices) != len(translated_lines):
        raise ValueError("dialogue delivery line voices are invalid")
    missing = sorted({
        voice_type
        for voice_type in line_voices
        if not is_calibrated_voice(voice_type)
    })
    if missing:
        raise ValueError(
            "voice baseline rates are missing for: " + ", ".join(missing)
        )
    lines = []
    for offset, (line, voice_type) in enumerate(
        zip(translated_lines, line_voices), 1
    ):
        window_s = max(
            float(line["end_s"]) - float(line["start_s"]), _MIN_WINDOW_S,
        )
        target_cps = round(count_letters(str(line["text"])) / window_s, 2)
        baseline_cps = doubao_voices.VOICE_BASELINE_RATES[voice_type]
        voice = catalog.voice(voice_type)
        lines.append({
            "line_index": offset,
            "text": line["text"],
            "window_s": round(window_s, 3),
            "voice": {
                "voice_type": voice_type,
                "name": voice["name"],
                "gender": voice["gender"],
                "scenes": voice["scenes"],
                "capabilities": voice["capabilities"],  # 仅参考，不作门禁
            },
            "voice_baseline_chars_per_s": baseline_cps,
            "target_chars_per_window_s": target_cps,
            "suggested_speech_rate": suggested_speech_rate(
                target_cps, baseline_cps,
            ),
            "features": (
                analysis_lines[offset - 1]
                if analysis_lines is not None
                and offset - 1 < len(analysis_lines)
                else {}
            ),
        })
    return {
        "target_language": target_language,
        "voice_catalog_sha256": catalog.digest,
        "lines": lines,
    }


# context_texts 语言策略：只允许中文/英文，小语种统一兜底英文。
# 实证事实（均有原始凭证）：
# - 小语种 context 会被当正文朗读：2026-09-27 车载湿巾 line-0012 的马来语
#   context 被 TTS 逐字念进音频（同参数 2/2 稳定复现，wav 7.3s/窗口 3.16s）。
# - 中文 context 配异语言台词会串语言：探针（原始凭证
#   /tmp/tts-ctx-probe/results.json）同一英文行 speech_rate=-13
#   emotion=happy explicit_language=en，中文 context 5 次合成漂移 2 次
#   （"60毫升"/"六十毫升"）。
# - 英文 context 安全：同一探针英文 context 与无 context 合计 10 次 0 次
#   漂移，且无续读泄漏。
# 机制原因（推断，未经火山文档证实，勿当事实引用）：TTS 2.0 的指令遵循
# 能力可能以中英语料为主训练，小语种指令因此不生效。此推断不影响上述
# 策略——策略完全由实证事实支撑。
def dialogue_delivery_instruction(language_label: str) -> str:
    """DeepSeek 系统指令：语速由窗口预算公式锁定；语气参考原音描述，但以
    广告口播表现力为下限（原音平淡时提升为带货腔，禁止整段 neutral 念稿）。"""
    return (
        "你是广告配音导演。视频原声是"
        f"{language_label or '源语'}"
        "口播，我们要用豆包 Seed-TTS 2.0 把目标语译文合成出来，贴在原时间窗里。\n"
        "我会给你每行：译文文本、窗口时长 window_s、目标语速 "
        "target_chars_per_window_s（译文有效字符数/窗口秒数）、该音色的基准语速 "
        "voice_baseline_chars_per_s（speech_rate=0 时实测合成语速）、以及原音频"
        "理解描述 features（为空对象表示暂不可用，仅凭文本与语境判断语气）。\n"
        "你的任务是为每行决定 TTS 参数，第一优先级是时长适配：\n"
        "1. speech_rate（整数，-50..100，相对基准语速的百分比调整）：必须让合成"
        "语音恰好填满 window_s。换算公式 speech_rate = "
        "round((target_chars_per_window_s / voice_baseline_chars_per_s - 1) * 100)，"
        "夹到 [-50, 100]。我给每行附了按此公式算好的 suggested_speech_rate，除有"
        "明确理由外不要偏离超过 ±5。绝不允许为了贴近原音语速而牺牲窗口适配。\n"
        "2. emotion：从 happy/sad/angry/surprised/fearful/disgusted/neutral 中选"
        "最贴合的一个。这是替换后商品的广告口播，表现力有下限：features 描述"
        "原音语气平淡（语调平缓、起伏小、无明显情绪）时，不允许照抄成 neutral "
        "念稿，必须结合带货语境选积极情绪——推荐、种草类台词用 happy，强调卖点"
        "惊喜感可用 surprised；不得整段 neutral。原音有明确情绪时尊重原音情绪；"
        "只有台词内容本身严肃（如安全警示、致歉声明）才可填 neutral。实在无法"
        "判断可填 null。\n"
        "3. context_texts：一条自然语言指令（单元素数组），描述语气、情绪、"
        "停顿节奏。硬性语言约束：指令只允许中文或英文——台词是中文就用中文写，"
        "其余所有目标语言（包括英语、马来语、西班牙语等）一律用英文写；禁止"
        "用台词本身的小语种语言写演绎指令（小语种 context 实测会被当作正文"
        "逐字朗读而不是指令——马来语 context 曾 2/2 复现被念进音频；英文 "
        "context 经探针验证安全）。语速快慢与停顿位置参考"
        "原音特征，语气风格同样适用广告表现力下限：原音平淡时必须写成亲切、"
        "有感染力的带货口播腔（英文指令例如“warm, excited friend-recommendation "
        "energy, punch the benefit words, lift the tail of each sentence”），"
        "禁止写“平静陈述”“语调平缓”这类念稿指令；没有原音描述则直接按带货"
        "口播语境写。不要在指令里要求加快或放慢语速（语速由 speech_rate 控制），"
        "只描述语气风格与停顿。实在无法判断可填 null。\n"
        "line_index 必须与我给的行一一对应，不得漏行。"
    )


def dialogue_delivery_instruction_sha256(language_label: str) -> str:
    """instruction 文本哈希，delivery plan 复用判据之一
    （pipeline._ensure_dialogue_delivery）：instruction 改了旧 plan 必须
    重规划，否则旧 plan 的 context_texts/参数还带着旧指令的行为。"""
    return hashlib.sha256(
        dialogue_delivery_instruction(language_label).encode("utf-8"),
    ).hexdigest()


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def dialogue_delivery_schema(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Closed schema：逐行 {speech_rate(-50..100), emotion|null,
    context_texts|null}，绑定 schema/version 与 input_sha256。"""
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list) or not lines:
        raise ValueError("dialogue delivery input is invalid")
    line_count = len(lines)
    input_sha256 = hashlib.sha256(canonical_input(payload)).hexdigest()
    line_schema = _closed_object({
        "line_index": {"type": "integer", "minimum": 1, "maximum": line_count},
        "speech_rate": {"type": "integer", "minimum": -50, "maximum": 100},
        "emotion": {"type": ["string", "null"]},
        "context_texts": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    })
    return _closed_object({
        "schema": {"type": "string", "enum": [DIALOGUE_DELIVERY_OUTPUT_NAME]},
        "version": {
            "type": "integer",
            "enum": [DIALOGUE_DELIVERY_OUTPUT_VERSION],
        },
        "input_sha256": {"type": "string", "enum": [input_sha256]},
        "lines": {
            "type": "array",
            "minItems": line_count,
            "maxItems": line_count,
            "items": line_schema,
        },
    }), input_sha256


def normalize_dialogue_delivery(
    value: object,
    *,
    input_sha256: str,
    line_count: int,
) -> list[dict[str, Any]]:
    """归一化模型输出为按行序的 [{speech_rate, emotion, context_texts}]。"""
    if not isinstance(value, dict) or not {
        "schema", "version", "input_sha256", "lines",
    } <= set(value):
        raise ValueError("dialogue delivery output is invalid")
    if (
        value["schema"] != DIALOGUE_DELIVERY_OUTPUT_NAME
        or type(value["version"]) is not int
        or value["version"] != DIALOGUE_DELIVERY_OUTPUT_VERSION
        or value["input_sha256"] != input_sha256
    ):
        raise ValueError("dialogue delivery output is not bound to its input")
    raw_lines = value["lines"]
    if not isinstance(raw_lines, list) or len(raw_lines) != line_count:
        raise ValueError("dialogue delivery lines are invalid")
    ordered: list[dict[str, Any] | None] = [None] * line_count
    for item in raw_lines:
        if not isinstance(item, dict) or not {
            "line_index", "speech_rate", "emotion", "context_texts",
        } <= set(item):
            raise ValueError("dialogue delivery line is invalid")
        line_index = item["line_index"]
        if (
            isinstance(line_index, bool)
            or not isinstance(line_index, int)
            or not 1 <= line_index <= line_count
            or ordered[line_index - 1] is not None
        ):
            raise ValueError("dialogue delivery line_index is invalid")
        speech_rate = item["speech_rate"]
        if (
            isinstance(speech_rate, bool)
            or not isinstance(speech_rate, int)
            or not -50 <= speech_rate <= 100
        ):
            raise ValueError("dialogue delivery speech_rate is invalid")
        emotion = item["emotion"]
        if emotion is not None and (
            not isinstance(emotion, str) or not emotion.strip()
        ):
            raise ValueError("dialogue delivery emotion is invalid")
        context_texts = item["context_texts"]
        if context_texts is not None and (
            not isinstance(context_texts, list)
            or any(
                not isinstance(entry, str) or not entry.strip()
                for entry in context_texts
            )
        ):
            raise ValueError("dialogue delivery context_texts is invalid")
        ordered[line_index - 1] = {
            "speech_rate": speech_rate,
            "emotion": emotion,
            "context_texts": context_texts,
        }
    return [item for item in ordered if item is not None]


def dialogue_tts_section_id(run_id: str, voice_type: str) -> str:
    """每次运行 × 每个说话人一个 section：uuid5(命名空间, run_id:voice_type)。"""
    return str(uuid.uuid5(TTS_SECTION_NAMESPACE, f"{run_id}:{voice_type}"))


def dialogue_tts_line_input_sha256(
    *,
    text: str,
    voice_type: str,
    start_s: float,
    end_s: float,
    speech_rate: int | None,
    emotion: str | None,
    context_texts: list[str] | None,
    explicit_language: str | None,
) -> str:
    """单行合成输入标识：文本 + 音色 + 窗口 + delivery 合成参数 +
    explicit_language 的规范化哈希。

    dialogue-tts 断点行复用判据之一：续跑重跑翻译（输出非确定）后行数
    相同、音色绑定相同、文本不同时，旧 wav 不得复用；音色绑定变化同理。
    explicit_language 必须入哈希：2026-09-27 全量锁修复前合成的不锁语种
    wav（概率性串语种）在修复后不得被复用。用逐行哈希而非 delivery 块级
    input_sha256：块级哈希把任一行的变化放大成全部行重合成（重复扣费），
    逐行哈希只让内容真正变化的行重新合成，粒度与付费单元（行）一致。
    """
    return hashlib.sha256(canonical_input({
        "text": text,
        "voice_type": voice_type,
        "start_s": start_s,
        "end_s": end_s,
        "speech_rate": speech_rate,
        "emotion": emotion,
        "context_texts": context_texts,
        "explicit_language": explicit_language,
    })).hexdigest()


def is_decodable_wav(path: Path) -> bool:
    """断点行复用判据之一：wav 存在、非空且可解码（PCM wav 头 + 至少一帧）。"""
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnframes() <= 0:
                return False
            return bool(handle.readframes(1))
    except (wave.Error, EOFError, OSError):
        return False


# ---------------------------------------------------------------------------
# 逐行音色绑定（2026-09-27 画面人物性别修复）
#
# 背景：原片旁白性别与画面演示者可以不一致（b6240a72 电热睫毛夹：全程男声
# 旁白，画面 seg1 男性演示、seg2/3 女性演示），且 Object Replace LLM 的
# dialogue_voices 绑定也会错配性别。因此音色锚定「画面里的成年演示者」：
# 每行台词按时间窗归属到 segment，取该段主口播人物的性别选音色；音频信号
# （方舟 gender_age）降级为无画面证据时的兜底。
#
# 主口播人物判定：行起声瞬间画面上的演示者，即该段 occurrences 里首帧序
# 最小的成年人物（口播按时间顺序叙述演示动作，行窗起点的演示者就是该行的
# 音色锚点）；首帧并列时帧数多者胜，再并列按 person id 字典序。注意不是
# 「帧数最多」——b6240a72 seg1 女性 6 帧多于男性 3 帧，但行起声时是男性在
# 演示，验收锚定 line1 男声。
# ---------------------------------------------------------------------------

# 儿童人物永不作为口播音色依据（儿童不出声念广告旁白）。
_PERSON_CHILD_KEYWORDS = ("儿童", "女童", "男童", "女孩", "男孩", "小孩", "幼童", "孩子")
# 性别关键词用双字词，避免「男/女」单字被无关词命中；两种性别都出现或都
# 缺失时返回 None（描述可能提到画面里的其他人物，如「与女巫帽女性并列」，
# 此时单性别命中属于噪声，但尚无更好的无模型判别，按命中处理）。
_PERSON_FEMALE_KEYWORDS = ("女性", "女子", "女人", "女士", "女声")
_PERSON_MALE_KEYWORDS = ("男性", "男子", "男人", "男士", "男声")


def person_description_gender(text: object) -> str | None:
    """人物自由文本描述 → female/male；两性都提到或都没提到返回 None。"""
    if not isinstance(text, str):
        return None
    female = any(word in text for word in _PERSON_FEMALE_KEYWORDS)
    male = any(word in text for word in _PERSON_MALE_KEYWORDS)
    if female == male:
        return None
    return "female" if female else "male"


def _person_record_text(record: dict[str, Any]) -> str:
    parts = [str(record.get("source_visual_description") or "")]
    preserve = record.get("preserve")
    if isinstance(preserve, list):
        parts.extend(str(item) for item in preserve)
    return "\n".join(parts)


def person_record_is_child(record: dict[str, Any]) -> bool:
    text = _person_record_text(record)
    return any(word in text for word in _PERSON_CHILD_KEYWORDS)


def ark_gender_age_vote(value: object) -> str | None:
    """方舟 gender_age 自由文本 → female/male；两性都提到或都没提到返回 None。"""
    return person_description_gender(value)


def ark_gender_majority(
    analysis_lines: list[dict[str, Any]],
) -> tuple[str | None, str]:
    """方舟逐行 gender_age 多数票。单行异票视为分类噪声；全缺失/平票返回
    (None, 依据)。只作无画面证据时的兜底信号。"""
    votes = [
        gender
        for gender in (ark_gender_age_vote(line.get("gender_age")) for line in analysis_lines)
        if gender is not None
    ]
    if not votes:
        return None, "ark gender_age 全部缺失"
    female = votes.count("female")
    male = votes.count("male")
    if female == male:
        return None, f"ark gender_age 平票 {female} 女/{male} 男"
    gender = "female" if female > male else "male"
    return gender, f"ark gender_age 多数票 {female} 女/{male} 男"


def _segment_for_line(
    line: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """行 → 段归属：行 start_s 落在的段窗（含段起点）；都不含时取时间重叠
    最大的段。不假设行窗与段窗一一对应。"""
    start_s = float(line["start_s"])
    end_s = float(line["end_s"])
    best: dict[str, Any] | None = None
    best_overlap = -1.0
    for segment in segments:
        seg_start = float(segment["start_s"])
        seg_end = float(segment["end_s"])
        if seg_start <= start_s < seg_end:
            return segment
        overlap = min(end_s, seg_end) - max(start_s, seg_start)
        if overlap > best_overlap:
            best = segment
            best_overlap = overlap
    if best is None:
        raise ValueError("segments are empty")
    return best


def _segment_primary_speaker(
    people: dict[str, Any], segment_index: int
) -> tuple[str, str] | None:
    """段内主口播人物（成年、性别可解析）：首帧序最小者，并列帧数多者，
    再并列 person id 字典序。返回 (person_id, gender) 或 None。"""
    candidates: list[tuple[int, int, str, str]] = []
    for person_id in sorted(people):
        record = people[person_id]
        if not isinstance(record, dict) or person_record_is_child(record):
            continue
        gender = person_description_gender(_person_record_text(record))
        if gender is None:
            continue
        occurrences = record.get("occurrences")
        if not isinstance(occurrences, list):
            continue
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            if occurrence.get("segment_index") != segment_index:
                continue
            frames = occurrence.get("frame_orders")
            frames = frames if isinstance(frames, list) else []
            first = min(frames) if frames else 1 << 30
            candidates.append((first, -len(frames), person_id, gender))
    if not candidates:
        return None
    _, _, person_id, gender = min(candidates)
    return person_id, gender


def _project_primary_speaker(
    people: dict[str, Any],
) -> tuple[str, str] | None:
    """项目级回退：全部段合计帧数最多的成年人物（并列取首帧最早）。"""
    totals: dict[str, tuple[int, int, str]] = {}
    for person_id in sorted(people):
        record = people[person_id]
        if not isinstance(record, dict) or person_record_is_child(record):
            continue
        gender = person_description_gender(_person_record_text(record))
        if gender is None:
            continue
        occurrences = record.get("occurrences")
        if not isinstance(occurrences, list):
            continue
        count = 0
        first = 1 << 30
        for occurrence in occurrences:
            frames = occurrence.get("frame_orders") if isinstance(occurrence, dict) else None
            if not isinstance(frames, list):
                continue
            count += len(frames)
            if frames:
                first = min(first, min(frames))
        if count:
            totals[person_id] = (count, first, gender)
    if not totals:
        return None
    person_id = min(totals, key=lambda pid: (-totals[pid][0], totals[pid][1], pid))
    return person_id, totals[person_id][2]


def resolve_line_speakers(
    *,
    lines: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    people: dict[str, Any],
) -> list[dict[str, Any]]:
    """逐行解析画面主口播人物。每行返回 {line_index, segment_index,
    person_id, gender, source, evidence}；source ∈
    segment/adjacent_segment/project/none。"""
    segment_indexes = [int(segment["index"]) for segment in segments]
    primary_by_segment = {
        index: _segment_primary_speaker(people, index) for index in segment_indexes
    }
    resolution: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines, 1):
        segment = _segment_for_line(line, segments)
        segment_index = int(segment["index"])
        entry: dict[str, Any] = {
            "line_index": line_index,
            "segment_index": segment_index,
            "person_id": None,
            "gender": None,
            "source": "none",
            "evidence": "画面无成年人物证据",
        }
        primary = primary_by_segment.get(segment_index)
        if primary is not None:
            entry.update(
                person_id=primary[0],
                gender=primary[1],
                source="segment",
                evidence=f"段 {segment_index} 首帧成年人物 {primary[0]}",
            )
        else:
            # 相邻段回退：段距离升序，同距离取段号小者。
            ordered = sorted(
                segment_indexes, key=lambda index: (abs(index - segment_index), index)
            )
            neighbor = next(
                (
                    (index, primary_by_segment[index])
                    for index in ordered
                    if index != segment_index and primary_by_segment[index] is not None
                ),
                None,
            )
            if neighbor is not None:
                neighbor_index, neighbor_primary = neighbor
                entry.update(
                    person_id=neighbor_primary[0],
                    gender=neighbor_primary[1],
                    source="neighbor_segment",
                    evidence=(
                        f"段 {segment_index} 无成年人物，取相邻段 "
                        f"{neighbor_index} 的 {neighbor_primary[0]}"
                    ),
                )
            else:
                project_primary = _project_primary_speaker(people)
                if project_primary is not None:
                    entry.update(
                        person_id=project_primary[0],
                        gender=project_primary[1],
                        source="project",
                        evidence=f"项目级合计帧数最多成年人物 {project_primary[0]}",
                    )
        resolution.append(entry)
    return resolution


# 豆包 V3 单向流 additions.explicit_language 的语种码（火山文档 6561/1257584）。
_EXPLICIT_LANGUAGE_CODES = {
    "zh": "zh-cn",
    "en": "en",
    "ja": "ja",
    "es": "es-mx",
    "id": "id",
    "pt": "pt-br",
    "ko": "ko",
    "it": "it",
    "de": "de",
    "fr": "fr",
    "th": "th",
    "vi": "vi",
    "ru": "ru",
    "tl": "fil",
    "ms": "ms",
    "ar": "ar",
}

# 语种子集缺某性别已标定音色时的跨语种兜底：火山音色列表注 3 记载部分中英
# 精品音色支持 30+ 语种识别（含马来语），必须搭配 explicit_language 锁定
# 朗读语种。目前只有女声兜底（马来语目录只有 ms_male_ham/naim 两条男声）；
# 兜底音色必须已入 VOICE_BASELINE_RATES，男声缺位时明确报错，不静默错配。
_MULTILINGUAL_FALLBACK_VOICES = {"female": "zh_female_vv_uranus_bigtts"}


def pick_voice_for_gender(
    catalog: DoubaoVoiceCatalog, language_label: str, gender: str
) -> str:
    """目标语言 × 性别选音色：只从 VOICE_BASELINE_RATES 已标定音色中选择
    （未标定音色的时长控制没有基准，不可用）；母语音色（voice_type 以语言码
    开头）优先于跨语种音色，场景优先级 通用场景 > 视频配音，目录序决胜；
    各层过滤后为空才走下一层，全部为空（含跨语种兜底音色未标定）时明确
    报错。只负责选音色；explicit_language 由 _explicit_language_for_voice
    统一决定（单一事实源）。"""
    language_key = doubao_voice_language_key(language_label)
    subset = [
        voice
        for voice in catalog.voices_for_language(language_label)
        if voice["gender"] == gender
        and is_calibrated_voice(voice["voice_type"])
    ]
    native = [
        voice
        for voice in subset
        if voice["voice_type"].startswith(f"{language_key}_")
    ]
    for pool in (native, subset):
        for scene in ("通用场景", "视频配音"):
            for voice in pool:
                if scene in voice["scenes"]:
                    return voice["voice_type"]
        if pool:
            return pool[0]["voice_type"]
    fallback = _MULTILINGUAL_FALLBACK_VOICES.get(gender)
    if (
        fallback is not None
        and catalog.has(fallback)
        and is_calibrated_voice(fallback)
    ):
        return fallback
    calibrated = ", ".join(calibrated_voice_types()) or "(空)"
    raise RuntimeError(
        f"no calibrated {gender} voice for {language_label}; "
        f"calibrated voices: {calibrated}"
    )


def _explicit_language_for_voice(voice_type: str, language_label: str) -> str | None:
    """explicit_language 全量锁（2026-09-27）：豆包 2.0 精品音色
    （*_uranus_bigtts）合成非中文台词时，数字+单位 token 会概率性串中文
    （实证：防晒霜项目 en_female_joanne 把 "60 ml" 读成中文"六十
    毫升"，裸探针偶发不复现；机制原因未经火山文档证实）。同语种合成也
    必须锁定朗读语种。接受度探针
    五组合（ms_male_ham+ms、es_female_ht_mx_f6+es-mx、en_female_joanne+en、
    zh_female_vv+ms、en_male_adam-imitation+en）全部 200：uranus 音色一律
    传目标语言码；非 uranus 音色未验证接受度，不传。"""
    if not voice_type.endswith("_uranus_bigtts"):
        return None
    return _EXPLICIT_LANGUAGE_CODES[doubao_voice_language_key(language_label)]


def resolve_line_voices(
    *,
    lines: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    people: dict[str, Any],
    inherited_voices: list[str],
    catalog: DoubaoVoiceCatalog,
    language_label: str,
    analysis_lines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """逐行音色决策：画面主口播人物性别与旧绑定性别一致则保留旧音色（不
    重折腾），不一致按语言 × 性别重选；无画面证据时方舟 gender_age 多数票
    兜底，仍无证据保留旧绑定。每行返回 {line_index, segment_index,
    person_id, gender, gender_source, evidence, previous_voice_type,
    voice_type, explicit_language, reselected}。"""
    speakers = resolve_line_speakers(lines=lines, segments=segments, people=people)
    ark_gender: str | None = None
    ark_evidence = "ark 分析未提供"
    if analysis_lines:
        ark_gender, ark_evidence = ark_gender_majority(analysis_lines)
    resolution: list[dict[str, Any]] = []
    for offset, (speaker, inherited) in enumerate(zip(speakers, inherited_voices), 1):
        gender = speaker["gender"]
        gender_source = speaker["source"]
        evidence = speaker["evidence"]
        if gender is None and ark_gender is not None:
            gender = ark_gender
            gender_source = "ark_majority"
            evidence = ark_evidence
        voice_type = inherited
        reselected = False
        if gender is not None and catalog.voice(inherited)["gender"] != gender:
            voice_type = pick_voice_for_gender(catalog, language_label, gender)
            reselected = True
        resolution.append(
            {
                "line_index": offset,
                "segment_index": speaker["segment_index"],
                "person_id": speaker["person_id"],
                "gender": gender,
                "gender_source": gender_source,
                "evidence": evidence,
                "previous_voice_type": inherited,
                "voice_type": voice_type,
                "explicit_language": _explicit_language_for_voice(
                    voice_type, language_label
                ),
                "reselected": reselected,
            }
        )
    return resolution


