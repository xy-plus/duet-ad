"""步骤3：DeepSeek 改写节点（tts20-local / tts20-ark 共用）。

第一目标：合成语音恰好填满台词窗口。目标语速=译文有效字符数/窗口秒数，
基准语速=该音色在 speech_rate=0 下的实测合成语速（取自 baseline 组
outputs/baseline/*/durations.json），speech_rate=(目标/基准-1)*100 夹 [-50,100]。
原音语速/情绪/停顿只作为语气风格参考写进 context_texts，不决定 speech_rate。
context_texts 与 emotion 对所有音色发送（实测 API 对 QA 音色同样接受）。
落 projects/<短id>/deepseek-plan-{local,ark}.json。

用法: python plan_deepseek.py [local|ark]   # 缺省两个都跑
"""

from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc")
from common import (  # noqa: E402
    OUTPUTS_DIR,
    PROJECTS,
    deepseek_structured,
    load_secrets,
    load_voice_catalog,
    project_dir,
    project_lines,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_index": {"type": "integer"},
                    "speech_rate": {"type": "integer", "minimum": -50, "maximum": 100},
                    "emotion": {"type": "string"},
                    "context_texts": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["line_index", "speech_rate", "emotion", "context_texts", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

INSTRUCTION_LOCAL = """你是广告配音导演。视频原声是{lang}口播，我们要用豆包 Seed-TTS 2.0 把目标语译文合成出来，贴在原时间窗 [start_s, end_s] 里。
我会给你每行：译文文本、窗口时长 window_s、目标语速 target_chars_per_window_s（译文有效字符数/窗口秒数）、该音色的基准语速 voice_baseline_chars_per_s（speech_rate=0 时实测合成语速）、以及本地声学分析（源语语速、基频、能量、停顿、SenseVoice 情绪标签）。
你的任务是为每行决定 TTS 参数，第一优先级是时长适配：
1. speech_rate（整数，-50..100，相对基准语速的百分比调整）：必须让合成语音恰好填满 window_s。换算公式 speech_rate = round((target_chars_per_window_s / voice_baseline_chars_per_s - 1) * 100)，夹到 [-50, 100]。我给每行附了按此公式算好的 suggested_speech_rate，除有明确理由（如行首末有长停顿可容纳偏差）外不要偏离超过 ±5。绝不允许为了贴近原音语速而牺牲窗口适配。
2. emotion（必填）：从 happy/sad/angry/surprised/fearful/disgusted/neutral 中选最贴合原音情绪标签与广告语境的一个，拿不准填 neutral。
3. context_texts（必填）：一句中文自然语言指令，描述语气、情绪、停顿节奏，参考原音特征（语速快慢、停顿位置、情绪）与广告带货语境。不要在指令里要求加快或放慢语速（语速由 speech_rate 控制），只描述语气风格与停顿。
4. rationale：一句话说明决策依据。
line_index 必须与我给的行一一对应，不得漏行。"""

INSTRUCTION_ARK = """你是广告配音导演。视频原声是{lang}口播，我们要用豆包 Seed-TTS 2.0 把目标语译文合成出来，贴在原时间窗 [start_s, end_s] 里。
我会给你每行：译文文本、窗口时长 window_s、目标语速 target_chars_per_window_s（译文有效字符数/窗口秒数）、该音色的基准语速 voice_baseline_chars_per_s（speech_rate=0 时实测合成语速）、以及音频理解模型的分析结果（若 features 里有 error 字段说明音频理解暂不可用，仅凭文本与语境判断语气）。
你的任务是为每行决定 TTS 参数，第一优先级是时长适配：
1. speech_rate（整数，-50..100，相对基准语速的百分比调整）：必须让合成语音恰好填满 window_s。换算公式 speech_rate = round((target_chars_per_window_s / voice_baseline_chars_per_s - 1) * 100)，夹到 [-50, 100]。我给每行附了按此公式算好的 suggested_speech_rate，除有明确理由外不要偏离超过 ±5。绝不允许为了贴近原音语速而牺牲窗口适配。
2. emotion（必填）：从 happy/sad/angry/surprised/fearful/disgusted/neutral 中选最贴合描述情绪与广告语境的一个，拿不准填 neutral。
3. context_texts（必填）：一句中文自然语言指令，描述语气、情绪、停顿节奏，参考音频理解描述（没有则参考广告带货语境）。不要在指令里要求加快或放慢语速（语速由 speech_rate 控制），只描述语气风格与停顿。
4. rationale：一句话说明决策依据。
line_index 必须与我给的行一一对应，不得漏行。"""

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def count_letters(text: str) -> int:
    return len(_LETTER.findall(text or ""))


def voice_baseline_rates() -> dict[str, float]:
    """从 baseline 组实测合成时长推每个音色 speech_rate=0 的字符/秒。"""
    totals: dict[str, list[float]] = {}
    for short_id in PROJECTS:
        durations_path = OUTPUTS_DIR / "baseline" / short_id / "durations.json"
        if not durations_path.exists():
            continue
        durations = {l["line_index"]: l for l in json.loads(durations_path.read_text())["lines"]}
        for line in project_lines(short_id):
            d = durations.get(line["index"])
            if not d or d["synth_s"] <= 0:
                continue
            t = totals.setdefault(line["voice_type"], [0.0, 0.0])
            t[0] += count_letters(line["text"])
            t[1] += d["synth_s"]
    return {v: round(c / s, 2) for v, (c, s) in totals.items() if s > 0}


def suggested_speech_rate(target_cps: float | None, baseline_cps: float | None) -> int | None:
    if not target_cps or not baseline_cps or baseline_cps <= 0:
        return None
    return max(-50, min(100, round((target_cps / baseline_cps - 1.0) * 100)))


def build_payload(short_id: str, variant: str, baseline_rates: dict[str, float]) -> dict[str, object]:
    catalog = load_voice_catalog()
    analysis_path = project_dir(short_id) / f"analysis-{variant}.json"
    analysis = {l["line_index"]: l for l in json.loads(analysis_path.read_text())["lines"]}
    lines = []
    for line in project_lines(short_id):
        voice = catalog.get(line["voice_type"], {})
        feats = analysis.get(line["index"], {})
        window = line["end_s"] - line["start_s"]
        target_cps = round(count_letters(line["text"]) / max(window, 0.01), 2)
        baseline_cps = baseline_rates.get(line["voice_type"])
        lines.append({
            "line_index": line["index"],
            "text": line["text"],
            "window_s": round(window, 3),
            "voice": {
                "voice_type": line["voice_type"],
                "name": voice.get("name"),
                "gender": voice.get("gender"),
                "scenes": voice.get("scenes"),
                "capabilities": voice.get("capabilities"),  # 仅参考，不作门禁
            },
            "voice_baseline_chars_per_s": baseline_cps,
            "target_chars_per_window_s": target_cps,
            "features": feats,
            "suggested_speech_rate": suggested_speech_rate(target_cps, baseline_cps),
        })
    return {"project": short_id, "feature_source": variant, "lines": lines}


def run_variant(secrets: dict[str, str], variant: str) -> None:
    baseline_rates = voice_baseline_rates()
    print(f"{variant}: 音色基准语速 {json.dumps(baseline_rates, ensure_ascii=False)}")
    for short_id in PROJECTS:
        out_path = project_dir(short_id) / f"deepseek-plan-{variant}.json"
        if out_path.exists():
            print(f"{short_id} {variant}: 已存在，跳过")
            continue
        payload = build_payload(short_id, variant, baseline_rates)
        instruction = (INSTRUCTION_LOCAL if variant == "local" else INSTRUCTION_ARK).format(lang="源语")
        result = deepseek_structured(
            secrets, instruction=instruction, payload=payload, schema=SCHEMA
        )
        result["feature_source"] = variant
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        n = len(result.get("lines", []))
        print(f"{short_id} {variant}: {n} 行计划落盘")


def main() -> None:
    variants = sys.argv[1:] or ["local", "ark"]
    secrets = load_secrets()
    for variant in variants:
        run_variant(secrets, variant)


if __name__ == "__main__":
    main()
