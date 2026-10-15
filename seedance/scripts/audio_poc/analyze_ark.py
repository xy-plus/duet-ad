"""步骤2b：方舟音频理解分析节点。

逐行把原音片段发给方舟 doubao-seed-2-0-lite-260428（base64 input_audio），
输出语速/语调情绪/停顿/整体语气的自然语言描述，落 analysis-ark.json。
单行失败记录原因并继续。

用法: python analyze_ark.py
"""

from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc")
from common import (  # noqa: E402
    ARK_AUDIO_MODEL,
    PROJECTS,
    ark_audio_understand,
    line_clip_path,
    load_secrets,
    load_state,
    project_dir,
    project_lines,
)

PROMPT = """这是一段广告视频的原声配音片段。请仔细听，然后只输出一个 JSON 对象（不要 markdown 代码围栏），字段如下：
{
  "transcript": "你听到的原文转写",
  "speech_rate_desc": "语速描述（慢/中等/偏快/快，并估计大致字符每秒）",
  "intonation_emotion": "语调与情绪描述（如热情、平稳、急促、上扬等）",
  "pause_pattern": "停顿模式描述（停顿位置、长短、节奏）",
  "overall_tone": "整体语气与风格（如带货激情、教学亲切等）",
  "gender_age": "说话人性别与年龄段猜测"
}"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_ark_json(raw: str) -> dict[str, object]:
    cleaned = _FENCE.sub("", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(cleaned, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("方舟输出不是 JSON 对象")
    return parsed


def main() -> None:
    secrets = load_secrets()
    for short_id in PROJECTS:
        out_path = project_dir(short_id) / "analysis-ark.json"
        prior: dict[int, dict[str, object]] = {}
        if out_path.exists():
            prior = {
                l["line_index"]: l
                for l in json.loads(out_path.read_text()).get("lines", [])
                if "description" in l  # 只保留成功行；错误行重试（如账号开通模型后重跑）
            }
        state = load_state(short_id)
        results = []
        for line in project_lines(short_id):
            if line["index"] in prior:
                results.append(prior[line["index"]])
                print(f"{short_id} line {line['index']}: 已有成功结果，复用")
                continue
            clip = line_clip_path(short_id, line["index"])
            entry: dict[str, object] = {
                "line_index": line["index"],
                "window_s": round(line["end_s"] - line["start_s"], 3),
                "model": ARK_AUDIO_MODEL,
            }
            try:
                raw = ark_audio_understand(secrets, audio_path=clip, prompt=PROMPT)
                entry["description"] = parse_ark_json(raw)
                entry["raw"] = raw
            except Exception as exc:  # POC：记录原因继续
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            print(f"{short_id} line {line['index']}: "
                  + ("ok" if "description" in entry else f"失败 {entry['error']}"))
        out_path.write_text(json.dumps({
            "project": short_id,
            "source_language_code": state.get("source_language_code"),
            "model": ARK_AUDIO_MODEL,
            "lines": results,
        }, ensure_ascii=False, indent=2))
        print(f"{short_id}: analysis-ark.json 落盘")


if __name__ == "__main__":
    main()
