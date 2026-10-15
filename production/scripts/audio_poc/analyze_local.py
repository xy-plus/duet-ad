"""步骤2a：本地分析节点。

逐行窗口内：Silero VAD 扣静音 → 语速(源语字符/有声时长)；numpy YIN 提 F0
（均值/范围/性别启发）；RMS dB；停顿列表；SenseVoice-Small 情绪+音频事件。
产出 projects/<短id>/analysis-local.json。

用法: python analyze_local.py
"""

from __future__ import annotations

import json
import math
import re
import sys

import numpy as np
import sherpa_onnx

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc")
from common import (  # noqa: E402
    MODELS_DIR,
    PROJECTS,
    line_clip_path,
    load_state,
    project_dir,
    project_lines,
    read_wav_float,
)

SENSE_VOICE_MODEL = MODELS_DIR / "model.int8.onnx"
SENSE_VOICE_TOKENS = MODELS_DIR / "tokens.txt"
SILERO_VAD_MODEL = MODELS_DIR / "silero_vad.onnx"

SENSE_EMOTIONS = {
    "HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED",
    "SURPRISED", "EMO_UNKNOWN", "CALM",
}
SENSE_EVENTS = {
    "Speech", "Applause", "BGM", "Laughter", "Cry", "Sneeze", "Breath",
    "Cough", "Silence", "Music", "Noise",
}
_TAG = re.compile(r"<\|([^|]+)\|>")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def count_letters(text: str) -> int:
    return len(_LETTER.findall(text or ""))


def vad_segments(samples: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(SILERO_VAD_MODEL)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_speech_duration = 0.1
    config.silero_vad.min_silence_duration = 0.2
    config.silero_vad.max_speech_duration = 20.0
    config.silero_vad.window_size = 512
    config.sample_rate = sample_rate
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)
    window = 512
    offset = 0
    while offset + window <= len(samples):
        vad.accept_waveform(samples[offset:offset + window])
        offset += window
    if offset < len(samples):
        tail = np.zeros(window, dtype=np.float32)
        tail[: len(samples) - offset] = samples[offset:]
        vad.accept_waveform(tail)
    vad.flush()
    segments = []
    while not vad.empty():
        seg = vad.front
        segments.append((seg.start / sample_rate, (seg.start + len(seg.samples)) / sample_rate))
        vad.pop()
    return segments


def yin_f0(samples: np.ndarray, sample_rate: int,
           fmin: float = 50.0, fmax: float = 500.0) -> np.ndarray:
    """numpy 手写 YIN：返回每帧 F0（Hz），无声帧为 0。"""
    frame = int(0.032 * sample_rate)
    hop = int(0.010 * sample_rate)
    tau_min = max(2, int(sample_rate / fmax))
    tau_max = int(sample_rate / fmin)
    span = frame + tau_max
    if len(samples) < span:
        return np.array([])
    f0s = []
    for start in range(0, len(samples) - span, hop):
        xw = samples[start:start + span]
        # difference function d(tau)，向量化：windows[tau] = xw[tau:tau+frame]
        windows = np.lib.stride_tricks.sliding_window_view(xw, frame)[: tau_max + 1]
        head = xw[:frame]
        diffs_full = ((windows - head) ** 2).sum(axis=1)
        diffs = diffs_full[tau_min:tau_max]
        # cumulative mean normalized difference
        cmnd = np.ones_like(diffs)
        running = np.cumsum(diffs)
        taus = np.arange(1, len(diffs) + 1)
        nonzero = running > 0
        cmnd[nonzero] = diffs[nonzero] * taus[nonzero] / running[nonzero]
        threshold = 0.15
        below = np.where(cmnd < threshold)[0]
        if len(below) == 0:
            f0s.append(0.0)
            continue
        tau_idx = below[0]
        while tau_idx + 1 < len(cmnd) and cmnd[tau_idx + 1] < cmnd[tau_idx]:
            tau_idx += 1
        tau = tau_idx + tau_min
        f0s.append(sample_rate / tau)
    return np.array(f0s)


def classify_f0_range(p10: float, p90: float, mean: float) -> str:
    if mean <= 0:
        return "unknown"
    semitones = 12.0 * math.log2(max(p90, 1e-3) / max(p10, 1e-3))
    if semitones < 3.0:
        return "narrow"
    if semitones < 7.0:
        return "medium"
    return "wide"


def parse_sense_voice(raw: str) -> dict[str, object]:
    tags = _TAG.findall(raw)
    emotions = [t for t in tags if t.upper() in SENSE_EMOTIONS]
    events = [t for t in tags if t in SENSE_EVENTS]
    text = _TAG.sub("", raw).strip()
    return {
        "emotion": emotions[0].lower() if emotions else "neutral",
        "events": sorted(set(events)),
        "transcript": text,
    }


def main() -> None:
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSE_VOICE_MODEL),
        tokens=str(SENSE_VOICE_TOKENS),
        num_threads=4,
        use_itn=False,
    )
    for short_id in PROJECTS:
        out_path = project_dir(short_id) / "analysis-local.json"
        if out_path.exists():
            print(f"{short_id}: 已存在，跳过")
            continue
        state = load_state(short_id)
        asr_lines = state.get("asr_voice_lines") or []
        results = []
        for line in project_lines(short_id):
            clip = line_clip_path(short_id, line["index"])
            rate, samples = read_wav_float(clip)
            duration = len(samples) / rate
            segments = vad_segments(samples, rate)
            voiced = sum(e - s for s, e in segments)
            pauses = [
                {"at_s": round(segments[i][1], 3),
                 "dur_s": round(segments[i + 1][0] - segments[i][1], 3)}
                for i in range(len(segments) - 1)
                if segments[i + 1][0] - segments[i][1] >= 0.2
            ]
            # 源语字符数：取与窗口有时间重叠的 ASR 行
            overlapping = [
                a["text"] for a in asr_lines
                if a.get("end_s", 0) > line["start_s"] and a.get("start_s", 0) < line["end_s"]
            ]
            source_chars = sum(count_letters(t) for t in overlapping)
            f0 = yin_f0(samples, rate)
            voiced_f0 = f0[f0 > 0]
            f0_info: dict[str, object] = {"available": False}
            if len(voiced_f0) >= 5:
                mean_f0 = float(np.mean(voiced_f0))
                f0_info = {
                    "available": True,
                    "mean_hz": round(mean_f0, 1),
                    "p10_hz": round(float(np.percentile(voiced_f0, 10)), 1),
                    "p90_hz": round(float(np.percentile(voiced_f0, 90)), 1),
                    "range": classify_f0_range(
                        float(np.percentile(voiced_f0, 10)),
                        float(np.percentile(voiced_f0, 90)),
                        mean_f0,
                    ),
                    "gender_heuristic": "male" if mean_f0 < 165.0 else "female",
                }
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
            stream = recognizer.create_stream()
            stream.accept_waveform(rate, samples.tolist())
            recognizer.decode_stream(stream)
            sense = parse_sense_voice(stream.result.text)
            results.append({
                "line_index": line["index"],
                "window_s": round(line["end_s"] - line["start_s"], 3),
                "clip_duration_s": round(duration, 3),
                "voiced_s": round(voiced, 3),
                "source_text": " ".join(overlapping),
                "source_chars": source_chars,
                "source_chars_per_voiced_s": round(source_chars / voiced, 2) if voiced > 0.05 else None,
                "target_chars": count_letters(line["text"]),
                "target_chars_per_window_s": round(
                    count_letters(line["text"]) / max(line["end_s"] - line["start_s"], 0.01), 2
                ),
                "f0": f0_info,
                "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-8)), 1),
                "pauses": pauses,
                "sense_voice": sense,
            })
            print(f"{short_id} line {line['index']}: voiced={voiced:.2f}s "
                  f"rate={results[-1]['source_chars_per_voiced_s']} "
                  f"f0={f0_info.get('mean_hz')} emo={sense['emotion']}")
        out_path.write_text(json.dumps({
            "project": short_id,
            "source_language_code": state.get("source_language_code"),
            "lines": results,
        }, ensure_ascii=False, indent=2))
        print(f"{short_id}: analysis-local.json 落盘")


if __name__ == "__main__":
    main()
