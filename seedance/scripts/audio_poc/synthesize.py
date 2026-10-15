"""步骤4：三方案 TTS 合成 + 装配 full-track。

方案标记：baseline（裸参数）、tts20-local（本地特征→DeepSeek 指令）、
tts20-ark（方舟描述→DeepSeek 指令）。
每行实际请求参数落 request-log.jsonl（脱敏）；时长偏差落 durations.json；
单行失败记录并继续。超长用 ffmpeg atempo 压窗口，总变速限 ±15%，超限记 warning。

用法: python synthesize.py [baseline|tts20-local|tts20-ark ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc")
from common import (  # noqa: E402
    DOUBAO_TTS_SAMPLE_RATE,
    OUTPUTS_DIR,
    doubao_synthesize,
    load_secrets,
    project_dir,
    project_lines,
)

FFMPEG = "/home/xy/.local/bin/ffmpeg"
MAX_TEMPO = 1.15  # atempo 总变速上限（+15%）
MIN_TEMPO = 0.87  # 下限（-13%≈-15%）


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def fit_to_window(src: Path, window_s: float) -> tuple[np.ndarray, list[str]]:
    """读 24kHz  wav，超长则 atempo 压进窗口（上限 ±15%，超限记 warning 并截尾）。"""
    warnings: list[str] = []
    duration = wav_duration(src)
    tempo = duration / max(window_s, 0.01)
    path = src
    if tempo > 1.0:
        applied = min(tempo, MAX_TEMPO)
        if tempo > MAX_TEMPO:
            warnings.append(
                f"需要 atempo={tempo:.2f} 超过 ±15% 上限，按 {MAX_TEMPO} 压缩后截尾"
            )
        fitted = src.with_suffix(".fitted.wav")
        result = subprocess.run(
            [FFMPEG, "-y", "-i", str(path), "-filter:a", f"atempo={applied:.4f}",
             str(fitted)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            warnings.append(f"atempo 失败，按原样放置: {result.stderr[-200:]}")
        else:
            path = fitted
            duration = wav_duration(path)
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").copy()
    limit = int(window_s * DOUBAO_TTS_SAMPLE_RATE)
    if len(samples) > limit:
        fade = int(0.02 * DOUBAO_TTS_SAMPLE_RATE)
        samples = samples[:limit]
        if fade > 0 and len(samples) > fade:
            ramp = np.linspace(1.0, 0.0, fade)
            samples[-fade:] = (samples[-fade:] * ramp).astype("<i2")
        warnings.append("音频仍超窗口，已截尾（20ms 淡出）")
    return samples.astype(np.float32) / 32768.0, warnings


def plan_for(short_id: str, plan: str) -> dict[int, dict[str, object]]:
    if plan == "baseline":
        return {}
    variant = plan.removeprefix("tts20-")
    raw = json.loads(
        (project_dir(short_id) / f"deepseek-plan-{variant}.json").read_text()
    )
    return {l["line_index"]: l for l in raw.get("lines", [])}


def run_plan(secrets: dict[str, str], plan: str) -> dict[str, object]:
    summary: dict[str, object] = {"plan": plan, "projects": {}, "failures": []}
    for short_id, lines_all in ((sid, project_lines(sid)) for sid in sorted_lines_keys()):
        out_dir = OUTPUTS_DIR / plan / short_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "request-log.jsonl"
        plans = plan_for(short_id, plan)
        durations = []
        track_end = max(l["end_s"] for l in lines_all)
        track = np.zeros(int(track_end * DOUBAO_TTS_SAMPLE_RATE) + 1, dtype=np.float32)
        with log_path.open("a") as log:
            for line in lines_all:
                idx = line["index"]
                window = line["end_s"] - line["start_s"]
                entry: dict[str, object] = {
                    "plan": plan, "project": short_id, "line_index": idx,
                    "window_s": round(window, 3),
                }
                kwargs: dict[str, object] = {}
                if plan != "baseline":
                    # context_texts/emotion 对所有音色发送：实测 API 对 QA 音色同样接受
                    p = plans.get(idx) or {}
                    rate = p.get("speech_rate")
                    if isinstance(rate, (int, float)):
                        kwargs["speech_rate"] = int(max(-50, min(100, rate)))
                    emotion = (p.get("emotion") or "").strip()
                    if emotion and emotion != "neutral":
                        kwargs["emotion"] = emotion
                    ctx = (p.get("context_texts") or "").strip()
                    if ctx:
                        kwargs["context_texts"] = ctx
                line_wav = out_dir / f"line-{idx:04d}.wav"
                try:
                    record = doubao_synthesize(
                        secrets,
                        text=line["text"],
                        voice_type=line["voice_type"],
                        destination=line_wav,
                        **kwargs,
                    )
                    entry["request_body"] = record["request"]["body"]
                    entry["outcome"] = "synthesized"
                    synth_dur = wav_duration(line_wav)
                    samples, warns = fit_to_window(line_wav, window)
                    fitted_dur = len(samples) / DOUBAO_TTS_SAMPLE_RATE
                    if warns:
                        entry["warnings"] = warns
                    offset = int(line["start_s"] * DOUBAO_TTS_SAMPLE_RATE)
                    end = min(offset + len(samples), len(track))
                    track[offset:end] += samples[: end - offset]
                    durations.append({
                        "line_index": idx,
                        "window_s": round(window, 3),
                        "synth_s": round(synth_dur, 3),
                        "fitted_s": round(fitted_dur, 3),
                        "deviation_s": round(synth_dur - window, 3),
                    })
                except Exception as exc:  # POC：记录原因继续其他行
                    entry["outcome"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    summary["failures"].append(
                        {"plan": plan, "project": short_id, "line_index": idx,
                         "error": entry["error"]}
                    )
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"{plan} {short_id} line {idx}: {entry['outcome']}")
        full = out_dir / "full-track.wav"
        pcm = (np.clip(track, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with wave.open(str(full), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(DOUBAO_TTS_SAMPLE_RATE)
            handle.writeframes(pcm)
        (out_dir / "durations.json").write_text(json.dumps({
            "plan": plan, "project": short_id, "lines": durations,
        }, ensure_ascii=False, indent=2))
        summary["projects"][short_id] = str(full)
    return summary


def sorted_lines_keys() -> list[str]:
    from common import PROJECTS

    return sorted(PROJECTS.keys())


def main() -> None:
    plans = sys.argv[1:] or ["baseline", "tts20-local", "tts20-ark"]
    secrets = load_secrets()
    summaries = []
    for plan in plans:
        summaries.append(run_plan(secrets, plan))
    report = POC_REPORT = OUTPUTS_DIR / "synthesis-summary.json"
    report.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    failures = [f for s in summaries for f in s["failures"]]
    print(f"完成。失败行 {len(failures)} 条；汇总 {report}")


if __name__ == "__main__":
    main()
