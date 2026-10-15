"""Seed-Audio 1.0 POC: 参考音频 + prompt 一步出声（离线支线，不动 app/）。

两个后端（--backend 选择，输出目录互不覆盖）：

- volcengine（默认）：`POST https://openspeech.bytedance.com/api/v3/tts/create`，
  model=seed-audio-1.0（文档
  https://www.volcengine.com/docs/DoubaoVoice/audio-generation-http ）。
  鉴权只有 X-Api-Key（+ 可选 X-Api-Request-Id），**不要**传 X-Api-Resource-Id：
  2026-09-25 实测传 `seed-audio-1.0` 会 403 code=45000030
  （resource_id=volc.service_type.10074 not granted），不传则由服务端按 model
  解析到同一资源——官方确认邀测制无法自助开通，此后端当前被授权阻塞。
- fal：fal.ai 托管的同一模型，endpoint id `bytedance/seed-audio-1.0`，
  同步 `POST https://fal.run/bytedance/seed-audio-1.0`，
  `Authorization: Key $FAL_KEY`（文档
  https://fal.ai/models/bytedance/seed-audio-1.0/api ）。
  参数与官方版不同：prompt（@AudioN 引用）/ audio_urls（公网 URL 或
  base64 data URI，≤3 条、各 ≤30s ≤10MB）/ output_format / sample_rate /
  speed / volume / pitch / multilingual（非英文要开）；返回
  {"audio": {"url", "duration", ...}}，无字级时间戳。计费 $0.1875/分钟。

用法：
    python3 /home/xy/duet-ad1-h3-max-r2v/scripts/audio_poc/seed_audio_scheme.py \
        [--backend volcengine|fal] [--projects b6240a72,c57d9d6c] \
        [--max-lines N] [--dry-run] [--neighbor-refs]

Key 从 /home/xy/.config/duet-ad1/service.env 与 /home/xy/duet-ad1-h3-max-r2v/.env
读取（volcengine 用 VOLC_VOICE_API_KEY，fal 用 FAL_KEY；后者覆盖前者）；
任何情况下不打印、不落盘 key 值。
付费接口不做重试：失败行记录后继续；授权/路由类致命错误（火山 45000030、
fal 401/403/404）首行即整体中止。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

SEED_AUDIO_API_URL = "https://openspeech.bytedance.com/api/v3/tts/create"
SEED_AUDIO_MODEL = "seed-audio-1.0"
FAL_ENDPOINT_ID = "bytedance/seed-audio-1.0"
FAL_SYNC_URL = f"https://fal.run/{FAL_ENDPOINT_ID}"
OUTPUT_SAMPLE_RATE = 24000
TRACK_SAMPLE_RATE = 16000  # original-track.wav / 参考片段统一 16kHz 单声道
MAX_REFERENCE_SECONDS = 30.0  # 接口上限：单条参考音频 ≤30s、≤10MB
MAX_TEMPO_RATIO = 1.15  # 超长压缩上限 ±15%，超限记 warning
REQUEST_TIMEOUT_S = 300

ENV_FILES = [
    "/home/xy/.config/duet-ad1/service.env",
    "/home/xy/duet-ad1-h3-max-r2v/.env",
]
POC_ROOT = "/home/xy/duet-ad1-h3-max-r2v/data-verify/audio-poc-20260925"
PROJECTS = {
    "b6240a72": "/home/xy/duet-ad1-h3-max-r2v/data/projects/b6240a721a244447abc697c55f6f9e1f",
    "c57d9d6c": "/home/xy/duet-ad1-h3-max-r2v/data/projects/c57d9d6c4c924c1a90b7fff35bdc6d64",
    "1cc3adb4": "/home/xy/duet-ad1-h3-max-r2v/data/projects/1cc3adb4e58442ea926242bc3ecb3063",
    "e46bf03d": "/home/xy/duet-ad1-h3-max-r2v/data/projects/e46bf03dcc514d06a6d56846ddb382ab",
    "6ecc60ca": "/home/xy/duet-ad1-h3-max-r2v/data/projects/6ecc60ca953147978c6926fdf3887e38",
}


def load_env_value(name: str) -> str:
    values: dict[str, str] = {}
    for path in ENV_FILES:
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    values[key.strip()] = value.strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    value = values.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} missing in service.env / .env")
    return value


def run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:400]}")


def ensure_original_track(project_dir: str, work_dir: Path) -> Path:
    """从 source.mp4 抽 16kHz 单声道原音轨；已存在且非空则复用
    （与并行任务共享同一文件，tmp+rename 原子落盘避免读到半成品）。"""
    destination = work_dir / "original-track.wav"
    work_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 44:
        return destination
    tmp = destination.with_suffix(".wav.tmp")
    run_ffmpeg([
        "-i", str(Path(project_dir) / "inputs" / "source.mp4"),
        "-vn", "-ac", "1", "-ar", str(TRACK_SAMPLE_RATE), "-f", "wav", str(tmp),
    ])
    os.replace(tmp, destination)
    return destination


def slice_line_clips(
    lines: list[dict], track: Path, clips_dir: Path
) -> list[Path]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for index, line in enumerate(lines):
        clip = clips_dir / f"line-{index:04d}.wav"
        if not (clip.exists() and clip.stat().st_size > 44):
            tmp = clip.with_suffix(".wav.tmp")
            run_ffmpeg([
                "-ss", f"{line['start_s']:.6f}", "-to", f"{line['end_s']:.6f}",
                "-i", str(track),
                "-ac", "1", "-ar", str(TRACK_SAMPLE_RATE), "-f", "wav", str(tmp),
            ])
            os.replace(tmp, clip)
        clips.append(clip)
    return clips


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def build_text_prompt(text: str, window_s: float, neighbor_count: int) -> str:
    """参考音频模式：@音频N 编号与 references 上传顺序严格对应（文档要求）。
    时间轴控制按文档用自然语言写明总时长目标。"""
    refs = "、".join(f"@音频{i + 1}" for i in range(1 + neighbor_count))
    if neighbor_count:
        style = f"{refs}是同一位说话人的相邻片段，请参考其中说话人的音色、语速、语调和语气"
    else:
        style = "请参考@音频1中说话人的音色、语速、语调和语气"
    return (
        f"{style}，以广告口播的感觉，用约{window_s:.1f}秒的总时长"
        f"说出以下文本：{text}"
    )


class GrantNotFoundError(RuntimeError):
    """code=45000030：当前 Key 未开通 seed-audio-1.0，重跑无意义，整体中止。"""


class FalFatalError(RuntimeError):
    """fal 侧 401/403/404：key 无效或模型路由不存在，重跑无意义，整体中止。"""


def build_fal_prompt(text: str, window_s: float, neighbor_count: int) -> str:
    """fal 版引用语法是 @AudioN（编号与 audio_urls 顺序对应）。"""
    refs = "、".join(f"@Audio{i + 1}" for i in range(1 + neighbor_count))
    if neighbor_count:
        style = f"{refs}是同一位说话人的相邻片段，请参考其中说话人的音色、语速、语调和语气"
    else:
        style = "请参考 @Audio1 中说话人的音色、语速、语调和语气"
    return (
        f"{style}，以广告口播的感觉，用约{window_s:.1f}秒的总时长"
        f"说出以下文本：{text}"
    )


def synthesize_line_fal(
    *,
    api_key: str,
    text: str,
    window_s: float,
    reference_clips: list[Path],
) -> dict:
    """fal 同步端点：input JSON 直接作 body；参考音频走 base64 data URI
    （≤30s 16kHz 单声道片段约 1MB 内，省去单独上传步骤）。"""
    audio_urls = []
    reference_records = []
    for clip in reference_clips:
        raw = clip.read_bytes()
        audio_urls.append("data:audio/wav;base64," + base64.b64encode(raw).decode())
        reference_records.append({
            "clip": clip.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
            "duration_s": round(wav_duration(clip), 3),
        })
    body = {
        "prompt": build_fal_prompt(text, window_s, len(reference_clips) - 1),
        "audio_urls": audio_urls,
        "output_format": "wav",
        "sample_rate": OUTPUT_SAMPLE_RATE,
        "multilingual": True,  # 译文为马来/印尼语，按 fal 文档开多语种变体
    }
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    record: dict = {
        "endpoint": FAL_SYNC_URL,
        "fal_endpoint_id": FAL_ENDPOINT_ID,
        "prompt": body["prompt"],
        "references": reference_records,
        "fal_input": {k: v for k, v in body.items() if k not in ("prompt", "audio_urls")},
    }
    request = urllib.request.Request(
        FAL_SYNC_URL, data=json.dumps(body).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, exc.read()
    record["http_status"] = status
    if status in (401, 403, 404):
        record["error"] = payload[:300].decode("utf-8", "replace")
        raise FalFatalError(f"HTTP {status}: {record['error']}")
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        record["error"] = f"non-JSON response head={payload[:120]!r}"
        return {"record": record, "audio": None}
    if status != 200:
        record["error"] = json.dumps(result, ensure_ascii=False)[:300]
        return {"record": record, "audio": None}
    audio_info = result.get("audio") or {}
    audio_url = audio_info.get("url")
    if not audio_url:
        record["error"] = f"no audio.url in response: {json.dumps(result)[:200]}"
        return {"record": record, "audio": None}
    record["audio_url_host"] = audio_url.split("/")[2]
    record["duration_s"] = audio_info.get("duration")
    record["file_size"] = audio_info.get("file_size")
    record["fal_request_id"] = result.get("requestId") or result.get("request_id")
    try:
        with urllib.request.urlopen(audio_url, timeout=REQUEST_TIMEOUT_S) as resp:
            audio = resp.read()
    except Exception as exc:
        record["error"] = f"audio download failed: {exc}"
        return {"record": record, "audio": None}
    return {"record": record, "audio": audio}


def synthesize_line(
    *,
    api_key: str,
    text: str,
    window_s: float,
    reference_clips: list[Path],
) -> dict:
    references = []
    reference_records = []
    for clip in reference_clips:
        raw = clip.read_bytes()
        references.append({"audio_data": base64.b64encode(raw).decode()})
        reference_records.append({
            "clip": clip.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
            "duration_s": round(wav_duration(clip), 3),
        })
    body = {
        "model": SEED_AUDIO_MODEL,
        "text_prompt": build_text_prompt(text, window_s, len(reference_clips) - 1),
        "references": references,
        "audio_config": {
            "format": "wav",
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "enable_subtitle": True,
        },
    }
    request_id = str(uuid.uuid4())
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Request-Id": request_id,
        "Content-Type": "application/json",
    }
    record: dict = {
        "request_id": request_id,
        "endpoint": SEED_AUDIO_API_URL,
        "model": SEED_AUDIO_MODEL,
        "text_prompt": body["text_prompt"],
        "references": reference_records,
        "audio_config": body["audio_config"],
    }
    request = urllib.request.Request(
        SEED_AUDIO_API_URL, data=json.dumps(body).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as resp:
            status, payload = resp.status, resp.read()
            record["http_status"] = status
            record["logid"] = resp.headers.get("X-Tt-Logid")
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, exc.read()
        record["http_status"] = status
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        record["error"] = f"non-JSON response head={payload[:120]!r}"
        return {"record": record, "audio": None}
    record["code"] = result.get("code")
    record["message"] = result.get("message")
    record["duration_s"] = result.get("duration")
    record["original_duration_s"] = result.get("original_duration")
    if record["code"] == 45000030:
        raise GrantNotFoundError(str(record["message"]))
    audio_b64 = result.get("audio")
    if not audio_b64:
        record["error"] = "no audio in response"
        return {"record": record, "audio": None}
    subtitle = result.get("subtitle") or {}
    sentences = subtitle.get("sentences") or []
    record["subtitle_sentences"] = len(sentences)
    record["subtitle_first"] = sentences[0] if sentences else None
    return {"record": record, "audio": base64.b64decode(audio_b64)}


def apply_atempo(source: Path, factor: float, destination: Path) -> None:
    run_ffmpeg([
        "-i", str(source),
        "-af", f"atempo={factor:.4f}",
        "-ac", "1", "-ar", str(OUTPUT_SAMPLE_RATE), "-f", "wav", str(destination),
    ])


def read_wav_samples(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16bit mono wav")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return rate, list(struct.unpack(f"<{len(frames) // 2}h", frames))


def assemble_full_track(
    lines: list[dict],
    synth_paths: dict[int, Path],
    window_fits: dict[int, float],
    total_s: float,
    destination: Path,
) -> None:
    """静音底 24kHz 单声道，按 start_s 叠加各行（溢出窗口的部分裁掉，
    裁切本身已在 durations.json 记 warning）。"""
    total_samples = int(round(total_s * OUTPUT_SAMPLE_RATE)) + OUTPUT_SAMPLE_RATE
    mix = [0] * total_samples
    for index, path in synth_paths.items():
        rate, samples = read_wav_samples(path)
        if rate != OUTPUT_SAMPLE_RATE:
            raise ValueError(f"{path} sample rate {rate} != {OUTPUT_SAMPLE_RATE}")
        start = int(round(lines[index]["start_s"] * OUTPUT_SAMPLE_RATE))
        window_samples = int(round(window_fits[index] * OUTPUT_SAMPLE_RATE))
        for offset, value in enumerate(samples[:window_samples]):
            pos = start + offset
            if pos >= total_samples:
                break
            mixed = mix[pos] + value
            mix[pos] = max(-32768, min(32767, mixed))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(OUTPUT_SAMPLE_RATE)
        handle.writeframes(struct.pack(f"<{len(mix)}h", *mix))


def process_project(
    short_id: str,
    project_dir: str,
    api_key: str,
    *,
    backend: str,
    max_lines: int | None,
    dry_run: bool,
    neighbor_refs: bool,
) -> dict:
    state = json.loads((Path(project_dir) / "state.json").read_text(encoding="utf-8"))
    lines = state.get("translated_voice_lines") or []
    if max_lines is not None:
        lines = lines[:max_lines]
    total_s = float(state.get("duration_s") or max(l["end_s"] for l in lines))

    work_dir = Path(POC_ROOT) / "projects" / short_id
    out_name = "seed-audio" if backend == "volcengine" else "seed-audio-fal"
    out_dir = Path(POC_ROOT) / "outputs" / out_name / short_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "request-log.jsonl"
    durations_path = out_dir / "durations.json"

    track = ensure_original_track(project_dir, work_dir)
    clips = slice_line_clips(lines, track, work_dir / "line-clips-seed")
    summary = {"short_id": short_id, "lines": len(lines), "ok": 0, "failed": []}
    if dry_run:
        summary["dry_run"] = True
        return summary

    durations: list[dict] = []
    synth_paths: dict[int, Path] = {}
    window_fits: dict[int, float] = {}
    with open(log_path, "w", encoding="utf-8") as log:
        for index, (line, clip) in enumerate(zip(lines, clips)):
            window_s = float(line["end_s"]) - float(line["start_s"])
            window_fits[index] = window_s
            reference_clips = [clip]
            if neighbor_refs:  # 相邻行片段作 @音频2/@音频3 增强（各 ≤30s，最多 3 条）
                for neighbor in (index - 1, index + 1):
                    if 0 <= neighbor < len(clips) and len(reference_clips) < 3:
                        if wav_duration(clips[neighbor]) <= MAX_REFERENCE_SECONDS:
                            reference_clips.append(clips[neighbor])
            entry = {"line_index": index, "text": line["text"],
                     "start_s": line["start_s"], "end_s": line["end_s"],
                     "window_s": round(window_s, 3)}
            try:
                if backend == "fal":
                    result = synthesize_line_fal(
                        api_key=api_key, text=line["text"], window_s=window_s,
                        reference_clips=reference_clips,
                    )
                else:
                    result = synthesize_line(
                        api_key=api_key, text=line["text"], window_s=window_s,
                        reference_clips=reference_clips,
                    )
            except (GrantNotFoundError, FalFatalError) as exc:
                entry["error"] = str(exc)
                entry["outcome"] = "failed_fatal"
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                durations.append(entry)
                durations_path.write_text(
                    json.dumps(durations, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise
            entry.update(result["record"])
            audio = result["audio"]
            if audio is None:
                summary["failed"].append({"line_index": index,
                                          "error": entry.get("error")
                                          or f"code={entry.get('code')} {entry.get('message')}"})
                entry["outcome"] = "failed"
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                durations.append(entry)
                continue
            wav_path = out_dir / f"line-{index:04d}.wav"
            wav_path.write_bytes(audio)
            # fal 实测返回 24kHz 立体声（官方版为单声道），统一转 16bit 单声道
            # 24kHz 再进装配（tmp+rename 原地替换）。
            try:
                read_wav_samples(wav_path)
            except ValueError:
                tmp = wav_path.with_suffix(".wav.tmp")
                run_ffmpeg([
                    "-i", str(wav_path), "-ac", "1",
                    "-ar", str(OUTPUT_SAMPLE_RATE), "-f", "wav", str(tmp),
                ])
                os.replace(tmp, wav_path)
            synth_s = wav_duration(wav_path)
            placed_path = wav_path
            warning = None
            if synth_s > window_s:
                factor = synth_s / window_s
                if factor > MAX_TEMPO_RATIO:
                    warning = (f"tempo ratio {factor:.3f} exceeds "
                               f"{MAX_TEMPO_RATIO}; capped, tail trimmed")
                    factor = MAX_TEMPO_RATIO
                fitted = out_dir / f"line-{index:04d}.fitted.wav"
                apply_atempo(wav_path, factor, fitted)
                placed_path = fitted
                entry["atempo_factor"] = round(factor, 4)
            entry["synth_duration_s"] = round(synth_s, 3)
            entry["deviation_s"] = round(synth_s - window_s, 3)
            entry["deviation_ratio"] = round(synth_s / window_s, 4) if window_s else None
            if warning:
                entry["warning"] = warning
            entry["outcome"] = "synthesized"
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            durations.append(entry)
            synth_paths[index] = placed_path
            summary["ok"] += 1
    if synth_paths:
        assemble_full_track(lines, synth_paths, window_fits, total_s,
                            out_dir / "full-track.wav")
        summary["full_track"] = str(out_dir / "full-track.wav")
    durations_path.write_text(
        json.dumps(durations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["volcengine", "fal"],
                        default="volcengine")
    parser.add_argument("--projects", default=",".join(PROJECTS))
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="只准备原音轨与参考片段，不调用付费接口")
    parser.add_argument("--neighbor-refs", action="store_true",
                        help="参考音频附带相邻行片段（@音频2/@Audio2、@音频3/@Audio3）")
    args = parser.parse_args()
    if args.dry_run:
        api_key = ""
    elif args.backend == "fal":
        api_key = load_env_value("FAL_KEY")
    else:
        api_key = load_env_value("VOLC_VOICE_API_KEY")

    summaries = []
    for short_id in args.projects.split(","):
        short_id = short_id.strip()
        if short_id not in PROJECTS:
            raise SystemExit(f"unknown project {short_id}")
        try:
            summaries.append(process_project(
                short_id, PROJECTS[short_id], api_key,
                backend=args.backend,
                max_lines=args.max_lines, dry_run=args.dry_run,
                neighbor_refs=args.neighbor_refs,
            ))
        except GrantNotFoundError as exc:
            print(f"[{short_id}] ABORT: key not granted for seed-audio-1.0 "
                  f"(volc.service_type.10074): {exc}", file=sys.stderr)
            print("需要在火山引擎控制台为当前 API Key 开通"
                  "「音频生成大模型 / Doubao-Seed-Audio-1.0」后重跑。", file=sys.stderr)
            return 2
        except FalFatalError as exc:
            print(f"[{short_id}] ABORT: fal fatal error: {exc}", file=sys.stderr)
            return 2
        print(f"[{short_id}] {json.dumps(summaries[-1], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
