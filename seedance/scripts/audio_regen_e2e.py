#!/usr/bin/env python
"""纯音频复刻 E2E：对 5 个生产项目只重跑新音频链（方舟分析 → delivery
规划 → 豆包 TTS → 装配 → ducking 混音 → 与生产成片合成 final.mp4）。

画面复用 3211 生产成片 output/generated.mp4（-c:v copy），不跑
Seedream/Picsart/H3 任何视频侧付费调用。对 data/projects 只读，全部中间
产物落输出目录 /home/xy/duet-ad1-h3-max-r2v/data-verify/audio-regen-20260926/<内容名>/。

与生产链的唯一差异（2026-09-27 音色逐行绑定修复）：不继承 state.json 的
旧 dialogue_voices 绑定，每行台词按时间窗归属到 segment，取该段「画面主
口播人物」（首帧最早的成年人物，儿童永不作依据）的性别选音色，与旧绑定
性别一致才保留，否则按语言 × 性别重选。explicit_language 全量锁：所有
*_uranus_bigtts 精品音色（多语种混合训练，数字/单位 token 概率性串语种，
防晒霜 "60 ml" 读成中文实证）一律锁定目标朗读语种。解析规则与生产共用
app.dialogue_delivery.resolve_line_voices；方舟 gender_age 只作无画面
证据时的兜底与报告参考信号。

用法（路径全绝对，key 只走 env）：
  ARK_AUDIO_API_KEY=... ARK_AUDIO_MODEL=doubao-seed-2-0-lite-260428 \
  ARK_AUDIO_API_BASE=https://ark.cn-beijing.volces.com/api/v3 \
  /home/xy/duet-ad1/.venv/bin/python /home/xy/duet-ad1-h3-max-r2v/scripts/audio_regen_e2e.py

VOLC_VOICE_API_KEY 从 /home/xy/.config/duet-ad1/service.env 读取，
DEEPSEEK_API_KEY 从 /home/xy/.config/claude/deepseek.env 读取，
DEEPSEEK_API_BASE/DEEPSEEK_MODEL 从 /home/xy/duet-ad1-h3-max-r2v/.env 读取
（均已在 os.environ 时优先用环境值；脚本永不打印 key）。

幂等续跑：accompaniment.wav、analysis.json、delivery.json、逐行 wav
（按合成输入哈希比对）存在即复用；装配/混音/合成片为本地 ffmpeg，
每次重跑。

干跑（不付费、不写产物，只输出逐行音色解析与现状对比）：
  /home/xy/duet-ad1/.venv/bin/python /home/xy/duet-ad1-h3-max-r2v/scripts/audio_regen_e2e.py --resolve-only

单项目重跑（其他项目不重新合成）：
  ... /home/xy/duet-ad1-h3-max-r2v/scripts/audio_regen_e2e.py --only 防晒霜
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, "/home/xy/duet-ad1-h3-max-r2v")

from app import dialogue_delivery, media, vocal_separation  # noqa: E402
from app.config import Settings  # noqa: E402
from app.doubao_tts import DoubaoTtsClient  # noqa: E402
from app.doubao_voices import (  # noqa: E402
    is_calibrated_voice,
    load_doubao_voice_catalog,
)
from app.pipeline import (  # noqa: E402
    ARK_AUDIO_ANALYSIS_PROMPT,
    _dialogue_language_display,
    parse_ark_audio_analysis,
)
from app.providers import ArkAudioClient, DeepSeekClient  # noqa: E402

PROJECTS_ROOT = Path("/home/xy/duet-ad1-h3-max-r2v/data/projects")
OUTPUT_ROOT = Path("/home/xy/duet-ad1-h3-max-r2v/data-verify/audio-regen-20260926")
SEPARATION_CLI = Path(
    "/home/xy/duet-ad1-h3-max-r2v/.venv-separation/bin/audio-separator"
)
SEPARATION_MODEL = Path(
    "/home/xy/duet-ad1-h3-max-r2v/models/UVR-MDX-NET-Inst_HQ_3.onnx"
)
SERVICE_ENV = Path("/home/xy/.config/duet-ad1/service.env")
DEEPSEEK_ENV = Path("/home/xy/.config/claude/deepseek.env")
REPO_ENV = Path("/home/xy/duet-ad1-h3-max-r2v/.env")

# ducking 终值（2026-09-26 POC 标定，与 pipeline mix-final-audio 节点一致，
# 显式传入不依赖 media 层缺省）。
DUCK_PARAMS = {
    "duck_threshold": 0.01,
    "duck_ratio": 12,
    "duck_attack_ms": 10,
    "duck_release_ms": 300,
    "accompaniment_gain_db": -6.0,
}

PROJECTS = [
    ("b6240a72", "b6240a721a244447abc697c55f6f9e1f", "电热睫毛夹"),
    ("c57d9d6c", "c57d9d6c4c924c1a90b7fff35bdc6d64", "户外灯串"),
    ("1cc3adb4", "1cc3adb4e58442ea926242bc3ecb3063", "车载湿巾"),
    ("e46bf03d", "e46bf03dcc514d06a6d56846ddb382ab", "庭院灯串"),
    ("6ecc60ca", "6ecc60ca953147978c6926fdf3887e38", "防晒霜"),
]


def _parse_env_file(path: Path, keys: set[str]) -> dict[str, str]:
    """从 KEY=VALUE / export KEY=VALUE 文件提取白名单 key；不打印值。"""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name in keys and name not in values:
            values[name] = value.strip().strip('"').strip("'")
    return values


def _load_credentials() -> dict[str, str]:
    wanted = {
        "VOLC_VOICE_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_MODEL",
        "ARK_AUDIO_API_KEY",
        "ARK_AUDIO_MODEL",
        "ARK_AUDIO_API_BASE",
    }
    values: dict[str, str] = {}
    for path in (SERVICE_ENV, DEEPSEEK_ENV, REPO_ENV):
        for name, value in _parse_env_file(path, wanted).items():
            values.setdefault(name, value)
    for name in wanted:
        env_value = os.environ.get(name, "").strip()
        if env_value:
            values[name] = env_value
    missing = [name for name in sorted(wanted) if not values.get(name)]
    if missing:
        raise RuntimeError(
            f"missing credentials/config: {', '.join(missing)} "
            "(ARK_AUDIO_* 须在命令行 env 注入，其余见脚本 docstring)"
        )
    return values


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_project_inputs(
    short_id: str, project_id: str, catalog
) -> dict[str, object]:
    """加载项目的只读输入：state 行/段、enhanced_element_index（work/ 下
    为准，缺失回退 state 内嵌副本）、语言标签与旧音色绑定。run_project 与
    --resolve-only 干跑共用。"""
    project_dir = PROJECTS_ROOT / project_id
    state = _read_json(project_dir / "state.json")
    translated_lines = state.get("translated_voice_lines")
    if not isinstance(translated_lines, list) or not translated_lines:
        raise RuntimeError(f"{short_id}: translated_voice_lines missing")
    segments = state.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError(f"{short_id}: segments missing")
    timeline_end = max(float(item["end_s"]) for item in segments)
    enhanced_index: object = None
    work_index = project_dir / "work" / "enhanced_element_index.json"
    if work_index.is_file():
        enhanced_index = _read_json(work_index)
    if not isinstance(enhanced_index, dict):
        enhanced_index = state.get("enhanced_element_index")
    if not isinstance(enhanced_index, dict):
        raise RuntimeError(f"{short_id}: enhanced_element_index missing")

    clamped_lines: list[dict[str, object]] = []
    for line in translated_lines:
        clamped_lines.append(
            {
                "text": str(line["text"]),
                "start_s": min(float(line["start_s"]), timeline_end),
                "end_s": min(float(line["end_s"]), timeline_end),
            }
        )
    language_label = _dialogue_language_display(
        str(state["effective_request"]["dialogue"]["target_language"]),
        translated_lines,
        state.get("source_language_code"),
    )
    dialogue_voices = enhanced_index.get("dialogue_voices")
    if not isinstance(dialogue_voices, list):
        dialogue_voices = []
    if len(dialogue_voices) < len(clamped_lines):
        dialogue_voices = dialogue_voices + [None] * (
            len(clamped_lines) - len(dialogue_voices)
        )
    default_voice_type = catalog.default_voice_type(language_label)
    # 与 app/pipeline.py 消费侧同一口径：无绑定、目录外或未标定的冻结
    # 绑定一律回落到该语种已标定默认音色。
    inherited_voices = [
        voice
        if catalog.has(voice) and is_calibrated_voice(voice)
        else default_voice_type
        for voice in dialogue_voices[: len(clamped_lines)]
    ]
    people = enhanced_index.get("people")
    return {
        "project_dir": project_dir,
        "segments": segments,
        "clamped_lines": clamped_lines,
        "language_label": language_label,
        "inherited_voices": inherited_voices,
        "people": people if isinstance(people, dict) else {},
    }


def _load_analysis_lines(out_dir: Path, line_count: int) -> list[dict[str, object]]:
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.is_file():
        return []
    cached = _read_json(analysis_path)
    return [
        item
        for item in (cached.get("lines", []) if isinstance(cached, dict) else [])
        if isinstance(item, dict)
    ][:line_count]


def run_project(
    short_id: str,
    project_id: str,
    content_name: str,
    settings: Settings,
    catalog,
) -> dict[str, object]:
    inputs = _load_project_inputs(short_id, project_id, catalog)
    project_dir = inputs["project_dir"]
    segments = inputs["segments"]
    clamped_lines = inputs["clamped_lines"]
    language_label = inputs["language_label"]
    inherited_voices = inputs["inherited_voices"]
    people = inputs["people"]
    out_dir = OUTPUT_ROOT / content_name
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "ark-clips"
    total_duration = sum(float(item["duration_s"]) for item in segments)

    # 2. 音源：优先生产 work/source.wav（只读复制语义，直接引用路径切片）。
    source_wav = project_dir / "work" / "source.wav"
    if not source_wav.is_file():
        source_wav = out_dir / "source.wav"
        if not source_wav.is_file():
            extracted = media.extract_audio(
                project_dir / "inputs" / "source.mp4", source_wav
            )
            if extracted is None:
                raise RuntimeError(f"{short_id}: source video has no audio track")

    # 3. 人声分离 → accompaniment.wav。
    accompaniment = out_dir / "accompaniment.wav"
    if not accompaniment.is_file():
        vocal_separation.separate_vocals(
            source_wav,
            accompaniment,
            out_dir / "vocals.wav",
            cli_path=SEPARATION_CLI,
            model_path=SEPARATION_MODEL,
        )

    # 4. 方舟逐行音频理解（按行缓存，续跑不重付）。
    analysis_path = out_dir / "analysis.json"
    cached_analysis = (
        _read_json(analysis_path) if analysis_path.is_file() else {"lines": []}
    )
    analysis_by_index = {
        int(item["line_index"]): item
        for item in cached_analysis.get("lines", [])
        if isinstance(item, dict)
    }
    ark_client: ArkAudioClient | None = None
    analysis_lines: list[dict[str, object]] = []
    for line_index, line in enumerate(clamped_lines, 1):
        start_s = float(line["start_s"])
        end_s = float(line["end_s"])
        cached = analysis_by_index.get(line_index)
        if (
            isinstance(cached, dict)
            and cached.get("start_s") == start_s
            and cached.get("end_s") == end_s
        ):
            analysis_lines.append(cached)
            continue
        if ark_client is None:
            ark_client = ArkAudioClient(settings)
        clip = clips_dir / f"line-{line_index:04d}.wav"
        media.slice_dialogue_track(source_wav, start_s, end_s - start_s, clip)
        # 与 pipeline _node 语义一致：模型自由 JSON 输出 malformed 属暂时性
        # 失败，原地重试（每次重新付费，故限制 3 次）。
        description: dict[str, str] | None = None
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                result = ark_client.analyze(clip, ARK_AUDIO_ANALYSIS_PROMPT)
                description = parse_ark_audio_analysis(result.value)
                break
            except Exception as exc:  # noqa: BLE001 - 重试后仍失败则抛出
                last_error = exc
        if description is None:
            raise last_error  # type: ignore[misc]
        item = {
            "line_index": line_index,
            "start_s": start_s,
            "end_s": end_s,
            **description,
        }
        analysis_lines.append(item)
        analysis_by_index[line_index] = item
        _write_json(
            analysis_path, {"lines": sorted(
                analysis_by_index.values(), key=lambda entry: entry["line_index"]
            )}
        )

    # 4.5 逐行音色绑定（bug 修复 2026-09-27 v2）：不继承 state.json 的旧
    # dialogue_voices 绑定逐字重放，也不按项目级单一音色一刀切（b6240a72
    # 电热睫毛夹：原片全程男声旁白，但画面 seg1 男性演示、seg2/3 女性演示，
    # 用户要求口播音色跟随画面里的成年演示者）。每行按时间窗归属 segment，
    # 取画面主口播人物性别：与旧绑定一致保留（不重折腾），不一致按语言 ×
    # 性别重选；方舟 gender_age 降级为无画面证据时的兜底与报告参考信号
    # （原片旁白与画面人物可以不一致，音频信号不可信）。规则与生产共用
    # app.dialogue_delivery.resolve_line_voices。
    voice_resolution = dialogue_delivery.resolve_line_voices(
        lines=clamped_lines,
        segments=[
            {
                "index": int(item["index"]),
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
            }
            for item in segments
        ],
        people=people,
        inherited_voices=list(inherited_voices),
        catalog=catalog,
        language_label=language_label,
        analysis_lines=analysis_lines,
    )
    line_voices = [entry["voice_type"] for entry in voice_resolution]
    line_explicit_languages = [
        entry["explicit_language"] for entry in voice_resolution
    ]
    print(
        f"[audio-regen] {content_name} voices: "
        + json.dumps(
            [
                {
                    "line": entry["line_index"],
                    "seg": entry["segment_index"],
                    "person": entry["person_id"],
                    "gender": entry["gender"],
                    "voice": entry["voice_type"],
                    "explicit_language": entry["explicit_language"],
                }
                for entry in voice_resolution
            ],
            ensure_ascii=False,
        ),
        flush=True,
    )

    # 5. delivery plan（按 input_sha256 + instruction_sha256 缓存，输入或
    # 指令文本任一变化都重付 DeepSeek；instruction 哈希与 pipeline 同判据，
    # 防止指令改了旧 plan 被复用）。
    payload = dialogue_delivery.build_dialogue_delivery_payload(
        target_language=language_label,
        translated_lines=clamped_lines,
        line_voices=line_voices,
        catalog=catalog,
        analysis_lines=analysis_lines,
    )
    output_schema, input_sha256 = dialogue_delivery.dialogue_delivery_schema(payload)
    instruction_sha256 = dialogue_delivery.dialogue_delivery_instruction_sha256(
        language_label,
    )
    delivery_path = out_dir / "delivery.json"
    plan_lines: list[dict[str, object]] | None = None
    if delivery_path.is_file():
        cached_delivery = _read_json(delivery_path)
        if (
            isinstance(cached_delivery, dict)
            and cached_delivery.get("input_sha256") == input_sha256
            and cached_delivery.get("instruction_sha256") == instruction_sha256
            and isinstance(cached_delivery.get("lines"), list)
            and len(cached_delivery["lines"]) == len(clamped_lines)
        ):
            plan_lines = cached_delivery["lines"]
    if plan_lines is None:
        deepseek = DeepSeekClient(settings)

        def normalize(value: object) -> list[dict[str, object]]:
            return dialogue_delivery.normalize_dialogue_delivery(
                value,
                input_sha256=input_sha256,
                line_count=len(clamped_lines),
            )

        result = deepseek.complete_structured(
            dialogue_delivery.dialogue_delivery_instruction(language_label),
            payload,
            output_schema,
            [],
            validator=normalize,
        )
        plan_lines = normalize(result.value)
        _write_json(
            delivery_path,
            {
                "input_sha256": input_sha256,
                "instruction_sha256": instruction_sha256,
                "lines": plan_lines,
            },
        )

    # 6. 逐行 TTS（按合成输入哈希复用已确认行 wav）。
    run_id = str(uuid.uuid4())
    tts = DoubaoTtsClient(settings)
    tts_cache_path = out_dir / "tts-lines.json"
    tts_cache = (
        _read_json(tts_cache_path) if tts_cache_path.is_file() else {"lines": []}
    )
    confirmed = {
        int(item["line_index"]): str(item["input_sha256"])
        for item in tts_cache.get("lines", [])
        if isinstance(item, dict)
    }
    section_ids = {
        voice_type: dialogue_delivery.dialogue_tts_section_id(run_id, voice_type)
        for voice_type in set(line_voices)
    }
    line_audios: list[dict[str, object]] = []
    for offset, (line, voice_type) in enumerate(zip(clamped_lines, line_voices), 1):
        start_s = float(line["start_s"])
        end_s = float(line["end_s"])
        if start_s >= end_s:
            continue
        plan = plan_lines[offset - 1]
        speech_rate = plan.get("speech_rate")
        emotion = plan.get("emotion")
        context_texts = plan.get("context_texts")
        destination = out_dir / f"line-{offset:04d}.wav"
        input_hash = dialogue_delivery.dialogue_tts_line_input_sha256(
            text=str(line["text"]),
            voice_type=voice_type,
            start_s=start_s,
            end_s=end_s,
            speech_rate=speech_rate,
            emotion=emotion,
            context_texts=context_texts,
            explicit_language=line_explicit_languages[offset - 1],
        )
        if confirmed.get(offset) == input_hash and (
            dialogue_delivery.is_decodable_wav(destination)
        ):
            line_audios.append(
                {"audio": destination, "start_s": start_s, "end_s": end_s}
            )
            continue
        tts.synthesize(
            text=str(line["text"]),
            voice_type=voice_type,
            destination=destination,
            speech_rate=speech_rate,
            emotion=emotion,
            context_texts=context_texts,
            section_id=section_ids[voice_type],
            explicit_language=line_explicit_languages[offset - 1],
        )
        confirmed[offset] = input_hash
        _write_json(
            tts_cache_path,
            {
                "lines": [
                    {"line_index": index, "input_sha256": value}
                    for index, value in sorted(confirmed.items())
                ]
            },
        )
        line_audios.append(
            {"audio": destination, "start_s": start_s, "end_s": end_s}
        )

    # 7. 装配 + ducking 混音。
    full_track = out_dir / "full-track.wav"
    assembled = media.assemble_dialogue_track(
        line_audios, total_duration, full_track
    )
    mixed_track = out_dir / "mixed-track.wav"
    media.mix_dialogue_with_accompaniment(
        full_track, accompaniment, mixed_track, **DUCK_PARAMS
    )

    # 8. 与生产成片合成（视频流 copy，音频 aac 48k，时长对齐视频）。
    generated = project_dir / "output" / "generated.mp4"
    final = out_dir / "final.mp4"
    media.replace_audio_track(generated, mixed_track, final)

    assembled_by_window = {
        (float(item["start_s"]), float(item["end_s"])): item
        for item in assembled.get("lines", [])
    }
    report_lines: list[dict[str, object]] = []
    for offset, (line, voice_type) in enumerate(zip(clamped_lines, line_voices), 1):
        start_s = float(line["start_s"])
        end_s = float(line["end_s"])
        plan = plan_lines[offset - 1]
        placement = assembled_by_window.get((start_s, end_s), {})
        audio_duration = placement.get("audio_duration_s")
        window_s = end_s - start_s
        ark = analysis_by_index.get(offset, {})
        report_lines.append(
            {
                "line_index": offset,
                "text": line["text"],
                "start_s": start_s,
                "end_s": end_s,
                "window_s": round(window_s, 6),
                "voice_type": voice_type,
                "explicit_language": line_explicit_languages[offset - 1],
                "previous_voice_type": inherited_voices[offset - 1],
                "speech_rate": plan.get("speech_rate"),
                "emotion": plan.get("emotion"),
                "context_texts": plan.get("context_texts"),
                "audio_duration_s": audio_duration,
                "deviation_s": (
                    round(float(audio_duration) - window_s, 6)
                    if isinstance(audio_duration, (int, float))
                    else None
                ),
                "tail_truncated_s": placement.get("tail_truncated_s", 0.0),
                "ark_analysis": {
                    field: ark.get(field)
                    for field in (
                        "speech_rate_desc",
                        "intonation_emotion",
                        "pause_pattern",
                        "overall_tone",
                        "gender_age",
                    )
                },
            }
        )
    speech_rates = [
        int(item["speech_rate"])
        for item in report_lines
        if isinstance(item["speech_rate"], int)
    ]
    placed = [
        item
        for item in report_lines
        if isinstance(item.get("audio_duration_s"), (int, float))
    ]
    report = {
        "project_id": project_id,
        "short_id": short_id,
        "content_name": content_name,
        "language_label": language_label,
        "run_id": run_id,
        # 逐行音色绑定：画面主口播人物优先，方舟 gender_age 仅参考/兜底。
        "voice_selection": {
            "strategy": "per_line_visual_speaker",
            "lines": [
                {
                    **{
                        key: entry[key]
                        for key in (
                            "line_index",
                            "segment_index",
                            "person_id",
                            "gender",
                            "gender_source",
                            "evidence",
                            "previous_voice_type",
                            "voice_type",
                            "explicit_language",
                            "reselected",
                        )
                    },
                    # 参考信号：原片旁白性别，不起决定作用。
                    "ark_gender_age": (
                        analysis_by_index.get(entry["line_index"], {}).get(
                            "gender_age"
                        )
                    ),
                }
                for entry in voice_resolution
            ],
        },
        "line_count": len(report_lines),
        "placed_line_count": len(placed),
        "speech_rate_distribution": {
            str(rate): speech_rates.count(rate) for rate in sorted(set(speech_rates))
        },
        "emotion_coverage": (
            sum(1 for item in report_lines if item.get("emotion"))
            / len(report_lines)
        ),
        "context_texts_coverage": (
            sum(1 for item in report_lines if item.get("context_texts"))
            / len(report_lines)
        ),
        "over_window_lines": [
            item["line_index"]
            for item in placed
            if float(item["deviation_s"]) > 0
        ],
        "truncated_lines": [
            item["line_index"]
            for item in placed
            if float(item.get("tail_truncated_s") or 0.0) > 0
        ],
        "max_abs_deviation_s": (
            max(abs(float(item["deviation_s"])) for item in placed) if placed else None
        ),
        "video_duration_s": media.probe(generated)["duration_s"],
        "final_duration_s": media.probe(final)["duration_s"],
        "lines": report_lines,
        "outputs": {
            "accompaniment": str(accompaniment),
            "full_track": str(full_track),
            "mixed_track": str(mixed_track),
            "final_mp4": str(final),
        },
    }
    _write_json(out_dir / "run-report.json", report)
    return report


def _resolve_only(catalog) -> int:
    """干跑：只输出逐行音色解析，与 state 旧绑定、上一次复刻 run-report 的
    现状音色三方对比。不付费、不写产物。"""
    results: list[dict[str, object]] = []
    for short_id, project_id, content_name in PROJECTS:
        try:
            inputs = _load_project_inputs(short_id, project_id, catalog)
            out_dir = OUTPUT_ROOT / content_name
            analysis_lines = _load_analysis_lines(
                out_dir, len(inputs["clamped_lines"])
            )
            resolution = dialogue_delivery.resolve_line_voices(
                lines=inputs["clamped_lines"],
                segments=[
                    {
                        "index": int(item["index"]),
                        "start_s": float(item["start_s"]),
                        "end_s": float(item["end_s"]),
                    }
                    for item in inputs["segments"]
                ],
                people=inputs["people"],
                inherited_voices=list(inputs["inherited_voices"]),
                catalog=catalog,
                language_label=inputs["language_label"],
                analysis_lines=analysis_lines,
            )
            current_voices: list[object] = []
            report_path = out_dir / "run-report.json"
            if report_path.is_file():
                report = _read_json(report_path)
                if isinstance(report, dict) and isinstance(report.get("lines"), list):
                    current_voices = [
                        item.get("voice_type") if isinstance(item, dict) else None
                        for item in report["lines"]
                    ]
            new_voices = [entry["voice_type"] for entry in resolution]
            results.append(
                {
                    "content_name": content_name,
                    "short_id": short_id,
                    "language_label": inputs["language_label"],
                    "lines": [
                        {
                            **entry,
                            "current_regen_voice": (
                                current_voices[entry["line_index"] - 1]
                                if entry["line_index"] - 1 < len(current_voices)
                                else None
                            ),
                        }
                        for entry in resolution
                    ],
                    "matches_current_regen": new_voices == current_voices,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 干跑汇总全部项目
            results.append(
                {
                    "content_name": content_name,
                    "short_id": short_id,
                    "error": repr(exc),
                }
            )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _selected_projects() -> list[tuple[str, str, str]]:
    """--only <内容名|short_id> 限定单项目重跑（如修复后只重跑防晒霜，
    其他项目不重新合成）；缺省全部。"""
    args = sys.argv[1:]
    if "--only" not in args:
        return list(PROJECTS)
    selector = args[args.index("--only") + 1]
    selected = [
        entry for entry in PROJECTS if selector in (entry[0], entry[2])
    ]
    if not selected:
        raise RuntimeError(f"--only {selector}: no matching project")
    return selected


def main() -> int:
    if "--resolve-only" in sys.argv[1:]:
        return _resolve_only(load_doubao_voice_catalog())
    credentials = _load_credentials()
    settings = Settings(
        access_token="audio-regen-e2e",
        public_base_url="http://127.0.0.1:3252",
        volc_voice_api_key=credentials["VOLC_VOICE_API_KEY"],
        deepseek_api_key=credentials["DEEPSEEK_API_KEY"],
        deepseek_api_base=credentials["DEEPSEEK_API_BASE"],
        deepseek_model=credentials["DEEPSEEK_MODEL"],
        ark_audio_api_key=credentials["ARK_AUDIO_API_KEY"],
        ark_audio_model=credentials["ARK_AUDIO_MODEL"],
        ark_audio_api_base=credentials["ARK_AUDIO_API_BASE"],
    )
    catalog = load_doubao_voice_catalog()
    failures: list[str] = []
    summaries: list[dict[str, object]] = []
    for short_id, project_id, content_name in _selected_projects():
        print(f"[audio-regen] {content_name} ({short_id}) ...", flush=True)
        try:
            report = run_project(
                short_id, project_id, content_name, settings, catalog
            )
        except Exception as exc:
            failures.append(f"{short_id} {content_name}: {exc!r}")
            print(f"[audio-regen] {content_name} FAILED: {exc!r}", flush=True)
            continue
        summaries.append(
            {
                "content_name": content_name,
                "short_id": short_id,
                "lines": report["line_count"],
                "truncated": report["truncated_lines"],
                "over_window": report["over_window_lines"],
                "max_abs_deviation_s": report["max_abs_deviation_s"],
                "final_mp4": report["outputs"]["final_mp4"],
            }
        )
        print(
            f"[audio-regen] {content_name} done: lines={report['line_count']} "
            f"truncated={report['truncated_lines']} "
            f"final={report['outputs']['final_mp4']}",
            flush=True,
        )
    print(json.dumps({"summaries": summaries, "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
