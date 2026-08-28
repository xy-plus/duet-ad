"""处理流水线：queued → processing → done|failed。

步骤：extract_keyframes --fps 4 抽帧 + 分页联系表 → （voice_mode ≠ none 时）抽音轨 +
codex 听写台词（voice_lines.json 白名单校验，落 meta.voice_lines）→ scenes.py 场景检测
（work/scenes.json）→ 按 segments 决定模式：
- 单段模式（segments 空）：codex 沙箱按 SKILL.md 选帧/写 prompt → 后端白名单校验 →
  meta 落盘（work/keyframes + work/prompt.txt，保持视觉 prompt 原文）。
- 多段模式（segments 非空）：ffmpeg 按段边界切源视频（work/segments/N/source.mp4，
  N 从 1 起），每段独立走抽帧 → codex prompt（单段/多段共用） → 校验 → 后端机械操作在 prompt 开头加
  「不要生成背景音乐」行；段间
  ThreadPoolExecutor 并发（每段目录独立，CodexRunner 自带信号量兜底）；meta.voice_lines
  按 start_s 归段（[start_s, end_s) 口径，恰在边界归后段），
  每段写 work/segments/N/work/voice_lines.json；任一段失败 → 整体 failed（error 指明段号）；
  meta.segments 落各段产物，顶层 keyframes/prompt 保持空值。段 codex 的 cwd 即段目录
  （物理隔离，看不到段外内容）；段目录内嵌套 work/（帧/台词/产物落段 work/，SKILL.md
  的 work/ 路径逐字适用，段 prompt 与单段逐字相同）；scripts/ 逐段拷入段目录，
  scenes.json 不拷入（段 codex 不需要知道全片）。
scenes 检测失败或 scenes.json 非法（含拆段不变量违规）→ 回退单段模式（meta.scenes_note
留痕），不做拆段；越界台词不归段并计数落 meta.voice_lines_dropped（内部字段）；翻译模式
的目标语言由后端写进 prompt（codex 不从台词反推）。
codex 超时/非零退出时先校验已落盘产物，完整则收养，不完整才判失败。
codex 运行前把 skill 的 scripts/ 拷进 codex 工作目录（单段=会话目录、多段=段目录；裁剪工具按 scripts/crop_image.py 相对引用）。
流水线复用 skills/video-maker 的脚本，不重造。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from app import asr, frame_fit, h3, h3_project, image_optimization, long_generation, long_video, prepared_input, scenes as scene_planner, storage, vocal, voice
from app.codex_runner import CodexError, CodexOutputError, clean_stderr
from app.config import Settings
from app.retry import RetryPolicy, run_with_retry

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "video-maker"
SCRIPTS_DIR = SKILL_DIR / "scripts"
EXTRACT_SCRIPT = SCRIPTS_DIR / "extract_keyframes.py"
SCENES_SCRIPT = ROOT / "app" / "scenes.py"
SKILL_MD = SKILL_DIR / "SKILL.md"
PROMPT_FUSION_SKILL_MD = ROOT / "skills" / "video-prompt-fusion" / "SKILL.md"
PROMPT_FUSION_FROZEN_SKILL_FILENAME = "video_prompt_fusion_skill.md"

MAX_PROMPT_BYTES = 32 * 1024
SCENES_TIMEOUT_S = 300  # scenes.py 场景检测超时（长视频 PySceneDetect 较慢）
CUT_DURATION_TOLERANCE_S = 0.1  # 切段时长允许误差（秒）
SCENE_BOUNDARY_ROUNDING_TOLERANCE_S = 0.001  # scenes.json 仅保留毫秒
_FLOAT_COMPARISON_EPS_S = 1e-12
NO_BGM_LINE = "不要生成背景音乐"  # 多段模式由后端机械加进 prompt 首行（不依赖 codex 写）
_SEG_TAIL_EPS_S = 0.01  # 台词 start_s 超出末段终点 ≤0.01s（与 voice 校验容差同口径）→ 归末段
# YAMNet 量化步长为 1/256；0.2 相邻下档 51/256 是线下真实 sung 样本。
# 该阈值只决定空听写是否重试，不改变逐句 classify_segment 规则。
EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN = 51 / 256
EMPTY_TRANSCRIPT_WARNING = (
    "voice_lines.json empty after automatic retries despite vocal evidence; "
    "continuing without dialogue"
)
UNRECOGNIZED_TRANSCRIPT_WARNING = (
    "自动听写只返回无法辨识占位符；已丢弃占位符，未将其作为台词"
)
VOICE_TIMELINE_WARNING = (
    "voice lines normalized to video duration {duration_s:.3f}s: "
    "clipped {clipped}, dropped {dropped}"
)
H3_DEFAULT_RATIO = "9:16"
H3_DEFAULT_FIT_MODE = "none"
H3_ENGINE_WORKFLOW = "minimax_h3_lightx2v_v5_15s"


def _remove_local_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _replace_scripts(cdir: Path, workdir: Path) -> None:
    """Install trusted skill scripts without retaining any previous directory bytes."""
    root = cdir.resolve()
    destination_root = workdir.resolve()
    try:
        destination_root.relative_to(root)
    except ValueError:
        raise PipelineError("scripts destination escapes conversation") from None
    if not destination_root.is_dir():
        raise PipelineError("scripts destination is not a directory")
    stage_root = Path(
        tempfile.mkdtemp(prefix=".scripts-stage-", dir=destination_root)
    )
    staged = stage_root / "scripts"
    target = destination_root / "scripts"
    try:
        shutil.copytree(SCRIPTS_DIR, staged)
        _remove_local_path(target)
        staged.replace(target)
    finally:
        _remove_local_path(stage_root)


def _load_receipt_dialogue(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines = payload["dialogue"]["lines"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise PipelineError("prepared input recovery invalid") from None
    if not isinstance(lines, list):
        raise PipelineError("prepared input recovery invalid")
    return lines


def _recovery_path(cdir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise PipelineError("long video plan recovery invalid")
    path = (cdir / value).resolve()
    try:
        path.relative_to(cdir)
    except ValueError:
        raise PipelineError("long video plan recovery invalid") from None
    return path


def _recover_prepared_input(cdir: Path, meta: dict) -> dict:
    receipt = cdir / prepared_input.RECEIPT_FILENAME
    lines = _load_receipt_dialogue(receipt)
    try:
        frozen = prepared_input.load_prepared_input(
            cdir, receipt, expected_dialogue=lines
        )
    except prepared_input.PreparedInputError:
        raise PipelineError("prepared input recovery invalid") from None
    names = [artifact.path.name for artifact in frozen.keyframes]
    if any(Path(name).name != name for name in names):
        raise PipelineError("prepared input recovery invalid")
    duration = meta.get("duration_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or abs(float(duration) - frozen.duration_s) > _FLOAT_COMPARISON_EPS_S
    ):
        raise PipelineError("prepared input recovery invalid")
    effective_meta = dict(meta)
    if not isinstance(effective_meta.get("source_width"), int):
        try:
            source_probe = storage.probe_video(frozen.source.path)
        except storage.UploadError:
            raise PipelineError("prepared input recovery invalid") from None
        effective_meta.update(
            source_width=source_probe.width,
            source_height=source_probe.height,
        )
    try:
        profiles, _recommended_aspect, _recommended_resolution, _recommended_fit = _generation_defaults(
            [cdir / "work" / "keyframes" / name for name in names],
            effective_meta,
        )
    except (PipelineError, KeyError):
        raise PipelineError("prepared input recovery invalid") from None
    engine_h3 = frozen.engine_request.get("h3")
    if not isinstance(engine_h3, dict):
        raise PipelineError("prepared input recovery invalid")
    raw_resolution = engine_h3.get("resolution")
    if raw_resolution in h3.H3_RESOLUTIONS:
        resolution = raw_resolution
    elif raw_resolution == h3.H3_RESOLUTION:
        resolution = h3.H3_DEFAULT_RESOLUTION
    else:
        raise PipelineError("prepared input recovery invalid")
    aspect_ratio = frozen.ratio
    fit_mode = frozen.fit_mode
    return {
        "status": "done",
        "error": None,
        "keyframes": names,
        "prompt": frozen.prompt_text,
        "fit_required": profiles[aspect_ratio]["fit_required"],
        "fit_profiles": profiles,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "fit_mode": fit_mode,
        "prepared_input_receipt": prepared_input.RECEIPT_FILENAME,
    }


def _recover_long_plan(cdir: Path, meta: dict, settings: Settings) -> dict:
    receipt = cdir / long_video.PLAN_RECEIPT_FILENAME
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        raw_segments = payload["segments"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        raise PipelineError("long video plan recovery invalid") from None
    if not isinstance(raw_segments, list) or not raw_segments:
        raise PipelineError("long video plan recovery invalid")
    segments = []
    for position, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, dict) or raw.get("index") != position:
            raise PipelineError("long video plan recovery invalid")
        segwork = cdir / "work" / "segments" / str(position) / "work"
        try:
            dialogue = json.loads(
                (segwork / "voice_lines.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            dialogue = []
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise PipelineError("long video plan recovery invalid") from None
        if not isinstance(dialogue, list):
            raise PipelineError("long video plan recovery invalid")
        try:
            key_paths = [item["path"] for item in raw["keyframes"]]
            first_path = raw["anchors"][0]["path"]
            last_path = raw["anchors"][1]["path"]
            visual_path = _recovery_path(cdir, raw["visual_prompt"]["path"])
            final_path = _recovery_path(cdir, raw["final_prompt"]["path"])
            visual = visual_path.read_text(encoding="utf-8")
            final = final_path.read_text(encoding="utf-8")
        except (KeyError, IndexError, TypeError, OSError, UnicodeDecodeError):
            raise PipelineError("long video plan recovery invalid") from None
        segments.append({
            "index": position,
            "start_s": raw.get("start_s"),
            "end_s": raw.get("end_s"),
            "chain_id": raw.get("chain_id"),
            "join_mode": raw.get("join_mode"),
            "source": f"segments/{position}/source.mp4",
            "keyframes": [Path(path).name for path in key_paths],
            "keyframe_paths": [
                Path(path).relative_to("work").as_posix() for path in key_paths
            ],
            **(
                {"keyframe_sources": raw["keyframe_sources"]}
                if "keyframe_sources" in raw else {}
            ),
            "first_frame_path": Path(first_path).relative_to("work").as_posix(),
            "last_frame_path": Path(last_path).relative_to("work").as_posix(),
            "visual_prompt": visual,
            "prompt": final,
            "dialogue": dialogue,
            "lines": [line.get("text") for line in dialogue if isinstance(line, dict)],
        })
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    recovered_meta = {
        **meta,
        "segments": segments,
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
    }
    try:
        plan = long_generation.freeze_plan(
            cdir, recovered_meta, digest, "none", meta.get("dialogue_mode", "auto"),
            settings=settings,
        )
        anchors = [
            anchor
            for segment in plan.segments
            for anchor in (
                (
                    (segment.first_frame_data,)
                    if segment.join_mode == "hard_cut"
                    else ()
                )
                + (segment.last_frame_data,)
            )
        ]
        if "source_width" in meta or "source_height" in meta:
            profiles, aspect_ratio, resolution, fit_mode = (
                _generation_defaults_from_bytes(anchors, meta)
            )
        else:
            profiles, aspect_ratio, resolution, fit_mode = (
                _legacy_generation_defaults_from_bytes(anchors)
            )
    except (frame_fit.FrameFitError, long_generation.LongGenerationError, ValueError):
        raise PipelineError("long video plan recovery invalid") from None
    return {
        "status": "done",
        "error": None,
        "segments": segments,
        "fit_required": profiles[aspect_ratio]["fit_required"],
        "fit_profiles": profiles,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "fit_mode": fit_mode,
        "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
    }


def reconcile_stale_pipeline(settings: Settings, cid: str, owner: object) -> None:
    """Read-only validate a half-committed pipeline freeze and reach a terminal state."""
    meta = storage.load_pipeline_claim(settings.data_dir, cid, owner)
    if meta is None:
        return
    cdir = (settings.data_dir / cid).resolve()
    try:
        has_prepared = (cdir / prepared_input.RECEIPT_FILENAME).is_file()
        has_plan = (cdir / long_video.PLAN_RECEIPT_FILENAME).is_file()
        if has_prepared == has_plan:
            raise PipelineError("pipeline recovery has ambiguous frozen input")
        changes = (
            _recover_prepared_input(cdir, meta)
            if has_prepared
            else _recover_long_plan(cdir, meta, settings)
        )
    except Exception:
        changes = {"status": "failed", "error": "input_recovery_required"}
    storage.finish_input_claim(settings.data_dir, cid, owner, **changes)
H3_BOUNDARY_WORKFLOW = "minimax_h3_lightx2v"

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """流水线单步失败（HTTP 层不感知，只进 meta.error）。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _EmptyTranscript(PipelineError):
    def __init__(self) -> None:
        super().__init__("voice transcript empty despite vocal evidence", retryable=True)


def _codex_output_error(stage: str) -> PipelineError:
    """把 Codex 成功退出后的产物问题归到正确阶段，不暴露内部文件校验细节。"""
    details = {
        "voice": "required voice_lines artifact is missing or invalid",
        "visual": "required keyframes/prompt artifacts are missing or invalid",
    }
    detail = details[stage]
    return PipelineError(f"codex {stage} output invalid: {detail}", retryable=True)


def _retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy(settings.retry_count, settings.retry_interval_s)


def _retryable_operation_error(exc: Exception) -> bool:
    return bool(
        isinstance(exc, (CodexError, PipelineError))
        and getattr(exc, "retryable", False)
    )


def _retry_logger(step: str, policy: RetryPolicy):
    def report(retry_number: int, exc: Exception) -> None:
        log.warning(
            "%s failed; retry %d/%d in %.1fs (%s)",
            step,
            retry_number,
            policy.retries,
            policy.interval_s,
            type(exc).__name__,
        )

    return report


def _run_cmd(argv: list[str], *, timeout: int, step: str, cwd: Path | None = None) -> None:
    """argv 列表子进程；超时/找不到可执行/非零退出 → PipelineError（stderr 已清洗）。"""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise PipelineError(f"{step} timed out after {timeout}s") from None
    except FileNotFoundError as e:
        raise PipelineError(f"{step} executable not found: {e.filename}") from None
    if proc.returncode != 0:
        raise PipelineError(f"{step} exit {proc.returncode}: {clean_stderr(proc.stderr)}")


def validate_work_dir(
    work: Path, *, expected_keyframe_count: int | None = None
) -> tuple[list[str], str]:
    """产物白名单校验；返回 (关键帧文件名列表, prompt 文本)。任一不过 → PipelineError。"""
    frames = (
        sorted(p.name for p in (work / "keyframes").glob("*.png"))
        if (work / "keyframes").is_dir()
        else []
    )
    if expected_keyframe_count is not None and len(frames) != expected_keyframe_count:
        raise PipelineError(
            f"keyframe count {len(frames)} does not equal {expected_keyframe_count}"
        )
    if not 1 <= len(frames) <= 9:
        raise PipelineError(f"keyframe count {len(frames)} not in 1..9")
    for name in frames:
        if cv2.imread(str(work / "keyframes" / name)) is None:
            raise PipelineError(f"keyframe undecodable: {name}")

    prompt_path = work / "prompt.txt"
    if not prompt_path.is_file():
        raise PipelineError("prompt.txt missing")
    raw = prompt_path.read_bytes()
    if not raw.strip():
        raise PipelineError("prompt.txt empty")
    if len(raw) > MAX_PROMPT_BYTES:
        raise PipelineError(f"prompt.txt exceeds {MAX_PROMPT_BYTES} bytes")
    prompt = raw.decode("utf-8", errors="replace")
    return frames, prompt


def _materialize_backend_keyframes(
    source: Path,
    work: Path,
    selection: list[dict],
) -> tuple[list[str], dict, tuple[bytes, ...]]:
    """Decode and freeze the backend planner's exact ordered nine frames."""
    if (
        not isinstance(selection, list)
        or len(selection) != scene_planner.KEYFRAMES_PER_SEGMENT
    ):
        raise PipelineError("backend keyframe selection must contain exactly nine frames")
    indices: list[int] = []
    for order, item in enumerate(selection, 1):
        decode_index = item.get("decode_frame_index") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("order") != order
            or isinstance(decode_index, bool)
            or not isinstance(decode_index, int)
            or decode_index < 0
        ):
            raise PipelineError("backend keyframe selection is invalid")
        indices.append(decode_index)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise PipelineError("source video cannot be decoded for backend keyframes")
    wanted = set(indices)
    encoded: dict[int, bytes] = {}
    decode_index = 0
    try:
        while wanted - set(encoded):
            ok, frame = capture.read()
            if not ok:
                break
            if decode_index in wanted:
                success, png = cv2.imencode(".png", frame)
                if not success:
                    raise PipelineError("backend keyframe PNG encoding failed")
                encoded[decode_index] = png.tobytes()
            decode_index += 1
    finally:
        capture.release()
    missing = sorted(wanted - set(encoded))
    if missing:
        raise PipelineError(
            f"source decode inventory changed before keyframe freeze: {missing}"
        )

    frozen = tuple(encoded[index] for index in indices)
    names = [f"{order:02d}.png" for order in range(1, 10)]
    stage_root = Path(tempfile.mkdtemp(prefix=".keyframes-stage-", dir=work))
    staged = stage_root / "keyframes"
    target = work / "keyframes"
    staged.mkdir()
    try:
        for name, data in zip(names, frozen):
            (staged / name).write_bytes(data)
        _remove_local_path(target)
        staged.replace(target)
    finally:
        _remove_local_path(stage_root)

    receipt_items = []
    for name, item, data in zip(names, selection, frozen):
        receipt_items.append({
            **item,
            "path": f"keyframes/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    receipt = {
        "schema": "duet.backend-keyframe-sampling",
        "version": 1,
        "selection_method": "scene-anchor-capacity-hamilton-v1",
        "keyframes": receipt_items,
    }
    staged_receipt = work / ".keyframe_sampling.tmp"
    staged_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staged_receipt, work / "keyframe_sampling.json")
    return names, receipt, frozen


def _restore_backend_keyframes(work: Path, frozen: tuple[bytes, ...]) -> None:
    if len(frozen) != scene_planner.KEYFRAMES_PER_SEGMENT or any(
        not isinstance(data, bytes) or not data for data in frozen
    ):
        raise PipelineError("backend frozen keyframes are invalid")
    target = work / "keyframes"
    _remove_local_path(target)
    target.mkdir()
    for order, data in enumerate(frozen, 1):
        (target / f"{order:02d}.png").write_bytes(data)


def _hard_rules(cdir: Path) -> str:
    """codex prompt 硬性禁令四条（口播/选帧各步共用）。"""
    return f"""硬性禁令：
- 运行 Python 脚本一律用 {sys.executable}（系统 python3 缺 cv2）。
- 只在 {cdir} 内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。"""


def _language_note(target_language: str) -> str:
    """翻译模式：目标语言由后端注入 prompt（codex 不从台词反推语言）。"""
    return f"提示词与台词使用目标语言：{target_language}。" if target_language else ""


def _codex_prompt(
    cdir: Path, target_language: str = "", *, visual_only: bool = False
) -> str:
    parts = [
        f"按技能文档执行：{SKILL_MD}（该文档只读，禁止修改；「只读」指文档本身，不是执行模式）。输入在 work/，产物（keyframes/ 与 prompt.txt）必须按文档写入 work/。"
    ]
    if target_language:
        parts.append(_language_note(target_language))
    if visual_only:
        parts.append(
            "本次只生成视觉叙事 prompt：不要读取、生成、推断、改写或编排 voice_lines；"
            "画面文字、OCR、字幕和备注只能作为可见视觉元素，禁止写成角色发声。"
            "最终台词由后端从结构化来源机械合成。"
        )
    parts.append(_hard_rules(cdir))
    return "\n\n".join(parts) + "\n"


def _run_visual_codex(
    runner,
    cdir: Path,
    prompt: str,
    work: Path,
    *,
    isolate_dialogue: bool,
) -> None:
    """新 H3 契约下让视觉 Codex 看不到 voice_lines，并丢弃它越权创建的同名文件。"""
    voice_path = work / "voice_lines.json"
    frozen_voice = voice_path.read_bytes() if isolate_dialogue and voice_path.is_file() else None
    if isolate_dialogue:
        voice_path.unlink(missing_ok=True)
    try:
        runner.run(cdir, prompt)
    finally:
        if isolate_dialogue:
            voice_path.unlink(missing_ok=True)
            if frozen_voice is not None:
                voice_path.write_bytes(frozen_voice)


def _clear_visual_outputs(
    cdir: Path,
    work: Path,
    *,
    frozen_keyframes: tuple[bytes, ...] | None = None,
) -> None:
    """Start every visual attempt without outputs from an earlier attempt."""
    keyframes = work / "keyframes"
    if keyframes.exists():
        shutil.rmtree(keyframes)
    if frozen_keyframes is not None:
        _restore_backend_keyframes(work, frozen_keyframes)
    (work / "prompt.txt").unlink(missing_ok=True)
    (cdir / "codex_last_message.txt").unlink(missing_ok=True)


def _run_visual_attempt(
    runner,
    cdir: Path,
    prompt: str,
    work: Path,
    *,
    isolate_dialogue: bool,
    frozen_keyframes: tuple[bytes, ...] | None = None,
) -> tuple[list[str], str]:
    """Run once, adopting complete outputs even when Codex exits abnormally."""
    _clear_visual_outputs(
        cdir, work, frozen_keyframes=frozen_keyframes
    )
    run_error: CodexError | None = None
    try:
        _run_visual_codex(
            runner,
            cdir,
            prompt,
            work,
            isolate_dialogue=isolate_dialogue,
        )
    except CodexError as exc:
        run_error = exc
    finally:
        if frozen_keyframes is not None:
            # The model can analyze the server-owned frames but has no authority
            # over their bytes, count, numbering or order.
            _restore_backend_keyframes(work, frozen_keyframes)
    try:
        return validate_work_dir(
            work,
            expected_keyframe_count=(
                scene_planner.KEYFRAMES_PER_SEGMENT
                if frozen_keyframes is not None
                else None
            ),
        )
    except PipelineError:
        if run_error is not None:
            raise run_error from None
        raise _codex_output_error("visual") from None


def _run_visual_with_retry(
    settings: Settings,
    runner,
    cdir: Path,
    prompt: str,
    work: Path,
    *,
    isolate_dialogue: bool,
    step: str,
    frozen_keyframes: tuple[bytes, ...] | None = None,
) -> tuple[list[str], str]:
    policy = _retry_policy(settings)
    return run_with_retry(
        lambda: _run_visual_attempt(
            runner,
            cdir,
            prompt,
            work,
            isolate_dialogue=isolate_dialogue,
            frozen_keyframes=frozen_keyframes,
        ),
        policy=policy,
        is_retryable=_retryable_operation_error,
        on_retry=_retry_logger(step, policy),
    )


def _write_image_optimization_prompt(work: Path, prompt: str) -> None:
    target = work / "image_optimization_prompt.txt"
    staged = work / ".image_optimization_prompt.tmp"
    try:
        with staged.open("w", encoding="utf-8") as stream:
            stream.write(prompt + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, target)
        directory = os.open(work, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staged.unlink(missing_ok=True)


def _frame_inventory(
    frame_paths: dict[int, list[Path]],
    *,
    segment_lineage: dict[int, dict] | None = None,
) -> list[dict]:
    inventory = []
    previous: dict | None = None
    for segment_index in sorted(frame_paths):
        paths = frame_paths[segment_index]
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or not isinstance(paths, list)
            or not paths
        ):
            raise PipelineError("image optimization frame inventory is invalid")
        lineage = None if segment_lineage is None else segment_lineage.get(segment_index)
        if segment_lineage is not None and (
            not isinstance(lineage, dict)
            or set(lineage) != {"chain_id", "join_mode"}
            or not isinstance(lineage["chain_id"], str)
            or not lineage["chain_id"]
            or lineage["join_mode"] not in {"hard_cut", "continue"}
        ):
            raise PipelineError("image optimization frame inventory is invalid")
        for frame_index, path in enumerate(paths, 1):
            if not isinstance(path, Path) or path.name != f"{frame_index:02d}.png":
                raise PipelineError("image optimization frame inventory is invalid")
            try:
                source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise PipelineError("image optimization frame inventory is invalid") from exc
            item = {
                "segment_index": segment_index,
                "frame_index": frame_index,
                "frame_name": path.name,
                "source_sha256": source_sha256,
            }
            if lineage is not None:
                transition = (
                    "start"
                    if previous is None
                    else (
                        "hard_cut"
                        if frame_index == 1 and lineage["join_mode"] == "hard_cut"
                        # Chain lineage proves continuity but contains no measured
                        # camera transform.  It can therefore establish only the
                        # conservative same-camera relation; motion needs a
                        # versioned, source-bound measurement before it may be
                        # frozen as evidence.
                        else "same_camera"
                    )
                )
                evidence = {
                    "chain_id": lineage["chain_id"],
                    "join_mode": lineage["join_mode"],
                    "previous": previous,
                    "current": {
                        "segment_index": segment_index,
                        "frame_index": frame_index,
                        "source_sha256": source_sha256,
                    },
                    "transition": transition,
                }
                item.update(
                    source_transition_from_previous=transition,
                    source_transition_evidence_sha256=hashlib.sha256(
                        json.dumps(
                            evidence, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                )
                previous = evidence["current"]
            inventory.append(item)
    if segment_lineage is not None and set(segment_lineage) != set(frame_paths):
        raise PipelineError("image optimization frame inventory is invalid")
    return inventory


def _freeze_image_optimization(
    settings: Settings,
    meta: dict,
    continuity: dict,
    prompts: dict,
    frame_paths: dict[int, list[Path]],
    *,
    require_dual_target: bool,
    segment_lineage: dict[int, dict] | None = None,
) -> tuple[dict, dict]:
    """Freeze either the legacy segment prompt or V3 source-frame prompts."""
    try:
        if continuity.get("version") in {3, 4}:
            inventory = _frame_inventory(
                frame_paths,
                segment_lineage=segment_lineage if continuity.get("version") == 4 else None,
            )
            frame_counts = {
                index: len(paths) for index, paths in frame_paths.items()
            }
            frozen_continuity = image_optimization.freeze_continuity(
                continuity, frame_counts=frame_counts
            )
            execution = image_optimization.freeze_execution_inputs(
                continuity,
                revision=1,
                profile={"id": "image-postprocess", "revision": 2},
                model=settings.seedream_model,
                frame_inventory=inventory,
            )
            frozen_prompts = image_optimization.freeze_frame_prompts(
                settings,
                execution,
                prompts,
                plan=continuity if continuity.get("version") == 4 else None,
            )
        else:
            frozen_continuity = image_optimization.freeze_continuity(continuity)
            frozen_prompts = image_optimization.freeze_prompts(
                settings, meta, prompts
            )
        candidate = {**meta, **frozen_continuity, **frozen_prompts}
        if (
            require_dual_target
            and image_optimization.dual_target_plan_receipt(candidate) is None
        ):
            raise image_optimization.ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        if image_optimization.receipt(candidate) is None:
            raise image_optimization.ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        return frozen_continuity, frozen_prompts
    except (
        ValueError,
        image_optimization.ImageOptimizationIneligibleError,
        image_optimization.ImageOptimizationOutputError,
    ) as exc:
        raise PipelineError(str(exc)) from None


def _generate_image_optimization_project(
    settings: Settings,
    runner,
    segments: list[dict],
    *,
    session_dir: Path,
    step: str,
) -> tuple[dict | None, dict]:
    policy = _retry_policy(settings)

    def attempt() -> tuple[dict | None, dict]:
        try:
            return image_optimization.generate_project_prompts(
                runner,
                segments,
                settings.seedream_edit_mode,
                session_dir=session_dir,
                expected_version=4,
            )
        except image_optimization.ImageOptimizationOutputError as exc:
            raise PipelineError(str(exc), retryable=True) from None
        except ValueError:
            raise
        except Exception:
            raise PipelineError(
                "image optimization planner unavailable", retryable=True,
            ) from None

    # There is exactly one compiler.  Operational retries repeat the same
    # frozen request; they never switch to a second prompt-generation scheme.
    return run_with_retry(
        attempt,
        policy=policy,
        is_retryable=_retryable_operation_error,
        on_retry=_retry_logger(step, policy),
    )


def _generate_segmented_image_prompts(
    settings: Settings,
    runner,
    segments: list[dict],
    seg_metas: list[dict],
    work: Path,
    *,
    session_dir: Path,
) -> tuple[dict, dict]:
    frame_paths = {}
    for segment, meta in zip(segments, seg_metas):
        directory = work / "segments" / str(segment["index"]) / "work" / "keyframes"
        names = meta.get("keyframes") if isinstance(meta, dict) else None
        paths = (
            [directory / name for name in names]
            if isinstance(names, list) and all(isinstance(name, str) for name in names)
            else sorted(path for path in directory.glob("*.png") if path.is_file())
        )
        frame_paths[segment["index"]] = paths
    transitions_by_segment = {}
    if all(frame_paths.values()):
        transition_inventory = _frame_inventory(
            frame_paths,
            segment_lineage={
                segment["index"]: {
                    "chain_id": segment["chain_id"], "join_mode": segment["join_mode"],
                }
                for segment in segments
            },
        )
        transitions_by_segment = {
            index: [item for item in transition_inventory if item["segment_index"] == index]
            for index in frame_paths
        }
    specs = [{
        "index": segment["index"],
        "chain_id": segment["chain_id"],
        "join_mode": segment["join_mode"],
        "keyframes_dir": (
            work / "segments" / str(segment["index"]) / "work" / "keyframes"
        ),
        **({"transition_skeleton": transitions_by_segment[segment["index"]]}
           if segment["index"] in transitions_by_segment else {}),
    } for segment in segments]
    continuity, prompts = _generate_image_optimization_project(
        settings,
        runner,
        specs,
        session_dir=session_dir,
        step="project image postprocess codex",
    )
    if continuity is None:
        raise PipelineError("image continuity was not generated")
    if continuity.get("version") not in {3, 4}:
        for segment in seg_metas:
            index = segment["index"]
            _write_image_optimization_prompt(
                work / "segments" / str(index) / "work", prompts[index]
            )
    return continuity, prompts


def _voice_prompt(_cdir: Path, voice_mode: str, target_language: str, duration_s: float) -> str:
    """口播步 codex prompt：只使用隔离区内的音频输入，不暴露原会话路径。"""
    if voice_mode == "keep":
        rule = "原文保持：只修正错别字与标点，不改写措辞。"
    elif voice_mode == "rewrite":
        rule = (
            "洗稿：把台词改写得更自然；句数不变、句序不变、每句时间边界不变；"
            "产品和工具使用准确的通用称呼，不保留未经确认的夸张别名。"
        )
    else:  # translate
        rule = f"翻译成{target_language}：句对句对齐，每句时间边界不变。"
    return f"""听写并处理视频台词。输入：work/voice.mp3（源视频音轨，16kHz 单声道）与 work/manifest.json（源视频元信息，供参考）。音频时长约 {duration_s:.3f} 秒。

任务：
- 自动识别实际语言并听写音频中的人声台词，按句切分；必须支持中文、英文、西班牙语、马来语及其他源音频语言；
- 每句标出起止时间（秒，从音频开头起算）；
- {rule}
- `[无法辨识]`、`[无法识别]`、`[inaudible]`、`[unintelligible]` 等只是失败占位符，禁止作为 text 输出；无法得到任何可靠原文时输出空数组 `[]`；
- 输出 work/voice_lines.json（UTF-8）：JSON 数组 [{{"text": "...", "start_s": 0.5, "end_s": 2.1}}]，0 ≤ start_s < end_s ≤ 音频时长，按 start_s 升序，覆盖人声区间；不写其他文件。

硬性禁令：
- 只在当前音频专用工作区内创建/修改文件。
- 禁止联网（沙箱已断网，联网必然失败）。
- 禁止打印、读取或记录任何环境变量。
"""


def _run_voice_attempt(
    runner,
    work: Path,
    prompt: str,
    duration_s: float,
) -> list[dict]:
    """在音频隔离区听写；成功退出后的缺失/非法产物稳定归因为 voice 输出错误。"""
    try:
        return runner.run_voice(
            work,
            prompt,
            duration_s=duration_s,
            validate_output=lambda raw: voice.validate_voice_lines(raw, duration_s),
        )
    except CodexError:
        raise
    except (CodexOutputError, PipelineError, OSError):
        raise _codex_output_error("voice") from None


def _transcribe_voice_attempt(
    settings: Settings,
    runner,
    work: Path,
    prompt: str,
    duration_s: float,
    voice_mode: str,
) -> list[dict]:
    """Use deterministic local multilingual ASR for keep mode when configured."""
    if (settings.asr_cli is None) != (settings.asr_model is None):
        raise PipelineError("local ASR configuration incomplete")
    if voice_mode == "keep" and settings.asr_cli is not None:
        try:
            lines = asr.transcribe(
                work / "voice.mp3",
                cli=settings.asr_cli,
                model=settings.asr_model,
                duration_s=duration_s,
                timeout_s=settings.asr_timeout_s,
                threads=settings.asr_threads,
            )
            raw = json.dumps(lines, ensure_ascii=False).encode("utf-8")
            return voice.validate_voice_lines(raw, duration_s)
        except asr.ASRError as exc:
            raise PipelineError(str(exc), retryable=exc.retryable) from None
    return _run_voice_attempt(runner, work, prompt, duration_s)


def _vocal_filter_enabled() -> bool:
    """唯一环境开关解析点；未知值按启用处理，避免误拼写静默放宽过滤。"""
    return os.environ.get("VOCAL_FILTER", "on").strip().lower() not in {
        "0", "off", "false"
    }


def _has_retryable_vocal_evidence(analysis: vocal.VocalAnalysis) -> bool:
    """空听写重试边界：覆盖真实 51/256 sung，排除 0.059 纯 BGM。"""
    return any(
        window.spoken >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN
        or window.sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN
        for window in analysis.windows
    )


def _transcribe_voice_with_retry(
    settings: Settings,
    runner,
    work: Path,
    prompt: str,
    duration_s: float,
    voice_mode: str,
    *,
    has_vocal: bool,
) -> tuple[list[dict], bool]:
    """Retry transient execution/output failures and suspicious empty transcripts."""
    policy = _retry_policy(settings)
    unrecognized = False

    def attempt() -> list[dict]:
        nonlocal unrecognized
        lines = _transcribe_voice_attempt(
            settings, runner, work, prompt, duration_s, voice_mode
        )
        unrecognized = unrecognized or any(
            voice.is_unrecognized_text(line["text"]) for line in lines
        )
        lines = [
            line for line in lines if not voice.is_unrecognized_text(line["text"])
        ]
        if not lines and has_vocal:
            raise _EmptyTranscript()
        return lines

    try:
        lines = run_with_retry(
            attempt,
            policy=policy,
            is_retryable=_retryable_operation_error,
            on_retry=_retry_logger("voice transcription", policy),
        )
    except _EmptyTranscript:
        lines = []
    return lines, unrecognized


def _classify_voice_line(
    line: dict,
    analysis: vocal.VocalAnalysis,
    *,
    only_line: bool,
) -> str | None:
    """逐句分类；唯一台词时间戳漂移时允许用明确的全轨人声证据兜底。"""
    classification = vocal.classify_segment(
        int(line["start_s"] * 1000), int(line["end_s"] * 1000), analysis.windows
    )
    if classification is not None or not only_line or not analysis.windows:
        return classification
    max_spoken = max(window.spoken for window in analysis.windows)
    max_sung = max(window.sung for window in analysis.windows)
    if max_sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN and max_sung > max_spoken:
        return "sung"
    if max_spoken >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN:
        return "spoken"
    if max_sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN:
        return "sung"
    return None


def _normalize_voice_timeline(
    decisions: list[dict], duration_s: float
) -> tuple[list[dict], list[dict], list[str]]:
    """把已完成 YAMNet 分类的 ASR 行裁到视频时间轴，再做一次完整白名单校验。"""
    normalized_decisions = []
    effective_lines = []
    clipped = 0
    dropped = 0
    for decision in decisions:
        current = dict(decision)
        start_s = float(current["start_s"])
        end_s = float(current["end_s"])
        if start_s >= duration_s:
            current["kept"] = False
            current["drop_reason"] = "starts_at_or_after_video_duration"
            dropped += 1
        elif end_s > duration_s:
            current["asr_start_s"] = start_s
            current["asr_end_s"] = end_s
            current["end_s"] = duration_s
            current["time_adjustment"] = "clipped_to_video_duration"
            clipped += 1
        normalized_decisions.append(current)
        if current["kept"]:
            effective_lines.append(
                {
                    "text": current["text"],
                    "start_s": current["start_s"],
                    "end_s": current["end_s"],
                }
            )

    normalized_lines = voice.validate_voice_lines(
        json.dumps(effective_lines, ensure_ascii=False).encode("utf-8"), duration_s
    )
    warnings = []
    if clipped or dropped:
        warnings.append(
            VOICE_TIMELINE_WARNING.format(
                duration_s=duration_s,
                clipped=clipped,
                dropped=dropped,
            )
        )
    return normalized_lines, normalized_decisions, warnings


def _voice_step(
    settings: Settings, cid: str, cdir: Path, work: Path, runner,
    voice_mode: str, target_language: str, *, allow_no_audio: bool = False,
) -> list[dict]:
    """口播步（抽帧后）：抽音轨 → 声学预判 → codex 听写 → 白名单校验 → voice_lines 落 meta。

    台词时间戳在音频时间轴上（codex 听 voice.mp3），校验基准与提示词时长用音频实际时长
    （音频流可比容器长几十 ms，常态）；YAMNet 分类后再把最终行裁到 manifest 视频时间轴。
    空台词数组且音轨有人声证据 → 按统一策略重试，再空则写 warning 后以无台词继续。新 auto
    契约下无音轨同样是合法空台词；旧 voice_mode 可保留严格失败行为。返回白名单净化后的台词列表。
    """
    target_language = (target_language or "").strip()  # 纯空白串视为缺失，不生成「翻译成   」prompt
    if voice_mode not in ("keep", "rewrite", "translate"):
        raise PipelineError(f"unknown voice_mode: {voice_mode}")
    if voice_mode == "translate" and not target_language:
        raise PipelineError("voice_mode=translate requires target_language")
    filter_enabled = _vocal_filter_enabled()
    audio = voice.extract_audio(cdir)
    if audio is None:
        if not allow_no_audio:
            raise PipelineError("source video has no audio track")
        (work / "voice.mp3").unlink(missing_ok=True)
        (work / "voice_lines.json").write_text("[]\n", encoding="utf-8")
        storage.update_meta(
            settings.data_dir,
            cid,
            voice_lines=[],
            has_bgm=False,
            vocal_filter_enabled=filter_enabled,
            voice_line_provenance=[],
            voice_warnings=[],
            voice_has_retryable_vocal_evidence=False,
            voice_lines_vocal_dropped=0,
            voice_lines_credit_dropped=0,
            voice_text_normalizations=[],
        )
        return []
    try:
        manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PipelineError("manifest.json missing or invalid") from None
    try:
        duration_s = float(manifest["duration_seconds"])
    except (KeyError, TypeError, ValueError):
        raise PipelineError("manifest.json missing or invalid") from None
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise PipelineError(f"manifest.json invalid duration: {duration_s}")
    # 台词校验基准 = 音频实际时长；probe 失败回退容器时长（旧行为）
    audio_duration_s = voice.probe_audio_duration(audio) or duration_s
    if audio.is_symlink():
        raise PipelineError("vocal analysis audio is invalid")
    try:
        audio_data = audio.read_bytes()
        analysis_audio_path = (
            audio.resolve().relative_to(cdir.resolve()).as_posix()
        )
    except (OSError, ValueError):
        raise PipelineError("vocal analysis audio is invalid") from None
    if not audio_data or analysis_audio_path != "work/voice.mp3":
        raise PipelineError("vocal analysis audio is invalid")
    analysis_audio_sha256 = hashlib.sha256(audio_data).hexdigest()
    # 声学分析提前到听写前：空台词数组时区分「音轨无口播（合法无台词）」与「codex 摆烂（重试兜底）」。
    # 分析前后重读同一路径，避免 receipt 绑定到分析期间被替换的另一份 bytes。
    try:
        analysis = vocal.analyze(audio)
    except Exception as e:
        raise PipelineError(f"vocal classification unavailable: {e}") from None
    try:
        analyzed_data = audio.read_bytes()
        analyzed_path = audio.resolve().relative_to(cdir.resolve()).as_posix()
    except (OSError, ValueError):
        raise PipelineError("vocal analysis audio is invalid") from None
    if (
        audio.is_symlink()
        or analyzed_path != analysis_audio_path
        or analyzed_data != audio_data
    ):
        raise PipelineError("vocal analysis audio drifted")
    has_vocal = _has_retryable_vocal_evidence(analysis)
    prompt = _voice_prompt(cdir, voice_mode, target_language, audio_duration_s)
    lines, unrecognized = _transcribe_voice_with_retry(
        settings,
        runner,
        work,
        prompt,
        audio_duration_s,
        voice_mode,
        has_vocal=has_vocal,
    )
    warnings = [EMPTY_TRANSCRIPT_WARNING] if not lines and has_vocal else []
    if unrecognized:
        warnings.append(UNRECOGNIZED_TRANSCRIPT_WARNING)
    # VOCAL_FILTER=off 只旁路 keep/drop，不旁路分类：receipt 必须解释每句为何被保留。
    decisions = []
    for line in lines:
        classification = _classify_voice_line(
            line,
            analysis,
            only_line=len(lines) == 1,
        )
        subtitle_credit = voice.is_subtitle_credit_text(line["text"])
        kept = (
            (not filter_enabled or classification in ("spoken", "sung"))
            and not subtitle_credit
        )
        decision = {
            **line,
            "classification": classification,
            "provenance": "asr",
            "kept": kept,
        }
        if subtitle_credit:
            decision["drop_reason"] = "subtitle_credit"
        decisions.append(decision)
    filtered_lines, decisions, timeline_warnings = _normalize_voice_timeline(
        decisions, duration_s
    )
    try:
        classification_evidence = long_generation.classification_evidence_sha256(
            audio_path=analysis_audio_path,
            audio_sha256=analysis_audio_sha256,
            has_bgm=bool(analysis.has_bgm),
            decisions=decisions,
        )
    except long_generation.LongGenerationError:
        raise PipelineError("vocal classification evidence is invalid") from None
    decisions = [{
        **decision,
        "analysis_audio_path": analysis_audio_path,
        "analysis_audio_sha256": analysis_audio_sha256,
        "analysis_has_bgm": bool(analysis.has_bgm),
        "classification_evidence_sha256": classification_evidence,
    } for decision in decisions]
    warnings.extend(timeline_warnings)
    vocal_dropped = sum(
        decision.get("classification") not in ("spoken", "sung")
        for decision in decisions
        if filter_enabled
    )
    credit_dropped = sum(
        decision.get("drop_reason") == "subtitle_credit" for decision in decisions
    )
    # 后续视觉步骤看到的 voice_lines 也只能是最终有效集，不能重新收养已过滤行。
    (work / "voice_lines.json").write_text(
        json.dumps(filtered_lines, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changes = {
        "voice_lines": filtered_lines,
        "has_bgm": bool(analysis.has_bgm),
        "vocal_filter_enabled": filter_enabled,
        "voice_line_provenance": decisions,
        "voice_warnings": warnings,
        "voice_has_retryable_vocal_evidence": has_vocal,
        "voice_lines_vocal_dropped": vocal_dropped,
        "voice_lines_credit_dropped": credit_dropped,
        "voice_text_normalizations": [],
    }
    storage.update_meta(settings.data_dir, cid, **changes)
    return filtered_lines


def _load_scenes(work: Path) -> list[dict]:
    """读并校验 work/scenes.json；返回 segments（空 = 单段模式）。缺失/非法 → PipelineError。

    结构不变量（与 scenes.py 同口径）：每段 provider 请求不超过 14s、相邻无缝（1e-6 容差，
    隐含单调有序）、首段 0 起且覆盖 [0, duration_s]。
    """
    try:
        data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PipelineError("scenes.json missing or invalid") from None
    segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(segments, list):
        raise PipelineError("scenes.json missing segments")
    out: list[dict] = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("index"), int):
            raise PipelineError(f"scenes.json segments[{i}] must be an object with int index")
        start_s, end_s = seg.get("start_s"), seg.get("end_s")
        for key, val in (("start_s", start_s), ("end_s", end_s)):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise PipelineError(f"scenes.json segments[{i}].{key} must be a number")
        start_s, end_s = float(start_s), float(end_s)
        if not (math.isfinite(start_s) and math.isfinite(end_s) and start_s < end_s):
            raise PipelineError(f"scenes.json segments[{i}] invalid bounds: {start_s}..{end_s}")
        out.append({"index": seg["index"], "start_s": start_s, "end_s": end_s})
    if not out:
        return out  # 空 segments = 单段模式（合法），无需时长与覆盖校验
    duration = data.get("duration_s")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise PipelineError("scenes.json missing valid duration_s")
    prev_end = 0.0
    for i, seg in enumerate(out):
        try:
            frozen_duration = long_video.segment_duration_s(
                seg["start_s"], seg["end_s"]
            )
        except long_video.LongVideoError:
            raise PipelineError(
                f"scenes.json segments[{i}] provider duration not in 1..14s"
            ) from None
        if (
            frozen_duration < long_video.SEGMENT_MIN_S
            or long_video.provider_duration_s(seg["start_s"], seg["end_s"])
            > long_video.SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            raise PipelineError(
                f"scenes.json segments[{i}] provider duration not in 1..14s"
            )
        if abs(seg["start_s"] - prev_end) > 1e-6:
            raise PipelineError(f"scenes.json segments[{i}] not contiguous with previous")
        prev_end = seg["end_s"]
    if abs(prev_end - float(duration)) > 1e-6:
        raise PipelineError("scenes.json segments do not cover [0, duration]")
    return out


def _detect_segments(settings: Settings, cid: str, source: Path, work: Path) -> list[dict]:
    """跑 app/scenes.py 检测场景并读拆段建议。

    检测失败或 scenes.json 非法（含拆段结构不变量违规）→ 回退空列表 = 单段模式，
    不判失败，meta.scenes_note 留痕；segments 空（≤15s）是合法单段结果，不留痕。
    """
    try:
        _run_cmd(
            [sys.executable, str(SCENES_SCRIPT), str(source), "--work-dir", str(work)],
            timeout=SCENES_TIMEOUT_S,
            step="scenes",
        )
    except PipelineError as e:
        print(f"scenes detection failed ({e}); falling back to single-segment mode")
        storage.update_meta(
            settings.data_dir, cid,
            scenes_note="scenes detection failed, single-segment fallback",
        )
        return []
    try:
        return _load_scenes(work)
    except PipelineError as e:
        print(f"scenes.json invalid ({e}); falling back to single-segment mode")
        storage.update_meta(
            settings.data_dir, cid,
            scenes_note="scenes.json invalid, single-segment fallback",
        )
        return []


def _scene_bounds_for_long_plan(work: Path, duration_s: float) -> list[dict]:
    """Load detected scene bounds; absence means one scene, never guessed hard cuts."""
    try:
        data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [{"start_s": 0.0, "end_s": duration_s}]
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list) or not scenes:
        return [{"start_s": 0.0, "end_s": duration_s}]
    bounds: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            return scenes
        try:
            bounds.append(
                {"start_s": float(scene["start_s"]), "end_s": float(scene["end_s"])}
            )
        except (KeyError, TypeError, ValueError):
            return scenes
    # scenes.py serializes boundaries to milliseconds, while upload ffprobe
    # preserves finer precision.  The source duration remains authoritative;
    # only the two outer boundaries may absorb that bounded serialization loss.
    tolerance = SCENE_BOUNDARY_ROUNDING_TOLERANCE_S + _FLOAT_COMPARISON_EPS_S
    if abs(bounds[0]["start_s"]) <= tolerance:
        bounds[0]["start_s"] = 0.0
    if abs(bounds[-1]["end_s"] - duration_s) <= tolerance:
        bounds[-1]["end_s"] = duration_s
    return bounds


def _source_scenes_for_timeline(work: Path, duration_s: float) -> list[dict]:
    """Return the detector-owned scene ids and exact normalized boundaries."""
    try:
        raw = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = None
    exact_scenes = raw.get("effective_scenes") if isinstance(raw, dict) else None
    if isinstance(exact_scenes, list) and exact_scenes:
        result = []
        for position, scene in enumerate(exact_scenes, 1):
            if (
                not isinstance(scene, dict)
                or scene.get("index") != position
                or not isinstance(scene.get("frames"), list)
                or not scene["frames"]
            ):
                raise PipelineError("source scene timeline is invalid")
            result.append(dict(scene))
        return result

    bounds = _scene_bounds_for_long_plan(work, duration_s)
    raw_scenes = raw.get("scenes") if isinstance(raw, dict) else None
    result: list[dict] = []
    for position, bound in enumerate(bounds, 1):
        source = (
            raw_scenes[position - 1]
            if isinstance(raw_scenes, list)
            and len(raw_scenes) == len(bounds)
            and isinstance(raw_scenes[position - 1], dict)
            else None
        )
        source_index = (
            source.get("index", position) if source is not None else position
        )
        if source_index != position:
            raise PipelineError("source scene timeline is invalid")
        result.append({
            "index": position,
            "start_s": bound["start_s"],
            "end_s": bound["end_s"],
        })
    return result


def _bind_keyframe_source_timeline(
    work: Path,
    segments: list[dict],
    segment_metas: list[dict],
    source_scenes: list[dict],
) -> list[dict]:
    """Bind selected frame bytes to source time, scene and exact hard cuts.

    The video-maker contract copies selected frames byte-for-byte.  Hash
    ambiguity is therefore a contract failure, never a reason to guess a time.
    """
    if (
        not segments
        or len(segments) != len(segment_metas)
        or not source_scenes
    ):
        raise PipelineError("keyframe source timeline is invalid")
    if all(isinstance(meta.get("keyframe_sampling"), dict) for meta in segment_metas):
        updated = [dict(meta) for meta in segment_metas]
        previous: dict | None = None
        seen_scenes: set[str] = set()
        for expected_index, (segment, meta) in enumerate(zip(segments, updated), 1):
            receipt = meta["keyframe_sampling"]
            items = receipt.get("keyframes")
            names = meta.get("keyframes")
            if (
                segment.get("index") != expected_index
                or meta.get("index") != expected_index
                or receipt.get("schema") != "duet.backend-keyframe-sampling"
                or receipt.get("version") != 1
                or not isinstance(items, list)
                or len(items) != scene_planner.KEYFRAMES_PER_SEGMENT
                or names != [f"{order:02d}.png" for order in range(1, 10)]
            ):
                raise PipelineError("backend keyframe source timeline is invalid")
            sources = []
            segwork = work / "segments" / str(expected_index) / "work"
            for order, item in enumerate(items, 1):
                name = names[order - 1]
                if (
                    not isinstance(item, dict)
                    or item.get("order") != order
                    or item.get("path") != f"keyframes/{name}"
                    or not isinstance(item.get("source_scene_id"), str)
                    or not item["source_scene_id"]
                    or isinstance(item.get("source_time_s"), bool)
                    or not isinstance(item.get("source_time_s"), (int, float))
                    or not math.isfinite(float(item["source_time_s"]))
                    or isinstance(item.get("repeated"), bool) is False
                ):
                    raise PipelineError("backend keyframe source timeline is invalid")
                try:
                    data = (segwork / "keyframes" / name).read_bytes()
                except OSError:
                    raise PipelineError("backend keyframe source timeline is invalid") from None
                if hashlib.sha256(data).hexdigest() != item.get("sha256"):
                    raise PipelineError("backend frozen keyframe changed")
                source_time_s = float(item["source_time_s"])
                if previous is None:
                    transition = {"type": "start", "at_s": source_time_s}
                elif item["source_scene_id"] != previous["source_scene_id"]:
                    at_s = item.get("source_scene_start_s")
                    if (
                        isinstance(at_s, bool)
                        or not isinstance(at_s, (int, float))
                        or not math.isfinite(float(at_s))
                    ):
                        raise PipelineError("backend keyframe source timeline is invalid")
                    transition = {"type": "hard_cut", "at_s": float(at_s)}
                else:
                    transition = {"type": "continuous", "at_s": None}
                if previous is not None and source_time_s < previous["source_time_s"]:
                    raise PipelineError("backend keyframe source timeline is invalid")
                if (
                    previous is not None
                    and source_time_s == previous["source_time_s"]
                    and not item["repeated"]
                ):
                    raise PipelineError("backend keyframe repeat provenance is invalid")
                source = {
                    "order": order,
                    "source_time_s": source_time_s,
                    "source_scene_id": item["source_scene_id"],
                    "transition": transition,
                }
                sources.append(source)
                seen_scenes.add(item["source_scene_id"])
                previous = source
            meta["keyframe_sources"] = sources
        expected_scenes = {
            f"SCENE_{scene['index']:02d}" for scene in source_scenes
        }
        if seen_scenes != expected_scenes:
            raise PipelineError("backend keyframe source timeline misses scene anchor")
        return updated
    scenes: list[dict] = []
    previous_end: float | None = None
    for position, raw in enumerate(source_scenes, 1):
        try:
            index = raw["index"]
            start_s = float(raw["start_s"])
            end_s = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            raise PipelineError("keyframe source timeline is invalid") from None
        if (
            index != position
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or start_s >= end_s
            or (previous_end is not None and abs(start_s - previous_end) > _FLOAT_COMPARISON_EPS_S)
        ):
            raise PipelineError("keyframe source timeline is invalid")
        scenes.append({
            "index": index,
            "id": f"SCENE_{index:02d}",
            "start_s": round(start_s, long_video.BOUNDARY_PRECISION),
            "end_s": round(end_s, long_video.BOUNDARY_PRECISION),
        })
        previous_end = end_s

    def source_scene(time_s: float) -> dict:
        for scene in scenes:
            if scene["start_s"] <= time_s < scene["end_s"]:
                return scene
        if abs(time_s - scenes[-1]["end_s"]) <= _FLOAT_COMPARISON_EPS_S:
            return scenes[-1]
        raise PipelineError("keyframe source timeline is invalid")

    resolved: list[tuple[int, int, float, dict]] = []
    updated: list[dict] = []
    for expected_index, (segment, meta) in enumerate(
        zip(segments, segment_metas), 1
    ):
        if (
            not isinstance(segment, dict)
            or not isinstance(meta, dict)
            or segment.get("index") != expected_index
            or meta.get("index") != expected_index
        ):
            raise PipelineError("keyframe source timeline is invalid")
        names = meta.get("keyframes")
        if (
            not isinstance(names, list)
            or len(names) != 9
            or names != [f"{order:02d}.png" for order in range(1, 10)]
        ):
            raise PipelineError("keyframe source timeline is invalid")
        segwork = work / "segments" / str(expected_index) / "work"
        try:
            manifest = json.loads(
                (segwork / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise PipelineError("keyframe source timeline is invalid") from None
        manifest_frames = manifest.get("frames") if isinstance(manifest, dict) else None
        if not isinstance(manifest_frames, list) or not manifest_frames:
            raise PipelineError("keyframe source timeline is invalid")
        candidates: dict[str, list[float]] = {}
        segroot = segwork.resolve()
        for raw_frame in manifest_frames:
            try:
                relative = raw_frame["file"]
                local_time = float(raw_frame["time_seconds"])
            except (KeyError, TypeError, ValueError):
                raise PipelineError("keyframe source timeline is invalid") from None
            if (
                not isinstance(relative, str)
                or not relative
                or not math.isfinite(local_time)
                or local_time < 0
            ):
                raise PipelineError("keyframe source timeline is invalid")
            candidate = (segwork / relative).resolve()
            try:
                candidate.relative_to(segroot)
                data = candidate.read_bytes()
            except (OSError, ValueError):
                raise PipelineError("keyframe source timeline is invalid") from None
            if not data:
                raise PipelineError("keyframe source timeline is invalid")
            digest = hashlib.sha256(data).hexdigest()
            candidates.setdefault(digest, []).append(local_time)
        start_s = float(segment["start_s"])
        for order, name in enumerate(names, 1):
            try:
                selected_data = (segwork / "keyframes" / name).read_bytes()
            except OSError:
                raise PipelineError("keyframe source timeline is invalid") from None
            matches = candidates.get(hashlib.sha256(selected_data).hexdigest(), [])
            if len(matches) != 1:
                raise PipelineError("keyframe source timeline is ambiguous")
            source_time_s = round(
                start_s + matches[0], long_video.BOUNDARY_PRECISION
            )
            scene = source_scene(source_time_s)
            resolved.append((expected_index, order, source_time_s, scene))
        updated.append(dict(meta))

    if (
        {scene["id"] for _segment, _order, _time, scene in resolved}
        != {scene["id"] for scene in scenes}
    ):
        raise PipelineError("keyframe source timeline misses scene anchor")

    previous: tuple[int, int, float, dict] | None = None
    per_segment: dict[int, list[dict]] = {item["index"]: [] for item in segments}
    for segment_index, order, source_time_s, scene in resolved:
        if previous is None:
            transition = {"type": "start", "at_s": source_time_s}
        else:
            previous_time = previous[2]
            if source_time_s <= previous_time:
                raise PipelineError("keyframe source timeline is invalid")
            crossed = [
                candidate for candidate in scenes[1:]
                if previous_time < candidate["start_s"] <= source_time_s
            ]
            if len(crossed) > 1:
                raise PipelineError("keyframe source timeline misses scene anchor")
            if crossed:
                if crossed[0]["id"] != scene["id"]:
                    raise PipelineError("keyframe source timeline misses scene anchor")
                transition = {
                    "type": "hard_cut", "at_s": crossed[0]["start_s"],
                }
            else:
                if previous[3]["id"] != scene["id"]:
                    raise PipelineError("keyframe source timeline is invalid")
                transition = {"type": "continuous", "at_s": None}
        per_segment[segment_index].append({
            "order": order,
            "source_time_s": source_time_s,
            "source_scene_id": scene["id"],
            "transition": transition,
        })
        previous = (segment_index, order, source_time_s, scene)
    for item in updated:
        item["keyframe_sources"] = per_segment[item["index"]]
    return updated


def _probe_duration(path: Path) -> float:
    """探测 v:0 视觉时长；音频/容器尾巴不得影响切段验收。"""
    try:
        return storage.probe_video(path).duration_s
    except storage.UploadError as exc:
        raise PipelineError(str(exc)) from None


def _manifest_duration(work: Path) -> float:
    try:
        raw = json.loads((work / "manifest.json").read_text(encoding="utf-8"))[
            "duration_seconds"
        ]
        duration = float(raw)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise PipelineError("manifest.json invalid duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError("manifest.json invalid duration")
    return duration


def _validate_calibrated_duration(meta: dict, duration_s: float) -> float:
    """Apply every gate whenever the authoritative visual duration changes."""
    long_video.plan_segments(duration_s, [], [])
    duration = float(duration_s)
    if (
        duration > long_video.SHORT_VIDEO_MAX_S
        and meta.get("voice_mode") != "keep"
    ):
        raise PipelineError("long_video_audio_mode_unsupported")
    return duration


def _cut_segment(source: Path, start_s: float, end_s: float, segdir: Path) -> None:
    """ffmpeg 按段边界切源视频（-ss 在 -i 前重编码）+ ffprobe 验证切出时长误差 <0.1s。"""
    segdir.mkdir(parents=True, exist_ok=True)
    out = segdir / "source.mp4"
    length = end_s - start_s
    _run_cmd(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(source),
         "-to", f"{length:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(out)],
        timeout=300,
        step="segment cut",
    )
    duration = _probe_duration(out)
    if abs(duration - length) > CUT_DURATION_TOLERANCE_S:
        raise PipelineError(
            f"segment cut duration {duration:.3f}s off target {length:.3f}s "
            f"beyond tolerance {CUT_DURATION_TOLERANCE_S}s"
        )


def attribute_lines(lines: list[dict], segments: list[dict]) -> dict[int, list[dict]]:
    """按台词 start_s 落入段 [start_s, end_s) 归段（恰在边界归后段）；返回 {index: [台词]}。

    start_s 超出末段终点但在 0.01s 浮点误差内（voice 校验同口径容差）→ 归末段；
    更远的越界台词不归任何段。
    """
    result: dict[int, list[dict]] = {seg["index"]: [] for seg in segments}
    if not segments:
        return result
    last_end = segments[-1]["end_s"]
    for line in lines:
        target: int | None = None
        for seg in segments:
            if seg["start_s"] <= line["start_s"] < seg["end_s"]:
                target = seg["index"]
                break
        if target is None and line["start_s"] <= last_end + _SEG_TAIL_EPS_S:
            target = segments[-1]["index"]
        if target is not None:
            result[target].append(line)
    return result


def _apply_no_bgm_prefix(prompt: str, prompt_path: Path, *, enabled: bool) -> str:
    """多段模式机械添加 BGM 禁令；单段保持原文。两者都复核最终长度。"""
    final_prompt = f"{NO_BGM_LINE}\n{prompt}" if enabled else prompt
    if len(final_prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise PipelineError(f"prompt.txt exceeds {MAX_PROMPT_BYTES} bytes after prefix")
    prompt_path.write_text(final_prompt, encoding="utf-8")
    return final_prompt


def _source_dimensions(meta: dict) -> tuple[int, int]:
    width, height = meta.get("source_width"), meta.get("source_height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise PipelineError("source dimensions missing for generation defaults")
    return width, height


def _generation_defaults(
    paths: list[Path], meta: dict,
) -> tuple[dict[str, dict[str, object]], str, str, str]:
    width, height = _source_dimensions(meta)
    try:
        profiles, aspect_ratio = frame_fit.generation_fit_profiles(
            paths, source_width=width, source_height=height
        )
        resolution = frame_fit.recommended_resolution(min(width, height))
    except frame_fit.FrameFitError as exc:
        raise PipelineError(str(exc)) from None
    fit_mode = str(profiles[aspect_ratio]["default_fit_mode"])
    return profiles, aspect_ratio, resolution, fit_mode


def _generation_defaults_from_bytes(
    frames: list[bytes], meta: dict,
) -> tuple[dict[str, dict[str, object]], str, str, str]:
    width, height = _source_dimensions(meta)
    try:
        profiles, aspect_ratio = frame_fit.generation_fit_profiles_from_bytes(
            frames, source_width=width, source_height=height
        )
        resolution = frame_fit.recommended_resolution(min(width, height))
    except frame_fit.FrameFitError as exc:
        raise PipelineError(str(exc)) from None
    fit_mode = str(profiles[aspect_ratio]["default_fit_mode"])
    return profiles, aspect_ratio, resolution, fit_mode


def _legacy_generation_defaults(
    paths: list[Path],
) -> tuple[dict[str, dict[str, object]], str, str, str]:
    try:
        required = frame_fit.frames_require_fit(
            paths, h3.H3_DEFAULT_ASPECT_RATIO
        )
    except frame_fit.FrameFitError as exc:
        raise PipelineError(str(exc)) from None
    profiles = {
        "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        "9:16": {
            "fit_required": required,
            "default_fit_mode": "crop" if required else "none",
        },
    }
    return (
        profiles,
        h3.H3_DEFAULT_ASPECT_RATIO,
        h3.H3_DEFAULT_RESOLUTION,
        profiles[h3.H3_DEFAULT_ASPECT_RATIO]["default_fit_mode"],
    )


def _legacy_generation_defaults_from_bytes(
    frames: list[bytes],
) -> tuple[dict[str, dict[str, object]], str, str, str]:
    try:
        required = frame_fit.frame_bytes_require_fit(
            frames, h3.H3_DEFAULT_ASPECT_RATIO
        )
    except frame_fit.FrameFitError as exc:
        raise PipelineError(str(exc)) from None
    profiles = {
        "16:9": {"fit_required": True, "default_fit_mode": "crop"},
        "9:16": {
            "fit_required": required,
            "default_fit_mode": "crop" if required else "none",
        },
    }
    return (
        profiles,
        h3.H3_DEFAULT_ASPECT_RATIO,
        h3.H3_DEFAULT_RESOLUTION,
        profiles[h3.H3_DEFAULT_ASPECT_RATIO]["default_fit_mode"],
    )


def _read_segment_anchor_frames(work: Path) -> tuple[bytes, bytes]:
    """Snapshot the first/end sampled source frames before Codex can mutate work/."""
    try:
        manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PipelineError("segment manifest missing or invalid for source anchors") from None
    frames = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames, list) or not frames:
        raise PipelineError("segment manifest has no frames for source anchors")
    ordered: list[tuple[float, int, Path]] = []
    root = work.resolve()
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("file"), str):
            raise PipelineError("segment manifest frame invalid for source anchors")
        try:
            time_s = float(frame["time_seconds"])
        except (KeyError, TypeError, ValueError):
            raise PipelineError("segment manifest frame invalid for source anchors") from None
        if not math.isfinite(time_s):
            raise PipelineError("segment manifest frame invalid for source anchors")
        path = (work / frame["file"]).resolve()
        if path.parent != root or not path.is_file():
            raise PipelineError("segment manifest frame path invalid for source anchors")
        ordered.append((time_s, position, path))
    ordered.sort(key=lambda item: (item[0], item[1]))
    try:
        first = ordered[0][2].read_bytes()
        last = ordered[-1][2].read_bytes()
    except OSError:
        raise PipelineError("cannot read segment source anchors") from None
    if not first or not last:
        raise PipelineError("segment source anchor is empty")
    return first, last


def _write_segment_anchors(work: Path, anchors: tuple[bytes, bytes]) -> tuple[Path, Path]:
    """Write stable server-owned anchor paths from the pre-Codex snapshots."""
    anchor_dir = work / "anchors"
    if anchor_dir.is_dir():
        shutil.rmtree(anchor_dir)
    elif anchor_dir.exists():
        anchor_dir.unlink()
    anchor_dir.mkdir()
    first_path = anchor_dir / "first.png"
    last_path = anchor_dir / "last.png"
    first_path.write_bytes(anchors[0])
    last_path.write_bytes(anchors[1])
    return first_path, last_path


def _process_segment(settings: Settings, work: Path, source: Path, seg: dict, runner,
                     lines: list[dict] | None, target_language: str = "",
                     *, new_input_contract: bool = False,
                     keyframe_selection: list[dict] | None = None) -> dict:
    """单段完整流程：切段 → 抽帧 → 写该段台词 → codex（cwd=段目录）→ 校验 → 后端加前缀。

    段目录内嵌套 work/：帧/台词/产物都在 segdir/work/，SKILL.md 的 work/ 路径逐字适用，
    段 prompt 与单段逐字相同（_codex_prompt）；codex 的 cwd 即段目录（物理隔离，看不到
    段外内容）；scripts/ 拷入段目录（裁剪工具按相对路径引用），scenes.json 不拷入。
    任一失败包装为 PipelineError 并指明段号；返回 meta.segments 条目。
    """
    index = seg["index"]
    segdir = work / "segments" / str(index)
    segwork = segdir / "work"
    try:
        _cut_segment(source, seg["start_s"], seg["end_s"], segdir)
        segwork.mkdir(parents=True, exist_ok=True)
        if new_input_contract and lines:
            if voice.extract_audio(segdir) is None:
                raise PipelineError(
                    f"segment {index} normalized voice reference is missing"
                )
        _run_cmd(
            [sys.executable, str(EXTRACT_SCRIPT), str(segdir / "source.mp4"),
             "--out-dir", str(segwork), "--fps", "4"],
            timeout=120,
            step=f"segment {index} extract",
        )
        sampling_receipt = None
        frozen_keyframes = None
        if keyframe_selection is not None:
            _names, sampling_receipt, frozen_keyframes = (
                _materialize_backend_keyframes(source, segwork, keyframe_selection)
            )
        anchor_frames = _read_segment_anchor_frames(segwork) if new_input_contract else None
        # 该段台词（白名单净化后；lines 为 None = 无口播，不写文件）
        if lines is not None:
            (segwork / "voice_lines.json").write_text(
                json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # 裁剪工具按 scripts/crop_image.py 相对 cwd 引用：scripts/ 拷入段目录
        _replace_scripts(work.parent, segdir)
        visual_request = _codex_prompt(
            segdir,
            target_language,
            visual_only=new_input_contract,
        )
        if frozen_keyframes is not None:
            visual_request += (
                "\n后端已按唯一场景采样算法冻结 work/keyframes/01.png 至 09.png。"
                "逐张读取这九帧并据其既定顺序写 prompt.txt；禁止增删、替换、重排或修改任何关键帧。\n"
            )
        keyframes, prompt = _run_visual_with_retry(
            settings,
            runner,
            segdir,
            visual_request,
            segwork,
            isolate_dialogue=new_input_contract,
            step=f"segment {index} visual codex",
            frozen_keyframes=frozen_keyframes,
        )
        visual_prompt = prompt
        if new_input_contract:
            first_anchor, last_anchor = _write_segment_anchors(segwork, anchor_frames)
            (segwork / "visual_prompt.txt").write_text(visual_prompt, encoding="utf-8")
            prompt = long_video.compose_segment_visual_prompt(visual_prompt)
            try:
                prompt = prepared_input.compose_final_prompt(prompt, lines or [])
            except prepared_input.PreparedInputError as exc:
                raise PipelineError(f"prepared segment prompt invalid: {exc}") from None
        prompt = _apply_no_bgm_prefix(prompt, segwork / "prompt.txt", enabled=True)
        result = {
            "index": index,
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "keyframes": keyframes,
            "prompt": prompt,
            "lines": [line["text"] for line in (lines or [])],
        }
        if new_input_contract:
            result.update(
                chain_id=seg["chain_id"],
                join_mode=seg["join_mode"],
                source=f"segments/{index}/source.mp4",
                keyframe_paths=[
                    f"segments/{index}/work/keyframes/{name}" for name in keyframes
                ],
                first_frame_path=f"segments/{index}/work/anchors/{first_anchor.name}",
                last_frame_path=f"segments/{index}/work/anchors/{last_anchor.name}",
                visual_prompt=visual_prompt,
                dialogue=list(lines or []),
            )
            if sampling_receipt is not None:
                result["keyframe_sampling"] = sampling_receipt
        return result
    except Exception as e:
        raise PipelineError(f"segment {index} failed: {e}") from None


def _dialogue_for_prepared_input(
    meta: dict,
    mode: str,
    voice_lines: list[dict],
) -> tuple[dict, ...]:
    """把 meta 中的当前有效台词转换为 prepared-input 的显式来源契约。"""
    duration_s, _engine_duration = _prepared_durations(meta)
    filter_enabled = meta.get("vocal_filter_enabled", _vocal_filter_enabled())
    if not isinstance(filter_enabled, bool):
        raise PipelineError("vocal_filter_enabled in meta must be bool")
    if mode == "auto":
        automatic = [
            {
                "text": item["text"],
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "classification": item["classification"],
                "provenance": "asr",
            }
            for item in meta.get("voice_line_provenance", [])
            if item.get("kept") is True
        ]
        if len(automatic) != len(voice_lines):
            raise PipelineError("automatic dialogue provenance does not match effective lines")
        return prepared_input.prepare_dialogue(
            "auto",
            duration_s,
            automatic_lines=automatic,
            vocal_filter_enabled=filter_enabled,
        )
    if mode in ("edit", "custom"):
        return prepared_input.prepare_dialogue(
            mode,
            duration_s,
            supplied_lines=voice_lines,
            vocal_filter_enabled=filter_enabled,
        )
    if mode == "none":
        return prepared_input.prepare_dialogue("none", duration_s)
    raise PipelineError(f"unknown dialogue_mode: {mode}")


def _planner_dialogue(meta: Mapping, supplied_lines: list[dict]) -> list[dict]:
    """Project the one dialogue authority used by planning and frozen segments."""
    mode = meta.get("dialogue_mode")
    if mode == "none":
        return []
    if mode in {"edit", "custom"}:
        return [dict(line) for line in supplied_lines]
    if mode != "auto":
        raise PipelineError(f"unknown dialogue_mode: {mode}")
    provenance = meta.get("voice_line_provenance")
    if not isinstance(provenance, list):
        raise PipelineError("automatic dialogue provenance is missing")
    return [
        {
            "text": line.get("text"),
            "start_s": line.get("start_s"),
            "end_s": line.get("end_s"),
            "classification": "spoken",
        }
        for line in provenance
        if isinstance(line, Mapping)
        and line.get("kept") is True
        and line.get("classification") == "spoken"
    ]


def _prepared_durations(meta: dict) -> tuple[float, int]:
    """返回 receipt 的实际时长与统一换算的 H3 整秒请求时长。"""
    raw = meta.get("duration_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PipelineError("duration_s in meta must be a positive finite number")
    actual = float(raw)
    if not math.isfinite(actual) or actual <= 0:
        raise PipelineError("duration_s in meta must be a positive finite number")
    engine_duration = long_video.provider_duration_s(0.0, actual)
    return actual, engine_duration


def _write_prepared_receipt(
    settings: Settings,
    cid: str,
    cdir: Path,
    work: Path,
    source: Path,
    keyframes: list[str],
    visual_prompt: str,
    dialogue_mode: str,
    voice_lines: list[dict],
    aspect_ratio: str,
    resolution: str,
    fit_mode: str,
) -> str:
    """冻结单段 H3 输入并把 receipt 位置写入 meta；不触发任何远程请求。"""
    current = storage.load_meta(settings.data_dir, cid)
    if current is None:
        raise PipelineError("conversation disappeared while preparing H3 input")
    dialogue = _dialogue_for_prepared_input(current, dialogue_mode, voice_lines)
    duration_s, engine_duration = _prepared_durations(current)
    visual_path = work / "visual_prompt.txt"
    visual_path.write_text(visual_prompt, encoding="utf-8")
    originals = [work / "keyframes" / name for name in keyframes]
    if fit_mode == "none":
        h3_keyframes = originals
    else:
        try:
            h3_keyframes = list(frame_fit.fit_frames(
                originals,
                work / "h3_frames" / aspect_ratio.replace(":", "x") / fit_mode,
                fit_mode,
                aspect_ratio,
            ))
        except frame_fit.FrameFitError as exc:
            raise PipelineError(str(exc)) from None
    filter_enabled = current.get("vocal_filter_enabled", _vocal_filter_enabled())
    if not isinstance(filter_enabled, bool):
        raise PipelineError("vocal_filter_enabled in meta must be bool")
    engine_request = {
        "h3": {
            "workflow": H3_ENGINE_WORKFLOW,
            "duration": engine_duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "provider_resolution": h3.provider_resolution(
                aspect_ratio, resolution
            ),
        },
    }
    try:
        frozen = prepared_input.write_prepared_input(
            root=cdir,
            source=source,
            audio=(work / "voice.mp3") if (work / "voice.mp3").is_file() else None,
            keyframes=h3_keyframes,
            visual=visual_path,
            final=work / "prompt.txt",
            dialogue_mode=dialogue_mode,
            dialogue=dialogue,
            vocal_filter_enabled=filter_enabled,
            duration_s=duration_s,
            ratio=aspect_ratio,
            fit_mode=fit_mode,
            engine_request=engine_request,
        )
    except prepared_input.PreparedInputError as exc:
        raise PipelineError(f"prepared input invalid: {exc}") from None
    final_prompt = frozen.final_prompt.data.decode("utf-8")
    storage.update_meta(
        settings.data_dir,
        cid,
        prompt=final_prompt,
        prepared_input_receipt=prepared_input.RECEIPT_FILENAME,
    )
    return final_prompt


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise PipelineError("prompt fusion artifact is invalid") from None


def _require_prompt_fusion_v2_input(input_data: bytes) -> dict:
    """Creation may consume only the current source-timeline contract."""
    try:
        payload = json.loads(input_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PipelineError("prompt fusion input is invalid") from None
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "version", "segments"}
        or payload.get("schema") != long_generation.PROMPT_FUSION_INPUT_SCHEMA
        or payload.get("version") != long_generation.PROMPT_FUSION_VERSION
        or not isinstance(segments, list)
        or not segments
    ):
        raise PipelineError("prompt fusion input is invalid")
    for index, segment in enumerate(segments, 1):
        frames = segment.get("new_keyframes") if isinstance(segment, dict) else None
        if (
            not isinstance(segment, dict)
            or segment.get("index") != index
            or not isinstance(frames, list)
            or len(frames) != 9
        ):
            raise PipelineError("prompt fusion input is invalid")
        try:
            long_generation._freeze_local_keyframe_sources(
                [{
                    key: frame[key]
                    for key in (
                        "order", "segment_time_s", "source_scene_id", "transition",
                    )
                } for frame in frames],
            )
        except (
            KeyError,
            TypeError,
            long_generation.LongGenerationError,
        ):
            raise PipelineError("prompt fusion input is invalid") from None
    return payload


def _copy_prompt_fusion_frame(
    *, root: Path, stage: Path, frame: Mapping,
) -> None:
    """Copy one receipt-bound regular image into an otherwise empty stage."""
    raw_path = frame.get("path")
    expected_sha256 = frame.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise PipelineError("prompt fusion input is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PipelineError("prompt fusion input is invalid")
    source_candidate = root / relative
    if source_candidate.is_symlink():
        raise PipelineError("prompt fusion input is invalid")
    try:
        source = source_candidate.resolve(strict=True)
        source.relative_to(root)
        descriptor = os.open(
            source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, ValueError):
        raise PipelineError("prompt fusion input is invalid") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PipelineError("prompt fusion input is invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
    finally:
        os.close(descriptor)
    if not data or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise PipelineError("prompt fusion input is invalid")
    destination = stage / relative
    try:
        destination.resolve().relative_to(stage)
    except ValueError:
        raise PipelineError("prompt fusion input is invalid") from None
    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(data)
    except OSError:
        raise PipelineError("prompt fusion input is invalid") from None


def _read_prompt_fusion_stage_output(path: Path, *, segment_count: int) -> bytes:
    max_bytes = 64 * 1024 + MAX_PROMPT_BYTES * segment_count
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        raise PipelineError("prompt fusion raw output is missing") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise PipelineError("prompt fusion raw output is invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if not data or len(data) > max_bytes:
        raise PipelineError("prompt fusion raw output is invalid")
    return data


def queue_prompt_fusion(
    settings: Settings,
    cid: str,
    *,
    input_data: bytes,
    image_acceptance_sha256: str,
) -> str:
    """Freeze the one project-level video-prompt-fusion invocation."""
    root = (settings.data_dir / cid).resolve()
    work = root / "work"
    if (
        not input_data
        or not isinstance(image_acceptance_sha256, str)
        or len(image_acceptance_sha256) != 64
    ):
        raise PipelineError("prompt fusion input is invalid")
    _require_prompt_fusion_v2_input(input_data)
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    manifest_path = work / h3_project.SOURCE_FILENAME
    input_sha256 = hashlib.sha256(input_data).hexdigest()
    current = storage.load_meta(settings.data_dir, cid)
    if current is None:
        return "missing"
    state = current.get("_prompt_fusion")
    if (
        isinstance(state, dict)
        and state.get("input_sha256") == input_sha256
        and state.get("image_acceptance_sha256") == image_acceptance_sha256
    ):
        if state.get("status") in {"queued", "running"}:
            return state["status"]
        if state.get("status") == "failed":
            return "failed"
        if state.get("status") == "done":
            try:
                long_generation.load_bound_prompt_fusion_manifest(
                    root=root,
                    meta=current,
                )
                return "done"
            except long_generation.LongGenerationError:
                storage.update_meta(
                    settings.data_dir,
                    cid,
                    _prompt_fusion={
                        **state,
                        "status": "failed",
                        "error": "prompt_fusion_manifest_invalid",
                    },
                )
                return "failed"
    _atomic_bytes(input_path, input_data)
    output_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    storage.update_meta(
        settings.data_dir,
        cid,
        _prompt_fusion={
            "version": long_generation.PROMPT_FUSION_VERSION,
            "status": "queued",
            "error": None,
            "input_sha256": input_sha256,
            "image_acceptance_sha256": image_acceptance_sha256,
            "manifest_sha256": None,
        },
    )
    return "queued"


def _publish_prompt_fusion_manifest(
    *, root: Path, meta: Mapping, state: Mapping,
) -> tuple[long_generation.FrozenPromptFusion, bytes]:
    """Revalidate frozen fusion authorities and publish the manifest last."""
    work = root / "work"
    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    frozen_skill_path = work / PROMPT_FUSION_FROZEN_SKILL_FILENAME
    manifest_path = work / h3_project.SOURCE_FILENAME
    try:
        input_data = input_path.read_bytes()
        source_skill_data = PROMPT_FUSION_SKILL_MD.read_bytes()
        frozen_skill_data = frozen_skill_path.read_bytes()
    except OSError:
        raise PipelineError("prompt fusion frozen authority is missing") from None
    if hashlib.sha256(input_data).hexdigest() != state.get("input_sha256"):
        raise PipelineError("prompt fusion input drifted")
    if not source_skill_data or frozen_skill_data != source_skill_data:
        raise PipelineError("prompt fusion Skill drifted")
    try:
        frozen = long_generation.load_prompt_fusion(
            input_path=input_path, output_path=output_path, root=root,
        )
    except long_generation.LongGenerationError as exc:
        raise PipelineError(exc.code) from None
    if (
        long_generation.prompt_fusion_image_authority_sha256(meta)
        != state.get("image_acceptance_sha256")
    ):
        raise PipelineError("image acceptance drifted during prompt fusion")
    manifest = {
        "schema": long_generation.PROMPT_FUSION_MANIFEST_SCHEMA,
        "version": long_generation.PROMPT_FUSION_MANIFEST_VERSION,
        "image_acceptance_sha256": state["image_acceptance_sha256"],
        "input": {
            "path": f"work/{h3_project.SKILL_INPUT_FILENAME}",
            "sha256": frozen.input_sha256,
        },
        "output": {
            "path": "work/h3_prompt_plan.json",
            "sha256": frozen.output_sha256,
        },
        "skill": {
            "source_path": "skills/video-prompt-fusion/SKILL.md",
            "frozen_path": f"work/{PROMPT_FUSION_FROZEN_SKILL_FILENAME}",
            "sha256": hashlib.sha256(source_skill_data).hexdigest(),
        },
        "segments": [
            {
                "index": index,
                "final_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            }
            for index, prompt in enumerate(frozen.final_prompts, 1)
        ],
    }
    manifest_data = _canonical_json_bytes(manifest)
    manifest_existed = manifest_path.exists()
    if manifest_existed:
        try:
            existing_manifest_data = manifest_path.read_bytes()
        except OSError:
            raise PipelineError("prompt fusion manifest is unreadable") from None
        if manifest_path.is_symlink() or existing_manifest_data != manifest_data:
            raise PipelineError("prompt fusion manifest drifted")
    else:
        try:
            _atomic_bytes(manifest_path, manifest_data)
        except Exception:
            try:
                if manifest_path.read_bytes() == manifest_data:
                    manifest_path.unlink()
            except OSError:
                pass
            raise
    try:
        long_generation.load_prompt_fusion_manifest(
            root=root, skill_source_path=PROMPT_FUSION_SKILL_MD,
        )
    except long_generation.LongGenerationError as exc:
        if not manifest_existed:
            try:
                if manifest_path.read_bytes() == manifest_data:
                    manifest_path.unlink()
            except OSError:
                pass
        raise PipelineError(exc.code) from None
    return frozen, manifest_data


def _recoverable_prompt_fusion_state(
    root: Path,
    meta: Mapping,
    *,
    expected_raw_output_sha256: str | None,
) -> dict:
    state = meta.get("_prompt_fusion")
    if (
        not isinstance(state, dict)
        or state.get("version") != long_generation.PROMPT_FUSION_VERSION
        or state.get("status") != "failed"
        or state.get("error") != "prompt_fusion_output_invalid"
        or state.get("manifest_sha256") is not None
        or meta.get("generation") is not None
    ):
        raise PipelineError("prompt fusion receipt finalization is not allowed")
    downstream_roots = (
        root / ".context-ir",
        root / ".h3",
        *tuple((root / "work" / "segments").glob("*/.context-ir")),
        *tuple((root / "work" / "segments").glob("*/.h3")),
    )
    if any(path.exists() for path in downstream_roots):
        raise PipelineError("prompt fusion downstream state already exists")
    frozen_state = dict(state)
    raw_output_path = state.get("raw_output_path")
    raw_output_sha256 = state.get("raw_output_sha256")
    legacy_recovery = raw_output_path is None and raw_output_sha256 is None
    if legacy_recovery:
        if (
            not isinstance(expected_raw_output_sha256, str)
            or len(expected_raw_output_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_raw_output_sha256
            )
        ):
            raise PipelineError("prompt fusion raw output authority is required")
        raw_output_path = "work/h3_prompt_plan.json"
        raw_output_sha256 = expected_raw_output_sha256
        frozen_state.update(
            raw_output_path=raw_output_path,
            raw_output_sha256=raw_output_sha256,
        )
    elif (
        expected_raw_output_sha256 is not None
        or raw_output_path != "work/h3_prompt_plan.json"
        or not isinstance(raw_output_sha256, str)
        or len(raw_output_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in raw_output_sha256
        )
    ):
        raise PipelineError("prompt fusion raw output authority is invalid")
    output_path = root / "work" / "h3_prompt_plan.json"
    try:
        raw_output_data = output_path.read_bytes()
    except OSError:
        raise PipelineError("prompt fusion raw output is missing") from None
    if (
        output_path.is_symlink()
        or hashlib.sha256(raw_output_data).hexdigest() != raw_output_sha256
    ):
        raise PipelineError("prompt fusion raw output drifted")
    return frozen_state


def _unlink_exact_prompt_fusion_manifest(path: Path, data: bytes) -> None:
    try:
        if not path.is_symlink() and path.read_bytes() == data:
            path.unlink()
    except OSError:
        pass


def finalize_prompt_fusion_receipt(
    settings: Settings,
    cid: str,
    *,
    expected_raw_output_sha256: str | None = None,
) -> str:
    """Finalize one exact terminal LF-envelope output without running a Skill."""
    root = (settings.data_dir / cid).resolve()
    manifest_path = root / "work" / h3_project.SOURCE_FILENAME
    publication: dict[str, object] = {}

    def commit(current: dict) -> None:
        state = _recoverable_prompt_fusion_state(
            root,
            current,
            expected_raw_output_sha256=expected_raw_output_sha256,
        )
        manifest_existed = manifest_path.exists()
        _frozen, manifest_data = _publish_prompt_fusion_manifest(
            root=root, meta=current, state=state,
        )
        publication.update(
            data=manifest_data,
            created=not manifest_existed,
        )
        try:
            refreshed_state = _recoverable_prompt_fusion_state(
                root,
                current,
                expected_raw_output_sha256=expected_raw_output_sha256,
            )
        except PipelineError:
            raise PipelineError(
                "prompt fusion state drifted during finalization"
            ) from None
        if refreshed_state != state:
            raise PipelineError("prompt fusion state drifted during finalization")
        manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
        current["_prompt_fusion"] = {
            **state,
            "status": "done",
            "error": None,
            "manifest_sha256": manifest_sha256,
            "recovered_error": state["error"],
        }

    try:
        committed = storage.mutate_meta(settings.data_dir, cid, commit)
    except Exception:
        if publication.get("created") is True and isinstance(
            publication.get("data"), bytes
        ):
            _unlink_exact_prompt_fusion_manifest(
                manifest_path, publication["data"],
            )
        raise
    if committed is None:
        raise PipelineError("prompt fusion receipt finalization is not allowed")
    return "done"


def produce_prompt_fusion(settings: Settings, cid: str, runner) -> str:
    """Run video-prompt-fusion once and publish its production manifest last."""
    root = (settings.data_dir / cid).resolve()
    work = root / "work"
    claimed: dict[str, object] = {}

    def claim(meta: dict) -> None:
        current = meta.get("_prompt_fusion")
        if not isinstance(current, dict):
            claimed["status"] = "missing"
            return
        if current.get("status") != "queued":
            claimed["status"] = current.get("status", "missing")
            return
        state = {**current, "status": "running", "error": None}
        meta["_prompt_fusion"] = state
        claimed["status"] = "claimed"
        claimed["state"] = state

    meta = storage.mutate_meta(settings.data_dir, cid, claim)
    if meta is None or claimed.get("status") != "claimed":
        return str(claimed.get("status", "missing"))
    state = claimed["state"]
    assert isinstance(state, dict)

    def persist(status: str, error: str | None = None, **changes) -> None:
        storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion={**state, "status": status, "error": error, **changes},
        )

    input_path = work / h3_project.SKILL_INPUT_FILENAME
    output_path = work / "h3_prompt_plan.json"
    frozen_skill_path = work / PROMPT_FUSION_FROZEN_SKILL_FILENAME
    manifest_path = work / h3_project.SOURCE_FILENAME
    try:
        input_data = input_path.read_bytes()
        if hashlib.sha256(input_data).hexdigest() != state.get("input_sha256"):
            raise PipelineError("prompt fusion input drifted")
        input_payload = _require_prompt_fusion_v2_input(input_data)
        skill_data = PROMPT_FUSION_SKILL_MD.read_bytes()
        if not skill_data:
            raise PipelineError("prompt fusion Skill is missing")
        if not callable(getattr(runner, "run_isolated", None)):
            raise PipelineError("prompt fusion isolation unavailable")
        _atomic_bytes(frozen_skill_path, skill_data)
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="duet-prompt-fusion-", dir="/tmp",
        ) as raw_stage:
            stage = Path(raw_stage).resolve(strict=True)
            stage_work = stage / "work"
            stage_work.mkdir(mode=0o700)
            (stage / "SKILL.md").write_bytes(skill_data)
            (stage_work / h3_project.SKILL_INPUT_FILENAME).write_bytes(input_data)
            stage_output = stage_work / "h3_prompt_plan.json"
            stage_output.touch(mode=0o600, exist_ok=False)
            for segment in input_payload["segments"]:
                for frame in segment["new_keyframes"]:
                    _copy_prompt_fusion_frame(
                        root=root, stage=stage, frame=frame,
                    )
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md；只读取 work/multimodal_input.json "
                "及其中 SHA 绑定的有序图片，只写 work/h3_prompt_plan.json。",
                session_dir=root,
                writable_paths=(stage_output,),
            )
            raw_output_data = _read_prompt_fusion_stage_output(
                stage_output,
                segment_count=len(input_payload["segments"]),
            )
        _atomic_bytes(output_path, raw_output_data)
        state = {
            **state,
            "raw_output_path": "work/h3_prompt_plan.json",
            "raw_output_sha256": hashlib.sha256(raw_output_data).hexdigest(),
        }
        persisted_meta = storage.update_meta(
            settings.data_dir,
            cid,
            _prompt_fusion=state,
        )
        if persisted_meta is None:
            raise PipelineError("prompt fusion raw output receipt was not persisted")
        _frozen, manifest_data = _publish_prompt_fusion_manifest(
            root=root, meta=persisted_meta, state=state,
        )
        persist(
            "done",
            manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
        )
        return "done"
    except Exception as exc:
        persist("failed", str(exc))
        return "failed"


def run(settings: Settings, cid: str, runner, *, claimed_owner: object = None) -> None:
    """后台任务入口；任何步骤失败 → status=failed + error，不抛异常。"""
    # data_dir 可能是相对路径（生产默认 "data"）：子进程带 cwd 时相对路径会错位，统一起点解析为绝对
    cdir = (settings.data_dir / cid).resolve()
    work = cdir / "work"
    claim_owner = claimed_owner
    try:
        meta = (
            storage.claim_pipeline_input(settings.data_dir, cid)
            if claim_owner is None
            else storage.load_pipeline_claim(settings.data_dir, cid, claim_owner)
        )
        if meta is None:
            return
        if claim_owner is None:
            claim_owner = meta["_input_owner"]
        sources = sorted(cdir.glob("source.*"))
        if not sources:
            raise PipelineError("source video missing")
        source = sources[0]
        # Reject an invalid/oversized new contract before extraction, ASR, Codex,
        # or any later provider can observe the input.
        new_input_contract = "dialogue_mode" in meta and meta.get("duration_s") is not None
        if new_input_contract:
            try:
                source_probe = storage.probe_video(source)
                probed_duration = source_probe.duration_s
            except storage.UploadError as exc:
                raise PipelineError(str(exc)) from None
            probed_duration = _validate_calibrated_duration(meta, probed_duration)
            meta = storage.update_meta(
                settings.data_dir, cid,
                duration_s=probed_duration,
                source_width=source_probe.width,
                source_height=source_probe.height,
            ) or meta
        _run_cmd(
            [
                sys.executable, str(EXTRACT_SCRIPT), str(source),
                "--out-dir", str(work),
                "--fps", "4",
            ],
            timeout=120,
            step="extract",
        )
        if new_input_contract:
            manifest_duration = _manifest_duration(work)
            manifest_duration = _validate_calibrated_duration(meta, manifest_duration)
            meta = storage.update_meta(
                settings.data_dir, cid, duration_s=manifest_duration
            ) or meta
        # 新 H3 会话以 dialogue_mode 为唯一产品契约；voice_mode 仅供旧会话内部兼容。
        # duration_s 是上传探测后才写入的新输入契约完成标记；仅有默认字段但尚未探测，
        # 或历史会话没有实际时长时，仍走旧流水线且不伪造 prepared receipt。
        dialogue_mode = meta.get("dialogue_mode") if new_input_contract else None
        voice_mode = meta.get("voice_mode", "none")
        voice_lines: list[dict] | None = None
        if new_input_contract:
            if dialogue_mode != "auto":
                raise PipelineError(f"unknown initial dialogue_mode: {dialogue_mode}")
            auto_voice_mode = meta.get("voice_mode")
            if auto_voice_mode in (None, ""):
                auto_voice_mode = "keep"
            if auto_voice_mode not in ("keep", "rewrite", "translate"):
                raise PipelineError(f"unknown voice_mode for auto dialogue: {auto_voice_mode}")
            voice_lines = _voice_step(
                settings,
                cid,
                cdir,
                work,
                runner,
                auto_voice_mode,
                meta.get("target_language") or "",
                allow_no_audio=True,
            )
        elif voice_mode != "none":
            voice_lines = _voice_step(
                settings, cid, cdir, work, runner, voice_mode,
                meta.get("target_language") or "",
            )
        translate_lang = ""
        if not new_input_contract and voice_mode == "translate":
            translate_lang = (meta.get("target_language") or "").strip()
        segments = _detect_segments(settings, cid, source, work)
        duration_s = float(meta["duration_s"]) if new_input_contract else None
        source_scenes: list[dict] | None = None
        backend_keyframe_selections: dict[int, list[dict]] = {}
        planner_dialogue = voice_lines or []
        if new_input_contract:
            refreshed = storage.load_meta(settings.data_dir, cid)
            if refreshed is None:
                raise PipelineError("conversation disappeared during dialogue planning")
            meta = refreshed
            planner_dialogue = _planner_dialogue(meta, voice_lines or [])
            source_scenes = _source_scenes_for_timeline(work, duration_s)
            exact_inventory = all(
                isinstance(scene.get("frames"), list)
                and scene["frames"]
                and all(
                    isinstance(frame, dict)
                    and isinstance(frame.get("decode_frame_index"), int)
                    and not isinstance(frame.get("decode_frame_index"), bool)
                    and isinstance(frame.get("pts"), int)
                    and not isinstance(frame.get("pts"), bool)
                    for frame in scene["frames"]
                )
                for scene in source_scenes
            )
            if exact_inventory:
                segments = scene_planner.plan_segments(
                    duration_s,
                    source_scenes,
                    planner_dialogue,
                )
                backend_keyframe_selections = {
                    segment["index"]: scene_planner.select_segment_keyframes(
                        source_scenes, segment
                    )
                    for segment in segments
                }
            else:
                # Read-only compatibility for pre-exact scene receipts.  New
                # scenes.py output always carries the exact inventory above.
                segments = long_video.plan_segments(
                    duration_s,
                    source_scenes,
                    planner_dialogue,
                )
        if not segments:
            # 单段模式：work/keyframes + work/prompt.txt，不注入 workaround 前缀；
            # 裁剪工具以 scripts/crop_image.py 相对工作目录引用，scripts/ 拷进会话目录
            _replace_scripts(cdir, cdir)
            keyframes, prompt = _run_visual_with_retry(
                settings,
                runner,
                cdir,
                _codex_prompt(cdir, translate_lang, visual_only=new_input_contract),
                work,
                isolate_dialogue=new_input_contract,
                step="visual codex",
            )
            if new_input_contract:
                frame_paths = [work / "keyframes" / name for name in keyframes]
                transition_skeleton = _frame_inventory(
                    {0: frame_paths},
                    segment_lineage={
                        0: {"chain_id": "short-000", "join_mode": "hard_cut"},
                    },
                )
                continuity, image_prompts = _generate_image_optimization_project(
                    settings,
                    runner,
                    [{
                        "index": 0,
                        "chain_id": "short-000",
                        "join_mode": "hard_cut",
                        "keyframes_dir": work / "keyframes",
                        "transition_skeleton": transition_skeleton,
                    }],
                    session_dir=cdir,
                    step="project image postprocess codex",
                )
                if continuity is None or set(image_prompts) != {0}:
                    raise PipelineError("image optimization output is missing or invalid")
                if continuity.get("version") not in {3, 4}:
                    image_prompt = image_prompts[0]
                    _write_image_optimization_prompt(work, image_prompt)
            prompt = _apply_no_bgm_prefix(prompt, work / "prompt.txt", enabled=False)
            frame_paths = [work / "keyframes" / name for name in keyframes]
            profiles, aspect_ratio, resolution, default_fit_mode = (
                _generation_defaults(frame_paths, meta)
                if new_input_contract
                else _legacy_generation_defaults(frame_paths)
            )
            fit_required = bool(profiles[aspect_ratio]["fit_required"])
            if new_input_contract:
                prompt = _write_prepared_receipt(
                    settings,
                    cid,
                    cdir,
                    work,
                    source,
                    keyframes,
                    prompt,
                    dialogue_mode,
                    voice_lines or [],
                    aspect_ratio,
                    resolution,
                    default_fit_mode,
                )
            completion = dict(
                status="done",
                keyframes=keyframes,
                prompt=prompt,
                fit_required=fit_required,
                fit_profiles=profiles,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                fit_mode=default_fit_mode,
            )
            if new_input_contract:
                frozen_continuity, frozen_prompts = _freeze_image_optimization(
                    settings,
                    {**meta, **completion},
                    continuity,
                    image_prompts,
                    {0: frame_paths},
                    require_dual_target=True,
                    segment_lineage={
                        0: {"chain_id": "short-000", "join_mode": "hard_cut"},
                    },
                )
                completion.update(frozen_continuity)
                completion.update(frozen_prompts)
            storage.finish_input_claim(
                settings.data_dir,
                cid,
                claim_owner,
                **completion,
            )
        else:
            # 多段模式：各段目录独立、无共享状态，线程池并发；CodexRunner.run 自带信号量兜底
            # lines_by_seg 为 None = 无口播：段目录不写 voice_lines.json
            if new_input_contract:
                lines_by_seg = {
                    seg["index"]: long_video.localize_dialogue(
                        planner_dialogue, seg
                    )
                    for seg in segments
                }
            else:
                lines_by_seg = (
                    attribute_lines(voice_lines, segments) if voice_lines is not None else None
                )
            if lines_by_seg is not None:
                # 越界台词不归段：计数留痕（meta 内部字段，不静默丢失）
                dropped = len(planner_dialogue) - sum(
                    len(v) for v in lines_by_seg.values()
                )
                if dropped:
                    storage.update_meta(settings.data_dir, cid, voice_lines_dropped=dropped)
            # 段并发上限 = codex_concurrency 的一半：一条长视频不得占满全部 codex 槽饿死其他会话
            workers = min(len(segments), max(1, settings.codex_concurrency // 2))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                seg_metas = list(
                    pool.map(
                        lambda seg: _process_segment(
                            settings, work, source, seg, runner,
                            lines_by_seg.get(seg["index"]) if lines_by_seg is not None else None,
                            translate_lang,
                            new_input_contract=new_input_contract,
                            **(
                                {"keyframe_selection": backend_keyframe_selections[seg["index"]]}
                                if seg["index"] in backend_keyframe_selections
                                else {}
                            ),
                        ),
                        segments,
                    )
                )
            if new_input_contract:
                assert source_scenes is not None
                seg_metas = _bind_keyframe_source_timeline(
                    work, segments, seg_metas, source_scenes,
                )
            image_prompts: dict[int, str] | None = None
            continuity: dict | None = None
            if new_input_contract:
                continuity, image_prompts = _generate_segmented_image_prompts(
                    settings,
                    runner,
                    segments,
                    seg_metas,
                    work,
                    session_dir=cdir,
                )
            changes: dict = {"status": "done", "segments": seg_metas}
            if new_input_contract:
                receipt_segments = []
                multimodal_intent: list[bool] = []
                multimodal_complete: list[bool] = []
                for seg in seg_metas:
                    segdir = work / "segments" / str(seg["index"])
                    segwork = segdir / "work"
                    manifest_path = segwork / h3_project.SOURCE_FILENAME
                    multimodal_intent.append(any(
                        (segwork / name).exists()
                        for name in (
                            h3_project.SKILL_INPUT_FILENAME,
                            "h3_prompt_plan.json",
                            h3_project.SOURCE_FILENAME,
                        )
                    ))
                    multimodal_complete.append(manifest_path.is_file())
                    receipt_segment = {
                            **seg,
                            "source_path": segdir / "source.mp4",
                            "keyframe_paths": [
                                segwork / "keyframes" / name for name in seg["keyframes"]
                            ],
                            "first_frame_path": segwork / "anchors" / "first.png",
                            "last_frame_path": segwork / "anchors" / "last.png",
                            "visual_prompt_path": segwork / "visual_prompt.txt",
                            "final_prompt_path": segwork / "prompt.txt",
                            "dialogue": seg["dialogue"],
                        }
                    if manifest_path.is_file():
                        receipt_segment["multimodal_manifest_path"] = manifest_path
                    receipt_segments.append(receipt_segment)
                if any(multimodal_intent) and not all(multimodal_complete):
                    raise PipelineError("long_video_multimodal_incomplete")
                receipt_workflow = (
                    h3.H3_MULTIMODAL_WORKFLOW
                    if all(multimodal_complete)
                    else H3_ENGINE_WORKFLOW
                )
                receipt_path = long_video.write_plan_receipt(
                    cdir,
                    source=source,
                    duration_s=duration_s,
                    segments=receipt_segments,
                    workflow=receipt_workflow,
                )
                changes["long_video_plan_receipt"] = receipt_path.name
                try:
                    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                    plan = long_generation.freeze_plan(
                        cdir, {**meta, **changes}, digest, "none", dialogue_mode,
                        settings=settings,
                    )
                    anchors = [
                        anchor
                        for segment in plan.segments
                        for anchor in (
                            (
                                (segment.first_frame_data,)
                                if segment.join_mode == "hard_cut"
                                else ()
                            )
                            + (segment.last_frame_data,)
                        )
                    ]
                    profiles, aspect_ratio, resolution, default_fit_mode = (
                        _generation_defaults_from_bytes(anchors, meta)
                    )
                except (
                    OSError,
                    frame_fit.FrameFitError,
                    long_generation.LongGenerationError,
                ) as exc:
                    raise PipelineError(str(exc)) from None
                changes.update(
                    fit_required=profiles[aspect_ratio]["fit_required"],
                    fit_profiles=profiles,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    fit_mode=default_fit_mode,
                )
            # 新 schema 保留 segments；短视频仍只写顶层 keyframes/prompt。
            if new_input_contract:
                if image_prompts is None or continuity is None:
                    raise PipelineError("image continuity was not generated")
                frozen_continuity, frozen_prompts = _freeze_image_optimization(
                    settings,
                    {**meta, **changes},
                    continuity,
                    image_prompts,
                    {
                        seg["index"]: [
                            work / "segments" / str(seg["index"]) / "work"
                            / "keyframes" / name
                            for name in seg["keyframes"]
                        ]
                        for seg in seg_metas
                    },
                    require_dual_target=False,
                    segment_lineage={
                        seg["index"]: {
                            "chain_id": seg["chain_id"],
                            "join_mode": seg["join_mode"],
                        }
                        for seg in seg_metas
                    },
                )
                changes.update(frozen_continuity)
                changes.update(frozen_prompts)
            storage.finish_input_claim(
                settings.data_dir, cid, claim_owner, **changes
            )
    except Exception as e:
        if claim_owner is not None:
            storage.finish_input_claim(
                settings.data_dir, cid, claim_owner,
                status="failed", error=str(e)[:500],
            )
