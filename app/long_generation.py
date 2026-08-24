"""Fail-closed orchestration for paid long-video H3 segment generation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from app import frame_fit, h3, long_video, prepared_input, stitch, storage

WORKFLOW = h3.H3_BOUNDARY_WORKFLOW
_PIPELINE_NO_BGM = "不要生成背景音乐"
_EPS = 1e-6


class LongGenerationError(RuntimeError):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


@dataclass(frozen=True)
class FrozenSegment:
    index: int
    start_s: float
    end_s: float
    chain_id: str
    join_mode: str
    workdir: Path
    first_frame: Path
    first_frame_data: bytes
    last_frame: Path
    last_frame_data: bytes
    prompt: str


@dataclass(frozen=True)
class FrozenPlan:
    root: Path
    source: Path
    receipt: str
    segments: tuple[FrozenSegment, ...]


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _canonical_digest(value: object) -> str:
    try:
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode()
    except (TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    return hashlib.sha256(data).hexdigest()


def plan_receipt(root: Path, meta: Mapping) -> str | None:
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        return None
    path = Path(root) / name
    if not path.is_file():
        return None
    try:
        return _digest(path)
    except LongGenerationError:
        return None


def _bound_path(root: Path, artifact: object) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None
    if not path.is_file() or _digest(path) != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path


def _bound_bytes(root: Path, artifact: object) -> tuple[Path, bytes]:
    """Read one receipt-bound artifact once and retain the verified bytes."""
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise LongGenerationError("long_video_plan_invalid")
    relative, expected = artifact.get("path"), artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise LongGenerationError("long_video_plan_invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
        data = path.read_bytes()
    except (ValueError, OSError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if hashlib.sha256(data).hexdigest() != expected:
        raise LongGenerationError("long_video_plan_invalid")
    return path, data


def _relative_to_work(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root / "work").as_posix()
    except ValueError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _fit_anchor(path: Path, data: bytes, output: Path, fit_mode: str,
                *, prepare: bool) -> tuple[Path, bytes]:
    if fit_mode == "none":
        return path, data
    try:
        fitted = frame_fit.fit_frame_bytes(data, fit_mode, label=path.name)
    except frame_fit.FrameFitError:
        raise LongGenerationError("frame_fit_failed") from None
    target = output / (path.stem + ".png")
    if prepare:
        output.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_bytes(fitted)
            temporary.replace(target)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise LongGenerationError("frame_fit_failed") from None
    else:
        try:
            if target.read_bytes() != fitted:
                raise LongGenerationError("frame_fit_failed")
        except OSError:
            raise LongGenerationError("frame_fit_failed") from None
    return target, fitted


def freeze_plan(root: Path, meta: Mapping, expected_receipt: str, fit_mode: str,
                dialogue_mode: str, *, prepare_fit: bool = True) -> FrozenPlan:
    """Validate every immutable plan fact and pre-fit every source anchor."""
    root = Path(root).resolve()
    name = meta.get("long_video_plan_receipt")
    if name != long_video.PLAN_RECEIPT_FILENAME:
        raise LongGenerationError("long_video_plan_invalid")
    receipt_path = root / name
    try:
        receipt_data = receipt_path.read_bytes()
    except OSError:
        raise LongGenerationError("long_video_plan_invalid") from None
    receipt = hashlib.sha256(receipt_data).hexdigest()
    if expected_receipt != receipt:
        raise LongGenerationError("long_video_plan_changed", 409)
    try:
        payload = json.loads(receipt_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LongGenerationError("long_video_plan_invalid") from None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "duet.long-video-plan"
        or payload.get("version") != 1
        or payload.get("workflow") != WORKFLOW
    ):
        raise LongGenerationError("long_video_plan_invalid")
    source = _bound_path(root, payload.get("source"))
    try:
        duration = float(payload["video"]["duration_s"])
        meta_duration = float(meta["duration_s"])
    except (KeyError, TypeError, ValueError):
        raise LongGenerationError("long_video_plan_invalid") from None
    raw_segments, meta_segments = payload.get("segments"), meta.get("segments")
    if (
        not math.isfinite(duration)
        or duration <= long_video.SHORT_VIDEO_MAX_S
        or duration > long_video.LONG_VIDEO_MAX_S
        or abs(duration - meta_duration) > _EPS
        or not isinstance(raw_segments, list)
        or not raw_segments
        or not isinstance(meta_segments, list)
        or len(raw_segments) != len(meta_segments)
    ):
        raise LongGenerationError("long_video_plan_invalid")

    frozen: list[FrozenSegment] = []
    previous_end = 0.0
    previous_chain = None
    for position, (raw, current) in enumerate(zip(raw_segments, meta_segments), 1):
        if not isinstance(raw, dict) or not isinstance(current, dict):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            index = raw["index"]
            start_s, end_s = float(raw["start_s"]), float(raw["end_s"])
            chain_id, join_mode = raw["chain_id"], raw["join_mode"]
        except (KeyError, TypeError, ValueError):
            raise LongGenerationError("long_video_plan_invalid") from None
        comparable = ("index", "start_s", "end_s", "chain_id", "join_mode")
        if (
            index != position
            or any(current.get(key) != raw.get(key) for key in comparable)
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or abs(start_s - previous_end) > _EPS
            or end_s - start_s < 1
            or math.ceil(end_s - start_s) > 15
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
            or (position == 1 and join_mode != "hard_cut")
            or (join_mode == "continue" and chain_id != previous_chain)
            or (position > 1 and join_mode == "hard_cut" and chain_id == previous_chain)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        if raw.get("source") is None:
            raise LongGenerationError("long_video_plan_invalid")
        segment_source = _bound_path(root, raw["source"])
        keys = raw.get("keyframes")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 9:
            raise LongGenerationError("long_video_plan_invalid")
        keyframe_paths = [_bound_path(root, artifact) for artifact in keys]
        anchors = raw.get("anchors")
        if (
            not isinstance(anchors, list)
            or len(anchors) != 2
            or [item.get("role") if isinstance(item, dict) else None for item in anchors]
            != ["first", "end"]
        ):
            raise LongGenerationError("long_video_plan_invalid")
        first_source, first_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[0].items() if k != "role"}
        )
        last_source, last_source_data = _bound_bytes(
            root, {k: v for k, v in anchors[1].items() if k != "role"}
        )
        expected_prefix = f"segments/{index}/"
        if (
            current.get("source") != expected_prefix + "source.mp4"
            or segment_source != root / "work" / current["source"]
            or current.get("keyframe_paths") != [
                _relative_to_work(root, path) for path in keyframe_paths
            ]
            or current.get("first_frame_path")
            != _relative_to_work(root, first_source)
            or current.get("last_frame_path")
            != _relative_to_work(root, last_source)
        ):
            raise LongGenerationError("long_video_plan_invalid")
        visual_path, visual_data = _bound_bytes(root, raw.get("visual_prompt"))
        final_path, final_data = _bound_bytes(root, raw.get("final_prompt"))
        try:
            visual = visual_data.decode("utf-8")
            final = final_data.decode("utf-8")
        except UnicodeDecodeError:
            raise LongGenerationError("long_video_plan_invalid") from None
        dialogue = current.get("dialogue")
        dialogue_binding = raw.get("dialogue")
        if (
            not isinstance(dialogue, list)
            or not isinstance(dialogue_binding, dict)
            or set(dialogue_binding) != {"count", "sha256"}
            or dialogue_binding.get("count") != len(dialogue)
            or dialogue_binding.get("sha256") != _canonical_digest(dialogue)
            or current.get("visual_prompt") != visual
            or current.get("prompt") != final
        ):
            raise LongGenerationError("long_video_plan_invalid")
        try:
            rebuilt_visual = long_video.compose_segment_visual_prompt(visual)
            auto_prompt = f"{_PIPELINE_NO_BGM}\n" + prepared_input.compose_final_prompt(
                rebuilt_visual, dialogue
            )
        except (prepared_input.PreparedInputError, long_video.LongVideoError):
            raise LongGenerationError("long_video_plan_invalid") from None
        if final != auto_prompt:
            raise LongGenerationError("long_video_plan_invalid")
        if dialogue_mode == "none":
            try:
                rebuilt = prepared_input.compose_final_prompt(
                    long_video.compose_segment_visual_prompt(visual), ()
                )
            except (prepared_input.PreparedInputError, long_video.LongVideoError):
                raise LongGenerationError("long_video_plan_invalid") from None
            prompt = f"{_PIPELINE_NO_BGM}\n{rebuilt}"
        else:
            prompt = final
        segdir = root / "work" / "segments" / str(index)
        fit_root = segdir / "work" / "h3_frames" / fit_mode
        # Complete all static transformations before the caller can make a POST.
        first, first_data = _fit_anchor(
            first_source, first_source_data, fit_root / "first", fit_mode,
            prepare=prepare_fit,
        )
        last, last_data = _fit_anchor(
            last_source, last_source_data, fit_root / "end", fit_mode,
            prepare=prepare_fit,
        )
        frozen.append(FrozenSegment(index, start_s, end_s, chain_id, join_mode,
                                    segdir, first, first_data, last, last_data,
                                    prompt))
        previous_end, previous_chain = end_s, chain_id
    if abs(previous_end - duration) > _EPS:
        raise LongGenerationError("long_video_plan_invalid")
    return FrozenPlan(root, source, receipt, tuple(frozen))


def child_request_id(parent_id: str, receipt: str, index: int) -> str:
    digest = hashlib.sha256(f"{parent_id}\0{receipt}\0{index}".encode()).hexdigest()
    return f"long-{digest[:59]}"  # 64 bytes, deterministic, provider-safe.


def _extract_last_frame(video: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", str(video),
         "-vf", "reverse", "-frames:v", "1", str(temporary)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise LongGenerationError("long_video_tail_frame_failed")
    temporary.replace(output)
    return output


def _request(settings, cid: str, plan: FrozenPlan, segment: FrozenSegment,
             parent_id: str, fit_mode: str, *, frozen_child_id: str | None = None,
             prepare_inputs: bool = True) -> h3.H3Request:
    first, first_data = segment.first_frame, segment.first_frame_data
    if segment.join_mode == "continue":
        upstream = plan.segments[segment.index - 2]
        tail = upstream.workdir / "work" / "generated_last.png"
        if not tail.is_file():
            if not prepare_inputs:
                raise LongGenerationError("long_video_tail_frame_missing")
            tail = _extract_last_frame(upstream.workdir / "generated.mp4", tail)
        try:
            tail_data = tail.read_bytes()
        except OSError:
            raise LongGenerationError("long_video_tail_frame_missing") from None
        continued = segment.workdir / "work" / "h3_frames" / fit_mode / "continued"
        first, first_data = _fit_anchor(
            tail, tail_data, continued, fit_mode, prepare=prepare_inputs
        )
    return h3.H3Request(
        cid=f"{cid}-segment-{segment.index}",
        workdir=segment.workdir,
        client_request_id=(
            frozen_child_id
            or child_request_id(parent_id, plan.receipt, segment.index)
        ),
        prompt=segment.prompt,
        keyframes=(),
        voice_texts=(),
        voice_receipt=h3.voice_texts_receipt(()),
        duration=max(1, math.ceil(segment.end_s - segment.start_s)),
        autodl_token=settings.autodl_art_token,
        timeouts=h3.Timeouts(
            request_s=settings.h3_request_timeout_s,
            h3_poll_s=settings.h3_poll_timeout_s,
            download_s=settings.h3_download_timeout_s,
            poll_interval_s=settings.h3_poll_interval_s,
            retry_count=settings.retry_count,
            retry_interval_s=settings.retry_interval_s,
        ),
        mode="boundary",
        first_frame=(first, first_data),
        last_frame=(segment.last_frame, segment.last_frame_data),
    )


def public_segments(generation: Mapping) -> list[dict]:
    result = []
    for item in generation.get("segments", []):
        if isinstance(item, dict):
            result.append({key: item.get(key) for key in
                           ("index", "chain_id", "join_mode", "status", "attempt", "error")})
    return result


def _result_status(result: h3.H3Result) -> tuple[str, str | None]:
    if result.status == "succeeded":
        return "succeeded", None
    if result.status in {"submission_unknown", "h3_submitting"}:
        return "submission_unknown", "submission_unknown"
    if result.status == "h3_running" or result.error_code in {
        "h3_query_failed", "h3_timeout", "download_failed", "download_dns_failed",
        "download_peer_unverified", "output_write_failed", "output_probe_failed",
    }:
        return "resume_required", result.error_code or result.status
    return "failed", result.error_code or "h3_failed"


def reusable_segment_indices(
    expected_indices: Sequence[int],
    prior_segments: object,
    artifact_exists: Callable[[int], bool],
) -> frozenset[int]:
    """Return paid segment outputs that are safe to reuse on retry."""
    expected = tuple(expected_indices)
    if len(set(expected)) != len(expected) or not isinstance(prior_segments, list):
        return frozenset()
    by_index: dict[int, dict] = {}
    observed: list[int] = []
    for item in prior_segments:
        if not isinstance(item, dict):
            return frozenset()
        index = item.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in expected
            or index in by_index
        ):
            return frozenset()
        observed.append(index)
        by_index[index] = item
    if observed != [index for index in expected if index in by_index]:
        return frozenset()

    reusable = set()
    for index, item in by_index.items():
        if item.get("status") != "succeeded":
            continue
        try:
            exists = artifact_exists(index)
        except OSError:
            exists = False
        if exists:
            reusable.add(index)
    return frozenset(reusable)


def generation_segments_are_valid(
    expected_segments: object,
    generation: Mapping,
) -> bool:
    """Validate the complete ordered persisted coordinator state."""
    raw = generation.get("segments")
    if not isinstance(expected_segments, (list, tuple)) or not isinstance(raw, list):
        return False
    if len(raw) != len(expected_segments) or not raw:
        return False
    keys = {
        "index", "chain_id", "join_mode", "status", "attempt", "error",
        "child_request_id",
    }
    statuses = {
        "not_started", "queued", "running", "resume_required", "succeeded",
        "failed", "submission_unknown",
    }
    for position, (expected, item) in enumerate(zip(expected_segments, raw), 1):
        if not isinstance(item, dict) or set(item) != keys:
            return False
        if isinstance(expected, FrozenSegment):
            expected_index = expected.index
            expected_chain = expected.chain_id
            expected_join = expected.join_mode
        elif isinstance(expected, Mapping):
            expected_index = expected.get("index")
            expected_chain = expected.get("chain_id")
            expected_join = expected.get("join_mode")
        else:
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        if (
            expected_index != position
            or item.get("index") != expected_index
            or item.get("chain_id") != expected_chain
            or item.get("join_mode") != expected_join
            or item.get("status") not in statuses
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 0
            or (
                item.get("error") is not None
                and not isinstance(item.get("error"), str)
            )
            or (
                child_id is not None
                and (not isinstance(child_id, str) or not child_id)
            )
        ):
            return False
    return True


def bound_reusable_segment_indices(
    settings,
    cid: str,
    plan: FrozenPlan,
    generation: Mapping,
) -> frozenset[int]:
    """Single source of truth for paid-count and execution reuse decisions."""
    segments = generation.get("segments")
    expected = tuple(item.index for item in plan.segments)
    if not generation_segments_are_valid(plan.segments, generation):
        return frozenset()
    meta = storage.load_meta(settings.data_dir, cid)
    fit_mode = meta.get("fit_mode") if isinstance(meta, dict) else None
    if fit_mode not in {"none", "crop", "pad"}:
        return frozenset()
    by_index = {
        item.get("index"): item for item in segments or [] if isinstance(item, dict)
    }
    reusable: set[int] = set()

    def valid(index: int) -> bool:
        item = by_index.get(index)
        segment = plan.segments[index - 1]
        if not isinstance(item, dict):
            return False
        attempt = item.get("attempt")
        child_id = item.get("child_request_id")
        if (
            item.get("status") != "succeeded"
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(child_id, str)
            or not child_id
            or (
                segment.join_mode == "continue"
                and segment.index - 1 not in reusable
            )
        ):
            return False
        try:
            request = _request(
                settings, cid, plan, segment,
                str(generation.get("client_request_id") or "frozen-parent"),
                fit_mode,
                frozen_child_id=child_id,
                prepare_inputs=False,
            )
            return h3.output_is_reusable(
                request, expected_duration_s=request.duration
            )
        except (OSError, h3.H3Error, LongGenerationError, ValueError):
            return False

    structurally_valid = reusable_segment_indices(expected, segments, lambda _index: True)
    if structurally_valid != frozenset(
        item.get("index") for item in segments or []
        if isinstance(item, dict) and item.get("status") == "succeeded"
    ):
        return frozenset()
    for index in expected:
        if valid(index):
            reusable.add(index)
    return frozenset(reusable)


def initial_generation(settings, cid: str, plan: FrozenPlan, parent_id: str, attempt: int,
                       old: Mapping | None = None) -> dict:
    plan_by_index = {segment.index: segment for segment in plan.segments}
    raw_old_segments = (old or {}).get("segments", [])
    old_segments = raw_old_segments if isinstance(raw_old_segments, list) else []
    reusable = bound_reusable_segment_indices(
        settings, cid, plan, old or {"segments": []}
    )
    old_by_index = {
        item.get("index"): item for item in old_segments
        if isinstance(item, dict)
    }
    items = []
    for segment in plan.segments:
        prior = old_by_index.get(segment.index, {})
        succeeded = segment.index in reusable
        items.append({
            "index": segment.index,
            "chain_id": segment.chain_id,
            "join_mode": segment.join_mode,
            "status": "succeeded" if succeeded else "not_started",
            "attempt": prior.get("attempt", 0) if succeeded else int(prior.get("attempt", 0) or 0),
            "error": None,
            "child_request_id": prior.get("child_request_id") if succeeded else None,
        })
    return {"status": "queued", "error": None, "attempt": attempt,
            "client_request_id": parent_id, "stage": "h3", "segments": items}


def _stitch(settings, cid: str, plan: FrozenPlan, dialogue_mode: str) -> None:
    stitch.stitch_video(
        segments=[stitch.StitchSegment(
            item.workdir / "generated.mp4", item.end_s - item.start_s, item.join_mode
        ) for item in plan.segments],
        source_video=plan.source,
        output=plan.root / "generated.mp4",
        audio_mode="keep" if dialogue_mode == "auto" else "mute",
    )
    if not stitched_output_is_reusable(plan, dialogue_mode):
        raise LongGenerationError("long_video_stitch_output_invalid")


def stitched_output_is_reusable(plan: FrozenPlan, dialogue_mode: str) -> bool:
    """Validate the published video against the exact local stitch receipt."""
    if dialogue_mode not in {"auto", "none"}:
        return False
    output = plan.root / "generated.mp4"
    receipt_path = plan.root / stitch.RECEIPT_FILENAME
    audio_mode = "keep" if dialogue_mode == "auto" else "mute"
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "version", "segments", "audio", "output"}
            or payload.get("schema") != "duet.stitch"
            or payload.get("version") != 1
        ):
            return False
        budgets = stitch._frame_budgets([
            stitch.StitchSegment(
                item.workdir / "generated.mp4",
                item.end_s - item.start_s,
                item.join_mode,
            )
            for item in plan.segments
        ])
        expected_segments = [
            {
                "index": index,
                "path": str((item.workdir / "generated.mp4").resolve()),
                "sha256": stitch._sha256(item.workdir / "generated.mp4"),
                "target_duration_s": item.end_s - item.start_s,
                "output_frames": budgets[index - 1],
                "join_mode": item.join_mode,
            }
            for index, item in enumerate(plan.segments, 1)
        ]
        if payload.get("segments") != expected_segments:
            return False
        source_info = stitch._probe(plan.source)
        expected_audio = {
            "mode": audio_mode,
            "source": str(plan.source.resolve()),
            "source_sha256": stitch._sha256(plan.source),
            "source_has_audio": source_info.has_audio,
        }
        if payload.get("audio") != expected_audio:
            return False
        output_receipt = payload.get("output")
        stat = output.stat()
        if (
            not output.is_file()
            or stat.st_size <= 0
            or not isinstance(output_receipt, dict)
            or set(output_receipt)
            != {"name", "sha256", "size", "duration_s", "fps"}
            or output_receipt.get("name") != "generated.mp4"
            or output_receipt.get("size") != stat.st_size
            or output_receipt.get("sha256") != stitch._sha256(output)
            or output_receipt.get("fps") != stitch.FPS
        ):
            return False
        expected_duration = sum(item.end_s - item.start_s for item in plan.segments)
        output_info = stitch._validate_output(
            output, expected_duration, audio_mode, source_info.has_audio
        )
        receipt_duration = float(output_receipt.get("duration_s"))
        return (
            math.isfinite(receipt_duration)
            and abs(receipt_duration - output_info.duration_s) <= _EPS
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        stitch.StitchError,
    ):
        return False


def run(settings, cid: str, plan: FrozenPlan, *, startup: bool = False) -> None:
    """Drive chains with max-two concurrency; only this coordinator writes meta."""
    meta = storage.load_meta(settings.data_dir, cid)
    if not meta or not isinstance(meta.get("generation"), dict):
        return
    generation = meta["generation"]
    if not generation_segments_are_valid(plan.segments, generation):
        storage.update_meta(
            settings.data_dir,
            cid,
            generation={
                **generation,
                "status": "submission_unknown",
                "error": "submission_unknown",
            },
        )
        return
    parent_id = generation.get("client_request_id")
    fit_mode = meta.get("fit_mode")
    dialogue_mode = meta.get("dialogue_mode")
    states = {item["index"]: dict(item) for item in generation.get("segments", [])}
    if not isinstance(parent_id, str) or fit_mode not in {"none", "crop", "pad"}:
        return

    def persist(status: str | None = None, error: str | None = None, stage: str = "h3") -> None:
        nonlocal generation
        ordered = [states[item.index] for item in plan.segments]
        generation = {**generation, "segments": ordered,
                      "status": status or generation.get("status"), "error": error, "stage": stage}
        storage.update_meta(settings.data_dir, cid, generation=generation)

    reusable = bound_reusable_segment_indices(settings, cid, plan, generation)
    for index, state in states.items():
        if state.get("status") == "succeeded" and index not in reusable:
            state.update(
                status="failed",
                error="long_video_segment_output_invalid",
            )

    if startup:
        for segment in plan.segments:
            state = states[segment.index]
            if state.get("status") not in {"queued", "running", "resume_required"}:
                continue
            try:
                request = _request(settings, cid, plan, segment, parent_id, fit_mode)
                result = h3.resume(request)
                if result.status == "not_started":
                    state["status"], state["error"] = (
                        "submission_unknown", "submission_unknown"
                    )
                else:
                    state["status"], state["error"] = _result_status(result)
            except Exception:
                state["status"], state["error"] = "submission_unknown", "submission_unknown"
        if any(item.get("status") == "submission_unknown" for item in states.values()):
            persist("submission_unknown", "submission_unknown")
        elif all(item.get("status") == "succeeded" for item in states.values()):
            try:
                _stitch(settings, cid, plan, dialogue_mode)
            except Exception:
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
        elif any(item.get("status") == "failed" for item in states.values()):
            persist("failed", "long_video_segment_failed")
        else:
            # A known attempt still needs GET recovery, or an unstarted child
            # awaits an explicit same-parent confirmation.  Startup never POSTs.
            persist("resume_required", "long_video_resume_required")
        return

    chains: dict[str, list[FrozenSegment]] = {}
    for segment in plan.segments:
        chains.setdefault(segment.chain_id, []).append(segment)

    attempted_indices: set[int] = set()

    def ready() -> list[FrozenSegment]:
        candidates = []
        for chain in chains.values():
            for segment in chain:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") in {"not_started", "queued", "resume_required"}:
                    if segment.index in attempted_indices:
                        break
                    prior = [states[item.index].get("status") for item in chain if item.index < segment.index]
                    if all(value == "succeeded" for value in prior):
                        candidates.append(segment)
                    break
                break
        return candidates

    def worker(segment: FrozenSegment, action: str):
        try:
            request = _request(settings, cid, plan, segment, parent_id, fit_mode)
        except LongGenerationError as exc:
            return None, ("failed", exc.code)
        except Exception:
            return None, ("failed", "long_video_request_invalid")
        try:
            result = h3.start(request) if action == "start" else h3.resume(request)
            if action == "resume" and result.status == "not_started":
                return request.client_request_id, (
                    "submission_unknown", "submission_unknown"
                )
            status = _result_status(result)
            if status[0] == "succeeded" and not h3.output_is_reusable(
                request, expected_duration_s=request.duration
            ):
                status = ("failed", "long_video_segment_output_invalid")
            return request.client_request_id, status
        except h3.H3Error as exc:
            try:
                inspected = h3.inspect(request)
                status = _result_status(inspected)
            except Exception:
                status = ("submission_unknown", "submission_unknown")
            if status[0] == "failed" and exc.code in {"submission_unknown", "state_persist_failed", "h3_internal_error"}:
                status = ("submission_unknown", "submission_unknown")
            return request.client_request_id, status
        except Exception:
            return request.client_request_id, ("submission_unknown", "submission_unknown")

    active = {}
    locked = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            if not locked:
                active_chains = {segment.chain_id for segment in active.values()}
                for segment in ready():
                    if len(active) >= 2:
                        break
                    if segment.chain_id in active_chains:
                        continue
                    state = states[segment.index]
                    is_new_child = state.get("status") == "not_started"
                    action = "start" if is_new_child else "resume"
                    state["status"], state["error"] = "running", None
                    if is_new_child:
                        state["attempt"] = int(state.get("attempt", 0) or 0) + 1
                    attempted_indices.add(segment.index)
                    persist("running", None)
                    future = pool.submit(worker, segment, action)
                    active[future] = segment
                    active_chains.add(segment.chain_id)
            if not active:
                break
            done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                segment = active.pop(future)
                child_id, (status, error) = future.result()
                states[segment.index].update(
                    status=status, error=error, child_request_id=child_id
                )
                if status == "submission_unknown":
                    locked = True
                persist("submission_unknown" if locked else "running",
                        "submission_unknown" if locked else None)

    if locked:
        persist("submission_unknown", "submission_unknown")
        return
    statuses = {item.get("status") for item in states.values()}
    if statuses == {"succeeded"}:
        try:
            _stitch(settings, cid, plan, dialogue_mode)
        except Exception:
            persist("failed", "long_video_stitch_failed", "stitch")
        else:
            persist("succeeded", None, "stitch")
    elif "resume_required" in statuses:
        persist("resume_required", "long_video_resume_required")
    else:
        persist("failed", "long_video_segment_failed")
