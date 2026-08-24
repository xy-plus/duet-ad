"""Fail-closed orchestration for paid long-video H3 segment generation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app import frame_fit, h3, long_video, prepared_input, stitch, storage

WORKFLOW = h3.H3_BOUNDARY_WORKFLOW
_PIPELINE_NO_BGM = "不要生成背景音乐"
_EPS = 1e-6
FIT_LAYOUT_LEGACY = "legacy-v0"
FIT_LAYOUT_ASPECT = "aspect-v1"
_FIT_LAYOUTS = frozenset({FIT_LAYOUT_LEGACY, FIT_LAYOUT_ASPECT})
_FAST_MODE_WORKERS = 8


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
    receipt_version: int = long_video.PLAN_RECEIPT_VERSION
    aspect_ratio: str = h3.H3_DEFAULT_ASPECT_RATIO
    resolution: str = h3.H3_DEFAULT_RESOLUTION
    legacy_layout: bool = False


def _segment_duration_s(plan: FrozenPlan, segment: FrozenSegment) -> float:
    """Interpret a segment boundary with its frozen plan receipt version."""
    try:
        return long_video.segment_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        )
    except long_video.LongVideoError:
        raise LongGenerationError("long_video_plan_invalid") from None


def _stitch_segments(plan: FrozenPlan) -> list[stitch.StitchSegment]:
    return [
        stitch.StitchSegment(
            item.workdir / "generated.mp4",
            _segment_duration_s(plan, item),
            item.join_mode,
        )
        for item in plan.segments
    ]


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


def _fit_anchor(
    path: Path, data: bytes, output: Path, fit_mode: str, aspect_ratio: str,
    *, prepare: bool,
) -> tuple[Path, bytes]:
    if fit_mode == "none":
        return path, data
    try:
        fitted = frame_fit.fit_frame_bytes(
            data, fit_mode, aspect_ratio, label=path.name
        )
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


def _persisted_fit_layout(meta: Mapping) -> str | None:
    generation = meta.get("generation")
    if not isinstance(generation, Mapping) or "fit_layout" not in generation:
        return None
    layout = generation.get("fit_layout")
    if layout not in _FIT_LAYOUTS:
        raise LongGenerationError("frame_fit_failed")
    return str(layout)


def _fit_outputs_complete(paths: tuple[Path, Path], aspect_ratio: str) -> bool:
    if not all(path.is_file() for path in paths):
        return False
    try:
        return not frame_fit.frames_require_fit(paths, aspect_ratio)
    except frame_fit.FrameFitError:
        return False


def freeze_plan(root: Path, meta: Mapping, expected_receipt: str, fit_mode: str,
                dialogue_mode: str, *, aspect_ratio: str | None = None,
                resolution: str | None = None,
                prepare_fit: bool = True) -> FrozenPlan:
    """Validate every immutable plan fact and pre-fit every source anchor."""
    root = Path(root).resolve()
    if (aspect_ratio is None) != (resolution is None):
        raise LongGenerationError("invalid_generation_parameters", 422)
    parameters_explicit = aspect_ratio is not None
    persisted_layout = _persisted_fit_layout(meta)
    detect_existing_layout = (
        persisted_layout is None and not prepare_fit and fit_mode != "none"
    )
    if persisted_layout is not None:
        legacy_layout: bool | None = persisted_layout == FIT_LAYOUT_LEGACY
    elif prepare_fit or fit_mode == "none":
        # Current callers explicitly supply semantic parameters and always use
        # the versioned path.  Calls without them are the historical contract.
        # initial_generation persists this choice before any provider POST.
        legacy_layout = not parameters_explicit
    else:
        # Pre-marker attempts are recovered from their complete, decodable
        # frozen files.  Never let fields added after the POST select a path.
        legacy_layout = None
    aspect_ratio = (
        meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        if aspect_ratio is None else aspect_ratio
    )
    resolution = (
        meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        if resolution is None else resolution
    )
    if aspect_ratio not in h3.H3_ASPECT_RATIOS:
        raise LongGenerationError("invalid_aspect_ratio", 422)
    if resolution not in h3.H3_RESOLUTIONS:
        raise LongGenerationError("invalid_resolution", 422)
    if fit_mode not in {"none", "crop", "pad"}:
        raise LongGenerationError("frame_fit_failed")
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
    receipt_version = payload.get("version") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "duet.long-video-plan"
        or isinstance(receipt_version, bool)
        or not isinstance(receipt_version, int)
        or receipt_version not in {
            long_video.LEGACY_PLAN_RECEIPT_VERSION,
            long_video.PLAN_RECEIPT_VERSION,
        }
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
    max_provider_duration = (
        long_video.SEGMENT_PROVIDER_MAX_DURATION_S
        if receipt_version == long_video.PLAN_RECEIPT_VERSION
        else long_video.LEGACY_PROVIDER_MAX_DURATION_S
    )
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
        try:
            frozen_duration = long_video.segment_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
        except long_video.LongVideoError:
            raise LongGenerationError("long_video_plan_invalid") from None
        comparable = ("index", "start_s", "end_s", "chain_id", "join_mode")
        if (
            index != position
            or any(current.get(key) != raw.get(key) for key in comparable)
            or not math.isfinite(start_s)
            or not math.isfinite(end_s)
            or abs(start_s - previous_end) > _EPS
            or frozen_duration < long_video.SEGMENT_MIN_S
            or long_video.provider_duration_s(
                start_s, end_s, receipt_version=receipt_version
            )
            > max_provider_duration
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
        fit_base = segdir / "work" / "h3_frames"
        legacy_root = fit_base / fit_mode
        aspect_root = fit_base / aspect_ratio.replace(":", "x") / fit_mode
        if detect_existing_layout:
            legacy_paths = (
                legacy_root / "first" / first_source.name,
                legacy_root / "end" / last_source.name,
            )
            aspect_paths = (
                aspect_root / "first" / first_source.name,
                aspect_root / "end" / last_source.name,
            )
            has_legacy = _fit_outputs_complete(legacy_paths, aspect_ratio)
            has_aspect = _fit_outputs_complete(aspect_paths, aspect_ratio)
            if has_legacy == has_aspect:
                raise LongGenerationError("frame_fit_failed")
            if legacy_layout is None:
                legacy_layout = has_legacy
            elif legacy_layout != has_legacy:
                raise LongGenerationError("frame_fit_failed")
        fit_root = legacy_root if legacy_layout else aspect_root
        # Complete all static transformations before the caller can make a POST.
        first, first_data = _fit_anchor(
            first_source, first_source_data, fit_root / "first", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        last, last_data = _fit_anchor(
            last_source, last_source_data, fit_root / "end", fit_mode,
            aspect_ratio,
            prepare=prepare_fit,
        )
        frozen.append(FrozenSegment(index, start_s, end_s, chain_id, join_mode,
                                    segdir, first, first_data, last, last_data,
                                    prompt))
        previous_end, previous_chain = end_s, chain_id
    if abs(previous_end - duration) > _EPS:
        raise LongGenerationError("long_video_plan_invalid")
    assert legacy_layout is not None
    return FrozenPlan(
        root=root,
        source=source,
        receipt=receipt,
        segments=tuple(frozen),
        receipt_version=receipt_version,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        legacy_layout=legacy_layout,
    )


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
             prepare_inputs: bool = True, fast_mode: bool = False) -> h3.H3Request:
    first, first_data = segment.first_frame, segment.first_frame_data
    if segment.join_mode == "continue":
        upstream = plan.segments[segment.index - 2]
        if fast_mode:
            # The upstream end anchor is already receipt-bound and fitted by
            # freeze_plan.  Reuse those exact immutable bytes; no generated
            # output is read and no duplicate anchor file is created.
            first, first_data = upstream.last_frame, upstream.last_frame_data
        else:
            tail = upstream.workdir / "work" / "generated_last.png"
            # A paid start belongs to the current parent attempt and must refresh
            # the dependency; resume may only reuse the already-frozen tail.
            if prepare_inputs:
                tail = _extract_last_frame(upstream.workdir / "generated.mp4", tail)
            elif not tail.is_file():
                raise LongGenerationError("long_video_tail_frame_missing")
            try:
                tail_data = tail.read_bytes()
            except OSError:
                raise LongGenerationError("long_video_tail_frame_missing") from None
            continued = segment.workdir / "work" / "h3_frames"
            if not plan.legacy_layout:
                continued = continued / plan.aspect_ratio.replace(":", "x")
            continued = continued / fit_mode / "continued"
            first, first_data = _fit_anchor(
                tail, tail_data, continued, fit_mode, plan.aspect_ratio,
                prepare=prepare_inputs,
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
        duration=long_video.provider_duration_s(
            segment.start_s,
            segment.end_s,
            receipt_version=plan.receipt_version,
        ),
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
        aspect_ratio=plan.aspect_ratio,
        resolution=plan.resolution,
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


def generation_segments_are_valid(
    expected_segments: object,
    generation: Mapping,
) -> bool:
    """Validate the complete ordered persisted coordinator state."""
    raw = generation.get("segments")
    if "fast_mode" in generation and not isinstance(generation.get("fast_mode"), bool):
        return False
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
    fast_mode = generation.get("fast_mode", False)
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
        status_can_revalidate = item.get("status") == "succeeded" or (
            item.get("status") == "failed"
            and item.get("error") == "long_video_segment_output_invalid"
        )
        if (
            not status_can_revalidate
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(child_id, str)
            or not child_id
            or (
                not fast_mode
                and
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
                fast_mode=fast_mode,
            )
            return h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
            )
        except (OSError, h3.H3Error, LongGenerationError, ValueError):
            return False

    for index in expected:
        if valid(index):
            reusable.add(index)
    return frozenset(reusable)


def initial_generation(settings, cid: str, plan: FrozenPlan, parent_id: str, attempt: int,
                       old: Mapping | None = None, *, fast_mode: bool = False) -> dict:
    if not isinstance(fast_mode, bool):
        raise LongGenerationError("invalid_fast_mode", 422)
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
    return {
        "status": "queued",
        "error": None,
        "attempt": attempt,
        "client_request_id": parent_id,
        "stage": "h3",
        "fit_layout": (
            FIT_LAYOUT_LEGACY if plan.legacy_layout else FIT_LAYOUT_ASPECT
        ),
        "fast_mode": fast_mode,
        "segments": items,
    }


def _stitch(settings, cid: str, plan: FrozenPlan, dialogue_mode: str) -> None:
    stitch.stitch_video(
        segments=_stitch_segments(plan),
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
        stitch_segments = _stitch_segments(plan)
        budgets = stitch._frame_budgets(stitch_segments)
        expected_segments = [
            {
                "index": index,
                "path": str((item.workdir / "generated.mp4").resolve()),
                "sha256": stitch._sha256(item.workdir / "generated.mp4"),
                "target_duration_s": stitch_segments[index - 1].target_duration_s,
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
        expected_duration = sum(
            item.target_duration_s for item in stitch_segments
        )
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
        LongGenerationError,
        stitch.StitchError,
    ):
        return False


def run(settings, cid: str, plan: FrozenPlan, *, startup: bool = False) -> None:
    """Drive frozen serial chains or fast fan-out; only this coordinator writes meta."""
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
    fast_mode = generation.get("fast_mode", False)
    fit_mode = meta.get("fit_mode")
    dialogue_mode = meta.get("dialogue_mode")
    states = {item["index"]: dict(item) for item in generation.get("segments", [])}
    if (
        not isinstance(parent_id, str)
        or fit_mode not in {"none", "crop", "pad"}
        or meta.get("aspect_ratio", h3.H3_DEFAULT_ASPECT_RATIO)
        != plan.aspect_ratio
        or meta.get("resolution", h3.H3_DEFAULT_RESOLUTION)
        != plan.resolution
    ):
        return

    def persist(status: str | None = None, error: str | None = None, stage: str = "h3") -> None:
        nonlocal generation
        ordered = [states[item.index] for item in plan.segments]
        generation = {**generation, "segments": ordered,
                      "status": status or generation.get("status"), "error": error, "stage": stage}
        storage.update_meta(settings.data_dir, cid, generation=generation)

    def parallel_update(segments, operation) -> None:
        with ThreadPoolExecutor(
            max_workers=min(_FAST_MODE_WORKERS, len(segments))
        ) as pool:
            futures = {pool.submit(operation, segment): segment for segment in segments}
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    segment = futures.pop(future)
                    status, error = future.result()
                    states[segment.index].update(status=status, error=error)
                    persist("running", None)

    reusable = bound_reusable_segment_indices(settings, cid, plan, generation)
    for index, state in states.items():
        if (
            index in reusable
            and state.get("status") == "failed"
            and state.get("error") == "long_video_segment_output_invalid"
        ):
            state.update(status="succeeded", error=None)
        elif state.get("status") == "succeeded" and index not in reusable:
            state.update(
                status="failed",
                error="long_video_segment_output_invalid",
            )

    if startup:
        recoverable = []
        for segment in plan.segments:
            state = states[segment.index]
            if state.get("status") not in {"queued", "running", "resume_required"}:
                continue
            recoverable.append(segment)

        def recover(segment: FrozenSegment):
            state = states[segment.index]
            try:
                request = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    prepare_inputs=False,
                    fast_mode=fast_mode,
                    frozen_child_id=state.get("child_request_id"),
                )
                result = h3.resume(request)
                if result.status == "not_started":
                    return (
                        ("queued", None)
                        if (
                            fast_mode
                            and state.get("status") == "queued"
                            and isinstance(result.attempt_id, str)
                            and bool(result.attempt_id)
                        )
                        else ("submission_unknown", "submission_unknown")
                    )
                status = _result_status(result)
                if (
                    fast_mode
                    and status[0] == "succeeded"
                    and not h3.output_is_reusable(
                        request,
                        expected_duration_s=_segment_duration_s(plan, segment),
                    )
                ):
                    return "failed", "long_video_segment_output_invalid"
                return status
            except Exception:
                return "submission_unknown", "submission_unknown"

        if recoverable and fast_mode:
            parallel_update(recoverable, recover)
        elif recoverable:
            for segment in recoverable:
                status, error = recover(segment)
                states[segment.index].update(status=status, error=error)
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

    if fast_mode:
        # Phase 1: construct every immutable request before creating any paid
        # attempt. A local validation failure therefore guarantees zero POSTs.
        requests: dict[int, h3.H3Request] = {}
        try:
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") == "succeeded":
                    continue
                if state.get("status") not in {
                    "not_started", "queued", "running", "resume_required",
                }:
                    continue
                child_id = state.get("child_request_id")
                if not isinstance(child_id, str) or not child_id:
                    child_id = child_request_id(parent_id, plan.receipt, segment.index)
                requests[segment.index] = _request(
                    settings, cid, plan, segment, parent_id, fit_mode,
                    frozen_child_id=child_id,
                    prepare_inputs=False,
                    fast_mode=True,
                )

            # Phase 2: persist every unpaid child receipt before the first POST.
            for segment in plan.segments:
                state = states[segment.index]
                if state.get("status") != "not_started":
                    continue
                result = h3.prepare(requests[segment.index])
                if result.status == "not_started":
                    prepared_status, prepared_error = "queued", None
                elif result.status == "h3_running":
                    prepared_status, prepared_error = "running", None
                elif result.status == "succeeded":
                    if not h3.output_is_reusable(
                        requests[segment.index],
                        expected_duration_s=_segment_duration_s(plan, segment),
                    ):
                        prepared_status, prepared_error = (
                            "failed", "long_video_segment_output_invalid"
                        )
                    else:
                        prepared_status, prepared_error = "succeeded", None
                else:
                    prepared_status, prepared_error = _result_status(result)
                state.update(
                    status=prepared_status,
                    attempt=int(state.get("attempt", 0) or 0) + 1,
                    error=prepared_error,
                    child_request_id=requests[segment.index].client_request_id,
                )
            persist("running", None)
        except (h3.H3Error, LongGenerationError, OSError, ValueError) as exc:
            code = exc.code if isinstance(exc, (h3.H3Error, LongGenerationError)) else "long_video_request_invalid"
            failed_index = next(
                (
                    segment.index for segment in plan.segments
                    if states[segment.index].get("status") == "not_started"
                ),
                None,
            )
            if failed_index is not None:
                states[failed_index].update(status="failed", error=code)
            persist("failed", "long_video_segment_failed")
            return

        prepared_statuses = {item.get("status") for item in states.values()}
        if "submission_unknown" in prepared_statuses:
            persist("submission_unknown", "submission_unknown")
            return
        if "failed" in prepared_statuses:
            persist("failed", "long_video_segment_failed")
            return

        def submit_one(segment: FrozenSegment):
            try:
                result = h3.submit(requests[segment.index])
                if result.status == "h3_running":
                    return "running", None
                status = _result_status(result)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    requests[segment.index],
                    expected_duration_s=_segment_duration_s(plan, segment),
                ):
                    return "failed", "long_video_segment_output_invalid"
                return status
            except h3.H3Error as exc:
                if exc.code == "attempt_not_prepared":
                    return "submission_unknown", "submission_unknown"
                try:
                    inspected = h3.inspect(requests[segment.index])
                    status = _result_status(inspected)
                except Exception:
                    status = ("submission_unknown", "submission_unknown")
                if status[0] == "failed" and exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                }:
                    return "submission_unknown", "submission_unknown"
                return status
            except Exception:
                return "submission_unknown", "submission_unknown"

        # Phase 3: fan out only the short POST boundary. No worker waits for a
        # provider result, so every queued child is submitted before polling.
        to_submit = [
            segment for segment in plan.segments
            if states[segment.index].get("status") == "queued"
        ]
        if to_submit:
            parallel_update(to_submit, submit_one)

        def poll_one(segment: FrozenSegment):
            request = requests[segment.index]
            try:
                result = h3.resume(request)
                if result.status == "not_started":
                    return "submission_unknown", "submission_unknown"
                status = _result_status(result)
                if status[0] == "succeeded" and not h3.output_is_reusable(
                    request,
                    expected_duration_s=_segment_duration_s(plan, segment),
                ):
                    return "failed", "long_video_segment_output_invalid"
                return status
            except h3.H3Error as exc:
                return (
                    "submission_unknown", "submission_unknown"
                ) if exc.code in {
                    "submission_unknown", "state_persist_failed", "h3_internal_error",
                } else ("resume_required", exc.code)
            except Exception:
                return "submission_unknown", "submission_unknown"

        # Phase 4: bounded long-lived GET polling. Unknown children never get a
        # second POST, while known siblings are still allowed to finish.
        to_poll = [
            segment for segment in plan.segments
            if states[segment.index].get("status") in {"running", "resume_required"}
        ]
        if to_poll:
            parallel_update(to_poll, poll_one)

        statuses = {item.get("status") for item in states.values()}
        if statuses == {"succeeded"}:
            try:
                _stitch(settings, cid, plan, dialogue_mode)
            except Exception:
                persist("failed", "long_video_stitch_failed", "stitch")
            else:
                persist("succeeded", None, "stitch")
        elif "submission_unknown" in statuses:
            persist("submission_unknown", "submission_unknown")
        elif "resume_required" in statuses or "running" in statuses:
            persist("resume_required", "long_video_resume_required")
        else:
            persist("failed", "long_video_segment_failed")
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
        if (
            action == "start"
            and long_video.provider_duration_s(
                segment.start_s,
                segment.end_s,
                receipt_version=plan.receipt_version,
            )
            > long_video.SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            return None, ("failed", "long_video_legacy_plan_read_only")
        existing_child_id = states[segment.index].get("child_request_id")
        try:
            request = _request(
                settings, cid, plan, segment, parent_id, fit_mode,
                prepare_inputs=action == "start",
            )
        except LongGenerationError as exc:
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown"
                )
            return None, ("failed", exc.code)
        except Exception:
            if action == "resume":
                return existing_child_id, (
                    "submission_unknown", "submission_unknown"
                )
            return None, ("failed", "long_video_request_invalid")
        try:
            result = h3.start(request) if action == "start" else h3.resume(request)
            if action == "resume" and result.status == "not_started":
                return request.client_request_id, (
                    "submission_unknown", "submission_unknown"
                )
            status = _result_status(result)
            if status[0] == "succeeded" and not h3.output_is_reusable(
                request,
                expected_duration_s=_segment_duration_s(plan, segment),
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
