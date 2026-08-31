"""Deterministic long-video planning and immutable plan receipts.

This module has no provider calls.  It turns server-owned facts into a plan that
later submission code can verify with a compare-and-swap receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app import h3, h3_project

SHORT_VIDEO_MAX_S = 10.0
PREVIOUS_SHORT_VIDEO_MAX_S = 10.0
LONG_VIDEO_MAX_S = 300.0
RECEIPT_COMPAT_SEGMENT_MIN_S = 1.0
# Historical public name retained for receipt readers outside this module.
SEGMENT_MIN_S = RECEIPT_COMPAT_SEGMENT_MIN_S
SEGMENT_PROVIDER_MIN_DURATION_S = 4
SEGMENT_PROVIDER_MAX_DURATION_S = 10
SEGMENT_SOURCE_MIN_S = float(SEGMENT_PROVIDER_MIN_DURATION_S)
PREVIOUS_SEGMENT_PROVIDER_MAX_DURATION_S = 10
LEGACY_PROVIDER_MAX_DURATION_S = 15
BOUNDARY_PRECISION = 6
PLAN_RECEIPT_FILENAME = "long_video_plan.json"
PLAN_RECEIPT_VERSION = 2
MULTIMODAL_PLAN_RECEIPT_VERSION = 3
VISUAL_PLAN_RECEIPT_VERSION = 4
VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION = 5
LEGACY_PLAN_RECEIPT_VERSION = 1

_EPS = 1e-6
_CONTINUITY_BLOCK = """【全局连续性（所有分段必须逐字遵守）】
- 同一 chain 内保持人物身份、脸部、服装、道具、场景布局、光线、色调和运动方向连续。
- continue 段承接上一段的末帧状态；hard_cut 段按当前源片段独立建立画面。
- 本段局部动作以本段视觉描述为准，不得被全局连续性要求抹除。
- 画面文字、OCR、字幕和备注只属于视觉元素，不得推断、扩写或升级为角色台词。
【全局连续性结束】"""


class LongVideoError(RuntimeError):
    """Stable fail-closed error consumed by the orchestration layer."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _finite_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongVideoError("long_video_invalid_duration")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise LongVideoError("long_video_invalid_duration")
    if duration > LONG_VIDEO_MAX_S:
        raise LongVideoError("long_video_duration_exceeded")
    return duration


def segment_duration_s(
    start_s: object,
    end_s: object,
    *,
    receipt_version: int = PLAN_RECEIPT_VERSION,
) -> float:
    """Return segment length using the receipt's frozen-boundary precision.

    Current segment boundaries are persisted with six decimal places.
    Normalize both endpoints and their difference before applying any duration
    constraint.  Receipt v1 preserves its historical raw-float calculation
    solely to reconstruct already-paid H3 attempts byte-for-byte.
    """
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise LongVideoError("long_video_invalid_segment_duration")
    if (
        isinstance(receipt_version, bool)
        or not isinstance(receipt_version, int)
        or receipt_version not in {
            LEGACY_PLAN_RECEIPT_VERSION,
            PLAN_RECEIPT_VERSION,
            MULTIMODAL_PLAN_RECEIPT_VERSION,
            VISUAL_PLAN_RECEIPT_VERSION,
            VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION,
        }
    ):
        raise LongVideoError("long_video_invalid_segment_duration")
    if receipt_version == LEGACY_PLAN_RECEIPT_VERSION:
        start = float(start_s)
        end = float(end_s)
        duration = end - start
    else:
        start = round(float(start_s), BOUNDARY_PRECISION)
        end = round(float(end_s), BOUNDARY_PRECISION)
        duration = round(end - start, BOUNDARY_PRECISION)
    if not (math.isfinite(start) and math.isfinite(end) and duration > 0):
        raise LongVideoError("long_video_invalid_segment_duration")
    return duration


def _provider_duration_for_duration(duration: float) -> int:
    return max(SEGMENT_PROVIDER_MIN_DURATION_S, math.ceil(duration))


def provider_duration_s(
    start_s: object,
    end_s: object,
    *,
    receipt_version: int = PLAN_RECEIPT_VERSION,
) -> int:
    """Return the provider integer duration for frozen segment boundaries.

    The shared frozen-boundary calculation keeps a binary artifact such as
    ``47.52 - 37.52 == 10.000000000000004`` at ten seconds, while a real
    10.000001-second segment remains eleven seconds.
    """
    duration = segment_duration_s(
        start_s,
        end_s,
        receipt_version=receipt_version,
    )
    return _provider_duration_for_duration(duration)


def _bounds(
    scenes: Iterable[Mapping | Sequence[float]], duration_s: float
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for raw in scenes:
        try:
            if isinstance(raw, Mapping):
                start, end = raw["start_s"], raw["end_s"]
            else:
                start, end = raw[0], raw[1]
            start, end = float(start), float(end)
        except (KeyError, IndexError, TypeError, ValueError):
            raise LongVideoError("long_video_invalid_scenes") from None
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise LongVideoError("long_video_invalid_scenes")
        result.append((start, end))
    if not result:
        return [(0.0, duration_s)]
    if abs(result[0][0]) > _EPS or abs(result[-1][1] - duration_s) > _EPS:
        raise LongVideoError("long_video_invalid_scenes")
    previous = 0.0
    for start, end in result:
        if abs(start - previous) > _EPS:
            raise LongVideoError("long_video_invalid_scenes")
        previous = end
    return result


def _dialogue_intervals(lines: Iterable[Mapping], duration_s: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    previous_start = -1.0
    for line in lines:
        try:
            start, end = float(line["start_s"]), float(line["end_s"])
        except (KeyError, TypeError, ValueError):
            raise LongVideoError("long_video_invalid_dialogue") from None
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or start >= end
            or end > duration_s + _EPS
            or start < previous_start
        ):
            raise LongVideoError("long_video_invalid_dialogue")
        intervals.append((start, min(end, duration_s)))
        previous_start = start
    return intervals


def plan_segments(
    duration_s: float,
    scenes: Iterable[Mapping | Sequence[float]],
    dialogue: Iterable[Mapping],
) -> list[dict]:
    """Plan provider-safe segments with hard cuts preferred over timed splits.

    Current projects always return at least one segment.  A short video is the
    one-element form of the same plan consumed by longer projects.
    """
    duration = _finite_duration(duration_s)
    if duration <= SHORT_VIDEO_MAX_S:
        _dialogue_intervals(dialogue, duration)
        return [{
            "index": 1,
            "start_s": 0.0,
            "end_s": round(duration, BOUNDARY_PRECISION),
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
        }]
    scene_bounds = _bounds(scenes, duration)
    _dialogue_intervals(dialogue, duration)
    hard_cuts = {end for _start, end in scene_bounds[:-1]}

    cuts = [0.0]
    while provider_duration_s(cuts[-1], duration) > SEGMENT_PROVIDER_MAX_DURATION_S:
        start = cuts[-1]
        target = min(
            round(start + SEGMENT_PROVIDER_MAX_DURATION_S, BOUNDARY_PRECISION),
            duration - SEGMENT_SOURCE_MIN_S,
        )
        minimum = start + SEGMENT_SOURCE_MIN_S
        eligible_hard_cuts = [
            cut
            for cut in hard_cuts
            if minimum <= cut <= target
            and segment_duration_s(cut, duration) >= SEGMENT_SOURCE_MIN_S
            and provider_duration_s(start, cut) <= SEGMENT_PROVIDER_MAX_DURATION_S
        ]
        boundary = max(eligible_hard_cuts) if eligible_hard_cuts else target
        if boundary <= start + _EPS:
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        cuts.append(round(boundary, 6))
    cuts.append(duration)

    # A preferred hard cut must never leave a source tail shorter than the
    # provider's real minimum.  Pull the boundary earlier; the source timeline
    # remains complete and the provider receives no invented duration.
    if segment_duration_s(cuts[-2], cuts[-1]) < SEGMENT_SOURCE_MIN_S:
        cuts.pop(-2)
        start = cuts[-2]
        target = duration - SEGMENT_SOURCE_MIN_S
        boundary = target
        if (
            provider_duration_s(boundary, duration)
            > SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        cuts.insert(-1, round(boundary, 6))

    segments: list[dict] = []
    chain_number = 1
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1):
        if (
            segment_duration_s(start, end) < SEGMENT_SOURCE_MIN_S
            or provider_duration_s(start, end) > SEGMENT_PROVIDER_MAX_DURATION_S
        ):
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        hard_cut = index == 1 or any(abs(start - cut) <= _EPS for cut in hard_cuts)
        if index > 1 and hard_cut:
            chain_number += 1
        segments.append(
            {
                "index": index,
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "chain_id": f"chain-{chain_number:03d}",
                "join_mode": "hard_cut" if hard_cut else "continue",
            }
        )
    return segments


def localize_dialogue(
    lines: Iterable[Mapping],
    segment: Mapping,
    *,
    segments: Sequence[Mapping],
) -> list[dict]:
    """Split dialogue text by timed overlap and shift it to segment-local time.

    A boundary-spanning line is partitioned into consecutive Unicode codepoint
    ranges proportional to cumulative overlap duration.  Concatenating the
    fragments in segment order reproduces the original text exactly, while
    dialogue never changes the visual cut plan.
    """
    try:
        current_index = int(segment["index"])
        normalized_segments = [
            (
                int(item["index"]),
                float(item["start_s"]),
                float(item["end_s"]),
            )
            for item in segments
        ]
    except (KeyError, TypeError, ValueError):
        raise LongVideoError("long_video_invalid_dialogue") from None
    if (
        not normalized_segments
        or any(
            not (math.isfinite(start) and math.isfinite(end) and start < end)
            for _index, start, end in normalized_segments
        )
        or [index for index, _start, _end in normalized_segments]
        != list(range(1, len(normalized_segments) + 1))
        or abs(normalized_segments[0][1]) > _EPS
        or any(
            abs(left[2] - right[1]) > _EPS
            for left, right in zip(normalized_segments, normalized_segments[1:])
        )
    ):
        raise LongVideoError("long_video_invalid_dialogue")
    try:
        current_index, start, end = next(
            item for item in normalized_segments if item[0] == current_index
        )
    except StopIteration:
        raise LongVideoError("long_video_invalid_dialogue") from None
    local: list[dict] = []
    for raw in lines:
        try:
            line_start = float(raw["start_s"])
            line_end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError):
            raise LongVideoError("long_video_invalid_dialogue") from None
        if not (
            math.isfinite(line_start)
            and math.isfinite(line_end)
            and 0 <= line_start < line_end
            and line_end <= normalized_segments[-1][2] + _EPS
            and isinstance(raw.get("text"), str)
            and raw["text"]
        ):
            raise LongVideoError("long_video_invalid_dialogue")
        intersections = [
            (
                item_index,
                max(line_start, item_start),
                min(line_end, item_end),
            )
            for item_index, item_start, item_end in normalized_segments
            if min(line_end, item_end) > max(line_start, item_start)
        ]
        if not intersections:
            raise LongVideoError("long_video_invalid_dialogue")
        total_overlap = sum(
            overlap_end - overlap_start
            for _index, overlap_start, overlap_end in intersections
        )
        text = raw["text"]
        codepoint_start = 0
        cumulative_overlap = 0.0
        for position, (owner_index, overlap_start, overlap_end) in enumerate(
            intersections, 1
        ):
            cumulative_overlap += overlap_end - overlap_start
            codepoint_end = (
                len(text)
                if position == len(intersections)
                else min(
                    len(text),
                    max(
                        codepoint_start,
                        int(math.floor(
                            len(text) * cumulative_overlap / total_overlap + 0.5
                        )),
                    ),
                )
            )
            if owner_index != current_index or codepoint_end == codepoint_start:
                codepoint_start = codepoint_end
                continue
            item = dict(raw)
            item["text"] = text[codepoint_start:codepoint_end]
            item["start_s"] = round(overlap_start - start, BOUNDARY_PRECISION)
            item["end_s"] = round(overlap_end - start, BOUNDARY_PRECISION)
            if not 0 <= item["start_s"] < item["end_s"] <= end - start + _EPS:
                raise LongVideoError("long_video_invalid_dialogue")
            local.append(item)
            codepoint_start = codepoint_end
    return local


def localize_keyframe_sources(
    value: object,
    *,
    segment_start_s: object,
    segment_end_s: object,
    provider_duration_s: object,
) -> tuple[list[dict], list[dict]]:
    """Project nine frozen global frame receipts onto one H3-local timeline.

    Paths, hashes and source-scene ids remain receipt authorities.  Timing and
    transitions are a deterministic backend projection: source coordinates are
    rounded to the plan precision, clamped to the segment, made strictly
    increasing, then shifted by the frozen segment start.  Recoverable timing
    anomalies never become a quality rejection; every normalization is returned
    as an ordered diagnostic for the coordinator to expose or persist.
    """
    if not isinstance(value, list) or len(value) != 9:
        raise LongVideoError("long_video_plan_invalid_keyframe_sources")
    try:
        duration = segment_duration_s(segment_start_s, segment_end_s)
        start_s = round(float(segment_start_s), BOUNDARY_PRECISION)
        end_s = round(float(segment_end_s), BOUNDARY_PRECISION)
    except (TypeError, ValueError, LongVideoError):
        raise LongVideoError("long_video_invalid_segment_duration") from None

    diagnostics: list[dict] = []
    canonical_provider_duration = _provider_duration_for_duration(duration)
    if not (
        isinstance(provider_duration_s, int)
        and not isinstance(provider_duration_s, bool)
        and provider_duration_s == canonical_provider_duration
    ):
        diagnostics.append({
            "order": 0,
            "code": "provider_duration_normalized",
            "from": provider_duration_s,
            "to": canonical_provider_duration,
        })

    quantum = 10 ** -BOUNDARY_PRECISION
    if duration < quantum * 8:
        raise LongVideoError("long_video_invalid_segment_duration")
    required = {
        "order", "path", "sha256", "source_time_s",
        "source_scene_id", "transition",
    }
    localized: list[dict] = []
    for order, raw in enumerate(value, 1):
        if not isinstance(raw, Mapping) or not required.issubset(raw):
            raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        path, digest = raw.get("path"), raw.get("sha256")
        scene_id = raw.get("source_scene_id")
        if (
            not isinstance(path, str) or not path
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(scene_id, str) or not scene_id.strip()
        ):
            raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        if raw.get("order") != order:
            diagnostics.append({
                "order": order, "code": "keyframe_order_normalized",
                "from": raw.get("order"), "to": order,
            })

        source_value = raw.get("source_time_s")
        if (
            isinstance(source_value, bool)
            or not isinstance(source_value, (int, float))
            or not math.isfinite(float(source_value))
        ):
            source_time = round(
                start_s + duration * (order - 1) / 8, BOUNDARY_PRECISION
            )
            diagnostics.append({
                "order": order, "code": "source_time_reconstructed",
                "from": None, "to": source_time,
            })
        else:
            original = round(float(source_value), BOUNDARY_PRECISION)
            source_time = min(end_s, max(start_s, original))
            if source_time != original:
                diagnostics.append({
                    "order": order, "code": "source_time_clamped",
                    "from": original, "to": source_time,
                })

        raw_local = round(source_time - start_s, BOUNDARY_PRECISION)
        previous = localized[-1] if localized else None
        if previous is None:
            # The first selected frame is the segment's visual origin even
            # when its decoded PTS lands after the frozen cut.
            # Keep the global source receipt intact and normalize only the
            # provider-facing segment-local coordinate.
            local_time = 0.0
        else:
            lower = round(
                previous["segment_time_s"] + quantum, BOUNDARY_PRECISION
            )
            upper = round(
                duration - quantum * (9 - order), BOUNDARY_PRECISION
            )
            local_time = round(
                min(upper, max(lower, raw_local)), BOUNDARY_PRECISION
            )
        local_time = 0.0 if local_time == -0.0 else local_time
        if local_time != raw_local:
            diagnostics.append({
                "order": order,
                "code": (
                    "segment_origin_normalized"
                    if previous is None
                    else "source_time_order_normalized"
                ),
                "from": raw_local, "to": local_time,
            })

        source_transition = raw.get("transition")
        transition = source_transition if isinstance(source_transition, Mapping) else {}
        if previous is None:
            local_transition = {"type": "start", "at_segment_s": local_time}
        else:
            raw_type = transition.get("type")
            local_type = (
                raw_type
                if raw_type in {"continuous", "hard_cut"}
                else (
                    "hard_cut"
                    if scene_id != previous["source_scene_id"]
                    else "continuous"
                )
            )
            if raw_type != local_type:
                diagnostics.append({
                    "order": order, "code": "transition_type_normalized",
                    "from": raw_type, "to": local_type,
                })
            raw_at_s = transition.get("at_s")
            if local_type == "continuous":
                if raw_at_s is not None:
                    diagnostics.append({
                        "order": order, "code": "continuous_cut_removed",
                        "from": raw_at_s, "to": None,
                    })
                local_transition = {
                    "type": "continuous", "at_segment_s": None,
                }
            else:
                raw_at_local = (
                    round(float(raw_at_s) - start_s, BOUNDARY_PRECISION)
                    if not isinstance(raw_at_s, bool)
                    and isinstance(raw_at_s, (int, float))
                    and math.isfinite(float(raw_at_s))
                    else None
                )
                at_local = (
                    raw_at_local
                    if raw_at_local is not None
                    and previous["segment_time_s"] < raw_at_local <= local_time
                    else local_time
                )
                if raw_at_local != at_local:
                    diagnostics.append({
                        "order": order, "code": "hard_cut_time_normalized",
                        "from": raw_at_local, "to": at_local,
                    })
                local_transition = {
                    "type": "hard_cut", "at_segment_s": at_local,
                }
        localized.append({
            "order": order,
            "path": path,
            "sha256": digest,
            "segment_time_s": local_time,
            "source_scene_id": scene_id,
            "transition": local_transition,
        })
    return localized, diagnostics


def build_continuity_block() -> str:
    """Return the server-owned block injected byte-for-byte into every segment."""
    return _CONTINUITY_BLOCK


def compose_segment_visual_prompt(local_prompt: str, continuity_block: str | None = None) -> str:
    if not isinstance(local_prompt, str) or not local_prompt.strip():
        raise LongVideoError("long_video_invalid_visual_prompt")
    block = _CONTINUITY_BLOCK if continuity_block is None else continuity_block
    if block != _CONTINUITY_BLOCK:
        raise LongVideoError("long_video_invalid_continuity_block")
    return f"{block}\n\n【本段局部动作】\n{local_prompt.strip()}\n"


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise LongVideoError("long_video_plan_not_canonical") from None
    return (text + "\n").encode("utf-8")


def _artifact(root: Path, path: Path) -> dict:
    root = root.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        raise LongVideoError("long_video_plan_artifact_outside_root") from None
    try:
        data = resolved.read_bytes()
    except OSError:
        raise LongVideoError("long_video_plan_artifact_missing") from None
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}


def freeze_keyframe_sources(
    value: object,
    *,
    expected_count: int,
    previous: Mapping | None = None,
) -> tuple[list[dict], dict]:
    """Validate one ordered source timeline without guessing visual facts."""
    if (
        expected_count != 9
        or not isinstance(value, list)
        or len(value) != expected_count
    ):
        raise LongVideoError("long_video_plan_invalid_keyframe_sources")
    frozen: list[dict] = []
    prior = dict(previous) if isinstance(previous, Mapping) else None
    for order, raw in enumerate(value, 1):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {
                "order", "source_time_s", "source_scene_id", "transition",
            }
            or raw.get("order") != order
            or isinstance(raw.get("source_time_s"), bool)
            or not isinstance(raw.get("source_time_s"), (int, float))
            or not math.isfinite(float(raw["source_time_s"]))
            or float(raw["source_time_s"]) < 0
            or not isinstance(raw.get("source_scene_id"), str)
            or not raw["source_scene_id"].strip()
            or not isinstance(raw.get("transition"), Mapping)
            or set(raw["transition"]) != {"type", "at_s"}
        ):
            raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        source_time_s = round(
            float(raw["source_time_s"]), BOUNDARY_PRECISION
        )
        transition_type = raw["transition"].get("type")
        at_s = raw["transition"].get("at_s")
        if transition_type not in {"start", "continuous", "hard_cut"}:
            raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        if prior is None:
            if transition_type != "start" or at_s != source_time_s:
                raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        else:
            previous_time_s = float(prior["source_time_s"])
            previous_scene_id = prior["source_scene_id"]
            # The scene sampler may deliberately repeat the nearest decoded
            # source frame when a valid segment contains fewer than nine
            # distinct frames.  Equal PTS within the same continuous scene is
            # therefore a canonical receipt, not a reason to stop A -> B.
            if source_time_s < previous_time_s or transition_type == "start":
                raise LongVideoError("long_video_plan_invalid_keyframe_sources")
            if transition_type == "hard_cut":
                if (
                    isinstance(at_s, bool)
                    or not isinstance(at_s, (int, float))
                    or not math.isfinite(float(at_s))
                ):
                    raise LongVideoError(
                        "long_video_plan_invalid_keyframe_sources"
                    )
                at_s = round(float(at_s), BOUNDARY_PRECISION)
                if not previous_time_s < at_s <= source_time_s:
                    raise LongVideoError(
                        "long_video_plan_invalid_keyframe_sources"
                    )
                if raw["source_scene_id"] == previous_scene_id:
                    raise LongVideoError(
                        "long_video_plan_invalid_keyframe_sources"
                    )
            elif at_s is not None or raw["source_scene_id"] != previous_scene_id:
                raise LongVideoError("long_video_plan_invalid_keyframe_sources")
        item = {
            "order": order,
            "source_time_s": source_time_s,
            "source_scene_id": raw["source_scene_id"],
            "transition": {"type": transition_type, "at_s": at_s},
        }
        frozen.append(item)
        prior = item
    assert prior is not None
    return frozen, prior


def freeze_source_cut_timeline(
    value: object, *, segment_start_s: object, segment_end_s: object,
) -> list[dict]:
    """Freeze complete global source cuts without conflating them with frames."""
    try:
        segment_start = round(float(segment_start_s), BOUNDARY_PRECISION)
        segment_end = round(float(segment_end_s), BOUNDARY_PRECISION)
    except (TypeError, ValueError):
        raise LongVideoError("long_video_plan_invalid_cut_timeline") from None
    if not isinstance(value, list) or not value:
        raise LongVideoError("long_video_plan_invalid_cut_timeline")
    frozen = []
    previous_end = segment_start
    previous_scene_id = None
    for order, raw in enumerate(value, 1):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"order", "start_s", "end_s", "source_scene_id"}
            or raw.get("order") != order
            or isinstance(raw.get("start_s"), bool)
            or isinstance(raw.get("end_s"), bool)
            or not isinstance(raw.get("start_s"), (int, float))
            or not isinstance(raw.get("end_s"), (int, float))
            or not isinstance(raw.get("source_scene_id"), str)
            or not raw["source_scene_id"].strip()
        ):
            raise LongVideoError("long_video_plan_invalid_cut_timeline")
        start = round(float(raw["start_s"]), BOUNDARY_PRECISION)
        end = round(float(raw["end_s"]), BOUNDARY_PRECISION)
        if (
            not math.isfinite(start) or not math.isfinite(end)
            or start != previous_end or end <= start
            or raw["source_scene_id"] == previous_scene_id
        ):
            raise LongVideoError("long_video_plan_invalid_cut_timeline")
        frozen.append({
            "order": order, "start_s": start, "end_s": end,
            "source_scene_id": raw["source_scene_id"],
        })
        previous_end = end
        previous_scene_id = raw["source_scene_id"]
    if previous_end != segment_end:
        raise LongVideoError("long_video_plan_invalid_cut_timeline")
    return frozen


def _write_plan_receipt(
    root: Path,
    *,
    source: Path,
    duration_s: float,
    segments: Sequence[Mapping],
    workflow: str,
    dialogue_mode: str = "auto",
    dialogue_delivery: str | None = None,
    resolved_dialogue_delivery: str | None = None,
    prompt_fusion_manifest_path: Path | None = None,
    minimum_source_duration_s: float,
    provider_max_duration_s: int,
) -> Path:
    """Write a canonical receipt binding the complete generated long-video plan."""
    root = root.resolve()
    duration = _finite_duration(duration_s)
    if not segments:
        raise LongVideoError("long_video_plan_requires_segments")
    if not isinstance(workflow, str) or not workflow.strip():
        raise LongVideoError("long_video_plan_invalid_workflow")
    has_multimodal = ["multimodal_manifest_path" in raw for raw in segments]
    if any(has_multimodal) and not all(has_multimodal):
        raise LongVideoError("long_video_multimodal_incomplete")
    has_prompt_fusion = prompt_fusion_manifest_path is not None
    if has_prompt_fusion and any(has_multimodal):
        raise LongVideoError("long_video_multimodal_ambiguous")
    receipt_segments = []
    previous_end = 0.0
    previous_keyframe_source: dict | None = None
    has_keyframe_sources = ["keyframe_sources" in raw for raw in segments]
    if any(has_keyframe_sources) and not all(has_keyframe_sources):
        raise LongVideoError("long_video_plan_invalid_keyframe_sources")
    for expected_index, raw in enumerate(segments, start=1):
        try:
            index = int(raw["index"])
            start_s = float(raw["start_s"])
            end_s = float(raw["end_s"])
            chain_id = raw["chain_id"]
            join_mode = raw["join_mode"]
        except (KeyError, TypeError, ValueError):
            raise LongVideoError("long_video_plan_invalid_segment") from None
        try:
            frozen_duration = segment_duration_s(start_s, end_s)
        except LongVideoError:
            raise LongVideoError("long_video_plan_invalid_segment") from None
        if (
            index != expected_index
            or not (math.isfinite(start_s) and math.isfinite(end_s))
            or abs(start_s - previous_end) > _EPS
            or frozen_duration < minimum_source_duration_s
            or provider_duration_s(start_s, end_s)
            > provider_max_duration_s
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
        ):
            raise LongVideoError("long_video_plan_invalid_segment")
        keyframe_paths = list(raw.get("keyframe_paths", []))
        if not 1 <= len(keyframe_paths) <= 9:
            raise LongVideoError("long_video_plan_invalid_keyframes")
        try:
            first_frame_path = Path(raw["first_frame_path"])
            last_frame_path = Path(raw["last_frame_path"])
        except (KeyError, TypeError):
            raise LongVideoError("long_video_plan_invalid_anchors") from None
        dialogue = list(raw.get("dialogue", []))
        receipt_segment = {
                "index": index,
                "start_s": start_s,
                "end_s": end_s,
                "chain_id": chain_id,
                "join_mode": join_mode,
                "source": _artifact(root, Path(raw["source_path"])),
                "keyframes": [
                    _artifact(root, Path(path)) for path in keyframe_paths
                ],
                "anchors": [
                    {"role": "first", **_artifact(root, first_frame_path)},
                    {"role": "end", **_artifact(root, last_frame_path)},
                ],
                "visual_prompt": _artifact(root, Path(raw["visual_prompt_path"])),
                "final_prompt": _artifact(root, Path(raw["final_prompt_path"])),
                "dialogue": {
                    "count": len(dialogue),
                    "sha256": hashlib.sha256(_canonical_bytes(dialogue)).hexdigest(),
                },
            }
        if all(has_keyframe_sources):
            keyframe_sources, previous_keyframe_source = freeze_keyframe_sources(
                raw.get("keyframe_sources"),
                expected_count=len(keyframe_paths),
                previous=previous_keyframe_source,
            )
            receipt_segment["keyframe_sources"] = keyframe_sources
        if raw.get("source_cut_timeline") is not None:
            receipt_segment["source_cut_timeline"] = freeze_source_cut_timeline(
                raw["source_cut_timeline"],
                segment_start_s=start_s,
                segment_end_s=end_s,
            )
        if all(has_multimodal):
            try:
                manifest_path = Path(raw["multimodal_manifest_path"]).resolve()
                frozen_multimodal = h3_project.freeze_optional(
                    root, manifest_path.parent
                )
                if (
                    frozen_multimodal is None
                    or frozen_multimodal.manifest_path != manifest_path
                    or workflow.strip() != h3.H3_MULTIMODAL_WORKFLOW
                ):
                    raise h3_project.ProjectMultimodalError(
                        "multimodal_source_invalid"
                    )
                receipt_segment["multimodal"] = h3_project.receipt_binding(
                    root, frozen_multimodal
                )
            except (KeyError, h3_project.ProjectMultimodalError) as exc:
                code = getattr(exc, "code", "multimodal_source_invalid")
                raise LongVideoError(code) from None
        receipt_segments.append(receipt_segment)
        previous_end = end_s
    if abs(previous_end - duration) > _EPS:
        raise LongVideoError("long_video_plan_invalid_segment")
    receipt = {
        "schema": "duet.long-video-plan",
        "version": (
            (
                VISUAL_MULTIMODAL_PLAN_RECEIPT_VERSION
                if all(has_keyframe_sources)
                else MULTIMODAL_PLAN_RECEIPT_VERSION
            )
            if all(has_multimodal) or has_prompt_fusion
            else (
                VISUAL_PLAN_RECEIPT_VERSION
                if all(has_keyframe_sources)
                else PLAN_RECEIPT_VERSION
            )
        ),
        "source": _artifact(root, source),
        "video": {"duration_s": duration},
        "workflow": workflow.strip(),
        "segments": receipt_segments,
    }
    if all(has_multimodal) or has_prompt_fusion:
        if dialogue_mode not in {"auto", "edit", "custom", "none"}:
            raise LongVideoError("long_video_plan_invalid_dialogue_mode")
        receipt["dialogue_mode"] = dialogue_mode
        if (dialogue_delivery is None) != (resolved_dialogue_delivery is None):
            raise LongVideoError("long_video_plan_invalid_dialogue_delivery")
        if dialogue_delivery is not None:
            if dialogue_delivery not in {"auto", "on_screen", "off_screen"}:
                raise LongVideoError("long_video_plan_invalid_dialogue_delivery")
            if resolved_dialogue_delivery not in {"on_screen", "off_screen"}:
                raise LongVideoError("long_video_plan_invalid_dialogue_delivery")
            receipt["dialogue_delivery"] = dialogue_delivery
            receipt["resolved_dialogue_delivery"] = resolved_dialogue_delivery
        if has_prompt_fusion:
            receipt["prompt_fusion"] = _artifact(
                root, prompt_fusion_manifest_path
            )
    path = root / PLAN_RECEIPT_FILENAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(receipt))
    temporary.replace(path)
    return path


def write_plan_receipt(
    root: Path,
    *,
    source: Path,
    duration_s: float,
    segments: Sequence[Mapping],
    workflow: str,
    dialogue_mode: str = "auto",
    dialogue_delivery: str | None = None,
    resolved_dialogue_delivery: str | None = None,
    prompt_fusion_manifest_path: Path | None = None,
) -> Path:
    """Write a canonical plan while keeping a short source timeline exact."""
    return _write_plan_receipt(
        root,
        source=source,
        duration_s=duration_s,
        segments=segments,
        workflow=workflow,
        dialogue_mode=dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        resolved_dialogue_delivery=resolved_dialogue_delivery,
        prompt_fusion_manifest_path=prompt_fusion_manifest_path,
        minimum_source_duration_s=min(
            _finite_duration(duration_s), SEGMENT_SOURCE_MIN_S,
        ),
        provider_max_duration_s=SEGMENT_PROVIDER_MAX_DURATION_S,
    )


def _write_frozen_v4_n1_plan_receipt(
    root: Path,
    *,
    source: Path,
    duration_s: float,
    segments: Sequence[Mapping],
    workflow: str,
    dialogue_mode: str = "auto",
    dialogue_delivery: str | None = None,
    resolved_dialogue_delivery: str | None = None,
    prompt_fusion_manifest_path: Path | None = None,
) -> Path:
    """Migrate one authority-validated pre-unification v4 N=1 plan.

    The orchestration caller must first validate the frozen private v4
    postprocess authority.  This private seam deliberately has no configurable
    limit and cannot write a general legacy multi-segment or non-H3 plan.
    """
    duration = _finite_duration(duration_s)
    if (
        not SHORT_VIDEO_MAX_S < duration <= LEGACY_PROVIDER_MAX_DURATION_S
        or workflow != h3.H3_WORKFLOW
        or len(segments) != 1
    ):
        raise LongVideoError("long_video_plan_invalid_segment")
    raw = segments[0]
    try:
        index = int(raw["index"])
        start_s = float(raw["start_s"])
        end_s = float(raw["end_s"])
        keyframe_paths = list(raw["keyframe_paths"])
        keyframe_sources = list(raw["keyframe_sources"])
    except (KeyError, TypeError, ValueError):
        raise LongVideoError("long_video_plan_invalid_segment") from None
    if (
        index != 1
        or start_s != 0.0
        or abs(end_s - duration) > _EPS
        or len(keyframe_paths) != 9
        or len(keyframe_sources) != 9
        or "multimodal_manifest_path" in raw
    ):
        raise LongVideoError("long_video_plan_invalid_segment")
    return _write_plan_receipt(
        root,
        source=source,
        duration_s=duration,
        segments=segments,
        workflow=workflow,
        dialogue_mode=dialogue_mode,
        dialogue_delivery=dialogue_delivery,
        resolved_dialogue_delivery=resolved_dialogue_delivery,
        prompt_fusion_manifest_path=prompt_fusion_manifest_path,
        minimum_source_duration_s=RECEIPT_COMPAT_SEGMENT_MIN_S,
        provider_max_duration_s=LEGACY_PROVIDER_MAX_DURATION_S,
    )
