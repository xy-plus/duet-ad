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

SHORT_VIDEO_MAX_S = 10.0
LONG_VIDEO_MAX_S = 300.0
SEGMENT_MIN_S = 1.0
SEGMENT_MAX_S = 15.0
PLAN_RECEIPT_FILENAME = "long_video_plan.json"

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
    if duration > LONG_VIDEO_MAX_S + _EPS:
        raise LongVideoError("long_video_duration_exceeded")
    return duration


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


def _is_safe_boundary(value: float, dialogue: Sequence[tuple[float, float]]) -> bool:
    return not any(start + _EPS < value < end - _EPS for start, end in dialogue)


def _safe_at_or_before(
    target: float,
    minimum: float,
    dialogue: Sequence[tuple[float, float]],
) -> float | None:
    """Return the latest safe point in ``[minimum, target]``.

    A sentence occupies an open interval for cut purposes: cutting exactly at
    its start or end keeps that whole sentence on one side.
    """
    candidate = target
    while True:
        containing = [(start, end) for start, end in dialogue if start < candidate < end]
        if not containing:
            return candidate if candidate >= minimum - _EPS else None
        candidate = min(start for start, _end in containing)
        if candidate < minimum - _EPS:
            return None


def plan_segments(
    duration_s: float,
    scenes: Iterable[Mapping | Sequence[float]],
    dialogue: Iterable[Mapping],
) -> list[dict]:
    """Plan 1..15 second segments with hard cuts preferred over timed splits.

    ``[]`` deliberately means the unchanged short-video path.  Every emitted
    segment carries chain semantics so downstream code never has to infer it.
    """
    duration = _finite_duration(duration_s)
    if duration <= SHORT_VIDEO_MAX_S + _EPS:
        return []
    scene_bounds = _bounds(scenes, duration)
    dialogue_intervals = _dialogue_intervals(dialogue, duration)
    hard_cuts = {end for _start, end in scene_bounds[:-1]}

    cuts = [0.0]
    while duration - cuts[-1] > SEGMENT_MAX_S + _EPS:
        start = cuts[-1]
        target = min(start + SEGMENT_MAX_S, duration - SEGMENT_MIN_S)
        minimum = start + SEGMENT_MIN_S
        eligible_hard_cuts = [
            cut
            for cut in hard_cuts
            if minimum - _EPS <= cut <= target + _EPS
            and duration - cut >= SEGMENT_MIN_S - _EPS
            and _is_safe_boundary(cut, dialogue_intervals)
        ]
        boundary = max(eligible_hard_cuts) if eligible_hard_cuts else _safe_at_or_before(
            target, minimum, dialogue_intervals
        )
        if boundary is None or boundary <= start + _EPS:
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        cuts.append(round(boundary, 6))
    cuts.append(duration)

    # The final remainder may be shorter than one second after a preferred hard
    # cut.  Fold that cut back and choose the latest safe ordinary boundary.
    if cuts[-1] - cuts[-2] < SEGMENT_MIN_S - _EPS:
        cuts.pop(-2)
        start = cuts[-2]
        target = duration - SEGMENT_MIN_S
        boundary = _safe_at_or_before(target, start + SEGMENT_MIN_S, dialogue_intervals)
        if boundary is None or duration - boundary > SEGMENT_MAX_S + _EPS:
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        cuts.insert(-1, round(boundary, 6))

    segments: list[dict] = []
    chain_number = 1
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1):
        length = end - start
        if length < SEGMENT_MIN_S - _EPS or length > SEGMENT_MAX_S + _EPS:
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


def localize_dialogue(lines: Iterable[Mapping], segment: Mapping) -> list[dict]:
    """Return dialogue wholly contained in a segment, shifted to local time."""
    start = float(segment["start_s"])
    end = float(segment["end_s"])
    local: list[dict] = []
    for raw in lines:
        line_start = float(raw["start_s"])
        line_end = float(raw["end_s"])
        overlaps = line_start < end - _EPS and line_end > start + _EPS
        contained = line_start >= start - _EPS and line_end <= end + _EPS
        if overlaps and not contained:
            raise LongVideoError("long_video_no_safe_dialogue_boundary")
        if contained and line_end > start + _EPS and line_start < end - _EPS:
            item = dict(raw)
            item["start_s"] = round(max(0.0, line_start - start), 6)
            item["end_s"] = round(min(end - start, line_end - start), 6)
            if not 0 <= item["start_s"] < item["end_s"] <= end - start + _EPS:
                raise LongVideoError("long_video_no_safe_dialogue_boundary")
            local.append(item)
    return local


def build_continuity_block() -> str:
    """Return the server-owned block injected byte-for-byte into every segment."""
    return _CONTINUITY_BLOCK


def compose_segment_visual_prompt(local_prompt: str, continuity_block: str | None = None) -> str:
    if not isinstance(local_prompt, str) or not local_prompt.strip():
        raise LongVideoError("long_video_invalid_visual_prompt")
    block = build_continuity_block() if continuity_block is None else continuity_block
    if block != build_continuity_block():
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


def write_plan_receipt(
    root: Path,
    *,
    source: Path,
    duration_s: float,
    segments: Sequence[Mapping],
    workflow: str,
) -> Path:
    """Write a canonical receipt binding the complete generated long-video plan."""
    root = root.resolve()
    duration = _finite_duration(duration_s)
    if duration <= SHORT_VIDEO_MAX_S + _EPS or not segments:
        raise LongVideoError("long_video_plan_requires_segments")
    if not isinstance(workflow, str) or not workflow.strip():
        raise LongVideoError("long_video_plan_invalid_workflow")
    receipt_segments = []
    previous_end = 0.0
    for expected_index, raw in enumerate(segments, start=1):
        try:
            index = int(raw["index"])
            start_s = float(raw["start_s"])
            end_s = float(raw["end_s"])
            chain_id = raw["chain_id"]
            join_mode = raw["join_mode"]
        except (KeyError, TypeError, ValueError):
            raise LongVideoError("long_video_plan_invalid_segment") from None
        if (
            index != expected_index
            or not (math.isfinite(start_s) and math.isfinite(end_s))
            or abs(start_s - previous_end) > _EPS
            or end_s - start_s < SEGMENT_MIN_S - _EPS
            or end_s - start_s > SEGMENT_MAX_S + _EPS
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
        ):
            raise LongVideoError("long_video_plan_invalid_segment")
        keyframe_paths = list(raw.get("keyframe_paths", []))
        if not 1 <= len(keyframe_paths) <= 9:
            raise LongVideoError("long_video_plan_invalid_keyframes")
        dialogue = list(raw.get("dialogue", []))
        receipt_segments.append(
            {
                "index": index,
                "start_s": start_s,
                "end_s": end_s,
                "chain_id": chain_id,
                "join_mode": join_mode,
                "source": _artifact(root, Path(raw["source_path"])),
                "keyframes": [
                    _artifact(root, Path(path)) for path in keyframe_paths
                ],
                "visual_prompt": _artifact(root, Path(raw["visual_prompt_path"])),
                "final_prompt": _artifact(root, Path(raw["final_prompt_path"])),
                "dialogue": {
                    "count": len(dialogue),
                    "sha256": hashlib.sha256(_canonical_bytes(dialogue)).hexdigest(),
                },
            }
        )
        previous_end = end_s
    if abs(previous_end - duration) > _EPS:
        raise LongVideoError("long_video_plan_invalid_segment")
    receipt = {
        "schema": "duet.long-video-plan",
        "version": 1,
        "source": _artifact(root, source),
        "video": {"duration_s": duration},
        "workflow": workflow.strip(),
        "segments": receipt_segments,
    }
    path = root / PLAN_RECEIPT_FILENAME
    path.write_bytes(_canonical_bytes(receipt))
    return path
