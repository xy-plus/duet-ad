#!/usr/bin/env python3
"""检测视频场景边界：把 manifest 中的帧按场景分组，输出 scenes.json 与拆段边界建议。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:  # direct script execution: expose repository package root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import long_video

try:
    from scenedetect import ContentDetector, detect
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit(
        "PySceneDetect is required. Install scenedetect>=0.7 in the server environment."
    ) from exc


KEYFRAMES_PER_SEGMENT = 9
MAX_EFFECTIVE_SCENES_PER_SEGMENT = KEYFRAMES_PER_SEGMENT
DISPLAY_TIME_PRECISION = 6


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _frame_time(frame: Mapping) -> Fraction:
    pts = _integer(frame.get("pts"), "frame pts", minimum=-(2**63))
    origin = _integer(
        frame.get("pts_origin", 0), "frame PTS origin", minimum=-(2**63)
    )
    numerator = _integer(frame.get("time_base_num"), "time base numerator", minimum=1)
    denominator = _integer(
        frame.get("time_base_den"), "time base denominator", minimum=1
    )
    return Fraction((pts - origin) * numerator, denominator)


def _display_time(value: Fraction) -> float:
    """Serialize time for humans; exact planning never consumes this value."""
    return round(float(value), DISPLAY_TIME_PRECISION)


def _exact_time_receipt(value: Fraction) -> dict:
    return {
        "pts": value.numerator,
        "time_base_num": 1,
        "time_base_den": value.denominator,
    }


def _receipt_time(value: object, label: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an exact time receipt")
    return Fraction(
        _integer(value.get("pts"), f"{label} pts", minimum=-(2**63))
        * _integer(value.get("time_base_num"), f"{label} numerator", minimum=1),
        _integer(value.get("time_base_den"), f"{label} denominator", minimum=1),
    )


def _scene_bound_time(bound: Mapping, edge: str, frames: list[dict]) -> Fraction:
    pts_key = f"{edge}_pts"
    if pts_key in bound:
        pts = _integer(bound.get(pts_key), pts_key, minimum=-(2**63))
        numerator = _integer(
            bound.get("time_base_num"), "scene time base numerator", minimum=1
        )
        denominator = _integer(
            bound.get("time_base_den"), "scene time base denominator", minimum=1
        )
        return Fraction(pts * numerator, denominator)
    display = bound.get(f"{edge}_s")
    if isinstance(display, bool) or not isinstance(display, (int, float)):
        if frames:
            return _frame_time(frames[0] if edge == "start" else frames[-1])
        return Fraction(0)
    if not math.isfinite(float(display)):
        raise ValueError(f"scene {edge}_s must be finite")
    return Fraction(str(display))


def normalize_scene_inventory(
    bounds: Iterable[Mapping], decoded_frames: Iterable[Mapping]
) -> list[dict]:
    """Assign decoded frames to exact half-open scene intervals.

    Integer decode indices determine ownership.  Rational PTS/time-base values
    determine source time and are never rounded until the JSON display fields
    are created.  Detector scenes without a decoded frame are normalized by one
    fixed rule: leading empty scenes join the next populated scene; every later
    empty scene joins the preceding populated scene.
    """
    frames: list[dict] = []
    previous_index: int | None = None
    previous_time: Fraction | None = None
    for position, raw in enumerate(decoded_frames, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"decoded frame {position} must be an object")
        item = dict(raw)
        decode_index = _integer(
            item.get("decode_frame_index"), "decode frame index"
        )
        exact_time = _frame_time(item)
        if previous_index is not None and decode_index <= previous_index:
            raise ValueError("decoded frame indices must be strictly increasing")
        if previous_time is not None and exact_time < previous_time:
            raise ValueError("decoded frame PTS values must be nondecreasing")
        item["source_time_s"] = _display_time(exact_time)
        frames.append(item)
        previous_index = decode_index
        previous_time = exact_time
    if not frames:
        raise ValueError("decoded frame inventory must not be empty")

    raw_scenes: list[dict] = []
    previous_end: int | None = None
    for position, raw in enumerate(bounds, 1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"scene bound {position} must be an object")
        source_index = _integer(raw.get("index", position), "scene index", minimum=1)
        if source_index != position:
            raise ValueError("scene indices must be contiguous and one-based")
        start = _integer(
            raw.get("start_decode_frame_index"), "scene start decode frame index"
        )
        end = _integer(
            raw.get("end_decode_frame_index"), "scene end decode frame index"
        )
        if start > end or (previous_end is not None and start != previous_end):
            raise ValueError("scene decode frame intervals must be contiguous")
        owned = [
            frame
            for frame in frames
            if start <= frame["decode_frame_index"] < end
        ]
        raw_scenes.append({
            "index": source_index,
            "source_scene_indices": [source_index],
            "start_decode_frame_index": start,
            "end_decode_frame_index": end,
            "start_exact": _scene_bound_time(raw, "start", owned),
            "end_exact": _scene_bound_time(raw, "end", owned),
            "frames": owned,
        })
        previous_end = end
    if not raw_scenes:
        raise ValueError("scene inventory must not be empty")
    assigned = [
        frame["decode_frame_index"]
        for scene in raw_scenes
        for frame in scene["frames"]
    ]
    if assigned != [frame["decode_frame_index"] for frame in frames]:
        raise ValueError("scene intervals do not cover every decoded frame exactly once")

    first_populated = next(
        (position for position, scene in enumerate(raw_scenes) if scene["frames"]),
        None,
    )
    if first_populated is None:  # defensive: frames are nonempty and coverage was verified
        raise ValueError("scene inventory contains no populated scene")
    if first_populated:
        leading = raw_scenes[:first_populated]
        target = raw_scenes[first_populated]
        target["source_scene_indices"] = [
            source
            for scene in leading + [target]
            for source in scene["source_scene_indices"]
        ]
        target["start_decode_frame_index"] = leading[0]["start_decode_frame_index"]
        target["start_exact"] = leading[0]["start_exact"]

    effective: list[dict] = []
    for scene in raw_scenes[first_populated:]:
        if scene["frames"]:
            effective.append(scene)
            continue
        previous = effective[-1]
        previous["source_scene_indices"].extend(scene["source_scene_indices"])
        previous["end_decode_frame_index"] = scene["end_decode_frame_index"]
        previous["end_exact"] = scene["end_exact"]

    result: list[dict] = []
    for index, scene in enumerate(effective, 1):
        result.append({
            "index": index,
            "source_scene_indices": scene["source_scene_indices"],
            "start_decode_frame_index": scene["start_decode_frame_index"],
            "end_decode_frame_index": scene["end_decode_frame_index"],
            "start_s": _display_time(scene["start_exact"]),
            "end_s": _display_time(scene["end_exact"]),
            "start_time": _exact_time_receipt(scene["start_exact"]),
            "end_time": _exact_time_receipt(scene["end_exact"]),
            "frames": scene["frames"],
        })
    return result


def _scene_indices_for_interval(
    scenes: list[dict], start_s: float, end_s: float
) -> list[int]:
    start = Fraction(str(start_s))
    end = Fraction(str(end_s))
    indices = []
    for scene in scenes:
        if any(start <= _frame_time(frame) < end for frame in scene["frames"]):
            indices.append(scene["index"])
    return indices


def plan_segments(
    duration_s: float,
    effective_scenes: Iterable[Mapping],
    dialogue: Iterable[Mapping],
) -> list[dict]:
    """Return the one segment plan constrained by provider and scene capacity.

    ``long_video.plan_segments`` remains the provider/dialogue planner.  This
    entry point adds the keyframe-capacity invariant to the same frozen result:
    no segment may contain more than nine populated scenes, and any required
    capacity split is an existing scene boundary (therefore a hard cut).
    """
    scenes = [dict(scene) for scene in effective_scenes]
    if not scenes or any(not scene.get("frames") for scene in scenes):
        raise ValueError("effective scenes must be populated")
    base = long_video.plan_segments(duration_s, scenes, dialogue)
    cut_points = {Fraction(str(segment["start_s"])) for segment in base[1:]}
    hard_cuts = {
        (
            _receipt_time(scene["start_time"], "scene start")
            if "start_time" in scene
            else Fraction(str(scene["start_s"]))
        )
        for scene in scenes[1:]
    }
    for segment in base:
        indices = _scene_indices_for_interval(
            scenes, float(segment["start_s"]), float(segment["end_s"])
        )
        for offset in range(MAX_EFFECTIVE_SCENES_PER_SEGMENT, len(indices), 9):
            next_scene = scenes[indices[offset] - 1]
            cut_points.add(
                _receipt_time(next_scene["start_time"], "scene start")
                if "start_time" in next_scene
                else Fraction(str(next_scene["start_s"]))
            )

    boundaries = [Fraction(0), *sorted(cut_points), Fraction(str(duration_s))]
    # A base-plan boundary may be identical to the outer boundaries.
    boundaries = list(dict.fromkeys(boundaries))
    planned: list[dict] = []
    chain = 0
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        if end <= start:
            raise ValueError("segment boundaries must be strictly increasing")
        scene_indices = _scene_indices_for_interval(scenes, float(start), float(end))
        if not scene_indices or len(scene_indices) > MAX_EFFECTIVE_SCENES_PER_SEGMENT:
            raise ValueError("segment scene capacity invariant violated")
        hard_cut = index == 1 or start in hard_cuts
        if hard_cut:
            chain += 1
        planned.append({
            "index": index,
            "start_s": round(float(start), long_video.BOUNDARY_PRECISION),
            "end_s": round(float(end), long_video.BOUNDARY_PRECISION),
            "chain_id": f"chain-{chain:03d}",
            "join_mode": "hard_cut" if hard_cut else "continue",
            "scene_indices": scene_indices,
        })
    return planned


def _hamilton_scene_counts(capacities: list[int], seats: int) -> list[int]:
    if not capacities or len(capacities) > seats or sum(capacities) < seats:
        raise ValueError("scene capacities cannot fill keyframe seats")
    counts = [1] * len(capacities)
    remaining = seats - len(capacities)
    spare = [capacity - 1 for capacity in capacities]
    total_spare = sum(spare)
    if not remaining:
        return counts
    quotas = [Fraction(remaining * value, total_spare) for value in spare]
    floors = [quota.numerator // quota.denominator for quota in quotas]
    counts = [count + extra for count, extra in zip(counts, floors)]
    leftovers = remaining - sum(floors)
    ranked = sorted(
        range(len(capacities)),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in ranked[:leftovers]:
        counts[index] += 1
    if any(count > capacity for count, capacity in zip(counts, capacities)):
        raise ValueError("Hamilton allocation exceeded scene capacity")
    return counts


def _equidistant_ordinals(capacity: int, count: int) -> list[int]:
    if count < 1 or count > capacity:
        raise ValueError("invalid frame selection count")
    if count == 1:
        return [(capacity - 1) // 2]
    result = []
    for position in range(count):
        exact = Fraction(position * (capacity - 1), count - 1)
        result.append((2 * exact.numerator + exact.denominator) // (2 * exact.denominator))
    if len(set(result)) != count:
        raise ValueError("ordinal spacing did not select distinct frames")
    return result


def _nearest_frame(target: Fraction, frames: list[dict]) -> int:
    return min(
        range(len(frames)),
        key=lambda index: (
            abs(_frame_time(frames[index]) - target),
            frames[index]["decode_frame_index"],
        ),
    )


def _under_capacity_counts(frames: list[dict], seats: int) -> list[int]:
    if not frames or len(frames) >= seats:
        raise ValueError("under-capacity repeat planner received invalid frame count")
    if len(frames) == 1:
        return [seats]
    first = _frame_time(frames[0])
    last = _frame_time(frames[-1])
    targets = [first + Fraction(position, seats - 1) * (last - first) for position in range(seats)]
    assignments = [_nearest_frame(target, frames) for target in targets]
    counts = [assignments.count(index) for index in range(len(frames))]

    # Extremely clustered PTS values can leave an actual decoded frame without
    # an ideal target.  Preserve every actual frame once, then remove the
    # farthest duplicate assignment; ties resolve by later target then frame.
    for missing in [index for index, count in enumerate(counts) if count == 0]:
        candidates = [
            (abs(_frame_time(frames[source]) - targets[target_index]), target_index, source)
            for target_index, source in enumerate(assignments)
            if counts[source] > 1
        ]
        if not candidates:
            raise ValueError("cannot preserve every decoded frame")
        _distance, target_index, source = max(candidates)
        assignments[target_index] = missing
        counts[source] -= 1
        counts[missing] += 1
    return counts


def select_segment_keyframes(
    effective_scenes: Iterable[Mapping], segment: Mapping
) -> list[dict]:
    """Select exactly nine deterministic backend-owned source frames."""
    try:
        start = Fraction(str(segment["start_s"]))
        end = Fraction(str(segment["end_s"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("segment bounds are invalid") from None
    if start < 0 or end <= start:
        raise ValueError("segment bounds are invalid")

    scenes: list[dict] = []
    for raw in effective_scenes:
        scene = dict(raw)
        frames = [
            dict(frame)
            for frame in scene.get("frames", [])
            if start <= _frame_time(frame) < end
        ]
        if frames:
            scene["frames"] = frames
            scenes.append(scene)
    if not scenes or len(scenes) > MAX_EFFECTIVE_SCENES_PER_SEGMENT:
        raise ValueError("segment effective scene count must be in 1..9")
    all_frames = [frame for scene in scenes for frame in scene["frames"]]
    if len(all_frames) >= KEYFRAMES_PER_SEGMENT:
        counts = _hamilton_scene_counts(
            [len(scene["frames"]) for scene in scenes], KEYFRAMES_PER_SEGMENT
        )
        chosen = [
            (scene, scene["frames"][ordinal], False)
            for scene, count in zip(scenes, counts)
            for ordinal in _equidistant_ordinals(len(scene["frames"]), count)
        ]
    else:
        repeat_counts = _under_capacity_counts(all_frames, KEYFRAMES_PER_SEGMENT)
        scene_by_decode_index = {
            frame["decode_frame_index"]: scene
            for scene in scenes
            for frame in scene["frames"]
        }
        chosen = []
        for frame, count in zip(all_frames, repeat_counts):
            scene = scene_by_decode_index[frame["decode_frame_index"]]
            chosen.extend((scene, frame, repeat > 0) for repeat in range(count))

    selected: list[dict] = []
    previous_scene: int | None = None
    for order, (scene, frame, repeated) in enumerate(chosen, 1):
        exact_time = _frame_time(frame)
        scene_id = f"SCENE_{scene['index']:02d}"
        if order == 1:
            transition = {"type": "start", "at_s": _display_time(exact_time)}
        elif scene["index"] != previous_scene:
            transition = {
                "type": "hard_cut",
                "at_s": round(float(scene["start_s"]), DISPLAY_TIME_PRECISION),
            }
        else:
            transition = {"type": "continuous", "at_s": None}
        selected.append({
            "order": order,
            "decode_frame_index": frame["decode_frame_index"],
            "pts": frame["pts"],
            "pts_origin": frame.get("pts_origin", 0),
            "time_base_num": frame["time_base_num"],
            "time_base_den": frame["time_base_den"],
            "source_time_s": _display_time(exact_time),
            "source_scene_id": scene_id,
            "source_scene_start_s": round(
                float(scene["start_s"]), DISPLAY_TIME_PRECISION
            ),
            "transition": transition,
            "repeated": repeated,
            "repeat_of_decode_frame_index": (
                frame["decode_frame_index"] if repeated else None
            ),
        })
        previous_scene = scene["index"]
    if len(selected) != KEYFRAMES_PER_SEGMENT:
        raise AssertionError("keyframe sampler did not produce exactly nine frames")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检测视频场景边界，把关键帧按场景分组，并给出拆段边界建议。"
    )
    parser.add_argument("video", help="Input MP4, MOV, or WebM file.")
    parser.add_argument(
        "--work-dir", required=True, help="目录：含 manifest.json，scenes.json 写到这里。"
    )
    parser.add_argument(
        "--threshold", type=float, default=27.0, help="ContentDetector 阈值（默认 27）。"
    )
    return parser.parse_args()


def load_manifest(work_dir: Path) -> tuple[float, list[dict]]:
    """读取并校验 manifest.json，返回 (时长秒数, 帧列表)。"""
    path = work_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取 {path}: {exc}") from exc
    duration = manifest.get("duration_seconds")
    frames = manifest.get("frames")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise SystemExit("manifest.json 缺少有效的 duration_seconds。")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("manifest.json 缺少非空的 frames 列表。")
    for frame in frames:
        if (
            not isinstance(frame, dict)
            or "file" not in frame
            or not isinstance(frame.get("time_seconds"), (int, float))
            or not math.isfinite(frame.get("time_seconds"))
        ):
            raise SystemExit("manifest.json 的 frames 元素缺少 file/time_seconds。")
    return float(duration), frames


def probe_decoded_frames(video: Path) -> list[dict]:
    """Read the original v:0 decoded-frame ordinal and rational PTS inventory."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SystemExit("ffprobe is required for exact scene/frame planning.")
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries",
        "stream=time_base:frame=best_effort_timestamp,pts,pkt_duration",
        "-of", "json",
        str(video),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=300
        )
        payload = json.loads(completed.stdout or "{}")
        streams = payload.get("streams")
        raw_frames = payload.get("frames")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("missing v:0 stream")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("missing decoded frames")
        raw_time_base = streams[0].get("time_base")
        numerator_text, denominator_text = str(raw_time_base).split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if numerator <= 0 or denominator <= 0:
            raise ValueError("invalid v:0 time base")
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"无法读取原始解码帧 PTS: {exc}") from exc

    frames: list[dict] = []
    previous_time: Fraction | None = None
    pts_origin: int | None = None
    for decode_index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise SystemExit("原始解码帧清单包含非法元素。")
        raw_pts = raw.get("best_effort_timestamp", raw.get("pts"))
        try:
            pts = int(raw_pts)
        except (TypeError, ValueError):
            raise SystemExit("原始解码帧缺少整数 PTS。") from None
        if pts_origin is None:
            pts_origin = pts
        exact_time = Fraction((pts - pts_origin) * numerator, denominator)
        if previous_time is not None and exact_time < previous_time:
            raise SystemExit("原始解码帧 PTS 不是单调时间轴。")
        duration_pts = raw.get("pkt_duration")
        try:
            duration_ticks = int(duration_pts) if duration_pts is not None else None
        except (TypeError, ValueError):
            duration_ticks = None
        frames.append({
            "decode_frame_index": decode_index,
            "pts": pts,
            "pts_origin": pts_origin,
            "duration_pts": duration_ticks if duration_ticks and duration_ticks > 0 else None,
            "time_base_num": numerator,
            "time_base_den": denominator,
            "source_time_s": _display_time(exact_time),
        })
        previous_time = exact_time
    return frames


def _exclusive_end_time(decoded_frames: list[dict], end_index: int) -> Fraction:
    if end_index < len(decoded_frames):
        return _frame_time(decoded_frames[end_index])
    last = decoded_frames[-1]
    duration_ticks = last.get("duration_pts")
    if isinstance(duration_ticks, int) and duration_ticks > 0:
        return _frame_time(last) + Fraction(
            duration_ticks * last["time_base_num"], last["time_base_den"]
        )
    if len(decoded_frames) > 1:
        step = _frame_time(decoded_frames[-1]) - _frame_time(decoded_frames[-2])
        if step > 0:
            return _frame_time(last) + step
    raise SystemExit("无法确定最后一帧的排他结束 PTS。")


def detect_exact_scene_bounds(
    video: Path, threshold: float, decoded_frames: list[dict]
) -> list[dict]:
    """Detect cuts once and bind every boundary to the decoded-frame axis."""
    diagnostics: list[str] = []
    try:
        scene_list = detect(
            str(video),
            ContentDetector(threshold=threshold),
            start_in_scene=True,
        )
    except Exception:
        scene_list = []
        diagnostics.append("scene_detector_error_normalized")
    if not scene_list:
        diagnostics.append("scene_detector_no_cut_normalized")
        start_exact = _frame_time(decoded_frames[0])
        end_exact = _exclusive_end_time(decoded_frames, len(decoded_frames))
        common_denominator = math.lcm(
            start_exact.denominator, end_exact.denominator
        )
        return [{
            "index": 1,
            "start_decode_frame_index": 0,
            "end_decode_frame_index": len(decoded_frames),
            "start_pts": start_exact.numerator * (
                common_denominator // start_exact.denominator
            ),
            "end_pts": end_exact.numerator * (
                common_denominator // end_exact.denominator
            ),
            "time_base_num": 1,
            "time_base_den": common_denominator,
            "start_s": _display_time(start_exact),
            "end_s": _display_time(end_exact),
            "diagnostics": diagnostics,
        }]
    bounds: list[dict] = []
    for index, (start, end) in enumerate(scene_list, 1):
        start_index = max(0, min(int(start.frame_num), len(decoded_frames)))
        end_index = max(start_index, min(int(end.frame_num), len(decoded_frames)))
        start_exact = (
            _frame_time(decoded_frames[start_index])
            if start_index < len(decoded_frames)
            else _exclusive_end_time(decoded_frames, start_index)
        )
        end_exact = _exclusive_end_time(decoded_frames, end_index)
        start_receipt = _exact_time_receipt(start_exact)
        end_receipt = _exact_time_receipt(end_exact)
        bounds.append({
            "index": index,
            "start_decode_frame_index": start_index,
            "end_decode_frame_index": end_index,
            "start_pts": start_receipt["pts"],
            "end_pts": end_receipt["pts"],
            "time_base_num": 1,
            "time_base_den": math.lcm(
                start_receipt["time_base_den"], end_receipt["time_base_den"]
            ),
            "start_s": _display_time(start_exact),
            "end_s": _display_time(end_exact),
            "diagnostics": [],
        })
        common_denominator = bounds[-1]["time_base_den"]
        bounds[-1]["start_pts"] = start_exact.numerator * (
            common_denominator // start_exact.denominator
        )
        bounds[-1]["end_pts"] = end_exact.numerator * (
            common_denominator // end_exact.denominator
        )
    return bounds


def detect_scene_bounds(video: Path, threshold: float) -> list[tuple[float, float]]:
    """用 PySceneDetect 检测场景，返回 [(start, end), ...] 时间边界列表。"""
    try:
        # detect() 直接返回 [(FrameTimecode, FrameTimecode), ...] 边界列表（>=0.7；requirements 锁 scenedetect>=0.7）
        scene_list = detect(
            str(video), ContentDetector(threshold=threshold), start_in_scene=True
        )
    except Exception as exc:
        raise SystemExit(f"场景检测失败: {exc}") from exc
    bounds = [(start.seconds, end.seconds) for start, end in scene_list]
    if not bounds:
        raise SystemExit("未检测到任何场景。")
    return bounds


def group_frames(
    frames: list[dict], bounds: list[tuple[float, float]]
) -> list[list[str]]:
    """按帧时间落在 [start, end) 区间把帧文件名分组到各场景。"""
    groups: list[list[str]] = [[] for _ in bounds]
    for frame in frames:
        t = float(frame["time_seconds"])
        for index, (start, end) in enumerate(bounds):
            if start <= t < end:
                groups[index].append(frame["file"])
                break
        else:
            # 帧时间恰等于末场景终点（浮点误差）时并入末场景
            if t <= bounds[-1][1] + 1e-3:
                groups[-1].append(frame["file"])
            else:
                raise SystemExit(f"帧 {frame['file']} 时间 {t}s 不在任何场景区间内。")
    return groups


def build_segments(
    bounds: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """按统一 planner 生成 provider-safe 的旧/辅助拆段建议。

    输入契约：bounds 须已 round(3)（main 侧已做，本函数内部对 duration 同口径 round）。"""
    duration = round(duration, 3)  # 统一 3 位口径：bounds 已 round(3)，断言两端才可比
    if duration <= long_video.SHORT_VIDEO_MAX_S:
        return []
    planned = long_video.plan_segments(duration, bounds, [])
    return [(item["start_s"], item["end_s"]) for item in planned]


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        raise SystemExit("--threshold must be a positive finite number.")
    work_dir.mkdir(parents=True, exist_ok=True)

    duration, frames = load_manifest(work_dir)
    decoded_frames = probe_decoded_frames(video)
    exact_bounds = detect_exact_scene_bounds(video, args.threshold, decoded_frames)
    effective_scenes = normalize_scene_inventory(exact_bounds, decoded_frames)

    # Legacy sampled-frame grouping remains a display/navigation projection.
    # Its millisecond values never feed the exact planner or sampler.
    bounds = [
        (round(float(bound["start_s"]), 3), round(float(bound["end_s"]), 3))
        for bound in exact_bounds
    ]
    bounds[-1] = (bounds[-1][0], round(duration, 3))

    groups = group_frames(frames, bounds)
    scenes = [
        {"index": index, "start_s": start, "end_s": end, "frames": group}
        for index, ((start, end), group) in enumerate(zip(bounds, groups), start=1)
    ]
    result = {
        "duration_s": round(duration, 3),  # 与 start_s/end_s 同为 3 位，round 口径一致
        "scenes": scenes,
        "effective_scenes": effective_scenes,
        "diagnostics": [
            diagnostic
            for bound in exact_bounds
            for diagnostic in bound.get("diagnostics", [])
        ],
        "segments": [],
    }
    chain = 0
    scene_starts = {start for start, _end in bounds}
    for index, (start, end) in enumerate(build_segments(bounds, duration), start=1):
        hard_cut = index == 1 or start in scene_starts
        if hard_cut:
            chain += 1
        result["segments"].append(
            {
                "index": index,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "chain_id": f"chain-{chain:03d}",
                "join_mode": "hard_cut" if hard_cut else "continue",
            }
        )
    (work_dir / "scenes.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
