#!/usr/bin/env python3
"""检测视频场景边界：把 manifest 中的帧按场景分组，输出 scenes.json 与拆段边界建议。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from scenedetect import ContentDetector, detect
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit(
        "PySceneDetect is required. Install scenedetect in a task-local environment."
    ) from exc

SEGMENT_MIN_S = 4.0
SEGMENT_MAX_S = 15.0
SEGMENT_ONLY_ABOVE_S = 20.0


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
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise SystemExit("manifest.json 缺少有效的 duration_seconds。")
    if not isinstance(frames, list) or not frames:
        raise SystemExit("manifest.json 缺少非空的 frames 列表。")
    for frame in frames:
        if (
            not isinstance(frame, dict)
            or "file" not in frame
            or not isinstance(frame.get("time_seconds"), (int, float))
        ):
            raise SystemExit("manifest.json 的 frames 元素缺少 file/time_seconds。")
    return float(duration), frames


def detect_scene_bounds(video: Path, threshold: float) -> list[tuple[float, float]]:
    """用 PySceneDetect 检测场景，返回 [(start, end), ...] 时间边界列表。"""
    try:
        detected = detect(str(video), ContentDetector(threshold=threshold))
        # scenedetect 0.6+ 的 detect 可能返回 SceneManager，兼容取列表
        scene_list = detected.get_scene_list() if hasattr(detected, "get_scene_list") else detected
    except Exception as exc:
        raise SystemExit(f"场景检测失败: {exc}") from exc
    bounds = [(float(start), float(end)) for start, end in scene_list]
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
    """按场景边界贪心聚合拆段建议：每段目标 4~15s，覆盖全程无缝隙、边界单调递增。"""
    if duration <= SEGMENT_ONLY_ABOVE_S:
        return []
    segments: list[tuple[float, float]] = []
    seg_start = seg_end = bounds[0][0]
    for start, end in bounds:
        if end - start > SEGMENT_MAX_S:
            # 单场景超 15s：动态均分为 ceil(时长/15) 段，每段必然落在 (7.5, 15]，避免固定硬切的短尾并入超长段
            if seg_end > seg_start:
                segments.append((seg_start, seg_end))
            piece_count = math.ceil((end - start) / SEGMENT_MAX_S)
            piece = (end - start) / piece_count
            for k in range(1, piece_count):
                segments.append((start + (k - 1) * piece, start + k * piece))
            seg_start, seg_end = start + (piece_count - 1) * piece, end
        elif end - seg_start > SEGMENT_MAX_S:
            # 累加该场景将超 15s：闭合当前段
            segments.append((seg_start, seg_end))
            seg_start, seg_end = start, end
        else:
            seg_end = end
    if seg_end > seg_start:
        segments.append((seg_start, seg_end))
    # 末段不足 4s 并入前段
    if len(segments) >= 2 and segments[-1][1] - segments[-1][0] < SEGMENT_MIN_S:
        (_, last_end) = segments.pop()
        (prev_start, _) = segments.pop()
        segments.append((prev_start, last_end))
    return segments


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    if args.threshold <= 0:
        raise SystemExit("--threshold must be positive.")
    work_dir.mkdir(parents=True, exist_ok=True)

    duration, frames = load_manifest(work_dir)
    bounds = detect_scene_bounds(video, args.threshold)
    # 末场景终点对齐 manifest 时长，保证帧分组覆盖全程
    bounds[-1] = (bounds[-1][0], duration)
    bounds = [(round(start, 3), round(end, 3)) for start, end in bounds]

    groups = group_frames(frames, bounds)
    scenes = [
        {"index": index, "start_s": start, "end_s": end, "frames": group}
        for index, ((start, end), group) in enumerate(zip(bounds, groups), start=1)
    ]
    result = {
        "duration_s": duration,
        "scenes": scenes,
        "segments": [
            {"index": index, "start_s": round(start, 3), "end_s": round(end, 3)}
            for index, (start, end) in enumerate(build_segments(bounds, duration), start=1)
        ],
    }
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
