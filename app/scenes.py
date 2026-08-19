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
        "PySceneDetect is required. Install scenedetect>=0.7 in the server environment."
    ) from exc

# 与当前生成链单段上限耦合：拆段目标每段 4~15s，仅时长 >20s 才计算
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


def detect_scene_bounds(video: Path, threshold: float) -> list[tuple[float, float]]:
    """用 PySceneDetect 检测场景，返回 [(start, end), ...] 时间边界列表。"""
    try:
        # detect() 直接返回 [(FrameTimecode, FrameTimecode), ...] 边界列表（>=0.7；requirements 锁 scenedetect>=0.7）
        scene_list = detect(str(video), ContentDetector(threshold=threshold))
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


def _assert_segments_valid(
    segments: list[tuple[float, float]], duration: float
) -> None:
    """防御性断言拆段不变量：每段 4~15s、相邻无缝、首 0 尾 duration（算法已保证，兜底）。"""
    if not segments:
        raise SystemExit("拆段结果为空。")
    prev_end = 0.0
    for start, end in segments:
        if abs(start - prev_end) > 1e-6:
            raise SystemExit(f"拆段边界不连续: {start:.6f}s 与 {prev_end:.6f}s")
        if end - start < SEGMENT_MIN_S - 1e-9 or end - start > SEGMENT_MAX_S + 1e-9:
            raise SystemExit(f"拆段长度违规: {start:.6f}s - {end:.6f}s")
        prev_end = end
    if abs(segments[0][0]) > 1e-6 or abs(prev_end - duration) > 1e-6:
        raise SystemExit(
            f"拆段未覆盖全程: [{segments[0][0]:.6f}s, {prev_end:.6f}s] vs {duration:.6f}s"
        )


def build_segments(
    bounds: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """按场景边界生成拆段建议：每段 4~15s、覆盖全程无缝隙、边界单调递增（算法级不变量）。

    输入契约：bounds 须已 round(3)（main 侧已做，本函数内部对 duration 同口径 round）。"""
    duration = round(duration, 3)  # 统一 3 位口径：bounds 已 round(3)，断言两端才可比
    if duration <= SEGMENT_ONLY_ABOVE_S:
        return []
    # 1. 预处理：>15s 场景均分为 ceil(时长/15) 块，得原子块列表（每块 ∈ (0, 15]）
    blocks: list[tuple[float, float]] = []
    for start, end in bounds:
        if end - start > SEGMENT_MAX_S:
            piece_count = math.ceil((end - start) / SEGMENT_MAX_S)
            piece = (end - start) / piece_count
            for k in range(piece_count):
                block_end = end if k == piece_count - 1 else start + (k + 1) * piece
                blocks.append((start + k * piece, block_end))
        else:
            blocks.append((start, end))
    # 2. 贪心装箱：累加块至将超 15s 时闭合当前段（闭合段长 ∈ (0, 15]）
    segments: list[tuple[float, float]] = []
    seg_start = 0.0  # 首段起点钉 0
    seg_end = 0.0
    for start, end in blocks:
        if end - seg_start > SEGMENT_MAX_S:
            segments.append((seg_start, seg_end))
            seg_start = start
        seg_end = end
    segments.append((seg_start, seg_end))
    # 3. 修复违规段（一轮收敛）：<4s 并入邻段（合并体 >15s 则均分成 2）
    # 步骤 2 不变量保证装箱段恒 ≤15，>15 只可能由合并产生、已在分支内均分，无需独立分支
    fixed: list[tuple[float, float]] = list(segments)
    i = 0
    while i < len(fixed):
        start, end = fixed[i]
        length = end - start
        if length < SEGMENT_MIN_S:
            if i == 0 and len(fixed) > 1:
                # 首段并入后段
                _, next_end = fixed[1]
                if next_end - start > SEGMENT_MAX_S:
                    half = (next_end - start) / 2
                    fixed[0:2] = [(start, start + half), (start + half, next_end)]
                else:
                    fixed[0:2] = [(start, next_end)]
            elif i > 0:
                # 并入前段
                prev_start, _ = fixed[i - 1]
                if end - prev_start > SEGMENT_MAX_S:
                    half = (end - prev_start) / 2
                    fixed[i - 1 : i + 1] = [
                        (prev_start, prev_start + half),
                        (prev_start + half, end),
                    ]
                else:
                    fixed[i - 1 : i + 1] = [(prev_start, end)]
        i += 1
    # 4. 防御性断言不变量
    _assert_segments_valid(fixed, duration)
    return fixed


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
        "duration_s": round(duration, 3),  # 与 start_s/end_s 同为 3 位，round 口径一致
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
