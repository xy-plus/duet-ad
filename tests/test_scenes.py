"""app/scenes.py 测试：场景检测、帧分组、拆段建议与错误处理。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app import scenes as scenes_module
from app.scenes import (
    build_segments,
    normalize_scene_inventory,
    plan_segments,
    select_segment_keyframes,
)
from app.long_video import provider_duration_s

ROOT = Path(__file__).resolve().parents[1]
SCENES_SCRIPT = ROOT / "app" / "scenes.py"

# 多段不同静态图案拼接，段间硬切，ContentDetector 默认阈值 27 可稳定检出
LONG_SEGMENTS = [
    ("color=c=red", 5.0),
    ("color=c=blue", 5.0),
    ("color=c=green", 5.0),
    ("color=c=white", 5.0),
    ("smptebars", 5.0),
    ("testsrc2", 5.0),
]
SHORT_SEGMENTS = [("color=c=red", 5.0), ("color=c=blue", 5.0)]


def build_scene_video(path: Path, segments: list[tuple[str, float]]) -> Path:
    """用 ffmpeg 拼接多段静态图案生成视频。"""
    inputs: list[str] = []
    for source, duration in segments:
        # 已带首个选项的源（如 color=c=red）用冒号续接，裸滤镜名（如 smptebars）用等号
        sep = ":" if "=" in source else "="
        inputs += ["-f", "lavfi", "-i", f"{source}{sep}s=320x240:r=10:d={duration:g}"]
    labels = ";".join(
        f"[{i}:v]scale=320:240,setsar=1[v{i}]" for i in range(len(segments))
    )
    filter_complex = (
        labels
        + ";"
        + "".join(f"[v{i}]" for i in range(len(segments)))
        + f"concat=n={len(segments)}:v=1:a=0[outv]"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def video_30s(tmp_path_factory):
    return build_scene_video(
        tmp_path_factory.mktemp("video") / "scenes_30s.mp4", LONG_SEGMENTS
    )


@pytest.fixture(scope="module")
def video_10s(tmp_path_factory):
    return build_scene_video(
        tmp_path_factory.mktemp("video") / "scenes_10s.mp4", SHORT_SEGMENTS
    )


@pytest.fixture(scope="module")
def video_24s(tmp_path_factory):
    """~24.03s 真实视频（含一切点），配 6 位小数帧量化时长（29.97fps 形态）的 manifest 用。"""
    return build_scene_video(
        tmp_path_factory.mktemp("video") / "scenes_24s.mp4",
        [("color=c=red", 12.0), ("color=c=blue", 12.033367)],
    )


def write_manifest(work: Path, duration: float) -> dict[str, float]:
    """按 4fps 抽帧节奏伪造 manifest，返回 file -> time_seconds 映射。"""
    frames = []
    for k in range(int(duration * 4)):
        t = round(k * 0.25, 6)
        frames.append(
            {"index": k + 1, "time_seconds": t, "file": f"{k + 1:03d}_frame_{t:07.3f}s.png"}
        )
    (work / "manifest.json").write_text(
        json.dumps({"duration_seconds": duration, "frames": frames}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {frame["file"]: frame["time_seconds"] for frame in frames}


def run_scenes(video: Path, work: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCENES_SCRIPT), str(video), "--work-dir", str(work)],
        capture_output=True,
        text=True,
    )


def test_scenes_json_and_frame_grouping(video_30s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    by_file = write_manifest(work, 30.0)

    result = run_scenes(video_30s, work)
    assert result.returncode == 0, result.stderr

    scenes_path = work / "scenes.json"
    assert scenes_path.is_file()
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    assert data["duration_s"] == 30.0
    assert isinstance(data["scenes"], list) and data["scenes"]

    grouped = [file for scene in data["scenes"] for file in scene["frames"]]
    assert set(grouped) == set(by_file)  # 帧分组无遗漏
    assert len(grouped) == len(by_file)  # 且无重复

    for scene in data["scenes"]:
        for file in scene["frames"]:
            assert scene["start_s"] <= by_file[file] < scene["end_s"]  # 帧时间落在区间内


def test_segments_cover_and_sized(video_30s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_manifest(work, 30.0)

    result = run_scenes(video_30s, work)
    assert result.returncode == 0, result.stderr
    data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))

    segments = data["segments"]
    assert segments  # >15s 视频必须给出拆段建议
    bounds = [(seg["start_s"], seg["end_s"]) for seg in segments]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == pytest.approx(data["duration_s"])  # 覆盖全程
    for prev, cur in zip(bounds, bounds[1:]):
        assert cur[0] == pytest.approx(prev[1])  # 无缝隙
        assert cur[0] >= prev[0]  # 单调递增
    for start, end in bounds:
        assert 1.0 <= end - start
        assert provider_duration_s(start, end) <= 14
    assert all(seg["chain_id"].startswith("chain-") for seg in segments)
    assert all(seg["join_mode"] in {"hard_cut", "continue"} for seg in segments)


def test_segments_empty_for_short_video(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_manifest(work, 10.0)

    result = run_scenes(video_10s, work)
    assert result.returncode == 0, result.stderr
    data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    assert data["segments"] == []  # ≤15s 不算拆段
    assert data["scenes"]  # 场景照常输出


def test_build_segments_starts_immediately_above_fifteen_seconds():
    segments = build_segments([(0.0, 15.001)], 15.001)
    assert segments == [(0.0, 14.0), (14.0, 15.001)]


def test_missing_manifest_fails(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    result = run_scenes(video_10s, work)
    assert result.returncode != 0
    assert not (work / "scenes.json").exists()


def test_corrupt_manifest_fails(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "manifest.json").write_text("{not json", encoding="utf-8")
    result = run_scenes(video_10s, work)
    assert result.returncode != 0


def test_manifest_missing_fields_fails(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "manifest.json").write_text('{"duration_seconds": 10.0}', encoding="utf-8")
    result = run_scenes(video_10s, work)
    assert result.returncode != 0


@pytest.mark.parametrize(
    "manifest_body",
    [
        # json.loads 默认接受 NaN/Infinity 字面量（Python 扩展），必须显式拒绝
        '{"duration_seconds": NaN, "frames": [{"file": "f1.png", "time_seconds": 0.0}]}',
        '{"duration_seconds": Infinity, "frames": [{"file": "f1.png", "time_seconds": 0.0}]}',
        '{"duration_seconds": 10.0, "frames": [{"file": "f1.png", "time_seconds": NaN}]}',
    ],
)
def test_manifest_non_finite_numbers_fail(video_10s, tmp_path, manifest_body):
    """NaN/Infinity 时长或帧时间 → 非零退出，不产出 scenes.json。"""
    work = tmp_path / "work"
    work.mkdir()
    (work / "manifest.json").write_text(manifest_body, encoding="utf-8")
    result = run_scenes(video_10s, work)
    assert result.returncode != 0
    assert not (work / "scenes.json").exists()


def test_nan_threshold_fails(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_manifest(work, 10.0)
    result = subprocess.run(
        [
            sys.executable, str(SCENES_SCRIPT), str(video_10s),
            "--work-dir", str(work), "--threshold", "nan",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_e2e_manifest_duration_six_decimals(video_30s, tmp_path):
    """真实 6 位小数 duration_seconds（extract_keyframes round(6) 口径）全脚本跑通。"""
    work = tmp_path / "work"
    work.mkdir()
    duration = 29.123456
    write_manifest(work, duration)

    result = run_scenes(video_30s, work)
    assert result.returncode == 0, result.stderr

    scenes_path = work / "scenes.json"
    assert scenes_path.is_file()
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    assert data["duration_s"] == round(duration, 3)
    segments = [(seg["start_s"], seg["end_s"]) for seg in data["segments"]]
    assert segments[-1][1] == round(duration, 3)
    assert_segments_valid(segments, data["duration_s"])


def test_e2e_manifest_duration_frame_quantized(video_24s, tmp_path):
    """帧量化 6 位小数 duration_seconds（29.97fps 时长形态）——本次 blocker 的回归锁。"""
    work = tmp_path / "work"
    work.mkdir()
    duration = 24.033367
    write_manifest(work, duration)

    result = run_scenes(video_24s, work)
    assert result.returncode == 0, result.stderr

    scenes_path = work / "scenes.json"
    assert scenes_path.is_file()
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    assert data["duration_s"] == round(duration, 3)
    segments = [(seg["start_s"], seg["end_s"]) for seg in data["segments"]]
    assert segments[-1][1] == round(duration, 3)
    assert_segments_valid(segments, data["duration_s"])


def test_build_segments_splits_long_scene_at_provider_safe_boundaries():
    assert build_segments([(0.0, 10.0), (10.0, 32.0)], 32.0) == [
        (0.0, 10.0),
        (10.0, 24.0),
        (24.0, 32.0),
    ]


def assert_segments_valid(segments, duration):
    """拆段不变量：非空、请求不超过 14s、首 0 尾 duration、相邻无缝。"""
    assert segments
    assert segments[0][0] == pytest.approx(0.0)
    assert segments[-1][1] == pytest.approx(duration)
    prev_end = 0.0
    for start, end in segments:
        assert start == pytest.approx(prev_end)  # 无缝隙
        assert 1.0 <= end - start
        assert provider_duration_s(start, end) <= 14
        assert end > start  # 单调
        prev_end = end


@pytest.mark.parametrize(
    "bounds,duration",
    [
        ([(0.0, 5.0), (5.0, 20.5)], 20.5),  # 单场景 15.5s
        ([(0.0, 5.0), (5.0, 21.0)], 21.0),  # 单场景 16s
        ([(0.0, 5.0), (5.0, 23.9)], 23.9),  # 单场景 18.9s
        ([(0.0, 29.0)], 29.0),  # 单场景 29s
        ([(0.0, 32.0)], 32.0),  # 单场景 32s（旧 15s 硬切尾段 2s 并入前段会超 15s）
    ],
)
def test_build_segments_long_scene_stays_within_limits(bounds, duration):
    """长单场景：provider 请求不超过 14s，覆盖全程且边界连续。"""
    assert_segments_valid(build_segments(bounds, duration), duration)


@pytest.mark.parametrize(
    "bounds,duration",
    [
        ([(0.0, 10.0), (10.0, 25.0), (25.0, 28.0)], 28.0),  # 反例：合并后 18s 段
        ([(0.0, 1.0), (1.0, 16.0), (16.0, 21.0)], 21.0),  # 反例：首段 1s
        ([(0.0, 8.0), (8.0, 23.0), (23.0, 26.0)], 26.0),  # [8,15,3]
        ([(0.0, 4.0), (4.0, 18.0), (18.0, 21.0)], 21.0),  # [4,14,3]
        ([(0.0, 14.0), (14.0, 17.0), (17.0, 23.0)], 23.0),  # [14,3] + 6s 尾
    ],
)
def test_build_segments_violation_cases(bounds, duration):
    """对抗审查反例与短场景变体：任意输入输出均满足拆段不变量。"""
    assert_segments_valid(build_segments(bounds, duration), duration)


def test_build_segments_merges_short_tail():
    """场景尾段较短时仍要保持 provider-safe 且全程连续。"""
    bounds = [(0.0, 8.0), (8.0, 14.0), (14.0, 27.5), (27.5, 29.5)]
    assert_segments_valid(build_segments(bounds, 29.5), 29.5)


def test_build_segments_round3_bounds_with_6dp_duration():
    """round(3) 的 bounds 配 6 位小数 duration（extract_keyframes round(6) 口径）不炸。"""
    segments = build_segments([(0.0, round(29.123456, 3))], 29.123456)
    assert_segments_valid(segments, round(29.123456, 3))


def _scene_length_sequences():
    """生成场景长 1..16s、总时长 21..48s 的所有有序组合（2~4 个场景）。"""

    def generate(prefix, total):
        if len(prefix) >= 2 and 21 <= total <= 48:
            yield tuple(prefix)
        if len(prefix) < 4:
            for length in range(1, 17):
                if total + length <= 48:
                    yield from generate(prefix + [length], total + length)

    return generate([], 0)


def test_build_segments_exhaustive_invariants():
    """穷举场景长 1..16s、总时长 21..48s、2~4 个场景的有序组合共 59,865 个断言不变量。

    仅对该有限域完备的回归，非任意输入完备证明。"""
    count = 0
    for lengths in _scene_length_sequences():
        duration = float(sum(lengths))
        bounds = []
        cum = 0.0
        for length in lengths:
            bounds.append((cum, cum + length))
            cum += length
        assert_segments_valid(build_segments(bounds, duration), duration)
        count += 1
    assert count == 59865


def _decoded_frame(index: int, pts: int, *, denominator: int = 1) -> dict:
    return {
        "decode_frame_index": index,
        "pts": pts,
        "time_base_num": 1,
        "time_base_den": denominator,
    }


def _exact_scene(index: int, start: int, end: int) -> dict:
    return {
        "index": index,
        "start_decode_frame_index": start,
        "end_decode_frame_index": end,
    }


def test_scene_inventory_uses_decode_indices_and_merges_empty_scenes_deterministically():
    frames = [
        _decoded_frame(1, 1001, denominator=30000),
        _decoded_frame(2, 2002, denominator=30000),
        _decoded_frame(4, 4004, denominator=30000),
        _decoded_frame(5, 5005, denominator=30000),
    ]
    bounds = [
        _exact_scene(1, 0, 1),       # leading empty -> merge into following
        _exact_scene(2, 1, 3),
        _exact_scene(3, 3, 4),       # later empty -> merge into preceding
        _exact_scene(4, 4, 6),
    ]

    scenes = normalize_scene_inventory(bounds, frames)

    assert [scene["source_scene_indices"] for scene in scenes] == [[1, 2, 3], [4]]
    assert [
        [frame["decode_frame_index"] for frame in scene["frames"]]
        for scene in scenes
    ] == [[1, 2], [4, 5]]
    assert scenes[0]["start_decode_frame_index"] == 0
    assert scenes[0]["end_decode_frame_index"] == 4
    # Display seconds may round; exact ownership remains the integer frame interval.
    assert scenes[0]["frames"][0]["source_time_s"] == round(1001 / 30000, 6)


def test_detector_exception_becomes_diagnostic_and_keeps_same_scene_algorithm(
    tmp_path, monkeypatch,
):
    frames = [_decoded_frame(index, index, denominator=10) for index in range(3)]
    monkeypatch.setattr(
        scenes_module,
        "detect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("detector")),
    )

    bounds = scenes_module.detect_exact_scene_bounds(
        tmp_path / "source.mp4", 27.0, frames
    )
    normalized = normalize_scene_inventory(bounds, frames)

    assert len(normalized) == 1
    assert [frame["decode_frame_index"] for frame in normalized[0]["frames"]] == [0, 1, 2]
    assert bounds[0]["diagnostics"] == [
        "scene_detector_error_normalized",
        "scene_detector_no_cut_normalized",
    ]


def test_unified_planner_splits_more_than_nine_effective_scenes_on_hard_cut():
    scenes = []
    for index in range(12):
        scenes.append({
            "index": index + 1,
            "start_decode_frame_index": index,
            "end_decode_frame_index": index + 1,
            "start_s": float(index),
            "end_s": float(index + 1),
            "frames": [_decoded_frame(index, index)],
        })

    segments = plan_segments(12.0, scenes, [])

    assert [(segment["start_s"], segment["end_s"]) for segment in segments] == [
        (0.0, 9.0),
        (9.0, 12.0),
    ]
    assert [segment["join_mode"] for segment in segments] == ["hard_cut", "hard_cut"]
    assert [segment["scene_indices"] for segment in segments] == [
        list(range(1, 10)),
        [10, 11, 12],
    ]


def test_keyframe_sampler_uses_anchor_then_capacity_hamilton_and_ordinal_spacing():
    capacities = (1, 2, 10)
    scenes = []
    decode_index = 0
    for scene_index, capacity in enumerate(capacities, 1):
        frames = [
            _decoded_frame(decode_index + offset, decode_index + offset)
            for offset in range(capacity)
        ]
        scenes.append({
            "index": scene_index,
            "start_decode_frame_index": decode_index,
            "end_decode_frame_index": decode_index + capacity,
            "start_s": float(decode_index),
            "end_s": float(decode_index + capacity),
            "frames": frames,
        })
        decode_index += capacity

    selected = select_segment_keyframes(
        scenes,
        {"index": 1, "start_s": 0.0, "end_s": 13.0},
    )

    assert len(selected) == 9
    assert [
        sum(item["source_scene_id"] == f"SCENE_{index:02d}" for item in selected)
        for index in range(1, 4)
    ] == [1, 2, 6]
    assert [
        item["decode_frame_index"]
        for item in selected
        if item["source_scene_id"] == "SCENE_03"
    ] == [3, 5, 7, 8, 10, 12]
    assert selected[1]["transition"] == {
        "type": "hard_cut",
        "at_s": 1.0,
    }
    assert selected[3]["transition"] == {
        "type": "hard_cut",
        "at_s": 3.0,
    }
    assert not any(item["repeated"] for item in selected)


def test_single_scene_anchor_uses_lower_median_actual_frame():
    scenes = []
    decode_index = 0
    for scene_index, capacity in enumerate((68, 257, 39, 70, 1), 1):
        frames = [
            _decoded_frame(decode_index + offset, decode_index + offset)
            for offset in range(capacity)
        ]
        scenes.append({
            "index": scene_index,
            "start_decode_frame_index": decode_index,
            "end_decode_frame_index": decode_index + capacity,
            "start_s": float(decode_index),
            "end_s": float(decode_index + capacity),
            "frames": frames,
        })
        decode_index += capacity

    selected = select_segment_keyframes(
        scenes,
        {"index": 1, "start_s": 0.0, "end_s": 435.0},
    )

    assert [
        sum(item["source_scene_id"] == f"SCENE_{index:02d}" for item in selected)
        for index in range(1, 6)
    ] == [2, 3, 1, 2, 1]
    scene_three = next(
        item for item in selected if item["source_scene_id"] == "SCENE_03"
    )
    assert scene_three["decode_frame_index"] == 325 + 19
    assert selected[-1]["decode_frame_index"] == 434


def test_keyframe_sampler_repeats_nearest_pts_when_source_has_under_nine_frames():
    scenes = [{
        "index": 1,
        "start_decode_frame_index": 0,
        "end_decode_frame_index": 3,
        "start_s": 0.0,
        "end_s": 1.001,
        "frames": [
            _decoded_frame(0, 0, denominator=10),
            _decoded_frame(1, 3, denominator=10),
            _decoded_frame(2, 10, denominator=10),
        ],
    }]

    selected = select_segment_keyframes(
        scenes,
        {"index": 1, "start_s": 0.0, "end_s": 1.001},
    )

    assert len(selected) == 9
    assert [item["decode_frame_index"] for item in selected] == [
        0, 0, 1, 1, 1, 1, 2, 2, 2,
    ]
    assert [item["repeated"] for item in selected] == [
        False, True, False, True, True, True, False, True, True,
    ]
    assert [item["repeat_of_decode_frame_index"] for item in selected] == [
        None, 0, None, 1, 1, 1, None, 2, 2,
    ]
    assert selected[0]["transition"] == {"type": "start", "at_s": 0.0}
    assert all(
        item["transition"] == {"type": "continuous", "at_s": None}
        for item in selected[1:]
    )
