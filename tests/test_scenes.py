"""app/scenes.py 测试：场景检测、帧分组、拆段建议与错误处理。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.scenes import build_segments

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
    assert segments  # >20s 视频必须给出拆段建议
    bounds = [(seg["start_s"], seg["end_s"]) for seg in segments]
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == pytest.approx(data["duration_s"])  # 覆盖全程
    for prev, cur in zip(bounds, bounds[1:]):
        assert cur[0] == pytest.approx(prev[1])  # 无缝隙
        assert cur[0] >= prev[0]  # 单调递增
    for start, end in bounds:
        assert 4.0 <= end - start <= 15.0  # 每段 4~15s


def test_segments_empty_for_short_video(video_10s, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    write_manifest(work, 10.0)

    result = run_scenes(video_10s, work)
    assert result.returncode == 0, result.stderr
    data = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    assert data["segments"] == []  # ≤20s 不算拆段
    assert data["scenes"]  # 场景照常输出


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


def test_build_segments_splits_long_scene_evenly():
    """单场景超 15s 动态均分：22s 场景 → 两段 11s。"""
    assert build_segments([(0.0, 10.0), (10.0, 32.0)], 32.0) == [
        (0.0, 10.0),
        (10.0, 21.0),
        (21.0, 32.0),
    ]


def assert_segments_valid(segments, duration):
    """拆段不变量：非空、每段 4~15s、首 0 尾 duration、相邻无缝、单调递增。"""
    assert segments
    assert segments[0][0] == pytest.approx(0.0)
    assert segments[-1][1] == pytest.approx(duration)
    prev_end = 0.0
    for start, end in segments:
        assert start == pytest.approx(prev_end)  # 无缝隙
        assert 4.0 <= end - start <= 15.0
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
    """超 15s 单场景：每段 4~15s、覆盖全程无缝隙、边界单调递增。"""
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
    """末段不足 4s 并入前段（合并体超 15s 则均分）。"""
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
