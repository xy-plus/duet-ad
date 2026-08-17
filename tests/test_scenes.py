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


def test_build_segments_hard_cut_long_scene():
    """单场景超 15s 按 15s 硬切，只算边界。"""
    assert build_segments([(0.0, 10.0), (10.0, 32.0)], 32.0) == [
        (0.0, 10.0),
        (10.0, 25.0),
        (25.0, 32.0),
    ]


def test_build_segments_merges_short_tail():
    """末段不足 4s 并入前段。"""
    bounds = [(0.0, 8.0), (8.0, 14.0), (14.0, 27.5), (27.5, 29.5)]
    assert build_segments(bounds, 29.5) == [(0.0, 14.0), (14.0, 29.5)]
