import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills" / "video-maker" / "scripts" / "extract_keyframes.py"
)
SPEC = importlib.util.spec_from_file_location("extract_keyframes_duration_test", SCRIPT)
assert SPEC and SPEC.loader
extract = importlib.util.module_from_spec(SPEC)
_dont_write_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    SPEC.loader.exec_module(extract)
finally:
    sys.dont_write_bytecode = _dont_write_bytecode


def _probe(monkeypatch, stream, packets=None):
    monkeypatch.setattr(extract.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        extract.subprocess,
        "run",
        lambda argv, *_a, **_kw: SimpleNamespace(
            stdout=json.dumps(
                {"packets": packets or []}
                if "-show_packets" in argv else {"streams": [stream]}
            ), returncode=0, stderr=""
        ),
    )
    return extract.probe_video_duration(Path("source.mp4"), decoded_duration=20.0)


def test_manifest_duration_prefers_video_stream_for_vfr(monkeypatch):
    assert _probe(monkeypatch, {"duration": "16.766667"}) == 16.766667


def test_manifest_duration_uses_duration_ts_time_base(monkeypatch):
    assert _probe(monkeypatch, {
        "duration": "N/A", "duration_ts": "503", "time_base": "1/30",
    }) == pytest.approx(503 / 30)


def test_manifest_duration_uses_packet_timeline_not_decoded_metadata(monkeypatch):
    assert _probe(monkeypatch, {}, [
        {"pts_time": "0"}, {"pts_time": "19.9", "duration_time": "0.1"},
    ]) == 20.0


def test_manifest_duration_rejects_bool_duration_and_uses_ticks(monkeypatch):
    assert _probe(monkeypatch, {
        "duration": True, "duration_ts": "503", "time_base": "1/30",
    }) == pytest.approx(503 / 30)


def test_manifest_duration_rejects_bool_ticks_and_uses_packet_timeline(monkeypatch):
    assert _probe(monkeypatch, {
        "duration": False, "duration_ts": True, "time_base": "1/30",
    }, [{"pts_time": "0"}, {"pts_time": "19.9", "duration_time": "0.1"}]) == 20.0


def test_manifest_duration_infers_missing_last_packet_duration_from_adjacent_pts(monkeypatch):
    assert _probe(
        monkeypatch,
        {"avg_frame_rate": "1/1"},
        [{"pts_time": "3"}, {"pts_time": "3.04"}, {"pts_time": "3.08"}],
    ) == pytest.approx(0.12)


def test_manifest_duration_infers_single_packet_duration_from_average_rate(monkeypatch):
    assert _probe(
        monkeypatch,
        {"avg_frame_rate": "25/1"},
        [{"pts_time": "7.5"}],
    ) == pytest.approx(0.04)


@pytest.mark.parametrize("bad_pts", [True, False, "nan", "inf"])
def test_manifest_duration_rejects_bool_or_non_finite_packet_pts(
    monkeypatch, bad_pts,
):
    with pytest.raises(SystemExit, match="video stream duration"):
        _probe(monkeypatch, {}, [{"pts_time": bad_pts}])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_real_webm_manifest_uses_visual_packets_not_longer_audio(tmp_path):
    source = tmp_path / "source.webm"
    work = tmp_path / "work"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=9.8",
            "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=10.2",
            "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
            "-c:a", "libopus", str(source),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--out-dir", str(work), "--fps", "4"],
        check=True,
    )

    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["duration_seconds"] == pytest.approx(9.8, abs=0.02)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_real_vfr_webm_extracts_by_presentation_timeline(tmp_path):
    source = tmp_path / "vfr.webm"
    work = tmp_path / "work"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=5:duration=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-fps_mode", "vfr", "-c:v", "libvpx-vp9",
            "-deadline", "realtime", "-cpu-used", "8", str(source),
        ],
        check=True,
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--out-dir", str(work), "--fps", "4"],
        check=True,
    )
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["duration_seconds"] == pytest.approx(2.0, abs=0.02)
    assert any(frame["time_seconds"] >= 1.5 for frame in manifest["frames"])
    assert all((work / frame["file"]).is_file() for frame in manifest["frames"])
