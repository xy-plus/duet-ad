import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import stitch


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required",
)


def _run(*argv: str) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def _make_video(path: Path, color: str, duration: float, *, codec: str, rate: int,
                audio: bool = False) -> None:
    argv = [
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"color=c={color}:s=160x120:r={rate}:d={duration}",
    ]
    if audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    argv += ["-c:v", codec, "-pix_fmt", "yuv420p"]
    if audio:
        argv += ["-c:a", "aac", "-shortest"]
    argv.append(str(path))
    _run(*argv)


def _make_leading_duplicate(path: Path, *, blue_duration: float = 0.75) -> None:
    _run(
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=12:d=0.083333",
        "-f", "lavfi", "-i", f"color=c=blue:s=160x120:r=12:d={blue_duration}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path),
    )


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,avg_frame_rate,duration:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _pixel(path: Path, timestamp: float) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    )
    return tuple(result.stdout[:3])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stitch_normalizes_order_duration_audio_and_receipt(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(first, "red", 0.8, codec="libx264", rate=30, audio=True)
    _make_video(second, "blue", 0.9, codec="mpeg4", rate=12)
    _make_video(source, "black", 1.0, codec="libx264", rate=25, audio=True)

    result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut"),
            stitch.StitchSegment(second, 0.5, "hard_cut"),
        ],
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    probe = _probe(output)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert video["avg_frame_rate"] == "24/1"
    assert any(s["codec_type"] == "audio" and s["codec_name"] == "aac"
               for s in probe["streams"])
    assert abs(float(video["duration"]) - 1.0) <= 1 / 24
    assert _pixel(output, 0.20)[0] > 200
    assert _pixel(output, 0.70)[2] > 200

    receipt = json.loads(result.receipt_path.read_text())
    assert result.output == output
    assert receipt["schema"] == "duet.stitch"
    assert receipt["version"] == 1
    assert receipt["audio"] == {
        "mode": "keep", "source": str(source.resolve()),
        "source_sha256": _sha256(source), "source_has_audio": True,
    }
    assert [item["sha256"] for item in receipt["segments"]] == [
        _sha256(first), _sha256(second),
    ]
    assert [item["join_mode"] for item in receipt["segments"]] == [
        "hard_cut", "hard_cut",
    ]
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["output"]["size"] == output.stat().st_size
    assert abs(receipt["output"]["duration_s"] - 1.0) <= 1 / 24


def test_continue_drops_first_decoded_frame_but_hard_cut_keeps_it(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    _make_video(first, "red", 0.7, codec="libx264", rate=24)
    _make_leading_duplicate(second)
    _make_video(source, "black", 1.0, codec="libx264", rate=24)

    continued = tmp_path / "continued.mp4"
    stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut"),
            stitch.StitchSegment(second, 0.5, "continue"),
        ],
        source_video=source,
        output=continued,
        audio_mode="mute",
    )
    hard = tmp_path / "hard.mp4"
    stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut"),
            stitch.StitchSegment(second, 0.5, "hard_cut"),
        ],
        source_video=source,
        output=hard,
        audio_mode="mute",
    )

    assert _pixel(continued, 0.51)[2] > 200
    assert _pixel(hard, 0.51)[0] > 200
    assert all(s["codec_type"] != "audio" for s in _probe(continued)["streams"])


def test_single_15_second_segment_keeps_its_first_frame(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_leading_duplicate(segment, blue_duration=15.0)
    _make_video(source, "black", 15.0, codec="libx264", rate=24)

    stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 15.0, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="mute",
    )

    assert _pixel(output, 0.001)[0] > 200
    video = next(s for s in _probe(output)["streams"] if s["codec_type"] == "video")
    assert abs(float(video["duration"]) - 15.0) <= 1 / 24


def test_keep_without_source_audio_is_valid_silent_output(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(segment, "green", 0.7, codec="mpeg4", rate=15, audio=True)
    _make_video(source, "black", 0.5, codec="libx264", rate=24)

    result = stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 0.5, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    assert all(s["codec_type"] != "audio" for s in _probe(output)["streams"])
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["audio"]["source_has_audio"] is False


def test_ffmpeg_failure_preserves_existing_output_and_receipt(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    source = tmp_path / "source.mp4"
    _make_video(source, "black", 0.5, codec="libx264", rate=24)
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"old generated video")
    receipt = tmp_path / stitch.RECEIPT_FILENAME
    receipt.write_bytes(b"old receipt")

    with pytest.raises(stitch.StitchError, match="ffmpeg|ffprobe"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(broken, 0.5, "hard_cut")],
            source_video=source,
            output=output,
            audio_mode="mute",
        )

    assert output.read_bytes() == b"old generated video"
    assert receipt.read_bytes() == b"old receipt"
    assert not list(tmp_path.glob(".stitch-*"))


@pytest.mark.parametrize("audio_mode", ["invalid", "KEEP", ""])
def test_rejects_invalid_audio_mode_before_running_ffmpeg(tmp_path, audio_mode):
    with pytest.raises(ValueError, match="audio_mode"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(tmp_path / "x.mp4", 1, "hard_cut")],
            source_video=tmp_path / "source.mp4",
            output=tmp_path / "generated.mp4",
            audio_mode=audio_mode,
        )
