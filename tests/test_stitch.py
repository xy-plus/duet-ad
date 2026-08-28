import hashlib
import json
import re
import shutil
import subprocess
from array import array
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


def _make_video_with_audio_duration(path: Path, video_duration: float,
                                    audio_duration: float) -> None:
    _run(
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"color=c=black:s=160x120:r=24:d={video_duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_duration}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )


def _make_offset_video(path: Path, offset: float) -> None:
    video_offset = max(0.0, -offset)
    audio_offset = max(0.0, offset)
    _run(
        "ffmpeg", "-v", "error", "-y", "-itsoffset", str(video_offset),
        "-f", "lavfi", "-i", "color=c=black:size=160x120:rate=24:d=2",
        "-itsoffset", str(audio_offset), "-f", "lavfi", "-i",
        "sine=frequency=1000:sample_rate=48000:duration=2.5",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-copyts", str(path),
    )


def _audible_start(path: Path) -> float:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path), "-af",
            "silencedetect=noise=-35dB:d=0.05", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    match = re.search(r"silence_end: ([0-9.]+)", result.stderr)
    return float(match.group(1)) if match else 0.0


def _make_priming_video(path: Path, audio_codec: str) -> None:
    video_codec = "libvpx-vp9" if path.suffix == ".webm" else "libx264"
    _run(
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:size=160x120:rate=24:d=2",
        "-f", "lavfi", "-i",
        "aevalsrc=if(lt(t\\,0.012)\\,sin(2*PI*1000*t)\\,0):s=48000:d=2",
        "-map", "0:v", "-map", "1:a", "-c:v", video_codec,
        "-pix_fmt", "yuv420p", "-c:a", audio_codec, str(path),
    )


def _initial_peak(path: Path) -> int:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-t", "0.012",
            "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
        ],
        check=True, capture_output=True,
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    return max(map(abs, samples), default=0)


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


def _native_audio_timeline() -> dict:
    return {
        "schema": "duet.h3.media_timeline",
        "version": 1,
        "decode_complete": True,
        "video": {"decoded_sha256": "1" * 64},
        "audio": {"decoded_sha256": "2" * 64},
    }


def _native_silent_timeline() -> dict:
    return {
        "schema": "duet.h3.media_timeline",
        "version": 1,
        "decode_complete": True,
        "video": {"decoded_sha256": "3" * 64},
        "audio": None,
    }


def test_provider_generated_stitch_supports_native_then_missing_audio_segment(
    tmp_path,
):
    native = tmp_path / "native.mp4"
    silent = tmp_path / "silent.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(native, "red", 1.0, codec="libx264", rate=24, audio=True)
    _make_video(silent, "blue", 1.0, codec="libx264", rate=24)
    _make_video(source, "black", 2.0, codec="libx264", rate=24, audio=True)
    segments = [
        stitch.StitchSegment(
            native, 1.0, "hard_cut", "000001", _native_audio_timeline(),
        ),
        stitch.StitchSegment(
            silent, 1.0, "hard_cut", "000002", _native_silent_timeline(),
        ),
    ]

    result = stitch.stitch_video(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    )

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert [item["source"] for item in receipt["audio"]["provider_segments"]] == [
        "h3", "h3",
    ]
    assert stitch.output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    )


def test_provider_generated_stitch_fills_missing_h3_audio_on_same_edl(
    tmp_path,
):
    silent = tmp_path / "h3-without-audio.mp4"
    source = tmp_path / "source-with-audio.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(silent, "blue", 1.0, codec="libx264", rate=24)
    _make_video(source, "black", 1.0, codec="libx264", rate=24, audio=True)
    segments = [stitch.StitchSegment(
        silent, 1.0, "hard_cut", "000001", _native_silent_timeline(),
    )]

    result = stitch.stitch_video(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    )

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["audio"]["provider_segments"] == [{
        "source": "h3",
        "attempt_id": "000001",
        "media_timeline_sha256": stitch._canonical_sha256(
            _native_silent_timeline()
        ),
        "decoded_audio_sha256": None,
    }]
    assert len([
        stream for stream in _probe(output)["streams"]
        if stream["codec_type"] == "audio"
    ]) == 1


@pytest.mark.parametrize("audio_duration", [0.35, 1.6])
def test_source_audio_overlay_mode_is_not_supported(tmp_path, audio_duration):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(segment, "red", 1.0, codec="libx264", rate=24)
    _make_video_with_audio_duration(source, 1.0, audio_duration)

    with pytest.raises(ValueError, match="mute.*provider_generated"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(segment, 1.0, "hard_cut")],
            source_video=source,
            output=output,
            audio_mode="keep",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("offset", [-0.5, 0.0, 0.5])
def test_source_audio_offset_never_enables_overlay(tmp_path, offset):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / f"source-{offset}.mp4"
    output = tmp_path / f"generated-{offset}.mp4"
    _make_video(segment, "red", 2.0, codec="libx264", rate=24)
    _make_offset_video(source, offset)

    with pytest.raises(ValueError, match="mute.*provider_generated"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(segment, 2.0, "hard_cut")],
            source_video=source,
            output=output,
            audio_mode="keep",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("suffix", "audio_codec"), [(".mp4", "aac"), (".webm", "libopus")])
def test_source_codec_never_enables_overlay(tmp_path, suffix, audio_codec):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / f"source{suffix}"
    output = tmp_path / f"generated-{audio_codec}.mp4"
    _make_video(segment, "red", 2.0, codec="libx264", rate=24)
    _make_priming_video(source, audio_codec)

    with pytest.raises(ValueError, match="mute.*provider_generated"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(segment, 2.0, "hard_cut")],
            source_video=source,
            output=output,
            audio_mode="keep",  # type: ignore[arg-type]
        )


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
        audio_mode="mute",
    )

    probe = _probe(output)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert video["avg_frame_rate"] == "24/1"
    assert all(s["codec_type"] != "audio" for s in probe["streams"])
    assert abs(float(video["duration"]) - 1.0) <= 1 / 24
    assert _pixel(output, 0.20)[0] > 200
    assert _pixel(output, 0.70)[2] > 200

    receipt = json.loads(result.receipt_path.read_text())
    assert result.output == output
    assert receipt["schema"] == "duet.stitch"
    assert receipt["version"] == 1
    assert receipt["audio"] == {
        "mode": "mute", "source": str(source.resolve()),
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


def test_continue_and_hard_cut_both_keep_first_decoded_frame(tmp_path):
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

    assert _pixel(continued, 0.51)[0] > 200
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


def test_mute_without_source_audio_is_valid_silent_output(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(segment, "green", 0.7, codec="mpeg4", rate=15, audio=True)
    _make_video(source, "black", 0.5, codec="libx264", rate=24)

    result = stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 0.5, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="mute",
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


def test_duration_limits_accept_15_second_segment_and_300_second_total(
    tmp_path, monkeypatch,
):
    media = tmp_path / "input.mp4"
    media.write_bytes(b"validation-only")
    segments = [stitch.StitchSegment(media, 15.0, "hard_cut")]
    segments += [stitch.StitchSegment(media, 15.0, "hard_cut") for _ in range(19)]

    class ReachedProbe(Exception):
        pass

    monkeypatch.setattr(
        stitch, "_probe", lambda _path: (_ for _ in ()).throw(ReachedProbe())
    )
    with pytest.raises(ReachedProbe):
        stitch.stitch_video(
            segments=segments,
            source_video=media,
            output=tmp_path / "generated.mp4",
            audio_mode="mute",
        )


@pytest.mark.parametrize(
    "durations, message",
    [
        ([15.001], "15"),
        ([15.0] * 20 + [0.001], "300"),
    ],
)
def test_duration_limits_reject_overflow_before_probe(
    tmp_path, monkeypatch, durations, message,
):
    media = tmp_path / "input.mp4"
    media.write_bytes(b"validation-only")
    probe_calls = []
    monkeypatch.setattr(stitch, "_probe", lambda path: probe_calls.append(path))

    with pytest.raises(ValueError, match=message):
        stitch.stitch_video(
            segments=[
                stitch.StitchSegment(media, duration, "hard_cut")
                for duration in durations
            ],
            source_video=media,
            output=tmp_path / "generated.mp4",
            audio_mode="mute",
        )

    assert probe_calls == []
