import hashlib
import json
import math
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


def _make_tone_video(
    path: Path, color: str, duration_s: float, frequency: int, *, rate: int = 24,
) -> None:
    _run(
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        f"color=c={color}:s=160x120:r={rate}:d={duration_s}",
        "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:sample_rate=48000:duration={duration_s}",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )


def _tone_score(path: Path, start_s: float, frequency: int) -> float:
    sample_rate = 8_000
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path),
            "-t", "0.20", "-map", "0:a:0", "-f", "s16le", "-ac", "1",
            "-ar", str(sample_rate), "-",
        ],
        check=True, capture_output=True,
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    real = sum(
        sample * math.cos(2 * math.pi * frequency * index / sample_rate)
        for index, sample in enumerate(samples)
    )
    imag = sum(
        sample * math.sin(2 * math.pi * frequency * index / sample_rate)
        for index, sample in enumerate(samples)
    )
    return math.hypot(real, imag)


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


def _make_leading_flash(path: Path, *, body_color: str = "blue") -> None:
    _run(
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=green:s=160x120:r=24:d=0.041667",
        "-f", "lavfi", "-i", f"color=c={body_color}:s=160x120:r=24:d=0.75",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    )


def _make_sparse_marker_video(path: Path) -> None:
    _run(
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "color=c=black:s=640x480:r=24:d=0.7",
        "-vf", "drawbox=x=1:y=1:w=1:h=1:color=white:t=fill",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path),
    )


def _make_vfr_video(path: Path) -> None:
    _run(
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "testsrc2=s=160x120:r=30:d=1.2",
        "-vf", "select='not(mod(n,3))',setpts=N/(10*TB)",
        "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    )


def _make_pulse_source(path: Path) -> None:
    _run(
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "color=c=black:s=160x120:r=24:d=1",
        "-f", "lavfi", "-i",
        "aevalsrc=if(between(t\\,0.24\\,0.27)+between(t\\,0.74\\,0.77)\\,"
        "0.9*sin(2*PI*1000*t)\\,0):s=48000:d=1",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )


def _audio_peak(path: Path, start_s: float, duration_s: float = 0.04) -> int:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path),
            "-t", str(duration_s), "-map", "0:a:0", "-f", "s16le", "-ac", "1",
            "-ar", "16000", "-",
        ],
        check=True, capture_output=True,
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    return max(map(abs, samples), default=0)


def _provider_evidence(path: Path, attempt_id: str) -> stitch.ProviderMediaEvidence:
    video = stitch._stream_timeline(path, "v:0")
    audio = stitch._stream_timeline(path, "a:0")
    receipt = path.parent / ".h3" / "attempts" / attempt_id / "attempt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps({"attempt_id": attempt_id, "output": path.name}),
        encoding="utf-8",
    )

    def upstream(local: dict) -> dict:
        return {
            "time_base": local["time_base"],
            "frame_count": local["decoded_units"],
            "packet_count": local["packet_count"],
            "first_packet_pts_s": local["first_packet_pts_s"],
            "last_packet_pts_s": local["last_packet_pts_s"],
            "packet_end_s": local["packet_end_s"],
            "first_frame_pts_s": local["first_pts_s"],
            "last_frame_pts_s": local["last_pts_s"],
            "frame_end_s": local["frame_end_s"],
            "presentation_monotonic": local["pts_monotonic"],
            "packet_dts_monotonic": local["dts_monotonic"],
        }

    audio_payload = None
    if audio is not None:
        audio_payload = {
            **upstream(audio),
            "decoded_sha256": stitch._decoded_audio_sha256(path, has_audio=True),
        }
    timeline = {
        "schema": "duet.h3.media_timeline",
        "version": 1,
        "decode_complete": True,
        "video": upstream(video),
        "audio": audio_payload,
    }
    return stitch.ProviderMediaEvidence(
        source="h3",
        attempt_id=attempt_id,
        receipt_path=receipt,
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        media_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_size=path.stat().st_size,
        media_timeline=timeline,
    )


def _anchor(kind: str, start_s: float, end_s: float, anchor_id: str):
    return stitch.TimelineAnchor(kind, start_s, end_s, anchor_id)


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


@pytest.mark.parametrize("audio_duration", [0.35, 1.6])
def test_keep_audio_is_trimmed_or_padded_to_visual_timeline(tmp_path, audio_duration):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(segment, "red", 1.0, codec="libx264", rate=24)
    _make_video_with_audio_duration(source, 1.0, audio_duration)

    stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 1.0, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    probe = _probe(output)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    assert abs(float(video["duration"]) - 1.0) <= 1 / 24
    assert abs(float(audio["duration"]) - float(video["duration"])) <= 0.05


@pytest.mark.parametrize("offset", [-0.5, 0.0, 0.5])
def test_keep_audio_preserves_relative_source_offset(tmp_path, offset):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / f"source-{offset}.mp4"
    output = tmp_path / f"generated-{offset}.mp4"
    _make_video(segment, "red", 2.0, codec="libx264", rate=24)
    _make_offset_video(source, offset)

    stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 2.0, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    onset = _audible_start(output)
    if offset > 0:
        assert onset == pytest.approx(0.5, abs=0.08)
    else:
        assert onset < 0.08
    video = next(s for s in _probe(output)["streams"] if s["codec_type"] == "video")
    assert float(video["duration"]) == pytest.approx(2.0, abs=1 / 24)


@pytest.mark.parametrize(("suffix", "audio_codec"), [(".mp4", "aac"), (".webm", "libopus")])
def test_keep_audio_does_not_trim_codec_priming_twice(tmp_path, suffix, audio_codec):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / f"source{suffix}"
    output = tmp_path / f"generated-{audio_codec}.mp4"
    _make_video(segment, "red", 2.0, codec="libx264", rate=24)
    _make_priming_video(source, audio_codec)

    stitch.stitch_video(
        segments=[stitch.StitchSegment(segment, 2.0, "hard_cut")],
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    assert _initial_peak(output) > 3_000


def _pixel(path: Path, timestamp: float) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True,
    )
    return tuple(result.stdout[:3])


def _pixel_frame(path: Path, frame: int) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vf",
            f"select='eq(n,{frame})'", "-frames:v", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
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
    assert receipt["version"] == 2
    assert receipt["audio"]["mode"] == "keep"
    assert receipt["audio"]["source_path"] == str(source.resolve())
    assert receipt["audio"]["source"]["container_sha256"] == _sha256(source)
    assert receipt["edl"]["source"]["sha256"] == _sha256(source)
    assert receipt["edl"]["source"]["video_timeline"]["time_base"]
    assert receipt["audio"]["source_has_audio"] is True
    assert [item["sha256"] for item in receipt["segments"]] == [
        _sha256(first), _sha256(second),
    ]
    assert [item["join_mode"] for item in receipt["segments"]] == [
        "hard_cut", "hard_cut",
    ]
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["output"]["size"] == output.stat().st_size
    assert abs(receipt["output"]["duration_s"] - 1.0) <= 1 / 24
    assert receipt["output"]["video_timeline"]["packet_count"] > 0
    assert receipt["output"]["video_timeline"]["dts_monotonic"] is True
    assert receipt["output"]["video_timeline"]["pts_monotonic"] is True
    assert receipt["output"]["video_timeline"]["frame_end_s"] == pytest.approx(
        1.0, abs=1 / 24
    )


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


def test_continue_does_not_treat_thumbnail_invisible_pixel_change_as_duplicate(
    tmp_path,
):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(first, "black", 0.7, codec="libx264", rate=24)
    _make_sparse_marker_video(second)
    _make_video(source, "black", 1.0, codec="libx264", rate=24)

    result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut", 0.0, 0.5),
            stitch.StitchSegment(second, 0.5, "continue", 0.5, 1.0),
        ],
        source_video=source,
        output=output,
        audio_mode="mute",
    )

    boundary = json.loads(result.receipt_path.read_text())["edl"]["entries"][1][
        "boundary"
    ]
    assert boundary["method"] == "decoded-rgb24-full-frame-exact-v2"
    assert boundary["duplicate_proven"] is False
    assert boundary["dropped_leading_frames"] == 0


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


def test_receipt_v2_freezes_audio_master_edl_and_media_timelines(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_video(first, "red", 0.8, codec="libx264", rate=30, audio=True)
    _make_vfr_video(second)
    _make_video(source, "black", 1.0, codec="libx264", rate=24, audio=True)

    segments = [
            stitch.StitchSegment(
                first, 0.5, "hard_cut", source_start_s=0.0, source_end_s=0.5,
                action_anchors=(_anchor("action", 0.20, 0.24, "flash"),),
            ),
            stitch.StitchSegment(
                second, 0.5, "hard_cut", source_start_s=0.5, source_end_s=1.0,
                action_anchors=(_anchor("action", 0.70, 0.75, "impact"),),
            ),
        ]
    result = stitch.stitch_video(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="keep",
    )

    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["version"] == 2
    assert receipt["algorithm"] == stitch.STITCH_ALGORITHM
    assert receipt["toolchain"]["ffmpeg_version"].startswith("ffmpeg version ")
    assert receipt["edl"]["master_clock"] == "source_audio"
    assert receipt["edl"]["total_frames"] == 24
    assert [entry["source_range_s"] for entry in receipt["edl"]["entries"]] == [
        {"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0},
    ]
    assert [entry["target_frame_range"] for entry in receipt["edl"]["entries"]] == [
        {"start": 0, "end": 12}, {"start": 12, "end": 24},
    ]
    assert receipt["audio"]["time_stretch_applied"] is False
    assert receipt["audio"]["source"]["stream_sha256"]
    assert receipt["audio"]["final"]["stream_sha256"]
    assert len(receipt["audio"]["providers"]) == 2
    assert receipt["output"]["video_timeline"]["pts_monotonic"] is True
    assert receipt["output"]["video_timeline"]["dts_monotonic"] is True
    assert receipt["output"]["video_timeline"]["decoded_units"] == 24
    assert receipt["output"]["audio_timeline"]["pts_monotonic"] is True
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="keep",
    ) is True

    valid_receipt = json.loads(json.dumps(receipt))
    receipt["unexpected"] = True
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="keep",
    ) is False

    receipt = json.loads(json.dumps(valid_receipt))
    receipt["edl"]["total_duration_s"] = "1.0"
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="keep",
    ) is False

    receipt = valid_receipt
    receipt["edl"]["entries"][1]["target_frame_range"]["start"] = 11
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="keep",
    ) is False


def test_continue_only_drops_a_locally_proven_duplicate_boundary_frame(tmp_path):
    first = tmp_path / "first.mp4"
    duplicate = tmp_path / "duplicate.mp4"
    flash = tmp_path / "flash.mp4"
    source = tmp_path / "source.mp4"
    _make_video(first, "red", 0.7, codec="libx264", rate=24)
    _make_leading_duplicate(duplicate)
    _make_leading_flash(flash)
    _make_video(source, "black", 1.0, codec="libx264", rate=24)

    duplicate_out = tmp_path / "duplicate-out.mp4"
    duplicate_result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut", 0.0, 0.5),
            stitch.StitchSegment(duplicate, 0.5, "continue", 0.5, 1.0),
        ],
        source_video=source, output=duplicate_out, audio_mode="mute",
        receipt_path=tmp_path / "duplicate-receipt.json",
    )
    duplicate_receipt = json.loads(duplicate_result.receipt_path.read_text())
    duplicate_boundary = duplicate_receipt["edl"]["entries"][1]["boundary"]
    assert duplicate_boundary["duplicate_proven"] is True
    assert duplicate_boundary["dropped_leading_frames"] == 1
    assert _pixel(duplicate_out, 0.51)[2] > 200

    flash_out = tmp_path / "flash-out.mp4"
    flash_result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut", 0.0, 0.5),
            stitch.StitchSegment(flash, 0.5, "continue", 0.5, 1.0),
        ],
        source_video=source, output=flash_out, audio_mode="mute",
        receipt_path=tmp_path / "flash-receipt.json",
    )
    flash_receipt = json.loads(flash_result.receipt_path.read_text())
    flash_boundary = flash_receipt["edl"]["entries"][1]["boundary"]
    assert flash_boundary["duplicate_proven"] is False
    assert flash_boundary["dropped_leading_frames"] == 0
    green = _pixel_frame(flash_out, 12)
    assert green[1] > green[0] and green[1] > green[2]


@pytest.mark.parametrize("join_mode", ["hard_cut", "continue"])
def test_flash_boundary_never_moves_source_master_audio_pulses(tmp_path, join_mode):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / f"{join_mode}.mp4"
    _make_video(first, "red", 0.7, codec="libx264", rate=30)
    _make_leading_flash(second)
    _make_pulse_source(source)

    result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(first, 0.5, "hard_cut", 0.0, 0.5),
            stitch.StitchSegment(second, 0.5, join_mode, 0.5, 1.0),
        ],
        source_video=source, output=output, audio_mode="keep",
    )

    assert _audio_peak(output, 0.23, 0.06) > 8_000
    assert _audio_peak(output, 0.73, 0.06) > 8_000
    assert _audio_peak(output, 0.45, 0.04) < 1_500
    green = _pixel_frame(output, 12)
    assert green[1] > green[0] and green[1] > green[2]
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["edl"]["entries"][1]["target_frame_range"] == {
        "start": 12, "end": 24,
    }
    assert receipt["audio"]["source"]["stream_sha256"]


def test_provider_generated_mode_stitches_each_h3_av_timeline_without_source_audio(
    tmp_path,
):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_tone_video(first, "red", 0.8, 440, rate=30)
    _make_tone_video(second, "blue", 0.8, 880, rate=12)
    _make_tone_video(source, "black", 1.0, 220)
    segments = [
        stitch.StitchSegment(
            first, 0.5, "hard_cut", 0.0, 0.5,
            dialogue_anchors=(_anchor("dialogue", 0.1, 0.4, "line-1"),),
            provider_evidence=_provider_evidence(first, "000001"),
        ),
        stitch.StitchSegment(
            second, 0.5, "hard_cut", 0.5, 1.0,
            provider_evidence=_provider_evidence(second, "000002"),
        ),
    ]

    result = stitch.stitch_video(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    )

    assert _tone_score(output, 0.10, 440) > _tone_score(output, 0.10, 220) * 5
    assert _tone_score(output, 0.10, 440) > _tone_score(output, 0.10, 880) * 5
    assert _tone_score(output, 0.60, 880) > _tone_score(output, 0.60, 220) * 5
    assert _tone_score(output, 0.60, 880) > _tone_score(output, 0.60, 440) * 5
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["edl"]["master_clock"] == "provider_generated_av"
    assert receipt["audio"]["master"] == "provider_segments"
    assert receipt["audio"]["source_role"] == "upstream_h3_material_only"
    assert [item["stream_sha256"] for item in receipt["audio"]["providers"]] == [
        receipt["segments"][0]["provider_media"]["audio"]["stream_sha256"],
        receipt["segments"][1]["provider_media"]["audio"]["stream_sha256"],
    ]
    assert receipt["audio"]["final"]["stream_sha256"]
    assert receipt["edl"]["entries"][0]["dialogue_anchors"][0]["anchor_id"] == "line-1"
    assert [item["upstream_receipt"]["attempt_id"] for item in receipt["segments"]] == [
        "000001", "000002",
    ]
    assert [item["upstream_receipt"]["media_sha256"] for item in receipt["segments"]] == [
        _sha256(first), _sha256(second),
    ]
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    ) is True
    segments[0].provider_evidence.receipt_path.write_text("tampered", encoding="utf-8")
    assert stitch.stitched_output_is_reusable(
        segments=segments,
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    ) is False


def test_provider_generated_mode_requires_exact_upstream_receipt(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    _make_tone_video(segment, "red", 1.0, 440)
    _make_tone_video(source, "black", 1.0, 220)

    with pytest.raises(ValueError, match="exact provider media evidence"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(
                segment, 1.0, "hard_cut", 0.0, 1.0,
            )],
            source_video=source,
            output=tmp_path / "generated.mp4",
            audio_mode="provider_generated",
        )


def test_provider_generated_mode_rejects_a_segment_without_joint_audio(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    _make_video(segment, "red", 1.0, codec="libx264", rate=24)
    _make_tone_video(source, "black", 1.0, 220)

    with pytest.raises(stitch.StitchError, match="provider_generated_audio_missing"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(
                segment, 1.0, "hard_cut", 0.0, 1.0,
                provider_evidence=_provider_evidence(segment, "000001"),
            )],
            source_video=source,
            output=tmp_path / "generated.mp4",
            audio_mode="provider_generated",
        )


def test_provider_generated_mode_rejects_provider_av_timeline_drift(tmp_path):
    segment = tmp_path / "segment.mp4"
    source = tmp_path / "source.mp4"
    _make_offset_video(segment, 0.5)
    _make_tone_video(source, "black", 1.0, 220)

    with pytest.raises(stitch.StitchError, match="provider_generated_av_timeline_invalid"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(
                segment, 1.0, "hard_cut", 0.0, 1.0,
                provider_evidence=_provider_evidence(segment, "000001"),
            )],
            source_video=source,
            output=tmp_path / "generated.mp4",
            audio_mode="provider_generated",
        )


def test_provider_generated_continue_keeps_joint_av_boundary_even_when_frame_repeats(
    tmp_path,
):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    source = tmp_path / "source.mp4"
    output = tmp_path / "generated.mp4"
    _make_tone_video(first, "red", 0.7, 440)
    _run(
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=24:d=0.041667",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=24:d=0.75",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=0.8",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a:0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(second),
    )
    _make_tone_video(source, "black", 1.0, 220)

    result = stitch.stitch_video(
        segments=[
            stitch.StitchSegment(
                first, 0.5, "hard_cut", 0.0, 0.5,
                provider_evidence=_provider_evidence(first, "000001"),
            ),
            stitch.StitchSegment(
                second, 0.5, "continue", 0.5, 1.0,
                provider_evidence=_provider_evidence(second, "000002"),
            ),
        ],
        source_video=source,
        output=output,
        audio_mode="provider_generated",
    )

    receipt = json.loads(result.receipt_path.read_text())
    boundary = receipt["edl"]["entries"][1]["boundary"]
    assert boundary["method"] == "joint-av-preserve-v1"
    assert boundary["dropped_leading_frames"] == 0
    assert _pixel_frame(output, 12)[0] > 200
    assert _tone_score(output, 0.52, 880) > _tone_score(output, 0.52, 440) * 5


def test_source_ranges_are_contiguous_and_match_frame_budget(tmp_path):
    media = tmp_path / "input.mp4"
    media.write_bytes(b"validation-only")

    with pytest.raises(ValueError, match="source timeline must be contiguous"):
        stitch.stitch_video(
            segments=[
                stitch.StitchSegment(media, 0.5, "hard_cut", 0.0, 0.5),
                stitch.StitchSegment(media, 0.5, "hard_cut", 0.6, 1.1),
            ],
            source_video=media, output=tmp_path / "gap.mp4", audio_mode="mute",
        )
    with pytest.raises(ValueError, match="source range.*target duration"):
        stitch.stitch_video(
            segments=[stitch.StitchSegment(
                media, 0.5, "hard_cut", 0.0, 0.8,
            )],
            source_video=media, output=tmp_path / "mismatch.mp4", audio_mode="mute",
        )
    with pytest.raises(ValueError, match="cumulative source EDL"):
        stitch.stitch_video(
            segments=[
                stitch.StitchSegment(media, 0.5, "hard_cut", 0.0, 0.52),
                stitch.StitchSegment(media, 0.5, "hard_cut", 0.52, 1.04),
                stitch.StitchSegment(media, 0.5, "hard_cut", 1.04, 1.56),
            ],
            source_video=media,
            output=tmp_path / "cumulative-drift.mp4",
            audio_mode="mute",
        )


def test_old_v1_receipt_is_never_reused(tmp_path):
    receipt = tmp_path / stitch.RECEIPT_FILENAME
    receipt.write_text(json.dumps({"schema": "duet.stitch", "version": 1}))
    assert stitch.receipt_is_v2(receipt) is False
