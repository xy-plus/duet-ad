from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import media


def _frame(index: int, pts: int, denominator: int = 1) -> dict:
    return {
        "decode_frame_index": index,
        "pts": pts,
        "time_base_num": 1,
        "time_base_den": denominator,
    }


def _scene(index: int, start: int, end: int) -> dict:
    return {
        "index": index,
        "start_s": float(start),
        "end_s": float(end),
        "frames": [_frame(value, value) for value in range(start, end)],
    }


def _png_bytes(width: int, height: int) -> bytes:
    image = np.full((height, width, 3), (10, 20, 30), dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    assert ok
    return encoded.tobytes()


def _assert_minimax_image_contract(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    assert 256 <= width <= 5760
    assert 256 <= height <= 5760
    assert 5 * width >= 2 * height
    assert 2 * width <= 5 * height


@pytest.mark.parametrize(
    ("width", "height", "expected_shape"),
    [
        (477, 848, (848, 477)),
        (512, 512, (256, 256)),
        (203, 360, (454, 256)),
        (200, 400, (512, 256)),
        (6000, 300, (1152, 2880)),
        (100, 1, (256, 256)),
        (1, 100, (256, 256)),
        (1285, 514, (258, 643)),
        (514, 1285, (643, 258)),
    ],
)
def test_provider_reference_png_uses_duet_minimax_bounds(
    tmp_path: Path,
    width: int,
    height: int,
    expected_shape: tuple[int, int],
) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "reference.png"
    source.write_bytes(_png_bytes(width, height))

    first = media.provider_reference_png(source, destination).read_bytes()
    second = media.provider_reference_png(source, destination).read_bytes()
    decoded = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)

    assert first == second
    assert decoded.shape[:2] == expected_shape
    _assert_minimax_image_contract(decoded)


def test_probe_uses_duration_ts_before_packet_fallback_and_never_format_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if "-show_packets" in argv:
            payload = {
                "packets": [
                    {"pts_time": "5.000", "duration_time": "0.040"},
                    {"pts_time": "5.040", "duration_time": "0.040"},
                ]
            }
        else:
            payload = {
                "format": {"duration": "99"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 64,
                        "height": 48,
                        "duration": "N/A",
                        "duration_ts": "25",
                        "time_base": "1/100",
                        "avg_frame_rate": "25/1",
                    }
                ],
            }
        return {
            "argv": argv,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": "",
        }

    monkeypatch.setattr(media, "run_command", fake_run)
    info = media.probe(tmp_path / "source.mp4")

    assert info["duration_s"] == pytest.approx(0.25)
    assert not any("-show_packets" in argv for argv in calls)


def test_probe_packet_timeline_is_last_visual_duration_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fake_run(argv, **_kwargs):
        payload = (
            {
                "packets": [
                    {"pts_time": "2.00", "duration_time": "0.04"},
                    {"pts_time": "2.04"},
                ]
            }
            if "-show_packets" in argv
            else {
                "format": {"duration": "99"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 64,
                        "height": 48,
                        "duration": "N/A",
                        "duration_ts": "N/A",
                        "time_base": "1/1000",
                        "avg_frame_rate": "25/1",
                        "r_frame_rate": "25/1",
                    }
                ],
            }
        )
        return {
            "argv": argv,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": "",
        }

    monkeypatch.setattr(media, "run_command", fake_run)

    assert media.probe(tmp_path / "source.mp4")["duration_s"] == pytest.approx(0.08)


def test_stream_start_packet_fallback_corrects_audio_skip_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    responses = iter(
        (
            {"streams": [{"sample_rate": "48000"}]},
            {
                "packets": [
                    {
                        "pts_time": "-0.021333333",
                        "side_data_list": [
                            {
                                "side_data_type": "Skip Samples",
                                "skip_samples": 1024,
                            }
                        ],
                    }
                ]
            },
        )
    )

    def fake_run(argv, **_kwargs):
        return {
            "argv": argv,
            "returncode": 0,
            "stdout": json.dumps(next(responses)),
            "stderr": "",
        }

    monkeypatch.setattr(media, "run_command", fake_run)

    assert media._stream_start_time(
        tmp_path / "source.mp4", "a:0"
    ) == pytest.approx(0.0, abs=1e-6)


def test_run_command_decodes_binary_diagnostics_with_replacement(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"out-\xff", stderr=b"err-\xfe"
        ),
    )

    result = media.run_command(["whisper-cli"], text=False)

    assert result["stdout"] == "out-\ufffd"
    assert result["stderr"] == "err-\ufffd"


def test_scene_first_planner_uses_latest_hard_cut_and_preserves_lineage():
    scenes = [
        _scene(1, 0, 4),
        _scene(2, 4, 9),
        _scene(3, 9, 15),
        _scene(4, 15, 20),
    ]

    planned = media.plan_segments(scenes, 20.0)

    assert [(item["start_s"], item["end_s"]) for item in planned] == [
        (0.0, 9.0),
        (9.0, 15.0),
        (15.0, 20.0),
    ]
    assert [item["join_mode"] for item in planned] == [
        "hard_cut",
        "hard_cut",
        "hard_cut",
    ]
    assert [item["chain_id"] for item in planned] == [
        "chain-001",
        "chain-002",
        "chain-003",
    ]
    assert planned[0]["scene_indices"] == [1, 2]
    assert [
        item["source_scene_id"] for item in planned[0]["source_cut_timeline"]
    ] == ["SCENE_01", "SCENE_02"]


def test_long_scene_fallback_and_short_tail_rollback_are_exact():
    scene = _scene(1, 0, 31)

    planned = media.plan_segments([scene], 31.0)

    assert [(item["start_s"], item["end_s"]) for item in planned] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 27.0),
        (27.0, 31.0),
    ]
    assert [item["join_mode"] for item in planned] == [
        "hard_cut",
        "continue",
        "continue",
        "continue",
    ]
    assert [item["chain_id"] for item in planned] == ["chain-001"] * 4


def test_dialogue_is_validation_only_and_crossing_text_is_split_by_overlap():
    scenes = [_scene(1, 0, 28)]
    line = {"text": "one sentence", "start_s": 9.0, "end_s": 11.0}

    without = media.plan_segments(scenes, 28.0)
    with_dialogue = media.plan_segments(scenes, 28.0, [line])

    assert with_dialogue == without
    first = media.localize_dialogue([line], with_dialogue[0], segments=with_dialogue)
    second = media.localize_dialogue([line], with_dialogue[1], segments=with_dialogue)
    assert first == [{"text": "one se", "start_s": 9.0, "end_s": 10.0}]
    assert second == [{"text": "ntence", "start_s": 0.0, "end_s": 1.0}]
    assert first[0]["text"] + second[0]["text"] == line["text"]


def test_dialogue_line_end_slightly_past_timeline_is_clamped():
    scenes = [_scene(1, 0, 28)]
    segments = media.plan_segments(scenes, 23.166666)
    # ASR follows the audio stream, which can end microseconds after the video.
    line = {"text": "tail", "start_s": 15.0, "end_s": 23.166688}

    localized = media.localize_dialogue([line], segments[-1], segments=segments)

    assert localized
    assert all(row["end_s"] <= 23.166666 for row in localized)


def test_hamilton_selection_uses_scene_anchors_and_ordinal_spacing():
    capacities = (1, 2, 10)
    scenes = []
    decode_index = 0
    for scene_index, capacity in enumerate(capacities, 1):
        frames = [
            _frame(decode_index + offset, decode_index + offset)
            for offset in range(capacity)
        ]
        scenes.append(
            {
                "index": scene_index,
                "start_s": float(decode_index),
                "end_s": float(decode_index + capacity),
                "frames": frames,
            }
        )
        decode_index += capacity

    selected = media.select_segment_keyframes(
        scenes, {"index": 1, "start_s": 0.0, "end_s": 13.0}
    )

    assert [
        sum(item["source_scene_id"] == f"SCENE_{index:02d}" for item in selected)
        for index in range(1, 4)
    ] == [1, 2, 6]
    assert [
        item["decode_frame_index"]
        for item in selected
        if item["source_scene_id"] == "SCENE_03"
    ] == [3, 5, 7, 8, 10, 12]


def test_sparse_selection_repeats_nearest_pts_deterministically():
    scenes = [
        {
            "index": 1,
            "start_s": 0.0,
            "end_s": 1.001,
            "frames": [
                _frame(0, 0, 10),
                _frame(1, 3, 10),
                _frame(2, 10, 10),
            ],
        }
    ]

    selected = media.select_segment_keyframes(
        scenes, {"index": 1, "start_s": 0.0, "end_s": 1.001}
    )

    assert [item["decode_frame_index"] for item in selected] == [
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    assert [item["repeated"] for item in selected] == [
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
    ]


def test_materialization_decodes_original_global_ordinals_in_order(tmp_path: Path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (48, 32)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder unavailable")
    for color in ((0, 0, 255), (0, 255, 0), (255, 0, 0)):
        writer.write(np.full((32, 48, 3), color, dtype=np.uint8))
    writer.release()
    indices = [0, 0, 1, 1, 1, 1, 2, 2, 2]
    selection = [
        {"order": order, "decode_frame_index": index}
        for order, index in enumerate(indices, 1)
    ]

    paths = media.materialize_keyframes(source, selection, tmp_path / "frames")

    assert [path.name for path in paths] == [f"{index:02d}.png" for index in range(1, 10)]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert paths[2].read_bytes() == paths[5].read_bytes()
    assert paths[6].read_bytes() == paths[8].read_bytes()
    assert len({paths[0].read_bytes(), paths[2].read_bytes(), paths[6].read_bytes()}) == 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_four_fps_extract_is_one_manifested_visual_timeline(tmp_path: Path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    manifest = media.extract_frames_4fps(source, tmp_path / "analysis")

    assert len(manifest["frames"]) == 5
    assert manifest["frames"][0]["time_seconds"] == 0.0
    assert manifest["frames"][-1]["time_seconds"] == pytest.approx(0.9)
    assert json.loads((tmp_path / "analysis" / "manifest.json").read_text()) == manifest
    assert all(
        (tmp_path / "analysis" / item["file"]).is_file()
        for item in manifest["frames"]
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_probe_real_video_ignores_longer_container_audio_timeline(tmp_path: Path):
    source = tmp_path / "long-audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=64x48:rate=24:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.8",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )

    info = media.probe(source)

    assert info["duration_s"] == pytest.approx(0.5, abs=0.02)
    assert info["has_audio"] is True


def test_visual_axis_audio_filter_uses_asetpts_resample_pad_and_trim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured = {}
    monkeypatch.setattr(
        media,
        "probe",
        lambda _path: {"has_audio": True, "duration_s": 6.25},
    )
    monkeypatch.setattr(media, "_stream_start_time", lambda _path, _stream: 1.5)

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(media, "run_command", fake_run)
    media.extract_audio(tmp_path / "source.mp4", tmp_path / "voice.wav")

    audio_filter = captured["argv"][captured["argv"].index("-filter:a") + 1]
    assert audio_filter == (
        "asetpts=PTS-(1.500000000)/TB,"
        "aresample=async=1:first_pts=0,apad,atrim=duration=6.250000000"
    )


def test_extract_audio_without_track_returns_none_and_removes_stale_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "voice.wav"
    destination.write_bytes(b"stale")
    monkeypatch.setattr(media, "probe", lambda _path: {"has_audio": False})
    monkeypatch.setattr(
        media,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("ffmpeg must not run without audio"),
    )

    assert media.extract_audio(tmp_path / "silent.mp4", destination) is None
    assert not destination.exists()


def test_transcribe_is_byte_safe_and_applies_master_non_dialogue_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"audio")
    destination = tmp_path / "voice_lines.json"
    texts = [
        "*tepuk tangan*",
        "［applause］",
        "♪ instrumental ♪",
        "ｕｍ, uh",
        "hmmm",
        "嗯……",
        "嗯，我知道了。",
        "Oh no!",
        "hello \ufffd world",
    ]
    payload = {
        "transcription": [
            {
                "text": text,
                "offsets": {"from": index * 1000, "to": (index + 1) * 1000},
            }
            for index, text in enumerate(texts)
        ]
    }

    def fake_run(argv, **kwargs):
        assert kwargs["text"] is False
        output = Path(argv[argv.index("-of") + 1]).with_suffix(".json")
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        output.write_bytes(raw.replace(b"hello ", b"hello \xff", 1))
        return {
            "argv": argv,
            "returncode": 0,
            "stdout": "diagnostic-\ufffd",
            "stderr": "",
        }

    monkeypatch.setattr(media, "run_command", fake_run)
    monkeypatch.setattr(media, "_probe_audio_duration", lambda _path: 20.0)

    lines, record = media.transcribe(
        audio,
        destination,
        tmp_path / "whisper-cli",
        tmp_path / "model.bin",
    )

    assert lines == [
        {"text": "嗯，我知道了。", "start_s": 6.0, "end_s": 7.0},
        {"text": "Oh no!", "start_s": 7.0, "end_s": 8.0},
        {"text": "hello  world", "start_s": 8.0, "end_s": 9.0},
    ]
    assert record["command"]["stdout"] == "diagnostic-\ufffd"
    assert json.loads(destination.read_text(encoding="utf-8")) == lines


def test_frame_budgets_are_cumulative_not_per_segment_rounding():
    assert media._frame_budgets([0.1, 0.1]) == [2, 3]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_concat_pads_short_outputs_and_normalizes_video_and_audio(tmp_path: Path):
    sources = []
    for index, color in enumerate(("red", "blue"), 1):
        source = tmp_path / f"source-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x48:r=24:d=0.05",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.05",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(source),
            ],
            check=True,
        )
        sources.append(source)

    output = tmp_path / "joined.mp4"
    result = media.concat(
        sources,
        output,
        target_durations=[1.0, 1.0],
        join_modes=["hard_cut", "continue"],
    )
    info = media.probe(output)

    assert result["frame_budgets"] == [24, 24]
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    assert info["fps"] == pytest.approx(24.0)
    assert info["duration_s"] == pytest.approx(2.0, abs=1 / 24)
    assert info["has_audio"] is True


def _sine_wav(path: Path, frequency: int, duration: float) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
            "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path


def _max_volume(path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    for line in result.stderr.splitlines():
        if "max_volume" in line:
            return float(line.split("max_volume:")[1].split("dB")[0].strip())
    raise AssertionError("volumedetect output is missing")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_assemble_dialogue_track_places_lines_on_a_silent_base(tmp_path: Path):
    first = _sine_wav(tmp_path / "line-0001.wav", 440, 0.3)
    second = _sine_wav(tmp_path / "line-0002.wav", 880, 0.5)

    destination = tmp_path / "full-track.wav"
    result = media.assemble_dialogue_track(
        [
            {"audio": first, "start_s": 0.1, "end_s": 0.5},
            {"audio": second, "start_s": 0.8, "end_s": 1.1},
        ],
        1.5,
        destination,
    )

    assert result["duration_s"] == 1.5
    assert [line["start_s"] for line in result["lines"]] == [0.1, 0.8]
    assert media._probe_audio_duration(destination) == pytest.approx(1.5, abs=0.05)
    # The 0.5s line is atempo-compressed into its 0.3s window, so the track
    # goes silent again before the end.
    assert _max_volume(destination) > -20.0
    tail = tmp_path / "tail.wav"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(destination),
            "-ss", "1.2", "-t", "0.3", "-c:a", "pcm_s16le", str(tail),
        ],
        check=True,
    )
    assert _max_volume(tail) < -60.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_silent_audio_track_has_requested_duration_and_no_signal(tmp_path: Path):
    destination = tmp_path / "no-dialogue-silence.wav"

    result = media.silent_audio_track(1.25, destination)

    assert result["duration_s"] == 1.25
    assert media._probe_audio_duration(destination) == pytest.approx(1.25, abs=0.05)
    assert _max_volume(destination) < -60.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_replace_audio_track_overwrites_audio_and_keeps_video(tmp_path: Path):
    video = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:size=64x48:rate=24:d=1.0",
            "-f", "lavfi", "-i",
            "sine=frequency=220:sample_rate=48000:duration=1.0",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video),
        ],
        check=True,
    )
    audio = _sine_wav(tmp_path / "full-track.wav", 880, 0.5)

    result = media.replace_audio_track(video, audio, video)

    assert result["duration_s"] == pytest.approx(1.0, abs=1 / 24)
    info = media.probe(video)
    assert info["codec_name"] == "h264"
    assert info["has_audio"] is True
    assert info["duration_s"] == pytest.approx(1.0, abs=0.05)
    assert _max_volume(video) > -30.0
