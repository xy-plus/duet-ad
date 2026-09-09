from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import media, subtitles


def _video(path: Path, audio: bool) -> None:
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=384x216:r=24:d=2"]
    if audio:
        argv += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000", "-c:a", "aac"]
    argv += ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path)]
    media.run_command(argv, timeout=30)


def _wav(path: Path, duration: float) -> None:
    media.run_command([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={duration}", "-c:a", "pcm_s16le", str(path),
    ], timeout=30)


def _frame(path: Path, seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        assert ok
        return frame
    finally:
        capture.release()


def _audio_hash(path: Path) -> str:
    return media.run_command([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-",
    ], timeout=30)["stdout"]


def test_cues_follow_actual_tts_duration_and_compressed_windows(tmp_path: Path) -> None:
    short, long = tmp_path / "short.wav", tmp_path / "long.wav"
    _wav(short, 0.25)
    _wav(long, 1.0)
    cues = subtitles.build_subtitle_cues([
        {"start_s": 0.1, "end_s": 1.0, "text": "short"},
        {"start_s": 0.8, "end_s": 1.3, "text": "compressed"},
        {"start_s": 1.8, "end_s": 2.8, "text": "clipped"},
        {"start_s": -0.1, "end_s": 0.4, "text": "negative"},
    ], [short, long, long, short], 2.0)
    assert [c["start_s"] for c in cues] == [0.1, 0.8, 1.8, 0]
    assert [c["end_s"] for c in cues] == pytest.approx([0.35, 1.3, 2.0, 0.25])
    assert subtitles.build_subtitle_cues([], [], 2.0) == []
    with pytest.raises(media.MediaError, match="count"):
        subtitles.build_subtitle_cues([{"text": "x"}], [], 2.0)
    with pytest.raises(media.MediaError, match="window"):
        subtitles.build_subtitle_cues([{"start_s": 1, "end_s": 0, "text": "x"}], [short], 2.0)
    with pytest.raises(media.MediaError, match="audio"):
        subtitles.build_subtitle_cues([{"start_s": 0, "end_s": 1, "text": "x"}], [tmp_path / "missing.wav"], 2.0)


@pytest.mark.parametrize("audio", [True, False])
def test_real_libass_renders_unicode_at_correct_times_and_preserves_media(tmp_path: Path, audio: bool) -> None:
    # Include filtergraph delimiters, quotes, colon and backslash in real paths.
    folder = tmp_path / "中文 quote' [:,;] back\\slash"
    folder.mkdir()
    source, output = folder / "source.mp4", folder / "captioned.mp4"
    _video(source, audio)
    result = subtitles.burn_subtitles(source, [{
        "start_s": 0.5, "end_s": 1.5,
        "text": "你好世界，这是中文自动换行字幕。Hello café Καλημέρα 😀 {\\an8}literal\\N\nDialogue: 0,0:00:00.00",
    }], output)
    assert result["subtitle_count"] == 1
    assert result["width"] == 384 and result["height"] == 216
    assert result["duration_s"] == pytest.approx(2.0, abs=1 / 24)
    assert result["frame_count"] == 48
    assert result["has_audio"] is audio
    assert _frame(output, 0.1).max() <= 4
    assert _frame(output, 1.8).max() <= 4
    middle = _frame(output, 1.0)
    assert (middle > 100).sum() > 100
    # User's top-alignment override must remain harmless text.
    assert middle[:45].max() <= 4
    assert middle[:, :8].max() <= 4 and middle[:, -8:].max() <= 4
    if audio:
        assert _audio_hash(source) == _audio_hash(output)
    assert not list(folder.glob(".subtitles-*"))


def test_empty_cues_copy_and_render_failure_keeps_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _video(source, False)
    subtitles.burn_subtitles(source, [], output)
    original = output.read_bytes()
    assert original == source.read_bytes()
    run = media.run_command

    def fail_renderer(argv, **kwargs):
        if argv[0] == "ffmpeg":
            raise subprocess.TimeoutExpired(argv, 600)
        return run(argv, **kwargs)

    monkeypatch.setattr(media, "run_command", fail_renderer)
    with pytest.raises(subprocess.TimeoutExpired):
        subtitles.burn_subtitles(source, [{"start_s": 0, "end_s": 1, "text": "hello"}], output)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".subtitles-*"))
    with pytest.raises(media.MediaError, match="separate"):
        subtitles.burn_subtitles(source, [], source)


def test_plain_text_cannot_create_ass_commands_or_events() -> None:
    text = subtitles._plain_ass_text("{\\an8}hi\\Nnext\nDialogue: evil\r\x00")
    assert "{" not in text and "}" not in text and "\\" not in text
    assert "\n" not in text and "\r" not in text and "\x00" not in text
    assert "｛＼an8｝hi＼Nnext" in text
    wrapped = subtitles._wrapped_text("这是一个没有空格且需要自动换行的中文台词测试", 12)
    assert r"\N" in wrapped
    assert all(len(row) <= 6 for row in wrapped.split(r"\N"))


def test_excessively_long_chinese_fails_without_truncating_or_replacing_output(tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _video(source, False)
    subtitles.burn_subtitles(source, [], output)
    original = output.read_bytes()
    with pytest.raises(media.MediaError, match="readable caption area"):
        subtitles.burn_subtitles(source, [{"start_s": 0, "end_s": 1, "text": "超长台词" * 300}], output)
    assert output.read_bytes() == original


def test_missing_required_font_fails_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _video(source, False)
    run = media.run_command

    def fallback_font(argv, **kwargs):
        if argv[0] == "fc-match":
            return {"stdout": "DejaVu Sans"}
        return run(argv, **kwargs)

    monkeypatch.setattr(media, "run_command", fallback_font)
    with pytest.raises(media.MediaError, match="fonts-noto-cjk"):
        subtitles.burn_subtitles(source, [{"start_s": 0, "end_s": 1, "text": "中文"}], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".subtitles-*"))
