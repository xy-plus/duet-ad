from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import media, subtitles

LANDSCAPE = {"width": 1024, "height": 576, "duration_s": 60.0}
PORTRAIT = {"width": 576, "height": 1024, "duration_s": 60.0}


def _parse_cs(stamp: str) -> int:
    hours, minutes, rest = stamp.split(":")
    seconds, centiseconds = rest.split(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 100 + int(centiseconds)


def _events(document: str) -> list[tuple[int, int, list[str]]]:
    events = []
    for line in document.splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(",", 9)
            events.append((_parse_cs(fields[1]), _parse_cs(fields[2]), fields[9].split(r"\N")))
    return events


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
    # The long line wraps past two rows and is split into two contiguous events.
    assert result["subtitle_count"] == 2
    assert result["width"] == 384 and result["height"] == 216
    assert result["duration_s"] == pytest.approx(2.0, abs=1 / 24)
    assert result["frame_count"] == 48
    assert result["has_audio"] is audio
    assert _frame(output, 0.1).max() <= 4
    assert _frame(output, 1.8).max() <= 4
    for seconds in (0.75, 1.25):
        middle = _frame(output, seconds)
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


def test_short_cue_stays_single_event_with_inward_rounded_window() -> None:
    document, count = subtitles._ass_document([
        {"start_s": 1.004, "end_s": 2.996, "text": "短句"},
        {"start_s": 3.0, "end_s": 4.0, "text": "   "},
        {"start_s": 5.0, "end_s": 6.0, "text": ""},
    ], LANDSCAPE)
    events = _events(document)
    assert count == 1 and len(events) == 1
    start, end, rows = events[0]
    assert (start, end) == (101, 299)
    assert rows == ["短句"]


def test_long_cue_splits_into_two_row_chunks_with_weighted_contiguous_windows() -> None:
    # 50 full-width chars then 50 narrow chars: three rows, two chunks whose
    # weights are 100 and 50, so the window splits ~2:1.
    document, count = subtitles._ass_document([
        {"start_s": 2.0, "end_s": 6.0, "text": "汉" * 50 + "a" * 50},
    ], LANDSCAPE)
    events = _events(document)
    assert count == 2 and len(events) == 2
    (start_a, end_a, rows_a), (start_b, end_b, rows_b) = events
    assert len(rows_a) == 2 and len(rows_b) == 1
    assert (start_a, end_b) == (200, 600)
    assert end_a == start_b
    total_weight = 150
    assert abs((end_a - start_a) - 400 * 100 / total_weight) <= 1
    assert abs((end_b - start_b) - 400 * 50 / total_weight) <= 1


def test_mixed_script_weight_gives_cjk_chunks_more_time() -> None:
    document, _ = subtitles._ass_document([
        {"start_s": 0.0, "end_s": 9.0, "text": "字" * 50 + "x" * 50},
    ], LANDSCAPE)
    (start_a, end_a, rows_a), (start_b, end_b, rows_b) = _events(document)
    assert all("字" in row for row in rows_a)
    assert all("x" in row for row in rows_b)
    assert end_a - start_a > 2 * (end_b - start_b) - 1


def test_split_never_exceeds_two_rows_in_portrait_or_landscape() -> None:
    text = "a" * 300
    for info in (LANDSCAPE, PORTRAIT):
        document, _ = subtitles._ass_document(
            [{"start_s": 0.0, "end_s": 10.0, "text": text}], info)
        events = _events(document)
        assert len(events) > 1
        assert all(len(rows) <= 2 for _, _, rows in events)
        assert events[0][0] == 0 and events[-1][1] == 1000
        for (_, end, _), (next_start, _, _) in zip(events, events[1:]):
            assert end == next_start


def test_wider_landscape_columns_split_into_fewer_events_than_portrait() -> None:
    text = "a" * 300
    _, landscape_count = subtitles._ass_document(
        [{"start_s": 0.0, "end_s": 10.0, "text": text}], LANDSCAPE)
    _, portrait_count = subtitles._ass_document(
        [{"start_s": 0.0, "end_s": 10.0, "text": text}], PORTRAIT)
    assert 1 < landscape_count < portrait_count


def test_tiny_window_never_produces_empty_events() -> None:
    # 6 rows want 3 chunks; a 3cs window can host exactly 3 non-empty events.
    document, count = subtitles._ass_document(
        [{"start_s": 0.0, "end_s": 0.03, "text": "a" * 300}], LANDSCAPE)
    events = _events(document)
    assert count == 3
    assert [start for start, _, _ in events] == [0, 1, 2]
    assert [end for _, end, _ in events] == [1, 2, 3]
    # A 1cs window cannot host 3 chunks: fall back to one tall event rather
    # than dropping text or emitting empty windows.
    document, count = subtitles._ass_document(
        [{"start_s": 0.0, "end_s": 0.01, "text": "a" * 300}], LANDSCAPE)
    events = _events(document)
    assert count == 1 and events[0][:2] == (0, 1)
    assert sum(len(row) for row in events[0][2]) == 300


def test_caption_area_too_small_still_fails() -> None:
    with pytest.raises(media.MediaError, match="caption area too small"):
        subtitles._ass_document([{"start_s": 0, "end_s": 1, "text": "hi"}],
                                {"width": 200, "height": 100, "duration_s": 2.0})


def test_excessively_long_chinese_splits_in_time_instead_of_failing(tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _video(source, False)
    result = subtitles.burn_subtitles(
        source, [{"start_s": 0, "end_s": 1, "text": "超长台词" * 300}], output)
    assert result["subtitle_count"] > 4
    assert result["width"] == 384 and result["height"] == 216
    assert not list(tmp_path.glob(".subtitles-*"))


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
