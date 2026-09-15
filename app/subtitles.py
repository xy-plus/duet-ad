"""Local subtitle timing and FFmpeg/libass rendering; no model calls."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app import media


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise media.MediaError(f"invalid subtitle {label}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise media.MediaError(f"invalid subtitle {label}") from None
    if not math.isfinite(result):
        raise media.MediaError(f"invalid subtitle {label}")
    return result


def build_subtitle_cues(
    voice_lines: Sequence[Mapping[str, object]],
    line_audio_paths: Sequence[Path],
    duration_s: float,
) -> list[dict[str, Any]]:
    """Match dialogue-track placement, including compression of long TTS lines.

    Audio files must be supplied in the same order as translated voice lines.
    Short speech ends at its real audio duration, not the end of its window.
    """
    total = _number(duration_s, "duration")
    if total <= 0 or len(voice_lines) != len(line_audio_paths):
        raise media.MediaError("invalid subtitle duration or line/audio count")
    cues: list[dict[str, Any]] = []
    for line, audio in zip(voice_lines, line_audio_paths):
        if not isinstance(line, Mapping) or not isinstance(line.get("text"), str):
            raise media.MediaError("invalid subtitle line")
        start = _number(line.get("start_s"), "start")
        end = _number(line.get("end_s"), "end")
        window = end - start
        if window <= 0:
            raise media.MediaError("invalid subtitle time window")
        audio_duration = media._dialogue_line_duration(audio)
        start = max(0.0, start)
        end = min(total, start + min(audio_duration, window))
        text = line["text"].strip()
        if text and end > start:
            cues.append({"start_s": start, "end_s": end, "text": text})
    return cues


def _plain_ass_text(text: str) -> str:
    # ASS has no portable literal escaping across parsers for all three syntax
    # characters. Full-width lookalikes preserve legibility without ever opening
    # an override block or interpreting user-supplied \N, \h, drawing commands.
    text = text.translate(str.maketrans({"{": "｛", "}": "｝", "\\": "＼"}))
    return " ".join(
        "".join(
            char if not unicodedata.category(char).startswith("C") else " "
            for char in text
        ).split()
    )


def _char_weight(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _wrapped_text(text: str, columns: int) -> str:
    # Explicit wrapping also supports libass builds without libunibreak.
    # Prefer word boundaries; split uninterrupted CJK or very long words.
    rows: list[str] = []
    current = ""
    width = 0
    for char in _plain_ass_text(text):
        weight = _char_weight(char)
        if current and width + weight > columns:
            split = current.rfind(" ")
            if split > len(current) // 2:
                rows.append(current[:split])
                current = current[split + 1:]
                width = sum(_char_weight(c) for c in current)
            else:
                rows.append(current.rstrip())
                current = ""
                width = 0
        current += char
        width += weight
    if current.strip():
        rows.append(current.strip())
    return r"\N".join(rows)


# A cue never shows more than two rows at once; longer text is split in time.
MAX_ROWS = 2


def _split_cue(start_cs: int, end_cs: int, rows: list[str]) -> list[tuple[int, int, str]]:
    """Split wrapped rows into contiguous sub-cues of at most MAX_ROWS rows.

    The window is divided proportionally to each chunk's East Asian character
    weight (same weighting as _wrapped_text). Boundaries are shared between
    neighbours, so chunks tile [start_cs, end_cs] without gaps or overlap.
    """
    window = end_cs - start_cs
    # Each chunk needs at least 1cs to display at all. A window shorter than
    # the chunk count is pathological (seconds of text in a centisecond); fall
    # back to fewer, taller chunks rather than dropping text or emitting
    # empty windows, accepting more than MAX_ROWS rows in that case only.
    chunk_count = min(-(-len(rows) // MAX_ROWS), window)
    if chunk_count <= 1:
        return [(start_cs, end_cs, r"\N".join(rows))]
    per, extra = divmod(len(rows), chunk_count)
    groups: list[list[str]] = []
    index = 0
    for i in range(chunk_count):
        size = per + (1 if i < extra else 0)
        groups.append(rows[index:index + size])
        index += size
    weights = [max(1, sum(_char_weight(c) for row in group for c in row)) for group in groups]
    total = sum(weights)
    events: list[tuple[int, int, str]] = []
    boundary, cumulative = start_cs, 0
    for i, (group, weight) in enumerate(zip(groups, weights)):
        remaining = chunk_count - i - 1
        cumulative += weight
        # Round-half-up of the ideal proportional boundary, clamped so every
        # remaining chunk keeps at least 1cs inside the original window.
        ideal = start_cs + (window * cumulative + total // 2) // total
        next_boundary = end_cs if not remaining else min(max(ideal, boundary + 1), end_cs - remaining)
        events.append((boundary, next_boundary, r"\N".join(group)))
        boundary = next_boundary
    return events


def _timestamp(centiseconds: int) -> str:
    seconds, fraction = divmod(centiseconds, 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}"


def _ass_document(cues: Sequence[Mapping[str, object]], info: Mapping[str, Any]) -> tuple[str, int]:
    width, height = info["width"], info["height"]
    # Size relative to the short edge keeps portrait captions comfortably sized.
    font_size = max(12, round(min(width, height) * 0.052))
    margin = max(8, round(width * 0.055))
    bottom = max(8, round(height * 0.065))
    prepared: list[tuple[int, int, str]] = []
    for cue in cues:
        if not isinstance(cue, Mapping) or not isinstance(cue.get("text"), str):
            raise media.MediaError("invalid subtitle cue")
        start = max(0.0, _number(cue.get("start_s"), "start"))
        end = min(info["duration_s"], _number(cue.get("end_s"), "end"))
        # Round inward so quantization never displays outside the cue window.
        start_cs, end_cs = math.ceil(start * 100), math.floor(end * 100)
        if end_cs > start_cs and _plain_ass_text(cue["text"]):
            prepared.append((start_cs, end_cs, cue["text"]))
    # Long text is split in time below, so shrinking only needs to fit two
    # rows vertically (ultrawide frames); the cap is a hard guarantee.
    while font_size > 12 and MAX_ROWS * font_size * 1.5 > height * 0.30:
        font_size -= 1
    if MAX_ROWS * font_size * 1.5 > height * 0.30:
        raise media.MediaError("subtitle caption area too small for legible captions")
    columns = max(8, int((width - margin * 2) / (font_size * 0.60)))
    events: list[tuple[int, int, str]] = []
    for start_cs, end_cs, text in prepared:
        events.extend(_split_cue(start_cs, end_cs, _wrapped_text(text, columns).split(r"\N")))
    document = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Noto Sans CJK SC,{font_size},&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{max(1, font_size / 16):.2f},0,2,{margin},{margin},{bottom},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    for start_cs, end_cs, text in events:
        document += f"Dialogue: 0,{_timestamp(start_cs)},{_timestamp(end_cs)},Default,,0,0,0,,{text}\n"
    return document, len(events)


def _require_font() -> None:
    try:
        result = media.run_command(
            ["fc-match", "-f", "%{family}", "Noto Sans CJK SC"], timeout=10,
        )
    except (FileNotFoundError, media.CommandError) as exc:
        raise media.MediaError("subtitle rendering requires fontconfig and Noto Sans CJK SC") from exc
    if "Noto Sans CJK SC" not in result["stdout"].split(","):
        raise media.MediaError("subtitle rendering requires the Noto Sans CJK SC font; install fonts-noto-cjk")


def _filter_filename(path: Path) -> str:
    # Escape both the filter-option layer and the filtergraph layer. No shell.
    first = str(path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return "".join("\\" + c if c in "\\'[],;" else c for c in first)


def burn_subtitles(
    video_path: Path,
    cues: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, Any]:
    """Burn captions into a separate MP4 atomically, preserving audio packets.

    Empty cues copy the original byte-for-byte. Existing output survives a
    renderer failure. H.264 quality matches media's default CRF 23 encoding.
    """
    if video_path.is_symlink() or not video_path.is_file():
        raise media.MediaError("invalid subtitle source video")
    video_path = video_path.resolve()
    if output_path.is_symlink() or output_path.resolve() == video_path:
        raise media.MediaError("subtitle output must be a separate regular file")
    if output_path.suffix.lower() != ".mp4":
        raise media.MediaError("subtitle output must be MP4")
    info = media.probe(video_path)
    document, count = _ass_document(cues, info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.absolute()
    with tempfile.TemporaryDirectory(prefix=".subtitles-", dir=output_path.parent) as scratch:
        scratch_path = Path(scratch)
        rendered = scratch_path / "rendered.mp4"
        result: dict[str, Any] = {}
        if count:
            _require_font()
            ass_path = scratch_path / "captions.ass"
            ass_path.write_text(document, encoding="utf-8")
            result = media.run_command(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-threads", "2", "-i", str(video_path),
                    "-map", "0:v:0", "-map", "0:a?",
                    "-filter_threads", "1", "-vf", f"ass=filename={_filter_filename(ass_path)}",
                    "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                    "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    "-movflags", "+faststart", str(rendered),
                ],
                timeout=600,
            )
        else:
            shutil.copyfile(video_path, rendered)
        rendered_info = media.probe(rendered)
        if (
            (rendered_info["width"], rendered_info["height"]) != (info["width"], info["height"])
            or abs(rendered_info["duration_s"] - info["duration_s"]) > max(0.05, 1 / max(info["fps"], 1))
            or rendered_info["has_audio"] != info["has_audio"]
        ):
            raise media.MediaError("subtitle output failed media preservation check")
        os.replace(rendered, output_path)
    return {**result, **rendered_info, "subtitle_count": count, "output_path": str(output_path)}
