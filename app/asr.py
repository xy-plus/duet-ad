"""Local multilingual speech-to-text adapter for whisper.cpp.

Only the fixed binary/model selected by deployment are executed.  The adapter
returns the same three-field line schema consumed by the existing YAMNet and
prepared-input pipeline.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path


class ASRError(RuntimeError):
    _RETRYABLE_CODES = {"asr_timeout", "asr_failed", "asr_output_invalid"}

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = code in self._RETRYABLE_CODES


def _lines_from_json(payload: object, duration_s: float) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("transcription"), list):
        raise ASRError("asr_output_invalid")
    lines = []
    for segment in payload["transcription"]:
        if not isinstance(segment, dict):
            raise ASRError("asr_output_invalid")
        text = segment.get("text")
        offsets = segment.get("offsets")
        if not isinstance(text, str) or not isinstance(offsets, dict):
            raise ASRError("asr_output_invalid")
        start_ms, end_ms = offsets.get("from"), offsets.get("to")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, (int, float))
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, (int, float))
        ):
            raise ASRError("asr_output_invalid")
        # whisper.cpp can truncate a multibyte token at a segment boundary.
        # The JSON remains structurally usable after replacement decoding; do
        # not leak the replacement marker into the frozen dialogue prompt.
        text = text.replace("\ufffd", "").strip()
        start_s = max(0.0, float(start_ms) / 1000.0)
        end_s = min(duration_s, float(end_ms) / 1000.0)
        if text and math.isfinite(start_s) and math.isfinite(end_s) and start_s < end_s:
            lines.append({"text": text, "start_s": start_s, "end_s": end_s})
    return lines


def transcribe(
    audio: Path,
    *,
    cli: Path,
    model: Path,
    duration_s: float,
    timeout_s: int,
    threads: int,
) -> list[dict]:
    """Transcribe one audio file locally with automatic language detection."""
    if not cli.is_file() or not model.is_file():
        raise ASRError("asr_not_configured")
    with tempfile.TemporaryDirectory(prefix="duet-asr-") as raw_tmp:
        tmp = Path(raw_tmp)
        wav = tmp / "input.wav"
        output = tmp / "result"
        try:
            converted = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(audio), "-ar", "16000",
                    "-ac", "1", "-c:a", "pcm_s16le", "-y", str(wav),
                ],
                capture_output=True,
                timeout=min(timeout_s, 120),
            )
            if converted.returncode != 0:
                raise ASRError("asr_audio_convert_failed")
            completed = subprocess.run(
                [
                    str(cli), "-m", str(model), "-f", str(wav), "-l", "auto",
                    "-ojf", "-of", str(output), "-ng", "-t", str(max(1, threads)),
                ],
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise ASRError("asr_timeout") from None
        except OSError:
            raise ASRError("asr_unavailable") from None
        if completed.returncode != 0:
            raise ASRError("asr_failed")
        try:
            raw = output.with_suffix(".json").read_bytes()
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            raise ASRError("asr_output_invalid") from None
        return _lines_from_json(payload, duration_s)
