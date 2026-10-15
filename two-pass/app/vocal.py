"""YAMNet vocal classification copied from the frozen Master baseline.

This module owns only deterministic audio decoding, inference, and per-ASR-line
classification.  It deliberately has no pipeline, provider, receipt, or retry
dependencies.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600
_FFMPEG_TIMEOUT_S = 120

YAMNET_SIZE_BYTES = 4_126_810
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
SPOKEN_INDICES = (0, 1, 2, 3, 4, 5, 12)
SUNG_INDICES = tuple(range(24, 33))
MUSIC_INDICES = (132, 261, 262, 263, 264, 265)

MUSIC_SCORE_MIN = 0.1
MUSIC_WINDOW_RATIO = 0.5
SPEECH_SCORE_MIN = 0.2
SPEECH_WINDOW_RATIO = 0.2
EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN = 51 / 256


class VocalError(RuntimeError):
    """The audio, calibrated model, or acoustic result is invalid."""


@dataclass(frozen=True)
class VocalWindow:
    start_ms: int
    end_ms: int
    sung: float
    spoken: float
    music: float


class WindowScores(NamedTuple):
    sung: float
    spoken: float
    music: float


@dataclass(frozen=True)
class VocalAnalysis:
    windows: list[VocalWindow]
    has_bgm: bool


def group_scores(scores: Sequence[float]) -> WindowScores:
    """Collapse the 521 AudioSet classes to the calibrated vocal groups."""
    try:
        if getattr(scores, "ndim", 1) != 1 or len(scores) != 521:
            raise VocalError("声学模型输出维度无效")
    except TypeError:
        raise VocalError("声学模型输出维度无效") from None
    try:
        values = [float(value) for value in scores]
    except (TypeError, ValueError):
        raise VocalError("声学模型输出必须是数值") from None
    if not all(math.isfinite(value) for value in values):
        raise VocalError("声学模型输出必须是有限数")
    if any(value < 0 or value > 1 for value in values):
        raise VocalError("声学模型输出必须在 0 到 1 之间")
    return WindowScores(
        sung=max(values[index] for index in SUNG_INDICES),
        spoken=max(values[index] for index in SPOKEN_INDICES),
        music=max(values[index] for index in MUSIC_INDICES),
    )


def detect_background_music(windows: Sequence[VocalWindow]) -> bool:
    if not windows:
        return False
    scored = sum(1 for window in windows if window.music >= MUSIC_SCORE_MIN)
    return scored >= len(windows) * MUSIC_WINDOW_RATIO


def classify_segment(
    start_ms: int,
    end_ms: int,
    windows: Sequence[VocalWindow],
) -> str | None:
    """Classify an ASR interval using exact overlap-duration weighting.

    Missing acoustic coverage is an absence of classification evidence, not an
    invalid ASR result.  Preserve the line for provenance and let callers move
    on with an unclassified result instead of failing the whole project.
    """
    overlapping = [
        window
        for window in windows
        if window.start_ms < end_ms and window.end_ms > start_ms
    ]
    if not overlapping:
        return None
    values = [
        value
        for window in overlapping
        for value in (window.sung, window.spoken)
    ]
    if not all(math.isfinite(value) for value in values):
        raise VocalError("声学分类分数必须是有限数")

    weights = [
        min(end_ms, window.end_ms) - max(start_ms, window.start_ms)
        for window in overlapping
    ]
    total_overlap = sum(weights)
    if total_overlap <= 0:
        raise VocalError("转录片段没有有效的声学分类重叠")
    weighted_mean_sung = sum(
        window.sung * weight for window, weight in zip(overlapping, weights)
    ) / total_overlap
    weighted_mean_spoken = sum(
        window.spoken * weight for window, weight in zip(overlapping, weights)
    ) / total_overlap
    sung_overlap_ratio = sum(
        weight
        for window, weight in zip(overlapping, weights)
        if window.sung > window.spoken
    ) / total_overlap
    if (
        weighted_mean_sung >= 0.04
        and weighted_mean_sung + 1e-12 >= weighted_mean_spoken + 0.02
        and sung_overlap_ratio >= 0.5
    ):
        return "sung"

    speech_overlap_ratio = sum(
        weight
        for window, weight in zip(overlapping, weights)
        if window.spoken >= SPEECH_SCORE_MIN
    ) / total_overlap
    if speech_overlap_ratio >= SPEECH_WINDOW_RATIO:
        return "spoken"
    return None


def _model_path() -> Path:
    configured = os.environ.get("YAMNET_MODEL_PATH")
    return Path(configured) if configured else ROOT / "models" / "yamnet.tflite"


def _verify_model(path: Path) -> None:
    if not path.is_file():
        raise VocalError("YAMNet 模型不存在")
    try:
        if path.stat().st_size != YAMNET_SIZE_BYTES:
            raise VocalError("YAMNet 模型校验失败")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except VocalError:
        raise
    except OSError:
        raise VocalError("YAMNet 模型读取失败") from None
    if digest != YAMNET_SHA256:
        raise VocalError("YAMNet 模型校验失败")


def _decode_waveform(audio: Path):
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(audio), "-vn", "-ac", "1",
                "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
            ],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise VocalError(f"音频解码超时（{_FFMPEG_TIMEOUT_S}s）") from None
    except OSError:
        raise VocalError("音频解码失败") from None
    if completed.returncode != 0:
        raise VocalError("音频解码失败")
    try:
        import numpy as np

        waveform = np.frombuffer(completed.stdout, dtype="<f4")
    except (ImportError, TypeError, ValueError):
        raise VocalError("音频解码结果无效") from None
    if not waveform.size or not np.isfinite(waveform).all():
        raise VocalError("音频解码结果无效")
    return waveform


def _analyze_windows(waveform, model_path: Path) -> list[VocalWindow]:
    try:
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        input_detail = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(
            input_detail["index"], [WINDOW_SAMPLES], strict=True
        )
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        score_outputs = [
            detail
            for detail in interpreter.get_output_details()
            if int(detail["shape"][-1]) == 521
        ]
        if len(score_outputs) != 1:
            raise VocalError("声学模型输出维度无效")
        score_output = score_outputs[0]
        windows: list[VocalWindow] = []
        for offset in range(0, len(waveform), WINDOW_SAMPLES):
            samples = waveform[offset:offset + WINDOW_SAMPLES]
            if len(samples) < WINDOW_SAMPLES:
                samples = np.pad(samples, (0, WINDOW_SAMPLES - len(samples)))
            interpreter.set_tensor(
                input_detail["index"],
                np.asarray(samples, dtype=input_detail["dtype"]),
            )
            interpreter.invoke()
            output = np.asarray(
                interpreter.get_tensor(score_output["index"])
            ).reshape(-1)
            scores = group_scores(output)
            windows.append(
                VocalWindow(
                    start_ms=round(offset * 1000 / SAMPLE_RATE),
                    end_ms=round(
                        min(offset + WINDOW_SAMPLES, len(waveform))
                        * 1000
                        / SAMPLE_RATE
                    ),
                    sung=float(scores.sung),
                    spoken=float(scores.spoken),
                    music=float(scores.music),
                )
            )
        return windows
    except VocalError:
        raise
    except Exception as exc:
        raise VocalError("声学模型执行失败") from exc


def analyze(audio: Path, model_path: Path | None = None) -> VocalAnalysis:
    """Decode audio once and run the frozen calibrated YAMNet model."""
    selected_model = Path(model_path) if model_path is not None else _model_path()
    _verify_model(selected_model)
    waveform = _decode_waveform(audio)
    windows = _analyze_windows(waveform, selected_model)
    return VocalAnalysis(windows=windows, has_bgm=detect_background_music(windows))


def analysis_from_mapping(value: Mapping[str, object]) -> VocalAnalysis:
    """Restore the JSON-friendly result returned by ``media.yamnet_analyze``."""
    raw_windows = value.get("windows")
    if not isinstance(raw_windows, list):
        raise VocalError("声学分类结果无效")
    try:
        windows = [
            VocalWindow(
                start_ms=int(item["start_ms"]),
                end_ms=int(item["end_ms"]),
                sung=float(item["sung"]),
                spoken=float(item["spoken"]),
                music=float(item["music"]),
            )
            for item in raw_windows
            if isinstance(item, Mapping)
        ]
    except (KeyError, TypeError, ValueError):
        raise VocalError("声学分类结果无效") from None
    if len(windows) != len(raw_windows):
        raise VocalError("声学分类结果无效")
    return VocalAnalysis(windows=windows, has_bgm=bool(value.get("has_bgm")))


def classify_line(
    line: Mapping[str, object],
    analysis: VocalAnalysis,
    *,
    only_line: bool = False,
) -> str | None:
    """Apply Master's interval rule plus its one-line timestamp-drift fallback."""
    try:
        classification = classify_segment(
            int(float(line["start_s"]) * 1000),
            int(float(line["end_s"]) * 1000),
            analysis.windows,
        )
    except (KeyError, TypeError, ValueError):
        raise VocalError("转录片段时间无效") from None
    if classification is not None or not only_line or not analysis.windows:
        return classification
    max_spoken = max(window.spoken for window in analysis.windows)
    max_sung = max(window.sung for window in analysis.windows)
    if max_sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN and max_sung > max_spoken:
        return "sung"
    if max_spoken >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN:
        return "spoken"
    if max_sung >= EMPTY_TRANSCRIPT_VOCAL_EVIDENCE_MIN:
        return "sung"
    return None
