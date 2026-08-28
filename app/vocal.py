"""口播台词声学验证：用 YAMNet 区分口播、唱歌与背景音乐。

本模块只负责音频解码、模型推理和纯函数判定；PipelineError 由 pipeline.py
归口，异常类放在本模块以便延迟导入时不形成循环。
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600
_FFMPEG_TIMEOUT_S = 120

YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
SPOKEN_INDICES = (0, 1, 2, 3, 4, 5, 12)
SUNG_INDICES = tuple(range(24, 33))
MUSIC_INDICES = (132, 261, 262, 263, 264, 265)

# 校准依据：TrendScout 2026-07-28 盘上 7 支素材实测。音乐组分达到 0.1
# 的窗口占比在无 BGM/有 BGM 两簇之间留有余量，因此沿用 0.1 与 0.5。
MUSIC_SCORE_MIN = 0.1
MUSIC_WINDOW_RATIO = 0.5
# 校准依据：同一批素材的真口播最低峰值窗为 0.414，纯 BGM 假转录最高为
# 0.082；沿用 Speech 分 0.2 与覆盖比 0.2。
SPEECH_SCORE_MIN = 0.2
SPEECH_WINDOW_RATIO = 0.2


class VocalError(RuntimeError):
    """音频、YAMNet 模型或声学分类结果无效。"""


@dataclass(frozen=True)
class VocalWindow:
    """一个 YAMNet 窗口及其三类聚合分数。"""

    start_ms: int
    end_ms: int
    sung: float
    spoken: float
    music: float


class WindowScores(NamedTuple):
    """从 521 个 AudioSet 类聚合出的判定分数。"""

    sung: float
    spoken: float
    music: float


@dataclass(frozen=True)
class VocalAnalysis:
    """一次音频解码和模型遍历的结果。"""

    windows: list[VocalWindow]
    has_bgm: bool


def group_scores(scores: Sequence[float]) -> WindowScores:
    """校验并聚合 YAMNet 的 521 类输出为唱歌、口播、音乐分数。"""
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
    """Music ≥ 0.1 的窗口占比达到 0.5 即判定整片有 BGM；无窗口为 False。"""
    if not windows:
        return False
    scored = sum(1 for window in windows if window.music >= MUSIC_SCORE_MIN)
    return scored >= len(windows) * MUSIC_WINDOW_RATIO


def classify_segment(
    start_ms: int,
    end_ms: int,
    windows: Sequence[VocalWindow],
) -> str | None:
    """按重叠时长判定片段为 sung、spoken 或 None（没有人声证据）。"""
    overlapping = [
        window
        for window in windows
        if window.start_ms < end_ms and window.end_ms > start_ms
    ]
    if not overlapping:
        raise VocalError("转录片段没有重叠的声学分类窗口")
    values = [value for window in overlapping for value in (window.sung, window.spoken)]
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
    """读取模型路径：环境变量优先，否则使用仓库内置模型。"""
    return Path(os.environ["YAMNET_MODEL_PATH"]) if os.environ.get("YAMNET_MODEL_PATH") else ROOT / "models" / "yamnet.tflite"


def _verify_model(path: Path) -> None:
    """校验模型存在且与校准过的 YAMNet 文件逐字节一致。"""
    if not path.is_file():
        raise VocalError("YAMNet 模型不存在")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise VocalError("YAMNet 模型读取失败") from None
    if digest != YAMNET_SHA256:
        raise VocalError("YAMNet 模型校验失败")


def _decode_waveform(audio: Path):
    """用 ffmpeg 解码为 16kHz、单声道、little-endian float32 波形。"""
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
    """加载一次 YAMNet，并对每个 15600 样本窗口推理。"""
    try:
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path), num_threads=1)
        input_detail = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_detail["index"], [WINDOW_SAMPLES], strict=True)
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
                input_detail["index"], np.asarray(samples, dtype=input_detail["dtype"])
            )
            interpreter.invoke()
            output = np.asarray(interpreter.get_tensor(score_output["index"])).reshape(-1)
            scores = group_scores(output)
            windows.append(
                VocalWindow(
                    start_ms=round(offset * 1000 / SAMPLE_RATE),
                    end_ms=round(min(offset + WINDOW_SAMPLES, len(waveform)) * 1000 / SAMPLE_RATE),
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


def analyze(audio: Path) -> VocalAnalysis:
    """解码音频并用内置 YAMNet 返回窗口分数和整片 BGM 判定。"""
    model_path = _model_path()
    _verify_model(model_path)
    waveform = _decode_waveform(audio)
    windows = _analyze_windows(waveform, model_path)
    return VocalAnalysis(windows=windows, has_bgm=detect_background_music(windows))
