"""YAMNet 口播/唱歌/BGM 声学验证测试。"""

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app import vocal


def _window(start_ms, end_ms, *, sung=0.0, spoken=0.0, music=0.0):
    return vocal.VocalWindow(start_ms, end_ms, sung, spoken, music)


def _scores(values):
    scores = [0.0] * 521
    for index, value in values.items():
        scores[index] = value
    return scores


class TestGroupScores:
    def test_collapses_521_classes(self):
        got = vocal.group_scores(
            _scores({0: 0.11, 5: 0.42, 12: 0.31, 24: 0.17, 32: 0.53, 132: 0.29})
        )
        assert got == vocal.WindowScores(sung=0.53, spoken=0.42, music=0.29)

    @pytest.mark.parametrize("scores", [[0.0] * 520, [0.0] * 522])
    def test_rejects_wrong_dimension(self, scores):
        with pytest.raises(vocal.VocalError, match="维度"):
            vocal.group_scores(scores)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
    def test_rejects_invalid_values(self, value):
        with pytest.raises(vocal.VocalError, match="有限数|0 到 1"):
            vocal.group_scores(_scores({0: value}))


class TestBackgroundMusic:
    def test_empty_is_false(self):
        assert vocal.detect_background_music([]) is False

    def test_score_floor_and_ratio_boundaries_are_inclusive(self):
        # 地板 0.1 恰在界上算达标；占比 1/2 = 0.5 恰在界上 → True
        assert vocal.detect_background_music(
            [_window(0, 100, music=0.1), _window(100, 200, music=0.0)]
        ) is True
        # 0.099 不达标；达标占比 1/3 < 0.5 → False
        assert vocal.detect_background_music(
            [_window(0, 100, music=0.1), _window(100, 200, music=0.099),
             _window(200, 300, music=0.0)]
        ) is False
        # 达标占比 2/4 = 0.5 恰在界上 → True
        assert vocal.detect_background_music(
            [_window(0, 100, music=0.1), _window(100, 200, music=0.1),
             _window(200, 300, music=0.0), _window(300, 400, music=0.0)]
        ) is True


class TestClassifySegment:
    def test_sung_three_conditions_at_boundary(self):
        windows = [
            _window(0, 500, sung=0.08, spoken=0.02),
            _window(500, 1000, sung=0.01, spoken=0.02),
        ]
        assert vocal.classify_segment(0, 1000, windows) == "sung"

    def test_sung_requires_half_overlap_ratio(self):
        windows = [
            _window(0, 499, sung=0.08, spoken=0.02),
            _window(499, 1000, sung=0.01, spoken=0.02),
        ]
        assert vocal.classify_segment(0, 1000, windows) is None

    def test_sung_mean_and_margin_are_required(self):
        assert vocal.classify_segment(
            0, 1000, [_window(0, 1000, sung=0.039999, spoken=0.0)]
        ) is None
        assert vocal.classify_segment(
            0, 1000, [_window(0, 1000, sung=0.04, spoken=0.02000000001)]
        ) is None

    def test_spoken_coverage_boundary_is_inclusive(self):
        windows = [
            _window(0, 200, spoken=0.2),
            _window(200, 1000, spoken=0.1),
        ]
        assert vocal.classify_segment(0, 1000, windows) == "spoken"
        windows[0] = _window(0, 199, spoken=0.2)
        assert vocal.classify_segment(0, 1000, windows) is None

    def test_no_overlap_is_an_error(self):
        with pytest.raises(vocal.VocalError, match="重叠"):
            vocal.classify_segment(1000, 1200, [_window(0, 500, spoken=0.5)])


class TestModel:
    def test_sha256_and_missing_model_are_checked(self, tmp_path):
        good = tmp_path / "yamnet.tflite"
        good.write_bytes(b"yamnet-test-model")
        old_digest = vocal.YAMNET_SHA256
        try:
            vocal.YAMNET_SHA256 = hashlib.sha256(good.read_bytes()).hexdigest()
            assert vocal._verify_model(good) is None
            bad = tmp_path / "bad.tflite"
            bad.write_bytes(b"wrong")
            with pytest.raises(vocal.VocalError, match="校验"):
                vocal._verify_model(bad)
            with pytest.raises(vocal.VocalError, match="不存在"):
                vocal._verify_model(tmp_path / "missing.tflite")
        finally:
            vocal.YAMNET_SHA256 = old_digest


class _FakeInterpreter:
    instances = []

    def __init__(self, *, model_path, num_threads):
        self.model_path = model_path
        self.num_threads = num_threads
        self.resizes = []
        self.inputs = []
        self.invocations = 0
        self._tensor = None
        self.__class__.instances.append(self)

    def get_input_details(self):
        return [{"index": 0, "shape": [1], "dtype": np.float32}]

    def resize_tensor_input(self, index, shape, strict):
        self.resizes.append((index, shape, strict))

    def allocate_tensors(self):
        pass

    def get_output_details(self):
        return [
            {"index": 1, "shape": [1, 521]},
            {"index": 2, "shape": [1, 3]},
        ]

    def set_tensor(self, index, value):
        self.inputs.append(np.array(value, copy=True))

    def invoke(self):
        self.invocations += 1

    def get_tensor(self, index):
        return np.asarray([_scores({0: 0.3, 132: 0.2})], dtype=np.float32)


def _install_fake_litert(monkeypatch):
    package = types.ModuleType("ai_edge_litert")
    package.__path__ = []
    interpreter = types.ModuleType("ai_edge_litert.interpreter")
    interpreter.Interpreter = _FakeInterpreter
    monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", interpreter)


def test_analyze_decodes_and_infers_each_window_with_padded_tail(tmp_path, monkeypatch):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model")
    monkeypatch.setenv("YAMNET_MODEL_PATH", str(model))
    monkeypatch.setattr(vocal, "YAMNET_SHA256", hashlib.sha256(b"model").hexdigest())
    _FakeInterpreter.instances.clear()
    _install_fake_litert(monkeypatch)
    waveform = np.arange(15_610, dtype=np.float32)
    ffmpeg_calls = []

    def fake_run(argv, **kwargs):
        ffmpeg_calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=waveform.astype("<f4").tobytes(), stderr=b"")

    monkeypatch.setattr(vocal.subprocess, "run", fake_run)
    analysis = vocal.analyze(tmp_path / "voice.mp3")

    assert len(analysis.windows) == 2
    assert analysis.windows[0].start_ms == 0
    assert analysis.windows[0].end_ms == 975
    assert analysis.windows[1].start_ms == 975
    assert analysis.windows[1].end_ms == 976
    assert analysis.has_bgm is True
    argv, kwargs = ffmpeg_calls[0]
    assert argv == [
        "ffmpeg", "-v", "error", "-i", str(tmp_path / "voice.mp3"), "-vn",
        "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
    ]
    assert kwargs["timeout"] == 120
    instance = _FakeInterpreter.instances[0]
    assert instance.model_path == str(model)
    assert instance.num_threads == 1
    assert instance.resizes == [(0, [15600], True)]
    assert instance.invocations == 2
    assert instance.inputs[0].shape == (15600,)
    assert instance.inputs[1].shape == (15600,)
    np.testing.assert_array_equal(instance.inputs[1][15_610 - 15_600:], np.zeros(15_590))


def test_analyze_real_model_e2e(tmp_path):
    """真实 ffmpeg + 真实 YAMNet：验证模型可加载并完成 1 秒音频推理。"""
    audio = tmp_path / "tone.wav"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(audio)],
        check=True,
        capture_output=True,
    )
    analysis = vocal.analyze(audio)
    assert analysis.windows
    assert all(window.end_ms > window.start_ms for window in analysis.windows)
    assert isinstance(analysis.has_bgm, bool)
