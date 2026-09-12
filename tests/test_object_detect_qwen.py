"""Qwen（阿里百炼）物品检测 provider 测试；自包含，不依赖 test_object_detect.py。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import ObjectDetection, QwenGroundingClient
from app.providers import ProviderFailure


_FRAME_WIDTH = 320
_FRAME_HEIGHT = 240


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://detect.test",
        "multiref_mode": "direct",
        "qwen_api_key": "qwen-secret",
        "qwen_api_base": "https://qwen.test/compatible-mode/v1",
        "qwen_model": "qwen-test-model",
        "request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(object_detect, "retry_sleep", lambda _seconds: None)


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    """真实可解码视频（cv2.VideoCapture 需要读到实际宽高做归一化）。"""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (_FRAME_WIDTH, _FRAME_HEIGHT),
    )
    assert writer.isOpened(), "cv2.VideoWriter cannot create the test video"
    for _ in range(4):
        writer.write(
            np.full((_FRAME_HEIGHT, _FRAME_WIDTH, 3), 127, dtype=np.uint8)
        )
    writer.release()
    return path


def _chat_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"choices": [{"message": {"role": "assistant", "content": text}}]},
    )


_QWEN_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox_2d": [32, 48, 160, 144]},'
    '{"timestamp_s": 1.5, "bbox_2d": [0, 0, 320, 240]}'
    "]}"
)

_QWEN_GOOD_DETECTIONS = [
    ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.5, 0.6)),
    ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
]


def _qwen_client(settings: Settings, handler) -> QwenGroundingClient:
    return QwenGroundingClient(settings, transport=httpx.MockTransport(handler))


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][0]["content"][1]["text"]


# --- 正常路径与请求形态 ---


def test_qwen_detect_normalizes_bbox_2d_absolute_pixels(
    tmp_path: Path, video: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL(
            "https://qwen.test/compatible-mode/v1/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer qwen-secret"
        body = json.loads(request.content)
        assert body["model"] == "qwen-test-model"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[0]["video_url"]["fps"] == 1
        assert content[1]["type"] == "text"
        assert "bottle" in content[1]["text"]
        # Qwen-VL 输出绝对像素坐标，提示词必须按此约定索取。
        assert "bbox_2d" in content[1]["text"]
        assert "pixel" in content[1]["text"]
        return _chat_response(_QWEN_GOOD)

    client = _qwen_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == _QWEN_GOOD_DETECTIONS


def test_qwen_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = "定位结果如下：\n```json\n" + _QWEN_GOOD + "\n```\n以上是全部结果。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(drifted)

    client = _qwen_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == _QWEN_GOOD_DETECTIONS


def test_qwen_detect_accepts_bare_detection_list(tmp_path: Path, video: Path) -> None:
    # Qwen-VL grounding 惯用输出：顶层 JSON 数组，无 detections 包裹。
    bare = (
        '[{"bbox_2d": [32, 48, 160, 144], "timestamp_s": 0},'
        ' {"bbox_2d": [0, 0, 320, 240], "timestamp_s": 1.5}]'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(bare)

    client = _qwen_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == _QWEN_GOOD_DETECTIONS


def test_qwen_detect_rejects_undecodable_video(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    client = _qwen_client(
        _settings(tmp_path), lambda request: _chat_response(_QWEN_GOOD)
    )
    with pytest.raises(ValueError, match="frame size"):
        client.detect(fake, ["bottle"])


# --- 重试与跳过语义（镜像豆包用例） ---


def test_qwen_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}
    bad = '{"detections": [{"timestamp_s": 0, "bbox_2d": [32, 48, 400, 144]}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response(bad)
        return _chat_response(_QWEN_GOOD)

    client = _qwen_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert result == _QWEN_GOOD_DETECTIONS


def test_qwen_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _chat_response(_QWEN_GOOD)

    client = _qwen_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == _QWEN_GOOD_DETECTIONS
    assert calls["count"] == 2


def test_qwen_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _qwen_client(_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert calls["count"] == 1


def test_qwen_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"failing": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "nosuchobject" in _prompt_of(request):
            calls["failing"] += 1
            return httpx.Response(503, json={"error": "unavailable"})
        calls["good"] += 1
        return _chat_response(_QWEN_GOOD)

    client = _qwen_client(_settings(tmp_path), handler)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    assert calls["failing"] == 3
    assert calls["good"] == 1
    assert result == _QWEN_GOOD_DETECTIONS
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


# --- 统一契约自断言（合并代理接线 CONTRACT_CASES 前的本地等价覆盖） ---


def test_qwen_unified_contract(tmp_path: Path, video: Path) -> None:
    client_class = object_detect.PROVIDER_REGISTRY["qwen"]
    assert client_class is QwenGroundingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(_QWEN_GOOD)

    client = client_class(_settings(tmp_path), transport=httpx.MockTransport(handler))
    result = client.detect(video, ["bottle"])
    assert isinstance(result, list)
    assert result, "qwen returned no detections"
    for detection in result:
        assert isinstance(detection, ObjectDetection)
        assert detection.label == "bottle"
        assert detection.timestamp_s >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


# --- config 校验 ---


def test_settings_qwen_defaults(tmp_path: Path) -> None:
    settings = Settings(
        access_token="test-access",
        data_dir=tmp_path,
        public_base_url="https://detect.test",
        multiref_mode="direct",
    )
    assert settings.qwen_api_key is None
    assert (
        settings.qwen_api_base
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert settings.qwen_model == "qwen3-vl-plus"


def test_settings_rejects_non_ascii_qwen_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="QWEN_API_KEY"):
        _settings(tmp_path, qwen_api_key="k€y")


def test_settings_rejects_blank_qwen_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="QWEN_API_KEY"):
        _settings(tmp_path, qwen_api_key="   ")


def test_settings_rejects_invalid_qwen_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="QWEN_MODEL"):
        _settings(tmp_path, qwen_model="bad model!")


def test_settings_rejects_non_https_qwen_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="QWEN_API_BASE"):
        _settings(tmp_path, qwen_api_base="http://qwen.test/v1")


def test_get_settings_reads_qwen_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("QWEN_API_KEY", "  qwen-key ")
    monkeypatch.delenv("QWEN_API_BASE", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    settings = get_settings()
    assert settings.qwen_api_key == "qwen-key"
    assert (
        settings.qwen_api_base
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert settings.qwen_model == "qwen3-vl-plus"


def test_get_settings_qwen_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    settings = get_settings()
    assert settings.qwen_api_key is None
