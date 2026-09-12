"""Kimi（Moonshot k2.6）物品检测 client 测试；自包含，不改统一契约框架。

解析格式假设：Kimi 官方未文档化 bbox 输出格式，采取保守解析——
沿用基类提示词要求的 ``<bbox>x1 y1 x2 y2</bbox>``（0–1000 归一化），
同时容忍模型输出漂移为纯数字串（"x1 y1 x2 y2"）或 JSON 数组
（[x1, y1, x2, y2]）；三种形态统一除以 1000 归一化到 0–1，
交给 ObjectDetection 校验。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import KimiGroundingClient, ObjectDetection
from app.providers import ProviderFailure


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://detect.test",
        "kimi_api_key": "kimi-secret",
        "kimi_api_base": "https://kimi.test/v1",
        "kimi_model": "kimi-test-model",
        "request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(object_detect, "retry_sleep", lambda _seconds: None)


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    return path


def _chat_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"choices": [{"message": {"role": "assistant", "content": text}}]},
    )


_KIMI_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox": "<bbox>100 200 300 400</bbox>"},'
    '{"timestamp_s": 1.5, "bbox": "<bbox>0 0 1000 1000</bbox>"}'
    "]}"
)


def _kimi_client(settings: Settings, handler) -> KimiGroundingClient:
    return KimiGroundingClient(settings, transport=httpx.MockTransport(handler))


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][0]["content"][1]["text"]


def test_kimi_detect_parses_bbox_tags(tmp_path: Path, video: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://kimi.test/v1/chat/completions")
        assert request.headers["authorization"] == "Bearer kimi-secret"
        body = json.loads(request.content)
        assert body["model"] == "kimi-test-model"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[0]["video_url"]["fps"] == 1
        assert content[1]["type"] == "text"
        assert "bottle" in content[1]["text"]
        return _chat_response(_KIMI_GOOD)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [
        ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.3, 0.4)),
        ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
    ]


def test_kimi_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = "定位结果：\n```json\n" + _KIMI_GOOD + "\n```\n以上。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(drifted)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert len(result) == 2
    assert result[0].bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)


def test_kimi_detect_tolerates_bbox_as_json_array(
    tmp_path: Path, video: Path
) -> None:
    # bbox 未文档化：模型可能漂移为 JSON 数组而非 <bbox> 标签，保守接受。
    array_form = (
        '{"detections": [{"timestamp_s": 2, "bbox": [100, 200, 300, 400]}]}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(array_form)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [ObjectDetection("bottle", 2.0, (0.1, 0.2, 0.3, 0.4))]


def test_kimi_detect_tolerates_bbox_as_plain_numbers(
    tmp_path: Path, video: Path
) -> None:
    # bbox 未文档化：模型可能只输出纯数字串，保守接受。
    plain = '{"detections": [{"timestamp_s": 2, "bbox": "100 200 300 400"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(plain)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [ObjectDetection("bottle", 2.0, (0.1, 0.2, 0.3, 0.4))]


def test_kimi_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}
    bad = '{"detections": [{"timestamp_s": 0, "bbox": "<bbox>100 200 1200 400</bbox>"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response(bad)
        return _chat_response(_KIMI_GOOD)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_kimi_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _chat_response(_KIMI_GOOD)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_kimi_detect_retries_transport_error(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return _chat_response(_KIMI_GOOD)

    client = _kimi_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_kimi_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _kimi_client(_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert calls["count"] == 1


def test_kimi_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"failing": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "nosuchobject" in _prompt_of(request):
            calls["failing"] += 1
            return httpx.Response(503, json={"error": "unavailable"})
        calls["good"] += 1
        return _chat_response(_KIMI_GOOD)

    client = _kimi_client(_settings(tmp_path), handler)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    # 失败目标首试 + 2 次重试 = 3 次推理上限，随后跳过；另一目标不受影响。
    assert calls["failing"] == 3
    assert calls["good"] == 1
    assert {detection.label for detection in result} == {"bottle"}
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


def test_kimi_detect_skips_target_with_persistent_empty_result(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _chat_response('{"detections": []}')

    client = _kimi_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == []
    assert calls["count"] == 3


def test_kimi_detect_rejects_oversized_inline_video(
    tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(object_detect, "INLINE_VIDEO_MAX_BYTES", 8)
    client = _kimi_client(_settings(tmp_path), lambda request: _chat_response(""))
    with pytest.raises(ValueError, match="Files API"):
        client.detect(video, ["bottle"])


# --- 统一契约断言（镜像 test_object_detect.py 的 test_unified_contract） ---


def test_kimi_unified_contract(tmp_path: Path, video: Path) -> None:
    settings = _settings(tmp_path)
    client_class = object_detect.PROVIDER_REGISTRY["kimi"]
    assert client_class is KimiGroundingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(_KIMI_GOOD)

    client = client_class(settings, transport=httpx.MockTransport(handler))
    result = client.detect(video, ["bottle"])
    assert isinstance(result, list)
    assert result, "kimi returned no detections"
    for detection in result:
        assert isinstance(detection, ObjectDetection)
        assert detection.label == "bottle"
        assert detection.timestamp_s >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


# --- config 校验 ---


def test_settings_kimi_api_key_defaults_to_none(tmp_path: Path) -> None:
    assert _settings(tmp_path, kimi_api_key=None).kimi_api_key is None


def test_settings_rejects_non_ascii_kimi_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="KIMI_API_KEY"):
        _settings(tmp_path, kimi_api_key="k€y")


def test_settings_rejects_invalid_kimi_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="KIMI_MODEL"):
        _settings(tmp_path, kimi_model="bad model!")


def test_settings_rejects_non_https_kimi_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="KIMI_API_BASE"):
        _settings(tmp_path, kimi_api_base="http://kimi.test/v1")


def test_get_settings_reads_kimi_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("KIMI_API_KEY", "  sk-moonshot ")
    monkeypatch.delenv("KIMI_API_BASE", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)
    settings = get_settings()
    assert settings.kimi_api_key == "sk-moonshot"
    assert settings.kimi_api_base == "https://api.moonshot.cn/v1"
    assert settings.kimi_model == "kimi-k2.6"


def test_get_settings_kimi_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    settings = get_settings()
    assert settings.kimi_api_key is None
