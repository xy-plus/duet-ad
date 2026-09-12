from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import (
    GlmGroundingClient,
    ObjectDetection,
    available_providers,
    select_provider,
)
from app.providers import ProviderFailure


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://detect.test",
        "multiref_mode": "direct",
        "glm_api_key": "glm-secret",
        "glm_api_base": "https://glm.test/api/paas/v4",
        "glm_model": "glm-test-model",
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


_GLM_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox": "<bbox>100 200 300 400</bbox>"},'
    '{"timestamp_s": 1.5, "bbox": "<bbox>0 0 1000 1000</bbox>"}'
    "]}"
)


def _glm_client(settings: Settings, handler) -> GlmGroundingClient:
    return GlmGroundingClient(settings, transport=httpx.MockTransport(handler))


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][0]["content"][1]["text"]


# --- GLM 正常路径与保守解析 ---


def test_glm_detect_parses_json_bbox_tags(tmp_path: Path, video: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL(
            "https://glm.test/api/paas/v4/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer glm-secret"
        body = json.loads(request.content)
        assert body["model"] == "glm-test-model"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[1]["type"] == "text"
        assert "bottle" in content[1]["text"]
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [
        ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.3, 0.4)),
        ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
    ]


def test_glm_detect_parses_numeric_bbox_arrays(tmp_path: Path, video: Path) -> None:
    # 格式漂移：bbox 直接给数字数组（0-1000 归一化），不带 <bbox> 标签。
    payload = (
        '{"detections": ['
        '{"timestamp_s": 2, "bbox": [100, 200, 300, 400]}'
        "]}"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(payload)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [ObjectDetection("bottle", 2.0, (0.1, 0.2, 0.3, 0.4))]


def test_glm_detect_accepts_already_normalized_floats(
    tmp_path: Path, video: Path
) -> None:
    # 格式漂移：模型直接给 0-1 浮点坐标（不再除 1000）。
    payload = (
        '{"detections": ['
        '{"timestamp_s": 0.5, "bbox": [0.1, 0.2, 0.3, 0.4]}'
        "]}"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(payload)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [ObjectDetection("bottle", 0.5, (0.1, 0.2, 0.3, 0.4))]


def test_glm_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = '好的，定位结果如下：\n```json\n' + _GLM_GOOD + "\n```\n以上是全部结果。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(drifted)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert len(result) == 2
    assert result[0].bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)


# --- 重试与致命错误 ---


def test_glm_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}
    bad = '{"detections": [{"timestamp_s": 0, "bbox": "<bbox>100 200 1200 400</bbox>"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response(bad)
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_glm_detect_retries_unparseable_output(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response("对不起，我无法定位该物体。")
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_glm_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_glm_detect_retries_transport_error(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_glm_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _glm_client(_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert calls["count"] == 1


def test_glm_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"failing": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "nosuchobject" in _prompt_of(request):
            calls["failing"] += 1
            return httpx.Response(503, json={"error": "unavailable"})
        calls["good"] += 1
        return _chat_response(_GLM_GOOD)

    client = _glm_client(_settings(tmp_path), handler)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    # 失败目标首试 + 2 次重试 = 3 次推理上限，随后跳过；另一目标不受影响。
    assert calls["failing"] == 3
    assert calls["good"] == 1
    assert {detection.label for detection in result} == {"bottle"}
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


# --- 统一契约断言（合并代理接线 CONTRACT_CASES 前的自包含版本） ---


def test_glm_unified_contract(tmp_path: Path, video: Path) -> None:
    client_class = object_detect.PROVIDER_REGISTRY["glm"]
    assert client_class is GlmGroundingClient
    client = GlmGroundingClient(
        _settings(tmp_path),
        transport=httpx.MockTransport(lambda request: _chat_response(_GLM_GOOD)),
    )
    result = client.detect(video, ["bottle"])
    assert isinstance(result, list)
    assert result
    for detection in result:
        assert isinstance(detection, ObjectDetection)
        assert detection.label == "bottle"
        assert detection.timestamp_s >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


def test_glm_available_and_selectable(tmp_path: Path) -> None:
    settings = _settings(tmp_path, object_detect_providers=("glm",))
    assert available_providers(settings) == ["glm"]
    assert isinstance(select_provider(settings), GlmGroundingClient)


def test_glm_unavailable_without_key(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, object_detect_providers=("glm",), glm_api_key=None
    )
    assert available_providers(settings) == []


def test_glm_detect_requires_configured_key(tmp_path: Path, video: Path) -> None:
    client = _glm_client(
        _settings(tmp_path, glm_api_key=None),
        lambda request: _chat_response(_GLM_GOOD),
    )
    with pytest.raises(ValueError, match="glm API key is not configured"):
        client.detect(video, ["bottle"])


# --- config 校验 ---


def test_settings_glm_api_key_defaults_to_none(tmp_path: Path) -> None:
    assert _settings(tmp_path, glm_api_key=None).glm_api_key is None


def test_settings_rejects_non_ascii_glm_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        _settings(tmp_path, glm_api_key="k€y")


def test_settings_rejects_blank_glm_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        _settings(tmp_path, glm_api_key="   ")


def test_settings_rejects_invalid_glm_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GLM_MODEL"):
        _settings(tmp_path, glm_model="bad model!")


def test_settings_rejects_non_https_glm_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GLM_API_BASE"):
        _settings(tmp_path, glm_api_base="http://glm.test/api/paas/v4")


def test_get_settings_reads_glm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("GLM_API_KEY", "  glm-key ")
    monkeypatch.delenv("GLM_API_BASE", raising=False)
    monkeypatch.delenv("GLM_MODEL", raising=False)
    settings = get_settings()
    assert settings.glm_api_key == "glm-key"
    assert settings.glm_api_base == "https://open.bigmodel.cn/api/paas/v4"
    assert settings.glm_model == "glm-4.5v"


def test_get_settings_glm_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    settings = get_settings()
    assert settings.glm_api_key is None
