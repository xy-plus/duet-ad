from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import (
    DoubaoGroundingClient,
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
        "doubao_api_key": "doubao-secret",
        "doubao_api_base": "https://doubao.test/api/v3",
        "doubao_model": "doubao-test-model",
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


_DOUBAO_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox": "<bbox>100 200 300 400</bbox>"},'
    '{"timestamp_s": 1.5, "bbox": "<bbox>0 0 1000 1000</bbox>"}'
    "]}"
)


def _doubao_client(settings: Settings, handler) -> DoubaoGroundingClient:
    return DoubaoGroundingClient(settings, transport=httpx.MockTransport(handler))


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][0]["content"][1]["text"]


# --- ObjectDetection 值对象校验 ---


def test_object_detection_accepts_valid_values() -> None:
    detection = ObjectDetection(
        label="bottle", timestamp_s=0.0,
        bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.4),
    )
    assert detection.label == "bottle"
    assert detection.timestamp_s == 0.0
    assert detection.bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)


@pytest.mark.parametrize(
    "bbox",
    [
        (-0.1, 0.2, 0.3, 0.4),  # 坐标 < 0
        (0.1, 0.2, 1.5, 0.4),  # 坐标 > 1
        (0.4, 0.2, 0.3, 0.4),  # x0 >= x1
        (0.1, 0.5, 0.3, 0.4),  # y0 >= y1
    ],
)
def test_object_detection_rejects_invalid_bbox(bbox: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        ObjectDetection(
            label="bottle", timestamp_s=0.0,
            bbox_xyxy_normalized=bbox,  # type: ignore[arg-type]
        )


def test_object_detection_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError):
        ObjectDetection(
            label="bottle", timestamp_s=-0.1,
            bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.4),
        )


# --- 豆包参考实现 ---


def test_doubao_detect_parses_bbox_tags(tmp_path: Path, video: Path) -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://doubao.test/api/v3/chat/completions")
        assert request.headers["authorization"] == "Bearer doubao-secret"
        body = json.loads(request.content)
        observed["body"] = body
        assert body["model"] == "doubao-test-model"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[0]["video_url"]["fps"] == 1
        assert content[1]["type"] == "text"
        assert "bottle" in content[1]["text"]
        return _chat_response(_DOUBAO_GOOD)

    client = _doubao_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [
        ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.3, 0.4)),
        ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
    ]


def test_doubao_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = '好的，定位结果如下：\n```json\n' + _DOUBAO_GOOD + "\n```\n以上是全部结果。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(drifted)

    client = _doubao_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert len(result) == 2
    assert result[0].bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)


def test_doubao_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}
    bad = '{"detections": [{"timestamp_s": 0, "bbox": "<bbox>100 200 1200 400</bbox>"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response(bad)
        return _chat_response(_DOUBAO_GOOD)

    client = _doubao_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_doubao_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _chat_response(_DOUBAO_GOOD)

    client = _doubao_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_doubao_detect_retries_transport_error(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return _chat_response(_DOUBAO_GOOD)

    client = _doubao_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_doubao_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _doubao_client(_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert calls["count"] == 1


def test_doubao_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"failing": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "nosuchobject" in _prompt_of(request):
            calls["failing"] += 1
            return httpx.Response(503, json={"error": "unavailable"})
        calls["good"] += 1
        return _chat_response(_DOUBAO_GOOD)

    client = _doubao_client(_settings(tmp_path), handler)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    # 失败目标首试 + 2 次重试 = 3 次推理上限，随后跳过；另一目标不受影响。
    assert calls["failing"] == 3
    assert calls["good"] == 1
    assert {detection.label for detection in result} == {"bottle"}
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


def test_doubao_detect_skips_target_with_persistent_empty_result(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _chat_response('{"detections": []}')

    client = _doubao_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == []
    assert calls["count"] == 3


def test_doubao_detect_rejects_oversized_inline_video(
    tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(object_detect, "INLINE_VIDEO_MAX_BYTES", 8)
    client = _doubao_client(_settings(tmp_path), lambda request: _chat_response(""))
    with pytest.raises(ValueError, match="Files API"):
        client.detect(video, ["bottle"])


# --- 注册表与选择函数 ---


def test_registry_reserves_seven_provider_names() -> None:
    assert tuple(object_detect.PROVIDER_REGISTRY) == (
        "doubao", "qwen", "glm", "hunyuan", "kimi", "step", "gemini",
    )
    assert object_detect.PROVIDER_REGISTRY["doubao"] is DoubaoGroundingClient


def test_available_providers_skips_missing_key_and_unimplemented(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(
        tmp_path,
        object_detect_providers=("doubao", "qwen", "gemini"),
        doubao_api_key=None,
    )
    with caplog.at_level(logging.INFO, logger="app.object_detect"):
        assert available_providers(settings) == []
    messages = [record.getMessage() for record in caplog.records]
    assert any("doubao" in message for message in messages)
    assert any("qwen" in message for message in messages)


def test_available_providers_preserves_configured_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path, object_detect_providers=("qwen", "doubao"))
    assert available_providers(settings) == ["doubao"]


def test_select_provider_returns_first_available_client(tmp_path: Path) -> None:
    settings = _settings(tmp_path, object_detect_providers=("qwen", "doubao"))
    client = select_provider(settings)
    assert isinstance(client, DoubaoGroundingClient)


def test_select_provider_returns_none_when_nothing_available(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        object_detect_providers=("qwen",),
        doubao_api_key=None,
    )
    assert select_provider(settings) is None


# --- 统一契约参数化框架：每家 provider 在 CONTRACT_CASES 注册一个正常用例 ---
# 后续 6 家代理：在 CONTRACT_CASES 加同名条目（settings 覆盖 + 正常响应文本），
# 本测试自动覆盖该家的统一输入输出契约。

CONTRACT_CASES: dict[str, dict[str, object]] = {
    "doubao": {
        "settings": {
            "doubao_api_key": "doubao-secret",
            "doubao_api_base": "https://doubao.test/api/v3",
            "doubao_model": "doubao-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
}


@pytest.mark.parametrize("provider_name", sorted(CONTRACT_CASES))
def test_unified_contract(provider_name: str, tmp_path: Path, video: Path) -> None:
    case = CONTRACT_CASES[provider_name]
    settings = _settings(tmp_path, **case["settings"])  # type: ignore[arg-type]
    client_class = object_detect.PROVIDER_REGISTRY[provider_name]
    assert client_class is not None

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(str(case["response_text"]))

    client = client_class(settings, transport=httpx.MockTransport(handler))
    result = client.detect(video, ["bottle"])
    assert isinstance(result, list)
    assert result, f"{provider_name} returned no detections"
    for detection in result:
        assert isinstance(detection, ObjectDetection)
        assert detection.label == "bottle"
        assert detection.timestamp_s >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


# --- config 校验 ---


def test_settings_rejects_unknown_object_detect_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OBJECT_DETECT_PROVIDERS"):
        _settings(tmp_path, object_detect_providers=("doubao", "unknown"))


def test_settings_accepts_known_object_detect_providers(tmp_path: Path) -> None:
    settings = _settings(tmp_path, object_detect_providers=("doubao", "qwen"))
    assert settings.object_detect_providers == ("doubao", "qwen")


def test_settings_object_detect_providers_default_empty(tmp_path: Path) -> None:
    assert _settings(tmp_path).object_detect_providers == ()


def test_settings_doubao_api_key_defaults_to_none(tmp_path: Path) -> None:
    assert _settings(tmp_path, doubao_api_key=None).doubao_api_key is None


def test_settings_rejects_non_ascii_doubao_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DOUBAO_API_KEY"):
        _settings(tmp_path, doubao_api_key="k€y")


def test_settings_rejects_invalid_doubao_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DOUBAO_MODEL"):
        _settings(tmp_path, doubao_model="bad model!")


def test_settings_rejects_non_https_doubao_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DOUBAO_API_BASE"):
        _settings(tmp_path, doubao_api_base="http://doubao.test/api/v3")


def test_get_settings_reads_object_detect_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("OBJECT_DETECT_PROVIDERS", "doubao, qwen")
    monkeypatch.setenv("DOUBAO_API_KEY", "  ark-key ")
    monkeypatch.delenv("DOUBAO_API_BASE", raising=False)
    monkeypatch.delenv("DOUBAO_MODEL", raising=False)
    settings = get_settings()
    assert settings.object_detect_providers == ("doubao", "qwen")
    assert settings.doubao_api_key == "ark-key"
    assert settings.doubao_api_base == "https://ark.cn-beijing.volces.com/api/v3"
    assert settings.doubao_model == "doubao-seed-2-1-pro-260628"


def test_get_settings_object_detect_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("OBJECT_DETECT_PROVIDERS", raising=False)
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    settings = get_settings()
    assert settings.object_detect_providers == ()
    assert settings.doubao_api_key is None
