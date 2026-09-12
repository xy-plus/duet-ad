"""腾讯混元物品检测 provider 测试（自包含；统一契约断言在本文件内）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import HunyuanGroundingClient, ObjectDetection
from app.providers import ProviderFailure


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://detect.test",
        "multiref_mode": "direct",
        "hunyuan_api_key": "hunyuan-secret",
        "hunyuan_api_base": "https://hunyuan.test/v1",
        "hunyuan_model": "hunyuan-test-model",
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


_HUNYUAN_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox": "<bbox>100 200 300 400</bbox>"},'
    '{"timestamp_s": 1.5, "bbox": "<bbox>0 0 1000 1000</bbox>"}'
    "]}"
)


def _hunyuan_client(settings: Settings, handler) -> HunyuanGroundingClient:
    return HunyuanGroundingClient(settings, transport=httpx.MockTransport(handler))


def _prompt_of(request: httpx.Request) -> str:
    body = json.loads(request.content)
    return body["messages"][0]["content"][1]["text"]


# --- 正常路径 ---


def test_hunyuan_detect_parses_bbox_tags(tmp_path: Path, video: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://hunyuan.test/v1/chat/completions")
        assert request.headers["authorization"] == "Bearer hunyuan-secret"
        body = json.loads(request.content)
        assert body["model"] == "hunyuan-test-model"
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[0]["video_url"]["fps"] == 1
        assert content[1]["type"] == "text"
        assert "bottle" in content[1]["text"]
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [
        ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.3, 0.4)),
        ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
    ]


def test_hunyuan_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = '定位结果如下：\n```json\n' + _HUNYUAN_GOOD + "\n```\n以上是全部结果。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(drifted)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert len(result) == 2
    assert result[0].bbox_xyxy_normalized == (0.1, 0.2, 0.3, 0.4)


@pytest.mark.parametrize(
    "bbox_field",
    [
        "100 200 300 400",  # 无标签纯数字字符串
        [100, 200, 300, 400],  # JSON 数组
    ],
)
def test_hunyuan_detect_parses_conservative_bbox_variants(
    tmp_path: Path, video: Path, bbox_field: object
) -> None:
    text = json.dumps({"detections": [{"timestamp_s": 2, "bbox": bbox_field}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(text)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert result == [ObjectDetection("bottle", 2.0, (0.1, 0.2, 0.3, 0.4))]


# --- 重试语义 ---


def test_hunyuan_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}
    bad = '{"detections": [{"timestamp_s": 0, "bbox": "<bbox>100 200 1200 400</bbox>"}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _chat_response(bad)
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_hunyuan_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_hunyuan_detect_retries_transport_error(tmp_path: Path, video: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
    result = client.detect(video, ["bottle"])
    assert calls["count"] == 2
    assert len(result) == 2


def test_hunyuan_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _hunyuan_client(_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert calls["count"] == 1


def test_hunyuan_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"failing": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "nosuchobject" in _prompt_of(request):
            calls["failing"] += 1
            return httpx.Response(503, json={"error": "unavailable"})
        calls["good"] += 1
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    # 失败目标首试 + 2 次重试 = 3 次推理上限，随后跳过；另一目标不受影响。
    assert calls["failing"] == 3
    assert calls["good"] == 1
    assert {detection.label for detection in result} == {"bottle"}
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


def test_hunyuan_detect_skips_target_with_persistent_empty_result(
    tmp_path: Path, video: Path
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _chat_response('{"detections": []}')

    client = _hunyuan_client(_settings(tmp_path), handler)
    assert client.detect(video, ["bottle"]) == []
    assert calls["count"] == 3


# --- 统一契约断言（镜像 tests/test_object_detect.py 的 CONTRACT_CASES 语义） ---


def test_hunyuan_unified_contract(tmp_path: Path, video: Path) -> None:
    assert object_detect.PROVIDER_REGISTRY["hunyuan"] is HunyuanGroundingClient

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(_HUNYUAN_GOOD)

    client = _hunyuan_client(_settings(tmp_path), handler)
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


# --- config 校验 ---


def test_settings_hunyuan_defaults(tmp_path: Path) -> None:
    settings = _settings(tmp_path, hunyuan_api_key=None)
    assert settings.hunyuan_api_key is None
    settings = _settings(
        tmp_path,
        hunyuan_api_base="https://api.hunyuan.cloud.tencent.com/v1",
        hunyuan_model="hunyuan-large-vision",
    )
    assert settings.hunyuan_api_base == "https://api.hunyuan.cloud.tencent.com/v1"
    assert settings.hunyuan_model == "hunyuan-large-vision"


def test_settings_rejects_non_ascii_hunyuan_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HUNYUAN_API_KEY"):
        _settings(tmp_path, hunyuan_api_key="k€y")


def test_settings_rejects_invalid_hunyuan_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HUNYUAN_MODEL"):
        _settings(tmp_path, hunyuan_model="bad model!")


def test_settings_rejects_non_https_hunyuan_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HUNYUAN_API_BASE"):
        _settings(tmp_path, hunyuan_api_base="http://hunyuan.test/v1")


def test_get_settings_reads_hunyuan_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("HUNYUAN_API_KEY", "  hunyuan-key ")
    monkeypatch.delenv("HUNYUAN_API_BASE", raising=False)
    monkeypatch.delenv("HUNYUAN_MODEL", raising=False)
    settings = get_settings()
    assert settings.hunyuan_api_key == "hunyuan-key"
    assert settings.hunyuan_api_base == "https://api.hunyuan.cloud.tencent.com/v1"
    assert settings.hunyuan_model == "hunyuan-large-vision"


def test_get_settings_hunyuan_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("HUNYUAN_API_KEY", raising=False)
    settings = get_settings()
    assert settings.hunyuan_api_key is None
