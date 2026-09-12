"""Gemini 物品检测 provider 自包含测试：File API 上传 + generateContent。

镜像 tests/test_object_detect.py 豆包用例维度：正常 / 格式漂移 / 坐标越界 /
重试 / 单目标跳过 + 统一契约断言；另覆盖 Gemini 特有语义：单次 detect 内
文件只上传一次（重试与多目标复用 file 句柄）、文件 ACTIVE 轮询、代理配置。
不改 tests/test_object_detect.py；CONTRACT_CASES 接线由合并代理统一完成。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from app import object_detect, providers
from app.config import Settings, get_settings
from app.object_detect import GeminiGroundingClient, ObjectDetection
from app.providers import ProviderFailure


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://detect.test",
        "gemini_api_key": "gemini-secret",
        "gemini_api_base": "https://gemini.test",
        "gemini_model": "gemini-2.5-flash",
        "request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(object_detect, "retry_sleep", lambda _seconds: None)
    monkeypatch.setattr(providers, "retry_sleep", lambda _seconds: None)


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    return path


_GEMINI_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "box_2d": [200, 100, 400, 300]},'
    '{"timestamp_s": 1.5, "box_2d": [0, 0, 1000, 1000]}'
    "]}"
)

# box_2d 顺序 [ymin, xmin, ymax, xmax] → xyxy = (0.1, 0.2, 0.3, 0.4)
_EXPECTED_GOOD = [
    ObjectDetection("bottle", 0.0, (0.1, 0.2, 0.3, 0.4)),
    ObjectDetection("bottle", 1.5, (0.0, 0.0, 1.0, 1.0)),
]


def _generate_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]},
    )


class _GeminiFlow:
    """模拟 File API resumable 上传（start → PUT finalize → ACTIVE 轮询）+ generateContent。"""

    def __init__(self, generate_responder, *, states: tuple[str, ...] = ("ACTIVE",)) -> None:
        self._generate_responder = generate_responder
        self._states = list(states)
        self.upload_start_calls = 0
        self.upload_finalize_calls = 0
        self.poll_calls = 0
        self.generate_calls = 0
        self.last_generate_body: dict[str, object] | None = None

    def _next_state(self) -> str:
        return self._states.pop(0) if self._states else "ACTIVE"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "gemini-secret"
        path = request.url.path
        if request.method == "POST" and path == "/upload/v1beta/files":
            self.upload_start_calls += 1
            assert request.headers["x-goog-upload-protocol"] == "resumable"
            return httpx.Response(
                200, headers={"X-Goog-Upload-URL": "https://gemini.test/upload/session"}
            )
        if request.method == "PUT" and path == "/upload/session":
            self.upload_finalize_calls += 1
            return httpx.Response(
                200,
                json={
                    "file": {
                        "name": "files/clip-1",
                        "uri": "https://files.test/clip-1",
                        "state": self._next_state(),
                    }
                },
            )
        if request.method == "GET" and path == "/v1beta/files/clip-1":
            self.poll_calls += 1
            return httpx.Response(
                200,
                json={
                    "name": "files/clip-1",
                    "uri": "https://files.test/clip-1",
                    "state": self._next_state(),
                },
            )
        if request.method == "POST" and ":generateContent" in path:
            self.generate_calls += 1
            self.last_generate_body = json.loads(request.content)
            return self._generate_responder(request)
        raise AssertionError(f"unexpected request {request.method} {request.url}")


def _client(
    settings: Settings, flow: _GeminiFlow
) -> GeminiGroundingClient:
    return GeminiGroundingClient(settings, transport=httpx.MockTransport(flow))


def _good_responder(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    return _generate_response(_GEMINI_GOOD)


# --- 正常流程与解析 ---


def test_gemini_detect_uploads_once_and_parses_box_2d_as_ymin_xmin(
    tmp_path: Path, video: Path
) -> None:
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == _EXPECTED_GOOD
    assert flow.upload_start_calls == 1
    assert flow.upload_finalize_calls == 1
    assert flow.poll_calls == 0  # finalize 直接 ACTIVE，无需轮询
    assert flow.generate_calls == 1
    body = flow.last_generate_body
    assert body is not None
    parts = body["contents"][0]["parts"]  # type: ignore[index]
    assert parts[0]["file_data"]["file_uri"] == "https://files.test/clip-1"
    assert "bottle" in parts[1]["text"]
    assert "box_2d" in parts[1]["text"]


def test_gemini_detect_tolerates_format_drift(tmp_path: Path, video: Path) -> None:
    drifted = "好的，定位结果如下：\n```json\n" + _GEMINI_GOOD + "\n```\n以上是全部结果。"
    flow = _GeminiFlow(lambda request: _generate_response(drifted))
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == _EXPECTED_GOOD


def test_gemini_detect_accepts_bare_detection_list(tmp_path: Path, video: Path) -> None:
    # Gemini 文档常见输出形态：顶层就是数组，条目带 label 字段（忽略 label，以目标为准）。
    bare = '[{"timestamp_s": 0, "box_2d": [200, 100, 400, 300], "label": "bottle"}]'
    flow = _GeminiFlow(lambda request: _generate_response(bare))
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == [_EXPECTED_GOOD[0]]


def test_gemini_detect_accepts_float_box_coordinates(tmp_path: Path, video: Path) -> None:
    text = '{"detections": [{"timestamp_s": 2, "box_2d": [200.5, 100.25, 400.75, 300.5]}]}'
    flow = _GeminiFlow(lambda request: _generate_response(text))
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == [
        ObjectDetection("bottle", 2.0, (0.10025, 0.2005, 0.3005, 0.40075))
    ]


# --- 重试语义（重试 = 重新推理，复用已上传文件句柄） ---


def test_gemini_detect_retries_out_of_range_coordinates(
    tmp_path: Path, video: Path
) -> None:
    bad = '{"detections": [{"timestamp_s": 0, "box_2d": [200, 100, 400, 1200]}]}'

    def responder(request: httpx.Request) -> httpx.Response:
        return _generate_response(bad if flow.generate_calls == 1 else _GEMINI_GOOD)

    flow = _GeminiFlow(responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert flow.generate_calls == 2
    assert result == _EXPECTED_GOOD
    assert flow.upload_start_calls == 1  # 重试不重复上传


def test_gemini_detect_retries_reversed_box(tmp_path: Path, video: Path) -> None:
    bad = '{"detections": [{"timestamp_s": 0, "box_2d": [400, 300, 200, 100]}]}'

    def responder(request: httpx.Request) -> httpx.Response:
        return _generate_response(bad if flow.generate_calls == 1 else _GEMINI_GOOD)

    flow = _GeminiFlow(responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert flow.generate_calls == 2
    assert result == _EXPECTED_GOOD


def test_gemini_detect_retries_gateway_status(tmp_path: Path, video: Path) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        if flow.generate_calls == 1:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return _generate_response(_GEMINI_GOOD)

    flow = _GeminiFlow(responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert flow.generate_calls == 2
    assert result == _EXPECTED_GOOD
    assert flow.upload_start_calls == 1


def test_gemini_detect_retries_transport_error(tmp_path: Path, video: Path) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        if flow.generate_calls == 1:
            raise httpx.ConnectError("boom", request=request)
        return _generate_response(_GEMINI_GOOD)

    flow = _GeminiFlow(responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert flow.generate_calls == 2
    assert result == _EXPECTED_GOOD


def test_gemini_detect_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    flow = _GeminiFlow(
        lambda request: httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    client = _client(_settings(tmp_path), flow)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert flow.generate_calls == 1


def test_gemini_detect_skips_target_after_retry_exhaustion(
    tmp_path: Path, video: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "nosuchobject" in body["contents"][0]["parts"][1]["text"]:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return _generate_response(_GEMINI_GOOD)

    flow = _GeminiFlow(responder)
    client = _client(_settings(tmp_path), flow)
    with caplog.at_level(logging.WARNING, logger="app.object_detect"):
        result = client.detect(video, ["nosuchobject", "bottle"])
    # 失败目标首试 + 2 次重试 = 3 次推理上限，随后跳过；另一目标不受影响。
    assert flow.generate_calls == 4
    assert {detection.label for detection in result} == {"bottle"}
    assert flow.upload_start_calls == 1  # 多目标复用同一文件句柄
    assert any("nosuchobject" in record.getMessage() for record in caplog.records)


def test_gemini_detect_skips_target_with_persistent_empty_result(
    tmp_path: Path, video: Path
) -> None:
    flow = _GeminiFlow(lambda request: _generate_response('{"detections": []}'))
    client = _client(_settings(tmp_path), flow)
    assert client.detect(video, ["bottle"]) == []
    assert flow.generate_calls == 3


# --- 上传与文件状态 ---


def test_gemini_detect_waits_for_file_active(tmp_path: Path, video: Path) -> None:
    flow = _GeminiFlow(_good_responder, states=("PROCESSING", "PROCESSING", "ACTIVE"))
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == _EXPECTED_GOOD
    assert flow.poll_calls == 2


def test_gemini_detect_retries_failed_upload(tmp_path: Path, video: Path) -> None:
    good = _good_responder

    class FlakyUpload(_GeminiFlow):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if (
                request.method == "POST"
                and request.url.path == "/upload/v1beta/files"
                and self.upload_start_calls < 2
            ):
                self.upload_start_calls += 1
                return httpx.Response(503, json={"error": {"message": "unavailable"}})
            return super().__call__(request)

    flow = FlakyUpload(good)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert result == _EXPECTED_GOOD
    assert flow.upload_start_calls == 3


def test_gemini_detect_upload_client_error_is_fatal(tmp_path: Path, video: Path) -> None:
    class BadUpload(_GeminiFlow):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/upload/v1beta/files":
                self.upload_start_calls += 1
                return httpx.Response(403, json={"error": {"message": "forbidden"}})
            return super().__call__(request)

    flow = BadUpload(_good_responder)
    client = _client(_settings(tmp_path), flow)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert flow.upload_start_calls == 1
    assert flow.generate_calls == 0


def test_gemini_detect_file_failed_state_is_fatal(tmp_path: Path, video: Path) -> None:
    flow = _GeminiFlow(_good_responder, states=("FAILED",))
    client = _client(_settings(tmp_path), flow)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert flow.generate_calls == 0


# --- 输入与配置守卫 ---


def test_gemini_detect_requires_api_key(tmp_path: Path, video: Path) -> None:
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path, gemini_api_key=None), flow)
    with pytest.raises(ValueError, match="gemini API key"):
        client.detect(video, ["bottle"])
    assert flow.upload_start_calls == 0


def test_gemini_detect_rejects_blank_targets(tmp_path: Path, video: Path) -> None:
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path), flow)
    with pytest.raises(ValueError, match="non-empty"):
        client.detect(video, ["  "])
    assert client.detect(video, []) == []
    assert flow.upload_start_calls == 0


def test_gemini_detect_rejects_empty_video(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path), flow)
    with pytest.raises(ValueError, match="empty"):
        client.detect(empty, ["bottle"])
    assert flow.upload_start_calls == 0


# --- 代理配置 ---


def _record_httpx_client(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    real_client = httpx.Client

    def factory(**kwargs: object) -> httpx.Client:
        captured.update(kwargs)
        kwargs.pop("proxy", None)  # MockTransport 下代理不生效，仅验证传参
        return real_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", factory)
    return captured


def test_gemini_proxy_passed_to_httpx_client(
    tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _record_httpx_client(monkeypatch)
    settings = _settings(tmp_path, gemini_proxy="http://127.0.0.1:7897")
    flow = _GeminiFlow(_good_responder)
    client = _client(settings, flow)
    assert client.detect(video, ["bottle"]) == _EXPECTED_GOOD
    assert captured.get("proxy") == "http://127.0.0.1:7897"


def test_gemini_without_proxy_omits_proxy_kwarg(
    tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _record_httpx_client(monkeypatch)
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path), flow)
    assert client.detect(video, ["bottle"]) == _EXPECTED_GOOD
    assert "proxy" not in captured


# --- 统一契约断言（与 tests/test_object_detect.py 的 test_unified_contract 同形） ---


def test_gemini_registered_in_provider_registry() -> None:
    assert object_detect.PROVIDER_REGISTRY["gemini"] is GeminiGroundingClient


def test_gemini_unified_contract(tmp_path: Path, video: Path) -> None:
    flow = _GeminiFlow(_good_responder)
    client = _client(_settings(tmp_path), flow)
    result = client.detect(video, ["bottle"])
    assert isinstance(result, list)
    assert result, "gemini returned no detections"
    for detection in result:
        assert isinstance(detection, ObjectDetection)
        assert detection.label == "bottle"
        assert detection.timestamp_s >= 0
        x0, y0, x1, y1 = detection.bbox_xyxy_normalized
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


# --- config 校验 ---


def test_settings_gemini_defaults(tmp_path: Path) -> None:
    # 不走 _settings（它会覆盖 gemini 字段），直接构造以验证 dataclass 默认值。
    settings = Settings(
        access_token="test-access",
        data_dir=tmp_path,
        public_base_url="https://detect.test",
    )
    assert settings.gemini_api_key is None
    assert settings.gemini_api_base == "https://generativelanguage.googleapis.com"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_proxy is None


def test_settings_rejects_non_ascii_gemini_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        _settings(tmp_path, gemini_api_key="k€y")


def test_settings_rejects_non_https_gemini_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GEMINI_API_BASE"):
        _settings(tmp_path, gemini_api_base="http://gemini.test")


def test_settings_rejects_invalid_gemini_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GEMINI_MODEL"):
        _settings(tmp_path, gemini_model="bad model!")


def test_settings_rejects_invalid_gemini_proxy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GEMINI_PROXY"):
        _settings(tmp_path, gemini_proxy="not a url")


def test_settings_accepts_socks5_gemini_proxy(tmp_path: Path) -> None:
    settings = _settings(tmp_path, gemini_proxy="socks5://127.0.0.1:7897")
    assert settings.gemini_proxy == "socks5://127.0.0.1:7897"


def test_get_settings_reads_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("GEMINI_API_KEY", "  gemini-key ")
    monkeypatch.setenv("GEMINI_API_BASE", "https://gemini.example.com")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("GEMINI_PROXY", " http://127.0.0.1:7897 ")
    settings = get_settings()
    assert settings.gemini_api_key == "gemini-key"
    assert settings.gemini_api_base == "https://gemini.example.com"
    assert settings.gemini_model == "gemini-2.5-pro"
    assert settings.gemini_proxy == "http://127.0.0.1:7897"


def test_get_settings_gemini_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    for name in ("GEMINI_API_KEY", "GEMINI_API_BASE", "GEMINI_MODEL", "GEMINI_PROXY"):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.gemini_api_key is None
    assert settings.gemini_api_base == "https://generativelanguage.googleapis.com"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_proxy is None
