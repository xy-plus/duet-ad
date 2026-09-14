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
from app.object_detect import (
    DoubaoGroundingClient,
    FalSam3GroundingClient,
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
    """真实可解码视频：qwen 需要 cv2.VideoCapture 读到实际宽高做归一化。"""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (320, 240)
    )
    assert writer.isOpened(), "cv2.VideoWriter cannot create the test video"
    for _ in range(4):
        writer.write(np.full((240, 320, 3), 127, dtype=np.uint8))
    writer.release()
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


def test_registry_reserves_eight_provider_names() -> None:
    assert tuple(object_detect.PROVIDER_REGISTRY) == (
        "doubao", "qwen", "glm", "hunyuan", "kimi", "step", "gemini", "fal_sam3",
    )
    assert object_detect.PROVIDER_REGISTRY["doubao"] is DoubaoGroundingClient
    assert object_detect.PROVIDER_REGISTRY["fal_sam3"] is FalSam3GroundingClient


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
# settings = Settings 覆盖项；response_text = 该家正常响应文本；flow = "gemini"
# 表示走 File API 上传三步流程而非 OpenAI chat completions。

_QWEN_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "bbox_2d": [32, 48, 160, 144]},'
    '{"timestamp_s": 1.5, "bbox_2d": [0, 0, 320, 240]}'
    "]}"
)

_GEMINI_GOOD = (
    '{"detections": ['
    '{"timestamp_s": 0, "box_2d": [200, 100, 400, 300]},'
    '{"timestamp_s": 1.5, "box_2d": [0, 0, 1000, 1000]}'
    "]}"
)


def _gemini_contract_handler(text: str):
    """File API resumable 上传（start → PUT finalize 即 ACTIVE）+ generateContent。"""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/upload/v1beta/files":
            return httpx.Response(
                200, headers={"X-Goog-Upload-URL": "https://gemini.test/upload/session"}
            )
        if request.method == "PUT" and path == "/upload/session":
            return httpx.Response(
                200,
                json={
                    "file": {
                        "name": "files/clip-1",
                        "uri": "https://files.test/clip-1",
                        "state": "ACTIVE",
                    }
                },
            )
        if request.method == "POST" and ":generateContent" in path:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"role": "model", "parts": [{"text": text}]}}
                    ]
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


# --- fal_sam3（fal.ai SAM 3.1 video-rle）真实派生 fixture 与队列 mock ---
# fixture 是 2026-09-14 真实校准响应的精简版（头 8 帧，单目标 handbag，
# 原值未改；来源与截取方式见文件内 _comment）。真实形态：rle/boxes/
# metadata 三个等长列表按下标=帧号对齐，box 为归一化 cxcywh。
_FAL_SAM3_FIXTURE = (
    Path(__file__).parent / "fixtures" / "fal_sam3_video_rle_response.json"
)
_FAL_SAM3_RESULT = json.loads(_FAL_SAM3_FIXTURE.read_text(encoding="utf-8"))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


# 期望的契约转换结果：cxcywh → xyxy（越界钳到 [0,1]），帧号 ÷ fps(=2)
# → timestamp_s；单目标 handbag 直接归属。
_FAL_SAM3_EXPECTED = [
    ObjectDetection(
        "handbag",
        index / 2.0,
        (
            _clamp01(box[0] - box[2] / 2),
            _clamp01(box[1] - box[3] / 2),
            _clamp01(box[0] + box[2] / 2),
            _clamp01(box[1] + box[3] / 2),
        ),
    )
    for index, box in enumerate(_FAL_SAM3_RESULT["boxes"])
    if box is not None
]


def _fal_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "fal_sam3_api_key": "fal-secret",
        "fal_sam3_api_base": "https://fal.test/fal-ai/sam-3-1",
        "fal_sam3_model": "video-rle",
        "provider_poll_interval_s": 0.01,
    }
    values.update(overrides)
    return _settings(tmp_path, **values)


def _fal_queue_handler(
    result: object,
    *,
    observed: dict[str, object] | None = None,
    statuses: tuple[str, ...] = ("IN_PROGRESS", "COMPLETED"),
    submit_status: int = 200,
) -> object:
    """fal 存储上传（initiate → PUT）+ 队列提交/轮询/取结果全链路 mock。"""
    state = {"polls": 0, "submits": 0, "uploads": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/storage/upload/initiate":
            state["uploads"] += 1
            assert request.headers["authorization"] == "Key fal-secret"
            body = json.loads(request.content)
            assert body["file_name"] == "clip.mp4"
            assert body["content_type"] == "video/mp4"
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://fal.test/upload/presigned",
                    "file_url": "https://fal.test/files/clip.mp4",
                },
            )
        if request.method == "PUT" and path == "/upload/presigned":
            assert request.headers["content-type"] == "video/mp4"
            assert request.content
            return httpx.Response(200)
        if request.method == "POST" and path == "/fal-ai/sam-3-1/video-rle":
            state["submits"] += 1
            assert request.headers["authorization"] == "Key fal-secret"
            if observed is not None:
                observed["submit_body"] = json.loads(request.content)
            if submit_status != 200:
                return httpx.Response(submit_status, json={"detail": "rejected"})
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "status_url": (
                        "https://fal.test/fal-ai/sam-3-1/video-rle"
                        "/requests/req-1/status"
                    ),
                    "response_url": (
                        "https://fal.test/fal-ai/sam-3-1/video-rle/requests/req-1"
                    ),
                },
            )
        if request.method == "GET" and path.endswith("/requests/req-1/status"):
            status = statuses[min(state["polls"], len(statuses) - 1)]
            state["polls"] += 1
            return httpx.Response(200, json={"status": status, "request_id": "req-1"})
        if request.method == "GET" and path.endswith("/requests/req-1"):
            return httpx.Response(200, json=result)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    handler.state = state  # type: ignore[attr-defined]
    return handler


CONTRACT_CASES: dict[str, dict[str, object]] = {
    "doubao": {
        "settings": {
            "doubao_api_key": "doubao-secret",
            "doubao_api_base": "https://doubao.test/api/v3",
            "doubao_model": "doubao-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
    "qwen": {
        "settings": {
            "qwen_api_key": "qwen-secret",
            "qwen_api_base": "https://qwen.test/compatible-mode/v1",
            "qwen_model": "qwen-test-model",
        },
        "response_text": _QWEN_GOOD,
    },
    "glm": {
        "settings": {
            "glm_api_key": "glm-secret",
            "glm_api_base": "https://glm.test/api/paas/v4",
            "glm_model": "glm-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
    "hunyuan": {
        "settings": {
            "hunyuan_api_key": "hunyuan-secret",
            "hunyuan_api_base": "https://hunyuan.test/v1",
            "hunyuan_model": "hunyuan-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
    "kimi": {
        "settings": {
            "kimi_api_key": "kimi-secret",
            "kimi_api_base": "https://kimi.test/v1",
            "kimi_model": "kimi-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
    "step": {
        "settings": {
            "step_api_key": "step-secret",
            "step_api_base": "https://step.test/v1",
            "step_model": "step-test-model",
        },
        "response_text": _DOUBAO_GOOD,
    },
    "gemini": {
        "settings": {
            "gemini_api_key": "gemini-secret",
            "gemini_api_base": "https://gemini.test",
            "gemini_model": "gemini-2.5-flash",
        },
        "response_text": _GEMINI_GOOD,
        "flow": "gemini",
    },
    "fal_sam3": {
        "settings": {
            "fal_sam3_api_key": "fal-secret",
            "fal_sam3_api_base": "https://fal.test/fal-ai/sam-3-1",
            "fal_sam3_model": "video-rle",
            "provider_poll_interval_s": 0.01,
        },
        "flow": "fal_sam3",
    },
}


@pytest.mark.parametrize("provider_name", sorted(CONTRACT_CASES))
def test_unified_contract(provider_name: str, tmp_path: Path, video: Path) -> None:
    case = CONTRACT_CASES[provider_name]
    settings = _settings(tmp_path, **case["settings"])  # type: ignore[arg-type]
    client_class = object_detect.PROVIDER_REGISTRY[provider_name]
    assert client_class is not None

    if case.get("flow") == "gemini":
        handler = _gemini_contract_handler(str(case["response_text"]))
    elif case.get("flow") == "fal_sam3":
        handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    else:

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
    monkeypatch.setenv("MULTIREF_MODE", "direct")
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
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    monkeypatch.delenv("OBJECT_DETECT_PROVIDERS", raising=False)
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    settings = get_settings()
    assert settings.object_detect_providers == ()
    assert settings.doubao_api_key is None


# ==== fal_sam3（fal.ai SAM 3.1 video-rle）====


def _fal_client(settings: Settings, handler) -> FalSam3GroundingClient:
    return FalSam3GroundingClient(settings, transport=httpx.MockTransport(handler))


def test_fal_sam3_detect_full_flow(tmp_path: Path, video: Path) -> None:
    """上传 → 提交 → 轮询 → 解析全链路；真实 fixture：cxcywh→xyxy、帧号→秒。"""
    observed: dict[str, object] = {}
    handler = _fal_queue_handler(_FAL_SAM3_RESULT, observed=observed)
    client = _fal_client(_fal_settings(tmp_path), handler)
    result = client.detect(video, ["handbag"])
    assert result == _FAL_SAM3_EXPECTED
    assert len(result) == 8  # fixture 头 8 帧全部有检出
    submit_body = observed["submit_body"]
    assert submit_body == {
        "video_url": "https://fal.test/files/clip.mp4",
        "prompt": "handbag",
        "detection_threshold": 0.3,
    }
    # 掩码 RLE 本期不消费，但逐帧解析记录（含 rle/score）保留供二期使用。
    assert [entry["rle"] for entry in client.last_frames] == _FAL_SAM3_RESULT["rle"]
    assert client.last_frames[0]["frame_index"] == 0
    assert client.last_frames[0]["score"] is None
    assert client.last_frames[0]["box"] == _FAL_SAM3_RESULT["boxes"][0]


def test_fal_sam3_multi_target_is_refused_before_any_http(
    tmp_path: Path, video: Path
) -> None:
    """多目标响应形态未校准：detect 直接拒绝，零 HTTP（不花钱、不猜归因）。"""
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ValueError, match="multi-target"):
        client.detect(video, ["bottle", "cup"])
    assert handler.state["submits"] == 0  # type: ignore[attr-defined]
    assert handler.state["uploads"] == 0  # type: ignore[attr-defined]


def test_fal_sam3_empty_result_is_not_retried(tmp_path: Path, video: Path) -> None:
    """专用检测模型的空结果是确定答案：返回空列表，不重提（省钱纪律）。"""
    handler = _fal_queue_handler({"rle": [], "boxes": [], "metadata": []})
    client = _fal_client(_fal_settings(tmp_path), handler)
    assert client.detect(video, ["handbag"]) == []
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_frames_without_box_are_skipped(
    tmp_path: Path, video: Path
) -> None:
    """无 box 的帧 = 该帧未检出（真实响应 253-301 帧的形态），跳过不是错误。"""
    result_payload = {
        "rle": [_FAL_SAM3_RESULT["rle"][0], None, _FAL_SAM3_RESULT["rle"][2]],
        "boxes": [_FAL_SAM3_RESULT["boxes"][0], None, _FAL_SAM3_RESULT["boxes"][2]],
        "metadata": [
            {"index": 0, "score": None, "box": _FAL_SAM3_RESULT["boxes"][0]},
            {"index": 1, "score": None, "box": None},
            {"index": 2, "score": None, "box": _FAL_SAM3_RESULT["boxes"][2]},
        ],
    }
    handler = _fal_queue_handler(result_payload)
    client = _fal_client(_fal_settings(tmp_path), handler)
    result = client.detect(video, ["handbag"])
    assert [detection.timestamp_s for detection in result] == [0.0, 1.0]
    # 未检出的帧也在 last_frames 里（box/rle 为 None），供二期完整消费。
    assert client.last_frames[1]["box"] is None
    assert client.last_frames[1]["rle"] is None


# --- 真实零检出形态（2026-09-15 项目 89f36eee 段 0001 落盘产物，见 fixture 注释）---
_FAL_SAM3_ZERO_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "fal_sam3_video_rle_zero_detection_response.json"
)
_FAL_SAM3_ZERO_RESULT = json.loads(_FAL_SAM3_ZERO_FIXTURE.read_text(encoding="utf-8"))


def test_fal_sam3_zero_detection_real_shape_returns_empty(
    tmp_path: Path, video: Path
) -> None:
    """真实零检出形态（rle 全 ""、boxes null、metadata box 全 null）是正常结果：
    返回空列表，不报错、不重提。"""
    handler = _fal_queue_handler(_FAL_SAM3_ZERO_RESULT)
    client = _fal_client(_fal_settings(tmp_path), handler)
    assert client.detect(video, ["camera bag"]) == []
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]
    # 零检出的帧也在 last_frames 里（box 为 None、rle 为 ""），供二期完整消费。
    assert len(client.last_frames) == len(_FAL_SAM3_ZERO_RESULT["rle"])
    assert all(frame["box"] is None for frame in client.last_frames)
    assert client.last_frames[0]["rle"] == ""


def test_fal_sam3_boxes_key_missing_derives_from_metadata(
    tmp_path: Path, video: Path
) -> None:
    """顶层 boxes 键整个缺失时从 metadata[i].box 派生（有检出校验证实
    metadata[i].box == boxes[i]，同一来源）。"""
    payload = {
        "rle": [_FAL_SAM3_RESULT["rle"][0], _FAL_SAM3_RESULT["rle"][1]],
        "metadata": [
            {"index": 0, "score": None, "box": _FAL_SAM3_RESULT["boxes"][0]},
            {"index": 1, "score": None, "box": _FAL_SAM3_RESULT["boxes"][1]},
        ],
    }
    handler = _fal_queue_handler(payload)
    client = _fal_client(_fal_settings(tmp_path), handler)
    result = client.detect(video, ["handbag"])
    assert [detection.timestamp_s for detection in result] == [0.0, 0.5]
    assert client.last_frames[0]["box"] == _FAL_SAM3_RESULT["boxes"][0]


def test_fal_sam3_empty_string_rle_means_not_detected(
    tmp_path: Path, video: Path
) -> None:
    """rle 空字符串 "" 与 null 同义 = 该帧未检出，即使同位置有 box 也不出检测。"""
    payload = {
        "rle": ["", _FAL_SAM3_RESULT["rle"][1]],
        "boxes": [_FAL_SAM3_RESULT["boxes"][0], _FAL_SAM3_RESULT["boxes"][1]],
        "metadata": [
            {"index": 0, "score": None, "box": _FAL_SAM3_RESULT["boxes"][0]},
            {"index": 1, "score": None, "box": _FAL_SAM3_RESULT["boxes"][1]},
        ],
    }
    handler = _fal_queue_handler(payload)
    client = _fal_client(_fal_settings(tmp_path), handler)
    result = client.detect(video, ["handbag"])
    assert [detection.timestamp_s for detection in result] == [0.5]
    assert client.last_frames[0]["rle"] == ""
    assert client.last_frames[0]["box"] == _FAL_SAM3_RESULT["boxes"][0]


def test_fal_sam3_rle_metadata_length_mismatch_is_fatal(
    tmp_path: Path, video: Path
) -> None:
    """rle/metadata 不等长 = 非预期形态：致命不重提，标记为确定性错误。"""
    handler = _fal_queue_handler(
        {
            "rle": ["", ""],
            "boxes": None,
            "metadata": [{"index": 0, "score": None, "box": None}],
        }
    )
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure) as excinfo:
        client.detect(video, ["camera bag"])
    assert isinstance(excinfo.value.cause, object_detect.UncalibratedResponseError)
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_neither_boxes_nor_metadata_is_fatal(
    tmp_path: Path, video: Path
) -> None:
    """box 既不在 boxes 也不在 metadata = 无处取框：非预期形态，确定性致命。"""
    handler = _fal_queue_handler({"rle": ["rle-a"], "boxes": None})
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure) as excinfo:
        client.detect(video, ["handbag"])
    assert isinstance(excinfo.value.cause, object_detect.UncalibratedResponseError)
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_out_of_range_box_is_clamped(tmp_path: Path, video: Path) -> None:
    """cxcywh 越界（w 超出画面）钳到 [0,1]：x0<0→0、x1>1→1。"""
    result_payload = {
        "rle": ["rle-wide"],
        "boxes": [[0.5, 0.5, 1.2, 0.6]],
        "metadata": [{"index": 0, "score": None, "box": [0.5, 0.5, 1.2, 0.6]}],
    }
    handler = _fal_queue_handler(result_payload)
    client = _fal_client(_fal_settings(tmp_path), handler)
    assert client.detect(video, ["handbag"]) == [
        ObjectDetection("handbag", 0.0, (0.0, 0.2, 1.0, 0.8))
    ]


def test_fal_sam3_rle_boxes_length_mismatch_is_fatal(
    tmp_path: Path, video: Path
) -> None:
    """rle/boxes 不等长 = 非预期形态：致命不重提，标记为确定性错误。"""
    handler = _fal_queue_handler({"rle": ["rle-a"], "boxes": []})
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure) as excinfo:
        client.detect(video, ["handbag"])
    assert isinstance(excinfo.value.cause, object_detect.UncalibratedResponseError)
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_poll_timeout_raises_provider_failure(
    tmp_path: Path, video: Path
) -> None:
    handler = _fal_queue_handler(
        _FAL_SAM3_RESULT, statuses=("IN_PROGRESS",)
    )
    settings = _fal_settings(
        tmp_path, provider_poll_interval_s=0.0, provider_poll_timeout_s=0.0
    )
    client = _fal_client(settings, handler)
    with pytest.raises(ProviderFailure) as excinfo:
        client.detect(video, ["bottle"])
    assert isinstance(excinfo.value.cause, TimeoutError)


def test_fal_sam3_task_error_is_fatal(tmp_path: Path, video: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/storage/upload/initiate":
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://fal.test/upload/presigned",
                    "file_url": "https://fal.test/files/clip.mp4",
                },
            )
        if request.method == "PUT" and path == "/upload/presigned":
            return httpx.Response(200)
        if request.method == "POST" and path == "/fal-ai/sam-3-1/video-rle":
            return httpx.Response(200, json={"request_id": "req-1"})
        if request.method == "GET" and path.endswith("/status"):
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "request_id": "req-1",
                    "error": "runner exploded",
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _fal_client(
        _fal_settings(tmp_path), handler
    )
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])


def test_fal_sam3_submit_client_error_is_fatal_without_retry(
    tmp_path: Path, video: Path
) -> None:
    handler = _fal_queue_handler(_FAL_SAM3_RESULT, submit_status=422)
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_submit_without_request_id_is_fatal(
    tmp_path: Path, video: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/storage/upload/initiate":
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://fal.test/upload/presigned",
                    "file_url": "https://fal.test/files/clip.mp4",
                },
            )
        if request.method == "PUT" and path == "/upload/presigned":
            return httpx.Response(200)
        if request.method == "POST" and path == "/fal-ai/sam-3-1/video-rle":
            return httpx.Response(200, json={"queue_position": 0})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure):
        client.detect(video, ["bottle"])


def test_fal_sam3_malformed_box_is_fatal_without_resubmit(
    tmp_path: Path, video: Path
) -> None:
    """确定性模型的畸形结果重提只会得到同样的结果：直接致命，不再花钱。"""
    bad = {
        "rle": ["rle-bad"],
        "boxes": [[0.1, 0.2, 0.3]],
        "metadata": [{"index": 0, "score": None, "box": [0.1, 0.2, 0.3]}],
    }
    handler = _fal_queue_handler(bad)
    client = _fal_client(_fal_settings(tmp_path), handler)
    with pytest.raises(ProviderFailure) as excinfo:
        client.detect(video, ["bottle"])
    assert isinstance(excinfo.value.cause, ValueError)
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


def test_fal_sam3_requires_api_key(tmp_path: Path, video: Path) -> None:
    client = _fal_client(
        _fal_settings(tmp_path, fal_sam3_api_key=None),
        lambda request: httpx.Response(200),
    )
    with pytest.raises(ValueError, match="fal_sam3 API key"):
        client.detect(video, ["bottle"])


def test_fal_sam3_empty_targets_return_empty(tmp_path: Path, video: Path) -> None:
    client = _fal_client(
        _fal_settings(tmp_path), lambda request: httpx.Response(200)
    )
    assert client.detect(video, []) == []


def test_available_providers_skips_fal_sam3_without_key(tmp_path: Path) -> None:
    settings = _fal_settings(
        tmp_path,
        object_detect_providers=("fal_sam3",),
        fal_sam3_api_key=None,
    )
    assert available_providers(settings) == []


def test_select_provider_returns_fal_sam3_client(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path, object_detect_providers=("fal_sam3",))
    client = select_provider(settings)
    assert isinstance(client, FalSam3GroundingClient)


# --- fal_sam3 产物落盘与复用（fal 调用很贵，成功结果绝不重复请求）---


def test_fal_sam3_cache_reuse_skips_http(tmp_path: Path, video: Path) -> None:
    """二次 detect 命中同指纹产物：不发任何 HTTP，结果与首次一致。"""
    cache_path = tmp_path / "work" / "object-detect-fal-sam3.json"
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        _fal_settings(tmp_path),
        transport=httpx.MockTransport(handler),
        cache_path=cache_path,
    )
    first = client.detect(video, ["handbag"])
    assert first == _FAL_SAM3_EXPECTED
    assert cache_path.is_file()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["fal_response"] == _FAL_SAM3_RESULT
    assert cached["fingerprint"]["prompt"] == "handbag"
    assert cached["fingerprint"]["detection_threshold"] == 0.3
    assert len(cached["fingerprint"]["video_sha256"]) == 64

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    reused = FalSam3GroundingClient(
        _fal_settings(tmp_path),
        transport=httpx.MockTransport(fail_handler),
        cache_path=cache_path,
    )
    assert reused.detect(video, ["handbag"]) == first
    assert [entry["rle"] for entry in reused.last_frames] == _FAL_SAM3_RESULT["rle"]


def test_fal_sam3_cache_fingerprint_mismatch_refetches(
    tmp_path: Path, video: Path
) -> None:
    """target/阈值/视频任一变化都算新任务：指纹不匹配则重新调用。"""
    cache_path = tmp_path / "object-detect-fal-sam3.json"
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        _fal_settings(tmp_path),
        transport=httpx.MockTransport(handler),
        cache_path=cache_path,
    )
    client.detect(video, ["handbag"])
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]
    client.detect(video, ["bottle"])
    assert handler.state["submits"] == 2  # type: ignore[attr-defined]


def test_fal_sam3_corrupt_cache_refetches(tmp_path: Path, video: Path) -> None:
    cache_path = tmp_path / "object-detect-fal-sam3.json"
    cache_path.write_text("not json", encoding="utf-8")
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        _fal_settings(tmp_path),
        transport=httpx.MockTransport(handler),
        cache_path=cache_path,
    )
    assert client.detect(video, ["handbag"]) == _FAL_SAM3_EXPECTED
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]


# --- fal_sam3 配置校验 ---


def test_settings_fal_sam3_threshold_defaults(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path)
    assert settings.fal_sam3_threshold == 0.3
    assert settings.fal_sam3_api_key == "fal-secret"
    assert settings.fal_sam3_api_base == "https://fal.test/fal-ai/sam-3-1"


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, True, "0.3"])
def test_settings_rejects_invalid_fal_sam3_threshold(
    tmp_path: Path, threshold: object
) -> None:
    with pytest.raises(ValueError, match="FAL_SAM3_THRESHOLD"):
        _fal_settings(tmp_path, fal_sam3_threshold=threshold)


def test_settings_rejects_invalid_fal_proxy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FAL_PROXY"):
        _fal_settings(tmp_path, fal_proxy="socks5://127.0.0.1:7897")
    with pytest.raises(ValueError, match="FAL_PROXY"):
        _fal_settings(tmp_path, fal_proxy="not a url")


def test_settings_accepts_fal_proxy(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path, fal_proxy="http://127.0.0.1:7897")
    assert settings.fal_proxy == "http://127.0.0.1:7897"


def test_get_settings_reads_fal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("FAL_KEY", "  fal-key ")
    monkeypatch.setenv("FAL_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("FAL_SAM3_THRESHOLD", "0.2")
    settings = get_settings()
    assert settings.fal_sam3_api_key == "fal-key"
    assert settings.fal_proxy == "http://127.0.0.1:7897"
    assert settings.fal_sam3_threshold == 0.2
    assert settings.fal_sam3_api_base == "https://queue.fal.run/fal-ai/sam-3-1"
    assert settings.fal_sam3_model == "video-rle"


def test_get_settings_fal_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_PROXY", raising=False)
    monkeypatch.delenv("FAL_SAM3_THRESHOLD", raising=False)
    settings = get_settings()
    assert settings.fal_sam3_api_key is None
    assert settings.fal_proxy is None
    assert settings.fal_sam3_threshold == 0.3
