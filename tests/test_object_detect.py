from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app import object_detect
from app.config import Settings, get_settings
from app.object_detect import (
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
        "request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    """真实可解码视频：fal client 需要 cv2.VideoCapture 读到实际帧率。"""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (320, 240)
    )
    assert writer.isOpened(), "cv2.VideoWriter cannot create the test video"
    for _ in range(4):
        writer.write(np.full((240, 320, 3), 127, dtype=np.uint8))
    writer.release()
    return path


# --- ObjectDetection 值对象校验 ---


def test_object_detection_accepts_valid_values() -> None:
    detection = ObjectDetection(
        label="bottle", timestamp_s=0.0,
        bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.4),
    )
    assert detection.label == "bottle"
    assert detection.timestamp_s == 0.0
    assert detection.rle is None
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
        _FAL_SAM3_RESULT["rle"][index],
    )
    for index, box in enumerate(_FAL_SAM3_RESULT["boxes"])
    if box is not None
]

# 2026-09-16 的唯一一次真实多目标（``face, necklace``）校准调用的精简
# 头 8 帧。响应仍为单个逐帧 rle/box/metadata 序列，完全没有 label、target
# id 或嵌套对象列表，故它只可解析为一个完整 prompt 的不可归因检测组。
_FAL_SAM3_MULTI_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "fal_sam3_video_rle_multi_target_response.json"
)
_FAL_SAM3_MULTI_RESULT = json.loads(
    _FAL_SAM3_MULTI_FIXTURE.read_text(encoding="utf-8")
)
_FAL_SAM3_MULTI_EXPECTED = [
    ObjectDetection(
        "face, necklace",
        index / 2.0,
        (
            _clamp01(box[0] - box[2] / 2),
            _clamp01(box[1] - box[3] / 2),
            _clamp01(box[0] + box[2] / 2),
            _clamp01(box[1] + box[3] / 2),
        ),
        rle,
    )
    for index, (rle, box) in enumerate(
        zip(_FAL_SAM3_MULTI_RESULT["rle"], _FAL_SAM3_MULTI_RESULT["boxes"])
    )
    if rle and box is not None
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
    assert [detection.rle for detection in result] == _FAL_SAM3_RESULT["rle"]
    assert client.last_frames[0]["frame_index"] == 0
    assert client.last_frames[0]["score"] is None
    assert client.last_frames[0]["box"] == _FAL_SAM3_RESULT["boxes"][0]


def test_fal_sam3_multi_target_uses_one_comma_separated_request(
    tmp_path: Path, video: Path
) -> None:
    """真实 multi fixture：多个 target 合成一个 prompt，只上传/提交一次；
    无归属字段时 label 保留完整 prompt，不能把框猜测归给 face 或 necklace。"""
    observed: dict[str, object] = {}
    handler = _fal_queue_handler(_FAL_SAM3_MULTI_RESULT, observed=observed)
    client = _fal_client(_fal_settings(tmp_path), handler)
    result = client.detect(video, ["face", "necklace"])
    assert result == _FAL_SAM3_MULTI_EXPECTED
    assert observed["submit_body"] == {
        "video_url": "https://fal.test/files/clip.mp4",
        "prompt": "face, necklace",
        "detection_threshold": 0.3,
    }
    assert handler.state["submits"] == 1  # type: ignore[attr-defined]
    assert handler.state["uploads"] == 1  # type: ignore[attr-defined]
    assert {frame["label"] for frame in client.last_frames} == {"face, necklace"}
    assert [entry["rle"] for entry in client.last_frames] == (
        _FAL_SAM3_MULTI_RESULT["rle"]
    )


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
        ObjectDetection("handbag", 0.0, (0.0, 0.2, 1.0, 0.8), "rle-wide")
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


# --- 可用性门禁与 client 构造（FAL_KEY 是唯一开关） ---


def test_available_providers_returns_fal_sam3_when_key_configured(
    tmp_path: Path,
) -> None:
    assert available_providers(_fal_settings(tmp_path)) == ["fal_sam3"]


def test_available_providers_empty_without_key(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path, fal_sam3_api_key=None)
    assert available_providers(settings) == []


def test_select_provider_returns_fal_sam3_client(tmp_path: Path) -> None:
    client = select_provider(_fal_settings(tmp_path))
    assert isinstance(client, FalSam3GroundingClient)


def test_select_provider_returns_none_without_key(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path, fal_sam3_api_key=None)
    assert select_provider(settings) is None


# --- fal_sam3 产物落盘与复用（fal 调用很贵，成功结果绝不重复请求）---
# 缓存布局（2026-09-15 起）：实例级共享目录
# <DATA_DIR>/shared/object-detect-cache/object-detect-response-<指纹sha256>.json，
# 跨项目同指纹零 HTTP；只写实例级（原子写），项目 work 目录里的旧版
# object-detect-response-<slug>.json 仅作回退读，不再写入、不做迁移。


def _shared_cache_files(settings: Settings) -> list[Path]:
    directory = settings.data_dir / "shared" / "object-detect-cache"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("object-detect-response-*.json"))


def _legacy_cache_payload(video: Path, prompt: str) -> dict[str, object]:
    return {
        "fingerprint": {
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "prompt": prompt,
            "detection_threshold": 0.3,
            "model": "video-rle",
        },
        "fal_response": _FAL_SAM3_RESULT,
    }


def test_fal_sam3_shared_cache_cross_project_reuse(
    tmp_path: Path, video: Path
) -> None:
    """实例级共享：项目 A 的 fal 响应落进 DATA_DIR/shared，项目 B 同指纹
    （同视频同 prompt）零 HTTP 复用；项目 work 目录不再写缓存文件。"""
    settings = _fal_settings(tmp_path)
    project_a_cache = (
        tmp_path / "projects" / "a" / "work" / "segments" / "0001"
        / "object-detect-response-target-a.json"
    )
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        settings,
        transport=httpx.MockTransport(handler),
        cache_path=project_a_cache,
    )
    first = client.detect(video, ["handbag"])
    assert first == _FAL_SAM3_EXPECTED
    assert not project_a_cache.exists()  # 只写实例级共享目录
    shared = _shared_cache_files(settings)
    assert len(shared) == 1
    cached = json.loads(shared[0].read_text(encoding="utf-8"))
    assert cached["fal_response"] == _FAL_SAM3_RESULT
    assert cached["fingerprint"]["prompt"] == "handbag"
    assert cached["fingerprint"]["detection_threshold"] == 0.3
    assert len(cached["fingerprint"]["video_sha256"]) == 64

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    project_b_cache = (
        tmp_path / "projects" / "b" / "work" / "segments" / "0001"
        / "object-detect-response-target-b.json"
    )
    reused = FalSam3GroundingClient(
        settings,
        transport=httpx.MockTransport(fail_handler),
        cache_path=project_b_cache,
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


def test_fal_sam3_legacy_project_cache_fallback(
    tmp_path: Path, video: Path
) -> None:
    """兼容期：旧版项目级缓存（work/segments 下的
    object-detect-response-<slug>.json）仍作回退读，命中即零 HTTP；
    回退命中只做读取，不向共享目录迁移。"""
    settings = _fal_settings(tmp_path)
    legacy = (
        tmp_path / "projects" / "a" / "work" / "segments" / "0001"
        / "object-detect-response-target-deadbeef.json"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(_legacy_cache_payload(video, "handbag")), encoding="utf-8"
    )

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = FalSam3GroundingClient(
        settings,
        transport=httpx.MockTransport(fail_handler),
        cache_path=legacy,
    )
    assert client.detect(video, ["handbag"]) == _FAL_SAM3_EXPECTED
    assert _shared_cache_files(settings) == []


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


def test_fal_sam3_corrupt_shared_cache_refetches(
    tmp_path: Path, video: Path
) -> None:
    """共享缓存文件损坏：按未命中处理重新调用，并把共享文件写回完整内容。"""
    settings = _fal_settings(tmp_path)
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        settings, transport=httpx.MockTransport(handler)
    )
    client.detect(video, ["handbag"])
    shared = _shared_cache_files(settings)
    assert len(shared) == 1
    shared[0].write_text("not json", encoding="utf-8")
    assert client.detect(video, ["handbag"]) == _FAL_SAM3_EXPECTED
    assert handler.state["submits"] == 2  # type: ignore[attr-defined]
    cached = json.loads(shared[0].read_text(encoding="utf-8"))
    assert cached["fal_response"] == _FAL_SAM3_RESULT


def test_fal_sam3_shared_cache_write_is_atomic(
    tmp_path: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写缓存 = 同目录唯一临时文件 + os.replace：replace 时临时文件已是完整
    合法 JSON（同目录 rename 才原子），replace 后无任何临时残留。"""
    settings = _fal_settings(tmp_path)
    handler = _fal_queue_handler(_FAL_SAM3_RESULT)
    client = FalSam3GroundingClient(
        settings, transport=httpx.MockTransport(handler)
    )
    replacements: list[tuple[Path, Path]] = []
    real_replace = object_detect.os.replace

    def spy_replace(src: object, dst: object) -> None:
        src_path, dst_path = Path(str(src)), Path(str(dst))
        assert src_path != dst_path
        assert src_path.parent == dst_path.parent
        payload = json.loads(src_path.read_text(encoding="utf-8"))
        assert payload["fal_response"] == _FAL_SAM3_RESULT
        replacements.append((src_path, dst_path))
        real_replace(src, dst)

    monkeypatch.setattr(object_detect.os, "replace", spy_replace)
    client.detect(video, ["handbag"])
    assert len(replacements) == 1
    temp, target = replacements[0]
    assert not temp.exists()
    assert target.is_file()
    assert list(target.parent.glob("*.tmp")) == []


def test_fal_sam3_shared_cache_concurrent_writers_leave_valid_file(
    tmp_path: Path, video: Path
) -> None:
    """并发项目同指纹同时写缓存：rename 原子性保证最终文件完整合法，
    不留半截文件或临时残留。"""
    settings = _fal_settings(tmp_path)
    transport = httpx.MockTransport(_fal_queue_handler(_FAL_SAM3_RESULT))
    clients = [
        FalSam3GroundingClient(settings, transport=transport) for _ in range(4)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda c: c.detect(video, ["handbag"]), clients))
    assert results == [_FAL_SAM3_EXPECTED] * 4
    shared = _shared_cache_files(settings)
    assert len(shared) == 1
    cached = json.loads(shared[0].read_text(encoding="utf-8"))
    assert cached["fal_response"] == _FAL_SAM3_RESULT
    assert list(shared[0].parent.glob("*.tmp")) == []


# --- fal_sam3 配置校验 ---


def test_settings_fal_sam3_defaults(tmp_path: Path) -> None:
    settings = _fal_settings(tmp_path)
    assert settings.fal_sam3_threshold == 0.3
    assert settings.fal_sam3_api_key == "fal-secret"
    assert settings.fal_sam3_api_base == "https://fal.test/fal-ai/sam-3-1"
    assert settings.fal_sam3_model == "video-rle"


def test_settings_fal_sam3_api_key_defaults_to_none(tmp_path: Path) -> None:
    assert _settings(tmp_path).fal_sam3_api_key is None


def test_settings_rejects_non_ascii_fal_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FAL_KEY"):
        _settings(tmp_path, fal_sam3_api_key="k€y")


def test_settings_rejects_invalid_fal_sam3_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FAL_SAM3_MODEL"):
        _settings(tmp_path, fal_sam3_model="bad model!")


def test_settings_rejects_non_https_fal_sam3_api_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FAL_SAM3_API_BASE"):
        _settings(tmp_path, fal_sam3_api_base="http://fal.test/fal-ai/sam-3-1")


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


def test_get_settings_ignores_removed_object_detect_providers_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBJECT_DETECT_PROVIDERS 已删除：遗留的环境变量不再被读取，也不报错。"""
    monkeypatch.setenv("ACCESS_TOKEN", "env-access")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-deepseek")
    monkeypatch.setenv("OBJECT_DETECT_PROVIDERS", "doubao,qwen")
    assert not hasattr(get_settings(), "object_detect_providers")
