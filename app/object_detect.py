"""物品检测：fal.ai SAM 3.1 video-rle，唯一的检测路径，不再有 provider 选择。

接口契约（object_mask 只面向本契约，不感知 HTTP 细节）：

- ``detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]``
  直接返回 list；致命失败（4xx 等确定性错误、配置错误、未校准响应形态）抛
  ``app.providers.ProviderFailure``。``ProviderResult`` 仅底层 HTTP 证据记录用，
  不作为 detect 返回形态。
- ``ObjectDetection``：``{label, timestamp_s, bbox_xyxy_normalized, rle}``；bbox 为
  0–1 浮点、相对帧宽/高，构造时校验（坐标 ∈[0,1]、x0<x1、y0<y1、timestamp≥0）。
- 多目标粒度：fal 端点的一次请求接受逗号分隔的 targets，``detect`` 将它们
  原顺序拼为一个 prompt，只发一次请求。已校准的响应仍是一套按帧对齐的
  rle/boxes/metadata，**没有每框所属 target 的字段**；因此多目标结果的
  ``ObjectDetection.label`` 是实际发送的完整 prompt（例如 ``"face, necklace"``），
  表示不可归因的合并检测组，而不是猜测性地归给任一 target。

为什么只剩 fal SAM-3 一家（2026-09-15 用户拍板）：此前平行接入的 7 家 VLM
（doubao/qwen/glm/hunyuan/kimi/step/gemini）输出文本 bbox，无法确定性转成
逐帧掩码，不符合需求，已全部删除；检测路径固定走 fal SAM-3，不再存在
"选择 provider"的概念。

缺 key 语义（与人脸节点"model is missing 才致命"对齐）：FAL_KEY 未配置 →
物品灰块节点不触发（等同该段无 OR 目标）；配置非法（非空但非法的
key/base/model）→ Settings 启动校验报错，即致命。provider 名称一律不出现在
公开 API 的参数与响应里。

确定性模型不做"重试 = 重新推理"：空结果即确定答案、畸形结果直接致命
（重提只会得到同样结果，白花钱）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx

from app import providers
from app.config import Settings
from app.providers import ProviderFailure


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    label: str
    timestamp_s: float
    bbox_xyxy_normalized: tuple[float, float, float, float]
    rle: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("ObjectDetection label must be a non-empty string")
        if (
            isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or self.timestamp_s < 0
        ):
            raise ValueError("ObjectDetection timestamp_s must be >= 0")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        if self.rle is not None and (not isinstance(self.rle, str) or not self.rle):
            raise ValueError("ObjectDetection rle must be a non-empty string or None")
        bbox = self.bbox_xyxy_normalized
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("ObjectDetection bbox must be a 4-tuple")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
            for value in bbox
        ):
            raise ValueError("ObjectDetection bbox coordinates must be within [0, 1]")
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if not x0 < x1 or not y0 < y1:
            raise ValueError("ObjectDetection bbox must satisfy x0 < x1 and y0 < y1")
        object.__setattr__(self, "bbox_xyxy_normalized", (x0, y0, x1, y1))


# ==== fal_sam3（fal.ai SAM 3.1 video-rle）：存储上传 + 队列轮询 ====
# 一次提交可含逗号分隔的多个 target，不按目标重试——确定性模型的空结果就是
# 确定答案，畸形结果重提只会得到同样的结果，两者都不重提（省钱纪律）。
# 流程：fal CDN 两步上传（initiate 拿预签名 PUT URL）→ POST 队列提交拿
# request_id → GET status 轮询（PROVIDER_POLL_INTERVAL_S /
# PROVIDER_POLL_TIMEOUT_S 语义）→ GET 取结果。
# 响应形态已按 2026-09-14 真实校准调用校准（fixture 来源
# data-verify/fal-sam31-calib-20260914/result.json，精简版在
# tests/fixtures/fal_sam3_video_rle_response.json）。真实响应与 fal OpenAPI
# 文档的差异：
# - 文档 Output 只声明 video（必填）与 boundingbox_frames_zip；真实响应
#   **没有 video 字段**（文档错误），但有文档未声明的逐帧结构化数据：
#   rle / boxes / metadata 三个等长列表，**下标 = 帧号**（rle 为 COCO RLE
#   字符串，boxes 为归一化 cxcywh，metadata[i].index == i 且 box 与
#   boxes[i] 同值、score 恒 null），另有 scores（恒 null）。
# - 有检出响应里未检出的帧：rle 与 boxes 同位置为 null（2026-09-14 校准
#   响应 302 帧中 49 帧如此），是正常语义不是错误。
# - 两种真实响应形态对照（均下标=帧号、rle/metadata 等长）：
#
#   | 字段          | 有检出（2026-09-14 校准，fixture tests/fixtures/fal_sam3_video_rle_response.json） | 零检出（2026-09-15 真实落盘 object-detect-response-target-9ea5aaab.json，fixture tests/fixtures/fal_sam3_video_rle_zero_detection_response.json） |
#   | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
#   | rle[i]        | 检出帧为非空 RLE 字符串，未检出帧为 null                      | 每帧都是空字符串 ""（不是 null）                              |
#   | boxes         | 等长列表，检出帧为归一化 cxcywh，未检出帧为 null               | 顶层整个为 null（键在、值 null；也可能整个键缺失）             |
#   | metadata[i]   | index==i，box 与 boxes[i] 同值（同一来源），score 恒 null      | index==i，box/score 全 null                                  |
#   | scores        | null                                                         | null                                                         |
#
#   零检出是正常结果（目标不在画面里），解析产出空 detections 而不是抛错；
#   boxes 缺失/null 时从 metadata[i].box 派生（同一来源，有检出校验证实
#   相等）；rle 空字符串 "" 与 null 同义 = 该帧未检出。
# - 多目标（逗号分隔 prompt）响应已按 2026-09-16 真实校准调用确认：仍为
#   一套 rle/boxes/metadata 按帧号对齐的合并列表，并无 target/label 字段或
#   第二维分组。也就是说盒子和 RLE 只能作为整条 prompt 的联合检测处理，
#   不能臆测其归属；详见
#   data-verify/fal-sam31-multitarget-calib-20260916/README.md 与精简 fixture
#   tests/fixtures/fal_sam3_video_rle_multi_target_response.json。
# video_url 也接受 base64 data URI，但大视频会显著拖慢请求（base64 膨胀
# 约 33% 且随请求体传输），仅作存储不可用时的兜底，未实现。
FAL_STORAGE_UPLOAD_URL = (
    "https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
)
_FAL_QUEUED_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS"})


def _probe_fps(video_path: Path) -> float:
    """cv2.VideoCapture 读视频帧率；读不到（不可解码）抛 ValueError。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"cannot read fps from video: {video_path}")
    return fps


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


class UncalibratedResponseError(ValueError):
    """fal 响应形态与已校准形态不符：确定性错误，重提只会得到同样结果。

    pipeline 的错误分类（``Pipeline._retryable_node_error`` 与
    mask-replaceable-objects 节点的 ``retry_if``）把本类型当作确定性
    4xx 同级处理：致命、不进 never-fail 退避重试。
    """


class FalSam3GroundingClient:
    """fal.ai SAM 3.1 video-rle：逐帧 RLE + 归一化 cxcywh box（下标=帧号）。

    产物复用：fal 原始响应以 {fingerprint, fal_response} 落盘，指纹（视频
    sha256 + prompt + 阈值 + 模型）匹配时直接复用，不发任何 HTTP（fal 按
    调用计费，同一指纹绝不重复付费）。缓存布局（2026-09-15 起）是**实例级
    共享目录** ``<DATA_DIR>/shared/object-detect-cache/
    object-detect-response-<指纹 sha256>.json``：同视频同 prompt 跨项目零
    HTTP（实测两轮不同项目指纹逐字节相同、fal 响应确定性相同，项目级缓存
    让第二轮白调 8 次）。只写实例级（同目录唯一临时文件 + os.replace 原子
    写，并发项目同指纹不会写出半截文件）；``cache_path`` 指向的旧版项目级
    缓存（work/segments/000N/object-detect-response-<slug>.json）仅作回退
    读，不再写入、不做迁移。契约没有 score 的对应字段，score 与 RLE 一起
    保留在落盘产物和 ``last_frames`` 里，供二期消费。

    落盘纪律（2026-09-15 用户定，同日更新为实例级共享布局）：全量落盘
    （原始响应每字段保留）是**测试期**策略——用存储换"绝不重复付费"；
    部署生产时按价值裁剪，裁剪点有两处：① 本类的共享 cache 产物（复用
    必需，保留——生产上它正是跨项目省钱的核心，可只留解析所需字段）；
    ② pipeline 节点 output-media 收编的原始响应 JSON（生产可不收编大
    字段，只留统计）。共享目录在 DATA_DIR 下、项目目录之外，
    artifact_retention 的 lean 裁剪只作用于项目目录、天然不碰它。裁剪时
    搜索本注释锚点 "落盘纪律"。
    """

    provider_name = "fal_sam3"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.cache_path = cache_path
        self.last_frames: list[dict[str, Any]] = []

    def with_cache(self, cache_path: Path) -> "FalSam3GroundingClient":
        """返回挂同一 settings/transport、带旧版项目级回退读路径的新实例。"""
        return FalSam3GroundingClient(
            self.settings, transport=self.transport, cache_path=cache_path
        )

    def detect(
        self, video_path: Path, targets: list[str]
    ) -> list[ObjectDetection]:
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise ValueError("object detect targets must be non-empty strings")
        if not targets:
            return []
        api_key = self.settings.fal_sam3_api_key
        if not api_key:
            raise ValueError(
                "fal_sam3 API key is not configured; "
                "the object mask node must not run"
            )
        content = Path(video_path).read_bytes()
        if not content:
            raise ValueError(f"object detect video input is empty: {video_path}")
        fps = _probe_fps(video_path)
        prompt = ", ".join(target.strip() for target in targets)
        fingerprint = {
            "video_sha256": hashlib.sha256(content).hexdigest(),
            "prompt": prompt,
            "detection_threshold": self.settings.fal_sam3_threshold,
            "model": self.settings.fal_sam3_model,
        }
        payload, record = self._load_cache(fingerprint)
        if payload is None:
            payload, record = self._run_flow(api_key, video_path, content, prompt)
            self._store_cache(fingerprint, payload)
        try:
            detections, self.last_frames = self._parse_result(payload, targets, fps)
        except (ValueError, TypeError, KeyError) as exc:
            # 解析失败对确定性模型是致命的（重提只会得到同样结果），
            # 包成 ProviderFailure 与契约对齐，并带上已收集的流程证据。
            raise ProviderFailure(record, [], exc) from exc
        return detections

    # --- 产物落盘与复用（实例级共享缓存 + 旧版项目级回退读） ---

    def _shared_cache_path(self, fingerprint: dict[str, Any]) -> Path:
        """实例级共享缓存路径：文件名 = 指纹规范 JSON 的 sha256。"""
        canonical = json.dumps(
            fingerprint, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return (
            self.settings.data_dir
            / "shared"
            / "object-detect-cache"
            / f"object-detect-response-{digest}.json"
        )

    def _load_cache_file(
        self, path: Path, fingerprint: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
            return None
        payload = cached.get("fal_response")
        if not isinstance(payload, dict):
            return None
        _LOGGER.info(
            "object detect provider fal_sam3: reusing cached fal response %s",
            path,
        )
        return payload

    def _load_cache(
        self, fingerprint: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """先查实例级共享缓存，再回退查旧版项目级 ``cache_path``（只读）。"""
        shared = self._shared_cache_path(fingerprint)
        payload = self._load_cache_file(shared, fingerprint)
        if payload is not None:
            return payload, {"reused_cache": str(shared)}
        if self.cache_path is not None and self.cache_path != shared:
            payload = self._load_cache_file(self.cache_path, fingerprint)
            if payload is not None:
                return payload, {"reused_cache": str(self.cache_path)}
        return None, {}

    def _store_cache(self, fingerprint: dict[str, Any], payload: Any) -> None:
        """只写实例级共享缓存，原子写：同目录唯一临时文件 + os.replace，
        并发项目同指纹也不会写出半截文件（内容相同，最后一个 rename 胜出）。"""
        path = self._shared_cache_path(fingerprint)
        temp: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".object-detect-response-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp = Path(handle.name)
                json.dump(
                    {"fingerprint": fingerprint, "fal_response": payload},
                    handle,
                    ensure_ascii=False,
                )
            os.replace(temp, path)
        except OSError:
            # 产物写盘失败不能让检测本身失败（结果已经在内存里）。
            if temp is not None:
                temp.unlink(missing_ok=True)
            _LOGGER.warning(
                "object detect provider fal_sam3: cache write failed: %s",
                path,
                exc_info=True,
            )

    # --- fal 队列流程（每步 HTTP 走 providers._send 语义） ---

    def _run_flow(
        self, api_key: str, video_path: Path, content: bytes, prompt: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record: dict[str, Any] = {}
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
        }
        if self.settings.fal_proxy:
            client_kwargs["proxy"] = self.settings.fal_proxy
        try:
            with httpx.Client(**client_kwargs) as client:
                file_url = self._upload(client, api_key, video_path, content, record)
                request_id, status_url, response_url = self._submit(
                    client, api_key, file_url, prompt, record
                )
                payload = self._poll_result(
                    client, api_key, request_id, status_url, response_url, record
                )
                return payload, record
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Key {api_key}"}

    def _upload(
        self,
        client: httpx.Client,
        api_key: str,
        video_path: Path,
        content: bytes,
        record: dict[str, Any],
    ) -> str:
        response, record["upload_initiate"] = providers._send(
            client,
            "POST",
            FAL_STORAGE_UPLOAD_URL,
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_body={
                "file_name": Path(video_path).name,
                "content_type": "video/mp4",
            },
        )
        providers._ensure_success(response, record["upload_initiate"])
        body = providers._json_body(response)
        upload_url = body.get("upload_url") if isinstance(body, dict) else None
        file_url = body.get("file_url") if isinstance(body, dict) else None
        if (
            not isinstance(upload_url, str)
            or not upload_url
            or not isinstance(file_url, str)
            or not file_url
        ):
            raise ValueError("fal storage initiate response lacks upload_url/file_url")
        put_response, record["upload_put"] = providers._send(
            client,
            "PUT",
            upload_url,
            headers={"Content-Type": "video/mp4"},
            content=content,
        )
        providers._ensure_success(put_response, record["upload_put"])
        return file_url

    def _submit(
        self,
        client: httpx.Client,
        api_key: str,
        file_url: str,
        prompt: str,
        record: dict[str, Any],
    ) -> tuple[str, str, str]:
        submit_url = (
            f"{self.settings.fal_sam3_api_base}/{self.settings.fal_sam3_model}"
        )
        # 可选的 max_num_objects（sam-3-1 新增）不传：用 fal 文档默认 16。
        body = {
            "video_url": file_url,
            "prompt": prompt,
            "detection_threshold": self.settings.fal_sam3_threshold,
        }
        response, record["submit"] = providers._send(
            client,
            "POST",
            submit_url,
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_body=body,
        )
        providers._ensure_success(response, record["submit"])
        payload = providers._json_body(response)
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("fal submit response has no request_id")
        status_url = payload.get("status_url")
        response_url = payload.get("response_url")
        if not isinstance(status_url, str) or not status_url:
            status_url = f"{submit_url}/requests/{request_id}/status"
        if not isinstance(response_url, str) or not response_url:
            response_url = f"{submit_url}/requests/{request_id}"
        return request_id, status_url, response_url

    def _poll_result(
        self,
        client: httpx.Client,
        api_key: str,
        request_id: str,
        status_url: str,
        response_url: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["polls"] = []
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        while True:
            response, exchange = providers._send(
                client, "GET", status_url, headers=self._headers(api_key)
            )
            record["polls"].append(exchange)
            providers._ensure_success(response, exchange)
            body = providers._json_body(response)
            if not isinstance(body, dict):
                raise ValueError("fal status response must be an object")
            error = body.get("error")
            if error:
                raise ValueError(f"fal task {request_id} failed: {error}")
            status = str(body.get("status") or "").upper()
            if status == "COMPLETED":
                break
            if status not in _FAL_QUEUED_STATUSES:
                raise ValueError(
                    f"fal task {request_id} has unknown status {status!r}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"fal task {request_id} timed out")
            time.sleep(self.settings.provider_poll_interval_s)
        response, record["result"] = providers._send(
            client, "GET", response_url, headers=self._headers(api_key)
        )
        providers._ensure_success(response, record["result"])
        payload = providers._json_body(response)
        if not isinstance(payload, dict):
            raise ValueError("fal result response must be an object")
        return payload

    # --- 响应解析（已按 2026-09-14 真实响应校准，见段落头注释） ---

    def _parse_result(
        self, payload: dict[str, Any], targets: list[str], fps: float
    ) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
        """真实形态：rle / boxes / metadata 按下标=帧号对齐（形态对照见段落头注释）。

        该帧未检出（box 为 null，或 rle 为 null/""）→ 跳过，不是错误；整段
        零检出 → 空 detections，正常返回。boxes 缺失/null 时从
        metadata[i].box 派生（同一来源）。列表缺失/类型不对/不等长/
        metadata 错位/box 两处皆无 = 未校准形态，抛 UncalibratedResponseError
        （确定性致命，不退避重试）。多 target 的真实响应没有归属字段，故每个
        检测的 label 是实际发送的完整逗号分隔 prompt，代表一个不可拆分的
        合并检测组；这避免将 face 或物品框猜测归给错误的 target。
        """
        rles = payload.get("rle")
        if not isinstance(rles, list):
            raise UncalibratedResponseError(
                "fal result must contain an rle list; uncalibrated response shape"
            )
        metadata_raw = payload.get("metadata")
        if metadata_raw is not None and (
            not isinstance(metadata_raw, list) or len(metadata_raw) != len(rles)
        ):
            raise UncalibratedResponseError(
                "fal result metadata length mismatch; uncalibrated response shape"
            )
        boxes_raw = payload.get("boxes")
        if boxes_raw is None:
            # 零检出真实形态：顶层 boxes 整个为 null/缺失，逐帧 box 只在
            # metadata[i].box（与 boxes[i] 同一来源，有检出校验证实相等）。
            if not isinstance(metadata_raw, list):
                raise UncalibratedResponseError(
                    "fal result has no box source (boxes and metadata both "
                    "absent); uncalibrated response shape"
                )
            boxes: list[Any] = [
                meta.get("box") if isinstance(meta, dict) else None
                for meta in metadata_raw
            ]
        elif isinstance(boxes_raw, list):
            if len(boxes_raw) != len(rles):
                raise UncalibratedResponseError(
                    "fal result rle/boxes length mismatch; "
                    "uncalibrated response shape"
                )
            boxes = boxes_raw
        else:
            raise UncalibratedResponseError(
                "fal result boxes must be a list or null; "
                "uncalibrated response shape"
            )
        metadata: list[Any] = metadata_raw or [None] * len(rles)
        prompt = ", ".join(target.strip() for target in targets)
        detections: list[ObjectDetection] = []
        frames: list[dict[str, Any]] = []
        for frame_index, rle in enumerate(rles):
            if rle is not None and not isinstance(rle, str):
                raise UncalibratedResponseError(
                    f"fal result frame {frame_index} rle must be a string or "
                    "null; uncalibrated response shape"
                )
            box = boxes[frame_index]
            meta = metadata[frame_index]
            if isinstance(meta, dict) and meta.get("index") != frame_index:
                raise UncalibratedResponseError(
                    f"fal result metadata index mismatch at frame {frame_index}; "
                    "uncalibrated response shape"
                )
            score = meta.get("score") if isinstance(meta, dict) else None
            frames.append({
                "frame_index": frame_index,
                "label": prompt,
                "score": score,
                "box": box,
                "rle": rle,
            })
            if box is None or not rle:
                continue  # 该帧未检出（box 为 null，或 rle 为 null/""——同义）
            bbox = self._cxcywh_to_xyxy(box, frame_index)
            if bbox is None:
                continue  # 钳位后退化（零面积），无框可画
            detections.append(
                ObjectDetection(
                    label=prompt,
                    timestamp_s=frame_index / fps,
                    bbox_xyxy_normalized=bbox,
                    rle=rle,
                )
            )
        return detections, frames

    @staticmethod
    def _cxcywh_to_xyxy(
        box: Any, position: int
    ) -> tuple[float, float, float, float] | None:
        """归一化 cxcywh → xyxy，越界钳到 [0,1]；畸形抛 ValueError，退化返回 None。"""
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in box
            )
        ):
            raise ValueError(f"fal result object {position - 1} has an invalid box")
        cx, cy, width, height = (float(value) for value in box)
        x0 = _clamp01(cx - width / 2)
        y0 = _clamp01(cy - height / 2)
        x1 = _clamp01(cx + width / 2)
        y1 = _clamp01(cy + height / 2)
        if not x0 < x1 or not y0 < y1:
            return None
        return (x0, y0, x1, y1)


def available_providers(settings: Settings) -> list[str]:
    """FAL_KEY 已配置返回 ["fal_sam3"]，未配置返回 []。

    只剩 fal SAM-3 一家后本函数实质是"检测是否可用"的门禁；保留列表形态
    与语义是为了 pipeline 的调用侧（有可用 provider 才触发物品灰块节点）
    不变。
    """
    if settings.fal_sam3_api_key:
        return ["fal_sam3"]
    _LOGGER.info("object detect: FAL_KEY is not configured; detection unavailable")
    return []


def select_provider(settings: Settings) -> FalSam3GroundingClient | None:
    """FAL_KEY 已配置返回 fal SAM-3 client；未配置返回 None。"""
    if not settings.fal_sam3_api_key:
        return None
    return FalSam3GroundingClient(settings)
