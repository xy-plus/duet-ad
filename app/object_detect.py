"""物品检测：SAM 3.1 video-rle，provider 链路 fal → wavespeed → sandbase。

接口契约（object_mask 只面向本契约，不感知 HTTP 细节）：

- ``detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]``
  直接返回 list；致命失败（4xx 等确定性错误、配置错误、未校准响应形态）抛
  ``app.providers.ProviderFailure``。``ProviderResult`` 仅底层 HTTP 证据记录用，
  不作为 detect 返回形态。
- ``ObjectDetection``：``{label, timestamp_s, bbox_xyxy_normalized, rle}``；bbox 为
  0–1 浮点、相对帧宽/高，构造时校验（坐标 ∈[0,1]、x0<x1、y0<y1、timestamp≥0）。
  ``rle`` 一律归一化为 fal 行主序 ``(start, length)`` 对 counts 字符串——下游
  object_mask._decode_rle_mask 的既有输入格式，各 provider 的原生 RLE 格式
  （fal 行主序对 / WaveSpeed pycocotools 列主序游程）由本模块的适配层转换，
  下游零改动。
- 多目标粒度：SAM 3.1 端点的一次请求接受逗号分隔的 targets，``detect`` 将它们
  原顺序拼为一个 prompt，只发一次请求。已校准的响应仍是一套按帧对齐的
  rle/boxes/metadata，**没有每框所属 target 的字段**；因此多目标结果的
  ``ObjectDetection.label`` 是实际发送的完整 prompt（例如 ``"face, necklace"``），
  表示不可归因的合并检测组，而不是猜测性地归给任一 target。

Provider 布局（2026-09-20）：fal 排第一（``FalSam3GroundingClient``，
FAL_KEY 配置即用），其后 wavespeed（``WavespeedSam3GroundingClient``，
WAVESPEED_API_KEY）、sandbase（``SandbaseSam3GroundingClient``，
SANDBASE_API_KEY）。背景：2026-09-19 fal 旧账号被官方 Admin lock、同日
接入的 sandbase 模型 worker 连续 generation_timeout，故临时接入 WaveSpeed
救急；2026-09-20 换新 fal 账号后回到 fal 优先——fal 跑 SAM 3.1 且
detection_threshold 可调（文档建议文本 prompt 失败时降到 0.2-0.3），
wavespeed 跑 SAM 3 且无阈值参数，实测词表稍长就整组零检出（校准 round4）。
三家 provider 的归一化输出形状完全一致，下游 object_mask/pipeline 零改动。

为什么只有 SAM 3.1 这一路（2026-09-15 用户拍板）：此前平行接入的 7 家 VLM
（doubao/qwen/glm/hunyuan/kimi/step/gemini）输出文本 bbox，无法确定性转成
逐帧掩码，不符合需求，已全部删除；检测路径固定走 SAM-3 video-rle。

缺 key 语义（与人脸节点"model is missing 才致命"对齐）：WAVESPEED_API_KEY、
SANDBASE_API_KEY 与 FAL_KEY 都未配置 → 物品灰块节点不触发（等同该段无 OR
目标）；配置非法（非空但非法的 key/base/model）→ Settings 启动校验报错，
即致命。provider 名称一律不出现在公开 API 的参数与响应里。

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
import numpy as np

from app import providers
from app.config import Settings
from app.providers import ProviderFailure


_LOGGER = logging.getLogger(__name__)


def _send_detect(
    settings: Settings,
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> tuple[httpx.Response, dict[str, Any]]:
    """providers._send + 429 限流长退避预算（检测三 provider 统一启用；
    退避序列/钳制/预算语义见 providers.RATE_LIMIT_BACKOFF_S 段头注释，
    预算值来自 settings.provider_rate_limit_budget_s）。"""
    return providers._send(
        client,
        method,
        url,
        rate_limit_budget_s=settings.provider_rate_limit_budget_s,
        **kwargs,
    )


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


def _probe_frame_size(video_path: Path) -> tuple[int, int]:
    """cv2.VideoCapture 读视频帧像素尺寸 (height, width)；读不到抛 ValueError。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot read frame size from video: {video_path}")
    return (height, width)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


# ==== 检测响应的实例级共享缓存（三个 provider 共用布局） ====
# 布局：<DATA_DIR>/shared/object-detect-cache/object-detect-response-<指纹
# sha256>.json，文件内容 {"fingerprint": ..., <payload_key>: 原始响应}。
# fal 的 payload_key 是 "fal_response"（2026-09-15 起的既有磁盘格式，不能改，
# 改名会让存量缓存全部失效、白花钱重调）；sandbase/wavespeed 用
# "provider_response"。指纹含 model 字段，三家 model 字符串不同
# （"video-rle" / "meta/sam-3-1/video-rle" / "wavespeed-ai/sam3-video-rle"），
# 同视频同 prompt 也不会互相命中。


def _shared_cache_path(settings: Settings, fingerprint: dict[str, Any]) -> Path:
    """实例级共享缓存路径：文件名 = 指纹规范 JSON 的 sha256。"""
    canonical = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return (
        settings.data_dir
        / "shared"
        / "object-detect-cache"
        / f"object-detect-response-{digest}.json"
    )


def _load_cache_file(
    path: Path,
    fingerprint: dict[str, Any],
    payload_key: str,
    provider_name: str,
) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
        return None
    payload = cached.get(payload_key)
    if not isinstance(payload, dict):
        return None
    _LOGGER.info(
        "object detect provider %s: reusing cached response %s",
        provider_name,
        path,
    )
    return payload


def _load_cached_response(
    settings: Settings,
    cache_path: Path | None,
    fingerprint: dict[str, Any],
    payload_key: str,
    provider_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """先查实例级共享缓存，再回退查旧版项目级 ``cache_path``（只读）。"""
    shared = _shared_cache_path(settings, fingerprint)
    payload = _load_cache_file(shared, fingerprint, payload_key, provider_name)
    if payload is not None:
        return payload, {"reused_cache": str(shared)}
    if cache_path is not None and cache_path != shared:
        payload = _load_cache_file(
            cache_path, fingerprint, payload_key, provider_name
        )
        if payload is not None:
            return payload, {"reused_cache": str(cache_path)}
    return None, {}


def _store_cached_response(
    settings: Settings,
    fingerprint: dict[str, Any],
    payload_key: str,
    payload: Any,
    provider_name: str,
) -> None:
    """只写实例级共享缓存，原子写：同目录唯一临时文件 + os.replace，
    并发项目同指纹也不会写出半截文件（内容相同，最后一个 rename 胜出）。"""
    path = _shared_cache_path(settings, fingerprint)
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
                {"fingerprint": fingerprint, payload_key: payload},
                handle,
                ensure_ascii=False,
            )
        os.replace(temp, path)
    except OSError:
        # 产物写盘失败不能让检测本身失败（结果已经在内存里）。
        if temp is not None:
            temp.unlink(missing_ok=True)
        _LOGGER.warning(
            "object detect provider %s: cache write failed: %s",
            provider_name,
            path,
            exc_info=True,
        )


class UncalibratedResponseError(ValueError):
    """检测 provider 响应形态与已校准形态不符：确定性错误，重提只会得到同样
    结果。

    pipeline 的错误分类（``Pipeline._retryable_node_error`` 与
    mask-replaceable-objects 节点的 ``retry_if``）把本类型当作确定性
    4xx 同级处理：致命、不进 never-fail 退避重试。
    """


def _cxcywh_to_xyxy(
    box: Any, position: int, provider_name: str
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
        raise ValueError(
            f"{provider_name} result object {position - 1} has an invalid box"
        )
    cx, cy, width, height = (float(value) for value in box)
    x0 = _clamp01(cx - width / 2)
    y0 = _clamp01(cy - height / 2)
    x1 = _clamp01(cx + width / 2)
    y1 = _clamp01(cy + height / 2)
    if not x0 < x1 or not y0 < y1:
        return None
    return (x0, y0, x1, y1)


def _parse_frame_aligned_result(
    payload: dict[str, Any], targets: list[str], fps: float, provider_name: str
) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
    """SAM 3.1 video-rle 的逐帧对齐形态解析（fal/sandbase 两个部署共用）。

    真实形态（fal 2026-09-14/15/16 三次真实校准，见 fal 段落头注释的形态
    对照表）：rle / boxes / metadata 按下标=帧号对齐。该帧未检出（box 为
    null，或 rle 为 null/""）→ 跳过，不是错误；整段零检出 → 空 detections，
    正常返回。boxes 缺失/null 时从 metadata[i].box 派生（同一来源）。列表
    缺失/类型不对/不等长/metadata 错位/box 两处皆无 = 未校准形态，抛
    UncalibratedResponseError（确定性致命，不退避重试）。多 target 的真实
    响应没有归属字段，故每个检测的 label 是实际发送的完整逗号分隔 prompt，
    代表一个不可拆分的合并检测组；这避免将 face 或物品框猜测归给错误的
    target。sandbase 侧按同一模型 + 官方文档结构复用本解析，尚未经真实
    completed 响应验证（2026-09-19 三次真实任务均 generation_timeout）。
    """
    rles = payload.get("rle")
    if not isinstance(rles, list):
        raise UncalibratedResponseError(
            f"{provider_name} result must contain an rle list; "
            "uncalibrated response shape"
        )
    metadata_raw = payload.get("metadata")
    if metadata_raw is not None and (
        not isinstance(metadata_raw, list) or len(metadata_raw) != len(rles)
    ):
        raise UncalibratedResponseError(
            f"{provider_name} result metadata length mismatch; "
            "uncalibrated response shape"
        )
    boxes_raw = payload.get("boxes")
    if boxes_raw is None:
        # 零检出真实形态：顶层 boxes 整个为 null/缺失，逐帧 box 只在
        # metadata[i].box（与 boxes[i] 同一来源，有检出校验证实相等）。
        if not isinstance(metadata_raw, list):
            raise UncalibratedResponseError(
                f"{provider_name} result has no box source (boxes and metadata "
                "both absent); uncalibrated response shape"
            )
        boxes: list[Any] = [
            meta.get("box") if isinstance(meta, dict) else None
            for meta in metadata_raw
        ]
    elif isinstance(boxes_raw, list):
        if len(boxes_raw) != len(rles):
            raise UncalibratedResponseError(
                f"{provider_name} result rle/boxes length mismatch; "
                "uncalibrated response shape"
            )
        boxes = boxes_raw
    else:
        raise UncalibratedResponseError(
            f"{provider_name} result boxes must be a list or null; "
            "uncalibrated response shape"
        )
    metadata: list[Any] = metadata_raw or [None] * len(rles)
    prompt = ", ".join(target.strip() for target in targets)
    detections: list[ObjectDetection] = []
    frames: list[dict[str, Any]] = []
    for frame_index, rle in enumerate(rles):
        if rle is not None and not isinstance(rle, str):
            raise UncalibratedResponseError(
                f"{provider_name} result frame {frame_index} rle must be a "
                "string or null; uncalibrated response shape"
            )
        box = boxes[frame_index]
        meta = metadata[frame_index]
        if isinstance(meta, dict) and meta.get("index") != frame_index:
            raise UncalibratedResponseError(
                f"{provider_name} result metadata index mismatch at frame "
                f"{frame_index}; uncalibrated response shape"
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
        bbox = _cxcywh_to_xyxy(box, frame_index, provider_name)
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
    # 实现抽为模块级函数（与 sandbase provider 共用布局）；payload_key 固定
    # "fal_response" 保持 2026-09-15 以来存量缓存文件的磁盘格式。

    _PAYLOAD_KEY = "fal_response"

    def _load_cache(
        self, fingerprint: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return _load_cached_response(
            self.settings,
            self.cache_path,
            fingerprint,
            self._PAYLOAD_KEY,
            self.provider_name,
        )

    def _store_cache(self, fingerprint: dict[str, Any], payload: Any) -> None:
        _store_cached_response(
            self.settings,
            fingerprint,
            self._PAYLOAD_KEY,
            payload,
            self.provider_name,
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
        response, record["upload_initiate"] = _send_detect(self.settings, 
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
        put_response, record["upload_put"] = _send_detect(self.settings, 
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
        response, record["submit"] = _send_detect(self.settings, 
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
            response, exchange = _send_detect(self.settings, 
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
        response, record["result"] = _send_detect(self.settings, 
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
        return _parse_frame_aligned_result(payload, targets, fps, "fal")


# ==== sandbase_sam3（sandbase.ai SAM 3.1 video-rle）：multipart 上传 + 轮询 ====
# 2026-09-19 接入时替代 fal 成为主 provider（fal 账号被官方 Admin lock，
# storage upload 403；fal 代码保留作后备）；同日因其模型 worker 连续
# generation_timeout 不可用，wavespeed 接入后排到链路第二位。接口事实来自
# 官方文档
# https://www.sandbase.ai/model/meta/sam-3-1/video-rle/docs 与 2026-09-19 真实
# 校准调用（落盘 data-verify/sandbase-sam31-calib-20260919/）：
# - 本地视频先 POST {base}/v1/upload（multipart 字段名 file，视频固定
#   video/mp4——pipeline 喂的永远是 source-muted.mp4），响应
#   {"url", "content_type", "filename", "size", "type"}，取 url 提交。
# - 提交 POST {base}/v1/run 返回 202 {"id", "status", "model"}；
#   轮询 GET {base}/v1/run/{id}，status ∈ pending/running 继续，
#   completed 取 outputs，failed/timeout 带 error {type, message}。
# - 提交/上传路径的错误体是扁平 {"error": "..."}（实测：401 invalid API
#   key、404 model not found / prediction not found、400 file type not
#   allowed / failed to parse multipart form）；HTTP 层语义与 fal 相同，
#   交给 providers._send/_ensure_success 分类（4xx 确定性、5xx 网关可重试）。
# - 任务级失败实测（2026-09-19 三次真实提交全中）：
#   status=failed、error={"type":"generation_timeout","message":"poll timeout
#   after 600 attempts"}——sandbase 的该模型 worker 等 600 次无果（约
#   45-65 分钟/次），连官方示例视频也一样，是 provider 侧故障。**因此
#   completed 的输出产物格式未经真实响应验证**：下载与解析按官方文档结构
#   （outputs[] {url, content_type}）+ 同一模型的 fal 部署已校准形态实现，
#   形态不符会以 UncalibratedResponseError 确定性致命而非静默错解析；
#   worker 恢复后须用一次真实 completed 响应复核（步骤见校准 README）。
# 与 fal 相同：确定性模型空结果即确定答案、畸形结果直接致命，都不重提。
SANDBASE_QUEUED_STATUSES = frozenset({"PENDING", "RUNNING"})


class SandbaseSam3GroundingClient:
    """sandbase.ai SAM 3.1 video-rle：归一化输出与 fal provider 完全一致。

    产物复用语义与 fal client 相同（实例级共享缓存，同指纹零 HTTP；缓存
    实现见模块级 _shared_cache_path 等函数，落盘纪律搜索锚点 "落盘纪律"）。
    指纹键与 fal 相同（video_sha256/prompt/detection_threshold/model），
    靠 model 字符串不同天然隔离；payload_key 用 "provider_response"。
    """

    provider_name = "sandbase_sam3"
    _PAYLOAD_KEY = "provider_response"

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

    def with_cache(self, cache_path: Path) -> "SandbaseSam3GroundingClient":
        """返回挂同一 settings/transport、带旧版项目级回退读路径的新实例。"""
        return SandbaseSam3GroundingClient(
            self.settings, transport=self.transport, cache_path=cache_path
        )

    def detect(
        self, video_path: Path, targets: list[str]
    ) -> list[ObjectDetection]:
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise ValueError("object detect targets must be non-empty strings")
        if not targets:
            return []
        api_key = self.settings.sandbase_sam3_api_key
        if not api_key:
            raise ValueError(
                "sandbase_sam3 API key is not configured; "
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
            "detection_threshold": self.settings.sandbase_sam3_threshold,
            "model": self.settings.sandbase_sam3_model,
        }
        payload, record = _load_cached_response(
            self.settings,
            self.cache_path,
            fingerprint,
            self._PAYLOAD_KEY,
            self.provider_name,
        )
        if payload is None:
            payload, record = self._run_flow(api_key, video_path, content, prompt)
            _store_cached_response(
                self.settings,
                fingerprint,
                self._PAYLOAD_KEY,
                payload,
                self.provider_name,
            )
        try:
            detections, self.last_frames = self._parse_result(payload, targets, fps)
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderFailure(record, [], exc) from exc
        return detections

    # --- sandbase 流程（每步 HTTP 走 providers._send 语义） ---

    def _run_flow(
        self, api_key: str, video_path: Path, content: bytes, prompt: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record: dict[str, Any] = {}
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
        }
        try:
            with httpx.Client(**client_kwargs) as client:
                file_url = self._upload(client, api_key, video_path, content, record)
                run_id = self._submit(client, api_key, file_url, prompt, record)
                completed = self._poll_result(client, api_key, run_id, record)
                payload = self._download_output(client, completed, record)
                return payload, record
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _upload(
        self,
        client: httpx.Client,
        api_key: str,
        video_path: Path,
        content: bytes,
        record: dict[str, Any],
    ) -> str:
        response, record["upload"] = _send_detect(self.settings, 
            client,
            "POST",
            f"{self.settings.sandbase_sam3_api_base}/v1/upload",
            headers=self._headers(api_key),
            files={
                "file": (Path(video_path).name, content, "video/mp4"),
            },
        )
        providers._ensure_success(response, record["upload"])
        body = providers._json_body(response)
        file_url = body.get("url") if isinstance(body, dict) else None
        if not isinstance(file_url, str) or not file_url.startswith("https://"):
            raise ValueError("sandbase upload response lacks a usable url")
        return file_url

    def _submit(
        self,
        client: httpx.Client,
        api_key: str,
        file_url: str,
        prompt: str,
        record: dict[str, Any],
    ) -> str:
        # 请求体按官方文档示例全字段显式提交（不用服务端缺省，避免误用）；
        # max_num_objects 16 与 fal 文档默认一致。
        body = {
            "model": self.settings.sandbase_sam3_model,
            "video": file_url,
            "prompt": prompt,
            "apply_mask": False,
            "box_prompts": [],
            "frame_index": 0,
            "point_prompts": [],
            "boundingbox_zip": False,
            "max_num_objects": 16,
            "detection_threshold": self.settings.sandbase_sam3_threshold,
        }
        response, record["submit"] = _send_detect(self.settings, 
            client,
            "POST",
            f"{self.settings.sandbase_sam3_api_base}/v1/run",
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_body=body,
        )
        providers._ensure_success(response, record["submit"])
        payload = providers._json_body(response)
        run_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("sandbase submit response has no id")
        return run_id

    def _poll_result(
        self,
        client: httpx.Client,
        api_key: str,
        run_id: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["polls"] = []
        poll_url = f"{self.settings.sandbase_sam3_api_base}/v1/run/{run_id}"
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        while True:
            response, exchange = _send_detect(self.settings, 
                client, "GET", poll_url, headers=self._headers(api_key)
            )
            record["polls"].append(exchange)
            providers._ensure_success(response, exchange)
            body = providers._json_body(response)
            if not isinstance(body, dict):
                raise ValueError("sandbase poll response must be an object")
            status = str(body.get("status") or "").upper()
            if status == "COMPLETED":
                if not isinstance(body.get("outputs"), list):
                    raise UncalibratedResponseError(
                        "sandbase completed run has no outputs list; "
                        "uncalibrated response shape"
                    )
                return body
            if status in {"FAILED", "TIMEOUT"}:
                error = body.get("error")
                raise ValueError(
                    f"sandbase run {run_id} {status.lower()}: "
                    + json.dumps(error, ensure_ascii=False)
                )
            if status not in SANDBASE_QUEUED_STATUSES:
                raise ValueError(
                    f"sandbase run {run_id} has unknown status {status!r}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"sandbase run {run_id} timed out")
            time.sleep(self.settings.provider_poll_interval_s)

    def _download_output(
        self,
        client: httpx.Client,
        completed: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """下载 completed run 的 JSON 输出产物。

        官方文档只声明 completed 后 ``outputs[]`` 每项 ``{url, content_type}``
        （https://www.sandbase.ai/model/meta/sam-3-1/video-rle/docs）；apply_mask
        =false 时产物应是逐帧 RLE 的结构化 JSON。**本形态未经真实 completed
        响应验证**（2026-09-19 三次真实任务均 provider 侧
        generation_timeout，见 data-verify/sandbase-sam31-calib-20260919/）：
        取 outputs 里第一个 content_type 含 "json" 的项下载；产物体按同一
        模型（SAM 3.1 video-rle）的 fal 部署已校准形态（rle/boxes/metadata
        逐帧对齐）交给 _parse_frame_aligned_result。真实响应落盘后若形态
        不同，会以 UncalibratedResponseError 确定性致命，不会静默错解析。
        """
        output: Any = None
        for item in completed["outputs"]:
            if not isinstance(item, dict):
                continue
            content_type = str(item.get("content_type") or "").lower()
            if "json" in content_type and isinstance(item.get("url"), str):
                output = item
                break
        if output is None:
            raise UncalibratedResponseError(
                "sandbase completed run has no JSON output; "
                "uncalibrated response shape"
            )
        response, record["output_download"] = _send_detect(self.settings, 
            client, "GET", output["url"]
        )
        providers._ensure_success(response, record["output_download"])
        payload = providers._json_body(response)
        if not isinstance(payload, dict):
            raise UncalibratedResponseError(
                "sandbase output artifact must be an object; "
                "uncalibrated response shape"
            )
        return payload

    # --- 响应解析（按官方文档结构 + 同模型 fal 已校准形态，未经真实验证） ---

    def _parse_result(
        self, payload: dict[str, Any], targets: list[str], fps: float
    ) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
        return _parse_frame_aligned_result(payload, targets, fps, "sandbase")


# ==== wavespeed_sam3（WaveSpeed SAM3 video-rle）：直传 ticket + 轮询 ====
# 2026-09-19 接入救急（当时 fal 被锁、sandbase worker 故障）；2026-09-20
# fal 换号恢复后降为链路第二位后备。接口事实
# 已经 2026-09-19 一次真实付费调用校准（原始响应落盘
# data-verify/wavespeed-calib-20260919/，结构逐项核对见该目录 README）：
# - 上传：视频输入只接受 URL，官方直传流程（文档
#   https://wavespeed.ai/docs/upload-files-api 与官方 ComfyUI 节点源码
#   WaveSpeedAI/wavespeed-comfyui py/wavespeed_upload.py，真实调用证实）：
#   POST {base}/api/v3/media/uploads（JSON {"filename", "size",
#   "content_type"}，Bearer 鉴权）拿一次性 ticket，响应 {"code": 200,
#   "data": {"download_url", "upload": {"url", "method", "headers"}}}，再按
#   ticket 的 method/headers 把文件字节 PUT 到 upload.url（不带 API key，
#   ticket URL 本身是临时凭证，按官方要求视为不透明值逐字使用），
#   download_url 即可作为模型输入。文档另载 legacy 端点
#   /api/v3/media/upload/binary（multipart），标注兼容保留，不采用。
# - 提交/轮询：提交 POST {base}/api/v3/{model}，body {"video": <URL>,
#   "prompt": <文本>, "apply_mask": false}（prompt 支持逗号分隔多目标）；
#   响应 {"data": {"id", "status", ...}}（官方示例兼容裸 {"id": ...}，本
#   实现按 body.get("data", body) 处理）。轮询
#   GET {base}/api/v3/predictions/{id}/result，data.status 实测
#   created → processing → completed；终态失败 failed / cancelled /
#   timeout / deleted。
# - completed 的 outputs：真实响应是 **list**，每项 {"rle_url",
#   "video_url"}（ playground 样例曾显示单个 dict，两种都接受并归一化为
#   list）。单 prompt 真实调用只产 1 项；多项目（多目标）的组织方式未经
#   真实验证，按"每项一组逐帧 mask"理解、逐帧并集合并（与 fal 已校准的
#   "整条 prompt 合并检测组"语义一致），多项目帧数不一致 = 未校准形态
#   致命。
# - rle_url 产物：真实响应是顶层 **list[str]**，下标 = 帧号（5 秒 30fps
#   视频 → 150 项），每帧一个 COCO **压缩** counts 字符串（pycocotools
#   ASCII 编码，不是官方 README 示例的空白分隔数字串；README 非压缩形态
#   也保留解析路径，未经真实验证），无 size 字段（尺寸取源视频帧尺寸），
#   未检出帧 = 编码全背景的空 mask 字符串（实测 150 帧全为 "PPPb0" =
#   [h*w] 单游程，即零检出；不是 null）。counts 与 fal 的行主序
#   (start,length) 对格式不同，由下方 _decode_wavespeed_counts /
#   _encode_fal_rle_pairs 适配转换，下游零改动。
# - 计价（官方 README）：$0.05/5 秒，按视频时长 5 秒向上取整，封顶 600 秒。
# 与 fal/sandbase 相同：确定性模型空结果即确定答案、畸形结果直接致命，
# 都不重提。
WAVESPEED_PENDING_STATUSES = frozenset(
    {"CREATED", "STARTING", "QUEUED", "PENDING", "RUNNING", "PROCESSING"}
)
WAVESPEED_FAILED_STATUSES = frozenset(
    {"FAILED", "CANCELLED", "TIMEOUT", "DELETED"}
)


def _runs_to_mask(
    values: list[int], height: int, width: int, frame_index: int, provider_name: str
) -> np.ndarray | None:
    """背景起始交替游程 → 列主序 bool mask；全零（该帧未检出）返回 None。"""
    if not values or any(value < 0 for value in values):
        raise UncalibratedResponseError(
            f"{provider_name} rle frame {frame_index} counts are invalid; "
            "uncalibrated response shape"
        )
    if sum(values) != height * width:
        raise UncalibratedResponseError(
            f"{provider_name} rle frame {frame_index} counts sum "
            f"{sum(values)} != {height * width} pixels; "
            "uncalibrated response shape"
        )
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    for run_index, run_length in enumerate(values):
        if run_index % 2 == 1:
            flat[offset:offset + run_length] = True
        offset += run_length
    mask = flat.reshape((height, width), order="F")
    return mask if mask.any() else None


def _decode_coco_compressed_counts(
    counts: str, height: int, width: int, frame_index: int, provider_name: str
) -> np.ndarray | None:
    """COCO 压缩 counts（pycocotools ASCII 编码）→ 游程 → bool mask。

    真实校准形态（2026-09-19 真实调用，见段头注释）：编码与 pycocotools
    maskApi.c 严格一致——每个 count 以 5bit 块编码，char = 块值 + 48，
    0x20 为续位、末块 0x10 为负号位；第 4 个（下标 ≥3）起的 count 存与
    cnts[i-2] 的差分。本实现已用 pycocotools 对拍验证（500 个随机 mask
    编解码全等，见 data-verify/wavespeed-calib-20260919/）。
    """
    values: list[int] = []
    position = 0
    length = len(counts)
    while position < length:
        value = 0
        shift = 0
        while True:
            code = ord(counts[position]) - 48
            position += 1
            if code < 0 or code > 63:
                raise UncalibratedResponseError(
                    f"{provider_name} rle frame {frame_index} compressed "
                    "counts contain an invalid char; uncalibrated response shape"
                )
            value |= (code & 0x1F) << shift
            shift += 5
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(values) > 2:
            value += values[-2]
        values.append(value)
    return _runs_to_mask(values, height, width, frame_index, provider_name)


def _decode_pycocotools_counts(
    counts: str, height: int, width: int, frame_index: int, provider_name: str
) -> np.ndarray | None:
    """pycocotools 非压缩 counts（空白分隔数字）→ 游程 → bool mask。

    官方 README 解码示例的形态（{"counts": "146301 3 147834 11..."}），
    **未经真实响应验证**（2026-09-19 真实调用实测为压缩 ASCII 编码）；
    保留本路径是防同一模型返回 README 形态，校验同样严格（sum == h*w）。
    """
    try:
        values = [int(value) for value in counts.split()]
    except ValueError as exc:
        raise UncalibratedResponseError(
            f"{provider_name} rle frame {frame_index} counts are not integers; "
            "uncalibrated response shape"
        ) from exc
    return _runs_to_mask(values, height, width, frame_index, provider_name)


def _decode_wavespeed_counts(
    counts: Any, height: int, width: int, frame_index: int, provider_name: str
) -> np.ndarray | None:
    """按内容判别 counts 编码：纯数字+空白 → README 非压缩形态；否则 →
    COCO 压缩 ASCII（2026-09-19 真实校准形态）。空字符串 = 该帧未检出。"""
    if not isinstance(counts, str):
        raise UncalibratedResponseError(
            f"{provider_name} rle frame {frame_index} counts must be a string; "
            "uncalibrated response shape"
        )
    if not counts.strip():
        return None
    if all(character.isdigit() or character.isspace() for character in counts):
        return _decode_pycocotools_counts(
            counts, height, width, frame_index, provider_name
        )
    return _decode_coco_compressed_counts(
        counts, height, width, frame_index, provider_name
    )


def _encode_fal_rle_pairs(mask: np.ndarray) -> str:
    """bool mask → fal 行主序 ``(start, length)`` 对 counts 字符串。

    归一化输出契约：ObjectDetection.rle 一律是下游
    object_mask._decode_rle_mask 的既有输入格式（行主序、C order），
    WaveSpeed 的 pycocotools 列主序游程在适配层转换，下游零改动。调用方
    保证 mask 非空，返回字符串必非空。
    """
    flat = mask.reshape(-1).view(np.int8)  # C order（行主序）
    edges = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return " ".join(
        f"{start} {end - start}" for start, end in zip(starts, ends)
    )


def _mask_bbox_xyxy_normalized(mask: np.ndarray) -> tuple[float, float, float, float]:
    """非空 bool mask 的最小外接框，归一化到 [0,1]；必满足 x0<x1、y0<y1。"""
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return (
        cols[0] / width,
        rows[0] / height,
        (cols[-1] + 1) / width,
        (rows[-1] + 1) / height,
    )


def _parse_wavespeed_rle_result(
    payload: dict[str, Any],
    targets: list[str],
    fps: float,
    frame_size: tuple[int, int],
    provider_name: str = "wavespeed",
) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
    """WaveSpeed rle_url 产物 → 归一化 detections（bbox 从 mask 派生）。

    真实校准形态（2026-09-19 真实调用，原始响应见
    data-verify/wavespeed-calib-20260919/）：每个 rle 产物是顶层
    list[str]，下标 = 帧号，每帧一个 counts 字符串（压缩 ASCII 为实测
    形态，README 非压缩数字串保留解析路径），无 size 字段（尺寸取源视频
    帧尺寸，sum(counts) 必须恰为 h*w，强校准锚点）；未检出帧 = 全背景空
    mask 字符串。outputs 多项目（多目标）的组织方式未经真实验证（单
    prompt 真实调用只产 1 项）：按"每项一组逐帧 mask"理解、逐帧并集合并
    （与 fal 合并检测组语义一致）；多项目帧数不一致 = 未校准形态致命。
    bbox 官方输出不提供，由并集 mask 的最小外接框确定性派生；rle 重编码
    为 fal 行主序对格式（下游零改动）。多 target 无归属字段，label 为
    实际发送的完整逗号分隔 prompt（与 fal/sandbase 同语义）。除此之外的
    一切形态都抛 UncalibratedResponseError 确定性致命，不做猜测性解析。
    """
    artifacts = payload.get("rle_artifacts") if isinstance(payload, dict) else None
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or any(
            not isinstance(artifact, list)
            or any(not isinstance(item, str) for item in artifact)
            for artifact in artifacts
        )
    ):
        raise UncalibratedResponseError(
            f"{provider_name} rle artifacts must be a non-empty list of "
            "per-frame counts string lists; uncalibrated response shape"
        )
    frame_counts = {len(artifact) for artifact in artifacts}
    if len(frame_counts) != 1:
        raise UncalibratedResponseError(
            f"{provider_name} rle artifacts have mismatched frame counts; "
            "uncalibrated response shape"
        )
    video_height, video_width = frame_size
    prompt = ", ".join(target.strip() for target in targets)
    detections: list[ObjectDetection] = []
    frames: list[dict[str, Any]] = []
    for frame_index in range(len(artifacts[0])):
        union: np.ndarray | None = None
        for artifact in artifacts:
            mask = _decode_wavespeed_counts(
                artifact[frame_index],
                video_height,
                video_width,
                frame_index,
                provider_name,
            )
            if mask is not None:
                union = mask if union is None else union | mask
        if union is None:
            frames.append({
                "frame_index": frame_index,
                "label": prompt,
                "score": None,
                "box": None,
                "rle": None,
            })
            continue
        bbox = _mask_bbox_xyxy_normalized(union)
        rle = _encode_fal_rle_pairs(union)
        frames.append({
            "frame_index": frame_index,
            "label": prompt,
            "score": None,
            "box": list(bbox),
            "rle": rle,
        })
        detections.append(
            ObjectDetection(
                label=prompt,
                timestamp_s=frame_index / fps,
                bbox_xyxy_normalized=bbox,
                rle=rle,
            )
        )
    return detections, frames


class WavespeedSam3GroundingClient:
    """WaveSpeed SAM3 video-rle：归一化输出与 fal/sandbase provider 完全一致。

    产物复用语义与 fal/sandbase client 相同（实例级共享缓存，同指纹零
    HTTP；缓存实现见模块级 _shared_cache_path 等函数，落盘纪律搜索锚点
    "落盘纪律"）。指纹键为 video_sha256/prompt/model（该模型无置信度阈值
    参数），靠 model 字符串与其他 provider 天然隔离；payload_key 用
    "provider_response"。缓存 payload = {"completed_outputs": 轮询
    completed 的原始 outputs 列表（每项 rle_url/video_url 原样保留）,
    "rle_artifacts": 各 rle_url 下载的原始 JSON}——原始响应字段一个不丢。
    """

    provider_name = "wavespeed_sam3"
    _PAYLOAD_KEY = "provider_response"

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

    def with_cache(self, cache_path: Path) -> "WavespeedSam3GroundingClient":
        """返回挂同一 settings/transport、带旧版项目级回退读路径的新实例。"""
        return WavespeedSam3GroundingClient(
            self.settings, transport=self.transport, cache_path=cache_path
        )

    def detect(
        self, video_path: Path, targets: list[str]
    ) -> list[ObjectDetection]:
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise ValueError("object detect targets must be non-empty strings")
        if not targets:
            return []
        api_key = self.settings.wavespeed_sam3_api_key
        if not api_key:
            raise ValueError(
                "wavespeed_sam3 API key is not configured; "
                "the object mask node must not run"
            )
        content = Path(video_path).read_bytes()
        if not content:
            raise ValueError(f"object detect video input is empty: {video_path}")
        fps = _probe_fps(video_path)
        frame_size = _probe_frame_size(video_path)
        prompt = ", ".join(target.strip() for target in targets)
        fingerprint = {
            "video_sha256": hashlib.sha256(content).hexdigest(),
            "prompt": prompt,
            "model": self.settings.wavespeed_sam3_model,
        }
        payload, record = _load_cached_response(
            self.settings,
            self.cache_path,
            fingerprint,
            self._PAYLOAD_KEY,
            self.provider_name,
        )
        if payload is None:
            payload, record = self._run_flow(api_key, video_path, content, prompt)
            _store_cached_response(
                self.settings,
                fingerprint,
                self._PAYLOAD_KEY,
                payload,
                self.provider_name,
            )
        try:
            detections, self.last_frames = self._parse_result(
                payload, targets, fps, frame_size
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderFailure(record, [], exc) from exc
        return detections

    # --- wavespeed 流程（每步 HTTP 走 providers._send 语义） ---

    def _run_flow(
        self, api_key: str, video_path: Path, content: bytes, prompt: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record: dict[str, Any] = {}
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
        }
        try:
            with httpx.Client(**client_kwargs) as client:
                file_url = self._upload(client, api_key, video_path, content, record)
                prediction_id = self._submit(client, api_key, file_url, prompt, record)
                outputs = self._poll_result(client, api_key, prediction_id, record)
                artifacts = self._download_rles(client, outputs, record)
                return {
                    "completed_outputs": outputs,
                    "rle_artifacts": artifacts,
                }, record
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _upload(
        self,
        client: httpx.Client,
        api_key: str,
        video_path: Path,
        content: bytes,
        record: dict[str, Any],
    ) -> str:
        """官方直传流程：POST ticket → 按 ticket 把字节 PUT 到一次性 URL。

        ticket 端点与响应形态来自官方文档 upload-files-api 与官方 ComfyUI
        节点源码（见段头注释），未经真实响应验证；ticket 的 upload.url 按
        官方要求视为不透明值逐字使用。PUT 不带 API key（ticket URL 本身是
        临时凭证）。
        """
        response, record["upload_ticket"] = _send_detect(self.settings, 
            client,
            "POST",
            f"{self.settings.wavespeed_sam3_api_base}/api/v3/media/uploads",
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_body={
                "filename": Path(video_path).name,
                "size": len(content),
                "content_type": "video/mp4",
            },
        )
        providers._ensure_success(response, record["upload_ticket"])
        body = providers._json_body(response)
        data = body.get("data") if isinstance(body, dict) else None
        upload = data.get("upload") if isinstance(data, dict) else None
        download_url = data.get("download_url") if isinstance(data, dict) else None
        if (
            not isinstance(download_url, str)
            or not download_url.startswith("https://")
            or not isinstance(upload, dict)
            or not isinstance(upload.get("url"), str)
            or not upload["url"]
        ):
            raise ValueError(
                "wavespeed upload ticket response lacks download_url/upload.url"
            )
        method = str(upload.get("method") or "PUT").upper()
        if method not in {"PUT", "POST"}:
            raise ValueError(
                f"wavespeed upload ticket has unsupported method {method!r}"
            )
        upload_headers = upload.get("headers")
        if not isinstance(upload_headers, dict):
            upload_headers = {}
        put_response, record["upload_put"] = _send_detect(self.settings, 
            client,
            method,
            upload["url"],
            headers={str(key): str(value) for key, value in upload_headers.items()},
            content=content,
        )
        providers._ensure_success(put_response, record["upload_put"])
        return download_url

    def _submit(
        self,
        client: httpx.Client,
        api_key: str,
        file_url: str,
        prompt: str,
        record: dict[str, Any],
    ) -> str:
        # 请求体按官方模型页示例显式提交：apply_mask=False（mask 不渲进视频，
        # 只要 RLE JSON）；point/box prompts 本链路永远没有坐标输入，不传。
        # 该模型 schema 无 detection_threshold 参数（与 fal/sandbase 不同）。
        body = {
            "video": file_url,
            "prompt": prompt,
            "apply_mask": False,
        }
        response, record["submit"] = _send_detect(self.settings, 
            client,
            "POST",
            f"{self.settings.wavespeed_sam3_api_base}/api/v3/"
            f"{self.settings.wavespeed_sam3_model}",
            headers={**self._headers(api_key), "Content-Type": "application/json"},
            json_body=body,
        )
        providers._ensure_success(response, record["submit"])
        payload = providers._json_body(response)
        # 官方示例代码按 body.get("data", body) 兼容两种响应包络。
        data = payload.get("data", payload) if isinstance(payload, dict) else None
        prediction_id = data.get("id") if isinstance(data, dict) else None
        if isinstance(prediction_id, bool):
            prediction_id = None
        elif isinstance(prediction_id, int):
            prediction_id = str(prediction_id)
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ValueError("wavespeed submit response has no id")
        return prediction_id

    def _poll_result(
        self,
        client: httpx.Client,
        api_key: str,
        prediction_id: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        record["polls"] = []
        poll_url = (
            f"{self.settings.wavespeed_sam3_api_base}/api/v3/predictions/"
            f"{prediction_id}/result"
        )
        deadline = time.monotonic() + self.settings.provider_poll_timeout_s
        while True:
            response, exchange = _send_detect(self.settings, 
                client, "GET", poll_url, headers=self._headers(api_key)
            )
            record["polls"].append(exchange)
            providers._ensure_success(response, exchange)
            body = providers._json_body(response)
            if not isinstance(body, dict):
                raise ValueError("wavespeed poll response must be an object")
            data = body.get("data", body)
            if not isinstance(data, dict):
                raise ValueError("wavespeed poll response data must be an object")
            status = str(data.get("status") or "").upper()
            if status == "COMPLETED":
                # 真实校准形态（2026-09-19）：outputs 是 list，每项
                # {"rle_url", "video_url"}；playground 样例曾显示单个
                # dict，两种都接受并归一化为 list。
                outputs = data.get("outputs")
                if isinstance(outputs, dict):
                    outputs = [outputs]
                if (
                    not isinstance(outputs, list)
                    or not outputs
                    or any(
                        not isinstance(item, dict)
                        or not isinstance(item.get("rle_url"), str)
                        or not item["rle_url"]
                        for item in outputs
                    )
                ):
                    raise UncalibratedResponseError(
                        "wavespeed completed prediction has no outputs list "
                        "with rle_url; uncalibrated response shape"
                    )
                return outputs
            if status in WAVESPEED_FAILED_STATUSES:
                error = data.get("error")
                raise ValueError(
                    f"wavespeed prediction {prediction_id} {status.lower()}: "
                    + json.dumps(error, ensure_ascii=False)
                )
            if status not in WAVESPEED_PENDING_STATUSES:
                raise ValueError(
                    f"wavespeed prediction {prediction_id} has unknown status "
                    f"{status!r}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"wavespeed prediction {prediction_id} timed out"
                )
            time.sleep(self.settings.provider_poll_interval_s)

    def _download_rles(
        self,
        client: httpx.Client,
        outputs: list[dict[str, Any]],
        record: dict[str, Any],
    ) -> list[Any]:
        """下载 completed prediction 各项目的 RLE JSON 产物（rle_url 为
        cloudfront 公开 URL，无需鉴权）。各项 video_url（预览 mp4）不下载，
        随 outputs 原始字段一起进缓存落盘。"""
        record["rle_downloads"] = []
        artifacts: list[Any] = []
        for item in outputs:
            response, exchange = _send_detect(self.settings, client, "GET", item["rle_url"])
            record["rle_downloads"].append(exchange)
            providers._ensure_success(response, exchange)
            payload = providers._json_body(response)
            if not isinstance(payload, list):
                raise UncalibratedResponseError(
                    "wavespeed rle artifact must be a per-frame list; "
                    "uncalibrated response shape"
                )
            artifacts.append(payload)
        return artifacts

    # --- 响应解析（真实校准形态见解析函数头注释） ---

    def _parse_result(
        self,
        payload: dict[str, Any],
        targets: list[str],
        fps: float,
        frame_size: tuple[int, int],
    ) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
        return _parse_wavespeed_rle_result(
            payload, targets, fps, frame_size, "wavespeed"
        )


def available_providers(settings: Settings) -> list[str]:
    """FAL_KEY 已配置返回 ["fal_sam3"]（链路第一位）；未配置则
    WAVESPEED_API_KEY 已配置返回 ["wavespeed_sam3"]；再未配置则
    SANDBASE_API_KEY 已配置返回 ["sandbase_sam3"]；都未配置返回 []。

    本函数实质是"检测是否可用"的门禁；保留列表形态与语义是为了 pipeline
    的调用侧（有可用 provider 才触发物品灰块节点）不变。provider 名称一律
    不出现在公开 API 的参数与响应里。
    """
    if settings.fal_sam3_api_key:
        return ["fal_sam3"]
    if settings.wavespeed_sam3_api_key:
        return ["wavespeed_sam3"]
    if settings.sandbase_sam3_api_key:
        return ["sandbase_sam3"]
    _LOGGER.info(
        "object detect: none of FAL_KEY, WAVESPEED_API_KEY, SANDBASE_API_KEY "
        "is configured; detection unavailable"
    )
    return []


def select_provider(
    settings: Settings,
) -> (
    WavespeedSam3GroundingClient
    | SandbaseSam3GroundingClient
    | FalSam3GroundingClient
    | None
):
    """按链路顺序返回第一个已配置 key 的 client：fal → wavespeed →
    sandbase；都未配置返回 None。"""
    if settings.fal_sam3_api_key:
        return FalSam3GroundingClient(settings)
    if settings.wavespeed_sam3_api_key:
        return WavespeedSam3GroundingClient(settings)
    if settings.sandbase_sam3_api_key:
        return SandbaseSam3GroundingClient(settings)
    return None
