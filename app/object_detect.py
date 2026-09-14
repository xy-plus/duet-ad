"""物品检测 provider 层：8 家平行可插拔，对外同一接口契约。

接口契约（所有 provider 必须严格遵守，pipeline 只面向本契约，不感知具体厂家）：

- ``detect(video_path: Path, targets: list[str]) -> list[ObjectDetection]``
  直接返回 list；致命失败（4xx 等确定性错误、配置错误）抛
  ``app.providers.ProviderFailure``。``ProviderResult`` 仅底层 HTTP 证据记录用，
  不作为 detect 返回形态。
- ``ObjectDetection``：``{label, timestamp_s, bbox_xyxy_normalized}``；bbox 为
  0–1 浮点、相对帧宽/高，构造时校验（坐标 ∈[0,1]、x0<x1、y0<y1、timestamp≥0）。
- 重试 = 重新推理：单目标上限 3 次（首试 + 2 次，backoff 2s/5s），只重试
  transport 错误、502/503/504、解析失败/坐标越界/无合法框；4xx 立即致命。
- 单目标粒度：某目标重试上限内仍无合法框 → 跳过该目标记日志，不挡其它目标。

已实现的 8 家与格式差异（endpoint/模型默认值见 app/config.py 的
``_OBJECT_DETECT_PROVIDER_SPECS``，各家输出一律归一化到 0–1 后进入统一契约）：

- doubao：火山方舟，OpenAI 兼容；``<bbox>x1 y1 x2 y2</bbox>``，0–1000 归一化。
- qwen：阿里百炼，OpenAI 兼容；``bbox_2d`` 绝对像素，÷ 视频实际宽高归一化。
- glm：智谱 BigModel GLM-4.5V，OpenAI 兼容；保守 JSON 解析。
- hunyuan：腾讯混元 TokenHub，OpenAI 兼容；只接受 0–1000 整数四元组。
- kimi：Moonshot，OpenAI 兼容，仅 k2.6 支持视频；解析容忍格式漂移。
- step：阶跃星辰，OpenAI 兼容；沿用 ``<bbox>`` 0–1000 约定，解析同豆包。
- gemini：File API 上传 + generateContent（非 OpenAI 兼容），``box_2d``
  0–1000 归一化；国内需 GEMINI_PROXY 代理；一次 detect 内重试与多目标
  复用已上传文件句柄，不重复上传。
- fal_sam3：fal.ai SAM 3 video-rle，第一家专用检测模型（前 7 家通用 VLM）。
  存储上传 + 队列提交/轮询，一次提交覆盖全部 targets（prompt 逗号分隔）；
  逐帧归一化 cxcywh box → xyxy 钳位，frame_index ÷ 视频 fps → timestamp_s，
  label 回映射 targets；track_id/score/RLE 不进契约（无对应字段），保留在
  落盘产物与 ``last_frames`` 供二期。确定性模型不做"重试 = 重新推理"：
  空结果即确定答案、畸形结果直接致命（重提只会得到同样结果，白花钱）。
  国内需 FAL_PROXY 代理。fal 原始响应落盘为 JSON 产物（cache_path 由
  pipeline 经 object_mask 注入），同项目重试/续跑指纹匹配直接复用，绝不
  重复调用。

缺 key 语义（与人脸节点"model is missing 才致命"对齐）：
``object_detect_providers`` 未配置（空）或所列 provider 全部缺 key → 物品
灰块节点不触发（缺 key 的 provider 跳过并记日志，等同该段无 OR 目标）；
配置非法（非空但非法的 key/base/model、未知名称）→ Settings 启动校验报错，
即致命。各家效果对比不在 pipeline 内并发，由研究期独立对比脚本逐家跑；
主力与回退顺序只调 OBJECT_DETECT_PROVIDERS 的顺序即可。provider 名称
一律不出现在公开 API 的参数与响应里。

重试成本：每次重试 = 重新上传 + 重新推理（Gemini 复用文件句柄除外），
单家单目标一次 detect 费用上限 = 3 次推理。

新增一家 provider 的步骤（6 个并行代理照做）：

1. ``app/config.py`` 在"物品检测 provider"独立段落追加三字段：
   ``<name>_api_key: str | None = field(default=None, repr=False)``（缺省 None =
   未配置）、``<name>_api_base``、``<name>_model``（默认值见 spec 调研表），
   并在 ``__post_init__`` 与 ``get_settings`` 的独立段落加同款校验/读取
   （环境变量 ``<NAME>_API_KEY`` / ``<NAME>_API_BASE`` / ``<NAME>_MODEL``）。
2. 本模块新增 client 类：OpenAI 兼容的 6 家继承
   ``OpenAICompatibleGroundingClient``，只需设 ``provider_name = "<name>"``
   并实现 ``parse_response(text, target)``（坐标一律归一化到 0–1 后交给
   ``ObjectDetection`` 校验，非法条目抛 ValueError 计入重试）；上传方式或
   提示词不同再覆写 ``_video_content_part`` / ``_prompt``。Gemini（File API）
   与 fal_sam3（队列轮询）各自单独实现，不继承基类但满足同一 detect 契约。
3. 在 ``PROVIDER_REGISTRY`` 把 ``"<name>": None`` 换成新 client 类
   （None = 已预留未实现，不进可选列表）。
4. ``tests/test_object_detect.py``：把该家正常响应样例注册进
   ``CONTRACT_CASES``（统一契约参数化测试自动覆盖），并镜像豆包用例补
   正常/格式漂移/坐标越界/重试/单目标跳过测试。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx

from app import providers
from app.config import OBJECT_DETECT_PROVIDER_NAMES, Settings
from app.providers import ProviderFailure, retry_sleep


_LOGGER = logging.getLogger(__name__)

# base64 inline 上传上限；超限应走 Files API（扩展点，本轮不实现）。
INLINE_VIDEO_MAX_BYTES = 50 * 1024 * 1024
# 视频抽帧采样率：VLM 视频理解本质是每秒抽帧，bbox 最多每秒一个。
DETECT_FPS = 1.0


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    label: str
    timestamp_s: float
    bbox_xyxy_normalized: tuple[float, float, float, float]

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


class _GroundingClientBase:
    """两类 grounding client 共享的单目标重试循环与重试日志。

    重试 = 重新推理，镜像 ``app.providers._send`` 语义：首试 + 2 次，
    backoff 2s/5s，只重试 transport 错误、502/503/504、解析失败/坐标越界/
    空结果；4xx 立即致命。子类提供 ``provider_name``、``_request_url``、
    ``_headers``、``_request_body``、``_response_text``、``parse_response``；
    ``media_ref`` 是单次 detect 内复用的视频引用（OpenAI 兼容家为 base64
    content part，Gemini 为 File API 的 file_uri）。
    """

    provider_name: str = ""

    def _request_url(self) -> str:
        raise NotImplementedError

    def _headers(self, api_key: str) -> dict[str, str]:
        raise NotImplementedError

    def _request_body(self, media_ref: Any, target: str) -> dict[str, Any]:
        raise NotImplementedError

    def _response_text(self, body: Any) -> str:
        raise NotImplementedError

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        """把模型输出文本解析为合法检测列表；格式非法/坐标越界抛 ValueError。"""
        raise NotImplementedError

    def _detect_target(
        self,
        client: httpx.Client,
        api_key: str,
        media_ref: Any,
        target: str,
    ) -> list[ObjectDetection] | None:
        """单目标检测；返回 None 表示重试上限内无合法框（调用方跳过该目标）。"""
        max_attempts = len(providers.RETRY_BACKOFF_S) + 1
        url = self._request_url()
        headers = self._headers(api_key)
        attempts: list[dict[str, Any]] = []
        for attempt_number in range(1, max_attempts + 1):
            body = self._request_body(media_ref, target)
            request = client.build_request("POST", url, headers=headers, json=body)
            try:
                response = client.send(request)
            except httpx.HTTPError as exc:
                exchange = {
                    "request": {
                        "method": "POST",
                        "url": url,
                        "body": body,
                    },
                    "response": None,
                    "attempt": attempt_number,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                attempts.append(exchange)
                if not providers.is_retryable_transport_error(exc):
                    failure = {**exchange, "attempts": attempts}
                    raise ProviderFailure(failure, [], exc) from exc
                if attempt_number < max_attempts:
                    self._log_retry(attempt_number, max_attempts, str(exc))
                    retry_sleep(providers.RETRY_BACKOFF_S[attempt_number - 1])
                    continue
                return None
            exchange = providers.response_record(response, request_body=body)
            exchange["attempt"] = attempt_number
            attempts.append(exchange)
            if response.status_code in providers.RETRYABLE_STATUSES:
                if attempt_number < max_attempts:
                    self._log_retry(
                        attempt_number, max_attempts, f"HTTP {response.status_code}"
                    )
                    retry_sleep(providers.RETRY_BACKOFF_S[attempt_number - 1])
                    continue
                return None
            if not response.is_success:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    failure = {**exchange, "attempts": attempts}
                    raise ProviderFailure(failure, [], exc) from exc
            try:
                detections = self.parse_response(
                    self._response_text(providers._json_body(response)), target
                )
                if not detections:
                    raise ValueError("model returned no valid bbox")
            except (ValueError, TypeError, KeyError) as exc:
                if attempt_number < max_attempts:
                    self._log_retry(attempt_number, max_attempts, str(exc))
                    retry_sleep(providers.RETRY_BACKOFF_S[attempt_number - 1])
                    continue
                return None
            return detections
        raise AssertionError("unreachable")

    def _log_retry(self, attempt_number: int, max_attempts: int, reason: str) -> None:
        _LOGGER.warning(
            "object detect provider %s attempt %d/%d failed (%s); retrying in %gs",
            self.provider_name,
            attempt_number,
            max_attempts,
            reason,
            providers.RETRY_BACKOFF_S[attempt_number - 1],
        )


class OpenAICompatibleGroundingClient(_GroundingClientBase):
    """OpenAI 兼容 chat completions 的视频 grounding 基类（覆盖 6 家）。

    子类契约：设 ``provider_name``（对应 config 的 ``<name>_api_key/_api_base/_model``
    三字段），实现 ``parse_response``。视频默认 base64 inline、fps 默认 1。
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not self.provider_name:
            raise TypeError("provider_name must be set by the subclass")
        self.settings = settings
        self.transport = transport

    def _api_key(self) -> str | None:
        return getattr(self.settings, f"{self.provider_name}_api_key")

    def _api_base(self) -> str:
        return getattr(self.settings, f"{self.provider_name}_api_base")

    def _model(self) -> str:
        return getattr(self.settings, f"{self.provider_name}_model")

    def _request_url(self) -> str:
        return f"{self._api_base()}/chat/completions"

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _video_content_part(self, data_url: str, fps: float) -> dict[str, Any]:
        return {"type": "video_url", "video_url": {"url": data_url, "fps": fps}}

    def _prompt(self, target: str, fps: float) -> str:
        return (
            f"Watch the video (sampled at {fps:g} fps) and locate this object: "
            f"{target}\n"
            "For each sampled second where the object is visible, report the "
            "second timestamp and its bounding box.\n"
            'Output JSON only: {"detections": [{"timestamp_s": <float seconds>, '
            '"bbox": "<bbox>x1 y1 x2 y2</bbox>"}]}\n'
            "Coordinates are integers normalized to 0-1000 relative to frame "
            "width/height.\n"
            'If the object is not visible, output {"detections": []}.'
        )

    def _response_text(self, body: Any) -> str:
        if not isinstance(body, dict):
            raise TypeError("chat completion response must be an object")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("chat completion response must contain one choice")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise TypeError("chat completion choice must contain a message")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("chat completion message content must be a string")
        return content

    def _load_video_part(self, video_path: Path) -> dict[str, Any]:
        content = Path(video_path).read_bytes()
        if not content:
            raise ValueError(f"object detect video input is empty: {video_path}")
        if len(content) > INLINE_VIDEO_MAX_BYTES:
            raise ValueError(
                f"object detect video exceeds the {INLINE_VIDEO_MAX_BYTES} byte "
                "inline budget; upload through the provider Files API instead "
                "(extension point, not implemented yet)"
            )
        encoded = base64.b64encode(content).decode("ascii")
        return self._video_content_part(f"data:video/mp4;base64,{encoded}", DETECT_FPS)

    def _request_body(self, video_part: dict[str, Any], target: str) -> dict[str, Any]:
        return {
            "model": self._model(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        video_part,
                        {"type": "text", "text": self._prompt(target, DETECT_FPS)},
                    ],
                }
            ],
        }

    def detect(
        self, video_path: Path, targets: list[str]
    ) -> list[ObjectDetection]:
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise ValueError("object detect targets must be non-empty strings")
        if not targets:
            return []
        api_key = self._api_key()
        if not api_key:
            raise ValueError(
                f"{self.provider_name} API key is not configured; "
                "this provider must not be selected"
            )
        video_part = self._load_video_part(video_path)
        results: list[ObjectDetection] = []
        with httpx.Client(
            timeout=self.settings.request_timeout_s,
            transport=self.transport,
        ) as client:
            for target in targets:
                detections = self._detect_target(client, api_key, video_part, target)
                if detections is None:
                    _LOGGER.warning(
                        "object detect provider %s: target %r skipped after %d "
                        "attempts without a valid bbox; other targets continue",
                        self.provider_name,
                        target,
                        len(providers.RETRY_BACKOFF_S) + 1,
                    )
                    continue
                results.extend(detections)
        return results


_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出提取 JSON 对象，容忍 markdown 围栏和前后散文（格式漂移）。"""
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if parsed is None:
        fence = _JSON_FENCE_RE.search(text)
        candidates = []
        if fence is not None:
            candidates.append(fence.group(1).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                break
            except (json.JSONDecodeError, ValueError):
                continue
    if not isinstance(parsed, dict):
        raise ValueError("model output does not contain a JSON object")
    return parsed


_DOUBAO_BBOX_RE = re.compile(
    r"<bbox>\s*(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s*</bbox>"
)


class DoubaoGroundingClient(OpenAICompatibleGroundingClient):
    """豆包 / 火山方舟参考实现：``<bbox>x1 y1 x2 y2</bbox>``，0–1000 归一化。"""

    provider_name = "doubao"

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        payload = extract_json_object(text)
        entries = payload.get("detections")
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            bbox_text = entry.get("bbox")
            if not isinstance(bbox_text, str):
                raise TypeError(f"detection entry {index} has no bbox")
            match = _DOUBAO_BBOX_RE.fullmatch(bbox_text.strip())
            if match is None:
                raise ValueError(
                    f"detection entry {index} bbox is not a <bbox> tag"
                )
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=tuple(
                            int(value) / 1000.0 for value in match.groups()
                        ),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections


# ==== qwen（阿里百炼 Qwen-VL）实现段落 ====


def _probe_frame_size(video_path: Path) -> tuple[int, int]:
    """cv2.VideoCapture 读视频实际宽高；读不到（不可解码）抛 ValueError。"""
    capture = cv2.VideoCapture(str(video_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot read frame size from video: {video_path}")
    return width, height


def _extract_json_array(text: str) -> list[Any]:
    """容忍顶层 JSON 数组输出（Qwen grounding 惯用形态）及 markdown 围栏。"""
    candidates = [text.strip()]
    fence = _JSON_FENCE_RE.search(text)
    if fence is not None:
        candidates.append(fence.group(1).strip())
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            return parsed
    raise ValueError("model output does not contain a detections list")


class QwenGroundingClient(OpenAICompatibleGroundingClient):
    """阿里百炼 Qwen-VL：``bbox_2d`` 绝对像素坐标，÷ 视频实际宽高归一化。

    基类 ``parse_response(text, target)`` 拿不到帧尺寸，因此覆写
    ``_load_video_part`` 在加载视频时用 cv2 探测宽高存到实例；
    对外 ``detect`` 契约不变。
    """

    provider_name = "qwen"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(settings, transport)
        self._frame_size: tuple[int, int] | None = None

    def _load_video_part(self, video_path: Path) -> dict[str, Any]:
        part = super()._load_video_part(video_path)
        self._frame_size = _probe_frame_size(video_path)
        return part

    def _prompt(self, target: str, fps: float) -> str:
        return (
            f"Watch the video (sampled at {fps:g} fps) and locate this object: "
            f"{target}\n"
            "For each sampled second where the object is visible, report the "
            "second timestamp and its bounding box.\n"
            'Output JSON only: {"detections": [{"timestamp_s": <float seconds>, '
            '"bbox_2d": [x1, y1, x2, y2]}]}\n'
            "bbox_2d coordinates are absolute pixel positions in the video "
            "frame.\n"
            'If the object is not visible, output {"detections": []}.'
        )

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        if self._frame_size is None:
            raise ValueError("qwen frame size is unknown; load the video first")
        width, height = self._frame_size
        entries = self._detection_entries(text)
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            bbox = entry.get("bbox_2d")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in bbox
                )
            ):
                raise ValueError(f"detection entry {index} has an invalid bbox_2d")
            x1, y1, x2, y2 = (float(value) for value in bbox)
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=(
                            x1 / width,
                            y1 / height,
                            x2 / width,
                            y2 / height,
                        ),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections

    @staticmethod
    def _detection_entries(text: str) -> list[Any]:
        try:
            payload = extract_json_object(text)
        except ValueError:
            payload = None
        entries: Any = (
            payload.get("detections")
            if payload is not None
            else _extract_json_array(text)
        )
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        return entries


# ---- glm（智谱 BigModel）----
# bbox 输出为 prompt 产物、官方未文档化，故采用保守解析：优先提取 JSON 对象
# （复用 extract_json_object，容忍 markdown 围栏与前后散文），detections 条目
# 的 bbox 兼容两种形态：``<bbox>x1 y1 x2 y2</bbox>`` 标签字符串、四元素数字
# 数组。坐标约定为 0–1000 归一化（除以 1000 转 0–1）；若四值全部落在
# [0, 1] 内则视为模型直接给了 0–1 浮点，原样采用。不扫描 JSON 之外的裸
# <bbox> 标签：缺少 timestamp_s 无法构造合法 ObjectDetection，宁可判解析
# 失败计入重试，也不猜测时间戳。
_GLM_BBOX_TAG_RE = re.compile(
    r"<bbox>\s*(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s*</bbox>"
)


class GlmGroundingClient(OpenAICompatibleGroundingClient):
    """智谱 BigModel GLM-4.5V：OpenAI 兼容端点，保守 JSON 解析（见上方段落注释）。"""

    provider_name = "glm"

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        payload = extract_json_object(text)
        entries = payload.get("detections")
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            raw_bbox = self._entry_bbox(entry.get("bbox"), index)
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=self._normalize_bbox(raw_bbox, index),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections

    def _entry_bbox(self, bbox: Any, index: int) -> tuple[float, ...]:
        if isinstance(bbox, str):
            match = _GLM_BBOX_TAG_RE.fullmatch(bbox.strip())
            if match is None:
                raise ValueError(f"detection entry {index} bbox is not a <bbox> tag")
            return tuple(float(value) for value in match.groups())
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and not any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in bbox
        ):
            return tuple(float(value) for value in bbox)
        raise TypeError(f"detection entry {index} has no usable bbox")

    def _normalize_bbox(
        self, raw: tuple[float, ...], index: int
    ) -> tuple[float, float, float, float]:
        if any(value < 0 for value in raw):
            raise ValueError(f"detection entry {index} has negative coordinates")
        scale = 1000.0 if any(value > 1.0 for value in raw) else 1.0
        x0, y0, x1, y1 = (value / scale for value in raw)
        return (x0, y0, x1, y1)


# ==== hunyuan（腾讯混元 TokenHub）段落开始 ====
# bbox 输出格式官方未文档化（spec 调研表），沿用基类 prompt 的
# ``<bbox>x1 y1 x2 y2</bbox>`` 0–1000 整数约定，解析端保守兼容三种形态。
_HUNYUAN_BBOX_TAG_RE = re.compile(
    r"<bbox>\s*(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s*</bbox>"
)
_HUNYUAN_BBOX_PLAIN_RE = re.compile(
    r"(\d{1,4})\s*[,\s]\s*(\d{1,4})\s*[,\s]\s*(\d{1,4})\s*[,\s]\s*(\d{1,4})"
)


def _hunyuan_bbox_values(bbox: Any, index: int) -> tuple[int, int, int, int]:
    """从模型输出取 0–1000 整数四元组；无法识别抛 ValueError/TypeError。"""
    if isinstance(bbox, str):
        match = _HUNYUAN_BBOX_TAG_RE.fullmatch(bbox.strip())
        if match is None:
            match = _HUNYUAN_BBOX_PLAIN_RE.fullmatch(bbox.strip())
        if match is None:
            raise ValueError(f"detection entry {index} bbox format is unrecognized")
        return tuple(int(value) for value in match.groups())  # type: ignore[return-value]
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
    ):
        return (bbox[0], bbox[1], bbox[2], bbox[3])
    raise TypeError(f"detection entry {index} has no recognizable bbox")


class HunyuanGroundingClient(OpenAICompatibleGroundingClient):
    """腾讯混元 TokenHub：OpenAI 兼容，bbox 保守解析。

    解析只接受 0–1000 整数四元组（``<bbox>`` 标签、纯数字串、四元素整数
    数组三种形态，见上方段落注释），统一除以 1000；与 glm 不同，不做
    0–1 浮点直收。
    """

    provider_name = "hunyuan"

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        payload = extract_json_object(text)
        entries = payload.get("detections")
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            values = _hunyuan_bbox_values(entry.get("bbox"), index)
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=tuple(
                            value / 1000.0 for value in values
                        ),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections
# ==== hunyuan（腾讯混元 TokenHub）段落结束 ====


# ==== Kimi（Moonshot）====

_KIMI_BBOX_TAG_RE = re.compile(
    r"<bbox>\s*(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s+(\d{1,4})\s*</bbox>"
)
_KIMI_BBOX_PLAIN_RE = re.compile(
    r"(\d{1,4})[\s,]+(\d{1,4})[\s,]+(\d{1,4})[\s,]+(\d{1,4})"
)


class KimiGroundingClient(OpenAICompatibleGroundingClient):
    """Kimi / Moonshot：bbox 输出格式官方未文档化，采取保守解析。

    提示词沿用基类（要求 ``<bbox>x1 y1 x2 y2</bbox>``，0–1000 归一化），
    解析时额外容忍两种格式漂移：纯数字串（"x1 y1 x2 y2"）与 JSON 数组
    （``[x1, y1, x2, y2]``）；三种形态统一除以 1000 后交给
    ``ObjectDetection`` 校验。
    """

    provider_name = "kimi"

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        payload = extract_json_object(text)
        entries = payload.get("detections")
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            values = self._bbox_values(entry.get("bbox"), index)
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=tuple(
                            value / 1000.0 for value in values
                        ),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections

    def _bbox_values(self, bbox: Any, index: int) -> tuple[int, ...]:
        if isinstance(bbox, str):
            match = _KIMI_BBOX_TAG_RE.fullmatch(bbox.strip())
            if match is None:
                match = _KIMI_BBOX_PLAIN_RE.fullmatch(bbox.strip())
            if match is not None:
                return tuple(int(value) for value in match.groups())
        elif isinstance(bbox, list) and len(bbox) == 4 and all(
            isinstance(value, int) and not isinstance(value, bool) for value in bbox
        ):
            return tuple(bbox)
        raise ValueError(f"detection entry {index} has an unrecognized bbox format")


# --- step（阶跃星辰）---

_STEP_BBOX_RE = _DOUBAO_BBOX_RE


class StepGroundingClient(OpenAICompatibleGroundingClient):
    """阶跃星辰：bbox 输出格式未文档化，保守沿用基类 prompt 约定的
    ``<bbox>x1 y1 x2 y2</bbox>``（0–1000 归一化整数），解析同豆包模式。"""

    provider_name = "step"

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        payload = extract_json_object(text)
        entries = payload.get("detections")
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
            ):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            bbox_text = entry.get("bbox")
            if not isinstance(bbox_text, str):
                raise TypeError(f"detection entry {index} has no bbox")
            match = _STEP_BBOX_RE.fullmatch(bbox_text.strip())
            if match is None:
                raise ValueError(
                    f"detection entry {index} bbox is not a <bbox> tag"
                )
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=tuple(
                            int(value) / 1000.0 for value in match.groups()
                        ),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections


# ==== Gemini（Google AI Studio）独立实现：File API resumable 上传 + generateContent ====
# 不继承 OpenAICompatibleGroundingClient（上传与请求形态完全不同），满足同一 detect 契约。
# 响应 box_2d 为 0–1000 归一化，顺序 [ymin, xmin, ymax, xmax]；转 xyxy 后除 1000。
# 文件在一次 detect 内只上传一次，重试与多目标复用同一 file_uri：重试 = 重新推理，
# 单目标上限 3 次推理，与其它家一致。
# 上传后视频文件先进入 PROCESSING，generateContent 要求 ACTIVE；轮询上限次数。
GEMINI_FILE_ACTIVE_POLLS = 10


class GeminiGroundingClient(_GroundingClientBase):
    """Gemini 视频 grounding：File API 上传 + generateContent，box_2d 解析。

    不继承 OpenAICompatibleGroundingClient（上传与请求形态完全不同），
    只共享重试循环，满足同一 detect 契约。
    """

    provider_name = "gemini"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    def detect(
        self, video_path: Path, targets: list[str]
    ) -> list[ObjectDetection]:
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise ValueError("object detect targets must be non-empty strings")
        if not targets:
            return []
        api_key = self.settings.gemini_api_key
        if not api_key:
            raise ValueError(
                "gemini API key is not configured; "
                "this provider must not be selected"
            )
        content = Path(video_path).read_bytes()
        if not content:
            raise ValueError(f"object detect video input is empty: {video_path}")
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
        }
        if self.settings.gemini_proxy:
            client_kwargs["proxy"] = self.settings.gemini_proxy
        with httpx.Client(**client_kwargs) as client:
            file_uri = self._upload_video(client, api_key, video_path, content)
            results: list[ObjectDetection] = []
            for target in targets:
                detections = self._detect_target(client, api_key, file_uri, target)
                if detections is None:
                    _LOGGER.warning(
                        "object detect provider %s: target %r skipped after %d "
                        "attempts without a valid bbox; other targets continue",
                        self.provider_name,
                        target,
                        len(providers.RETRY_BACKOFF_S) + 1,
                    )
                    continue
                results.extend(detections)
        return results

    # --- File API resumable 上传 ---

    def _upload_video(
        self,
        client: httpx.Client,
        api_key: str,
        video_path: Path,
        content: bytes,
    ) -> str:
        """resumable 上传：start 会话 → PUT 字节 finalize → 轮询至 ACTIVE。

        每个 HTTP 步走 ``providers._send`` 语义（transport + 502/503/504，
        backoff 2s/5s）；上传失败是致命的（无文件则所有目标都无法推理）。
        """
        base = self.settings.gemini_api_base
        start_response, record = providers._send(
            client,
            "POST",
            f"{base}/upload/v1beta/files",
            headers={
                "x-goog-api-key": api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(content)),
                "X-Goog-Upload-Header-Content-Type": "video/mp4",
                "Content-Type": "application/json",
            },
            json_body={"file": {"display_name": Path(video_path).name}},
        )
        providers._ensure_success(start_response, record)
        upload_url = start_response.headers.get("X-Goog-Upload-URL", "").strip()
        if not upload_url:
            raise ProviderFailure(
                record,
                [],
                ValueError("gemini upload start response has no X-Goog-Upload-URL"),
            )
        finalize_response, record = providers._send(
            client,
            "PUT",
            upload_url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Length": str(len(content)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=content,
        )
        providers._ensure_success(finalize_response, record)
        body = providers._json_body(finalize_response)
        file_info = body.get("file") if isinstance(body, dict) else None
        if not isinstance(file_info, dict):
            raise ProviderFailure(
                record, [], TypeError("gemini upload response has no file object")
            )
        name = file_info.get("name")
        uri = file_info.get("uri")
        state = file_info.get("state")
        if not isinstance(name, str) or not isinstance(uri, str):
            raise ProviderFailure(
                record, [], TypeError("gemini upload file object lacks name/uri")
            )
        if state == "FAILED":
            raise ProviderFailure(
                record, [], ValueError(f"gemini file {name} processing FAILED")
            )
        if state != "ACTIVE":
            self._wait_until_active(client, api_key, name)
        return uri

    def _wait_until_active(
        self, client: httpx.Client, api_key: str, file_name: str
    ) -> None:
        """轮询文件状态直到 ACTIVE；FAILED 立即致命，超轮询预算抛 ProviderFailure。"""
        url = f"{self.settings.gemini_api_base}/v1beta/{file_name}"
        record: dict[str, Any] = {}
        for _poll in range(GEMINI_FILE_ACTIVE_POLLS):
            response, record = providers._send(
                client, "GET", url, headers={"x-goog-api-key": api_key}
            )
            providers._ensure_success(response, record)
            body = providers._json_body(response)
            state = body.get("state") if isinstance(body, dict) else None
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise ProviderFailure(
                    record, [], ValueError(f"gemini file {file_name} processing FAILED")
                )
            retry_sleep(2.0)
        raise ProviderFailure(
            record,
            [],
            ValueError(
                f"gemini file {file_name} not ACTIVE after "
                f"{GEMINI_FILE_ACTIVE_POLLS} polls"
            ),
        )

    # --- generateContent 推理（重试 = 重新推理，复用已上传 file_uri） ---

    def _request_url(self) -> str:
        return (
            f"{self.settings.gemini_api_base}/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _prompt(self, target: str, fps: float) -> str:
        return (
            f"Watch the video (sampled at {fps:g} fps) and locate this object: "
            f"{target}\n"
            "For each sampled second where the object is visible, report the "
            "second timestamp and its bounding box.\n"
            'Output JSON only: {"detections": [{"timestamp_s": <float seconds>, '
            '"box_2d": [ymin, xmin, ymax, xmax]}]}\n'
            "box_2d coordinates are normalized to 0-1000 in [ymin, xmin, ymax, "
            "xmax] order.\n"
            'If the object is not visible, output {"detections": []}.'
        )

    def _request_body(self, file_uri: str, target: str) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": file_uri,
                                "mime_type": "video/mp4",
                            }
                        },
                        {"text": self._prompt(target, DETECT_FPS)},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0},
        }

    def _response_text(self, body: Any) -> str:
        if not isinstance(body, dict):
            raise TypeError("generateContent response must be an object")
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("generateContent response must contain candidates")
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise TypeError("generateContent candidate must contain content parts")
        texts = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if not texts:
            raise TypeError("generateContent candidate has no text part")
        return "".join(texts)

    def parse_response(
        self, text: str, target: str
    ) -> list[ObjectDetection]:
        """把模型输出文本解析为合法检测列表；格式非法/坐标越界抛 ValueError。

        接受两种顶层形态：``{"detections": [...]}``（prompt 约定）或文档常见的
        裸数组 ``[{...}]``。条目必须有数值 ``timestamp_s`` 与 4 元数值
        ``box_2d``（[ymin, xmin, ymax, xmax]，0–1000），转 xyxy 除 1000 后交
        ``ObjectDetection`` 校验。
        """
        payload = _extract_gemini_json(text)
        entries = payload.get("detections") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise TypeError("model output must contain a detections list")
        detections: list[ObjectDetection] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"detection entry {index} must be an object")
            timestamp = entry.get("timestamp_s")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(f"detection entry {index} has an invalid timestamp")
            box = entry.get("box_2d")
            if (
                not isinstance(box, (list, tuple))
                or len(box) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in box
                )
            ):
                raise ValueError(f"detection entry {index} has an invalid box_2d")
            ymin, xmin, ymax, xmax = (float(value) / 1000.0 for value in box)
            try:
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=float(timestamp),
                        bbox_xyxy_normalized=(xmin, ymin, xmax, ymax),
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"detection entry {index} has invalid coordinates: {exc}"
                ) from exc
        return detections

def _extract_gemini_json(text: str) -> Any:
    """从模型输出提取 JSON（对象或数组），容忍 markdown 围栏和前后散文。"""
    candidates = [text.strip()]
    fence = _JSON_FENCE_RE.search(text)
    if fence is not None:
        candidates.append(fence.group(1).strip())
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    raise ValueError("model output does not contain a JSON object or array")


# ==== fal_sam3（fal.ai SAM 3 video-rle）独立实现：存储上传 + 队列轮询 ====
# 第一家专用检测模型（前 7 家是通用 VLM）：一次提交覆盖全部 targets
# （prompt 逗号分隔），不按目标重试——确定性模型的空结果就是确定答案，
# 畸形结果重提只会得到同样的结果，两者都不重提（省钱纪律）。
# 流程：fal CDN 两步上传（initiate 拿预签名 PUT URL）→ POST 队列提交拿
# request_id → GET status 轮询（PROVIDER_POLL_INTERVAL_S /
# PROVIDER_POLL_TIMEOUT_S 语义）→ GET 取结果。
# 逐帧结构化输出的顶层字段名未经真实响应校准：fal 文档 Output 节只声明
# video 与 boundingbox_frames_zip，这里按 SAM3VideoObjectFrame
# （frame_index + objects[SAM3ObjectMask{track_id, rle}]）与 MaskMetadata
# （index + score + 归一化 cxcywh box）假设最合理结构。校准入口 =
# 罐头 fixture tests/fixtures/fal_sam3_video_rle_response.json +
# 下面两个字段名常量。video_url 也接受 base64 data URI，但大视频会显著
# 拖慢请求（base64 膨胀约 33% 且随请求体传输），仅作存储不可用时的兜底，
# 未实现。
FAL_SAM3_FRAMES_KEY = "frames"
FAL_SAM3_METADATA_KEY = "metadata"
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


class FalSam3GroundingClient:
    """fal.ai SAM 3 video-rle：逐帧 track_id + RLE + 归一化 cxcywh box。

    满足同一 detect 契约但不继承任何基类（队列模式与单次推理完全不同）。
    产物复用：``cache_path`` 指向项目 work 目录里的 JSON 产物，内容是
    {fingerprint, fal_response}；指纹（视频 sha256 + prompt + 阈值 + 模型）
    匹配时直接复用 fal 原始响应，不发任何 HTTP（fal 调用很贵，同一项目
    重试/续跑绝不重复调用）。契约没有 track_id/score 的对应字段，它们与
    RLE 一起保留在落盘产物和 ``last_frames`` 里，供二期消费。
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
        """返回挂同一 settings/transport、带产物复用路径的新实例。"""
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
                "this provider must not be selected"
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
        payload = self._load_cache(fingerprint)
        if payload is None:
            payload, record = self._run_flow(api_key, video_path, content, prompt)
            self._store_cache(fingerprint, payload)
        else:
            record = {"reused_cache": str(self.cache_path)}
        try:
            detections, self.last_frames = self._parse_result(payload, targets, fps)
        except (ValueError, TypeError, KeyError) as exc:
            # 解析失败对确定性模型是致命的（重提只会得到同样结果），
            # 包成 ProviderFailure 与契约对齐，并带上已收集的流程证据。
            raise ProviderFailure(record, [], exc) from exc
        return detections

    # --- 产物落盘与复用 ---

    def _load_cache(self, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
        if self.cache_path is None:
            return None
        try:
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
            return None
        payload = cached.get("fal_response")
        if not isinstance(payload, dict):
            return None
        _LOGGER.info(
            "object detect provider fal_sam3: reusing cached fal response %s",
            self.cache_path,
        )
        return payload

    def _store_cache(self, fingerprint: dict[str, Any], payload: Any) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.cache_path.with_name(self.cache_path.name + ".tmp")
            temp.write_text(
                json.dumps(
                    {"fingerprint": fingerprint, "fal_response": payload},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(temp, self.cache_path)
        except OSError:
            # 产物写盘失败不能让检测本身失败（结果已经在内存里）。
            _LOGGER.warning(
                "object detect provider fal_sam3: cache write failed: %s",
                self.cache_path,
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

    # --- 响应解析（字段名以首次真实响应校准，见段落头注释） ---

    def _parse_result(
        self, payload: dict[str, Any], targets: list[str], fps: float
    ) -> tuple[list[ObjectDetection], list[dict[str, Any]]]:
        frames_raw = payload.get(FAL_SAM3_FRAMES_KEY)
        if frames_raw is None:
            raise ValueError(
                f"fal result has no {FAL_SAM3_FRAMES_KEY!r} field; the assumed "
                "response shape does not match, recalibrate against the fixture"
            )
        if not isinstance(frames_raw, list):
            raise ValueError(f"fal result {FAL_SAM3_FRAMES_KEY!r} must be a list")
        metadata_raw = payload.get(FAL_SAM3_METADATA_KEY)
        metadata = metadata_raw if isinstance(metadata_raw, list) else []
        target_lookup = {target.strip().lower(): target for target in targets}
        detections: list[ObjectDetection] = []
        frames: list[dict[str, Any]] = []
        position = 0
        for frame_position, frame in enumerate(frames_raw):
            if not isinstance(frame, dict):
                raise ValueError(f"fal result frame {frame_position} must be an object")
            frame_index = frame.get("frame_index")
            if frame_index is None:
                frame_index = frame_position
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise ValueError(
                    f"fal result frame {frame_position} has an invalid frame_index"
                )
            objects = frame.get("objects") or []
            if not isinstance(objects, list):
                raise ValueError(
                    f"fal result frame {frame_position} objects must be a list"
                )
            for entry in objects:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"fal result object {position} must be an object"
                    )
                meta = metadata[position] if position < len(metadata) else None
                if not isinstance(meta, dict):
                    meta = {}
                position += 1
                box = entry.get("box")
                if box is None:
                    box = meta.get("box")
                score = entry.get("score")
                if score is None:
                    score = meta.get("score")
                label = entry.get("label")
                if label is None:
                    label = meta.get("label")
                frames.append({
                    "frame_index": frame_index,
                    "track_id": entry.get("track_id"),
                    "label": label,
                    "score": score,
                    "box": box,
                    "rle": entry.get("rle"),
                })
                if box is None:
                    continue  # 该帧此对象无检出
                target = self._map_label(label, targets, target_lookup, position)
                if target is None:
                    continue
                bbox = self._cxcywh_to_xyxy(box, position)
                if bbox is None:
                    continue  # 钳位后退化（零面积），无框可画
                detections.append(
                    ObjectDetection(
                        label=target,
                        timestamp_s=frame_index / fps,
                        bbox_xyxy_normalized=bbox,
                    )
                )
        detections.sort(key=lambda item: (item.timestamp_s, item.label))
        return detections, frames

    def _map_label(
        self,
        label: Any,
        targets: list[str],
        target_lookup: dict[str, str],
        position: int,
    ) -> str | None:
        """把 fal 返回的 label 回映射到 targets；无法映射的跳过（记日志）。

        响应完全不带 label 时：单目标全部归该目标；多目标无法归因，
        属响应形态假设失配，抛 ValueError 触发校准。
        """
        if isinstance(label, str) and label.strip():
            target = target_lookup.get(label.strip().lower())
            if target is None:
                _LOGGER.warning(
                    "object detect provider fal_sam3: object %d label %r maps to "
                    "no target; skipped",
                    position - 1,
                    label,
                )
                return None
            return target
        if len(targets) == 1:
            return targets[0]
        raise ValueError(
            "fal result objects carry no label; cannot attribute tracks to "
            "multiple targets, recalibrate against the fixture"
        )

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


# 名称 → client 类；None = 已预留未实现（不进可选列表）。
# 与 config.OBJECT_DETECT_PROVIDER_NAMES 必须一一对应。
PROVIDER_REGISTRY: dict[str, type | None] = {
    "doubao": DoubaoGroundingClient,
    "qwen": QwenGroundingClient,  # 阿里百炼，bbox_2d 绝对像素 ÷ 帧宽高归一化
    "glm": GlmGroundingClient,
    "hunyuan": HunyuanGroundingClient,
    "kimi": KimiGroundingClient,
    "step": StepGroundingClient,
    "gemini": GeminiGroundingClient,
    "fal_sam3": FalSam3GroundingClient,  # fal.ai SAM 3 video-rle，专用检测模型
}

if set(PROVIDER_REGISTRY) != set(OBJECT_DETECT_PROVIDER_NAMES):
    raise RuntimeError("provider registry and config provider names diverged")


def available_providers(settings: Settings) -> list[str]:
    """按 ``settings.object_detect_providers`` 顺序返回可用 provider 名称。

    未实现（注册项为 None）或未配 key 的跳过并记日志；全部不可用返回空列表
    （由调用方决定物品灰块节点不触发）。
    """
    available: list[str] = []
    for name in settings.object_detect_providers:
        client_class = PROVIDER_REGISTRY[name]
        if client_class is None:
            _LOGGER.info(
                "object detect provider %r is reserved but not implemented; skipped",
                name,
            )
            continue
        if not getattr(settings, f"{name}_api_key", None):
            _LOGGER.info(
                "object detect provider %r has no API key configured; skipped",
                name,
            )
            continue
        available.append(name)
    return available


def select_provider(settings: Settings) -> Any | None:
    """返回顺序最前的可用 provider client；无可用返回 None。"""
    available = available_providers(settings)
    if not available:
        return None
    client_class = PROVIDER_REGISTRY[available[0]]
    assert client_class is not None
    return client_class(settings)
