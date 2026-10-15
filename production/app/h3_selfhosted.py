"""自托管 MiniMax-H3（HyperFlow 蒸馏）同步生成 provider。

契约实测自 http://23.176.24.81:8000（/health 与真实生成，2026-10-10）：
- POST /v1/videos/generations 同步返回完整结果（无 task 轮询）；
  响应 {id, request: {prompt 原文回显, prompt_tokens}, data: [{url}], ...}，
  视频 GET {base}{data[0].url}（需 Bearer）。
- prompt 上限 12000 字符（超出 400 硬拒）；num_frames 仅 124/243/362
  （5s/10s/15s）；steps 固定 8；performance_mode=speed 跳过服务端 PE
  改写（冻结 prompt 必须逐字生效，quality/speed 生成同 8 步等价）。
- references[] 结构化参考（image ≤9 / video ≤3 / audio ≤3，合计 ≤12，
  body ≤80MiB）：{type,id,name,label,description,has_audio,data(dataURI)}；
  video 加 include_audio；带声音项加 audio_id/audio_label。编号规则与
  pipeline 冻结 prompt 的 "Image N / Video 1 / Audio 1" 约定对齐。
- X-H3-Client-Request-ID 幂等头（同一请求重提用同一 ID）。

与 PicsartClient 的对接：实现同名 submit/poll/download 三件套
（duck type），Pipeline 按 settings.h3_provider 选择 client，
_generate_picsart_segment 的回执账本/断点续跑/自动重提逻辑零改动：
- submit = 预检 + 同步生成，成功返回 vid id（充当 task_id）；
- poll = 直接由 vid id 推导内容 URL 返回 completed；
- download = GET 内容写盘。
失败语义：该 provider 不按次扣费（自托管 GPU），一切失败（4xx/5xx/
传输）都记 outcome="rejected"（未拿到 vid id = 服务端未交付结果），
交由既有的有限自动重提（≤3 次）与断点续跑处理。
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from app.providers import ProviderFailure, ProviderResult

# 服务端 /health 的 resolution_options（standard 档），2026-10-10 实测。
_ASPECT_SIZES = {
    "16:9": (1344, 756),
    "9:16": (756, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "1:1": (992, 992),
}
# 服务端仅接受 5s/10s/15s 三档；分段窗口 4-10s，向上靠档后由
# media.trim_24fps 裁回精确帧预算。
_NUM_FRAMES_OPTIONS = (124, 243, 362)
_NUM_FRAMES_MAX_SECONDS = (5, 10, 15)

_PROMPT_MAX_CHARS = 12000


class SelfhostedH3Client:
    """与 PicsartClient 同形的自托管 H3 同步 client（duck type）。"""

    def __init__(self, settings) -> None:
        base_url = str(getattr(settings, "h3_selfhosted_base_url", "") or "").strip()
        api_key = str(getattr(settings, "h3_selfhosted_api_key", "") or "").strip()
        if not base_url:
            raise ValueError("H3_PROVIDER=selfhosted 需要 H3_SELFHOSTED_BASE_URL")
        if not api_key:
            raise ValueError("H3_PROVIDER=selfhosted 需要 H3_SELFHOSTED_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = float(getattr(settings, "h3_selfhosted_timeout_s", 3600.0))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    def _num_frames(duration_s: int) -> int:
        for limit, frames in zip(_NUM_FRAMES_MAX_SECONDS, _NUM_FRAMES_OPTIONS):
            if duration_s <= limit:
                return frames
        return _NUM_FRAMES_OPTIONS[-1]

    @staticmethod
    def _size(aspect_ratio: str) -> tuple[int, int]:
        size = _ASPECT_SIZES.get(str(aspect_ratio).strip())
        if size is None:
            raise ValueError(f"unsupported aspect_ratio for selfhosted H3: {aspect_ratio!r}")
        return size

    def _health_preflight(self, record: dict[str, Any]) -> None:
        try:
            response = httpx.get(f"{self._base_url}/health", timeout=(10, 20))
            health = response.json() if response.status_code == 200 else {}
        except Exception as exc:
            record["health_error"] = repr(exc)
            raise ProviderFailure(record, [], exc) from exc
        record["health"] = {
            "status": health.get("status"),
            "queue": health.get("queue"),
            "model": (health.get("backend") or {}).get("model"),
        }
        problems: list[str] = []
        if health.get("status") != "ok":
            problems.append(f"status={health.get('status')!r}")
        scheduler = health.get("gpu_scheduler") or {}
        if scheduler.get("paused") or scheduler.get("fault"):
            problems.append("gpu_scheduler unavailable")
        queue = health.get("queue") or {}
        if int(queue.get("length", 0)) >= int(queue.get("max", 500)):
            problems.append("queue full")
        if problems:
            record["health_problems"] = problems
            raise ProviderFailure(
                record, [], RuntimeError("selfhosted H3 预检未通过: " + "; ".join(problems))
            )

    def _fetch_reference(self, url: str) -> tuple[str, bytes, str]:
        """provider-asset 公网 URL → (name, bytes, mime)。"""
        with httpx.Client(timeout=(10, 120), verify=False, follow_redirects=True) as client:
            response = client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"reference asset fetch failed: HTTP {response.status_code} {url[:120]}")
        name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "reference"
        mime = response.headers.get("Content-Type", "").split(";")[0].strip()
        if not mime:
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return name, response.content, mime

    @staticmethod
    def _data_uri(mime: str, content: bytes) -> str:
        return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"

    def _build_references(
        self,
        *,
        video_url: str,
        image_urls: list[str],
        audio_urls: list[str],
    ) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for order, url in enumerate(image_urls, start=1):
            name, content, mime = self._fetch_reference(url)
            references.append({
                "type": "image",
                "id": order,
                "name": name,
                "label": f"<Picture {order}>",
                "description": "",
                "has_audio": False,
                "data": self._data_uri(mime, content),
            })
        video_name, video_content, video_mime = self._fetch_reference(video_url)
        references.append({
            "type": "video",
            "id": 1,
            "name": video_name,
            "label": "<Video 1>",
            "description": "",
            "has_audio": False,
            "include_audio": False,
            "data": self._data_uri(video_mime, video_content),
        })
        for order, url in enumerate(audio_urls, start=1):
            name, content, mime = self._fetch_reference(url)
            references.append({
                "type": "audio",
                "id": order,
                "name": name,
                "label": f"<Audio {order}>",
                "description": "",
                "has_audio": True,
                "audio_id": order,
                "audio_label": f"<Audio {order}>",
                "data": self._data_uri(mime, content),
            })
        return references

    def submit(
        self,
        *,
        prompt: str,
        video_url: str,
        image_urls: list[str] | None,
        audio_urls: list[str] | None = None,
        duration: int,
        resolution: str,
        aspect_ratio: str,
    ) -> ProviderResult[str]:
        record: dict[str, Any] = {"provider": "selfhosted-h3", "api_key_role": "selfhosted"}
        image_urls = list(image_urls or [])
        audio_urls = list(audio_urls or [])
        try:
            if len(prompt) > _PROMPT_MAX_CHARS:
                record["outcome"] = "rejected"
                record["reason"] = "prompt_over_12000_chars"
                raise ProviderFailure(
                    record, [],
                    ValueError(f"prompt {len(prompt)} chars > {_PROMPT_MAX_CHARS}"),
                )
            record["outcome"] = "rejected"
            self._health_preflight(record)
            width, height = self._size(aspect_ratio)
            references = self._build_references(
                video_url=video_url, image_urls=image_urls, audio_urls=audio_urls,
            )
            payload: dict[str, Any] = {
                "prompt": prompt,
                "task": "ref2v",
                "performance_mode": "speed",
                "aspect_ratio": aspect_ratio,
                "resolution": "standard",
                "width": width,
                "height": height,
                "num_frames": self._num_frames(int(duration)),
                "steps": 8,
                "seed": 0,
                "reference_kv_cache": "auto",
                "references": references,
            }
            client_request_id = hashlib.sha256(
                "|".join(
                    [prompt, video_url, *image_urls, *audio_urls, str(duration), aspect_ratio]
                ).encode("utf-8")
            ).hexdigest()[:40]
            record["client_request_id"] = client_request_id
            started = time.monotonic()
            with httpx.Client(timeout=(10, self._timeout_s), verify=False) as client:
                response = client.post(
                    f"{self._base_url}/v1/videos/generations",
                    json=payload,
                    headers={
                        **self._headers(),
                        "X-H3-Client-Request-ID": client_request_id,
                    },
                )
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            record["http_status"] = response.status_code
            if response.status_code != 200:
                record["error_body"] = response.text[:2000]
                raise ProviderFailure(
                    record, [],
                    RuntimeError(f"selfhosted H3 HTTP {response.status_code}"),
                )
            body = response.json()
            video_id = body.get("id")
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError("selfhosted H3 response missing id")
            data = body.get("data") or []
            content_url = data[0].get("url") if data and isinstance(data[0], dict) else None
            if not isinstance(content_url, str) or not content_url.strip():
                raise ValueError("selfhosted H3 response missing data[0].url")
            record["outcome"] = "accepted"
            record["response"] = {
                "id": video_id,
                "elapsed_s": body.get("elapsed_s"),
                "prompt_tokens": (body.get("request") or {}).get("prompt_tokens"),
                "content_url": content_url,
            }
            self._content_urls = getattr(self, "_content_urls", {})
            self._content_urls[video_id] = content_url
            return ProviderResult(video_id, record)
        except ProviderFailure:
            raise
        except Exception as exc:
            record.setdefault("outcome", "rejected")
            raise ProviderFailure(record, [], exc) from exc

    def poll(self, task_id: str, *, api_key_role: str = "primary") -> ProviderResult[str]:
        content_urls = getattr(self, "_content_urls", {})
        content_url = content_urls.get(task_id) or f"/v1/videos/{task_id}/content"
        return ProviderResult(
            content_url,
            {
                "task_id": task_id,
                "polls": [],
                "api_key_role": api_key_role,
                "provider": "selfhosted-h3",
            },
        )

    def download(
        self, url: str, destination: Path, *, api_key_role: str = "primary"
    ) -> ProviderResult[Path]:
        record: dict[str, Any] = {
            "provider": "selfhosted-h3",
            "api_key_role": api_key_role,
            "url": url,
        }
        full_url = url if url.startswith("http") else f"{self._base_url}{url}"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(timeout=(10, 300), verify=False, follow_redirects=True) as client:
                with client.stream("GET", full_url, headers=self._headers()) as response:
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"selfhosted H3 download HTTP {response.status_code}"
                        )
                    with destination.open("wb") as fh:
                        for chunk in response.iter_bytes(1 << 20):
                            fh.write(chunk)
            record["size_bytes"] = destination.stat().st_size
            return ProviderResult(destination, record, [destination])
        except ProviderFailure:
            raise
        except Exception as exc:
            raise ProviderFailure(record, [], exc) from exc
