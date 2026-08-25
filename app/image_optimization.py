"""Frozen per-segment image-optimization prompts and strict CAS editing."""

from __future__ import annotations

import hashlib
from copy import deepcopy

from app.config import (
    SEEDREAM_EDIT_MODES,
    SEEDREAM_MODELS,
    SEEDREAM_PROMPT_TEMPLATES,
    Settings,
)

MAX_PROMPT_BYTES = 32 * 1024

_INTENSITY = {
    "light": "轻度优化：尽量少改，只做满足下列约束所需的最小替换。",
    "balanced": "均衡优化：在保持镜头叙事不变的前提下，清晰完成主体、服装和环境替换。",
    "strong": "强力优化：明显重设计人物、服装、场景和道具，但严格保持镜头与动作语义。",
}

_RULES = (
    "人物替换为同性别、同族裔、同风格、同年龄段、相近体型但完全不同的新面孔；"
    "服装保持同色同风格但换成不同款式；场景和道具保持同类但换成不同具体设计。"
    "锁定原画幅、构图、人物动作、机位、镜头语言和光线。"
    "跨帧重复出现的同一元素必须保持同一套新设计。"
    "画面中禁止出现任何文字、字幕、logo、watermark 或 icon。"
)


class ImageOptimizationError(ValueError):
    def __init__(self, status: int, detail: str | dict[str, str]):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_prompt(source_prompt: object, template: str) -> str:
    if template not in _INTENSITY:
        raise ValueError("unsupported prompt template")
    context = source_prompt.strip() if isinstance(source_prompt, str) else ""
    prefix = f"原镜头语义参考：{context}\n" if context else ""
    return prefix + _INTENSITY[template] + _RULES + "只输出编辑后的图1。"


def _segment_indices(meta: dict) -> list[int]:
    segments = meta.get("segments")
    if not segments:
        if segments is not None and not isinstance(segments, list):
            raise ValueError("invalid image optimization segments")
        return [0]
    if not isinstance(segments, list) or any(not isinstance(item, dict) for item in segments):
        raise ValueError("invalid image optimization segments")
    indices = [item.get("index") for item in segments]
    if (
        any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or indices != list(range(1, len(indices) + 1))
    ):
        raise ValueError("invalid image optimization segment indices")
    return indices


def freeze_prompts(settings: Settings, meta: dict) -> dict:
    """Build the private receipt to commit in the caller's existing atomic meta write."""
    segments = meta.get("segments")
    indices = _segment_indices(meta)
    sources = (
        [(item.get("index"), item.get("prompt")) for item in segments]
        if indices != [0]
        else [(0, meta.get("prompt"))]
    )
    if [item[0] for item in sources] != indices:
        raise ValueError("invalid image optimization segment indices")
    frozen = []
    for index, source in sources:
        text = default_prompt(source, settings.seedream_prompt_template)
        frozen.append({
            "segment_index": index,
            "default": text,
            "current": text,
            "sha256": sha256(text),
        })
    return {"_image_optimization": {
        "version": 1,
        "model": settings.seedream_model,
        "edit_mode": settings.seedream_edit_mode,
        "prompt_template": settings.seedream_prompt_template,
        "segments": frozen,
    }}


def receipt(meta: dict, settings: Settings | None = None) -> dict | None:
    raw = meta.get("_image_optimization")
    if isinstance(raw, dict):
        segments = raw.get("segments")
        if (
            raw.get("version") != 1
            or raw.get("model") not in SEEDREAM_MODELS
            or raw.get("edit_mode") not in SEEDREAM_EDIT_MODES
            or raw.get("prompt_template") not in SEEDREAM_PROMPT_TEMPLATES
            or not isinstance(segments, list) or not segments
        ):
            return None
        seen = set()
        for item in segments:
            if not isinstance(item, dict) or set(item) != {
                "segment_index", "default", "current", "sha256"
            }:
                return None
            index = item.get("segment_index")
            current = item.get("current")
            if (
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                or index in seen
                or not isinstance(item.get("default"), str) or not item["default"].strip()
                or not isinstance(current, str) or not current.strip()
                or item.get("sha256") != sha256(current)
            ):
                return None
            seen.add(index)
        try:
            expected = _segment_indices(meta)
        except ValueError:
            return None
        if seen != set(expected):
            return None
        return deepcopy(raw)
    if meta.get("schema_version") == 2 and meta.get("status") == "done" and settings:
        try:
            return freeze_prompts(settings, meta)["_image_optimization"]
        except ValueError:
            return None
    return None


def public_prompts(meta: dict, settings: Settings) -> dict[int, dict[str, str]]:
    raw = receipt(meta, settings)
    result = {}
    for item in (raw or {}).get("segments", []):
        if not isinstance(item, dict):
            continue
        index, current, default, digest = (
            item.get("segment_index"), item.get("current"),
            item.get("default"), item.get("sha256"),
        )
        if isinstance(index, int) and all(isinstance(x, str) for x in (current, default, digest)):
            result[index] = {"text": current, "default_text": default, "sha256": digest}
    return result


def replace(meta: dict, settings: Settings, segment_index: int,
            expected_sha256: str, prompt: str) -> dict:
    if meta.get("schema_version") != 2:
        raise ImageOptimizationError(409, "read_only")
    if meta.get("status") != "done":
        raise ImageOptimizationError(409, "artifacts not ready")
    if (
        meta.get("_input_owner")
        or isinstance(meta.get("generation"), dict)
        or isinstance(meta.get("postprocess"), dict)
    ):
        raise ImageOptimizationError(409, {
            "code": "image_optimization_prompt_frozen",
            "message": "图片优化提示词已冻结，请刷新页面。",
        })
    replacement = prompt.strip()
    if not replacement or len(replacement.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ImageOptimizationError(422, "invalid_image_optimization_prompt")
    raw = receipt(meta, settings)
    if raw is None:
        raise ImageOptimizationError(409, "image_optimization_prompt_invalid")
    matched = None
    for item in raw.get("segments", []):
        if item.get("segment_index") == segment_index:
            matched = item
            break
    if matched is None:
        raise ImageOptimizationError(422, "invalid_segment_index")
    if matched.get("sha256") != expected_sha256:
        raise ImageOptimizationError(409, {
            "code": "image_optimization_prompt_changed",
            "message": "图片优化提示词已更新，请刷新页面后重试。",
        })
    matched["current"] = replacement
    matched["sha256"] = sha256(replacement)
    return raw
