"""提示词审核/编辑：三段结构的落盘、读取与历史项目主体剥离。

H3 最终提示词 = prefix（后端机械注入：编辑框架 editing_block + 遮盖条款
mask_block + 人物替换块 face_block）+ body（DeepSeek fusion 生成的主体，
唯一可被用户审核/编辑的部分）+ suffix（后端机械注入：Overlay removal +
Dialogue requirements 尾部）。注入部分一律不展示、不可编辑；confirm 时由
后端用冻结的 prefix/suffix 重新机械拼接。

无论是否开启审核开关，_run_h3_segments 在冻结 H3 产物后都把三段结构落盘
work/prompt-review.json：审核暂停的项目靠它在 confirm 时重建最终提示词并
重算 h3_prepared_artifacts 回执；直通/已完成项目靠它在前端只读展示冻结的
DS 主体。历史项目没有该文件，用 split_frozen_prompt_body 从冻结
h3_prompt.txt 做 best-effort 剥离；剥不出（注入边界无法确定）就返回空
segments，绝不展示注入内容。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROMPT_REVIEW_VERSION = 1
PROMPT_REVIEW_RELATIVE_PATH = "work/prompt-review.json"

_SEGMENT_KEYS = frozenset(
    {"segment_index", "prefix", "prefix_sha256", "body", "suffix", "suffix_sha256"}
)

# 已知注入块的开头文案（新旧两版）。冻结提示词以这些块开头时逐块剥掉；
# 块都以 "\n\n" 收尾（见 pipeline._prepare_h3_segment 的组装）。
_PREFIX_BLOCK_STARTS = (
    "This is an editing task on Video 1",
    "Dark gray (64,64,64)",
    "Solid black (0,0,0)",
    "Semi-transparent white (255,255,255)",
    "FACE REPLACEMENT IS MANDATORY",
    "PERSON REPLACEMENT IS MANDATORY",
)

# 注入尾部的锚点：Overlay removal 段与 Dialogue requirements/无台词尾。
_SUFFIX_ANCHORS = (
    "\n\nOverlay removal:",
    "\n\nDialogue requirements:",
    "\n\nDo not add captions",
)


class PromptReviewError(ValueError):
    """提示词审核的确定性拒绝；code 直接映射到内外 API 的错误码。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_frozen_prompt_body(text: str) -> str | None:
    """从冻结的完整 H3 提示词剥出 DS 主体；无法确定注入边界时返回 None。"""
    if not isinstance(text, str):
        return None
    rest = text
    while True:
        matched = next(
            (start for start in _PREFIX_BLOCK_STARTS if rest.startswith(start)),
            None,
        )
        if matched is None:
            break
        boundary = rest.find("\n\n")
        if boundary < 0:
            return None
        rest = rest[boundary + 2:]
    cut = len(rest)
    for anchor in _SUFFIX_ANCHORS:
        position = rest.find(anchor)
        if position >= 0:
            cut = min(cut, position)
    body = rest[:cut]
    if not body.strip():
        return None
    # 主体里不应再出现注入块开头文案；出现说明块顺序未识别，宁可不展示。
    if any(start in body for start in _PREFIX_BLOCK_STARTS):
        return None
    return body


def write_review(
    project: Path,
    segments: list[dict[str, Any]],
) -> Path:
    """把每段 {segment_index, prefix, body, suffix} 落盘 work/prompt-review.json。"""
    records = []
    for segment in segments:
        records.append({
            "segment_index": segment["segment_index"],
            "prefix": segment["prefix"],
            "prefix_sha256": _sha256_text(segment["prefix"]),
            "body": segment["body"],
            "suffix": segment["suffix"],
            "suffix_sha256": _sha256_text(segment["suffix"]),
        })
    payload = (
        json.dumps(
            {"version": PROMPT_REVIEW_VERSION, "segments": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    target = project / PROMPT_REVIEW_RELATIVE_PATH
    temporary = target.with_name(".prompt-review.json.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def read_review(project: Path) -> dict[str, Any] | None:
    """读取并校验冻结的审核记录；缺失或损坏返回 None（confirm 侧再拒绝）。"""
    target = project / PROMPT_REVIEW_RELATIVE_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("version") != PROMPT_REVIEW_VERSION
        or not isinstance(raw.get("segments"), list)
        or not raw["segments"]
    ):
        return None
    for segment in raw["segments"]:
        if not isinstance(segment, dict) or set(segment) != _SEGMENT_KEYS:
            return None
        index = segment["segment_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            return None
        for field in ("prefix", "body", "suffix"):
            if not isinstance(segment[field], str):
                return None
        if (
            segment["prefix_sha256"] != _sha256_text(segment["prefix"])
            or segment["suffix_sha256"] != _sha256_text(segment["suffix"])
        ):
            return None
    return raw


def display_segments(
    project: Path, state: dict[str, Any],
) -> list[dict[str, Any]]:
    """只读展示视图：[{"segment_index": N, "prompt": DS 主体}]。

    优先读 prompt-review.json；没有则从冻结清单逐段剥离 h3_prompt.txt
    （剥离前先校验文件与清单回执一致）。任何一段剥不出来整个项目返回空，
    避免部分段展示注入内容、部分段缺失的混合视图。
    """
    review = read_review(project)
    if review is not None:
        return [
            {"segment_index": segment["segment_index"], "prompt": segment["body"]}
            for segment in sorted(
                review["segments"], key=lambda item: item["segment_index"]
            )
        ]
    manifest = state.get("h3_prepared_artifacts")
    segments = manifest.get("segments") if isinstance(manifest, dict) else None
    if not isinstance(segments, dict) or not segments:
        return []
    result: list[dict[str, Any]] = []
    try:
        ordered = sorted(segments, key=int)
    except (TypeError, ValueError):
        return []
    for key in ordered:
        receipt = segments[key].get("prompt") if isinstance(segments[key], dict) else None
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("path"), str)
            or not isinstance(receipt.get("sha256"), str)
        ):
            return []
        path = project / receipt["path"]
        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or project.resolve() not in path.resolve().parents
            ):
                return []
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != receipt["sha256"]:
            return []
        body = split_frozen_prompt_body(text)
        if body is None:
            return []
        result.append({"segment_index": int(key), "prompt": body})
    return result
