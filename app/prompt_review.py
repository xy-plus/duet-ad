"""提示词审核/编辑：四段结构的落盘、读取与历史项目主体剥离。

H3 最终提示词 = prefix（后端机械注入：编辑框架 editing_block + 遮盖条款
mask_block + 人物替换块 face_block）+ editable（DeepSeek fusion 主体中可审
核/编辑的前半）+ hidden（主体末尾的权威替换计划块：从整行
"BEGIN AUTHORITATIVE OBJECT REPLACEMENT AND VOICE PLAN" 起（含该行）到主体
末尾，含 END 标记整段）+ suffix（后端机械注入：Overlay removal +
Dialogue requirements 尾部）。prefix/suffix/hidden 一律不展示、不可编辑；
confirm 时由后端用冻结原文重新机械拼接，用户输入无法影响它们。

无论是否开启审核开关，_run_h3_segments 在冻结 H3 产物后都把四段结构落盘
work/prompt-review.json：审核暂停的项目靠它在 confirm 时重建最终提示词并
重算 h3_prepared_artifacts 回执；直通/已完成项目靠它在前端只读展示冻结的
可编辑主体。历史项目没有该文件，用 split_frozen_prompt_body 从冻结
h3_prompt.txt 做 best-effort 剥离，再按 BEGIN 行截断；剥不出（注入边界无
法确定）就返回空 segments，绝不展示注入内容。body 里没有 BEGIN 行的段
hidden 为空、整段可编辑，重建结果与拆分前逐字节一致。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from app.fusion_schema import AUTHORITY_BEGIN

PROMPT_REVIEW_VERSION = 2
PROMPT_REVIEW_RELATIVE_PATH = "work/prompt-review.json"

_SEGMENT_KEYS = frozenset({
    "segment_index", "prefix", "prefix_sha256",
    "editable", "hidden", "suffix", "suffix_sha256",
    "start_s", "end_s",
})

# 权威替换计划块的开始标记（整行；fusion_schema.AUTHORITY_BEGIN，由
# normalize_fusion_prompt 拼接到主体末尾）。该行起（含该行）到 body 末尾
# 为 hidden。
HIDDEN_BLOCK_START = AUTHORITY_BEGIN

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


def _valid_time_range(
    start: Any, end: Any,
) -> tuple[float | None, float | None]:
    """段的 start_s/end_s：非负有限数且 start < end 才有效，否则视为 null。"""
    for value in (start, end):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return None, None
    if not start < end:
        return None, None
    return float(start), float(end)


def split_body(body: str) -> tuple[str, str]:
    """把 DS 主体在首个 BEGIN 整行处拆成 (editable, hidden)。

    hidden 从 BEGIN 行开头（含该行）到 body 末尾；没有 BEGIN 行时 hidden 为
    空、editable 为整个 body（与拆分前逐字节一致）。
    """
    position = 0
    for line in body.splitlines(keepends=True):
        if line.strip() == HIDDEN_BLOCK_START:
            return body[:position], body[position:]
        position += len(line)
    return body, ""


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
    """把每段 {segment_index, prefix, editable, hidden, suffix, 时间} 落盘。"""
    records = []
    for segment in segments:
        start_s, end_s = _valid_time_range(
            segment.get("start_s"), segment.get("end_s"),
        )
        records.append({
            "segment_index": segment["segment_index"],
            "prefix": segment["prefix"],
            "prefix_sha256": _sha256_text(segment["prefix"]),
            "editable": segment["editable"],
            "hidden": segment["hidden"],
            "suffix": segment["suffix"],
            "suffix_sha256": _sha256_text(segment["suffix"]),
            "start_s": start_s,
            "end_s": end_s,
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
    """读取并校验冻结的审核记录；缺失、旧版或损坏返回 None。"""
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
        for field in ("prefix", "editable", "hidden", "suffix"):
            if not isinstance(segment[field], str):
                return None
        if (
            segment["prefix_sha256"] != _sha256_text(segment["prefix"])
            or segment["suffix_sha256"] != _sha256_text(segment["suffix"])
        ):
            return None
        # 脏时间数据不进响应：就地归一化为 null，不整文件判废。
        segment["start_s"], segment["end_s"] = _valid_time_range(
            segment.get("start_s"), segment.get("end_s"),
        )
    return raw


def display_segments(
    project: Path, state: dict[str, Any],
) -> list[dict[str, Any]]:
    """只读展示视图：[{"segment_index": N, "prompt": 可编辑主体前半}]。

    优先读 prompt-review.json；没有则从冻结清单逐段剥离 h3_prompt.txt
    （剥离前先校验文件与清单回执一致），再按 BEGIN 行截断。任何一段剥不
    出来整个项目返回空，避免部分段展示注入内容、部分段缺失的混合视图。
    """
    review = read_review(project)
    if review is not None:
        return [
            {
                "segment_index": segment["segment_index"],
                "prompt": segment["editable"],
                "start_s": segment["start_s"],
                "end_s": segment["end_s"],
            }
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
        editable, _hidden = split_body(body)
        # 剥离路径拿不到段时间：字段为 null，前端回退显示"分段 N"。
        result.append({
            "segment_index": int(key),
            "prompt": editable,
            "start_s": None,
            "end_s": None,
        })
    return result


def generation_history(
    project: Path, state: dict[str, Any],
) -> list[dict[str, Any]]:
    """视频版本历史：归档版本 + 当前版本（current=True 居末）。

    每项 {version, completed_at, current, video_file, segments}：
    video_file 是相对 /api/conversations/{id}/files/ 的键；segments 是各段
    冻结提示词的可编辑部分（归档版本从归档的 final prompt 全文用现有剥离
    器剥出，剥不出给空列表但保留版本；时间取归档时记录的值）。
    generation_history 字段损坏时 fail closed 返回空历史（只藏不炸）。
    """
    raw_history = state.get("generation_history")
    versions: list[dict[str, Any]] = []
    if raw_history is not None:
        if not isinstance(raw_history, list):
            return []
        for entry in raw_history:
            if (
                not isinstance(entry, dict)
                or isinstance(entry.get("version"), bool)
                or not isinstance(entry.get("version"), int)
                or entry["version"] < 1
                or not isinstance(entry.get("segments"), list)
            ):
                return []
            video = entry.get("video")
            if (
                not isinstance(video, dict)
                or video.get("path")
                != f"output/history/v{entry['version']}/generated.mp4"
            ):
                return []
            segments: list[dict[str, Any]] = []
            for segment in entry["segments"]:
                if (
                    not isinstance(segment, dict)
                    or not isinstance(segment.get("segment_index"), int)
                    or not isinstance(segment.get("prompt_path"), str)
                    or not isinstance(segment.get("prompt_sha256"), str)
                ):
                    return []
                prompt_path = project / segment["prompt_path"]
                try:
                    if (
                        prompt_path.is_symlink()
                        or not prompt_path.is_file()
                        or project.resolve() not in prompt_path.resolve().parents
                    ):
                        continue
                    text = prompt_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if (
                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                    != segment["prompt_sha256"]
                ):
                    continue
                body = split_frozen_prompt_body(text)
                if body is None:
                    continue
                editable, _hidden = split_body(body)
                start_s, end_s = _valid_time_range(
                    segment.get("start_s"), segment.get("end_s"),
                )
                segments.append({
                    "segment_index": segment["segment_index"],
                    "prompt": editable,
                    "start_s": start_s,
                    "end_s": end_s,
                })
            segments.sort(key=lambda item: item["segment_index"])
            versions.append({
                "version": entry["version"],
                "completed_at": entry.get("completed_at"),
                "current": False,
                "video_file": f"history/v{entry['version']}/generated.mp4",
                "segments": segments,
            })
    if state.get("status") == "succeeded" and state.get("has_video") is True:
        versions.append({
            "version": len(versions) + 1,
            "completed_at": state.get("updated_at"),
            "current": True,
            "video_file": "generated.mp4",
            "segments": display_segments(project, state),
        })
    return versions
