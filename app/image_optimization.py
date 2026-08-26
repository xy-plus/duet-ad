"""Frozen per-segment image-optimization prompts and strict CAS editing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from copy import deepcopy
from pathlib import Path

from app.codex_runner import CodexError
from app.config import (
    SEEDREAM_EDIT_MODES,
    SEEDREAM_MODELS,
    Settings,
)

MAX_PROMPT_BYTES = 32 * 1024
MAX_CONTINUITY_BYTES = 32 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _ROOT / "skills" / "image-postprocess" / "SKILL.md"
_CONTINUITY_SKILL = _ROOT / "skills" / "image-continuity" / "SKILL.md"
_ELEMENT_KINDS = {
    "PERSON": "person",
    "SUBJECT": "subject",
    "OUTFIT": "outfit",
    "SCENE": "scene",
    "PROP": "prop",
    "PRODUCT": "product",
}
_ELEMENT_ID_RE = re.compile(
    r"^(PERSON|SUBJECT|OUTFIT|SCENE|PROP|PRODUCT)_([0-9]{2})$"
)


class ImageOptimizationError(ValueError):
    def __init__(self, status: int, detail: str | dict[str, str]):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


class ImageOptimizationOutputError(ValueError):
    pass


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _copy_regular(source: Path, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError:
        raise ValueError("invalid image optimization keyframe") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("invalid image optimization keyframe")
        with os.fdopen(fd, "rb", closefd=False) as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        os.close(fd)


def _read_output(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise ImageOptimizationOutputError("image optimization output is missing or invalid") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROMPT_BYTES:
            raise ImageOptimizationOutputError("image optimization output is missing or invalid")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_PROMPT_BYTES + 1)
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ImageOptimizationOutputError("image optimization output is missing or invalid") from None
    if not text or len(text.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ImageOptimizationOutputError("image optimization output is missing or invalid")
    return text


def _read_json_output(path: Path) -> object:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise ImageOptimizationOutputError(
            "image continuity output is missing or invalid"
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CONTINUITY_BYTES:
            raise ImageOptimizationOutputError(
                "image continuity output is missing or invalid"
            )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_CONTINUITY_BYTES + 1)
    finally:
        os.close(fd)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ImageOptimizationOutputError(
            "image continuity output is missing or invalid"
        ) from None


def _canonical_continuity(value: object, expected_indices: list[int] | None = None) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "version", "segment_indices", "elements"
    } or value.get("version") != 1:
        raise ImageOptimizationOutputError("image continuity output is missing or invalid")
    indices = value.get("segment_indices")
    if (
        not isinstance(indices, list) or not indices
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 1 for index in indices)
        or indices != sorted(set(indices))
        or (expected_indices is not None and indices != expected_indices)
    ):
        raise ImageOptimizationOutputError("image continuity output is missing or invalid")
    elements = value.get("elements")
    if not isinstance(elements, list) or len(elements) > 100:
        raise ImageOptimizationOutputError("image continuity output is missing or invalid")
    seen: set[str] = set()
    counters: dict[str, int] = {prefix: 0 for prefix in _ELEMENT_KINDS}
    canonical = []
    for element in elements:
        if not isinstance(element, dict) or set(element) != {
            "id", "kind", "source", "replacement", "segments"
        }:
            raise ImageOptimizationOutputError("image continuity output is missing or invalid")
        identifier = element.get("id")
        matched = _ELEMENT_ID_RE.fullmatch(identifier) if isinstance(identifier, str) else None
        source = element.get("source")
        replacement = element.get("replacement")
        segments = element.get("segments")
        if (
            matched is None or identifier in seen
            or element.get("kind") != _ELEMENT_KINDS[matched.group(1)]
            or int(matched.group(2)) != counters[matched.group(1)] + 1
            or not isinstance(source, str) or source != source.strip() or not source
            or len(source.encode("utf-8")) > 2048
            or not isinstance(replacement, str) or replacement != replacement.strip() or not replacement
            or len(replacement.encode("utf-8")) > 2048
            or not isinstance(segments, list) or len(segments) < 2
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index not in indices
                for index in segments
            )
            or segments != sorted(set(segments))
        ):
            raise ImageOptimizationOutputError("image continuity output is missing or invalid")
        counters[matched.group(1)] += 1
        seen.add(identifier)
        canonical.append({
            "id": identifier,
            "kind": element["kind"],
            "source": source,
            "replacement": replacement,
            "segments": list(segments),
        })
    if [element["id"] for element in canonical] != sorted(seen):
        raise ImageOptimizationOutputError("image continuity output is missing or invalid")
    return {"version": 1, "segment_indices": list(indices), "elements": canonical}


def _validated_frames(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError("invalid image optimization keyframes directory")
    frames = sorted(source.glob("[0-9][0-9].png"))
    expected = [f"{index:02d}.png" for index in range(1, len(frames) + 1)]
    if not frames or len(frames) > 9 or [frame.name for frame in frames] != expected:
        raise ValueError("invalid image optimization keyframes")
    return frames


def generate_continuity_plan(runner, segments: list[dict], *, session_dir: Path) -> dict:
    """Generate one frozen cross-segment element map before per-segment prompts."""
    try:
        session = Path(session_dir).resolve(strict=True)
        skill = _CONTINUITY_SKILL.resolve(strict=True)
    except OSError:
        raise ValueError("invalid image continuity input") from None
    if not isinstance(segments, list) or len(segments) < 2 or not skill.is_file():
        raise ValueError("invalid image continuity segments")
    indices = [segment.get("index") for segment in segments if isinstance(segment, dict)]
    if len(indices) != len(segments) or indices != list(range(1, len(segments) + 1)):
        raise ValueError("invalid image continuity segments")
    prepared = []
    for segment in segments:
        if set(segment) != {"index", "chain_id", "join_mode", "keyframes_dir"}:
            raise ValueError("invalid image continuity segments")
        chain_id, join_mode = segment["chain_id"], segment["join_mode"]
        if (
            not isinstance(chain_id, str) or not chain_id or len(chain_id) > 128
            or join_mode not in {"hard_cut", "continue"}
        ):
            raise ValueError("invalid image continuity segments")
        try:
            source = Path(segment["keyframes_dir"]).resolve(strict=True)
            source.relative_to(session)
        except (OSError, TypeError, ValueError):
            raise ValueError("invalid image continuity segments") from None
        prepared.append((segment, _validated_frames(source)))

    with tempfile.TemporaryDirectory(prefix="duet-image-continuity-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        work = stage / "work"
        _copy_regular(skill, stage / "SKILL.md")
        request_segments = []
        for segment, frames in prepared:
            destination = work / "segments" / str(segment["index"]) / "keyframes"
            destination.mkdir(parents=True, mode=0o700)
            for frame in frames:
                _copy_regular(frame, destination / frame.name)
            request_segments.append({
                "index": segment["index"],
                "chain_id": segment["chain_id"],
                "join_mode": segment["join_mode"],
            })
        (work / "request.json").write_text(
            json.dumps({"segments": request_segments}, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        run_error: CodexError | None = None
        try:
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md；只读取允许的输入，并写入规定的唯一输出文件。",
                session_dir=session,
            )
        except CodexError as error:
            run_error = error
        try:
            return _canonical_continuity(
                _read_json_output(work / "continuity.json"), indices
            )
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


def generate_prompt(
    runner,
    keyframes_dir: Path,
    edit_mode: str,
    *,
    session_dir: Path,
    segment_index: int | None = None,
    continuity: dict | None = None,
) -> str:
    """Run the independent Skill with only this segment's frames and edit mode."""
    if edit_mode not in SEEDREAM_EDIT_MODES:
        raise ValueError("unsupported image optimization edit mode")
    try:
        source = Path(keyframes_dir).resolve(strict=True)
        session = Path(session_dir).resolve(strict=True)
        skill = _SKILL.resolve(strict=True)
        source.relative_to(session)
    except (OSError, ValueError):
        raise ValueError("invalid image optimization keyframes directory") from None
    if not skill.is_file():
        raise ValueError("invalid image optimization keyframes directory")
    frames = _validated_frames(source)
    if (continuity is None) != (segment_index is None):
        raise ValueError("invalid image continuity binding")
    segment_continuity = None
    if continuity is not None:
        canonical = _canonical_continuity(continuity)
        if (
            isinstance(segment_index, bool) or not isinstance(segment_index, int)
            or segment_index not in canonical["segment_indices"]
        ):
            raise ValueError("invalid image continuity binding")
        segment_continuity = {
            "version": 1,
            "segment_index": segment_index,
            "elements": [
                element for element in canonical["elements"]
                if segment_index in element["segments"]
            ],
        }

    with tempfile.TemporaryDirectory(prefix="duet-image-postprocess-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        work = stage / "work"
        staged_frames = work / "keyframes"
        staged_frames.mkdir(parents=True, mode=0o700)
        _copy_regular(skill, stage / "SKILL.md")
        for frame in frames:
            _copy_regular(frame, staged_frames / frame.name)
        request = {"edit_mode": edit_mode}
        if segment_continuity is not None:
            request["segment_index"] = segment_index
            (work / "continuity.json").write_text(
                json.dumps(segment_continuity, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        (work / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        run_error: CodexError | None = None
        try:
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md；只读取其中允许的输入，并写入规定的唯一输出文件。",
                session_dir=session,
            )
        except CodexError as error:
            run_error = error
        try:
            return _read_output(work / "image_optimization_prompt.txt")
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


def freeze_continuity(plan: dict) -> dict:
    canonical = _canonical_continuity(plan)
    raw = json.dumps(
        {
            "segment_indices": canonical["segment_indices"],
            "elements": canonical["elements"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"_image_continuity": {
        **canonical,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }}


def continuity_receipt(meta: dict) -> dict | None:
    raw = meta.get("_image_continuity")
    if not isinstance(raw, dict) or set(raw) != {
        "version", "segment_indices", "elements", "sha256"
    }:
        return None
    try:
        canonical = _canonical_continuity({
            "version": raw.get("version"),
            "segment_indices": raw.get("segment_indices"),
            "elements": raw.get("elements"),
        })
    except ImageOptimizationOutputError:
        return None
    expected = freeze_continuity(canonical)["_image_continuity"]
    return deepcopy(raw) if raw == expected else None


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


def freeze_prompts(settings: Settings, meta: dict, prompts: dict[int, str]) -> dict:
    """Build the private receipt to commit in the caller's existing atomic meta write."""
    indices = _segment_indices(meta)
    if (
        not isinstance(prompts, dict)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in prompts)
        or set(prompts) != set(indices)
    ):
        raise ValueError("invalid image optimization prompt segments")
    frozen = []
    for index in indices:
        source = prompts[index]
        text = source.strip() if isinstance(source, str) else ""
        if not text or len(text.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("invalid image optimization prompt output")
        frozen.append({
            "segment_index": index,
            "default": text,
            "current": text,
            "sha256": sha256(text),
        })
    return {"_image_optimization": {
        "version": 2,
        "model": settings.seedream_model,
        "edit_mode": settings.seedream_edit_mode,
        "segments": frozen,
    }}


def receipt(meta: dict, settings: Settings | None = None) -> dict | None:
    raw = meta.get("_image_optimization")
    if isinstance(raw, dict):
        segments = raw.get("segments")
        if (
            set(raw) != {"version", "model", "edit_mode", "segments"}
            or raw.get("version") != 2
            or raw.get("model") not in SEEDREAM_MODELS
            or raw.get("edit_mode") not in SEEDREAM_EDIT_MODES
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
                or len(item["default"].encode("utf-8")) > MAX_PROMPT_BYTES
                or len(current.encode("utf-8")) > MAX_PROMPT_BYTES
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
