"""Portable, read-only publication receipts for already validated previews.

The original generation receipts remain authoritative at ``source_locator``.
This module only binds an exact, public copy of their bytes; it never turns the
copy into a generation input or rewrites a paid-provider receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "duet.published-preview"
VERSION = 1
RECEIPT_FILENAME = "published_preview.json"
_RECEIPT_KEYS = {"schema", "version", "source_locator", "binding", "binding_sha256"}
_BINDING_KEYS = {"source", "target"}
_SOURCE_KEYS = {"cid", "meta_sha256", "receipts", "media_timeline_sha256"}
_TARGET_KEYS = {"cid", "title", "note", "read_only", "artifacts"}
_ENTRY_KEYS = {"kind", "path", "sha256", "size"}
_ARTIFACT_KEYS = {"path", "sha256", "size"}


class PublishedPreviewError(ValueError):
    pass


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise PublishedPreviewError("published_preview_invalid") from None
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise PublishedPreviewError("published_preview_file_invalid") from None
    return digest.hexdigest()


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PublishedPreviewError("published_preview_path_invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise PublishedPreviewError("published_preview_path_invalid")
    return value


def _bound_file(root: Path, relative: object) -> Path:
    rel = _relative_path(relative)
    base = root.resolve(strict=True)
    if not base.is_dir():
        raise PublishedPreviewError("published_preview_root_invalid")
    current = base
    try:
        for part in PurePosixPath(rel).parts:
            current = current / part
            if current.is_symlink():
                raise PublishedPreviewError("published_preview_path_invalid")
        resolved = current.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError):
        raise PublishedPreviewError("published_preview_path_invalid") from None
    if not resolved.is_file():
        raise PublishedPreviewError("published_preview_file_invalid")
    return resolved


def _entry(root: Path, kind: str, relative: object) -> dict[str, object]:
    if not isinstance(kind, str) or not kind:
        raise PublishedPreviewError("published_preview_invalid")
    path = _bound_file(root, relative)
    return {
        "kind": kind,
        "path": _relative_path(relative),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _artifact(root: Path, relative: object) -> dict[str, object]:
    path = _bound_file(root, relative)
    return {
        "path": _relative_path(relative),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PublishedPreviewError("published_preview_invalid") from None
    if not isinstance(value, dict):
        raise PublishedPreviewError("published_preview_invalid")
    return value


def _relative_to_root(root: Path, value: object) -> str:
    if not isinstance(value, str):
        raise PublishedPreviewError("published_preview_source_invalid")
    base = root.resolve(strict=True)
    try:
        relative = Path(value).resolve(strict=True).relative_to(base).as_posix()
    except (OSError, ValueError):
        raise PublishedPreviewError("published_preview_source_invalid") from None
    return _relative_path(relative)


def _source_receipts(root: Path, meta: Mapping[str, object]) -> tuple[list[dict[str, object]], str]:
    generation = meta.get("generation")
    prepared = meta.get("prepared_input_receipt")
    if (
        not isinstance(generation, Mapping)
        or generation.get("status") != "succeeded"
        or not isinstance(prepared, str)
        or prepared != Path(prepared).name
    ):
        raise PublishedPreviewError("published_preview_source_invalid")
    context = generation.get("context_ir")
    context_attempt_id = context.get("attempt_id") if isinstance(context, Mapping) else None
    context_receipt = context.get("receipt_path") if isinstance(context, Mapping) else None
    h3_attempt_id = generation.get("h3_attempt_id")
    if (
        not isinstance(context_attempt_id, str)
        or len(context_attempt_id) != 6
        or not context_attempt_id.isdigit()
        or not isinstance(h3_attempt_id, str)
        or len(h3_attempt_id) != 6
        or not h3_attempt_id.isdigit()
    ):
        raise PublishedPreviewError("published_preview_source_invalid")
    context_receipt_rel = _relative_to_root(root, context_receipt)
    expected_context_suffix = (
        f"work/h3-native/.context-ir/attempts/{context_attempt_id}/receipt.json"
    )
    if context_receipt_rel != expected_context_suffix:
        raise PublishedPreviewError("published_preview_source_invalid")
    h3_attempt_rel = f"work/h3-native/.h3/attempts/{h3_attempt_id}/attempt.json"
    fixed = (
        ("prepared_input", prepared),
        ("multimodal_input", "work/multimodal_input.json"),
        ("multimodal_source", "work/h3_multimodal_source.json"),
        ("h3_prompt_plan", "work/h3_prompt_plan.json"),
        (
            "context_ir_attempt",
            f"work/h3-native/.context-ir/attempts/{context_attempt_id}/attempt.json",
        ),
        ("context_ir_receipt", context_receipt_rel),
        ("h3_attempt", h3_attempt_rel),
        ("stitch_receipt", "stitch-receipt.json"),
    )
    entries = sorted((_entry(root, kind, path) for kind, path in fixed), key=lambda item: item["path"])
    if len({item["path"] for item in entries}) != len(entries):
        raise PublishedPreviewError("published_preview_source_invalid")
    h3_attempt = _read_json(_bound_file(root, h3_attempt_rel))
    timeline = (
        h3_attempt.get("h3", {}).get("output", {}).get("media_timeline")
        if isinstance(h3_attempt.get("h3"), Mapping)
        else None
    )
    if (
        not isinstance(timeline, Mapping)
        or timeline.get("schema") != "duet.h3.media_timeline"
        or timeline.get("version") != 1
        or timeline.get("decode_complete") is not True
    ):
        raise PublishedPreviewError("published_preview_source_invalid")
    return entries, _canonical_sha256(timeline)


def _public_artifact_paths(root: Path, meta: Mapping[str, object]) -> list[str]:
    try:
        sources = sorted(
            path.relative_to(root.resolve(strict=True)).as_posix()
            for path in root.resolve(strict=True).glob("source.*")
            if path.is_file() and not path.is_symlink()
        )
    except OSError:
        raise PublishedPreviewError("published_preview_source_invalid") from None
    if len(sources) != 1:
        raise PublishedPreviewError("published_preview_source_invalid")
    paths = {sources[0], "generated.mp4"}
    keyframes = meta.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        raise PublishedPreviewError("published_preview_source_invalid")
    for name in keyframes:
        if not isinstance(name, str) or name != Path(name).name:
            raise PublishedPreviewError("published_preview_source_invalid")
        paths.add(f"work/keyframes/{name}")
    postprocess = meta.get("postprocess")
    frames = postprocess.get("frames") if isinstance(postprocess, Mapping) else None
    if not isinstance(frames, list) or not frames:
        raise PublishedPreviewError("published_preview_source_invalid")
    for name in frames:
        if not isinstance(name, str) or name != Path(name).name:
            raise PublishedPreviewError("published_preview_source_invalid")
        paths.add(f"work/postprocessed/{name}")
    return sorted(_relative_path(path) for path in paths)


def _build(
    *, source_root: Path, target_root: Path, target_meta: Mapping[str, object]
) -> dict[str, object]:
    """Deterministically bind the complete public copy to its source chain."""
    source = Path(source_root).resolve(strict=True)
    target = Path(target_root).resolve(strict=True)
    source_meta_path = _bound_file(source, "meta.json")
    source_meta = _read_json(source_meta_path)
    cid = source_meta.get("id")
    target_generation = target_meta.get("generation")
    if (
        source == target
        or not isinstance(cid, str)
        or target_meta.get("id") != cid
        or source_meta.get("schema_version") != 2
        or target_meta.get("schema_version") != 2
        or not isinstance(target_meta.get("title"), str)
        or not isinstance(target_meta.get("note"), str)
        or not isinstance(target_generation, Mapping)
        or target_generation.get("status") != "succeeded"
        or "published_preview_receipt" in source_meta
    ):
        raise PublishedPreviewError("published_preview_invalid")
    receipts, timeline_sha256 = _source_receipts(source, source_meta)
    artifact_paths = _public_artifact_paths(source, source_meta)
    source_artifacts = [_artifact(source, path) for path in artifact_paths]
    target_artifacts = [_artifact(target, path) for path in artifact_paths]
    if target_artifacts != source_artifacts:
        raise PublishedPreviewError("published_preview_artifact_mismatch")
    binding = {
        "source": {
            "cid": cid,
            "meta_sha256": _sha256(source_meta_path),
            "receipts": receipts,
            "media_timeline_sha256": timeline_sha256,
        },
        "target": {
            "cid": cid,
            "title": target_meta["title"],
            "note": target_meta["note"],
            "read_only": True,
            "artifacts": target_artifacts,
        },
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "source_locator": {"root": str(source)},
        "binding": binding,
        "binding_sha256": _canonical_sha256(binding),
    }


def write(root: Path, receipt: Mapping[str, object]) -> Path:
    destination = Path(root).resolve(strict=True) / RECEIPT_FILENAME
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".published-preview-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def load(root: Path, meta: Mapping[str, object]) -> tuple[Path, dict[str, Any]]:
    if meta.get("published_preview_receipt") != RECEIPT_FILENAME:
        raise PublishedPreviewError("published_preview_invalid")
    receipt = _read_json(_bound_file(root, RECEIPT_FILENAME))
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema") != SCHEMA
        or receipt.get("version") != VERSION
        or not isinstance(receipt.get("source_locator"), Mapping)
        or set(receipt["source_locator"]) != {"root"}
        or not isinstance(receipt.get("binding"), Mapping)
        or set(receipt["binding"]) != _BINDING_KEYS
        or receipt.get("binding_sha256") != _canonical_sha256(receipt["binding"])
    ):
        raise PublishedPreviewError("published_preview_invalid")
    source = receipt["binding"].get("source")
    target = receipt["binding"].get("target")
    if (
        not isinstance(source, Mapping)
        or set(source) != _SOURCE_KEYS
        or not isinstance(target, Mapping)
        or set(target) != _TARGET_KEYS
        or target.get("read_only") is not True
        or target.get("cid") != meta.get("id")
        or target.get("title") != meta.get("title")
        or target.get("note") != meta.get("note")
        or not isinstance(source.get("receipts"), list)
        or not isinstance(target.get("artifacts"), list)
    ):
        raise PublishedPreviewError("published_preview_invalid")
    for entries, keys in (
        (source["receipts"], _ENTRY_KEYS),
        (target["artifacts"], _ARTIFACT_KEYS),
    ):
        if any(not isinstance(item, Mapping) or set(item) != keys for item in entries):
            raise PublishedPreviewError("published_preview_invalid")
        paths = [item.get("path") for item in entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise PublishedPreviewError("published_preview_invalid")
        for path in paths:
            _relative_path(path)
    locator = receipt["source_locator"].get("root")
    if not isinstance(locator, str) or not Path(locator).is_absolute():
        raise PublishedPreviewError("published_preview_invalid")
    try:
        source_root = Path(locator).resolve(strict=True)
    except OSError:
        raise PublishedPreviewError("published_preview_source_invalid") from None
    expected = _build(source_root=source_root, target_root=root, target_meta=meta)
    if expected["binding"] != receipt["binding"]:
        raise PublishedPreviewError("published_preview_mismatch")
    return source_root, receipt
