#!/usr/bin/env python3
"""Freeze the minimal image-postprocess/Fusion evaluation corpus.

This command is deliberately offline.  It copies an immutable, SHA-addressed
input closure from completed projects and publishes the whole corpus with one
same-parent rename.  Original project paths are never persisted in the
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn


CORPUS_SCHEMA = "duet.skill-eval-corpus"
CORPUS_VERSION = 1
ALLOWED_SPLITS = frozenset({"train", "regression", "holdout"})
_PROJECT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})


@dataclass(frozen=True)
class CaseSource:
    split: str
    project_root: Path


@dataclass(frozen=True)
class FreezeReport:
    output_root: Path
    manifest_sha256: str
    case_count: int
    unique_blob_count: int
    unique_blob_bytes: int


@dataclass(frozen=True)
class MaterializeReport:
    output_root: Path
    manifest_sha256: str
    source_project_id: str
    file_count: int
    copied_bytes: int


@dataclass(frozen=True)
class _ReadResult:
    data: bytes
    sha256: str
    size: int
    device: int
    inode: int


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_value(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_absolute_source_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("project root must be absolute")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise ValueError("project root is missing or invalid") from exc
    if resolved != path or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("project root must be a real canonical directory")
    return path


def _prepare_output_parent(output_root: Path) -> Path:
    if not output_root.is_absolute():
        raise ValueError("output root must be absolute")
    if output_root == Path("/") or output_root.name in {"", ".", ".."}:
        raise ValueError("output root is invalid")
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = output_root.parent.resolve(strict=True)
        parent_info = output_root.parent.lstat()
    except OSError as exc:
        raise ValueError("output parent is missing or invalid") from exc
    if (
        parent != output_root.parent
        or not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
    ):
        raise ValueError("output parent must be a real canonical directory")
    return parent


def _safe_relative_path(raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label} is invalid")
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} must be a safe relative path")
    if relative.as_posix() != raw:
        raise ValueError(f"{label} is not canonical")
    return relative


def _file_under(root: Path, relative: Path, label: str) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ValueError(f"{label} is missing") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{label} must not contain symlinks")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return current


def _stable_read(path: Path, label: str) -> _ReadResult:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{label} changed before reading")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed while reading") from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after_read, field)
        or getattr(before, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise ValueError(f"{label} changed while reading")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise ValueError(f"{label} byte count changed while reading")
    return _ReadResult(
        data=data,
        sha256=_sha256(data),
        size=len(data),
        device=before.st_dev,
        inode=before.st_ino,
    )


def _write_new_file(path: Path, data: bytes, mode: int = 0o444) -> None:
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short corpus write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _png_metadata(data: bytes) -> tuple[str, int, int] | None:
    if not data.startswith(_PNG_SIGNATURE):
        return None
    if len(data) < 33 or data[12:16] != b"IHDR" or struct.unpack(">I", data[8:12])[0] != 13:
        raise ValueError("PNG image has an invalid IHDR")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise ValueError("PNG image dimensions are invalid")
    return "image/png", width, height


def _jpeg_metadata(data: bytes) -> tuple[str, int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    cursor = 2
    while cursor < len(data):
        if data[cursor] != 0xFF:
            raise ValueError("JPEG marker stream is invalid")
        while cursor < len(data) and data[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(data):
            break
        marker = data[cursor]
        cursor += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if cursor + 2 > len(data):
            break
        length = struct.unpack(">H", data[cursor:cursor + 2])[0]
        if length < 2 or cursor + length > len(data):
            raise ValueError("JPEG segment length is invalid")
        if marker in _JPEG_SOF_MARKERS:
            if length < 8:
                raise ValueError("JPEG SOF segment is invalid")
            height, width = struct.unpack(">HH", data[cursor + 3:cursor + 7])
            if width < 1 or height < 1:
                raise ValueError("JPEG image dimensions are invalid")
            return "image/jpeg", width, height
        if marker == 0xDA:
            break
        cursor += length
    raise ValueError("JPEG image has no supported SOF marker")


def _image_metadata(data: bytes) -> dict[str, object]:
    detected = _png_metadata(data) or _jpeg_metadata(data)
    if detected is None:
        raise ValueError("unsupported or invalid image bytes")
    media_type, width, height = detected
    return {"media_type": media_type, "width": width, "height": height}


class _BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, mode=0o700)
        self.records: dict[str, dict[str, object]] = {}

    def add(
        self, source: Path, label: str, *, media_kind: str,
    ) -> tuple[dict[str, str], bytes]:
        before = _stable_read(source, label)
        destination = self._root / before.sha256
        if destination.exists():
            existing = _stable_read(destination, f"existing blob {before.sha256}")
            if existing.sha256 != before.sha256 or existing.size != before.size:
                raise ValueError("content-addressed blob collision")
        else:
            _write_new_file(destination, before.data)

        source_after = _stable_read(source, label)
        frozen = _stable_read(destination, f"frozen blob {before.sha256}")
        if (
            (source_after.sha256, source_after.size) != (before.sha256, before.size)
            or (frozen.sha256, frozen.size) != (before.sha256, before.size)
            or (frozen.device, frozen.inode) == (before.device, before.inode)
        ):
            raise ValueError(f"{label} did not freeze as an independent byte copy")

        metadata: dict[str, object] = {"bytes": before.size}
        if media_kind == "image":
            metadata.update(_image_metadata(frozen.data))
        elif media_kind == "json":
            _json_value(frozen.data, label)
            metadata["media_type"] = "application/json"
        else:
            raise ValueError("unsupported corpus media kind")
        prior = self.records.setdefault(before.sha256, metadata)
        if prior != metadata:
            raise ValueError("one blob has conflicting metadata")
        return {"blob_sha256": before.sha256}, frozen.data


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _json_blob(
    store: _BlobStore, root: Path, relative: str, label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = _file_under(root, _safe_relative_path(relative, label), label)
    reference, data = store.add(path, label, media_kind="json")
    return reference, _object(_json_value(data, label), label)


def _image_blob(
    store: _BlobStore, root: Path, relative: str, label: str,
) -> tuple[dict[str, str], bytes]:
    path = _file_under(root, _safe_relative_path(relative, label), label)
    return store.add(path, label, media_kind="image")


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return _sha256(value.encode("utf-8"))


def _freeze_topology(
    long_plan: dict[str, Any], project_id: str,
) -> list[dict[str, object]]:
    raw_segments = _list(long_plan.get("segments"), f"{project_id} long video segments")
    topology: list[dict[str, object]] = []
    for expected_index, raw in enumerate(raw_segments, 1):
        segment = _object(raw, f"{project_id} long video segment")
        chain_id = segment.get("chain_id")
        join_mode = segment.get("join_mode")
        if (
            segment.get("index") != expected_index
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
        ):
            raise ValueError(f"{project_id} long video topology is invalid")
        topology.append({
            "index": expected_index,
            "chain_id": chain_id,
            "join_mode": join_mode,
        })
    if not topology:
        raise ValueError(f"{project_id} long video topology is empty")
    return topology


def _segment_directories(project_root: Path) -> dict[int, Path]:
    segment_root = project_root / "work" / "segments"
    try:
        root_info = segment_root.lstat()
    except OSError as exc:
        raise ValueError("project segment root is missing") from exc
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("project segment root must be a real directory")
    result: dict[int, Path] = {}
    for entry in segment_root.iterdir():
        if not entry.name.isdigit():
            continue
        info = entry.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("project segment directory must not be a symlink")
        index = int(entry.name)
        if index < 1 or index in result:
            raise ValueError("project segment indices are invalid")
        result[index] = entry
    return result


def _freeze_source_segments(
    *,
    project_root: Path,
    project_id: str,
    topology: list[dict[str, object]],
    store: _BlobStore,
) -> list[dict[str, object]]:
    directories = _segment_directories(project_root)
    indices = [int(item["index"]) for item in topology]
    if sorted(directories) != indices:
        raise ValueError(f"{project_id} segment directories do not match topology")
    frozen_segments: list[dict[str, object]] = []
    for topology_item in topology:
        segment_index = int(topology_item["index"])
        base = f"work/segments/{segment_index}/work"
        sampling_ref, sampling = _json_blob(
            store,
            project_root,
            f"{base}/keyframe_sampling.json",
            f"{project_id} segment {segment_index} keyframe sampling",
        )
        raw_sampling_frames = _list(
            sampling.get("keyframes"),
            f"{project_id} segment {segment_index} sampled frames",
        )
        if not raw_sampling_frames:
            raise ValueError(f"{project_id} segment {segment_index} has no sampled frames")
        keyframe_dir = project_root / base / "keyframes"
        try:
            keyframe_info = keyframe_dir.lstat()
        except OSError as exc:
            raise ValueError("keyframe directory is missing") from exc
        if not stat.S_ISDIR(keyframe_info.st_mode) or stat.S_ISLNK(keyframe_info.st_mode):
            raise ValueError("keyframe directory must be a real directory")
        actual_names = sorted(
            entry.name for entry in keyframe_dir.iterdir()
            if entry.name.endswith(".png")
        )
        expected_names = [f"{order:02d}.png" for order in range(1, len(raw_sampling_frames) + 1)]
        if actual_names != expected_names:
            raise ValueError(f"{project_id} segment {segment_index} keyframe set is invalid")

        frozen_frames: list[dict[str, object]] = []
        for order, raw_frame in enumerate(raw_sampling_frames, 1):
            frame = _object(raw_frame, f"{project_id} sampled frame")
            transition = _object(frame.get("transition"), f"{project_id} frame transition")
            expected_path = f"keyframes/{order:02d}.png"
            declared_sha = frame.get("sha256")
            source_time_s = frame.get("source_time_s")
            source_scene_id = frame.get("source_scene_id")
            if (
                frame.get("order") != order
                or frame.get("path") != expected_path
                or not isinstance(declared_sha, str)
                or _SHA256_RE.fullmatch(declared_sha) is None
                or isinstance(source_time_s, bool)
                or not isinstance(source_time_s, (int, float))
                or not isinstance(source_scene_id, str)
                or not source_scene_id
                or transition.get("type") not in {"start", "continuous", "hard_cut"}
            ):
                raise ValueError(f"{project_id} sampled frame contract is invalid")
            logical_path = f"{base}/{expected_path}"
            frame_ref, _data = _image_blob(
                store,
                project_root,
                logical_path,
                f"{project_id} source keyframe {segment_index}:{order}",
            )
            if frame_ref["blob_sha256"] != declared_sha:
                raise ValueError(f"{project_id} sampled frame SHA does not match its file")
            frozen_frames.append({
                "order": order,
                "logical_path": logical_path,
                **frame_ref,
                "source_time_s": source_time_s,
                "source_scene_id": source_scene_id,
                "transition": {
                    "type": transition["type"],
                    "at_s": transition.get("at_s"),
                },
            })
        frozen_segments.append({
            **topology_item,
            "keyframe_sampling": {
                **sampling_ref,
                "schema": sampling.get("schema"),
                "version": sampling.get("version"),
            },
            "keyframes": frozen_frames,
        })
    return frozen_segments


def _freeze_fusion(
    *,
    project_root: Path,
    project_id: str,
    topology: list[dict[str, object]],
    store: _BlobStore,
) -> tuple[dict[str, object], str, list[int]]:
    input_ref, payload = _json_blob(
        store,
        project_root,
        "work/multimodal_input.json",
        f"{project_id} Fusion input",
    )
    raw_segments = _list(payload.get("segments"), f"{project_id} Fusion segments")
    expected_indices = [int(item["index"]) for item in topology]
    if [item.get("index") if isinstance(item, dict) else None for item in raw_segments] != expected_indices:
        raise ValueError(f"{project_id} Fusion segments do not match topology")
    frames: list[dict[str, object]] = []
    frame_counts: list[int] = []
    seen_paths: set[str] = set()
    for segment_index, raw_segment in zip(expected_indices, raw_segments, strict=True):
        segment = _object(raw_segment, f"{project_id} Fusion segment")
        raw_frames = _list(segment.get("new_keyframes"), f"{project_id} Fusion frames")
        raw_prompts = _list(
            segment.get("image_optimization_prompt"),
            f"{project_id} image optimization prompts",
        )
        if not raw_frames or len(raw_prompts) != len(raw_frames):
            raise ValueError(f"{project_id} Fusion frame/prompt cardinality is invalid")
        frame_counts.append(len(raw_frames))
        for order, raw_frame in enumerate(raw_frames, 1):
            frame = _object(raw_frame, f"{project_id} Fusion frame")
            raw_path = frame.get("path")
            declared_sha = frame.get("sha256")
            if (
                frame.get("order") != order
                or not isinstance(raw_path, str)
                or raw_path in seen_paths
                or not isinstance(declared_sha, str)
                or _SHA256_RE.fullmatch(declared_sha) is None
            ):
                raise ValueError(f"{project_id} Fusion frame contract is invalid")
            _safe_relative_path(raw_path, f"{project_id} Fusion frame path")
            seen_paths.add(raw_path)
            frame_ref, _data = _image_blob(
                store,
                project_root,
                raw_path,
                f"{project_id} Fusion frame {segment_index}:{order}",
            )
            if frame_ref["blob_sha256"] != declared_sha:
                raise ValueError(f"{project_id} Fusion frame SHA does not match its file")
            frames.append({
                "segment_index": segment_index,
                "order": order,
                "logical_path": raw_path,
                **frame_ref,
            })
    frozen = {
        "input": {
            **input_ref,
            "schema": payload.get("schema"),
            "version": payload.get("version"),
        },
        "new_keyframes": frames,
    }
    return frozen, input_ref["blob_sha256"], frame_counts


def _freeze_case(source: CaseSource, store: _BlobStore) -> dict[str, object]:
    if source.split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported corpus split: {source.split}")
    project_root = _require_absolute_source_root(source.project_root)
    project_id = project_root.name
    if _PROJECT_ID_RE.fullmatch(project_id) is None:
        raise ValueError("project directory name must be a 32-character lowercase hex ID")

    meta_path = _file_under(project_root, Path("meta.json"), f"{project_id} metadata")
    meta_read = _stable_read(meta_path, f"{project_id} metadata")
    meta = _object(_json_value(meta_read.data, f"{project_id} metadata"), "project metadata")
    if meta.get("id") != project_id or meta.get("status") != "done":
        raise ValueError(f"{project_id} is not a completed matching project")

    effective_request = _object(meta.get("effective_request"), f"{project_id} effective request")
    replacement_guidance = _object(
        effective_request.get("replacement_guidance"),
        f"{project_id} replacement guidance",
    )
    prompt = replacement_guidance.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
        raise ValueError(f"{project_id} replacement prompt is invalid")

    raw_reference_path = meta.get("_minimal_replacement_image_path")
    reference_relative = _safe_relative_path(
        raw_reference_path, f"{project_id} reference image path",
    )
    reference_path = _file_under(
        project_root, reference_relative, f"{project_id} reference image",
    )
    reference_ref, _reference_data = store.add(
        reference_path, f"{project_id} reference image", media_kind="image",
    )
    receipt = _object(meta.get("input_receipt"), f"{project_id} input receipt")
    reference_receipt = _object(
        receipt.get("replacement_image"), f"{project_id} reference receipt",
    )
    if reference_receipt != {
        "sha256": reference_ref["blob_sha256"],
        "bytes": store.records[reference_ref["blob_sha256"]]["bytes"],
    }:
        raise ValueError(f"{project_id} reference image receipt does not match")

    config_path = _file_under(
        project_root,
        Path("work/generation-config.json"),
        f"{project_id} generation config",
    )
    config_read = _stable_read(config_path, f"{project_id} generation config")
    config_document = _object(
        _json_value(config_read.data, f"{project_id} generation config"),
        f"{project_id} generation config",
    )
    full_config = _object(
        config_document.get("generation_config"),
        f"{project_id} generation config values",
    )
    generation_config = {
        key: full_config.get(key) for key in ("remove_subtitle", "remove_watermark")
    }
    if any(not isinstance(value, bool) for value in generation_config.values()):
        raise ValueError(f"{project_id} generation config values are invalid")

    long_plan_ref, long_plan = _json_blob(
        store,
        project_root,
        "long_video_plan.json",
        f"{project_id} long video plan",
    )
    topology = _freeze_topology(long_plan, project_id)
    source_segments = _freeze_source_segments(
        project_root=project_root,
        project_id=project_id,
        topology=topology,
        store=store,
    )
    element_index_ref, element_index = _json_blob(
        store,
        project_root,
        "work/element_index.json",
        f"{project_id} element index",
    )
    if set(element_index) != {"people", "entities", "scenes", "relations"} or any(
        not isinstance(element_index[key], dict)
        for key in ("people", "entities", "scenes", "relations")
    ):
        raise ValueError(f"{project_id} element index contract is invalid")

    fusion, fusion_input_sha256, fusion_frame_counts = _freeze_fusion(
        project_root=project_root,
        project_id=project_id,
        topology=topology,
        store=store,
    )
    if fusion_frame_counts != [len(item["keyframes"]) for item in source_segments]:
        raise ValueError(f"{project_id} source/Fusion frame cardinality differs")

    baseline_ref, baseline = _json_blob(
        store,
        project_root,
        "work/h3_prompt_plan.json",
        f"{project_id} historical Fusion output",
    )
    baseline_segments = _list(
        baseline.get("segments"), f"{project_id} historical Fusion segments",
    )
    if (
        baseline.get("input_sha256") != fusion_input_sha256
        or [item.get("index") if isinstance(item, dict) else None for item in baseline_segments]
        != [int(item["index"]) for item in topology]
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("visual"), list)
            or len(item["visual"]) != expected_count
            for item, expected_count in zip(baseline_segments, fusion_frame_counts, strict=True)
        )
    ):
        raise ValueError(f"{project_id} historical Fusion output is not bound to its input")

    return {
        "source_project_id": project_id,
        "split": source.split,
        "request": {
            "generation_config": generation_config,
            "user_replacement_prompt": prompt,
            "user_replacement_prompt_sha256": _sha256_text(
                prompt, f"{project_id} replacement prompt",
            ),
            "user_reference_image": reference_ref,
        },
        "image_postprocess": {
            "long_video_plan": long_plan_ref,
            "topology": topology,
            "element_index": element_index_ref,
            "segments": source_segments,
        },
        "video_prompt_fusion": fusion,
        "baseline": {
            "h3_prompt_plan": {
                **baseline_ref,
                "schema": baseline.get("schema"),
                "version": baseline.get("version"),
                "input_sha256": baseline.get("input_sha256"),
            },
        },
    }


def freeze_corpus(cases: Iterable[CaseSource], output_root: Path) -> FreezeReport:
    sources = list(cases)
    if not sources:
        raise ValueError("at least one corpus case is required")
    parent = _prepare_output_parent(output_root)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=parent))
    published = False
    try:
        blob_store = _BlobStore(staging / "blobs" / "sha256")
        frozen_cases = [_freeze_case(source, blob_store) for source in sources]
        frozen_cases.sort(key=lambda item: str(item["source_project_id"]))
        project_ids = [str(item["source_project_id"]) for item in frozen_cases]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("corpus project IDs must be unique")
        unique_blob_bytes = sum(int(item["bytes"]) for item in blob_store.records.values())
        manifest = {
            "schema": CORPUS_SCHEMA,
            "version": CORPUS_VERSION,
            "stats": {
                "case_count": len(frozen_cases),
                "unique_blob_count": len(blob_store.records),
                "unique_blob_bytes": unique_blob_bytes,
            },
            "blobs": blob_store.records,
            "cases": frozen_cases,
        }
        manifest_data = _canonical_json_bytes(manifest)
        manifest_sha256 = _sha256(manifest_data)
        _write_new_file(staging / "manifest.json", manifest_data)
        _write_new_file(
            staging / "manifest.sha256",
            f"{manifest_sha256}  manifest.json\n".encode("ascii"),
        )
        if os.path.lexists(output_root):
            raise FileExistsError(f"output root appeared during freeze: {output_root}")
        os.rename(staging, output_root)
        published = True
        return FreezeReport(
            output_root=output_root,
            manifest_sha256=manifest_sha256,
            case_count=len(frozen_cases),
            unique_blob_count=len(blob_store.records),
            unique_blob_bytes=unique_blob_bytes,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _require_absolute_manifest(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("corpus manifest path must be absolute")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise ValueError("corpus manifest is missing or invalid") from exc
    if (
        resolved != path
        or path.name != "manifest.json"
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
    ):
        raise ValueError("corpus manifest must be a canonical, single-link regular file")
    return path


def _manifest_digest(manifest_path: Path, manifest_data: bytes) -> str:
    digest_path = manifest_path.with_name("manifest.sha256")
    try:
        digest_info = digest_path.lstat()
    except OSError as exc:
        raise ValueError("corpus manifest digest is missing") from exc
    if (
        not stat.S_ISREG(digest_info.st_mode)
        or stat.S_ISLNK(digest_info.st_mode)
        or digest_info.st_nlink != 1
    ):
        raise ValueError("corpus manifest digest must be a single-link regular file")
    digest_read = _stable_read(digest_path, "corpus manifest digest")
    expected = f"{_sha256(manifest_data)}  manifest.json\n".encode("ascii")
    if digest_read.data != expected:
        raise ValueError("corpus manifest SHA-256 does not match")
    return _sha256(manifest_data)


def _validated_blob_record(digest: str, value: Any) -> dict[str, object]:
    if _SHA256_RE.fullmatch(digest) is None or not isinstance(value, dict):
        raise ValueError("corpus blob manifest is invalid")
    media_type = value.get("media_type")
    byte_count = value.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(media_type, str)
    ):
        raise ValueError("corpus blob metadata is invalid")
    expected_keys = (
        {"bytes", "media_type"}
        if media_type == "application/json"
        else {"bytes", "media_type", "width", "height"}
    )
    if set(value) != expected_keys:
        raise ValueError("corpus blob metadata fields are invalid")
    if media_type != "application/json" and (
        media_type not in {"image/png", "image/jpeg"}
        or isinstance(value.get("width"), bool)
        or not isinstance(value.get("width"), int)
        or value["width"] < 1
        or isinstance(value.get("height"), bool)
        or not isinstance(value.get("height"), int)
        or value["height"] < 1
    ):
        raise ValueError("corpus image metadata is invalid")
    return value


def _verify_corpus(
    manifest_path: Path,
) -> tuple[dict[str, Any], str, Path, dict[str, dict[str, object]]]:
    manifest_path = _require_absolute_manifest(manifest_path)
    manifest_read = _stable_read(manifest_path, "corpus manifest")
    manifest_sha256 = _manifest_digest(manifest_path, manifest_read.data)
    manifest = _object(
        _json_value(manifest_read.data, "corpus manifest"), "corpus manifest",
    )
    if (
        set(manifest) != {"schema", "version", "stats", "blobs", "cases"}
        or manifest.get("schema") != CORPUS_SCHEMA
        or manifest.get("version") != CORPUS_VERSION
    ):
        raise ValueError("corpus manifest contract is invalid")
    raw_blobs = _object(manifest.get("blobs"), "corpus blobs")
    blobs = {
        digest: _validated_blob_record(digest, value)
        for digest, value in raw_blobs.items()
    }
    blob_root = manifest_path.parent / "blobs" / "sha256"
    try:
        blob_root_resolved = blob_root.resolve(strict=True)
        blob_root_info = blob_root.lstat()
    except OSError as exc:
        raise ValueError("corpus blob directory is missing") from exc
    if (
        blob_root_resolved != blob_root
        or not stat.S_ISDIR(blob_root_info.st_mode)
        or stat.S_ISLNK(blob_root_info.st_mode)
    ):
        raise ValueError("corpus blob directory must be canonical and symlink-free")
    entries = list(blob_root.iterdir())
    if {entry.name for entry in entries} != set(blobs):
        raise ValueError("corpus blob directory does not match the manifest")
    for entry in entries:
        info = entry.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
        ):
            raise ValueError("corpus blobs must be single-link regular files")
        frozen = _stable_read(entry, f"corpus blob {entry.name}")
        metadata = blobs[entry.name]
        if frozen.sha256 != entry.name or frozen.size != metadata["bytes"]:
            raise ValueError("corpus blob SHA-256 or byte count does not match")
        if metadata["media_type"] == "application/json":
            _json_value(frozen.data, f"corpus JSON blob {entry.name}")
        elif _image_metadata(frozen.data) != {
            "media_type": metadata["media_type"],
            "width": metadata["width"],
            "height": metadata["height"],
        }:
            raise ValueError("corpus image metadata does not match its bytes")
    cases = _list(manifest.get("cases"), "corpus cases")
    validated_case_ids: list[str] = []
    for raw_case in cases:
        case = _object(raw_case, "corpus case")
        _copy_plan_for_case(case, blobs)
        validated_case_ids.append(str(case.get("source_project_id")))
    if len(validated_case_ids) != len(set(validated_case_ids)):
        raise ValueError("corpus source project IDs must be unique")
    stats = _object(manifest.get("stats"), "corpus stats")
    expected_stats = {
        "case_count": len(cases),
        "unique_blob_count": len(blobs),
        "unique_blob_bytes": sum(int(value["bytes"]) for value in blobs.values()),
    }
    if stats != expected_stats:
        raise ValueError("corpus statistics do not match its blobs")
    return manifest, manifest_sha256, blob_root, blobs


def _blob_digest(reference: Any, blobs: dict[str, dict[str, object]], label: str) -> str:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} blob reference is invalid")
    digest = reference.get("blob_sha256")
    if not isinstance(digest, str) or digest not in blobs:
        raise ValueError(f"{label} references an unknown blob")
    return digest


def _reference_layout_path(
    digest: str, blobs: dict[str, dict[str, object]],
) -> str:
    media_type = blobs[digest]["media_type"]
    if media_type == "image/jpeg":
        return "inputs/replacement_image.jpg"
    if media_type == "image/png":
        return "inputs/replacement_image.png"
    raise ValueError("user reference blob is not a supported image")


def _copy_plan_for_case(
    case: dict[str, Any], blobs: dict[str, dict[str, object]],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    project_id = case.get("source_project_id")
    split = case.get("split")
    if (
        not isinstance(project_id, str)
        or _PROJECT_ID_RE.fullmatch(project_id) is None
        or split not in ALLOWED_SPLITS
    ):
        raise ValueError("corpus case identity is invalid")
    request = _object(case.get("request"), f"{project_id} request")
    config = _object(request.get("generation_config"), f"{project_id} config")
    if set(config) != {"remove_subtitle", "remove_watermark"} or any(
        not isinstance(value, bool) for value in config.values()
    ):
        raise ValueError(f"{project_id} generation config is invalid")
    prompt = request.get("user_replacement_prompt")
    prompt_sha = request.get("user_replacement_prompt_sha256")
    if (
        not isinstance(prompt, str)
        or not prompt
        or prompt_sha != _sha256(prompt.encode("utf-8"))
    ):
        raise ValueError(f"{project_id} replacement prompt digest is invalid")
    reference_sha = _blob_digest(
        request.get("user_reference_image"), blobs, f"{project_id} reference image",
    )
    reference_path = _reference_layout_path(reference_sha, blobs)

    image = _object(case.get("image_postprocess"), f"{project_id} image inputs")
    long_plan_sha = _blob_digest(
        image.get("long_video_plan"), blobs, f"{project_id} long video plan",
    )
    element_index_sha = _blob_digest(
        image.get("element_index"), blobs, f"{project_id} element index",
    )
    topology = _list(image.get("topology"), f"{project_id} topology")
    source_segments = _list(image.get("segments"), f"{project_id} source segments")
    for expected_index, raw_topology in enumerate(topology, 1):
        topology_item = _object(raw_topology, f"{project_id} topology item")
        if (
            topology_item.get("index") != expected_index
            or not isinstance(topology_item.get("chain_id"), str)
            or not topology_item["chain_id"]
            or topology_item.get("join_mode") not in {"hard_cut", "continue"}
        ):
            raise ValueError(f"{project_id} topology is invalid")
    if not topology or len(source_segments) != len(topology):
        raise ValueError(f"{project_id} topology is invalid")

    plan: list[tuple[str, str]] = [
        ("long_video_plan.json", long_plan_sha),
        ("work/element_index.json", element_index_sha),
        (reference_path, reference_sha),
    ]
    for expected_index, raw_segment in enumerate(source_segments, 1):
        segment = _object(raw_segment, f"{project_id} source segment")
        if (
            segment.get("index") != expected_index
            or segment.get("chain_id") != topology[expected_index - 1].get("chain_id")
            or segment.get("join_mode") != topology[expected_index - 1].get("join_mode")
        ):
            raise ValueError(f"{project_id} source segment topology is invalid")
        sampling_sha = _blob_digest(
            segment.get("keyframe_sampling"),
            blobs,
            f"{project_id} segment {expected_index} sampling",
        )
        plan.append((
            f"work/segments/{expected_index}/work/keyframe_sampling.json",
            sampling_sha,
        ))
        frames = _list(segment.get("keyframes"), f"{project_id} source keyframes")
        if not frames:
            raise ValueError(f"{project_id} source keyframes are empty")
        for expected_order, raw_frame in enumerate(frames, 1):
            frame = _object(raw_frame, f"{project_id} source keyframe")
            expected_path = (
                f"work/segments/{expected_index}/work/keyframes/{expected_order:02d}.png"
            )
            if frame.get("order") != expected_order or frame.get("logical_path") != expected_path:
                raise ValueError(f"{project_id} source keyframe path is invalid")
            _safe_relative_path(expected_path, f"{project_id} source keyframe path")
            plan.append((
                expected_path,
                _blob_digest(frame, blobs, f"{project_id} source keyframe"),
            ))

    fusion = _object(case.get("video_prompt_fusion"), f"{project_id} Fusion inputs")
    fusion_input_sha = _blob_digest(
        fusion.get("input"), blobs, f"{project_id} Fusion input",
    )
    plan.append(("work/multimodal_input.json", fusion_input_sha))
    fusion_frames = _list(fusion.get("new_keyframes"), f"{project_id} Fusion keyframes")
    if not fusion_frames:
        raise ValueError(f"{project_id} Fusion keyframes are empty")
    for raw_frame in fusion_frames:
        frame = _object(raw_frame, f"{project_id} Fusion keyframe")
        digest = _blob_digest(frame, blobs, f"{project_id} Fusion keyframe")
        expected_path = f"work/prompt-fusion-proxies/{digest}.png"
        raw_path = frame.get("logical_path")
        if raw_path != expected_path:
            raise ValueError(f"{project_id} Fusion keyframe path is invalid")
        _safe_relative_path(raw_path, f"{project_id} Fusion keyframe path")
        plan.append((raw_path, digest))

    baseline = _object(case.get("baseline"), f"{project_id} baseline")
    h3_reference = _object(
        baseline.get("h3_prompt_plan"), f"{project_id} H3 prompt baseline",
    )
    h3_sha = _blob_digest(h3_reference, blobs, f"{project_id} H3 prompt baseline")
    if h3_reference.get("input_sha256") != fusion_input_sha:
        raise ValueError(f"{project_id} baseline is not bound to its Fusion input")
    plan.append(("work/h3_prompt_plan.json", h3_sha))

    paths = [item[0] for item in plan]
    if len(paths) != len(set(paths)) or "case.json" in paths:
        raise ValueError(f"{project_id} materialized paths collide")
    for path in paths:
        _safe_relative_path(path, f"{project_id} materialized path")
    case_document = {
        "schema": "duet.skill-eval-case",
        "version": 1,
        "source_project_id": project_id,
        "split": split,
        "request": {
            "generation_config": config,
            "user_replacement_prompt": prompt,
            "user_replacement_prompt_sha256": prompt_sha,
            "user_reference_image": {
                "path": reference_path,
                "sha256": reference_sha,
            },
        },
        "topology": topology,
        "inputs": {
            "long_video_plan": "long_video_plan.json",
            "element_index": "work/element_index.json",
            "fusion_input": "work/multimodal_input.json",
            "baseline_h3_prompt_plan": "work/h3_prompt_plan.json",
        },
    }
    return plan, case_document


def _copy_blob_independently(source: Path, destination: Path, label: str) -> int:
    source_info = source.lstat()
    if source_info.st_nlink != 1:
        raise ValueError(f"{label} source blob must not be hard-linked")
    before = _stable_read(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_new_file(destination, before.data)
    source_after = _stable_read(source, label)
    copied = _stable_read(destination, f"materialized {label}")
    copied_info = destination.lstat()
    if (
        (source_after.sha256, source_after.size) != (before.sha256, before.size)
        or (copied.sha256, copied.size) != (before.sha256, before.size)
        or (copied.device, copied.inode) == (before.device, before.inode)
        or copied_info.st_nlink != 1
        or stat.S_ISLNK(copied_info.st_mode)
    ):
        raise ValueError(f"{label} was not materialized as an independent byte copy")
    return copied.size


def materialize_case(
    manifest_path: Path,
    source_project_id: str,
    output_root: Path,
) -> MaterializeReport:
    if _PROJECT_ID_RE.fullmatch(source_project_id) is None:
        raise ValueError("source project ID must be 32 lowercase hexadecimal characters")
    parent = _prepare_output_parent(output_root)
    manifest, manifest_sha256, blob_root, blobs = _verify_corpus(manifest_path)
    raw_cases = _list(manifest.get("cases"), "corpus cases")
    matches = [
        item for item in raw_cases
        if isinstance(item, dict) and item.get("source_project_id") == source_project_id
    ]
    if len(matches) != 1:
        raise ValueError("source project ID does not select exactly one corpus case")
    case = _object(matches[0], "selected corpus case")
    copy_plan, case_document = _copy_plan_for_case(case, blobs)
    case_document["corpus_manifest_sha256"] = manifest_sha256

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=parent))
    published = False
    try:
        copied_bytes = 0
        for logical_path, digest in copy_plan:
            destination = staging / _safe_relative_path(logical_path, "materialized path")
            copied_bytes += _copy_blob_independently(
                blob_root / digest,
                destination,
                f"blob {digest}",
            )
        case_data = _canonical_json_bytes(case_document)
        _write_new_file(staging / "case.json", case_data)
        copied_bytes += len(case_data)
        if os.path.lexists(output_root):
            raise FileExistsError(f"output root appeared during materialization: {output_root}")
        os.rename(staging, output_root)
        published = True
        return MaterializeReport(
            output_root=output_root,
            manifest_sha256=manifest_sha256,
            source_project_id=source_project_id,
            file_count=len(copy_plan) + 1,
            copied_bytes=copied_bytes,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _case_argument(value: str) -> CaseSource:
    split, separator, raw_path = value.partition("=")
    if not separator or split not in ALLOWED_SPLITS or not raw_path:
        raise argparse.ArgumentTypeError(
            "case must be SPLIT=/absolute/project/root; SPLIT is train, regression, or holdout"
        )
    project_root = Path(raw_path)
    if not project_root.is_absolute():
        raise argparse.ArgumentTypeError("case project root must be absolute")
    return CaseSource(split=split, project_root=project_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or materialize a SHA-addressed offline Skill evaluation corpus."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser(
        "freeze", help="freeze completed projects into a new corpus",
    )
    freeze_parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_case_argument,
        help="repeat SPLIT=/absolute/project/root",
    )
    freeze_parser.add_argument("--output-root", required=True, type=Path)
    materialize_parser = commands.add_parser(
        "materialize", help="materialize one corpus case into a new input tree",
    )
    materialize_parser.add_argument("--manifest", required=True, type=Path)
    materialize_parser.add_argument("--source-project-id", required=True)
    materialize_parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.output_root.is_absolute():
        parser.error("--output-root must be absolute")
    try:
        if args.command == "freeze":
            report = freeze_corpus(args.case, args.output_root)
            payload = {
                "output_root": str(report.output_root),
                "manifest_sha256": report.manifest_sha256,
                "case_count": report.case_count,
                "unique_blob_count": report.unique_blob_count,
                "unique_blob_bytes": report.unique_blob_bytes,
            }
        else:
            if not args.manifest.is_absolute():
                parser.error("--manifest must be absolute")
            materialized = materialize_case(
                args.manifest,
                args.source_project_id,
                args.output_root,
            )
            payload = {
                "output_root": str(materialized.output_root),
                "manifest_sha256": materialized.manifest_sha256,
                "source_project_id": materialized.source_project_id,
                "file_count": materialized.file_count,
                "copied_bytes": materialized.copied_bytes,
            }
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
