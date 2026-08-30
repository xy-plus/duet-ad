"""Explicit continuation of a failed project image phase.

This module is intentionally a narrow operator seam.  It does not recover a
generic pipeline failure: the caller must first present the exact manifest
produced by :func:`inspect`, and the source project must still be the terminal
pre-image state.  The image compiler and all canonical freezing remain owned
by ``pipeline``/``image_optimization``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import cv2

from app import image_optimization, long_video, pipeline, storage
from app.config import Settings
from app.codex_runner import CodexRunner


SCHEMA = "duet.image-phase-resume"
VERSION = 1
IMAGE_FAILURE = "image optimization output is missing or invalid"
_CID_RE = r"[0-9a-f]{32}"
_PROMPT_BYTES = 32 * 1024
_JSON_BYTES = 4 * 1024 * 1024
_SHA_RE = r"[0-9a-f]{64}"
_META_FIELDS = (
    "schema_version", "id", "duration_s", "dialogue_mode", "voice_mode",
    "target_language", "voice_lines", "voice_line_provenance",
    "vocal_filter_enabled", "has_bgm", "source_width", "source_height",
)
_DOWNSTREAM_KEYS = (
    "generation", "postprocess", "_postprocess_receipt", "_image_continuity",
    "_image_optimization", "_image_verification", "_v4_canvas_execution",
    "_image_acceptance", "long_video_plan_receipt", "frozen_plan_receipt",
    "prepared_input_receipt", "_prompt_fusion",
)


class ResumeRejected(ValueError):
    """The explicit operator precondition or artifact manifest is invalid."""


class ResumeExecutionError(RuntimeError):
    """The image phase failed after all preconditions were proven."""


class _DiagnosticRunner:
    """Delegate execution while preserving exact phase protocol artifacts."""

    def __init__(self, inner: object, destination: Path) -> None:
        self._inner = inner
        self._destination = destination
        if destination.exists() or destination.is_symlink():
            raise ResumeRejected("diagnostics directory already exists")
        destination.mkdir(parents=True, mode=0o700)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    @staticmethod
    def _label(stage: Path, output: Path) -> str:
        if output.name == "global_plan.json":
            return "global-plan"
        matched = re.fullmatch(r"duet-image-segment-(\d+)-.+", stage.name)
        if output.name == "segment_frames.json" and matched is not None:
            return f"segment-{int(matched.group(1)):04d}"
        raise ResumeExecutionError("unexpected image diagnostic phase")

    @staticmethod
    def _read_regular(path: Path, maximum: int) -> bytes | None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError:
            return None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                return None
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                value = stream.read(maximum + 1)
            return value if len(value) <= maximum else None
        finally:
            os.close(descriptor)

    def _capture(
        self, stage: Path, output: Path, maximum: int, error: BaseException | None,
    ) -> None:
        label = self._label(stage, output)
        request = self._read_regular(stage / "work" / "request.json", _JSON_BYTES)
        raw_output = self._read_regular(output, maximum)
        if request is not None:
            _publish_bytes(self._destination / f"{label}.request.json", request)
        if raw_output is not None:
            _publish_bytes(self._destination / f"{label}.output.json", raw_output)
        if error is not None:
            _publish_bytes(
                self._destination / f"{label}.error.txt",
                (f"{type(error).__name__}: {error}\n").encode("utf-8"),
            )

    def run_isolated_until_output(
        self,
        workdir: Path,
        prompt: str,
        *,
        session_dir: Path,
        output_path: Path,
        max_output_bytes: int,
        validate_output,
    ):
        try:
            result = self._inner.run_isolated_until_output(
                workdir,
                prompt,
                session_dir=session_dir,
                output_path=output_path,
                max_output_bytes=max_output_bytes,
                validate_output=validate_output,
            )
        except BaseException as exc:
            self._capture(workdir, output_path, max_output_bytes, exc)
            raise
        self._capture(workdir, output_path, max_output_bytes, None)
        return result


@dataclass(frozen=True)
class _Snapshot:
    settings: Settings
    cid: str
    root: Path
    meta: dict
    source: Path
    element_index: Path
    segments: list[dict]
    segment_metas: list[dict]
    manifest: dict


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResumeRejected("resume manifest is not canonical JSON") from exc


def _regular(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ResumeRejected(f"{label} path is invalid")
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise ResumeRejected(f"{label} is a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ResumeRejected(f"{label} path is outside conversation") from None
    if not path.is_file():
        raise ResumeRejected(f"{label} is missing or not a regular file")
    return path


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ResumeRejected(f"cannot read artifact: {path.name}") from exc
    return digest.hexdigest(), size


def _record(root: Path, relative: str, *, label: str) -> dict:
    path = _regular(root, relative, label=label)
    sha256, size = _digest(path)
    return {"path": relative, "sha256": sha256, "size": size}


def _json(path: Path, *, label: str) -> object:
    try:
        if path.stat().st_size > _JSON_BYTES:
            raise ResumeRejected(f"{label} is too large")
        return json.loads(path.read_text(encoding="utf-8"))
    except ResumeRejected:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeRejected(f"{label} is missing or invalid") from exc


def _text(path: Path, *, label: str) -> str:
    try:
        data = path.read_bytes()
        if not data or len(data) > _PROMPT_BYTES:
            raise ResumeRejected(f"{label} is empty or too large")
        value = data.decode("utf-8")
    except ResumeRejected:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise ResumeRejected(f"{label} is missing or invalid") from exc
    if not value.strip():
        raise ResumeRejected(f"{label} is empty")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResumeRejected(f"{label} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ResumeRejected(f"{label} is invalid")
    return result


def _guard_meta(meta: dict, cid: str) -> float:
    if (
        not isinstance(cid, str)
        or re.fullmatch(_CID_RE, cid) is None
        or meta.get("id") != cid
        or meta.get("schema_version") != 2
    ):
        raise ResumeRejected("conversation identity is invalid")
    if meta.get("status") != "failed" or meta.get("error") != IMAGE_FAILURE:
        raise ResumeRejected("conversation is not a terminal image-phase failure")
    if meta.get("_input_owner") is not None:
        raise ResumeRejected("conversation still has an active input claim")
    if any(meta.get(key) is not None for key in _DOWNSTREAM_KEYS):
        raise ResumeRejected("downstream state already exists")
    if "segments" in meta:
        raise ResumeRejected("partial segment state is not resumable")
    duration = _finite(meta.get("duration_s"), "duration_s")
    if duration <= 0 or meta.get("dialogue_mode") not in {"auto", "edit", "custom", "none"}:
        raise ResumeRejected("conversation input facts are invalid")
    if not isinstance(meta.get("voice_lines"), list):
        raise ResumeRejected("conversation dialogue facts are invalid")
    return duration


def _load_segments(
    root: Path, duration: float, meta_voice_lines: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    work = root / "work"
    scenes_path = _regular(root, "work/scenes.json", label="scenes")
    raw = _json(scenes_path, label="scenes")
    if not isinstance(raw, dict) or set(raw) != {
        "duration_s", "scenes", "effective_scenes", "diagnostics", "segments",
    }:
        raise ResumeRejected("scenes shape is invalid")
    if abs(_finite(raw.get("duration_s"), "scenes duration") - duration) > 1e-6:
        raise ResumeRejected("scenes duration does not match conversation")
    try:
        pipeline._load_scenes(work)
        source_scenes = pipeline._source_scenes_for_timeline(work, duration)
    except pipeline.PipelineError as exc:
        raise ResumeRejected(f"scene boundary facts are invalid: {exc}") from None
    if not isinstance(raw.get("segments"), list) or not raw["segments"]:
        raise ResumeRejected("scene segments are missing")
    try:
        segments = pipeline.scene_planner.plan_segments(
            duration, source_scenes, meta_voice_lines,
        )
    except (ValueError, long_video.LongVideoError) as exc:
        raise ResumeRejected(f"scene segment plan is invalid: {exc}") from None
    if not segments:
        raise ResumeRejected("scene segment plan is empty")
    previous_end = 0.0
    for expected, item in enumerate(segments, 1):
        if not isinstance(item, dict) or not {
            "index", "start_s", "end_s", "chain_id", "join_mode",
        }.issubset(item):
            raise ResumeRejected("scene segment shape is invalid")
        start = _finite(item.get("start_s"), f"segment {expected} start")
        end = _finite(item.get("end_s"), f"segment {expected} end")
        if (
            item.get("index") != expected
            or start < 0 or end <= start
            or abs(start - previous_end) > 1e-6
            or not isinstance(item.get("chain_id"), str)
            or not item["chain_id"]
            or item.get("join_mode") not in {"hard_cut", "continue"}
        ):
            raise ResumeRejected("scene segment boundaries are invalid")
        previous_end = end
    if abs(previous_end - duration) > 1e-6:
        raise ResumeRejected("scene segments do not cover the source duration")
    segments = [
        {
            **{key: item[key] for key in (
                "index", "start_s", "end_s", "chain_id", "join_mode",
            )},
            **({"scene_indices": item["scene_indices"]}
               if "scene_indices" in item else {}),
            **({"source_cut_timeline": item["source_cut_timeline"]}
               if "source_cut_timeline" in item else {}),
        }
        for item in segments
    ]

    segment_metas = []
    for segment in segments:
        index = segment["index"]
        segroot = root / "work" / "segments" / str(index)
        segwork = segroot / "work"
        source_relative = f"segments/{index}/source.mp4"
        source_artifact_relative = f"work/{source_relative}"
        # The project currently writes MP4 segment sources.  Restricting this
        # operator to that exact output prevents an accidental alternate input.
        _regular(root, source_artifact_relative, label=f"segment {index} source")
        manifest = _json(
            _regular(root, f"work/segments/{index}/work/manifest.json", label=f"segment {index} manifest"),
            label=f"segment {index} manifest",
        )
        if not isinstance(manifest, dict):
            raise ResumeRejected(f"segment {index} manifest is invalid")
        segment_duration = _finite(
            manifest.get("duration_seconds"), f"segment {index} duration"
        )
        if abs(segment_duration - (segment["end_s"] - segment["start_s"])) > 0.11:
            raise ResumeRejected(f"segment {index} duration does not match boundary")

        keyframe_dir = segwork / "keyframes"
        names = [f"{order:02d}.png" for order in range(1, 10)]
        if not keyframe_dir.is_dir() or sorted(
            path.name for path in keyframe_dir.glob("*.png") if path.is_file()
        ) != names:
            raise ResumeRejected(f"segment {index} keyframe set is incomplete")
        for name in names:
            path = keyframe_dir / name
            if path.is_symlink() or cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
                raise ResumeRejected(f"segment {index} keyframe {name} is invalid")

        sampling_path = _regular(
            root, f"work/segments/{index}/work/keyframe_sampling.json",
            label=f"segment {index} keyframe sampling",
        )
        sampling = _json(sampling_path, label=f"segment {index} keyframe sampling")
        if not isinstance(sampling, dict) or set(sampling) != {
            "schema", "version", "selection_method", "keyframes",
        } or sampling.get("schema") != "duet.backend-keyframe-sampling" \
                or sampling.get("version") != 1 \
                or not isinstance(sampling.get("keyframes"), list) \
                or len(sampling["keyframes"]) != 9:
            raise ResumeRejected(f"segment {index} keyframe sampling is invalid")
        for order, item in enumerate(sampling["keyframes"], 1):
            if not isinstance(item, dict) or not {
                "order", "path", "source_scene_id", "source_time_s", "repeated",
                "sha256",
            }.issubset(item) or item.get("order") != order \
                    or item.get("path") != f"keyframes/{order:02d}.png" \
                    or not isinstance(item.get("source_scene_id"), str) \
                    or not item["source_scene_id"] \
                    or isinstance(item.get("repeated"), bool) is False \
                    or re.fullmatch(_SHA_RE, str(item.get("sha256"))) is None:
                raise ResumeRejected(f"segment {index} keyframe sampling is invalid")
            sha256, _size = _digest(keyframe_dir / f"{order:02d}.png")
            if sha256 != item["sha256"]:
                raise ResumeRejected(f"segment {index} keyframe bytes drifted")

        dialogue_path = _regular(
            root, f"work/segments/{index}/work/voice_lines.json",
            label=f"segment {index} dialogue",
        )
        dialogue = _json(dialogue_path, label=f"segment {index} dialogue")
        if not isinstance(dialogue, list):
            raise ResumeRejected(f"segment {index} dialogue is invalid")
        try:
            expected_dialogue = long_video.localize_dialogue(
                meta_voice_lines, segment, segments=segments
            )
        except long_video.LongVideoError:
            raise ResumeRejected("conversation dialogue boundaries are invalid") from None
        if dialogue != expected_dialogue:
            raise ResumeRejected(f"segment {index} dialogue drifted")

        visual = _text(
            _regular(root, f"work/segments/{index}/work/visual_prompt.txt", label=f"segment {index} visual prompt"),
            label=f"segment {index} visual prompt",
        )
        prompt = _text(
            _regular(root, f"work/segments/{index}/work/prompt.txt", label=f"segment {index} prompt"),
            label=f"segment {index} prompt",
        )
        _regular(root, f"work/segments/{index}/work/anchors/first.png", label=f"segment {index} first anchor")
        _regular(root, f"work/segments/{index}/work/anchors/last.png", label=f"segment {index} last anchor")
        segment_metas.append({
            **segment,
            "source": source_relative,
            "keyframes": names,
            "keyframe_paths": [f"segments/{index}/work/keyframes/{name}" for name in names],
            "first_frame_path": f"segments/{index}/work/anchors/first.png",
            "last_frame_path": f"segments/{index}/work/anchors/last.png",
            "visual_prompt": visual,
            "prompt": prompt,
            "dialogue": dialogue,
            "lines": [item.get("text") for item in dialogue if isinstance(item, dict)],
            "keyframe_sampling": sampling,
        })
    return segments, segment_metas, source_scenes


def _meta_projection(meta: Mapping) -> dict:
    return {key: meta.get(key) for key in _META_FIELDS if key in meta}


def _collect(settings: Settings, cid: str) -> _Snapshot:
    if not isinstance(settings.data_dir, Path) or not settings.data_dir.is_absolute():
        raise ResumeRejected("data_dir must be absolute")
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        raise ResumeRejected("conversation not found")
    duration = _guard_meta(meta, cid)
    root = (settings.data_dir / cid).resolve()
    if any((root / name).exists() for name in (
        long_video.PLAN_RECEIPT_FILENAME,
        "prepared_input.json",
    )):
        raise ResumeRejected("frozen downstream input already exists")
    source_paths = sorted(
        path for path in root.glob("source.*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in storage.ALLOWED_EXT
    )
    if len(source_paths) != 1:
        raise ResumeRejected("source input is missing or ambiguous")
    source = source_paths[0]
    _regular(root, "work/manifest.json", label="root manifest")
    root_manifest = _json(root / "work" / "manifest.json", label="root manifest")
    if not isinstance(root_manifest, dict) or abs(
        _finite(root_manifest.get("duration_seconds"), "root duration") - duration
    ) > 1e-6:
        raise ResumeRejected("root manifest duration does not match conversation")

    raw_index_path = _regular(root, "work/element_index.json", label="element index")
    raw_index = _json(raw_index_path, label="element index")
    canonical_index = image_optimization._canonical_element_index(raw_index)
    if raw_index != canonical_index:
        raise ResumeRejected("element index is not canonical")
    planner_dialogue = pipeline._planner_dialogue(
        meta, meta.get("voice_lines", [])
    )
    segments, segment_metas, source_scenes = _load_segments(
        root, duration, planner_dialogue
    )
    # This is the existing backend binding check.  It verifies every selected
    # frame hash, source scene id, transition, and scene-anchor coverage.
    try:
        segment_metas = pipeline._bind_keyframe_source_timeline(
            root / "work", segments, segment_metas, source_scenes,
        )
    except pipeline.PipelineError as exc:
        raise ResumeRejected(f"keyframe source facts are invalid: {exc}") from None

    # Recheck dialogue with the immutable conversation lines now that the
    # segment list is known.  The earlier loader deliberately stays local.
    for segment, segment_meta in zip(segments, segment_metas):
        try:
            expected = long_video.localize_dialogue(
                planner_dialogue, segment, segments=segments,
            )
        except long_video.LongVideoError:
            raise ResumeRejected("conversation dialogue boundaries are invalid") from None
        if segment_meta["dialogue"] != expected:
            raise ResumeRejected(f"segment {segment['index']} dialogue drifted")

    artifact_paths = [
        f"source{source.suffix}",
        "work/manifest.json", "work/scenes.json", "work/element_index.json",
    ]
    for segment in segments:
        index = segment["index"]
        artifact_paths.extend([
            f"work/segments/{index}/source.mp4",
            f"work/segments/{index}/work/manifest.json",
            f"work/segments/{index}/work/keyframe_sampling.json",
            f"work/segments/{index}/work/voice_lines.json",
            f"work/segments/{index}/work/visual_prompt.txt",
            f"work/segments/{index}/work/prompt.txt",
            f"work/segments/{index}/work/anchors/first.png",
            f"work/segments/{index}/work/anchors/last.png",
            *[f"work/segments/{index}/work/keyframes/{order:02d}.png" for order in range(1, 10)],
        ])
    artifacts = [
        _record(root, relative, label="resume artifact")
        for relative in artifact_paths
    ]
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "cid": cid,
        "failure": {"status": "failed", "error": IMAGE_FAILURE},
        "meta": _meta_projection(meta),
        "segments": segments,
        "artifacts": artifacts,
    }
    # Preserve source scene parsing as part of collection even though its
    # bytes are represented by the scenes artifact digest above.
    del canonical_index, root_manifest
    return _Snapshot(
        settings=settings, cid=cid, root=root, meta=meta, source=source,
        element_index=raw_index_path, segments=segments,
        segment_metas=segment_metas, manifest=manifest,
    )


def inspect(settings: Settings, cid: str) -> dict:
    """Read-only proof of the exact artifacts that would be reused."""
    return _collect(settings, cid).manifest


def _same_manifest(expected: object, observed: dict) -> bool:
    if not isinstance(expected, dict):
        return False
    try:
        return _canonical(expected) == _canonical(observed)
    except ResumeRejected:
        return False


@contextmanager
def _exclusive(root: Path) -> Iterator[None]:
    lock_path = root / ".image-phase-resume.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        raise ResumeRejected("cannot acquire image-phase operator lock") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ResumeRejected("another image-phase resume is already running") from None
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _receipt_segments(root: Path, segment_metas: list[dict]) -> list[dict]:
    result = []
    for item in segment_metas:
        index = item["index"]
        segroot = root / "work" / "segments" / str(index)
        result.append({
            **item,
            "source_path": segroot / "source.mp4",
            "keyframe_paths": [
                segroot / "work" / "keyframes" / name for name in item["keyframes"]
            ],
            "first_frame_path": segroot / "work" / "anchors" / "first.png",
            "last_frame_path": segroot / "work" / "anchors" / "last.png",
            "visual_prompt_path": segroot / "work" / "visual_prompt.txt",
            "final_prompt_path": segroot / "work" / "prompt.txt",
        })
    return result


def _execute_snapshot(snapshot: _Snapshot, runner) -> dict:
    settings = snapshot.settings
    root = snapshot.root
    work = root / "work"
    plan_path = root / long_video.PLAN_RECEIPT_FILENAME
    plan_written = False
    committed = False
    try:
        continuity, prompts = pipeline._generate_segmented_image_prompts(
            settings,
            runner,
            snapshot.segments,
            snapshot.segment_metas,
            work,
            session_dir=root,
            element_index_path=snapshot.element_index,
        )
        if (
            not isinstance(continuity, dict)
            or continuity.get("version") not in {3, 4}
            or not isinstance(prompts, dict)
            or set(prompts) != {item["index"] for item in snapshot.segments}
        ):
            raise ResumeExecutionError("image phase returned incomplete canonical output")

        observed = _collect(settings, snapshot.cid)
        if not _same_manifest(snapshot.manifest, observed.manifest):
            raise ResumeRejected("resume artifacts changed during image generation")

        receipt_segments = _receipt_segments(root, snapshot.segment_metas)
        long_video.write_plan_receipt(
            root,
            source=snapshot.source,
            duration_s=float(snapshot.meta["duration_s"]),
            segments=receipt_segments,
            workflow=pipeline.H3_ENGINE_WORKFLOW,
        )
        plan_written = True
        candidate = {
            **snapshot.meta,
            "segments": snapshot.segment_metas,
            "long_video_plan_receipt": long_video.PLAN_RECEIPT_FILENAME,
        }
        changes = pipeline._recover_long_plan(root, candidate, settings)
        frozen_continuity, frozen_prompts = pipeline._freeze_image_optimization(
            settings,
            {**candidate, **changes},
            continuity,
            prompts,
            {
                item["index"]: [
                    work / "segments" / str(item["index"]) / "work" / "keyframes" / name
                    for name in item["keyframes"]
                ]
                for item in snapshot.segment_metas
            },
            require_dual_target=False,
            segment_lineage={
                item["index"]: {
                    "chain_id": item["chain_id"],
                    "join_mode": item["join_mode"],
                }
                for item in snapshot.segments
            },
            keyframe_sources={
                item["index"]: item["keyframe_sources"]
                for item in snapshot.segment_metas
            },
        )
        changes.update(frozen_continuity)
        changes.update(frozen_prompts)
        changes.update(status="done", error=None)
        updated = storage.update_meta(settings.data_dir, snapshot.cid, **changes)
        if updated is None:
            raise ResumeExecutionError("conversation disappeared before resume commit")
        committed = True
        return updated
    except ResumeExecutionError:
        raise
    except Exception as exc:
        raise ResumeExecutionError(str(exc)) from None
    finally:
        if plan_written and not committed:
            plan_path.unlink(missing_ok=True)


def execute(settings: Settings, cid: str, expected_manifest: dict, *, runner) -> dict:
    """Execute exactly one image-phase continuation after a manifest match."""
    first = _collect(settings, cid)
    if not _same_manifest(expected_manifest, first.manifest):
        raise ResumeRejected("resume manifest does not match current artifacts")
    with _exclusive(first.root):
        snapshot = _collect(settings, cid)
        if not _same_manifest(expected_manifest, snapshot.manifest):
            raise ResumeRejected("resume artifacts changed before execution")
        return _execute_snapshot(snapshot, runner)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receipt-bound image-phase continuation operator")
    parser.add_argument("--data-dir", type=_absolute, required=True)
    parser.add_argument("--cid", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", type=_absolute, required=True)
    parser.add_argument("--seedream-model", default="doubao-seedream-5-0-pro-260628")
    parser.add_argument("--seedream-edit-mode", default="independent_parallel")
    parser.add_argument("--codex-timeout-s", type=int, default=3600)
    parser.add_argument("--codex-concurrency", type=int, default=4)
    parser.add_argument("--diagnostics-dir", type=_absolute)
    return parser


def _publish_manifest(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ResumeRejected("manifest output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        data = _canonical(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except FileExistsError:
        raise ResumeRejected("manifest staging path already exists") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _publish_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ResumeExecutionError("diagnostic output already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except FileExistsError:
        raise ResumeExecutionError("diagnostic staging path already exists") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings(
        access_token="",
        data_dir=args.data_dir,
        seedream_model=args.seedream_model,
        seedream_edit_mode=args.seedream_edit_mode,
        codex_timeout_s=args.codex_timeout_s,
        codex_concurrency=args.codex_concurrency,
        retry_count=0,
        retry_interval_s=0,
    )
    if args.dry_run:
        manifest = inspect(settings, args.cid)
        _publish_manifest(args.manifest, manifest)
        print(json.dumps({"id": args.cid, "manifest": str(args.manifest)}, ensure_ascii=False))
        return 0
    manifest_path = args.manifest
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("manifest is missing or not a regular file")
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"manifest is invalid: {exc}") from exc
    runner = CodexRunner(
        timeout_s=settings.codex_timeout_s, concurrency=settings.codex_concurrency,
    )
    if args.diagnostics_dir is not None:
        runner = _DiagnosticRunner(runner, args.diagnostics_dir)
    result = execute(settings, args.cid, expected, runner=runner)
    print(json.dumps({"id": args.cid, "status": result.get("status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
