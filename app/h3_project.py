"""Project-level freeze and native-audio assembly for H3 multimodal input.

The video Skill owns semantic facts.  This module only binds its approved JSON,
ordered reference bytes, and the existing H3/receipt/stitch lifecycle.  It has
no provider-specific state machine of its own.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from app import (
    context_ir_bridge,
    dialogue_timing,
    error_trace,
    h3,
    h3_multimodal,
    stitch,
)


SOURCE_FILENAME = "h3_multimodal_source.json"
SKILL_INPUT_FILENAME = "multimodal_input.json"
SPEAKER_TIMING_FILENAME = "speaker_timing.json"
SPEAKER_TIMING_PRODUCTION_FILENAME = "speaker_timing_production.json"
FINAL_ACCEPTANCE_FILENAME = "dialogue-av-acceptance.json"
SOURCE_SCHEMA = "duet.h3-multimodal-source"
SOURCE_VERSION = 3
LEGACY_SOURCE_VERSION = 2
SKILL_INPUT_SCHEMA = "duet.h3-multimodal-input"
SKILL_INPUT_VERSION = 2
LEGACY_SKILL_INPUT_VERSION = 1
RECEIPT_SCHEMA = "duet.h3-project-multimodal"
RECEIPT_VERSION = 3
LEGACY_RECEIPT_VERSION = 2
AUDIO_ROUTE = {
    "schema": "duet.h3-project.audio-route",
    "version": 1,
    "mode": "h3_native",
    "reference_audio_role": "conditioning_only",
    "stitch_audio_mode": "provider_generated",
}
CONTEXT_IR_BINDING_SCHEMA = "duet.h3-project.context-ir"
CONTEXT_IR_BINDING_VERSION = 1
_CONTEXT_IR_BINDING_KEYS = {
    "schema", "version", "status", "attempt_id", "provider_task_id",
    "source_prompt_sha256", "effective_prompt_sha256",
    "context_ir_attempt_sha256", "context_ir_request_sha256",
    "context_ir_task_sha256", "receipt_path", "receipt_sha256",
}


class ProjectMultimodalError(RuntimeError):
    """Stable pre-provider project/receipt contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class FrozenProjectMultimodal:
    root: Path
    manifest_path: Path
    manifest_data: bytes
    manifest_sha256: str
    source_version: int
    mode: str
    skill_input_path: Path
    skill_input_data: bytes
    skill_input_sha256: str
    skill_input: Mapping[str, Any]
    skill_plan_path: Path
    skill_plan_data: bytes
    skill_plan_data_sha256: str
    skill_plan: Mapping[str, Any]
    skill_plan_sha256: str
    speaker_timing_path: Path | None
    speaker_timing_data: bytes | None
    speaker_timing_data_sha256: str | None
    speaker_timing: Mapping[str, Any] | None
    speaker_timing_production_path: Path | None
    speaker_timing_production_data: bytes | None
    speaker_timing_production: Mapping[str, Any] | None
    speaker_timing_frame_sha256s: tuple[str, ...] | None
    reference_audios: h3.FrozenReferenceAudios


RequestFactory = Callable[..., h3.H3Request]


def _fail(code: str) -> None:
    raise ProjectMultimodalError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, code: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError:
        _fail(code)
    if not data:
        _fail(code)
    return data


def _inside(root: Path, base: Path, relative: object, code: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or Path(relative).name != relative
    ):
        _fail(code)
    path = (base / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        _fail(code)
    if not path.is_file():
        _fail(code)
    return path


def _relative_file(root: Path, base: Path, relative: object, code: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
    ):
        _fail(code)
    path = (base / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        _fail(code)
    if not path.is_file():
        _fail(code)
    return path


def _json_object(data: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("multimodal_input_invalid")


def _plan_has_on_screen_dialogue(plan: Mapping[str, Any]) -> bool:
    bindings = plan.get("speech_bindings")
    if not isinstance(bindings, list):
        _fail("skill_plan_shape_invalid")
    return any(
        isinstance(item, Mapping) and item.get("delivery") == "on_screen"
        for item in bindings
    )


def _timing_binding(
    *, root: Path, workdir: Path, manifest: Mapping[str, Any],
) -> tuple[Path | None, bytes | None, str | None, Mapping[str, Any] | None]:
    binding = manifest.get("speaker_timing")
    if binding is None:
        return None, None, None, None
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        _fail("speaker_timing_binding_invalid")
    path = _inside(
        root, workdir, binding.get("path"), "speaker_timing_binding_invalid"
    )
    data = _read(path, "speaker_timing_binding_invalid")
    digest = _sha256(data)
    if path.name != SPEAKER_TIMING_FILENAME or digest != binding.get("sha256"):
        _fail("speaker_timing_binding_invalid")
    return (
        path,
        data,
        digest,
        _json_object(data, "speaker_timing_binding_invalid"),
    )


def _speaker_timing_production_binding(
    *,
    root: Path,
    workdir: Path,
    manifest: Mapping[str, Any],
    timing: Mapping[str, Any] | None,
) -> tuple[Path | None, bytes | None, Mapping[str, Any] | None, tuple[str, ...] | None]:
    binding = manifest.get("speaker_timing_producer")
    if binding is None:
        return None, None, None, None
    if (
        timing is None
        or not isinstance(binding, dict)
        or set(binding) != {"path", "sha256"}
    ):
        _fail("speaker_timing_production_binding_invalid")
    receipt_path = _inside(
        root, workdir, binding.get("path"),
        "speaker_timing_production_binding_invalid",
    )
    receipt_data = _read(
        receipt_path, "speaker_timing_production_binding_invalid"
    )
    if (
        receipt_path.name != SPEAKER_TIMING_PRODUCTION_FILENAME
        or _sha256(receipt_data) != binding.get("sha256")
    ):
        _fail("speaker_timing_production_binding_invalid")
    receipt = _json_object(receipt_data, "speaker_timing_production_invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "producer_input", "raw_output", "skill", "speaker_timing",
    }:
        _fail("speaker_timing_production_invalid")

    expected_names = {
        "producer_input": "speaker_visibility_input.json",
        "raw_output": "speaker_visibility_output.json",
        "skill": "speaker_visibility_skill.md",
        "speaker_timing": SPEAKER_TIMING_FILENAME,
    }
    loaded: dict[str, tuple[Path, bytes]] = {}
    for role, expected_name in expected_names.items():
        artifact = artifacts.get(role)
        expected_keys = (
            {"path", "sha256", "canonical_sha256"}
            if role == "speaker_timing"
            else {"path", "sha256"}
        )
        if not isinstance(artifact, dict) or set(artifact) != expected_keys:
            _fail("speaker_timing_production_invalid")
        path = _inside(
            root, workdir, artifact.get("path"),
            "speaker_timing_production_invalid",
        )
        data = _read(path, "speaker_timing_production_invalid")
        if path.name != expected_name or _sha256(data) != artifact.get("sha256"):
            if role == "raw_output":
                _fail("speaker_visibility_output_hash_mismatch")
            _fail(f"speaker_timing_{role}_hash_mismatch")
        loaded[role] = (path, data)

    producer_input = _json_object(
        loaded["producer_input"][1], "speaker_visibility_input_invalid"
    )
    raw_frames = producer_input.get("frames")
    raw_sheets = producer_input.get("contact_sheets")
    raw_persons = producer_input.get("persons")
    if (
        not isinstance(raw_frames, list)
        or not isinstance(raw_sheets, list)
        or not isinstance(raw_persons, list)
    ):
        _fail("speaker_visibility_input_invalid")
    frame_data: dict[str, bytes] = {}
    raw_identity_refs = []
    for person in raw_persons:
        if not isinstance(person, dict) or not isinstance(
            person.get("identity_refs"), list
        ):
            _fail("speaker_visibility_input_invalid")
        raw_identity_refs.extend(person["identity_refs"])
    raw_cut_source = producer_input.get("cut_source")
    if not isinstance(raw_cut_source, dict):
        _fail("speaker_visibility_input_invalid")
    for item in [*raw_frames, *raw_sheets, *raw_identity_refs, raw_cut_source]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            _fail("speaker_visibility_input_invalid")
        path = _relative_file(
            root, workdir, item["path"], "speaker_visibility_input_invalid"
        )
        frame_data[item["path"]] = _read(
            path, "speaker_visibility_input_invalid"
        )
    if workdir.resolve() == (root / "work").resolve():
        sources = sorted(
            path for path in root.glob("source.*") if path.is_file()
        )
    else:
        segment_root = workdir.resolve().parent
        try:
            relative_segment = segment_root.relative_to(
                (root / "work" / "segments").resolve()
            )
        except ValueError:
            _fail("speaker_visibility_source_mismatch")
        source = segment_root / "source.mp4"
        sources = (
            [source]
            if len(relative_segment.parts) == 1
            and relative_segment.parts[0].isdigit()
            and source.is_file()
            else []
        )
    if len(sources) != 1:
        _fail("speaker_visibility_source_mismatch")
    try:
        production = dialogue_timing.freeze_speaker_visibility(
            producer_input_data=loaded["producer_input"][1],
            skill_output_data=loaded["raw_output"][1],
            source_data=_read(sources[0], "speaker_visibility_source_mismatch"),
            frame_data=frame_data,
            skill_data=loaded["skill"][1],
        )
    except dialogue_timing.DialogueTimingError as exc:
        raise ProjectMultimodalError(exc.code) from None
    if production.receipt != receipt or production.speaker_timing != timing:
        _fail("speaker_timing_production_mismatch")
    timeline = timing.get("timeline")
    timeline_frames = timeline.get("keyframes") if isinstance(timeline, dict) else None
    if not isinstance(timeline_frames, list):
        _fail("speaker_timing_production_mismatch")
    return (
        receipt_path,
        receipt_data,
        receipt,
        tuple(str(item.get("sha256")) for item in timeline_frames),
    )


def refresh_skill_input(
    *,
    root: Path,
    workdir: Path,
    visual_prompt_path: Path,
    keyframes: tuple[Path, ...],
    dialogue_source_sha256: str,
) -> bool:
    """Freeze the current authoritative inputs before the external Skill runs.

    Returns true only when the input snapshot changed.  The caller must stop
    the current paid submit and wait for the Skill plan/source manifest to be
    regenerated against the new file.
    """
    root = Path(root).resolve()
    workdir = Path(workdir).resolve()
    manifest_path = workdir / SOURCE_FILENAME
    if not manifest_path.is_file():
        return False
    manifest = _json_object(
        _read(manifest_path, "multimodal_source_invalid"),
        "multimodal_source_invalid",
    )
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("version") not in {
            LEGACY_SOURCE_VERSION, SOURCE_VERSION,
        }
        or manifest.get("mode") != "multimodal"
    ):
        _fail("multimodal_source_invalid")
    plan_binding = manifest.get("skill_plan")
    if not isinstance(plan_binding, dict) or set(plan_binding) != {"path", "sha256"}:
        _fail("skill_plan_binding_invalid")
    plan_path = _inside(
        root, workdir, plan_binding.get("path"), "skill_plan_binding_invalid"
    )
    plan_data = _read(plan_path, "skill_plan_binding_invalid")
    if _sha256(plan_data) != plan_binding.get("sha256"):
        _fail("skill_plan_hash_mismatch")
    plan = _json_object(plan_data, "skill_plan_invalid")
    on_screen_required = _plan_has_on_screen_dialogue(plan)
    if on_screen_required and manifest.get("version") == LEGACY_SOURCE_VERSION:
        _fail("speaker_timing_refresh_required")
    timing_binding = manifest.get("speaker_timing")
    if on_screen_required and timing_binding is None:
        _fail("speaker_timing_refresh_required")
    if timing_binding is not None:
        _path, _data, _digest, timing = _timing_binding(
            root=root, workdir=workdir, manifest=manifest
        )
        _speaker_timing_production_binding(
            root=root, workdir=workdir, manifest=manifest, timing=timing
        )
    raw_audios = manifest.get("reference_audios")
    if not isinstance(raw_audios, list) or not 0 <= len(raw_audios) <= 3:
        _fail("reference_audio_binding_invalid")
    for expected_order, raw in enumerate(raw_audios, 1):
        if (
            not isinstance(raw, dict)
            or set(raw) != {"order", "path", "sha256", "purpose"}
            or raw.get("order") != expected_order
            or raw.get("purpose") not in {"voice", "ambience", "effect"}
        ):
            _fail("reference_audio_binding_invalid")
        audio_path = _inside(
            root, workdir, raw.get("path"), "reference_audio_binding_invalid"
        )
        if _sha256(_read(audio_path, "reference_audio_binding_invalid")) != raw.get(
            "sha256"
        ):
            _fail("reference_audio_hash_mismatch")
    try:
        visual_relative = visual_prompt_path.resolve().relative_to(workdir).as_posix()
        frozen_keyframes = [
            (
                path.resolve().relative_to(workdir).as_posix(),
                _sha256(_read(path.resolve(), "multimodal_input_invalid")),
            )
            for path in keyframes
        ]
    except ValueError:
        _fail("multimodal_input_invalid")
    visual_data = _read(visual_prompt_path.resolve(), "multimodal_input_invalid")
    desired = {
        "schema": SKILL_INPUT_SCHEMA,
        "version": (
            SKILL_INPUT_VERSION
            if manifest.get("version") == SOURCE_VERSION
            else LEGACY_SKILL_INPUT_VERSION
        ),
        "visual_prompt": {
            "path": visual_relative,
            "sha256": _sha256(visual_data),
        },
        "keyframes": [
            {"order": order, "path": path, "sha256": sha256}
            for order, (path, sha256) in enumerate(frozen_keyframes, 1)
        ],
        "dialogue_source_sha256": dialogue_source_sha256,
        "reference_audios": raw_audios,
    }
    if timing_binding is not None:
        desired["speaker_timing"] = timing_binding
    input_path = workdir / SKILL_INPUT_FILENAME
    if input_path.is_file():
        try:
            if _json_object(input_path.read_bytes(), "multimodal_input_invalid") == desired:
                return False
        except OSError:
            _fail("multimodal_input_invalid")
    temporary = input_path.with_name(input_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(desired, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(input_path)
    except OSError:
        _fail("multimodal_input_write_failed")
    return True


def freeze_optional(
    root: Path,
    workdir: Path,
    *,
    allow_legacy_on_screen_read: bool = False,
) -> FrozenProjectMultimodal | None:
    """Freeze an opt-in project source; partial intent fails closed.

    Absence of both source manifest and Skill output is the unchanged legacy
    path.  Presence of either makes the multimodal contract mandatory.  The
    legacy exception only reloads an already-bound v2 receipt; request
    construction still requires refresh before any new on-screen paid work.
    """
    root = Path(root).resolve()
    workdir = Path(workdir).resolve()
    try:
        workdir.relative_to(root)
    except ValueError:
        _fail("multimodal_source_invalid")
    manifest_path = workdir / SOURCE_FILENAME
    default_plan = workdir / "h3_prompt_plan.json"
    skill_input = workdir / SKILL_INPUT_FILENAME
    if not any(
        path.exists() for path in (manifest_path, default_plan, skill_input)
    ):
        return None
    if not manifest_path.is_file():
        _fail("multimodal_source_missing")
    manifest_data = _read(manifest_path, "multimodal_source_invalid")
    manifest = _json_object(manifest_data, "multimodal_source_invalid")
    source_version = manifest.get("version")
    base_manifest_keys = {
        "schema",
        "version",
        "mode",
        "approved_skill_plan_sha256",
        "multimodal_input",
        "skill_plan",
        "reference_audios",
    }
    optional_manifest_keys = {"speaker_timing", "speaker_timing_producer"}
    manifest_keys = set(manifest)
    manifest_extras = manifest_keys - base_manifest_keys
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or source_version not in {LEGACY_SOURCE_VERSION, SOURCE_VERSION}
        or manifest.get("mode") != "multimodal"
        or not base_manifest_keys.issubset(manifest_keys)
        or not manifest_extras.issubset(optional_manifest_keys)
        or (
            "speaker_timing_producer" in manifest
            and "speaker_timing" not in manifest
        )
        or (
            source_version == LEGACY_SOURCE_VERSION
            and "speaker_timing" in manifest
        )
    ):
        _fail("multimodal_source_invalid")

    input_binding = manifest.get("multimodal_input")
    if not isinstance(input_binding, dict) or set(input_binding) != {"path", "sha256"}:
        _fail("multimodal_input_binding_invalid")
    skill_input_path = _inside(
        root,
        workdir,
        input_binding.get("path"),
        "multimodal_input_binding_invalid",
    )
    if skill_input_path.name != SKILL_INPUT_FILENAME:
        _fail("multimodal_input_binding_invalid")
    skill_input_data = _read(
        skill_input_path, "multimodal_input_binding_invalid"
    )
    skill_input_sha256 = _sha256(skill_input_data)
    if input_binding.get("sha256") != skill_input_sha256:
        _fail("multimodal_input_hash_mismatch")
    skill_input = _json_object(skill_input_data, "multimodal_input_invalid")
    expected_input_version = (
        SKILL_INPUT_VERSION
        if source_version == SOURCE_VERSION
        else LEGACY_SKILL_INPUT_VERSION
    )
    base_input_keys = {
        "schema",
        "version",
        "visual_prompt",
        "keyframes",
        "dialogue_source_sha256",
        "reference_audios",
    }
    if set(skill_input) not in {
        frozenset(base_input_keys),
        frozenset(base_input_keys | {"speaker_timing"}),
    } or (
        skill_input.get("schema") != SKILL_INPUT_SCHEMA
        or skill_input.get("version") != expected_input_version
        or (
            expected_input_version == LEGACY_SKILL_INPUT_VERSION
            and "speaker_timing" in skill_input
        )
    ):
        _fail("multimodal_input_invalid")

    plan_binding = manifest.get("skill_plan")
    if not isinstance(plan_binding, dict) or set(plan_binding) != {"path", "sha256"}:
        _fail("skill_plan_binding_invalid")
    skill_plan_path = _inside(
        root,
        workdir,
        plan_binding.get("path"),
        "skill_plan_binding_invalid",
    )
    skill_plan_data = _read(skill_plan_path, "skill_plan_binding_invalid")
    skill_plan_data_sha256 = _sha256(skill_plan_data)
    if plan_binding.get("sha256") != skill_plan_data_sha256:
        _fail("skill_plan_hash_mismatch")
    skill_plan = _json_object(skill_plan_data, "skill_plan_invalid")
    skill_plan_sha256 = h3.canonical_json_sha256(skill_plan)
    if manifest.get("approved_skill_plan_sha256") != skill_plan_sha256:
        _fail("skill_plan_approval_mismatch")
    if skill_plan.get("dialogue_source_sha256") != skill_input.get(
        "dialogue_source_sha256"
    ):
        _fail("multimodal_input_dialogue_mismatch")

    timing_binding = manifest.get("speaker_timing")
    on_screen_required = _plan_has_on_screen_dialogue(skill_plan)
    if (
        on_screen_required
        and source_version == LEGACY_SOURCE_VERSION
        and not allow_legacy_on_screen_read
    ):
        _fail("speaker_timing_refresh_required")
    if (
        on_screen_required
        and timing_binding is None
        and not (
            source_version == LEGACY_SOURCE_VERSION
            and allow_legacy_on_screen_read
        )
    ):
        _fail("speaker_timing_refresh_required")
    if skill_input.get("speaker_timing") != timing_binding:
        _fail("speaker_timing_binding_invalid")
    (
        speaker_timing_path,
        speaker_timing_data,
        speaker_timing_data_sha256,
        speaker_timing,
    ) = _timing_binding(
        root=root, workdir=workdir, manifest=manifest
    )
    (
        speaker_timing_production_path,
        speaker_timing_production_data,
        speaker_timing_production,
        speaker_timing_frame_sha256s,
    ) = _speaker_timing_production_binding(
        root=root,
        workdir=workdir,
        manifest=manifest,
        timing=speaker_timing,
    )

    visual_binding = skill_input.get("visual_prompt")
    if not isinstance(visual_binding, dict) or set(visual_binding) != {
        "path", "sha256"
    }:
        _fail("multimodal_input_invalid")
    visual_path = _relative_file(
        root, workdir, visual_binding.get("path"), "multimodal_input_invalid"
    )
    if _sha256(_read(visual_path, "multimodal_input_invalid")) != visual_binding.get(
        "sha256"
    ):
        _fail("multimodal_input_visual_mismatch")
    raw_keyframes = skill_input.get("keyframes")
    if not isinstance(raw_keyframes, list) or not 1 <= len(raw_keyframes) <= 9:
        _fail("multimodal_input_invalid")
    for expected_order, raw in enumerate(raw_keyframes, 1):
        if not isinstance(raw, dict) or set(raw) != {"order", "path", "sha256"}:
            _fail("multimodal_input_invalid")
        if raw.get("order") != expected_order:
            _fail("multimodal_input_keyframe_mismatch")
        keyframe_path = _relative_file(
            root, workdir, raw.get("path"), "multimodal_input_invalid"
        )
        if _sha256(_read(keyframe_path, "multimodal_input_invalid")) != raw.get(
            "sha256"
        ):
            _fail("multimodal_input_keyframe_mismatch")

    raw_audios = manifest.get("reference_audios")
    if not isinstance(raw_audios, list) or not 1 <= len(raw_audios) <= 3:
        _fail("reference_audio_binding_invalid")
    if skill_input.get("reference_audios") != raw_audios:
        _fail("multimodal_input_audio_mismatch")
    sources: list[tuple[Path, h3.ReferenceAudioPurpose]] = []
    expected_hashes: list[str] = []
    for expected_order, raw in enumerate(raw_audios, 1):
        if not isinstance(raw, dict) or set(raw) != {
            "order", "path", "sha256", "purpose"
        }:
            _fail("reference_audio_binding_invalid")
        if raw.get("order") != expected_order:
            _fail("reference_audio_order_invalid")
        purpose = raw.get("purpose")
        if purpose not in {"voice", "ambience", "effect"}:
            _fail("reference_audio_binding_invalid")
        path = _inside(
            root,
            workdir,
            raw.get("path"),
            "reference_audio_binding_invalid",
        )
        expected_sha = raw.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            _fail("reference_audio_binding_invalid")
        sources.append((path, purpose))
        expected_hashes.append(expected_sha)
    try:
        audios = (
            h3.freeze_reference_audios(tuple(sources)) if sources else ()
        )
    except h3.H3Error as exc:
        raise ProjectMultimodalError(exc.code) from None
    if [audio.sha256 for audio in audios] != expected_hashes:
        _fail("reference_audio_hash_mismatch")

    return FrozenProjectMultimodal(
        root=root,
        manifest_path=manifest_path,
        manifest_data=manifest_data,
        manifest_sha256=_sha256(manifest_data),
        source_version=int(source_version),
        mode=manifest["mode"],
        skill_input_path=skill_input_path,
        skill_input_data=skill_input_data,
        skill_input_sha256=skill_input_sha256,
        skill_input=skill_input,
        skill_plan_path=skill_plan_path,
        skill_plan_data=skill_plan_data,
        skill_plan_data_sha256=skill_plan_data_sha256,
        skill_plan=skill_plan,
        skill_plan_sha256=skill_plan_sha256,
        speaker_timing_path=speaker_timing_path,
        speaker_timing_data=speaker_timing_data,
        speaker_timing_data_sha256=speaker_timing_data_sha256,
        speaker_timing=speaker_timing,
        speaker_timing_production_path=speaker_timing_production_path,
        speaker_timing_production_data=speaker_timing_production_data,
        speaker_timing_production=speaker_timing_production,
        speaker_timing_frame_sha256s=speaker_timing_frame_sha256s,
        reference_audios=audios,
    )


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        _fail("multimodal_receipt_invalid")


def receipt_binding(
    root: Path, frozen: FrozenProjectMultimodal,
) -> dict[str, Any]:
    """Return the canonical project receipt section for one frozen snapshot."""
    root = Path(root).resolve()
    if not isinstance(frozen, FrozenProjectMultimodal) or frozen.root != root:
        _fail("multimodal_receipt_invalid")
    binding = {
        "schema": RECEIPT_SCHEMA,
        "version": (
            RECEIPT_VERSION
            if frozen.source_version == SOURCE_VERSION
            else LEGACY_RECEIPT_VERSION
        ),
        "mode": frozen.mode,
        "manifest": {
            "path": _relative(root, frozen.manifest_path),
            "sha256": frozen.manifest_sha256,
        },
        "multimodal_input": {
            "path": _relative(root, frozen.skill_input_path),
            "sha256": frozen.skill_input_sha256,
        },
        "skill_plan": {
            "path": _relative(root, frozen.skill_plan_path),
            "sha256": frozen.skill_plan_data_sha256,
            "canonical_sha256": frozen.skill_plan_sha256,
        },
        "reference_audios": [
            {
                "order": audio.order,
                "path": _relative(root, audio.path),
                "sha256": audio.sha256,
                "purpose": audio.purpose,
                "format": audio.format,
                "duration_s": audio.duration_s,
                "size": len(audio.data),
            }
            for audio in frozen.reference_audios
        ],
    }
    if frozen.speaker_timing_path is not None:
        binding["speaker_timing"] = {
            "path": _relative(root, frozen.speaker_timing_path),
            "sha256": frozen.speaker_timing_data_sha256,
            "canonical_sha256": dialogue_timing.canonical_sha256(
                frozen.speaker_timing
            ),
        }
    if frozen.speaker_timing_production_path is not None:
        raw_output = frozen.speaker_timing_production["artifacts"]["raw_output"]
        binding["speaker_timing_producer"] = {
            "path": _relative(root, frozen.speaker_timing_production_path),
            "sha256": _sha256(frozen.speaker_timing_production_data),
            "raw_output_path": raw_output["path"],
            "raw_output_sha256": raw_output["sha256"],
        }
    return binding


def load_bound(root: Path, binding: object) -> FrozenProjectMultimodal:
    """Reload exact bound files and reject any path, bytes, hash, or order drift."""
    root = Path(root).resolve()
    base_keys = {
        "schema", "version", "mode", "manifest", "multimodal_input",
        "skill_plan", "reference_audios"
    }
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != RECEIPT_SCHEMA
        or binding.get("version") not in {
            LEGACY_RECEIPT_VERSION, RECEIPT_VERSION,
        }
        or set(binding) not in {
            frozenset(base_keys),
            frozenset(base_keys | {"speaker_timing"}),
            frozenset(base_keys | {
                "speaker_timing", "speaker_timing_producer",
            }),
        }
        or (
            binding.get("version") == LEGACY_RECEIPT_VERSION
            and "speaker_timing" in binding
        )
    ):
        _fail("multimodal_receipt_invalid")
    manifest = binding.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"path", "sha256"}:
        _fail("multimodal_receipt_invalid")
    relative = manifest.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        _fail("multimodal_receipt_invalid")
    manifest_path = (root / relative).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError:
        _fail("multimodal_receipt_invalid")
    if manifest_path.name != SOURCE_FILENAME:
        _fail("multimodal_receipt_invalid")
    frozen = freeze_optional(
        root,
        manifest_path.parent,
        allow_legacy_on_screen_read=True,
    )
    if frozen is None or receipt_binding(root, frozen) != binding:
        _fail("multimodal_receipt_mismatch")
    return frozen


def revalidate_production_authority(
    root: Path,
    workdir: Path,
    request: h3.H3Request,
    *,
    expected_production_sha256: str | None,
) -> None:
    """Reload sampled speaker evidence immediately before an H3 boundary."""
    if not request.on_screen_dialogue:
        return
    if request.speaker_timing_authority_version == 0:
        if expected_production_sha256 is not None:
            _fail("speaker_timing_production_authority_invalid")
        return
    if (
        request.speaker_timing_authority_version != 1
        or expected_production_sha256 is None
        or request.speaker_timing_production_sha256
        != expected_production_sha256
    ):
        _fail("speaker_timing_production_authority_invalid")
    frozen = freeze_optional(root, workdir)
    if (
        frozen is None
        or frozen.speaker_timing_production is None
        or frozen.speaker_timing_production_data is None
        or _sha256(frozen.speaker_timing_production_data)
        != expected_production_sha256
        or frozen.speaker_timing is None
        or dialogue_timing.canonical_sha256(frozen.speaker_timing)
        != request.speaker_timing_sha256
    ):
        _fail("speaker_timing_refresh_required")


def build_request(
    *,
    frozen: Any,
    cid: str,
    workdir: Path,
    client_request_id: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    autodl_token: str,
    timeouts: h3.Timeouts = h3.Timeouts(),
    seed: int | None = None,
    request_factory: RequestFactory = h3_multimodal.build_h3_request,
) -> h3.H3Request:
    """Project adapter's sole Context-IR/H3 request injection point.

    ``request_factory`` is deliberately the only seam for the separate
    Context-IR bridge.  The coordinator does not maintain or guess semantic
    state; the factory must consume the frozen Skill plan and return H3Request.
    """
    multimodal = getattr(frozen, "multimodal", None)
    visual_prompt = getattr(frozen, "visual_prompt", None)
    keyframes = getattr(frozen, "frozen_keyframes", None)
    if (
        not isinstance(multimodal, FrozenProjectMultimodal)
        or visual_prompt is None
        or not isinstance(getattr(visual_prompt, "data", None), bytes)
        or not isinstance(keyframes, tuple)
    ):
        _fail("multimodal_prepared_input_invalid")
    try:
        prompt = visual_prompt.data.decode("utf-8")
    except UnicodeDecodeError:
        _fail("multimodal_prepared_input_invalid")
    return build_request_from_parts(
        multimodal=multimodal,
        visual_prompt=prompt,
        keyframes=keyframes,
        upstream_dialogue=frozen.dialogue,
        upstream_dialogue_receipt_sha256=frozen.dialogue_sha256,
        source_sha256=frozen.source.sha256,
        source_duration_s=frozen.duration_s,
        cid=cid,
        workdir=workdir,
        client_request_id=client_request_id,
        duration=duration,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        autodl_token=autodl_token,
        timeouts=timeouts,
        seed=seed,
        request_factory=request_factory,
    )


def _speaker_timing_request_authority(
    multimodal: FrozenProjectMultimodal,
) -> dict[str, Any]:
    if multimodal.speaker_timing_production_path is None:
        if multimodal.source_version != LEGACY_SOURCE_VERSION:
            _fail("speaker_timing_refresh_required")
        return {
            "speaker_timing_authority_version": 0,
            "speaker_timing_production_required": False,
            "speaker_timing_legacy_source_version": LEGACY_SOURCE_VERSION,
            "speaker_timing_legacy_receipt_path": _relative(
                multimodal.root, multimodal.manifest_path
            ),
            "speaker_timing_legacy_receipt_sha256": (
                multimodal.manifest_sha256
            ),
            "speaker_timing_production_path": None,
            "speaker_timing_production_sha256": None,
            "speaker_timing_authority_artifacts": (),
            "speaker_timing_authority_root": None,
        }
    receipt = multimodal.speaker_timing_production
    receipt_data = multimodal.speaker_timing_production_data
    if not isinstance(receipt, Mapping) or not isinstance(receipt_data, bytes):
        _fail("speaker_timing_production_authority_invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        _fail("speaker_timing_production_authority_invalid")
    root = multimodal.root
    work = multimodal.speaker_timing_production_path.parent
    frozen: dict[str, str] = {
        _relative(root, multimodal.speaker_timing_production_path): _sha256(
            receipt_data
        ),
    }
    role_paths: dict[str, Path] = {}
    for role in ("producer_input", "raw_output", "skill", "speaker_timing"):
        artifact = artifacts.get(role)
        if not isinstance(artifact, Mapping):
            _fail("speaker_timing_production_authority_invalid")
        path = _relative_file(
            root, work, artifact.get("path"),
            "speaker_timing_production_authority_invalid",
        )
        data = _read(path, "speaker_timing_production_authority_invalid")
        digest = artifact.get("sha256")
        if digest != _sha256(data):
            _fail("speaker_timing_production_authority_invalid")
        relative = _relative(root, path)
        if relative in frozen and frozen[relative] != digest:
            _fail("speaker_timing_production_authority_invalid")
        frozen[relative] = str(digest)
        role_paths[role] = path
    producer_input = _json_object(
        _read(
            role_paths["producer_input"],
            "speaker_timing_production_authority_invalid",
        ),
        "speaker_timing_production_authority_invalid",
    )
    source = producer_input.get("source")
    source_path = (role_paths["producer_input"].parent.parent / "source.mp4").resolve()
    try:
        source_path.relative_to(root)
    except ValueError:
        _fail("speaker_timing_production_authority_invalid")
    source_data = _read(source_path, "speaker_timing_production_authority_invalid")
    source_digest = source.get("sha256") if isinstance(source, Mapping) else None
    if source_digest != _sha256(source_data):
        _fail("speaker_timing_production_authority_invalid")
    frozen[_relative(root, source_path)] = str(source_digest)
    evidence: list[object] = []
    for key in ("frames", "contact_sheets"):
        values = producer_input.get(key)
        if not isinstance(values, list):
            _fail("speaker_timing_production_authority_invalid")
        evidence.extend(values)
    persons = producer_input.get("persons")
    if not isinstance(persons, list):
        _fail("speaker_timing_production_authority_invalid")
    for person in persons:
        refs = person.get("identity_refs") if isinstance(person, Mapping) else None
        if not isinstance(refs, list):
            _fail("speaker_timing_production_authority_invalid")
        evidence.extend(refs)
    cut_source = producer_input.get("cut_source")
    if not isinstance(cut_source, Mapping):
        _fail("speaker_timing_production_authority_invalid")
    evidence.append(cut_source)
    for artifact in evidence:
        if not isinstance(artifact, Mapping):
            _fail("speaker_timing_production_authority_invalid")
        path = _relative_file(
            root, role_paths["producer_input"].parent,
            artifact.get("path"),
            "speaker_timing_production_authority_invalid",
        )
        data = _read(path, "speaker_timing_production_authority_invalid")
        digest = artifact.get("sha256")
        if digest != _sha256(data):
            _fail("speaker_timing_production_authority_invalid")
        relative = _relative(root, path)
        if relative in frozen and frozen[relative] != digest:
            _fail("speaker_timing_production_authority_invalid")
        frozen[relative] = str(digest)
    production_relative = _relative(
        root, multimodal.speaker_timing_production_path
    )
    return {
        "speaker_timing_authority_version": 1,
        "speaker_timing_production_required": True,
        "speaker_timing_legacy_source_version": None,
        "speaker_timing_legacy_receipt_path": None,
        "speaker_timing_legacy_receipt_sha256": None,
        "speaker_timing_production_path": production_relative,
        "speaker_timing_production_sha256": frozen[production_relative],
        "speaker_timing_authority_artifacts": tuple(sorted(frozen.items())),
        "speaker_timing_authority_root": None,
    }


def build_request_from_parts(
    *,
    multimodal: FrozenProjectMultimodal,
    visual_prompt: str,
    keyframes: h3.FrozenKeyframes,
    upstream_dialogue: tuple[dict, ...],
    upstream_dialogue_receipt_sha256: str,
    source_sha256: str,
    source_duration_s: float,
    cid: str,
    workdir: Path,
    client_request_id: str,
    duration: int,
    resolution: str,
    aspect_ratio: str,
    autodl_token: str,
    timeouts: h3.Timeouts = h3.Timeouts(),
    seed: int | None = None,
    request_factory: RequestFactory = h3_multimodal.build_h3_request,
) -> h3.H3Request:
    """Build short or segmented requests through the same semantic seam."""
    if not isinstance(multimodal, FrozenProjectMultimodal):
        _fail("multimodal_prepared_input_invalid")
    skill_input = multimodal.skill_input
    visual_binding = skill_input.get("visual_prompt")
    input_keyframes = skill_input.get("keyframes")
    if (
        not isinstance(visual_binding, Mapping)
        or visual_binding.get("sha256")
        != _sha256(visual_prompt.encode("utf-8"))
        or skill_input.get("dialogue_source_sha256")
        != upstream_dialogue_receipt_sha256
        or not isinstance(input_keyframes, list)
        or len(input_keyframes) != len(keyframes)
    ):
        _fail("multimodal_input_runtime_mismatch")
    input_base = multimodal.skill_input_path.parent
    for expected_order, (binding, frozen_frame) in enumerate(
        zip(input_keyframes, keyframes, strict=True), 1
    ):
        if not isinstance(binding, Mapping):
            _fail("multimodal_input_runtime_mismatch")
        source_path, source_data = frozen_frame
        expected_path = (input_base / str(binding.get("path"))).resolve()
        if (
            binding.get("order") != expected_order
            or source_path.resolve() != expected_path
            or binding.get("sha256") != _sha256(source_data)
        ):
            _fail("multimodal_input_runtime_mismatch")
    visual = h3_multimodal.FrozenVisualInput(
        prompt=visual_prompt,
        keyframes=keyframes,
    )
    speaker_timing = None
    if _plan_has_on_screen_dialogue(multimodal.skill_plan):
        if multimodal.speaker_timing is None:
            _fail("speaker_timing_refresh_required")
        try:
            speaker_timing = dialogue_timing.freeze_speaker_timing(
                multimodal.speaker_timing,
                source_sha256=source_sha256,
                keyframe_sha256s=tuple(
                    multimodal.speaker_timing_frame_sha256s
                    or tuple(_sha256(data) for _path, data in keyframes)
                ),
                source_duration_s=source_duration_s,
            )
        except dialogue_timing.DialogueTimingError as exc:
            raise ProjectMultimodalError(exc.code) from None
    try:
        request = request_factory(
            skill_plan=multimodal.skill_plan,
            approved_skill_plan_sha256=multimodal.skill_plan_sha256,
            upstream_dialogue=upstream_dialogue,
            upstream_dialogue_receipt_sha256=(
                upstream_dialogue_receipt_sha256
            ),
            upstream_dialogue_content_sha256=h3.canonical_json_sha256(
                list(upstream_dialogue)
            ),
            speaker_timing=speaker_timing,
            visual=visual,
            reference_audios=multimodal.reference_audios,
            mode=multimodal.mode,
            cid=cid,
            workdir=Path(workdir),
            client_request_id=client_request_id,
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            autodl_token=autodl_token,
            timeouts=timeouts,
            seed=seed,
        )
    except h3.H3Error:
        _fail("context_ir_request_authority_mismatch")
    except h3_multimodal.MultimodalContractError as exc:
        raise ProjectMultimodalError(exc.code) from None
    if not isinstance(request, h3.H3Request) or not h3.is_multimodal_request(request):
        _fail("context_ir_request_invalid")
    try:
        h3_multimodal.validate_h3_request_authority(
            request,
            skill_plan=multimodal.skill_plan,
            approved_skill_plan_sha256=multimodal.skill_plan_sha256,
            upstream_dialogue=upstream_dialogue,
            upstream_dialogue_receipt_sha256=(
                upstream_dialogue_receipt_sha256
            ),
            visual=visual,
            reference_audios=multimodal.reference_audios,
            speaker_timing=speaker_timing,
        )
    except (h3.H3Error, h3_multimodal.MultimodalContractError) as exc:
        raise ProjectMultimodalError(
            getattr(exc, "code", "context_ir_request_authority_mismatch")
        ) from None
    if request.on_screen_dialogue:
        try:
            request = replace(
                request, **_speaker_timing_request_authority(multimodal)
            )
            h3.validate_request_authority(request)
        except h3.H3Error as exc:
            raise ProjectMultimodalError(exc.code) from None
    return request


def freeze_context_ir(
    *,
    source_request: h3.H3Request,
    upstream_dialogue_sha256: str,
    upstream_artifact_path: Path,
    upstream_artifact_sha256: str,
    upstream_dialogue_sha256_path: tuple[str | int, ...],
    minimax_api_key: str,
    request_timeout_s: float,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> context_ir_bridge.FrozenContextIrRequest:
    """Bind an authoritative project receipt to one deterministic H3 prompt."""
    try:
        return context_ir_bridge.freeze_context_ir_request(
            source_h3_request=source_request,
            upstream_dialogue_sha256=upstream_dialogue_sha256,
            upstream_artifact_path=upstream_artifact_path,
            upstream_artifact_sha256=upstream_artifact_sha256,
            upstream_dialogue_sha256_path=upstream_dialogue_sha256_path,
            source_prompt_sha256=_sha256(source_request.prompt.encode("utf-8")),
            minimax_api_key=minimax_api_key,
            timeouts=context_ir_bridge.ContextIrTimeouts(
                request_s=request_timeout_s,
                poll_total_s=poll_timeout_s,
                poll_interval_s=poll_interval_s,
            ),
        )
    except context_ir_bridge.ContextIrError as exc:
        raise ProjectMultimodalError(exc.code) from None
def context_ir_binding(
    result: context_ir_bridge.ContextIrResult,
) -> dict[str, Any]:
    if not isinstance(result, context_ir_bridge.ContextIrResult):
        _fail("context_ir_result_invalid")
    receipt_path = result.receipt_path
    return {
        "schema": CONTEXT_IR_BINDING_SCHEMA,
        "version": CONTEXT_IR_BINDING_VERSION,
        "status": result.status,
        "attempt_id": result.attempt_id,
        "provider_task_id": result.provider_task_id,
        "source_prompt_sha256": result.source_prompt_sha256,
        "effective_prompt_sha256": result.effective_prompt_sha256,
        "context_ir_attempt_sha256": result.context_ir_attempt_sha256,
        "context_ir_request_sha256": result.context_ir_request_sha256,
        "context_ir_task_sha256": result.context_ir_task_sha256,
        "receipt_path": str(receipt_path.resolve()) if receipt_path is not None else None,
        "receipt_sha256": result.receipt_sha256,
    }


def apply_bound_context_ir(
    frozen: context_ir_bridge.FrozenContextIrRequest,
    binding: object,
) -> h3.H3Request:
    """Pure read/verify seam; no Context IR network request is possible here."""
    if not isinstance(binding, Mapping) or set(binding) != _CONTEXT_IR_BINDING_KEYS:
        _fail("context_ir_binding_invalid")
    if (
        binding.get("schema") != CONTEXT_IR_BINDING_SCHEMA
        or binding.get("version") != CONTEXT_IR_BINDING_VERSION
        or binding.get("status") != "succeeded"
        or not isinstance(binding.get("receipt_path"), str)
        or not isinstance(binding.get("receipt_sha256"), str)
    ):
        _fail("context_ir_binding_invalid")
    try:
        receipt = context_ir_bridge.load_effective_prompt_receipt(
            frozen, Path(str(binding["receipt_path"]))
        )
    except context_ir_bridge.ContextIrError as exc:
        raise ProjectMultimodalError(exc.code) from None
    expected = {
        "attempt_id": receipt.attempt_id,
        "provider_task_id": receipt.provider_task_id,
        "source_prompt_sha256": receipt.source_prompt_sha256,
        "effective_prompt_sha256": receipt.effective_prompt_sha256,
        "context_ir_attempt_sha256": receipt.context_ir_attempt_sha256,
        "context_ir_request_sha256": receipt.context_ir_request_sha256,
        "context_ir_task_sha256": receipt.context_ir_task_sha256,
        "receipt_sha256": receipt.receipt_sha256,
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        _fail("context_ir_binding_mismatch")
    try:
        return context_ir_bridge.apply_effective_prompt(
            frozen, receipt.receipt_path
        )
    except context_ir_bridge.ContextIrError as exc:
        raise ProjectMultimodalError(exc.code) from None


def context_ir_progress_binding_matches(
    source_request: h3.H3Request,
    binding: object,
    *,
    frozen_context: context_ir_bridge.FrozenContextIrRequest | None = None,
) -> bool:
    """Bind a resumable coordinator marker to its exact persisted attempt."""
    allowed_binding_statuses = {"running", "query_unknown"}
    if frozen_context is not None:
        allowed_binding_statuses.add("failed")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != _CONTEXT_IR_BINDING_KEYS
        or binding.get("schema") != CONTEXT_IR_BINDING_SCHEMA
        or binding.get("version") != CONTEXT_IR_BINDING_VERSION
        or binding.get("status") not in allowed_binding_statuses
    ):
        return False
    attempt_id = binding.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
    ):
        return False
    path = (
        source_request.workdir
        / ".context-ir"
        / "attempts"
        / attempt_id
        / "attempt.json"
    )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(state, Mapping):
        return False
    binding_status = binding.get("status")
    state_status = state.get("status")
    status_matches = (
        binding_status in {"running", "query_unknown"}
        and state_status in {"polling", "query_unknown"}
    ) or (
        binding_status == "failed"
        and state_status == "failed"
        and state.get("error") in {
            "context_ir_result_invalid",
            "context_ir_semantic_mismatch",
        }
    )
    if not status_matches:
        return False
    if frozen_context is not None:
        expected_input = {
            "cid": frozen_context.cid,
            "source_h3_client_request_id": (
                frozen_context.source_h3_request.client_request_id
            ),
            "skill_plan_sha256": frozen_context.skill_plan_sha256,
            "source_prompt_sha256": frozen_context.source_prompt_sha256,
            "semantic_contract_sha256": frozen_context.semantic_contract_sha256,
            "references_sha256": frozen_context.references_sha256,
            "voice_texts_sha256": frozen_context.voice_texts_sha256,
            "source_h3_request_sha256": frozen_context.source_h3_request_sha256,
            "upstream_dialogue_sha256": frozen_context.upstream_dialogue_sha256,
            "upstream_artifact_path": str(frozen_context.upstream_artifact_path),
            "upstream_artifact_sha256": frozen_context.upstream_artifact_sha256,
            "upstream_dialogue_sha256_path": list(
                frozen_context.upstream_dialogue_sha256_path
            ),
            "duration": frozen_context.duration,
            "ratio": frozen_context.ratio,
        }
        if (
            frozen_context.source_h3_request != source_request
            or state.get("schema") != context_ir_bridge.ATTEMPT_SCHEMA
            or state.get("version") != context_ir_bridge.SCHEMA_VERSION
            or state.get("cid") != source_request.cid
            or state.get("client_request_id") != source_request.client_request_id
            or state.get("input") != expected_input
            or state.get("context_ir_attempt_sha256")
            != frozen_context.context_ir_attempt_sha256
        ):
            return False
    return (
        state.get("attempt_id") == attempt_id
        and state.get("provider_task_id") == binding.get("provider_task_id")
        and state.get("context_ir_attempt_sha256")
        == binding.get("context_ir_attempt_sha256")
        and state.get("context_ir_request_sha256")
        == binding.get("context_ir_request_sha256")
        and state.get("context_ir_task_sha256")
        == binding.get("context_ir_task_sha256")
    )
def _native_segment(
    *, request: h3.H3Request, attempt_id: object, target_duration_s: float,
) -> stitch.StitchSegment:
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
    ):
        _fail("h3_native_attempt_invalid")
    try:
        timeline = h3.load_media_timeline_receipt(request, attempt_id)
    except h3.H3Error:
        _fail("h3_native_timeline_invalid")
    if not isinstance(timeline.get("audio"), Mapping):
        _fail("h3_native_audio_missing")
    output = request.workdir / "generated.mp4"
    if not output.is_file():
        _fail("h3_native_output_missing")
    return stitch.StitchSegment(
        output,
        target_duration_s,
        "hard_cut",
        attempt_id,
        timeline,
    )


def validate_dialogue_acceptance(
    *,
    request: h3.H3Request,
    output: Path,
    timeline: Mapping[str, Any],
) -> str | None:
    """Bind independently produced ASR/lip evidence to exact H3 bytes."""
    on_screen_dialogue = getattr(request, "on_screen_dialogue", ())
    if not on_screen_dialogue:
        return None
    path = request.workdir / FINAL_ACCEPTANCE_FILENAME
    data = _read(path, "final_dialogue_evidence_missing")
    artifact = _json_object(data, "final_dialogue_evidence_invalid")
    try:
        stat = output.stat()
        dialogue_timing.validate_final_acceptance(
            artifact,
            dialogue=tuple(
                {
                    "text": line["text"],
                    "start_s": line["start_s"],
                    "end_s": line["end_s"],
                }
                for line in on_screen_dialogue
            ),
            subjects=tuple(
                str(line["subject_id"])
                for line in on_screen_dialogue
            ),
            output_sha256=_sha256(output.read_bytes()),
            output_size=stat.st_size,
            media_timeline_sha256=h3.canonical_json_sha256(timeline),
            dialogue_sha256=str(request.upstream_dialogue_receipt_sha256),
            speaker_timing_sha256=str(request.speaker_timing_sha256),
        )
    except (OSError, dialogue_timing.DialogueTimingError) as exc:
        code = getattr(exc, "code", "final_dialogue_evidence_invalid")
        _fail(code)
    return _sha256(data)


def stitch_short_native(
    *,
    request: h3.H3Request,
    result: h3.H3Result,
    source_video: Path,
    output: Path,
    target_duration_s: float,
) -> stitch.StitchResult:
    """Publish one short H3 result using provider audio only."""
    if result.status != "succeeded":
        _fail("h3_native_result_incomplete")
    segment = _native_segment(
        request=request,
        attempt_id=result.attempt_id,
        target_duration_s=target_duration_s,
    )
    try:
        return stitch.stitch_video(
            segments=(segment,),
            source_video=source_video,
            output=output,
            audio_mode="provider_generated",
        )
    except (OSError, TypeError, ValueError, stitch.StitchError) as exc:
        attempt_id = result.attempt_id or "unknown"
        directory = request.workdir / "errors" / "attempts" / attempt_id
        sequence = 1
        while (directory / f"h3-short-stitch-{sequence:06d}.json").exists():
            sequence += 1
        error_trace.record(
            directory / f"h3-short-stitch-{sequence:06d}.json",
            call_path=[
                "generation", "h3", "short-stitch", f"attempt:{attempt_id}",
            ],
            error=exc,
            secrets=(request.autodl_token,),
        )
        raise ProjectMultimodalError("h3_native_stitch_failed") from exc


def short_output_is_reusable(
    *,
    request: h3.H3Request,
    expected_attempt_id: str,
    source_video: Path,
    output: Path,
    target_duration_s: float,
) -> bool:
    """Bind published short output to the exact successful H3 attempt/timeline."""
    try:
        result = h3.inspect(request)
        if (
            result.status != "succeeded"
            or result.attempt_id != expected_attempt_id
        ):
            return False
        segment = _native_segment(
            request=request,
            attempt_id=expected_attempt_id,
            target_duration_s=target_duration_s,
        )
        return stitch.output_is_reusable(
            segments=(segment,),
            source_video=source_video,
            output=output,
            audio_mode="provider_generated",
        )
    except (OSError, TypeError, ValueError, h3.H3Error, ProjectMultimodalError):
        return False
