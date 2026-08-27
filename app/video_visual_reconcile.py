"""Closed-world freeze and deterministic projection for optimized video visuals.

This module does not run a Skill, edit media, build provider prose, or submit a
request.  It accepts only the exact source dynamic IR, canonical image plan,
verified optimized-output receipt, and the Skill's unified visual IR.  Every
projection is a receipt-bound copy of those facts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app import image_optimization


SOURCE_SCHEMA = "duet.source-visual-ir"
SOURCE_VERSION = 1
VERIFIED_OUTPUT_SCHEMA = "duet.image-optimization-verified-outputs"
VERIFIED_OUTPUT_VERSION = 1
OUTPUT_RECEIPT_SCHEMA = "duet.image-optimization-output"
OUTPUT_RECEIPT_VERSION = 1
RECEIPT_SCHEMA = "duet.video-visual-reconcile"
RECEIPT_VERSION = 1
PROJECTION_SCHEMA = "duet.video-visual-projection"
PROJECTION_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PERSON_ID_RE = re.compile(r"^PERSON_[0-9]{2,}$")
_SCENE_ID_RE = re.compile(r"^SCENE_[0-9]{2,}$")
_ENTITY_ID_RE = re.compile(r"^ENTITY_[0-9]{2,}$")
_REFERENCE_IN_TEXT_RE = re.compile(r"(?:PERSON|SCENE|ENTITY)_[0-9]{2,}")
_EVIDENCE_REF_RE = re.compile(
    r"^(?:[0-9a-f]{64}|(?:PERSON|SCENE|ENTITY)_[0-9]{2,}|"
    r"frame:[0-9]+:[1-9][0-9]*|receipt:[0-9a-f]{64})$"
)

_SOURCE_KEYS = {
    "schema", "version", "phase", "frame_manifest_sha256",
    "old_visual_prompt_sha256", "frames", "events",
}
_SOURCE_FRAME_KEYS = {
    "segment_index", "frame_index", "source_file", "source_frame_sha256",
    "source_pts", "source_time_base",
}
_EVENT_KEYS = {
    "event_index", "segment_index", "frame_refs", "actor_refs", "scene_ref",
    "entity_refs", "action", "camera", "timing",
}
_ENTITY_REF_KEYS = {"frame_index", "entity_id"}
_ACTION_KEYS = ("initial_state", "motion", "result_state")
_CAMERA_KEYS = ("shot_scale", "angle", "movement", "composition", "focus")
_TIMING_KEYS = {
    "start_source_pts", "end_source_pts", "source_time_base", "pace",
    "transition",
}
_TIME_BASE_KEYS = {"numerator", "denominator"}
_VERIFIED_KEYS = {
    "schema", "version", "plan_sha256", "verification_sha256", "passed",
    "frames",
}
_VERIFIED_FRAME_KEYS = {
    "segment_index", "frame_index", "source_frame_sha256", "optimized_file",
    "optimized_image_sha256", "output_receipt_file", "output_receipt_sha256",
}
_OUTPUT_RECEIPT_KEYS = {
    "schema", "version", "plan_sha256", "segment_index", "frame_index",
    "source_frame_sha256", "output",
}
_OUTPUT_KEYS = {"path", "sha256"}
_SUCCESS_KEYS = {
    "version", "phase", "eligible", "reason", "source_evidence_binding",
    "target_static_plan_binding", "frame_bindings", "preserved_beats",
    "conflicts",
}
_SOURCE_BINDING_KEYS = {"frame_manifest_sha256", "old_visual_prompt_sha256"}
_TARGET_BINDING_KEYS = {"image_plan_sha256", "image_verification_sha256"}
_FRAME_BINDING_KEYS = {
    "segment_index", "frame_index", "source_frame_sha256", "source_pts",
    "source_time_base", "optimized_image_sha256", "output_receipt_sha256",
}
_CONFLICT_KEYS = {
    "code", "segment_index", "frame_index", "evidence_refs",
}
_RECONCILE_REASONS = {
    "phase_input_conflict", "unexpected_dialogue_input",
    "receipt_binding_mismatch", "frame_mapping_missing",
    "optimized_action_changed", "physical_support_unclosed",
    "source_static_semantics_leaked", "reconciliation_unknown",
}
_ARTIFACT_NAMES = (
    "source_visual_ir", "image_plan", "verified_output_receipt",
    "unified_visual_ir",
)
_ARTIFACT_BINDING_KEYS = {"path", "sha256", "canonical_sha256"}
_RECEIPT_KEYS = {"schema", "version", *_ARTIFACT_NAMES}


class VideoVisualReconcileError(RuntimeError):
    """Stable pre-provider reconciliation or receipt failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _SemanticFailure(Exception):
    def __init__(self, code: str, evidence: object) -> None:
        super().__init__(code)
        self.code = code
        self.evidence = evidence


@dataclass(frozen=True, slots=True)
class FrozenVideoVisualReconcile:
    root: Path
    source_visual_ir_path: Path
    source_visual_ir_data: bytes
    source_visual_ir_data_sha256: str
    source_visual_ir_sha256: str
    image_plan_path: Path
    image_plan_data: bytes
    image_plan_data_sha256: str
    image_plan_sha256: str
    verified_output_receipt_path: Path
    verified_output_receipt_data: bytes
    verified_output_receipt_data_sha256: str
    verified_output_receipt_sha256: str
    unified_visual_ir_path: Path
    unified_visual_ir_data: bytes
    unified_visual_ir_data_sha256: str
    unified_visual_ir_sha256: str


def _fail(code: str) -> None:
    raise VideoVisualReconcileError(code)


def _semantic(code: str, evidence: object) -> None:
    raise _SemanticFailure(code, evidence)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object, code: str) -> bytes:
    try:
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
    except (TypeError, ValueError):
        _fail(code)


def canonical_json_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value, "canonical_json_invalid"))


def _json_object(data: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _is_int(value: object, minimum: int | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and (minimum is None or value >= minimum)
    )


def _digest(value: object) -> str:
    try:
        return canonical_json_sha256(value)
    except VideoVisualReconcileError:
        return _sha256(repr(type(value)).encode("utf-8"))


def _hash(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _text(value: object, code: str, *, max_bytes: int = 2048) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value.encode("utf-8")) > max_bytes
        or any(ord(character) < 32 for character in value)
    ):
        _fail(code)
    return value


def _dynamic_text(value: object, code: str) -> str:
    text = _text(value, code)
    if _REFERENCE_IN_TEXT_RE.search(text):
        _fail(code)
    return text


def _relative_text(value: object, code: str) -> str:
    text = _text(value, code, max_bytes=1024)
    path = Path(text)
    if (
        path.is_absolute()
        or "\\" in text
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        _fail(code)
    return text


def _time_base(value: object, code: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _TIME_BASE_KEYS:
        _fail(code)
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        not _is_int(numerator, 1)
        or not _is_int(denominator, 1)
        or math.gcd(numerator, denominator) != 1
    ):
        _fail(code)
    return {"numerator": numerator, "denominator": denominator}


def _canonical_action(value: object, code: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_ACTION_KEYS):
        _fail(code)
    return {key: _dynamic_text(value.get(key), code) for key in _ACTION_KEYS}


def _canonical_camera(value: object, code: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_CAMERA_KEYS):
        _fail(code)
    return {key: _dynamic_text(value.get(key), code) for key in _CAMERA_KEYS}


def _canonical_timing(value: object, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _TIMING_KEYS:
        _fail(code)
    start = value.get("start_source_pts")
    end = value.get("end_source_pts")
    if not _is_int(start) or not _is_int(end) or start > end:
        _fail(code)
    return {
        "start_source_pts": start,
        "end_source_pts": end,
        "source_time_base": _time_base(value.get("source_time_base"), code),
        "pace": _dynamic_text(value.get("pace"), code),
        "transition": _dynamic_text(value.get("transition"), code),
    }


def _canonical_event(
    value: object, code: str, *, index_key: str,
) -> dict[str, Any]:
    expected = (_EVENT_KEYS - {"event_index"}) | {index_key}
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code)
    index = value.get(index_key)
    segment = value.get("segment_index")
    frame_refs = value.get("frame_refs")
    actor_refs = value.get("actor_refs")
    scene_ref = value.get("scene_ref")
    entity_refs = value.get("entity_refs")
    if (
        not _is_int(index, 1)
        or not _is_int(segment, 0)
        or not isinstance(frame_refs, list)
        or not frame_refs
        or any(not _is_int(item, 1) for item in frame_refs)
        or frame_refs != sorted(set(frame_refs))
        or not isinstance(actor_refs, list)
        or any(
            not isinstance(item, str) or _PERSON_ID_RE.fullmatch(item) is None
            for item in actor_refs
        )
        or actor_refs != sorted(set(actor_refs))
        or not isinstance(scene_ref, str)
        or _SCENE_ID_RE.fullmatch(scene_ref) is None
        or not isinstance(entity_refs, list)
    ):
        _fail(code)
    canonical_refs = []
    for item in entity_refs:
        if not isinstance(item, dict) or set(item) != _ENTITY_REF_KEYS:
            _fail(code)
        frame_index = item.get("frame_index")
        entity_id = item.get("entity_id")
        if (
            not _is_int(frame_index, 1)
            or frame_index not in frame_refs
            or not isinstance(entity_id, str)
            or _ENTITY_ID_RE.fullmatch(entity_id) is None
        ):
            _fail(code)
        canonical_refs.append({
            "frame_index": frame_index, "entity_id": entity_id,
        })
    if canonical_refs != sorted(
        canonical_refs, key=lambda item: (item["frame_index"], item["entity_id"])
    ) or len({(item["frame_index"], item["entity_id"]) for item in canonical_refs}) != len(
        canonical_refs
    ):
        _fail(code)
    return {
        index_key: index,
        "segment_index": segment,
        "frame_refs": list(frame_refs),
        "actor_refs": list(actor_refs),
        "scene_ref": scene_ref,
        "entity_refs": canonical_refs,
        "action": _canonical_action(value.get("action"), code),
        "camera": _canonical_camera(value.get("camera"), code),
        "timing": _canonical_timing(value.get("timing"), code),
    }


def _canonical_image_plan(value: object) -> dict[str, Any]:
    try:
        plan = image_optimization.canonical_plan_v3(value)
    except (image_optimization.ImageOptimizationOutputError, TypeError, ValueError):
        _fail("image_plan_invalid")
    if plan.get("eligible") is not True:
        _fail("image_plan_ineligible")
    return plan


def canonical_source_visual_ir(value: object) -> dict[str, Any]:
    code = "source_visual_ir_invalid"
    if (
        not isinstance(value, dict)
        or set(value) != _SOURCE_KEYS
        or value.get("schema") != SOURCE_SCHEMA
        or value.get("version") != SOURCE_VERSION
        or value.get("phase") != "source_visual"
    ):
        _fail(code)
    frame_manifest = _hash(value.get("frame_manifest_sha256"), code)
    old_prompt = _hash(value.get("old_visual_prompt_sha256"), code)
    raw_frames = value.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        _fail(code)
    frames = []
    segment_frames: dict[int, list[dict[str, Any]]] = {}
    for item in raw_frames:
        if not isinstance(item, dict) or set(item) != _SOURCE_FRAME_KEYS:
            _fail(code)
        segment = item.get("segment_index")
        frame_index = item.get("frame_index")
        source_pts = item.get("source_pts")
        if (
            not _is_int(segment, 0)
            or not _is_int(frame_index, 1)
            or not _is_int(source_pts)
        ):
            _fail(code)
        frame = {
            "segment_index": segment,
            "frame_index": frame_index,
            "source_file": _relative_text(item.get("source_file"), code),
            "source_frame_sha256": _hash(item.get("source_frame_sha256"), code),
            "source_pts": source_pts,
            "source_time_base": _time_base(item.get("source_time_base"), code),
        }
        frames.append(frame)
        segment_frames.setdefault(segment, []).append(frame)
    segment_indices = sorted(segment_frames)
    if segment_indices not in ([0], list(range(1, len(segment_indices) + 1))):
        _fail(code)
    if frames != [
        frame for segment in segment_indices for frame in segment_frames[segment]
    ]:
        _fail(code)
    for segment in segment_indices:
        items = segment_frames[segment]
        if (
            [item["frame_index"] for item in items]
            != list(range(1, len(items) + 1))
            or any(
                current["source_pts"] <= previous["source_pts"]
                for previous, current in zip(items, items[1:])
            )
            or len({
                (item["source_time_base"]["numerator"],
                 item["source_time_base"]["denominator"])
                for item in items
            }) != 1
        ):
            _fail(code)

    raw_events = value.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        _fail(code)
    events = [_canonical_event(item, code, index_key="event_index") for item in raw_events]
    segment_events: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        segment_events.setdefault(event["segment_index"], []).append(event)
    if set(segment_events) != set(segment_frames) or events != [
        event for segment in segment_indices for event in segment_events[segment]
    ]:
        _fail(code)
    for segment in segment_indices:
        items = segment_events[segment]
        frames_by_index = {
            item["frame_index"]: item for item in segment_frames[segment]
        }
        if [item["event_index"] for item in items] != list(
            range(1, len(items) + 1)
        ):
            _fail(code)
        covered = set()
        for event in items:
            if any(index not in frames_by_index for index in event["frame_refs"]):
                _fail(code)
            first = frames_by_index[event["frame_refs"][0]]
            last = frames_by_index[event["frame_refs"][-1]]
            if (
                event["timing"]["start_source_pts"] != first["source_pts"]
                or event["timing"]["end_source_pts"] != last["source_pts"]
                or event["timing"]["source_time_base"]
                != first["source_time_base"]
            ):
                _fail(code)
            covered.update(event["frame_refs"])
        if covered != set(frames_by_index):
            _fail(code)
    return {
        "schema": SOURCE_SCHEMA,
        "version": SOURCE_VERSION,
        "phase": "source_visual",
        "frame_manifest_sha256": frame_manifest,
        "old_visual_prompt_sha256": old_prompt,
        "frames": frames,
        "events": events,
    }


def _plan_index(plan: Mapping[str, Any]) -> dict[str, Any]:
    persons = {item["id"] for item in plan["person_plans"]}
    scenes = {item["id"] for item in plan["scene_plans"]}
    segments: dict[int, dict[str, Any]] = {}
    for segment in plan["segments"]:
        person_frames = {
            item["id"]: set(item["observable_frames"])
            for item in segment["persons"] if item["state"] == "replace"
        }
        entities = {
            constraint["frame_index"]: {
                item["entity_id"]
                for item in constraint["non_person_entity_ledger"]["entities"]
            }
            for constraint in segment["frame_constraints"]
        }
        segments[segment["segment_index"]] = {
            "scene_id": segment["scene"]["scene_id"],
            "person_frames": person_frames,
            "entities": entities,
        }
    return {"persons": persons, "scenes": scenes, "segments": segments}


def _static_plan_texts(plan: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for item in plan["person_plans"]:
        for key in (
            "source_identity", "replacement_identity", "wardrobe_change",
            "local_color_change",
        ):
            texts.append(item[key])
    for item in plan["scene_plans"]:
        for key in (
            "source_scene", "replacement_scene", "semantic_change",
            "local_color_change",
        ):
            texts.append(item[key])
        for key in ("geometry_changes", "depth_changes", "layout_changes"):
            texts.extend(item[key])
    for segment in plan["segments"]:
        for person in segment["persons"]:
            for key in ("target_region", "boundary"):
                if isinstance(person[key], str):
                    texts.append(person[key])
        texts.extend(segment["protected_non_target_people"])
        texts.extend(segment["protected_relations"])
        texts.extend(
            value for key, value in segment["scene"].items()
            if key in {"target_region", "boundary"} and isinstance(value, str)
        )
        for constraint in segment["frame_constraints"]:
            texts.extend(constraint[key] for key in (
                "visible_body_parts", "pose_skeleton", "contact_points",
                "occlusion_order", "out_of_frame_crop",
            ))
            texts.extend(
                item["description"]
                for item in constraint["non_person_entity_ledger"]["entities"]
            )
        texts.extend(segment["photometric_contract"].values())
    return tuple(sorted({text for text in texts if isinstance(text, str) and text}))


def _dynamic_strings(event: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(event["action"].values()) + tuple(event["camera"].values()) + (
        event["timing"]["pace"], event["timing"]["transition"],
    )


def _validate_source_plan(source: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    indexed = _plan_index(plan)
    source_segments = {
        item["segment_index"] for item in source["frames"]
    }
    if source_segments != set(indexed["segments"]):
        _semantic("closed_world_mapping_incomplete", source_segments)
    frame_sets = {
        segment: {
            item["frame_index"] for item in source["frames"]
            if item["segment_index"] == segment
        }
        for segment in source_segments
    }
    for segment, frames in frame_sets.items():
        if frames != set(indexed["segments"][segment]["entities"]):
            _semantic("closed_world_mapping_incomplete", [segment, sorted(frames)])
    static_texts = _static_plan_texts(plan)
    for event in source["events"]:
        segment = indexed["segments"].get(event["segment_index"])
        if segment is None or event["scene_ref"] != segment["scene_id"]:
            _semantic("closed_world_mapping_incomplete", event)
        if event["scene_ref"] not in indexed["scenes"]:
            _semantic("closed_world_mapping_incomplete", event)
        for actor in event["actor_refs"]:
            visible = segment["person_frames"].get(actor)
            if (
                actor not in indexed["persons"]
                or visible is None
                or not set(event["frame_refs"]).issubset(visible)
            ):
                _semantic("closed_world_mapping_incomplete", event)
        for reference in event["entity_refs"]:
            if reference["entity_id"] not in segment["entities"].get(
                reference["frame_index"], set()
            ):
                _semantic("closed_world_mapping_incomplete", event)
        for dynamic in _dynamic_strings(event):
            if any(static in dynamic for static in static_texts):
                _semantic("static_semantics_leaked", event)


def _canonical_verified_output_receipt(
    value: object, plan: Mapping[str, Any], source: Mapping[str, Any],
) -> dict[str, Any]:
    code = "verified_output_receipt_invalid"
    if (
        not isinstance(value, dict)
        or set(value) != _VERIFIED_KEYS
        or value.get("schema") != VERIFIED_OUTPUT_SCHEMA
        or value.get("version") != VERIFIED_OUTPUT_VERSION
    ):
        _fail(code)
    plan_sha = image_optimization.plan_sha256(plan)
    if value.get("plan_sha256") != plan_sha:
        _semantic("hash_drift", value)
    verification_sha = _hash(value.get("verification_sha256"), code)
    if value.get("passed") is not True:
        _semantic("receipt_binding_mismatch", value)
    raw_frames = value.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        _fail(code)
    frames = []
    for item in raw_frames:
        if not isinstance(item, dict) or set(item) != _VERIFIED_FRAME_KEYS:
            _fail(code)
        segment = item.get("segment_index")
        frame_index = item.get("frame_index")
        if not _is_int(segment, 0) or not _is_int(frame_index, 1):
            _fail(code)
        frames.append({
            "segment_index": segment,
            "frame_index": frame_index,
            "source_frame_sha256": _hash(item.get("source_frame_sha256"), code),
            "optimized_file": _relative_text(item.get("optimized_file"), code),
            "optimized_image_sha256": _hash(
                item.get("optimized_image_sha256"), code
            ),
            "output_receipt_file": _relative_text(
                item.get("output_receipt_file"), code
            ),
            "output_receipt_sha256": _hash(
                item.get("output_receipt_sha256"), code
            ),
        })
    expected = [
        (item["segment_index"], item["frame_index"], item["source_frame_sha256"])
        for item in source["frames"]
    ]
    actual = [
        (item["segment_index"], item["frame_index"], item["source_frame_sha256"])
        for item in frames
    ]
    if actual != expected:
        _semantic("frame_mapping_missing", [expected, actual])
    return {
        "schema": VERIFIED_OUTPUT_SCHEMA,
        "version": VERIFIED_OUTPUT_VERSION,
        "plan_sha256": plan_sha,
        "verification_sha256": verification_sha,
        "passed": True,
        "frames": frames,
    }


def _canonical_conflict(value: object, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONFLICT_KEYS:
        _fail(code)
    reason = value.get("code")
    segment = value.get("segment_index")
    frame = value.get("frame_index")
    evidence = value.get("evidence_refs")
    if (
        reason not in _RECONCILE_REASONS
        or (segment is not None and not _is_int(segment, 0))
        or (frame is not None and not _is_int(frame, 1))
        or not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, str)
            or _EVIDENCE_REF_RE.fullmatch(item) is None
            for item in evidence
        )
    ):
        _fail(code)
    return {
        "code": reason,
        "segment_index": segment,
        "frame_index": frame,
        "evidence_refs": list(evidence),
    }


def _canonical_unified_shape(value: object) -> dict[str, Any]:
    code = "unified_visual_ir_invalid"
    if (
        not isinstance(value, dict)
        or set(value) != _SUCCESS_KEYS
        or value.get("version") != 1
        or value.get("phase") != "reconcile_after_image_optimization"
        or not isinstance(value.get("eligible"), bool)
    ):
        _fail(code)
    if value["eligible"] is False:
        conflicts = value.get("conflicts")
        if (
            value.get("reason") not in _RECONCILE_REASONS
            or value.get("source_evidence_binding") is not None
            or value.get("target_static_plan_binding") is not None
            or value.get("frame_bindings") != []
            or value.get("preserved_beats") != []
            or not isinstance(conflicts, list)
            or not conflicts
        ):
            _fail(code)
        canonical_conflicts = [
            _canonical_conflict(item, code) for item in conflicts
        ]
        if canonical_conflicts[0]["code"] != value["reason"]:
            _fail(code)
        return {
            "version": 1,
            "phase": "reconcile_after_image_optimization",
            "eligible": False,
            "reason": value["reason"],
            "source_evidence_binding": None,
            "target_static_plan_binding": None,
            "frame_bindings": [],
            "preserved_beats": [],
            "conflicts": canonical_conflicts,
        }
    if value.get("reason") is not None or value.get("conflicts") != []:
        _fail(code)
    source_binding = value.get("source_evidence_binding")
    target_binding = value.get("target_static_plan_binding")
    if (
        not isinstance(source_binding, dict)
        or set(source_binding) != _SOURCE_BINDING_KEYS
        or not isinstance(target_binding, dict)
        or set(target_binding) != _TARGET_BINDING_KEYS
    ):
        _fail(code)
    canonical_source_binding = {
        key: _hash(source_binding.get(key), code)
        for key in ("frame_manifest_sha256", "old_visual_prompt_sha256")
    }
    canonical_target_binding = {
        key: _hash(target_binding.get(key), code)
        for key in ("image_plan_sha256", "image_verification_sha256")
    }
    raw_frames = value.get("frame_bindings")
    raw_beats = value.get("preserved_beats")
    if (
        not isinstance(raw_frames, list) or not raw_frames
        or not isinstance(raw_beats, list) or not raw_beats
    ):
        _fail(code)
    frames = []
    for item in raw_frames:
        if not isinstance(item, dict) or set(item) != _FRAME_BINDING_KEYS:
            _fail(code)
        segment = item.get("segment_index")
        frame = item.get("frame_index")
        pts = item.get("source_pts")
        if not _is_int(segment, 0) or not _is_int(frame, 1) or not _is_int(pts):
            _fail(code)
        frames.append({
            "segment_index": segment,
            "frame_index": frame,
            "source_frame_sha256": _hash(item.get("source_frame_sha256"), code),
            "source_pts": pts,
            "source_time_base": _time_base(item.get("source_time_base"), code),
            "optimized_image_sha256": _hash(
                item.get("optimized_image_sha256"), code
            ),
            "output_receipt_sha256": _hash(
                item.get("output_receipt_sha256"), code
            ),
        })
    beats = [
        _canonical_event(item, code, index_key="beat_index") for item in raw_beats
    ]
    return {
        "version": 1,
        "phase": "reconcile_after_image_optimization",
        "eligible": True,
        "reason": None,
        "source_evidence_binding": canonical_source_binding,
        "target_static_plan_binding": canonical_target_binding,
        "frame_bindings": frames,
        "preserved_beats": beats,
        "conflicts": [],
    }


def _failure(reason: str, evidence: object) -> dict[str, Any]:
    mapped = {
        "hash_drift": "receipt_binding_mismatch",
        "receipt_binding_mismatch": "receipt_binding_mismatch",
        "frame_mapping_missing": "frame_mapping_missing",
        "static_semantics_leaked": "source_static_semantics_leaked",
        "closed_world_mapping_incomplete": "reconciliation_unknown",
        "event_set_drift": "reconciliation_unknown",
        "event_content_drift": "reconciliation_unknown",
    }.get(reason, "reconciliation_unknown")
    return {
        "version": 1,
        "phase": "reconcile_after_image_optimization",
        "eligible": False,
        "reason": mapped,
        "source_evidence_binding": None,
        "target_static_plan_binding": None,
        "frame_bindings": [],
        "preserved_beats": [],
        "conflicts": [{
            "code": mapped,
            "segment_index": None,
            "frame_index": None,
            "evidence_refs": [_digest(evidence)],
        }],
    }


def canonical_unified_visual_ir(
    value: object,
    *,
    source_visual_ir: object,
    image_plan: object,
    verified_output_receipt: object,
) -> dict[str, Any]:
    """Canonicalize an exact Skill result or return a fixed ineligible IR."""
    plan = _canonical_image_plan(image_plan)
    source = canonical_source_visual_ir(source_visual_ir)
    try:
        _validate_source_plan(source, plan)
        verified = _canonical_verified_output_receipt(
            verified_output_receipt, plan, source
        )
        unified = _canonical_unified_shape(value)
        if unified["eligible"] is False:
            return deepcopy(unified)
        if unified["source_evidence_binding"] != {
            "frame_manifest_sha256": source["frame_manifest_sha256"],
            "old_visual_prompt_sha256": source["old_visual_prompt_sha256"],
        } or unified["target_static_plan_binding"] != {
            "image_plan_sha256": image_optimization.plan_sha256(plan),
            "image_verification_sha256": verified["verification_sha256"],
        }:
            _semantic("hash_drift", value)
        expected_frames = [
            {
                "segment_index": frame["segment_index"],
                "frame_index": frame["frame_index"],
                "source_frame_sha256": frame["source_frame_sha256"],
                "source_pts": frame["source_pts"],
                "source_time_base": deepcopy(frame["source_time_base"]),
                "optimized_image_sha256": output["optimized_image_sha256"],
                "output_receipt_sha256": output["output_receipt_sha256"],
            }
            for frame, output in zip(
                source["frames"], verified["frames"], strict=True
            )
        ]
        if unified["frame_bindings"] != expected_frames:
            _semantic("frame_mapping_missing", unified["frame_bindings"])
        expected_beats = [
            {
                "beat_index": event["event_index"],
                **{
                    key: deepcopy(event[key])
                    for key in (
                        "segment_index", "frame_refs", "actor_refs", "scene_ref",
                        "entity_refs", "action", "camera", "timing",
                    )
                },
            }
            for event in source["events"]
        ]
        actual_signature = [
            (item["beat_index"], item["segment_index"], item["frame_refs"])
            for item in unified["preserved_beats"]
        ]
        expected_signature = [
            (item["beat_index"], item["segment_index"], item["frame_refs"])
            for item in expected_beats
        ]
        if actual_signature != expected_signature:
            _semantic("event_set_drift", actual_signature)
        if unified["preserved_beats"] != expected_beats:
            _semantic("event_content_drift", unified["preserved_beats"])
        return deepcopy(unified)
    except _SemanticFailure as exc:
        return _failure(exc.code, exc.evidence)


def _safe_file(root: Path, path: Path | str, code: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        _fail(code)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                _fail(code)
        except OSError:
            _fail(code)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
        descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                _fail(code)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except OSError:
        _fail(code)
    data = b"".join(chunks)
    if not data:
        _fail(code)
    return resolved, data


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        _fail("receipt_invalid")


def _canonical_output_receipt(
    value: object,
    *,
    plan_sha256: str,
    source: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    code = "output_receipt_invalid"
    if (
        not isinstance(value, dict)
        or set(value) != _OUTPUT_RECEIPT_KEYS
        or value.get("schema") != OUTPUT_RECEIPT_SCHEMA
        or value.get("version") != OUTPUT_RECEIPT_VERSION
        or value.get("plan_sha256") != plan_sha256
        or value.get("segment_index") != source["segment_index"]
        or value.get("frame_index") != source["frame_index"]
        or value.get("source_frame_sha256") != source["source_frame_sha256"]
    ):
        _fail(code)
    output = value.get("output")
    if not isinstance(output, dict) or set(output) != _OUTPUT_KEYS:
        _fail(code)
    canonical_output = {
        "path": _relative_text(output.get("path"), code),
        "sha256": _hash(output.get("sha256"), code),
    }
    if canonical_output != {
        "path": verified["optimized_file"],
        "sha256": verified["optimized_image_sha256"],
    }:
        _fail("hash_drift")
    return {
        "schema": OUTPUT_RECEIPT_SCHEMA,
        "version": OUTPUT_RECEIPT_VERSION,
        "plan_sha256": plan_sha256,
        "segment_index": source["segment_index"],
        "frame_index": source["frame_index"],
        "source_frame_sha256": source["source_frame_sha256"],
        "output": canonical_output,
    }


def _validate_bound_media(
    root: Path,
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> None:
    plan_sha = image_optimization.plan_sha256(plan)
    seen_paths: set[Path] = set()
    for source_frame, output in zip(
        source["frames"], verified["frames"], strict=True
    ):
        source_path, source_data = _safe_file(
            root, source_frame["source_file"], "source_frame_invalid"
        )
        optimized_path, optimized_data = _safe_file(
            root, output["optimized_file"], "optimized_output_invalid"
        )
        receipt_path, receipt_data = _safe_file(
            root, output["output_receipt_file"], "output_receipt_invalid"
        )
        if (
            len({source_path, optimized_path, receipt_path}) != 3
            or any(path in seen_paths for path in (
                source_path, optimized_path, receipt_path
            ))
        ):
            _fail("receipt_path_invalid")
        seen_paths.update((source_path, optimized_path, receipt_path))
        if (
            _sha256(source_data) != source_frame["source_frame_sha256"]
            or _sha256(optimized_data) != output["optimized_image_sha256"]
            or _sha256(receipt_data) != output["output_receipt_sha256"]
        ):
            _fail("hash_drift")
        receipt = _json_object(receipt_data, "output_receipt_invalid")
        _canonical_output_receipt(
            receipt,
            plan_sha256=plan_sha,
            source=source_frame,
            verified=output,
        )


def freeze(
    root: Path,
    *,
    source_visual_ir_path: Path,
    image_plan_path: Path,
    verified_output_receipt_path: Path,
    unified_visual_ir_path: Path,
) -> FrozenVideoVisualReconcile:
    """Freeze four canonical artifacts plus every receipt-bound image byte."""
    try:
        root = Path(root).resolve(strict=True)
    except OSError:
        _fail("receipt_root_invalid")
    if not root.is_dir():
        _fail("receipt_root_invalid")
    paths_and_data = {
        name: _safe_file(root, path, f"{name}_invalid")
        for name, path in (
            ("source_visual_ir", source_visual_ir_path),
            ("image_plan", image_plan_path),
            ("verified_output_receipt", verified_output_receipt_path),
            ("unified_visual_ir", unified_visual_ir_path),
        )
    }
    if len({item[0] for item in paths_and_data.values()}) != 4:
        _fail("receipt_path_invalid")
    source = canonical_source_visual_ir(_json_object(
        paths_and_data["source_visual_ir"][1], "source_visual_ir_invalid"
    ))
    plan = _canonical_image_plan(_json_object(
        paths_and_data["image_plan"][1], "image_plan_invalid"
    ))
    try:
        _validate_source_plan(source, plan)
        verified = _canonical_verified_output_receipt(
            _json_object(
                paths_and_data["verified_output_receipt"][1],
                "verified_output_receipt_invalid",
            ),
            plan,
            source,
        )
    except _SemanticFailure as exc:
        _fail(exc.code)
    raw_unified = _json_object(
        paths_and_data["unified_visual_ir"][1], "unified_visual_ir_invalid"
    )
    unified = canonical_unified_visual_ir(
        raw_unified,
        source_visual_ir=source,
        image_plan=plan,
        verified_output_receipt=verified,
    )
    if unified != _canonical_unified_shape(raw_unified):
        _fail("unified_visual_ir_semantic_mismatch")
    _validate_bound_media(root, source, plan, verified)
    values = {
        "source_visual_ir": source,
        "image_plan": plan,
        "verified_output_receipt": verified,
        "unified_visual_ir": unified,
    }
    fields: dict[str, Any] = {"root": root}
    for name in _ARTIFACT_NAMES:
        path, data = paths_and_data[name]
        fields[f"{name}_path"] = path
        fields[f"{name}_data"] = data
        fields[f"{name}_data_sha256"] = _sha256(data)
        fields[f"{name}_sha256"] = (
            image_optimization.plan_sha256(plan)
            if name == "image_plan"
            else canonical_json_sha256(values[name])
        )
    return FrozenVideoVisualReconcile(**fields)


def receipt_binding(
    root: Path, frozen: FrozenVideoVisualReconcile,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not isinstance(frozen, FrozenVideoVisualReconcile) or frozen.root != root:
        _fail("receipt_invalid")
    result: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
    }
    for name in _ARTIFACT_NAMES:
        result[name] = {
            "path": _relative(root, getattr(frozen, f"{name}_path")),
            "sha256": getattr(frozen, f"{name}_data_sha256"),
            "canonical_sha256": getattr(frozen, f"{name}_sha256"),
        }
    return result


def load_bound(root: Path, binding: object) -> FrozenVideoVisualReconcile:
    """Reload an exact frozen snapshot and reject path, byte, or hash drift."""
    root = Path(root).resolve()
    if (
        not isinstance(binding, dict)
        or set(binding) != _RECEIPT_KEYS
        or binding.get("schema") != RECEIPT_SCHEMA
        or binding.get("version") != RECEIPT_VERSION
    ):
        _fail("receipt_invalid")
    paths: dict[str, Path] = {}
    for name in _ARTIFACT_NAMES:
        item = binding.get(name)
        if not isinstance(item, dict) or set(item) != _ARTIFACT_BINDING_KEYS:
            _fail("receipt_invalid")
        paths[f"{name}_path"] = root / _relative_text(
            item.get("path"), "receipt_invalid"
        )
        _hash(item.get("sha256"), "receipt_invalid")
        _hash(item.get("canonical_sha256"), "receipt_invalid")
    frozen = freeze(root, **paths)
    if receipt_binding(root, frozen) != binding:
        _fail("receipt_mismatch")
    return frozen


def project(frozen: FrozenVideoVisualReconcile) -> dict[str, Any]:
    """Project only bound dynamic events, target IDs, hashes, and output paths."""
    if not isinstance(frozen, FrozenVideoVisualReconcile):
        _fail("receipt_invalid")
    current = freeze(
        frozen.root,
        source_visual_ir_path=frozen.source_visual_ir_path,
        image_plan_path=frozen.image_plan_path,
        verified_output_receipt_path=frozen.verified_output_receipt_path,
        unified_visual_ir_path=frozen.unified_visual_ir_path,
    )
    if receipt_binding(frozen.root, current) != receipt_binding(
        frozen.root, frozen
    ):
        _fail("receipt_mismatch")
    frozen = current
    source = canonical_source_visual_ir(_json_object(
        frozen.source_visual_ir_data, "source_visual_ir_invalid"
    ))
    plan = _canonical_image_plan(_json_object(
        frozen.image_plan_data, "image_plan_invalid"
    ))
    try:
        verified = _canonical_verified_output_receipt(
            _json_object(
                frozen.verified_output_receipt_data,
                "verified_output_receipt_invalid",
            ),
            plan,
            source,
        )
    except _SemanticFailure as exc:
        _fail(exc.code)
    unified = canonical_unified_visual_ir(
        _json_object(frozen.unified_visual_ir_data, "unified_visual_ir_invalid"),
        source_visual_ir=source,
        image_plan=plan,
        verified_output_receipt=verified,
    )
    if unified["eligible"] is not True:
        _fail(unified["reason"])
    frames = [
        {
            **deepcopy(binding),
            "optimized_file": output["optimized_file"],
        }
        for binding, output in zip(
            unified["frame_bindings"], verified["frames"], strict=True
        )
    ]
    return {
        "schema": PROJECTION_SCHEMA,
        "version": PROJECTION_VERSION,
        "source_visual_ir_sha256": frozen.source_visual_ir_sha256,
        "unified_visual_ir_sha256": frozen.unified_visual_ir_sha256,
        "target_static_plan_binding": deepcopy(
            unified["target_static_plan_binding"]
        ),
        "frames": frames,
        "events": deepcopy(source["events"]),
    }
