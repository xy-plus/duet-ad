"""Project-level prompt generation, frozen segment prompts, and strict CAS editing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
from copy import deepcopy
from pathlib import Path

import cv2

from app.codex_runner import CodexError
from app.config import (
    SEEDREAM_EDIT_MODES,
    SEEDREAM_MODELS,
    Settings,
)


_LOGGER = logging.getLogger(__name__)

MAX_PROMPT_BYTES = 32 * 1024
MAX_CONTINUITY_BYTES = 32 * 1024
MAX_PROJECT_OUTPUT_OVERHEAD_BYTES = 64 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _ROOT / "skills" / "image-postprocess" / "SKILL.md"
PALETTE_METRIC_ALGORITHM = "area-weighted-cie-lab-hsv-v1"
PALETTE_METRIC_THRESHOLDS = {
    "lab_b_star_neutral": 128.0,
    "lab_b_star_scale": 127.0,
    "warm_cool_delta": 0.05,
    "muted_saturation": 0.16,
    "vivid_saturation": 0.58,
}
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
_PERSON_ID_RE = re.compile(r"^PERSON_([0-9]{2})$")
_SCENE_ID_RE = re.compile(r"^SCENE_([0-9]{2})$")
_ENTITY_ID_RE = re.compile(r"^ENTITY_([0-9]{2})$")
_COMPONENT_ID_RE = re.compile(r"^COMPONENT_([0-9]{2})$")
_ENTITY_ID_MENTION_RE = re.compile(r"(?<![A-Z0-9_])ENTITY_[0-9]{2}(?![A-Z0-9_])")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_KEY_PREFIX = "stable_key="
_INELIGIBLE_REASONS = {
    "no_observable_narrative_person",
    "narrative_person_tracks_ambiguous",
    "person_replacement_unsafe",
    "scene_components_ambiguous",
    "scene_structure_replacement_unsafe",
}
_VERIFY_STATUSES = {"pass", "not_applicable", "fail", "unknown"}
_PROJECT_CHECKS = (
    "narrative_person_completeness",
    "no_identity_swap",
    "no_unplanned_person",
    "person_identity_continuity",
    "scene_continuity",
)
_PACK_PERSON_CHECKS = (
    "identity_changed",
    "source_identity_absent",
    "multiview",
    "local_color",
)
_PACK_SCENE_CHECKS = (
    "semantic",
    "geometry",
    "depth",
    "layout",
    "local_color",
)
_PACK_PROJECT_CHECKS = (
    "light_direction_preservation",
    "exposure_preservation",
    "wb_cct_preservation",
    "tone_curve_preservation",
)


class ImageOptimizationError(ValueError):
    def __init__(self, status: int, detail: str | dict[str, str]):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


class ImageOptimizationOutputError(ValueError):
    pass


class ImageOptimizationIneligibleError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def verification_skill_path() -> Path:
    """Resolve the shared plan/verify Skill without accepting a symlink."""
    try:
        info = _SKILL.lstat()
        resolved = _SKILL.resolve(strict=True)
    except OSError:
        raise ValueError("invalid image verification skill") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or resolved != _SKILL
    ):
        raise ValueError("invalid image verification skill")
    return resolved


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


def _sha256_regular(source: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError:
        raise ValueError("invalid image optimization keyframe") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("invalid image optimization keyframe")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _read_json_output(path: Path, max_bytes: int) -> object:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        ) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
    finally:
        os.close(fd)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        ) from None


def _canonical_continuity_v1(
    value: object, expected_indices: list[int] | None = None
) -> dict:
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


def _canonical_text(value: object, *, max_bytes: int = 2048) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return value


def _canonical_text_list(
    value: object, *, minimum: int = 0, maximum: int = 20
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    result = [_canonical_text(item, max_bytes=1024) for item in value]
    if len(set(result)) != len(result):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return result


def _canonical_plan_indices(
    value: object, expected_indices: list[int] | None = None
) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(index, bool) or not isinstance(index, int) for index in value)
        or value not in ([0], list(range(1, len(value) + 1)))
        or (expected_indices is not None and value != expected_indices)
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return list(value)


def _canonical_reference(
    value: object,
    indices: list[int],
    frame_counts: dict[int, int] | None,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"segment_index", "frame_index"}:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    segment_index = value.get("segment_index")
    frame_index = value.get("frame_index")
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index not in indices
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 1
        or (
            frame_counts is not None
            and frame_index > frame_counts.get(segment_index, 0)
        )
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {"segment_index": segment_index, "frame_index": frame_index}


def _canonical_plan_v2(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
    *,
    allow_empty_people: bool = False,
) -> dict:
    keys = {
        "version",
        "phase",
        "segment_indices",
        "eligible",
        "reason",
        "person_plans",
        "scene_plans",
        "segments",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("version") != 2
        or value.get("phase") != "plan"
        or not isinstance(value.get("eligible"), bool)
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    indices = _canonical_plan_indices(value.get("segment_indices"), expected_indices)
    if frame_counts is not None and set(frame_counts) != set(indices):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    if not value["eligible"]:
        if (
            value.get("reason") not in _INELIGIBLE_REASONS
            or value.get("person_plans") != []
            or value.get("scene_plans") != []
            or value.get("segments") != []
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        return {
            "version": 2,
            "phase": "plan",
            "segment_indices": indices,
            "eligible": False,
            "reason": value["reason"],
            "person_plans": [],
            "scene_plans": [],
            "segments": [],
        }
    if value.get("reason") is not None:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )

    raw_people = value.get("person_plans")
    if (
        not isinstance(raw_people, list)
        or len(raw_people) > 20
        or not allow_empty_people and not raw_people
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    people = []
    for number, item in enumerate(raw_people, 1):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "source_identity",
            "replacement_identity",
            "wardrobe_change",
            "local_color_change",
            "reference",
            "observable_segments",
        }:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        identifier = item.get("id")
        matched = _PERSON_ID_RE.fullmatch(identifier) if isinstance(identifier, str) else None
        observable = item.get("observable_segments")
        if (
            matched is None
            or int(matched.group(1)) != number
            or not isinstance(observable, list)
            or not observable
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in indices
                for index in observable
            )
            or observable != sorted(set(observable))
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        source = _canonical_text(item.get("source_identity"))
        replacement = _canonical_text(item.get("replacement_identity"))
        if source == replacement:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        people.append(
            {
                "id": identifier,
                "source_identity": source,
                "replacement_identity": replacement,
                "wardrobe_change": _canonical_text(item.get("wardrobe_change")),
                "local_color_change": _canonical_text(item.get("local_color_change")),
                "reference": _canonical_reference(
                    item.get("reference"), indices, frame_counts
                ),
                "observable_segments": list(observable),
            }
        )

    raw_scenes = value.get("scene_plans")
    if not isinstance(raw_scenes, list) or not 1 <= len(raw_scenes) <= 50:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    scenes = []
    covered: list[int] = []
    for number, item in enumerate(raw_scenes, 1):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "source_scene",
            "replacement_scene",
            "semantic_change",
            "geometry_changes",
            "depth_changes",
            "layout_changes",
            "local_color_change",
            "reference",
            "segments",
        }:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        identifier = item.get("id")
        matched = _SCENE_ID_RE.fullmatch(identifier) if isinstance(identifier, str) else None
        members = item.get("segments")
        if (
            matched is None
            or int(matched.group(1)) != number
            or not isinstance(members, list)
            or not members
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in indices
                for index in members
            )
            or members != sorted(set(members))
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        source = _canonical_text(item.get("source_scene"))
        replacement = _canonical_text(item.get("replacement_scene"))
        reference = _canonical_reference(item.get("reference"), indices, frame_counts)
        if source == replacement or reference["segment_index"] not in members:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        scenes.append(
            {
                "id": identifier,
                "source_scene": source,
                "replacement_scene": replacement,
                "semantic_change": _canonical_text(item.get("semantic_change")),
                "geometry_changes": _canonical_text_list(
                    item.get("geometry_changes"), minimum=1, maximum=8
                ),
                "depth_changes": _canonical_text_list(
                    item.get("depth_changes"), minimum=1, maximum=8
                ),
                "layout_changes": _canonical_text_list(
                    item.get("layout_changes"), minimum=1, maximum=8
                ),
                "local_color_change": _canonical_text(item.get("local_color_change")),
                "reference": reference,
                "segments": list(members),
            }
        )
        covered.extend(members)
    if sorted(covered) != indices or len(covered) != len(set(covered)):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )

    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(indices):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    person_ids = [item["id"] for item in people]
    scene_by_id = {item["id"]: item for item in scenes}
    segments = []
    observed_by_person = {identifier: [] for identifier in person_ids}
    person_frames: dict[tuple[int, str], list[int]] = {}
    for expected, item in zip(indices, raw_segments):
        if not isinstance(item, dict) or set(item) != {
            "segment_index",
            "persons",
            "scene",
            "protected_non_target_people",
            "protected_relations",
        } or item.get("segment_index") != expected:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        raw_segment_people = item.get("persons")
        if (
            not isinstance(raw_segment_people, list)
            or len(raw_segment_people) != len(person_ids)
            or [person.get("id") for person in raw_segment_people if isinstance(person, dict)]
            != person_ids
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        segment_people = []
        for person in raw_segment_people:
            if set(person) != {
                "id",
                "state",
                "observable_frames",
                "target_region",
                "boundary",
            }:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            state = person.get("state")
            frames = person.get("observable_frames")
            if (
                state not in {"replace", "not_observable"}
                or not isinstance(frames, list)
                or any(
                    isinstance(frame, bool) or not isinstance(frame, int) or frame < 1
                    for frame in frames
                )
                or frames != sorted(set(frames))
                or (
                    frame_counts is not None
                    and any(frame > frame_counts[expected] for frame in frames)
                )
            ):
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            if state == "replace":
                if not frames:
                    raise ImageOptimizationOutputError(
                        "image optimization output is missing or invalid"
                    )
                target = _canonical_text(person.get("target_region"))
                boundary = _canonical_text(person.get("boundary"))
                observed_by_person[person["id"]].append(expected)
            else:
                if frames or person.get("target_region") is not None or person.get("boundary") is not None:
                    raise ImageOptimizationOutputError(
                        "image optimization output is missing or invalid"
                    )
                target = None
                boundary = None
            person_frames[(expected, person["id"])] = list(frames)
            segment_people.append(
                {
                    "id": person["id"],
                    "state": state,
                    "observable_frames": list(frames),
                    "target_region": target,
                    "boundary": boundary,
                }
            )
        scene = item.get("scene")
        if not isinstance(scene, dict) or set(scene) != {
            "scene_id",
            "target_region",
            "boundary",
            "layout_reference_frame_index",
        }:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        scene_id = scene.get("scene_id")
        layout_frame = scene.get("layout_reference_frame_index")
        if (
            scene_id not in scene_by_id
            or expected not in scene_by_id[scene_id]["segments"]
            or isinstance(layout_frame, bool)
            or not isinstance(layout_frame, int)
            or layout_frame < 1
            or (
                frame_counts is not None
                and layout_frame > frame_counts[expected]
            )
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        segments.append(
            {
                "segment_index": expected,
                "persons": segment_people,
                "scene": {
                    "scene_id": scene_id,
                    "target_region": _canonical_text(scene.get("target_region")),
                    "boundary": _canonical_text(scene.get("boundary")),
                    "layout_reference_frame_index": layout_frame,
                },
                "protected_non_target_people": _canonical_text_list(
                    item.get("protected_non_target_people"), maximum=20
                ),
                "protected_relations": _canonical_text_list(
                    item.get("protected_relations"), maximum=30
                ),
            }
        )
    for person in people:
        if observed_by_person[person["id"]] != person["observable_segments"]:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        reference = person["reference"]
        if reference["segment_index"] not in person["observable_segments"] or reference[
            "frame_index"
        ] not in person_frames[(reference["segment_index"], person["id"])]:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
    return {
        "version": 2,
        "phase": "plan",
        "segment_indices": indices,
        "eligible": True,
        "reason": None,
        "person_plans": people,
        "scene_plans": scenes,
        "segments": segments,
    }


def canonical_plan_v2(
    value: object,
    segment_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
    """Return an isolated canonical v2 plan for all downstream consumers."""
    return deepcopy(_canonical_plan_v2(value, segment_indices, frame_counts))


_FRAME_TEXT_CONSTRAINT_KEYS = (
    "visible_body_parts",
    "pose_skeleton",
    "contact_points",
    "occlusion_order",
    "out_of_frame_crop",
)
_FRAME_CONSTRAINT_KEYS = (
    "frame_index",
    *_FRAME_TEXT_CONSTRAINT_KEYS,
    "non_person_entity_ledger",
    "dominant_palette_contract",
)
_NON_PERSON_ENTITY_LEDGER_KEYS = {"entities", "relations"}
_ENTITY_LEDGER_ENTITY_KEYS = {"entity_id", "description", "visibility"}
_ENTITY_LEDGER_RELATION_KEYS = {"subject_id", "predicate", "object_id"}
_ENTITY_VISIBILITIES = {
    "full", "partial", "edge_fragment", "occluded", "out_of_frame",
    "source_preserve",
}
_ENTITY_RELATION_PREDICATES = {
    "supports", "contacts", "separate_from", "occludes", "owned_by",
}
_PHYSICAL_ENTITY_RELATION_PREDICATES = {
    "supports", "contacts", "separate_from",
}
_PHOTOMETRIC_CONTRACT_KEYS = (
    "light_direction",
    "light_quality",
    "exposure_or_intensity",
    "wb_cct",
    "global_contrast",
    "tone_curve",
)
_DOMINANT_PALETTE_CONTRACT_KEYS = (
    "area_weighted_warm_cool_family",
    "saturation_style",
)
_AREA_WEIGHTED_WARM_COOL_FAMILIES = {"warm", "cool", "balanced"}
_SATURATION_STYLES = {"muted", "natural", "vivid"}
_SCENE_CONTINUITY_GRAPH_KEYS = {"components", "topology", "views"}
_SCENE_CONTINUITY_CORE_KEYS = {"components", "topology"}
_SCENE_CONTINUITY_COMPONENT_KEYS = {"component_id", "target_spec"}
_SCENE_CONTINUITY_TOPOLOGY_PREDICATES = {
    "supports", "contacts", "separate_from",
}
_SCENE_CONTINUITY_VIEW_KEYS = {
    "segment_index", "frame_index", "transition_from_previous",
    "observations", "view_relations",
}
_SCENE_CONTINUITY_OBSERVATION_KEYS = {"component_id", "visibility"}
_SCENE_CONTINUITY_TRANSITIONS = {
    "start", "same_camera", "camera_motion", "hard_cut",
}
_SCENE_CONTINUITY_VISIBILITIES = {
    "full", "partial", "edge_fragment", "occluded", "out_of_view",
}
_SCENE_CONTINUITY_VISIBLE = {"full", "partial", "edge_fragment"}
_SCENE_CONTINUITY_VIEW_PREDICATES = {"in_front_of", "occludes"}
_PROJECT_ENTITY_OWNER_ID = "PROJECT"
_SEMANTIC_ENTITY_VISIBILITIES = {
    "visible": "full",
    "occluded": "occluded",
    "out_of_frame": "out_of_frame",
}
_SEMANTIC_DERIVED_APPEARANCE_MODES = {
    "optical_projection",
    "temporal_residual",
}
_NON_PHYSICAL_APPEARANCE_CLAUSE = (
    "源帧中的光学投影、屏幕成像、阴影、时间采样残影、运动拖影和失焦回波，"
    "都只是已冻结主实体的非物理成像派生，不得实例化为新的物理人物或人体结构；"
    "其源载体随场景替换而消失时不得复制，只保持当前帧直接可见主实体的数量、"
    "区域、边界和物理关系"
)


def _contains_directed_cycle(edges: list[tuple[str, str]]) -> bool:
    graph: dict[str, set[str]] = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in graph[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def _canonical_component_relations(
    value: object,
    component_ids: tuple[str, ...],
    predicates: set[str],
    *,
    topology: bool,
    visibility_by_component: dict[str, str] | None = None,
) -> list[dict]:
    if not isinstance(value, list) or len(value) > 60:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    allowed = set(component_ids)
    pair_keys = set()
    cycle_edges = []
    relations = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _ENTITY_LEDGER_RELATION_KEYS:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        subject = item.get("subject_id")
        predicate = item.get("predicate")
        object_ = item.get("object_id")
        if (
            not isinstance(subject, str)
            or not isinstance(object_, str)
            or subject == object_
            or subject not in allowed
            or object_ not in allowed
            or not isinstance(predicate, str)
            or predicate not in predicates
            or (
                topology
                and predicate in {"contacts", "separate_from"}
                and subject >= object_
            )
            or (
                visibility_by_component is not None
                and (
                    visibility_by_component[subject] == "out_of_view"
                    or visibility_by_component[object_] == "out_of_view"
                    or (
                        predicate == "occludes"
                        and visibility_by_component[subject]
                        not in _SCENE_CONTINUITY_VISIBLE
                    )
                )
            )
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        pair = tuple(sorted((subject, object_))) if topology else (subject, object_)
        if pair in pair_keys:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        pair_keys.add(pair)
        if (topology and predicate == "supports") or not topology:
            cycle_edges.append((subject, object_))
        relations.append({
            "subject_id": subject,
            "predicate": predicate,
            "object_id": object_,
        })
    if relations != sorted(
        relations,
        key=lambda item: (item["subject_id"], item["predicate"], item["object_id"]),
    ) or _contains_directed_cycle(cycle_edges):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return relations


def _canonical_scene_continuity_core(
    value: object,
) -> tuple[dict, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != _SCENE_CONTINUITY_CORE_KEYS:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    raw_components = value.get("components")
    raw_topology = value.get("topology")
    if (
        not isinstance(raw_components, list)
        or not raw_components
        or len(raw_components) > 30
        or not isinstance(raw_topology, list)
        or len(raw_topology) > 60
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    components = []
    for index, item in enumerate(raw_components, 1):
        if (
            not isinstance(item, dict)
            or set(item) != _SCENE_CONTINUITY_COMPONENT_KEYS
            or item.get("component_id") != f"COMPONENT_{index:02d}"
            or _COMPONENT_ID_RE.fullmatch(item["component_id"]) is None
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        target_spec = _canonical_text(item.get("target_spec"), max_bytes=1024)
        if _ENTITY_ID_MENTION_RE.search(target_spec) is not None:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        components.append({
            "component_id": item["component_id"],
            "target_spec": target_spec,
        })
    component_ids = tuple(item["component_id"] for item in components)
    topology = _canonical_component_relations(
        raw_topology,
        component_ids,
        _SCENE_CONTINUITY_TOPOLOGY_PREDICATES,
        topology=True,
    )
    return {"components": components, "topology": topology}, component_ids


def _canonical_scene_continuity_view(
    value: object, component_ids: tuple[str, ...],
) -> dict:
    if not isinstance(value, dict) or set(value) != _SCENE_CONTINUITY_VIEW_KEYS:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    segment_index = value.get("segment_index")
    frame_index = value.get("frame_index")
    transition = value.get("transition_from_previous")
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index < 0
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 1
        or not isinstance(transition, str)
        or transition not in _SCENE_CONTINUITY_TRANSITIONS
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    raw_observations = value.get("observations")
    if not isinstance(raw_observations, list) or len(raw_observations) != len(
        component_ids
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    observations = []
    visibility_by_component = {}
    for expected, item in zip(component_ids, raw_observations):
        if (
            not isinstance(item, dict)
            or set(item) != _SCENE_CONTINUITY_OBSERVATION_KEYS
            or item.get("component_id") != expected
            or not isinstance(item.get("visibility"), str)
            or item.get("visibility") not in _SCENE_CONTINUITY_VISIBILITIES
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        visibility_by_component[expected] = item["visibility"]
        observations.append({
            "component_id": expected,
            "visibility": item["visibility"],
        })
    raw_relations = value.get("view_relations")
    relations = _canonical_component_relations(
        raw_relations,
        component_ids,
        _SCENE_CONTINUITY_VIEW_PREDICATES,
        topology=False,
        visibility_by_component=visibility_by_component,
    )
    occluded_components = {
        item["object_id"] for item in relations if item["predicate"] == "occludes"
    }
    if any(
        (
            visibility in {"partial", "occluded"}
            and component_id not in occluded_components
        )
        or (
            component_id in occluded_components
            and visibility not in {"partial", "occluded", "edge_fragment"}
        )
        for component_id, visibility in visibility_by_component.items()
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {
        "segment_index": segment_index,
        "frame_index": frame_index,
        "transition_from_previous": transition,
        "observations": observations,
        "view_relations": relations,
    }


def _canonical_scene_continuity_graph(
    value: object,
    *,
    expected_view_keys: list[tuple[int, int]],
    previous_global_keys: dict[tuple[int, int], tuple[int, int] | None],
) -> dict:
    if not isinstance(value, dict) or set(value) != _SCENE_CONTINUITY_GRAPH_KEYS:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    core, component_ids = _canonical_scene_continuity_core({
        "components": value.get("components"),
        "topology": value.get("topology"),
    })
    raw_views = value.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != len(expected_view_keys):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    views = [
        _canonical_scene_continuity_view(item, component_ids)
        for item in raw_views
    ]
    view_keys = [
        (item["segment_index"], item["frame_index"])
        for item in views
    ]
    if view_keys != expected_view_keys:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    view_by_key = dict(zip(view_keys, views))
    for key, view in view_by_key.items():
        previous = previous_global_keys.get(key)
        if (view["transition_from_previous"] == "start") != (previous is None):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        if (
            view["transition_from_previous"] == "same_camera"
            and previous not in view_by_key
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
    if any(
        all(
            next(
                item["visibility"]
                for item in view["observations"]
                if item["component_id"] == component_id
            ) not in _SCENE_CONTINUITY_VISIBLE
            for view in views
        )
        for component_id in component_ids
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {**core, "views": views}


def _canonical_non_person_entity_ledger(
    value: object, allowed_person_ids: set[str], *, allow_sparse: bool = False,
) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != _NON_PERSON_ENTITY_LEDGER_KEYS
        or any(
            not isinstance(identifier, str)
            or _PERSON_ID_RE.fullmatch(identifier) is None
            for identifier in allowed_person_ids
        )
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    raw_entities = value.get("entities")
    raw_relations = value.get("relations")
    if (
        not isinstance(raw_entities, list)
        or len(raw_entities) > 30
        or not isinstance(raw_relations, list)
        or len(raw_relations) > 60
        or not allow_sparse and (not raw_entities or not raw_relations)
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    entities = []
    descriptions = set()
    entity_ids = set()
    for index, item in enumerate(raw_entities, 1):
        if not isinstance(item, dict) or set(item) != _ENTITY_LEDGER_ENTITY_KEYS:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        entity_id = item.get("entity_id")
        if entity_id != f"ENTITY_{index:02d}" or _ENTITY_ID_RE.fullmatch(entity_id) is None:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        description = _canonical_text(item.get("description"), max_bytes=512)
        visibility = item.get("visibility")
        if (
            description in descriptions
            or not isinstance(visibility, str)
            or visibility not in _ENTITY_VISIBILITIES
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        descriptions.add(description)
        entity_ids.add(entity_id)
        entities.append({
            "entity_id": entity_id,
            "description": description,
            "visibility": visibility,
        })

    identifiers = entity_ids | allowed_person_ids | {_PROJECT_ENTITY_OWNER_ID}
    relations = []
    physical_pairs = set()
    occlusion_pairs = set()
    ownership_subjects = set()
    directed_edges = {"supports": [], "occludes": []}
    for item in raw_relations:
        if not isinstance(item, dict) or set(item) != _ENTITY_LEDGER_RELATION_KEYS:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        subject = item.get("subject_id")
        predicate = item.get("predicate")
        object_ = item.get("object_id")
        if (
            not isinstance(subject, str)
            or not isinstance(object_, str)
            or subject == object_
            or subject not in identifiers
            or object_ not in identifiers
            or (subject not in entity_ids and object_ not in entity_ids)
            or not isinstance(predicate, str)
            or predicate not in _ENTITY_RELATION_PREDICATES
            or (
                predicate == "owned_by"
                and (
                    subject not in entity_ids
                    or object_ not in (
                        allowed_person_ids | {_PROJECT_ENTITY_OWNER_ID}
                    )
                )
            )
            or (
                predicate != "owned_by"
                and _PROJECT_ENTITY_OWNER_ID in {subject, object_}
            )
            or (
                predicate in {"contacts", "separate_from"}
                and subject >= object_
            )
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        pair = tuple(sorted((subject, object_)))
        if predicate in _PHYSICAL_ENTITY_RELATION_PREDICATES:
            if pair in physical_pairs:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            physical_pairs.add(pair)
        elif predicate == "occludes":
            if pair in occlusion_pairs:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            occlusion_pairs.add(pair)
        else:
            if subject in ownership_subjects:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            ownership_subjects.add(subject)
        if predicate in directed_edges:
            directed_edges[predicate].append((subject, object_))
        relations.append({
            "subject_id": subject,
            "predicate": predicate,
            "object_id": object_,
        })
    if relations != sorted(
        relations,
        key=lambda item: (item["subject_id"], item["predicate"], item["object_id"]),
    ) or any(_contains_directed_cycle(edges) for edges in directed_edges.values()) or (
        not allow_sparse and {
            identifier
            for relation in relations
            for identifier in (relation["subject_id"], relation["object_id"])
            if identifier in entity_ids
        } != entity_ids
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {"entities": entities, "relations": relations}


def _canonical_dominant_palette_contract(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != set(_DOMINANT_PALETTE_CONTRACT_KEYS)
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    family = value.get("area_weighted_warm_cool_family")
    saturation = value.get("saturation_style")
    if (
        not isinstance(family, str)
        or family not in _AREA_WEIGHTED_WARM_COOL_FAMILIES
        or not isinstance(saturation, str)
        or saturation not in _SATURATION_STYLES
    ):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {
        "area_weighted_warm_cool_family": family,
        "saturation_style": saturation,
    }


def _canonical_frame_constraint(
    value: object, allowed_person_ids: set[str], *, allow_sparse: bool = False,
) -> dict:
    if not isinstance(value, dict) or set(value) != set(_FRAME_CONSTRAINT_KEYS):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    frame_index = value.get("frame_index")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 1:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {
        "frame_index": frame_index,
        **{
            key: _canonical_text(value.get(key))
            for key in _FRAME_TEXT_CONSTRAINT_KEYS
        },
        "non_person_entity_ledger": _canonical_non_person_entity_ledger(
            value.get("non_person_entity_ledger"), allowed_person_ids,
            allow_sparse=allow_sparse,
        ),
        "dominant_palette_contract": _canonical_dominant_palette_contract(
            value.get("dominant_palette_contract")
        ),
    }


def _canonical_plan_v3(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
    *,
    allow_sparse_facts: bool = False,
) -> dict:
    if not isinstance(value, dict) or value.get("version") != 3:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    v2_value = deepcopy(value)
    v2_value["version"] = 2
    v2_segments = []
    for segment in raw_segments:
        if not isinstance(segment, dict) or set(segment) != {
            "segment_index",
            "persons",
            "scene",
            "protected_non_target_people",
            "protected_relations",
            "frame_constraints",
            "photometric_contract",
        }:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        v2_segments.append({
            key: deepcopy(segment[key])
            for key in (
                "segment_index",
                "persons",
                "scene",
                "protected_non_target_people",
                "protected_relations",
            )
        })
    v2_value["segments"] = v2_segments
    canonical = _canonical_plan_v2(
        v2_value, expected_indices, frame_counts,
        allow_empty_people=allow_sparse_facts,
    )
    if not canonical["eligible"]:
        return {**canonical, "version": 3}
    segments = []
    for base, raw in zip(canonical["segments"], raw_segments):
        raw_constraints = raw["frame_constraints"]
        if not isinstance(raw_constraints, list) or not raw_constraints:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        constraints = []
        seen = set()
        for item in raw_constraints:
            raw_index = item.get("frame_index") if isinstance(item, dict) else None
            allowed_person_ids = {
                person["id"]
                for person in base["persons"]
                if raw_index in person["observable_frames"]
            }
            constraint = _canonical_frame_constraint(
                item, allowed_person_ids, allow_sparse=allow_sparse_facts,
            )
            index = constraint["frame_index"]
            if (
                index in seen
                or (
                    frame_counts is not None
                    and index > frame_counts[base["segment_index"]]
                )
            ):
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            seen.add(index)
            constraints.append(constraint)
        if [item["frame_index"] for item in constraints] != sorted(seen):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        if frame_counts is not None and seen != set(
            range(1, frame_counts[base["segment_index"]] + 1)
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        raw_photo = raw["photometric_contract"]
        if not isinstance(raw_photo, dict) or set(raw_photo) != set(
            _PHOTOMETRIC_CONTRACT_KEYS
        ):
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        segments.append({
            **base,
            "frame_constraints": constraints,
            "photometric_contract": {
                key: _canonical_text(raw_photo.get(key))
                for key in _PHOTOMETRIC_CONTRACT_KEYS
            },
        })
    return {**canonical, "version": 3, "segments": segments}


def canonical_plan_v3(
    value: object,
    segment_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
    """Return an isolated v3 plan with one hard constraint record per frame."""
    return deepcopy(_canonical_plan_v3(value, segment_indices, frame_counts))


def _canonical_plan_v4(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
    if not isinstance(value, dict) or value.get("version") != 4:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    raw_scenes = value.get("scene_plans")
    if not isinstance(raw_scenes, list):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    v3_value = deepcopy(value)
    v3_value["version"] = 3
    v3_scenes = []
    for scene in raw_scenes:
        if not isinstance(scene, dict) or "continuity_graph" not in scene:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        projected = deepcopy(scene)
        projected.pop("continuity_graph")
        v3_scenes.append(projected)
    v3_value["scene_plans"] = v3_scenes
    canonical = _canonical_plan_v3(
        v3_value, expected_indices, frame_counts, allow_sparse_facts=True,
    )
    if not canonical["eligible"]:
        return {**canonical, "version": 4}

    global_view_keys = [
        (segment["segment_index"], constraint["frame_index"])
        for segment in canonical["segments"]
        for constraint in segment["frame_constraints"]
    ]
    previous_global_keys = {
        key: global_view_keys[index - 1] if index else None
        for index, key in enumerate(global_view_keys)
    }
    scene_for_key = {
        (segment["segment_index"], constraint["frame_index"]):
        segment["scene"]["scene_id"]
        for segment in canonical["segments"]
        for constraint in segment["frame_constraints"]
    }
    scenes = []
    for base, raw in zip(canonical["scene_plans"], raw_scenes):
        expected_view_keys = [
            key for key in global_view_keys if scene_for_key[key] == base["id"]
        ]
        graph = _canonical_scene_continuity_graph(
            raw.get("continuity_graph"),
            expected_view_keys=expected_view_keys,
            previous_global_keys=previous_global_keys,
        )
        scenes.append({**base, "continuity_graph": graph})
    return {**canonical, "version": 4, "scene_plans": scenes}


def canonical_plan_v4(
    value: object,
    segment_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
    """Return an isolated v4 plan with scene-level target continuity authority."""
    return deepcopy(_canonical_plan_v4(value, segment_indices, frame_counts))


def _canonical_plan(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
    if isinstance(value, dict) and value.get("version") == 4:
        return _canonical_plan_v4(value, expected_indices, frame_counts)
    if isinstance(value, dict) and value.get("version") == 3:
        return _canonical_plan_v3(value, expected_indices, frame_counts)
    return _canonical_plan_v2(value, expected_indices, frame_counts)


def _validated_frames(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError("invalid image optimization keyframes directory")
    frames = sorted(source.glob("[0-9][0-9].png"))
    expected = [f"{index:02d}.png" for index in range(1, len(frames) + 1)]
    if not frames or len(frames) > 9 or [frame.name for frame in frames] != expected:
        raise ValueError("invalid image optimization keyframes")
    return frames


def source_palette_metric(path: Path) -> dict:
    """Measure one decoded source canvas for backend-owned prompt facts."""
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        delta = float((
            lab[:, :, 2].mean() - PALETTE_METRIC_THRESHOLDS["lab_b_star_neutral"]
        ) / PALETTE_METRIC_THRESHOLDS["lab_b_star_scale"])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean() / 255.0)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (cv2.error, OSError, ValueError):
        raise ValueError("invalid image optimization source frame") from None
    threshold = PALETTE_METRIC_THRESHOLDS["warm_cool_delta"]
    family = "warm" if delta > threshold else "cool" if delta < -threshold else "balanced"
    muted = PALETTE_METRIC_THRESHOLDS["muted_saturation"]
    vivid = PALETTE_METRIC_THRESHOLDS["vivid_saturation"]
    style = "muted" if saturation < muted else "vivid" if saturation > vivid else "natural"
    return {
        "bytes_sha256": digest,
        "warm_cool_family": family,
        "saturation_style": style,
        "mean_lab_b_star": round(delta, 6),
        "mean_saturation": round(saturation, 6),
    }


def _bind_source_palette_contracts(
    plan: dict, source_frames: dict[int, list[Path]],
) -> dict:
    if plan.get("version") != 4:
        return plan
    bound = deepcopy(plan)
    segments = {item["segment_index"]: item for item in bound["segments"]}
    if set(source_frames) != set(segments):
        raise ValueError("invalid image optimization source frames")
    for index, frames in source_frames.items():
        constraints = segments[index]["frame_constraints"]
        if len(frames) != len(constraints):
            raise ValueError("invalid image optimization source frames")
        for position, (path, constraint) in enumerate(zip(frames, constraints), 1):
            if path.name != f"{position:02d}.png" or constraint["frame_index"] != position:
                raise ValueError("invalid image optimization source frames")
            metric = source_palette_metric(path)
            constraint["dominant_palette_contract"] = {
                "area_weighted_warm_cool_family": metric["warm_cool_family"],
                "saturation_style": metric["saturation_style"],
            }
    return bound


def _semantic_text(value: object, default: str, *, max_bytes: int = 1000) -> str:
    """Return bounded model semantics without turning incompleteness into control flow."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        text = default
    raw = text.encode("utf-8")
    if len(raw) > max_bytes:
        text = raw[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    return text or default


def _canonical_element_index(value: object) -> dict:
    """Keep valid first-Skill facts without turning missing semantics into control flow."""
    raw_index = value if isinstance(value, dict) else {}
    result = {"people": {}, "entities": {}, "scenes": {}}
    for category in ("people", "entities", "scenes"):
        raw_items = raw_index.get(category)
        if not isinstance(raw_items, dict):
            continue
        items = {}
        for stable_key, raw in raw_items.items():
            if (
                not isinstance(stable_key, str)
                or not stable_key.strip()
                or stable_key != stable_key.strip()
                or not isinstance(raw, dict)
                or not isinstance(raw.get("source_visual_description"), str)
                or not raw["source_visual_description"].strip()
            ):
                continue
            raw_occurrences = raw.get("occurrences")
            raw_replaceable = raw.get("replaceable")
            raw_preserve = raw.get("preserve")
            if not isinstance(raw_occurrences, list):
                raw_occurrences = []
            if not isinstance(raw_replaceable, list):
                raw_replaceable = []
            if not isinstance(raw_preserve, list):
                raw_preserve = []
            occurrences = []
            seen_segments = set()
            for occurrence in raw_occurrences:
                if (
                    not isinstance(occurrence, dict)
                    or isinstance(occurrence.get("segment_index"), bool)
                    or not isinstance(occurrence.get("segment_index"), int)
                    or occurrence["segment_index"] < 0
                    or occurrence["segment_index"] in seen_segments
                    or not isinstance(occurrence.get("frame_orders"), list)
                    or not occurrence["frame_orders"]
                    or any(
                        isinstance(order, bool) or not isinstance(order, int)
                        or order < 1 for order in occurrence["frame_orders"]
                    )
                    or occurrence["frame_orders"]
                    != sorted(set(occurrence["frame_orders"]))
                ):
                    continue
                seen_segments.add(occurrence["segment_index"])
                occurrences.append({
                    "segment_index": occurrence["segment_index"],
                    "frame_orders": list(occurrence["frame_orders"]),
                })
            occurrences.sort(key=lambda item: item["segment_index"])
            items[stable_key] = {
                "source_visual_description": raw["source_visual_description"].strip(),
                "occurrences": occurrences,
                "replaceable": [
                    text.strip() for text in raw_replaceable
                    if isinstance(text, str) and text.strip()
                ],
                "preserve": [
                    text.strip() for text in raw_preserve
                    if isinstance(text, str) and text.strip()
                ],
            }
        result[category] = items
    return result


def _tag_stable_key(stable_key: str, text: str) -> str:
    return f"{_STABLE_KEY_PREFIX}{stable_key}；{text}"


def _split_stable_key(value: object) -> tuple[str | None, str]:
    text = value.strip() if isinstance(value, str) else ""
    if not text.startswith(_STABLE_KEY_PREFIX) or "；" not in text:
        return None, text
    tagged, description = text.split("；", 1)
    stable_key = tagged[len(_STABLE_KEY_PREFIX):].strip()
    return (stable_key or None), description.strip()


def _semantic_slots(
    segment_specs: list[dict],
    *,
    source_frames: dict[int, list[Path]] | None = None,
    element_index: dict | None = None,
) -> dict:
    """Construct every stable scene/frame key and transition from backend input."""
    if not isinstance(segment_specs, list) or not segment_specs:
        raise ValueError("invalid image semantic compiler input")
    scene_key_by_chain: dict[str, str] = {}
    scene_slots: list[dict] = []
    frame_slots: list[dict] = []
    frame_ordinal = 0
    prior_exists = False
    indexed_scenes = (
        element_index.get("scenes", {})
        if isinstance(element_index, dict) else {}
    )

    def indexed_scene_key(segment_index: int) -> str | None:
        matches = [
            stable_key
            for stable_key, item in indexed_scenes.items()
            if any(
                occurrence.get("segment_index") == segment_index
                for occurrence in item.get("occurrences", [])
                if isinstance(occurrence, dict)
            )
        ]
        return sorted(matches)[0] if matches else None

    for segment in segment_specs:
        if not isinstance(segment, dict):
            raise ValueError("invalid image semantic compiler input")
        index = segment.get("index")
        chain_id = segment.get("chain_id")
        join_mode = segment.get("join_mode")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(chain_id, str)
            or not chain_id
            or join_mode not in {"hard_cut", "continue"}
        ):
            raise ValueError("invalid image semantic compiler input")
        scene_key = indexed_scene_key(index) or scene_key_by_chain.get(chain_id)
        if scene_key is None:
            scene_key = f"scene-{len(scene_slots) + 1:03d}"
        scene_key_by_chain[chain_id] = scene_key
        if not any(item["key"] == scene_key for item in scene_slots):
            scene_slots.append({
                "key": scene_key,
                "chain_id": chain_id,
                "segment_indices": [],
            })
        next(
            item for item in scene_slots if item["key"] == scene_key
        )["segment_indices"].append(index)

        skeleton = segment.get("transition_skeleton")
        paths = None if source_frames is None else source_frames.get(index)
        if not isinstance(skeleton, list) or not skeleton:
            if not isinstance(paths, list) or not paths:
                raise ValueError("invalid image semantic compiler input")
            skeleton = [
                {
                    "frame_index": position,
                    "frame_name": path.name,
                    "source_transition_from_previous": (
                        "start" if not prior_exists and position == 1
                        else "hard_cut" if position == 1 and join_mode == "hard_cut"
                        else "same_camera"
                    ),
                }
                for position, path in enumerate(paths, 1)
            ]
        if paths is not None and len(paths) != len(skeleton):
            raise ValueError("invalid image semantic compiler input")
        for position, item in enumerate(skeleton, 1):
            if not isinstance(item, dict):
                raise ValueError("invalid image semantic compiler input")
            transition = item.get("source_transition_from_previous")
            if transition not in _SCENE_CONTINUITY_TRANSITIONS:
                raise ValueError("invalid image semantic compiler input")
            frame_ordinal += 1
            frame_slots.append({
                "key": f"frame-{frame_ordinal:03d}",
                "scene_key": scene_key,
                "segment_index": index,
                "frame_index": position,
                "frame_name": item.get("frame_name", f"{position:02d}.png"),
                "transition_from_previous": transition,
            })
            prior_exists = True
    indices = [item.get("index") for item in segment_specs]
    if indices not in ([0], list(range(1, len(indices) + 1))):
        raise ValueError("invalid image semantic compiler input")
    return {
        "scenes": scene_slots,
        "frames": frame_slots,
    }


def semantic_slot_manifest(
    segment_specs: list[dict], *, element_index: dict | None = None,
) -> dict:
    """Expose backend-owned stable keys that the Skill associates with semantics."""
    slots = _semantic_slots(segment_specs, element_index=element_index)
    return {
        "scenes": [
            {"key": item["key"], "chain_id": item["chain_id"]}
            for item in slots["scenes"]
        ],
        "frames": [
            {
                "key": item["key"],
                "scene_key": item["scene_key"],
                "path": (
                    f"work/segments/{item['segment_index']}/keyframes/"
                    f"{item['frame_name']}"
                ),
            }
            for item in slots["frames"]
        ],
    }


def compile_semantic_plan(
    value: object,
    segment_specs: list[dict],
    *,
    source_frames: dict[int, list[Path]] | None = None,
    element_index: dict | None = None,
) -> tuple[dict, dict]:
    """Compile tolerant visual semantics into the one canonical executable v4 plan."""
    slots = _semantic_slots(
        segment_specs,
        source_frames=source_frames,
        element_index=element_index,
    )
    raw = value if isinstance(value, dict) else {}
    raw_people = raw.get("people") if isinstance(raw.get("people"), dict) else {}
    raw_entities = (
        raw.get("entities") if isinstance(raw.get("entities"), dict) else {}
    )
    raw_scenes = raw.get("scenes") if isinstance(raw.get("scenes"), dict) else {}
    raw_frames = raw.get("frames") if isinstance(raw.get("frames"), dict) else {}
    indexed_people = set((element_index or {}).get("people", {}))
    indexed_entities = set((element_index or {}).get("entities", {}))
    indexed_scenes = set((element_index or {}).get("scenes", {}))
    frame_by_key = {item["key"]: item for item in slots["frames"]}
    frames_by_segment: dict[int, list[dict]] = {}
    for slot in slots["frames"]:
        frames_by_segment.setdefault(slot["segment_index"], []).append(slot)

    expected_fields = 0
    present_fields = 0
    issues: list[str] = []
    ignored_mechanical_fields: list[str] = []
    entity_source_preserve_defaults: list[str] = []
    appearance_source_preserve_defaults: list[str] = []
    classified_derived_appearances = 0

    def field(
        container: object,
        key: str,
        default: str,
        *,
        path: str,
    ) -> str:
        nonlocal expected_fields, present_fields
        expected_fields += 1
        candidate = container.get(key) if isinstance(container, dict) else None
        if isinstance(candidate, str) and candidate.strip():
            present_fields += 1
        else:
            issues.append(f"missing:{path}.{key}")
        return _semantic_text(candidate, default)

    for key in ("version", "phase", "segment_indices", "eligible", "reason"):
        if key in raw:
            ignored_mechanical_fields.append(key)
    if not isinstance(raw.get("entities"), dict):
        entity_source_preserve_defaults.append("entities")
    for entity_key, design in raw_entities.items():
        if not isinstance(design, dict):
            continue
        for key in design:
            if key in {"entity_id", "relations", "graph", "visibility"}:
                ignored_mechanical_fields.append(f"entities.{entity_key}.{key}")
    for frame_key, frame in raw_frames.items():
        if not isinstance(frame, dict):
            continue
        for key in frame:
            if "palette" in key.lower() or key in {
                "frame_index", "segment_index", "transition_from_previous",
            }:
                ignored_mechanical_fields.append(f"frames.{frame_key}.{key}")
        frame_entities = frame.get("entities")
        if not isinstance(frame_entities, dict):
            continue
        for entity_key, observation in frame_entities.items():
            if not isinstance(observation, dict):
                continue
            for key in observation:
                if key in {
                    "entity_id", "relations", "graph", "predicate", "owner_id",
                }:
                    ignored_mechanical_fields.append(
                        f"frames.{frame_key}.entities.{entity_key}.{key}"
                    )
        frame_people = frame.get("people")
        if not isinstance(frame_people, dict):
            continue
        for person_key, person in frame_people.items():
            if not isinstance(person, dict):
                continue
            derived = person.get("derived_observations")
            if not isinstance(derived, dict):
                continue
            for observation_key, observation in derived.items():
                if not isinstance(observation, dict):
                    continue
                for key in observation:
                    if key in {
                        "observation_id", "person_id", "physicality",
                        "instantiation",
                    }:
                        ignored_mechanical_fields.append(
                            f"frames.{frame_key}.people.{person_key}."
                            f"derived_observations.{observation_key}.{key}"
                        )

    observations: dict[str, dict[str, dict]] = {}
    person_keys = {
        key for key in raw_people if isinstance(key, str) and key.strip()
    }
    for frame_key in frame_by_key:
        frame = raw_frames.get(frame_key)
        frame_people = frame.get("people") if isinstance(frame, dict) else None
        if not isinstance(frame_people, dict):
            continue
        for person_key, observation in frame_people.items():
            if not isinstance(person_key, str) or not person_key.strip():
                continue
            person_keys.add(person_key)
            observations.setdefault(person_key, {})[frame_key] = (
                observation if isinstance(observation, dict) else {}
            )

    entity_observations: dict[str, dict[str, dict]] = {}
    for frame_key in frame_by_key:
        frame = raw_frames.get(frame_key)
        frame_entities = frame.get("entities") if isinstance(frame, dict) else None
        if not isinstance(frame_entities, dict):
            continue
        for entity_key, observation in frame_entities.items():
            if not isinstance(entity_key, str) or not entity_key.strip():
                continue
            entity_observations.setdefault(entity_key, {})[frame_key] = (
                observation if isinstance(observation, dict) else {}
            )

    observed_person_keys = sorted(
        key for key in person_keys if observations.get(key)
    )
    for key in sorted(person_keys - set(observed_person_keys)):
        issues.append(f"unused_person:{key}")
    person_id_by_key = {
        key: f"PERSON_{position:02d}"
        for position, key in enumerate(observed_person_keys, 1)
    }
    person_plans = []
    for key in observed_person_keys:
        design = raw_people.get(key)
        path = f"people.{key}"
        source_identity = field(
            design, "source_identity", f"源帧中由 {key} 标识的可见叙事人物",
            path=path,
        )
        replacement_identity = field(
            design, "replacement_identity", "与源身份明显不同且跨帧稳定的新人物",
            path=path,
        )
        if replacement_identity == source_identity:
            replacement_identity = f"与源身份不同的新人物：{replacement_identity}"
            issues.append(f"unchanged_identity:{key}")
        if key in indexed_people:
            replacement_identity = _tag_stable_key(key, replacement_identity)
        first_key = next(
            slot["key"] for slot in slots["frames"]
            if slot["key"] in observations[key]
        )
        first = frame_by_key[first_key]
        observable_segments = sorted({
            frame_by_key[frame_key]["segment_index"]
            for frame_key in observations[key]
            if frame_key in frame_by_key
        })
        person_plans.append({
            "id": person_id_by_key[key],
            "source_identity": source_identity,
            "replacement_identity": replacement_identity,
            "wardrobe_change": field(
                design, "wardrobe_change",
                "保持源可见覆盖边界并使用明显不同的新服装设计",
                path=path,
            ),
            "local_color_change": field(
                design, "local_color_change",
                "改变人物局部固有色并保持当前源帧全局光色",
                path=path,
            ),
            "reference": {
                "segment_index": first["segment_index"],
                "frame_index": first["frame_index"],
            },
            "observable_segments": observable_segments,
        })

    entity_keys_by_scene: dict[str, list[str]] = {}
    for entity_key, frame_observations in entity_observations.items():
        for scene_key in {
            frame_by_key[frame_key]["scene_key"]
            for frame_key in frame_observations
            if frame_key in frame_by_key
        }:
            entity_keys_by_scene.setdefault(scene_key, []).append(entity_key)
    for entity_key in sorted(set(raw_entities) - set(entity_observations)):
        issues.append(f"unused_entity:{entity_key}")
    for scene_key, entity_keys in entity_keys_by_scene.items():
        ordered = sorted(set(entity_keys))
        if len(ordered) > 30:
            issues.append(f"entity_limit:{scene_key}")
        entity_keys_by_scene[scene_key] = ordered[:30]

    entity_designs: dict[str, dict] = {}
    compiled_entity_keys = sorted({
        entity_key
        for entity_keys in entity_keys_by_scene.values()
        for entity_key in entity_keys
    })
    for entity_key in compiled_entity_keys:
        design = raw_entities.get(entity_key)
        path = f"entities.{entity_key}"
        if not isinstance(design, dict):
            entity_source_preserve_defaults.append(path)
            design = {}
        description = design.get("description")
        if not isinstance(description, str) or not description.strip():
            entity_source_preserve_defaults.append(f"{path}.description")
        owner = design.get("owner")
        if not isinstance(owner, str) or not owner.strip():
            entity_source_preserve_defaults.append(f"{path}.owner")
            owner = "source-preserve"
        persistence = design.get("persistence")
        if not isinstance(persistence, str) or not persistence.strip():
            entity_source_preserve_defaults.append(f"{path}.persistence")
        association = design.get("association")
        if not isinstance(association, str) or not association.strip():
            entity_source_preserve_defaults.append(f"{path}.association")
        owner_id = (
            _PROJECT_ENTITY_OWNER_ID if owner.strip() == "project"
            else person_id_by_key.get(owner.strip())
        )
        if owner_id is None:
            issues.append(f"source_preserve:{path}.owner")
        entity_designs[entity_key] = {
            "description": _semantic_text(
                (
                    _tag_stable_key(entity_key, description.strip())
                    if entity_key in indexed_entities
                    and isinstance(description, str) and description.strip()
                    else description
                ),
                "source-preserve/no-invention：保持源帧中该持久实体的外观身份",
                max_bytes=512,
            ),
            "owner_id": owner_id,
            "association": _semantic_text(
                association,
                "source-preserve/no-invention：保持源帧中该实体的归属方式",
                max_bytes=512,
            ),
            "persistence": _semantic_text(
                persistence,
                "source-preserve/no-invention：不因暂时不可见而新增、删除或替换实体",
                max_bytes=512,
            ),
        }

    entity_ids_by_scene: dict[str, dict[str, str]] = {}
    entity_descriptions_by_scene: dict[str, dict[str, str]] = {}
    for scene_key, entity_keys in entity_keys_by_scene.items():
        entity_ids_by_scene[scene_key] = {
            entity_key: f"ENTITY_{position:02d}"
            for position, entity_key in enumerate(entity_keys, 1)
        }
        descriptions = {}
        seen_descriptions = set()
        for position, entity_key in enumerate(entity_keys, 1):
            description = entity_designs[entity_key]["description"]
            if description in seen_descriptions:
                description = f"{description}；独立持久实体 {position}"
                issues.append(f"duplicate_entity_description:{scene_key}")
            seen_descriptions.add(description)
            descriptions[entity_key] = description
        entity_descriptions_by_scene[scene_key] = descriptions

    scene_id_by_key = {
        item["key"]: f"SCENE_{position:02d}"
        for position, item in enumerate(slots["scenes"], 1)
    }
    scene_plans = []
    for scene_slot in slots["scenes"]:
        key = scene_slot["key"]
        design = raw_scenes.get(key)
        path = f"scenes.{key}"
        source_scene = field(
            design, "source_scene", "当前源帧可见的真实环境", path=path,
        )
        replacement_scene = field(
            design, "replacement_scene",
            "叙事用途相同、结构和设计明显不同的真实新环境",
            path=path,
        )
        if replacement_scene == source_scene:
            replacement_scene = f"结构与源环境不同的真实新环境：{replacement_scene}"
            issues.append(f"unchanged_scene:{key}")
        if key in indexed_scenes:
            replacement_scene = _tag_stable_key(key, replacement_scene)
        semantic_change = field(
            design, "semantic_change", "保持叙事用途并改变环境语义", path=path,
        )
        geometry_change = field(
            design, "geometry_change", "改变可见环境形状与空间结构", path=path,
        )
        depth_change = field(
            design, "depth_change", "改变前中后景的纵深组织", path=path,
        )
        layout_change = field(
            design, "layout_change", "改变功能区域和实体的布局", path=path,
        )
        local_color_change = field(
            design, "local_color_change",
            "改变场景局部材质固有色并保持源帧全局光色", path=path,
        )
        members = list(scene_slot["segment_indices"])
        scene_frames = [
            item for item in slots["frames"] if item["scene_key"] == key
        ]
        reference = scene_frames[0]
        component_spec = _semantic_text(
            "；".join((
                replacement_scene,
                semantic_change,
                geometry_change,
                depth_change,
                layout_change,
                local_color_change,
            )),
            "跨帧稳定的真实新环境",
        )
        scene_plans.append({
            "id": scene_id_by_key[key],
            "source_scene": source_scene,
            "replacement_scene": replacement_scene,
            "semantic_change": semantic_change,
            "geometry_changes": [geometry_change],
            "depth_changes": [depth_change],
            "layout_changes": [layout_change],
            "local_color_change": local_color_change,
            "reference": {
                "segment_index": reference["segment_index"],
                "frame_index": reference["frame_index"],
            },
            "segments": members,
            "continuity_graph": {
                "components": [{
                    "component_id": "COMPONENT_01",
                    "target_spec": component_spec,
                }],
                "topology": [],
                "views": [{
                    "segment_index": frame["segment_index"],
                    "frame_index": frame["frame_index"],
                    "transition_from_previous": frame["transition_from_previous"],
                    "observations": [{
                        "component_id": "COMPONENT_01",
                        "visibility": "edge_fragment",
                    }],
                    "view_relations": [],
                } for frame in scene_frames],
            },
        })

    segment_scene_key = {
        index: item["key"]
        for item in slots["scenes"]
        for index in item["segment_indices"]
    }
    segments = []
    for segment_spec in segment_specs:
        index = segment_spec["index"]
        segment_frames = frames_by_segment[index]
        segment_people = []
        for person_key in observed_person_keys:
            observed = [
                frame["frame_index"] for frame in segment_frames
                if frame["key"] in observations[person_key]
            ]
            if observed:
                regions = []
                boundaries = []
                for frame in segment_frames:
                    detail = observations[person_key].get(frame["key"])
                    if detail is None:
                        continue
                    detail_path = f"frames.{frame['key']}.people.{person_key}"
                    regions.append(field(
                        detail, "visible_region", "当前帧可见人物区域",
                        path=detail_path,
                    ))
                    boundaries.append(field(
                        detail, "boundary", "当前帧可见人物边界与裁切",
                        path=detail_path,
                    ))
                    field(
                        detail, "body_and_pose", "保持当前帧可见身体和姿态",
                        path=detail_path,
                    )
                target_region = _semantic_text("；".join(dict.fromkeys(regions)), "可见人物区域")
                boundary = _semantic_text("；".join(dict.fromkeys(boundaries)), "可见人物边界")
                state = "replace"
            else:
                target_region = None
                boundary = None
                state = "not_observable"
            segment_people.append({
                "id": person_id_by_key[person_key],
                "state": state,
                "observable_frames": observed,
                "target_region": target_region,
                "boundary": boundary,
            })

        scene_key = segment_scene_key[index]
        scene_entity_keys = entity_keys_by_scene.get(scene_key, [])
        scene_entity_ids = entity_ids_by_scene.get(scene_key, {})
        scene_entity_descriptions = entity_descriptions_by_scene.get(
            scene_key, {}
        )
        constraints = []
        for frame in segment_frames:
            frame_value = raw_frames.get(frame["key"])
            frame_path = f"frames.{frame['key']}"
            relationships = field(
                frame_value, "relationships",
                "保持当前源帧可见的接触、支撑、遮挡与前后关系；不补造未知关系",
                path=frame_path,
            )
            crop = field(
                frame_value, "crop",
                "保持当前源帧画幅、出画裁切和可见边界；不补全画外内容",
                path=frame_path,
            )
            entity_notes = _semantic_text(
                frame_value.get("entities")
                if isinstance(frame_value, dict)
                and isinstance(frame_value.get("entities"), str)
                else None,
                "保持当前源帧可见非人物实体的数量、位置与边界",
            )
            frame_entity_semantics = (
                frame_value.get("entities")
                if isinstance(frame_value, dict)
                and isinstance(frame_value.get("entities"), dict)
                else {}
            )
            current_person_ids = {
                person_id_by_key[person_key]
                for person_key in observed_person_keys
                if frame["key"] in observations[person_key]
            }
            ledger_entities = []
            ledger_relations = []
            entity_relation_notes = []
            for entity_key in scene_entity_keys:
                observation = frame_entity_semantics.get(entity_key)
                observation_path = f"{frame_path}.entities.{entity_key}"
                missing_observation = not isinstance(observation, dict)
                if missing_observation:
                    entity_source_preserve_defaults.append(observation_path)
                    observation = {}
                raw_visibility = observation.get("visibility")
                visibility = (
                    _SEMANTIC_ENTITY_VISIBILITIES.get(
                        raw_visibility.strip().lower()
                    )
                    if isinstance(raw_visibility, str)
                    else None
                )
                if visibility is None:
                    visibility = "source_preserve"
                    if not missing_observation:
                        entity_source_preserve_defaults.append(
                            f"{observation_path}.visibility"
                        )
                relationship = observation.get("relationship")
                if not isinstance(relationship, str) or not relationship.strip():
                    relationship = (
                        "source-preserve/no-invention：保持当前源帧中该实体的可见关系"
                    )
                    if not missing_observation:
                        entity_source_preserve_defaults.append(
                            f"{observation_path}.relationship"
                        )
                entity_id = scene_entity_ids[entity_key]
                ledger_entities.append({
                    "entity_id": entity_id,
                    "description": scene_entity_descriptions[entity_key],
                    "visibility": visibility,
                })
                owner_id = entity_designs[entity_key]["owner_id"]
                if owner_id == _PROJECT_ENTITY_OWNER_ID or owner_id in current_person_ids:
                    ledger_relations.append({
                        "subject_id": entity_id,
                        "predicate": "owned_by",
                        "object_id": owner_id,
                    })
                elif owner_id is not None:
                    entity_source_preserve_defaults.append(
                        f"{observation_path}.owner_relation"
                    )
                entity_relation_notes.append(
                    f"{entity_id}：{entity_designs[entity_key]['association']}；"
                    f"{_semantic_text(relationship, '')}；"
                    f"{entity_designs[entity_key]['persistence']}"
                )
            ledger_relations.sort(
                key=lambda item: (
                    item["subject_id"], item["predicate"], item["object_id"]
                )
            )
            visible_parts = []
            poses = []
            derived_appearances = []
            for person_key in observed_person_keys:
                detail = observations[person_key].get(frame["key"])
                if detail is None:
                    continue
                body_pose = _semantic_text(
                    detail.get("body_and_pose") if isinstance(detail, dict) else None,
                    "保持当前帧可见身体和姿态",
                )
                visible_region = _semantic_text(
                    detail.get("visible_region") if isinstance(detail, dict) else None,
                    "当前帧可见人物区域",
                )
                boundary = _semantic_text(
                    detail.get("boundary") if isinstance(detail, dict) else None,
                    "当前帧人物、服装、遮挡与画边共同形成的可见边界",
                )
                visible_parts.append(
                    f"{person_id_by_key[person_key]}：{visible_region}；"
                    f"当前可见边界={boundary}"
                )
                poses.append(f"{person_id_by_key[person_key]}：{body_pose}")
                derived_path = (
                    f"frames.{frame['key']}.people.{person_key}."
                    "derived_observations"
                )
                raw_derived = (
                    detail.get("derived_observations")
                    if isinstance(detail, dict) else None
                )
                if not isinstance(raw_derived, dict):
                    appearance_source_preserve_defaults.append(derived_path)
                    raw_derived = {}
                for observation_key in sorted(raw_derived):
                    observation = raw_derived[observation_key]
                    observation_path = f"{derived_path}.{observation_key}"
                    if not isinstance(observation, dict):
                        appearance_source_preserve_defaults.append(
                            observation_path
                        )
                        continue
                    raw_mode = observation.get("mode")
                    mode = (
                        raw_mode.strip().lower()
                        if isinstance(raw_mode, str) else ""
                    )
                    if mode not in _SEMANTIC_DERIVED_APPEARANCE_MODES:
                        mode = "source_preserve"
                        appearance_source_preserve_defaults.append(
                            f"{observation_path}.mode"
                        )
                    source_carrier = observation.get("source_carrier")
                    if not isinstance(source_carrier, str) or not source_carrier.strip():
                        appearance_source_preserve_defaults.append(
                            f"{observation_path}.source_carrier"
                        )
                    derived_region = observation.get("visible_region")
                    if not isinstance(derived_region, str) or not derived_region.strip():
                        appearance_source_preserve_defaults.append(
                            f"{observation_path}.visible_region"
                        )
                    derived_boundary = observation.get("boundary")
                    if not isinstance(derived_boundary, str) or not derived_boundary.strip():
                        appearance_source_preserve_defaults.append(
                            f"{observation_path}.boundary"
                        )
                    relationship = observation.get("relationship")
                    if not isinstance(relationship, str) or not relationship.strip():
                        appearance_source_preserve_defaults.append(
                            f"{observation_path}.relationship"
                        )
                    derived_appearances.append({
                        "observation_id": (
                            f"OBSERVATION_{len(derived_appearances) + 1:02d}"
                        ),
                        "mode": mode,
                        "source_person": person_id_by_key[person_key],
                        "source_carrier": _semantic_text(
                            source_carrier,
                            "source-preserve/non-physical",
                            max_bytes=512,
                        ),
                        "visible_region": _semantic_text(
                            derived_region,
                            "source-preserve/non-physical",
                            max_bytes=512,
                        ),
                        "boundary": _semantic_text(
                            derived_boundary,
                            "source-preserve/non-physical",
                            max_bytes=512,
                        ),
                        "relationship": _semantic_text(
                            relationship,
                            "source-preserve/non-physical",
                            max_bytes=512,
                        ),
                        "physicality": "non_physical",
                        "instantiation": "source_carrier_bound",
                    })
                    classified_derived_appearances += 1
            derived_clause = (
                f"derived_observations={_plan_json(derived_appearances)}"
                if derived_appearances else
                "derived_observations=source-preserve/non-physical"
            )
            constraints.append({
                "frame_index": frame["frame_index"],
                "visible_body_parts": _semantic_text(
                    "；".join(visible_parts),
                    "当前源帧没有明确可观察的目标人物；不得新增人物",
                ),
                "pose_skeleton": _semantic_text(
                    "；".join(poses),
                    "不从其他帧补造当前源帧不可见的人体或姿态",
                ),
                "contact_points": _semantic_text(
                    "；".join((
                        relationships,
                        entity_notes,
                        *entity_relation_notes,
                        derived_clause,
                    )),
                    "保持当前源帧可见接触与实体关系",
                ),
                "occlusion_order": _semantic_text(
                    f"{relationships}；{derived_clause}",
                    "保持当前源帧可见遮挡与成像派生关系",
                ),
                "out_of_frame_crop": crop,
                "non_person_entity_ledger": {
                    "entities": ledger_entities,
                    "relations": ledger_relations,
                },
                "dominant_palette_contract": {
                    "area_weighted_warm_cool_family": "balanced",
                    "saturation_style": "natural",
                },
            })
        segments.append({
            "segment_index": index,
            "persons": segment_people,
            "scene": {
                "scene_id": scene_id_by_key[scene_key],
                "target_region": "当前源帧中人物目标以外的完整可见环境区域",
                "boundary": "在人物、非目标前景实体和画框的当前可见边界处停止",
                "layout_reference_frame_index": 1,
            },
            "protected_non_target_people": [],
            "protected_relations": [
                "source-preserve/no-invention：保持每帧可见构图、实体边界和物理关系；未见部分不补造"
            ],
            "frame_constraints": constraints,
            "photometric_contract": {
                "light_direction": "逐帧保持当前源帧全局光源方向",
                "light_quality": "逐帧保持当前源帧全局光线软硬",
                "exposure_or_intensity": "逐帧保持当前源帧全局曝光与强度",
                "wb_cct": "逐帧保持当前源帧白平衡与色温",
                "global_contrast": "逐帧保持当前源帧全局对比",
                "tone_curve": "逐帧保持当前源帧整体明暗曲线",
            },
        })

    indices = [item["index"] for item in segment_specs]
    frame_counts = {
        index: len(items) for index, items in frames_by_segment.items()
    }
    plan = _canonical_plan_v4({
        "version": 4,
        "phase": "plan",
        "segment_indices": indices,
        "eligible": True,
        "reason": None,
        "person_plans": person_plans,
        "scene_plans": scene_plans,
        "segments": segments,
    }, indices, frame_counts)
    if source_frames is not None:
        plan = _bind_source_palette_contracts(plan, source_frames)
    diagnostics = {
        "score": round(
            present_fields / expected_fields if expected_fields else 1.0, 6
        ),
        "issues": sorted(set(issues)),
        "ignored_mechanical_fields": sorted(set(ignored_mechanical_fields)),
        "entity_continuity": {
            "stable_entity_count": len(compiled_entity_keys),
            "source_preserve_defaults": sorted(set(
                entity_source_preserve_defaults
            )),
        },
        "person_observation_continuity": {
            "classified_derived_count": classified_derived_appearances,
            "source_preserve_defaults": sorted(set(
                appearance_source_preserve_defaults
            )),
        },
    }
    return plan, diagnostics


def _canonical_prompt(value: object) -> str:
    if not isinstance(value, str):
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    prompt = value.strip()
    if not prompt or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return prompt


def _plan_json(plan: dict) -> str:
    return json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_plan_sha256(canonical: dict) -> str:
    return hashlib.sha256(_plan_json(canonical).encode("utf-8")).hexdigest()


def plan_sha256(plan: dict) -> str:
    canonical = _canonical_plan(plan)
    return _canonical_plan_sha256(canonical)


def compile_segment_prompts(
    plan: dict,
    edit_mode: str,
    *,
    allow_empty_people: bool = False,
    _observable_frame_by_segment: dict[int, int] | None = None,
    _compiler_revision: int = 2,
) -> dict[int, str]:
    """Compile an eligible semantic v2 plan into immutable provider prompts."""
    if edit_mode not in SEEDREAM_EDIT_MODES:
        raise ValueError("unsupported image optimization edit mode")
    if _compiler_revision not in {1, 2}:
        raise ValueError("unsupported image optimization prompt compiler revision")
    canonical = _canonical_plan_v2(
        plan, allow_empty_people=allow_empty_people,
    )
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    people = {item["id"]: item for item in canonical["person_plans"]}
    scenes = {item["id"]: item for item in canonical["scene_plans"]}
    prompts = {}
    for segment in canonical["segments"]:
        person_edits = []
        hidden_people = False
        for member in segment["persons"]:
            selected_frame = (
                _observable_frame_by_segment.get(segment["segment_index"])
                if _observable_frame_by_segment is not None else None
            )
            if (
                member["state"] == "not_observable"
                or selected_frame is not None
                and selected_frame not in member["observable_frames"]
            ):
                hidden_people = True
                continue
            design = people[member["id"]]
            target = member["target_region"]
            boundary = member["boundary"]
            if selected_frame is not None and _compiler_revision >= 2:
                target = (
                    f"当前帧中由 {member['id']} 标识的直接可见唯一物理人物区域"
                )
                boundary = (
                    "仅限当前帧硬约束中冻结的人物、服装、遮挡与画边共同形成的"
                    "直接可见边界"
                )
            person_edits.append(
                "将{target}完整替换为{identity}，{wardrobe}，{color}，编辑边界为{boundary}".format(
                    target=target,
                    identity=design["replacement_identity"],
                    wardrobe=design["wardrobe_change"],
                    color=design["local_color_change"],
                    boundary=boundary,
                )
            )
        person_clause = (
            "替换人物：" + "；".join(person_edits)
            if person_edits else "不替换人物"
        )
        if hidden_people:
            person_clause += "；本段不可观察的冻结主人物不得被新增或补造"
        scene_ref = segment["scene"]
        scene = scenes[scene_ref["scene_id"]]
        scene_clause = (
            "替换场景：将{target}完整替换为{replacement}；{semantic}；"
            "形成不同空间结构和真实新环境，形状变化为{geometry}，"
            "纵深变化为{depth}，布局变化为{layout}，{color}；"
            "编辑边界为{boundary}，禁止仅调色、换材质或保留原场景结构"
        ).format(
            target=scene_ref["target_region"],
            replacement=scene["replacement_scene"],
            semantic=scene["semantic_change"],
            geometry="、".join(scene["geometry_changes"]),
            depth="、".join(scene["depth_changes"]),
            layout="、".join(scene["layout_changes"]),
            color=scene["local_color_change"],
            boundary=scene_ref["boundary"],
        )
        protected = "、".join(segment["protected_relations"]) or "现有空间与物理关系"
        non_targets = "、".join(segment["protected_non_target_people"])
        non_target_clause = (
            f"；背景非目标人物（{non_targets}）保持不变" if non_targets else ""
        )
        appearance_clause = (
            f"；{_NON_PHYSICAL_APPEARANCE_CLAUSE}"
            if _compiler_revision >= 2 else ""
        )
        invariant_clause = (
            "保持当前源图的画幅、裁切、机位、镜头、透视、构图、焦点、景深、"
            "人物数量、姿态、动作、视线及核心实体不变；光源方向、曝光、白平衡、"
            "色温、全局色调曲线保持与当前源图一致，只允许新几何产生物理正确的"
            f"局部阴影；严格保持{protected}以及持握、接触、遮挡、前后关系和动作目的"
            f"{non_target_clause}{appearance_clause}；"
            "禁止文字、Logo、水印、畸变、融合、增删实体或画质美化"
        )
        mode_clause = (
            "图1始终是唯一编辑画布；其他输入图只提供冻结人物身份、场景设计和"
            "本段布局，不传递构图、机位、动作、光线、实体关系。"
        )
        prompts[segment["segment_index"]] = _canonical_prompt(
            f"{person_clause}。{scene_clause}。{invariant_clause}。{mode_clause}"
        )
    return prompts


def compile_frame_prompts(
    plan: dict,
    edit_mode: str,
    *,
    _compiler_revision: int = 2,
) -> dict[int, dict[int, str]]:
    """Compile an eligible v3/v4 plan into one immutable prompt per source frame."""
    canonical = (
        _canonical_plan_v4(plan)
        if isinstance(plan, dict) and plan.get("version") == 4
        else _canonical_plan_v3(plan)
    )
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    base_plan = deepcopy(canonical)
    base_plan["version"] = 2
    if canonical["version"] == 4:
        for scene in base_plan["scene_plans"]:
            scene.pop("continuity_graph")
    for segment in base_plan["segments"]:
        segment.pop("frame_constraints")
        segment.pop("photometric_contract")
    segment_prompts = (
        compile_segment_prompts(
            base_plan,
            edit_mode,
            _compiler_revision=_compiler_revision,
        )
        if canonical["version"] == 3 else None
    )
    continuity_by_scene = {
        scene["id"]: scene.get("continuity_graph")
        for scene in canonical["scene_plans"]
    }
    board = composite_replacement_board_spec(canonical)
    tile_by_key = {tile["stable_key"]: tile for tile in board["tiles"]}
    person_key_by_id = {
        person["id"]: _split_stable_key(person["replacement_identity"])[0]
        for person in canonical["person_plans"]
    }
    scene_key_by_id = {
        scene["id"]: _split_stable_key(scene["replacement_scene"])[0]
        for scene in canonical["scene_plans"]
    }
    prompts: dict[int, dict[int, str]] = {}
    for segment in canonical["segments"]:
        photo = segment["photometric_contract"]
        photo_clause = "；".join(
            f"{key}={photo[key]}" for key in _PHOTOMETRIC_CONTRACT_KEYS
        )
        per_frame = {}
        for constraint in segment["frame_constraints"]:
            visible_keys = [
                person_key_by_id.get(member["id"])
                for member in segment["persons"]
                if member["state"] == "replace"
                and constraint["frame_index"] in member["observable_frames"]
            ]
            visible_keys.extend(
                _split_stable_key(entity["description"])[0]
                for entity in constraint["non_person_entity_ledger"]["entities"]
                if entity["visibility"] != "out_of_frame"
            )
            visible_keys.append(scene_key_by_id.get(segment["scene"]["scene_id"]))
            frame_tiles = [
                tile_by_key[stable_key]
                for stable_key in dict.fromkeys(visible_keys)
                if stable_key in tile_by_key
            ]
            board_clause = ""
            if frame_tiles:
                bindings = "；".join(
                    f"{tile['stable_key']} -> {tile['tile_id']} -> "
                    f"{tile['replacement_description']}"
                    for tile in frame_tiles
                )
                board_clause = (
                    f"。全项目共享替换参考板绑定：{bindings}。"
                    "所有片段使用同一张参考板和同一替换描述；只替换当前源帧直接"
                    "可见的已绑定元素，严格保持当前源帧构图、表现形式、色调、光照、"
                    "动作和关系不变"
                )
            base_prompt = (
                segment_prompts[segment["segment_index"]]
                if segment_prompts is not None else compile_segment_prompts(
                    base_plan,
                    edit_mode,
                    allow_empty_people=True,
                    _observable_frame_by_segment={
                        segment["segment_index"]: constraint["frame_index"]
                    },
                    _compiler_revision=_compiler_revision,
                )[segment["segment_index"]]
            )
            frame_clause = "；".join(
                f"{key}={constraint[key]}"
                for key in ("frame_index", *_FRAME_TEXT_CONSTRAINT_KEYS)
            )
            frame_clause = (
                f"{frame_clause}；non_person_entity_ledger="
                f"{_plan_json(constraint['non_person_entity_ledger'])}"
                f"；dominant_palette_contract="
                f"{_plan_json(constraint['dominant_palette_contract'])}"
            )
            continuity_clause = ""
            if canonical["version"] == 4:
                graph = continuity_by_scene[segment["scene"]["scene_id"]]
                view = next(
                    item for item in graph["views"]
                    if item["segment_index"] == segment["segment_index"]
                    and item["frame_index"] == constraint["frame_index"]
                )
                continuity_clause = (
                    "。冻结场景连续性目标图："
                    f"{_plan_json({key: graph[key] for key in ('components', 'topology')})}"
                    "。当前帧场景视图："
                    f"{_plan_json(view)}"
                )
            per_frame[constraint["frame_index"]] = _canonical_prompt(
                f"{base_prompt}。仅当前源帧硬约束："
                f"{frame_clause}。全局光色硬约束：{photo_clause}。"
                f"{continuity_clause}"
                f"不得从其他帧补全，不得重布全局光线。{board_clause}"
            )
        prompts[segment["segment_index"]] = per_frame
    return prompts


def composite_replacement_board_spec(plan: dict) -> dict:
    """Derive one numbered project board from existing v4 plan fields only."""
    canonical = _canonical_plan(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    candidates = []
    for person in canonical["person_plans"]:
        stable_key, description = _split_stable_key(
            person["replacement_identity"]
        )
        if stable_key is not None:
            candidates.append({
                "stable_key": stable_key,
                "kind": "person",
                "replacement_description": description,
                "reference": deepcopy(person["reference"]),
            })

    seen_entities = set()
    for segment in canonical["segments"]:
        for frame in segment.get("frame_constraints", []):
            for entity in frame["non_person_entity_ledger"]["entities"]:
                stable_key, description = _split_stable_key(
                    entity["description"]
                )
                if stable_key is None or stable_key in seen_entities:
                    continue
                seen_entities.add(stable_key)
                candidates.append({
                    "stable_key": stable_key,
                    "kind": "entity",
                    "replacement_description": description,
                    "reference": {
                        "segment_index": segment["segment_index"],
                        "frame_index": frame["frame_index"],
                    },
                })

    for scene in canonical["scene_plans"]:
        stable_key, description = _split_stable_key(scene["replacement_scene"])
        if stable_key is not None:
            candidates.append({
                "stable_key": stable_key,
                "kind": "scene",
                "replacement_description": description,
                "reference": deepcopy(scene["reference"]),
            })

    tiles = [
        {"tile_id": f"TILE_{position:02d}", **candidate}
        for position, candidate in enumerate(candidates, 1)
    ]
    return {"tiles": tiles}


def composite_replacement_board_prompt(plan: dict) -> str:
    board = composite_replacement_board_spec(plan)
    columns = max(1, int(len(board["tiles"]) ** 0.5 + 0.999999))
    rows = max(1, (len(board["tiles"]) + columns - 1) // columns)
    bindings = "；".join(
        f"{tile['tile_id']}={tile['stable_key']}："
        f"{tile['replacement_description']}"
        for tile in board["tiles"]
    )
    return _canonical_prompt(
        "在图1的空白画布上生成一张全项目共享的合并替换参考图；"
        f"按 {columns} 列 {rows} 行的固定网格编号分块，且每个元素恰好一个 tile："
        f"{bindings}。"
        "图2是按相同 tile 顺序合成的唯一源视觉证据图。每个 tile 只需一块能清晰展示"
        "对应替换元素特征的代表图，不生成三视图或四视图，不重复、不遗漏、"
        "不合并不同 stable key。"
    )


def semantic_context(plan: dict) -> dict:
    canonical = _canonical_plan(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    people = {item["id"]: item for item in canonical["person_plans"]}
    scenes = {item["id"]: item for item in canonical["scene_plans"]}
    return {
        "version": canonical["version"],
        "plan_sha256": plan_sha256(canonical),
        "person_plans": deepcopy(canonical["person_plans"]),
        "segments": [
            {
                "segment_index": segment["segment_index"],
                "observable_person_ids": [
                    item["id"] for item in segment["persons"]
                    if item["state"] == "replace"
                ],
                "persons": [
                    {
                        **deepcopy(item),
                        "design": deepcopy(people[item["id"]]),
                    }
                    for item in segment["persons"]
                ],
                "scene": {
                    **deepcopy(segment["scene"]),
                    "design": deepcopy(scenes[segment["scene"]["scene_id"]]),
                },
                "protected_non_target_people": list(
                    segment["protected_non_target_people"]
                ),
                "protected_relations": list(segment["protected_relations"]),
            }
            for segment in canonical["segments"]
        ],
    }


def reference_slots(plan: dict) -> dict[str, list[dict]]:
    canonical = _canonical_plan(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    identity = [
        {
            "role": f"identity:{person['id']}",
            "person_id": person["id"],
            **person["reference"],
        }
        for person in canonical["person_plans"]
    ]
    scene = [
        {
            "role": f"scene:{item['id']}",
            "scene_id": item["id"],
            **item["reference"],
        }
        for item in canonical["scene_plans"]
    ]
    layout = [
        {
            "segment_index": segment["segment_index"],
            "role": f"layout:{segment['scene']['scene_id']}",
            "scene_id": segment["scene"]["scene_id"],
            "frame_index": segment["scene"]["layout_reference_frame_index"],
        }
        for segment in canonical["segments"]
    ]
    return {"identity": identity, "scene": scene, "layout": layout}


def _canonical_plan_with_frame_inventory(
    plan: dict, frame_inventory: list[dict]
) -> tuple[dict, list[dict]]:
    canonical = _canonical_plan(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    if not isinstance(frame_inventory, list) or not frame_inventory:
        raise ValueError("invalid image optimization execution inputs")
    inventory = []
    lookup: dict[tuple[int, int], dict] = {}
    prior: tuple[int, int] | None = None
    for item in frame_inventory:
        inventory_keys = {
            "segment_index", "frame_index", "frame_name", "source_sha256"
        }
        if canonical["version"] == 4:
            inventory_keys |= {
                "source_transition_from_previous",
                "source_transition_evidence_sha256",
            }
        if not isinstance(item, dict) or set(item) != inventory_keys:
            raise ValueError("invalid image optimization execution inputs")
        segment_index = item.get("segment_index")
        frame_index = item.get("frame_index")
        key = (segment_index, frame_index)
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index not in canonical["segment_indices"]
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 1
            or item.get("frame_name") != f"{frame_index:02d}.png"
            or not isinstance(item.get("source_sha256"), str)
            or _SHA256_RE.fullmatch(item["source_sha256"]) is None
            or (
                canonical["version"] == 4
                and (
                    not isinstance(
                        item.get("source_transition_from_previous"), str
                    )
                    or item["source_transition_from_previous"]
                    not in _SCENE_CONTINUITY_TRANSITIONS
                    or not isinstance(
                        item.get("source_transition_evidence_sha256"), str
                    )
                    or _SHA256_RE.fullmatch(
                        item["source_transition_evidence_sha256"]
                    ) is None
                )
            )
            or key in lookup
            or (prior is not None and key <= prior)
        ):
            raise ValueError("invalid image optimization execution inputs")
        current = deepcopy(item)
        inventory.append(current)
        lookup[key] = current
        prior = key
    for index in canonical["segment_indices"]:
        frames = [item["frame_index"] for item in inventory if item["segment_index"] == index]
        if frames != list(range(1, len(frames) + 1)):
            raise ValueError("invalid image optimization execution inputs")
    try:
        canonical = _canonical_plan(
            canonical,
            canonical["segment_indices"],
            {
                index: sum(item["segment_index"] == index for item in inventory)
                for index in canonical["segment_indices"]
            },
        )
    except ImageOptimizationOutputError:
        raise ValueError("invalid image optimization execution inputs") from None
    if canonical["version"] == 4:
        planned_transitions = {
            (view["segment_index"], view["frame_index"]):
            view["transition_from_previous"]
            for scene in canonical["scene_plans"]
            for view in scene["continuity_graph"]["views"]
        }
        if any(
            item["source_transition_from_previous"]
            != planned_transitions[(item["segment_index"], item["frame_index"])]
            for item in inventory
        ):
            raise ValueError("invalid image optimization execution inputs")
    return canonical, inventory


def freeze_execution_inputs(
    plan: dict,
    *,
    revision: int,
    profile: dict,
    model: str,
    frame_inventory: list[dict],
) -> dict:
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(profile, dict)
        or set(profile) != {"id", "revision"}
        or not isinstance(profile.get("id"), str)
        or profile["id"] != profile["id"].strip()
        or not profile["id"]
        or isinstance(profile.get("revision"), bool)
        or not isinstance(profile.get("revision"), int)
        or profile["revision"] < 1
        or model not in SEEDREAM_MODELS
    ):
        raise ValueError("invalid image optimization execution inputs")
    canonical, inventory = _canonical_plan_with_frame_inventory(plan, frame_inventory)
    lookup = {
        (item["segment_index"], item["frame_index"]): item
        for item in inventory
    }

    slots = reference_slots(canonical)

    def freeze_slot(item: dict) -> dict:
        source = lookup.get((item["segment_index"], item["frame_index"]))
        if source is None:
            raise ValueError("invalid image optimization execution inputs")
        return {**deepcopy(item), "source_sha256": source["source_sha256"]}

    segments = {item["segment_index"]: item for item in canonical["segments"]}
    scene_plans = {item["id"]: item for item in canonical["scene_plans"]}
    frames = []
    for source in inventory:
        segment = segments[source["segment_index"]]
        observable = [
            person["id"]
            for person in segment["persons"]
            if source["frame_index"] in person["observable_frames"]
        ]
        current = {
            **deepcopy(source),
            "observable_person_ids": observable,
            "scene_id": segment["scene"]["scene_id"],
        }
        if canonical["version"] in {3, 4}:
            constraint = next(
                item
                for item in segment["frame_constraints"]
                if item["frame_index"] == source["frame_index"]
            )
            current.update(
                frame_constraint=deepcopy(constraint),
                photometric_contract=deepcopy(segment["photometric_contract"]),
            )
        if canonical["version"] == 4:
            graph = scene_plans[segment["scene"]["scene_id"]]["continuity_graph"]
            view = next(
                item for item in graph["views"]
                if item["segment_index"] == source["segment_index"]
                and item["frame_index"] == source["frame_index"]
            )
            current["scene_continuity_view"] = deepcopy(view)
        frames.append(current)
    payload = {
        "version": canonical["version"],
        "plan_sha256": plan_sha256(canonical),
        "profile": deepcopy(profile),
        "revision": revision,
        "model": model,
        "identity_slots": [freeze_slot(item) for item in slots["identity"]],
        "scene_slots": [freeze_slot(item) for item in slots["scene"]],
        "layout_slots": [freeze_slot(item) for item in slots["layout"]],
        "frames": frames,
    }
    if canonical["version"] == 4:
        payload["continuity_sha256"] = _scene_graph_digest(canonical, inventory)
        payload["sha256"] = sha256(_plan_json(payload))
    return payload


def _scene_anchor_schedule(plan: dict, execution_inputs: dict) -> dict:
    """Derive the only v4 scene-anchor DAG from frozen plan and frame inputs."""
    canonical = _canonical_plan_v4(plan)
    if not canonical["eligible"] or execution_inputs.get("version") != 4:
        raise ValueError("invalid image optimization anchor schedule")
    frames = execution_inputs.get("frames")
    if not isinstance(frames, list):
        raise ValueError("invalid image optimization anchor schedule")
    frame_by_key = {
        (frame.get("segment_index"), frame.get("frame_index")): frame
        for frame in frames if isinstance(frame, dict)
    }
    if len(frame_by_key) != len(frames):
        raise ValueError("invalid image optimization anchor schedule")

    def frozen_anchor(segment_index: int, frame_index: int, order: int) -> dict:
        frame = frame_by_key.get((segment_index, frame_index))
        if not isinstance(frame, dict) or set(frame) != {
            "segment_index", "frame_index", "frame_name", "source_sha256",
            "observable_person_ids", "scene_id", "frame_constraint",
            "photometric_contract", "source_transition_from_previous",
            "source_transition_evidence_sha256", "scene_continuity_view",
        }:
            raise ValueError("invalid image optimization anchor schedule")
        return {
            "order": order,
            "segment_index": segment_index,
            "frame_index": frame_index,
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
        }

    segments = {item["segment_index"]: item for item in canonical["segments"]}
    # Every paid node is ordered and named before runtime.  Runtime may only
    # replay this immutable DAG, never allocate an order with a dynamic max().
    # Quality-review-only alternate/person packs are deliberately excluded:
    # valid generation must not depend on a model-facing acceptance topology.
    node_specs: list[tuple[str, str, int, int]] = []
    for scene in canonical["scene_plans"]:
        reference = scene["reference"]
        node_specs.append((scene["id"], "global", reference["segment_index"], reference["frame_index"]))
    layout_keys = set()
    for scene in canonical["scene_plans"]:
        for segment_index in scene["segments"]:
            segment = segments.get(segment_index)
            if segment is None or segment["scene"]["scene_id"] != scene["id"]:
                raise ValueError("invalid image optimization anchor schedule")
            frame_index = segment["scene"]["layout_reference_frame_index"]
            layout_keys.add((segment_index, frame_index))
            node_specs.append((scene["id"], f"layout-{segment_index:04d}", segment_index, frame_index))
    for segment in canonical["segments"]:
        index = segment["segment_index"]
        scene_id = segment["scene"]["scene_id"]
        for frame in segment["frame_constraints"]:
            frame_index = frame["frame_index"]
            if (index, frame_index) not in layout_keys:
                node_specs.append((scene_id, f"fanout-{index:04d}-{frame_index:04d}", index, frame_index))
    if len({(scene_id, label) for scene_id, label, _segment, _frame in node_specs}) != len(node_specs):
        raise ValueError("invalid image optimization anchor schedule")
    nodes = [
        {"scene_id": scene_id, "label": label, "anchor": frozen_anchor(segment_index, frame_index, order)}
        for order, (scene_id, label, segment_index, frame_index) in enumerate(node_specs, 1)
    ]
    node_by_identity = {(item["scene_id"], item["label"]): item["anchor"] for item in nodes}
    scenes = []
    for scene in canonical["scene_plans"]:
        scenes.append({
            "scene_id": scene["id"],
            "global_anchor": node_by_identity[(scene["id"], "global")],
            "segment_layout_anchors": [
                node_by_identity[(scene["id"], f"layout-{segment_index:04d}")]
                for segment_index in scene["segments"]
            ],
        })
    return {
        "version": 4,
        "plan_sha256": execution_inputs["plan_sha256"],
        "execution_input_sha256": execution_inputs["sha256"],
        "scenes": scenes,
        "nodes": nodes,
    }


def _scene_graph_digest(plan: dict, inventory: list[dict]) -> str:
    """Bind v4 graph/views to the authoritative source-transition evidence."""
    canonical = _canonical_plan_v4(plan)
    if not canonical["eligible"] or not isinstance(inventory, list):
        raise ValueError("invalid image optimization scene graph")
    transitions = []
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("invalid image optimization scene graph")
        try:
            transitions.append({
                key: item[key]
                for key in (
                    "segment_index", "frame_index",
                    "source_transition_from_previous",
                    "source_transition_evidence_sha256",
                )
            })
        except KeyError:
            raise ValueError("invalid image optimization scene graph") from None
    payload = {
        "version": 4,
        "scenes": [
            {"scene_id": scene["id"], "continuity_graph": scene["continuity_graph"]}
            for scene in canonical["scene_plans"]
        ],
        "transitions": transitions,
    }
    return sha256(_plan_json(payload))


def _frame_prompt_compiler_revision(profile: object) -> int:
    if profile == {"id": "image-postprocess", "revision": 1}:
        return 1
    return 2


def _freeze_frame_prompts_v4(
    settings: Settings,
    execution_inputs: dict,
    prompts: dict[int, dict[int, str]],
    plan: dict | None,
) -> dict:
    expected_keys = {
        "version", "plan_sha256", "profile", "revision", "model",
        "identity_slots", "scene_slots", "layout_slots", "frames",
        "continuity_sha256", "sha256",
    }
    if (
        not isinstance(execution_inputs, dict)
        or set(execution_inputs) != expected_keys
        or execution_inputs.get("version") != 4
        or execution_inputs.get("model") != settings.seedream_model
        or not isinstance(plan, dict)
        or not isinstance(prompts, dict)
        or not isinstance(execution_inputs.get("frames"), list)
        or not execution_inputs["frames"]
    ):
        raise ValueError("invalid image optimization frame prompts")
    inventory = []
    for frame in execution_inputs["frames"]:
        if not isinstance(frame, dict) or set(frame) != {
            "segment_index", "frame_index", "frame_name", "source_sha256",
            "observable_person_ids", "scene_id", "frame_constraint",
            "photometric_contract", "source_transition_from_previous",
            "source_transition_evidence_sha256", "scene_continuity_view",
        }:
            raise ValueError("invalid image optimization frame prompts")
        inventory.append({
            key: frame[key] for key in (
                "segment_index", "frame_index", "frame_name", "source_sha256",
                "source_transition_from_previous",
                "source_transition_evidence_sha256",
            )
        })
    try:
        expected_inputs = freeze_execution_inputs(
            plan,
            revision=execution_inputs["revision"],
            profile=execution_inputs["profile"],
            model=execution_inputs["model"],
            frame_inventory=inventory,
        )
        expected_prompts = compile_frame_prompts(
            plan,
            settings.seedream_edit_mode,
            _compiler_revision=_frame_prompt_compiler_revision(
                execution_inputs["profile"]
            ),
        )
    except (
        KeyError, TypeError, ValueError, ImageOptimizationOutputError,
        ImageOptimizationIneligibleError,
    ):
        raise ValueError("invalid image optimization frame prompts") from None
    if execution_inputs != expected_inputs or prompts != expected_prompts:
        raise ValueError("invalid image optimization frame prompts")
    frozen = []
    for frame in execution_inputs["frames"]:
        text = expected_prompts[frame["segment_index"]][frame["frame_index"]]
        frozen.append({
            "segment_index": frame["segment_index"],
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
            "default": text,
            "current": text,
            "sha256": sha256(text),
        })
    receipt = {
        "version": 4,
        "plan_sha256": execution_inputs["plan_sha256"],
        "continuity_sha256": execution_inputs["continuity_sha256"],
        "execution_input_sha256": execution_inputs["sha256"],
        "execution_inputs": deepcopy(execution_inputs),
        "model": settings.seedream_model,
        "edit_mode": settings.seedream_edit_mode,
        "scene_anchor_schedule": _scene_anchor_schedule(plan, execution_inputs),
        "frames": frozen,
    }
    return {"_image_optimization": {
        **receipt,
        "sha256": sha256(_plan_json(receipt)),
    }}


def freeze_frame_prompts(
    settings: Settings,
    execution_inputs: dict,
    prompts: dict[int, dict[int, str]],
    *,
    plan: dict | None = None,
) -> dict:
    """Freeze v3/v4 prompts against one exact source frame identity per call."""
    if isinstance(execution_inputs, dict) and execution_inputs.get("version") == 4:
        return _freeze_frame_prompts_v4(settings, execution_inputs, prompts, plan)
    expected_keys = {
        "version", "plan_sha256", "profile", "revision", "model",
        "identity_slots", "scene_slots", "layout_slots", "frames",
    }
    if (
        not isinstance(execution_inputs, dict)
        or set(execution_inputs) != expected_keys
        or execution_inputs.get("version") != 3
        or execution_inputs.get("model") != settings.seedream_model
        or not isinstance(execution_inputs.get("plan_sha256"), str)
        or _SHA256_RE.fullmatch(execution_inputs["plan_sha256"]) is None
        or not isinstance(prompts, dict)
    ):
        raise ValueError("invalid image optimization frame prompts")
    raw_frames = execution_inputs.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("invalid image optimization frame prompts")
    expected: dict[tuple[int, int], dict] = {}
    for frame in raw_frames:
        if not isinstance(frame, dict) or set(frame) != {
            "segment_index", "frame_index", "frame_name", "source_sha256",
            "observable_person_ids", "scene_id", "frame_constraint",
            "photometric_contract",
        }:
            raise ValueError("invalid image optimization frame prompts")
        index, number = frame.get("segment_index"), frame.get("frame_index")
        key = (index, number)
        if (
            isinstance(index, bool) or not isinstance(index, int)
            or isinstance(number, bool) or not isinstance(number, int)
            or number < 1 or key in expected
            or frame.get("frame_name") != f"{number:02d}.png"
            or not isinstance(frame.get("source_sha256"), str)
            or _SHA256_RE.fullmatch(frame["source_sha256"]) is None
            or not isinstance(frame.get("observable_person_ids"), list)
            or any(
                not isinstance(identifier, str)
                or _PERSON_ID_RE.fullmatch(identifier) is None
                for identifier in frame["observable_person_ids"]
            )
            or len(set(frame["observable_person_ids"])) != len(
                frame["observable_person_ids"]
            )
            or not isinstance(frame.get("frame_constraint"), dict)
            or not isinstance(frame.get("photometric_contract"), dict)
            or set(frame["photometric_contract"]) != set(_PHOTOMETRIC_CONTRACT_KEYS)
        ):
            raise ValueError("invalid image optimization frame prompts")
        try:
            canonical_constraint = _canonical_frame_constraint(
                frame["frame_constraint"], set(frame["observable_person_ids"])
            )
        except ImageOptimizationOutputError:
            raise ValueError("invalid image optimization frame prompts") from None
        if (
            canonical_constraint != frame["frame_constraint"]
            or canonical_constraint["frame_index"] != number
        ):
            raise ValueError("invalid image optimization frame prompts")
        expected[key] = frame
    expected_by_segment: dict[int, set[int]] = {}
    for index, number in expected:
        expected_by_segment.setdefault(index, set()).add(number)
    if set(prompts) != set(expected_by_segment):
        raise ValueError("invalid image optimization frame prompts")
    for index, frames in prompts.items():
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(frames, dict)
            or set(frames) != expected_by_segment[index]
            or any(isinstance(number, bool) or not isinstance(number, int) for number in frames)
        ):
            raise ValueError("invalid image optimization frame prompts")
    frozen = []
    for key, frame in sorted(expected.items()):
        text = prompts[key[0]].get(key[1])
        if not isinstance(text, str):
            raise ValueError("invalid image optimization frame prompts")
        text = _canonical_prompt(text)
        frozen.append({
            "segment_index": key[0],
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
            "default": text,
            "current": text,
            "sha256": sha256(text),
        })
    receipt = {
        "version": 3,
        "plan_sha256": execution_inputs["plan_sha256"],
        "model": settings.seedream_model,
        "edit_mode": settings.seedream_edit_mode,
        "frames": frozen,
    }
    return {"_image_optimization": {
        **receipt,
        "sha256": sha256(_plan_json(receipt)),
    }}


def _canonical_project_output(
    value: object,
    indices: list[int],
    edit_mode: str,
    frame_counts: dict[int, int],
    *,
    source_frames: dict[int, list[Path]] | None = None,
    segment_specs: list[dict] | None = None,
    element_index: dict | None = None,
) -> tuple[dict, dict]:
    semantic_output = isinstance(value, dict) and value.get("version") is None
    if semantic_output:
        if segment_specs is None:
            raise ValueError("image semantic compiler requires backend segment input")
        plan, diagnostics = compile_semantic_plan(
            value,
            segment_specs,
            source_frames=source_frames,
            element_index=element_index,
        )
        _LOGGER.info(
            "image semantic compiler score=%s issues=%s ignored=%s "
            "entity_continuity=%s",
            diagnostics["score"],
            diagnostics["issues"],
            diagnostics["ignored_mechanical_fields"],
            diagnostics["entity_continuity"],
        )
    else:
        # Frozen v2-v4 plans remain readable; current generation uses only the
        # semantic compiler above.
        plan = _canonical_plan(value, indices, frame_counts)
    if not plan["eligible"]:
        # A plan-phase model is a compiler, not the product's content judge.
        # Refusal-shaped legacy output is a retryable protocol violation.
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    if source_frames is not None and not semantic_output:
        plan = _bind_source_palette_contracts(plan, source_frames)
    if plan["version"] in {3, 4}:
        return plan, compile_frame_prompts(plan, edit_mode)
    return plan, compile_segment_prompts(plan, edit_mode)


def _project_segment_inputs(
    segments: list[dict], session_dir: Path, *, expected_version: int,
) -> tuple[Path, list[int], list[tuple[dict, list[Path]]]]:
    try:
        session = Path(session_dir).resolve(strict=True)
    except OSError:
        raise ValueError("invalid image optimization input") from None
    if not isinstance(segments, list) or not segments:
        raise ValueError("invalid image optimization segments")
    indices = [item.get("index") for item in segments if isinstance(item, dict)]
    if len(indices) != len(segments) or indices not in (
        [0], list(range(1, len(segments) + 1))
    ):
        raise ValueError("invalid image optimization segments")
    prepared = []
    for segment in segments:
        allowed = {"index", "chain_id", "join_mode", "keyframes_dir"}
        if set(segment) != allowed and set(segment) != allowed | {
            "transition_skeleton"
        }:
            raise ValueError("invalid image optimization segments")
        if expected_version == 4 and "transition_skeleton" not in segment:
            raise ValueError("invalid image optimization segments")
        if (
            not isinstance(segment["chain_id"], str)
            or not segment["chain_id"]
            or len(segment["chain_id"]) > 128
            or segment["join_mode"] not in {"hard_cut", "continue"}
        ):
            raise ValueError("invalid image optimization segments")
        try:
            source = Path(segment["keyframes_dir"]).resolve(strict=True)
            source.relative_to(session)
        except (OSError, TypeError, ValueError):
            raise ValueError("invalid image optimization segments") from None
        frames = _validated_frames(source)
        skeleton = segment.get("transition_skeleton")
        if skeleton is not None:
            if not isinstance(skeleton, list) or len(skeleton) != len(frames):
                raise ValueError("invalid image optimization segments")
            expected = [
                {
                    "segment_index": segment["index"],
                    "frame_index": position,
                    "frame_name": frame.name,
                    "source_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "source_transition_from_previous": item.get(
                        "source_transition_from_previous"
                    ) if isinstance(item, dict) else None,
                    "source_transition_evidence_sha256": item.get(
                        "source_transition_evidence_sha256"
                    ) if isinstance(item, dict) else None,
                }
                for position, (frame, item) in enumerate(zip(frames, skeleton), 1)
            ]
            if skeleton != expected or any(
                item["source_transition_from_previous"]
                not in _SCENE_CONTINUITY_TRANSITIONS
                or not isinstance(
                    item["source_transition_evidence_sha256"], str
                )
                or _SHA256_RE.fullmatch(
                    item["source_transition_evidence_sha256"]
                ) is None
                for item in skeleton
            ):
                raise ValueError("invalid image optimization segments")
        prepared.append((segment, frames))
    return session, indices, prepared


def generate_project_prompts(
    runner,
    segments: list[dict],
    edit_mode: str,
    *,
    session_dir: Path,
    expected_version: int = 4,
    element_index_path: Path | None = None,
) -> tuple[dict, dict]:
    """Run the plan phase once, then compile immutable provider prompts."""
    if edit_mode not in SEEDREAM_EDIT_MODES or expected_version not in {2, 3, 4}:
        raise ValueError("unsupported image optimization edit mode")
    session, indices, prepared = _project_segment_inputs(
        segments, session_dir, expected_version=expected_version,
    )
    skill = verification_skill_path()
    if not skill.is_file():
        raise ValueError("invalid image optimization segments")
    element_index = None
    if element_index_path is not None:
        try:
            element_index = _canonical_element_index(
                _read_json_output(Path(element_index_path), MAX_CONTINUITY_BYTES)
            )
        except ImageOptimizationOutputError:
            element_index = _canonical_element_index(None)

    with tempfile.TemporaryDirectory(prefix="duet-image-postprocess-", dir="/tmp") as raw:
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
                **({"transition_skeleton": segment["transition_skeleton"]}
                   if "transition_skeleton" in segment else {}),
            })
        request = {
                "phase": "plan",
                "edit_mode": edit_mode,
                "segments": request_segments,
                "semantic_slots": semantic_slot_manifest(
                    request_segments, element_index=element_index,
                ),
        }
        if element_index is not None:
            request["element_index"] = element_index
        (work / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n",
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
            max_bytes = (
                MAX_CONTINUITY_BYTES
                + MAX_PROMPT_BYTES * len(indices)
                + MAX_PROJECT_OUTPUT_OVERHEAD_BYTES
            )
            plan, prompts = _canonical_project_output(
                _read_json_output(work / "image_optimization.json", max_bytes),
                indices,
                edit_mode,
                {segment["index"]: len(frames) for segment, frames in prepared},
                source_frames={
                    segment["index"]: frames for segment, frames in prepared
                },
                segment_specs=[segment for segment, _frames in prepared],
                element_index=element_index,
            )
            if plan.get("version") != expected_version:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            skeletons = {
                segment["index"]: segment.get("transition_skeleton")
                for segment, _frames in prepared
                if "transition_skeleton" in segment
            }
            if plan.get("version") == 4 and skeletons:
                actual = {
                    scene_view["segment_index"]: []
                    for scene in plan["scene_plans"]
                    for scene_view in scene["continuity_graph"]["views"]
                }
                for scene in plan["scene_plans"]:
                    for view in scene["continuity_graph"]["views"]:
                        actual[view["segment_index"]].append({
                            "frame_index": view["frame_index"],
                            "transition_from_previous": view["transition_from_previous"],
                        })
                expected = {
                    index: [
                        {
                            "frame_index": item["frame_index"],
                            "transition_from_previous": item[
                                "source_transition_from_previous"
                            ],
                        }
                        for item in skeleton
                    ]
                    for index, skeleton in skeletons.items()
                }
                if actual != expected:
                    raise ImageOptimizationOutputError(
                        "image optimization output is missing or invalid"
                    )
            return plan, prompts
        except ImageOptimizationIneligibleError:
            raise
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


def freeze_continuity(
    plan: dict, *, frame_counts: dict[int, int] | None = None
) -> dict:
    if isinstance(plan, dict) and plan.get("version") in {2, 3, 4}:
        canonical = _canonical_plan(plan, frame_counts=frame_counts)
        if canonical["version"] in {3, 4} and frame_counts is None:
            raise ImageOptimizationOutputError(
                "image optimization output is missing or invalid"
            )
        if not canonical["eligible"]:
            raise ImageOptimizationIneligibleError(canonical["reason"])
        raw = _plan_json(canonical)
    else:
        canonical = _canonical_continuity_v1(plan)
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


def freeze_plan_audit_inputs(plan: dict, *, frame_inventory: list[dict]) -> dict:
    """Freeze the source-frame evidence a pre-provider v3/v4 audit may inspect."""
    try:
        canonical, inventory = _canonical_plan_with_frame_inventory(
            plan, frame_inventory
        )
    except ValueError:
        raise ValueError("invalid image plan audit inputs") from None
    if canonical["version"] not in {3, 4}:
        raise ValueError("invalid image plan audit inputs")
    frame_counts = {
        index: sum(item["segment_index"] == index for item in inventory)
        for index in canonical["segment_indices"]
    }
    continuity_sha256 = (
        _scene_graph_digest(canonical, inventory)
        if canonical["version"] == 4
        else freeze_continuity(canonical, frame_counts=frame_counts)[
            "_image_continuity"
        ]["sha256"]
    )
    payload = {
        "version": 1,
        "plan_sha256": plan_sha256(canonical),
        "continuity_sha256": continuity_sha256,
        "frames": inventory,
    }
    return {**payload, "sha256": sha256(_plan_json(payload))}


def _canonical_plan_audit_inputs(plan: dict, value: object) -> tuple[dict, dict]:
    if (
        not isinstance(value, dict)
        or set(value) != {
            "version", "plan_sha256", "continuity_sha256", "frames", "sha256"
        }
        or value.get("version") != 1
        or not isinstance(value.get("plan_sha256"), str)
        or _SHA256_RE.fullmatch(value["plan_sha256"]) is None
        or not isinstance(value.get("continuity_sha256"), str)
        or _SHA256_RE.fullmatch(value["continuity_sha256"]) is None
        or not isinstance(value.get("sha256"), str)
        or _SHA256_RE.fullmatch(value["sha256"]) is None
        or not isinstance(value.get("frames"), list)
    ):
        raise ImageOptimizationOutputError("image plan audit input is missing or invalid")
    try:
        canonical, _ = _canonical_plan_with_frame_inventory(plan, value["frames"])
        expected = freeze_plan_audit_inputs(plan, frame_inventory=value["frames"])
    except (ValueError, ImageOptimizationIneligibleError):
        raise ImageOptimizationOutputError(
            "image plan audit input is missing or invalid"
        ) from None
    if canonical["version"] not in {3, 4} or value != expected:
        raise ImageOptimizationOutputError("image plan audit input is missing or invalid")
    return canonical, expected


def canonical_plan_audit_verdict(value: object, plan: dict, audit_inputs: dict) -> dict:
    """Validate a source-bound, exact per-frame pre-provider audit verdict."""
    canonical_plan, receipt = _canonical_plan_audit_inputs(plan, audit_inputs)
    if (
        not isinstance(value, dict)
        or set(value) != {
            "version", "phase", "plan_sha256", "continuity_sha256",
            "audit_input_sha256", "passed", "reason", "frame_checks",
        }
        or value.get("version") != canonical_plan["version"]
        or value.get("phase") != "plan_audit"
        or value.get("plan_sha256") != receipt["plan_sha256"]
        or value.get("continuity_sha256") != receipt["continuity_sha256"]
        or value.get("audit_input_sha256") != receipt["sha256"]
        or not isinstance(value.get("passed"), bool)
        or not isinstance(value.get("frame_checks"), list)
        or len(value["frame_checks"]) != len(receipt["frames"])
    ):
        raise ImageOptimizationOutputError("image plan audit output is missing or invalid")
    checks = []
    statuses = []
    closure_keys = (
        "body_closure", "scene_closure", "entity_closure", "relation_closure",
    )
    if canonical_plan["version"] == 4:
        closure_keys += ("scene_continuity_closure",)
    for expected, raw in zip(receipt["frames"], value["frame_checks"]):
        frame_check_keys = {
            "segment_index", "frame_index", "source_sha256", *closure_keys,
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != frame_check_keys
            or raw.get("segment_index") != expected["segment_index"]
            or raw.get("frame_index") != expected["frame_index"]
            or raw.get("source_sha256") != expected["source_sha256"]
        ):
            raise ImageOptimizationOutputError(
                "image plan audit output is missing or invalid"
            )
        item = {
            "segment_index": expected["segment_index"],
            "frame_index": expected["frame_index"],
            "source_sha256": expected["source_sha256"],
        }
        for key in closure_keys:
            item[key] = _canonical_quality_check(raw.get(key))
            statuses.append(item[key]["status"])
        checks.append(item)
    passed = all(status == "pass" for status in statuses)
    reason = None if passed else (
        "plan_audit_unknown" if "unknown" in statuses else "plan_audit_failed"
    )
    if value["passed"] != passed or value.get("reason") != reason:
        raise ImageOptimizationOutputError("image plan audit output is missing or invalid")
    return {
        "version": canonical_plan["version"],
        "phase": "plan_audit",
        "plan_sha256": receipt["plan_sha256"],
        "continuity_sha256": receipt["continuity_sha256"],
        "audit_input_sha256": receipt["sha256"],
        "passed": passed,
        "reason": reason,
        "frame_checks": checks,
    }


def generate_plan_audit_verdict(
    runner,
    plan: dict,
    audit_inputs: dict,
    segments: list[dict],
    *,
    session_dir: Path,
) -> dict:
    """Audit only frozen source frames before any provider submission."""
    try:
        session = Path(session_dir).resolve(strict=True)
        skill = verification_skill_path()
        canonical_plan, receipt = _canonical_plan_audit_inputs(plan, audit_inputs)
    except (OSError, TypeError, ValueError, ImageOptimizationOutputError):
        raise ValueError("invalid image plan audit input") from None
    if (
        not isinstance(segments, list)
        or len(segments) != len(canonical_plan["segment_indices"])
    ):
        raise ValueError("invalid image plan audit input")
    prepared = []
    actual_inventory = []
    receipt_by_frame = {
        (item["segment_index"], item["frame_index"]): item
        for item in receipt["frames"]
    }
    for expected, segment in zip(canonical_plan["segment_indices"], segments):
        if (
            not isinstance(segment, dict)
            or set(segment) != {"index", "source_keyframes_dir"}
            or segment.get("index") != expected
        ):
            raise ValueError("invalid image plan audit input")
        try:
            source = Path(segment["source_keyframes_dir"]).resolve(strict=True)
            source.relative_to(session)
            frames = _validated_frames(source)
        except (OSError, TypeError, ValueError):
            raise ValueError("invalid image plan audit input") from None
        prepared.append((expected, frames))
        for frame_index, frame in enumerate(frames, 1):
            item = {
                "segment_index": expected,
                "frame_index": frame_index,
                "frame_name": frame.name,
                "source_sha256": _sha256_regular(frame),
            }
            if canonical_plan["version"] == 4:
                frozen = receipt_by_frame.get((expected, frame_index), {})
                item.update(
                    source_transition_from_previous=frozen.get(
                        "source_transition_from_previous"
                    ),
                    source_transition_evidence_sha256=frozen.get(
                        "source_transition_evidence_sha256"
                    ),
                )
            actual_inventory.append(item)
    if actual_inventory != receipt["frames"]:
        raise ValueError("invalid image plan audit input")
    frame_counts = {index: len(frames) for index, frames in prepared}
    with tempfile.TemporaryDirectory(prefix="duet-image-plan-audit-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        work = stage / "work"
        work.mkdir(parents=True, mode=0o700)
        _copy_regular(skill, stage / "SKILL.md")
        (work / "request.json").write_text(
            json.dumps(
                {
                    "phase": "plan_audit",
                    "plan_sha256": receipt["plan_sha256"],
                    "continuity_sha256": receipt["continuity_sha256"],
                    "audit_input_sha256": receipt["sha256"],
                    "segment_indices": canonical_plan["segment_indices"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "frozen_plan.json").write_text(
            json.dumps(
                freeze_continuity(
                    canonical_plan, frame_counts=frame_counts
                )["_image_continuity"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "audit_inputs.json").write_text(
            _plan_json(receipt) + "\n", encoding="utf-8"
        )
        for index, frames in prepared:
            destination = work / "segments" / str(index) / "source"
            destination.mkdir(parents=True, mode=0o700)
            for frame in frames:
                _copy_regular(frame, destination / frame.name)
        run_error: CodexError | None = None
        try:
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md 的 plan_audit 阶段；只读取允许的输入，"
                "并写入规定的唯一输出文件。",
                session_dir=session,
            )
        except CodexError as error:
            run_error = error
        try:
            return canonical_plan_audit_verdict(
                _read_json_output(
                    work / "plan_audit.json",
                    MAX_CONTINUITY_BYTES
                    + MAX_PROJECT_OUTPUT_OVERHEAD_BYTES
                    + len(receipt["frames"])
                    * (5 if canonical_plan["version"] == 4 else 4)
                    * 2048,
                ),
                canonical_plan,
                receipt,
            )
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


def continuity_receipt(meta: dict) -> dict | None:
    raw = meta.get("_image_continuity")
    if not isinstance(raw, dict):
        return None
    if raw.get("version") == 1:
        if set(raw) != {"version", "segment_indices", "elements", "sha256"}:
            return None
        candidate = {
            "version": raw.get("version"),
            "segment_indices": raw.get("segment_indices"),
            "elements": raw.get("elements"),
        }
    elif raw.get("version") in {2, 3, 4}:
        if set(raw) != {
            "version", "phase", "segment_indices", "eligible", "reason",
            "person_plans", "scene_plans", "segments", "sha256",
        }:
            return None
        candidate = {key: value for key, value in raw.items() if key != "sha256"}
    else:
        return None
    try:
        frame_counts = None
        if candidate.get("version") in {3, 4}:
            raw_segments = candidate.get("segments")
            if not isinstance(raw_segments, list):
                return None
            frame_counts = {
                item.get("segment_index"): len(item.get("frame_constraints", []))
                for item in raw_segments if isinstance(item, dict)
            }
        expected = freeze_continuity(
            candidate, frame_counts=frame_counts
        )["_image_continuity"]
    except (ImageOptimizationOutputError, ImageOptimizationIneligibleError):
        return None
    return deepcopy(raw) if raw == expected else None


def dual_target_plan_receipt(meta: dict) -> dict | None:
    """Return a valid v2/v3/v4 dual-target receipt; reject corrupt state fail-closed."""
    if "_image_continuity" not in meta:
        return None
    raw = meta.get("_image_continuity")
    valid = continuity_receipt(meta)
    if valid is not None and valid.get("version") == 1:
        return None
    if valid is not None and valid.get("version") in {2, 3, 4}:
        return valid
    raise ImageOptimizationOutputError("image continuity receipt is invalid")


def _canonical_quality_check(value: object, *, allow_na: bool = False) -> dict:
    if not isinstance(value, dict) or set(value) != {"status", "evidence"}:
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    status = value.get("status")
    if status not in _VERIFY_STATUSES or (status == "not_applicable" and not allow_na):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    return {
        "status": status,
        "evidence": _canonical_text(value.get("evidence"), max_bytes=2048),
    }


def _verification_reason(segments: list[dict], project_checks: dict) -> str | None:
    checks = []
    for segment in segments:
        for person in segment["person_checks"]:
            checks.extend(
                [
                    person["identity_changed"],
                    person["source_identity_absent"],
                    person["local_color_change"],
                ]
            )
        checks.extend(segment["scene_checks"].values())
        checks.extend(segment["invariants"].values())
    checks.extend(project_checks.values())
    if any(item["status"] == "unknown" for item in checks):
        return "verification_unknown"
    for key, reason in (
        ("narrative_person_completeness", "narrative_person_incomplete"),
        ("no_identity_swap", "identity_swap_detected"),
        ("no_unplanned_person", "unplanned_person_detected"),
    ):
        if project_checks[key]["status"] == "fail":
            return reason
    for segment in segments:
        if any(
            person[key]["status"] == "fail"
            for person in segment["person_checks"]
            for key in ("identity_changed", "source_identity_absent")
        ):
            return "person_replacement_failed"
    for key, reason in (
        ("semantic_change", "scene_semantic_change_failed"),
        ("geometry_change", "scene_geometry_change_failed"),
        ("depth_change", "scene_depth_change_failed"),
        ("layout_change", "scene_layout_change_failed"),
    ):
        if any(segment["scene_checks"][key]["status"] == "fail" for segment in segments):
            return reason
    if any(
        check["status"] == "fail"
        for segment in segments
        for check in (
            [person["local_color_change"] for person in segment["person_checks"]]
            + [segment["scene_checks"]["local_color_change"]]
        )
    ):
        return "local_color_change_failed"
    for key, reason in (
        ("lighting_preservation", "lighting_preservation_failed"),
        ("interaction_preservation", "interaction_preservation_failed"),
        ("cross_frame_continuity", "cross_frame_continuity_failed"),
    ):
        if any(segment["invariants"][key]["status"] == "fail" for segment in segments):
            return reason
    if project_checks["person_identity_continuity"]["status"] == "fail":
        return "person_identity_continuity_failed"
    if project_checks["scene_continuity"]["status"] == "fail":
        return "scene_continuity_failed"
    return None


def _canonical_verification_v2(
    value: object, plan: dict, *, allow_empty_people: bool = False,
) -> dict:
    canonical_plan = _canonical_plan_v2(
        plan, allow_empty_people=allow_empty_people,
    )
    if not canonical_plan["eligible"]:
        raise ImageOptimizationIneligibleError(canonical_plan["reason"])
    if not isinstance(value, dict) or set(value) != {
        "version",
        "phase",
        "plan_sha256",
        "segment_indices",
        "passed",
        "reason",
        "segments",
        "project_checks",
    } or value.get("version") != 2 or value.get("phase") != "verify":
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    if (
        value.get("plan_sha256") != _canonical_plan_sha256(canonical_plan)
        or value.get("segment_indices") != canonical_plan["segment_indices"]
        or not isinstance(value.get("passed"), bool)
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(
        canonical_plan["segments"]
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    segments = []
    for planned, item in zip(canonical_plan["segments"], raw_segments):
        if not isinstance(item, dict) or set(item) != {
            "segment_index",
            "passed",
            "person_checks",
            "scene_checks",
            "invariants",
        } or item.get("segment_index") != planned["segment_index"] or not isinstance(
            item.get("passed"), bool
        ):
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        raw_people = item.get("person_checks")
        if not isinstance(raw_people, list) or len(raw_people) != len(planned["persons"]):
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        people = []
        for planned_person, check in zip(planned["persons"], raw_people):
            if not isinstance(check, dict) or set(check) != {
                "person_id",
                "identity_changed",
                "source_identity_absent",
                "local_color_change",
            } or check.get("person_id") != planned_person["id"]:
                raise ImageOptimizationOutputError(
                    "image verification output is missing or invalid"
                )
            allow_na = planned_person["state"] == "not_observable"
            canonical_checks = {
                key: _canonical_quality_check(check.get(key), allow_na=allow_na)
                for key in (
                    "identity_changed",
                    "source_identity_absent",
                    "local_color_change",
                )
            }
            statuses = {entry["status"] for entry in canonical_checks.values()}
            if (allow_na and statuses != {"not_applicable"}) or (
                not allow_na and "not_applicable" in statuses
            ):
                raise ImageOptimizationOutputError(
                    "image verification output is missing or invalid"
                )
            people.append({"person_id": planned_person["id"], **canonical_checks})
        raw_scene = item.get("scene_checks")
        raw_invariants = item.get("invariants")
        if not isinstance(raw_scene, dict) or set(raw_scene) != {
            "semantic_change",
            "geometry_change",
            "depth_change",
            "layout_change",
            "local_color_change",
        } or not isinstance(raw_invariants, dict) or set(raw_invariants) != {
            "lighting_preservation", "interaction_preservation", "cross_frame_continuity"
        }:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        scene_checks = {
            key: _canonical_quality_check(raw_scene.get(key))
            for key in (
                "semantic_change",
                "geometry_change",
                "depth_change",
                "layout_change",
                "local_color_change",
            )
        }
        invariants = {
            key: _canonical_quality_check(raw_invariants.get(key))
            for key in (
                "lighting_preservation",
                "interaction_preservation",
                "cross_frame_continuity",
            )
        }
        applicable = [
            check
            for person in people
            for check in (
                person["identity_changed"],
                person["source_identity_absent"],
                person["local_color_change"],
            )
            if check["status"] != "not_applicable"
        ] + list(scene_checks.values()) + list(invariants.values())
        segment_passed = all(check["status"] == "pass" for check in applicable)
        if item["passed"] != segment_passed:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        segments.append(
            {
                "segment_index": planned["segment_index"],
                "passed": segment_passed,
                "person_checks": people,
                "scene_checks": scene_checks,
                "invariants": invariants,
            }
        )
    raw_project = value.get("project_checks")
    if not isinstance(raw_project, dict) or set(raw_project) != set(_PROJECT_CHECKS):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    project_checks = {
        key: _canonical_quality_check(raw_project.get(key)) for key in _PROJECT_CHECKS
    }
    passed = all(segment["passed"] for segment in segments) and all(
        check["status"] == "pass" for check in project_checks.values()
    )
    reason = _verification_reason(segments, project_checks)
    if value["passed"] != passed or value.get("reason") != reason:
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    return {
        "version": 2,
        "phase": "verify",
        "plan_sha256": _canonical_plan_sha256(canonical_plan),
        "segment_indices": list(canonical_plan["segment_indices"]),
        "passed": passed,
        "reason": reason,
        "segments": segments,
        "project_checks": project_checks,
    }


def _reference_pack_verification_reason(
    persons: list[dict], scenes: list[dict], project: dict
) -> str | None:
    checks = [
        check
        for item in [*persons, *scenes]
        for check in item["checks"].values()
    ] + list(project.values())
    if any(check["status"] == "unknown" for check in checks):
        return "pack_verification_unknown"
    for key, reason in (
        ("identity_changed", "person_identity_change_failed"),
        ("source_identity_absent", "source_identity_residual"),
        ("multiview", "person_multiview_failed"),
        ("local_color", "person_local_color_failed"),
    ):
        if any(item["checks"][key]["status"] == "fail" for item in persons):
            return reason
    for key, reason in (
        ("semantic", "scene_semantic_failed"),
        ("geometry", "scene_geometry_failed"),
        ("depth", "scene_depth_failed"),
        ("layout", "scene_layout_failed"),
        ("local_color", "scene_local_color_failed"),
    ):
        if any(item["checks"][key]["status"] == "fail" for item in scenes):
            return reason
    for key, reason in (
        ("light_direction_preservation", "light_direction_preservation_failed"),
        ("exposure_preservation", "exposure_preservation_failed"),
        ("wb_cct_preservation", "wb_cct_preservation_failed"),
        ("tone_curve_preservation", "tone_curve_preservation_failed"),
    ):
        if project[key]["status"] == "fail":
            return reason
    return None


def _scene_continuity_evidence_tokens(scene: dict) -> tuple[str, ...]:
    graph = scene["continuity_graph"]
    scene_id = scene["id"]
    return tuple(
        [
            f"{scene_id}/{component['component_id']}"
            for component in graph["components"]
        ]
        + [
            f"{scene_id}/{relation['subject_id']} {relation['predicate']} "
            f"{scene_id}/{relation['object_id']}"
            for relation in graph["topology"]
        ]
    )


def _require_scene_continuity_evidence(evidence: str, scene: dict) -> None:
    if any(
        token not in evidence for token in _scene_continuity_evidence_tokens(scene)
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )


def canonical_reference_pack_verdict(value: object, plan: dict) -> dict:
    """Validate and derive the exact semantic replacement-pack verdict."""
    canonical_plan = _canonical_plan(plan)
    if not canonical_plan["eligible"]:
        raise ImageOptimizationIneligibleError(canonical_plan["reason"])
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "version",
            "phase",
            "plan_sha256",
            "passed",
            "reason",
            "persons",
            "scenes",
            "project",
        }
        or value.get("version") != canonical_plan["version"]
        or value.get("phase") != "verify_pack"
        or value.get("plan_sha256") != plan_sha256(canonical_plan)
        or not isinstance(value.get("passed"), bool)
    ):
        raise ImageOptimizationOutputError(
            "reference pack verification output is missing or invalid"
        )

    def canonical_entities(
        raw: object, planned: list[dict], id_key: str, check_keys: tuple[str, ...]
    ) -> list[dict]:
        if not isinstance(raw, list) or len(raw) != len(planned):
            raise ImageOptimizationOutputError(
                "reference pack verification output is missing or invalid"
            )
        result = []
        for item, design in zip(raw, planned):
            if (
                not isinstance(item, dict)
                or set(item) != {id_key, "passed", "checks"}
                or item.get(id_key) != design["id"]
                or not isinstance(item.get("passed"), bool)
                or not isinstance(item.get("checks"), dict)
                or set(item["checks"]) != set(check_keys)
            ):
                raise ImageOptimizationOutputError(
                    "reference pack verification output is missing or invalid"
                )
            checks = {
                key: _canonical_quality_check(item["checks"].get(key))
                for key in check_keys
            }
            passed = all(check["status"] == "pass" for check in checks.values())
            if item["passed"] != passed:
                raise ImageOptimizationOutputError(
                    "reference pack verification output is missing or invalid"
                )
            result.append({id_key: design["id"], "passed": passed, "checks": checks})
        return result

    persons = canonical_entities(
        value.get("persons"),
        canonical_plan["person_plans"],
        "person_id",
        _PACK_PERSON_CHECKS,
    )
    scenes = canonical_entities(
        value.get("scenes"),
        canonical_plan["scene_plans"],
        "scene_id",
        _PACK_SCENE_CHECKS,
    )
    if canonical_plan["version"] == 4:
        for result, design in zip(scenes, canonical_plan["scene_plans"]):
            _require_scene_continuity_evidence(
                " ".join(
                    result["checks"][key]["evidence"]
                    for key in ("geometry", "depth", "layout")
                ),
                design,
            )
    raw_project = value.get("project")
    if not isinstance(raw_project, dict) or set(raw_project) != set(
        _PACK_PROJECT_CHECKS
    ):
        raise ImageOptimizationOutputError(
            "reference pack verification output is missing or invalid"
        )
    project = {
        key: _canonical_quality_check(raw_project.get(key))
        for key in _PACK_PROJECT_CHECKS
    }
    passed = (
        all(item["passed"] for item in persons)
        and all(item["passed"] for item in scenes)
        and all(check["status"] == "pass" for check in project.values())
    )
    reason = _reference_pack_verification_reason(persons, scenes, project)
    if value["passed"] != passed or value.get("reason") != reason:
        raise ImageOptimizationOutputError(
            "reference pack verification output is missing or invalid"
        )
    return {
        "version": canonical_plan["version"],
        "phase": "verify_pack",
        "plan_sha256": plan_sha256(canonical_plan),
        "passed": passed,
        "reason": reason,
        "persons": persons,
        "scenes": scenes,
        "project": project,
    }


def _reference_pack_inputs(
    raw: object,
    planned: list[dict],
    id_key: str,
    session: Path,
) -> list[tuple[str, tuple[Path, Path, Path]]]:
    if not isinstance(raw, list) or len(raw) != len(planned):
        raise ValueError("invalid reference pack verification input")
    prepared = []
    for item, design in zip(raw, planned):
        if not isinstance(item, dict) or set(item) != {
            id_key,
            "source_path",
            "primary_path",
            "alternate_path",
        } or item.get(id_key) != design["id"]:
            raise ValueError("invalid reference pack verification input")
        paths = []
        for key in ("source_path", "primary_path", "alternate_path"):
            try:
                path = Path(item[key]).resolve(strict=True)
                path.relative_to(session)
            except (OSError, TypeError, ValueError):
                raise ValueError("invalid reference pack verification input") from None
            if path.suffix.lower() != ".png":
                raise ValueError("invalid reference pack verification input")
            paths.append(path)
        prepared.append((design["id"], tuple(paths)))
    return prepared


def generate_reference_pack_verdict(
    runner,
    plan: dict,
    person_packs: list[dict],
    scene_packs: list[dict],
    deterministic_metrics: dict,
    *,
    session_dir: Path,
) -> dict:
    """Run semantic replacement-pack verification in an isolated workspace."""
    try:
        session = Path(session_dir).resolve(strict=True)
        skill = verification_skill_path()
    except (OSError, TypeError, ValueError):
        raise ValueError("invalid reference pack verification input") from None
    canonical_plan = _canonical_plan(plan)
    if not canonical_plan["eligible"]:
        raise ImageOptimizationIneligibleError(canonical_plan["reason"])
    people = _reference_pack_inputs(
        person_packs, canonical_plan["person_plans"], "person_id", session
    )
    scenes = _reference_pack_inputs(
        scene_packs, canonical_plan["scene_plans"], "scene_id", session
    )
    if not isinstance(deterministic_metrics, dict):
        raise ValueError("invalid reference pack verification metrics")
    try:
        metrics_raw = json.dumps(
            deterministic_metrics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ValueError("invalid reference pack verification metrics") from None
    if len(metrics_raw.encode("utf-8")) > MAX_PROJECT_OUTPUT_OVERHEAD_BYTES:
        raise ValueError("invalid reference pack verification metrics")

    with tempfile.TemporaryDirectory(prefix="duet-image-pack-verify-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        work = stage / "work"
        work.mkdir(parents=True, mode=0o700)
        _copy_regular(skill, stage / "SKILL.md")
        digest = plan_sha256(canonical_plan)
        (work / "request.json").write_text(
            json.dumps(
                {
                    "phase": "verify_pack",
                    "plan_sha256": digest,
                    "person_ids": [item[0] for item in people],
                    "scene_ids": [item[0] for item in scenes],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "frozen_plan.json").write_text(
            json.dumps(
                freeze_continuity(
                    canonical_plan,
                    frame_counts=(
                        {
                            segment["segment_index"]: len(
                                segment["frame_constraints"]
                            )
                            for segment in canonical_plan["segments"]
                        }
                        if canonical_plan["version"] == 4
                        else None
                    ),
                )["_image_continuity"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "metrics.json").write_text(metrics_raw + "\n", encoding="utf-8")
        for label, prepared in (("persons", people), ("scenes", scenes)):
            for identifier, paths in prepared:
                destination = work / "reference_packs" / label / identifier
                destination.mkdir(parents=True, mode=0o700)
                for name, source in zip(("source", "primary", "alternate"), paths):
                    _copy_regular(source, destination / f"{name}.png")
        run_error: CodexError | None = None
        try:
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md 的 verify_pack 阶段；只读取允许的输入，"
                "并写入规定的唯一输出文件。",
                session_dir=session,
            )
        except CodexError as error:
            run_error = error
        try:
            return canonical_reference_pack_verdict(
                _read_json_output(
                    work / "reference_pack_verification.json",
                    MAX_CONTINUITY_BYTES + MAX_PROJECT_OUTPUT_OVERHEAD_BYTES,
                ),
                canonical_plan,
            )
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise
def _v3_common_projection(value: dict, plan: dict) -> dict:
    base_plan = deepcopy(plan)
    base_plan["version"] = 2
    for segment in base_plan["segments"]:
        segment.pop("frame_constraints")
        segment.pop("photometric_contract")
    base = deepcopy(value)
    base["version"] = 2
    base["plan_sha256"] = _canonical_plan_sha256(base_plan)
    try:
        for segment in base["segments"]:
            segment.pop("frame_checks")
            checks = [
                item[key]
                for item in segment["person_checks"]
                for key in (
                    "identity_changed", "source_identity_absent", "local_color_change",
                )
            ] + list(segment["scene_checks"].values()) + list(
                segment["invariants"].values()
            )
            segment["passed"] = all(
                isinstance(check, dict) and check.get("status") in {
                    "pass", "not_applicable",
                }
                for check in checks
            )
        project_checks = base["project_checks"]
        base["passed"] = all(item["passed"] for item in base["segments"]) and all(
            isinstance(check, dict) and check.get("status") == "pass"
            for check in project_checks.values()
        )
        base["reason"] = _verification_reason(
            base["segments"], project_checks
        )
    except (KeyError, TypeError, AttributeError):
        pass
    return base, base_plan


def _canonical_verification_v3(
    value: object, plan: dict, *, derive_claims: bool = False,
    allow_sparse_facts: bool = False,
) -> dict:
    canonical_plan = _canonical_plan_v3(
        plan, allow_sparse_facts=allow_sparse_facts,
    )
    keys = {
        "version", "phase", "plan_sha256", "segment_indices", "passed", "reason",
        "segments", "project_checks",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("version") != 3
        or value.get("phase") != "verify"
        or value.get("plan_sha256") != _canonical_plan_sha256(canonical_plan)
        or value.get("segment_indices") != canonical_plan["segment_indices"]
        or not isinstance(value.get("passed"), bool)
        or not isinstance(value.get("segments"), list)
        or len(value["segments"]) != len(canonical_plan["segments"])
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    common_value, common_plan = _v3_common_projection(value, canonical_plan)
    common = _canonical_verification_v2(
        common_value, common_plan, allow_empty_people=allow_sparse_facts,
    )
    segments = []
    any_unknown = common["reason"] == "verification_unknown"
    any_body_failure = False
    any_photo_failure = False
    any_palette_failure = False
    for expected, raw, common_segment in zip(
        canonical_plan["segments"], value["segments"], common["segments"]
    ):
        if not isinstance(raw, dict) or set(raw) != {
            "segment_index", "passed", "person_checks", "scene_checks", "invariants",
            "frame_checks",
        } or raw.get("segment_index") != expected["segment_index"]:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        raw_checks = raw.get("frame_checks")
        if not isinstance(raw_checks, list) or len(raw_checks) != len(
            expected["frame_constraints"]
        ):
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        frame_checks = []
        for constraint, check in zip(expected["frame_constraints"], raw_checks):
            if not isinstance(check, dict) or set(check) != {
                "frame_index", "visible_body_parts", "pose_skeleton", "contact_points",
                "occlusion_order", "out_of_frame_crop", "non_person_entity_ledger",
                "dominant_palette_contract",
                "photometric_contract",
            } or check.get("frame_index") != constraint["frame_index"]:
                raise ImageOptimizationOutputError(
                    "image verification output is missing or invalid"
                )
            canonical_checks = {
                key: _canonical_quality_check(check.get(key))
                for key in _FRAME_CONSTRAINT_KEYS[1:] + ("photometric_contract",)
            }
            statuses = [item["status"] for item in canonical_checks.values()]
            any_unknown = any_unknown or "unknown" in statuses
            any_body_failure = any_body_failure or any(
                canonical_checks[key]["status"] == "fail"
                for key in _FRAME_TEXT_CONSTRAINT_KEYS + (
                    "non_person_entity_ledger",
                )
            )
            any_palette_failure = any_palette_failure or (
                canonical_checks["dominant_palette_contract"]["status"] == "fail"
            )
            any_photo_failure = any_photo_failure or (
                canonical_checks["photometric_contract"]["status"] == "fail"
            )
            frame_checks.append({
                "frame_index": constraint["frame_index"], **canonical_checks,
            })
        segment_passed = common_segment["passed"] and all(
            check["status"] == "pass"
            for frame in frame_checks
            for key, check in frame.items()
            if key != "frame_index"
        )
        if not derive_claims and raw.get("passed") != segment_passed:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        segments.append({**common_segment, "passed": segment_passed, "frame_checks": frame_checks})
    passed = all(segment["passed"] for segment in segments) and all(
        check["status"] == "pass" for check in common["project_checks"].values()
    )
    reason = (
        "verification_unknown" if any_unknown else common["reason"]
    )
    if reason is None and any_body_failure:
        reason = "interaction_preservation_failed"
    if reason is None and any_palette_failure:
        reason = "dominant_palette_preservation_failed"
    if reason is None and any_photo_failure:
        reason = "lighting_preservation_failed"
    if not derive_claims and (
        value["passed"] != passed or value.get("reason") != reason
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    return {
        "version": 3,
        "phase": "verify",
        "plan_sha256": _canonical_plan_sha256(canonical_plan),
        "segment_indices": list(canonical_plan["segment_indices"]),
        "passed": passed,
        "reason": reason,
        "segments": segments,
        "project_checks": common["project_checks"],
    }


def _v4_common_projection(value: dict, plan: dict) -> dict:
    v3_plan = deepcopy(plan)
    v3_plan["version"] = 3
    for scene in v3_plan["scene_plans"]:
        scene.pop("continuity_graph")
    v3_value = deepcopy(value)
    v3_value["version"] = 3
    v3_value["plan_sha256"] = _canonical_plan_sha256(v3_plan)
    try:
        for segment in v3_value["segments"]:
            for frame in segment["frame_checks"]:
                frame.pop("scene_continuity_view")
        return _canonical_verification_v3(
            v3_value, v3_plan, derive_claims=True, allow_sparse_facts=True,
        )
    except (KeyError, TypeError, AttributeError, ImageOptimizationOutputError):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        ) from None


def _canonical_verification_v4(value: object, plan: dict) -> dict:
    canonical_plan = _canonical_plan_v4(plan)
    keys = {
        "version", "phase", "plan_sha256", "segment_indices", "passed", "reason",
        "segments", "project_checks",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("version") != 4
        or value.get("phase") != "verify"
        or value.get("plan_sha256") != plan_sha256(canonical_plan)
        or value.get("segment_indices") != canonical_plan["segment_indices"]
        or not isinstance(value.get("passed"), bool)
        or not isinstance(value.get("segments"), list)
        or len(value["segments"]) != len(canonical_plan["segments"])
    ):
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    common = _v4_common_projection(value, canonical_plan)
    for scene in canonical_plan["scene_plans"]:
        _require_scene_continuity_evidence(
            common["project_checks"]["scene_continuity"]["evidence"], scene
        )
    segments = []
    any_unknown = common["reason"] == "verification_unknown"
    any_continuity_failure = False
    for expected, raw, common_segment in zip(
        canonical_plan["segments"], value["segments"], common["segments"]
    ):
        if not isinstance(raw, dict) or set(raw) != {
            "segment_index", "passed", "person_checks", "scene_checks", "invariants",
            "frame_checks",
        } or raw.get("segment_index") != expected["segment_index"]:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        raw_checks = raw.get("frame_checks")
        if not isinstance(raw_checks, list) or len(raw_checks) != len(
            expected["frame_constraints"]
        ):
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        frame_checks = []
        for constraint, raw_check, common_check in zip(
            expected["frame_constraints"], raw_checks, common_segment["frame_checks"]
        ):
            expected_keys = {
                "frame_index", *_FRAME_CONSTRAINT_KEYS[1:], "photometric_contract",
                "scene_continuity_view",
            }
            if (
                not isinstance(raw_check, dict)
                or set(raw_check) != expected_keys
                or raw_check.get("frame_index") != constraint["frame_index"]
            ):
                raise ImageOptimizationOutputError(
                    "image verification output is missing or invalid"
                )
            continuity_checks = {
                "scene_continuity_view": _canonical_quality_check(
                    raw_check.get("scene_continuity_view")
                )
            }
            statuses = [item["status"] for item in continuity_checks.values()]
            any_unknown = any_unknown or "unknown" in statuses
            any_continuity_failure = any_continuity_failure or "fail" in statuses
            frame_checks.append({**common_check, **continuity_checks})
        segment_passed = common_segment["passed"] and all(
            check["status"] == "pass"
            for frame in frame_checks
            for key, check in frame.items()
            if key != "frame_index"
        )
        if raw.get("passed") != segment_passed:
            raise ImageOptimizationOutputError(
                "image verification output is missing or invalid"
            )
        segments.append({
            **common_segment,
            "passed": segment_passed,
            "frame_checks": frame_checks,
        })
    passed = all(segment["passed"] for segment in segments) and all(
        check["status"] == "pass" for check in common["project_checks"].values()
    )
    reason = "verification_unknown" if any_unknown else common["reason"]
    if reason is None and any_continuity_failure:
        reason = "scene_continuity_failed"
    if value["passed"] != passed or value.get("reason") != reason:
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    return {
        "version": 4,
        "phase": "verify",
        "plan_sha256": plan_sha256(canonical_plan),
        "segment_indices": list(canonical_plan["segment_indices"]),
        "passed": passed,
        "reason": reason,
        "segments": segments,
        "project_checks": common["project_checks"],
    }


def canonical_verification(value: object, plan: dict) -> dict:
    if isinstance(plan, dict) and plan.get("version") == 4:
        return _canonical_verification_v4(value, plan)
    if isinstance(plan, dict) and plan.get("version") == 3:
        return _canonical_verification_v3(value, plan)
    return _canonical_verification_v2(value, plan)
def generate_project_verdict(
    runner,
    plan: dict,
    segments: list[dict],
    deterministic_metrics: dict,
    *,
    session_dir: Path,
) -> dict:
    """Run verify in an isolated workspace and return a strict verdict."""
    try:
        session = Path(session_dir).resolve(strict=True)
        skill = verification_skill_path()
    except OSError:
        raise ValueError("invalid image verification input") from None
    canonical_plan = _canonical_plan(plan)
    if not canonical_plan["eligible"]:
        raise ImageOptimizationIneligibleError(canonical_plan["reason"])
    if (
        not skill.is_file()
        or not isinstance(segments, list)
        or len(segments) != len(canonical_plan["segment_indices"])
        or not isinstance(deterministic_metrics, dict)
    ):
        raise ValueError("invalid image verification input")
    try:
        metrics_raw = json.dumps(
            deterministic_metrics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ValueError("invalid image verification metrics") from None
    if len(metrics_raw.encode("utf-8")) > MAX_PROJECT_OUTPUT_OVERHEAD_BYTES:
        raise ValueError("invalid image verification metrics")
    prepared = []
    frame_counts = {}
    for expected, segment in zip(canonical_plan["segment_indices"], segments):
        if not isinstance(segment, dict) or set(segment) != {
            "index", "source_keyframes_dir", "output_keyframes_dir"
        } or segment.get("index") != expected:
            raise ValueError("invalid image verification input")
        pair = []
        for key in ("source_keyframes_dir", "output_keyframes_dir"):
            try:
                path = Path(segment[key]).resolve(strict=True)
                path.relative_to(session)
            except (OSError, TypeError, ValueError):
                raise ValueError("invalid image verification input") from None
            pair.append(_validated_frames(path))
        if [item.name for item in pair[0]] != [item.name for item in pair[1]]:
            raise ValueError("invalid image verification frames")
        prepared.append((expected, pair[0], pair[1]))
        frame_counts[expected] = len(pair[0])
    canonical_plan = _canonical_plan(
        canonical_plan, canonical_plan["segment_indices"], frame_counts
    )
    with tempfile.TemporaryDirectory(prefix="duet-image-verify-", dir="/tmp") as raw:
        stage = Path(raw).resolve(strict=True)
        work = stage / "work"
        work.mkdir(parents=True, mode=0o700)
        _copy_regular(skill, stage / "SKILL.md")
        (work / "request.json").write_text(
            json.dumps(
                {
                    "phase": "verify",
                    "segment_indices": canonical_plan["segment_indices"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "frozen_plan.json").write_text(
            json.dumps(
                freeze_continuity(
                    canonical_plan, frame_counts=frame_counts
                )["_image_continuity"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (work / "metrics.json").write_text(metrics_raw + "\n", encoding="utf-8")
        for index, source_frames, output_frames in prepared:
            for label, frames in (("source", source_frames), ("output", output_frames)):
                destination = work / "segments" / str(index) / label
                destination.mkdir(parents=True, mode=0o700)
                for frame in frames:
                    _copy_regular(frame, destination / frame.name)
        run_error: CodexError | None = None
        try:
            runner.run_isolated(
                stage,
                "严格执行当前目录 SKILL.md 的 verify 阶段；只读取允许的输入，"
                "并写入规定的唯一输出文件。",
                session_dir=session,
            )
        except CodexError as error:
            run_error = error
        try:
            return canonical_verification(
                _read_json_output(
                    work / "image_verification.json",
                    MAX_CONTINUITY_BYTES + MAX_PROJECT_OUTPUT_OVERHEAD_BYTES,
                ),
                canonical_plan,
            )
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


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


def _receipt_v4(raw: dict, meta: dict) -> dict | None:
    if (
        set(raw) != {
            "version", "plan_sha256", "continuity_sha256",
            "execution_input_sha256", "execution_inputs", "model", "edit_mode",
            "scene_anchor_schedule", "frames", "sha256",
        }
        or raw.get("model") not in SEEDREAM_MODELS
        or raw.get("edit_mode") not in SEEDREAM_EDIT_MODES
        or not isinstance(raw.get("execution_inputs"), dict)
        or raw.get("execution_input_sha256")
        != raw["execution_inputs"].get("sha256")
        or raw.get("sha256") != sha256(_plan_json({
            key: value for key, value in raw.items() if key != "sha256"
        }))
    ):
        return None
    try:
        frozen_plan = dual_target_plan_receipt(meta)
        if frozen_plan is None or frozen_plan.get("version") != 4:
            return None
        plan = {key: value for key, value in frozen_plan.items() if key != "sha256"}
        execution = raw["execution_inputs"]
        inventory = [
            {
                key: frame[key] for key in (
                    "segment_index", "frame_index", "frame_name", "source_sha256",
                    "source_transition_from_previous",
                    "source_transition_evidence_sha256",
                )
            }
            for frame in execution["frames"]
        ]
        expected_execution = freeze_execution_inputs(
            plan,
            revision=execution["revision"],
            profile=execution["profile"],
            model=execution["model"],
            frame_inventory=inventory,
        )
        expected_prompts = compile_frame_prompts(
            plan,
            raw["edit_mode"],
            _compiler_revision=_frame_prompt_compiler_revision(
                execution["profile"]
            ),
        )
        expected_schedule = _scene_anchor_schedule(plan, expected_execution)
    except (
        KeyError, TypeError, ValueError, ImageOptimizationOutputError,
        ImageOptimizationIneligibleError,
    ):
        return None
    if (
        execution != expected_execution
        or raw.get("plan_sha256") != expected_execution["plan_sha256"]
        or raw["continuity_sha256"] != expected_execution["continuity_sha256"]
        or raw.get("scene_anchor_schedule") != expected_schedule
        or not isinstance(raw.get("frames"), list)
        or len(raw["frames"]) != len(execution["frames"])
    ):
        return None
    for frozen, frame in zip(raw["frames"], execution["frames"]):
        if not isinstance(frozen, dict) or set(frozen) != {
            "segment_index", "frame_name", "source_sha256",
            "default", "current", "sha256",
        }:
            return None
        try:
            text = expected_prompts[frame["segment_index"]][frame["frame_index"]]
        except (KeyError, TypeError):
            return None
        if frozen != {
            "segment_index": frame["segment_index"],
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
            "default": text,
            "current": text,
            "sha256": sha256(text),
        }:
            return None
    return deepcopy(raw)


def receipt(meta: dict, settings: Settings | None = None) -> dict | None:
    raw = meta.get("_image_optimization")
    if isinstance(raw, dict):
        if raw.get("version") == 4:
            return _receipt_v4(raw, meta)
        if raw.get("version") == 3:
            if (
                set(raw) != {
                    "version", "plan_sha256", "model", "edit_mode", "frames", "sha256",
                }
                or raw.get("model") not in SEEDREAM_MODELS
                or raw.get("edit_mode") not in SEEDREAM_EDIT_MODES
                or not isinstance(raw.get("frames"), list)
                or not raw["frames"]
                or not isinstance(raw.get("sha256"), str)
                or raw["sha256"] != sha256(_plan_json({
                    key: value for key, value in raw.items() if key != "sha256"
                }))
            ):
                return None
            try:
                plan = dual_target_plan_receipt(meta)
            except ImageOptimizationOutputError:
                return None
            if (
                plan is None
                or plan.get("version") != raw.get("version")
                or raw.get("plan_sha256") != plan_sha256(
                    {key: value for key, value in plan.items() if key != "sha256"}
                )
            ):
                return None
            seen = set()
            for item in raw["frames"]:
                if not isinstance(item, dict) or set(item) != {
                    "segment_index", "frame_name", "source_sha256",
                    "default", "current", "sha256",
                }:
                    return None
                index, name, source, current = (
                    item.get("segment_index"), item.get("frame_name"),
                    item.get("source_sha256"), item.get("current"),
                )
                if (
                    isinstance(index, bool) or not isinstance(index, int)
                    or not isinstance(name, str) or not re.fullmatch(
                        r"[0-9]{2}\.png", name
                    )
                    or (index, name) in seen
                    or not isinstance(source, str)
                    or _SHA256_RE.fullmatch(source) is None
                    or not isinstance(item.get("default"), str)
                    or not item["default"].strip()
                    or not isinstance(current, str) or not current.strip()
                    or len(item["default"].encode("utf-8")) > MAX_PROMPT_BYTES
                    or len(current.encode("utf-8")) > MAX_PROMPT_BYTES
                    or item.get("sha256") != sha256(current)
                ):
                    return None
                seen.add((index, name))
            expected = {
                (segment["segment_index"], f"{constraint['frame_index']:02d}.png")
                for segment in plan["segments"]
                for constraint in segment["frame_constraints"]
            }
            if seen != expected:
                return None
            return deepcopy(raw)
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
    try:
        compiled_plan = dual_target_plan_receipt(meta)
    except ImageOptimizationOutputError:
        raise ImageOptimizationError(
            409, "image_optimization_plan_invalid"
        ) from None
    if compiled_plan is not None:
        raise ImageOptimizationError(409, "image_optimization_prompt_compiled")
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
