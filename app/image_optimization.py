"""Project-level prompt generation, frozen segment prompts, and strict CAS editing."""

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
MAX_PROJECT_OUTPUT_OVERHEAD_BYTES = 64 * 1024
_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _ROOT / "skills" / "image-postprocess" / "SKILL.md"
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    if not isinstance(raw_people, list) or not 1 <= len(raw_people) <= 20:
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
)
_NON_PERSON_ENTITY_LEDGER_KEYS = {"entities", "relations"}
_ENTITY_LEDGER_ENTITY_KEYS = {"entity_id", "description", "visibility"}
_ENTITY_LEDGER_RELATION_KEYS = {"subject_id", "predicate", "object_id"}
_ENTITY_VISIBILITIES = {"full", "partial", "edge_fragment"}
_ENTITY_RELATION_PREDICATES = {
    "supports", "contacts", "separate_from", "occludes",
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


def _canonical_non_person_entity_ledger(
    value: object, allowed_person_ids: set[str],
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
        or not raw_entities
        or len(raw_entities) > 30
        or not isinstance(raw_relations, list)
        or not raw_relations
        or len(raw_relations) > 60
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

    identifiers = entity_ids | allowed_person_ids
    relations = []
    physical_pairs = set()
    occlusion_pairs = set()
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
        else:
            if pair in occlusion_pairs:
                raise ImageOptimizationOutputError(
                    "image optimization output is missing or invalid"
                )
            occlusion_pairs.add(pair)
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
    ) or any(_contains_directed_cycle(edges) for edges in directed_edges.values()) or {
        identifier
        for relation in relations
        for identifier in (relation["subject_id"], relation["object_id"])
        if identifier in entity_ids
    } != entity_ids:
        raise ImageOptimizationOutputError(
            "image optimization output is missing or invalid"
        )
    return {"entities": entities, "relations": relations}


def _canonical_frame_constraint(value: object, allowed_person_ids: set[str]) -> dict:
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
            value.get("non_person_entity_ledger"), allowed_person_ids
        ),
    }


def _canonical_plan_v3(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
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
    canonical = _canonical_plan_v2(v2_value, expected_indices, frame_counts)
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
            constraint = _canonical_frame_constraint(item, allowed_person_ids)
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


def _canonical_plan(
    value: object,
    expected_indices: list[int] | None = None,
    frame_counts: dict[int, int] | None = None,
) -> dict:
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


def plan_sha256(plan: dict) -> str:
    canonical = _canonical_plan(plan)
    return hashlib.sha256(_plan_json(canonical).encode("utf-8")).hexdigest()


def compile_segment_prompts(plan: dict, edit_mode: str) -> dict[int, str]:
    """Compile an eligible semantic v2 plan into immutable provider prompts."""
    if edit_mode not in SEEDREAM_EDIT_MODES:
        raise ValueError("unsupported image optimization edit mode")
    canonical = _canonical_plan_v2(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    people = {item["id"]: item for item in canonical["person_plans"]}
    scenes = {item["id"]: item for item in canonical["scene_plans"]}
    prompts = {}
    for segment in canonical["segments"]:
        person_edits = []
        hidden_people = False
        for member in segment["persons"]:
            if member["state"] == "not_observable":
                hidden_people = True
                continue
            design = people[member["id"]]
            person_edits.append(
                "将{target}完整替换为{identity}，{wardrobe}，{color}，编辑边界为{boundary}".format(
                    target=member["target_region"],
                    identity=design["replacement_identity"],
                    wardrobe=design["wardrobe_change"],
                    color=design["local_color_change"],
                    boundary=member["boundary"],
                )
            )
        person_clause = "替换人物：" + "；".join(person_edits)
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
        invariant_clause = (
            "保持当前源图的画幅、裁切、机位、镜头、透视、构图、焦点、景深、"
            "人物数量、姿态、动作、视线及核心实体不变；光源方向、曝光、白平衡、"
            "色温、全局色调曲线保持与当前源图一致，只允许新几何产生物理正确的"
            f"局部阴影；严格保持{protected}以及持握、接触、遮挡、前后关系和动作目的"
            f"{non_target_clause}；禁止文字、Logo、水印、畸变、融合、增删实体或画质美化"
        )
        mode_clause = (
            "图1始终是唯一编辑画布；其他输入图只提供冻结人物身份、场景设计和"
            "本段布局，不传递构图、机位、动作、光线、实体关系。"
        )
        prompts[segment["segment_index"]] = _canonical_prompt(
            f"{person_clause}。{scene_clause}。{invariant_clause}。{mode_clause}"
        )
    return prompts


def compile_frame_prompts(plan: dict, edit_mode: str) -> dict[int, dict[int, str]]:
    """Compile an eligible v3 plan into one immutable prompt per source frame."""
    canonical = _canonical_plan_v3(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
    base_plan = deepcopy(canonical)
    base_plan["version"] = 2
    for segment in base_plan["segments"]:
        segment.pop("frame_constraints")
        segment.pop("photometric_contract")
    segment_prompts = compile_segment_prompts(base_plan, edit_mode)
    prompts: dict[int, dict[int, str]] = {}
    for segment in canonical["segments"]:
        photo = segment["photometric_contract"]
        photo_clause = "；".join(
            f"{key}={photo[key]}" for key in _PHOTOMETRIC_CONTRACT_KEYS
        )
        per_frame = {}
        for constraint in segment["frame_constraints"]:
            frame_clause = "；".join(
                f"{key}={constraint[key]}"
                for key in ("frame_index", *_FRAME_TEXT_CONSTRAINT_KEYS)
            )
            frame_clause = (
                f"{frame_clause}；non_person_entity_ledger="
                f"{_plan_json(constraint['non_person_entity_ledger'])}"
            )
            per_frame[constraint["frame_index"]] = _canonical_prompt(
                f"{segment_prompts[segment['segment_index']]}。仅当前源帧硬约束："
                f"{frame_clause}。全局光色硬约束：{photo_clause}。"
                "不得从其他帧补全，不得重布全局光线。"
            )
        prompts[segment["segment_index"]] = per_frame
    return prompts


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


def freeze_execution_inputs(
    plan: dict,
    *,
    revision: int,
    profile: dict,
    model: str,
    frame_inventory: list[dict],
) -> dict:
    canonical = _canonical_plan(plan)
    if not canonical["eligible"]:
        raise ImageOptimizationIneligibleError(canonical["reason"])
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
        or not isinstance(frame_inventory, list)
        or not frame_inventory
    ):
        raise ValueError("invalid image optimization execution inputs")
    inventory = []
    lookup: dict[tuple[int, int], dict] = {}
    prior: tuple[int, int] | None = None
    for item in frame_inventory:
        if not isinstance(item, dict) or set(item) != {
            "segment_index", "frame_index", "frame_name", "source_sha256"
        }:
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

    slots = reference_slots(canonical)

    def freeze_slot(item: dict) -> dict:
        source = lookup.get((item["segment_index"], item["frame_index"]))
        if source is None:
            raise ValueError("invalid image optimization execution inputs")
        return {**deepcopy(item), "source_sha256": source["source_sha256"]}

    segments = {item["segment_index"]: item for item in canonical["segments"]}
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
        if canonical["version"] == 3:
            constraint = next(
                item
                for item in segment["frame_constraints"]
                if item["frame_index"] == source["frame_index"]
            )
            current.update(
                frame_constraint=deepcopy(constraint),
                photometric_contract=deepcopy(segment["photometric_contract"]),
            )
        frames.append(current)
    return {
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


def freeze_frame_prompts(
    settings: Settings,
    execution_inputs: dict,
    prompts: dict[int, dict[int, str]],
) -> dict:
    """Freeze v3 prompts against one exact source frame identity per call."""
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
) -> tuple[dict, dict]:
    plan = _canonical_plan(value, indices, frame_counts)
    if not plan["eligible"]:
        raise ImageOptimizationIneligibleError(plan["reason"])
    if plan["version"] == 3:
        return plan, compile_frame_prompts(plan, edit_mode)
    return plan, compile_segment_prompts(plan, edit_mode)


def generate_project_prompts(
    runner,
    segments: list[dict],
    edit_mode: str,
    *,
    session_dir: Path,
) -> tuple[dict, dict]:
    """Run the plan phase once, then compile immutable provider prompts."""
    if edit_mode not in SEEDREAM_EDIT_MODES:
        raise ValueError("unsupported image optimization edit mode")
    try:
        session = Path(session_dir).resolve(strict=True)
        skill = verification_skill_path()
    except OSError:
        raise ValueError("invalid image optimization input") from None
    if not skill.is_file() or not isinstance(segments, list) or not segments:
        raise ValueError("invalid image optimization segments")
    indices = [item.get("index") for item in segments if isinstance(item, dict)]
    if len(indices) != len(segments) or indices not in ([0], list(range(1, len(segments) + 1))):
        raise ValueError("invalid image optimization segments")
    prepared = []
    for segment in segments:
        if set(segment) != {"index", "chain_id", "join_mode", "keyframes_dir"}:
            raise ValueError("invalid image optimization segments")
        if (
            not isinstance(segment["chain_id"], str) or not segment["chain_id"]
            or len(segment["chain_id"]) > 128
            or segment["join_mode"] not in {"hard_cut", "continue"}
        ):
            raise ValueError("invalid image optimization segments")
        try:
            source = Path(segment["keyframes_dir"]).resolve(strict=True)
            source.relative_to(session)
        except (OSError, TypeError, ValueError):
            raise ValueError("invalid image optimization segments") from None
        prepared.append((segment, _validated_frames(source)))

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
            })
        (work / "request.json").write_text(
            json.dumps({
                "phase": "plan",
                "edit_mode": edit_mode,
                "segments": request_segments,
            }, ensure_ascii=False, separators=(",", ":")) + "\n",
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
            return _canonical_project_output(
                _read_json_output(work / "image_optimization.json", max_bytes),
                indices,
                edit_mode,
                {segment["index"]: len(frames) for segment, frames in prepared},
            )
        except ImageOptimizationIneligibleError:
            raise
        except ImageOptimizationOutputError:
            if run_error is not None:
                raise run_error from None
            raise


def freeze_continuity(
    plan: dict, *, frame_counts: dict[int, int] | None = None
) -> dict:
    if isinstance(plan, dict) and plan.get("version") in {2, 3}:
        canonical = _canonical_plan(plan, frame_counts=frame_counts)
        if canonical["version"] == 3 and frame_counts is None:
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
    elif raw.get("version") in {2, 3}:
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
        if candidate.get("version") == 3:
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
    """Return a valid v2/v3 dual-target receipt; reject corrupt state fail-closed."""
    if "_image_continuity" not in meta:
        return None
    raw = meta.get("_image_continuity")
    valid = continuity_receipt(meta)
    if valid is not None and valid.get("version") == 1:
        return None
    if valid is not None and valid.get("version") in {2, 3}:
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


def _canonical_verification_v2(value: object, plan: dict) -> dict:
    canonical_plan = _canonical_plan(plan)
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
        value.get("plan_sha256") != plan_sha256(canonical_plan)
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
        "plan_sha256": plan_sha256(canonical_plan),
        "segment_indices": list(canonical_plan["segment_indices"]),
        "passed": passed,
        "reason": reason,
        "segments": segments,
        "project_checks": project_checks,
    }


def _v3_common_projection(value: dict, plan: dict) -> dict:
    base_plan = deepcopy(plan)
    base_plan["version"] = 2
    for segment in base_plan["segments"]:
        segment.pop("frame_constraints")
        segment.pop("photometric_contract")
    base = deepcopy(value)
    base["version"] = 2
    base["plan_sha256"] = plan_sha256(base_plan)
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


def _canonical_verification_v3(value: object, plan: dict) -> dict:
    canonical_plan = _canonical_plan_v3(plan)
    keys = {
        "version", "phase", "plan_sha256", "segment_indices", "passed", "reason",
        "segments", "project_checks",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("version") != 3
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
    common_value, common_plan = _v3_common_projection(value, canonical_plan)
    common = _canonical_verification_v2(common_value, common_plan)
    segments = []
    any_unknown = common["reason"] == "verification_unknown"
    any_body_failure = False
    any_photo_failure = False
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
                for key in _FRAME_CONSTRAINT_KEYS[1:]
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
        if raw.get("passed") != segment_passed:
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
    if reason is None and any_photo_failure:
        reason = "lighting_preservation_failed"
    if value["passed"] != passed or value.get("reason") != reason:
        raise ImageOptimizationOutputError(
            "image verification output is missing or invalid"
        )
    return {
        "version": 3,
        "phase": "verify",
        "plan_sha256": plan_sha256(canonical_plan),
        "segment_indices": list(canonical_plan["segment_indices"]),
        "passed": passed,
        "reason": reason,
        "segments": segments,
        "project_checks": common["project_checks"],
    }


def canonical_verification(value: object, plan: dict) -> dict:
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


def receipt(meta: dict, settings: Settings | None = None) -> dict | None:
    raw = meta.get("_image_optimization")
    if isinstance(raw, dict):
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
                or plan.get("version") != 3
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
