#!/usr/bin/env python3
"""Offline scorer for real image-postprocess artifacts.

The source frames and element index are the evidence.  Structural numbers are
descriptive; visual numbers come from a separate frame-by-frame review and
``decision`` is intentionally always ``None``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STABLE_KEY = re.compile(r"stable_key=([A-Za-z0-9_.-]+)")
VISUAL_AXES = (
    "person_and_object_identity_stability",
    "composition_preservation",
    "camera_preservation",
    "perspective_preservation",
    "tone_preservation",
    "lighting_preservation",
    "action_and_pose_preservation",
    "occlusion_and_contact_preservation",
    "no_cross_frame_completion",
    "no_new_entities",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 100.0


def _frames(source_root: Path) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    segments = source_root / "work" / "segments"
    for segment in sorted(
        (item for item in segments.iterdir() if item.is_dir() and item.name.isdigit()),
        key=lambda item: int(item.name),
    ):
        for candidate in (
            segment / "work" / "keyframes",
            segment / "keyframes",
        ):
            if candidate.is_dir():
                for frame in sorted(
                    candidate.glob("*.png"),
                    key=lambda item: int(item.stem) if item.stem.isdigit() else 10**9,
                ):
                    if frame.stem.isdigit():
                        result[(int(segment.name), int(frame.stem))] = frame
                break
    return result


def _expected(index: dict, frames: dict[tuple[int, int], Path]) -> dict:
    result = {
        key: {"people": set(), "entities": set(), "scenes": set()}
        for key in frames
    }
    for kind in ("people", "entities", "scenes"):
        for stable_key, spec in index.get(kind, {}).items():
            for occurrence in spec.get("occurrences", []):
                segment = occurrence.get("segment_index")
                for frame in occurrence.get("frame_orders", []):
                    key = (segment, frame)
                    if key in result:
                        result[key][kind].add(stable_key)
    return result


def _transitions(
    source_root: Path,
    frames: dict[tuple[int, int], Path],
    request: dict | None,
) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    if request is not None:
        for segment in request.get("segments", []):
            for item in segment.get("transition_skeleton", []):
                result[(item["segment_index"], item["frame_index"])] = item[
                    "source_transition_from_previous"
                ]
        return result
    for (segment, frame) in frames:
        if frame == 1 and segment == min(item[0] for item in frames):
            result[(segment, frame)] = "start"
            continue
        sampling_path = (
            source_root
            / "work"
            / "segments"
            / str(segment)
            / "work"
            / "keyframe_sampling.json"
        )
        transition = "same_camera"
        if sampling_path.is_file():
            sampling = _load(sampling_path).get("keyframes", [])
            item = sampling[frame - 1] if frame <= len(sampling) else {}
            transition_type = item.get("transition", {}).get("type")
            previous = sampling[frame - 2].get("source_scene_id") if frame > 1 else None
            current = item.get("source_scene_id")
            if transition_type == "hard_cut" or (
                frame > 1 and previous is not None and current != previous
            ):
                transition = "hard_cut"
        result[(segment, frame)] = transition
    return result


def _keys(value: object) -> list[str]:
    return STABLE_KEY.findall(json.dumps(value, ensure_ascii=False))


def _actual(plan: dict) -> tuple[dict, dict, dict]:
    person_map: dict[str, str] = {}
    scene_map: dict[str, str] = {}
    for item in plan.get("person_plans", []):
        values = _keys(item)
        if values and isinstance(item.get("id"), str):
            person_map[item["id"]] = values[0]
    for item in plan.get("scene_plans", []):
        values = _keys(item)
        if values and isinstance(item.get("id"), str):
            scene_map[item["id"]] = values[0]

    actual: dict[tuple[int, int], dict[str, set[str]]] = {}
    for segment in plan.get("segments", []):
        segment_index = segment.get("segment_index")
        if not isinstance(segment_index, int):
            continue
        for person in segment.get("persons", []):
            stable_key = person_map.get(person.get("id"), person.get("id"))
            for frame in person.get("observable_frames", []):
                if isinstance(frame, int):
                    actual.setdefault(
                        (segment_index, frame),
                        {"people": set(), "entities": set(), "scenes": set()},
                    )["people"].add(stable_key)
        for frame in segment.get("frame_constraints", []):
            frame_index = frame.get("frame_index")
            if not isinstance(frame_index, int):
                continue
            target = actual.setdefault(
                (segment_index, frame_index),
                {"people": set(), "entities": set(), "scenes": set()},
            )
            ledger = (frame.get("non_person_entity_ledger") or {}).get("entities", [])
            for entity in ledger:
                values = _keys(entity.get("description", ""))
                target["entities"].add(values[0] if values else entity.get("entity_id"))
    return actual, person_map, scene_map


def _score(
    index: dict,
    plan: dict,
    frames: dict[tuple[int, int], Path],
    transitions: dict[tuple[int, int], str],
) -> tuple[dict, dict]:
    expected = _expected(index, frames)
    actual, person_map, scene_map = _actual(plan)
    for key in frames:
        actual.setdefault(key, {"people": set(), "entities": set(), "scenes": set()})

    person_exact = sum(actual[key]["people"] == expected[key]["people"] for key in frames)
    entity_exact = sum(actual[key]["entities"] == expected[key]["entities"] for key in frames)
    indexed = set().union(*(set(index.get(kind, {})) for kind in ("people", "entities", "scenes")))
    represented = set(_keys(plan))
    plan_coverage = len(indexed & represented)
    extras = represented - indexed

    entity_descriptions: dict[str, set[str]] = {}
    for segment in plan.get("segments", []):
        for frame in segment.get("frame_constraints", []):
            for entity in ((frame.get("non_person_entity_ledger") or {}).get("entities", [])):
                values = _keys(entity.get("description", ""))
                if values:
                    entity_descriptions.setdefault(values[0], set()).add(
                        str(entity.get("description"))
                    )
    consistent = sum(len(values) == 1 for values in entity_descriptions.values())
    special = {
        key
        for key, transition in transitions.items()
        if transition == "hard_cut"
    }
    special |= {
        (segment, frame + delta)
        for segment, frame in list(special)
        for delta in (-1, 1)
        if (segment, frame + delta) in frames
    }
    direct_special = sum(
        actual[key]["people"] <= expected[key]["people"]
        and actual[key]["entities"] <= expected[key]["entities"]
        for key in special
    )
    scene_keys = set(scene_map.values())
    if not scene_keys and isinstance(plan.get("scenes"), dict):
        scene_keys = set(plan["scenes"])
    frame_keys = {
        (segment.get("segment_index"), frame.get("frame_index"))
        for segment in plan.get("segments", [])
        for frame in segment.get("frame_constraints", [])
        if isinstance(segment.get("segment_index"), int)
        and isinstance(frame.get("frame_index"), int)
    }
    expected_scene_keys = set(index.get("scenes", {}))
    evidence = {
        "person_plan_stable_keys": person_map,
        "scene_plan_stable_keys": scene_map,
        "frame_membership": {
            f"{segment}:{frame}": {
                "expected_people": sorted(expected[(segment, frame)]["people"]),
                "actual_people": sorted(actual[(segment, frame)]["people"]),
                "expected_entities": sorted(expected[(segment, frame)]["entities"]),
                "actual_entities": sorted(actual[(segment, frame)]["entities"]),
                "transition": transitions.get((segment, frame)),
                "source_frame_sha256": hashlib.sha256(
                    frames[(segment, frame)].read_bytes()
                ).hexdigest(),
            }
            for segment, frame in sorted(frames)
        },
        "extra_stable_keys": sorted(extras),
        "hard_cut_scope_frames": [
            {"segment": segment, "frame": frame}
            for segment, frame in sorted(special)
        ],
    }
    scores = {
        "indexed_element_replacement_coverage": _ratio(
            plan_coverage, len(indexed)
        ),
        "stable_key_cross_segment_consistency": _ratio(
            consistent, len(entity_descriptions)
        ),
        "shared_board_tile_mapping_completeness": _ratio(
            len(indexed & represented), len(indexed)
        ),
        "per_frame_visible_element_binding": _ratio(
            person_exact + entity_exact, 2 * len(frames)
        ),
        "no_unindexed_top_level_entities": _ratio(
            len(indexed), len(indexed) + len(extras)
        ),
        "scene_and_frame_key_closure": _ratio(
            len(expected_scene_keys & scene_keys) + len(set(frames) & frame_keys),
            len(expected_scene_keys | scene_keys)
            + len(set(frames) | frame_keys),
        ),
        "hard_cut_direct_evidence_scope": _ratio(direct_special, len(special)),
    }
    return scores, evidence


def evaluate(
    source_root: Path,
    element_index_path: Path,
    output_path: Path,
    source_sha256: str | None = None,
    request_path: Path | None = None,
    visual_path: Path | None = None,
) -> dict:
    index = _load(element_index_path)
    plan = _load(output_path)
    frames = _frames(source_root)
    request = _load(request_path) if request_path is not None else None
    transitions = _transitions(source_root, frames, request)
    scores, evidence = _score(index, plan, frames, transitions)
    visual = _load(visual_path) if visual_path is not None else {}
    visual_scores = visual.get("scores", {}) if isinstance(visual, dict) else {}
    for axis in VISUAL_AXES:
        value = visual_scores.get(axis)
        if value is not None and (
            not isinstance(value, (int, float)) or not 0 <= value <= 100
        ):
            raise ValueError(f"visual score must be within 0..100: {axis}")
    all_scores = {**scores, **{axis: visual_scores.get(axis) for axis in VISUAL_AXES}}
    numeric = [float(value) for value in all_scores.values() if isinstance(value, (int, float))]
    return {
        "source": {
            "root": str(source_root),
            "source_mp4_sha256": source_sha256,
            "element_index_sha256": _sha256(element_index_path),
            "frame_count": len(frames),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "visual_review": (
            {"path": str(visual_path), "sha256": _sha256(visual_path)}
            if visual_path is not None
            else None
        ),
        "scores": all_scores,
        "descriptive_mean": round(sum(numeric) / len(numeric), 2) if numeric else None,
        "structural_evidence": evidence,
        "visual_evidence": visual.get("evidence", {}) if isinstance(visual, dict) else {},
        "method": "source-frame evidence; visual scores are external review inputs, not gates",
        "decision": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--element-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.source_root,
        args.element_index,
        args.output,
        source_sha256=args.source_sha256,
        request_path=args.request,
        visual_path=args.visual_review,
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
