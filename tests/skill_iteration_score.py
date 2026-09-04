#!/usr/bin/env python3
"""Strict offline scorer for image-postprocess and prompt-fusion iterations.

This module evaluates frozen experiment reports only.  It is deliberately not
imported by the application and never changes pipeline eligibility or status.
Semantic ratings are retained, while deterministic facts can force a related
axis to zero so a reviewer cannot hide a concrete contract violation behind a
high subjective score.
"""

from __future__ import annotations

import argparse
import json
import statistics
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "duet.skill-iteration-evaluation"
REPORT_VERSION = 1
RUNS_PER_CASE = 3
SIMPLIFY_MIN_REDUCTION_PERCENT = 10
SKILLS = ("image-postprocess", "video-prompt-fusion")

IMAGE_WEIGHTS = {
    "schema_and_binding": 10,
    "user_replacement_binding": 10,
    "all_people_replaced": 15,
    "person_face_identity_shift": 10,
    "person_style_and_clothing_similarity": 10,
    "minimum_non_person_replacements": 15,
    "source_target_material_difference": 10,
    "scene_same_kind_different_instance": 5,
    "camera_light_lens_preservation": 5,
    "cross_frame_stable_consistency": 10,
}

FUSION_WEIGHTS = {
    "schema_and_order": 10,
    "new_keyframe_static_authority": 20,
    "old_static_fact_exclusion": 15,
    "replacement_target_propagation": 15,
    "action_camera_rhythm_fidelity": 15,
    "hard_cut_non_projection": 10,
    "relation_preservation": 5,
    "audio_visual_separation": 5,
    "cross_segment_stable_consistency": 5,
}

WEIGHTS = {
    "image-postprocess": IMAGE_WEIGHTS,
    "video-prompt-fusion": FUSION_WEIGHTS,
}

IMAGE_FACT_KEYS = {
    "people_keys",
    "replaced_people_keys",
    "people_demographic_style_evaluable_keys",
    "people_demographic_style_preserved_keys",
    "people_face_evaluable_keys",
    "people_face_changed_keys",
    "people_clothing_evaluable_keys",
    "people_clothing_palette_style_preserved_cut_changed_keys",
    "non_person_candidate_keys",
    "replaced_non_person_keys",
    "scene_replacement_keys",
    "same_kind_different_scene_keys",
    "user_replacement_required",
    "user_replacement_key",
    "user_prompt_binding_keys",
    "user_reference_binding_keys",
    "source_target_noop_keys",
    "camera_light_lens_changed",
    "inconsistent_stable_keys",
}

FUSION_FACT_KEYS = {
    "expected_visual_count",
    "actual_visual_count",
    "new_frame_contradictions",
    "old_static_leaks",
    "expected_replacement_keys",
    "missing_replacement_keys",
    "camera_light_lens_inventions",
    "action_direction_conflicts",
    "hard_cut_projection_count",
    "relation_conflicts",
    "audio_text_leaks",
    "binding_token_leaks",
    "inconsistent_stable_keys",
}

_TOP_KEYS = {
    "schema",
    "version",
    "skill",
    "skill_sha256",
    "skill_bytes",
    "dataset_manifest_sha256",
    "oracle_sha256",
    "runner_sha256",
    "evaluator_sha256",
    "review_prompt_sha256",
    "model_config_sha256",
    "model",
    "runs",
}
_RUN_KEYS = {
    "case_id",
    "repetition",
    "input_sha256",
    "artifacts",
    "schema_valid",
    "facts",
    "ratings",
}
_ARTIFACT_KEYS = {"kind", "artifact_id", "sha256"}
_EVIDENCE_KEYS = {
    "artifact_sha256",
    "json_pointer",
    "stable_key",
    "segment_index",
    "frame_order",
}
_HEX = set("0123456789abcdef")


class ScoreContractError(ValueError):
    """The frozen experiment report is malformed or incomparable."""


def _raw_sha256(path: Path) -> str:
    if not path.is_absolute():
        raise ScoreContractError("all context paths must be absolute")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ScoreContractError(f"cannot read frozen context: {path}") from exc


def _context_cases(manifest: Any, oracle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(oracle, dict) or oracle.get("schema") != "duet.skill-iteration-oracle" or oracle.get("version") != 1:
        raise ScoreContractError("unsupported oracle schema/version")
    cases = oracle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ScoreContractError("oracle.cases must be a non-empty list")
    om: dict[str, Any] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
            raise ScoreContractError("oracle cases must have non-empty case_id")
        if case["case_id"] in om:
            raise ScoreContractError("oracle case_id must be unique")
        if not isinstance(case.get("source_project_id"), str) or not isinstance(case.get("split"), str):
            raise ScoreContractError("oracle cases require source_project_id and split")
        om[case["case_id"]] = case
    mcases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(mcases, list):
        raise ScoreContractError("manifest.cases must be a list")
    mm: dict[str, Any] = {}
    manifest_projects: dict[str, Any] = {}
    for case in mcases:
        if not isinstance(case, dict) or not isinstance(case.get("source_project_id"), str) or case["source_project_id"] in manifest_projects:
            raise ScoreContractError("manifest source_project_id must be unique and non-empty")
        manifest_projects[case["source_project_id"]] = case
    for cid, oracle_case in om.items():
        source_id = oracle_case["source_project_id"]
        if source_id not in manifest_projects:
            raise ScoreContractError(f"manifest missing source_project_id: {source_id}")
        item = manifest_projects[source_id]
        mm[cid] = item
        if item.get("source_project_id") != oracle_case["source_project_id"] or item.get("split") != oracle_case["split"]:
            raise ScoreContractError(f"manifest/oracle mapping mismatch: {cid}")
    return mm, om


def validate_frozen_context(report: Any, manifest_path: Path, oracle_path: Path) -> dict[str, Any]:
    """Validate report against the exact bytes and declarations supplied to the CLI."""
    if not manifest_path.is_absolute() or not oracle_path.is_absolute():
        raise ScoreContractError("all context paths must be absolute")
    manifest = json.loads(manifest_path.read_bytes())
    oracle = json.loads(oracle_path.read_bytes())
    if report["dataset_manifest_sha256"] != _raw_sha256(manifest_path):
        raise ScoreContractError("report dataset_manifest_sha256 does not match manifest bytes")
    if report["oracle_sha256"] != _raw_sha256(oracle_path):
        raise ScoreContractError("report oracle_sha256 does not match oracle bytes")
    mm, om = _context_cases(manifest, oracle)
    report_ids = {run["case_id"] for run in report["runs"]}
    if report_ids != set(om):
        raise ScoreContractError("report case set must equal oracle")
    for run in report["runs"]:
        cid = run["case_id"]
        oc, mc = om[cid], mm[cid]
        facts = run["facts"]
        if report["skill"] == "image-postprocess":
            for field in ("people_keys", "non_person_candidate_keys", "people_demographic_style_evaluable_keys", "people_face_evaluable_keys", "people_clothing_evaluable_keys"):
                if set(facts[field]) != set(oc[field]):
                    raise ScoreContractError(f"{cid}.{field} does not match oracle")
            expected = oc["user_replacement_expected_key"]
            if facts["user_replacement_key"] != expected:
                raise ScoreContractError(f"{cid}.user_replacement_key does not match oracle")
        else:
            if set(facts["expected_replacement_keys"]) != set(oc["fusion_expected_replacement_keys"]):
                raise ScoreContractError(f"{cid}.expected_replacement_keys does not match oracle")
            image = mc.get("image_postprocess", {})
            image_segments = image.get("segments", []) if isinstance(image, dict) else []
            expected_visual_count = sum(
                1 + sum(
                    isinstance(frame, dict)
                    and isinstance(frame.get("transition"), dict)
                    and frame["transition"].get("type") == "hard_cut"
                    for frame in segment.get("keyframes", [])[1:]
                )
                for segment in image_segments
                if isinstance(segment, dict)
                and isinstance(segment.get("keyframes"), list)
                and segment["keyframes"]
            )
            if facts["expected_visual_count"] != expected_visual_count:
                raise ScoreContractError(f"{cid}.expected_visual_count does not match manifest")
        fusion = mc.get("video_prompt_fusion", {})
        blob = fusion.get("input", {}).get("blob_sha256") if isinstance(fusion, dict) and isinstance(fusion.get("input"), dict) else None
        if report["skill"] == "video-prompt-fusion" and run["input_sha256"] != blob:
            raise ScoreContractError(f"{cid}.input_sha256 does not match manifest input blob")
    return report


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ScoreContractError(f"{label} must contain exactly {sorted(expected)}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise ScoreContractError(f"{label} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoreContractError(f"{label} must be a non-negative integer")
    return value


def _stable_keys(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise ScoreContractError(f"{label} must be a unique non-empty string list")
    return value


def _structured_evidence(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ScoreContractError(f"{label} must be a non-empty evidence list")
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        item = _exact_keys(item, _EVIDENCE_KEYS, item_label)
        _sha256(item["artifact_sha256"], f"{item_label}.artifact_sha256")
        if not isinstance(item["json_pointer"], str) or not item["json_pointer"].startswith("/"):
            raise ScoreContractError(f"{item_label}.json_pointer must be an absolute JSON Pointer")
        if item["stable_key"] is not None and (
            not isinstance(item["stable_key"], str) or not item["stable_key"].strip()
        ):
            raise ScoreContractError(f"{item_label}.stable_key must be null or non-empty")
        for field in ("segment_index", "frame_order"):
            number = item[field]
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, int) or number < 1
            ):
                raise ScoreContractError(f"{item_label}.{field} must be null or positive")
    return value


def _validate_ratings(
    ratings: Any, weights: dict[str, int], label: str
) -> dict[str, dict[str, Any]]:
    ratings = _exact_keys(ratings, set(weights), label)
    for axis, rating in ratings.items():
        rating = _exact_keys(rating, {"score", "evidence"}, f"{label}.{axis}")
        score = rating["score"]
        if isinstance(score, bool) or not isinstance(score, int) or score not in range(5):
            raise ScoreContractError(f"{label}.{axis}.score must be an integer 0..4")
        _structured_evidence(rating["evidence"], f"{label}.{axis}.evidence")
    return ratings


def _validate_image_facts(facts: Any, label: str) -> dict[str, Any]:
    facts = _exact_keys(facts, IMAGE_FACT_KEYS, label)
    list_fields = IMAGE_FACT_KEYS - {
        "user_replacement_required",
        "user_replacement_key",
        "camera_light_lens_changed",
    }
    for field in list_fields:
        _stable_keys(facts[field], f"{label}.{field}")
    for field in (
        "user_replacement_required",
        "camera_light_lens_changed",
    ):
        if not isinstance(facts[field], bool):
            raise ScoreContractError(f"{label}.{field} must be boolean")
    user_key = facts["user_replacement_key"]
    if user_key is not None and (not isinstance(user_key, str) or not user_key.strip()):
        raise ScoreContractError(f"{label}.user_replacement_key must be null or non-empty")
    if facts["user_replacement_required"] != (user_key is not None):
        raise ScoreContractError(
            f"{label}.user_replacement_required must match user_replacement_key presence"
        )

    people = set(facts["people_keys"])
    non_people = set(facts["non_person_candidate_keys"])
    if people & non_people:
        raise ScoreContractError(f"{label}: people and non-person candidates must be disjoint")
    subset_fields = {
        "replaced_people_keys": people,
        "people_demographic_style_evaluable_keys": people,
        "people_demographic_style_preserved_keys": people,
        "people_face_evaluable_keys": people,
        "people_face_changed_keys": people,
        "people_clothing_evaluable_keys": people,
        "people_clothing_palette_style_preserved_cut_changed_keys": people,
        "replaced_non_person_keys": non_people,
        "scene_replacement_keys": non_people,
        "same_kind_different_scene_keys": set(facts["scene_replacement_keys"]),
    }
    for field, allowed in subset_fields.items():
        if not set(facts[field]) <= allowed:
            raise ScoreContractError(f"{label}.{field} contains an unknown stable key")
    known = people | non_people
    for field in ("source_target_noop_keys", "inconsistent_stable_keys"):
        if not set(facts[field]) <= known:
            raise ScoreContractError(f"{label}.{field} contains an unknown stable key")
    if user_key is not None and user_key not in known:
        raise ScoreContractError(f"{label}.user_replacement_key is not indexed")
    for field in ("user_prompt_binding_keys", "user_reference_binding_keys"):
        if not set(facts[field]) <= known:
            raise ScoreContractError(f"{label}.{field} contains an unknown stable key")
        if user_key is None and facts[field]:
            raise ScoreContractError(f"{label}.{field} must be empty without a user target")
    return facts


def _validate_fusion_facts(facts: Any, label: str) -> dict[str, Any]:
    facts = _exact_keys(facts, FUSION_FACT_KEYS, label)
    for field in ("expected_visual_count", "actual_visual_count", "hard_cut_projection_count"):
        _non_negative_int(facts[field], f"{label}.{field}")
    for field in FUSION_FACT_KEYS - {
        "expected_visual_count",
        "actual_visual_count",
        "hard_cut_projection_count",
    }:
        _stable_keys(facts[field], f"{label}.{field}")
    if not set(facts["missing_replacement_keys"]) <= set(facts["expected_replacement_keys"]):
        raise ScoreContractError(f"{label}.missing_replacement_keys contains an unexpected key")
    return facts


def validate_report(report: Any) -> dict[str, Any]:
    report = _exact_keys(report, _TOP_KEYS, "report")
    if report["schema"] != REPORT_SCHEMA or report["version"] != REPORT_VERSION:
        raise ScoreContractError("unsupported report schema/version")
    skill = report["skill"]
    if skill not in SKILLS:
        raise ScoreContractError(f"unsupported skill: {skill}")
    _sha256(report["skill_sha256"], "report.skill_sha256")
    for field in (
        "dataset_manifest_sha256",
        "oracle_sha256",
        "runner_sha256",
        "evaluator_sha256",
        "review_prompt_sha256",
        "model_config_sha256",
    ):
        _sha256(report[field], f"report.{field}")
    if not isinstance(report["model"], str) or not report["model"].strip():
        raise ScoreContractError("report.model must be non-empty")
    if isinstance(report["skill_bytes"], bool) or not isinstance(report["skill_bytes"], int) or report["skill_bytes"] <= 0:
        raise ScoreContractError("report.skill_bytes must be a positive integer")
    runs = report["runs"]
    if not isinstance(runs, list) or not runs:
        raise ScoreContractError("report.runs must be a non-empty list")

    seen: set[tuple[str, int]] = set()
    repetitions: dict[str, set[int]] = defaultdict(set)
    for index, run in enumerate(runs):
        label = f"report.runs[{index}]"
        run = _exact_keys(run, _RUN_KEYS, label)
        case_id = run["case_id"]
        repetition = run["repetition"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ScoreContractError(f"{label}.case_id must be non-empty")
        if isinstance(repetition, bool) or repetition not in range(1, RUNS_PER_CASE + 1):
            raise ScoreContractError(f"{label}.repetition must be 1..{RUNS_PER_CASE}")
        identity = (case_id, repetition)
        if identity in seen:
            raise ScoreContractError(f"duplicate run identity: {identity}")
        seen.add(identity)
        repetitions[case_id].add(repetition)
        _sha256(run["input_sha256"], f"{label}.input_sha256")
        artifacts = run["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ScoreContractError(f"{label}.artifacts must be non-empty")
        artifact_ids: set[tuple[str, str]] = set()
        artifact_hashes: set[str] = set()
        kinds: list[str] = []
        for artifact_index, artifact in enumerate(artifacts):
            artifact_label = f"{label}.artifacts[{artifact_index}]"
            artifact = _exact_keys(artifact, _ARTIFACT_KEYS, artifact_label)
            kind = artifact["kind"]
            artifact_id = artifact["artifact_id"]
            if not isinstance(kind, str) or not kind.strip():
                raise ScoreContractError(f"{artifact_label}.kind must be non-empty")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                raise ScoreContractError(f"{artifact_label}.artifact_id must be non-empty")
            identity_key = (kind, artifact_id)
            if identity_key in artifact_ids:
                raise ScoreContractError(f"{artifact_label} duplicates an artifact identity")
            artifact_ids.add(identity_key)
            artifact_hashes.add(_sha256(artifact["sha256"], f"{artifact_label}.sha256"))
            kinds.append(kind)
        required_kinds = (
            {"global_plan", "segment_frames", "compiled_plan", "compiled_prompts"}
            if skill == "image-postprocess"
            else {"h3_prompt_plan"}
        )
        if not required_kinds <= set(kinds):
            raise ScoreContractError(f"{label}.artifacts is missing {sorted(required_kinds - set(kinds))}")
        if not isinstance(run["schema_valid"], bool):
            raise ScoreContractError(f"{label}.schema_valid must be boolean")
        if skill == "image-postprocess":
            _validate_image_facts(run["facts"], f"{label}.facts")
        else:
            _validate_fusion_facts(run["facts"], f"{label}.facts")
        ratings = _validate_ratings(run["ratings"], WEIGHTS[skill], f"{label}.ratings")
        for axis, rating in ratings.items():
            for evidence in rating["evidence"]:
                if evidence["artifact_sha256"] not in artifact_hashes:
                    raise ScoreContractError(
                        f"{label}.ratings.{axis}.evidence references an unfrozen artifact"
                    )
    expected_repetitions = set(range(1, RUNS_PER_CASE + 1))
    incomplete = sorted(case for case, values in repetitions.items() if values != expected_repetitions)
    if incomplete:
        raise ScoreContractError(
            f"each case must have exactly {RUNS_PER_CASE} repetitions: {incomplete}"
        )
    return report


def _image_violations(facts: dict[str, Any]) -> tuple[list[str], list[str], set[str]]:
    people = set(facts["people_keys"])
    non_people = set(facts["non_person_candidate_keys"])
    required_non_people = min(2, len(non_people))
    violations: list[str] = []
    zero_axes: set[str] = set()

    expected_user_binding = [facts["user_replacement_key"]] if facts["user_replacement_required"] else []
    if (
        facts["user_prompt_binding_keys"] != expected_user_binding
        or facts["user_reference_binding_keys"] != expected_user_binding
    ):
        violations.append("user_replacement_not_bound")
        zero_axes.add("user_replacement_binding")
    person_checks = (
        ("replaced_people_keys", "people_not_all_replaced", "all_people_replaced"),
    )
    for field, code, axis in person_checks:
        if set(facts[field]) != people:
            violations.append(code)
            zero_axes.add(axis)
    evaluable_checks = (
        (
            "people_demographic_style_evaluable_keys",
            "people_demographic_style_preserved_keys",
            "people_demographic_or_style_changed",
            "person_style_and_clothing_similarity",
        ),
        (
            "people_face_evaluable_keys",
            "people_face_changed_keys",
            "people_face_not_changed",
            "person_face_identity_shift",
        ),
        (
            "people_clothing_evaluable_keys",
            "people_clothing_palette_style_preserved_cut_changed_keys",
            "people_clothing_rule_not_met",
            "person_style_and_clothing_similarity",
        ),
    )
    for evaluable_field, actual_field, code, axis in evaluable_checks:
        if set(facts[actual_field]) != set(facts[evaluable_field]):
            violations.append(code)
            zero_axes.add(axis)
    if len(facts["replaced_non_person_keys"]) < required_non_people:
        violations.append("fewer_than_required_non_person_replacements")
        zero_axes.add("minimum_non_person_replacements")
    elif len(non_people) > 1 and facts["user_replacement_key"] in non_people and not (
        set(facts["replaced_non_person_keys"]) - {facts["user_replacement_key"]}
    ):
        violations.append("no_autonomous_non_person_replacement")
        zero_axes.add("minimum_non_person_replacements")
    if facts["source_target_noop_keys"]:
        violations.append("source_target_noop")
        zero_axes.add("source_target_material_difference")
    if set(facts["same_kind_different_scene_keys"]) != set(facts["scene_replacement_keys"]):
        violations.append("scene_not_same_kind_different_instance")
        zero_axes.add("scene_same_kind_different_instance")
    if facts["camera_light_lens_changed"]:
        violations.append("camera_light_or_lens_changed")
        zero_axes.add("camera_light_lens_preservation")
    if facts["inconsistent_stable_keys"]:
        violations.append("cross_frame_stable_key_inconsistent")
        zero_axes.add("cross_frame_stable_consistency")
    return violations, [], zero_axes


def _fusion_violations(facts: dict[str, Any]) -> tuple[list[str], list[str], set[str]]:
    critical_checks = (
        (
            facts["expected_visual_count"] != facts["actual_visual_count"],
            "visual_count_mismatch",
            "schema_and_order",
        ),
        (bool(facts["new_frame_contradictions"]), "new_frame_contradiction", "new_keyframe_static_authority"),
        (bool(facts["old_static_leaks"]), "old_static_fact_leak", "old_static_fact_exclusion"),
        (bool(facts["missing_replacement_keys"]), "replacement_target_missing", "replacement_target_propagation"),
        (
            bool(facts["camera_light_lens_inventions"] or facts["action_direction_conflicts"]),
            "action_camera_or_rhythm_conflict",
            "action_camera_rhythm_fidelity",
        ),
        (facts["hard_cut_projection_count"] > 0, "hard_cut_projection", "hard_cut_non_projection"),
        (bool(facts["relation_conflicts"]), "relation_conflict", "relation_preservation"),
        (
            bool(facts["inconsistent_stable_keys"]),
            "cross_segment_stable_key_inconsistent",
            "cross_segment_stable_consistency",
        ),
    )
    quality_checks = (
        (bool(facts["audio_text_leaks"]), "audio_text_leak", "audio_visual_separation"),
        (bool(facts["binding_token_leaks"]), "binding_token_leak", "schema_and_order"),
    )
    critical = [code for failed, code, _axis in critical_checks if failed]
    quality = [code for failed, code, _axis in quality_checks if failed]
    zero_axes = {
        axis
        for failed, _code, axis in (*critical_checks, *quality_checks)
        if failed
    }
    return critical, quality, zero_axes


def score_run(skill: str, run: dict[str, Any]) -> dict[str, Any]:
    weights = WEIGHTS[skill]
    critical_failures, quality_failures, zero_axes = (
        _image_violations(run["facts"])
        if skill == "image-postprocess"
        else _fusion_violations(run["facts"])
    )
    if not run["schema_valid"]:
        critical_failures.insert(0, "schema_invalid")
        zero_axes.update(weights)
    violations = critical_failures + quality_failures
    axis_scores = {
        axis: 0 if axis in zero_axes else run["ratings"][axis]["score"]
        for axis in weights
    }
    raw_total = round(
        sum(axis_scores[axis] * weight / 4 for axis, weight in weights.items()),
        2,
    )
    cap = 59 if critical_failures else (79 if quality_failures else 100)
    total = min(raw_total, cap)
    return {
        "case_id": run["case_id"],
        "repetition": run["repetition"],
        "input_sha256": run["input_sha256"],
        "artifacts": run["artifacts"],
        "schema_valid": run["schema_valid"],
        "policy_met": run["schema_valid"] and not violations,
        "violations": violations,
        "critical_failures": critical_failures,
        "quality_failures": quality_failures,
        "axis_scores": axis_scores,
        "raw_total": raw_total,
        "cap": cap,
        "total": total,
    }


def summarize(report: Any) -> dict[str, Any]:
    report = validate_report(report)
    skill = report["skill"]
    scored = [score_run(skill, run) for run in report["runs"]]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in scored:
        by_case[run["case_id"]].append(run)
    case_summaries = {}
    for case_id, runs in sorted(by_case.items()):
        values = [run["total"] for run in runs]
        case_axis_medians = {
            axis: round(float(statistics.median(run["axis_scores"][axis] for run in runs)), 2)
            for axis in WEIGHTS[skill]
        }
        case_summaries[case_id] = {
            "median": round(float(statistics.median(values)), 2),
            "worst": round(min(values), 2),
            "schema_valid_rate": round(sum(run["schema_valid"] for run in runs) / len(runs), 4),
            "policy_met_rate": round(sum(run["policy_met"] for run in runs) / len(runs), 4),
            "axis_medians": case_axis_medians,
            "axis_worsts": {
                axis: min(run["axis_scores"][axis] for run in runs)
                for axis in WEIGHTS[skill]
            },
        }
    axis_medians = {
        axis: round(
            sum(case["axis_medians"][axis] for case in case_summaries.values())
            / len(case_summaries),
            2,
        )
        for axis in WEIGHTS[skill]
    }
    return {
        "schema": "duet.skill-iteration-summary",
        "version": 1,
        "skill": skill,
        "skill_sha256": report["skill_sha256"],
        "skill_bytes": report["skill_bytes"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "oracle_sha256": report["oracle_sha256"],
        "runner_sha256": report["runner_sha256"],
        "evaluator_sha256": report["evaluator_sha256"],
        "review_prompt_sha256": report["review_prompt_sha256"],
        "model_config_sha256": report["model_config_sha256"],
        "model": report["model"],
        "case_count": len(by_case),
        "run_count": len(scored),
        "schema_valid_rate": round(sum(run["schema_valid"] for run in scored) / len(scored), 4),
        "policy_met_rate": round(sum(run["policy_met"] for run in scored) / len(scored), 4),
        "median_total": round(
            sum(case["median"] for case in case_summaries.values()) / len(case_summaries),
            2,
        ),
        "worst_total": round(min(case["worst"] for case in case_summaries.values()), 2),
        "axis_medians": axis_medians,
        "axis_worsts": {
            axis: min(case["axis_worsts"][axis] for case in case_summaries.values())
            for axis in WEIGHTS[skill]
        },
        "cases": case_summaries,
        "runs": scored,
        "runtime_gate": False,
    }


def _run_identity(report: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (run["case_id"], run["repetition"]): run["input_sha256"]
        for run in report["runs"]
    }


def compare(baseline: Any, candidate: Any, *, phase: str) -> dict[str, Any]:
    baseline = validate_report(baseline)
    candidate = validate_report(candidate)
    if phase not in {"simplify", "semantic"}:
        raise ScoreContractError("phase must be simplify or semantic")
    if baseline["skill"] != candidate["skill"]:
        raise ScoreContractError("reports target different skills")
    frozen_fields = (
        "dataset_manifest_sha256",
        "oracle_sha256",
        "runner_sha256",
        "evaluator_sha256",
        "review_prompt_sha256",
        "model_config_sha256",
        "model",
    )
    for field in frozen_fields:
        if baseline[field] != candidate[field]:
            raise ScoreContractError(f"reports use different {field}")
    if _run_identity(baseline) != _run_identity(candidate):
        raise ScoreContractError("reports do not contain the same frozen run inputs")

    before = summarize(baseline)
    after = summarize(candidate)
    reasons: list[str] = []
    if after["schema_valid_rate"] < before["schema_valid_rate"]:
        reasons.append("schema_valid_rate_regressed")
    if after["policy_met_rate"] < before["policy_met_rate"]:
        reasons.append("policy_met_rate_regressed")
    if after["median_total"] < before["median_total"]:
        reasons.append("median_total_regressed")
    if after["worst_total"] < before["worst_total"]:
        reasons.append("worst_total_regressed")
    for axis in WEIGHTS[baseline["skill"]]:
        if after["axis_medians"][axis] < before["axis_medians"][axis]:
            reasons.append(f"axis_regressed:{axis}")
        if after["axis_worsts"][axis] < before["axis_worsts"][axis]:
            reasons.append(f"axis_worst_regressed:{axis}")
    for case_id in before["cases"]:
        for metric in ("schema_valid_rate", "policy_met_rate", "median", "worst"):
            if after["cases"][case_id][metric] < before["cases"][case_id][metric]:
                reasons.append(f"case_regressed:{case_id}:{metric}")
    if phase == "simplify":
        reduction = baseline["skill_bytes"] - candidate["skill_bytes"]
        if reduction <= 0:
            reasons.append("skill_not_smaller")
        elif reduction * 100 < baseline["skill_bytes"] * SIMPLIFY_MIN_REDUCTION_PERCENT:
            reasons.append("skill_reduction_below_10_percent")
    if phase == "semantic":
        before_met_cases = sum(value["policy_met_rate"] == 1.0 for value in before["cases"].values())
        after_met_cases = sum(value["policy_met_rate"] == 1.0 for value in after["cases"].values())
        if before_met_cases < before["case_count"]:
            improved = after_met_cases > before_met_cases
        else:
            improved = after["median_total"] >= before["median_total"] + 3
        if not improved:
            reasons.append("no_semantic_improvement")
    return {
        "schema": "duet.skill-iteration-comparison",
        "version": 1,
        "skill": baseline["skill"],
        "phase": phase,
        "baseline_skill_sha256": baseline["skill_sha256"],
        "candidate_skill_sha256": candidate["skill_sha256"],
        "baseline_skill_bytes": baseline["skill_bytes"],
        "candidate_skill_bytes": candidate["skill_bytes"],
        "baseline": before,
        "candidate": after,
        "recommended_for_next_offline_round": not reasons,
        "reasons": reasons,
        "runtime_gate": False,
    }


def _load(path: Path) -> Any:
    if not path.is_absolute():
        raise ScoreContractError("all paths must be absolute")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    if not path.is_absolute():
        raise ScoreContractError("all paths must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--report", required=True, type=Path)
    score_parser.add_argument("--dataset-manifest", required=True, type=Path)
    score_parser.add_argument("--oracle", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--phase", choices=("simplify", "semantic"), required=True)
    compare_parser.add_argument("--output", required=True, type=Path)
    compare_parser.add_argument("--dataset-manifest", required=True, type=Path)
    compare_parser.add_argument("--oracle", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "score":
        report = _load(args.report)
        validate_frozen_context(validate_report(report), args.dataset_manifest, args.oracle)
        result = summarize(report)
    else:
        baseline, candidate = _load(args.baseline), _load(args.candidate)
        validate_frozen_context(validate_report(baseline), args.dataset_manifest, args.oracle)
        validate_frozen_context(validate_report(candidate), args.dataset_manifest, args.oracle)
        result = compare(baseline, candidate, phase=args.phase)
    _write(args.output, result)


if __name__ == "__main__":
    main()
