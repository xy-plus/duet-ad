import copy
import hashlib
import json
from pathlib import Path

import pytest

from skill_iteration_score import ScoreContractError, compare, summarize, validate_frozen_context


SHA = hashlib.sha256(b"fixture").hexdigest()
ROOT = Path(__file__).parents[1]


def _ratings(axes, score=4):
    return {
        axis: {
            "score": score,
            "evidence": [
                {
                    "artifact_sha256": SHA,
                    "json_pointer": "/fixture",
                    "stable_key": None,
                    "segment_index": None,
                    "frame_order": None,
                }
            ],
        }
        for axis in axes
    }


def _image_facts():
    return {
        "people_keys": ["person-01", "person-02"],
        "replaced_people_keys": ["person-01", "person-02"],
        "people_demographic_style_evaluable_keys": ["person-01", "person-02"],
        "people_demographic_style_preserved_keys": ["person-01", "person-02"],
        "people_face_evaluable_keys": ["person-01", "person-02"],
        "people_face_changed_keys": ["person-01", "person-02"],
        "people_clothing_evaluable_keys": ["person-01", "person-02"],
        "people_clothing_palette_style_preserved_cut_changed_keys": ["person-01", "person-02"],
        "non_person_candidate_keys": ["entity-01", "scene-01", "entity-02"],
        "replaced_non_person_keys": ["entity-01", "scene-01"],
        "scene_replacement_keys": ["scene-01"],
        "same_kind_different_scene_keys": ["scene-01"],
        "user_replacement_required": True,
        "user_replacement_key": "entity-01",
        "user_prompt_binding_keys": ["entity-01"],
        "user_reference_binding_keys": ["entity-01"],
        "source_target_noop_keys": [],
        "camera_light_lens_changed": False,
        "inconsistent_stable_keys": [],
    }


def _fusion_facts():
    return {
        "expected_visual_count": 9,
        "actual_visual_count": 9,
        "new_frame_contradictions": [],
        "old_static_leaks": [],
        "expected_replacement_keys": ["person-01", "entity-01", "scene-01"],
        "missing_replacement_keys": [],
        "camera_light_lens_inventions": [],
        "action_direction_conflicts": [],
        "hard_cut_projection_count": 0,
        "relation_conflicts": [],
        "audio_text_leaks": [],
        "binding_token_leaks": [],
        "inconsistent_stable_keys": [],
    }


def _report(skill="image-postprocess", *, skill_bytes=1000, score=4):
    from skill_iteration_score import WEIGHTS

    facts = _image_facts() if skill == "image-postprocess" else _fusion_facts()
    runs = []
    for case_id in ("train-person-and-prop", "holdout-scene"):
        for repetition in (1, 2, 3):
            runs.append(
                {
                    "case_id": case_id,
                    "repetition": repetition,
                    "input_sha256": SHA,
                    "artifacts": (
                        [
                            {"kind": "global_plan", "artifact_id": "global", "sha256": SHA},
                            {"kind": "segment_frames", "artifact_id": "segment-1", "sha256": SHA},
                            {"kind": "compiled_plan", "artifact_id": "plan", "sha256": SHA},
                            {"kind": "compiled_prompts", "artifact_id": "prompts", "sha256": SHA},
                        ]
                        if skill == "image-postprocess"
                        else [{"kind": "h3_prompt_plan", "artifact_id": "output", "sha256": SHA}]
                    ),
                    "schema_valid": True,
                    "facts": copy.deepcopy(facts),
                    "ratings": _ratings(WEIGHTS[skill], score),
                }
            )
    return {
        "schema": "duet.skill-iteration-evaluation",
        "version": 1,
        "skill": skill,
        "skill_sha256": hashlib.sha256(f"skill:{skill_bytes}".encode()).hexdigest(),
        "skill_bytes": skill_bytes,
        "dataset_manifest_sha256": SHA,
        "oracle_sha256": SHA,
        "runner_sha256": SHA,
        "evaluator_sha256": SHA,
        "review_prompt_sha256": SHA,
        "model_config_sha256": SHA,
        "model": "deepseek-test",
        "runs": runs,
    }


def test_image_policy_requires_every_person_and_two_non_person_replacements():
    report = _report()
    report["runs"][0]["facts"]["replaced_people_keys"] = ["person-01"]
    report["runs"][0]["facts"]["replaced_non_person_keys"] = ["entity-01"]

    summary = summarize(report)
    run = summary["runs"][0]

    assert run["policy_met"] is False
    assert run["axis_scores"]["all_people_replaced"] == 0
    assert run["axis_scores"]["minimum_non_person_replacements"] == 0
    assert run["cap"] == 59
    assert run["total"] == 59
    assert "people_not_all_replaced" in run["violations"]
    assert "fewer_than_required_non_person_replacements" in run["violations"]


def test_non_person_requirement_uses_all_candidates_when_fewer_than_two():
    report = _report()
    for run in report["runs"]:
        run["facts"]["non_person_candidate_keys"] = ["scene-01"]
        run["facts"]["replaced_non_person_keys"] = ["scene-01"]
        run["facts"]["user_replacement_key"] = "scene-01"
        run["facts"]["user_prompt_binding_keys"] = ["scene-01"]
        run["facts"]["user_reference_binding_keys"] = ["scene-01"]

    summary = summarize(report)

    assert summary["policy_met_rate"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "axis", "violation"),
    [
        ("people_face_changed_keys", ["person-01"], "person_face_identity_shift", "people_face_not_changed"),
        (
            "people_clothing_palette_style_preserved_cut_changed_keys",
            ["person-01"],
            "person_style_and_clothing_similarity",
            "people_clothing_rule_not_met",
        ),
        ("source_target_noop_keys", ["entity-01"], "source_target_material_difference", "source_target_noop"),
        ("camera_light_lens_changed", True, "camera_light_lens_preservation", "camera_light_or_lens_changed"),
        ("inconsistent_stable_keys", ["person-02"], "cross_frame_stable_consistency", "cross_frame_stable_key_inconsistent"),
    ],
)
def test_image_deterministic_fact_caps_related_axis(field, value, axis, violation):
    report = _report()
    report["runs"][0]["facts"][field] = value

    run = summarize(report)["runs"][0]

    assert run["axis_scores"][axis] == 0
    assert violation in run["violations"]


def test_user_reference_binding_is_required_only_when_present():
    report = _report()
    report["runs"][0]["facts"]["user_reference_binding_keys"] = []

    failed = summarize(report)["runs"][0]
    assert failed["axis_scores"]["user_replacement_binding"] == 0

    for run in report["runs"]:
        run["facts"]["user_replacement_required"] = False
        run["facts"]["user_replacement_key"] = None
        run["facts"]["user_prompt_binding_keys"] = []
        run["facts"]["user_reference_binding_keys"] = []
    assert summarize(report)["policy_met_rate"] == 1.0


def test_hidden_face_and_clothing_are_oracle_marked_not_evaluable():
    report = _report()
    for run in report["runs"]:
        run["facts"]["people_face_evaluable_keys"] = ["person-01"]
        run["facts"]["people_face_changed_keys"] = ["person-01"]
        run["facts"]["people_clothing_evaluable_keys"] = []
        run["facts"]["people_clothing_palette_style_preserved_cut_changed_keys"] = []

    assert summarize(report)["policy_met_rate"] == 1.0


def test_fusion_static_leak_and_missing_target_cannot_be_hidden_by_ratings():
    report = _report("video-prompt-fusion")
    report["runs"][0]["facts"]["old_static_leaks"] = ["gold necklace"]
    report["runs"][0]["facts"]["missing_replacement_keys"] = ["entity-01"]

    run = summarize(report)["runs"][0]

    assert run["axis_scores"]["old_static_fact_exclusion"] == 0
    assert run["axis_scores"]["replacement_target_propagation"] == 0
    assert run["policy_met"] is False


def test_fusion_text_leak_uses_quality_cap_not_core_cap():
    report = _report("video-prompt-fusion")
    report["runs"][0]["facts"]["binding_token_leaks"] = ["stable_key=person-01"]

    run = summarize(report)["runs"][0]

    assert run["critical_failures"] == []
    assert run["quality_failures"] == ["binding_token_leak"]
    assert run["cap"] == 79
    assert run["policy_met"] is False


def test_simplification_recommends_only_a_smaller_non_regressing_skill():
    baseline = _report(skill_bytes=1200, score=3)
    candidate = _report(skill_bytes=700, score=3)

    result = compare(baseline, candidate, phase="simplify")

    assert result["recommended_for_next_offline_round"] is True
    assert result["runtime_gate"] is False

    candidate["skill_bytes"] = 1300
    result = compare(baseline, candidate, phase="simplify")
    assert result["recommended_for_next_offline_round"] is False
    assert "skill_not_smaller" in result["reasons"]

    candidate["skill_bytes"] = 1199
    result = compare(baseline, candidate, phase="simplify")
    assert result["recommended_for_next_offline_round"] is False
    assert "skill_reduction_below_10_percent" in result["reasons"]

    candidate["skill_bytes"] = 1080
    assert compare(baseline, candidate, phase="simplify")[
        "recommended_for_next_offline_round"
    ] is True


def test_semantic_round_requires_improvement_without_any_axis_regression():
    baseline = _report(score=2)
    candidate = _report(score=3)

    assert compare(baseline, candidate, phase="semantic")[
        "recommended_for_next_offline_round"
    ] is True

    axis = next(iter(candidate["runs"][0]["ratings"]))
    for run in candidate["runs"]:
        run["ratings"][axis]["score"] = 1
    result = compare(baseline, candidate, phase="semantic")
    assert result["recommended_for_next_offline_round"] is False
    assert f"axis_regressed:{axis}" in result["reasons"]


def test_comparison_requires_identical_frozen_inputs():
    baseline = _report()
    candidate = _report(skill_bytes=900)
    candidate["runs"][0]["input_sha256"] = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(ScoreContractError, match="same frozen run inputs"):
        compare(baseline, candidate, phase="simplify")


def test_contract_rejects_missing_repetition_and_extra_fact():
    report = _report()
    report["runs"].pop()
    with pytest.raises(ScoreContractError, match="exactly 3 repetitions"):
        summarize(report)


def test_schema_invalid_run_scores_zero():
    report = _report()
    report["runs"][0]["schema_valid"] = False

    run = summarize(report)["runs"][0]

    assert run["raw_total"] == 0
    assert run["total"] == 0
    assert run["policy_met"] is False


def test_rating_evidence_must_bind_a_frozen_artifact():
    report = _report()
    first_axis = next(iter(report["runs"][0]["ratings"]))
    report["runs"][0]["ratings"][first_axis]["evidence"][0]["artifact_sha256"] = hashlib.sha256(b"unfrozen").hexdigest()

    with pytest.raises(ScoreContractError, match="unfrozen artifact"):
        summarize(report)

    report = _report()
    report["runs"][0]["facts"]["self_score_comment"] = "excellent"
    with pytest.raises(ScoreContractError, match="must contain exactly"):
        summarize(report)


def test_seed_oracle_freezes_current_cases_and_missing_holdout_coverage():
    oracle = json.loads(
        (ROOT / "tests/skill_iteration_oracle.v1.json").read_text(encoding="utf-8")
    )

    assert oracle["schema"] == "duet.skill-iteration-oracle"
    assert oracle["version"] == 1
    assert {case["source_project_id"] for case in oracle["cases"]} == {
        "03ed892b63bd44ba85d43553f8e7a40e",
        "88da16ebf6604fa19d94f5c8346735d2",
        "6920ff446dad40a7b77695aeba4be6c9",
    }
    for case in oracle["cases"]:
        assert len(case["people_keys"]) == 1
        assert len(case["non_person_candidate_keys"]) >= 2
        assert case["user_replacement_expected_key"] in case["non_person_candidate_keys"]
    assert "multiple fully visible people recurring across segments" in oracle[
        "missing_holdout_coverage"
    ]


def test_frozen_context_accepts_real_oracle_shape(tmp_path):
    oracle_path = ROOT / "tests/skill_iteration_oracle.v1.json"
    oracle = json.loads(oracle_path.read_text())
    manifest = {"cases": [{"source_project_id": c["source_project_id"], "split": c["split"],
                             "video_prompt_fusion": {"input": {"blob_sha256": SHA}, "new_keyframes": [{"x": 1}]}}
                            for c in oracle["cases"]]}
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
    copied_oracle = tmp_path / "oracle.json"; copied_oracle.write_bytes(oracle_path.read_bytes())
    report = _report(); report["dataset_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report["oracle_sha256"] = hashlib.sha256(copied_oracle.read_bytes()).hexdigest()
    for run, case in zip(report["runs"], oracle["cases"] * 3):
        run["case_id"] = case["case_id"]
        run["input_sha256"] = SHA
        run["facts"]["people_keys"] = case["people_keys"]
        run["facts"]["non_person_candidate_keys"] = case["non_person_candidate_keys"]
        run["facts"]["people_demographic_style_evaluable_keys"] = case["people_demographic_style_evaluable_keys"]
        run["facts"]["people_face_evaluable_keys"] = case["people_face_evaluable_keys"]
        run["facts"]["people_clothing_evaluable_keys"] = case["people_clothing_evaluable_keys"]
        run["facts"]["user_replacement_key"] = case["user_replacement_expected_key"]
    validate_frozen_context(report, manifest_path, copied_oracle)


@pytest.mark.parametrize("mutation", ["people", "nonpeople", "count", "input", "manifest_hash", "oracle_hash"])
def test_frozen_context_rejects_drift(tmp_path, mutation):
    oracle_path = ROOT / "tests/skill_iteration_oracle.v1.json"; oracle = json.loads(oracle_path.read_text())
    manifest = {"cases": [{"source_project_id": c["source_project_id"], "split": c["split"],
                             "video_prompt_fusion": {"input": {"blob_sha256": SHA}, "new_keyframes": [{"x": 1}]}}
                            for c in oracle["cases"]]}
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(manifest)); op = tmp_path / "o.json"; op.write_bytes(oracle_path.read_bytes())
    report = _report("video-prompt-fusion"); report["dataset_manifest_sha256"] = hashlib.sha256(mp.read_bytes()).hexdigest(); report["oracle_sha256"] = hashlib.sha256(op.read_bytes()).hexdigest()
    for run, case in zip(report["runs"], oracle["cases"] * 3):
        run["case_id"] = case["case_id"]; run["input_sha256"] = SHA; run["facts"]["expected_replacement_keys"] = case["fusion_expected_replacement_keys"]; run["facts"]["expected_visual_count"] = 1
    if mutation == "people": report["runs"][0]["facts"]["expected_replacement_keys"] = []
    if mutation == "nonpeople": report["runs"][0]["facts"]["expected_replacement_keys"] = report["runs"][0]["facts"]["expected_replacement_keys"][:-1]
    if mutation == "count": report["runs"][0]["facts"]["expected_visual_count"] = 2
    if mutation == "input": report["runs"][0]["input_sha256"] = "0" * 64
    if mutation == "manifest_hash": report["dataset_manifest_sha256"] = "0" * 64
    if mutation == "oracle_hash": report["oracle_sha256"] = "0" * 64
    with pytest.raises(ScoreContractError): validate_frozen_context(report, mp, op)


def test_frozen_context_rejects_relative_context_paths(tmp_path):
    with pytest.raises(ScoreContractError, match="absolute"):
        validate_frozen_context({}, Path("manifest.json"), Path("oracle.json"))
