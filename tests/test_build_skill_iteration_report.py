from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_skill_iteration_report import ReportBridgeError, build_report
from skill_iteration_score import FUSION_FACT_KEYS, IMAGE_FACT_KEYS, WEIGHTS


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8"); return path


def _files(tmp_path: Path, skill: str) -> dict[str, Path]:
    files = {name: tmp_path / f"{name}.json" for name in ("dataset", "runner", "evaluator", "config")}
    for path in files.values(): path.write_text("{}", encoding="utf-8")
    files["skill"] = tmp_path / "SKILL.md"; files["skill"].write_text("skill", encoding="utf-8")
    files["prompt"] = tmp_path / "prompt.md"; files["prompt"].write_text("prompt", encoding="utf-8")
    files["oracle"] = _json(tmp_path / "oracle.json", {"schema":"duet.skill-iteration-oracle","version":1,"cases":[{"case_id":"case","people_keys":["person-01"],"people_demographic_style_evaluable_keys":[],"people_face_evaluable_keys":[],"people_clothing_evaluable_keys":[],"non_person_candidate_keys":["entity-01","scene-01"],"user_replacement_expected_key":"entity-01","fusion_expected_replacement_keys":["person-01","entity-01"]}]})
    return files


def _review(skill: str, sha: str, *, dynamic: dict | None = None) -> dict:
    evidence = {"artifact_sha256":sha,"json_pointer":"/evidence","stable_key":None,"segment_index":None,"frame_order":None}
    keys = (IMAGE_FACT_KEYS - {"people_keys", "people_demographic_style_evaluable_keys", "people_face_evaluable_keys", "people_clothing_evaluable_keys", "non_person_candidate_keys", "user_replacement_required", "user_replacement_key"}) if skill == "image-postprocess" else (FUSION_FACT_KEYS - {"expected_visual_count", "expected_replacement_keys"})
    facts = {key: ([] if key not in {"camera_light_lens_changed", "actual_visual_count", "hard_cut_projection_count"} else False if key == "camera_light_lens_changed" else 0) for key in keys}
    facts.update(dynamic or {})
    return {"schema_valid":True,"facts":facts,"ratings":{axis:{"score":4,"evidence":[evidence]} for axis in WEIGHTS[skill]}}


def _build(tmp_path: Path, skill: str = "image-postprocess", **kwargs):
    files = _files(tmp_path, skill); input_path = tmp_path / "input.json"; input_path.write_text("{}")
    names = ["global_plan","segment_frames","compiled_plan","compiled_prompts"] if skill == "image-postprocess" else ["multimodal_input","h3_prompt_plan"]
    artifacts=[]
    for name in names:
        value = (
            {
                "evidence": name,
                "segments": [{
                    "new_keyframes": [
                        {"transition": {"type": "start"}},
                        {"transition": {"type": "hard_cut"}},
                    ],
                }],
            }
            if name == "multimodal_input"
            else {"evidence": name}
        )
        path=tmp_path/f"{name}.json"; _json(path, value); artifacts.append({"kind":name,"artifact_id":name,"path":str(path)})
    sha=hashlib.sha256((tmp_path/f"{names[0]}.json").read_bytes()).hexdigest()
    review=tmp_path/"review.json"; _json(review, _review(skill, sha, **kwargs))
    experiment=tmp_path/"experiment.json"; _json(experiment, {"model":"blind","runs":[{"case_id":"case","repetition":i,"input_path":str(input_path),"artifacts":artifacts,"review_path":str(review)} for i in (1,2,3)]})
    return build_report(skill=skill, experiment_path=experiment, oracle_path=files["oracle"], skill_path=files["skill"], dataset_manifest_path=files["dataset"], runner_path=files["runner"], evaluator_path=files["evaluator"], review_prompt_path=files["prompt"], model_config_path=files["config"])


def test_image_bridge_hashes_bytes_and_injects_oracle_facts(tmp_path: Path):
    report=_build(tmp_path, dynamic={"replaced_people_keys":["person-01"],"replaced_non_person_keys":["entity-01","scene-01"],"user_prompt_binding_keys":["entity-01"],"user_reference_binding_keys":["entity-01"]})
    run=report["runs"][0]
    assert {x["kind"] for x in run["artifacts"]} == {"global_plan","segment_frames","compiled_plan","compiled_prompts"}
    assert run["facts"]["people_keys"] == ["person-01"]
    assert run["facts"]["user_replacement_key"] == "entity-01"


def test_bridge_rejects_reviewer_claiming_oracle_owned_fact(tmp_path: Path):
    with pytest.raises(ReportBridgeError, match="all dynamic"):
        _build(tmp_path, dynamic={"people_keys":[]})


def test_bridge_rejects_missing_dynamic_fact_and_nonexistent_evidence_pointer(tmp_path: Path):
    files = _files(tmp_path, "image-postprocess")
    input_path = tmp_path / "input.json"; input_path.write_text("{}")
    artifact = _json(tmp_path / "global.json", {"evidence": True})
    artifacts = [{"kind": kind, "artifact_id": kind, "path": str(artifact)} for kind in ("global_plan", "segment_frames", "compiled_plan", "compiled_prompts")]
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    review = _review("image-postprocess", sha)
    review["facts"].pop("replaced_people_keys")
    review_path = _json(tmp_path / "review.json", review)
    experiment = _json(tmp_path / "experiment.json", {"model":"blind", "runs":[{"case_id":"case", "repetition": i, "input_path":str(input_path), "artifacts":artifacts, "review_path":str(review_path)} for i in (1,2,3)]})
    kwargs = dict(skill="image-postprocess", experiment_path=experiment, oracle_path=files["oracle"], skill_path=files["skill"], dataset_manifest_path=files["dataset"], runner_path=files["runner"], evaluator_path=files["evaluator"], review_prompt_path=files["prompt"], model_config_path=files["config"])
    with pytest.raises(ReportBridgeError, match="all dynamic"):
        build_report(**kwargs)
    review = _review("image-postprocess", sha); review["ratings"][next(iter(WEIGHTS["image-postprocess"]))]["evidence"][0]["json_pointer"] = "/missing"
    _json(review_path, review)
    with pytest.raises(ReportBridgeError, match="does not identify"):
        build_report(**kwargs)


def test_fusion_registers_input_and_counts_hard_cut_intervals(tmp_path: Path):
    report=_build(tmp_path, "video-prompt-fusion", dynamic={"actual_visual_count":2})
    run=report["runs"][0]
    assert {x["kind"] for x in run["artifacts"]} == {"multimodal_input","h3_prompt_plan"}
    assert run["facts"]["expected_visual_count"] == 2


def test_bridge_rejects_non_oracle_case(tmp_path: Path):
    files=_files(tmp_path, "image-postprocess")
    experiment=_json(tmp_path/"experiment.json", {"model":"x","runs":[]})
    with pytest.raises(ReportBridgeError, match="non-empty"):
        build_report(skill="image-postprocess", experiment_path=experiment, oracle_path=files["oracle"], skill_path=files["skill"], dataset_manifest_path=files["dataset"], runner_path=files["runner"], evaluator_path=files["evaluator"], review_prompt_path=files["prompt"], model_config_path=files["config"])
