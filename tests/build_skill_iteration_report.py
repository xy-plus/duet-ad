#!/usr/bin/env python3
"""Build a scorer-v1 report from frozen runs and independent blind reviews.

This is deliberately an offline adapter: it reads bytes and JSON only, never
imports application code or invokes a model.  The experiment manifest is JSON:
``{"model":"...","runs":[{"case_id", "repetition", "input_path",
"artifacts":[{"kind","artifact_id","path"}], "review_path"}]}``.
All paths in it, and every CLI path, must be absolute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from skill_iteration_score import FUSION_FACT_KEYS, IMAGE_FACT_KEYS, RUNS_PER_CASE, WEIGHTS, validate_report


class ReportBridgeError(ValueError):
    pass


STATIC_IMAGE = {"people_keys", "people_demographic_style_evaluable_keys", "people_face_evaluable_keys", "people_clothing_evaluable_keys", "non_person_candidate_keys", "user_replacement_required", "user_replacement_key"}
STATIC_FUSION = {"expected_visual_count", "expected_replacement_keys"}
REQUIRED = {
    "image-postprocess": {"global_plan", "segment_frames", "compiled_plan", "compiled_prompts"},
    "video-prompt-fusion": {"multimodal_input", "h3_prompt_plan"},
}


def _path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ReportBridgeError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ReportBridgeError(f"{label} must be a regular file")
    return path


def _load(path: str | Path, label: str) -> Any:
    file = _path(path, label)
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportBridgeError(f"{label} must be valid JSON") from exc


def _stable_bytes(path: str | Path, label: str) -> bytes:
    """Read one immutable artifact; reject a file that changed while read."""
    file = _path(path, label)
    try:
        before = file.stat()
        data = file.read_bytes()
        after = file.stat()
    except OSError as exc:
        raise ReportBridgeError(f"{label} cannot be read") from exc
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after):
        raise ReportBridgeError(f"{label} changed while being frozen")
    return data


def _sha(path: str | Path, label: str) -> str:
    return hashlib.sha256(_stable_bytes(path, label)).hexdigest()


def _oracle(path: str | Path) -> dict[str, dict[str, Any]]:
    value = _load(path, "oracle")
    if not isinstance(value, dict) or value.get("schema") != "duet.skill-iteration-oracle" or value.get("version") != 1 or not isinstance(value.get("cases"), list):
        raise ReportBridgeError("oracle schema is invalid")
    result = {}
    for item in value["cases"]:
        case_id = item.get("case_id") if isinstance(item, dict) else None
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise ReportBridgeError("oracle case_id is invalid")
        result[case_id] = item
    return result


def _artifacts(spec: Any, skill: str) -> tuple[list[dict[str, str]], dict[str, Path], dict[str, bytes]]:
    if not isinstance(spec, list) or not spec:
        raise ReportBridgeError("run artifacts are required")
    output, paths, contents, identities = [], {}, {}, set()
    for item in spec:
        if not isinstance(item, dict) or set(item) != {"kind", "artifact_id", "path"}:
            raise ReportBridgeError("artifact must contain exactly kind, artifact_id, path")
        kind, artifact_id = item["kind"], item["artifact_id"]
        if not isinstance(kind, str) or not kind or not isinstance(artifact_id, str) or not artifact_id or (kind, artifact_id) in identities:
            raise ReportBridgeError("artifact identity is invalid")
        file = _path(item["path"], f"artifact {kind}/{artifact_id}")
        identities.add((kind, artifact_id)); paths.setdefault(kind, file)
        data = _stable_bytes(file, f"artifact {kind}/{artifact_id}")
        digest = hashlib.sha256(data).hexdigest()
        contents[digest] = data
        output.append({"kind": kind, "artifact_id": artifact_id, "sha256": digest})
    missing = REQUIRED[skill] - set(paths)
    if missing:
        raise ReportBridgeError(f"run is missing frozen artifacts: {sorted(missing)}")
    return output, paths, contents


def _pointer(value: Any, pointer: str) -> None:
    """Resolve the RFC6901 pointer enough to prove the cited node exists."""
    if pointer == "":
        return
    if not pointer.startswith("/"):
        raise ReportBridgeError("review evidence pointer is not absolute")
    current = value
    for raw in pointer[1:].split("/"):
        token = ""; index = 0
        while index < len(raw):
            if raw[index] != "~":
                token += raw[index]; index += 1; continue
            if index + 1 >= len(raw) or raw[index + 1] not in "01":
                raise ReportBridgeError("review evidence pointer has invalid RFC6901 escape")
            token += "/" if raw[index + 1] == "1" else "~"; index += 2
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and (token == "0" or not token.startswith("0")) and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ReportBridgeError("review evidence pointer does not identify an artifact node")


def _review(value: Any, axes: set[str], artifact_contents: dict[str, bytes], dynamic_keys: set[str]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"schema_valid", "facts", "ratings"} or not isinstance(value["schema_valid"], bool) or not isinstance(value["facts"], dict) or not isinstance(value["ratings"], dict):
        raise ReportBridgeError("review must contain exactly schema_valid, facts, ratings")
    if set(value["facts"]) != dynamic_keys:
        raise ReportBridgeError("review facts must contain exactly all dynamic scorer facts")
    if set(value["ratings"]) != axes:
        raise ReportBridgeError("review ratings axes do not match scorer")
    for axis, rating in value["ratings"].items():
        if not isinstance(rating, dict) or set(rating) != {"score", "evidence"} or not isinstance(rating["evidence"], list) or not rating["evidence"]:
            raise ReportBridgeError(f"review rating {axis} is invalid")
        for evidence in rating["evidence"]:
            digest = evidence.get("artifact_sha256") if isinstance(evidence, dict) else None
            if digest not in artifact_contents or not isinstance(evidence.get("json_pointer"), str):
                raise ReportBridgeError(f"review rating {axis} has unbound evidence")
            try:
                artifact_json = json.loads(artifact_contents[digest].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReportBridgeError(f"review rating {axis} cites a non-JSON artifact") from exc
            _pointer(artifact_json, evidence["json_pointer"])
    return value["schema_valid"], value["facts"], value["ratings"]


def _fusion_count(path: Path) -> int:
    value = _load(path, "frozen multimodal_input")
    segments = value.get("segments") if isinstance(value, dict) else None
    if not isinstance(segments, list):
        raise ReportBridgeError("frozen multimodal_input has no segments")
    counts = [
        1 + sum(
            isinstance(frame, dict)
            and isinstance(frame.get("transition"), dict)
            and frame["transition"].get("type") == "hard_cut"
            for frame in item["new_keyframes"][1:]
        )
        for item in segments
        if isinstance(item, dict)
        and isinstance(item.get("new_keyframes"), list)
        and item["new_keyframes"]
    ]
    if len(counts) != len(segments):
        raise ReportBridgeError("frozen multimodal_input new_keyframes is invalid")
    return sum(counts)


def _facts(skill: str, oracle: dict[str, Any], dynamic: dict[str, Any], fusion_input: Path | None) -> dict[str, Any]:
    allowed = IMAGE_FACT_KEYS if skill == "image-postprocess" else FUSION_FACT_KEYS
    static = STATIC_IMAGE if skill == "image-postprocess" else STATIC_FUSION
    if skill == "image-postprocess":
        facts = dict(dynamic)
        facts.update({key: oracle[key] for key in ("people_keys", "people_demographic_style_evaluable_keys", "people_face_evaluable_keys", "people_clothing_evaluable_keys", "non_person_candidate_keys")})
        facts["user_replacement_required"] = oracle["user_replacement_expected_key"] is not None
        facts["user_replacement_key"] = oracle["user_replacement_expected_key"]
        return facts
    assert fusion_input is not None
    facts = dict(dynamic)
    facts["expected_visual_count"] = _fusion_count(fusion_input)
    facts["expected_replacement_keys"] = oracle["fusion_expected_replacement_keys"]
    return facts


def build_report(*, skill: str, experiment_path: str | Path, oracle_path: str | Path, skill_path: str | Path, dataset_manifest_path: str | Path, runner_path: str | Path, evaluator_path: str | Path, review_prompt_path: str | Path, model_config_path: str | Path) -> dict[str, Any]:
    if skill not in WEIGHTS:
        raise ReportBridgeError("unsupported skill")
    experiment = _load(experiment_path, "experiment")
    if not isinstance(experiment, dict) or set(experiment) != {"model", "runs"} or not isinstance(experiment["model"], str) or not experiment["model"].strip() or not isinstance(experiment["runs"], list):
        raise ReportBridgeError("experiment must contain exactly model and runs")
    cases = _oracle(oracle_path); runs = []
    for spec in experiment["runs"]:
        if not isinstance(spec, dict) or set(spec) != {"case_id", "repetition", "input_path", "artifacts", "review_path"}:
            raise ReportBridgeError("run descriptor shape is invalid")
        case_id, repetition = spec["case_id"], spec["repetition"]
        if case_id not in cases or isinstance(repetition, bool) or repetition not in range(1, RUNS_PER_CASE + 1):
            raise ReportBridgeError("run case_id/repetition is not oracle-declared")
        artifacts, paths, contents = _artifacts(spec["artifacts"], skill)
        static = STATIC_IMAGE if skill == "image-postprocess" else STATIC_FUSION
        schema_valid, dynamic, ratings = _review(_load(spec["review_path"], "blind review"), set(WEIGHTS[skill]), contents, (IMAGE_FACT_KEYS if skill == "image-postprocess" else FUSION_FACT_KEYS) - static)
        runs.append({"case_id": case_id, "repetition": repetition, "input_sha256": _sha(spec["input_path"], "run input"), "artifacts": artifacts, "schema_valid": schema_valid, "facts": _facts(skill, cases[case_id], dynamic, paths.get("multimodal_input")), "ratings": ratings})
    report = {"schema": "duet.skill-iteration-evaluation", "version": 1, "skill": skill, "skill_sha256": _sha(skill_path, "skill"), "skill_bytes": len(_path(skill_path, "skill").read_bytes()), "dataset_manifest_sha256": _sha(dataset_manifest_path, "dataset manifest"), "oracle_sha256": _sha(oracle_path, "oracle"), "runner_sha256": _sha(runner_path, "runner"), "evaluator_sha256": _sha(evaluator_path, "evaluator"), "review_prompt_sha256": _sha(review_prompt_path, "review prompt"), "model_config_sha256": _sha(model_config_path, "model config"), "model": experiment["model"], "runs": runs}
    try:
        return validate_report(report)
    except ValueError as exc:
        raise ReportBridgeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=tuple(WEIGHTS), required=True)
    for name in ("experiment", "oracle", "skill-path", "dataset-manifest", "runner", "evaluator", "review-prompt", "model-config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = build_report(skill=args.skill, experiment_path=args.experiment, oracle_path=args.oracle, skill_path=args.skill_path, dataset_manifest_path=args.dataset_manifest, runner_path=args.runner, evaluator_path=args.evaluator, review_prompt_path=args.review_prompt, model_config_path=args.model_config)
    output = Path(args.output)
    if not output.is_absolute():
        raise ReportBridgeError("output must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
