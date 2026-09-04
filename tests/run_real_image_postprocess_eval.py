#!/usr/bin/env python3
"""Run and score the image-postprocess Skill on frozen real-frame inputs.

This is an explicit offline experiment helper.  It never participates in the
web pipeline and never calls an image provider: only the Skill's two structured
model phases run, followed by the sidecar structural evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Import storage first so this standalone CLI enters the app's intentional
# dialogue_review/voice/storage cycle through its stable production order.
from app import storage as _storage
from app import image_optimization, pipeline
from app.codex_runner import CodexRunner
from app.deepseek_runner import DeepSeekRunner
from eval_image_postprocess_artifact import evaluate


RUNNER_CHOICES = ("deepseek", "codex")
EVAL_GENERATION_CONFIG = {
    "remove_subtitle": True,
    "remove_watermark": True,
}


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_runner(
    runner_name: str,
    *,
    timeout_s: int,
    concurrency: int,
    credential_file: Path | None,
):
    if runner_name == "deepseek":
        if credential_file is None:
            raise ValueError("DeepSeek requires an explicit credential file")
        credential = _require_absolute(
            Path(credential_file), "DeepSeek credential file"
        )
        return DeepSeekRunner(
            timeout_s=timeout_s,
            concurrency=concurrency,
            credential_file=credential,
        )
    if runner_name == "codex":
        return CodexRunner(timeout_s=timeout_s, concurrency=concurrency)
    raise ValueError(f"unsupported runner: {runner_name}")


def _sampling_sources(path: Path) -> list[dict] | None:
    sampling_path = path / "keyframe_sampling.json"
    if not sampling_path.is_file():
        return None
    value = json.loads(sampling_path.read_text(encoding="utf-8"))
    frames = value.get("keyframes") if isinstance(value, dict) else None
    if not isinstance(frames, list) or not frames:
        return None
    result = []
    for item in frames:
        if not isinstance(item, dict):
            return None
        transition = item.get("transition")
        if not isinstance(transition, dict):
            return None
        result.append({
            "order": item.get("order"),
            "source_time_s": item.get("source_time_s"),
            "source_scene_id": item.get("source_scene_id"),
            "transition": {
                "type": transition.get("type"),
                "at_s": transition.get("at_s"),
            },
        })
    return result


def _transition_sources(
    segment_sources: list[list[dict] | None],
    segment_lineage: dict[int, dict[str, str]],
) -> dict[int, list[dict]] | None:
    """Normalize per-segment sampling into one valid cross-segment timeline."""
    if any(source is None for source in segment_sources):
        return None
    result: dict[int, list[dict]] = {}
    previous_scene: str | None = None
    global_first = True
    for segment_index, source_items in enumerate(segment_sources, 1):
        assert source_items is not None
        lineage = segment_lineage[segment_index]
        normalized = []
        for frame_index, source in enumerate(source_items, 1):
            scene = source.get("source_scene_id")
            source_time = source.get("source_time_s")
            if (
                not isinstance(scene, str)
                or not scene
                or isinstance(source_time, bool)
                or not isinstance(source_time, (int, float))
            ):
                return None
            if global_first:
                transition_type = "start"
            elif frame_index == 1:
                if lineage["join_mode"] == "hard_cut":
                    transition_type = "hard_cut"
                elif scene == previous_scene:
                    transition_type = "continuous"
                else:
                    return None
            elif scene != previous_scene:
                transition_type = "hard_cut"
            else:
                transition_type = "continuous"
            normalized.append({
                "order": frame_index,
                "source_time_s": float(source_time),
                "source_scene_id": scene,
                "transition": {
                    "type": transition_type,
                    "at_s": float(source_time) if transition_type != "continuous" else None,
                },
            })
            previous_scene = scene
            global_first = False
        result[segment_index] = normalized
    return result


def _segment_lineage(
    source_root: Path, segment_indices: list[int],
) -> dict[int, dict[str, str]]:
    """Read exact chain topology from the frozen long-video receipt."""
    plan_path = source_root / "long_video_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("frozen long_video_plan.json is missing")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("frozen long_video_plan.json is invalid") from None
    raw_segments = plan.get("segments") if isinstance(plan, dict) else None
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != "duet.long-video-plan"
        or not isinstance(raw_segments, list)
    ):
        raise ValueError("frozen long_video_plan.json is invalid")

    result: dict[int, dict[str, str]] = {}
    receipt_indices: list[int] = []
    previous_chain: str | None = None
    for position, item in enumerate(raw_segments, 1):
        index = item.get("index") if isinstance(item, dict) else None
        chain_id = item.get("chain_id") if isinstance(item, dict) else None
        join_mode = item.get("join_mode") if isinstance(item, dict) else None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in result
            or not isinstance(chain_id, str)
            or not chain_id
            or len(chain_id) > 128
            or join_mode not in {"hard_cut", "continue"}
            or (position == 1 and join_mode != "hard_cut")
            or (join_mode == "continue" and chain_id != previous_chain)
            or (
                position > 1
                and join_mode == "hard_cut"
                and chain_id == previous_chain
            )
        ):
            raise ValueError("frozen long_video_plan.json topology is invalid")
        receipt_indices.append(index)
        result[index] = {"chain_id": chain_id, "join_mode": join_mode}
        previous_chain = chain_id
    if receipt_indices != segment_indices:
        raise ValueError(
            "frozen long_video_plan.json segments do not match source segments"
        )
    return result


def _segments(source_root: Path) -> tuple[list[dict], dict[int, list[Path]]]:
    root = _require_absolute(source_root / "work" / "segments", "segment root")
    directories = sorted(
        (item for item in root.iterdir() if item.is_dir() and item.name.isdigit()),
        key=lambda item: int(item.name),
    )
    if not directories:
        raise ValueError("source root has no segment directories")
    if [int(item.name) for item in directories] != list(range(1, len(directories) + 1)):
        raise ValueError("source segment indices must be contiguous from 1")
    segment_indices = [int(item.name) for item in directories]
    segment_lineage = _segment_lineage(source_root, segment_indices)
    paths_by_segment: dict[int, list[Path]] = {}
    raw_sources = []
    for directory in directories:
        keyframes = directory / "work" / "keyframes"
        frames = sorted(
            (item for item in keyframes.iterdir() if item.is_file() and item.suffix == ".png"),
            key=lambda item: int(item.stem) if item.stem.isdigit() else 10**9,
        )
        if not frames or [item.name for item in frames] != [f"{n:02d}.png" for n in range(1, len(frames) + 1)]:
            raise ValueError(f"invalid frozen keyframes in {keyframes}")
        paths_by_segment[int(directory.name)] = frames
        raw_sources.append(_sampling_sources(directory / "work"))
    sources = _transition_sources(raw_sources, segment_lineage)
    specs = []
    inventory = None
    if sources is not None:
        inventory = pipeline._frame_inventory(
            paths_by_segment,
            segment_lineage=segment_lineage,
            keyframe_sources=sources,
        )
    for segment_index, directory in enumerate(directories, 1):
        lineage = segment_lineage[segment_index]
        spec = {
            "index": segment_index,
            **lineage,
            "keyframes_dir": directory / "work" / "keyframes",
        }
        if inventory is not None:
            spec["transition_skeleton"] = [
                item for item in inventory
                if item["segment_index"] == segment_index
            ]
        specs.append(spec)
    return specs, paths_by_segment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_case(
    source_root: Path,
    element_index: Path,
    skill_path: Path,
    evidence_dir: Path,
    *,
    timeout_s: int,
    blind_holdout: bool = False,
    runner_name: str = "deepseek",
    credential_file: Path | None = None,
    user_reference_image: Path | None = None,
    user_replacement_prompt: str | None = None,
) -> dict:
    source_root = _require_absolute(source_root.resolve(strict=True), "source root")
    element_index = _require_absolute(element_index.resolve(strict=True), "element index")
    skill_path = _require_absolute(skill_path.resolve(strict=True), "Skill path")
    evidence_dir = _require_absolute(evidence_dir, "evidence directory")
    if not element_index.is_file() or not skill_path.is_file():
        raise ValueError("element index and Skill must be regular files")
    if (user_reference_image is None) != (user_replacement_prompt is None):
        raise ValueError(
            "user reference image and replacement prompt must be provided together"
        )
    if user_reference_image is not None:
        requested_reference = _require_absolute(
            Path(user_reference_image), "user reference image"
        )
        if requested_reference.is_symlink():
            raise ValueError("user reference image must be a regular file")
        user_reference_image = requested_reference.resolve(strict=True)
        if not user_reference_image.is_file():
            raise ValueError("user reference image must be a regular file")
    skill_bytes = skill_path.read_bytes()
    skill_sha256 = hashlib.sha256(skill_bytes).hexdigest()
    skill_lines = len(skill_bytes.decode("utf-8").splitlines())
    evidence_dir.mkdir(parents=True, exist_ok=True)

    specs, _paths_by_segment = _segments(source_root)
    runner = _build_runner(
        runner_name,
        timeout_s=timeout_s,
        concurrency=max(1, len(specs)),
        credential_file=credential_file,
    )
    captured = evidence_dir / "phase-outputs"
    captured.mkdir(parents=True, exist_ok=True)
    original = image_optimization._run_image_skill_phase

    def capture(runner, **kwargs):
        value = original(runner, **kwargs)
        request = json.loads(
            (kwargs["stage"] / "work" / "request.json").read_text(encoding="utf-8")
        )
        phase = request.get("phase", "unknown")
        segment = request.get("segment", {}).get("index")
        suffix = f"segment-{segment:02d}" if isinstance(segment, int) else "project"
        shutil.copy2(
            kwargs["stage"] / "work" / kwargs["output_name"],
            captured / f"{phase}-{suffix}.json",
        )
        return value

    image_optimization._run_image_skill_phase = capture
    try:
        plan, prompts = image_optimization.generate_project_prompts(
            runner,
            specs,
            "independent_parallel",
            session_dir=source_root,
            expected_version=4,
            element_index_path=element_index,
            skill_bytes=skill_bytes,
            generation_config=dict(EVAL_GENERATION_CONFIG),
            user_reference_image_path=user_reference_image,
            user_replacement_prompt=user_replacement_prompt,
        )
    finally:
        image_optimization._run_image_skill_phase = original

    output_path = evidence_dir / "image_optimization.json"
    prompts_path = evidence_dir / "compiled-prompts.json"
    _write_json(output_path, plan)
    _write_json(prompts_path, prompts)
    source_mp4 = source_root / "source.mp4"
    report = evaluate(
        source_root,
        element_index,
        output_path,
        source_sha256=_sha256(source_mp4) if source_mp4.is_file() else None,
    )
    report["experiment"] = {
        "skill_sha256": skill_sha256,
        "skill_lines": skill_lines,
        "skill_bytes": len(skill_bytes),
        "case_root": str(source_root),
        "blind_holdout": blind_holdout,
    }
    _write_json(evidence_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=_absolute_path)
    parser.add_argument("--element-index", required=True, type=_absolute_path)
    parser.add_argument("--skill", required=True, type=_absolute_path)
    parser.add_argument("--evidence-dir", required=True, type=_absolute_path)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--blind-holdout", action="store_true")
    parser.add_argument("--runner", choices=RUNNER_CHOICES, default="deepseek")
    parser.add_argument("--credential-file", type=_absolute_path)
    parser.add_argument("--user-reference-image", type=_absolute_path)
    parser.add_argument("--user-replacement-prompt")
    args = parser.parse_args()
    report = run_case(
        args.source_root,
        args.element_index,
        args.skill,
        args.evidence_dir,
        timeout_s=args.timeout_s,
        blind_holdout=args.blind_holdout,
        runner_name=args.runner,
        credential_file=args.credential_file,
        user_reference_image=args.user_reference_image,
        user_replacement_prompt=args.user_replacement_prompt,
    )
    print(json.dumps({
        "descriptive_mean": report.get("descriptive_mean"),
        "scores": report.get("scores"),
        "output_sha256": report.get("output", {}).get("sha256"),
        "skill_sha256": report.get("experiment", {}).get("skill_sha256"),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
