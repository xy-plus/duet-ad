"""Isolated three-frame Seedream proof runner for a frozen dual-target plan."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2

from app import image_optimization, seedream
try:
    from app import replacement_packs
except ImportError:  # The isolated runner stays unavailable until its pack module lands.
    replacement_packs = None
from app.config import Settings, get_settings


SCHEMA = "duet.dual-target-poc"
VERSION = 1
MAX_FRAMES = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DualTargetPocError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class PocOutput:
    receipt_path: Path
    result_dir: Path
    frames: tuple[Path, ...]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_root(path: Path, *, create: bool) -> Path:
    requested = Path(path)
    if not requested.is_absolute() or requested.is_symlink():
        raise DualTargetPocError("invalid_path")
    try:
        if create:
            requested.mkdir(parents=True, exist_ok=True)
        resolved = requested.resolve(strict=True)
    except OSError:
        raise DualTargetPocError("invalid_path") from None
    if not resolved.is_dir() or resolved.is_symlink():
        raise DualTargetPocError("invalid_path")
    return resolved


def _read_meta(project_dir: Path) -> dict:
    path = project_dir / "meta.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise DualTargetPocError("plan_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise DualTargetPocError("plan_invalid") from None
    if not isinstance(value, dict):
        raise DualTargetPocError("plan_invalid")
    return value


def _safe_frame(project_dir: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str) or not relative or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or PurePosixPath(relative).as_posix() != relative
        or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
        or PurePosixPath(relative).suffix.lower() != ".png"
    ):
        raise DualTargetPocError("frame_inventory_invalid")
    current = project_dir
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise DualTargetPocError("frame_inventory_invalid")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(project_dir)
    except (OSError, ValueError):
        raise DualTargetPocError("frame_inventory_invalid") from None
    if not resolved.is_file() or cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED) is None:
        raise DualTargetPocError("frame_inventory_invalid")
    return resolved


def _project_frames(project_dir: Path, meta: dict, plan: Mapping[str, Any]) -> list[dict]:
    indices = plan.get("segment_indices")
    if not isinstance(indices, list) or not indices:
        raise DualTargetPocError("plan_invalid")
    names_by_segment: dict[int, list[str]] = {}
    if indices == [0]:
        names_by_segment[0] = meta.get("keyframes")
    else:
        segments = meta.get("segments")
        if not isinstance(segments, list):
            raise DualTargetPocError("frame_inventory_invalid")
        for item in segments:
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                names_by_segment[item["index"]] = item.get("keyframes")
    frames = []
    for segment_index in indices:
        names = names_by_segment.get(segment_index)
        if not isinstance(names, list) or not names:
            raise DualTargetPocError("frame_inventory_invalid")
        for frame_index, name in enumerate(names, 1):
            prefix = "work/keyframes" if segment_index == 0 else (
                f"work/segments/{segment_index}/work/keyframes"
            )
            source = _safe_frame(project_dir, f"{prefix}/{name}")
            frames.append({
                "segment_index": segment_index,
                "frame_index": frame_index,
                "frame_name": source.name,
                "source_sha256": _sha(source),
                "path": source,
                "relative_path": source.relative_to(project_dir).as_posix(),
            })
    return frames


def _selected_positions(frames: list[dict], requested: Sequence[tuple[int, int]] | None) -> list[tuple[int, int]]:
    available = [(item["segment_index"], item["frame_index"]) for item in frames]
    if requested is not None:
        positions = list(requested)
        if (
            not positions or len(positions) > MAX_FRAMES or len(set(positions)) != len(positions)
            or any(
                not isinstance(item, tuple) or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
                or item not in available
                for item in positions
            )
        ):
            raise DualTargetPocError("frame_selection_invalid")
        return positions
    if len(available) <= MAX_FRAMES:
        return available
    offsets = (0, (len(available) - 1) // 2, len(available) - 1)
    return [available[offset] for offset in offsets]


def _pack_image(entity: object, role: str, project_dir: Path) -> tuple[Path, str]:
    images = getattr(entity, "images", None)
    matches = [item for item in images or () if getattr(item, "role", None) == role]
    if len(matches) != 1:
        raise DualTargetPocError("replacement_pack_invalid")
    item = matches[0]
    path = getattr(item, "path", None)
    digest = getattr(item, "sha256", None)
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise DualTargetPocError("replacement_pack_invalid")
    try:
        path.resolve(strict=True).relative_to(project_dir)
    except (OSError, ValueError):
        raise DualTargetPocError("replacement_pack_invalid") from None
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None or _sha(path) != digest:
        raise DualTargetPocError("replacement_pack_invalid")
    return path, digest


def _frame_inputs(
    frame: Mapping[str, Any], pack: object, source: Path,
) -> tuple[list[Path], list[str]]:
    people = frame.get("observable_person_ids")
    scene_id = frame.get("scene_id")
    if (
        not isinstance(people, list)
        or any(not isinstance(person_id, str) or not person_id for person_id in people)
        or people != sorted(set(people))
        or not isinstance(scene_id, str) or not scene_id
    ):
        raise DualTargetPocError("execution_invalid")
    paths = [source]
    roles = ["current_frame"]
    for person_id in people:
        try:
            entity = pack.people[person_id]
        except (KeyError, TypeError):
            raise DualTargetPocError("replacement_pack_invalid") from None
        for role in ("primary", "alternate"):
            path, _digest = _pack_image(entity, role, pack.project_dir)
            paths.append(path)
            roles.append(f"identity:{person_id}:{role}")
    try:
        scene = pack.scenes[scene_id]
    except (KeyError, TypeError):
        raise DualTargetPocError("replacement_pack_invalid") from None
    for role in ("primary", "alternate"):
        path, _digest = _pack_image(scene, role, pack.project_dir)
        paths.append(path)
        roles.append(f"scene:{scene_id}:{role}")
    if len(paths) > 10 or len(paths) != len(roles) or len(set(roles)) != len(roles):
        raise DualTargetPocError("seedream_reference_limit_exceeded")
    return paths, roles


def _frame_prompt(base: str, roles: Sequence[str], observable_people: Sequence[str]) -> str:
    visibility = (
        "只替换当前帧可见主人物：" + "、".join(observable_people) + "；同时真实更换场景。"
        if observable_people else "当前帧不得新增人物；仍须真实更换场景。"
    )
    return (
        base.strip() + "\n\n" + visibility
        + "图1是唯一编辑画布；其余图只提供冻结的新人物身份或新场景设计，"
        "不得传递构图、机位、动作、光线或实体关系。保持图1的剧情、构图、"
        "动作、物体功能关系和全局光色，输出自然且无扭曲。输入角色顺序："
        + "；".join(f"图{index}={role}" for index, role in enumerate(roles, 1))
    )


def _receipt(path: Path, value: dict) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    seedream._atomic_json(path, {**unsigned, "sha256": _hash_json(unsigned)})


def _load_existing(path: Path, frozen: dict) -> dict | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise DualTargetPocError("poc_receipt_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise DualTargetPocError("poc_receipt_invalid") from None
    unsigned = {key: item for key, item in value.items() if key != "sha256"} \
        if isinstance(value, dict) else None
    if (
        not isinstance(unsigned, dict)
        or value.get("sha256") != _hash_json(unsigned)
        or any(value.get(key) != item for key, item in frozen.items())
    ):
        raise DualTargetPocError("poc_receipt_invalid")
    return value


def _published_output(evaluation_dir: Path, frame: dict) -> Path:
    return evaluation_dir / "results" / (
        f"s{frame['segment_index']:04d}-f{frame['frame_index']:04d}.png"
    )


async def run_three_frame_seedream_poc(
    settings: Settings,
    project_dir: Path,
    evaluation_dir: Path,
    *,
    frame_positions: Sequence[tuple[int, int]] | None = None,
    transport: Any = None,
) -> PocOutput:
    """Edit up to three frozen frames without touching project metadata or outputs."""
    project = _safe_root(project_dir, create=False)
    requested_evaluation = Path(evaluation_dir)
    if not requested_evaluation.is_absolute() or requested_evaluation.is_symlink():
        raise DualTargetPocError("invalid_path")
    try:
        requested_evaluation.resolve(strict=False).relative_to(project)
    except (OSError, ValueError):
        pass
    else:
        raise DualTargetPocError("evaluation_dir_not_isolated")
    evaluation = _safe_root(requested_evaluation, create=True)
    meta_before = (project / "meta.json").read_bytes()
    if replacement_packs is None:
        raise DualTargetPocError("replacement_pack_invalid")
    meta = _read_meta(project)
    try:
        plan = image_optimization.dual_target_plan_receipt(meta)
    except Exception:
        raise DualTargetPocError("plan_invalid") from None
    if not isinstance(plan, dict) or plan.get("version") != 2 or plan.get("eligible") is not True:
        raise DualTargetPocError("plan_invalid")
    person_ids = tuple(item["id"] for item in plan["person_plans"])
    scene_ids = tuple(item["id"] for item in plan["scene_plans"])
    if not person_ids or not scene_ids:
        raise DualTargetPocError("plan_not_dual_target")
    revision = 1
    try:
        initial_pack = replacement_packs.load_replacement_pack(
            project,
            expected_upstream_plan_sha256=plan["sha256"],
            expected_model=settings.seedream_model,
            expected_revision=revision,
            expected_person_plan_ids=person_ids,
            expected_scene_plan_ids=scene_ids,
        )
        profile = dict(initial_pack.execution_profile)
    except Exception:
        raise DualTargetPocError("replacement_pack_invalid") from None
    if (
        set(profile) != {"id", "revision"}
        or not isinstance(profile["id"], str) or not profile["id"].strip()
        or isinstance(profile["revision"], bool) or not isinstance(profile["revision"], int)
        or profile["revision"] < 1
    ):
        raise DualTargetPocError("replacement_pack_invalid")
    frames = _project_frames(project, meta, plan)
    inventory = [{key: item[key] for key in (
        "segment_index", "frame_index", "frame_name", "source_sha256"
    )} for item in frames]
    try:
        execution = image_optimization.freeze_execution_inputs(
            plan, revision=revision, profile=profile,
            model=settings.seedream_model, frame_inventory=inventory,
        )
        source_inventory_sha256 = (
            replacement_packs.canonical_source_inventory_sha256(execution["frames"])
        )
        profile_sha256 = _hash_json(profile)
        pack = replacement_packs.load_replacement_pack(
            project,
            expected_upstream_plan_sha256=execution["plan_sha256"],
            expected_upstream_source_inventory_sha256=source_inventory_sha256,
            expected_execution_profile_sha256=profile_sha256,
            expected_model=settings.seedream_model,
            expected_revision=revision,
            expected_person_plan_ids=person_ids,
            expected_scene_plan_ids=scene_ids,
        )
        prompts = image_optimization.compile_segment_prompts(plan, "independent_parallel")
    except Exception:
        raise DualTargetPocError("execution_invalid") from None
    selected = _selected_positions(frames, frame_positions)
    by_position = {(item["segment_index"], item["frame_index"]): item for item in frames}
    execution_by_position = {
        (item["segment_index"], item["frame_index"]): item
        for item in execution["frames"]
    }
    if not any(
        execution_by_position.get(position, {}).get("observable_person_ids")
        for position in selected
    ):
        visible = next((
            position for position in (
                (item["segment_index"], item["frame_index"]) for item in frames
            )
            if execution_by_position.get(position, {}).get("observable_person_ids")
        ), None)
        if visible is None or frame_positions is not None:
            raise DualTargetPocError("frame_selection_not_dual_target")
        selected[min(1, len(selected) - 1)] = visible
        order = {
            (item["segment_index"], item["frame_index"]): index
            for index, item in enumerate(frames)
        }
        selected = sorted(set(selected), key=order.__getitem__)
    work = evaluation / "work"
    work.mkdir(exist_ok=True)
    frozen_frames = []
    runtime = []
    for segment_index, frame_index in selected:
        source_item = by_position[(segment_index, frame_index)]
        frame = execution_by_position.get((segment_index, frame_index))
        base = prompts.get(segment_index)
        if not isinstance(frame, dict) or not isinstance(base, str) or not base.strip():
            raise DualTargetPocError("execution_invalid")
        paths, roles = _frame_inputs(frame, pack, source_item["path"])
        prompt = _frame_prompt(base, roles, frame["observable_person_ids"])
        name = f"s{segment_index:04d}-f{frame_index:04d}.png"
        output = work / name
        attempt = work / f"{name}.attempt.json"
        frozen_frames.append({
            "segment_index": segment_index,
            "frame_index": frame_index,
            "source": {
                "path": source_item["relative_path"],
                "sha256": source_item["source_sha256"],
            },
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "input_order": [
                {"position": index, "role": role, "sha256": _sha(path)}
                for index, (role, path) in enumerate(zip(roles, paths), 1)
            ],
            "attempt_receipt": attempt.relative_to(evaluation).as_posix(),
            "staged_output": output.relative_to(evaluation).as_posix(),
            "published_output": f"results/{name}",
        })
        runtime.append((paths, roles, prompt, output, attempt))
    frozen = {
        "schema": SCHEMA,
        "version": VERSION,
        "plan_sha256": execution["plan_sha256"],
        "replacement_pack_candidate_sha256": pack.candidate_sha256,
        "replacement_pack_receipt_sha256": pack.receipt_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "model": settings.seedream_model,
        "profile": profile,
        "revision": revision,
        "frames": frozen_frames,
    }
    receipt_path = evaluation / "run.json"
    existing = _load_existing(receipt_path, frozen)
    if existing is not None and existing.get("status") == "done":
        outputs = tuple(_published_output(evaluation, item) for item in frozen_frames)
        stored_results = existing.get("results")
        if (
            isinstance(stored_results, list) and len(stored_results) == len(outputs)
            and all(
                isinstance(result, dict)
                and set(result) == {"path", "sha256"}
                and result["path"] == frame["published_output"]
                and path.is_file() and _sha(path) == result["sha256"]
                for path, frame, result in zip(outputs, frozen_frames, stored_results)
            )
            and (project / "meta.json").read_bytes() == meta_before
        ):
            return PocOutput(receipt_path, evaluation / "results", outputs)
        raise DualTargetPocError("poc_receipt_invalid")
    if existing is not None and existing.get("status") in {"failed", "submission_unknown"}:
        raise DualTargetPocError(existing["status"])
    _receipt(receipt_path, {**frozen, "status": "running", "results": []})

    async def edit(item: tuple[list[Path], list[str], str, Path, Path]) -> Path:
        paths, roles, prompt, output, attempt = item
        return await seedream.edit(
            settings,
            [path.read_bytes() for path in paths],
            prompt,
            output,
            receipt_path=attempt,
            execution_binding={
                "plan_sha256": execution["plan_sha256"],
                "profile": profile,
                "revision": revision,
                "input_roles": roles,
                "reference_pack_candidate_sha256": pack.candidate_sha256,
            },
            transport=transport,
        )

    results = await asyncio.gather(*(edit(item) for item in runtime), return_exceptions=True)
    errors = [item for item in results if isinstance(item, BaseException)]
    if errors:
        error = errors[0]
        code = (
            "submission_unknown"
            if isinstance(error, seedream.SeedreamError)
            and error.code == "submission_unknown"
            else "failed"
        )
        _receipt(receipt_path, {**frozen, "status": code, "results": []})
        raise DualTargetPocError(code)
    staged = [Path(item) for item in results]
    publish_tmp = evaluation / ".results.publishing"
    if publish_tmp.exists():
        shutil.rmtree(publish_tmp)
    publish_tmp.mkdir()
    result_receipts = []
    for source_item, output in zip(frozen_frames, staged):
        if not output.is_file() or cv2.imread(str(output), cv2.IMREAD_UNCHANGED) is None:
            raise DualTargetPocError("output_invalid")
        destination = publish_tmp / Path(source_item["published_output"]).name
        shutil.copyfile(output, destination)
        result_receipts.append({
            "path": source_item["published_output"], "sha256": _sha(destination),
        })
    result_dir = evaluation / "results"
    if result_dir.exists():
        raise DualTargetPocError("poc_receipt_invalid")
    os.replace(publish_tmp, result_dir)
    seedream._fsync_dir(evaluation)
    if (project / "meta.json").read_bytes() != meta_before:
        raise DualTargetPocError("project_mutated")
    _receipt(receipt_path, {**frozen, "status": "done", "results": result_receipts})
    return PocOutput(
        receipt_path, result_dir,
        tuple(_published_output(evaluation, item) for item in frozen_frames),
    )


def _cli_position(value: str) -> tuple[int, int]:
    try:
        segment, frame = value.split(":", 1)
        parsed = (int(segment), int(frame))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("frame must use SEGMENT:FRAME") from None
    if parsed[0] < 0 or parsed[1] < 1:
        raise argparse.ArgumentTypeError("frame must use SEGMENT>=0 and FRAME>=1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--frame", action="append", type=_cli_position)
    args = parser.parse_args()
    try:
        result = asyncio.run(run_three_frame_seedream_poc(
            get_settings(),
            args.project_dir,
            args.evaluation_dir,
            frame_positions=args.frame,
        ))
    except DualTargetPocError as error:
        raise SystemExit(error.code) from None
    print(json.dumps({
        "receipt_path": str(result.receipt_path),
        "result_dir": str(result.result_dir),
        "frames": [str(path) for path in result.frames],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["DualTargetPocError", "PocOutput", "run_three_frame_seedream_poc"]
