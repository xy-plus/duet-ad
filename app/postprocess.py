"""Fail-closed, segment-scoped MediaKit -> Seedream postprocessing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2

from app import image_optimization, mediakit, seedream, storage
from app.config import Settings
from app.sanitize import sanitize

OPTION_KEYS = ("remove_subtitle", "remove_brand", "optimize_image")
_OLD_OPTION_KEYS = frozenset({"remove_subtitle", "remove_brand"})
_OPTION_SET = frozenset(OPTION_KEYS)
_STALE_KEYS = frozenset({"change_bg", "face_hold"})
_PUBLIC_STATUSES = frozenset({"running", "done", "failed"})
_PUBLIC_STAGES = frozenset({"queued", "text", "brand", "seedream", "publishing", "done"})
_PUBLIC_ERROR_CODES = frozenset({
    "cancelled", "submission_unknown", "provider_rejected",
    "provider_protocol_error", "postprocess_artifacts_invalid",
    "postprocess_canonical_conflict", "image_optimization_prompt_invalid",
    "postprocess_receipt_invalid", "image_plan_audit_failed",
    "image_verification_failed", "segment_failed",
})
_FRAME_ERROR_RE = re.compile(r"^frame ([A-Za-z0-9_.-]{1,128}) failed(?:$|:)")
_PUBLIC_FRAME_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]{1,128}|segments/[1-9]\d*/work/postprocessed/[A-Za-z0-9_.-]{1,128})$"
)


class PostprocessError(Exception):
    def __init__(self, status: int, detail: str | dict[str, str]) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _structured(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _public_error(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value in _PUBLIC_ERROR_CODES:
        return value
    if isinstance(value, str):
        matched = _FRAME_ERROR_RE.match(value)
        if matched and matched.group(1) not in {".", ".."}:
            return f"frame {matched.group(1)} failed"
    return "postprocess_failed"


def public_state(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    raw_options = value.get("options")
    options = {
        key: raw_options.get(key) if isinstance(raw_options, dict)
        and isinstance(raw_options.get(key), bool) else False
        for key in OPTION_KEYS
    }
    raw_frames = value.get("frames")
    result = {
        "status": (
            status if isinstance(status, str) and status in _PUBLIC_STATUSES else "failed"
        ),
        "options": options,
        "frames": [
            item for item in raw_frames
            if isinstance(item, str) and _PUBLIC_FRAME_RE.match(item)
            and item.rsplit("/", 1)[-1] not in {".", ".."}
        ] if isinstance(raw_frames, list) else [],
        "error": _public_error(value.get("error")),
    }
    raw_segments = value.get("segments")
    if raw_segments is None:
        raw_segments = []
        valid_segments = True
    else:
        valid_segments = isinstance(raw_segments, list)
    indices = [
        item.get("index") for item in raw_segments
        if isinstance(item, dict)
    ] if isinstance(raw_segments, list) else []
    valid_segments = (
        valid_segments and len(indices) == len(raw_segments)
        and all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        and (
        indices == [0] or indices == list(range(1, len(indices) + 1))
        )
    )
    if valid_segments:
        for item in raw_segments:
            completed, total, revision = (
                item.get("completed_frames"), item.get("total_frames"), item.get("revision")
            )
            if (
                isinstance(completed, bool) or not isinstance(completed, int)
                or isinstance(total, bool) or not isinstance(total, int)
                or isinstance(revision, bool) or not isinstance(revision, int)
                or completed < 0 or total < 0 or completed > total or revision < 1
            ):
                valid_segments = False
                break
    result["segments"] = [
        {
            "index": item.get("index"),
            "status": (
                item.get("status")
                if isinstance(item.get("status"), str)
                and item.get("status") in _PUBLIC_STATUSES else "failed"
            ),
            "stage": (
                item.get("stage")
                if isinstance(item.get("stage"), str)
                and item.get("stage") in _PUBLIC_STAGES else "unknown"
            ),
            "completed_frames": item["completed_frames"],
            "total_frames": item["total_frames"],
            "revision": item["revision"],
            "error": _public_error(item.get("error")),
        }
        for item in raw_segments
    ] if valid_segments else []
    if not valid_segments:
        result.update(status="failed", frames=[], error="postprocess_receipt_invalid")
    return result


def _private_receipt(meta: dict) -> dict:
    raw = meta.get("_postprocess_receipt")
    if not isinstance(raw, dict) or raw.get("version") not in {2, 3, 4}:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    expected_keys = (
        {"version", "options", "model", "edit_mode", "timeout_s", "prompts"}
        if raw["version"] == 2 else ({
            "version", "options", "model", "edit_mode", "timeout_s",
            "plan_sha256", "frames", "receipt_sha256",
        } if raw["version"] == 3 else {
            "version", "options", "model", "edit_mode", "timeout_s",
            "plan_sha256", "continuity_sha256", "execution_input_sha256",
            "execution_inputs", "scene_anchor_schedule", "frames",
            "receipt_sha256",
        })
    )
    if set(raw) != expected_keys:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    options = raw.get("options")
    post = meta.get("postprocess")
    if not isinstance(post, dict):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    public_options = post.get("options")
    if (
        not isinstance(options, dict) or set(options) != _OPTION_SET
        or any(not isinstance(options[key], bool) for key in OPTION_KEYS)
        or not any(options.values())
        or not isinstance(public_options, dict) or set(public_options) != _OPTION_SET
        or any(not isinstance(public_options[key], bool) for key in OPTION_KEYS)
        or any(public_options[key] != options[key] for key in OPTION_KEYS)
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    timeout = raw.get("timeout_s")
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout)) or timeout <= 0
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    project_frozen = image_optimization.receipt(meta)
    private_frozen = (
        {
            "version": 2, "model": raw.get("model"),
            "edit_mode": raw.get("edit_mode"), "segments": raw.get("prompts"),
        }
        if raw["version"] == 2 else ({
            "version": 3, "plan_sha256": raw.get("plan_sha256"),
            "model": raw.get("model"), "edit_mode": raw.get("edit_mode"),
            "frames": raw.get("frames"), "sha256": raw.get("receipt_sha256"),
        } if raw["version"] == 3 else {
            "version": 4,
            "plan_sha256": raw.get("plan_sha256"),
            "continuity_sha256": raw.get("continuity_sha256"),
            "execution_input_sha256": raw.get("execution_input_sha256"),
            "execution_inputs": raw.get("execution_inputs"),
            "model": raw.get("model"),
            "edit_mode": raw.get("edit_mode"),
            "scene_anchor_schedule": raw.get("scene_anchor_schedule"),
            "frames": raw.get("frames"), "sha256": raw.get("receipt_sha256"),
        })
    )
    if project_frozen is None or private_frozen != project_frozen:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    result = {**raw, "options": {key: options[key] for key in OPTION_KEYS}}
    if raw["version"] == 2:
        result["prompts"] = [dict(item) for item in project_frozen["segments"]]
    else:
        result["frames"] = [dict(item) for item in project_frozen["frames"]]
        if raw["version"] == 4:
            result["execution_inputs"] = deepcopy(project_frozen["execution_inputs"])
            result["scene_anchor_schedule"] = deepcopy(
                project_frozen["scene_anchor_schedule"]
            )
    return result


def _parse_options(payload: dict) -> dict[str, bool]:
    raw = payload.get("options")
    if not isinstance(raw, dict):
        raise PostprocessError(422, "at least one option required")
    if not raw:
        raise PostprocessError(422, "at least one option required")
    if any(not isinstance(value, bool) for value in raw.values()):
        raise PostprocessError(422, "options must be booleans")
    keys = frozenset(raw)
    stale = keys - _OPTION_SET
    if stale and stale <= _STALE_KEYS:
        raise PostprocessError(409, "页面版本已更新，请刷新页面后重试。")
    if keys not in {_OLD_OPTION_KEYS, _OPTION_SET}:
        unknown = sorted(keys - _OPTION_SET)
        if unknown:
            raise PostprocessError(422, f"unknown options: {', '.join(unknown)}")
        raise PostprocessError(422, "invalid_postprocess_options")
    options = {key: bool(raw.get(key, False)) for key in OPTION_KEYS}
    if not any(options.values()):
        raise PostprocessError(422, "at least one option required")
    return options


def _capability_gate(settings: Settings, options: dict[str, bool]) -> None:
    if (options["remove_subtitle"] or options["remove_brand"]) and not settings.enable_mediakit_erase:
        raise PostprocessError(501, "MediaKit erase is disabled.")
    if options["optimize_image"] and not os.environ.get("ARK_API_KEY", "").strip():
        raise PostprocessError(501, "Seedream image optimization is disabled.")


def _options_match(previous: object, current: dict[str, bool]) -> bool:
    if not isinstance(previous, dict):
        return False
    if set(previous) and not any(previous.get(key) is True for key in OPTION_KEYS):
        return True
    return all(bool(previous.get(key, False)) == value for key, value in current.items())


def _pure_legacy(previous: object) -> bool:
    return (
        isinstance(previous, dict)
        and bool(set(previous) & _STALE_KEYS)
        and not any(previous.get(key) is True for key in OPTION_KEYS)
    )


def _clear_canonical(cdir: Path, grouped: dict[int, list[tuple[Path, Path]]]) -> None:
    for targets in grouped.values():
        destination = targets[0][1].parent
        if destination.is_dir():
            shutil.rmtree(destination)


def _valid_png(candidate: Path, source: Path) -> bool:
    try:
        with candidate.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
        decoded = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    except OSError:
        return False
    return (
        decoded is not None and original is not None
        and decoded.shape == original.shape
    )


_PALETTE_METRIC_ALGORITHM = image_optimization.PALETTE_METRIC_ALGORITHM
_PALETTE_METRIC_THRESHOLDS = image_optimization.PALETTE_METRIC_THRESHOLDS


def _receipt_sha256(payload: dict) -> str:
    """Digest a JSON receipt without permitting NaN or ordering ambiguity."""
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _area_weighted_palette_metric(path: Path) -> dict:
    """Return the backend-owned whole-frame palette proxy."""
    try:
        return image_optimization.source_palette_metric(path)
    except ValueError:
        raise PostprocessError(409, "dominant_palette_metric_invalid") from None


def _v4_palette_metrics(
    plan: dict,
    sources: dict[tuple[int, int], Path],
    outputs: dict[tuple[int, int], Path] | None = None,
) -> dict:
    """Measure v4 source/output palettes without turning quality into eligibility."""
    constraints = {
        (segment["segment_index"], frame["frame_index"]): frame
        for segment in plan["segments"]
        for frame in segment["frame_constraints"]
    }
    if (
        set(sources) != set(constraints)
        or outputs is not None and not set(outputs).issubset(sources)
    ):
        raise PostprocessError(409, "dominant_palette_metric_invalid")
    frames = []
    for key in sorted(sources):
        contract = constraints[key]["dominant_palette_contract"]
        source = _area_weighted_palette_metric(sources[key])
        record = {
            "segment_index": key[0],
            "frame_index": key[1],
            "contract": contract,
            "source": source,
        }
        if outputs is not None and key in outputs:
            output = _area_weighted_palette_metric(outputs[key])
            record["output"] = output
        frames.append(record)
    payload = {
        "version": 1,
        "algorithm": _PALETTE_METRIC_ALGORITHM,
        "thresholds": _PALETTE_METRIC_THRESHOLDS,
        "frames": frames,
    }
    return {**payload, "sha256": _receipt_sha256(payload)}


def _v4_endpoint_palette_metric(
    plan: dict, sources: dict[tuple[int, int], Path], key: tuple[int, int], output: Path,
) -> dict:
    constraints = {
        (segment["segment_index"], frame["frame_index"]): frame
        for segment in plan["segments"]
        for frame in segment["frame_constraints"]
    }
    source = sources.get(key)
    constraint = constraints.get(key)
    if source is None or not isinstance(constraint, dict):
        raise PostprocessError(409, "dominant_palette_metric_invalid")
    contract = constraint["dominant_palette_contract"]
    source_metric = _area_weighted_palette_metric(source)
    output_metric = _area_weighted_palette_metric(output)
    if (
        source_metric["warm_cool_family"] != contract["area_weighted_warm_cool_family"]
        or source_metric["saturation_style"] != contract["saturation_style"]
    ):
        raise PostprocessError(409, "dominant_palette_source_mismatch")
    if (
        output_metric["warm_cool_family"] != contract["area_weighted_warm_cool_family"]
        or output_metric["saturation_style"] != contract["saturation_style"]
    ):
        raise PostprocessError(409, "dominant_palette_verification_failed")
    return {
        "segment_index": key[0], "frame_index": key[1],
        "contract": deepcopy(contract), "source": source_metric, "output": output_metric,
    }


def _v4_pack_palette_metrics(
    plan: dict, sources: dict[tuple[int, int], Path], cdir: Path,
    anchor_receipts: list[dict], person_packs: list[dict], scene_packs: list[dict],
) -> dict:
    """Freeze every semantic endpoint's Lab metrics against its own source view."""
    endpoints = []
    for kind, id_key, packs in (
        ("person", "person_id", person_packs), ("scene", "scene_id", scene_packs),
    ):
        for pack in packs:
            identifier = pack[id_key]
            for role in ("primary", "alternate"):
                output = Path(pack[f"{role}_path"]).resolve()
                matches = [
                    receipt for receipt in anchor_receipts
                    if receipt.get("output_sha256") == _sha256_path(output)
                    and _anchor_output_path(cdir, receipt) == output
                ]
                if len(matches) != 1:
                    raise PostprocessError(409, "image_reference_pack_failed")
                anchor = matches[0]["anchor"]
                key = (anchor["segment_index"], anchor["frame_index"])
                endpoints.append({
                    "kind": kind, "id": identifier, "role": role,
                    **_v4_endpoint_palette_metric(plan, sources, key, output),
                })
    payload = {
        "version": 1, "algorithm": _PALETTE_METRIC_ALGORITHM,
        "thresholds": _PALETTE_METRIC_THRESHOLDS, "endpoints": endpoints,
    }
    return {**payload, "sha256": _receipt_sha256(payload)}


def _write_json_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_complete(targets: list[tuple[Path, Path]]) -> bool:
    destination = targets[0][1].parent
    if not destination.is_dir():
        return False
    existing = sorted(path for path in destination.iterdir() if path.is_file())
    expected = sorted(canonical.name for _, canonical in targets)
    sources = {canonical.name: source for source, canonical in targets}
    return (
        [path.name for path in existing] == expected
        and all(_valid_png(path, sources[path.name]) for path in existing)
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _group_targets(cdir: Path, meta: dict) -> dict[int, list[tuple[Path, Path]]]:
    segments = meta.get("segments")
    grouped: dict[int, list[tuple[Path, Path]]] = {}
    if isinstance(segments, list) and segments:
        for segment in segments:
            index = segment.get("index") if isinstance(segment, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or index < 1:
                raise PostprocessError(409, "artifacts not ready")
            src_dir = cdir / "work" / "segments" / str(index) / "work" / "keyframes"
            dst_dir = cdir / "work" / "segments" / str(index) / "work" / "postprocessed"
            files = sorted(path for path in src_dir.glob("*.png") if path.is_file())
            if not files:
                raise PostprocessError(409, "artifacts not ready")
            grouped[index] = [(path, dst_dir / path.name) for path in files]
    else:
        src_dir = cdir / "work" / "keyframes"
        files = sorted(path for path in src_dir.glob("*.png") if path.is_file())
        if not files:
            raise PostprocessError(409, "artifacts not ready")
        grouped[0] = [(path, cdir / "work" / "postprocessed" / path.name) for path in files]
    return grouped


def _v3_plan_audit_inputs(
    meta: dict, private: dict, grouped: dict[int, list[tuple[Path, Path]]],
) -> tuple[dict, dict, list[dict]]:
    """Bind a v3/v4 audit to its stored plan receipt and source frames."""
    try:
        frozen = image_optimization.dual_target_plan_receipt(meta)
        if frozen is None or frozen.get("version") not in {3, 4}:
            raise ValueError
        plan = {key: value for key, value in frozen.items() if key != "sha256"}
        if image_optimization.plan_sha256(plan) != private.get("plan_sha256"):
            raise ValueError
        inventory = []
        segments = []
        for index in sorted(grouped):
            targets = grouped[index]
            if not targets or len({source.parent for source, _ in targets}) != 1:
                raise ValueError
            segments.append({
                "index": index,
                "source_keyframes_dir": targets[0][0].parent,
            })
            for frame_index, (source, _) in enumerate(targets, 1):
                item = {
                    "segment_index": index,
                    "frame_index": frame_index,
                    "frame_name": source.name,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
                if frozen["version"] == 4:
                    execution = private.get("execution_inputs")
                    expected_frame = next((
                        frame for frame in execution.get("frames", [])
                        if isinstance(frame, dict)
                        and frame.get("segment_index") == index
                        and frame.get("frame_index") == frame_index
                    ), None) if isinstance(execution, dict) else None
                    if (
                        not isinstance(expected_frame, dict)
                        or expected_frame.get("frame_name") != source.name
                        or expected_frame.get("source_sha256") != item["source_sha256"]
                    ):
                        raise ValueError
                    item.update(
                        source_transition_from_previous=expected_frame.get(
                            "source_transition_from_previous"
                        ),
                        source_transition_evidence_sha256=expected_frame.get(
                            "source_transition_evidence_sha256"
                        ),
                    )
                inventory.append(item)
        audit_inputs = image_optimization.freeze_plan_audit_inputs(
            plan, frame_inventory=inventory
        )
        expected = sorted(
            [{
                "segment_index": item["segment_index"],
                "frame_index": int(item["frame_name"][:2]),
                "frame_name": item["frame_name"],
                "source_sha256": item["source_sha256"],
                **({
                    "source_transition_from_previous": next(
                        frame["source_transition_from_previous"]
                        for frame in private["execution_inputs"]["frames"]
                        if frame["segment_index"] == item["segment_index"]
                        and frame["frame_index"] == int(item["frame_name"][:2])
                    ),
                    "source_transition_evidence_sha256": next(
                        frame["source_transition_evidence_sha256"]
                        for frame in private["execution_inputs"]["frames"]
                        if frame["segment_index"] == item["segment_index"]
                        and frame["frame_index"] == int(item["frame_name"][:2])
                    ),
                } if frozen["version"] == 4 else {}),
            } for item in private["frames"]],
            key=lambda item: (item["segment_index"], item["frame_index"]),
        )
        if audit_inputs["frames"] != expected:
            raise ValueError
    except (
        OSError,
        ValueError,
        image_optimization.ImageOptimizationIneligibleError,
        image_optimization.ImageOptimizationOutputError,
    ):
        raise PostprocessError(409, "image_plan_audit_failed") from None
    return plan, audit_inputs, segments


def _mark_plan_audit_failed(
    settings: Settings, cid: str, indices: set[int],
) -> None:
    def mutate(_meta: dict, post: dict) -> None:
        segments = [dict(item) for item in post.get("segments", [])]
        for item in segments:
            if item.get("index") in indices and item.get("status") != "done":
                item.update(status="failed", error="image_plan_audit_failed")
        post["segments"] = segments
        post["status"] = "failed"
        post["error"] = "image_plan_audit_failed"
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(settings.data_dir / cid, item["index"])
        )

    _mutate_postprocess(settings, cid, mutate)


def _mark_image_verification_failed(
    settings: Settings, cid: str, indices: set[int], *, error: str = "image_verification_failed",
) -> None:
    public_error = error if error in _PUBLIC_ERROR_CODES else "image_verification_failed"
    def mutate(_meta: dict, post: dict) -> None:
        segments = [dict(item) for item in post.get("segments", [])]
        for item in segments:
            if item.get("index") in indices and item.get("status") != "done":
                item.update(status="failed", error=public_error)
        post["segments"] = segments
        post["status"] = "failed"
        post["error"] = public_error
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(settings.data_dir / cid, item["index"])
        )

    _mutate_postprocess(settings, cid, mutate)


def _segment_state(index: int, total: int, revision: int = 1) -> dict:
    return {
        "index": index, "status": "running", "stage": "queued",
        "completed_frames": 0, "total_frames": total,
        "revision": revision, "error": None,
    }


async def start(settings: Settings, cid: str, payload: dict,
                locks: dict[str, asyncio.Lock]) -> None:
    if set(payload) != {"confirm", "options"}:
        raise PostprocessError(422, "invalid_postprocess_request")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    options = _parse_options(payload)
    _capability_gate(settings, options)
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        cdir = (settings.data_dir / cid).resolve()
        def mutate(meta: dict) -> None:
            if meta.get("schema_version") != 2:
                raise PostprocessError(409, "read_only")
            if meta.get("status") != "done":
                raise PostprocessError(409, "artifacts not ready")
            if isinstance(meta.get("generation"), dict) or meta.get("_input_owner"):
                raise PostprocessError(409, "generation_already_started")
            previous = meta.get("postprocess")
            if isinstance(previous, dict) and previous.get("status") == "running":
                raise PostprocessError(409, "already running")
            if isinstance(previous, dict) and previous.get("status") == "failed":
                raise PostprocessError(409, _structured(
                    "postprocess_segment_retry_required",
                    "后处理存在失败分段，请使用分段重试。",
                ))
            if (
                isinstance(previous, dict) and previous.get("status") == "done"
                and not _options_match(previous.get("options"), options)
            ):
                raise PostprocessError(409, _structured(
                    "postprocess_options_locked", "后处理选项已锁定，请刷新页面后按原选项重试。"
                ))
            grouped = _group_targets(cdir, meta)
            for frames in grouped.values():
                destination = frames[0][1].parent
                if destination.is_dir():
                    if not _canonical_complete(frames) and not (
                        isinstance(previous, dict)
                        and _pure_legacy(previous.get("options"))
                    ):
                        raise PostprocessError(409, "postprocess_canonical_conflict")
            if isinstance(previous, dict) and _pure_legacy(previous.get("options")):
                _clear_canonical(cdir, grouped)
            optimization = image_optimization.receipt(meta, settings)
            if optimization is None:
                raise PostprocessError(409, "image_optimization_prompt_invalid")
            private = (
                {
                    "version": 2, "options": options,
                    "model": optimization["model"],
                    "edit_mode": optimization["edit_mode"],
                    "timeout_s": settings.seedream_timeout_s,
                    "prompts": optimization["segments"],
                }
                if optimization["version"] == 2 else ({
                    "version": 3, "options": options,
                    "model": optimization["model"],
                    "edit_mode": optimization["edit_mode"],
                    "timeout_s": settings.seedream_timeout_s,
                    "plan_sha256": optimization["plan_sha256"],
                    "receipt_sha256": optimization["sha256"],
                    "frames": optimization["frames"],
                } if optimization["version"] == 3 else {
                    "version": 4, "options": options,
                    "model": optimization["model"],
                    "edit_mode": optimization["edit_mode"],
                    "timeout_s": settings.seedream_timeout_s,
                    "plan_sha256": optimization["plan_sha256"],
                    "continuity_sha256": optimization["continuity_sha256"],
                    "execution_input_sha256": optimization[
                        "execution_input_sha256"
                    ],
                    "execution_inputs": optimization["execution_inputs"],
                    "scene_anchor_schedule": optimization[
                        "scene_anchor_schedule"
                    ],
                    "receipt_sha256": optimization["sha256"],
                    "frames": optimization["frames"],
                })
            )
            states = [
                _segment_state(index, len(frames)) for index, frames in grouped.items()
            ]
            reuse_done = (
                isinstance(previous, dict) and previous.get("status") == "done"
                and _options_match(previous.get("options"), options)
                and all(_canonical_complete(frames) for frames in grouped.values())
            )
            if reuse_done and private["version"] == 4 and options["optimize_image"]:
                # Canonical PNGs are not a paid-output receipt.  A v4 same-
                # options request can be a pure reuse only if the immutable
                # project verification, semantic packs, typed anchors and
                # H3 source gate remain completely replayable.
                try:
                    if _private_receipt(meta) != private:
                        raise ValueError
                    generation_keyframes(
                        cdir, meta,
                        [source for index in sorted(grouped) for source, _ in grouped[index]],
                        settings=settings,
                    )
                except (PostprocessError, ValueError):
                    raise PostprocessError(409, "postprocess_artifacts_invalid") from None
            if reuse_done:
                for item in states:
                    item.update(
                        status="done", stage="done",
                        completed_frames=item["total_frames"],
                    )
            else:
                meta.pop("_image_verification", None)
            meta["_image_optimization"] = optimization
            meta["_postprocess_receipt"] = private
            meta["postprocess"] = {
                "status": "done" if reuse_done else "running",
                "options": options,
                "frames": sorted(
                    _frame_ref(index, canonical.name)
                    for index, frames in grouped.items()
                    for _, canonical in frames if reuse_done
                ),
                "segments": states, "error": None,
            }

        if storage.mutate_meta(settings.data_dir, cid, mutate) is None:
            raise PostprocessError(404, "not found")
def _prompt(private: dict, index: int) -> str:
    for item in private.get("prompts", []):
        if item.get("segment_index") == index:
            return item["current"]
    raise PostprocessError(409, "image_optimization_prompt_invalid")


def _frame_prompt(private: dict, index: int, name: str, source_sha256: str) -> str:
    if private.get("version") == 2:
        return _prompt(private, index)
    for item in private.get("frames", []):
        if item.get("segment_index") == index and item.get("frame_name") == name:
            if item.get("source_sha256") != source_sha256:
                break
            return item["current"]
    raise PostprocessError(409, "image_optimization_prompt_invalid")


def _frame_ref(index: int, name: str) -> str:
    return name if index == 0 else f"segments/{index}/work/postprocessed/{name}"


def _private_dir(cdir: Path, index: int) -> Path:
    return cdir / "work" / ".postprocess-private" / str(index)


def _mutate_postprocess(settings: Settings, cid: str, mutator) -> dict | None:
    def mutate(meta: dict) -> None:
        raw = meta.get("postprocess")
        post = dict(raw) if isinstance(raw, dict) else {}
        mutator(meta, post)
        meta["postprocess"] = post

    return storage.mutate_meta(settings.data_dir, cid, mutate)


def _update_segment(settings: Settings, cid: str, index: int, **changes) -> None:
    def mutate(_meta: dict, post: dict) -> None:
        segments = [dict(item) for item in post.get("segments", [])]
        for item in segments:
            if item.get("index") == index:
                item.update(changes)
                break
        post["segments"] = segments
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(settings.data_dir / cid, item["index"])
        )

    _mutate_postprocess(settings, cid, mutate)


def _canonical_files(cdir: Path, index: int) -> list[Path]:
    root = (
        cdir / "work" / "postprocessed" if index == 0
        else cdir / "work" / "segments" / str(index) / "work" / "postprocessed"
    )
    return sorted(path for path in root.glob("*.png") if path.is_file()) if root.is_dir() else []


async def _mediakit_stage(settings: Settings, cdir: Path, index: int,
                          inputs: list[Path], stage: str, scene: str,
                          sem: asyncio.Semaphore) -> list[Path]:
    root = _private_dir(cdir, index) / stage
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / path.name for path in inputs]

    async def one(source: Path, output: Path) -> None:
        if output.is_file():
            # A PNG alone is never evidence that MediaKit did not reach an
            # ambiguous paid state.  Reuse requires the provider's terminal
            # receipt to bind this exact source and scene.
            _v4_mediakit_success(source, output, scene)
            return
        async with sem:
            await mediakit.erase_image(settings, cdir, source, output, True, (scene,))

    results = await asyncio.gather(
        *(one(source, output) for source, output in zip(inputs, outputs)),
        return_exceptions=True,
    )
    errors = [
        f"frame {output.name} failed: {sanitize(str(result))}"
        for output, result in zip(outputs, results) if isinstance(result, BaseException)
    ]
    if errors:
        raise PostprocessError(502, errors[0])
    return outputs


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v4_canvas_stage_path(cdir: Path, index: int, stage: str, name: str) -> Path:
    """Keep v4 derived canvases out of legacy per-segment postprocess paths."""
    return (
        cdir / "work" / ".postprocess-private" / "v4-canvases"
        / f"{index:04d}" / stage / name
    )


def _v4_mediakit_success(source: Path, output: Path, scene: str) -> dict | None:
    """Return one replay-safe MediaKit stage receipt, never infer success from bytes."""
    receipt_path = output.parent / ".mediakit" / f"{output.name}.json"
    if not output.exists() and not receipt_path.exists():
        return None
    if not output.exists():
        # A durable response_received receipt has already crossed the paid
        # boundary.  Hand it back to MediaKit's own GET/download recovery
        # rather than treating it as permission for a second erase POST.
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                isinstance(receipt, dict)
                and receipt.get("version") == mediakit.RECEIPT_VERSION
                and receipt.get("state") == "response_received"
                and receipt.get("scenes") == [scene]
                and isinstance(receipt.get("source"), dict)
                and receipt["source"].get("sha256") == _sha256_path(source)
                and isinstance(receipt.get("stages"), list)
                and len(receipt["stages"]) == 1
                and receipt["stages"][0].get("state") == "response_received"
                and receipt["stages"][0].get("scene") == scene
            ):
                return None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        raise PostprocessError(409, "submission_unknown")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source_sha256 = _sha256_path(source)
        output_sha256 = _sha256_path(output)
        stages = receipt["stages"]
        if (
            not isinstance(receipt, dict)
            or receipt.get("version") != mediakit.RECEIPT_VERSION
            or receipt.get("state") != "succeeded"
            or receipt.get("output") != output.name
            or receipt.get("scenes") != [scene]
            or not isinstance(receipt.get("source"), dict)
            or receipt["source"].get("sha256") != source_sha256
            or not isinstance(stages, list)
            or len(stages) != 1
            or stages[0].get("scene") != scene
            or stages[0].get("state") != "succeeded"
            or not _valid_png(output, source)
        ):
            raise ValueError
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        # A preexisting artifact without an exact terminal receipt may be a
        # paid request in its crash window.  It is strictly GET-only.
        raise PostprocessError(409, "submission_unknown") from None
    return {
        "scene": scene,
        "input_sha256": source_sha256,
        "output_sha256": output_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_path(receipt_path),
    }


def _v4_canvas_record(
    cdir: Path, index: int, frame_index: int, source: Path, canvas: Path,
    stages: list[dict],
) -> dict:
    try:
        canvas_path = str(canvas.resolve().relative_to(cdir.resolve()))
    except ValueError:
        raise PostprocessError(409, "postprocess_receipt_invalid") from None
    return {
        "segment_index": index,
        "frame_index": frame_index,
        "frame_name": source.name,
        "source_sha256": _sha256_path(source),
        "canvas_path": canvas_path,
        "canvas_sha256": _sha256_path(canvas),
        "stages": stages,
    }


def _v4_derive_canvas_optimization(
    settings: Settings, meta: dict, private: dict, records: list[dict],
) -> dict:
    """Re-freeze the same v4 plan against the deterministic derived canvases."""
    plan = _v4_frozen_plan(meta, private)
    record_by_key = {
        (item["segment_index"], item["frame_index"]): item for item in records
    }
    inventory = []
    for frame in private["execution_inputs"]["frames"]:
        key = (frame["segment_index"], frame["frame_index"])
        record = record_by_key.get(key)
        if record is None or record["frame_name"] != frame["frame_name"]:
            raise PostprocessError(409, "postprocess_receipt_invalid")
        inventory.append({
            "segment_index": frame["segment_index"],
            "frame_index": frame["frame_index"],
            "frame_name": frame["frame_name"],
            "source_sha256": record["canvas_sha256"],
            "source_transition_from_previous": frame["source_transition_from_previous"],
            "source_transition_evidence_sha256": frame[
                "source_transition_evidence_sha256"
            ],
        })
    try:
        execution = image_optimization.freeze_execution_inputs(
            plan,
            revision=private["execution_inputs"]["revision"],
            profile=private["execution_inputs"]["profile"],
            model=private["model"],
            frame_inventory=inventory,
        )
        task_settings = replace(
            settings, seedream_model=private["model"],
            seedream_edit_mode=private["edit_mode"],
        )
        return image_optimization.freeze_frame_prompts(
            task_settings,
            execution,
            image_optimization.compile_frame_prompts(plan, private["edit_mode"]),
            plan=plan,
        )["_image_optimization"]
    except (
        KeyError, TypeError, ValueError, image_optimization.ImageOptimizationOutputError,
        image_optimization.ImageOptimizationIneligibleError,
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid") from None


def _v4_effective_private(private: dict, derived: dict, canvas_sha256: str) -> dict:
    required = {
        "version", "plan_sha256", "continuity_sha256", "execution_input_sha256",
        "execution_inputs", "model", "edit_mode", "scene_anchor_schedule", "frames", "sha256",
    }
    payload = {key: value for key, value in derived.items() if key != "sha256"}
    if (
        not isinstance(derived, dict)
        or set(derived) != required
        or derived.get("version") != 4
        or derived.get("plan_sha256") != private.get("plan_sha256")
        or derived.get("model") != private.get("model")
        or derived.get("edit_mode") != private.get("edit_mode")
        or derived.get("sha256") != _receipt_sha256(payload)
    ):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    result = deepcopy(private)
    for key in (
        "continuity_sha256", "execution_input_sha256", "execution_inputs",
        "scene_anchor_schedule", "frames",
    ):
        result[key] = deepcopy(derived[key])
    result["canvas_execution_sha256"] = canvas_sha256
    return result


async def _v4_prepare_canvases(
    settings: Settings, cdir: Path, cid: str, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], sem: asyncio.Semaphore,
) -> tuple[dict, dict[int, list[tuple[Path, Path]]]]:
    """Complete and freeze all requested MediaKit stages before v4 audit/POSTs."""
    if not (private["options"]["remove_subtitle"] or private["options"]["remove_brand"]):
        return private, grouped
    # This verifies the raw frozen source inventory before any MediaKit reuse
    # or submission.  A partial segment retry cannot turn a project DAG into
    # an independent image edit.
    raw_sources = _v4_frame_sources(grouped, private)
    stages = [
        ("text", mediakit.TEXT_SCENE, private["options"]["remove_subtitle"]),
        ("brand", mediakit.ICON_SCENE, private["options"]["remove_brand"]),
    ]
    existing = meta.get("_v4_canvas_execution")
    expected_records: list[dict] = []
    current = dict(raw_sources)
    for stage, scene, enabled in stages:
        if not enabled:
            continue
        next_current: dict[tuple[int, int], Path] = {}
        for key in sorted(current):
            index, frame_index = key
            source = current[key]
            output = _v4_canvas_stage_path(cdir, index, stage, source.name)
            stage_receipt = _v4_mediakit_success(source, output, scene)
            if stage_receipt is None:
                async with sem:
                    try:
                        await mediakit.erase_image(
                            settings, cdir, source, output, True, (scene,)
                        )
                    except mediakit.MediaKitError as exc:
                        raise PostprocessError(exc.status, exc.detail) from None
                stage_receipt = _v4_mediakit_success(source, output, scene)
                if stage_receipt is None:
                    raise PostprocessError(409, "submission_unknown")
            next_current[key] = output
            # Build records only after all stages are terminally receipt-bound.
        current = next_current
    for key in sorted(raw_sources):
        index, frame_index = key
        source = raw_sources[key]
        canvas = current[key]
        chain = []
        stage_source = source
        for stage, scene, enabled in stages:
            if not enabled:
                continue
            output = _v4_canvas_stage_path(cdir, index, stage, source.name)
            receipt = _v4_mediakit_success(stage_source, output, scene)
            if receipt is None:
                raise PostprocessError(409, "submission_unknown")
            try:
                receipt["receipt_path"] = str(
                    Path(receipt["receipt_path"]).resolve().relative_to(cdir.resolve())
                )
            except ValueError:
                raise PostprocessError(409, "postprocess_receipt_invalid") from None
            chain.append(receipt)
            stage_source = output
        expected_records.append(_v4_canvas_record(
            cdir, index, frame_index, source, canvas, chain,
        ))
    derived = _v4_derive_canvas_optimization(settings, meta, private, expected_records)
    payload = {
        "version": 1,
        "postprocess_receipt_sha256": private["receipt_sha256"],
        "options": deepcopy(private["options"]),
        "frames": expected_records,
        "derived_optimization": derived,
    }
    receipt = {**payload, "sha256": _receipt_sha256(payload)}
    if existing is not None and existing != receipt:
        # Existing receipts are immutable.  Drift is never repaired by a new
        # MediaKit submission, because that would change paid input evidence.
        raise PostprocessError(409, "postprocess_receipt_invalid")
    if existing is None:
        storage.update_meta(settings.data_dir, cid, _v4_canvas_execution=receipt)
    effective = _v4_effective_private(private, derived, receipt["sha256"])
    effective_grouped = {
        index: [
            (
                (cdir / next(
                    item["canvas_path"] for item in expected_records
                    if item["segment_index"] == index and item["frame_name"] == source.name
                )).resolve(),
                target,
            )
            for source, target in targets
        ]
        for index, targets in grouped.items()
    }
    return effective, effective_grouped


async def _seedream_stage(settings: Settings, cdir: Path, cid: str, index: int,
                          inputs: list[Path], private: dict,
                          source_sha256s: dict[str, str],
                          sem: asyncio.Semaphore) -> list[Path]:
    root = _private_dir(cdir, index) / "seedream"
    attempts = _private_dir(cdir, index) / "attempts"
    root.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
    outputs = [root / path.name for path in inputs]
    latest = storage.load_meta(settings.data_dir, cid) or {}
    revision = next(
        (item.get("revision", 1) for item in (latest.get("postprocess") or {}).get("segments", [])
         if item.get("index") == index),
        1,
    )
    task_settings = replace(
        settings, seedream_model=private["model"], seedream_edit_mode=private["edit_mode"],
        seedream_timeout_s=private["timeout_s"],
    )

    async def call(position: int, image_inputs: list[Path], output: Path) -> None:
        if output.is_file():
            return
        source = image_inputs[0]
        source_sha256 = source_sha256s.get(source.name)
        if source_sha256 is None:
            raise PostprocessError(409, "image_optimization_prompt_invalid")
        prompt = _frame_prompt(private, index, source.name, source_sha256)
        async with sem:
            await seedream.edit(
                task_settings, [path.read_bytes() for path in image_inputs], prompt, output,
                receipt_path=attempts / f"{position:04d}-r{revision}.json",
            )

    if private["edit_mode"] == "independent_parallel":
        results = await asyncio.gather(
            *(call(i, [source], output) for i, (source, output) in enumerate(zip(inputs, outputs), 1)),
            return_exceptions=True,
        )
    else:
        try:
            await call(1, inputs, outputs[0])
            results = await asyncio.gather(
                *(call(i, [source, outputs[0]], output)
                  for i, (source, output) in enumerate(zip(inputs[1:], outputs[1:]), 2)),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            results = [exc]
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        error = errors[0]
        if isinstance(error, asyncio.CancelledError):
            raise error
        if isinstance(error, seedream.SeedreamError):
            raise PostprocessError(502, error.code)
        raise PostprocessError(502, sanitize(str(error)))
    return outputs


def _v4_frozen_plan(meta: dict, private: dict) -> dict:
    try:
        frozen = image_optimization.dual_target_plan_receipt(meta)
        if (
            frozen is None
            or frozen.get("version") != 4
            or private.get("continuity_sha256")
            != private.get("execution_inputs", {}).get("continuity_sha256")
        ):
            raise ValueError
        plan = {key: value for key, value in frozen.items() if key != "sha256"}
        if image_optimization.plan_sha256(plan) != private.get("plan_sha256"):
            raise ValueError
    except (ValueError, image_optimization.ImageOptimizationOutputError):
        raise PostprocessError(409, "postprocess_receipt_invalid") from None
    return plan


def _v4_frame_sources(
    grouped: dict[int, list[tuple[Path, Path]]], private: dict,
) -> dict[tuple[int, int], Path]:
    sources = {}
    frozen = {
        (item.get("segment_index"), item.get("frame_name")): item
        for item in private.get("frames", []) if isinstance(item, dict)
    }
    for index, targets in grouped.items():
        for frame_index, (source, _canonical) in enumerate(targets, 1):
            expected = frozen.get((index, source.name))
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            if (
                not isinstance(expected, dict)
                or expected.get("source_sha256") != source_sha256
                or source.name != f"{frame_index:02d}.png"
            ):
                raise PostprocessError(409, "postprocess_receipt_invalid")
            sources[(index, frame_index)] = source
    if len(sources) != len(frozen):
        raise PostprocessError(409, "postprocess_receipt_invalid")
    return sources


def _anchor_receipt_path(cdir: Path, scene_id: str, label: str) -> Path:
    return cdir / "work" / ".postprocess-private" / "scene-anchors" / scene_id / (
        f"{label}.json"
    )


def _anchor_receipt_payload(
    *,
    private: dict,
    scene_id: str,
    label: str,
    anchor: dict,
    input_roles: list[str],
    inputs: list[Path],
    output: Path,
) -> dict:
    if (
        not output.is_file()
        or len(input_roles) != len(inputs)
        or not input_roles
        or input_roles[0] != "canvas"
    ):
        raise PostprocessError(409, "scene_anchor_verification_failed")
    payload = {
        "version": 1,
        "plan_sha256": private["plan_sha256"],
        "continuity_sha256": private["continuity_sha256"],
        "scene_id": scene_id,
        "label": label,
        "anchor": anchor,
        "input_roles": input_roles,
        "input_sha256s": [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs],
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    return {**payload, "sha256": hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()}


def _write_anchor_receipt(path: Path, receipt: dict) -> None:
    _write_json_receipt(path, receipt)


def _load_anchor_receipt(
    path: Path, *, private: dict, scene_id: str, label: str,
    anchor: dict, input_roles: list[str], inputs: list[Path], output: Path,
) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = _anchor_receipt_payload(
            private=private,
            scene_id=scene_id,
            label=label,
            anchor=anchor,
            input_roles=input_roles,
            inputs=inputs,
            output=output,
        )
    except (OSError, ValueError, PostprocessError):
        return None
    return raw if raw == expected else None


def _semantic_receipt_path(cdir: Path, label: str) -> Path:
    return cdir / "work" / ".postprocess-private" / "scene-anchors" / "semantic" / (
        f"{label}.json"
    )


def _semantic_receipt(
    *, cdir: Path, private: dict, label: str, verdict: dict, metrics: dict,
    anchor_receipts: list[dict], person_packs: list[dict], scene_packs: list[dict],
) -> dict:
    if verdict.get("passed") is not True:
        raise PostprocessError(409, "image_reference_pack_failed")
    metric_payload = {key: value for key, value in metrics.items() if key != "sha256"}
    if metrics.get("sha256") != _receipt_sha256(metric_payload):
        raise PostprocessError(409, "dominant_palette_metric_invalid")
    try:
        metric_by_endpoint = {
            (item["kind"], item["id"], item["role"]): item
            for item in metrics["endpoints"]
        }
        if len(metric_by_endpoint) != len(metrics["endpoints"]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise PostprocessError(409, "dominant_palette_metric_invalid") from None

    def binding(kind: str, identifier: str, pack: dict) -> dict:
        def output_binding(value: object, role: str) -> dict:
            try:
                path = Path(value)
                output_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, TypeError, ValueError):
                raise PostprocessError(409, "image_reference_pack_failed") from None
            matches = [
                item for item in anchor_receipts
                if (
                    item.get("output_sha256") == output_sha256
                    and _anchor_output_path(cdir, item) == path.resolve()
                )
            ]
            if len(matches) != 1:
                raise PostprocessError(409, "image_reference_pack_failed")
            match = matches[0]
            metric = metric_by_endpoint.get((kind, identifier, role))
            if not isinstance(metric, dict):
                raise PostprocessError(409, "dominant_palette_metric_invalid")
            # Keep a canonical snapshot of the exact typed input receipt in
            # every semantic binding.  A receipt digest alone cannot prove
            # that primary and alternate roles have not been interchanged.
            return {
                "scene_id": match["scene_id"],
                "label": match["label"],
                "anchor": deepcopy(match["anchor"]),
                "input_roles": deepcopy(match["input_roles"]),
                "input_sha256s": deepcopy(match["input_sha256s"]),
                "anchor_receipt_sha256": match["sha256"],
                "output_sha256": output_sha256,
                "palette_metric": deepcopy(metric),
            }
        try:
            source_sha256 = hashlib.sha256(Path(pack["source_path"]).read_bytes()).hexdigest()
            primary = output_binding(pack["primary_path"], "primary")
            alternate = output_binding(pack["alternate_path"], "alternate")
        except KeyError:
            raise PostprocessError(409, "image_reference_pack_failed") from None
        if primary == alternate:
            raise PostprocessError(409, "image_reference_pack_failed")
        return {
            "kind": kind,
            "id": identifier,
            "source_sha256": source_sha256,
            "primary": primary,
            "alternate": alternate,
        }

    pack_bindings = [
        binding("person", item["person_id"], item) for item in person_packs
    ] + [
        binding("scene", item["scene_id"], item) for item in scene_packs
    ]
    endpoint_receipts = [
        side["anchor_receipt_sha256"]
        for item in pack_bindings for side in (item["primary"], item["alternate"])
    ]
    if len(endpoint_receipts) != len(set(endpoint_receipts)):
        raise PostprocessError(409, "image_reference_pack_failed")
    payload = {
        "version": 1,
        "plan_sha256": private["plan_sha256"],
        "continuity_sha256": private["continuity_sha256"],
        "label": label,
        "pack_bindings": pack_bindings,
        "metrics_sha256": metrics["sha256"],
        "verdict": verdict,
    }
    return {**payload, "sha256": _receipt_sha256(payload)}


def _anchor_output_path(cdir: Path, receipt: dict) -> Path:
    label = receipt["label"]
    if (
        label in {"global", "pack-alternate"}
        or label.startswith("person-")
    ):
        return _anchor_receipt_path(cdir, receipt["scene_id"], label).with_suffix(
            ".png"
        ).resolve()
    anchor = receipt["anchor"]
    return (
        _private_dir(cdir, anchor["segment_index"])
        / "seedream" / anchor["frame_name"]
    ).resolve()


def _append_anchor_receipt(receipts: list[dict], receipt: dict) -> None:
    if not any(item.get("sha256") == receipt.get("sha256") for item in receipts):
        receipts.append(receipt)


def _load_json_receipt(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: item for key, item in value.items() if key != "sha256"}
        if not isinstance(value, dict) or value.get("sha256") != _receipt_sha256(payload):
            return None
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return value


def _v4_person_alternate_key(plan: dict, person: dict) -> tuple[int, int]:
    reference = person["reference"]
    primary = (reference["segment_index"], reference["frame_index"])
    candidates = sorted(
        (segment["segment_index"], frame_index)
        for segment in plan["segments"]
        for instance in segment["persons"]
        if instance["id"] == person["id"] and instance["state"] == "replace"
        for frame_index in instance["observable_frames"]
        if (segment["segment_index"], frame_index) != primary
    )
    if not candidates:
        raise ValueError
    return candidates[0]


def _v4_expected_anchor_descriptors(plan: dict, schedule: dict) -> list[dict]:
    """Derive the whole v4 DAG from frozen plan/schedule, never from files."""
    nodes = schedule.get("nodes") if isinstance(schedule, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise ValueError
    descriptors = [
        {
            "scene_id": item["scene_id"], "label": item["label"],
            "anchor": item["anchor"],
        }
        for item in nodes
        if isinstance(item, dict)
    ]
    if (
        len(descriptors) != len(nodes)
        or [item["anchor"].get("order") for item in descriptors]
        != list(range(1, len(descriptors) + 1))
        or len({(item["scene_id"], item["label"]) for item in descriptors})
        != len(descriptors)
    ):
        raise ValueError
    return descriptors


def _v4_semantic_labels(schedule: dict) -> list[str]:
    try:
        return ["bootstrap", *[
            f"layout-{layout['segment_index']:04d}"
            for scene in schedule["scenes"]
            for layout in scene["segment_layout_anchors"]
        ]]
    except (KeyError, TypeError):
        raise ValueError from None


def _v4_anchor_manifest(
    plan: dict, schedule: dict, receipts: list[dict],
) -> list[dict]:
    """Return the only legal ordered, complete anchor receipt manifest."""
    descriptors = _v4_expected_anchor_descriptors(plan, schedule)
    if not isinstance(receipts, list) or len(receipts) != len(descriptors):
        raise ValueError
    manifest = []
    for descriptor, receipt in zip(descriptors, receipts):
        if not isinstance(receipt, dict) or (
            receipt.get("scene_id"), receipt.get("label")
        ) != (descriptor["scene_id"], descriptor["label"]):
            raise ValueError
        sha = receipt.get("sha256")
        if not isinstance(sha, str):
            raise ValueError
        manifest.append({
            "scene_id": descriptor["scene_id"],
            "label": descriptor["label"],
            "sha256": sha,
        })
    if len({item["sha256"] for item in manifest}) != len(manifest):
        raise ValueError
    return manifest


def _v4_scheduled_anchor(schedule: dict, scene_id: str, label: str) -> dict:
    matches = [
        item["anchor"] for item in _v4_expected_anchor_descriptors({}, schedule)
        if item["scene_id"] == scene_id and item["label"] == label
    ]
    if len(matches) != 1:
        raise ValueError
    return deepcopy(matches[0])


def _v4_anchor_receipt_index(cdir: Path, plan: dict, schedule: dict) -> dict[str, dict]:
    """Load exactly the frozen DAG's receipts; PERSON labels never fan out by scene."""
    result = {}
    for descriptor in _v4_expected_anchor_descriptors(plan, schedule):
        receipt = _load_json_receipt(_anchor_receipt_path(
            cdir, descriptor["scene_id"], descriptor["label"]
        ))
        if receipt is None:
            raise ValueError
        sha = receipt.get("sha256")
        if (
            not isinstance(sha, str)
            or sha in result
            or receipt.get("scene_id") != descriptor["scene_id"]
            or receipt.get("label") != descriptor["label"]
            or receipt.get("anchor") != descriptor["anchor"]
        ):
            raise ValueError
        result[sha] = receipt
    return result


def _valid_v4_anchor_receipt(
    cdir: Path, receipt: dict, *, plan_sha256: str, continuity_sha256: str,
    source_sha256s: dict[tuple[int, int], str],
    source_paths: dict[tuple[int, int], Path], anchors: dict[str, dict],
) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != {
        "version", "plan_sha256", "continuity_sha256", "scene_id", "label",
        "anchor", "input_roles", "input_sha256s", "output_sha256", "sha256",
    } or receipt.get("version") != 1 or receipt.get("plan_sha256") != plan_sha256 \
            or receipt.get("continuity_sha256") != continuity_sha256:
        return False
    try:
        anchor = receipt["anchor"]
        key = (anchor["segment_index"], anchor["frame_index"])
        source_sha256 = source_sha256s[key]
        if (
            set(anchor) != {
                "order", "segment_index", "frame_index", "frame_name", "source_sha256",
            }
            or anchor["source_sha256"] != source_sha256
            or not isinstance(receipt["input_roles"], list)
            or not isinstance(receipt["input_sha256s"], list)
            or len(receipt["input_roles"]) != len(receipt["input_sha256s"])
            or receipt["input_roles"][:1] != ["canvas"]
            or receipt["input_sha256s"][:1] != [source_sha256]
            or hashlib.sha256(_anchor_output_path(cdir, receipt).read_bytes()).hexdigest()
            != receipt["output_sha256"]
            or not _valid_png(_anchor_output_path(cdir, receipt), source_paths[key])
        ):
            return False
        roles = receipt["input_roles"]
        if receipt["label"] == "global":
            return roles == ["canvas"]
        if receipt["label"].startswith("fanout-"):
            layout_sha = next(
                item["output_sha256"] for item in anchors.values()
                if item["label"] == f"layout-{anchor['segment_index']:04d}"
                and item["scene_id"] == receipt["scene_id"]
            )
            return roles == ["canvas", "segment_layout_anchor"] and receipt[
                "input_sha256s"
            ][1:] == [layout_sha]
        global_sha = next(
            item["output_sha256"] for item in anchors.values()
            if item["label"] == "global" and item["scene_id"] == receipt["scene_id"]
        )
        return roles == ["canvas", "global_scene_anchor"] and receipt[
            "input_sha256s"
        ][1:] == [global_sha]
    except (KeyError, OSError, StopIteration, TypeError, ValueError):
        return False


def _valid_semantic_pack_bindings(
    cdir: Path, plan: dict, schedule: dict, receipt: dict,
    source_sha256s: dict[tuple[int, int], str],
    source_paths: dict[tuple[int, int], Path],
) -> bool:
    if set(receipt) != {
        "version", "plan_sha256", "continuity_sha256", "label", "pack_bindings",
        "metrics_sha256", "verdict", "sha256",
    }:
        return False
    bindings = receipt.get("pack_bindings")
    expected = [
        ("person", item["id"], item["reference"])
        for item in plan["person_plans"]
    ] + [
        ("scene", item["id"], item["reference"])
        for item in plan["scene_plans"]
    ]
    if not isinstance(bindings, list) or len(bindings) != len(expected):
        return False
    try:
        anchors = _v4_anchor_receipt_index(cdir, plan, schedule)
        scene_by_segment = {
            segment["segment_index"]: segment["scene"]["scene_id"]
            for segment in plan["segments"]
        }
        schedule_by_scene = {
            item["scene_id"]: item for item in schedule["scenes"]
        }
        endpoint_receipts = []
        endpoint_metrics = []
        for binding, (kind, identifier, reference) in zip(bindings, expected):
            if not isinstance(binding, dict) or set(binding) != {
                "kind", "id", "source_sha256", "primary", "alternate",
            } or binding["kind"] != kind or binding["id"] != identifier:
                return False
            key = (reference["segment_index"], reference["frame_index"])
            if binding["source_sha256"] != source_sha256s.get(key):
                return False
            for role in ("primary", "alternate"):
                side = binding[role]
                if not isinstance(side, dict) or set(side) != {
                    "scene_id", "label", "anchor", "input_roles", "input_sha256s",
                    "anchor_receipt_sha256", "output_sha256", "palette_metric",
                }:
                    return False
                anchor = anchors.get(side["anchor_receipt_sha256"])
                if (
                    anchor is None
                    or any(anchor.get(name) != side[name] for name in (
                        "scene_id", "label", "anchor", "input_roles", "input_sha256s",
                        "output_sha256",
                    ))
                    or not _valid_v4_anchor_receipt(
                        cdir, anchor,
                        plan_sha256=receipt["plan_sha256"],
                        continuity_sha256=receipt["continuity_sha256"],
                        source_sha256s=source_sha256s,
                        source_paths=source_paths,
                        anchors=anchors,
                    )
                ):
                    return False
                key = (anchor["anchor"]["segment_index"], anchor["anchor"]["frame_index"])
                metric = {
                    "kind": kind, "id": identifier, "role": role,
                    **_v4_endpoint_palette_metric(
                        plan, source_paths, key, _anchor_output_path(cdir, anchor),
                    ),
                }
                if side["palette_metric"] != metric:
                    return False
                endpoint_metrics.append(metric)
                endpoint_receipts.append(side["anchor_receipt_sha256"])
            primary = binding["primary"]
            alternate = binding["alternate"]
            if primary == alternate:
                return False
            if kind == "person":
                alternate_key = _v4_person_alternate_key(
                    plan, next(item for item in plan["person_plans"] if item["id"] == identifier)
                )
                primary_key = (reference["segment_index"], reference["frame_index"])
                if (
                    primary["label"] != f"person-{identifier}-primary"
                    or alternate["label"] != f"person-{identifier}-alternate"
                    or (primary["anchor"]["segment_index"], primary["anchor"]["frame_index"])
                    != primary_key
                    or (alternate["anchor"]["segment_index"], alternate["anchor"]["frame_index"])
                    != alternate_key
                    or primary["scene_id"] != scene_by_segment[primary_key[0]]
                    or alternate["scene_id"] != scene_by_segment[alternate_key[0]]
                ):
                    return False
            else:
                scene_schedule = schedule_by_scene[identifier]
                global_anchor = scene_schedule["global_anchor"]
                global_key = (
                    global_anchor["segment_index"], global_anchor["frame_index"]
                )
                alternate_key = (
                    alternate["anchor"]["segment_index"],
                    alternate["anchor"]["frame_index"],
                )
                if (
                    primary["scene_id"] != identifier
                    or alternate["scene_id"] != identifier
                    or primary["label"] != "global"
                    or (primary["anchor"]["segment_index"], primary["anchor"]["frame_index"])
                    != global_key
                    or primary["anchor_receipt_sha256"] == alternate["anchor_receipt_sha256"]
                    or not (
                        alternate["label"] == "pack-alternate"
                        or alternate["label"].startswith("layout-")
                    )
                ):
                    return False
        metrics_payload = {
            "version": 1, "algorithm": _PALETTE_METRIC_ALGORITHM,
            "thresholds": _PALETTE_METRIC_THRESHOLDS,
            "endpoints": endpoint_metrics,
        }
        metrics = {**metrics_payload, "sha256": _receipt_sha256(metrics_payload)}
        return (
            len(endpoint_receipts) == len(set(endpoint_receipts))
            and receipt.get("metrics_sha256") == metrics["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _valid_palette_metrics_for_outputs(
    metrics: object,
    original_output_pairs: list[tuple[int, Path, Path]],
) -> bool:
    if not isinstance(metrics, dict):
        return False
    payload = {key: value for key, value in metrics.items() if key != "sha256"}
    if (
        set(metrics) != {"version", "algorithm", "thresholds", "frames", "sha256"}
        or metrics.get("version") != 1
        or metrics.get("algorithm") != _PALETTE_METRIC_ALGORITHM
        or metrics.get("thresholds") != _PALETTE_METRIC_THRESHOLDS
        or metrics.get("sha256") != _receipt_sha256(payload)
        or not isinstance(metrics.get("frames"), list)
    ):
        return False
    expected = {(index, source.name): (source, output) for index, source, output in original_output_pairs}
    seen = set()
    for item in metrics["frames"]:
        if not isinstance(item, dict):
            return False
        key = (item.get("segment_index"), item.get("frame_index"))
        # frame indices are decoded from the locked 01.png-style names below.
        if not isinstance(key[0], int) or not isinstance(key[1], int):
            return False
        name = f"{key[1]:02d}.png"
        pair = expected.get((key[0], name))
        if pair is None or key in seen:
            return False
        source, output = pair
        if set(item) != {"segment_index", "frame_index", "contract", "source", "output"}:
            return False
        if item["source"] != _area_weighted_palette_metric(source):
            return False
        if item["output"] != _area_weighted_palette_metric(output):
            return False
        contract = item.get("contract")
        if not isinstance(contract, dict) or set(contract) != {
            "area_weighted_warm_cool_family", "saturation_style",
        }:
            return False
        if (
            item["source"]["warm_cool_family"] != contract["area_weighted_warm_cool_family"]
            or item["source"]["saturation_style"] != contract["saturation_style"]
            or item["output"]["warm_cool_family"] != contract["area_weighted_warm_cool_family"]
            or item["output"]["saturation_style"] != contract["saturation_style"]
        ):
            return False
        seen.add(key)
    return len(seen) == len(expected)


async def _v4_anchor(
    settings: Settings, cdir: Path, cid: str, private: dict,
    seedream_sem: asyncio.Semaphore, *, scene_id: str, label: str,
    anchor: dict, canvas: Path, references: list[tuple[str, Path]], output: Path,
) -> tuple[Path, dict]:
    """Generate one typed anchor; no dependent may use an unverified result."""
    input_roles = ["canvas", *[role for role, _ in references]]
    inputs = [canvas, *[path for _, path in references]]
    receipt_path = _anchor_receipt_path(cdir, scene_id, label)
    existing = _load_anchor_receipt(
        receipt_path,
        private=private,
        scene_id=scene_id,
        label=label,
        anchor=anchor,
        input_roles=input_roles,
        inputs=inputs,
        output=output,
    )
    if existing is not None and _valid_png(output, canvas):
        return output, existing
    attempts = receipt_path.parent / "attempts"
    latest = storage.load_meta(settings.data_dir, cid) or {}
    revisions = {
        item.get("revision")
        for item in (latest.get("postprocess") or {}).get("segments", [])
        if isinstance(item, dict)
    }
    if len(revisions) != 1 or next(iter(revisions), 0) is None:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    revision = next(iter(revisions))
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise PostprocessError(409, "postprocess_receipt_invalid")
    attempt_path = attempts / f"{anchor['order']:04d}-{label}-r{revision}.json"
    task_settings = replace(
        settings,
        seedream_model=private["model"],
        seedream_edit_mode=private["edit_mode"],
        seedream_timeout_s=private["timeout_s"],
    )
    prompt = _frame_prompt(
        private,
        anchor["segment_index"],
        anchor["frame_name"],
        anchor["source_sha256"],
    )
    # The provider may have atomically persisted its exact succeeded attempt
    # and output just before the business anchor receipt is written.  Replay
    # that local attempt only; an absent/ambiguous attempt remains GET-only.
    if output.exists() or receipt_path.exists() or attempt_path.exists():
        if not attempt_path.is_file():
            raise PostprocessError(409, "submission_unknown")
        try:
            await seedream.edit(
                task_settings, [path.read_bytes() for path in inputs], prompt, output,
                receipt_path=attempt_path,
            )
        except seedream.SeedreamError as exc:
            raise PostprocessError(502, exc.code) from None
        if not _valid_png(output, canvas):
            raise PostprocessError(409, "scene_anchor_verification_failed")
        receipt = _anchor_receipt_payload(
            private=private, scene_id=scene_id, label=label, anchor=anchor,
            input_roles=input_roles, inputs=inputs, output=output,
        )
        _write_anchor_receipt(receipt_path, receipt)
        return output, receipt
    if receipt_path.exists():
        raise PostprocessError(409, "submission_unknown")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
    try:
        async with seedream_sem:
            await seedream.edit(
                task_settings,
                [path.read_bytes() for path in inputs],
                prompt,
                output,
                receipt_path=attempt_path,
            )
    except seedream.SeedreamError as exc:
        raise PostprocessError(502, exc.code) from None
    except OSError as exc:
        raise PostprocessError(502, sanitize(str(exc))) from None
    if not _valid_png(output, canvas):
        raise PostprocessError(409, "scene_anchor_verification_failed")
    receipt = _anchor_receipt_payload(
        private=private,
        scene_id=scene_id,
        label=label,
        anchor=anchor,
        input_roles=input_roles,
        inputs=inputs,
        output=output,
    )
    _write_anchor_receipt(receipt_path, receipt)
    return output, receipt


def _v4_visible_scene_candidate(
    private: dict, scene_id: str, excluded: tuple[int, int],
) -> dict | None:
    candidates = []
    for frame in private["execution_inputs"]["frames"]:
        if frame["scene_id"] != scene_id:
            continue
        if (frame["segment_index"], frame["frame_index"]) == excluded:
            continue
        observations = frame["scene_continuity_view"]["observations"]
        if any(item["visibility"] in {"full", "partial", "edge_fragment"}
               for item in observations):
            candidates.append(frame)
    return min(
        candidates,
        key=lambda item: (item["segment_index"], item["frame_index"]),
        default=None,
    )


def _v4_preflight(
    plan: dict, private: dict, sources: dict[tuple[int, int], Path],
) -> None:
    """Reject only malformed or source-drifted paid nodes before generation."""
    frames = {
        (frame["segment_index"], frame["frame_index"]): frame
        for frame in private["execution_inputs"]["frames"]
    }

    def source_for(key: tuple[int, int], error: str) -> None:
        frame = frames.get(key)
        source = sources.get(key)
        try:
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
        except (AttributeError, OSError):
            raise PostprocessError(409, error) from None
        if frame is None or frame.get("source_sha256") != actual:
            raise PostprocessError(409, error)

    # Every remaining node is part of the generation DAG.  Continuity views,
    # target semantics and palette contracts stay frozen in prompts, but are
    # never reinterpreted here as content eligibility.
    try:
        descriptors = _v4_expected_anchor_descriptors(
            plan, private["scene_anchor_schedule"],
        )
    except ValueError:
        raise PostprocessError(409, "scene_anchor_preflight_failed") from None
    for descriptor in descriptors:
        anchor = descriptor["anchor"]
        key = (anchor["segment_index"], anchor["frame_index"])
        source_for(key, "scene_anchor_preflight_failed")
        if frames[key].get("source_sha256") != anchor.get("source_sha256"):
            raise PostprocessError(409, "scene_anchor_preflight_failed")


async def _v4_bootstrap_scene_anchors(
    settings: Settings, cdir: Path, cid: str, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
) -> tuple[dict[tuple[int, int], Path], list[dict]]:
    """Build the global roots used by layout and frame generation."""
    sources = _v4_frame_sources(grouped, private)
    outputs: dict[tuple[int, int], Path] = {}
    receipts: list[dict] = []
    for scene in private["scene_anchor_schedule"]["scenes"]:
        scene_id = scene["scene_id"]
        global_anchor = scene["global_anchor"]
        global_source = sources[(
            global_anchor["segment_index"], global_anchor["frame_index"]
        )]
        global_output, receipt = await _v4_anchor(
            settings, cdir, cid, private, seedream_sem,
            scene_id=scene_id,
            label="global",
            anchor=global_anchor,
            canvas=global_source,
            references=[],
            output=_anchor_receipt_path(cdir, scene_id, "global").with_suffix(".png"),
        )
        receipts.append(receipt)
    return outputs, receipts


async def _v4_generate_layout_anchors(
    settings: Settings, cdir: Path, cid: str, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
    bootstrap_outputs: dict[tuple[int, int], Path], anchor_receipts: list[dict],
) -> None:
    sources = _v4_frame_sources(grouped, private)
    for scene in private["scene_anchor_schedule"]["scenes"]:
        scene_id = scene["scene_id"]
        global_output = _anchor_receipt_path(cdir, scene_id, "global").with_suffix(".png")
        for layout in scene["segment_layout_anchors"]:
            key = (layout["segment_index"], layout["frame_index"])
            output, receipt = await _v4_anchor(
                settings, cdir, cid, private, seedream_sem,
                scene_id=scene_id,
                label=f"layout-{layout['segment_index']:04d}",
                anchor=layout,
                canvas=sources[key],
                references=[("global_scene_anchor", global_output)],
                output=_private_dir(cdir, layout["segment_index"])
                / "seedream" / layout["frame_name"],
            )
            bootstrap_outputs[key] = output
            _append_anchor_receipt(anchor_receipts, receipt)


async def _v4_verify_bootstrap_packs(
    settings: Settings, cdir: Path, cid: str, private: dict, meta: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
    runner, bootstrap_outputs: dict[tuple[int, int], Path],
    anchor_receipts: list[dict],
    *, label: str, scene_alternates: dict[str, tuple[tuple[int, int], Path]] | None = None,
) -> dict:
    """Use independently generated target views; never duplicate one image as a pack."""
    if not callable(getattr(runner, "run_isolated", None)):
        raise PostprocessError(409, "image_reference_pack_failed")
    plan = _v4_frozen_plan(meta, private)
    sources = _v4_frame_sources(grouped, private)
    schedules = {
        item["scene_id"]: item for item in private["scene_anchor_schedule"]["scenes"]
    }
    scene_by_segment = {
        segment["segment_index"]: segment["scene"]["scene_id"]
        for segment in plan["segments"]
    }
    global_outputs = {
        scene_id: _anchor_receipt_path(cdir, scene_id, "global").with_suffix(".png")
        for scene_id in schedules
    }
    person_targets: dict[tuple[tuple[int, int], str], Path] = {}

    async def target_for(key: tuple[int, int], label: str) -> Path:
        scene_id = scene_by_segment[key[0]]
        existing = person_targets.get((key, label))
        if existing is not None:
            return existing
        try:
            anchor = _v4_scheduled_anchor(
                private["scene_anchor_schedule"], scene_id, label
            )
        except ValueError:
            raise PostprocessError(409, "postprocess_receipt_invalid") from None
        if (anchor["segment_index"], anchor["frame_index"]) != key:
            raise PostprocessError(409, "postprocess_receipt_invalid")
        output, receipt = await _v4_anchor(
            settings, cdir, cid, private, seedream_sem,
            scene_id=scene_id,
            label=label,
            anchor=anchor,
            canvas=sources[key],
            references=[("global_scene_anchor", global_outputs[scene_id])],
            output=_anchor_receipt_path(cdir, scene_id, label).with_suffix(".png"),
        )
        _append_anchor_receipt(anchor_receipts, receipt)
        person_targets[(key, label)] = output
        return output

    person_packs = []
    for person in plan["person_plans"]:
        reference = person["reference"]
        primary_key = (reference["segment_index"], reference["frame_index"])
        candidates = sorted(
            (segment["segment_index"], frame_index)
            for segment in plan["segments"]
            for instance in segment["persons"]
            if instance["id"] == person["id"] and instance["state"] == "replace"
            for frame_index in instance["observable_frames"]
            if (segment["segment_index"], frame_index) != primary_key
        )
        if not candidates:
            raise PostprocessError(409, "person_pack_alternate_unavailable")
        alternate_key = candidates[0]
        primary = await target_for(primary_key, f"person-{person['id']}-primary")
        alternate = await target_for(alternate_key, f"person-{person['id']}-alternate")
        if primary.resolve() == alternate.resolve():
            raise PostprocessError(409, "person_pack_alternate_unavailable")
        person_packs.append({
            "person_id": person["id"],
            "source_path": str(sources[primary_key]),
            "primary_path": str(primary),
            "alternate_path": str(alternate),
        })

    scene_packs = []
    for scene in plan["scene_plans"]:
        schedule = schedules[scene["id"]]
        global_anchor = schedule["global_anchor"]
        primary_key = (
            global_anchor["segment_index"], global_anchor["frame_index"]
        )
        selected = (scene_alternates or {}).get(scene["id"])
        if selected is None:
            alternate = _v4_visible_scene_candidate(
                private, scene["id"], primary_key
            )
            if alternate is None:
                raise PostprocessError(409, "scene_anchor_alternate_unavailable")
            alternate_key = (alternate["segment_index"], alternate["frame_index"])
            alternate_output = bootstrap_outputs.get(alternate_key)
            if alternate_output is None:
                raise PostprocessError(409, "scene_anchor_alternate_unavailable")
        else:
            alternate_key, alternate_output = selected
        primary = global_outputs[scene["id"]]
        if primary.resolve() == alternate_output.resolve():
            raise PostprocessError(409, "scene_anchor_alternate_unavailable")
        scene_packs.append({
            "scene_id": scene["id"],
            "source_path": str(sources[primary_key]),
            "primary_path": str(primary),
            "alternate_path": str(alternate_output),
        })
    metrics = _v4_pack_palette_metrics(
        plan, sources, cdir, anchor_receipts, person_packs, scene_packs,
    )
    try:
        verdict = await asyncio.to_thread(
            image_optimization.generate_reference_pack_verdict,
            runner,
            plan,
            person_packs,
            scene_packs,
            metrics,
            session_dir=cdir,
        )
    except Exception:
        raise PostprocessError(409, "image_reference_pack_failed") from None
    if verdict.get("passed") is not True:
        raise PostprocessError(409, "image_reference_pack_failed")
    receipt = _semantic_receipt(
        cdir=cdir,
        private=private,
        label=label,
        verdict=verdict,
        metrics=metrics,
        anchor_receipts=anchor_receipts,
        person_packs=person_packs,
        scene_packs=scene_packs,
    )
    _write_json_receipt(_semantic_receipt_path(cdir, label), receipt)
    return receipt


async def _v4_fan_out(
    settings: Settings, cdir: Path, cid: str, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
    bootstrap_outputs: dict[tuple[int, int], Path], anchor_receipts: list[dict],
) -> list[Path]:
    sources = _v4_frame_sources(grouped, private)
    layout_outputs = {
        (layout["segment_index"], layout["frame_index"]): bootstrap_outputs[
            (layout["segment_index"], layout["frame_index"])
        ]
        for scene in private["scene_anchor_schedule"]["scenes"]
        for layout in scene["segment_layout_anchors"]
    }
    layout_by_segment = {
        segment_index: frame_index
        for segment_index, frame_index in layout_outputs
    }
    scene_by_segment = {
        frame["segment_index"]: frame["scene_id"]
        for frame in private["execution_inputs"]["frames"]
    }
    frame_by_key = {
        (frame["segment_index"], frame["frame_index"]): frame
        for frame in private["execution_inputs"]["frames"]
    }
    pending = [
        key for key in sorted(sources)
        if key not in layout_outputs
    ]
    async def one(key: tuple[int, int]) -> dict:
        index, frame_index = key
        frame = frame_by_key[key]
        scene_id = scene_by_segment[index]
        try:
            anchor = _v4_scheduled_anchor(
                private["scene_anchor_schedule"], scene_id,
                f"fanout-{index:04d}-{frame_index:04d}",
            )
        except ValueError:
            raise PostprocessError(409, "postprocess_receipt_invalid") from None
        if (
            anchor["segment_index"], anchor["frame_index"], anchor["frame_name"],
            anchor["source_sha256"],
        ) != (index, frame_index, frame["frame_name"], frame["source_sha256"]):
            raise PostprocessError(409, "postprocess_receipt_invalid")
        output, receipt = await _v4_anchor(
            settings, cdir, cid, private, seedream_sem,
            scene_id=scene_id,
            label=f"fanout-{index:04d}-{frame_index:04d}",
            anchor=anchor,
            canvas=sources[key],
            references=[("segment_layout_anchor", layout_outputs[(
                index, layout_by_segment[index]
            )])],
            output=_private_dir(cdir, index) / "seedream" / frame["frame_name"],
        )
        return {"key": key, "output": output, "receipt": receipt}

    results = await asyncio.gather(
        *(one(key) for key in pending),
        return_exceptions=True,
    )
    errors = [item for item in results if isinstance(item, BaseException)]
    if errors:
        error = errors[0]
        if isinstance(error, asyncio.CancelledError):
            raise error
        if isinstance(error, PostprocessError):
            raise error
        raise PostprocessError(502, sanitize(str(error)))
    for result in results:
        _append_anchor_receipt(anchor_receipts, result["receipt"])
    outputs = dict(layout_outputs)
    outputs.update({result["key"]: result["output"] for result in results})
    return [
        outputs[(index, frame_index)]
        for index in sorted(grouped)
        for frame_index, _target in enumerate(grouped[index], 1)
    ]


def _publish_segment(outputs: list[Path], targets: list[tuple[Path, Path]]) -> None:
    # Publish the complete directory in one rename; no partial canonical set is observable.
    output_roots = {path.parent for path in outputs}
    staged_files = (
        sorted(path.name for path in next(iter(output_roots)).iterdir() if path.is_file())
        if len(output_roots) == 1 else []
    )
    expected_files = sorted(canonical.name for _, canonical in targets)
    if (
        len(outputs) != len(targets)
        or staged_files != expected_files
        or any(
            not output.is_file() or output.name != canonical.name
            or not _valid_png(output, source)
            for output, (source, canonical) in zip(outputs, targets)
        )
    ):
        raise PostprocessError(502, "postprocess_artifacts_invalid")
    destination = targets[0][1].parent
    if destination.is_dir():
        if _canonical_complete(targets):
            return
        raise PostprocessError(409, "postprocess_canonical_conflict")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        for output, (_, canonical) in zip(outputs, targets):
            copied = temporary / canonical.name
            shutil.copyfile(output, copied)
            _fsync_file(copied)
        _fsync_dir(temporary)
        os.replace(temporary, destination)
        _fsync_dir(destination.parent)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)


async def _run_segment(settings: Settings, cid: str, cdir: Path, index: int,
                       targets: list[tuple[Path, Path]], options: dict[str, bool],
                       private: dict, mediakit_sem: asyncio.Semaphore,
                       seedream_sem: asyncio.Semaphore,
                       *, defer_publish: bool = False) -> None:
    inputs = [source for source, _ in targets]
    source_sha256s = {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest()
        for source in inputs
    }
    try:
        if options["remove_subtitle"]:
            _update_segment(settings, cid, index, stage="text")
            inputs = await _mediakit_stage(
                settings, cdir, index, inputs, "text", mediakit.TEXT_SCENE, mediakit_sem
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        if options["remove_brand"]:
            _update_segment(settings, cid, index, stage="brand")
            inputs = await _mediakit_stage(
                settings, cdir, index, inputs, "brand", mediakit.ICON_SCENE, mediakit_sem
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        if options["optimize_image"]:
            _update_segment(settings, cid, index, stage="seedream", completed_frames=0)
            inputs = await _seedream_stage(
                settings, cdir, cid, index, inputs, private, source_sha256s,
                seedream_sem,
            )
            _update_segment(settings, cid, index, completed_frames=len(inputs))
        _update_segment(settings, cid, index, stage="publishing")
        if defer_publish:
            return
        _publish_segment(inputs, targets)
        _update_segment(
            settings, cid, index, status="done", stage="done",
            completed_frames=len(targets), error=None,
        )
    except asyncio.CancelledError:
        latest = storage.load_meta(settings.data_dir, cid) or {}
        ambiguous = index in _ambiguous_segments(cdir, latest.get("postprocess"))
        _update_segment(
            settings, cid, index, status="failed",
            error="submission_unknown" if ambiguous else "cancelled",
        )
        raise
    except Exception as exc:
        detail = exc.detail if isinstance(exc, PostprocessError) else sanitize(str(exc))
        _update_segment(settings, cid, index, status="failed", error=detail)


async def _run_v4_task(
    settings: Settings, cid: str, cdir: Path, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
) -> None:
    plan = _v4_frozen_plan(meta, private)
    sources = _v4_frame_sources(grouped, private)
    _v4_preflight(plan, private, sources)
    source_metrics = _v4_palette_metrics(plan, sources)
    source_payload = {
        "version": 1,
        "plan_sha256": private["plan_sha256"],
        "continuity_sha256": private["continuity_sha256"],
        "metrics": source_metrics,
    }
    source_palette_receipt = {
        **source_payload, "sha256": _receipt_sha256(source_payload),
    }
    _write_json_receipt(
        cdir / "work" / ".postprocess-private" / "scene-anchors" / "palette-source.json",
        source_palette_receipt,
    )
    bootstrap_outputs, anchor_receipts = await _v4_bootstrap_scene_anchors(
        settings, cdir, cid, private, grouped, seedream_sem
    )
    await _v4_generate_layout_anchors(
        settings, cdir, cid, private, grouped, seedream_sem,
        bootstrap_outputs, anchor_receipts,
    )
    outputs = await _v4_fan_out(
        settings, cdir, cid, private, grouped, seedream_sem,
        bootstrap_outputs, anchor_receipts,
    )
    offset = 0
    for index in sorted(grouped):
        targets = grouped[index]
        _publish_segment(outputs[offset:offset + len(targets)], targets)
        offset += len(targets)


async def run_task(settings: Settings, cid: str, mediakit_sem: asyncio.Semaphore,
                   seedream_sem: asyncio.Semaphore | None = None,
                   only_segments: set[int] | None = None, *, audit_runner=None,
                   verification_runner=None) -> None:
    cdir = (settings.data_dir / cid).resolve()
    seedream_sem = seedream_sem or asyncio.Semaphore(settings.seedream_concurrency)
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
        return
    if isinstance(meta.get("postprocess"), dict) and meta["postprocess"].get("status") == "done":
        # `start` may schedule a background task for an exact done reuse.  No
        # worker may reinterpret canonical files as permission to replay a
        # paid v4 DAG.
        return
    try:
        private = _private_receipt(meta)
    except PostprocessError:
        _mutate_postprocess(
            settings, cid,
            lambda _meta, post: post.update(
                status="failed", error="postprocess_receipt_invalid"
            ),
        )
        return
    options = private["options"]
    # Both V3 and V4 are receipt-bound verification contracts.  Only legacy
    # V2 may use the old immediate per-frame publication path.
    defer_v3_publish = options["optimize_image"] and private["version"] in {3, 4}
    audit_grouped = _group_targets(cdir, meta)
    grouped = audit_grouped
    v4_project_dag = private["version"] == 4 and options["optimize_image"]
    if only_segments is not None and not v4_project_dag:
        grouped = {index: targets for index, targets in grouped.items() if index in only_segments}
    states = {
        item.get("index"): item.get("status")
        for item in (meta.get("postprocess") or {}).get("segments", [])
        if isinstance(item, dict)
    }
    if not v4_project_dag:
        grouped = {
            index: targets for index, targets in grouped.items()
            if states.get(index) != "done"
        }
    runtime_private = private
    runtime_grouped = grouped
    if v4_project_dag and runtime_grouped:
        try:
            # Only immutable schedule/source integrity belongs before a paid
            # edge.  Continuity, target and Lab semantics remain prompt input.
            plan = _v4_frozen_plan(meta, private)
            raw_sources = _v4_frame_sources(runtime_grouped, private)
            _v4_preflight(
                plan, private, raw_sources,
            )
        except PostprocessError:
            _mark_plan_audit_failed(settings, cid, set(grouped))
            return
    if v4_project_dag and (options["remove_subtitle"] or options["remove_brand"]):
        try:
            runtime_private, runtime_grouped = await _v4_prepare_canvases(
                settings, cdir, cid, meta, private, grouped, mediakit_sem,
            )
        except asyncio.CancelledError:
            raise
        except PostprocessError as exc:
            _mark_image_verification_failed(
                settings, cid, set(grouped),
                error=exc.detail if isinstance(exc.detail, str) else "image_verification_failed",
            )
            return
        except Exception:
            _mark_image_verification_failed(settings, cid, set(grouped))
            return
    if private["version"] == 4 and options["optimize_image"] and runtime_grouped:
        try:
            await _run_v4_task(
                settings, cid, cdir, meta, runtime_private, runtime_grouped, seedream_sem,
            )
        except asyncio.CancelledError:
            raise
        except PostprocessError as exc:
            _mark_image_verification_failed(
                settings, cid, set(grouped),
                error=exc.detail if isinstance(exc.detail, str) else "image_verification_failed",
            )
            _mutate_postprocess(
                settings, cid,
                lambda _meta, post: post.update(error=exc.detail),
            )
            return
        except Exception:
            _mark_image_verification_failed(settings, cid, set(grouped))
            return
        def finalize_v4(_meta: dict, post: dict) -> None:
            # Manifest-last: canonical directories may have been staged, but no
            # segment becomes observable as done until every v4 publish and the
            # complete technical generation DAG has succeeded.
            for item in post.get("segments", []):
                if item.get("index") in audit_grouped:
                    item.update(
                        status="done", stage="done",
                        completed_frames=item.get("total_frames"), error=None,
                    )
            post.update(status="done", error=None)
            post["frames"] = [
                _frame_ref(item["index"], path.name)
                for item in post.get("segments", []) if item.get("status") == "done"
                for path in _canonical_files(settings.data_dir / cid, item["index"])
            ]
        _mutate_postprocess(settings, cid, finalize_v4)
        return
    try:
        await asyncio.gather(*(
            _run_segment(settings, cid, cdir, index, targets, options, private,
                         mediakit_sem, seedream_sem, defer_publish=defer_v3_publish)
            for index, targets in grouped.items()
        ))
    except asyncio.CancelledError:
        def cancel(_meta: dict, post: dict) -> None:
            unknown = any(
                item.get("error") == "submission_unknown"
                for item in post.get("segments", []) if isinstance(item, dict)
            )
            post.update(
                status="failed",
                error="submission_unknown" if unknown else "cancelled",
            )

        _mutate_postprocess(settings, cid, cancel)
        raise

    if defer_v3_publish and grouped:
        current = storage.load_meta(settings.data_dir, cid) or {}
        current_states = {
            item.get("index"): item.get("status")
            for item in (current.get("postprocess") or {}).get("segments", [])
            if isinstance(item, dict)
        }
        unpublished = {
            index: targets for index, targets in audit_grouped.items()
            if current_states.get(index) != "done"
        }
        if not any(current_states.get(index) == "failed" for index in grouped):
            try:
                for index in sorted(unpublished):
                    targets = unpublished[index]
                    outputs = [
                        _private_dir(cdir, index) / "seedream" / canonical.name
                        for _, canonical in targets
                    ]
                    _publish_segment(outputs, targets)
                    _update_segment(
                        settings, cid, index, status="done", stage="done",
                        completed_frames=len(targets), error=None,
                    )
            except Exception:
                _mark_image_verification_failed(settings, cid, set(unpublished))
                return

    def finalize(_meta: dict, post: dict) -> None:
        segments = post.get("segments") or []
        failed = [item for item in segments if item.get("status") == "failed"]
        running = [item for item in segments if item.get("status") == "running"]
        post["status"] = "running" if running else ("failed" if failed else "done")
        post["error"] = (
            "submission_unknown"
            if any(item.get("error") == "submission_unknown" for item in failed)
            else ("segment_failed" if failed else None)
        )
        post["frames"] = sorted(
            _frame_ref(item["index"], path.name)
            for item in segments if item.get("status") == "done"
            for path in _canonical_files(cdir, item["index"])
        )

    _mutate_postprocess(settings, cid, finalize)


async def retry_segment(settings: Settings, cid: str, index: int, payload: dict,
                        locks: dict[str, asyncio.Lock]) -> None:
    if set(payload) != {"confirm", "expected_revision"}:
        raise PostprocessError(422, "invalid_postprocess_retry_request")
    if payload.get("confirm") is not True:
        raise PostprocessError(409, "confirmation required")
    expected = payload.get("expected_revision")
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise PostprocessError(422, "invalid_postprocess_retry_request")
    lock = locks.setdefault(cid, asyncio.Lock())
    async with lock:
        def mutate(meta: dict) -> None:
            private = _private_receipt(meta)
            post = dict(meta["postprocess"])
            segments = [dict(item) for item in post.get("segments", [])]
            canonical = private["options"]
            _capability_gate(settings, canonical)
            target = next((item for item in segments if item.get("index") == index), None)
            if target is None or target.get("status") != "failed":
                raise PostprocessError(409, "segment_not_retryable")
            v4_project_dag = private["version"] == 4 and canonical["optimize_image"]
            if v4_project_dag and target.get("error") == "submission_unknown":
                # A paid anchor may have reached the provider.  This remains a
                # GET/recovery case, never a retry that could create a second
                # project DAG submission.
                raise PostprocessError(409, "submission_unknown")
            if v4_project_dag and target.get("error") != "provider_rejected":
                raise PostprocessError(409, "segment_not_retryable")
            if target.get("revision") != expected:
                raise PostprocessError(409, _structured(
                    "postprocess_revision_changed", "分段状态已更新，请刷新页面后重试。"
                ))
            if v4_project_dag:
                # The schedule is one project DAG.  A segment endpoint may
                # start a retry, but the resumed run must revalidate/reuse the
                # exact complete graph rather than isolate that segment.
                if any(item.get("revision") != expected for item in segments):
                    raise PostprocessError(409, "postprocess_revision_changed")
                for item in segments:
                    item.update(
                        status="running", error=None, stage="queued", revision=expected + 1,
                    )
            else:
                target.update(
                    status="running", error=None, revision=expected + 1,
                    stage=target.get("stage") or "queued",
                )
            post.update(status="running", error=None, segments=segments)
            meta["postprocess"] = post

        updated = storage.mutate_meta(settings.data_dir, cid, mutate)
        if updated is None:
            raise PostprocessError(404, "not found")


def _v4_canvas_execution_for_h3(
    settings: Settings | None, cdir: Path, meta: dict, private: dict, originals: list[Path],
) -> tuple[dict, dict[tuple[int, int], Path]]:
    """Rebuild the only legal v4 canvas source map from terminal MediaKit receipts."""
    combined = private["options"]["remove_subtitle"] or private["options"]["remove_brand"]
    if not combined:
        return private, {
            (0 if len(path.resolve().relative_to(cdir / "work").parts) == 2
             else int(path.resolve().relative_to(cdir / "work").parts[1]), int(path.stem)): path
            for path in originals
        }
    if settings is None:
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    raw = meta.get("_v4_canvas_execution")
    if not isinstance(raw, dict):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    payload = {key: value for key, value in raw.items() if key != "sha256"}
    if (
        set(raw) != {
            "version", "postprocess_receipt_sha256", "options", "frames",
            "derived_optimization", "sha256",
        }
        or raw.get("version") != 1
        or raw.get("sha256") != _receipt_sha256(payload)
        or raw.get("postprocess_receipt_sha256") != private.get("receipt_sha256")
        or raw.get("options") != private.get("options")
        or not isinstance(raw.get("frames"), list)
        or not isinstance(raw.get("derived_optimization"), dict)
    ):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    raw_sources = {}
    work = (cdir / "work").resolve()
    for path in originals:
        relative = path.resolve().relative_to(work)
        index = 0 if len(relative.parts) == 2 else int(relative.parts[1])
        raw_sources[(index, int(path.stem))] = path
    expected_records = []
    canvas_sources = {}
    for index, frame_index in sorted(raw_sources):
        source = raw_sources[(index, frame_index)]
        matches = [
            item for item in raw["frames"] if isinstance(item, dict)
            and (item.get("segment_index"), item.get("frame_index")) == (index, frame_index)
        ]
        if len(matches) != 1:
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        record = matches[0]
        try:
            canvas = (cdir / record["canvas_path"]).resolve()
            if not canvas.is_relative_to(cdir.resolve()):
                raise ValueError
            stages = record["stages"]
            if (
                record.get("frame_name") != source.name
                or record.get("source_sha256") != _sha256_path(source)
                or record.get("canvas_sha256") != _sha256_path(canvas)
                or not isinstance(stages, list)
            ):
                raise ValueError
            stage_source = source
            expected_stages = [
                ("text", mediakit.TEXT_SCENE, private["options"]["remove_subtitle"]),
                ("brand", mediakit.ICON_SCENE, private["options"]["remove_brand"]),
            ]
            rebuilt = []
            for stage, scene, enabled in expected_stages:
                if not enabled:
                    continue
                output = _v4_canvas_stage_path(cdir, index, stage, source.name)
                entry = _v4_mediakit_success(stage_source, output, scene)
                if entry is None:
                    raise ValueError
                entry["receipt_path"] = str(
                    Path(entry["receipt_path"]).resolve().relative_to(cdir.resolve())
                )
                rebuilt.append(entry)
                stage_source = output
            if rebuilt != stages or canvas != stage_source:
                raise ValueError
            expected_records.append(_v4_canvas_record(
                cdir, index, frame_index, source, canvas, rebuilt,
            ))
            canvas_sources[(index, frame_index)] = canvas
        except (KeyError, OSError, TypeError, ValueError):
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
    if raw["frames"] != expected_records:
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    # Re-derive the entire execution/prompt/schedule receipt from the
    # receipt-bound canvas records.  A self-consistent replacement JSON must
    # not redirect H3 to a different graph, prompt, or transition view.
    expected_derived = _v4_derive_canvas_optimization(
        settings, meta, private, expected_records,
    )
    if raw["derived_optimization"] != expected_derived:
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    effective = _v4_effective_private(
        private, expected_derived, raw["sha256"],
    )
    execution_frames = effective["execution_inputs"]["frames"]
    if [
        (frame["segment_index"], frame["frame_index"], frame["source_sha256"])
        for frame in execution_frames
    ] != [
        (record["segment_index"], record["frame_index"], record["canvas_sha256"])
        for record in expected_records
    ]:
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    return effective, canvas_sources


def _validate_mediakit_only_outputs(
    cdir: Path, private: dict, originals: list[Path], selected: list[Path],
) -> None:
    """Keep MediaKit-only H3 reuse bound to its durable provider receipts."""
    if private["options"]["optimize_image"]:
        return
    stages = [
        ("text", mediakit.TEXT_SCENE, private["options"]["remove_subtitle"]),
        ("brand", mediakit.ICON_SCENE, private["options"]["remove_brand"]),
    ]
    if not any(enabled for _stage, _scene, enabled in stages):
        return
    work = (cdir / "work").resolve()
    try:
        for original, canonical in zip(originals, selected):
            relative = original.resolve().relative_to(work)
            index = 0 if len(relative.parts) == 2 else int(relative.parts[1])
            current = original
            for stage, scene, enabled in stages:
                if not enabled:
                    continue
                output = _private_dir(cdir, index) / stage / original.name
                if _v4_mediakit_success(current, output, scene) is None:
                    raise ValueError
                current = output
            if _sha256_path(canonical) != _sha256_path(current):
                raise ValueError
    except (OSError, ValueError, PostprocessError):
        raise PostprocessError(409, "postprocess_artifacts_invalid") from None


def generation_keyframes(
    cdir: Path, meta: dict, originals: list[Path], *, settings: Settings | None = None,
) -> list[Path]:
    state = meta.get("postprocess")
    if state is None:
        # Once any postprocess authority has been frozen, absence of its
        # public state is corruption, not a reason to silently hand H3 the
        # original inputs.  This also covers MediaKit-only selections.
        if any(key in meta for key in (
            "_postprocess_receipt", "_image_optimization",
            "_v4_canvas_execution", "_image_verification",
        )):
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        return originals
    if not isinstance(state, dict) or state.get("status") != "done":
        raise PostprocessError(409, "postprocess_not_ready")
    frame_refs = state.get("frames")
    if not isinstance(frame_refs, list) or any(not isinstance(item, str) for item in frame_refs):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    selected = []
    expected_refs = []
    work = cdir.resolve() / "work"
    for original in originals:
        source = original.resolve()
        try:
            relative = source.relative_to(work)
        except ValueError:
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
        parts = relative.parts
        if len(parts) == 2 and parts[0] == "keyframes":
            output, ref = work / "postprocessed" / source.name, source.name
        elif len(parts) == 5 and parts[0] == "segments" and parts[1].isdigit() \
                and parts[2:4] == ("work", "keyframes"):
            output = work / "segments" / parts[1] / "work" / "postprocessed" / source.name
            ref = f"segments/{parts[1]}/work/postprocessed/{source.name}"
        else:
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        if not output.is_file():
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        expected_refs.append(ref)
        selected.append(output)
    # The public postprocess manifest is an ordered frozen input contract for
    # H3, not a set membership hint.  Reordering, omitting, or adding frames
    # must therefore be rejected before any generated image is exposed.
    if frame_refs != expected_refs:
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    if len({path.resolve() for path in selected}) != len(selected):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    frozen_intent = meta.get("_postprocess_receipt")
    optimization = image_optimization.receipt(meta)
    public_options = state.get("options") if isinstance(state, dict) else None
    intent_options = (
        frozen_intent.get("options") if isinstance(frozen_intent, dict) else public_options
    )
    expected_v3 = (
        isinstance(intent_options, dict)
        and intent_options.get("optimize_image") is True
        and (
            isinstance(frozen_intent, dict) and frozen_intent.get("version") == 3
            or isinstance(optimization, dict) and optimization.get("version") == 3
        )
    )
    expected_v4 = (
        isinstance(intent_options, dict)
        and intent_options.get("optimize_image") is True
        and (
            isinstance(frozen_intent, dict) and frozen_intent.get("version") == 4
            or isinstance(optimization, dict) and optimization.get("version") == 4
        )
    )
    try:
        # `_postprocess_receipt` is immutable start intent.  Do not let a
        # deleted image receipt or a copied PNG downgrade a media-only job.
        media_private = _private_receipt(meta)
        _validate_mediakit_only_outputs(cdir, media_private, originals, selected)
    except PostprocessError:
        raise PostprocessError(409, "postprocess_artifacts_invalid") from None
    if expected_v3:
        raw = meta.get("_image_verification")
        try:
            if not isinstance(optimization, dict) or optimization.get("version") != 3:
                raise ValueError
            frozen_plan = image_optimization.dual_target_plan_receipt(meta)
            if not isinstance(frozen_plan, dict) or frozen_plan.get("version") != 3:
                raise ValueError
            plan = {key: value for key, value in frozen_plan.items() if key != "sha256"}
            payload = {key: value for key, value in raw.items() if key != "sha256"}
            if (
                not isinstance(raw, dict)
                or set(raw) != {
                    "version", "plan_version", "plan_sha256", "frames",
                    "verdict", "sha256",
                }
                or raw["version"] != 1
                or raw["plan_version"] != 3
                or raw["plan_sha256"] != optimization["plan_sha256"]
                or raw["sha256"] != _receipt_sha256(payload)
                or image_optimization.canonical_verification(
                    raw["verdict"], plan,
                ).get("passed") is not True
            ):
                raise ValueError
            expected_frames = []
            for original, output in zip(originals, selected):
                relative = original.resolve().relative_to(work)
                segment_index = 0 if len(relative.parts) == 2 else int(relative.parts[1])
                if not _valid_png(output, original):
                    raise ValueError
                expected_frames.append({
                    "segment_index": segment_index,
                    "frame_name": output.name,
                    "source_sha256": _sha256_path(original),
                    "output_sha256": _sha256_path(output),
                })
            if raw["frames"] != expected_frames:
                raise ValueError
        except (
            AttributeError, KeyError, OSError, TypeError, ValueError,
            image_optimization.ImageOptimizationIneligibleError,
            image_optimization.ImageOptimizationOutputError,
        ):
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
    if expected_v4:
        raw = meta.get("_image_verification")
        try:
            base_private = _private_receipt(meta)
            if not isinstance(optimization, dict) or optimization.get("version") != 4:
                raise ValueError
            # The public continuity receipt carries its own SHA.  Canonical
            # verifiers accept the exact plan payload, never that wrapper.
            plan = _v4_frozen_plan(meta, base_private)
            if not isinstance(raw, dict):
                raise ValueError
            runtime_private, runtime_sources = _v4_canvas_execution_for_h3(
                settings, cdir, meta, base_private, originals,
            )
            payload = {key: value for key, value in raw.items() if key != "sha256"}
            expected_sha = _receipt_sha256(payload)
            expected_schedule_sha = _receipt_sha256(
                runtime_private["scene_anchor_schedule"]
            )
            verified = {
                (item["segment_index"], item["frame_name"]): item
                for item in raw["frames"]
            }
            if (
                set(raw) != {
                    "version", "plan_sha256", "continuity_sha256",
                    "scene_anchor_schedule_sha256",
                    "canvas_execution_sha256",
                    "semantic_receipts", "anchor_receipts", "source_palette_receipt_sha256",
                    "palette_metrics", "palette_metrics_sha256", "frames", "verdict", "sha256",
                }
                or raw["version"] != 1
                or raw["sha256"] != expected_sha
                or raw["plan_sha256"] != runtime_private["plan_sha256"]
                or raw["continuity_sha256"] != runtime_private["continuity_sha256"]
                or raw["scene_anchor_schedule_sha256"] != expected_schedule_sha
                or raw["canvas_execution_sha256"] != runtime_private.get("canvas_execution_sha256")
                or raw["palette_metrics_sha256"] != raw["palette_metrics"].get("sha256")
                or len(verified) != len(selected)
            ):
                raise ValueError
            source_receipt = _load_json_receipt(
                cdir / "work" / ".postprocess-private" / "scene-anchors" / "palette-source.json"
            )
            if (
                source_receipt is None
                or source_receipt.get("sha256") != raw["source_palette_receipt_sha256"]
                or source_receipt.get("plan_sha256") != runtime_private["plan_sha256"]
                or source_receipt.get("continuity_sha256") != runtime_private["continuity_sha256"]
            ):
                raise ValueError
            if raw["semantic_receipts"] != []:
                raise ValueError
            if image_optimization.canonical_verification(raw["verdict"], plan).get("passed") is not True:
                raise ValueError
            source_sha256s = {}
            for original in originals:
                relative = original.resolve().relative_to(work)
                segment_index = 0 if len(relative.parts) == 2 else int(relative.parts[1])
                source_sha256s[(segment_index, int(original.stem))] = _sha256_path(
                    runtime_sources[(segment_index, int(original.stem))]
                )
            anchors = _v4_anchor_receipt_index(
                cdir, plan, runtime_private["scene_anchor_schedule"]
            )
            expected_anchor_manifest = [
                {
                    "scene_id": descriptor["scene_id"],
                    "label": descriptor["label"],
                    "sha256": next(
                        receipt["sha256"] for receipt in anchors.values()
                        if receipt["scene_id"] == descriptor["scene_id"]
                        and receipt["label"] == descriptor["label"]
                    ),
                }
                for descriptor in _v4_expected_anchor_descriptors(
                    plan, runtime_private["scene_anchor_schedule"]
                )
            ]
            if raw["anchor_receipts"] != expected_anchor_manifest:
                raise ValueError
            if not all(_valid_v4_anchor_receipt(
                cdir, receipt,
                plan_sha256=runtime_private["plan_sha256"],
                continuity_sha256=runtime_private["continuity_sha256"],
                    source_sha256s=source_sha256s,
                    source_paths=runtime_sources,
                    anchors=anchors,
            ) for receipt in anchors.values()):
                raise ValueError
            pairs = []
            expected_verified_frames = []
            for original, output in zip(originals, selected):
                segment_index = 0
                relative = original.resolve().relative_to(work)
                if len(relative.parts) == 5:
                    segment_index = int(relative.parts[1])
                item = verified[(segment_index, original.name)]
                if (
                    item["source_sha256"] != _sha256_path(
                        runtime_sources[(segment_index, int(original.stem))]
                    )
                    or item["output_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()
                    or not _valid_png(output, runtime_sources[(segment_index, int(original.stem))])
                ):
                    raise ValueError
                expected_verified_frames.append({
                    "segment_index": segment_index,
                    "frame_name": original.name,
                    "source_sha256": item["source_sha256"],
                    "output_sha256": item["output_sha256"],
                })
                pairs.append((segment_index, runtime_sources[(segment_index, int(original.stem))], output))
            if raw["frames"] != expected_verified_frames:
                raise ValueError
            if not _valid_palette_metrics_for_outputs(raw["palette_metrics"], pairs):
                raise ValueError
        except (
            AttributeError, KeyError, OSError, StopIteration, TypeError, ValueError,
            image_optimization.ImageOptimizationIneligibleError,
            image_optimization.ImageOptimizationOutputError,
        ):
            raise PostprocessError(409, "postprocess_artifacts_invalid") from None
    return selected


_ATTEMPT_RE = re.compile(r"^\d+-r([1-9]\d*)\.json$")


def _ambiguous_segments(cdir: Path, post: object) -> set[int]:
    ambiguous: set[int] = set()
    if not isinstance(post, dict):
        return ambiguous
    current = {
        item.get("index"): (item.get("revision"), item.get("status"))
        for item in post.get("segments", []) if isinstance(item, dict)
    }
    attempts_root = cdir / "work" / ".postprocess-private"
    for attempt in attempts_root.glob("*/attempts/*.json") if attempts_root.is_dir() else ():
        matched = _ATTEMPT_RE.match(attempt.name)
        try:
            index = int(attempt.parents[1].name)
        except ValueError:
            continue
        revision, status = current.get(index, (None, None))
        if matched is None or status == "done" or int(matched.group(1)) != revision:
            continue
        try:
            payload = json.loads(attempt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"status": "submission_unknown"}
        if payload.get("status") in {"submitting", "submission_unknown"}:
            ambiguous.add(index)
    return ambiguous


def _v4_project_revision(post: object) -> int:
    if not isinstance(post, dict):
        raise ValueError
    segments = post.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError
    revisions = {
        item.get("revision") for item in segments if isinstance(item, dict)
    }
    if len(revisions) != 1:
        raise ValueError
    revision = next(iter(revisions))
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError
    return revision


def _ambiguous_v4_anchor_attempts(cdir: Path, private: dict, post: object) -> bool:
    """Inspect only frozen typed DAG paths, never a permissive filename pattern."""
    try:
        revision = _v4_project_revision(post)
        descriptors = _v4_expected_anchor_descriptors(
            {}, private["scene_anchor_schedule"],
        )
    except (KeyError, TypeError, ValueError):
        return True
    for descriptor in descriptors:
        attempts = _anchor_receipt_path(
            cdir, descriptor["scene_id"], descriptor["label"],
        ).parent / "attempts"
        order = descriptor["anchor"].get("order")
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            return True
        # Revisions are append-only.  An unresolved older typed attempt is
        # still ambiguous; an explicit retry can only follow a determinate
        # provider rejection, never an unknown submission.
        for attempt_revision in range(1, revision + 1):
            attempt = attempts / (
                f"{order:04d}-{descriptor['label']}-r{attempt_revision}.json"
            )
            if not attempt.is_file():
                continue
            try:
                payload = json.loads(attempt.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return True
            if payload.get("status") in {"submitting", "submission_unknown"}:
                return True
    return False


def recover_running(settings: Settings) -> list[str]:
    """Fail ambiguous Seedream submissions; return only locally safe jobs to resume."""
    jobs = []
    for meta in storage.list_conversations(settings.data_dir):
        post = meta.get("postprocess")
        if not isinstance(post, dict) or post.get("status") not in {"running", "failed"}:
            continue
        cid = meta["id"]
        try:
            private = _private_receipt(meta)
        except PostprocessError:
            _mutate_postprocess(
                settings, cid,
                lambda _meta, current: current.update(
                    status="failed", error="postprocess_receipt_invalid"
                ),
            )
            continue
        ambiguous = _ambiguous_segments(settings.data_dir / cid, post)
        anchor_ambiguous = private["version"] == 4 and _ambiguous_v4_anchor_attempts(
            settings.data_dir / cid, private, post
        )
        if anchor_ambiguous:
            ambiguous.update(
                item.get("index") for item in post.get("segments", [])
                if isinstance(item, dict) and isinstance(item.get("index"), int)
            )
        if ambiguous or anchor_ambiguous:
            for index in ambiguous:
                _update_segment(
                    settings, cid, index, status="failed", error="submission_unknown"
                )
            _mutate_postprocess(
                settings, cid,
                lambda _meta, current: current.update(
                    status="failed", error="submission_unknown"
                ),
            )
            continue
        if post.get("status") == "failed":
            continue
        jobs.append(cid)
    return jobs
