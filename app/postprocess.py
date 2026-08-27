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
        and decoded.shape[:2] == original.shape[:2]
    )


_PALETTE_METRIC_ALGORITHM = "area-weighted-cie-lab-hsv-v1"
_PALETTE_METRIC_THRESHOLDS = {
    # OpenCV uint8 Lab encodes neutral b* as 128.  Dividing by 127 keeps
    # the stored whole-frame b* coordinate stable and explicitly portable.
    "lab_b_star_neutral": 128.0,
    "lab_b_star_scale": 127.0,
    "warm_cool_delta": 0.05,
    "muted_saturation": 0.16,
    "vivid_saturation": 0.58,
}


def _receipt_sha256(payload: dict) -> str:
    """Digest a JSON receipt without permitting NaN or ordering ambiguity."""
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _area_weighted_palette_metric(path: Path) -> dict:
    """Return the frozen whole-frame palette proxy for one decoded PNG.

    This intentionally measures the entire pixel canvas rather than a semantic
    crop: the contract protects global perceived warmth/coolness and saturation.
    """
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        delta = float((
            lab[:, :, 2].mean() - _PALETTE_METRIC_THRESHOLDS["lab_b_star_neutral"]
        ) / _PALETTE_METRIC_THRESHOLDS["lab_b_star_scale"])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean() / 255.0)
    except (cv2.error, OSError, ValueError):
        raise PostprocessError(409, "dominant_palette_metric_invalid") from None
    threshold = _PALETTE_METRIC_THRESHOLDS["warm_cool_delta"]
    family = "warm" if delta > threshold else "cool" if delta < -threshold else "balanced"
    muted = _PALETTE_METRIC_THRESHOLDS["muted_saturation"]
    vivid = _PALETTE_METRIC_THRESHOLDS["vivid_saturation"]
    style = "muted" if saturation < muted else "vivid" if saturation > vivid else "natural"
    return {
        "bytes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "warm_cool_family": family,
        "saturation_style": style,
        "mean_lab_b_star": round(delta, 6),
        "mean_saturation": round(saturation, 6),
    }


def _v4_palette_metrics(
    plan: dict,
    sources: dict[tuple[int, int], Path],
    outputs: dict[tuple[int, int], Path] | None = None,
) -> dict:
    """Measure and enforce v4's source/output global palette contract."""
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
        if (
            source["warm_cool_family"]
            != contract["area_weighted_warm_cool_family"]
            or source["saturation_style"] != contract["saturation_style"]
        ):
            raise PostprocessError(409, "dominant_palette_source_mismatch")
        record = {
            "segment_index": key[0],
            "frame_index": key[1],
            "contract": contract,
            "source": source,
        }
        if outputs is not None and key in outputs:
            output = _area_weighted_palette_metric(outputs[key])
            if (
                output["warm_cool_family"]
                != contract["area_weighted_warm_cool_family"]
                or output["saturation_style"] != contract["saturation_style"]
            ):
                raise PostprocessError(409, "dominant_palette_verification_failed")
            record["output"] = output
        frames.append(record)
    payload = {
        "version": 1,
        "algorithm": _PALETTE_METRIC_ALGORITHM,
        "thresholds": _PALETTE_METRIC_THRESHOLDS,
        "frames": frames,
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


async def _run_v3_plan_audit(
    runner, cdir: Path, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]],
) -> None:
    if not callable(getattr(runner, "run_isolated", None)):
        raise PostprocessError(409, "image_plan_audit_failed")
    plan, audit_inputs, segments = _v3_plan_audit_inputs(meta, private, grouped)
    try:
        verdict = await asyncio.to_thread(
            image_optimization.generate_plan_audit_verdict,
            runner,
            plan,
            audit_inputs,
            segments,
            session_dir=cdir,
        )
    except Exception:
        raise PostprocessError(409, "image_plan_audit_failed") from None
    if verdict.get("passed") is not True:
        raise PostprocessError(409, "image_plan_audit_failed")


def _v3_verification_inputs(
    cdir: Path, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]],
) -> tuple[dict, list[dict]]:
    """Bind verify to the same frozen v3 plan and unpublishable staged outputs."""
    plan, _audit_inputs, _audit_segments = _v3_plan_audit_inputs(
        meta, private, grouped
    )
    segments = []
    for index in sorted(grouped):
        targets = grouped[index]
        if not targets:
            raise PostprocessError(409, "image_verification_failed")
        output = _private_dir(cdir, index) / "seedream"
        expected = [canonical.name for _, canonical in targets]
        actual = (
            [path.name for path in sorted(output.glob("*.png"))]
            if output.is_dir() else []
        )
        if actual != expected:
            raise PostprocessError(409, "image_verification_failed")
        segments.append({
            "index": index,
            "source_keyframes_dir": targets[0][0].parent,
            "output_keyframes_dir": output,
        })
    return plan, segments


async def _run_v3_verification(
    runner, cdir: Path, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], deterministic_metrics: dict | None = None,
) -> dict:
    if not callable(getattr(runner, "run_isolated", None)):
        raise PostprocessError(409, "image_verification_failed")
    try:
        plan, segments = _v3_verification_inputs(cdir, meta, private, grouped)
        verdict = await asyncio.to_thread(
            image_optimization.generate_project_verdict,
            runner,
            plan,
            segments,
            {} if deterministic_metrics is None else deterministic_metrics,
            session_dir=cdir,
        )
    except Exception:
        raise PostprocessError(409, "image_verification_failed") from None
    if verdict.get("passed") is not True:
        raise PostprocessError(409, "image_verification_failed")
    return verdict


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
    settings: Settings, cid: str, indices: set[int],
) -> None:
    def mutate(_meta: dict, post: dict) -> None:
        segments = [dict(item) for item in post.get("segments", [])]
        for item in segments:
            if item.get("index") in indices and item.get("status") != "done":
                item.update(status="failed", error="image_verification_failed")
        post["segments"] = segments
        post["status"] = "failed"
        post["error"] = "image_verification_failed"
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
            if reuse_done:
                for item in states:
                    item.update(
                        status="done", stage="done",
                        completed_frames=item["total_frames"],
                    )
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
    def binding(kind: str, identifier: str, pack: dict) -> dict:
        def output_binding(value: object) -> dict:
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
            return {
                "anchor_receipt_sha256": matches[0]["sha256"],
                "output_sha256": output_sha256,
            }
        try:
            source_sha256 = hashlib.sha256(Path(pack["source_path"]).read_bytes()).hexdigest()
            primary = output_binding(pack["primary_path"])
            alternate = output_binding(pack["alternate_path"])
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


def _v4_anchor_receipt_index(cdir: Path, plan: dict, schedule: dict) -> dict[str, dict]:
    """Load only schedule-derived anchor receipts; never discover arbitrary files."""
    labels: dict[str, set[str]] = {
        scene["id"]: {"global", "pack-alternate"}
        for scene in plan["scene_plans"]
    }
    for scene in schedule["scenes"]:
        scene_id = scene["scene_id"]
        labels.setdefault(scene_id, set()).update(
            f"layout-{item['segment_index']:04d}"
            for item in scene["segment_layout_anchors"]
        )
    for person in plan["person_plans"]:
        for scene_id in labels:
            labels[scene_id].update({
                f"person-{person['id']}-primary",
                f"person-{person['id']}-alternate",
            })
    result = {}
    for scene_id, scene_labels in labels.items():
        for label in scene_labels:
            receipt = _load_json_receipt(_anchor_receipt_path(cdir, scene_id, label))
            if receipt is None:
                continue
            sha = receipt.get("sha256")
            if not isinstance(sha, str) or sha in result:
                raise ValueError
            result[sha] = receipt
    return result


def _valid_semantic_pack_bindings(
    cdir: Path, plan: dict, schedule: dict, receipt: dict,
    source_sha256s: dict[tuple[int, int], str],
) -> bool:
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
        endpoint_receipts = []
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
                    "anchor_receipt_sha256", "output_sha256",
                }:
                    return False
                anchor = anchors.get(side["anchor_receipt_sha256"])
                if anchor is None or anchor.get("output_sha256") != side["output_sha256"]:
                    return False
                endpoint_receipts.append(side["anchor_receipt_sha256"])
        return len(endpoint_receipts) == len(set(endpoint_receipts))
    except (KeyError, TypeError, ValueError):
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
    if output.exists() or receipt_path.exists():
        raise PostprocessError(409, "submission_unknown")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = receipt_path.parent / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
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
    try:
        async with seedream_sem:
            await seedream.edit(
                task_settings,
                [path.read_bytes() for path in inputs],
                prompt,
                output,
                receipt_path=attempts / f"{anchor['order']:04d}.json",
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


async def _v4_bootstrap_scene_anchors(
    settings: Settings, cdir: Path, cid: str, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
) -> tuple[dict[tuple[int, int], Path], list[dict]]:
    """Build global -> per-segment layout -> real alternate anchor outputs."""
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
        alternate = _v4_visible_scene_candidate(
            private,
            scene_id,
            (global_anchor["segment_index"], global_anchor["frame_index"]),
        )
        if alternate is None:
            raise PostprocessError(409, "scene_anchor_alternate_unavailable")
        alternate_key = (alternate["segment_index"], alternate["frame_index"])
        if alternate_key not in outputs:
            alternate_output, receipt = await _v4_anchor(
                settings, cdir, cid, private, seedream_sem,
                scene_id=scene_id,
                label="pack-alternate",
                anchor={
                    "order": max(item["anchor"]["order"] for item in receipts) + 1,
                    "segment_index": alternate["segment_index"],
                    "frame_index": alternate["frame_index"],
                    "frame_name": alternate["frame_name"],
                    "source_sha256": alternate["source_sha256"],
                },
                canvas=sources[alternate_key],
                references=[("global_scene_anchor", global_output)],
                output=_anchor_receipt_path(cdir, scene_id, "pack-alternate").with_suffix(".png"),
            )
            outputs[alternate_key] = alternate_output
            receipts.append(receipt)
    return outputs, receipts


async def _v4_generate_layout_anchors(
    settings: Settings, cdir: Path, cid: str, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
    bootstrap_outputs: dict[tuple[int, int], Path], anchor_receipts: list[dict],
    *, meta: dict, runner, semantic_receipts: list[dict],
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
            global_anchor = scene["global_anchor"]
            alternate_override = (
                {} if key == (
                    global_anchor["segment_index"], global_anchor["frame_index"]
                ) else {scene_id: (key, output)}
            )
            semantic_receipts.append(await _v4_verify_bootstrap_packs(
                settings, cdir, cid, private, meta, grouped, seedream_sem, runner,
                bootstrap_outputs, anchor_receipts,
                label=f"layout-{layout['segment_index']:04d}",
                scene_alternates=alternate_override,
            ))


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

    def next_order() -> int:
        return max(item["anchor"]["order"] for item in anchor_receipts) + 1

    async def target_for(key: tuple[int, int], label: str) -> Path:
        scene_id = scene_by_segment[key[0]]
        existing = person_targets.get((key, label))
        if existing is not None:
            return existing
        frame = next(
            item for item in private["execution_inputs"]["frames"]
            if (item["segment_index"], item["frame_index"]) == key
        )
        anchor = {
            "order": next_order(),
            "segment_index": key[0], "frame_index": key[1],
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
        }
        prior = _load_json_receipt(_anchor_receipt_path(cdir, scene_id, label))
        if isinstance(prior, dict) and isinstance(prior.get("anchor"), dict):
            frozen_anchor = prior["anchor"]
            if all(
                frozen_anchor.get(field) == anchor[field]
                for field in ("segment_index", "frame_index", "frame_name", "source_sha256")
            ) and isinstance(frozen_anchor.get("order"), int):
                anchor = frozen_anchor
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
    metric_outputs = dict(bootstrap_outputs)
    for scene_id, schedule in schedules.items():
        global_anchor = schedule["global_anchor"]
        metric_outputs.setdefault(
            (global_anchor["segment_index"], global_anchor["frame_index"]),
            global_outputs[scene_id],
        )
    metrics = _v4_palette_metrics(plan, sources, metric_outputs)
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
    first_order = max(item["anchor"]["order"] for item in anchor_receipts) + 1

    async def one(order: int, key: tuple[int, int]) -> dict:
        index, frame_index = key
        frame = frame_by_key[key]
        scene_id = scene_by_segment[index]
        anchor = {
            "order": order,
            "segment_index": index,
            "frame_index": frame_index,
            "frame_name": frame["frame_name"],
            "source_sha256": frame["source_sha256"],
        }
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
        *(one(first_order + position, key) for position, key in enumerate(pending)),
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


def _write_v4_verification_receipt(
    settings: Settings, cid: str, private: dict, grouped: dict[int, list[tuple[Path, Path]]],
    outputs: list[Path], semantic_receipts: list[dict],
    source_palette_receipt: dict, palette_metrics: dict, verdict: dict,
) -> None:
    expected = [
        (index, source, canonical)
        for index in sorted(grouped)
        for source, canonical in grouped[index]
    ]
    if len(expected) != len(outputs):
        raise PostprocessError(409, "image_verification_failed")
    payload = {
        "version": 1,
        "plan_sha256": private["plan_sha256"],
        "continuity_sha256": private["continuity_sha256"],
        "scene_anchor_schedule_sha256": hashlib.sha256(json.dumps(
            private["scene_anchor_schedule"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "semantic_receipts": [
            {"label": item["label"], "sha256": item["sha256"]}
            for item in semantic_receipts
        ],
        "source_palette_receipt_sha256": source_palette_receipt["sha256"],
        "palette_metrics": palette_metrics,
        "palette_metrics_sha256": palette_metrics["sha256"],
        "frames": [
            {
                "segment_index": index,
                "frame_name": canonical.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            for (index, source, canonical), output in zip(expected, outputs)
        ],
        "verdict": verdict,
    }
    receipt = {**payload, "sha256": hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()}
    storage.update_meta(settings.data_dir, cid, _image_verification=receipt)


async def _run_v4_task(
    settings: Settings, cid: str, cdir: Path, meta: dict, private: dict,
    grouped: dict[int, list[tuple[Path, Path]]], seedream_sem: asyncio.Semaphore,
    audit_runner, verification_runner,
) -> None:
    if private["options"]["remove_subtitle"] or private["options"]["remove_brand"]:
        raise PostprocessError(409, "v4_anchor_preprocess_unavailable")
    plan = _v4_frozen_plan(meta, private)
    sources = _v4_frame_sources(grouped, private)
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
    semantic_receipts = [await _v4_verify_bootstrap_packs(
        settings, cdir, cid, private, meta, grouped, seedream_sem,
        verification_runner or audit_runner, bootstrap_outputs, anchor_receipts,
        label="bootstrap",
    )]
    await _v4_generate_layout_anchors(
        settings, cdir, cid, private, grouped, seedream_sem,
        bootstrap_outputs, anchor_receipts, meta=meta,
        runner=verification_runner or audit_runner,
        semantic_receipts=semantic_receipts,
    )
    outputs = await _v4_fan_out(
        settings, cdir, cid, private, grouped, seedream_sem,
        bootstrap_outputs, anchor_receipts,
    )
    output_by_key: dict[tuple[int, int], Path] = {}
    output_cursor = 0
    for index in sorted(grouped):
        for frame_index, _target in enumerate(grouped[index], 1):
            output_by_key[(index, frame_index)] = outputs[output_cursor]
            output_cursor += 1
    palette_metrics = _v4_palette_metrics(plan, sources, output_by_key)
    verdict = await _run_v3_verification(
        verification_runner or audit_runner, cdir, meta, private, grouped,
        palette_metrics,
    )
    _write_v4_verification_receipt(
        settings, cid, private, grouped, outputs, semantic_receipts,
        source_palette_receipt, palette_metrics, verdict,
    )
    offset = 0
    for index in sorted(grouped):
        targets = grouped[index]
        _publish_segment(outputs[offset:offset + len(targets)], targets)
        offset += len(targets)
        _update_segment(
            settings, cid, index, status="done", stage="done",
            completed_frames=len(targets), error=None,
        )


async def run_task(settings: Settings, cid: str, mediakit_sem: asyncio.Semaphore,
                   seedream_sem: asyncio.Semaphore | None = None,
                   only_segments: set[int] | None = None, *, audit_runner=None,
                   verification_runner=None) -> None:
    cdir = (settings.data_dir / cid).resolve()
    seedream_sem = seedream_sem or asyncio.Semaphore(settings.seedream_concurrency)
    meta = storage.load_meta(settings.data_dir, cid)
    if meta is None:
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
    if only_segments is not None:
        grouped = {index: targets for index, targets in grouped.items() if index in only_segments}
    states = {
        item.get("index"): item.get("status")
        for item in (meta.get("postprocess") or {}).get("segments", [])
        if isinstance(item, dict)
    }
    grouped = {
        index: targets for index, targets in grouped.items()
        if states.get(index) != "done"
    }
    if options["optimize_image"] and private["version"] in {3, 4} and grouped:
        try:
            await _run_v3_plan_audit(
                audit_runner, cdir, meta, private, audit_grouped
            )
        except PostprocessError:
            _mark_plan_audit_failed(settings, cid, set(grouped))
            return
    if private["version"] == 4 and grouped:
        try:
            await _run_v4_task(
                settings, cid, cdir, meta, private, grouped, seedream_sem,
                audit_runner, verification_runner,
            )
        except asyncio.CancelledError:
            raise
        except PostprocessError as exc:
            _mark_image_verification_failed(settings, cid, set(grouped))
            _mutate_postprocess(
                settings, cid,
                lambda _meta, post: post.update(error=exc.detail),
            )
            return
        except Exception:
            _mark_image_verification_failed(settings, cid, set(grouped))
            return
        def finalize_v4(_meta: dict, post: dict) -> None:
            post.update(status="done", error=None)
            post["frames"] = sorted(
                _frame_ref(item["index"], path.name)
                for item in post.get("segments", []) if item.get("status") == "done"
                for path in _canonical_files(settings.data_dir / cid, item["index"])
            )
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
                await _run_v3_verification(
                    verification_runner or audit_runner,
                    cdir,
                    current,
                    private,
                    audit_grouped,
                )
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
            if target.get("revision") != expected:
                raise PostprocessError(409, _structured(
                    "postprocess_revision_changed", "分段状态已更新，请刷新页面后重试。"
                ))
            target.update(
                status="running", error=None, revision=expected + 1,
                stage=target.get("stage") or "queued",
            )
            post.update(status="running", error=None, segments=segments)
            meta["postprocess"] = post

        updated = storage.mutate_meta(settings.data_dir, cid, mutate)
        if updated is None:
            raise PostprocessError(404, "not found")


def generation_keyframes(cdir: Path, meta: dict, originals: list[Path]) -> list[Path]:
    state = meta.get("postprocess")
    if state is None:
        return originals
    if not isinstance(state, dict) or state.get("status") != "done":
        raise PostprocessError(409, "postprocess_not_ready")
    frame_refs = state.get("frames")
    if not isinstance(frame_refs, list) or any(not isinstance(item, str) for item in frame_refs):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    available = set(frame_refs)
    selected = []
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
        if ref not in available or not output.is_file():
            raise PostprocessError(409, "postprocess_artifacts_invalid")
        selected.append(output)
    if len({path.resolve() for path in selected}) != len(selected):
        raise PostprocessError(409, "postprocess_artifacts_invalid")
    optimization = image_optimization.receipt(meta)
    if optimization is not None and optimization.get("version") == 4:
        raw = meta.get("_image_verification")
        try:
            plan = image_optimization.dual_target_plan_receipt(meta)
            if not isinstance(plan, dict) or plan.get("version") != 4:
                raise ValueError
            if not isinstance(raw, dict):
                raise ValueError
            payload = {key: value for key, value in raw.items() if key != "sha256"}
            expected_sha = _receipt_sha256(payload)
            expected_schedule_sha = _receipt_sha256(
                optimization["scene_anchor_schedule"]
            )
            verified = {
                (item["segment_index"], item["frame_name"]): item
                for item in raw["frames"]
            }
            if (
                set(raw) != {
                    "version", "plan_sha256", "continuity_sha256",
                    "scene_anchor_schedule_sha256",
                    "semantic_receipts", "source_palette_receipt_sha256",
                    "palette_metrics", "palette_metrics_sha256", "frames", "verdict", "sha256",
                }
                or raw["version"] != 1
                or raw["sha256"] != expected_sha
                or raw["plan_sha256"] != optimization["plan_sha256"]
                or raw["continuity_sha256"] != optimization["continuity_sha256"]
                or raw["scene_anchor_schedule_sha256"] != expected_schedule_sha
                or raw["verdict"].get("passed") is not True
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
                or source_receipt.get("plan_sha256") != optimization["plan_sha256"]
                or source_receipt.get("continuity_sha256") != optimization["continuity_sha256"]
            ):
                raise ValueError
            if not isinstance(raw["semantic_receipts"], list) or not raw["semantic_receipts"]:
                raise ValueError
            source_sha256s = {}
            for original in originals:
                relative = original.resolve().relative_to(work)
                segment_index = 0 if len(relative.parts) == 2 else int(relative.parts[1])
                source_sha256s[(segment_index, int(original.stem))] = hashlib.sha256(
                    original.read_bytes()
                ).hexdigest()
            for item in raw["semantic_receipts"]:
                if not isinstance(item, dict) or set(item) != {"label", "sha256"}:
                    raise ValueError
                receipt = _load_json_receipt(_semantic_receipt_path(cdir, item["label"]))
                if (
                    receipt is None or receipt.get("sha256") != item["sha256"]
                    or receipt.get("plan_sha256") != optimization["plan_sha256"]
                    or receipt.get("continuity_sha256") != optimization["continuity_sha256"]
                    or receipt.get("verdict", {}).get("passed") is not True
                    or not _valid_semantic_pack_bindings(
                        cdir, plan, optimization["scene_anchor_schedule"],
                        receipt, source_sha256s,
                    )
                ):
                    raise ValueError
            pairs = []
            for original, output in zip(originals, selected):
                segment_index = 0
                relative = original.resolve().relative_to(work)
                if len(relative.parts) == 5:
                    segment_index = int(relative.parts[1])
                item = verified[(segment_index, original.name)]
                if (
                    item["source_sha256"] != hashlib.sha256(original.read_bytes()).hexdigest()
                    or item["output_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()
                ):
                    raise ValueError
                pairs.append((segment_index, original, output))
            if not _valid_palette_metrics_for_outputs(raw["palette_metrics"], pairs):
                raise ValueError
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
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
        if ambiguous:
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
