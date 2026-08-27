"""Receipt-bound client contract for a remote scene-component mask worker.

The worker is deliberately treated as untrusted.  This module freezes every
frame and hard-cut boundary before POST, resumes accepted work with GET only,
and validates the complete local PNG set before publishing success.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote, urlsplit

import cv2
import httpx
import numpy as np


DEFAULT_BACKEND = "sam2_birefnet"
_REQUEST_SCHEMA = "duet.scene-mask.request"
_CLIENT_SCHEMA = "duet.scene-mask.client"
_PRODUCER_SCHEMA = "duet.scene-mask.producer"
_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_TASK_ID = re.compile(r"[A-Za-z0-9_.:-]{1,256}")
_SAFE_TEXT_MAX = 256

_PUBLIC_MESSAGES = {
    "invalid_input": "Scene mask input is invalid",
    "invalid_project_path": "Scene mask path is invalid",
    "prompt_missing": "Scene mask prompt is missing",
    "people_protection_unknown": "People protection is unknown",
    "receipt_invalid": "Scene mask receipt is invalid",
    "receipt_mismatch": "Scene mask receipt does not match the frozen request",
    "state_persist_failed": "Scene mask state could not be persisted",
    "submission_unknown": "Scene mask submission could not be confirmed",
    "worker_submit_rejected": "Scene mask worker rejected the request",
    "worker_query_failed": "Scene mask worker status could not be queried",
    "worker_protocol_error": "Scene mask worker returned an invalid response",
    "worker_failed": "Scene mask worker failed",
    "worker_output_invalid": "Scene mask worker output is invalid",
}


class SceneMaskError(RuntimeError):
    """Stable, non-sensitive public error."""

    def __init__(self, code: str):
        self.code = code if code in _PUBLIC_MESSAGES else "worker_protocol_error"
        super().__init__(_PUBLIC_MESSAGES[self.code])


@dataclass(frozen=True)
class Frame:
    frame_id: str
    path: str
    sha256: str
    width: int
    height: int
    pts: int


@dataclass(frozen=True)
class HardCutShot:
    shot_id: str
    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceFrame:
    component_id: str
    shot_id: str
    frame_id: str


@dataclass(frozen=True)
class BoxPrompt:
    component_id: str
    frame_id: str
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class PointPrompt:
    component_id: str
    frame_id: str
    x: int
    y: int
    positive: bool


@dataclass(frozen=True)
class ProtectionMask:
    frame_id: str
    path: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class SceneMaskPlan:
    plan_sha256: str
    scene_id: str
    components: tuple[str, ...]
    frames: tuple[Frame, ...]
    hard_cut_chain: tuple[HardCutShot, ...]
    references: tuple[ReferenceFrame, ...]
    box_prompts: tuple[BoxPrompt, ...]
    point_prompts: tuple[PointPrompt, ...]
    people_count: int
    people_protection_known: bool
    protection_masks: tuple[ProtectionMask, ...]
    model: str
    model_version: str
    endpoint_identity: str
    backend: str = DEFAULT_BACKEND


@dataclass(frozen=True)
class SceneMaskItem:
    purpose: str
    channel: str
    component_id: str
    shot_id: str
    frame_id: str
    path: str
    sha256: str
    byte_size: int
    width: int
    height: int
    producer_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class SceneMaskResult:
    status: str
    task_id: str | None = None
    masks: tuple[SceneMaskItem, ...] = ()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def request_payload(plan: SceneMaskPlan) -> dict[str, Any]:
    """Validate and serialize the complete provider-neutral worker request."""

    _validate_plan(plan)
    frames = [
        {
            "frame_id": frame.frame_id,
            "path": frame.path,
            "sha256": frame.sha256,
            "width": frame.width,
            "height": frame.height,
            "pts": frame.pts,
        }
        for frame in plan.frames
    ]
    protection_by_frame = {
        item.frame_id: {
            "frame_id": item.frame_id,
            "path": item.path,
            "sha256": item.sha256,
            "width": item.width,
            "height": item.height,
        }
        for item in plan.protection_masks
    }
    reference_by_job = {
        (item.component_id, item.shot_id): item.frame_id
        for item in plan.references
    }
    jobs: list[dict[str, Any]] = []
    for component_id in plan.components:
        for shot in plan.hard_cut_chain:
            frame_ids = set(shot.frame_ids)
            boxes = [
                _box_payload(item)
                for item in plan.box_prompts
                if item.component_id == component_id and item.frame_id in frame_ids
            ]
            points = [
                _point_payload(item)
                for item in plan.point_prompts
                if item.component_id == component_id and item.frame_id in frame_ids
            ]
            jobs.append(
                {
                    "component_id": component_id,
                    "shot_id": shot.shot_id,
                    "frame_ids": list(shot.frame_ids),
                    "reference_frame_id": reference_by_job[(component_id, shot.shot_id)],
                    "box_prompts": boxes,
                    "point_prompts": points,
                    "protection_masks": [
                        protection_by_frame[frame_id]
                        for frame_id in shot.frame_ids
                        if frame_id in protection_by_frame
                    ],
                }
            )
    return {
        "schema": _REQUEST_SCHEMA,
        "version": _VERSION,
        "plan_sha256": plan.plan_sha256,
        "scene_id": plan.scene_id,
        "backend": plan.backend,
        "model": {"name": plan.model, "version": plan.model_version},
        "endpoint_identity": plan.endpoint_identity,
        "components": list(plan.components),
        "frames": frames,
        "hard_cut_chain": [
            {"shot_id": shot.shot_id, "frame_ids": list(shot.frame_ids)}
            for shot in plan.hard_cut_chain
        ],
        "people_protection": {
            "known": True,
            "people_count": plan.people_count,
            "masks": [protection_by_frame[item.frame_id] for item in plan.protection_masks],
        },
        "propagation_jobs": jobs,
        "contracts": {
            "propagation_scope": "hard_cut_shot_only",
            "membership_engine": "sam2",
            "edge_refinement": "birefnet_uncertain_edges_only",
            "fallback": "none",
        },
    }


def advance(
    plan: SceneMaskPlan,
    *,
    endpoint: str,
    output_root: Path,
    receipt_path: Path,
    client: httpx.Client | None = None,
    timeout_s: float = 30,
) -> SceneMaskResult:
    """Submit once or resume an accepted task, then validate one GET result.

    A returned ``running`` result is intentionally non-blocking.  Calling this
    function again with the same receipt performs GET only.  There is no API
    that can turn ``submission_unknown`` into another POST.
    """

    payload = request_payload(plan)
    request_sha256 = canonical_json_sha256(payload)
    root, receipt = _validate_local_paths(output_root, receipt_path)
    submit_url = _worker_tasks_url(endpoint)
    if not _valid_timeout(timeout_s):
        raise SceneMaskError("invalid_input")
    with _receipt_lease(receipt):
        return _advance_locked(
            plan,
            payload,
            request_sha256,
            root,
            receipt,
            submit_url,
            client,
            timeout_s,
        )


def _advance_locked(
    plan: SceneMaskPlan,
    payload: dict[str, Any],
    request_sha256: str,
    root: Path,
    receipt: Path,
    submit_url: str,
    client: httpx.Client | None,
    timeout_s: float,
) -> SceneMaskResult:

    state = _load_state(receipt, payload, request_sha256)
    if state is not None:
        status = state["status"]
        if status == "succeeded":
            masks = _validate_masks(root, plan, payload, state.get("masks"))
            return SceneMaskResult("succeeded", state["task_id"], masks)
        if status == "submission_unknown":
            raise SceneMaskError("submission_unknown")
        if status == "failed":
            raise SceneMaskError(str(state["error"]))
        if status == "submitting":
            unknown = dict(state)
            unknown["status"] = "submission_unknown"
            _persist_state(receipt, unknown)
            raise SceneMaskError("submission_unknown")
        task_id = _validated_task_id(state.get("task_id"))
        return _query_once(
            plan,
            payload,
            root,
            receipt,
            state,
            submit_url,
            task_id,
            client,
            timeout_s,
        )

    state = {
        "schema": _CLIENT_SCHEMA,
        "version": _VERSION,
        "status": "submitting",
        "request": payload,
        "request_sha256": request_sha256,
    }
    _persist_state(receipt, state)
    response: httpx.Response
    try:
        response = _request(
            client,
            "POST",
            submit_url,
            json_payload=payload,
            timeout_s=timeout_s,
        )
    except httpx.HTTPError:
        state["status"] = "submission_unknown"
        _persist_state(receipt, state)
        raise SceneMaskError("submission_unknown") from None
    if not 200 <= response.status_code < 300:
        state.update(status="failed", error="worker_submit_rejected")
        _persist_state(receipt, state)
        raise SceneMaskError("worker_submit_rejected")
    try:
        response_payload = _response_object(response)
        task_id = _validated_task_id(response_payload.get("task_id"))
    except (TypeError, ValueError, SceneMaskError):
        state.update(status="failed", error="worker_protocol_error")
        _persist_state(receipt, state)
        raise SceneMaskError("worker_protocol_error") from None
    state.update(status="running", task_id=task_id)
    _persist_state(receipt, state)
    return _query_once(
        plan,
        payload,
        root,
        receipt,
        state,
        submit_url,
        task_id,
        client,
        timeout_s,
    )


def _query_once(
    plan: SceneMaskPlan,
    payload: Mapping[str, Any],
    root: Path,
    receipt: Path,
    state: dict[str, Any],
    submit_url: str,
    task_id: str,
    client: httpx.Client | None,
    timeout_s: float,
) -> SceneMaskResult:
    query_url = f"{submit_url}/{quote(task_id, safe='')}"
    try:
        response = _request(
            client,
            "GET",
            query_url,
            json_payload=None,
            timeout_s=timeout_s,
        )
    except httpx.HTTPError:
        raise SceneMaskError("worker_query_failed") from None
    if response.status_code != 200:
        raise SceneMaskError("worker_query_failed")
    try:
        response_payload = _response_object(response)
        worker_status = response_payload.get("status")
        if not isinstance(worker_status, str):
            raise ValueError("missing status")
        worker_status = worker_status.strip().lower()
    except (TypeError, ValueError):
        raise SceneMaskError("worker_protocol_error") from None
    if worker_status in {"queued", "pending", "running", "processing"}:
        state.pop("error", None)
        state["status"] = "running"
        _persist_state(receipt, state)
        return SceneMaskResult("running", task_id)
    if worker_status in {"failed", "error"}:
        state.update(status="failed", error="worker_failed")
        _persist_state(receipt, state)
        raise SceneMaskError("worker_failed")
    if worker_status != "succeeded":
        raise SceneMaskError("worker_protocol_error")
    result = response_payload.get("result")
    raw_masks = result.get("masks") if isinstance(result, Mapping) else None
    try:
        masks = _validate_masks(root, plan, payload, raw_masks)
    except SceneMaskError:
        state.update(status="failed", error="worker_output_invalid")
        _persist_state(receipt, state)
        raise
    normalized = [_mask_item_payload(item) for item in masks]
    state.pop("error", None)
    state.update(status="succeeded", masks=normalized)
    _persist_state(receipt, state)
    return SceneMaskResult("succeeded", task_id, masks)


def _validate_plan(plan: SceneMaskPlan) -> None:
    if not isinstance(plan, SceneMaskPlan):
        raise SceneMaskError("invalid_input")
    if not _is_sha256(plan.plan_sha256) or not _is_identifier(plan.scene_id):
        raise SceneMaskError("invalid_input")
    for text in (plan.backend, plan.model, plan.model_version, plan.endpoint_identity):
        if not _is_safe_identity(text):
            raise SceneMaskError("invalid_input")
    if not plan.components or any(not _is_identifier(item) for item in plan.components):
        raise SceneMaskError("invalid_input")
    if len(set(plan.components)) != len(plan.components):
        raise SceneMaskError("invalid_input")
    if not plan.frames:
        raise SceneMaskError("invalid_input")

    frames: dict[str, Frame] = {}
    previous_pts: int | None = None
    for frame in plan.frames:
        if (
            not isinstance(frame, Frame)
            or not _is_identifier(frame.frame_id)
            or frame.frame_id in frames
            or not _is_project_relative(frame.path)
            or not _is_sha256(frame.sha256)
            or not _positive_int(frame.width)
            or not _positive_int(frame.height)
            or not _nonnegative_int(frame.pts)
            or (previous_pts is not None and frame.pts <= previous_pts)
        ):
            raise SceneMaskError("invalid_input")
        frames[frame.frame_id] = frame
        previous_pts = frame.pts

    if not plan.hard_cut_chain:
        raise SceneMaskError("invalid_input")
    shot_ids: set[str] = set()
    flattened: list[str] = []
    shot_frames: dict[str, set[str]] = {}
    for shot in plan.hard_cut_chain:
        if (
            not isinstance(shot, HardCutShot)
            or not _is_identifier(shot.shot_id)
            or shot.shot_id in shot_ids
            or not shot.frame_ids
            or any(not _is_identifier(frame_id) for frame_id in shot.frame_ids)
            or len(set(shot.frame_ids)) != len(shot.frame_ids)
            or any(frame_id not in frames for frame_id in shot.frame_ids)
        ):
            raise SceneMaskError("invalid_input")
        shot_ids.add(shot.shot_id)
        flattened.extend(shot.frame_ids)
        shot_frames[shot.shot_id] = set(shot.frame_ids)
    if flattened != list(frames):
        raise SceneMaskError("invalid_input")

    references: dict[tuple[str, str], str] = {}
    for reference in plan.references:
        if (
            not isinstance(reference, ReferenceFrame)
            or not _is_identifier(reference.component_id)
            or not _is_identifier(reference.shot_id)
            or not _is_identifier(reference.frame_id)
            or reference.component_id not in plan.components
            or reference.shot_id not in shot_ids
            or reference.frame_id not in shot_frames.get(reference.shot_id, set())
            or (reference.component_id, reference.shot_id) in references
        ):
            raise SceneMaskError("invalid_input")
        references[(reference.component_id, reference.shot_id)] = reference.frame_id
    expected_jobs = {
        (component_id, shot.shot_id)
        for component_id in plan.components
        for shot in plan.hard_cut_chain
    }
    if set(references) != expected_jobs:
        raise SceneMaskError("prompt_missing")

    for box in plan.box_prompts:
        if not _valid_box_prompt(box, plan.components, frames):
            raise SceneMaskError("invalid_input")
    for point in plan.point_prompts:
        if not _valid_point_prompt(point, plan.components, frames):
            raise SceneMaskError("invalid_input")
    for (component_id, shot_id), reference_frame_id in references.items():
        has_anchor = any(
            box.component_id == component_id and box.frame_id == reference_frame_id
            for box in plan.box_prompts
        ) or any(
            point.component_id == component_id
            and point.frame_id == reference_frame_id
            and point.positive
            for point in plan.point_prompts
        )
        if not has_anchor:
            raise SceneMaskError("prompt_missing")
        if reference_frame_id not in shot_frames[shot_id]:
            raise SceneMaskError("prompt_missing")

    if plan.people_protection_known is not True:
        raise SceneMaskError("people_protection_unknown")
    if not _nonnegative_int(plan.people_count):
        raise SceneMaskError("invalid_input")
    protections: dict[str, ProtectionMask] = {}
    for item in plan.protection_masks:
        frame = frames.get(item.frame_id) if isinstance(item, ProtectionMask) else None
        if (
            frame is None
            or item.frame_id in protections
            or not _is_project_relative(item.path)
            or not _is_sha256(item.sha256)
            or not _positive_int(item.width)
            or not _positive_int(item.height)
            or item.width != frame.width
            or item.height != frame.height
        ):
            raise SceneMaskError("invalid_input")
        protections[item.frame_id] = item
    if plan.people_count == 0:
        if protections:
            raise SceneMaskError("invalid_input")
    elif set(protections) != set(frames):
        raise SceneMaskError("people_protection_unknown")


def _valid_box_prompt(
    box: Any, components: Sequence[str], frames: Mapping[str, Frame]
) -> bool:
    if not isinstance(box, BoxPrompt) or box.component_id not in components:
        return False
    frame = frames.get(box.frame_id)
    return bool(
        frame
        and all(_nonnegative_int(value) for value in (box.left, box.top))
        and all(_positive_int(value) for value in (box.right, box.bottom))
        and box.left < box.right <= frame.width
        and box.top < box.bottom <= frame.height
    )


def _valid_point_prompt(
    point: Any, components: Sequence[str], frames: Mapping[str, Frame]
) -> bool:
    if (
        not isinstance(point, PointPrompt)
        or point.component_id not in components
        or type(point.positive) is not bool
    ):
        return False
    frame = frames.get(point.frame_id)
    return bool(
        frame
        and _nonnegative_int(point.x)
        and _nonnegative_int(point.y)
        and point.x < frame.width
        and point.y < frame.height
    )


def _validate_masks(
    root: Path,
    plan: SceneMaskPlan,
    payload: Mapping[str, Any],
    raw_masks: Any,
) -> tuple[SceneMaskItem, ...]:
    if not isinstance(raw_masks, list):
        raise SceneMaskError("worker_output_invalid")
    expected = [
        (component_id, shot.shot_id, frame_id)
        for component_id in plan.components
        for shot in plan.hard_cut_chain
        for frame_id in shot.frame_ids
    ]
    if len(raw_masks) != len(expected):
        raise SceneMaskError("worker_output_invalid")
    frame_by_id = {frame.frame_id: frame for frame in plan.frames}
    jobs = {
        (job["component_id"], job["shot_id"]): job
        for job in payload["propagation_jobs"]
    }
    request_sha256 = canonical_json_sha256(payload)
    by_key: dict[tuple[str, str, str], SceneMaskItem] = {}
    seen_paths: set[str] = set()
    for raw in raw_masks:
        if not isinstance(raw, Mapping) or set(raw) != {
            "purpose",
            "channel",
            "component_id",
            "shot_id",
            "frame_id",
            "path",
            "sha256",
            "byte_size",
            "width",
            "height",
            "producer_receipt",
        }:
            raise SceneMaskError("worker_output_invalid")
        component_id = raw.get("component_id")
        shot_id = raw.get("shot_id")
        frame_id = raw.get("frame_id")
        if raw.get("purpose") != "scene_component" or raw.get("channel") != "grayscale_alpha":
            raise SceneMaskError("worker_output_invalid")
        if not all(isinstance(value, str) for value in (component_id, shot_id, frame_id)):
            raise SceneMaskError("worker_output_invalid")
        key = (component_id, shot_id, frame_id)
        if key not in expected or key in by_key:
            raise SceneMaskError("worker_output_invalid")
        frame = frame_by_id[frame_id]
        path = raw.get("path")
        if not isinstance(path, str) or path in seen_paths or not path.endswith(".png"):
            raise SceneMaskError("worker_output_invalid")
        seen_paths.add(path)
        data = _read_contained_regular(root, path)
        byte_size = raw.get("byte_size")
        width = raw.get("width")
        height = raw.get("height")
        sha256 = raw.get("sha256")
        if (
            not _positive_int(byte_size)
            or byte_size != len(data)
            or width != frame.width
            or height != frame.height
            or not _is_sha256(sha256)
            or hashlib.sha256(data).hexdigest() != sha256
            or not _valid_png_mask(data, frame.width, frame.height)
        ):
            raise SceneMaskError("worker_output_invalid")
        producer = raw.get("producer_receipt")
        job = jobs.get((component_id, shot_id))
        if not _valid_producer_receipt(
            producer,
            plan,
            frame,
            component_id,
            shot_id,
            job,
            request_sha256,
        ):
            raise SceneMaskError("worker_output_invalid")
        by_key[key] = SceneMaskItem(
            purpose="scene_component",
            channel="grayscale_alpha",
            component_id=component_id,
            shot_id=shot_id,
            frame_id=frame_id,
            path=path,
            sha256=sha256,
            byte_size=byte_size,
            width=width,
            height=height,
            producer_receipt=dict(producer),
        )
    if set(by_key) != set(expected):
        raise SceneMaskError("worker_output_invalid")
    return tuple(by_key[key] for key in expected)


def _valid_producer_receipt(
    producer: Any,
    plan: SceneMaskPlan,
    frame: Frame,
    component_id: str,
    shot_id: str,
    job: Mapping[str, Any] | None,
    request_sha256: str,
) -> bool:
    expected_keys = {
        "schema",
        "version",
        "backend",
        "model",
        "model_version",
        "endpoint_identity",
        "plan_sha256",
        "scene_id",
        "component_id",
        "shot_id",
        "frame_id",
        "frame_sha256",
        "reference_frame_id",
        "request_sha256",
        "propagation_job_sha256",
        "propagation_scope",
        "membership_engine",
        "edge_refiner",
        "edge_refinement_scope",
        "fallback",
    }
    if not isinstance(producer, Mapping) or set(producer) != expected_keys or job is None:
        return False
    return producer == {
        "schema": _PRODUCER_SCHEMA,
        "version": _VERSION,
        "backend": plan.backend,
        "model": plan.model,
        "model_version": plan.model_version,
        "endpoint_identity": plan.endpoint_identity,
        "plan_sha256": plan.plan_sha256,
        "scene_id": plan.scene_id,
        "component_id": component_id,
        "shot_id": shot_id,
        "frame_id": frame.frame_id,
        "frame_sha256": frame.sha256,
        "reference_frame_id": job["reference_frame_id"],
        "request_sha256": request_sha256,
        "propagation_job_sha256": canonical_json_sha256(job),
        "propagation_scope": "hard_cut_shot_only",
        "membership_engine": "sam2",
        "edge_refiner": "birefnet",
        "edge_refinement_scope": "sam2_uncertain_edges_only",
        "fallback": "none",
    }


def _valid_png_mask(data: bytes, width: int, height: int) -> bool:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.shape != (height, width):
        return False
    active = image > 0
    return bool(np.any(active) and not np.all(active))


def _read_contained_regular(root: Path, relative: str) -> bytes:
    if not _is_project_relative(relative):
        raise SceneMaskError("worker_output_invalid")
    candidate = root
    try:
        for part in PurePosixPath(relative).parts:
            candidate = candidate / part
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode):
                raise SceneMaskError("worker_output_invalid")
        if not stat.S_ISREG(info.st_mode):
            raise SceneMaskError("worker_output_invalid")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(candidate, flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                info.st_dev,
                info.st_ino,
            ):
                raise SceneMaskError("worker_output_invalid")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
    except SceneMaskError:
        raise
    except (OSError, ValueError):
        raise SceneMaskError("worker_output_invalid") from None
    data = b"".join(chunks)
    if not data:
        raise SceneMaskError("worker_output_invalid")
    return data


def _load_state(
    receipt: Path,
    payload: Mapping[str, Any],
    request_sha256: str,
) -> dict[str, Any] | None:
    if not receipt.exists():
        return None
    try:
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("not regular")
        raw = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        raise SceneMaskError("receipt_invalid") from None
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != _CLIENT_SCHEMA
        or raw.get("version") != _VERSION
        or not isinstance(raw.get("request"), dict)
        or not _is_sha256(raw.get("request_sha256"))
        or canonical_json_sha256(raw["request"]) != raw["request_sha256"]
    ):
        raise SceneMaskError("receipt_invalid")
    if raw["request"] != payload or raw["request_sha256"] != request_sha256:
        raise SceneMaskError("receipt_mismatch")
    status = raw.get("status")
    if status not in {"submitting", "running", "submission_unknown", "failed", "succeeded"}:
        raise SceneMaskError("receipt_invalid")
    base_keys = {"schema", "version", "status", "request", "request_sha256"}
    valid_keys = {
        "submitting": base_keys,
        "submission_unknown": base_keys,
        "running": base_keys | {"task_id"},
        "succeeded": base_keys | {"task_id", "masks"},
    }
    if status == "failed":
        if frozenset(raw) not in {
            frozenset(base_keys | {"error"}),
            frozenset(base_keys | {"task_id", "error"}),
        }:
            raise SceneMaskError("receipt_invalid")
    elif set(raw) != valid_keys[status]:
        raise SceneMaskError("receipt_invalid")
    if status in {"running", "succeeded"}:
        _validated_task_id(raw.get("task_id"), receipt_error=True)
    if status == "failed" and raw.get("error") not in {
        "worker_submit_rejected",
        "worker_protocol_error",
        "worker_failed",
        "worker_output_invalid",
    }:
        raise SceneMaskError("receipt_invalid")
    if status == "succeeded" and not isinstance(raw.get("masks"), list):
        raise SceneMaskError("receipt_invalid")
    return raw


def _persist_state(receipt: Path, state: Mapping[str, Any]) -> None:
    try:
        _atomic_write(receipt, _canonical_json_bytes(state) + b"\n")
    except (OSError, TypeError, ValueError):
        raise SceneMaskError("state_persist_failed") from None


@contextmanager
def _receipt_lease(receipt: Path) -> Iterator[None]:
    """Serialize callers sharing a receipt so only one can cross POST."""

    lock_path = receipt.with_name(f".{receipt.name}.lock")
    fd: int | None = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("lock is not regular")
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        if fd is not None:
            os.close(fd)
        raise SceneMaskError("state_persist_failed") from None
    try:
        yield
    finally:
        assert fd is not None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _validate_local_paths(output_root: Path, receipt_path: Path) -> tuple[Path, Path]:
    try:
        root_path = Path(output_root)
        if root_path.is_symlink():
            raise ValueError("symlink root")
        root = root_path.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("not directory")
        receipt = Path(os.path.abspath(receipt_path))
        relative = receipt.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() or current.is_symlink():
                if current.is_symlink():
                    raise ValueError("symlink")
                if current != receipt and not current.is_dir():
                    raise ValueError("non-directory parent")
        if receipt.exists() and not receipt.is_file():
            raise ValueError("not regular")
    except (OSError, RuntimeError, ValueError):
        raise SceneMaskError("invalid_project_path") from None
    return root, receipt


def _worker_tasks_url(endpoint: str) -> str:
    if not isinstance(endpoint, str):
        raise SceneMaskError("invalid_input")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SceneMaskError("invalid_input")
    return endpoint.rstrip("/") + "/tasks"


def _request(
    client: httpx.Client | None,
    method: str,
    url: str,
    *,
    json_payload: Mapping[str, Any] | None,
    timeout_s: float,
) -> httpx.Response:
    if client is not None:
        return client.request(method, url, json=json_payload, timeout=timeout_s)
    with httpx.Client(trust_env=False) as owned:
        return owned.request(method, url, json=json_payload, timeout=timeout_s)


def _response_object(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    return payload


def _validated_task_id(value: Any, *, receipt_error: bool = False) -> str:
    if type(value) is int and value >= 0:
        value = str(value)
    if not isinstance(value, str) or not _TASK_ID.fullmatch(value):
        raise SceneMaskError("receipt_invalid" if receipt_error else "worker_protocol_error")
    return value


def _box_payload(item: BoxPrompt) -> dict[str, Any]:
    return {
        "component_id": item.component_id,
        "frame_id": item.frame_id,
        "left": item.left,
        "top": item.top,
        "right": item.right,
        "bottom": item.bottom,
    }


def _point_payload(item: PointPrompt) -> dict[str, Any]:
    return {
        "component_id": item.component_id,
        "frame_id": item.frame_id,
        "x": item.x,
        "y": item.y,
        "positive": item.positive,
    }


def _mask_item_payload(item: SceneMaskItem) -> dict[str, Any]:
    return {
        "purpose": item.purpose,
        "channel": item.channel,
        "component_id": item.component_id,
        "shot_id": item.shot_id,
        "frame_id": item.frame_id,
        "path": item.path,
        "sha256": item.sha256,
        "byte_size": item.byte_size,
        "width": item.width,
        "height": item.height,
        "producer_receipt": dict(item.producer_receipt),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_project_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return bool(
        not path.is_absolute()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == value
    )


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _is_safe_identity(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and 0 < len(value) <= _SAFE_TEXT_MAX
        and value.strip() == value
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _valid_timeout(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) and 0 < value <= 3600
