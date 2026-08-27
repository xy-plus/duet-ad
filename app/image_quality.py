"""Fail-closed image-edit quality gates and versioned receipts.

This module deliberately does not call an image provider.  It evaluates frozen
source/output pairs, optionally consumes independently-produced pixel masks, and
can invoke the existing isolated Codex runner for the semantic ``phase=verify``
contract of ``image-postprocess``.  A caller may publish only when every enabled
deterministic gate and the semantic verifier pass.

``POC_PROFILE_V1`` is intentionally labelled uncalibrated.  Production must
inject a calibrated, versioned ``QualityProfile``; thresholds do not live in a
Skill or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

try:  # Optional at import time so a missing runtime capability fails closed.
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by capability checks/mocking.
    cv2 = None
    np = None


RECEIPT_SCHEMA = "duet.image_quality.receipt"
RECEIPT_VERSION = 1
ALGORITHM_VERSION = "image-quality-v1"
MASK_SCHEMA = "duet.image_edit_masks"
MASK_VERSION = 1
SEMANTIC_VERSION = 2
_HEX = frozenset("0123456789abcdef")
_MAX_SEMANTIC_OUTPUT_BYTES = 256 * 1024
_MAX_STAGE_IMAGE_BYTES = 64 * 1024 * 1024

_THRESHOLD_KEYS = frozenset({
    "max_global_exposure_delta_l",
    "max_protected_exposure_delta_l",
    "max_global_white_point_delta_e",
    "max_protected_white_point_delta_e",
    "max_global_contrast_relative_delta",
    "max_protected_contrast_relative_delta",
    "max_global_cct_proxy_delta",
    "max_protected_cct_proxy_delta",
    "min_person_local_delta_e",
    "min_scene_local_delta_e",
    "min_scene_edge_change_ratio",
    "max_scene_edge_change_ratio",
    "min_protected_edge_iou",
    "max_composition_centroid_shift",
    "min_mask_pixels",
})


def _json_roundtrip(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("value must be finite JSON data") from None


def _canonical_bytes(value: Any) -> bytes:
    canonical = _json_roundtrip(value)
    return json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical finite JSON data."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    """Bind either a raw v2 plan or its frozen receipt without re-hashing the hash."""
    supplied = plan.get("sha256")
    if supplied is None:
        return canonical_sha256(plan)
    if not _valid_sha(supplied):
        raise ValueError("invalid frozen plan hash")
    unsigned = {key: value for key, value in plan.items() if key != "sha256"}
    if canonical_sha256(unsigned) != supplied:
        raise ValueError("frozen plan hash mismatch")
    return supplied


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in _HEX for char in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QualityProfile:
    """Versioned thresholds.  There is no implicit default profile."""

    name: str
    version: str
    calibration: str
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value and len(value) <= 128
            for value in (self.name, self.version, self.calibration)
        ):
            raise ValueError("invalid quality profile identity")
        if not isinstance(self.thresholds, Mapping) or set(self.thresholds) != _THRESHOLD_KEYS:
            raise ValueError("quality profile thresholds do not match algorithm version")
        normalized: dict[str, float] = {}
        for key in sorted(_THRESHOLD_KEYS):
            value = self.thresholds[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("quality thresholds must be finite numbers")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("quality thresholds must be finite non-negative numbers")
            normalized[key] = number
        if normalized["min_scene_edge_change_ratio"] > normalized["max_scene_edge_change_ratio"]:
            raise ValueError("invalid scene edge threshold range")
        object.__setattr__(self, "thresholds", MappingProxyType(normalized))

    def thresholds_dict(self) -> dict[str, float]:
        return dict(self.thresholds)

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "name": self.name,
            "version": self.version,
            "calibration": self.calibration,
            "thresholds": self.thresholds_dict(),
        }
        return {**raw, "sha256": canonical_sha256(raw)}


POC_PROFILE_V1 = QualityProfile(
    name="poc",
    version="poc-v1",
    calibration="uncalibrated_poc",
    thresholds={
        "max_global_exposure_delta_l": 18.0,
        "max_protected_exposure_delta_l": 8.0,
        "max_global_white_point_delta_e": 22.0,
        "max_protected_white_point_delta_e": 7.0,
        "max_global_contrast_relative_delta": 0.45,
        "max_protected_contrast_relative_delta": 0.25,
        "max_global_cct_proxy_delta": 0.42,
        "max_protected_cct_proxy_delta": 0.18,
        "min_person_local_delta_e": 6.0,
        "min_scene_local_delta_e": 6.0,
        "min_scene_edge_change_ratio": 0.08,
        "max_scene_edge_change_ratio": 0.85,
        "min_protected_edge_iou": 0.55,
        "max_composition_centroid_shift": 0.08,
        "min_mask_pixels": 16.0,
    },
)


@dataclass(frozen=True)
class MaskArtifact:
    path: Path
    relative_path: str
    sha256: str
    width: int
    height: int
    producer_receipt: Mapping[str, Any] | None = None

    @classmethod
    def from_path(cls, path: Path, relative_path: str) -> "MaskArtifact":
        resolved = Path(path).resolve(strict=True)
        if cv2 is None:
            raise ValueError("mask dependency unavailable")
        image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim not in (2, 3):
            raise ValueError("mask is not a decodable image")
        return cls(
            path=resolved,
            relative_path=relative_path,
            sha256=_file_sha256(resolved),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            producer_receipt=None,
        )


@dataclass(frozen=True)
class FrameMasks:
    version: int
    segment_index: int
    frame_index: int
    persons: Mapping[str, MaskArtifact]
    scene: MaskArtifact
    protected_non_target: MaskArtifact
    producer_receipt_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.version != MASK_VERSION:
            raise ValueError("unsupported frame mask version")
        if (
            isinstance(self.segment_index, bool) or not isinstance(self.segment_index, int)
            or self.segment_index < 0
            or isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("invalid frame mask identity")
        if not isinstance(self.persons, Mapping):
            raise ValueError("invalid person masks")
        normalized = dict(sorted(self.persons.items()))
        if not all(isinstance(key, str) and key and isinstance(value, MaskArtifact)
                   for key, value in normalized.items()):
            raise ValueError("invalid person masks")
        if not isinstance(self.scene, MaskArtifact) or not isinstance(self.protected_non_target, MaskArtifact):
            raise ValueError("invalid scene/protected masks")
        if not _valid_sha(self.producer_receipt_sha256) or not _valid_sha(self.manifest_sha256):
            raise ValueError("invalid mask receipt binding")
        object.__setattr__(self, "persons", MappingProxyType(normalized))


@dataclass(frozen=True)
class GateResult:
    name: str
    version: str
    status: str
    code: str | None
    metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid gate status")
        if self.status == "pass" and self.code is not None:
            raise ValueError("passing gate cannot have a failure code")
        if self.status != "pass" and (not isinstance(self.code, str) or not self.code):
            raise ValueError("non-passing gate requires a stable code")
        object.__setattr__(self, "metrics", MappingProxyType(_json_roundtrip(dict(self.metrics))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "code": self.code,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class SemanticVerdict:
    status: str
    code: str | None
    checks: Mapping[str, str]
    verdict_sha256: str | None

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid semantic status")
        if self.status == "pass" and self.code is not None:
            raise ValueError("passing semantic result cannot have a code")
        if self.status != "pass" and (not isinstance(self.code, str) or not self.code):
            raise ValueError("non-passing semantic result requires a stable code")
        if self.verdict_sha256 is not None and not _valid_sha(self.verdict_sha256):
            raise ValueError("invalid semantic verdict hash")
        normalized = dict(sorted(self.checks.items()))
        if not all(isinstance(key, str) and key and value in {
            "pass", "fail", "unknown", "not_applicable"
        } for key, value in normalized.items()):
            raise ValueError("invalid semantic checks")
        object.__setattr__(self, "checks", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "checks": dict(self.checks),
            "verdict_sha256": self.verdict_sha256,
        }


@dataclass(frozen=True)
class QualityReceipt:
    plan_sha256: str
    mask_manifest_sha256: str | None
    control_mode: str
    profile: Mapping[str, Any]
    status: str
    publishable: bool
    frames: tuple[Mapping[str, Any], ...]
    gates: tuple[GateResult, ...]
    semantic: SemanticVerdict
    failures: tuple[Mapping[str, str], ...]

    @property
    def provider_retry_allowed(self) -> bool:
        return False

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(item["code"] for item in self.failures)

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "schema": RECEIPT_SCHEMA,
            "version": RECEIPT_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "plan_sha256": self.plan_sha256,
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "control_mode": self.control_mode,
            "profile": _json_roundtrip(dict(self.profile)),
            "status": self.status,
            "publishable": self.publishable,
            "provider_retry_allowed": False,
            "frames": [_json_roundtrip(dict(item)) for item in self.frames],
            "gates": [item.to_dict() for item in self.gates],
            "semantic": self.semantic.to_dict(),
            "failures": [_json_roundtrip(dict(item)) for item in self.failures],
        }
        return {**raw, "sha256": canonical_sha256(raw)}


class QualityGate(Protocol):
    name: str
    version: str

    def evaluate(self, context: "_Context", profile: QualityProfile) -> GateResult: ...


class SemanticVerifier(Protocol):
    def verify(
        self, plan: Mapping[str, Any], source_frames: Sequence[Path], output_frames: Sequence[Path],
        *, deterministic_metrics: Mapping[str, Any],
    ) -> SemanticVerdict: ...


class ReferencePackSemanticVerifier(Protocol):
    def verify_reference_packs(
        self,
        plan: Mapping[str, Any],
        source_slots: Mapping[str, Sequence[Path]],
        generated_packs: Mapping[str, Sequence[Path]],
        *,
        deterministic_metrics: Mapping[str, Any],
    ) -> SemanticVerdict: ...


def _canonical_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "width", "height"}:
        raise ValueError("invalid mask artifact")
    path = raw["path"]
    if not isinstance(path, str) or not path or len(path) > 1024:
        raise ValueError("invalid mask artifact path")
    if not _valid_sha(raw["sha256"]):
        raise ValueError("invalid mask artifact hash")
    for key in ("width", "height"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int) or raw[key] <= 0:
            raise ValueError("invalid mask artifact dimensions")
    return dict(raw)


def _canonical_mask_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "path", "sha256", "width", "height", "producer_receipt"
    } or not isinstance(raw.get("producer_receipt"), dict):
        raise ValueError("invalid mask artifact producer receipt")
    _canonical_artifact({key: raw[key] for key in ("path", "sha256", "width", "height")})
    return _json_roundtrip(raw)


def _canonical_scene_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "purpose", "channel", "component_id", "shot_id", "frame_id", "path",
        "sha256", "byte_size", "width", "height", "producer_receipt",
    }:
        raise ValueError("invalid scene mask artifact")
    if raw.get("purpose") != "scene_component" or raw.get("channel") != "grayscale_alpha":
        raise ValueError("invalid scene mask purpose")
    if not all(isinstance(raw.get(key), str) and raw[key] for key in (
        "component_id", "shot_id", "frame_id"
    )):
        raise ValueError("invalid scene mask identity")
    if isinstance(raw.get("byte_size"), bool) or not isinstance(raw.get("byte_size"), int) or raw["byte_size"] <= 0:
        raise ValueError("invalid scene mask byte size")
    _canonical_mask_artifact({
        "path": raw["path"], "sha256": raw["sha256"], "width": raw["width"],
        "height": raw["height"], "producer_receipt": raw["producer_receipt"],
    })
    return _json_roundtrip(raw)


def _validate_person_producer_receipt(
    receipt: Any,
    *,
    source: Mapping[str, Any],
    artifact: Mapping[str, Any],
    purpose: str,
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "version", "producer", "purpose", "source", "request_sha256",
        "params", "cache_version", "mask",
    } or receipt.get("schema") != "duet.image-mask-producer" or receipt.get("version") != 1:
        raise ValueError("invalid person mask producer receipt")
    producer = receipt.get("producer")
    if not isinstance(producer, dict) or set(producer) != {"provider", "action", "model"}:
        raise ValueError("invalid person mask producer receipt")
    if not all(isinstance(value, str) and value and len(value) <= 128 for value in producer.values()):
        raise ValueError("invalid person mask producer receipt")
    receipt_source = receipt.get("source")
    if not isinstance(receipt_source, dict) or set(receipt_source) != {
        "path", "sha256", "width", "height", "frame_pts"
    }:
        raise ValueError("invalid person mask producer receipt")
    if (
        receipt.get("purpose") != purpose
        or {key: receipt_source[key] for key in ("path", "sha256", "width", "height")} != source
        or not isinstance(receipt_source["frame_pts"], str) or not receipt_source["frame_pts"]
        or not _valid_sha(receipt.get("request_sha256"))
        or not isinstance(receipt.get("params"), dict)
        or not isinstance(receipt.get("cache_version"), str) or not receipt["cache_version"]
    ):
        raise ValueError("person mask producer receipt binding mismatch")
    mask = receipt.get("mask")
    if not isinstance(mask, dict) or set(mask) != {
        "path", "sha256", "size", "width", "height", "mime_type",
        "alpha_nonzero_pixels", "alpha_transparent_pixels",
    }:
        raise ValueError("invalid person mask producer receipt")
    if (
        {key: mask[key] for key in ("path", "sha256", "width", "height")} != {
            key: artifact[key] for key in ("path", "sha256", "width", "height")
        }
        or mask["mime_type"] != "image/png"
        or isinstance(mask["size"], bool) or not isinstance(mask["size"], int) or mask["size"] <= 0
        or isinstance(mask["alpha_nonzero_pixels"], bool)
        or not isinstance(mask["alpha_nonzero_pixels"], int)
        or isinstance(mask["alpha_transparent_pixels"], bool)
        or not isinstance(mask["alpha_transparent_pixels"], int)
        or mask["alpha_nonzero_pixels"] <= 0
        or mask["alpha_transparent_pixels"] <= 0
        or mask["alpha_nonzero_pixels"] + mask["alpha_transparent_pixels"]
        != artifact["width"] * artifact["height"]
    ):
        raise ValueError("person mask producer receipt binding mismatch")
    expected_request = {
        "provider": producer["provider"],
        "action": producer["action"],
        "model": producer["model"],
        "purpose": purpose,
        "source_sha256": source["sha256"],
        "width": source["width"],
        "height": source["height"],
        "frame_pts": receipt_source["frame_pts"],
        "params": receipt["params"],
        "cache_version": receipt["cache_version"],
    }
    if canonical_sha256(expected_request) != receipt["request_sha256"]:
        raise ValueError("person mask producer request hash mismatch")


def _validate_scene_producer_receipt(
    receipt: Any,
    *,
    source: Mapping[str, Any],
    artifact: Mapping[str, Any],
    plan_sha256: str,
) -> None:
    expected_keys = {
        "schema", "version", "backend", "model", "model_version",
        "endpoint_identity", "plan_sha256", "scene_id", "component_id", "shot_id",
        "frame_id", "frame_sha256", "reference_frame_id", "request_sha256",
        "propagation_job_sha256", "propagation_scope", "membership_engine",
        "edge_refiner", "edge_refinement_scope", "fallback",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ValueError("invalid scene mask producer receipt")
    if (
        receipt.get("schema") != "duet.scene-mask.producer" or receipt.get("version") != 1
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("frame_sha256") != source["sha256"]
        or receipt.get("component_id") != artifact["component_id"]
        or receipt.get("shot_id") != artifact["shot_id"]
        or receipt.get("frame_id") != artifact["frame_id"]
        or receipt.get("propagation_scope") != "hard_cut_shot_only"
        or receipt.get("membership_engine") != "sam2"
        or receipt.get("edge_refiner") != "birefnet"
        or receipt.get("edge_refinement_scope") != "sam2_uncertain_edges_only"
        or receipt.get("fallback") != "none"
        or not all(_valid_sha(receipt.get(key)) for key in (
            "plan_sha256", "frame_sha256", "request_sha256", "propagation_job_sha256"
        ))
    ):
        raise ValueError("scene mask producer receipt binding mismatch")
    if not all(isinstance(receipt.get(key), str) and receipt[key] for key in (
        "backend", "model", "model_version", "endpoint_identity", "scene_id",
        "component_id", "shot_id", "frame_id", "reference_frame_id",
    )):
        raise ValueError("invalid scene mask producer receipt")


def _canonical_inventory(frame_inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(frame_inventory, Sequence) or isinstance(frame_inventory, (str, bytes)):
        raise ValueError("invalid frame inventory")
    result = []
    seen = set()
    for raw in frame_inventory:
        if not isinstance(raw, Mapping) or set(raw) != {
            "segment_index", "frame_index", "source", "person_ids"
        }:
            raise ValueError("invalid frame inventory")
        segment = raw["segment_index"]
        frame = raw["frame_index"]
        if (
            isinstance(segment, bool) or not isinstance(segment, int) or segment < 0
            or isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
            or (segment, frame) in seen
        ):
            raise ValueError("invalid frame inventory identity")
        seen.add((segment, frame))
        source = _canonical_artifact(raw["source"])
        people = raw["person_ids"]
        if (
            not isinstance(people, list)
            or any(not isinstance(item, str) or not item for item in people)
            or people != sorted(set(people))
        ):
            raise ValueError("invalid frame inventory persons")
        result.append({
            "segment_index": segment,
            "frame_index": frame,
            "source": source,
            "person_ids": list(people),
        })
    if not result:
        raise ValueError("frame inventory is empty")
    return result


def mask_manifest_receipt(
    meta: Mapping[str, Any], *, plan_sha256: str,
    frame_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return a canonical, inventory-bound ``_image_edit_masks`` receipt or None.

    This is a pure receipt check.  ``load_frame_masks`` performs filesystem,
    PNG, dimensions, coverage and overlap validation.
    """
    if not isinstance(meta, Mapping) or not _valid_sha(plan_sha256):
        return None
    raw = meta.get("_image_edit_masks")
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "version", "plan_sha256", "frames", "sha256"
    }:
        return None
    try:
        inventory = _canonical_inventory(frame_inventory)
        if raw["schema"] != MASK_SCHEMA or raw["version"] != MASK_VERSION:
            return None
        if raw["plan_sha256"] != plan_sha256 or not _valid_sha(raw["sha256"]):
            return None
        frames = raw["frames"]
        if not isinstance(frames, list) or len(frames) != len(inventory):
            return None
        for item, expected in zip(frames, inventory):
            if not isinstance(item, dict) or set(item) != {
                "segment_index", "frame_index", "source", "persons", "scene",
                "protected_non_target",
            }:
                return None
            if (
                item["segment_index"] != expected["segment_index"]
                or item["frame_index"] != expected["frame_index"]
                or _canonical_artifact(item["source"]) != expected["source"]
            ):
                return None
            persons = item["persons"]
            if not isinstance(persons, list):
                return None
            canonical_people = []
            for person in persons:
                if not isinstance(person, dict) or set(person) != {
                    "person_id", "path", "sha256", "width", "height", "producer_receipt"
                }:
                    return None
                person_id = person["person_id"]
                if not isinstance(person_id, str) or not person_id:
                    return None
                canonical_people.append(person_id)
                _canonical_mask_artifact({key: value for key, value in person.items() if key != "person_id"})
            if canonical_people != expected["person_ids"]:
                return None
            _canonical_scene_artifact(item["scene"])
            _canonical_mask_artifact(item["protected_non_target"])
            source = expected["source"]
            for person in persons:
                _validate_person_producer_receipt(
                    person["producer_receipt"],
                    source=source,
                    artifact=person,
                    purpose="person",
                )
            _validate_person_producer_receipt(
                item["protected_non_target"]["producer_receipt"],
                source=source,
                artifact=item["protected_non_target"],
                purpose="protected_non_target_people",
            )
            _validate_scene_producer_receipt(
                item["scene"]["producer_receipt"],
                source=source,
                artifact=item["scene"],
                plan_sha256=plan_sha256,
            )
        unsigned = {key: value for key, value in raw.items() if key != "sha256"}
        if canonical_sha256(unsigned) != raw["sha256"]:
            return None
        return _json_roundtrip(raw)
    except (KeyError, TypeError, ValueError):
        return None


def _safe_project_file(project_dir: Path, relative: str) -> Path:
    if "\\" in relative:
        raise ValueError("mask path must be project-relative")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("mask path must be project-relative")
    candidate = project_dir.joinpath(*pure.parts)
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project_dir)
    except (OSError, ValueError):
        raise ValueError("mask path must be project-relative") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("mask path must reference a regular non-symlink file")
    return resolved


def _read_binary_mask(path: Path, artifact: Mapping[str, Any]):
    if cv2 is None or np is None:
        raise ValueError("mask dependency unavailable")
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("mask artifact must be PNG")
    if _file_sha256(path) != artifact["sha256"]:
        raise ValueError("mask artifact hash mismatch")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("mask artifact is not decodable")
    if image.ndim == 2:
        alpha = image
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
    else:
        raise ValueError("mask artifact must be grayscale-alpha or BGRA PNG")
    if alpha.shape != (artifact["height"], artifact["width"]):
        raise ValueError("mask artifact dimensions mismatch")
    count = int(np.count_nonzero(alpha))
    if count == 0:
        raise ValueError("mask artifact is empty")
    if count == int(alpha.size):
        raise ValueError("whole-frame mask is forbidden")
    receipt = artifact.get("producer_receipt")
    if isinstance(receipt, Mapping) and receipt.get("schema") == "duet.image-mask-producer":
        mask = receipt.get("mask")
        if (
            not isinstance(mask, Mapping)
            or mask.get("size") != path.stat().st_size
            or mask.get("alpha_nonzero_pixels") != count
            or mask.get("alpha_transparent_pixels") != int(alpha.size) - count
        ):
            raise ValueError("person mask alpha receipt mismatch")
    if "byte_size" in artifact and artifact["byte_size"] != path.stat().st_size:
        raise ValueError("scene mask byte size mismatch")
    return alpha > 0


def load_frame_masks(
    project_dir: Path, manifest: Mapping[str, Any],
    frame_inventory: Sequence[Mapping[str, Any]],
) -> list[FrameMasks]:
    """Load and validate project-relative, receipt-bound alpha PNG masks."""
    requested_root = Path(project_dir)
    try:
        root = requested_root.resolve(strict=True)
    except OSError:
        raise ValueError("invalid mask project directory") from None
    if not requested_root.is_absolute() or requested_root != root or not root.is_dir():
        raise ValueError("invalid mask project directory")
    inventory = _canonical_inventory(frame_inventory)
    canonical = mask_manifest_receipt(
        {"_image_edit_masks": manifest},
        plan_sha256=manifest.get("plan_sha256") if isinstance(manifest, Mapping) else "",
        frame_inventory=inventory,
    )
    if canonical is None:
        raise ValueError("invalid mask manifest receipt")

    result = []
    for frame, expected in zip(canonical["frames"], inventory):
        height = expected["source"]["height"]
        width = expected["source"]["width"]
        person_artifacts: dict[str, MaskArtifact] = {}
        target_arrays = []
        for raw in frame["persons"]:
            artifact_raw = {key: value for key, value in raw.items() if key != "person_id"}
            if (artifact_raw["width"], artifact_raw["height"]) != (width, height):
                raise ValueError("mask artifact dimensions mismatch")
            path = _safe_project_file(root, artifact_raw["path"])
            target_arrays.append(_read_binary_mask(path, artifact_raw))
            person_artifacts[raw["person_id"]] = MaskArtifact(
                path, artifact_raw["path"], artifact_raw["sha256"], width, height,
                artifact_raw["producer_receipt"],
            )
        scene_raw = frame["scene"]
        protected_raw = frame["protected_non_target"]
        if (
            (scene_raw["width"], scene_raw["height"]) != (width, height)
            or (protected_raw["width"], protected_raw["height"]) != (width, height)
        ):
            raise ValueError("mask artifact dimensions mismatch")
        scene_path = _safe_project_file(root, scene_raw["path"])
        protected_path = _safe_project_file(root, protected_raw["path"])
        scene_array = _read_binary_mask(scene_path, scene_raw)
        protected_array = _read_binary_mask(protected_path, protected_raw)
        target_arrays.append(scene_array)
        union = np.zeros((height, width), dtype=bool)
        for target in target_arrays:
            if np.any(union & target):
                raise ValueError("target masks overlap")
            union |= target
        if np.any(union & protected_array):
            raise ValueError("target and protected masks overlap")
        result.append(FrameMasks(
            version=MASK_VERSION,
            segment_index=frame["segment_index"],
            frame_index=frame["frame_index"],
            persons=person_artifacts,
            scene=MaskArtifact(
                scene_path, scene_raw["path"], scene_raw["sha256"], width, height,
                scene_raw["producer_receipt"],
            ),
            protected_non_target=MaskArtifact(
                protected_path, protected_raw["path"], protected_raw["sha256"], width, height,
                protected_raw["producer_receipt"],
            ),
            producer_receipt_sha256=canonical_sha256([
                *[raw["producer_receipt"] for raw in frame["persons"]],
                scene_raw["producer_receipt"],
                protected_raw["producer_receipt"],
            ]),
            manifest_sha256=canonical["sha256"],
        ))
    return result


@dataclass
class _Frame:
    segment_index: int
    frame_index: int
    source_path: Path
    output_path: Path
    source_sha256: str
    output_sha256: str
    source: Any
    output: Any
    persons: Mapping[str, Any] | None
    scene: Any | None
    protected: Any | None


@dataclass
class _Context:
    plan: Mapping[str, Any]
    frames: list[_Frame]
    masks_available: bool


def _read_color(path: Path):
    if cv2 is None or np is None:
        raise RuntimeError("dependency_unavailable")
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError:
        raise ValueError("input_frame_unreadable") from None
    if not resolved.is_file():
        raise ValueError("input_frame_unreadable")
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("input_frame_undecodable")
    return resolved, image


def _load_mask(artifact: MaskArtifact, shape: tuple[int, int]):
    if artifact.width != shape[1] or artifact.height != shape[0]:
        raise ValueError("input_mask_dimension_mismatch")
    if _file_sha256(artifact.path) != artifact.sha256:
        raise ValueError("input_mask_hash_mismatch")
    if cv2 is None or np is None:
        raise RuntimeError("dependency_unavailable")
    image = cv2.imread(str(artifact.path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise ValueError("input_mask_dimension_mismatch")
    values = set(int(value) for value in np.unique(image))
    if not values.issubset({0, 255}):
        raise ValueError("input_mask_not_binary")
    mask = image > 0
    if not np.any(mask) or np.all(mask):
        raise ValueError("input_mask_invalid_coverage")
    return mask


def _mask_unknown(name: str, version: str) -> GateResult:
    return GateResult(name, version, "unknown", "input_masks_missing", {})


def _lab(image):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    converted[..., 0] *= 100.0 / 255.0
    converted[..., 1:] -= 128.0
    return converted


def _appearance(image, mask=None) -> dict[str, float]:
    pixels = image if mask is None else image[mask]
    lab_pixels = _lab(image).reshape(-1, 3) if mask is None else _lab(image)[mask]
    bgr = pixels.reshape(-1, 3).astype(np.float32)
    l_values = lab_pixels[:, 0]
    mean_b = float(np.mean(bgr[:, 0]))
    mean_r = float(np.mean(bgr[:, 2]))
    return {
        "exposure_l": float(np.mean(l_values)),
        "contrast_l": float(np.std(l_values)),
        "white_a": float(np.mean(lab_pixels[:, 1])),
        "white_b": float(np.mean(lab_pixels[:, 2])),
        "cct_proxy": float(math.log((mean_r + 1.0) / (mean_b + 1.0))),
    }


def _appearance_delta(source: dict[str, float], output: dict[str, float]) -> dict[str, float]:
    return {
        "exposure_delta_l": abs(output["exposure_l"] - source["exposure_l"]),
        "white_point_delta_e": math.hypot(
            output["white_a"] - source["white_a"],
            output["white_b"] - source["white_b"],
        ),
        "contrast_relative_delta": abs(output["contrast_l"] - source["contrast_l"])
        / max(source["contrast_l"], 1.0),
        "cct_proxy_delta": abs(output["cct_proxy"] - source["cct_proxy"]),
    }


class DimensionsGate:
    name = "dimensions"
    version = "dimensions-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        metrics = []
        for frame in context.frames:
            source_hw = [int(frame.source.shape[0]), int(frame.source.shape[1])]
            output_hw = [int(frame.output.shape[0]), int(frame.output.shape[1])]
            metrics.append({"frame_index": frame.frame_index, "source_hw": source_hw, "output_hw": output_hw})
            if source_hw != output_hw:
                return GateResult(self.name, self.version, "fail", "dimension_mismatch", {"frames": metrics})
        return GateResult(self.name, self.version, "pass", None, {"frames": metrics})


class GlobalAppearanceGate:
    name = "global_appearance"
    version = "global-appearance-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        rows = [_appearance_delta(_appearance(frame.source), _appearance(frame.output)) for frame in context.frames]
        checks = (
            ("exposure_delta_l", "max_global_exposure_delta_l", "global_exposure_drift"),
            ("white_point_delta_e", "max_global_white_point_delta_e", "global_white_point_drift"),
            ("contrast_relative_delta", "max_global_contrast_relative_delta", "global_contrast_drift"),
            ("cct_proxy_delta", "max_global_cct_proxy_delta", "global_color_temperature_drift"),
        )
        for metric, threshold, code in checks:
            if max(row[metric] for row in rows) > profile.thresholds[threshold]:
                return GateResult(self.name, self.version, "fail", code, {"frames": rows})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


class ProtectedAppearanceGate:
    name = "protected_appearance"
    version = "protected-appearance-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        if not context.masks_available:
            return _mask_unknown(self.name, self.version)
        rows = []
        for frame in context.frames:
            if frame.protected is None or frame.output.shape[:2] != frame.source.shape[:2]:
                return GateResult(self.name, self.version, "unknown", "input_mask_dimension_mismatch", {})
            rows.append(_appearance_delta(
                _appearance(frame.source, frame.protected), _appearance(frame.output, frame.protected)
            ))
        checks = (
            ("exposure_delta_l", "max_protected_exposure_delta_l", "protected_exposure_drift"),
            ("white_point_delta_e", "max_protected_white_point_delta_e", "protected_white_point_drift"),
            ("contrast_relative_delta", "max_protected_contrast_relative_delta", "protected_contrast_drift"),
            ("cct_proxy_delta", "max_protected_cct_proxy_delta", "protected_color_temperature_drift"),
        )
        for metric, threshold, code in checks:
            if max(row[metric] for row in rows) > profile.thresholds[threshold]:
                return GateResult(self.name, self.version, "fail", code, {"frames": rows})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


class LocalPerceptualColorGate:
    name = "local_perceptual_color"
    version = "delta-e76-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        if not context.masks_available:
            return _mask_unknown(self.name, self.version)
        person_seen: set[str] = set()
        rows = []
        min_pixels = int(profile.thresholds["min_mask_pixels"])
        for frame in context.frames:
            if frame.output.shape[:2] != frame.source.shape[:2] or frame.scene is None or frame.persons is None:
                return GateResult(self.name, self.version, "unknown", "input_mask_dimension_mismatch", {})
            delta = np.linalg.norm(_lab(frame.source) - _lab(frame.output), axis=2)
            if int(np.count_nonzero(frame.scene)) < min_pixels:
                return GateResult(self.name, self.version, "unknown", "scene_mask_too_small", {})
            scene_delta = float(np.median(delta[frame.scene]))
            if scene_delta < profile.thresholds["min_scene_local_delta_e"]:
                return GateResult(self.name, self.version, "fail", "scene_local_color_change_too_small", {"frames": rows + [{"scene_delta_e": scene_delta}]})
            people = {}
            for person_id, mask in sorted(frame.persons.items()):
                person_seen.add(person_id)
                if int(np.count_nonzero(mask)) < min_pixels:
                    return GateResult(self.name, self.version, "unknown", "person_mask_too_small", {})
                value = float(np.median(delta[mask]))
                people[person_id] = value
                if value < profile.thresholds["min_person_local_delta_e"]:
                    return GateResult(self.name, self.version, "fail", "person_local_color_change_too_small", {"frames": rows + [{"persons": people}]})
            rows.append({"frame_index": frame.frame_index, "scene_delta_e": scene_delta, "persons": people})
        expected = {
            item.get("id") for item in context.plan.get("person_plans", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if expected and not expected.issubset(person_seen):
            return GateResult(self.name, self.version, "unknown", "person_masks_missing", {"missing_person_ids": sorted(expected - person_seen)})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


def _edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 80, 160) > 0


class SceneEdgeChangeGate:
    name = "scene_edge_change"
    version = "scene-edge-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        if not context.masks_available:
            return _mask_unknown(self.name, self.version)
        rows = []
        for frame in context.frames:
            if frame.scene is None or frame.output.shape[:2] != frame.source.shape[:2]:
                return GateResult(self.name, self.version, "unknown", "input_mask_dimension_mismatch", {})
            source = _edges(frame.source) & frame.scene
            output = _edges(frame.output) & frame.scene
            union_count = int(np.count_nonzero(source | output))
            if union_count == 0:
                return GateResult(self.name, self.version, "unknown", "scene_edge_metric_unavailable", {})
            ratio = float(np.count_nonzero(source ^ output) / union_count)
            rows.append({"frame_index": frame.frame_index, "change_ratio": ratio})
            if ratio < profile.thresholds["min_scene_edge_change_ratio"]:
                return GateResult(self.name, self.version, "fail", "scene_edge_change_too_small", {"frames": rows})
            if ratio > profile.thresholds["max_scene_edge_change_ratio"]:
                return GateResult(self.name, self.version, "fail", "scene_edge_change_too_large", {"frames": rows})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


class ProtectedStructureGate:
    name = "protected_structure"
    version = "protected-edge-iou-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        if not context.masks_available:
            return _mask_unknown(self.name, self.version)
        rows = []
        for frame in context.frames:
            if frame.protected is None or frame.output.shape[:2] != frame.source.shape[:2]:
                return GateResult(self.name, self.version, "unknown", "input_mask_dimension_mismatch", {})
            source = _edges(frame.source) & frame.protected
            output = _edges(frame.output) & frame.protected
            union = int(np.count_nonzero(source | output))
            iou = 1.0 if union == 0 else float(np.count_nonzero(source & output) / union)
            rows.append({"frame_index": frame.frame_index, "edge_iou": iou})
            if iou < profile.thresholds["min_protected_edge_iou"]:
                return GateResult(self.name, self.version, "fail", "protected_structure_drift", {"frames": rows})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


def _saliency_centroid(image, mask) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    weight = np.hypot(gx, gy) * mask.astype(np.float32)
    total = float(np.sum(weight))
    if total <= 1e-6:
        return 0.5, 0.5
    yy, xx = np.indices(weight.shape)
    return float(np.sum(xx * weight) / total / max(weight.shape[1] - 1, 1)), float(
        np.sum(yy * weight) / total / max(weight.shape[0] - 1, 1)
    )


class CompositionGate:
    name = "composition"
    version = "protected-saliency-v1"

    def evaluate(self, context: _Context, profile: QualityProfile) -> GateResult:
        if not context.masks_available:
            return _mask_unknown(self.name, self.version)
        rows = []
        for frame in context.frames:
            if frame.protected is None or frame.output.shape[:2] != frame.source.shape[:2]:
                return GateResult(self.name, self.version, "unknown", "input_mask_dimension_mismatch", {})
            source = _saliency_centroid(frame.source, frame.protected)
            output = _saliency_centroid(frame.output, frame.protected)
            shift = math.hypot(source[0] - output[0], source[1] - output[1])
            rows.append({"frame_index": frame.frame_index, "centroid_shift": shift})
            if shift > profile.thresholds["max_composition_centroid_shift"]:
                return GateResult(self.name, self.version, "fail", "composition_drift", {"frames": rows})
        return GateResult(self.name, self.version, "pass", None, {"frames": rows})


DEFAULT_GATES: tuple[QualityGate, ...] = (
    DimensionsGate(),
    GlobalAppearanceGate(),
    ProtectedAppearanceGate(),
    LocalPerceptualColorGate(),
    SceneEdgeChangeGate(),
    ProtectedStructureGate(),
    CompositionGate(),
)


def _semantic_missing() -> SemanticVerdict:
    return SemanticVerdict("unknown", "semantic_verifier_missing", {}, None)


def _receipt_status(gates: Sequence[GateResult], semantic: SemanticVerdict) -> str:
    statuses = [item.status for item in gates] + [semantic.status]
    if "fail" in statuses:
        return "fail"
    if "unknown" in statuses:
        return "unknown"
    return "pass"


def evaluate_image_quality(
    plan: Mapping[str, Any],
    source_frames: Sequence[Path],
    output_frames: Sequence[Path],
    *,
    frame_masks: Sequence[FrameMasks] | None,
    profile: QualityProfile,
    semantic_verifier: SemanticVerifier | None,
    gates: Sequence[QualityGate] | None = None,
    receipt_path: Path | None = None,
) -> QualityReceipt:
    """Evaluate source/output frames.  No failure in this function is retryable.

    ``frame_masks=None`` is accepted only for staged integration.  Local and
    structure gates become ``unknown`` and the result cannot be published.
    """
    if not isinstance(profile, QualityProfile):
        raise TypeError("a versioned QualityProfile is required")
    if not isinstance(plan, Mapping) or plan.get("version") != 2 or plan.get("phase") != "plan":
        raise ValueError("quality evaluation requires a v2 plan")
    canonical_plan = _json_roundtrip(dict(plan))
    plan_hash = _plan_sha256(canonical_plan)
    if (
        not isinstance(source_frames, Sequence) or isinstance(source_frames, (str, bytes))
        or not isinstance(output_frames, Sequence) or isinstance(output_frames, (str, bytes))
        or not source_frames or len(source_frames) != len(output_frames)
    ):
        raise ValueError("source/output frame inventory mismatch")
    if frame_masks is not None and len(frame_masks) != len(source_frames):
        raise ValueError("frame mask inventory mismatch")

    loaded = []
    manifest_hashes = set()
    for index, (source_raw, output_raw) in enumerate(zip(source_frames, output_frames), 1):
        source_path, source = _read_color(Path(source_raw))
        output_path, output = _read_color(Path(output_raw))
        masks = None if frame_masks is None else frame_masks[index - 1]
        persons = scene = protected = None
        segment_index = 0
        frame_index = index
        if masks is not None:
            segment_index = masks.segment_index
            frame_index = masks.frame_index
            manifest_hashes.add(masks.manifest_sha256)
            shape = source.shape[:2]
            persons = {key: _load_mask(value, shape) for key, value in masks.persons.items()}
            scene = _load_mask(masks.scene, shape)
            declared_protected_people = _load_mask(masks.protected_non_target, shape)
            union = scene.copy()
            for mask in persons.values():
                if np.any(union & mask):
                    raise ValueError("target masks overlap")
                union |= mask
            if np.any(union & declared_protected_people):
                raise ValueError("target and protected masks overlap")
            # The gateway mask identifies protected people.  The deterministic
            # non-target domain is the complete complement of all editable
            # person/scene targets, so core props and interaction objects are
            # covered as well instead of silently disappearing from the gates.
            protected = ~union
        loaded.append(_Frame(
            segment_index, frame_index, source_path, output_path,
            _file_sha256(source_path), _file_sha256(output_path),
            source, output, persons, scene, protected,
        ))
    if len(manifest_hashes) > 1:
        raise ValueError("frame masks come from different manifests")

    context = _Context(canonical_plan, loaded, frame_masks is not None)
    selected_gates = DEFAULT_GATES if gates is None else tuple(gates)
    gate_results: list[GateResult] = []
    for gate in selected_gates:
        if not hasattr(gate, "evaluate") or not isinstance(getattr(gate, "name", None), str):
            raise TypeError("invalid quality gate")
        try:
            result = gate.evaluate(context, profile)
        except Exception:
            result = GateResult(
                getattr(gate, "name", "invalid_gate"),
                getattr(gate, "version", "unknown"),
                "unknown", "deterministic_gate_error", {},
            )
        if not isinstance(result, GateResult):
            raise TypeError("quality gate must return GateResult")
        gate_results.append(result)

    if semantic_verifier is None:
        semantic = _semantic_missing()
    else:
        try:
            semantic = semantic_verifier.verify(
                canonical_plan,
                source_frames,
                output_frames,
                deterministic_metrics={
                    "algorithm_version": ALGORITHM_VERSION,
                    "profile": profile.to_dict(),
                    "gates": [item.to_dict() for item in gate_results],
                },
            )
            if not isinstance(semantic, SemanticVerdict):
                raise TypeError("semantic verifier returned invalid value")
        except Exception:
            semantic = SemanticVerdict("unknown", "semantic_verifier_unavailable", {}, None)

    status = _receipt_status(gate_results, semantic)
    failures = [
        {"code": item.code or "deterministic_gate_error", "gate": item.name}
        for item in gate_results if item.status != "pass"
    ]
    if semantic.status != "pass":
        failures.append({"code": semantic.code or "semantic_unknown", "gate": "semantic"})
    frames_payload = tuple({
        "segment_index": item.segment_index,
        "frame_index": item.frame_index,
        "source_sha256": item.source_sha256,
        "output_sha256": item.output_sha256,
        "width": int(item.source.shape[1]),
        "height": int(item.source.shape[0]),
    } for item in loaded)
    receipt = QualityReceipt(
        plan_sha256=plan_hash,
        mask_manifest_sha256=next(iter(manifest_hashes)) if manifest_hashes else None,
        control_mode="pixel_masks" if frame_masks is not None else "soft_control",
        profile=profile.to_dict(),
        status=status,
        publishable=status == "pass",
        frames=frames_payload,
        gates=tuple(gate_results),
        semantic=semantic,
        failures=tuple(failures),
    )
    if receipt_path is not None:
        _write_receipt(Path(receipt_path), receipt.to_dict())
    return receipt


def _reference_target_ids(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    people = plan.get("person_plans")
    scenes = plan.get("scene_plans")
    if not isinstance(people, list) or not isinstance(scenes, list):
        raise ValueError("reference packs require person and scene plans")
    person_ids = tuple(item.get("id") for item in people if isinstance(item, Mapping))
    scene_ids = tuple(item.get("id") for item in scenes if isinstance(item, Mapping))
    if (
        not person_ids or not scene_ids
        or any(not isinstance(item, str) or not item for item in (*person_ids, *scene_ids))
        or len(set((*person_ids, *scene_ids))) != len(person_ids) + len(scene_ids)
    ):
        raise ValueError("reference pack plan targets are invalid")
    return person_ids, scene_ids


def _reference_checks(person_ids: Sequence[str], scene_ids: Sequence[str]) -> set[str]:
    checks = {
        f"person.{person_id}.{check}"
        for person_id in person_ids
        for check in (
            "identity_changed", "source_identity_absent", "multi_view_consistency",
            "local_color_change",
        )
    }
    checks.update({
        f"scene.{scene_id}.{check}"
        for scene_id in scene_ids
        for check in (
            "semantic_change", "geometry_change", "depth_change", "layout_change",
            "local_color_change",
        )
    })
    checks.update({
        "project.global_light_direction_preservation",
        "project.global_exposure_preservation",
        "project.global_wb_cct_preservation",
        "project.global_tone_curve_preservation",
    })
    return checks


def evaluate_reference_packs(
    plan: Mapping[str, Any],
    source_slots: Mapping[str, Sequence[Path]],
    generated_packs: Mapping[str, Sequence[Path]],
    *,
    mask_manifest: Mapping[str, Any],
    profile: QualityProfile,
    semantic_verifier: ReferencePackSemanticVerifier | None,
    receipt_path: Path | None = None,
) -> QualityReceipt:
    """Gate generated target packs before any per-frame paid image-edit POST.

    Source slots are evidence only.  Passing this function never authorizes a
    caller to submit them as provider identity/scene references.
    """
    if not isinstance(profile, QualityProfile):
        raise TypeError("a versioned QualityProfile is required")
    if not isinstance(plan, Mapping) or plan.get("version") != 2 or plan.get("phase") != "plan":
        raise ValueError("reference packs require a v2 plan")
    canonical_plan = _json_roundtrip(dict(plan))
    plan_hash = _plan_sha256(canonical_plan)
    if (
        not isinstance(mask_manifest, Mapping)
        or set(mask_manifest) != {"schema", "version", "plan_sha256", "frames", "sha256"}
        or mask_manifest.get("schema") != MASK_SCHEMA
        or mask_manifest.get("version") != MASK_VERSION
        or mask_manifest.get("plan_sha256") != plan_hash
        or not _valid_sha(mask_manifest.get("sha256"))
        or not isinstance(mask_manifest.get("frames"), list)
        or not mask_manifest["frames"]
    ):
        raise ValueError("reference packs require a plan-bound mask manifest")
    if canonical_sha256({
        key: value for key, value in mask_manifest.items() if key != "sha256"
    }) != mask_manifest["sha256"]:
        raise ValueError("reference packs require a plan-bound mask manifest")
    person_ids, scene_ids = _reference_target_ids(canonical_plan)
    target_ids = set((*person_ids, *scene_ids))
    if not isinstance(source_slots, Mapping) or not isinstance(generated_packs, Mapping):
        raise ValueError("invalid reference pack inventory")
    if set(source_slots) != target_ids or set(generated_packs) != target_ids:
        raise ValueError("reference pack targets do not match the frozen plan")

    inventory_failures = []
    pairs: list[tuple[str, int, Path, Path, Any, Any]] = []
    frames_payload = []
    appearance_rows = []
    local_rows = []
    scene_edge_rows = []
    ordinal = 0
    for target_id in (*person_ids, *scene_ids):
        sources = source_slots[target_id]
        outputs = generated_packs[target_id]
        if (
            not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources
            or not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or not outputs
        ):
            inventory_failures.append("reference_pack_missing")
            continue
        if target_id in person_ids and len(outputs) < 2:
            inventory_failures.append("reference_person_multiview_missing")
        if target_id in person_ids:
            try:
                output_hashes = {
                    _file_sha256(Path(item).resolve(strict=True)) for item in outputs
                }
            except OSError:
                raise ValueError("reference pack image is unreadable") from None
            if len(output_hashes) != len(outputs):
                inventory_failures.append("reference_person_multiview_duplicate")
        for view_index, output_raw in enumerate(outputs, 1):
            source_raw = sources[(view_index - 1) % len(sources)]
            source_path, source = _read_color(Path(source_raw))
            output_path, output = _read_color(Path(output_raw))
            ordinal += 1
            frames_payload.append({
                "segment_index": 0,
                "frame_index": ordinal,
                "source_sha256": _file_sha256(source_path),
                "output_sha256": _file_sha256(output_path),
                "width": int(source.shape[1]),
                "height": int(source.shape[0]),
            })
            pairs.append((target_id, view_index, source_path, output_path, source, output))
            if source.shape != output.shape:
                inventory_failures.append("reference_pack_dimension_mismatch")
                continue
            appearance = _appearance_delta(_appearance(source), _appearance(output))
            appearance_rows.append({"target_id": target_id, "view_index": view_index, **appearance})
            delta_e = float(np.median(np.linalg.norm(_lab(source) - _lab(output), axis=2)))
            local_rows.append({"target_id": target_id, "view_index": view_index, "delta_e": delta_e})
            if target_id in scene_ids:
                source_edges = _edges(source)
                output_edges = _edges(output)
                union = int(np.count_nonzero(source_edges | output_edges))
                change = None if union == 0 else float(np.count_nonzero(source_edges ^ output_edges) / union)
                scene_edge_rows.append({"target_id": target_id, "view_index": view_index, "change_ratio": change})

    gates = []
    if inventory_failures:
        gates.append(GateResult(
            "reference_pack_inventory", "reference-inventory-v1", "fail",
            sorted(set(inventory_failures))[0], {"codes": sorted(set(inventory_failures))},
        ))
    else:
        gates.append(GateResult(
            "reference_pack_inventory", "reference-inventory-v1", "pass", None,
            {"person_views_min": 2, "targets": sorted(target_ids)},
        ))

    appearance_code = None
    appearance_checks = (
        ("exposure_delta_l", "max_global_exposure_delta_l", "reference_pack_exposure_drift"),
        ("white_point_delta_e", "max_global_white_point_delta_e", "reference_pack_white_point_drift"),
        ("contrast_relative_delta", "max_global_contrast_relative_delta", "reference_pack_contrast_drift"),
        ("cct_proxy_delta", "max_global_cct_proxy_delta", "reference_pack_color_temperature_drift"),
    )
    for metric, threshold, code in appearance_checks:
        if appearance_rows and max(row[metric] for row in appearance_rows) > profile.thresholds[threshold]:
            appearance_code = code
            break
    gates.append(GateResult(
        "reference_pack_global_appearance", "reference-appearance-v1",
        "fail" if appearance_code else "pass", appearance_code,
        {"views": appearance_rows},
    ))

    local_code = None
    for row in local_rows:
        minimum = profile.thresholds[
            "min_person_local_delta_e" if row["target_id"] in person_ids
            else "min_scene_local_delta_e"
        ]
        if row["delta_e"] < minimum:
            local_code = "reference_pack_local_color_change_too_small"
            break
    gates.append(GateResult(
        "reference_pack_local_color", "reference-delta-e76-v1",
        "fail" if local_code else "pass", local_code, {"views": local_rows},
    ))

    edge_code = None
    if any(row["change_ratio"] is None for row in scene_edge_rows):
        edge_status = "unknown"
        edge_code = "reference_scene_edge_metric_unavailable"
    elif scene_edge_rows and any(
        row["change_ratio"] < profile.thresholds["min_scene_edge_change_ratio"]
        for row in scene_edge_rows
    ):
        edge_status = "fail"
        edge_code = "reference_scene_edge_change_too_small"
    else:
        edge_status = "pass"
    gates.append(GateResult(
        "reference_pack_scene_edge_change", "reference-scene-edge-v1",
        edge_status, edge_code, {"views": scene_edge_rows},
    ))

    metrics = {
        "algorithm_version": ALGORITHM_VERSION,
        "profile": profile.to_dict(),
        "gates": [item.to_dict() for item in gates],
    }
    if semantic_verifier is None:
        semantic = SemanticVerdict("unknown", "semantic_reference_pack_verifier_missing", {}, None)
    else:
        try:
            semantic = semantic_verifier.verify_reference_packs(
                canonical_plan, source_slots, generated_packs,
                deterministic_metrics=metrics,
            )
            expected_checks = _reference_checks(person_ids, scene_ids)
            if not isinstance(semantic, SemanticVerdict) or set(semantic.checks) != expected_checks:
                raise ValueError("invalid reference pack semantic checks")
            values = list(semantic.checks.values())
            derived_semantic = (
                "unknown" if "unknown" in values else "fail" if "fail" in values else "pass"
            )
            if semantic.status != derived_semantic or any(
                value == "not_applicable" for value in values
            ):
                raise ValueError("inconsistent reference pack semantic verdict")
        except Exception:
            semantic = SemanticVerdict("unknown", "semantic_reference_pack_verdict_invalid", {}, None)
    status = _receipt_status(gates, semantic)
    failures = [
        {"code": item.code or "reference_pack_gate_error", "gate": item.name}
        for item in gates if item.status != "pass"
    ]
    if semantic.status != "pass":
        failures.append({"code": semantic.code or "semantic_unknown", "gate": "semantic"})
    receipt = QualityReceipt(
        plan_sha256=plan_hash,
        mask_manifest_sha256=mask_manifest["sha256"],
        control_mode="reference_packs",
        profile=profile.to_dict(),
        status=status,
        publishable=status == "pass",
        frames=tuple(frames_payload),
        gates=tuple(gates),
        semantic=semantic,
        failures=tuple(failures),
    )
    if receipt_path is not None:
        _write_receipt(Path(receipt_path), receipt.to_dict())
    return receipt


def quality_receipt(
    path: Path,
    *,
    plan_sha256: str,
    mask_manifest_sha256: str | None,
    source_frames: Sequence[Path],
    output_frames: Sequence[Path],
) -> dict[str, Any] | None:
    """Re-read and bind a durable quality receipt before publication."""
    if not _valid_sha(plan_sha256) or (
        mask_manifest_sha256 is not None and not _valid_sha(mask_manifest_sha256)
    ) or len(source_frames) != len(output_frames) or not source_frames:
        return None
    requested = Path(path)
    try:
        if not requested.is_absolute():
            return None
        info = requested.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
            return None
        raw = json.loads(requested.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or set(raw) != {
        "schema", "version", "algorithm_version", "plan_sha256",
        "mask_manifest_sha256", "control_mode", "profile", "status",
        "publishable", "provider_retry_allowed", "frames", "gates", "semantic",
        "failures", "sha256",
    }:
        return None
    try:
        if (
            raw["schema"] != RECEIPT_SCHEMA or raw["version"] != RECEIPT_VERSION
            or raw["algorithm_version"] != ALGORITHM_VERSION
            or raw["plan_sha256"] != plan_sha256
            or raw["mask_manifest_sha256"] != mask_manifest_sha256
            or raw["control_mode"] not in {"pixel_masks", "soft_control", "reference_packs"}
            or raw["status"] not in {"pass", "fail", "unknown"}
            or not isinstance(raw["publishable"], bool)
            or raw["provider_retry_allowed"] is not False
            or not _valid_sha(raw["sha256"])
        ):
            return None
        unsigned = {key: value for key, value in raw.items() if key != "sha256"}
        if canonical_sha256(unsigned) != raw["sha256"]:
            return None
        profile = raw["profile"]
        if not isinstance(profile, dict) or set(profile) != {
            "name", "version", "calibration", "thresholds", "sha256"
        }:
            return None
        profile_unsigned = {key: value for key, value in profile.items() if key != "sha256"}
        if not _valid_sha(profile["sha256"]) or canonical_sha256(profile_unsigned) != profile["sha256"]:
            return None
        frames = raw["frames"]
        if not isinstance(frames, list) or len(frames) != len(source_frames):
            return None
        for item, source, output in zip(frames, source_frames, output_frames):
            if not isinstance(item, dict) or set(item) != {
                "segment_index", "frame_index", "source_sha256", "output_sha256",
                "width", "height",
            }:
                return None
            if item["source_sha256"] != _file_sha256(Path(source).resolve(strict=True)):
                return None
            if item["output_sha256"] != _file_sha256(Path(output).resolve(strict=True)):
                return None
        gates = raw["gates"]
        if not isinstance(gates, list):
            return None
        gate_statuses = []
        expected_failures = []
        for gate in gates:
            if not isinstance(gate, dict) or set(gate) != {
                "name", "version", "status", "code", "metrics"
            } or gate["status"] not in {"pass", "fail", "unknown"}:
                return None
            if (gate["status"] == "pass") != (gate["code"] is None):
                return None
            gate_statuses.append(gate["status"])
            if gate["status"] != "pass":
                expected_failures.append({"code": gate["code"], "gate": gate["name"]})
        semantic = raw["semantic"]
        if not isinstance(semantic, dict) or set(semantic) != {
            "status", "code", "checks", "verdict_sha256"
        } or semantic["status"] not in {"pass", "fail", "unknown"}:
            return None
        if (semantic["status"] == "pass") != (semantic["code"] is None):
            return None
        statuses = gate_statuses + [semantic["status"]]
        derived = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"
        if raw["status"] != derived or raw["publishable"] != (derived == "pass"):
            return None
        if semantic["status"] != "pass":
            expected_failures.append({"code": semantic["code"], "gate": "semantic"})
        if raw["failures"] != expected_failures:
            return None
        return _json_roundtrip(raw)
    except (OSError, TypeError, ValueError):
        return None


def _copy_regular(source: Path, destination: Path) -> None:
    try:
        info = source.lstat()
    except OSError:
        raise ValueError("semantic verifier input is unreadable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_STAGE_IMAGE_BYTES:
        raise ValueError("semantic verifier input must be a bounded regular file")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _read_bounded_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError
        if info.st_size > _MAX_SEMANTIC_OUTPUT_BYTES:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("semantic verdict invalid") from None
    if not isinstance(value, dict):
        raise ValueError("semantic verdict invalid")
    return value


def _check_object(raw: Any) -> tuple[str, str]:
    if not isinstance(raw, dict) or set(raw) != {"status", "evidence"}:
        raise ValueError("semantic verdict invalid")
    if raw["status"] not in {"pass", "fail", "unknown", "not_applicable"}:
        raise ValueError("semantic verdict invalid")
    if not isinstance(raw["evidence"], str) or not raw["evidence"] or len(raw["evidence"]) > 4000:
        raise ValueError("semantic verdict invalid")
    return raw["status"], raw["evidence"]


_PROJECT_CHECKS = (
    "narrative_person_completeness",
    "no_identity_swap",
    "no_unplanned_person",
    "person_identity_continuity",
    "scene_continuity",
)
_SCENE_CHECKS = (
    "semantic_change", "geometry_change", "depth_change", "layout_change",
    "local_color_change",
)
_INVARIANT_CHECKS = (
    "lighting_preservation", "interaction_preservation", "cross_frame_continuity"
)


def _semantic_failure_code(name: str, status: str) -> str:
    base = {
        "identity_changed": "semantic_person_identity_not_changed",
        "source_identity_absent": "semantic_source_identity_residual",
        "person_local_color_change": "semantic_person_local_color_failed",
        "semantic_change": "semantic_scene_semantic_change_failed",
        "geometry_change": "semantic_scene_geometry_change_failed",
        "depth_change": "semantic_scene_depth_change_failed",
        "layout_change": "semantic_scene_layout_change_failed",
        "scene_local_color_change": "semantic_scene_local_color_failed",
        "lighting_preservation": "semantic_global_lighting_failed",
        "interaction_preservation": "semantic_interaction_or_occlusion_failed",
        "cross_frame_continuity": "semantic_cross_frame_continuity_failed",
        "narrative_person_completeness": "semantic_narrative_person_missing",
        "no_identity_swap": "semantic_identity_swap",
        "no_unplanned_person": "semantic_unplanned_person",
        "person_identity_continuity": "semantic_person_continuity_failed",
        "scene_continuity": "semantic_scene_continuity_failed",
    }.get(name, "semantic_verification_failed")
    return base if status == "fail" else f"{base}_unknown"


def _expected_segment_people(plan: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    result = {}
    segments = plan.get("segments")
    if not isinstance(segments, list):
        raise ValueError("semantic verdict invalid")
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("segment_index"), int):
            raise ValueError("semantic verdict invalid")
        people = segment.get("persons")
        if not isinstance(people, list):
            raise ValueError("semantic verdict invalid")
        mapped = {}
        for person in people:
            if not isinstance(person, Mapping) or not {"id", "state"}.issubset(person):
                raise ValueError("semantic verdict invalid")
            person_id = person.get("id")
            state = person.get("state")
            if not isinstance(person_id, str) or state not in {"replace", "not_observable"}:
                raise ValueError("semantic verdict invalid")
            mapped[person_id] = state
        result[segment["segment_index"]] = mapped
    return result


def _semantic_reason(checks: Mapping[str, str]) -> str | None:
    if "unknown" in checks.values():
        return "verification_unknown"
    ordered = (
        ("project.narrative_person_completeness", "narrative_person_incomplete"),
        ("project.no_identity_swap", "identity_swap_detected"),
        ("project.no_unplanned_person", "unplanned_person_detected"),
    )
    for name, reason in ordered:
        if checks.get(name) == "fail":
            return reason
    if any(
        value == "fail" and (name.endswith(".identity_changed") or name.endswith(".source_identity_absent"))
        for name, value in checks.items()
    ):
        return "person_replacement_failed"
    for suffix, reason in (
        (".scene.semantic_change", "scene_semantic_change_failed"),
        (".scene.geometry_change", "scene_geometry_change_failed"),
        (".scene.depth_change", "scene_depth_change_failed"),
        (".scene.layout_change", "scene_layout_change_failed"),
    ):
        if any(name.endswith(suffix) and value == "fail" for name, value in checks.items()):
            return reason
    if any(name.endswith(".local_color_change") and value == "fail" for name, value in checks.items()):
        return "local_color_change_failed"
    for suffix, reason in (
        (".lighting_preservation", "lighting_preservation_failed"),
        (".interaction_preservation", "interaction_preservation_failed"),
        (".cross_frame_continuity", "cross_frame_continuity_failed"),
    ):
        if any(name.endswith(suffix) and value == "fail" for name, value in checks.items()):
            return reason
    if checks.get("project.person_identity_continuity") == "fail":
        return "person_identity_continuity_failed"
    if checks.get("project.scene_continuity") == "fail":
        return "scene_continuity_failed"
    return None


def _parse_semantic_verdict(
    raw: Mapping[str, Any], plan: Mapping[str, Any], plan_hash: str
) -> SemanticVerdict:
    if set(raw) != {
        "version", "phase", "plan_sha256", "segment_indices", "passed", "reason",
        "segments", "project_checks",
    }:
        raise ValueError("semantic verdict invalid")
    if (
        raw["version"] != SEMANTIC_VERSION or raw["phase"] != "verify"
        or raw["plan_sha256"] != plan_hash or not isinstance(raw["passed"], bool)
        or (raw["reason"] is not None and (
            not isinstance(raw["reason"], str) or not raw["reason"] or len(raw["reason"]) > 256
        ))
        or raw["segment_indices"] != plan.get("segment_indices")
    ):
        raise ValueError("semantic verdict invalid")
    expected_people = _expected_segment_people(plan)
    segments = raw["segments"]
    if not isinstance(segments, list) or [item.get("segment_index") for item in segments if isinstance(item, dict)] != raw["segment_indices"]:
        raise ValueError("semantic verdict invalid")
    checks: dict[str, str] = {}
    nonpasses: list[tuple[str, str]] = []

    def record(
        name: str, value: Any, *, allow_na: bool = False, code_key: str | None = None
    ) -> None:
        status, _evidence = _check_object(value)
        if status == "not_applicable" and not allow_na:
            raise ValueError("semantic verdict invalid")
        checks[name] = status
        if status in {"fail", "unknown"}:
            nonpasses.append((code_key or name.rsplit(".", 1)[-1], status))

    for segment in segments:
        if set(segment) != {
            "segment_index", "passed", "person_checks", "scene_checks", "invariants"
        } or not isinstance(segment["passed"], bool):
            raise ValueError("semantic verdict invalid")
        index = segment["segment_index"]
        person_checks = segment["person_checks"]
        if not isinstance(person_checks, list):
            raise ValueError("semantic verdict invalid")
        expected = expected_people.get(index)
        if expected is None or [item.get("person_id") for item in person_checks if isinstance(item, dict)] != list(expected):
            raise ValueError("semantic verdict invalid")
        segment_statuses = []
        for person in person_checks:
            if set(person) != {
                "person_id", "identity_changed", "source_identity_absent", "local_color_change"
            }:
                raise ValueError("semantic verdict invalid")
            person_id = person["person_id"]
            allow_na = expected[person_id] == "not_observable"
            for key in ("identity_changed", "source_identity_absent", "local_color_change"):
                name = f"segment.{index}.person.{person_id}.{key}"
                record(
                    name,
                    person[key],
                    allow_na=allow_na,
                    code_key="person_local_color_change" if key == "local_color_change" else key,
                )
                status = checks[name]
                if allow_na and status != "not_applicable":
                    raise ValueError("semantic verdict invalid")
                if not allow_na and status == "not_applicable":
                    raise ValueError("semantic verdict invalid")
                segment_statuses.append(status)
        scene = segment["scene_checks"]
        invariants = segment["invariants"]
        if not isinstance(scene, dict) or set(scene) != set(_SCENE_CHECKS):
            raise ValueError("semantic verdict invalid")
        if not isinstance(invariants, dict) or set(invariants) != set(_INVARIANT_CHECKS):
            raise ValueError("semantic verdict invalid")
        for key in _SCENE_CHECKS:
            qualified = "scene_local_color_change" if key == "local_color_change" else key
            name = f"segment.{index}.scene.{qualified}"
            record(name, scene[key], code_key=qualified)
            segment_statuses.append(checks[name])
        for key in _INVARIANT_CHECKS:
            name = f"segment.{index}.invariant.{key}"
            record(name, invariants[key])
            segment_statuses.append(checks[name])
        expected_pass = not any(status in {"fail", "unknown"} for status in segment_statuses)
        if segment["passed"] != expected_pass:
            raise ValueError("semantic verdict invalid")

    project = raw["project_checks"]
    if not isinstance(project, dict) or set(project) != set(_PROJECT_CHECKS):
        raise ValueError("semantic verdict invalid")
    for key in _PROJECT_CHECKS:
        record(f"project.{key}", project[key])

    selected = next((item for item in nonpasses if item[1] == "unknown"), None)
    if selected is None and nonpasses:
        selected = nonpasses[0]
    derived = "pass" if selected is None else selected[1]
    if raw["passed"] != (derived == "pass"):
        raise ValueError("semantic verdict invalid")
    if raw["reason"] != _semantic_reason(checks):
        raise ValueError("semantic verdict invalid")
    code = None if derived == "pass" else _semantic_failure_code(*selected)
    return SemanticVerdict(derived, code, checks, canonical_sha256(raw))


class CodexSemanticVerifier:
    """Isolated wrapper for the same ``image-postprocess`` Skill, phase=verify."""

    def __init__(
        self,
        runner: Any,
        *,
        skill_path: Path,
        session_dir: Path,
        frame_inventory: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._runner = runner
        self._skill_path = Path(skill_path)
        self._session_dir = Path(session_dir)
        self._frame_inventory = (
            None if frame_inventory is None else _canonical_inventory(frame_inventory)
        )

    def verify(
        self,
        plan: Mapping[str, Any],
        source_frames: Sequence[Path],
        output_frames: Sequence[Path],
        *,
        deterministic_metrics: Mapping[str, Any],
    ) -> SemanticVerdict:
        plan_hash = _plan_sha256(plan)
        run_error: Exception | None = None
        try:
            session = self._session_dir.resolve(strict=True)
            skill = self._skill_path.resolve(strict=True)
            if (
                not session.is_dir() or not skill.is_file()
                or len(source_frames) != len(output_frames) or not source_frames
                or not isinstance(deterministic_metrics, Mapping)
            ):
                raise ValueError("invalid semantic verifier input")
            metrics = _json_roundtrip(dict(deterministic_metrics))
            if len(_canonical_bytes(metrics)) > _MAX_SEMANTIC_OUTPUT_BYTES:
                raise ValueError("invalid semantic verifier metrics")
            inventory = self._frame_inventory
            if inventory is None:
                indices = plan.get("segment_indices")
                if not isinstance(indices, list) or len(indices) != 1:
                    raise ValueError("multi-segment semantic verification requires frame inventory")
                inventory = [{
                    "segment_index": indices[0],
                    "frame_index": ordinal,
                    "source": {
                        "path": Path(source).name,
                        "sha256": _file_sha256(Path(source).resolve(strict=True)),
                        "width": 1,
                        "height": 1,
                    },
                    "person_ids": [],
                } for ordinal, source in enumerate(source_frames, 1)]
            if len(inventory) != len(source_frames):
                raise ValueError("semantic verifier frame inventory mismatch")
            with tempfile.TemporaryDirectory(prefix="duet-image-verify-", dir="/tmp") as raw_stage:
                stage = Path(raw_stage).resolve(strict=True)
                work = stage / "work"
                work.mkdir(parents=True, mode=0o700)
                _copy_regular(skill, stage / "SKILL.md")
                seen_names = set()
                for item, source_raw, output_raw in zip(inventory, source_frames, output_frames):
                    source = Path(source_raw).resolve(strict=True)
                    output = Path(output_raw).resolve(strict=True)
                    source.relative_to(session)
                    output.relative_to(session)
                    if _file_sha256(source) != item["source"]["sha256"]:
                        raise ValueError("semantic verifier source binding mismatch")
                    name = (item["segment_index"], item["frame_index"])
                    if name in seen_names:
                        raise ValueError("semantic verifier duplicate frame")
                    seen_names.add(name)
                    base = work / "segments" / str(item["segment_index"])
                    source_dir = base / "source"
                    output_dir = base / "output"
                    source_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
                    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
                    filename = f"{item['frame_index']:02d}.png"
                    _copy_regular(source, source_dir / filename)
                    _copy_regular(output, output_dir / filename)
                (work / "request.json").write_bytes(_canonical_bytes({
                    "phase": "verify",
                    "segment_indices": plan.get("segment_indices"),
                }) + b"\n")
                frozen_plan = _json_roundtrip(plan)
                if "sha256" not in frozen_plan:
                    frozen_plan["sha256"] = plan_hash
                (work / "frozen_plan.json").write_bytes(_canonical_bytes(frozen_plan) + b"\n")
                (work / "metrics.json").write_bytes(_canonical_bytes(metrics) + b"\n")
                try:
                    self._runner.run_isolated(
                        stage,
                        "严格执行当前目录 SKILL.md 的 verify 阶段；只读取允许的输入，并写入规定的唯一输出文件。",
                        session_dir=session,
                    )
                except Exception as error:
                    run_error = error
                try:
                    raw = _read_bounded_json(work / "image_verification.json")
                    return _parse_semantic_verdict(raw, plan, plan_hash)
                except ValueError:
                    code = "semantic_verifier_unavailable" if run_error is not None else "semantic_verdict_invalid"
                    return SemanticVerdict("unknown", code, {}, None)
        except Exception:
            return SemanticVerdict("unknown", "semantic_verdict_invalid", {}, None)


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    requested = Path(path)
    if not requested.is_absolute() or not requested.parent.is_dir():
        raise ValueError("receipt path must be absolute with an existing parent")
    raw = _canonical_bytes(payload) + b"\n"
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{requested.name}.", suffix=".tmp",
        dir=requested.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, requested)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "ALGORITHM_VERSION",
    "CodexSemanticVerifier",
    "DEFAULT_GATES",
    "FrameMasks",
    "GateResult",
    "MaskArtifact",
    "POC_PROFILE_V1",
    "QualityGate",
    "QualityProfile",
    "QualityReceipt",
    "ReferencePackSemanticVerifier",
    "SemanticVerdict",
    "canonical_sha256",
    "evaluate_image_quality",
    "evaluate_reference_packs",
    "load_frame_masks",
    "mask_manifest_receipt",
    "quality_receipt",
]
