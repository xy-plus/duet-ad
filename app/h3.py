"""Recoverable direct H3 generation primitives.

The caller owns review/confirmation and supplies an immutable request.  This
module owns the paid-provider crash boundary: every provider submission is
claimed on disk before POST and provider task identifiers are persisted before
polling. Recovery creates a new task only for a fully persisted provider
terminal failure while its bounded automatic-attempt budget remains.
"""

from __future__ import annotations

import base64
from bisect import bisect_right
import fcntl
import hashlib
import ipaddress
import json
import logging
import math
import os
import selectors
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence, TypeVar
from urllib.parse import urlsplit

import httpx

from app.retry import RetryPolicy, run_with_retry
from app.sanitize import sanitize
from app import dialogue_timing, storage, voice


SCHEMA_VERSION = 1
H3_WORKFLOW = "minimax_h3_lightx2v_v5_15s"
H3_PREVIOUS_WORKFLOW = "minimax_h3_lightx2v_v5"
H3_MULTIMODAL_WORKFLOW = "minimax_h3_image_audio_to_video_v2_15s"
H3_MULTIMODAL_HD_WORKFLOW = "minimax_h3_image_audio_to_video_v2"
H3_MULTIMODAL_WORKFLOWS = frozenset(
    {H3_MULTIMODAL_WORKFLOW, H3_MULTIMODAL_HD_WORKFLOW}
)
H3_REFERENCE_WORKFLOWS = frozenset(
    {H3_WORKFLOW, H3_PREVIOUS_WORKFLOW, *H3_MULTIMODAL_WORKFLOWS}
)
H3_BOUNDARY_WORKFLOW = "minimax_h3_lightx2v"
H3_ASPECT_RATIOS = frozenset({"16:9", "9:16"})
H3_RESOLUTIONS = frozenset({"480p", "768p"})
H3_DEFAULT_ASPECT_RATIO = "9:16"
H3_DEFAULT_RESOLUTION = "768p"
MEDIA_TIMELINE_SCHEMA = "duet.h3.media_timeline"
MEDIA_TIMELINE_VERSION = 1
MAX_AV_TIMELINE_DELTA_S = 0.1


def provider_resolution(aspect_ratio: str, resolution: str) -> str:
    """Project closed product semantics to the provider's single enum."""
    if aspect_ratio not in H3_ASPECT_RATIOS:
        raise H3Error("invalid_aspect_ratio")
    if resolution not in H3_RESOLUTIONS:
        raise H3Error("invalid_resolution")
    return resolution + ("横" if aspect_ratio == "16:9" else "竖")


# Historical public constant kept for exact legacy attempt recovery only.
H3_RESOLUTION = "768p竖"
H3_MAX_DURATION_S = 15
H3_PREVIOUS_MAX_DURATION_S = 10
H3_BOUNDARY_MAX_DURATION_S = 15
AUTODL_BASE_URL = "https://autodl.art"
H3_GATEWAY_BASE_URL = "http://127.0.0.1:31000"
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_MEDIA_PROBE_STDOUT_BYTES = 16 * 1024 * 1024
MAX_MEDIA_TIMELINE_EVENTS = 20_000
MAX_REFERENCE_AUDIO_BYTES = 15 * 1024 * 1024
H3_OUTPUT_FRAME_DURATION_S = 1 / 24
_DURATION_EPS_S = 1e-6

_SAFE_ERROR_CODES = {
    "h3_submit_rejected",
    "h3_query_failed",
    "h3_result_missing",
    "h3_provider_failed",
    "h3_timeout",
    "download_failed",
    "download_dns_failed",
    "download_peer_unverified",
    "download_url_rejected",
    "download_redirect_rejected",
    "download_too_large",
    "download_invalid_video",
    "output_probe_failed",
    "output_audio_missing",
    "output_write_failed",
}

log = logging.getLogger(__name__)
_T = TypeVar("_T")

FrozenKeyframes = tuple[tuple[Path, bytes], ...]
FrozenFrame = tuple[Path, bytes]
FrozenVoiceTexts = tuple[str, ...]
H3Mode = Literal["reference", "boundary"]
ReferenceAudioPurpose = Literal["voice", "ambience", "effect"]


class H3Error(RuntimeError):
    """A safe, stable error whose text never includes provider data."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class ReceiptError(H3Error):
    """Persisted state or caller-supplied frozen input does not match."""


class H3BusyError(H3Error):
    """Another process/thread currently owns this session."""


class _AutomaticRetryH3Error(H3Error):
    """A safe same-attempt failure that may be retried without another POST."""


class _DNSLookupFailed(Exception):
    pass


class _ProbeUnavailable(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FrozenReferenceAudio:
    """One probed, immutable conditioning reference for native H3 audio."""

    path: Path
    data: bytes = field(repr=False)
    order: int
    purpose: ReferenceAudioPurpose
    format: Literal["mp3", "wav"]
    sha256: str
    duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, int)
            or self.order < 1
        ):
            raise H3Error("invalid_reference_audio_order")
        if self.purpose not in {"voice", "ambience", "effect"}:
            raise H3Error("invalid_reference_audio_purpose")
        if self.format not in {"mp3", "wav"}:
            raise H3Error("invalid_reference_audio_format")
        if not isinstance(self.data, bytes) or not self.data:
            raise H3Error("invalid_reference_audio")
        if len(self.data) > MAX_REFERENCE_AUDIO_BYTES:
            raise H3Error("reference_audio_too_large")
        if self.sha256 != hashlib.sha256(self.data).hexdigest():
            raise ReceiptError("reference_audio_hash_mismatch")
        if (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, (int, float))
            or not math.isfinite(float(self.duration_s))
            or not 2 - _DURATION_EPS_S <= float(self.duration_s) <= 15 + _DURATION_EPS_S
        ):
            raise H3Error("invalid_reference_audio_duration")


FrozenReferenceAudios = tuple[FrozenReferenceAudio, ...]


@dataclass(frozen=True)
class Timeouts:
    request_s: float = 30.0
    h3_poll_s: float = 1500.0
    download_s: float = 180.0
    poll_interval_s: float = 3.0
    probe_s: float = 30.0
    retry_count: int = 2
    retry_interval_s: float = 15.0

    def __post_init__(self) -> None:
        positive = (
            self.request_s,
            self.h3_poll_s,
            self.download_s,
            self.probe_s,
        )
        if any(isinstance(value, bool) or value <= 0 for value in positive):
            raise H3Error("invalid_timeout")
        if isinstance(self.poll_interval_s, bool) or self.poll_interval_s < 0:
            raise H3Error("invalid_timeout")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise H3Error("invalid_timeout")
        if (
            isinstance(self.retry_interval_s, bool)
            or not isinstance(self.retry_interval_s, (int, float))
            or not math.isfinite(float(self.retry_interval_s))
            or self.retry_interval_s < 0
        ):
            raise H3Error("invalid_timeout")


@dataclass(frozen=True)
class H3Request:
    cid: str
    workdir: Path
    client_request_id: str
    prompt: str
    keyframes: FrozenKeyframes
    voice_texts: FrozenVoiceTexts
    voice_receipt: str
    duration: int
    autodl_token: str
    timeouts: Timeouts = Timeouts()
    mode: H3Mode = "reference"
    first_frame: FrozenFrame | None = None
    last_frame: FrozenFrame | None = None
    seed: int | None = None
    aspect_ratio: str = H3_DEFAULT_ASPECT_RATIO
    resolution: str = H3_DEFAULT_RESOLUTION
    workflow: str | None = None
    reference_audios: FrozenReferenceAudios = ()
    skill_plan_sha256: str | None = None
    multimodal_compiler_version: str | None = None
    upstream_dialogue_receipt_sha256: str | None = None
    speaker_timing_sha256: str | None = None
    speaker_timing_authority_version: int | None = None
    speaker_timing_production_required: bool = False
    speaker_timing_legacy_source_version: int | None = None
    speaker_timing_legacy_receipt_path: str | None = None
    speaker_timing_legacy_receipt_sha256: str | None = None
    speaker_timing_production_path: str | None = None
    speaker_timing_production_sha256: str | None = None
    speaker_timing_authority_artifacts: tuple[tuple[str, str], ...] = ()
    on_screen_dialogue: tuple[Mapping[str, Any], ...] = ()
    on_screen_dialogue_sha256: str | None = None
    audio_required: bool = False
    context_ir_receipt_path: Path | None = None
    context_ir_receipt_sha256: str | None = None
    gateway_storage_root: Path | None = None
    speaker_timing_authority_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workdir", Path(self.workdir))
        if self.gateway_storage_root is not None:
            storage_root = Path(self.gateway_storage_root)
            if not storage_root.is_absolute():
                raise H3Error("invalid_gateway_storage_root")
            object.__setattr__(self, "gateway_storage_root", storage_root)
        if self.speaker_timing_authority_root is not None:
            authority_root = Path(self.speaker_timing_authority_root)
            if not authority_root.is_absolute():
                raise H3Error("invalid_speaker_timing_authority_root")
            object.__setattr__(
                self, "speaker_timing_authority_root", authority_root
            )
        if not isinstance(self.cid, str) or not self.cid.strip():
            raise H3Error("invalid_cid")
        if not isinstance(self.client_request_id, str) or not self.client_request_id.strip():
            raise H3Error("invalid_client_request_id")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise H3Error("invalid_prompt")
        if not isinstance(self.mode, str) or self.mode not in {"reference", "boundary"}:
            raise H3Error("invalid_mode")
        workflow = _workflow(self)
        if self.aspect_ratio not in H3_ASPECT_RATIOS:
            raise H3Error("invalid_aspect_ratio")
        is_multimodal = workflow in H3_MULTIMODAL_WORKFLOWS
        allowed_resolutions = (
            {"1080p"}
            if workflow == H3_MULTIMODAL_HD_WORKFLOW
            else H3_RESOLUTIONS
        )
        if self.resolution not in allowed_resolutions:
            raise H3Error("invalid_resolution")
        if self.mode == "reference":
            if workflow not in H3_REFERENCE_WORKFLOWS:
                raise H3Error("invalid_workflow")
            if workflow in {H3_WORKFLOW, H3_MULTIMODAL_WORKFLOW}:
                max_duration = H3_MAX_DURATION_S
            else:
                max_duration = H3_PREVIOUS_MAX_DURATION_S
        else:
            if workflow != H3_BOUNDARY_WORKFLOW:
                raise H3Error("invalid_workflow")
            max_duration = H3_BOUNDARY_MAX_DURATION_S
        if (
            not isinstance(self.duration, int)
            or isinstance(self.duration, bool)
            or not 1 <= self.duration <= max_duration
        ):
            raise H3Error("invalid_duration")
        if not isinstance(self.autodl_token, str) or not self.autodl_token.strip():
            raise H3Error("missing_autodl_credential")
        if not isinstance(self.keyframes, tuple):
            raise H3Error("invalid_keyframes")
        if self.mode == "reference":
            if self.first_frame is not None or self.last_frame is not None:
                raise H3Error("mixed_h3_inputs")
            if not 1 <= len(self.keyframes) <= 9:
                raise H3Error("invalid_keyframes")
            names = [
                _validate_frame(item, "invalid_keyframes")[0].name
                for item in self.keyframes
            ]
            if len(names) != len(set(names)):
                raise H3Error("duplicate_keyframe_name")
            if self.seed is not None and (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, int)
                or not 1 <= self.seed <= 999_999_999_999_999
            ):
                raise H3Error("invalid_seed")
        else:
            if self.keyframes:
                raise H3Error("mixed_h3_inputs")
            if self.first_frame is None or self.last_frame is None:
                raise H3Error("invalid_boundary_frames")
            _validate_frame(self.first_frame, "invalid_boundary_frames")
            _validate_frame(self.last_frame, "invalid_boundary_frames")
            if self.seed is not None:
                raise H3Error("seed_not_supported")
        if not isinstance(self.voice_texts, tuple):
            raise H3Error("invalid_voice_texts")
        if any(not isinstance(text, str) or not text for text in self.voice_texts):
            raise H3Error("invalid_voice_texts")
        if self.voice_receipt != voice_texts_receipt(self.voice_texts):
            raise ReceiptError("voice_receipt_mismatch")
        _validate_request_audio_contract(self, is_multimodal=is_multimodal)


def _validate_frame(item: Any, code: str) -> FrozenFrame:
    if not isinstance(item, tuple) or len(item) != 2:
        raise H3Error(code)
    path, blob = item
    if not isinstance(path, Path) or not isinstance(blob, bytes) or not blob:
        raise H3Error(code)
    if not path.name or path.name != Path(path.name).name:
        raise H3Error(code)
    return path, blob


def _validate_request_audio_contract(
    request: H3Request, *, is_multimodal: bool,
) -> None:
    audios = request.reference_audios
    if not isinstance(audios, tuple):
        raise H3Error("invalid_reference_audio")
    if not is_multimodal:
        if (
            audios
            or request.skill_plan_sha256 is not None
            or request.multimodal_compiler_version is not None
            or request.upstream_dialogue_receipt_sha256 is not None
            or request.speaker_timing_sha256 is not None
            or request.speaker_timing_authority_version is not None
            or request.speaker_timing_production_required is not False
            or request.speaker_timing_legacy_source_version is not None
            or request.speaker_timing_legacy_receipt_path is not None
            or request.speaker_timing_legacy_receipt_sha256 is not None
            or request.speaker_timing_production_path is not None
            or request.speaker_timing_production_sha256 is not None
            or request.speaker_timing_authority_artifacts
            or request.speaker_timing_authority_root is not None
            or request.on_screen_dialogue
            or request.on_screen_dialogue_sha256 is not None
            or request.audio_required is not False
            or request.context_ir_receipt_path is not None
            or request.context_ir_receipt_sha256 is not None
            or request.gateway_storage_root is not None
        ):
            raise H3Error("mixed_h3_inputs")
        return
    if request.mode != "reference":
        raise H3Error("mixed_h3_inputs")
    if not 4 <= request.duration:
        raise H3Error("invalid_duration")
    if not 1 <= len(audios) <= 3:
        raise H3Error("invalid_reference_audio_count")
    for expected_order, audio in enumerate(audios, 1):
        if not isinstance(audio, FrozenReferenceAudio) or audio.order != expected_order:
            raise H3Error("invalid_reference_audio_order")
        if audio.sha256 != hashlib.sha256(audio.data).hexdigest():
            raise ReceiptError("reference_audio_hash_mismatch")
    if len({audio.sha256 for audio in audios}) != len(audios):
        raise H3Error("duplicate_reference_audio")
    if any(
        path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}
        for path, _blob in request.keyframes
    ):
        raise H3Error("invalid_keyframe_format")
    if sum(float(audio.duration_s) for audio in audios) > 15 + _DURATION_EPS_S:
        raise H3Error("invalid_reference_audio_duration")
    if request.audio_required is not True:
        raise H3Error("audio_required")
    if not _is_sha256(request.skill_plan_sha256):
        raise H3Error("invalid_skill_plan_receipt")
    if not _is_sha256(request.upstream_dialogue_receipt_sha256):
        raise H3Error("invalid_upstream_dialogue_receipt")
    if request.on_screen_dialogue and not _is_sha256(request.speaker_timing_sha256):
        raise H3Error("invalid_speaker_timing_receipt")
    if not request.on_screen_dialogue and request.speaker_timing_sha256 is not None and not _is_sha256(request.speaker_timing_sha256):
        raise H3Error("invalid_speaker_timing_receipt")
    authority_values = (
        request.speaker_timing_authority_version,
        request.speaker_timing_production_required,
        request.speaker_timing_legacy_source_version,
        request.speaker_timing_legacy_receipt_path,
        request.speaker_timing_legacy_receipt_sha256,
        request.speaker_timing_production_path,
        request.speaker_timing_production_sha256,
        request.speaker_timing_authority_artifacts,
        request.speaker_timing_authority_root,
    )
    if not request.on_screen_dialogue:
        if authority_values != (
            None, False, None, None, None, None, None, (), None,
        ):
            raise H3Error("mixed_speaker_timing_authority")
    elif request.speaker_timing_authority_version == 0:
        legacy_values = authority_values[2:5]
        if (
            authority_values[1] is not False
            or authority_values[5:8] != (None, None, ())
            or (
                legacy_values != (None, None, None)
                and (
                    legacy_values[0] != 2
                    or not _safe_relative_authority_path(legacy_values[1])
                    or not _is_sha256(legacy_values[2])
                )
            )
            or (legacy_values == (None, None, None) and authority_values[8] is not None)
        ):
            raise H3Error("invalid_legacy_speaker_timing_authority")
    elif request.speaker_timing_authority_version == 1:
        production_path = request.speaker_timing_production_path
        artifacts = request.speaker_timing_authority_artifacts
        if (
            request.speaker_timing_production_required is not True
            or authority_values[2:5] != (None, None, None)
            or not isinstance(production_path, str)
            or not _safe_relative_authority_path(production_path)
            or not _is_sha256(request.speaker_timing_production_sha256)
            or not isinstance(artifacts, tuple)
            or not artifacts
        ):
            raise H3Error("invalid_speaker_timing_production_authority")
        normalized: list[tuple[str, str]] = []
        for artifact in artifacts:
            if (
                not isinstance(artifact, tuple)
                or len(artifact) != 2
                or not _safe_relative_authority_path(artifact[0])
                or not _is_sha256(artifact[1])
            ):
                raise H3Error("invalid_speaker_timing_production_authority")
            normalized.append((artifact[0], artifact[1]))
        if (
            tuple(sorted(normalized)) != tuple(normalized)
            or len({path for path, _sha in normalized}) != len(normalized)
            or (production_path, request.speaker_timing_production_sha256)
            not in normalized
        ):
            raise H3Error("invalid_speaker_timing_production_authority")
    else:
        raise H3Error("speaker_timing_authority_required")
    if not isinstance(request.on_screen_dialogue, tuple):
        raise H3Error("invalid_on_screen_dialogue")
    normalized_dialogue: list[dict[str, Any]] = []
    for expected_order, line in enumerate(request.on_screen_dialogue, 1):
        if (
            not isinstance(line, Mapping)
            or set(line) != {
                "order", "line_index", "subject_id", "text",
                "start_s", "end_s",
            }
            or line.get("order") != expected_order
            or isinstance(line.get("line_index"), bool)
            or not isinstance(line.get("line_index"), int)
            or line["line_index"] < 1
            or not isinstance(line.get("subject_id"), str)
            or not line["subject_id"]
            or not isinstance(line.get("text"), str)
            or not line["text"]
        ):
            raise H3Error("invalid_on_screen_dialogue")
        try:
            start_s, end_s = float(line["start_s"]), float(line["end_s"])
        except (TypeError, ValueError):
            raise H3Error("invalid_on_screen_dialogue") from None
        if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s < end_s):
            raise H3Error("invalid_on_screen_dialogue")
        normalized_dialogue.append({
            "order": expected_order,
            "line_index": line["line_index"],
            "subject_id": line["subject_id"],
            "text": line["text"],
            "start_s": start_s,
            "end_s": end_s,
        })
    object.__setattr__(request, "on_screen_dialogue", tuple(normalized_dialogue))
    dialogue_sha256 = (
        canonical_json_sha256(normalized_dialogue)
        if normalized_dialogue
        else None
    )
    if request.on_screen_dialogue_sha256 != dialogue_sha256:
        raise ReceiptError("on_screen_dialogue_receipt_mismatch")
    if (
        not isinstance(request.multimodal_compiler_version, str)
        or not request.multimodal_compiler_version.strip()
    ):
        raise H3Error("invalid_multimodal_compiler")


def validate_request_authority(request: H3Request) -> None:
    """Revalidate nested multimodal authority at every paid/read boundary."""
    if not isinstance(request, H3Request):
        raise H3Error("invalid_request")
    _validate_request_audio_contract(
        request,
        is_multimodal=_workflow(request) in H3_MULTIMODAL_WORKFLOWS,
    )


def speaker_timing_authority_manifest(request: H3Request) -> dict[str, Any] | None:
    version = request.speaker_timing_authority_version
    if version is None or (
        version == 0 and request.speaker_timing_legacy_receipt_path is None
    ):
        return None
    manifest = {
        "version": version,
        "required": request.speaker_timing_production_required,
        "production_path": request.speaker_timing_production_path,
        "production_sha256": request.speaker_timing_production_sha256,
        "artifacts": [
            {"path": path, "sha256": sha256}
            for path, sha256 in request.speaker_timing_authority_artifacts
        ],
    }
    if version == 0:
        manifest["legacy_source"] = {
            "version": request.speaker_timing_legacy_source_version,
            "receipt_path": request.speaker_timing_legacy_receipt_path,
            "receipt_sha256": request.speaker_timing_legacy_receipt_sha256,
        }
    return manifest


def _require_speaker_timing_production_authority(request: H3Request) -> None:
    if not request.on_screen_dialogue:
        return
    if request.speaker_timing_authority_version == 0:
        # A v2 source manifest alone does not prove its nested input, skill
        # plan, dialogue, keyframes, or audio still match this request.  Until
        # an existing historical attempt can be fully rebound to those bytes,
        # every public H3 paid/read/reuse boundary must fail closed.
        raise ReceiptError("legacy_speaker_timing_authority_unverifiable")
    if request.speaker_timing_authority_version != 1:
        raise ReceiptError("speaker_timing_production_authority_required")
    root = request.speaker_timing_authority_root
    if not isinstance(root, Path):
        raise ReceiptError("speaker_timing_production_authority_root_required")
    root = root.resolve()
    frozen = dict(request.speaker_timing_authority_artifacts)
    loaded: dict[str, bytes] = {}
    try:
        for relative, expected_sha256 in frozen.items():
            path = (root / relative).resolve()
            path.relative_to(root)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ReceiptError(
                    "speaker_timing_production_authority_mismatch"
                )
            loaded[relative] = data
    except (OSError, ValueError):
        raise ReceiptError(
            "speaker_timing_production_authority_invalid"
        ) from None
    production_relative = request.speaker_timing_production_path
    if not isinstance(production_relative, str):
        raise ReceiptError("speaker_timing_production_authority_invalid")
    try:
        production = json.loads(loaded[production_relative].decode("utf-8"))
        artifacts = production["artifacts"]
    except (
        KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError,
    ):
        raise ReceiptError(
            "speaker_timing_production_authority_invalid"
        ) from None
    if (
        not isinstance(production, Mapping)
        or production.get("schema") != "duet.speaker-timing-production"
        or production.get("version") != 1
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {
        "producer_input", "raw_output", "skill", "speaker_timing",
        }
    ):
        raise ReceiptError("speaker_timing_production_authority_invalid")

    def bound(
        base: str,
        artifact: object,
        *,
        timing: bool = False,
        exact: bool = True,
    ) -> str:
        keys = {"path", "sha256", "canonical_sha256"} if timing else {
            "path", "sha256"
        }
        if (
            not isinstance(artifact, Mapping)
            or (set(artifact) != keys if exact else not keys <= set(artifact))
        ):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        relative = artifact.get("path")
        if not _safe_relative_authority_path(relative):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        joined = (Path(base).parent / str(relative)).as_posix()
        if not _safe_relative_authority_path(joined):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        return joined

    expected = {
        production_relative: str(request.speaker_timing_production_sha256),
    }
    role_paths: dict[str, str] = {}
    for role in ("producer_input", "raw_output", "skill", "speaker_timing"):
        artifact = artifacts[role]
        path = bound(
            production_relative, artifact, timing=role == "speaker_timing"
        )
        digest = artifact.get("sha256")
        if not _is_sha256(digest) or (
            path in expected and expected[path] != digest
        ):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        expected[path] = str(digest)
        role_paths[role] = path
        if (
            role == "speaker_timing"
            and artifact.get("canonical_sha256")
            != request.speaker_timing_sha256
        ):
            raise ReceiptError("speaker_timing_production_authority_invalid")
    try:
        producer_input = json.loads(
            loaded[role_paths["producer_input"]].decode("utf-8")
        )
    except (
        KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError,
    ):
        raise ReceiptError(
            "speaker_timing_production_authority_invalid"
        ) from None
    evidence: list[object] = []
    for key in ("frames", "contact_sheets"):
        values = producer_input.get(key) if isinstance(producer_input, Mapping) else None
        if not isinstance(values, list):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        evidence.extend(values)
    persons = producer_input.get("persons") if isinstance(producer_input, Mapping) else None
    if not isinstance(persons, list):
        raise ReceiptError("speaker_timing_production_authority_invalid")
    for person in persons:
        refs = person.get("identity_refs") if isinstance(person, Mapping) else None
        if not isinstance(refs, list):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        evidence.extend(refs)
    cut_source = producer_input.get("cut_source") if isinstance(producer_input, Mapping) else None
    if not isinstance(cut_source, Mapping):
        raise ReceiptError("speaker_timing_production_authority_invalid")
    evidence.append(cut_source)
    input_base = role_paths["producer_input"]
    source = producer_input.get("source") if isinstance(producer_input, Mapping) else None
    source_sha256 = source.get("sha256") if isinstance(source, Mapping) else None
    source_relative = (
        Path(input_base).parent.parent / "source.mp4"
    ).as_posix()
    if not _is_sha256(source_sha256) or not _safe_relative_authority_path(
        source_relative
    ):
        raise ReceiptError("speaker_timing_production_authority_invalid")
    expected[source_relative] = str(source_sha256)
    frame_data: dict[str, bytes] = {}
    for artifact in evidence:
        path = bound(input_base, artifact, exact=False)
        digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
        if not _is_sha256(digest) or (
            path in expected and expected[path] != digest
        ):
            raise ReceiptError("speaker_timing_production_authority_invalid")
        expected[path] = str(digest)
        frame_data[str(artifact["path"])] = loaded[path]
    if expected != frozen or set(loaded) != set(expected):
        raise ReceiptError("speaker_timing_production_authority_mismatch")
    try:
        projected = dialogue_timing.freeze_speaker_visibility(
            producer_input_data=loaded[role_paths["producer_input"]],
            skill_output_data=loaded[role_paths["raw_output"]],
            source_data=loaded[source_relative],
            frame_data=frame_data,
            skill_data=loaded[role_paths["skill"]],
        )
        frozen_timing = json.loads(
            loaded[role_paths["speaker_timing"]].decode("utf-8")
        )
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        dialogue_timing.DialogueTimingError,
    ):
        raise ReceiptError(
            "speaker_timing_production_authority_invalid"
        ) from None
    if (
        projected.receipt != production
        or projected.speaker_timing != frozen_timing
        or dialogue_timing.canonical_sha256(projected.speaker_timing)
        != request.speaker_timing_sha256
    ):
        raise ReceiptError("speaker_timing_production_authority_mismatch")


def _require_h3_boundary(request: H3Request) -> None:
    _require_context_ir_receipt(request)
    _require_speaker_timing_production_authority(request)


def _context_ir_reference_receipt(request: H3Request) -> str:
    references: list[dict[str, Any]] = []
    image_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    audio_types = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
    for order, (path, data) in enumerate(request.keyframes, 1):
        references.append({
            "order": order,
            "type": "image_url",
            "role": "reference_image",
            "name": path.name,
            "mime_type": image_types.get(path.suffix.lower()),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    for audio in request.reference_audios:
        references.append({
            "order": audio.order,
            "type": "audio_url",
            "role": "reference_audio",
            "name": audio.path.name,
            "mime_type": audio_types.get(audio.path.suffix.lower()),
            "sha256": audio.sha256,
            "size": len(audio.data),
        })
    if any(item["mime_type"] is None for item in references):
        raise ReceiptError("context_ir_receipt_invalid")
    return canonical_json_sha256(references)


def _context_ir_source_request_receipt(
    request: H3Request,
    *,
    source_prompt_sha256: str,
    references_sha256: str,
) -> str:
    def frame_manifest(frame: FrozenFrame | None) -> dict[str, Any] | None:
        if frame is None:
            return None
        path, data = frame
        return {
            "name": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    return canonical_json_sha256({
        "cid": request.cid,
        "workdir": str(request.workdir.resolve()),
        "client_request_id": request.client_request_id,
        "source_prompt_sha256": source_prompt_sha256,
        "references_sha256": references_sha256,
        "voice_texts_sha256": request.voice_receipt,
        "duration": request.duration,
        "mode": request.mode,
        "first_frame": frame_manifest(request.first_frame),
        "last_frame": frame_manifest(request.last_frame),
        "seed": request.seed,
        "aspect_ratio": request.aspect_ratio,
        "resolution": request.resolution,
        "workflow": request.workflow,
        "skill_plan_sha256": request.skill_plan_sha256,
        "multimodal_compiler_version": request.multimodal_compiler_version,
        "speaker_timing_sha256": request.speaker_timing_sha256,
        **(
            {"speaker_timing_authority": speaker_timing_authority_manifest(request)}
            if speaker_timing_authority_manifest(request) is not None
            else {}
        ),
        "on_screen_dialogue_sha256": request.on_screen_dialogue_sha256,
        "audio_required": request.audio_required,
        "reference_audio_metadata": [
            {
                "order": audio.order,
                "purpose": audio.purpose,
                "format": audio.format,
                "sha256": audio.sha256,
                "duration_s": audio.duration_s,
            }
            for audio in request.reference_audios
        ],
    })


def _require_context_ir_receipt(request: H3Request) -> None:
    """Reject raw multimodal requests at every H3 provider boundary."""
    validate_request_authority(request)
    if not is_multimodal_request(request):
        return
    path = request.context_ir_receipt_path
    expected_sha256 = request.context_ir_receipt_sha256
    if not isinstance(path, Path) or not _is_sha256(expected_sha256):
        raise ReceiptError("context_ir_receipt_required")
    resolved = path.resolve()
    attempts = (request.workdir / ".context-ir" / "attempts").resolve()
    try:
        relative = resolved.relative_to(attempts)
    except ValueError:
        raise ReceiptError("context_ir_receipt_path_invalid") from None
    if (
        len(relative.parts) != 2
        or relative.name != "receipt.json"
        or len(relative.parts[0]) != 6
        or not relative.parts[0].isdigit()
    ):
        raise ReceiptError("context_ir_receipt_path_invalid")
    try:
        receipt = _read_json(resolved)
        attempt = _read_json(resolved.with_name("attempt.json"))
    except H3Error:
        raise ReceiptError("context_ir_receipt_invalid") from None
    if not isinstance(receipt, dict) or not isinstance(attempt, dict):
        raise ReceiptError("context_ir_receipt_invalid")
    unhashed = dict(receipt)
    receipt_sha256 = unhashed.pop("receipt_sha256", None)
    source_prompt = receipt.get("source_prompt")
    source_prompt_sha256 = receipt.get("source_prompt_sha256")
    references_sha256 = _context_ir_reference_receipt(request)
    if (
        receipt.get("schema") != "duet.context-ir.effective-prompt"
        or receipt.get("version") != 1
        or receipt.get("cid") != request.cid
        or receipt.get("client_request_id") != request.client_request_id
        or receipt.get("attempt_id") != relative.parts[0]
        or receipt_sha256 != expected_sha256
        or receipt_sha256 != canonical_json_sha256(unhashed)
        or not isinstance(source_prompt, str)
        or hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
        != source_prompt_sha256
        or receipt.get("effective_prompt") != request.prompt
        or hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        != receipt.get("effective_prompt_sha256")
        or receipt.get("skill_plan_sha256") != request.skill_plan_sha256
        or receipt.get("voice_texts_sha256") != request.voice_receipt
        or receipt.get("upstream_dialogue_sha256")
        != request.upstream_dialogue_receipt_sha256
        or receipt.get("references_sha256") != references_sha256
        or receipt.get("source_h3_request_sha256")
        != _context_ir_source_request_receipt(
            request,
            source_prompt_sha256=str(source_prompt_sha256),
            references_sha256=references_sha256,
        )
        or attempt.get("schema") != "duet.context-ir.attempt"
        or attempt.get("version") != 1
        or attempt.get("cid") != request.cid
        or attempt.get("attempt_id") != relative.parts[0]
        or attempt.get("client_request_id") != request.client_request_id
        or attempt.get("status") != "succeeded"
        or attempt.get("receipt")
        != {"path": "receipt.json", "sha256": receipt_sha256}
        or attempt.get("provider_task_id") != receipt.get("provider_task_id")
        or attempt.get("context_ir_request_sha256")
        != receipt.get("context_ir_request_sha256")
        or attempt.get("context_ir_task_sha256")
        != receipt.get("context_ir_task_sha256")
        or attempt.get("context_ir_attempt_sha256")
        != receipt.get("context_ir_attempt_sha256")
    ):
        raise ReceiptError("context_ir_receipt_mismatch")


@dataclass(frozen=True)
class H3Result:
    status: str
    attempt_id: str | None
    output: Path | None = None
    retryable: bool = False
    error_code: str | None = None
    media_timeline: Mapping[str, Any] | None = None


def _retryable_http_status(status: int) -> bool:
    return status in {408, 425, 429} or 500 <= status <= 599


def _provider_error_detail(payload: Mapping[str, Any], *, secret: str) -> str:
    """Extract only provider-owned error fields; never log the full response."""

    values: list[str] = []

    def append(value: Any) -> None:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value).strip()
            if text and text not in values:
                values.append(text)

    for key in ("code", "msg", "message"):
        append(payload.get(key))
    error = payload.get("error")
    if isinstance(error, Mapping):
        for key in ("code", "msg", "message"):
            append(error.get(key))
    else:
        append(error)
    return sanitize(" | ".join(values), secrets=(secret,))


def _provider_failure_diagnostic(
    payload: Mapping[str, Any], *, status: str, secret: str
) -> dict[str, str]:
    """Keep only bounded provider-owned fields needed to investigate a failure."""

    diagnostic = {
        "status": status,
        "detail": _safe_provider_field(
            _provider_error_detail(payload, secret=secret), limit=300, secret=secret
        ),
    }
    request_id = payload.get("request_id")
    if isinstance(request_id, (str, int)) and not isinstance(request_id, bool):
        safe_request_id = _safe_provider_field(request_id, limit=128, secret=secret)
        if safe_request_id:
            diagnostic["request_id"] = safe_request_id
    return diagnostic


def _safe_provider_field(value: Any, *, limit: int, secret: str) -> str:
    return " ".join(
        sanitize(str(value).strip(), limit=limit, secrets=(secret,)).splitlines()
    )


def _run_automatic_retry(
    timeouts: Timeouts,
    operation: Callable[[], _T],
    *,
    step: str,
    deadline: float | None = None,
) -> _T:
    policy = RetryPolicy(timeouts.retry_count, timeouts.retry_interval_s)

    def retryable(exc: Exception) -> bool:
        if not isinstance(exc, _AutomaticRetryH3Error):
            return False
        return deadline is None or time.monotonic() + policy.interval_s < deadline

    def report(retry_number: int, _exc: Exception) -> None:
        log.warning(
            "%s failed; retry %d/%d in %.1fs",
            step,
            retry_number,
            policy.retries,
            policy.interval_s,
        )

    return run_with_retry(
        operation,
        policy=policy,
        is_retryable=retryable,
        on_retry=report,
    )


def freeze_keyframes(paths: Sequence[Path]) -> FrozenKeyframes:
    """Read the ordered image batch once; all later stages reuse these bytes."""
    try:
        frozen = tuple((Path(path), Path(path).read_bytes()) for path in paths)
    except OSError:
        raise H3Error("keyframe_read_failed") from None
    return frozen


def freeze_reference_audios(
    sources: Sequence[tuple[Path, ReferenceAudioPurpose]],
) -> FrozenReferenceAudios:
    """Read and probe 1–3 ordered MP3/WAV references exactly once."""
    if not 1 <= len(sources) <= 3:
        raise H3Error("invalid_reference_audio_count")
    frozen: list[FrozenReferenceAudio] = []
    for order, source in enumerate(sources, 1):
        if not isinstance(source, tuple) or len(source) != 2:
            raise H3Error("invalid_reference_audio")
        path, purpose = source
        path = Path(path)
        audio_format = path.suffix.lower().lstrip(".")
        if audio_format not in {"mp3", "wav"}:
            raise H3Error("invalid_reference_audio_format")
        if purpose not in {"voice", "ambience", "effect"}:
            raise H3Error("invalid_reference_audio_purpose")
        try:
            data = path.read_bytes()
        except OSError:
            raise H3Error("reference_audio_read_failed") from None
        if not data:
            raise H3Error("invalid_reference_audio")
        if len(data) > MAX_REFERENCE_AUDIO_BYTES:
            raise H3Error("reference_audio_too_large")
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}") as probe_file:
                probe_file.write(data)
                probe_file.flush()
                if not storage.probe_audio(Path(probe_file.name)):
                    raise H3Error("invalid_reference_audio")
                duration = voice.probe_audio_duration(Path(probe_file.name))
        except (OSError, storage.UploadError):
            raise H3Error("reference_audio_probe_failed") from None
        if duration is None:
            raise H3Error("reference_audio_probe_failed")
        frozen.append(
            FrozenReferenceAudio(
                path=path,
                data=data,
                order=order,
                purpose=purpose,
                format=audio_format,
                sha256=hashlib.sha256(data).hexdigest(),
                duration_s=duration,
            )
        )
    result = tuple(frozen)
    if len({audio.sha256 for audio in result}) != len(result):
        raise H3Error("duplicate_reference_audio")
    if sum(audio.duration_s for audio in result) > 15 + _DURATION_EPS_S:
        raise H3Error("invalid_reference_audio_duration")
    return result


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def voice_texts_receipt(voice_texts: Sequence[str]) -> str:
    return canonical_json_sha256(list(voice_texts))


def start(request: H3Request, *, client: httpx.Client | None = None) -> H3Result:
    """Start or idempotently advance one client request.

    A repeated ``client_request_id`` resolves to its existing attempt. It may
    query an already persisted task, but never repeats a provider POST whose
    outcome is unknown.
    """
    _require_h3_boundary(request)
    with _session_lease(request):
        existing = _find_attempt(request, request.client_request_id)
        if output_is_reusable(request, existing):
            return _output_result(request, existing)
        if existing is None:
            state = _create_attempt(request, request.client_request_id)
            is_new = True
        else:
            state = existing
            is_new = False
        with _client(client) as active_client:
            return _advance_with_provider_retry(
                request,
                state,
                active_client,
                allow_submit=True,
                new_attempt=is_new,
            )


def prepare(request: H3Request) -> H3Result:
    """Persist an exact unpaid attempt without contacting the provider.

    Fast fan-out callers prepare every child first.  A persisted ``ready``
    state proves that no POST has started; ``submitting`` remains ambiguous.
    """
    _require_h3_boundary(request)
    with _session_lease(request):
        existing = _find_attempt(request, request.client_request_id)
        if output_is_reusable(request, existing):
            return _output_result(request, existing)
        state = (
            _create_attempt(request, request.client_request_id)
            if existing is None
            else existing
        )
        if state.get("status") == "ready_to_submit":
            return H3Result("not_started", str(state["attempt_id"]))
        return _result(state)


def submit(request: H3Request, *, client: httpx.Client | None = None) -> H3Result:
    """POST one previously prepared attempt and return before any GET poll."""
    _require_h3_boundary(request)
    with _session_lease(request):
        state = _find_attempt(request, request.client_request_id)
        if state is None:
            raise H3Error("attempt_not_prepared")
        if output_is_reusable(request, state):
            return _output_result(request, state)
        status = str(state.get("status") or "")
        if status in {"submission_unknown", "failed", "retryable_failure"}:
            return _result(state)
        h3_state = state["h3"]
        task_id = _task_id(h3_state.get("task_id"), required=False)
        if task_id is not None:
            return _result(state)
        if status != "ready_to_submit" or h3_state.get("status") != "ready":
            _submission_unknown(request, state, "h3")
            return _result(state)
        with _client(client) as active_client:
            _submit_h3(request, state, active_client)
        return _result(state)

def inspect(request: H3Request) -> H3Result:
    """Read the latest attempt for UI/startup decisions, without any writes."""
    _require_h3_boundary(request)
    root = _state_root(request)
    marker = root / "session.json"
    if not root.exists():
        return H3Result(status="not_started", attempt_id=None)
    latest = _latest_attempt(request)
    if marker.is_file():
        expected = {"schema_version": SCHEMA_VERSION, "cid": request.cid}
        if _read_json(marker) != expected:
            raise ReceiptError("session_cid_mismatch")
    elif latest is not None:
        raise ReceiptError("state_invalid")

    if latest is not None:
        _validate_state(request, latest, require_client_request_id=False)
    if latest is None:
        return H3Result(status="not_started", attempt_id=None)
    if output_is_reusable(request, latest):
        return _output_result(request, latest)
    return _result(latest)


def timeout_attempt_is_get_only_resumable(
    request: H3Request, expected_attempt_id: str,
) -> bool:
    """Prove that the latest exact attempt still owns a running H3 task."""
    _require_h3_boundary(request)
    try:
        state = _find_attempt(request, request.client_request_id)
    except ReceiptError:
        return False
    h3_state = state.get("h3") if isinstance(state, Mapping) else None
    return bool(
        isinstance(expected_attempt_id, str)
        and isinstance(state, Mapping)
        and state.get("attempt_id") == expected_attempt_id
        and state.get("status") == "retryable_failure"
        and state.get("retryable") is True
        and state.get("error") == {"code": "h3_timeout"}
        and isinstance(h3_state, Mapping)
        and h3_state.get("status") == "running"
        and _task_id(h3_state.get("task_id"), required=False) is not None
    )


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _legacy_input_and_ir_are_bound(state: Mapping[str, Any]) -> bool:
    legacy_input = state.get("input")
    ir_state = state.get("ir")
    legacy_request = (
        legacy_input.get("request") if isinstance(legacy_input, dict) else None
    )
    base_request_keys = frozenset(
        {"duration", "h3_workflow", "ir_model", "ratio", "resolution"}
    )
    request_keys = (
        frozenset(legacy_request) if isinstance(legacy_request, dict) else frozenset()
    )
    if (
        not isinstance(legacy_input, dict)
        or set(legacy_input)
        != {"prompt_sha256", "keyframes", "voice_texts_sha256", "request"}
        or not _is_lower_sha256(legacy_input.get("prompt_sha256"))
        or not _is_lower_sha256(legacy_input.get("voice_texts_sha256"))
        or not isinstance(legacy_input.get("keyframes"), list)
        or not 1 <= len(legacy_input["keyframes"]) <= 9
        or any(
            not isinstance(frame, dict)
            or set(frame) != {"name", "sha256"}
            or not isinstance(frame.get("name"), str)
            or not frame["name"]
            or Path(frame["name"]).name != frame["name"]
            or "\\" in frame["name"]
            or not _is_lower_sha256(frame.get("sha256"))
            for frame in legacy_input["keyframes"]
        )
        or len({frame["name"] for frame in legacy_input["keyframes"]})
        != len(legacy_input["keyframes"])
        or not isinstance(legacy_request, dict)
        or request_keys
        not in {base_request_keys, base_request_keys | {"context_ir_enabled"}}
        or isinstance(legacy_request.get("duration"), bool)
        or not isinstance(legacy_request.get("duration"), int)
        or not 1 <= legacy_request["duration"] <= 15
        or legacy_request.get("h3_workflow") not in H3_REFERENCE_WORKFLOWS
        or legacy_request.get("ir_model") != "MiniMax-H3"
        or legacy_request.get("ratio") != H3_DEFAULT_ASPECT_RATIO
        or legacy_request.get("resolution") != H3_RESOLUTION
        or state.get("input_receipt") != canonical_json_sha256(legacy_input)
        or not isinstance(ir_state, dict)
    ):
        return False
    optimized = ir_state.get("optimized_prompt")
    optimized_sha = ir_state.get("optimized_prompt_sha256")
    if (
        not isinstance(optimized, str)
        or not optimized
        or not _is_lower_sha256(optimized_sha)
        or hashlib.sha256(optimized.encode("utf-8")).hexdigest() != optimized_sha
    ):
        return False
    if set(ir_state) == {
        "mode", "optimized_prompt", "optimized_prompt_sha256", "status",
    }:
        return (
            ir_state.get("status") == "succeeded"
            and ir_state.get("mode") == "skipped"
            and request_keys == base_request_keys | {"context_ir_enabled"}
            and legacy_request.get("context_ir_enabled") is False
            and optimized_sha == legacy_input.get("prompt_sha256")
        )
    if set(ir_state) != {
        "optimized_prompt", "optimized_prompt_sha256", "receipt", "status",
        "task_id",
    }:
        return False
    task_id = ir_state.get("task_id")
    receipt = ir_state.get("receipt")
    return (
        ir_state.get("status") == "succeeded"
        and request_keys == base_request_keys
        and isinstance(task_id, str)
        and bool(task_id.strip())
        and isinstance(receipt, dict)
        and set(receipt)
        == {
            "input_receipt", "keyframes", "prompt_sha256", "request",
            "task_id", "voice_texts_sha256",
        }
        and receipt.get("input_receipt") == state.get("input_receipt")
        and receipt.get("keyframes") == legacy_input.get("keyframes")
        and receipt.get("prompt_sha256") == legacy_input.get("prompt_sha256")
        and receipt.get("voice_texts_sha256")
        == legacy_input.get("voice_texts_sha256")
        and receipt.get("task_id") == task_id
        and receipt.get("request")
        == {
            "duration": legacy_request.get("duration"),
            "model": legacy_request.get("ir_model"),
            "ratio": legacy_request.get("ratio"),
        }
    )


def legacy_h3_is_provably_unsubmitted(
    workdir: Path,
    *,
    cid: str,
    attempt: int,
    client_request_id: str,
) -> bool:
    """Accept a removed pre-H3 flow only with one explicit unpaid receipt.

    Missing, malformed, extra, or paid attempt evidence is ambiguous and must
    stay locked.  This function is deliberately read-only.
    """
    if (
        not isinstance(cid, str)
        or not cid
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= 999999
        or not isinstance(client_request_id, str)
        or not client_request_id
    ):
        return False
    root = Path(workdir) / ".h3"
    attempts = root / "attempts"
    expected_id = f"{attempt:06d}"
    try:
        if _read_json(root / "session.json") != {
            "schema_version": SCHEMA_VERSION,
            "cid": cid,
        }:
            return False
        entries = list(attempts.iterdir())
        if len(entries) != 1:
            return False
        directory = entries[0]
        if not directory.is_dir() or directory.name != expected_id:
            return False
        children = list(directory.iterdir())
        if len(children) != 1 or children[0].name != "attempt.json":
            return False
        state = _read_json(children[0])
    except (OSError, ReceiptError):
        return False
    h3_state = state.get("h3")
    return (
        set(state)
        == {
            "schema_version", "cid", "attempt_id", "client_request_id",
            "input", "input_receipt", "status", "retryable", "ir", "h3",
        }
        and state.get("schema_version") == SCHEMA_VERSION
        and state.get("cid") == cid
        and state.get("attempt_id") == expected_id
        and state.get("client_request_id") == client_request_id
        and state.get("status") == "ready_for_h3"
        and state.get("retryable") is False
        and _legacy_input_and_ir_are_bound(state)
        and isinstance(h3_state, dict)
        and set(h3_state) == {"status"}
        and h3_state.get("status") in {"not_started", "ready"}
        and not (Path(workdir) / "generated.mp4").exists()
    )


def resume(request: H3Request, *, client: httpx.Client | None = None) -> H3Result:
    """Recover existing work, including the strict provider-failure exception."""
    _require_h3_boundary(request)
    with _session_lease(request):
        state = _find_attempt(request, request.client_request_id)
        if output_is_reusable(request, state):
            return _output_result(request, state)
        if state is None:
            return H3Result(status="not_started", attempt_id=None)
        with _client(client) as active_client:
            return _advance_with_provider_retry(
                request,
                state,
                active_client,
                allow_submit=False,
                new_attempt=False,
            )


def retry(
    request: H3Request,
    client_request_id: str,
    *,
    client: httpx.Client | None = None,
) -> H3Result:
    """Explicitly create a paid retry, keyed by a new idempotency key."""
    retried = replace(request, client_request_id=client_request_id)
    _require_h3_boundary(retried)
    with _session_lease(retried):
        existing = _find_attempt(retried, client_request_id)
        if output_is_reusable(retried, existing):
            return _output_result(retried, existing)
        is_new = existing is None
        state = _create_attempt(retried, client_request_id) if is_new else existing
        with _client(client) as active_client:
            return _advance_with_provider_retry(
                retried,
                state,
                active_client,
                allow_submit=True,
                new_attempt=is_new,
            )


def controlled_storage_rejection_is_safely_retryable(
    request: H3Request,
    *,
    legacy_attempt_sha256: str = "",
    legacy_evidence_sha256: str = "",
) -> bool:
    """Recognize only a definite local Gateway storage rejection.

    A legacy exception is operator-authorized by the exact immutable attempt
    bytes plus an external rejection-evidence digest.  It is intentionally not
    inferred from the generic ``h3_submit_rejected`` code.
    """
    _require_h3_boundary(request)
    try:
        state = _find_attempt(request, request.client_request_id)
    except ReceiptError:
        return False
    return _controlled_storage_retry_state_is_valid(
        request,
        state,
        legacy_attempt_sha256=legacy_attempt_sha256,
        legacy_evidence_sha256=legacy_evidence_sha256,
    )


def retry_controlled_storage_rejection(
    request: H3Request,
    *,
    legacy_attempt_sha256: str = "",
    legacy_evidence_sha256: str = "",
    client: httpx.Client | None = None,
) -> H3Result:
    """Append one same-client attempt after a proven local 400 rejection."""
    _require_h3_boundary(request)
    with _session_lease(request):
        previous = _find_attempt(request, request.client_request_id)
        if not _controlled_storage_retry_state_is_valid(
            request,
            previous,
            legacy_attempt_sha256=legacy_attempt_sha256,
            legacy_evidence_sha256=legacy_evidence_sha256,
        ):
            raise H3Error("controlled_storage_retry_not_allowed")
        state = _create_attempt(request, request.client_request_id)
        with _client(client) as active_client:
            return _advance_with_provider_retry(
                request,
                state,
                active_client,
                allow_submit=True,
                new_attempt=True,
            )


def _state_root(request: H3Request) -> Path:
    return request.workdir / ".h3"


def _attempt_path(request: H3Request, attempt_id: str) -> Path:
    return _state_root(request) / "attempts" / attempt_id / "attempt.json"


def load_media_timeline_receipt(
    request: H3Request,
    attempt_id: str,
) -> dict[str, Any]:
    """Load one exact successful attempt's provider media timeline.

    Callers must persist ``attempt_id`` instead of inferring the newest
    attempt.  The helper validates the frozen H3 input, the versioned timeline
    shape, and the exact output bytes before returning a detached JSON object.
    """
    _require_h3_boundary(request)
    if (
        not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
    ):
        raise ReceiptError("state_invalid")
    if _read_json(_state_root(request) / "session.json") != {
        "schema_version": SCHEMA_VERSION,
        "cid": request.cid,
    }:
        raise ReceiptError("state_invalid")
    state = _read_json(_attempt_path(request, attempt_id))
    _validate_state(request, state, require_client_request_id=False)
    h3_state = state.get("h3")
    output_receipt = h3_state.get("output") if isinstance(h3_state, dict) else None
    task_id = (
        _task_id(h3_state.get("task_id"), required=True)
        if isinstance(h3_state, dict)
        else None
    )
    if (
        state.get("attempt_id") != attempt_id
        or state.get("status") != "succeeded"
        or not isinstance(h3_state, dict)
        or h3_state.get("status") != "succeeded"
        or h3_state.get("receipt")
        != _h3_receipt(
            request,
            task_id,
            legacy=_state_uses_legacy_generation_parameters(request, state),
            workflow=_state_workflow(request, state),
        )
        or not isinstance(output_receipt, dict)
        or not _output_receipt_matches_file(
            request.workdir / "generated.mp4", output_receipt
        )
    ):
        raise ReceiptError("state_invalid")
    timeline = output_receipt.get("media_timeline")
    if not _media_timeline_receipt_is_valid(timeline):
        raise ReceiptError("media_timeline_missing")
    return json.loads(json.dumps(timeline, ensure_ascii=False))


def output_is_reusable(
    request: H3Request,
    state: Mapping[str, Any] | None = None,
    *,
    expected_duration_s: float | None = None,
    allow_provider_duration_ceiling: bool = False,
) -> bool:
    """Validate a local output against its exact paid attempt and frozen input."""
    _require_h3_boundary(request)
    if state is None:
        state = _find_attempt(request, request.client_request_id)
    if state is None:
        return False
    marker = _state_root(request) / "session.json"
    if _read_json(marker) != {"schema_version": SCHEMA_VERSION, "cid": request.cid}:
        raise ReceiptError("session_cid_mismatch")
    _validate_state(request, state)
    h3_state = state.get("h3")
    if (
        state.get("status") != "succeeded"
        or not isinstance(h3_state, dict)
        or h3_state.get("status") != "succeeded"
    ):
        return False
    receipt = h3_state.get("output")
    if not isinstance(receipt, dict):
        return False
    output = request.workdir / "generated.mp4"
    try:
        if not _output_receipt_matches_file(output, receipt):
            return False
        duration = _probe_video_duration(output, request.timeouts.probe_s)
        if duration is None:
            return False
        expected = float(
            request.duration if expected_duration_s is None else expected_duration_s
        )
        if not math.isfinite(expected) or expected <= 0:
            return False
        if request.mode == "reference" and not allow_provider_duration_ceiling:
            if abs(duration - expected) > 0.5:
                return False
        elif (
            expected < request.duration - 1 - _DURATION_EPS_S
            or expected > request.duration + _DURATION_EPS_S
            or duration < expected - H3_OUTPUT_FRAME_DURATION_S - _DURATION_EPS_S
            or duration > request.duration + 1
        ):
            return False
        return True
    except OSError:
        return False
    except _ProbeUnavailable:
        return False


def _output_receipt_matches_file(path: Path, receipt: Mapping[str, Any]) -> bool:
    try:
        stat = path.stat()
        if (
            not path.is_file()
            or stat.st_size <= 0
            or receipt.get("size") != stat.st_size
        ):
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == receipt.get("sha256")
    except OSError:
        return False


def legacy_succeeded_output_is_valid(
    workdir: Path,
    *,
    cid: str,
    client_request_id: str,
    attempt: int,
    probe_timeout_s: float = 30.0,
) -> bool:
    """Validate display-only evidence from the removed Context IR contract.

    The legacy-only ``ir``/``keyframes`` discriminator prevents a current
    receipt-aware attempt from bypassing ``output_is_reusable``.
    """
    if (
        not isinstance(cid, str)
        or not cid.strip()
        or not isinstance(client_request_id, str)
        or not client_request_id.strip()
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= 999999
        or isinstance(probe_timeout_s, bool)
        or not isinstance(probe_timeout_s, (int, float))
        or not math.isfinite(float(probe_timeout_s))
        or probe_timeout_s <= 0
    ):
        return False
    attempt_id = f"{attempt:06d}"
    path = (
        Path(workdir) / ".h3" / "attempts" / attempt_id / "attempt.json"
    )
    try:
        state = _read_json(path)
        if set(state) != {
            "schema_version",
            "cid",
            "attempt_id",
            "client_request_id",
            "input",
            "input_receipt",
            "status",
            "retryable",
            "ir",
            "h3",
        }:
            return False
        legacy_input = state.get("input")
        h3_state = state.get("h3")
        if (
            state.get("schema_version") != 1
            or state.get("cid") != cid
            or state.get("attempt_id") != attempt_id
            or state.get("client_request_id") != client_request_id
            or state.get("status") != "succeeded"
            or state.get("retryable") is not False
            or not _legacy_input_and_ir_are_bound(state)
            or not isinstance(h3_state, dict)
            or set(h3_state) != {"status", "task_id", "receipt", "output"}
            or h3_state.get("status") != "succeeded"
            or not isinstance(h3_state.get("task_id"), str)
            or not h3_state["task_id"].strip()
        ):
            return False
        legacy_request = legacy_input["request"]
        optimized_prompt_sha256 = state["ir"]["optimized_prompt_sha256"]
        h3_receipt = h3_state.get("receipt")
        if (
            not isinstance(h3_receipt, dict)
            or set(h3_receipt) != {
                "input_receipt", "keyframes", "prompt_sha256", "request", "task_id",
            }
            or h3_receipt.get("input_receipt") != state.get("input_receipt")
            or h3_receipt.get("keyframes") != legacy_input.get("keyframes")
            or h3_receipt.get("prompt_sha256") != optimized_prompt_sha256
            or h3_receipt.get("task_id") != h3_state.get("task_id")
            or h3_receipt.get("request") != {
                "duration": min(legacy_request["duration"], H3_MAX_DURATION_S),
                "resolution": legacy_request.get("resolution"),
                "workflow": legacy_request.get("h3_workflow"),
            }
        ):
            return False
        receipt = h3_state.get("output")
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"name", "sha256", "size"}
            or receipt.get("name") != "generated.mp4"
            or not isinstance(receipt.get("sha256"), str)
            or len(receipt["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in receipt["sha256"])
            or isinstance(receipt.get("size"), bool)
            or not isinstance(receipt.get("size"), int)
            or receipt["size"] <= 0
        ):
            return False
        output = Path(workdir) / "generated.mp4"
        stat = output.stat()
        if not output.is_file() or stat.st_size != receipt["size"]:
            return False
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != receipt["sha256"]:
            return False
        duration = _probe_video_duration(output, float(probe_timeout_s))
        return duration is not None and math.isfinite(duration) and duration > 0
    except (OSError, ReceiptError, _ProbeUnavailable):
        return False


def _output_result(request: H3Request, state: Mapping[str, Any] | None) -> H3Result:
    attempt_id = str(state["attempt_id"]) if state is not None else None
    return H3Result(
        status="succeeded",
        attempt_id=attempt_id,
        output=request.workdir / "generated.mp4",
        media_timeline=_state_media_timeline(state),
    )


@contextmanager
def _client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    if client is not None:
        yield client
        return
    with httpx.Client(trust_env=False) as owned:
        yield owned


@contextmanager
def _session_lease(request: H3Request) -> Iterator[None]:
    root = _state_root(request)
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(root / "session.lock", os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        raise H3Error("state_unavailable") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise H3BusyError("session_busy") from None
        _ensure_session_marker(request)
        yield
    finally:
        os.close(fd)


def _ensure_session_marker(request: H3Request) -> None:
    marker = _state_root(request) / "session.json"
    payload = {"schema_version": SCHEMA_VERSION, "cid": request.cid}
    try:
        _atomic_create_json(marker, payload)
        return
    except FileExistsError:
        pass
    except OSError:
        raise H3Error("state_unavailable") from None
    existing = _read_json(marker)
    if existing != payload:
        raise ReceiptError("session_cid_mismatch")


def _input_manifest(
    request: H3Request, *, workflow: str | None = None,
) -> dict[str, Any]:
    selected_workflow = workflow or _workflow(request)
    projected = _provider_resolution(request)
    if request.mode == "reference":
        provider_request = {
            "h3_workflow": selected_workflow,
            "duration": request.duration,
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
            "provider_resolution": projected,
        }
        if request.seed is not None:
            provider_request["seed"] = request.seed
        manifest = {
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "keyframes": [
                {"name": path.name, "sha256": hashlib.sha256(blob).hexdigest()}
                for path, blob in request.keyframes
            ],
            "voice_texts_sha256": request.voice_receipt,
            "request": provider_request,
        }
        if selected_workflow in H3_MULTIMODAL_WORKFLOWS:
            provider_request["payload_sha256"] = canonical_json_sha256(
                _provider_body(request)
            )
            manifest["multimodal"] = {
                "skill_plan_sha256": request.skill_plan_sha256,
                "upstream_dialogue_receipt_sha256": (
                    request.upstream_dialogue_receipt_sha256
                ),
                "compiler_version": request.multimodal_compiler_version,
                "speaker_timing_sha256": request.speaker_timing_sha256,
                **(
                    {"speaker_timing_authority": speaker_timing_authority_manifest(request)}
                    if speaker_timing_authority_manifest(request) is not None
                    else {}
                ),
                "on_screen_dialogue": list(request.on_screen_dialogue),
                "on_screen_dialogue_sha256": (
                    request.on_screen_dialogue_sha256
                ),
                "audio_required": request.audio_required,
                "reference_audio_semantics": "conditioning_only",
                "reference_audios": _reference_audio_manifest(request),
                "context_ir": {
                    "receipt_path": str(request.context_ir_receipt_path),
                    "receipt_sha256": request.context_ir_receipt_sha256,
                },
                "gateway_inputs": _gateway_input_manifest(request),
            }
        return manifest
    provider_request = {
        "mode": request.mode,
        "h3_workflow": selected_workflow,
        "duration": request.duration,
        "aspect_ratio": request.aspect_ratio,
        "resolution": request.resolution,
        "provider_resolution": projected,
    }
    return {
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "images": _image_manifest(request),
        "voice_texts_sha256": request.voice_receipt,
        "request": provider_request,
    }


def _pre_controlled_storage_input_manifest(
    request: H3Request, *, workflow: str | None = None,
) -> dict[str, Any] | None:
    selected_workflow = workflow or _workflow(request)
    if selected_workflow not in H3_MULTIMODAL_WORKFLOWS:
        return None
    manifest = _input_manifest(request, workflow=selected_workflow)
    legacy_root = (
        request.workdir / ".h3" / "multimodal-inputs"
        / str(request.skill_plan_sha256)
    )
    images = tuple(
        legacy_root
        / f"image-{index}-{hashlib.sha256(blob).hexdigest()}{path.suffix.lower()}"
        for index, (path, blob) in enumerate(request.keyframes, 1)
    )
    audios = tuple(
        legacy_root / f"audio-{audio.order}-{audio.sha256}.{audio.format}"
        for audio in request.reference_audios
    )
    body = {
        "mode": (
            "multimodal_hd"
            if selected_workflow == H3_MULTIMODAL_HD_WORKFLOW
            else "multimodal"
        ),
        "prompt": request.prompt,
        "duration_sec": request.duration,
        "aspect_ratio": request.aspect_ratio,
        "resolution": request.resolution,
        "images": [str(path) for path in images],
        "audios": [
            {
                "path": str(path),
                "kind": "voice" if audio.purpose == "voice" else "sound",
                "label": f"{audio.purpose}-{audio.order}",
            }
            for audio, path in zip(request.reference_audios, audios, strict=True)
        ],
    }
    manifest["request"]["payload_sha256"] = canonical_json_sha256(body)
    del manifest["multimodal"]["gateway_inputs"]
    return manifest


def _controlled_storage_retry_state_is_valid(
    request: H3Request,
    state: Mapping[str, Any] | None,
    *,
    legacy_attempt_sha256: str,
    legacy_evidence_sha256: str,
) -> bool:
    if not isinstance(state, Mapping) or request.gateway_storage_root is None:
        return False
    if (
        state.get("status") != "failed"
        or state.get("retryable") is not False
        or state.get("client_request_id") != request.client_request_id
        or state.get("h3") != {"status": "failed"}
    ):
        return False
    error = state.get("error")
    if error == {
        "code": "h3_submit_rejected",
        "gateway": {"http_status": 400, "reason": "controlled_storage"},
    }:
        return True
    legacy_manifest = _pre_controlled_storage_input_manifest(request)
    if (
        not _is_sha256(legacy_attempt_sha256)
        or not _is_sha256(legacy_evidence_sha256)
        or state.get("error") != {"code": "h3_submit_rejected"}
        or state.get("input") != legacy_manifest
        or state.get("input_receipt") != canonical_json_sha256(legacy_manifest)
    ):
        return False
    attempt_id = state.get("attempt_id")
    if not isinstance(attempt_id, str):
        return False
    try:
        raw = _attempt_path(request, attempt_id).read_bytes()
    except OSError:
        return False
    return hashlib.sha256(raw).hexdigest() == legacy_attempt_sha256


def _workflow(request: H3Request) -> str:
    if request.workflow is not None:
        return request.workflow
    return H3_WORKFLOW if request.mode == "reference" else H3_BOUNDARY_WORKFLOW


def is_multimodal_request(request: H3Request) -> bool:
    """Return the receipt-stable discriminator for native H3 audio requests."""
    return (
        isinstance(request, H3Request)
        and _workflow(request) in H3_MULTIMODAL_WORKFLOWS
        and request.audio_required is True
        and bool(request.reference_audios)
    )


def _provider_resolution(request: H3Request) -> str:
    if _workflow(request) == H3_MULTIMODAL_HD_WORKFLOW:
        return request.resolution + ("横" if request.aspect_ratio == "16:9" else "竖")
    return provider_resolution(request.aspect_ratio, request.resolution)


def _state_workflow(request: H3Request, state: Mapping[str, Any]) -> str:
    stored_input = state.get("input")
    stored_request = (
        stored_input.get("request") if isinstance(stored_input, Mapping) else None
    )
    stored_workflow = (
        stored_request.get("h3_workflow")
        if isinstance(stored_request, Mapping)
        else None
    )
    allowed = (
        H3_REFERENCE_WORKFLOWS
        if request.mode == "reference"
        else frozenset({H3_BOUNDARY_WORKFLOW})
    )
    if stored_workflow not in allowed:
        raise ReceiptError("receipt_mismatch")
    return stored_workflow


def _image_inputs(request: H3Request) -> tuple[tuple[str, FrozenFrame], ...]:
    if request.mode == "reference":
        return tuple(
            (f"ref_image_{index}", frame)
            for index, frame in enumerate(request.keyframes)
        )
    assert request.first_frame is not None and request.last_frame is not None
    return (
        ("first_frame", request.first_frame),
        ("last_frame", request.last_frame),
    )


def _image_manifest(request: H3Request) -> list[dict[str, str]]:
    return [
        {
            "role": role,
            "name": path.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        for role, (path, blob) in _image_inputs(request)
    ]


def _reference_audio_manifest(request: H3Request) -> list[dict[str, Any]]:
    return [
        {
            "role": f"ref_audio_{audio.order - 1}",
            "name": audio.path.name,
            "order": audio.order,
            "purpose": audio.purpose,
            "format": audio.format,
            "sha256": audio.sha256,
            "size": len(audio.data),
            "duration_s": audio.duration_s,
        }
        for audio in request.reference_audios
    ]


def _gateway_media_paths(request: H3Request) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    assert request.skill_plan_sha256 is not None
    if request.gateway_storage_root is None:
        raise H3Error("gateway_storage_root_required")
    root = (
        request.gateway_storage_root
        / hashlib.sha256(request.cid.encode("utf-8")).hexdigest()
        / request.skill_plan_sha256
    )
    images = tuple(
        root / f"image-{index}-{hashlib.sha256(blob).hexdigest()}{path.suffix.lower()}"
        for index, (path, blob) in enumerate(request.keyframes, 1)
    )
    audios = tuple(
        root / f"audio-{audio.order}-{audio.sha256}.{audio.format}"
        for audio in request.reference_audios
    )
    return images, audios


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ReceiptError("gateway_input_symlink")


def _gateway_input_manifest(request: H3Request) -> list[dict[str, Any]]:
    images, audios = _gateway_media_paths(request)
    entries: list[dict[str, Any]] = []
    for order, ((source, blob), provider) in enumerate(
        zip(request.keyframes, images, strict=True), 1
    ):
        _reject_symlink_components(source.absolute())
        digest = hashlib.sha256(blob).hexdigest()
        entries.append({
            "order": order,
            "role": "reference_image",
            "source_path": str(source.absolute()),
            "source_sha256": digest,
            "provider_path": str(provider),
            "provider_sha256": digest,
        })
    for audio, provider in zip(request.reference_audios, audios, strict=True):
        _reject_symlink_components(audio.path.absolute())
        entries.append({
            "order": audio.order,
            "role": "reference_audio",
            "purpose": audio.purpose,
            "source_path": str(audio.path.absolute()),
            "source_sha256": audio.sha256,
            "provider_path": str(provider),
            "provider_sha256": audio.sha256,
        })
    return entries


def _provider_body(request: H3Request) -> dict[str, Any]:
    if _workflow(request) in H3_MULTIMODAL_WORKFLOWS:
        images, audios = _gateway_media_paths(request)
        return {
            "mode": (
                "multimodal_hd"
                if _workflow(request) == H3_MULTIMODAL_HD_WORKFLOW
                else "multimodal"
            ),
            "prompt": request.prompt,
            "duration_sec": request.duration,
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
            "images": [str(path) for path in images],
            "audios": [
                {
                    "path": str(path),
                    "kind": "voice" if audio.purpose == "voice" else "sound",
                    "label": f"{audio.purpose}-{audio.order}",
                }
                for audio, path in zip(request.reference_audios, audios, strict=True)
            ],
        }
    body: dict[str, Any] = {
        "prompt": request.prompt,
        "duration": request.duration,
        "resolution": _provider_resolution(request),
    }
    if request.seed is not None:
        body["seed"] = request.seed
    for role, (_path, blob) in _image_inputs(request):
        body[role] = "data:image/png;base64," + base64.b64encode(blob).decode("ascii")
    return body


def _materialize_gateway_inputs(request: H3Request) -> None:
    root = request.gateway_storage_root
    if root is None:
        raise H3Error("gateway_storage_root_required")
    _reject_symlink_components(root.absolute())
    bound = _gateway_input_manifest(request)
    blobs = tuple(blob for _path, blob in request.keyframes) + tuple(
        audio.data for audio in request.reference_audios
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve(strict=True)
        for item, blob in zip(bound, blobs, strict=True):
            path = Path(item["provider_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            _reject_symlink_components(path.absolute())
            try:
                path.parent.resolve(strict=True).relative_to(root_resolved)
            except ValueError:
                raise ReceiptError("gateway_input_path_invalid") from None
            if path.exists():
                if not path.is_file() or path.is_symlink():
                    raise ReceiptError("gateway_input_symlink")
                if path.read_bytes() != blob:
                    raise ReceiptError("receipt_mismatch") from None
                continue
            _atomic_write_bytes(path, blob)
            if (
                path.resolve(strict=True).parent != path.parent.resolve(strict=True)
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != item["provider_sha256"]
            ):
                raise ReceiptError("receipt_mismatch")
    except ReceiptError:
        raise
    except OSError:
        raise H3Error("state_unavailable") from None


def _new_state(request: H3Request, attempt_id: str, client_request_id: str) -> dict[str, Any]:
    manifest = _input_manifest(request)
    return {
        "schema_version": SCHEMA_VERSION,
        "cid": request.cid,
        "attempt_id": attempt_id,
        "client_request_id": client_request_id,
        "input": manifest,
        "input_receipt": canonical_json_sha256(manifest),
        "status": "ready_to_submit",
        "retryable": False,
        "h3": {"status": "ready"},
    }


def _create_attempt(request: H3Request, client_request_id: str) -> dict[str, Any]:
    attempts = _state_root(request) / "attempts"
    try:
        attempts.mkdir(parents=True, exist_ok=True)
        numbers = [
            int(path.name)
            for path in attempts.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
        attempt_id = f"{max(numbers, default=0) + 1:06d}"
        directory = attempts / attempt_id
        directory.mkdir()
        state = _new_state(request, attempt_id, client_request_id)
        _atomic_create_json(directory / "attempt.json", state)
        return state
    except (OSError, ValueError):
        raise H3Error("attempt_claim_failed") from None


def _find_attempt(request: H3Request, client_request_id: str) -> dict[str, Any] | None:
    attempts = _state_root(request) / "attempts"
    if not attempts.is_dir():
        return None
    try:
        paths = sorted(attempts.glob("*/attempt.json"), reverse=True)
    except OSError:
        raise H3Error("state_unavailable") from None
    for path in paths:
        raw = _read_json(path)
        if raw.get("client_request_id") == client_request_id:
            _validate_state(request, raw)
            return raw
    return None


def _attempt_chain(
    request: H3Request, state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load the exact same-request/input attempt chain in creation order."""
    ledger = _validated_attempt_ledger(request)
    chain = []
    for raw in ledger:
        if raw.get("client_request_id") != request.client_request_id:
            continue
        _validate_state(request, raw)
        if raw.get("input_receipt") == state.get("input_receipt"):
            chain.append(raw)
    return chain


def _validated_attempt_ledger(request: H3Request) -> list[dict[str, Any]]:
    """Validate the append-only attempt directory structure without input drift.

    Unrelated client ids need only structural validation here. Matching records
    receive the full current-request receipt validation in ``_attempt_chain``.
    """
    root = _state_root(request)
    if _read_json(root / "session.json") != {
        "schema_version": SCHEMA_VERSION,
        "cid": request.cid,
    }:
        raise ReceiptError("state_invalid")
    attempts = root / "attempts"
    try:
        numbered = sorted(
            (path for path in attempts.iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except OSError:
        raise H3Error("state_unavailable") from None
    if not numbered:
        raise ReceiptError("state_invalid")
    expected_names = [f"{index:06d}" for index in range(1, len(numbered) + 1)]
    if [path.name for path in numbered] != expected_names:
        raise ReceiptError("state_invalid")
    ledger = []
    for path in numbered:
        attempt_path = path / "attempt.json"
        if not path.is_dir() or not attempt_path.is_file():
            raise ReceiptError("state_invalid")
        raw = _read_json(attempt_path)
        if (
            raw.get("schema_version") != SCHEMA_VERSION
            or raw.get("cid") != request.cid
            or raw.get("attempt_id") != path.name
        ):
            raise ReceiptError("state_invalid")
        ledger.append(raw)
    return ledger


def _is_complete_provider_failure(state: Mapping[str, Any]) -> bool:
    error = state.get("error")
    h3_state = state.get("h3")
    return (
        state.get("status") == "failed"
        and state.get("retryable") is False
        and isinstance(error, dict)
        and set(error) == {"code", "provider"}
        and error.get("code") == "h3_provider_failed"
        and isinstance(error.get("provider"), dict)
        and isinstance(h3_state, dict)
        and set(h3_state) == {"status", "task_id", "receipt"}
        and h3_state.get("status") == "failed"
        and _task_id(h3_state.get("task_id"), required=True) is not None
        and isinstance(h3_state.get("receipt"), dict)
    )


def _is_ready_automatic_attempt(
    state: Mapping[str, Any], chain: Sequence[Mapping[str, Any]],
) -> bool:
    if len(chain) < 2 or chain[-1].get("attempt_id") != state.get("attempt_id"):
        return False
    previous = chain[-2]
    h3_state = state.get("h3")
    try:
        sequential = int(str(state.get("attempt_id"))) == int(
            str(previous.get("attempt_id"))
        ) + 1
    except ValueError:
        return False
    return (
        sequential
        and _is_complete_provider_failure(previous)
        and state.get("status") == "ready_to_submit"
        and state.get("retryable") is False
        and "error" not in state
        and isinstance(h3_state, dict)
        and h3_state == {"status": "ready"}
    )


def _latest_attempt(request: H3Request) -> dict[str, Any] | None:
    attempts = _state_root(request) / "attempts"
    if not attempts.is_dir():
        return None
    try:
        path = next(iter(sorted(attempts.glob("*/attempt.json"), reverse=True)), None)
    except OSError:
        raise H3Error("state_unavailable") from None
    return _read_json(path) if path is not None else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReceiptError("state_invalid") from None
    if not isinstance(value, dict):
        raise ReceiptError("state_invalid")
    return value


def _legacy_input_manifest(
    request: H3Request, *, workflow: str | None = None,
) -> dict[str, Any] | None:
    if request.reference_audios:
        return None
    selected_workflow = workflow or _workflow(request)
    if (
        request.aspect_ratio != H3_DEFAULT_ASPECT_RATIO
        or request.resolution != H3_DEFAULT_RESOLUTION
    ):
        return None
    if request.mode == "reference":
        provider_request = {
            "h3_workflow": selected_workflow,
            "duration": request.duration,
            "resolution": H3_RESOLUTION,
        }
        if request.seed is not None:
            provider_request["seed"] = request.seed
        return {
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "keyframes": [
                {"name": path.name, "sha256": hashlib.sha256(blob).hexdigest()}
                for path, blob in request.keyframes
            ],
            "voice_texts_sha256": request.voice_receipt,
            "request": provider_request,
        }
    return {
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "images": _image_manifest(request),
        "voice_texts_sha256": request.voice_receipt,
        "request": {
            "mode": request.mode,
            "h3_workflow": selected_workflow,
            "duration": request.duration,
            "resolution": H3_RESOLUTION,
        },
    }


def _output_receipt_is_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) not in (
        {"name", "sha256", "size"},
        {"name", "sha256", "size", "media_timeline"},
    ):
        return False
    if (
        value.get("name") != "generated.mp4"
        or not _is_sha256(value.get("sha256"))
        or isinstance(value.get("size"), bool)
        or not isinstance(value.get("size"), int)
        or value["size"] <= 0
    ):
        return False
    return (
        "media_timeline" not in value
        or _media_timeline_receipt_is_valid(value["media_timeline"])
    )


def _media_timeline_receipt_is_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "version",
        "decode_complete",
        "container",
        "video",
        "audio",
        "av_delta_s",
    }:
        return False
    if (
        value.get("schema") != MEDIA_TIMELINE_SCHEMA
        or value.get("version") != MEDIA_TIMELINE_VERSION
        or value.get("decode_complete") is not True
    ):
        return False
    container = value.get("container")
    if (
        not isinstance(container, dict)
        or set(container) != {"format_name", "start_time_s", "duration_s"}
        or not isinstance(container.get("format_name"), str)
        or not container["format_name"].strip()
        or not _optional_receipt_seconds(container.get("start_time_s"))
        or not _optional_receipt_seconds(
            container.get("duration_s"), positive=True
        )
        or not _stream_timeline_receipt_is_valid(value.get("video"), kind="video")
    ):
        return False
    audio = value.get("audio")
    av_delta = value.get("av_delta_s")
    if audio is None:
        return av_delta is None
    if (
        not _stream_timeline_receipt_is_valid(audio, kind="audio")
        or not isinstance(av_delta, dict)
        or set(av_delta) != {"start", "end"}
        or not _finite_receipt_number(av_delta.get("start"))
        or not _finite_receipt_number(av_delta.get("end"))
    ):
        return False
    video = value["video"]
    expected_start = _round_seconds(
        audio["first_frame_pts_s"] - video["first_frame_pts_s"]
    )
    expected_end = _round_seconds(audio["frame_end_s"] - video["frame_end_s"])
    return (
        av_delta["start"] == expected_start
        and av_delta["end"] == expected_end
        and abs(expected_start) <= MAX_AV_TIMELINE_DELTA_S
        and abs(expected_end) <= MAX_AV_TIMELINE_DELTA_S
    )


def _stream_timeline_receipt_is_valid(
    value: Any,
    *,
    kind: Literal["video", "audio"],
) -> bool:
    common = {
        "index",
        "codec_name",
        "time_base",
        "start_time_s",
        "duration_s",
        "packet_count",
        "first_packet_pts_s",
        "last_packet_pts_s",
        "packet_end_s",
        "packet_dts_monotonic",
        "frame_count",
        "first_frame_pts_s",
        "last_frame_pts_s",
        "frame_end_s",
        "presentation_monotonic",
    }
    expected = common | (
        {"avg_frame_rate", "r_frame_rate"}
        if kind == "video"
        else {"sample_rate", "channels", "decoded_sha256"}
    )
    if not isinstance(value, dict) or set(value) != expected:
        return False
    if (
        isinstance(value.get("index"), bool)
        or not isinstance(value.get("index"), int)
        or value["index"] < 0
        or not isinstance(value.get("codec_name"), str)
        or not value["codec_name"].strip()
        or value.get("packet_dts_monotonic") is not True
        or value.get("presentation_monotonic") is not True
    ):
        return False
    try:
        _positive_fraction(value.get("time_base"))
        if kind == "video":
            avg = _fraction_value(_positive_fraction(value.get("avg_frame_rate")))
            nominal = _fraction_value(_positive_fraction(value.get("r_frame_rate")))
            if avg > 240 or nominal > 240:
                return False
    except H3Error:
        return False
    if any(
        isinstance(value.get(key), bool)
        or not isinstance(value.get(key), int)
        or value[key] <= 0
        for key in ("packet_count", "frame_count")
    ):
        return False
    numeric_keys = (
        "start_time_s",
        "duration_s",
        "first_packet_pts_s",
        "last_packet_pts_s",
        "packet_end_s",
        "first_frame_pts_s",
        "last_frame_pts_s",
        "frame_end_s",
    )
    if not all(_finite_receipt_number(value.get(key)) for key in numeric_keys):
        return False
    if (
        value["duration_s"] <= 0
        or value["first_packet_pts_s"] > value["last_packet_pts_s"]
        or value["last_packet_pts_s"] >= value["packet_end_s"]
        or value["first_frame_pts_s"] > value["last_frame_pts_s"]
        or value["last_frame_pts_s"] >= value["frame_end_s"]
        or abs(value["start_time_s"] - value["first_frame_pts_s"])
        > MAX_AV_TIMELINE_DELTA_S
        or abs(
            value["start_time_s"] + value["duration_s"] - value["frame_end_s"]
        )
        > MAX_AV_TIMELINE_DELTA_S
    ):
        return False
    if kind == "audio" and (
        isinstance(value.get("sample_rate"), bool)
        or not isinstance(value.get("sample_rate"), int)
        or not 8000 <= value["sample_rate"] <= 384000
        or isinstance(value.get("channels"), bool)
        or not isinstance(value.get("channels"), int)
        or not 1 <= value["channels"] <= 32
        or not _is_sha256(value.get("decoded_sha256"))
    ):
        return False
    return True


def _finite_receipt_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _optional_receipt_seconds(value: Any, *, positive: bool = False) -> bool:
    return value is None or (
        _finite_receipt_number(value) and (not positive or value > 0)
    )


def _validate_state(
    request: H3Request,
    state: Mapping[str, Any],
    *,
    require_client_request_id: bool = True,
) -> None:
    stored_workflow = _state_workflow(request, state)
    manifest = _input_manifest(request, workflow=stored_workflow)
    legacy_manifest = _legacy_input_manifest(request, workflow=stored_workflow)
    legacy = legacy_manifest is not None and state.get("input") == legacy_manifest
    storage_legacy_manifest = _pre_controlled_storage_input_manifest(
        request, workflow=stored_workflow
    )
    storage_legacy = (
        storage_legacy_manifest is not None
        and state.get("input") == storage_legacy_manifest
    )
    if legacy:
        manifest = legacy_manifest
    elif storage_legacy:
        manifest = storage_legacy_manifest
    attempt_id = state.get("attempt_id")
    stored_request_id = state.get("client_request_id")
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("cid") != request.cid
        or not isinstance(attempt_id, str)
        or len(attempt_id) != 6
        or not attempt_id.isdigit()
        or not isinstance(stored_request_id, str)
        or not stored_request_id.strip()
        or (
            require_client_request_id
            and stored_request_id != request.client_request_id
        )
        or state.get("input") != manifest
        or state.get("input_receipt") != canonical_json_sha256(manifest)
    ):
        raise ReceiptError("receipt_mismatch")
    h3_state = state.get("h3")
    if not isinstance(h3_state, dict):
        raise ReceiptError("state_invalid")
    error = state.get("error")
    if error is not None and (
        not isinstance(error, dict) or error.get("code") not in _SAFE_ERROR_CODES
    ):
        raise ReceiptError("state_invalid")
    if isinstance(error, dict) and "provider" in error:
        provider = error["provider"]
        if (
            error.get("code") != "h3_provider_failed"
            or set(error) != {"code", "provider"}
            or not isinstance(provider, dict)
            or not {"status", "detail"}.issubset(provider)
            or not set(provider).issubset({"status", "detail", "request_id"})
            or provider.get("status") not in {"FAILED", "ERROR", "FAIL"}
            or not isinstance(provider.get("detail"), str)
            or len(provider["detail"]) > 300
            or provider["detail"]
            != _safe_provider_field(
                provider["detail"], limit=300, secret=request.autodl_token
            )
            or (
                "request_id" in provider
                and (
                    not isinstance(provider["request_id"], str)
                    or not provider["request_id"]
                    or len(provider["request_id"]) > 128
                    or provider["request_id"]
                    != _safe_provider_field(
                        provider["request_id"],
                        limit=128,
                        secret=request.autodl_token,
                    )
                )
            )
        ):
            raise ReceiptError("state_invalid")
    if isinstance(error, dict) and "gateway" in error:
        if error != {
            "code": "h3_submit_rejected",
            "gateway": {"http_status": 400, "reason": "controlled_storage"},
        }:
            raise ReceiptError("state_invalid")
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    if h3_task_id is not None:
        if h3_state.get("receipt") != _h3_receipt(
            request, h3_task_id, legacy=legacy, workflow=stored_workflow
        ):
            raise ReceiptError("receipt_mismatch")
    if "result_url" in h3_state:
        raise ReceiptError("state_invalid")
    output_receipt = h3_state.get("output")
    if output_receipt is not None and not _output_receipt_is_valid(output_receipt):
        raise ReceiptError("state_invalid")


def _state_uses_legacy_generation_parameters(
    request: H3Request, state: Mapping[str, Any],
) -> bool:
    stored_input = state.get("input")
    stored_request = (
        stored_input.get("request") if isinstance(stored_input, Mapping) else None
    )
    stored_workflow = (
        stored_request.get("h3_workflow")
        if isinstance(stored_request, Mapping)
        else None
    )
    legacy = _legacy_input_manifest(request, workflow=stored_workflow)
    return legacy is not None and state.get("input") == legacy


def _task_id(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReceiptError("state_invalid")
    normalized = str(value).strip()
    if not normalized:
        raise ReceiptError("state_invalid")
    return normalized


def _h3_receipt(
    request: H3Request,
    task_id: str,
    *,
    legacy: bool = False,
    workflow: str | None = None,
) -> dict[str, Any]:
    selected_workflow = workflow or _workflow(request)
    manifest = (
        _legacy_input_manifest(request, workflow=selected_workflow)
        if legacy
        else _input_manifest(request, workflow=selected_workflow)
    )
    if manifest is None:
        raise ReceiptError("receipt_mismatch")
    projected = (
        H3_RESOLUTION
        if legacy
        else _provider_resolution(request)
    )
    if request.mode == "reference":
        provider_request = {
            "workflow": selected_workflow,
            "duration": request.duration,
            "resolution": projected,
        }
        if not legacy:
            provider_request.update(
                aspect_ratio=request.aspect_ratio,
                semantic_resolution=request.resolution,
            )
        if request.seed is not None:
            provider_request["seed"] = request.seed
        receipt = {
            "task_id": task_id,
            "input_receipt": canonical_json_sha256(manifest),
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "keyframes": _input_manifest(request)["keyframes"],
            "request": provider_request,
        }
        if selected_workflow in H3_MULTIMODAL_WORKFLOWS:
            provider_request["payload_sha256"] = manifest["request"][
                "payload_sha256"
            ]
            receipt["multimodal"] = manifest["multimodal"]
        return receipt
    provider_request = {
        "mode": request.mode,
        "workflow": _workflow(request),
        "duration": request.duration,
        "resolution": projected,
    }
    if not legacy:
        provider_request.update(
            aspect_ratio=request.aspect_ratio,
            semantic_resolution=request.resolution,
        )
    return {
        "task_id": task_id,
        "input_receipt": canonical_json_sha256(manifest),
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "images": _image_manifest(request),
        "request": provider_request,
    }


def _advance(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    *,
    allow_submit: bool,
    new_attempt: bool,
) -> H3Result:
    _validate_state(request, state)
    if output_is_reusable(request, state):
        return _output_result(request, state)
    status = str(state.get("status") or "")
    if status == "submission_unknown":
        return _result(state)
    if status == "failed":
        return _result(state)

    h3_state = state["h3"]
    h3_task_id = _task_id(h3_state.get("task_id"), required=False)
    if h3_task_id is None:
        if h3_state.get("status") == "ready":
            if allow_submit:
                h3_task_id = _submit_h3(request, state, client)
                return _poll_h3(request, state, client, h3_task_id)
            return H3Result("not_started", str(state["attempt_id"]))
        if h3_state.get("status") == "submitting":
            if new_attempt and allow_submit:
                h3_task_id = _submit_h3(request, state, client)
                return _poll_h3(request, state, client, h3_task_id)
            state["status"] = "submission_unknown"
            state["retryable"] = False
            h3_state["status"] = "submission_unknown"
            _save_state(request, state)
            return _result(state)
        raise ReceiptError("state_invalid")

    return _poll_h3(request, state, client, h3_task_id)


def _advance_with_provider_retry(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    *,
    allow_submit: bool,
    new_attempt: bool,
) -> H3Result:
    """Advance while the strictly verified provider-failure budget permits.

    This is the sole automatic new-POST path. It deliberately excludes the
    public ``submit`` boundary so a prepared fan-out child still crosses only
    one provider POST per submit call.
    """
    chain = _attempt_chain(request, state)
    limit = request.timeouts.retry_count + 1
    # A ready automatic attempt already consumed its quota when its receipt was
    # created. Lowering the live limit stops further creation, but must not
    # strand that exactly identified unpaid attempt after a crash.
    resume_ready = _is_ready_automatic_attempt(state, chain)
    result = _advance(
        request,
        state,
        client,
        allow_submit=allow_submit or resume_ready,
        new_attempt=new_attempt,
    )
    while _is_complete_provider_failure(state):
        chain = _attempt_chain(request, state)
        if (
            not chain
            or chain[-1].get("attempt_id") != state.get("attempt_id")
            or len(chain) >= limit
        ):
            return result
        log.warning(
            "H3 provider failure retry cid=%s attempt=%s next=%d/%d in %.1fs",
            request.cid,
            state.get("attempt_id"),
            len(chain) + 1,
            limit,
            request.timeouts.retry_interval_s,
        )
        _pause(request.timeouts.retry_interval_s)
        state = _create_attempt(request, request.client_request_id)
        result = _advance(
            request,
            state,
            client,
            allow_submit=True,
            new_attempt=True,
        )
    return result


def _query_json_with_retry(
    request: H3Request,
    operation: Callable[[], httpx.Response],
    *,
    code: str,
    step: str,
    deadline: float,
) -> tuple[httpx.Response, dict[str, Any]]:
    """Retry only same-task GET failures; provider POSTs never enter here."""

    def attempt() -> tuple[httpx.Response, dict[str, Any]]:
        try:
            response = operation()
        except httpx.HTTPError:
            raise _AutomaticRetryH3Error(code, retryable=True) from None
        if response.status_code != 200:
            error_type = (
                _AutomaticRetryH3Error
                if _retryable_http_status(response.status_code)
                else H3Error
            )
            raise error_type(code, retryable=True)
        try:
            payload = _response_json(response)
        except (ValueError, TypeError):
            raise _AutomaticRetryH3Error(code, retryable=True) from None
        return response, payload

    return _run_automatic_retry(
        request.timeouts,
        attempt,
        step=step,
        deadline=deadline,
    )


def _submit_h3(request: H3Request, state: dict[str, Any], client: httpx.Client) -> str:
    workflow = _state_workflow(request, state)
    use_gateway = workflow in H3_MULTIMODAL_WORKFLOWS
    body = _provider_body(request)
    if use_gateway:
        _materialize_gateway_inputs(request)
    state["h3"] = {"status": "submitting"}
    state["status"] = "h3_submitting"
    state["retryable"] = False
    _save_state(request, state)
    try:
        response = client.post(
            (
                f"{H3_GATEWAY_BASE_URL}/v1/videos"
                if use_gateway
                else f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/{workflow}"
            ),
            headers={} if use_gateway else {"Authorization": request.autodl_token},
            json=body,
            timeout=request.timeouts.request_s,
        )
        payload = _response_json(response)
    except (httpx.HTTPError, ValueError, TypeError):
        _submission_unknown(request, state, "h3")
        raise H3Error("submission_unknown") from None
    data = payload.get("data")
    task_value = payload.get("task_id") if use_gateway else (
        data.get("task_id") if isinstance(data, dict) else None
    )
    expected_status = 201 if use_gateway else 200
    if response.status_code != expected_status or isinstance(task_value, bool) or not isinstance(
        task_value, (str, int)
    ) or not str(task_value).strip():
        detail = _provider_error_detail(payload, secret=request.autodl_token)
        log.warning(
            "H3 submission rejected cid=%s attempt=%s http_status=%d detail=%s",
            request.cid,
            state.get("attempt_id"),
            response.status_code,
            detail or "no_safe_detail",
        )
        gateway_diagnostic = None
        if (
            use_gateway
            and response.status_code == 400
            and payload.get("error") == "图片不在受控存储内"
        ):
            gateway_diagnostic = {
                "http_status": 400,
                "reason": "controlled_storage",
            }
        _fail(
            request,
            state,
            "h3_submit_rejected",
            retryable=False,
            gateway_diagnostic=gateway_diagnostic,
        )
        raise H3Error("h3_submit_rejected")
    task_id = str(task_value).strip()
    state["h3"] = {
        "status": "running",
        "task_id": task_id,
        "receipt": _h3_receipt(request, task_id, workflow=workflow),
    }
    state["status"] = "h3_running"
    state["retryable"] = False
    _save_state(request, state)
    return task_id


def _poll_h3(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    task_id: str,
) -> H3Result:
    workflow = _state_workflow(request, state)
    if state["h3"].get("receipt") != _h3_receipt(
        request,
        task_id,
        legacy=_state_uses_legacy_generation_parameters(request, state),
        workflow=workflow,
    ):
        raise ReceiptError("receipt_mismatch")
    if workflow in H3_MULTIMODAL_WORKFLOWS:
        return _poll_gateway_h3(request, state, client, task_id)
    deadline = time.monotonic() + request.timeouts.h3_poll_s
    headers = {"Authorization": request.autodl_token}
    while True:
        try:
            _, payload = _query_json_with_retry(
                request,
                lambda: client.get(
                    f"{AUTODL_BASE_URL}/api/v1/comfyui/comfyui_workflow/result/{task_id}",
                    headers=headers,
                    timeout=request.timeouts.request_s,
                ),
                code="h3_query_failed",
                step="H3 result query",
                deadline=deadline,
            )
        except H3Error:
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True) from None
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        provider_status = str(data.get("status") or "").upper()
        if provider_status in {"SUCCESS", "COMPLETED"}:
            results = data.get("results")
            url = next(
                (
                    item.get("url")
                    for item in results
                    if isinstance(item, dict) and isinstance(item.get("url"), str)
                ),
                None,
            ) if isinstance(results, list) else None
            if not url:
                _fail(request, state, "h3_result_missing", retryable=False, keep_task=True)
                raise H3Error("h3_result_missing")
            output_receipt = _download(request, state, client, url)
            state["h3"]["status"] = "succeeded"
            state["h3"]["output"] = output_receipt
            state["status"] = "succeeded"
            state["retryable"] = False
            _save_state(request, state)
            return _result(state, output=request.workdir / "generated.mp4")
        if provider_status in {"FAILED", "ERROR", "FAIL"}:
            diagnostic = _provider_failure_diagnostic(
                payload, status=provider_status, secret=request.autodl_token
            )
            log.warning(
                "H3 provider failed cid=%s attempt=%s request_id=%s detail=%s",
                request.cid,
                state.get("attempt_id"),
                diagnostic.get("request_id") or "unavailable",
                diagnostic.get("detail") or "no_safe_detail",
            )
            state["h3"]["status"] = "failed"
            _fail(
                request,
                state,
                "h3_provider_failed",
                retryable=False,
                keep_task=True,
                provider_diagnostic=diagnostic,
            )
            return _result(state)
        if time.monotonic() >= deadline:
            _fail(request, state, "h3_timeout", retryable=True, keep_task=True)
            return _result(state)
        _pause(request.timeouts.poll_interval_s)


def _poll_gateway_h3(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    task_id: str,
) -> H3Result:
    deadline = time.monotonic() + request.timeouts.h3_poll_s
    while True:
        try:
            _, payload = _query_json_with_retry(
                request,
                lambda: client.get(
                    f"{H3_GATEWAY_BASE_URL}/v1/videos/{task_id}",
                    timeout=request.timeouts.request_s,
                ),
                code="h3_query_failed",
                step="H3 gateway result query",
                deadline=deadline,
            )
        except H3Error:
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True) from None
        provider_status = str(payload.get("status") or "").lower()
        if provider_status == "succeeded":
            output_receipt = _download(
                request,
                state,
                client,
                f"{H3_GATEWAY_BASE_URL}/v1/videos/{task_id}/content",
                trusted_loopback=True,
            )
            state["h3"]["status"] = "succeeded"
            state["h3"]["output"] = output_receipt
            state["status"] = "succeeded"
            state["retryable"] = False
            _save_state(request, state)
            return _result(state, output=request.workdir / "generated.mp4")
        if provider_status == "failed":
            diagnostic = _provider_failure_diagnostic(
                payload, status="FAILED", secret=request.autodl_token
            )
            state["h3"]["status"] = "failed"
            _fail(
                request,
                state,
                "h3_provider_failed",
                retryable=False,
                keep_task=True,
                provider_diagnostic=diagnostic,
            )
            return _result(state)
        if provider_status not in {"queued", "running"}:
            _fail(request, state, "h3_query_failed", retryable=True, keep_task=True)
            raise H3Error("h3_query_failed", retryable=True)
        if time.monotonic() >= deadline:
            _fail(request, state, "h3_timeout", retryable=True, keep_task=True)
            return _result(state)
        _pause(request.timeouts.poll_interval_s)


def _download(
    request: H3Request,
    state: dict[str, Any],
    client: httpx.Client,
    url: str,
    *,
    trusted_loopback: bool = False,
) -> dict[str, Any]:
    try:
        receipt = _run_automatic_retry(
            request.timeouts,
            lambda: _download_once(
                request, client, url, trusted_loopback=trusted_loopback
            ),
            step="H3 result download",
        )
    except H3Error as exc:
        _fail(request, state, exc.code, retryable=exc.retryable, keep_task=True)
        raise
    # Earlier failed attempts persist a recoverable state.  A later success in
    # the same process must remove that stale error before the caller commits
    # the final succeeded state.
    state.pop("error", None)
    state["status"] = "h3_running"
    state["retryable"] = False
    state["h3"]["status"] = "running"
    return receipt


def _download_once(
    request: H3Request,
    client: httpx.Client,
    url: str,
    *,
    trusted_loopback: bool = False,
) -> dict[str, Any]:
    if not trusted_loopback:
        try:
            public_url = _is_public_https_url(url)
        except _DNSLookupFailed:
            _raise_download_error("download_dns_failed", retryable=True)
        if not public_url:
            _raise_download_error("download_url_rejected", retryable=False)

    destination = request.workdir / "generated.mp4"
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    size = 0
    digest = hashlib.sha256()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with client.stream(
                "GET",
                url,
                timeout=request.timeouts.download_s,
                follow_redirects=False,
            ) as response:
                if not trusted_loopback:
                    public_peer = _response_has_public_peer(response)
                    if public_peer is None:
                        _raise_download_error(
                            "download_peer_unverified",
                            retryable=True,
                        )
                    if not public_peer:
                        _raise_download_error(
                            "download_url_rejected",
                            retryable=False,
                        )
                if 300 <= response.status_code < 400:
                    _raise_download_error(
                        "download_redirect_rejected",
                        retryable=False,
                    )
                if response.status_code != 200:
                    _raise_download_error(
                        "download_failed",
                        retryable=True,
                        automatic_retryable=_retryable_http_status(response.status_code),
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        _raise_download_error("download_failed", retryable=True)
                    if declared_size < 0:
                        _raise_download_error("download_failed", retryable=True)
                    if declared_size > MAX_VIDEO_BYTES:
                        _raise_download_error("download_too_large", retryable=False)
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        _raise_download_error("download_too_large", retryable=False)
                    _write_all(fd, chunk)
                    digest.update(chunk)
        except httpx.HTTPError:
            _raise_download_error("download_failed", retryable=True)
        if size <= 0:
            _raise_download_error("download_failed", retryable=True)
        os.fsync(fd)
        os.close(fd)
        fd = None

        def probe_attempt() -> dict[str, Any]:
            try:
                return _probe_media_timeline(temporary, request.timeouts.probe_s)
            except _ProbeUnavailable:
                raise _AutomaticRetryH3Error(
                    "output_probe_failed", retryable=True
                ) from None

        try:
            media_timeline = _run_automatic_retry(
                request.timeouts,
                probe_attempt,
                step="downloaded media probe",
            )
        except _AutomaticRetryH3Error:
            _raise_download_error(
                "output_probe_failed",
                retryable=True,
                automatic_retryable=False,
            )
        if request.audio_required and media_timeline.get("audio") is None:
            _raise_download_error("output_audio_missing", retryable=False)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except OSError:
        raise _AutomaticRetryH3Error("output_write_failed", retryable=True) from None
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "name": "generated.mp4",
        "sha256": digest.hexdigest(),
        "size": size,
        "media_timeline": media_timeline,
    }


def _raise_download_error(
    code: str,
    *,
    retryable: bool,
    automatic_retryable: bool | None = None,
) -> None:
    automatic = retryable if automatic_retryable is None else automatic_retryable
    error_type = _AutomaticRetryH3Error if automatic else H3Error
    raise error_type(code, retryable=retryable)


def _is_public_https_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        port = parsed.port or 443
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except UnicodeError:
            return False
        except OSError:
            raise _DNSLookupFailed from None
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except (IndexError, TypeError, ValueError):
                return False
    else:
        addresses = [literal]
    return bool(addresses) and all(
        address.is_global and not address.is_multicast for address in addresses
    )


def _response_has_public_peer(response: httpx.Response) -> bool | None:
    network_stream = response.extensions.get("network_stream")
    get_extra_info = getattr(network_stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return None
    try:
        peer = get_extra_info("server_addr")
    except Exception:
        return None
    host = peer[0] if isinstance(peer, (tuple, list)) and peer else peer
    if not isinstance(host, str):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    return address.is_global and not address.is_multicast


def _probe_media_timeline(
    path: Path,
    timeout_s: float,
    *,
    max_duration_s: float = H3_MAX_DURATION_S + 1,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    probe_prefix = [
        "ffprobe",
        "-v",
        "error",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file",
    ]
    summary_command = [
        *probe_prefix,
        "-show_entries",
        (
            "format=format_name,start_time,duration:"
            "stream=index,codec_type,codec_name,width,height,time_base,start_pts,"
            "start_time,duration,duration_ts,avg_frame_rate,r_frame_rate,"
            "sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    summary = _run_bounded_media_command(
        summary_command,
        deadline,
        max_stdout_bytes=64 * 1024,
    )
    summary_payload = _decode_probe_json(summary)
    has_audio = _validate_media_summary(
        summary_payload,
        max_duration_s=max_duration_s,
    )
    full_command = [
        *probe_prefix,
        "-show_entries",
        (
            "format=format_name,start_time,duration:"
            "stream=index,codec_type,codec_name,time_base,start_pts,start_time,"
            "duration,duration_ts,avg_frame_rate,r_frame_rate,sample_rate,channels:"
            "packet=stream_index,pts,pts_time,dts,dts_time,duration,duration_time:"
            "frame=stream_index,media_type,best_effort_timestamp,"
            "best_effort_timestamp_time,pts,pts_time,duration,duration_time,"
            "pkt_duration,pkt_duration_time,nb_samples"
        ),
        "-show_packets",
        "-show_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = _decode_probe_json(
        _run_bounded_media_command(
            full_command,
            deadline,
            max_stdout_bytes=MAX_MEDIA_PROBE_STDOUT_BYTES,
        )
    )
    if _validate_media_stream_inventory(payload) != has_audio:
        raise H3Error("download_invalid_video", retryable=False)
    decoded_audio_sha256 = _decode_media_and_hash_audio(
        path,
        deadline,
        has_audio=has_audio,
    )
    return _parse_media_timeline(
        payload,
        decoded_audio_sha256=decoded_audio_sha256,
    )


def _run_bounded_media_command(
    command: Sequence[str],
    deadline: float,
    *,
    max_stdout_bytes: int,
) -> bytes:
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        raise _ProbeUnavailable from None
    chunks: list[bytes] = []
    total = 0
    selector = selectors.DefaultSelector()
    try:
        if process.stdout is None:
            raise _ProbeUnavailable
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = _remaining_probe_timeout(deadline)
            if not selector.select(remaining):
                raise _ProbeUnavailable
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_stdout_bytes:
                raise H3Error("download_invalid_video", retryable=False)
            chunks.append(chunk)
        try:
            return_code = process.wait(timeout=_remaining_probe_timeout(deadline))
        except subprocess.TimeoutExpired:
            raise _ProbeUnavailable from None
    except BaseException as exc:
        if process.poll() is None:
            process.kill()
        process.wait()
        if isinstance(exc, OSError):
            raise _ProbeUnavailable from None
        raise
    finally:
        selector.close()
        if process.stdout is not None:
            process.stdout.close()
    if return_code != 0:
        raise H3Error("download_invalid_video", retryable=False)
    return b"".join(chunks)


def _remaining_probe_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _ProbeUnavailable
    return remaining


def _decode_probe_json(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        raise H3Error("download_invalid_video", retryable=False) from None
    if not isinstance(payload, dict):
        raise H3Error("download_invalid_video", retryable=False)
    return payload


def _validate_media_summary(payload: Any, *, max_duration_s: float) -> bool:
    has_audio = _validate_media_stream_inventory(payload)
    streams = payload["streams"]
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    width = video.get("width")
    height = video.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= 8192
        or isinstance(height, bool)
        or not isinstance(height, int)
        or not 1 <= height <= 8192
    ):
        raise H3Error("download_invalid_video", retryable=False)
    time_base = _positive_fraction(video.get("time_base"))
    duration = _stream_duration(video, time_base)
    avg_rate = _fraction_value(_positive_fraction(video.get("avg_frame_rate")))
    nominal_rate = _fraction_value(_positive_fraction(video.get("r_frame_rate")))
    if (
        duration is None
        or duration > max_duration_s + _DURATION_EPS_S
        or avg_rate > 240
        or nominal_rate > 240
    ):
        raise H3Error("download_invalid_video", retryable=False)
    return has_audio


def _validate_media_stream_inventory(payload: Any) -> bool:
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        raise H3Error("download_invalid_video", retryable=False)
    videos = [
        stream for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audios = [
        stream for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    supported = len(videos) + len(audios)
    if (
        len(videos) != 1
        or len(audios) > 1
        or supported != len(streams)
    ):
        raise H3Error("download_invalid_video", retryable=False)
    return bool(audios)


def _decode_media_and_hash_audio(
    path: Path,
    deadline: float,
    *,
    has_audio: bool,
) -> str | None:
    decode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-max_error_rate",
        "0",
        "-err_detect",
        "explode",
        "-nostdin",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(path),
        "-map",
        "0:V:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        "-abort_on",
        "empty_output_stream",
        "-f",
        "null",
        "-",
    ]
    _run_bounded_media_command(decode_command, deadline, max_stdout_bytes=1024)
    if not has_audio:
        return None
    hash_command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "-",
    ]
    try:
        output = _run_bounded_media_command(
            hash_command,
            deadline,
            max_stdout_bytes=1024,
        ).decode("ascii", "strict").strip()
    except UnicodeDecodeError:
        raise H3Error("download_invalid_video", retryable=False) from None
    prefix = "SHA256="
    digest = output[len(prefix):] if output.startswith(prefix) else ""
    if not _is_sha256(digest):
        raise H3Error("download_invalid_video", retryable=False)
    return digest


def _parse_media_timeline(
    payload: Any,
    *,
    decoded_audio_sha256: str | None,
) -> dict[str, Any]:
    has_audio = _validate_media_stream_inventory(payload)
    streams = payload["streams"]
    events = _media_events(payload)
    video_source = next(
        stream for stream in streams if stream.get("codec_type") == "video"
    )
    audio_source = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if has_audio != (audio_source is not None):
        raise H3Error("download_invalid_video", retryable=False)
    video = _parse_stream_timeline(video_source, events, kind="video")
    if audio_source is None:
        if decoded_audio_sha256 is not None:
            raise H3Error("download_invalid_video", retryable=False)
        audio = None
        av_delta = None
    else:
        if not _is_sha256(decoded_audio_sha256):
            raise H3Error("download_invalid_video", retryable=False)
        audio = _parse_stream_timeline(audio_source, events, kind="audio")
        audio["decoded_sha256"] = decoded_audio_sha256
        start_delta = _round_seconds(
            audio["first_frame_pts_s"] - video["first_frame_pts_s"]
        )
        end_delta = _round_seconds(
            audio["frame_end_s"] - video["frame_end_s"]
        )
        if (
            abs(start_delta) > MAX_AV_TIMELINE_DELTA_S
            or abs(end_delta) > MAX_AV_TIMELINE_DELTA_S
        ):
            raise H3Error("download_invalid_video", retryable=False)
        av_delta = {"start": start_delta, "end": end_delta}
    raw_format = payload.get("format")
    if not isinstance(raw_format, dict):
        raise H3Error("download_invalid_video", retryable=False)
    format_name = raw_format.get("format_name")
    if not isinstance(format_name, str) or not format_name.strip():
        raise H3Error("download_invalid_video", retryable=False)
    return {
        "schema": MEDIA_TIMELINE_SCHEMA,
        "version": MEDIA_TIMELINE_VERSION,
        "decode_complete": True,
        "container": {
            "format_name": format_name,
            "start_time_s": _optional_seconds(raw_format.get("start_time")),
            "duration_s": _optional_positive_seconds(raw_format.get("duration")),
        },
        "video": video,
        "audio": audio,
        "av_delta_s": av_delta,
    }


def _media_events(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    combined = payload.get("packets_and_frames")
    if isinstance(combined, list):
        events = combined
    else:
        packets = payload.get("packets")
        frames = payload.get("frames")
        if not isinstance(packets, list) or not isinstance(frames, list):
            raise H3Error("download_invalid_video", retryable=False)
        if len(packets) + len(frames) > MAX_MEDIA_TIMELINE_EVENTS:
            raise H3Error("download_invalid_video", retryable=False)
        events = [
            *({**event, "type": "packet"} for event in packets),
            *({**event, "type": "frame"} for event in frames),
        ]
    if len(events) > MAX_MEDIA_TIMELINE_EVENTS or not all(
        isinstance(event, dict) for event in events
    ):
        raise H3Error("download_invalid_video", retryable=False)
    return events


def _parse_stream_timeline(
    source: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    kind: Literal["video", "audio"],
) -> dict[str, Any]:
    index = source.get("index")
    codec_name = source.get("codec_name")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or not isinstance(codec_name, str)
        or not codec_name.strip()
    ):
        raise H3Error("download_invalid_video", retryable=False)
    time_base = _positive_fraction(source.get("time_base"))
    packets = [
        event for event in events
        if event.get("type") == "packet" and event.get("stream_index") == index
    ]
    frames = [
        event for event in events
        if event.get("type") == "frame" and event.get("stream_index") == index
    ]
    if not packets or not frames:
        raise H3Error("download_invalid_video", retryable=False)
    packet_dts = [
        _event_seconds(event, "dts", "dts_time", time_base)
        for event in packets
    ]
    if not _monotonic_nondecreasing(packet_dts):
        raise H3Error("download_invalid_video", retryable=False)
    packet_pts = [
        _event_seconds(event, "pts", "pts_time", time_base)
        for event in packets
    ]
    frame_pts = [
        _event_seconds(
            event,
            "best_effort_timestamp",
            "best_effort_timestamp_time",
            time_base,
            fallback_tick_key="pts",
            fallback_time_key="pts_time",
        )
        for event in frames
    ]
    if not _monotonic_nondecreasing(frame_pts):
        raise H3Error("download_invalid_video", retryable=False)
    frame_durations = [
        _event_positive_duration(event, time_base, kind=kind, source=source)
        for event in frames
    ]
    first_frame = frame_pts[0]
    last_frame = frame_pts[-1]
    frame_end = _round_seconds(last_frame + frame_durations[-1])
    packet_durations = _derive_packet_durations(
        packets,
        packet_pts=packet_pts,
        packet_dts=packet_dts,
        time_base=time_base,
        frame_end=frame_end,
    )
    start_time = _optional_seconds(source.get("start_time"))
    if start_time is None:
        start_time = first_frame
    duration = _stream_duration(source, time_base)
    if duration is None:
        duration = _round_seconds(frame_end - first_frame)
    if (
        duration <= 0
        or abs(start_time - first_frame) > MAX_AV_TIMELINE_DELTA_S
        or abs((start_time + duration) - frame_end) > MAX_AV_TIMELINE_DELTA_S
    ):
        raise H3Error("download_invalid_video", retryable=False)
    receipt: dict[str, Any] = {
        "index": index,
        "codec_name": codec_name,
        "time_base": time_base,
        "start_time_s": _round_seconds(start_time),
        "duration_s": _round_seconds(duration),
        "packet_count": len(packets),
        "first_packet_pts_s": _round_seconds(min(packet_pts)),
        "last_packet_pts_s": _round_seconds(max(packet_pts)),
        "packet_end_s": _round_seconds(
            max(pts + duration_s for pts, duration_s in zip(packet_pts, packet_durations))
        ),
        "packet_dts_monotonic": True,
        "frame_count": len(frames),
        "first_frame_pts_s": _round_seconds(first_frame),
        "last_frame_pts_s": _round_seconds(last_frame),
        "frame_end_s": frame_end,
        "presentation_monotonic": True,
    }
    if kind == "video":
        avg_frame_rate = _positive_fraction(source.get("avg_frame_rate"))
        r_frame_rate = _positive_fraction(source.get("r_frame_rate"))
        if (
            _fraction_value(avg_frame_rate) > 240
            or _fraction_value(r_frame_rate) > 240
        ):
            raise H3Error("download_invalid_video", retryable=False)
        receipt.update(
            avg_frame_rate=avg_frame_rate,
            r_frame_rate=r_frame_rate,
        )
    else:
        sample_rate = source.get("sample_rate")
        channels = source.get("channels")
        try:
            normalized_rate = int(sample_rate)
        except (TypeError, ValueError):
            raise H3Error("download_invalid_video", retryable=False) from None
        if (
            isinstance(sample_rate, bool)
            or not 8000 <= normalized_rate <= 384000
            or isinstance(channels, bool)
            or not isinstance(channels, int)
            or not 1 <= channels <= 32
        ):
            raise H3Error("download_invalid_video", retryable=False)
        receipt.update(sample_rate=normalized_rate, channels=channels)
    return receipt


def _stream_duration(
    source: Mapping[str, Any],
    time_base: str,
) -> float | None:
    direct = _optional_positive_seconds(source.get("duration"))
    if direct is not None:
        return direct
    ticks = _optional_positive_seconds(source.get("duration_ts"))
    if ticks is None:
        return None
    try:
        duration = ticks * _fraction_value(time_base)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return _round_seconds(duration) if math.isfinite(duration) else None


def _positive_fraction(value: Any) -> str:
    if not isinstance(value, str):
        raise H3Error("download_invalid_video", retryable=False)
    try:
        numerator, denominator = value.split("/", 1)
        normalized_numerator = int(numerator)
        normalized_denominator = int(denominator)
    except (TypeError, ValueError):
        raise H3Error("download_invalid_video", retryable=False) from None
    if normalized_numerator <= 0 or normalized_denominator <= 0:
        raise H3Error("download_invalid_video", retryable=False)
    return f"{normalized_numerator}/{normalized_denominator}"


def _fraction_value(value: str) -> float:
    try:
        numerator, denominator = value.split("/", 1)
        normalized = int(numerator) / int(denominator)
    except (AttributeError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        raise H3Error("download_invalid_video", retryable=False) from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise H3Error("download_invalid_video", retryable=False)
    return normalized


def _monotonic_nondecreasing(values: Sequence[float]) -> bool:
    return all(
        current + _DURATION_EPS_S >= previous
        for previous, current in zip(values, values[1:])
    )


def _event_seconds(
    event: Mapping[str, Any],
    tick_key: str,
    time_key: str,
    time_base: str,
    *,
    fallback_tick_key: str | None = None,
    fallback_time_key: str | None = None,
) -> float:
    for candidate_tick, candidate_time in (
        (tick_key, time_key),
        (fallback_tick_key, fallback_time_key),
    ):
        if candidate_tick is not None:
            ticks = event.get(candidate_tick)
            if not isinstance(ticks, bool) and isinstance(ticks, int):
                return _ticks_to_seconds(ticks, time_base)
        if candidate_time is not None:
            seconds = _optional_seconds(event.get(candidate_time))
            if seconds is not None:
                return seconds
    raise H3Error("download_invalid_video", retryable=False)


def _event_positive_duration(
    event: Mapping[str, Any],
    time_base: str,
    *,
    kind: Literal["video", "audio"] | None = None,
    source: Mapping[str, Any] | None = None,
) -> float:
    duration = _event_optional_duration(
        event,
        time_base,
        kind=kind,
        source=source,
    )
    if duration is None:
        raise H3Error("download_invalid_video", retryable=False)
    return duration


def _event_optional_duration(
    event: Mapping[str, Any],
    time_base: str,
    *,
    kind: Literal["video", "audio"] | None = None,
    source: Mapping[str, Any] | None = None,
) -> float | None:
    for tick_key, time_key in (
        ("duration", "duration_time"),
        ("pkt_duration", "pkt_duration_time"),
    ):
        ticks = event.get(tick_key)
        if not isinstance(ticks, bool) and isinstance(ticks, int) and ticks > 0:
            return _ticks_to_seconds(ticks, time_base)
        seconds = _optional_positive_seconds(event.get(time_key))
        if seconds is not None:
            return seconds
    if kind == "audio" and source is not None:
        samples = event.get("nb_samples")
        sample_rate = source.get("sample_rate")
        try:
            normalized_samples = int(samples)
            normalized_rate = int(sample_rate)
        except (TypeError, ValueError):
            pass
        else:
            if normalized_samples > 0 and normalized_rate > 0:
                return _round_seconds(normalized_samples / normalized_rate)
    if kind == "video" and source is not None:
        for rate_key in ("avg_frame_rate", "r_frame_rate"):
            try:
                rate = _fraction_value(_positive_fraction(source.get(rate_key)))
            except H3Error:
                continue
            if rate > 0:
                return _round_seconds(1 / rate)
    return None


def _derive_packet_durations(
    packets: Sequence[Mapping[str, Any]],
    *,
    packet_pts: Sequence[float],
    packet_dts: Sequence[float],
    time_base: str,
    frame_end: float,
) -> list[float]:
    durations = [
        _event_optional_duration(packet, time_base) for packet in packets
    ]
    sorted_pts = sorted(set(packet_pts))
    for index, duration in enumerate(durations):
        if duration is not None:
            continue
        candidates: list[float] = []
        if index + 1 < len(packet_dts):
            candidates.append(packet_dts[index + 1] - packet_dts[index])
        next_position = bisect_right(
            sorted_pts,
            packet_pts[index] + _DURATION_EPS_S,
        )
        if next_position < len(sorted_pts):
            candidates.append(sorted_pts[next_position] - packet_pts[index])
        candidates.append(frame_end - packet_pts[index])
        inferred = next(
            (
                _round_seconds(candidate)
                for candidate in candidates
                if math.isfinite(candidate) and candidate > _DURATION_EPS_S
            ),
            None,
        )
        if inferred is None:
            raise H3Error("download_invalid_video", retryable=False)
        durations[index] = inferred
    return [
        duration
        for duration in durations
        if duration is not None
    ]


def _ticks_to_seconds(ticks: int, time_base: str) -> float:
    try:
        value = ticks * _fraction_value(time_base)
    except (OverflowError, ValueError, ZeroDivisionError):
        raise H3Error("download_invalid_video", retryable=False) from None
    if not math.isfinite(value):
        raise H3Error("download_invalid_video", retryable=False)
    return _round_seconds(value)


def _optional_seconds(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return _round_seconds(normalized) if math.isfinite(normalized) else None


def _optional_positive_seconds(value: Any) -> float | None:
    normalized = _optional_seconds(value)
    return normalized if normalized is not None and normalized > 0 else None


def _round_seconds(value: float) -> float:
    return round(float(value), 9)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_authority_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and value == path.as_posix()
        and value not in {".", ".."}
        and ".." not in path.parts
    )


def _probe_video_duration(path: Path, timeout_s: float) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,duration,duration_ts,time_base",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise _ProbeUnavailable from None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams")
    except (
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return None
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        return None
    try:
        raw_duration = stream.get("duration")
        if isinstance(raw_duration, bool):
            return None
        duration = float(raw_duration)
        if math.isfinite(duration) and duration > 0:
            return duration
    except (TypeError, ValueError):
        pass
    try:
        raw_duration_ts = stream.get("duration_ts")
        if isinstance(raw_duration_ts, bool):
            return None
        ticks = float(raw_duration_ts)
        numerator, denominator = str(stream.get("time_base")).split("/", 1)
        duration = ticks * float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _probe_video(path: Path, timeout_s: float) -> bool:
    return _probe_video_duration(path, timeout_s) is not None


def _submission_unknown(request: H3Request, state: dict[str, Any], stage: str) -> None:
    state["status"] = "submission_unknown"
    state["retryable"] = False
    state[stage]["status"] = "submission_unknown"
    _save_state(request, state)


def _fail(
    request: H3Request,
    state: dict[str, Any],
    code: str,
    *,
    retryable: bool,
    keep_task: bool = False,
    provider_diagnostic: Mapping[str, str] | None = None,
    gateway_diagnostic: Mapping[str, Any] | None = None,
) -> None:
    state["status"] = "retryable_failure" if retryable else "failed"
    state["retryable"] = retryable
    state["error"] = {"code": code}
    if provider_diagnostic is not None:
        state["error"]["provider"] = dict(provider_diagnostic)
    if gateway_diagnostic is not None:
        state["error"]["gateway"] = dict(gateway_diagnostic)
    if not keep_task:
        if state["h3"].get("status") == "submitting":
            state["h3"]["status"] = "failed"
    _save_state(request, state)


def _result(
    state: Mapping[str, Any], *, output: Path | None = None
) -> H3Result:
    error = state.get("error")
    error_code = error.get("code") if isinstance(error, dict) else None
    if not isinstance(error_code, str):
        error_code = None
    return H3Result(
        status=str(state.get("status") or "failed"),
        attempt_id=str(state.get("attempt_id")),
        output=output,
        retryable=bool(state.get("retryable")),
        error_code=error_code,
        media_timeline=_state_media_timeline(state),
    )


def _state_media_timeline(
    state: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    h3_state = state.get("h3") if isinstance(state, Mapping) else None
    output = h3_state.get("output") if isinstance(h3_state, Mapping) else None
    timeline = output.get("media_timeline") if isinstance(output, Mapping) else None
    return timeline if _media_timeline_receipt_is_valid(timeline) else None


def _response_json(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("non-object response")
    return payload


def _pause(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _save_state(request: H3Request, state: Mapping[str, Any]) -> None:
    try:
        _atomic_write_json(_attempt_path(request, str(state["attempt_id"])), state)
    except OSError:
        raise H3Error("state_persist_failed") from None


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_bytes(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
