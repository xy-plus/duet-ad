"""Receipt-first, provider-neutral gateway for experimental post-video lip sync.

This module is deliberately not wired into the application.  It has no default HTTP
client: a coordinating layer must inject ``send`` explicitly, which keeps unit tests
offline and makes accidental provider traffic impossible.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, Mapping, Protocol
from urllib.parse import urlparse


RECEIPT_SCHEMA = "duet.lipsync.request"
RECEIPT_VERSION = 1
COMPARISON_SCHEMA = "duet.av-generation"
COMPARISON_VERSION = 1
COMPARISON_ROUTE = "post_h3_lipsync"
DEFAULT_RECEIPT_PATH = "work/lipsync/receipt.json"
TENCENT_MULTI_PERSON_PROVIDER = "tencent_video_no_train_multi"

_TENCENT_BASE_URL = "https://gw.tvs.qq.com"
_TENCENT_SUBMIT_PATH = "/v2/ivh/videomaker/broadcastservice/videomakenotrain"
_TENCENT_QUERY_PATH = "/v2/ivh/videomaker/broadcastservice/getprogress"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ALLOWED_STATUSES = {
    "prepared",
    "submitting",
    "submission_unknown",
    "accepted",
    "processing",
    "succeeded",
    "failed",
}
_RECEIPT_KEYS = {
    "schema",
    "version",
    "provider",
    "status",
    "input",
    "input_receipt",
    "comparison",
    "comparison_receipt",
    "provider_request_sha256",
    "asset_urls_sha256",
    "task_id",
    "provider_code",
    "error",
    "media_url",
    "duration_ms",
}


class LipSyncError(RuntimeError):
    """Public failure containing only a stable local code and safe detail."""

    def __init__(self, code: str, detail: str = "Lip-sync request failed"):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class AudioInterval:
    speaker_id: str
    audio_path: str
    start_pts: int
    end_pts: int


@dataclass(frozen=True)
class LipSyncInput:
    """Inputs frozen before any potentially paid provider submission.

    ``target_audio`` is the experiment's audio truth.  It may originate from source
    audio, rewritten speech, translation, inserted dialogue, or voice replacement.
    Per-speaker interval files are provider projections of that same upstream-frozen
    target and never redefine the final audio track.
    """

    video_path: str
    visual_receipt_path: str
    visual_attempt_id: str | None
    target_audio_path: str
    target_audio_receipt_path: str
    target_audio_decoded_sha256: str
    target_audio_sample_rate: int
    target_audio_channels: int
    speaker_to_face: Mapping[str, str]
    intervals: tuple[AudioInterval, ...]
    reference_frames: Mapping[str, str]
    pts_time_base_num: int
    pts_time_base_den: int
    timeline_start_pts: int
    timeline_end_pts: int
    provider: str
    provider_params: Mapping[str, object]
    idempotency_key: str
    workflow: str


@dataclass(frozen=True)
class TencentCredentials:
    app_key: str
    access_token: str


@dataclass(frozen=True)
class ProviderHttpRequest:
    """A request description consumed only by an injected transport."""

    operation: str
    method: str
    url: str
    query: Mapping[str, str]
    body: Mapping[str, object]
    timeout_s: float


@dataclass(frozen=True)
class ProviderHttpResponse:
    status_code: int
    body: Mapping[str, object]


@dataclass(frozen=True)
class LipSyncResult:
    status: str
    task_id: str | None = None
    media_url: str | None = None
    duration_ms: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Submission:
    task_id: str


@dataclass(frozen=True)
class _Progress:
    status: str
    media_url: str | None = None
    duration_ms: int | None = None
    provider_code: int | None = None


class _ProviderRejected(Exception):
    def __init__(self, provider_code: int | None = None):
        self.provider_code = provider_code


class _SubmissionAmbiguous(Exception):
    pass


class _QueryUnavailable(Exception):
    pass


class LipSyncProvider(Protocol):
    name: str

    def build_submission(
        self,
        frozen: Mapping[str, object],
        credentials: object,
        asset_urls: Mapping[str, str] | None,
    ) -> ProviderHttpRequest: ...

    def parse_submission(self, response: ProviderHttpResponse) -> _Submission: ...

    def build_query(self, task_id: str, credentials: object) -> ProviderHttpRequest: ...

    def parse_query(self, response: ProviderHttpResponse) -> _Progress: ...


RequestSender = Callable[[ProviderHttpRequest], Awaitable[ProviderHttpResponse]]


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _project_root(project_root: Path) -> Path:
    root = Path(project_root)
    if not root.is_absolute():
        raise LipSyncError("invalid_project_root")
    root = root.resolve()
    if not root.is_dir():
        raise LipSyncError("invalid_project_root")
    return root


def _project_path(
    root: Path,
    relative: str,
    *,
    require_file: bool,
) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise LipSyncError("invalid_project_path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise LipSyncError("invalid_project_path")
    resolved = (root / Path(*pure.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise LipSyncError("invalid_project_path")
    if require_file and not resolved.is_file():
        raise LipSyncError("input_file_missing")
    return relative, resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_file(root: Path, relative: str) -> dict[str, object]:
    canonical, path = _project_path(root, relative, require_file=True)
    return {
        "path": canonical,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _valid_integer(value: object, *, minimum: int | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and (minimum is None or value >= minimum)
    )


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and bool(_HASH_RE.fullmatch(value))


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER_RE.fullmatch(value))


def _normalize_provider_params(provider: str, raw: Mapping[str, object]) -> dict[str, object]:
    if provider != TENCENT_MULTI_PERSON_PROVIDER:
        raise LipSyncError("unsupported_lipsync_provider")
    if not isinstance(raw, Mapping):
        raise LipSyncError("provider_params_invalid")
    if not set(raw).issubset({"silent_mouth_mode", "face_match_mode", "resolution"}):
        raise LipSyncError("provider_params_invalid")
    params = {
        "silent_mouth_mode": raw.get("silent_mouth_mode", "ForceClosed"),
        "face_match_mode": raw.get("face_match_mode", "Strict"),
        "resolution": raw.get("resolution", 0),
    }
    if params["silent_mouth_mode"] not in {"ForceClosed", "KeepOriginal"}:
        raise LipSyncError("provider_params_invalid")
    if params["face_match_mode"] not in {"Strict", "Loose"}:
        raise LipSyncError("provider_params_invalid")
    if not _valid_integer(params["resolution"], minimum=0) or params["resolution"] not in {0, 1}:
        raise LipSyncError("provider_params_invalid")
    return params


def _validate_interval_contract(
    *,
    speaker_to_face: Mapping[str, str],
    intervals: list[dict[str, object]],
    reference_frames: Mapping[str, object],
) -> None:
    speakers = set(speaker_to_face)
    interval_speakers = {interval.get("speaker_id") for interval in intervals}
    faces = set(speaker_to_face.values())
    if not speakers or len(speakers) > 3:
        raise LipSyncError("too_many_speakers" if len(speakers) > 3 else "speaker_face_mapping_invalid")
    if interval_speakers != speakers or set(reference_frames) != faces:
        raise LipSyncError("speaker_face_mapping_invalid")
    if len(faces) != len(speakers):
        raise LipSyncError("speaker_face_mapping_invalid")
    if len(intervals) > 30:
        raise LipSyncError("intervals_invalid")
    counts = {speaker: 0 for speaker in speakers}
    previous_end: int | None = None
    for interval in intervals:
        if set(interval) != {"speaker_id", "audio", "start_pts", "end_pts"}:
            raise LipSyncError("intervals_invalid")
        speaker = interval["speaker_id"]
        start = interval["start_pts"]
        end = interval["end_pts"]
        if (
            speaker not in speakers
            or not _valid_integer(start, minimum=0)
            or not _valid_integer(end, minimum=0)
            or start >= end
            or (previous_end is not None and start < previous_end)
        ):
            raise LipSyncError("intervals_invalid")
        counts[speaker] += 1
        if counts[speaker] > 10:
            raise LipSyncError("intervals_invalid")
        previous_end = end


def _artifact_shape(value: object, *, decoded: bool = False) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {"path", "sha256", "size"}
    if decoded:
        expected.add("decoded_sha256")
    return (
        set(value) == expected
        and isinstance(value.get("path"), str)
        and _valid_hash(value.get("sha256"))
        and _valid_integer(value.get("size"), minimum=0)
        and (not decoded or _valid_hash(value.get("decoded_sha256")))
    )


def _validate_relative_string(relative: object) -> None:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise LipSyncError("receipt_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise LipSyncError("receipt_invalid")


def _validate_frozen(frozen: object) -> dict[str, object]:
    if not isinstance(frozen, dict) or set(frozen) != {
        "video",
        "visual_receipt",
        "visual_attempt_id",
        "target_audio",
        "target_audio_receipt",
        "target_audio_sample_rate",
        "target_audio_channels",
        "speaker_to_face",
        "intervals",
        "reference_frames",
        "pts_time_base",
        "timeline",
        "provider",
        "provider_params",
        "idempotency_key",
        "workflow",
    }:
        raise LipSyncError("receipt_invalid")
    artifacts: list[object] = [
        frozen["video"],
        frozen["visual_receipt"],
        frozen["target_audio_receipt"],
    ]
    if not all(_artifact_shape(item) for item in artifacts) or not _artifact_shape(
        frozen["target_audio"], decoded=True
    ):
        raise LipSyncError("receipt_invalid")
    for item in [*artifacts, frozen["target_audio"]]:
        _validate_relative_string(item["path"])
    attempt_id = frozen["visual_attempt_id"]
    if attempt_id is not None and not _valid_identifier(attempt_id):
        raise LipSyncError("receipt_invalid")
    if (
        not _valid_integer(frozen["target_audio_sample_rate"], minimum=1)
        or not _valid_integer(frozen["target_audio_channels"], minimum=1)
    ):
        raise LipSyncError("receipt_invalid")
    speaker_to_face = frozen["speaker_to_face"]
    references = frozen["reference_frames"]
    intervals = frozen["intervals"]
    if (
        not isinstance(speaker_to_face, dict)
        or not isinstance(references, dict)
        or not isinstance(intervals, list)
        or not all(_valid_identifier(key) and _valid_identifier(value) for key, value in speaker_to_face.items())
        or not all(_valid_identifier(key) and _artifact_shape(value) for key, value in references.items())
    ):
        raise LipSyncError("receipt_invalid")
    for value in references.values():
        _validate_relative_string(value["path"])
    for interval in intervals:
        if not isinstance(interval, dict) or not _artifact_shape(interval.get("audio")):
            raise LipSyncError("receipt_invalid")
        _validate_relative_string(interval["audio"]["path"])
    _validate_interval_contract(
        speaker_to_face=speaker_to_face,
        intervals=intervals,
        reference_frames=references,
    )
    time_base = frozen["pts_time_base"]
    timeline = frozen["timeline"]
    if (
        not isinstance(time_base, dict)
        or set(time_base) != {"num", "den"}
        or not _valid_integer(time_base["num"], minimum=1)
        or not _valid_integer(time_base["den"], minimum=1)
        or not isinstance(timeline, dict)
        or set(timeline) != {"start_pts", "end_pts"}
        or not _valid_integer(timeline["start_pts"], minimum=0)
        or not _valid_integer(timeline["end_pts"], minimum=1)
        or timeline["start_pts"] >= timeline["end_pts"]
        or intervals[0]["start_pts"] < timeline["start_pts"]
        or intervals[-1]["end_pts"] > timeline["end_pts"]
    ):
        raise LipSyncError("receipt_invalid")
    # Tencent accepts time values to millisecond precision.  Reject rather than round.
    for pts in {
        timeline["start_pts"],
        timeline["end_pts"],
        *(item[key] for item in intervals for key in ("start_pts", "end_pts")),
    }:
        if pts * time_base["num"] * 1000 % time_base["den"]:
            raise LipSyncError("receipt_invalid")
    if frozen["provider"] != TENCENT_MULTI_PERSON_PROVIDER:
        raise LipSyncError("receipt_invalid")
    try:
        normalized_params = _normalize_provider_params(frozen["provider"], frozen["provider_params"])
    except LipSyncError:
        raise LipSyncError("receipt_invalid") from None
    if frozen["provider_params"] != normalized_params:
        raise LipSyncError("receipt_invalid")
    if not _valid_identifier(frozen["idempotency_key"]):
        raise LipSyncError("receipt_invalid")
    workflow = frozen["workflow"]
    if (
        not isinstance(workflow, str)
        or not workflow
        or len(workflow) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in workflow)
    ):
        raise LipSyncError("receipt_invalid")
    return frozen


def _comparison(frozen: Mapping[str, object]) -> dict[str, object]:
    video = frozen["video"]
    target = frozen["target_audio"]
    visual_receipt = frozen["visual_receipt"]
    target_receipt = frozen["target_audio_receipt"]
    time_base = frozen["pts_time_base"]
    timeline = frozen["timeline"]
    speaker_face_receipt = canonical_json_sha256(
        {
            "speaker_to_face": frozen["speaker_to_face"],
            "intervals": [
                {
                    "speaker_id": item["speaker_id"],
                    "start_pts": item["start_pts"],
                    "end_pts": item["end_pts"],
                }
                for item in frozen["intervals"]
            ],
        }
    )
    dialogue_receipt = canonical_json_sha256(
        [
            {
                "speaker_id": item["speaker_id"],
                "audio_sha256": item["audio"]["sha256"],
                "start_pts": item["start_pts"],
                "end_pts": item["end_pts"],
            }
            for item in frozen["intervals"]
        ]
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "route": COMPARISON_ROUTE,
        "workflow": frozen["workflow"],
        "visual_input": {
            "receipt_sha256": visual_receipt["sha256"],
            "items": [
                {
                    "order": 0,
                    "name": PurePosixPath(video["path"]).name,
                    "sha256": video["sha256"],
                    "size": video["size"],
                },
                *(
                    {
                        "order": order,
                        "name": PurePosixPath(reference["path"]).name,
                        "sha256": reference["sha256"],
                        "size": reference["size"],
                    }
                    for order, (_, reference) in enumerate(
                        sorted(frozen["reference_frames"].items()), start=1
                    )
                ),
            ],
        },
        "target_audio_materials": {
            "receipt_sha256": target_receipt["sha256"],
            "items": [
                {
                    "order": 0,
                    "role": "target_dialogue",
                    "name": PurePosixPath(target["path"]).name,
                    "content_type": "audio/wav"
                    if str(target["path"]).lower().endswith(".wav")
                    else "application/octet-stream",
                    "size": target["size"],
                    "sha256": target["sha256"],
                    "decoded_sha256": target["decoded_sha256"],
                    "sample_rate": frozen["target_audio_sample_rate"],
                    "channels": frozen["target_audio_channels"],
                    "time_base": time_base,
                    "start_pts": timeline["start_pts"],
                    "end_pts": timeline["end_pts"],
                    "dialogue_sha256": dialogue_receipt,
                    "speaker_face_map_sha256": speaker_face_receipt,
                }
            ],
        },
        "upstream": {
            "receipt_sha256": visual_receipt["sha256"],
            "attempt_id": frozen["visual_attempt_id"],
        },
        # Provider SUCCESS does not prove downloaded output bytes or decoded PTS.
        # A downstream downloader/prober must create the comparable output receipt.
        "output": None,
    }


def _freeze_input(root: Path, request: LipSyncInput) -> dict[str, object]:
    if not _valid_hash(request.target_audio_decoded_sha256):
        raise LipSyncError("target_audio_evidence_invalid")
    if (
        not _valid_integer(request.target_audio_sample_rate, minimum=1)
        or not _valid_integer(request.target_audio_channels, minimum=1)
    ):
        raise LipSyncError("target_audio_evidence_invalid")
    if not isinstance(request.speaker_to_face, Mapping) or not isinstance(
        request.reference_frames, Mapping
    ):
        raise LipSyncError("speaker_face_mapping_invalid")
    speaker_to_face = dict(sorted(request.speaker_to_face.items()))
    if not all(_valid_identifier(key) and _valid_identifier(value) for key, value in speaker_to_face.items()):
        raise LipSyncError("speaker_face_mapping_invalid")
    if not isinstance(request.intervals, tuple) or not request.intervals:
        raise LipSyncError("intervals_invalid")
    raw_intervals = []
    for interval in request.intervals:
        if not isinstance(interval, AudioInterval):
            raise LipSyncError("intervals_invalid")
        raw_intervals.append(
            {
                "speaker_id": interval.speaker_id,
                "audio_path": interval.audio_path,
                "start_pts": interval.start_pts,
                "end_pts": interval.end_pts,
            }
        )
    try:
        raw_intervals.sort(
            key=lambda item: (
                item["start_pts"], item["end_pts"], item["speaker_id"], item["audio_path"]
            )
        )
    except TypeError:
        raise LipSyncError("intervals_invalid") from None
    intervals = [
        {
            "speaker_id": item["speaker_id"],
            "audio": _freeze_file(root, item["audio_path"]),
            "start_pts": item["start_pts"],
            "end_pts": item["end_pts"],
        }
        for item in raw_intervals
    ]
    references = {
        face: _freeze_file(root, path)
        for face, path in sorted(request.reference_frames.items())
        if isinstance(face, str) and isinstance(path, str)
    }
    if len(references) != len(request.reference_frames):
        raise LipSyncError("speaker_face_mapping_invalid")
    _validate_interval_contract(
        speaker_to_face=speaker_to_face,
        intervals=intervals,
        reference_frames=references,
    )
    if (
        not _valid_integer(request.pts_time_base_num, minimum=1)
        or not _valid_integer(request.pts_time_base_den, minimum=1)
        or not _valid_integer(request.timeline_start_pts, minimum=0)
        or not _valid_integer(request.timeline_end_pts, minimum=1)
        or request.timeline_start_pts >= request.timeline_end_pts
        or intervals[0]["start_pts"] < request.timeline_start_pts
        or intervals[-1]["end_pts"] > request.timeline_end_pts
    ):
        raise LipSyncError("intervals_invalid")
    for pts in {
        request.timeline_start_pts,
        request.timeline_end_pts,
        *(item[key] for item in intervals for key in ("start_pts", "end_pts")),
    }:
        if pts * request.pts_time_base_num * 1000 % request.pts_time_base_den:
            raise LipSyncError("intervals_invalid")
    if not _valid_identifier(request.idempotency_key):
        raise LipSyncError("idempotency_key_invalid")
    if (
        not isinstance(request.workflow, str)
        or not request.workflow
        or len(request.workflow) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in request.workflow)
    ):
        raise LipSyncError("workflow_invalid")
    if request.visual_attempt_id is not None and not _valid_identifier(request.visual_attempt_id):
        raise LipSyncError("visual_receipt_invalid")
    target_audio = _freeze_file(root, request.target_audio_path)
    target_audio["decoded_sha256"] = request.target_audio_decoded_sha256
    frozen = {
        "video": _freeze_file(root, request.video_path),
        "visual_receipt": _freeze_file(root, request.visual_receipt_path),
        "visual_attempt_id": request.visual_attempt_id,
        "target_audio": target_audio,
        "target_audio_receipt": _freeze_file(root, request.target_audio_receipt_path),
        "target_audio_sample_rate": request.target_audio_sample_rate,
        "target_audio_channels": request.target_audio_channels,
        "speaker_to_face": speaker_to_face,
        "intervals": intervals,
        "reference_frames": references,
        "pts_time_base": {"num": request.pts_time_base_num, "den": request.pts_time_base_den},
        "timeline": {
            "start_pts": request.timeline_start_pts,
            "end_pts": request.timeline_end_pts,
        },
        "provider": request.provider,
        "provider_params": _normalize_provider_params(request.provider, request.provider_params),
        "idempotency_key": request.idempotency_key,
        "workflow": request.workflow,
    }
    _validate_frozen(frozen)
    return frozen


def _receipt_path(root: Path, relative: str) -> Path:
    _, path = _project_path(root, relative, require_file=False)
    return path


@contextmanager
def _receipt_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LipSyncError("lipsync_request_busy") from None
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise LipSyncError("receipt_invalid") from None
    if not isinstance(payload, dict):
        raise LipSyncError("receipt_invalid")
    return payload


def _validate_receipt(receipt: object) -> dict[str, object]:
    if (
        not isinstance(receipt, dict)
        or not set(receipt).issubset(_RECEIPT_KEYS)
        or not {
            "schema",
            "version",
            "provider",
            "status",
            "input",
            "input_receipt",
            "comparison",
            "comparison_receipt",
        }.issubset(receipt)
        or receipt["schema"] != RECEIPT_SCHEMA
        or receipt["version"] != RECEIPT_VERSION
        or receipt["status"] not in _ALLOWED_STATUSES
        or not _valid_hash(receipt["input_receipt"])
        or not _valid_hash(receipt["comparison_receipt"])
    ):
        raise LipSyncError("receipt_invalid")
    frozen = _validate_frozen(receipt["input"])
    if (
        receipt["provider"] != frozen["provider"]
        or receipt["input_receipt"] != canonical_json_sha256(frozen)
        or receipt["comparison"] != _comparison(frozen)
        or receipt["comparison_receipt"] != canonical_json_sha256(receipt["comparison"])
    ):
        raise LipSyncError("receipt_invalid")
    for field in ("provider_request_sha256", "asset_urls_sha256"):
        if field in receipt and not _valid_hash(receipt[field]):
            raise LipSyncError("receipt_invalid")
    task_id = receipt.get("task_id")
    if task_id is not None and (
        not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id)
    ):
        raise LipSyncError("receipt_invalid")
    status = receipt["status"]
    if status in {"prepared", "submitting", "submission_unknown"} and task_id is not None:
        raise LipSyncError("receipt_invalid")
    if status in {"accepted", "processing", "succeeded"} and task_id is None:
        raise LipSyncError("receipt_invalid")
    if status == "succeeded":
        if not _valid_https_url(receipt.get("media_url")) or not _valid_integer(
            receipt.get("duration_ms"), minimum=0
        ):
            raise LipSyncError("receipt_invalid")
    if "provider_code" in receipt and not _valid_integer(receipt["provider_code"], minimum=0):
        raise LipSyncError("receipt_invalid")
    if "error" in receipt and receipt["error"] not in {
        "submission_unknown",
        "provider_rejected",
        "lipsync_provider_failed",
    }:
        raise LipSyncError("receipt_invalid")
    return receipt


def _load_receipt(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise LipSyncError("receipt_missing")
    return _validate_receipt(_read_json(path))


def _result(receipt: Mapping[str, object]) -> LipSyncResult:
    return LipSyncResult(
        status=receipt["status"],
        task_id=receipt.get("task_id"),
        media_url=receipt.get("media_url"),
        duration_ms=receipt.get("duration_ms"),
        error=receipt.get("error"),
    )


def freeze_request(
    project_root: Path,
    receipt_path: str,
    request: LipSyncInput,
) -> LipSyncResult:
    """Validate and atomically freeze experiment-B inputs before submission."""

    root = _project_root(project_root)
    path = _receipt_path(root, receipt_path)
    frozen = _freeze_input(root, request)
    input_paths = {
        (root / item["path"]).resolve()
        for item in _frozen_artifacts(frozen)
    }
    if path in input_paths:
        raise LipSyncError("receipt_path_conflict")
    comparison = _comparison(frozen)
    new_receipt = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "provider": frozen["provider"],
        "status": "prepared",
        "input": frozen,
        "input_receipt": canonical_json_sha256(frozen),
        "comparison": comparison,
        "comparison_receipt": canonical_json_sha256(comparison),
    }
    with _receipt_lock(path):
        if path.exists():
            existing = _load_receipt(path)
            if (
                existing["input_receipt"] != new_receipt["input_receipt"]
                or existing["comparison_receipt"] != new_receipt["comparison_receipt"]
            ):
                raise LipSyncError("receipt_conflict")
            return _result(existing)
        _atomic_json(path, new_receipt)
    return _result(new_receipt)


def _frozen_artifacts(frozen: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        frozen["video"],
        frozen["visual_receipt"],
        frozen["target_audio"],
        frozen["target_audio_receipt"],
        *(item["audio"] for item in frozen["intervals"]),
        *frozen["reference_frames"].values(),
    ]


def _verify_frozen_files(root: Path, frozen: Mapping[str, object]) -> None:
    for artifact in _frozen_artifacts(frozen):
        try:
            _, path = _project_path(root, artifact["path"], require_file=True)
        except LipSyncError:
            raise LipSyncError("frozen_input_changed") from None
        if path.stat().st_size != artifact["size"] or _sha256_file(path) != artifact["sha256"]:
            raise LipSyncError("frozen_input_changed")


def _valid_https_url(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 4096:
        return False
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )


def _safe_provider_code(body: object) -> int | None:
    if not isinstance(body, Mapping):
        return None
    header = body.get("Header")
    if not isinstance(header, Mapping):
        return None
    code = header.get("Code")
    return code if _valid_integer(code, minimum=0) else None


class TencentMultiPersonProvider:
    """Tencent aPaaS projection for the multi-speaker no-training API."""

    name = TENCENT_MULTI_PERSON_PROVIDER

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        base_url: str = _TENCENT_BASE_URL,
        timeout_s: float = 60.0,
    ):
        if not _valid_https_url(base_url):
            raise ValueError("base_url must be an HTTPS origin")
        parsed = urlparse(base_url)
        if parsed.path not in {"", "/"} or parsed.query:
            raise ValueError("base_url must be an HTTPS origin")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._clock = clock
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)

    def _credentials(self, credentials: object) -> TencentCredentials:
        if (
            not isinstance(credentials, TencentCredentials)
            or not isinstance(credentials.app_key, str)
            or not credentials.app_key.strip()
            or not isinstance(credentials.access_token, str)
            or not credentials.access_token.strip()
        ):
            raise LipSyncError("lipsync_not_configured")
        return credentials

    def _signed_query(self, credentials: object) -> dict[str, str]:
        current = self._credentials(credentials)
        timestamp = int(self._clock())
        signing_content = f"appkey={current.app_key}&timestamp={timestamp}"
        digest = hmac.new(
            current.access_token.encode("utf-8"),
            signing_content.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return {
            "appkey": current.app_key,
            "timestamp": str(timestamp),
            "signature": base64.b64encode(digest).decode("ascii"),
        }

    @staticmethod
    def _asset_urls(
        frozen: Mapping[str, object], asset_urls: Mapping[str, str] | None
    ) -> dict[str, str]:
        required = {
            frozen["video"]["path"],
            *(item["audio"]["path"] for item in frozen["intervals"]),
            *(item["path"] for item in frozen["reference_frames"].values()),
        }
        if not isinstance(asset_urls, Mapping) or set(asset_urls) != required:
            raise LipSyncError("provider_assets_invalid")
        normalized = dict(sorted(asset_urls.items()))
        if not all(_valid_https_url(value) for value in normalized.values()):
            raise LipSyncError("provider_assets_invalid")
        return normalized

    @staticmethod
    def _seconds(pts: int, time_base: Mapping[str, int]) -> int | float:
        milliseconds = pts * time_base["num"] * 1000 // time_base["den"]
        if milliseconds % 1000 == 0:
            return milliseconds // 1000
        return float(f"{milliseconds / 1000:.3f}")

    def build_submission(
        self,
        frozen: Mapping[str, object],
        credentials: object,
        asset_urls: Mapping[str, str] | None,
    ) -> ProviderHttpRequest:
        assets = self._asset_urls(frozen, asset_urls)
        params = frozen["provider_params"]
        time_base = frozen["pts_time_base"]
        speakers = []
        for speaker_id, face_id in sorted(frozen["speaker_to_face"].items()):
            audio_segments = [
                {
                    "AudioUrl": assets[item["audio"]["path"]],
                    "StartTime": self._seconds(item["start_pts"], time_base),
                    "EndTime": self._seconds(item["end_pts"], time_base),
                }
                for item in frozen["intervals"]
                if item["speaker_id"] == speaker_id
            ]
            speakers.append(
                {
                    "IdPhotoUrl": assets[frozen["reference_frames"][face_id]["path"]],
                    "AudioSegments": audio_segments,
                }
            )
        body = {
            "Header": {},
            "Payload": {
                "RefVideoUrl": assets[frozen["video"]["path"]],
                "DriverType": "OriginalVoice",
                "InputAudioUrl": "",
                "InputSsml": "",
                "SpeechParam": {},
                "VideoParam": {
                    "DisableIdDetect": 0,
                    "MakeType": "Default",
                    "StartTime": 0,
                    "EndTime": 0,
                    "Resolution": params["resolution"],
                    "FaceMatchMode": params["face_match_mode"],
                    "RefPhotoUrl": "",
                    "RefPhotoUrls": [],
                },
                "MultiSpeakerParam": {
                    "Speakers": speakers,
                    "NarrationSegments": [],
                    "SilentMouthMode": params["silent_mouth_mode"],
                },
            },
        }
        return ProviderHttpRequest(
            operation="submit",
            method="POST",
            url=self._base_url + _TENCENT_SUBMIT_PATH,
            query=self._signed_query(credentials),
            body=body,
            timeout_s=self._timeout_s,
        )

    def parse_submission(self, response: ProviderHttpResponse) -> _Submission:
        if not isinstance(response, ProviderHttpResponse):
            raise _SubmissionAmbiguous
        body = response.body
        task_id = None
        if isinstance(body, Mapping) and isinstance(body.get("Payload"), Mapping):
            candidate = body["Payload"].get("TaskId")
            if isinstance(candidate, str) and _TASK_ID_RE.fullmatch(candidate):
                task_id = candidate
        # A persisted task ID dominates response framing: recovery is query-only from here.
        if task_id is not None:
            return _Submission(task_id)
        provider_code = _safe_provider_code(body)
        if (
            _valid_integer(response.status_code, minimum=400)
            and 400 <= response.status_code < 500
        ) or (provider_code is not None and provider_code != 0):
            raise _ProviderRejected(provider_code)
        raise _SubmissionAmbiguous

    def build_query(self, task_id: str, credentials: object) -> ProviderHttpRequest:
        if not _TASK_ID_RE.fullmatch(task_id):
            raise LipSyncError("receipt_invalid")
        return ProviderHttpRequest(
            operation="query",
            # Tencent calls its progress endpoint with POST; this is a query-only operation,
            # never a repeat of the paid videomakenotrain submission.
            method="POST",
            url=self._base_url + _TENCENT_QUERY_PATH,
            query=self._signed_query(credentials),
            body={"Header": {}, "Payload": {"TaskId": task_id}},
            timeout_s=self._timeout_s,
        )

    def parse_query(self, response: ProviderHttpResponse) -> _Progress:
        if not isinstance(response, ProviderHttpResponse):
            raise _QueryUnavailable
        provider_code = _safe_provider_code(response.body)
        if (
            not _valid_integer(response.status_code, minimum=100)
            or response.status_code >= 400
            or provider_code != 0
            or not isinstance(response.body, Mapping)
            or not isinstance(response.body.get("Payload"), Mapping)
        ):
            raise _QueryUnavailable
        payload = response.body["Payload"]
        status = payload.get("Status")
        if status in {"COMMIT", "MAKING"}:
            return _Progress("processing")
        if status == "FAIL":
            fail_code = payload.get("FailCode")
            return _Progress(
                "failed",
                provider_code=fail_code if _valid_integer(fail_code, minimum=0) else None,
            )
        if status != "SUCCESS" or not _valid_https_url(payload.get("MediaUrl")):
            raise _QueryUnavailable
        duration = payload.get("Duration")
        if not _valid_integer(duration, minimum=0):
            raise _QueryUnavailable
        return _Progress("succeeded", payload["MediaUrl"], duration)


def load_status(project_root: Path, receipt_path: str) -> LipSyncResult:
    root = _project_root(project_root)
    return _result(_load_receipt(_receipt_path(root, receipt_path)))


def _transition(
    path: Path,
    receipt: Mapping[str, object],
    status: str,
    **changes: object,
) -> dict[str, object]:
    updated = dict(receipt)
    updated.update(changes)
    updated["status"] = status
    _validate_receipt(updated)
    _atomic_json(path, updated)
    return updated


async def advance(
    project_root: Path,
    receipt_path: str,
    *,
    provider: LipSyncProvider,
    credentials: object,
    asset_urls: Mapping[str, str] | None,
    send: RequestSender,
) -> LipSyncResult:
    """Advance exactly one safe state-machine edge.

    A prepared receipt may submit once.  ``submitting`` without a task ID becomes
    permanently ``submission_unknown``.  Any receipt with a task ID can only call the
    provider's query operation.  Query failures are safe to retry because they never
    create a new paid task.
    """

    root = _project_root(project_root)
    path = _receipt_path(root, receipt_path)
    with _receipt_lock(path):
        receipt = _load_receipt(path)
        if receipt["provider"] != provider.name:
            raise LipSyncError("provider_mismatch")
        status = receipt["status"]
        if status == "succeeded":
            return _result(receipt)
        if status == "failed":
            detail = (
                "Lip-sync provider rejected the request"
                if receipt.get("error") == "provider_rejected"
                else "Lip-sync provider failed the task"
            )
            raise LipSyncError(receipt.get("error", "lipsync_provider_failed"), detail)
        if status == "submission_unknown":
            raise LipSyncError("submission_unknown", "Lip-sync submission state is unknown")
        if status == "submitting":
            receipt = _transition(
                path,
                receipt,
                "submission_unknown",
                error="submission_unknown",
            )
            raise LipSyncError("submission_unknown", "Lip-sync submission state is unknown")

        task_id = receipt.get("task_id")
        if task_id is not None:
            query = provider.build_query(task_id, credentials)
            if query.operation != "query":
                raise LipSyncError("provider_protocol_error")
            try:
                response = await send(query)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise LipSyncError(
                    "provider_query_unavailable", "Lip-sync provider query is unavailable"
                ) from None
            try:
                progress = provider.parse_query(response)
            except _QueryUnavailable:
                raise LipSyncError(
                    "provider_query_unavailable", "Lip-sync provider query is unavailable"
                ) from None
            if progress.status == "processing":
                receipt = _transition(path, receipt, "processing")
                return _result(receipt)
            if progress.status == "failed":
                changes: dict[str, object] = {"error": "lipsync_provider_failed"}
                if progress.provider_code is not None:
                    changes["provider_code"] = progress.provider_code
                receipt = _transition(path, receipt, "failed", **changes)
                raise LipSyncError(
                    "lipsync_provider_failed", "Lip-sync provider failed the task"
                )
            receipt = _transition(
                path,
                receipt,
                "succeeded",
                media_url=progress.media_url,
                duration_ms=progress.duration_ms,
            )
            return _result(receipt)

        if status != "prepared":
            raise LipSyncError("receipt_invalid")
        _verify_frozen_files(root, receipt["input"])
        submission = provider.build_submission(receipt["input"], credentials, asset_urls)
        if submission.operation != "submit":
            raise LipSyncError("provider_protocol_error")
        receipt = _transition(
            path,
            receipt,
            "submitting",
            provider_request_sha256=canonical_json_sha256(submission.body),
            asset_urls_sha256=canonical_json_sha256(dict(sorted((asset_urls or {}).items()))),
        )
        try:
            response = await send(submission)
        except asyncio.CancelledError:
            _transition(
                path,
                receipt,
                "submission_unknown",
                error="submission_unknown",
            )
            raise
        except Exception:
            _transition(
                path,
                receipt,
                "submission_unknown",
                error="submission_unknown",
            )
            raise LipSyncError("submission_unknown", "Lip-sync submission state is unknown") from None
        try:
            submitted = provider.parse_submission(response)
        except _ProviderRejected as rejected:
            changes = {"error": "provider_rejected"}
            if rejected.provider_code is not None:
                changes["provider_code"] = rejected.provider_code
            _transition(path, receipt, "failed", **changes)
            raise LipSyncError(
                "provider_rejected", "Lip-sync provider rejected the request"
            ) from None
        except _SubmissionAmbiguous:
            _transition(
                path,
                receipt,
                "submission_unknown",
                error="submission_unknown",
            )
            raise LipSyncError("submission_unknown", "Lip-sync submission state is unknown") from None
        receipt = _transition(path, receipt, "accepted", task_id=submitted.task_id)
        return _result(receipt)
