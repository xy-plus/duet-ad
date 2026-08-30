"""Provider-neutral, receipt-first image-mask generation.

The gateway owns the paid POST boundary and local artifact publication.  A
provider adapter receives one immutable :class:`ProviderMaskRequest`; adapters
must return only the small normalized DTOs defined here.  Network clients,
credentials, SDK objects, and HTTP downloaders are injected by the caller, so
this module never performs implicit egress and is straightforward to fake.

Only person masks are admitted.  A scene mask is a different semantic product
and must be produced by the remote SAM2 scene-mask path; this gateway never
falls back to a full-frame or bounding-box mask.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import uuid
import zlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np

from app import error_trace


log = logging.getLogger(__name__)


ATTEMPT_SCHEMA = "duet.image-mask-attempt"
ATTEMPT_VERSION = 2
REQUEST_SCHEMA = "duet.image-mask-request"
REQUEST_VERSION = 2
PERSON_ROSTER_SCHEMA = "duet.person-roster"
PERSON_ROSTER_VERSION = 1
PRODUCER_SCHEMA = "duet.image-mask-producer"
PRODUCER_VERSION = 2
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SOURCE_BYTES = 40 * 1024 * 1024
MAX_MASK_BYTES = 64 * 1024 * 1024
MASK_PURPOSES = frozenset({"person", "protected_non_target_people"})
MaskPurpose = Literal["person", "protected_non_target_people"]
MaskScope = Literal["all_people_union", "person_instance"]
IdentityBinding = Literal["sole_visible_person", "provider_person_id"]
_PURPOSE_ORDER: tuple[MaskPurpose, ...] = ("person", "protected_non_target_people")

_SAFE_CODES = frozenset(
    {
        "invalid_mask_purpose",
        "invalid_frame_pts",
        "invalid_cache_version",
        "invalid_provider_descriptor",
        "invalid_provider_capability",
        "invalid_provider_params",
        "invalid_person_instance",
        "person_roster_receipt_invalid",
        "person_roster_receipt_mismatch",
        "person_instance_unavailable",
        "provider_identity_binding_invalid",
        "invalid_source_image",
        "source_image_too_large",
        "unsafe_project_path",
        "project_root_invalid",
        "mask_receipt_invalid",
        "mask_receipt_mismatch",
        "mask_artifact_mismatch",
        "submission_unknown",
        "provider_pending",
        "provider_query_failed",
        "provider_request_invalid",
        "provider_protocol_error",
        "provider_rejected",
        "provider_get_unsupported",
        "mask_download_failed",
        "mask_download_too_large",
        "mask_not_png",
        "mask_dimensions_mismatch",
        "mask_alpha_missing",
        "mask_alpha_empty",
        "mask_alpha_full_frame",
        "mask_output_write_failed",
    }
)
_DESCRIPTOR_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_PARAM_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MaskError(RuntimeError):
    """Stable public error.  Its message never includes provider data or URLs."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        safe_code = code if code in _SAFE_CODES else "provider_protocol_error"
        super().__init__(safe_code)
        self.code = safe_code
        self.retryable = retryable


class SubmissionUncertain(Exception):
    """The adapter could not confirm a POST response.

    A reliably observed task ID allows GET-only recovery.  A request ID is kept
    solely as private receipt evidence and is not assumed to be pollable.
    """

    def __init__(self, *, request_id: str | None = None, task_id: str | None = None):
        super().__init__("submission_unknown")
        self.request_id = request_id
        self.task_id = task_id


@dataclass(frozen=True)
class ProviderDescriptor:
    provider: str
    action: str
    model: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not _DESCRIPTOR_RE.fullmatch(value)
            for value in (self.provider, self.action, self.model)
        ):
            raise MaskError("invalid_provider_descriptor")


@dataclass(frozen=True)
class ProviderCapabilities:
    """Identity guarantees a provider must satisfy before any paid POST."""

    mask_scope: MaskScope
    identity_binding: IdentityBinding
    supported_purposes: tuple[MaskPurpose, ...]
    person_id_param: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.supported_purposes, tuple)
            or not self.supported_purposes
            or len(self.supported_purposes) != len(set(self.supported_purposes))
            or any(purpose not in MASK_PURPOSES for purpose in self.supported_purposes)
        ):
            raise MaskError("invalid_provider_capability")
        ordered = tuple(
            purpose for purpose in _PURPOSE_ORDER if purpose in self.supported_purposes
        )
        object.__setattr__(self, "supported_purposes", ordered)
        if self.mask_scope == "all_people_union":
            if (
                self.identity_binding != "sole_visible_person"
                or self.person_id_param is not None
                or ordered != ("person",)
            ):
                raise MaskError("invalid_provider_capability")
        elif self.mask_scope == "person_instance":
            if (
                self.identity_binding != "provider_person_id"
                or not isinstance(self.person_id_param, str)
                or not _PARAM_NAME_RE.fullmatch(self.person_id_param)
            ):
                raise MaskError("invalid_provider_capability")
        else:
            raise MaskError("invalid_provider_capability")


@dataclass(frozen=True)
class PersonInstanceRequest:
    """Frozen upstream identity roster binding for one requested person."""

    person_id: str
    visible_person_ids: tuple[str, ...]
    person_roster_receipt_path: str
    person_roster_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.person_id, str) or not _DESCRIPTOR_RE.fullmatch(
            self.person_id
        ):
            raise MaskError("invalid_person_instance")
        if (
            not isinstance(self.visible_person_ids, tuple)
            or not self.visible_person_ids
            or len(self.visible_person_ids) != len(set(self.visible_person_ids))
            or any(
                not isinstance(person_id, str)
                or not _DESCRIPTOR_RE.fullmatch(person_id)
                for person_id in self.visible_person_ids
            )
            or self.person_id not in self.visible_person_ids
        ):
            raise MaskError("invalid_person_instance")
        object.__setattr__(self, "visible_person_ids", tuple(sorted(self.visible_person_ids)))
        object.__setattr__(
            self,
            "person_roster_receipt_path",
            _relative_path(self.person_roster_receipt_path),
        )
        if (
            not isinstance(self.person_roster_receipt_sha256, str)
            or not _SHA256_RE.fullmatch(self.person_roster_receipt_sha256)
        ):
            raise MaskError("invalid_person_instance")


@dataclass(frozen=True)
class ProviderMaskRequest:
    """Frozen, provider-neutral request presented to an adapter."""

    provider: str
    action: str
    model: str
    purpose: MaskPurpose
    provider_capability: ProviderCapabilities
    person_instance: PersonInstanceRequest
    source_sha256: str
    width: int
    height: int
    frame_pts: str
    request_sha256: str
    params: Mapping[str, Any]
    cache_version: str

    def __post_init__(self) -> None:
        try:
            frozen_params = _freeze_params(self.params)
            normalized_pts = _frame_pts(self.frame_pts)
            expected = request_sha256(
                provider=self.provider,
                action=self.action,
                model=self.model,
                purpose=self.purpose,
                provider_capability=self.provider_capability,
                person_instance=self.person_instance,
                source_sha256=self.source_sha256,
                width=self.width,
                height=self.height,
                frame_pts=normalized_pts,
                params=frozen_params,
                cache_version=self.cache_version,
            )
        except MaskError:
            raise MaskError("provider_request_invalid") from None
        if self.request_sha256 != expected:
            raise MaskError("provider_request_invalid")
        object.__setattr__(self, "frame_pts", normalized_pts)
        object.__setattr__(self, "params", frozen_params)


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized POST/GET response; URLs remain private receipt state."""

    request_id: str | None = None
    task_id: str | None = None
    result_url: str | None = None


@dataclass(frozen=True)
class MaskResult:
    path: Path
    receipt_path: Path
    producer_receipt: dict[str, Any]


@dataclass(frozen=True)
class MaskSourceExpectation:
    path: str
    sha256: str
    width: int
    height: int
    frame_pts: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "frame_pts", _frame_pts(self.frame_pts))
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise MaskError("invalid_source_image")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise MaskError("invalid_source_image")


@dataclass(frozen=True)
class MaskRosterExpectation:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise MaskError("invalid_person_instance")


@dataclass(frozen=True)
class LoadedMaskArtifact:
    """Fully revalidated consumer artifact with an immutable packed bool mask."""

    canonical_receipt: bytes
    canonical_receipt_sha256: str
    project_relative_path: str
    mask_sha256: str
    width: int
    height: int
    foreground_pixels: int
    packed_mask: bytes
    packed_encoding: str = "row-major-alpha-gt-zero-packbits-little-v1"


class MaskProvider(Protocol):
    """Minimal adapter boundary used by :func:`generate_mask`."""

    descriptor: ProviderDescriptor
    capabilities: ProviderCapabilities

    def submit(self, request: ProviderMaskRequest) -> ProviderResponse: ...

    def get(self, task_id: str) -> ProviderResponse: ...

    def download(self, url: str) -> bytes: ...


class AliyunVIAPISegmentHDBody:
    """SegmentHDBody adapter over injected SDK/request and download callables.

    ``request`` receives ``(action, params)`` and returns the decoded API body.
    ``download`` receives the temporary result URL and returns its bytes.  The
    official synchronous API returns ``RequestId`` and ``Data.ImageURL``; it has
    no task polling method, so :meth:`get` fails closed unless a GET callable is
    explicitly supplied for a compatible proxy.
    """

    descriptor = ProviderDescriptor(
        provider="aliyun_viapi",
        action="SegmentHDBody",
        model="imageseg-20191230",
    )
    capabilities = ProviderCapabilities(
        mask_scope="all_people_union",
        identity_binding="sole_visible_person",
        supported_purposes=("person",),
    )

    def __init__(
        self,
        *,
        request: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
        download: Callable[[str], bytes],
        get_request: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        if not callable(request) or not callable(download):
            raise MaskError("provider_request_invalid")
        self._request = request
        self._download = download
        self._get_request = get_request

    def submit(self, request: ProviderMaskRequest) -> ProviderResponse:
        if (
            request.provider != self.descriptor.provider
            or request.action != self.descriptor.action
            or request.model != self.descriptor.model
            or request.provider_capability != self.capabilities
            or not _SHA256_RE.fullmatch(request.request_sha256)
        ):
            raise MaskError("provider_request_invalid")
        params = _freeze_params(request.params)
        _person_instance_receipt(
            self.capabilities, request.purpose, request.person_instance, params
        )
        if set(params) != {"ImageURL"}:
            raise MaskError("provider_request_invalid")
        _private_https_url(params["ImageURL"])
        payload = self._request(self.descriptor.action, params)
        return self._decode(payload)

    def get(self, task_id: str) -> ProviderResponse:
        task = _private_identifier(task_id)
        if self._get_request is None:
            raise MaskError("provider_get_unsupported")
        return self._decode(self._get_request(task))

    def download(self, url: str) -> bytes:
        return self._download(_private_https_url(url))

    @staticmethod
    def _decode(payload: Mapping[str, Any]) -> ProviderResponse:
        if not isinstance(payload, Mapping):
            raise MaskError("provider_protocol_error")
        request_id = _optional_private_identifier(payload.get("RequestId"))
        data = payload.get("Data")
        if not isinstance(data, Mapping):
            raise MaskError("provider_protocol_error")
        result_url = _aliyun_result_url(data.get("ImageURL"))
        return ProviderResponse(request_id=request_id, result_url=result_url)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise MaskError("invalid_provider_params") from None


def _freeze_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise MaskError("invalid_provider_params")
    try:
        frozen = json.loads(_canonical_json(dict(params)).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):  # defensive; dumps output is UTF-8 JSON
        raise MaskError("invalid_provider_params") from None
    if not isinstance(frozen, dict) or any(not isinstance(key, str) for key in frozen):
        raise MaskError("invalid_provider_params")
    return frozen


def _frame_pts(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise MaskError("invalid_frame_pts")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation:
        raise MaskError("invalid_frame_pts") from None
    if not decimal.is_finite() or decimal < 0:
        raise MaskError("invalid_frame_pts")
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _capability_receipt(capability: ProviderCapabilities) -> dict[str, Any]:
    return {
        "mask_scope": capability.mask_scope,
        "identity_binding": capability.identity_binding,
        "person_id_param": capability.person_id_param,
        "supported_purposes": list(capability.supported_purposes),
    }


def _person_instance_receipt(
    capability: ProviderCapabilities,
    purpose: str,
    instance: PersonInstanceRequest,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(instance, PersonInstanceRequest):
        raise MaskError("invalid_person_instance")
    if purpose not in capability.supported_purposes:
        raise MaskError("person_instance_unavailable")
    provider_person_id: str | None = None
    if capability.mask_scope == "all_people_union":
        if (
            purpose != "person"
            or instance.visible_person_ids != (instance.person_id,)
        ):
            raise MaskError("person_instance_unavailable")
    else:
        parameter = capability.person_id_param
        provider_person_id = params.get(parameter) if parameter is not None else None
        if provider_person_id != instance.person_id:
            raise MaskError("provider_identity_binding_invalid")
    return {
        "person_id": instance.person_id,
        "visible_person_ids": list(instance.visible_person_ids),
        "person_roster_receipt_path": instance.person_roster_receipt_path,
        "person_roster_receipt_sha256": instance.person_roster_receipt_sha256,
        "provider_person_id": provider_person_id,
    }


def request_sha256(
    *,
    provider: str,
    action: str,
    model: str,
    purpose: str,
    provider_capability: ProviderCapabilities,
    person_instance: PersonInstanceRequest,
    source_sha256: str,
    width: int,
    height: int,
    frame_pts: Any,
    params: Mapping[str, Any],
    cache_version: str,
) -> str:
    """Hash every provider/cache semantic used to decide mask reuse."""
    ProviderDescriptor(provider=provider, action=action, model=model)
    if purpose not in MASK_PURPOSES:
        raise MaskError("invalid_mask_purpose")
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(source_sha256):
        raise MaskError("invalid_source_image")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise MaskError("invalid_source_image")
    if not isinstance(cache_version, str) or not cache_version or len(cache_version) > 128:
        raise MaskError("invalid_cache_version")
    if not isinstance(provider_capability, ProviderCapabilities):
        raise MaskError("invalid_provider_capability")
    frozen_params = _freeze_params(params)
    frozen = {
        "schema": REQUEST_SCHEMA,
        "version": REQUEST_VERSION,
        "provider": provider,
        "action": action,
        "model": model,
        "purpose": purpose,
        "provider_capability": _capability_receipt(provider_capability),
        "person_instance": _person_instance_receipt(
            provider_capability, purpose, person_instance, frozen_params
        ),
        "source_sha256": source_sha256,
        "width": width,
        "height": height,
        "frame_pts": _frame_pts(frame_pts),
        "params": frozen_params,
        "cache_version": cache_version,
    }
    return sha256_bytes(_canonical_json(frozen))


def _private_identifier(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 0x20 for character in value)
    ):
        raise MaskError("provider_protocol_error")
    return value


def _optional_private_identifier(value: Any) -> str | None:
    return None if value is None else _private_identifier(value)


def _private_https_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 8192:
        raise MaskError("provider_protocol_error")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        raise MaskError("provider_protocol_error") from None
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise MaskError("provider_protocol_error")
    return value


def _aliyun_result_url(value: Any) -> str:
    url = _private_https_url(value)
    hostname = urlsplit(url).hostname
    if hostname is None or (
        hostname != "aliyuncs.com" and not hostname.endswith(".aliyuncs.com")
    ):
        raise MaskError("provider_protocol_error")
    return url


def _relative_path(value: str | Path) -> str:
    if not isinstance(value, (str, Path)):
        raise MaskError("unsafe_project_path")
    text = str(value)
    raw_parts = text.split("/")
    if (
        not text
        or "\\" in text
        or text.startswith("/")
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise MaskError("unsafe_project_path")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.name:
        raise MaskError("unsafe_project_path")
    return path.as_posix()


def _project_root(value: Path) -> Path:
    root = Path(value)
    try:
        info = root.lstat()
    except OSError:
        raise MaskError("project_root_invalid") from None
    if not root.is_absolute() or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise MaskError("project_root_invalid")
    return root


def _open_parent(root: Path, relative: str, *, create: bool) -> tuple[int, str]:
    parts = PurePosixPath(relative).parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        current = os.open(root, flags)
    except OSError:
        raise MaskError("unsafe_project_path") from None
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=current)
                os.fsync(current)
            os.close(current)
            current = child
        return current, parts[-1]
    except Exception:
        os.close(current)
        raise


def _leaf_info(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _check_destination(root: Path, relative: str) -> None:
    try:
        parent_fd, name = _open_parent(root, relative, create=True)
        try:
            info = _leaf_info(parent_fd, name)
            if info is not None and not stat.S_ISREG(info.st_mode):
                raise MaskError("unsafe_project_path")
        finally:
            os.close(parent_fd)
    except MaskError:
        raise
    except OSError:
        raise MaskError("unsafe_project_path") from None


def _read_project_file(root: Path, relative: str, *, limit: int) -> bytes:
    try:
        parent_fd, name = _open_parent(root, relative, create=False)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(name, flags, dir_fd=parent_fd)
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise MaskError("unsafe_project_path")
                if info.st_size > limit:
                    raise MaskError(
                        "source_image_too_large"
                        if limit == MAX_SOURCE_BYTES
                        else "mask_download_too_large"
                    )
                chunks: list[bytes] = []
                remaining = limit + 1
                while remaining:
                    chunk = os.read(file_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if len(payload) > limit:
                    raise MaskError(
                        "source_image_too_large"
                        if limit == MAX_SOURCE_BYTES
                        else "mask_download_too_large"
                    )
                return payload
            finally:
                os.close(file_fd)
        finally:
            os.close(parent_fd)
    except MaskError:
        raise
    except OSError:
        raise MaskError("unsafe_project_path") from None


def _read_optional(root: Path, relative: str, *, limit: int) -> bytes | None:
    try:
        return _read_project_file(root, relative, limit=limit)
    except MaskError as exc:
        if exc.code != "unsafe_project_path":
            raise
        # Distinguish a missing leaf from an unsafe path without following links.
        try:
            parent_fd, name = _open_parent(root, relative, create=False)
            try:
                if _leaf_info(parent_fd, name) is None:
                    return None
            finally:
                os.close(parent_fd)
        except (OSError, FileNotFoundError):
            return None
        raise


def _atomic_project_bytes(root: Path, relative: str, payload: bytes) -> None:
    parent_fd = -1
    temporary = f".{PurePosixPath(relative).name}.{uuid.uuid4().hex}.tmp"
    try:
        parent_fd, name = _open_parent(root, relative, create=True)
        existing = _leaf_info(parent_fd, name)
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise MaskError("unsafe_project_path")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                written += os.write(file_fd, view[written:])
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    except MaskError:
        raise
    except OSError:
        raise MaskError("mask_output_write_failed") from None
    finally:
        if parent_fd >= 0:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            os.close(parent_fd)


def _atomic_json(root: Path, relative: str, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError):
        raise MaskError("mask_receipt_invalid") from None
    _atomic_project_bytes(root, relative, encoded)


def _transition(
    root: Path,
    receipt_path: str,
    receipt: dict[str, Any],
    status: str,
    **updates: Any,
) -> dict[str, Any]:
    history = list(receipt.get("history") or [])
    if not history or history[-1] != status:
        history.append(status)
    current = {**receipt, **updates, "status": status, "history": history}
    _atomic_json(root, receipt_path, current)
    return current


def _decode_source(payload: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in {2, 3}:
        raise MaskError("invalid_source_image")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise MaskError("invalid_source_image")
    return width, height


def _validated_mask(
    payload: bytes, *, width: int, height: int
) -> tuple[dict[str, Any], np.ndarray]:
    png_width, png_height = _png_dimensions(payload)
    if png_width != width or png_height != height:
        raise MaskError("mask_dimensions_mismatch")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MaskError("mask_not_png")
    if image.shape[1] != width or image.shape[0] != height:
        raise MaskError("mask_dimensions_mismatch")
    if image.ndim != 3 or image.shape[2] != 4:
        raise MaskError("mask_alpha_missing")
    alpha = image[:, :, 3]
    nonzero = int(np.count_nonzero(alpha))
    pixels = width * height
    if nonzero == 0:
        raise MaskError("mask_alpha_empty")
    if nonzero == pixels:
        raise MaskError("mask_alpha_full_frame")
    return (
        {
            "sha256": sha256_bytes(payload),
            "size": len(payload),
            "width": width,
            "height": height,
            "mime_type": "image/png",
            "alpha_nonzero_pixels": nonzero,
            "alpha_transparent_pixels": pixels - nonzero,
        },
        np.ascontiguousarray(alpha > 0),
    )


def _validate_mask(payload: bytes, *, width: int, height: int) -> dict[str, Any]:
    metadata, _binary = _validated_mask(payload, width=width, height=height)
    return metadata


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    """Validate the complete PNG chunk stream before asking OpenCV to decode."""
    if not payload.startswith(PNG_SIGNATURE):
        raise MaskError("mask_not_png")
    offset = len(PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    first = True
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise MaskError("mask_not_png")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if data_end < data_start or chunk_end > len(payload):
            raise MaskError("mask_not_png")
        expected_crc = int.from_bytes(payload[data_end:chunk_end], "big")
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload[data_start:data_end], actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise MaskError("mask_not_png")
        if first:
            if chunk_type != b"IHDR" or length != 13:
                raise MaskError("mask_not_png")
            png_width = int.from_bytes(payload[data_start : data_start + 4], "big")
            png_height = int.from_bytes(payload[data_start + 4 : data_start + 8], "big")
            if png_width <= 0 or png_height <= 0:
                raise MaskError("mask_not_png")
            dimensions = (png_width, png_height)
            first = False
        elif chunk_type == b"IHDR":
            raise MaskError("mask_not_png")
        if chunk_type == b"IEND":
            if length != 0 or chunk_end != len(payload) or dimensions is None:
                raise MaskError("mask_not_png")
            return dimensions
        offset = chunk_end
    raise MaskError("mask_not_png")


def _provider_state(response: ProviderResponse) -> dict[str, str]:
    if not isinstance(response, ProviderResponse):
        raise MaskError("provider_protocol_error")
    state: dict[str, str] = {}
    if response.request_id is not None:
        state["request_id"] = _private_identifier(response.request_id)
    if response.task_id is not None:
        state["task_id"] = _private_identifier(response.task_id)
    if response.result_url is not None:
        state["result_url"] = _private_https_url(response.result_url)
    if not state.get("result_url") and not state.get("task_id"):
        raise MaskError("provider_protocol_error")
    return state


def _producer_receipt(
    request: Mapping[str, Any], output_path: str, mask: Mapping[str, Any]
) -> dict[str, Any]:
    source = dict(request["source"])
    source["frame_pts"] = request["frame_pts"]
    return {
        "schema": PRODUCER_SCHEMA,
        "version": PRODUCER_VERSION,
        "producer": {
            "provider": request["provider"],
            "action": request["action"],
            "model": request["model"],
        },
        "purpose": request["purpose"],
        "provider_capability": request["provider_capability"],
        "person_instance": request["person_instance"],
        "source": source,
        "request_sha256": request["request_sha256"],
        "params": request["params"],
        "cache_version": request["cache_version"],
        "mask": {"path": output_path, **dict(mask)},
    }


def _load_receipt(root: Path, relative: str) -> dict[str, Any] | None:
    raw = _read_optional(root, relative, limit=4 * 1024 * 1024)
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise MaskError("mask_receipt_invalid") from None
    if not isinstance(payload, dict):
        raise MaskError("mask_receipt_invalid")
    return payload


def _validate_person_roster_receipt(
    root: Path,
    instance: PersonInstanceRequest,
    *,
    source_path: str,
    source_sha256: str,
    width: int,
    height: int,
    frame_pts: str,
) -> None:
    raw = _read_project_file(
        root, instance.person_roster_receipt_path, limit=4 * 1024 * 1024
    )
    if sha256_bytes(raw) != instance.person_roster_receipt_sha256:
        raise MaskError("person_roster_receipt_mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise MaskError("person_roster_receipt_invalid") from None
    expected = {
        "schema": PERSON_ROSTER_SCHEMA,
        "version": PERSON_ROSTER_VERSION,
        "source": {
            "path": source_path,
            "sha256": source_sha256,
            "width": width,
            "height": height,
            "frame_pts": frame_pts,
        },
        "person_ids": list(instance.visible_person_ids),
    }
    if payload != expected:
        raise MaskError("person_roster_receipt_mismatch")


def _producer_artifact(
    root: Path, artifact: Mapping[str, Any] | str | Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        if isinstance(artifact, Mapping):
            payload = json.loads(_canonical_json(dict(artifact)).decode("utf-8"))
        else:
            relative = _relative_path(artifact)
            raw = _read_project_file(root, relative, limit=4 * 1024 * 1024)
            payload = json.loads(raw.decode("utf-8"))
    except (MaskError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise MaskError("mask_artifact_mismatch") from None
    if not isinstance(payload, dict):
        raise MaskError("mask_artifact_mismatch")
    if payload.get("schema") == ATTEMPT_SCHEMA:
        producer = payload.get("producer_receipt")
        if (
            payload.get("version") != ATTEMPT_VERSION
            or payload.get("status") != "succeeded"
            or not isinstance(payload.get("history"), list)
            or not payload["history"]
            or payload["history"][-1] != "succeeded"
            or not isinstance(producer, dict)
        ):
            raise MaskError("mask_artifact_mismatch")
        return producer, payload
    return payload, None


def load_validated_mask(
    project_root: Path,
    artifact: Mapping[str, Any] | str | Path,
    *,
    expected_source: MaskSourceExpectation,
    expected_person_id: str,
    expected_visible_person_ids: tuple[str, ...],
    expected_roster: MaskRosterExpectation,
    expected_purpose: MaskPurpose,
) -> LoadedMaskArtifact:
    """Load one v2 producer/attempt receipt and fully revalidate its mask.

    ``artifact`` may be the producer mapping or a project-relative path to a
    producer/attempt receipt.  The returned bool mask is row-major, thresholded
    as ``alpha > 0``, and packed with ``numpy.packbits(bitorder="little")``.
    """
    root = _project_root(Path(project_root))
    if not isinstance(expected_source, MaskSourceExpectation) or not isinstance(
        expected_roster, MaskRosterExpectation
    ):
        raise MaskError("mask_artifact_mismatch")
    producer, attempt = _producer_artifact(root, artifact)
    try:
        if producer.get("schema") != PRODUCER_SCHEMA or producer.get(
            "version"
        ) != PRODUCER_VERSION:
            raise MaskError("mask_artifact_mismatch")
        descriptor_payload = producer["producer"]
        if not isinstance(descriptor_payload, dict) or set(descriptor_payload) != {
            "provider",
            "action",
            "model",
        }:
            raise MaskError("mask_artifact_mismatch")
        descriptor = ProviderDescriptor(**descriptor_payload)
        capability_payload = producer["provider_capability"]
        if not isinstance(capability_payload, dict) or set(capability_payload) != {
            "mask_scope",
            "identity_binding",
            "person_id_param",
            "supported_purposes",
        }:
            raise MaskError("mask_artifact_mismatch")
        purposes = capability_payload["supported_purposes"]
        if not isinstance(purposes, list):
            raise MaskError("mask_artifact_mismatch")
        capability = ProviderCapabilities(
            mask_scope=capability_payload["mask_scope"],
            identity_binding=capability_payload["identity_binding"],
            person_id_param=capability_payload["person_id_param"],
            supported_purposes=tuple(purposes),
        )
        instance = PersonInstanceRequest(
            person_id=expected_person_id,
            visible_person_ids=expected_visible_person_ids,
            person_roster_receipt_path=expected_roster.path,
            person_roster_receipt_sha256=expected_roster.sha256,
        )
        params = _freeze_params(producer["params"])
        expected_instance = _person_instance_receipt(
            capability, expected_purpose, instance, params
        )
        if producer.get("purpose") != expected_purpose or producer.get(
            "person_instance"
        ) != expected_instance:
            raise MaskError("mask_artifact_mismatch")
        source = {
            "path": expected_source.path,
            "sha256": expected_source.sha256,
            "width": expected_source.width,
            "height": expected_source.height,
            "frame_pts": expected_source.frame_pts,
        }
        if producer.get("source") != source:
            raise MaskError("mask_artifact_mismatch")
        request_hash = request_sha256(
            provider=descriptor.provider,
            action=descriptor.action,
            model=descriptor.model,
            purpose=expected_purpose,
            provider_capability=capability,
            person_instance=instance,
            source_sha256=expected_source.sha256,
            width=expected_source.width,
            height=expected_source.height,
            frame_pts=expected_source.frame_pts,
            params=params,
            cache_version=producer["cache_version"],
        )
        if producer.get("request_sha256") != request_hash:
            raise MaskError("mask_artifact_mismatch")
        mask_payload = producer["mask"]
        if not isinstance(mask_payload, dict) or set(mask_payload) != {
            "path",
            "sha256",
            "size",
            "width",
            "height",
            "mime_type",
            "alpha_nonzero_pixels",
            "alpha_transparent_pixels",
        }:
            raise MaskError("mask_artifact_mismatch")
        mask_path = _relative_path(mask_payload["path"])
    except (KeyError, TypeError, ValueError, MaskError):
        raise MaskError("mask_artifact_mismatch") from None

    source_bytes = _read_project_file(root, expected_source.path, limit=MAX_SOURCE_BYTES)
    if sha256_bytes(source_bytes) != expected_source.sha256 or _decode_source(
        source_bytes
    ) != (expected_source.width, expected_source.height):
        raise MaskError("mask_artifact_mismatch")
    _validate_person_roster_receipt(
        root,
        instance,
        source_path=expected_source.path,
        source_sha256=expected_source.sha256,
        width=expected_source.width,
        height=expected_source.height,
        frame_pts=expected_source.frame_pts,
    )
    if mask_path in {expected_source.path, expected_roster.path}:
        raise MaskError("mask_artifact_mismatch")
    mask_bytes = _read_project_file(root, mask_path, limit=MAX_MASK_BYTES)
    mask_metadata, binary = _validated_mask(
        mask_bytes, width=expected_source.width, height=expected_source.height
    )
    request = {
        "schema": REQUEST_SCHEMA,
        "version": REQUEST_VERSION,
        "provider": descriptor.provider,
        "action": descriptor.action,
        "model": descriptor.model,
        "purpose": expected_purpose,
        "provider_capability": _capability_receipt(capability),
        "person_instance": expected_instance,
        "source": {key: value for key, value in source.items() if key != "frame_pts"},
        "frame_pts": expected_source.frame_pts,
        "params": params,
        "cache_version": producer["cache_version"],
        "request_sha256": request_hash,
    }
    canonical = _producer_receipt(request, mask_path, mask_metadata)
    if producer != canonical:
        raise MaskError("mask_artifact_mismatch")
    if attempt is not None and (
        attempt.get("request") != request
        or attempt.get("output_path") != mask_path
        or attempt.get("producer_receipt") != canonical
    ):
        raise MaskError("mask_artifact_mismatch")
    canonical_bytes = _canonical_json(canonical)
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    return LoadedMaskArtifact(
        canonical_receipt=canonical_bytes,
        canonical_receipt_sha256=sha256_bytes(canonical_bytes),
        project_relative_path=mask_path,
        mask_sha256=mask_metadata["sha256"],
        width=expected_source.width,
        height=expected_source.height,
        foreground_pixels=mask_metadata["alpha_nonzero_pixels"],
        packed_mask=packed,
    )


def _failed(
    root: Path,
    receipt_path: str,
    receipt: dict[str, Any],
    error: MaskError,
) -> None:
    _transition(
        root,
        receipt_path,
        receipt,
        "failed",
        error_code=error.code,
        public_error={"code": error.code},
    )


def _provider_request(
    request: Mapping[str, Any],
    provider_capability: ProviderCapabilities,
    person_instance: PersonInstanceRequest,
) -> ProviderMaskRequest:
    source = request["source"]
    return ProviderMaskRequest(
        provider=request["provider"],
        action=request["action"],
        model=request["model"],
        purpose=request["purpose"],
        provider_capability=provider_capability,
        person_instance=person_instance,
        source_sha256=source["sha256"],
        width=source["width"],
        height=source["height"],
        frame_pts=request["frame_pts"],
        request_sha256=request["request_sha256"],
        params=_freeze_params(request["params"]),
        cache_version=request["cache_version"],
    )


def generate_mask(
    *,
    project_root: Path,
    source_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    provider: MaskProvider,
    purpose: MaskPurpose,
    person_instance: PersonInstanceRequest,
    frame_pts: str | int | float | Decimal,
    params: Mapping[str, Any],
    cache_version: str,
) -> MaskResult:
    """Generate or GET-only recover one receipt-bound person mask.

    All artifact paths are project-relative.  The attempt receipt is written as
    ``prepared`` and then ``submitting`` before the only allowed POST.  A crash
    or timeout after that boundary without a task ID becomes terminal
    ``submission_unknown``.  Receipts with a task ID use only ``provider.get``.
    """
    if purpose not in MASK_PURPOSES:
        raise MaskError("invalid_mask_purpose")
    descriptor = getattr(provider, "descriptor", None)
    if not isinstance(descriptor, ProviderDescriptor):
        raise MaskError("invalid_provider_descriptor")
    capability = getattr(provider, "capabilities", None)
    if not isinstance(capability, ProviderCapabilities):
        raise MaskError("invalid_provider_capability")
    for method in ("submit", "get", "download"):
        if not callable(getattr(provider, method, None)):
            raise MaskError("invalid_provider_descriptor")
    normalized_pts = _frame_pts(frame_pts)
    frozen_params = _freeze_params(params)
    frozen_instance = _person_instance_receipt(
        capability, purpose, person_instance, frozen_params
    )
    if not isinstance(cache_version, str) or not cache_version or len(cache_version) > 128:
        raise MaskError("invalid_cache_version")
    root = _project_root(Path(project_root))
    source_relative = _relative_path(source_path)
    output_relative = _relative_path(output_path)
    receipt_relative = _relative_path(receipt_path)
    receipt_posix = PurePosixPath(receipt_relative)
    landing_relative = str(
        receipt_posix.parent / f".{receipt_posix.name}.provider-result"
    )
    if len({source_relative, output_relative, receipt_relative, landing_relative}) != 4:
        raise MaskError("unsafe_project_path")

    source_bytes = _read_project_file(root, source_relative, limit=MAX_SOURCE_BYTES)
    width, height = _decode_source(source_bytes)
    source_sha = sha256_bytes(source_bytes)
    instance_path = person_instance.person_roster_receipt_path
    if instance_path in {
        source_relative, output_relative, receipt_relative, landing_relative
    }:
        raise MaskError("unsafe_project_path")
    _validate_person_roster_receipt(
        root,
        person_instance,
        source_path=source_relative,
        source_sha256=source_sha,
        width=width,
        height=height,
        frame_pts=normalized_pts,
    )
    request_hash = request_sha256(
        provider=descriptor.provider,
        action=descriptor.action,
        model=descriptor.model,
        purpose=purpose,
        provider_capability=capability,
        person_instance=person_instance,
        source_sha256=source_sha,
        width=width,
        height=height,
        frame_pts=normalized_pts,
        params=frozen_params,
        cache_version=cache_version,
    )
    frozen_request = {
        "schema": REQUEST_SCHEMA,
        "version": REQUEST_VERSION,
        "provider": descriptor.provider,
        "action": descriptor.action,
        "model": descriptor.model,
        "purpose": purpose,
        "provider_capability": _capability_receipt(capability),
        "person_instance": frozen_instance,
        "source": {
            "path": source_relative,
            "sha256": source_sha,
            "width": width,
            "height": height,
        },
        "frame_pts": normalized_pts,
        "params": frozen_params,
        "cache_version": cache_version,
        "request_sha256": request_hash,
    }
    _check_destination(root, output_relative)
    _check_destination(root, receipt_relative)
    _check_destination(root, landing_relative)

    receipt = _load_receipt(root, receipt_relative)
    if receipt is None:
        receipt = {
            "schema": ATTEMPT_SCHEMA,
            "version": ATTEMPT_VERSION,
            "status": "prepared",
            "history": ["prepared"],
            "request": frozen_request,
            "output_path": output_relative,
            "landing_path": landing_relative,
        }
        _atomic_json(root, receipt_relative, receipt)
    else:
        if (
            receipt.get("schema") != ATTEMPT_SCHEMA
            or receipt.get("version") != ATTEMPT_VERSION
            or not isinstance(receipt.get("history"), list)
            or not isinstance(receipt.get("status"), str)
        ):
            raise MaskError("mask_receipt_invalid")
        if (
            receipt.get("request") != frozen_request
            or receipt.get("output_path") != output_relative
            or receipt.get("landing_path") != landing_relative
        ):
            raise MaskError("mask_receipt_mismatch")

    status = receipt["status"]
    if status == "succeeded":
        output = _read_project_file(root, output_relative, limit=MAX_MASK_BYTES)
        mask = _validate_mask(output, width=width, height=height)
        expected = _producer_receipt(frozen_request, output_relative, mask)
        if receipt.get("producer_receipt") != expected:
            raise MaskError("mask_receipt_invalid")
        return MaskResult(root / output_relative, root / receipt_relative, expected)
    if status == "failed":
        code = receipt.get("error_code")
        if not isinstance(code, str) or code not in _SAFE_CODES:
            raise MaskError("mask_receipt_invalid")
        raise MaskError(code)
    if status == "submission_unknown":
        raise MaskError("submission_unknown")
    if status == "submitting":
        receipt = _transition(
            root,
            receipt_relative,
            receipt,
            "submission_unknown",
            error_code="submission_unknown",
            public_error={"code": "submission_unknown"},
        )
        raise MaskError("submission_unknown")
    if status not in {"prepared", "accepted", "response_received", "downloaded", "validated"}:
        raise MaskError("mask_receipt_invalid")

    if status == "prepared":
        receipt = _transition(root, receipt_relative, receipt, "submitting")
        try:
            response = provider.submit(
                _provider_request(frozen_request, capability, person_instance)
            )
            provider_state = _provider_state(response)
        except SubmissionUncertain as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "submit"],
                error=exc,
                logger=log,
            )
            request_id = _optional_private_identifier(exc.request_id)
            task_id = _optional_private_identifier(exc.task_id)
            if task_id is None:
                private_state = {"request_id": request_id} if request_id else {}
                receipt = _transition(
                    root,
                    receipt_relative,
                    receipt,
                    "submission_unknown",
                    provider_response=private_state,
                    error_code="submission_unknown",
                    public_error={"code": "submission_unknown"},
                )
                raise MaskError("submission_unknown") from None
            provider_state = {"task_id": task_id}
            if request_id:
                provider_state["request_id"] = request_id
            receipt = _transition(
                root, receipt_relative, receipt, "accepted", provider_response=provider_state
            )
            status = "accepted"
        except MaskError as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "submit"],
                error=exc,
                logger=log,
            )
            _failed(root, receipt_relative, receipt, exc)
            raise
        except Exception as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "submit"],
                error=exc,
                logger=log,
            )
            receipt = _transition(
                root,
                receipt_relative,
                receipt,
                "submission_unknown",
                error_code="submission_unknown",
                public_error={"code": "submission_unknown"},
            )
            raise MaskError("submission_unknown") from None
        else:
            if "result_url" in provider_state:
                receipt = _transition(
                    root,
                    receipt_relative,
                    receipt,
                    "response_received",
                    provider_response=provider_state,
                )
                status = "response_received"
            else:
                receipt = _transition(
                    root,
                    receipt_relative,
                    receipt,
                    "accepted",
                    provider_response=provider_state,
                )
                status = "accepted"

    if status == "accepted":
        state = receipt.get("provider_response")
        task_id = state.get("task_id") if isinstance(state, Mapping) else None
        if not isinstance(task_id, str):
            raise MaskError("mask_receipt_invalid")
        try:
            response = provider.get(task_id)
            next_state = _provider_state(response)
        except MaskError as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "get"],
                error=exc,
                logger=log,
            )
            raise
        except Exception as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "get"],
                error=exc,
                logger=log,
            )
            raise MaskError("provider_query_failed", retryable=True) from None
        if next_state.get("task_id") not in {None, task_id}:
            raise MaskError("provider_protocol_error")
        combined_state = {**dict(state), **next_state, "task_id": task_id}
        if "result_url" not in combined_state:
            raise MaskError("provider_pending", retryable=True)
        receipt = _transition(
            root,
            receipt_relative,
            receipt,
            "response_received",
            provider_response=combined_state,
        )
        status = "response_received"

    if status == "response_received":
        state = receipt.get("provider_response")
        result_url = state.get("result_url") if isinstance(state, Mapping) else None
        try:
            private_url = _private_https_url(result_url)
            downloaded = provider.download(private_url)
            if not isinstance(downloaded, bytes) or not downloaded:
                raise MaskError("mask_download_failed")
            if len(downloaded) > MAX_MASK_BYTES:
                raise MaskError("mask_download_too_large")
            _atomic_project_bytes(root, landing_relative, downloaded)
        except MaskError as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "download"],
                error=exc,
                logger=log,
            )
            raise
        except Exception as exc:
            error_trace.record(
                root / f"{receipt_relative}.error.json",
                call_path=["image_masks", "provider", "download"],
                error=exc,
                logger=log,
            )
            raise MaskError("mask_download_failed", retryable=True) from None
        receipt = _transition(
            root,
            receipt_relative,
            receipt,
            "downloaded",
            downloaded={
                "path": landing_relative,
                "sha256": sha256_bytes(downloaded),
                "size": len(downloaded),
            },
        )
        status = "downloaded"

    if status == "downloaded":
        downloaded_receipt = receipt.get("downloaded")
        if not isinstance(downloaded_receipt, Mapping):
            raise MaskError("mask_receipt_invalid")
        downloaded = _read_project_file(root, landing_relative, limit=MAX_MASK_BYTES)
        if (
            downloaded_receipt.get("path") != landing_relative
            or downloaded_receipt.get("sha256") != sha256_bytes(downloaded)
            or downloaded_receipt.get("size") != len(downloaded)
        ):
            raise MaskError("mask_receipt_invalid")
        try:
            mask = _validate_mask(downloaded, width=width, height=height)
        except MaskError as exc:
            _failed(root, receipt_relative, receipt, exc)
            raise
        producer_receipt = _producer_receipt(frozen_request, output_relative, mask)
        receipt = _transition(
            root,
            receipt_relative,
            receipt,
            "validated",
            validated_mask=mask,
            producer_receipt=producer_receipt,
        )
        status = "validated"

    if status == "validated":
        downloaded = _read_project_file(root, landing_relative, limit=MAX_MASK_BYTES)
        mask = _validate_mask(downloaded, width=width, height=height)
        producer_receipt = _producer_receipt(frozen_request, output_relative, mask)
        if (
            receipt.get("validated_mask") != mask
            or receipt.get("producer_receipt") != producer_receipt
        ):
            raise MaskError("mask_receipt_invalid")
        _atomic_project_bytes(root, output_relative, downloaded)
        receipt = _transition(
            root,
            receipt_relative,
            receipt,
            "succeeded",
            producer_receipt=producer_receipt,
        )
        return MaskResult(root / output_relative, root / receipt_relative, producer_receipt)

    raise MaskError("mask_receipt_invalid")
