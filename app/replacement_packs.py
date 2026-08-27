"""Durable project-level replacement packs, generated before frame editing.

Source images are generation conditions and quality-gate evidence only.  They
never appear in the executor-facing target image list.  Paid retry semantics are
owned by the existing Seedream adapter; semantic acceptance is owned by the
single injected image-quality gate.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

from app import seedream


CANDIDATE_SCHEMA = "duet.replacement-pack-candidate"
PACK_SCHEMA = "duet.replacement-packs"
SCHEMA_VERSION = 1
CANDIDATE_RECEIPT_PATH = "work/replacement-packs/candidate.json"
PACK_RECEIPT_PATH = "work/replacement-packs/pack.json"
QUALITY_RECEIPT_PATH = "work/replacement-packs/quality-receipt.json"
ROLES = ("primary", "alternate")
SOURCE_REFERENCE_POLICY = (
    "Source frames are soft observations for attributes, composition, relationships, "
    "lighting/layout and negative evidence only; never use them as target identity or "
    "target scene references."
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PNG = b"\x89PNG\r\n\x1a\n"
_MAX_IMAGE_BYTES = 64 * 1024 * 1024

PackKind = Literal["person", "scene"]
BuildStatus = Literal["ready", "submission_unknown", "unknown", "failed"]


class ReplacementPackError(ValueError):
    pass


class PackGenerationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class PersonPlan:
    plan_id: str
    source_frames: tuple[str, ...]
    prompt: str
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class ScenePlan:
    plan_id: str
    source_frames: tuple[str, ...]
    prompt: str
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class ProjectReplacementPlan:
    people: tuple[PersonPlan, ...]
    scenes: tuple[ScenePlan, ...]
    upstream_plan_sha256: str
    upstream_source_inventory_sha256: str
    execution_profile: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenSource:
    relative_path: str
    sha256: str
    size: int
    width: int
    height: int
    png_bytes: bytes


@dataclass(frozen=True)
class GenerationRequest:
    kind: PackKind
    plan_id: str
    model: str
    revision: int
    user_prompt: str
    prompt: str
    roles: tuple[str, str]
    role_prompts: tuple[str, str]
    profile: Mapping[str, Any]
    sources: tuple[FrozenSource, ...]
    upstream_plan_sha256: str
    upstream_source_inventory_sha256: str
    execution_profile: Mapping[str, Any]
    neutral_png: bytes
    neutral_sha256: str
    width: int
    height: int
    input_sha256: str
    work_dir: Path


@dataclass(frozen=True)
class GeneratedImage:
    role: str
    png_bytes: bytes


@dataclass(frozen=True)
class GenerationResult:
    images: tuple[GeneratedImage, GeneratedImage]
    producer_receipt: Mapping[str, Any]


class PackGenerator(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


@dataclass(frozen=True)
class SourceImageDTO:
    path: Path
    relative_path: str
    sha256: str
    size: int
    width: int
    height: int


@dataclass(frozen=True)
class ReferenceImageDTO:
    role: str
    path: Path
    relative_path: str
    sha256: str
    size: int
    width: int
    height: int


@dataclass(frozen=True)
class EntityReferencePackDTO:
    kind: PackKind
    plan_id: str
    plan_sha256: str
    profile_sha256: str
    producer_receipt_path: Path
    producer_receipt_relative_path: str
    producer_receipt_sha256: str
    sources: tuple[SourceImageDTO, ...]
    images: tuple[ReferenceImageDTO, ReferenceImageDTO]


@dataclass(frozen=True)
class ReplacementPackCandidateDTO:
    schema: str
    version: int
    project_dir: Path
    receipt_path: str
    candidate_sha256: str
    input_sha256: str
    output_sha256: str
    plan_sha256: str
    profile_sha256: str
    source_inventory_sha256: str
    upstream_plan_sha256: str
    upstream_source_inventory_sha256: str
    execution_profile: Mapping[str, Any]
    execution_profile_sha256: str
    model: str
    revision: int
    people: Mapping[str, EntityReferencePackDTO]
    scenes: Mapping[str, EntityReferencePackDTO]


@dataclass(frozen=True)
class ReplacementPackDTO(ReplacementPackCandidateDTO):
    quality_receipt_path: str
    quality_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class PackQualityResult:
    status: Literal["pass", "fail", "unknown"]
    publishable: bool
    receipt: Mapping[str, Any]


class PackQualityGate(Protocol):
    def evaluate(
        self, candidate: ReplacementPackCandidateDTO, *, receipt_path: Path
    ) -> PackQualityResult: ...


@dataclass(frozen=True)
class PackBuildResult:
    status: BuildStatus
    pack: ReplacementPackDTO | None = None
    issues: tuple[str, ...] = ()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ReplacementPackError("receipt values must be finite JSON") from None


def _copy_json(value: Any) -> Any:
    return json.loads(_canonical(value))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_source_inventory_sha256(
    frames: Sequence[Mapping[str, Any]],
) -> str:
    """Hash ordered frozen frames projected to the four cross-module fields."""
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence) or not frames:
        raise ReplacementPackError("source inventory must be a non-empty ordered sequence")
    items = []
    positions = set()
    for raw in frames:
        if not isinstance(raw, Mapping):
            raise ReplacementPackError("source inventory frame is invalid")
        segment = raw.get("segment_index")
        frame = raw.get("frame_index")
        name = raw.get("frame_name")
        digest = raw.get("source_sha256")
        if (
            isinstance(segment, bool) or not isinstance(segment, int) or segment < 0
            or isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
            or not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).name != name
            or not isinstance(digest, str) or _SHA.fullmatch(digest) is None
            or (segment, frame, name) in positions
        ):
            raise ReplacementPackError("source inventory frame is invalid")
        positions.add((segment, frame, name))
        items.append({
            "segment_index": segment, "frame_index": frame,
            "frame_name": name, "source_sha256": digest,
        })
    return _hash(items)


def _root(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ReplacementPackError("project root must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise ReplacementPackError("project root is unavailable") from None
    if not resolved.is_dir():
        raise ReplacementPackError("project root is unavailable")
    return resolved


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReplacementPackError("path must be a safe project-relative path")
    parts = value.split("/")
    if (
        PurePosixPath(value).is_absolute()
        or PurePosixPath(value).as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ReplacementPackError("path must be a safe project-relative path")
    return value


def _path(root: Path, relative: str, *, must_exist: bool) -> Path:
    relative = _relative(relative)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ReplacementPackError("project paths must not contain symlinks")
    try:
        resolved = current.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ReplacementPackError("path escapes the project") from None
    return resolved


def _directory(root: Path, relative: str) -> Path:
    relative = _relative(relative)
    current = root
    for part in PurePosixPath(relative).parts:
        child = current / part
        if child.is_symlink():
            raise ReplacementPackError("project paths must not contain symlinks")
        existed = child.exists()
        try:
            child.mkdir(exist_ok=True)
        except OSError:
            raise ReplacementPackError("pack storage is unavailable") from None
        if not child.is_dir():
            raise ReplacementPackError("pack storage is unavailable")
        if not existed:
            seedream._fsync_dir(current)
        current = child
    return current.resolve(strict=True)


def _read(root: Path, relative: str) -> bytes:
    path = _path(root, relative, must_exist=True)
    if not path.is_file():
        raise ReplacementPackError("project file is unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read()
    except OSError:
        raise ReplacementPackError("project file is unavailable") from None


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(_read(root, relative).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ReplacementPackError("receipt is invalid JSON") from None
    if not isinstance(value, dict):
        raise ReplacementPackError("receipt must be an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    seedream._atomic_json(path, _copy_json(value))


def _image(data: bytes) -> tuple[int, int]:
    if not isinstance(data, bytes) or not data.startswith(_PNG) or not 0 < len(data) <= _MAX_IMAGE_BYTES:
        raise ReplacementPackError("image must be a bounded PNG")
    decoded = seedream.cv2.imdecode(
        seedream.np.frombuffer(data, seedream.np.uint8), seedream.cv2.IMREAD_UNCHANGED
    )
    if decoded is None:
        raise ReplacementPackError("image must be a valid PNG")
    return int(decoded.shape[1]), int(decoded.shape[0])


def _neutral_png(width: int, height: int) -> bytes:
    canvas = seedream.np.full((height, width, 3), 127, dtype=seedream.np.uint8)
    ok, encoded = seedream.cv2.imencode(".png", canvas)
    if not ok:
        raise ReplacementPackError("neutral canvas could not be encoded")
    return encoded.tobytes()


def _source(root: Path, relative: str) -> FrozenSource:
    relative = _relative(relative)
    if PurePosixPath(relative).suffix.lower() != ".png":
        raise ReplacementPackError("source reference must be PNG")
    data = _read(root, relative)
    width, height = _image(data)
    return FrozenSource(
        relative, _bytes_hash(data), len(data), width, height, data,
    )


def _manifest(source: FrozenSource) -> dict[str, Any]:
    return {
        "path": source.relative_path, "sha256": source.sha256,
        "size": source.size, "width": source.width, "height": source.height,
    }


def _validate_id(value: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ReplacementPackError("plan id is invalid")
    return value


def _validate_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ReplacementPackError(f"{label} must be a lowercase SHA-256")
    return value


def _requests(
    root: Path, plan: ProjectReplacementPlan, model: str, revision: int
) -> tuple[GenerationRequest, ...]:
    if not isinstance(plan, ProjectReplacementPlan):
        raise ReplacementPackError("project plan is invalid")
    if (
        not isinstance(model, str) or not model.strip()
        or isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
    ):
        raise ReplacementPackError("model and revision are required")
    upstream_plan = _validate_sha(plan.upstream_plan_sha256, "upstream_plan_sha256")
    upstream_sources = _validate_sha(
        plan.upstream_source_inventory_sha256, "upstream_source_inventory_sha256"
    )
    if not isinstance(plan.execution_profile, Mapping):
        raise ReplacementPackError("execution_profile must be a frozen profile")
    execution_profile = _copy_json(dict(plan.execution_profile))
    if (
        not isinstance(execution_profile, dict)
        or set(execution_profile) != {"id", "revision"}
        or not isinstance(execution_profile.get("id"), str)
        or not execution_profile["id"].strip()
        or execution_profile["id"] != execution_profile["id"].strip()
        or isinstance(execution_profile.get("revision"), bool)
        or not isinstance(execution_profile.get("revision"), int)
        or execution_profile["revision"] < 1
    ):
        raise ReplacementPackError("execution_profile must be a frozen profile")
    entries: list[tuple[PackKind, PersonPlan | ScenePlan]] = [
        *(("person", item) for item in plan.people),
        *(("scene", item) for item in plan.scenes),
    ]
    if not entries:
        raise ReplacementPackError("project plan must not be empty")
    ids = [item.plan_id for _, item in entries]
    if len(ids) != len(set(ids)):
        raise ReplacementPackError("plan ids must be unique")
    requests = []
    for kind, item in entries:
        plan_id = _validate_id(item.plan_id)
        if not isinstance(item.prompt, str) or not item.prompt.strip():
            raise ReplacementPackError("plan prompt is required")
        if not isinstance(item.profile, Mapping):
            raise ReplacementPackError("plan profile must be an object")
        profile = _copy_json(dict(item.profile))
        if not isinstance(item.source_frames, tuple) or not item.source_frames:
            raise ReplacementPackError("source_frames must be a non-empty tuple")
        if len(item.source_frames) > 8:
            raise ReplacementPackError("source_frames exceed Seedream's 10-image limit")
        sources = tuple(_source(root, path) for path in item.source_frames)
        policy = (
            "Create one new identity, internally consistent across both views, dissimilar "
            "to every source and non-confusable with other planned people."
            if kind == "person"
            else "Create one new environment whose semantics, spatial structure and color "
            "differ from the source; it must not be a color-grade-only change. Preserve "
            "only global light direction, exposure, white balance and grading style."
        )
        prompt = f"{item.prompt.strip()}\n\n{policy}\n{SOURCE_REFERENCE_POLICY}"
        role_prompts = (
            f"{prompt}\nOutput role: primary. Establish the new target from the neutral canvas.",
            f"{prompt}\nOutput role: alternate. Preserve exactly the same new target as the "
            "primary target reference and render a complementary view.",
        )
        neutral = _neutral_png(sources[0].width, sources[0].height)
        neutral_manifest = {
            "sha256": _bytes_hash(neutral), "size": len(neutral),
            "width": sources[0].width, "height": sources[0].height,
            "content": "solid-neutral-canvas",
        }
        frozen = {
            "kind": kind, "plan_id": plan_id, "model": model.strip(),
            "revision": revision, "user_prompt": item.prompt.strip(),
            "prompt": prompt, "role_prompts": [
                {"role": role, "prompt": text, "sha256": _bytes_hash(text.encode())}
                for role, text in zip(ROLES, role_prompts)
            ],
            "profile": profile, "profile_sha256": _hash(profile),
            "sources": [_manifest(source) for source in sources],
            "neutral_canvas": neutral_manifest,
            "upstream_plan_sha256": upstream_plan,
            "upstream_source_inventory_sha256": upstream_sources,
            "execution_profile": execution_profile,
            "execution_profile_sha256": _hash(execution_profile),
            "reference_policy": SOURCE_REFERENCE_POLICY,
        }
        digest = _hash(frozen)
        work = _path(
            root, f"work/replacement-packs/entities/{kind}-{plan_id}-{digest[:16]}",
            must_exist=False,
        )
        requests.append(GenerationRequest(
            kind, plan_id, model.strip(), revision, item.prompt.strip(), prompt,
            ROLES, role_prompts, _freeze(profile), sources, upstream_plan,
            upstream_sources, _freeze(execution_profile), neutral,
            neutral_manifest["sha256"], sources[0].width, sources[0].height,
            digest, work,
        ))
    return tuple(requests)


def _request_receipt(request: GenerationRequest) -> dict[str, Any]:
    return {
        "kind": request.kind, "plan_id": request.plan_id, "model": request.model,
        "revision": request.revision, "user_prompt": request.user_prompt,
        "prompt": request.prompt, "role_prompts": [
            {"role": role, "prompt": text, "sha256": _bytes_hash(text.encode())}
            for role, text in zip(request.roles, request.role_prompts)
        ],
        "profile": _copy_json(request.profile),
        "profile_sha256": _hash(request.profile),
        "sources": [_manifest(source) for source in request.sources],
        "neutral_canvas": {
            "sha256": request.neutral_sha256, "size": len(request.neutral_png),
            "width": request.width, "height": request.height,
            "content": "solid-neutral-canvas",
        },
        "upstream_plan_sha256": request.upstream_plan_sha256,
        "upstream_source_inventory_sha256": request.upstream_source_inventory_sha256,
        "execution_profile": _copy_json(request.execution_profile),
        "execution_profile_sha256": _hash(request.execution_profile),
        "reference_policy": SOURCE_REFERENCE_POLICY,
        "input_sha256": request.input_sha256,
    }


class SeedreamPackGenerator:
    """Seedream v2 adapter: neutral canvas first, primary anchors alternate."""

    def __init__(self, project_root: Path, settings: Any, *, transport: Any = None) -> None:
        self._root = _root(project_root)
        self._settings = settings
        self._transport = transport

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.model != self._settings.seedream_model:
            raise PackGenerationError("generator_model_mismatch")
        provider_dir = _directory(
            self._root,
            f"work/replacement-packs/provider/{request.kind}-{request.plan_id}-{request.input_sha256[:16]}",
        )
        neutral_path = provider_dir / "neutral.png"
        if neutral_path.exists() or neutral_path.is_symlink():
            if _bytes_hash(_read(self._root, neutral_path.relative_to(self._root).as_posix())) != request.neutral_sha256:
                raise PackGenerationError("neutral_canvas_mismatch")
        else:
            seedream._atomic_bytes(neutral_path, request.neutral_png)
        images: list[GeneratedImage] = []
        receipts = []
        for role, prompt in zip(request.roles, request.role_prompts):
            output = provider_dir / f"{role}.png"
            provider_receipt = provider_dir / f"{role}.json"
            request_receipt = provider_dir / f"{role}.request.json"
            input_bytes = [request.neutral_png]
            input_roles = ["current_frame"]
            if role == "alternate":
                if not images or images[0].role != "primary":
                    raise PackGenerationError("primary_reference_unavailable")
                input_bytes.append(images[0].png_bytes)
                input_roles.append(
                    f"target_reference:{request.kind}:{request.plan_id}:primary"
                )
            input_bytes.extend(source.png_bytes for source in request.sources)
            input_roles.extend(
                f"source_negative:{request.kind}:{request.plan_id}:{index}"
                for index in range(1, len(request.sources) + 1)
            )
            binding = {
                "plan_sha256": request.upstream_plan_sha256,
                "profile": _copy_json(request.execution_profile),
                "revision": request.revision,
                "input_roles": input_roles,
            }
            frozen = {
                "schema": "duet.replacement-pack-paid-request", "version": 1,
                "kind": request.kind, "plan_id": request.plan_id, "role": role,
                "model": request.model, "mode": self._settings.seedream_edit_mode,
                "prompt": prompt, "prompt_sha256": _bytes_hash(prompt.encode("utf-8")),
                "input_order": [
                    {"position": index, "role": input_role, "sha256": _bytes_hash(data)}
                    for index, (input_role, data) in enumerate(zip(input_roles, input_bytes), 1)
                ],
                "execution_binding": binding,
                "entity_input_sha256": request.input_sha256,
                "upstream_source_inventory_sha256": request.upstream_source_inventory_sha256,
                "reference_policy": SOURCE_REFERENCE_POLICY,
            }
            frozen["sha256"] = _hash(frozen)
            relative_request = request_receipt.relative_to(self._root).as_posix()
            if request_receipt.exists() or request_receipt.is_symlink():
                if _read_json(self._root, relative_request) != frozen:
                    raise PackGenerationError("paid_request_receipt_mismatch")
            else:
                _write_json(request_receipt, frozen)
            try:
                await seedream.edit(
                    self._settings,
                    input_bytes,
                    prompt,
                    output,
                    receipt_path=provider_receipt,
                    transport=self._transport,
                    execution_binding=binding,
                )
            except TypeError:
                raise PackGenerationError("seedream_v2_required") from None
            except seedream.SeedreamError as error:
                raise PackGenerationError(error.code) from None
            data = _read(self._root, output.relative_to(self._root).as_posix())
            _image(data)
            raw_receipt = _read_json(
                self._root, provider_receipt.relative_to(self._root).as_posix()
            )
            if (
                raw_receipt.get("version") != 2
                or raw_receipt.get("status") != "succeeded"
                or raw_receipt.get("plan_sha256") != request.upstream_plan_sha256
                or raw_receipt.get("profile") != _copy_json(request.execution_profile)
                or raw_receipt.get("revision") != request.revision
                or [item.get("role") for item in raw_receipt.get("input_order", [])]
                != input_roles
            ):
                raise PackGenerationError("provider_receipt_invalid")
            images.append(GeneratedImage(role, data))
            receipts.append({
                "role": role,
                "request_path": relative_request,
                "request_sha256": _bytes_hash(_read(self._root, relative_request)),
                "path": provider_receipt.relative_to(self._root).as_posix(),
                "sha256": _bytes_hash(_read(
                    self._root, provider_receipt.relative_to(self._root).as_posix()
                )),
                "status": "succeeded",
            })
        return GenerationResult(
            images=tuple(images),  # type: ignore[arg-type]
            producer_receipt={
                "adapter": "seedream", "model": request.model,
                "mode": self._settings.seedream_edit_mode, "roles": receipts,
            },
        )


class ImageQualityPackGate:
    """Thin adapter to the sole image-quality reference-pack gate."""

    def __init__(
        self, *, plan: Mapping[str, Any], frame_masks: Sequence[Any],
        profile: Any, semantic_verifier: Any,
    ) -> None:
        self._plan = _copy_json(dict(plan))
        self._frame_masks = tuple(frame_masks)
        self._profile = profile
        self._semantic_verifier = semantic_verifier

    def evaluate(
        self, candidate: ReplacementPackCandidateDTO, *, receipt_path: Path
    ) -> PackQualityResult:
        from app.image_quality import evaluate_reference_packs

        receipt = evaluate_reference_packs(
            candidate,
            plan=self._plan,
            frame_masks=self._frame_masks,
            profile=self._profile,
            semantic_verifier=self._semantic_verifier,
            receipt_path=receipt_path,
        )
        raw = receipt.to_dict()
        return PackQualityResult(receipt.status, receipt.publishable, raw)


async def _entity(
    root: Path, request: GenerationRequest, generator: PackGenerator
) -> dict[str, Any]:
    _directory(root, request.work_dir.relative_to(root).as_posix())
    receipt_path = request.work_dir / "generation.json"
    relative_receipt = receipt_path.relative_to(root).as_posix()
    prepared = {
        "schema": "duet.replacement-pack-generation", "version": 1,
        "status": "prepared", "request": _request_receipt(request),
    }
    prepared["sha256"] = _hash(prepared)
    if receipt_path.is_file() or receipt_path.is_symlink():
        raw = _read_json(root, relative_receipt)
        unsigned = dict(raw)
        if (
            unsigned.pop("sha256", None) != _hash(unsigned)
            or raw.get("schema") != prepared["schema"]
            or raw.get("version") != 1
            or raw.get("request") != prepared["request"]
        ):
            raise ReplacementPackError("generation receipt is invalid")
        if raw.get("status") == "prepared" and set(raw) == set(prepared):
            pass
        elif raw.get("status") == "completed" and set(raw) == {
            "schema", "version", "status", "request", "producer",
            "producer_receipt_sha256", "outputs", "sha256",
        }:
            outputs = raw.get("outputs")
            producer = raw.get("producer")
            if (
                not isinstance(producer, dict)
                or raw.get("producer_receipt_sha256") != _hash(producer)
                or not isinstance(outputs, list) or len(outputs) != 2
                or any(not isinstance(output, dict) for output in outputs)
                or [output.get("role") for output in outputs] != list(ROLES)
            ):
                raise ReplacementPackError("generation receipt is invalid")
            for output in outputs:
                if set(output) != {"role", "path", "sha256", "size", "width", "height"}:
                    raise ReplacementPackError("generation output receipt is invalid")
                data = _read(root, output.get("path", ""))
                width, height = _image(data)
                if (
                    output.get("sha256") != _bytes_hash(data)
                    or output.get("size") != len(data)
                    or output.get("width") != width or output.get("height") != height
                ):
                    raise ReplacementPackError("generation output binding mismatch")
            return raw
        else:
            raise ReplacementPackError("generation receipt is invalid")
    else:
        _write_json(receipt_path, prepared)
    try:
        result = await generator.generate(request)
    except PackGenerationError:
        raise
    except Exception:
        raise PackGenerationError("generator_unknown") from None
    if (
        not isinstance(result, GenerationResult)
        or tuple(image.role for image in result.images) != ROLES
        or not isinstance(result.producer_receipt, Mapping)
        or not result.producer_receipt
    ):
        raise PackGenerationError("generator_output_invalid")
    output_dir = _directory(root, request.work_dir.relative_to(root).as_posix() + "/outputs")
    outputs = []
    for image in result.images:
        width, height = _image(image.png_bytes)
        path = output_dir / f"{image.role}.png"
        seedream._atomic_bytes(path, image.png_bytes)
        outputs.append({
            "role": image.role, "path": path.relative_to(root).as_posix(),
            "sha256": _bytes_hash(image.png_bytes), "size": len(image.png_bytes),
            "width": width, "height": height,
        })
    producer = _copy_json(dict(result.producer_receipt))
    raw = {
        "schema": "duet.replacement-pack-generation", "version": 1,
        "status": "completed", "request": _request_receipt(request),
        "producer": producer, "producer_receipt_sha256": _hash(producer),
        "outputs": outputs,
    }
    raw["sha256"] = _hash(raw)
    _write_json(receipt_path, raw)
    return raw


def _candidate_receipt(requests: Sequence[GenerationRequest], entities: Sequence[dict]) -> dict:
    items = []
    for request, generated in zip(requests, entities):
        producer_path = request.work_dir / "generation.json"
        items.append({
            "kind": request.kind, "plan_id": request.plan_id,
            "plan": {
                "kind": request.kind, "plan_id": request.plan_id,
                "prompt": request.user_prompt, "sources": [s.relative_path for s in request.sources],
                "roles": list(ROLES),
            },
            "plan_sha256": _hash({
                "kind": request.kind, "plan_id": request.plan_id,
                "prompt": request.user_prompt, "sources": [s.relative_path for s in request.sources],
                "roles": list(ROLES),
            }),
            "profile": _copy_json(request.profile), "profile_sha256": _hash(request.profile),
            "sources": [_manifest(source) for source in request.sources],
            "outputs": generated["outputs"],
            "producer_receipt_path": producer_path.relative_to(
                request.work_dir.parents[3]
            ).as_posix(),
            "producer_receipt_sha256": _bytes_hash(producer_path.read_bytes()),
        })
    receipt = {
        "schema": CANDIDATE_SCHEMA, "version": SCHEMA_VERSION,
        "model": requests[0].model, "revision": requests[0].revision,
        "upstream_plan_sha256": requests[0].upstream_plan_sha256,
        "upstream_source_inventory_sha256": requests[0].upstream_source_inventory_sha256,
        "execution_profile": _copy_json(requests[0].execution_profile),
        "execution_profile_sha256": _hash(requests[0].execution_profile),
        "ordered_plans": [f"{item['kind']}:{item['plan_id']}" for item in items],
        "input_sha256": _hash([request.input_sha256 for request in requests]),
        "output_sha256": _hash([item["outputs"] for item in items]),
        "plan_sha256": _hash([item["plan_sha256"] for item in items]),
        "profile_sha256": _hash([item["profile_sha256"] for item in items]),
        "source_inventory_sha256": _hash([item["sources"] for item in items]),
        "entities": items,
    }
    receipt["candidate_sha256"] = _hash(receipt)
    return receipt


def _image_dto(root: Path, raw: Mapping[str, Any], *, role: str | None) -> Any:
    expected = {"path", "sha256", "size", "width", "height"} | ({"role"} if role else set())
    if (
        not isinstance(raw, Mapping) or set(raw) != expected
        or (role is not None and raw.get("role") != role)
    ):
        raise ReplacementPackError("candidate image receipt is invalid")
    data = _read(root, raw["path"])
    width, height = _image(data)
    if (
        raw["sha256"] != _bytes_hash(data) or raw["size"] != len(data)
        or raw["width"] != width or raw["height"] != height
    ):
        raise ReplacementPackError("candidate image binding mismatch")
    values = dict(
        path=_path(root, raw["path"], must_exist=True), relative_path=raw["path"],
        sha256=raw["sha256"], size=len(data), width=width, height=height,
    )
    return SourceImageDTO(**values) if role is None else ReferenceImageDTO(role=role, **values)


def _candidate_dto(root: Path, raw: dict[str, Any]) -> ReplacementPackCandidateDTO:
    keys = {
        "schema", "version", "model", "revision", "upstream_plan_sha256",
        "upstream_source_inventory_sha256", "execution_profile",
        "execution_profile_sha256", "ordered_plans", "input_sha256", "output_sha256",
        "plan_sha256", "profile_sha256", "source_inventory_sha256", "entities",
        "candidate_sha256",
    }
    if set(raw) != keys or raw.get("schema") != CANDIDATE_SCHEMA or raw.get("version") != 1:
        raise ReplacementPackError("candidate receipt shape is invalid")
    unsigned = dict(raw)
    if unsigned.pop("candidate_sha256") != _hash(unsigned):
        raise ReplacementPackError("candidate receipt sha256 mismatch")
    if raw["execution_profile_sha256"] != _hash(raw["execution_profile"]):
        raise ReplacementPackError("candidate execution profile mismatch")
    if (
        not isinstance(raw.get("model"), str) or not raw["model"].strip()
        or isinstance(raw.get("revision"), bool)
        or not isinstance(raw.get("revision"), int) or raw["revision"] < 1
        or not isinstance(raw.get("execution_profile"), dict)
        or set(raw["execution_profile"]) != {"id", "revision"}
        or not isinstance(raw["execution_profile"].get("id"), str)
        or not raw["execution_profile"]["id"].strip()
        or raw["execution_profile"]["id"] != raw["execution_profile"]["id"].strip()
        or isinstance(raw["execution_profile"].get("revision"), bool)
        or not isinstance(raw["execution_profile"].get("revision"), int)
        or raw["execution_profile"]["revision"] < 1
        or not isinstance(raw.get("entities"), list) or not raw["entities"]
    ):
        raise ReplacementPackError("candidate execution binding is invalid")
    for field in (
        "upstream_plan_sha256", "upstream_source_inventory_sha256", "input_sha256",
        "output_sha256", "plan_sha256", "profile_sha256", "source_inventory_sha256",
        "candidate_sha256",
    ):
        _validate_sha(raw[field], field)
    entities = []
    input_hashes = []
    for item in raw["entities"] if isinstance(raw.get("entities"), list) else []:
        if not isinstance(item, dict) or set(item) != {
            "kind", "plan_id", "plan", "plan_sha256", "profile", "profile_sha256",
            "sources", "outputs", "producer_receipt_path", "producer_receipt_sha256",
        } or item["kind"] not in {"person", "scene"}:
            raise ReplacementPackError("candidate entity is invalid")
        _validate_id(item["plan_id"])
        if item["plan_sha256"] != _hash(item["plan"]) or item["profile_sha256"] != _hash(item["profile"]):
            raise ReplacementPackError("candidate entity plan/profile mismatch")
        if (
            not isinstance(item["sources"], list) or not item["sources"]
            or not isinstance(item["outputs"], list) or len(item["outputs"]) != 2
        ):
            raise ReplacementPackError("candidate entity inventory is invalid")
        sources = tuple(_image_dto(root, source, role=None) for source in item["sources"])
        images = tuple(
            _image_dto(root, output, role=role) for role, output in zip(ROLES, item["outputs"])
        )
        if not sources or len(images) != 2 or [out.get("role") for out in item["outputs"]] != list(ROLES):
            raise ReplacementPackError("candidate entity inventory is invalid")
        producer = _read(root, item["producer_receipt_path"])
        if item["producer_receipt_sha256"] != _bytes_hash(producer):
            raise ReplacementPackError("candidate producer receipt mismatch")
        try:
            producer_raw = json.loads(producer.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ReplacementPackError("candidate producer receipt is invalid") from None
        producer_unsigned = dict(producer_raw)
        if (
            producer_unsigned.pop("sha256", None) != _hash(producer_unsigned)
            or producer_raw.get("status") != "completed"
            or producer_raw.get("request", {}).get("kind") != item["kind"]
            or producer_raw.get("request", {}).get("plan_id") != item["plan_id"]
            or producer_raw.get("request", {}).get("model") != raw["model"]
            or producer_raw.get("request", {}).get("revision") != raw["revision"]
            or producer_raw.get("request", {}).get("upstream_plan_sha256")
            != raw["upstream_plan_sha256"]
            or producer_raw.get("request", {}).get("upstream_source_inventory_sha256")
            != raw["upstream_source_inventory_sha256"]
            or producer_raw.get("request", {}).get("execution_profile")
            != raw["execution_profile"]
            or producer_raw.get("outputs") != item["outputs"]
        ):
            raise ReplacementPackError("candidate producer receipt binding mismatch")
        input_hashes.append(producer_raw["request"].get("input_sha256"))
        entities.append(EntityReferencePackDTO(
            item["kind"], item["plan_id"], item["plan_sha256"], item["profile_sha256"],
            _path(root, item["producer_receipt_path"], must_exist=True),
            item["producer_receipt_path"], item["producer_receipt_sha256"],
            sources, images,  # type: ignore[arg-type]
        ))
    ordered = [f"{item.kind}:{item.plan_id}" for item in entities]
    if raw["ordered_plans"] != ordered or len(ordered) != len(set(ordered)):
        raise ReplacementPackError("candidate plan order is invalid")
    if raw["input_sha256"] != _hash(input_hashes):
        raise ReplacementPackError("candidate input binding mismatch")
    if raw["output_sha256"] != _hash([item["outputs"] for item in raw["entities"]]):
        raise ReplacementPackError("candidate output binding mismatch")
    if raw["plan_sha256"] != _hash([item["plan_sha256"] for item in raw["entities"]]):
        raise ReplacementPackError("candidate plan binding mismatch")
    if raw["profile_sha256"] != _hash([item["profile_sha256"] for item in raw["entities"]]):
        raise ReplacementPackError("candidate profile binding mismatch")
    if raw["source_inventory_sha256"] != _hash([item["sources"] for item in raw["entities"]]):
        raise ReplacementPackError("candidate source binding mismatch")
    people = {item.plan_id: item for item in entities if item.kind == "person"}
    scenes = {item.plan_id: item for item in entities if item.kind == "scene"}
    return ReplacementPackCandidateDTO(
        CANDIDATE_SCHEMA, 1, root, CANDIDATE_RECEIPT_PATH, raw["candidate_sha256"],
        raw["input_sha256"], raw["output_sha256"], raw["plan_sha256"],
        raw["profile_sha256"], raw["source_inventory_sha256"],
        raw["upstream_plan_sha256"], raw["upstream_source_inventory_sha256"],
        _freeze(raw["execution_profile"]), raw["execution_profile_sha256"],
        raw["model"], raw["revision"], MappingProxyType(people), MappingProxyType(scenes),
    )


def load_replacement_pack_candidate(project_root: Path) -> ReplacementPackCandidateDTO:
    root = _root(project_root)
    return _candidate_dto(root, _read_json(root, CANDIDATE_RECEIPT_PATH))


def _quality_bound(candidate: ReplacementPackCandidateDTO, result: PackQualityResult) -> bool:
    if not isinstance(result, PackQualityResult) or not isinstance(result.receipt, Mapping):
        return False
    try:
        raw = _copy_json(dict(result.receipt))
    except ReplacementPackError:
        return False
    digest = raw.pop("sha256", None)
    return (
        result.status in {"pass", "fail", "unknown"}
        and result.publishable is (result.status == "pass")
        and raw.get("status") == result.status
        and raw.get("publishable") is result.publishable
        and raw.get("provider_retry_allowed") is False
        and raw.get("plan_sha256") == candidate.upstream_plan_sha256
        and raw.get("reference_pack_candidate_sha256") == candidate.candidate_sha256
        and digest == _hash(raw)
    )


def load_replacement_pack(
    project_root: Path,
    *,
    expected_upstream_plan_sha256: str | None = None,
    expected_upstream_source_inventory_sha256: str | None = None,
    expected_execution_profile_sha256: str | None = None,
    expected_model: str | None = None,
    expected_revision: int | None = None,
    expected_person_plan_ids: Sequence[str] | None = None,
    expected_scene_plan_ids: Sequence[str] | None = None,
) -> ReplacementPackDTO:
    root = _root(project_root)
    final = _read_json(root, PACK_RECEIPT_PATH)
    if set(final) != {
        "schema", "version", "candidate_receipt_path", "candidate_sha256",
        "quality_receipt_path", "quality_sha256", "sha256",
    } or final["schema"] != PACK_SCHEMA or final["version"] != 1:
        raise ReplacementPackError("published pack receipt is invalid")
    unsigned = dict(final)
    if unsigned.pop("sha256") != _hash(unsigned):
        raise ReplacementPackError("published pack receipt sha256 mismatch")
    if final["candidate_receipt_path"] != CANDIDATE_RECEIPT_PATH or final["quality_receipt_path"] != QUALITY_RECEIPT_PATH:
        raise ReplacementPackError("published pack path binding mismatch")
    candidate = load_replacement_pack_candidate(root)
    quality = _read_json(root, QUALITY_RECEIPT_PATH)
    result = PackQualityResult(quality.get("status"), quality.get("publishable"), quality)
    if (
        final["candidate_sha256"] != candidate.candidate_sha256
        or final["quality_sha256"] != quality.get("sha256")
        or not _quality_bound(candidate, result) or result.status != "pass"
    ):
        raise ReplacementPackError("published pack quality binding mismatch")
    values = {
        "upstream_plan_sha256": expected_upstream_plan_sha256,
        "upstream_source_inventory_sha256": expected_upstream_source_inventory_sha256,
        "execution_profile_sha256": expected_execution_profile_sha256,
        "model": expected_model, "revision": expected_revision,
    }
    for field, expected in values.items():
        if expected is not None and getattr(candidate, field) != expected:
            raise ReplacementPackError(f"expected {field} mismatch")
    if expected_person_plan_ids is not None and tuple(candidate.people) != tuple(expected_person_plan_ids):
        raise ReplacementPackError("expected person plan ids mismatch")
    if expected_scene_plan_ids is not None and tuple(candidate.scenes) != tuple(expected_scene_plan_ids):
        raise ReplacementPackError("expected scene plan ids mismatch")
    fields = {**candidate.__dict__, "schema": PACK_SCHEMA, "receipt_path": PACK_RECEIPT_PATH}
    return ReplacementPackDTO(
        **fields, quality_receipt_path=QUALITY_RECEIPT_PATH,
        quality_sha256=quality["sha256"], receipt_sha256=final["sha256"],
    )


def _lock(root: Path) -> int:
    directory = _directory(root, "work/replacement-packs")
    path = directory / ".lock"
    if path.is_symlink():
        raise ReplacementPackError("pack lock must not be a symlink")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


async def prepare_replacement_packs(
    project_root: Path,
    plan: ProjectReplacementPlan,
    *,
    model: str,
    revision: int,
    generator: PackGenerator,
    quality_gate: PackQualityGate,
) -> PackBuildResult:
    root = _root(project_root)
    requests = _requests(root, plan, model, revision)
    expected_input = _hash([request.input_sha256 for request in requests])
    descriptor = await asyncio.to_thread(_lock, root)
    try:
        if (root / PACK_RECEIPT_PATH).exists() or (root / PACK_RECEIPT_PATH).is_symlink():
            try:
                existing = _read_json(root, CANDIDATE_RECEIPT_PATH)
                if existing.get("input_sha256") == expected_input:
                    return PackBuildResult("ready", load_replacement_pack(root))
            except ReplacementPackError:
                raise
        generated = []
        for request in requests:
            try:
                generated.append(await _entity(root, request, generator))
            except PackGenerationError as error:
                status: BuildStatus = (
                    "submission_unknown" if error.code == "submission_unknown"
                    else "unknown" if error.code.endswith("_unknown") else "failed"
                )
                return PackBuildResult(status, issues=(error.code,))
        candidate_raw = _candidate_receipt(requests, generated)
        _write_json(root / CANDIDATE_RECEIPT_PATH, candidate_raw)
        candidate = _candidate_dto(root, candidate_raw)
        try:
            quality = quality_gate.evaluate(candidate, receipt_path=root / QUALITY_RECEIPT_PATH)
        except Exception:
            return PackBuildResult("unknown", issues=("quality_unknown",))
        if not _quality_bound(candidate, quality):
            return PackBuildResult("unknown", issues=("quality_receipt_invalid",))
        quality_raw = _copy_json(dict(quality.receipt))
        _write_json(root / QUALITY_RECEIPT_PATH, quality_raw)
        if quality.status != "pass":
            status = "unknown" if quality.status == "unknown" else "failed"
            return PackBuildResult(status, issues=(f"quality_{quality.status}",))
        final = {
            "schema": PACK_SCHEMA, "version": 1,
            "candidate_receipt_path": CANDIDATE_RECEIPT_PATH,
            "candidate_sha256": candidate.candidate_sha256,
            "quality_receipt_path": QUALITY_RECEIPT_PATH,
            "quality_sha256": quality_raw["sha256"],
        }
        final["sha256"] = _hash(final)
        _write_json(root / PACK_RECEIPT_PATH, final)
        return PackBuildResult("ready", load_replacement_pack(root))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


async def prepare_replacement_packs_with_seedream(
    project_root: Path,
    plan: ProjectReplacementPlan,
    *,
    settings: Any,
    revision: int,
    quality_gate: PackQualityGate,
    transport: Any = None,
) -> PackBuildResult:
    """Executor-ready entrypoint; tests inject an HTTP transport, never real calls."""
    return await prepare_replacement_packs(
        project_root,
        plan,
        model=settings.seedream_model,
        revision=revision,
        generator=SeedreamPackGenerator(project_root, settings, transport=transport),
        quality_gate=quality_gate,
    )


prepare_replacement_pack = prepare_replacement_packs
