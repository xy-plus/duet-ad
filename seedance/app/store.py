from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# awaiting_prompt_review：创建时打开提示词审核开关的项目在 H3 冻结后暂停、
# 等待用户确认提示词。它不是 queued/running，重启恢复
# （fail_incomplete_after_restart/auto-resume）天然不动它，只能由
# prompt-review/confirm 显式继续（等价 resume-picsart 入口）。
PUBLIC_STATUSES = frozenset(
    {"queued", "running", "succeeded", "failed", "awaiting_prompt_review"}
)
PROJECT_STATE_SCHEMA = "duet.project-state"
PROJECT_STATE_VERSION = 1
IDEMPOTENCY_INDEX_VERSION = 1
IDEMPOTENCY_INDEX_NAME = ".client-request-index.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStateError(RuntimeError):
    """Persisted project state is structurally unsafe to expose or update."""


class ClientRequestIdConflict(RuntimeError):
    """A client request id is already bound to different immutable inputs."""


class IdempotencyIndexUnavailable(RuntimeError):
    """The durable create claim cannot be trusted, so creation must fail closed."""


def generation_request_sha256(value: Any) -> str:
    """The optional off switches preserve the fingerprint of legacy requests."""
    if isinstance(value, dict):
        value = dict(value)
        if value.get("add_subtitles") is False:
            value.pop("add_subtitles")
        if value.get("prompt_review") is False:
            value.pop("prompt_review")
    return canonical_json_sha256(value)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def creation_request_digest(
    generation_request_sha256: str,
    source_sha256: str,
    replacement_image_sha256: str | None,
) -> str:
    """Hash the immutable v1 idempotency tuple without retaining input data."""

    for digest in (generation_request_sha256, source_sha256):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ValueError("invalid creation input digest")
    if (
        replacement_image_sha256 is not None
        and (
            not isinstance(replacement_image_sha256, str)
            or SHA256_RE.fullmatch(replacement_image_sha256) is None
        )
    ):
        raise ValueError("invalid creation input digest")
    return canonical_json_sha256(
        [
            generation_request_sha256,
            source_sha256,
            replacement_image_sha256,
        ]
    )


def replacement_receipts(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered original references, with the v1 first-image projection intact."""
    first = receipt.get("replacement_image")
    values = receipt.get("replacement_images", [] if first is None else [first])
    if not isinstance(values, list) or len(values) > 4 or (values[0] if values else None) != first:
        raise ValueError("invalid replacement image receipts")
    for item in values:
        if (not isinstance(item, dict) or set(item) != {"sha256", "bytes"}
                or not isinstance(item["sha256"], str)
                or SHA256_RE.fullmatch(item["sha256"]) is None
                or type(item["bytes"]) is not int or item["bytes"] < 0):
            raise ValueError("invalid replacement image receipt")
    return values


def replacement_receipt_digest(receipt: dict[str, Any]) -> str | None:
    values = replacement_receipts(receipt)
    if len(values) < 2:
        return values[0]["sha256"] if values else None
    return canonical_json_sha256([item["sha256"] for item in values])


def replacement_input_paths(count: int) -> list[str]:
    return ["inputs/replacement-image" if index == 1 else f"inputs/replacement-image-{index}"
            for index in range(1, count + 1)]


class ProjectStore:
    """Atomic, single-process filesystem storage for research projects.

    Signed provider assets are process-local, so overlapping application workers
    cannot safely share this store.  The advisory lock makes that deployment
    constraint executable instead of relying on operator convention.
    """

    def __init__(self, root: Path, *, acquire_process_lock: bool = True) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.idempotency_index_path = self.root / IDEMPOTENCY_INDEX_NAME
        self._lock = threading.RLock()
        self._process_lock_fd: int | None = None
        if acquire_process_lock:
            self._acquire_process_lock()

    def _acquire_process_lock(self) -> None:
        lock_path = self.root / ".process.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError(
                f"project store is already owned by another application worker: {self.root}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        self._process_lock_fd = descriptor

    def close(self) -> None:
        descriptor = self._process_lock_fd
        if descriptor is None:
            return
        self._process_lock_fd = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _file_receipt(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
        return total, digest.hexdigest()

    @staticmethod
    def _validated_receipt_digest(
        receipt: Any,
        *,
        client_request_id: str,
    ) -> str:
        if (
            not isinstance(receipt, dict)
            or set(receipt) != ({
                "version",
                "client_request_id",
                "generation_request_sha256",
                "source",
                "replacement_image",
            } | ({"replacement_images"} if receipt.get("version") == 2 else set()))
            or type(receipt.get("version")) is not int
            or receipt.get("version") not in (1, 2)
        ):
            raise IdempotencyIndexUnavailable("project input receipt is missing or invalid")
        if receipt.get("client_request_id") != client_request_id:
            raise IdempotencyIndexUnavailable("project input receipt belongs to another request")
        generation_digest = receipt.get("generation_request_sha256")
        source = receipt.get("source")
        replacement = receipt.get("replacement_image")
        if not isinstance(source, dict) or set(source) != {"sha256", "bytes"}:
            raise IdempotencyIndexUnavailable("project source receipt is missing")
        source_digest = source.get("sha256")
        source_bytes = source.get("bytes")
        if (
            not isinstance(source_digest, str)
            or SHA256_RE.fullmatch(source_digest) is None
            or isinstance(source_bytes, bool)
            or not isinstance(source_bytes, int)
            or source_bytes < 0
        ):
            raise IdempotencyIndexUnavailable("project source receipt is invalid")
        replacement_digest: str | None
        if replacement is None:
            replacement_digest = None
        elif isinstance(replacement, dict) and set(replacement) == {"sha256", "bytes"}:
            replacement_digest = replacement.get("sha256")
            replacement_bytes = replacement.get("bytes")
            if (
                not isinstance(replacement_digest, str)
                or SHA256_RE.fullmatch(replacement_digest) is None
                or isinstance(replacement_bytes, bool)
                or not isinstance(replacement_bytes, int)
                or replacement_bytes < 0
            ):
                raise IdempotencyIndexUnavailable("project replacement receipt is invalid")
        else:
            raise IdempotencyIndexUnavailable("project replacement receipt is invalid")
        try:
            values = replacement_receipts(receipt)
            if receipt["version"] == 2 and len(values) < 2:
                raise ValueError("multi-image receipt requires multiple images")
            return creation_request_digest(
                generation_digest,
                source_digest,
                replacement_receipt_digest(receipt),
            )
        except ValueError as exc:
            raise IdempotencyIndexUnavailable(
                "project generation request receipt is invalid"
            ) from exc

    def _has_persisted_idempotent_project_unlocked(self) -> bool:
        """Distinguish a fresh store from a lost index after prior commits."""

        for project in self.root.iterdir():
            if not project.is_dir() or PROJECT_ID_RE.fullmatch(project.name) is None:
                continue
            try:
                raw = json.loads((project / "state.json").read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            receipt = raw.get("input_receipt") if isinstance(raw, dict) else None
            if (
                isinstance(receipt, dict)
                and isinstance(receipt.get("client_request_id"), str)
                and receipt["client_request_id"]
            ):
                return True
        return False

    def _read_idempotency_index_unlocked(self) -> dict[str, Any]:
        try:
            descriptor = os.open(
                self.idempotency_index_path,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError:
            if self._has_persisted_idempotent_project_unlocked():
                raise IdempotencyIndexUnavailable(
                    "idempotency index is missing after committed projects"
                ) from None
            return {"version": IDEMPOTENCY_INDEX_VERSION, "entries": {}}
        except OSError as exc:
            raise IdempotencyIndexUnavailable("idempotency index cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise IdempotencyIndexUnavailable("idempotency index is not a regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
                descriptor = -1
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IdempotencyIndexUnavailable("idempotency index is not valid JSON") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "entries"}
            or type(value.get("version")) is not int
            or value.get("version") != IDEMPOTENCY_INDEX_VERSION
            or not isinstance(value.get("entries"), dict)
        ):
            raise IdempotencyIndexUnavailable("idempotency index schema is invalid")
        project_ids: set[str] = set()
        for key_digest, entry in value["entries"].items():
            if (
                not isinstance(key_digest, str)
                or SHA256_RE.fullmatch(key_digest) is None
                or not isinstance(entry, dict)
                or set(entry) != {"project_id", "request_digest"}
                or not isinstance(entry.get("project_id"), str)
                or PROJECT_ID_RE.fullmatch(entry["project_id"]) is None
                or not isinstance(entry.get("request_digest"), str)
                or SHA256_RE.fullmatch(entry["request_digest"]) is None
                or entry["project_id"] in project_ids
            ):
                raise IdempotencyIndexUnavailable("idempotency index entry is invalid")
            project_ids.add(entry["project_id"])
        return value

    def _write_idempotency_index_unlocked(self, value: dict[str, Any]) -> None:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = self.idempotency_index_path.with_name(
            f".{self.idempotency_index_path.name}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.idempotency_index_path)
            self._fsync_directory(self.root)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _read_claimed_project_unlocked(
        self,
        *,
        project_id: str,
        client_request_id: str,
        request_digest: str,
    ) -> tuple[Path, dict[str, Any]]:
        try:
            state = self._read_unlocked(project_id)
        except (FileNotFoundError, OSError, ProjectStateError) as exc:
            raise IdempotencyIndexUnavailable(
                "idempotency claim does not reference a readable project"
            ) from exc
        if state.get("client_request_id") != client_request_id:
            raise IdempotencyIndexUnavailable(
                "idempotency claim and project request id disagree"
            )
        receipt = state.get("input_receipt")
        receipt_digest = self._validated_receipt_digest(
            receipt,
            client_request_id=client_request_id,
        )
        if receipt_digest != request_digest:
            raise IdempotencyIndexUnavailable(
                "idempotency claim and project receipt disagree"
            )
        if generation_request_sha256(state.get("effective_request")) != receipt.get(
            "generation_request_sha256"
        ):
            raise IdempotencyIndexUnavailable(
                "project effective request no longer matches its receipt"
            )
        if state.get("has_source") is not True or state.get("source_path") != "inputs/source.mp4":
            raise IdempotencyIndexUnavailable("claimed project source is incomplete")
        project = self.path(project_id)
        try:
            source_path = self.resolve_project_file(
                project_id,
                project / "inputs" / "source.mp4",
            )
            source_bytes, source_digest = self._file_receipt(source_path)
        except (FileNotFoundError, OSError) as exc:
            raise IdempotencyIndexUnavailable("claimed project source is unreadable") from exc
        if (
            source_bytes != receipt["source"]["bytes"]
            or source_digest != receipt["source"]["sha256"]
        ):
            raise IdempotencyIndexUnavailable("claimed project source does not match its receipt")
        replacement_receipt = receipt.get("replacement_image")
        if replacement_receipt is None:
            if state.get("replacement_image_path") is not None:
                raise IdempotencyIndexUnavailable(
                    "claimed project replacement state is inconsistent"
                )
        else:
            if state.get("replacement_image_path") != "inputs/replacement-image":
                raise IdempotencyIndexUnavailable(
                    "claimed project replacement image is incomplete"
                )
            try:
                replacement_path = self.resolve_project_file(
                    project_id,
                    project / "inputs" / "replacement-image",
                )
                replacement_bytes, replacement_digest = self._file_receipt(replacement_path)
            except (FileNotFoundError, OSError) as exc:
                raise IdempotencyIndexUnavailable(
                    "claimed project replacement image is unreadable"
                ) from exc
            if (
                replacement_bytes != replacement_receipt["bytes"]
                or replacement_digest != replacement_receipt["sha256"]
            ):
                raise IdempotencyIndexUnavailable(
                    "claimed project replacement image does not match its receipt"
                )
        receipts = replacement_receipts(receipt)
        if "replacement_image_paths" in state or len(receipts) > 1:
            if state.get("replacement_image_paths") != replacement_input_paths(len(receipts)):
                raise IdempotencyIndexUnavailable("claimed replacement image list is inconsistent")
            for relative, expected in zip(state["replacement_image_paths"], receipts):
                try:
                    path = self.resolve_project_file(project_id, project / relative)
                    size, digest = self._file_receipt(path)
                except (FileNotFoundError, OSError) as exc:
                    raise IdempotencyIndexUnavailable("claimed replacement image is unreadable") from exc
                if size != expected["bytes"] or digest != expected["sha256"]:
                    raise IdempotencyIndexUnavailable("claimed replacement image does not match receipt")
        return project, state

    def lookup_claim(self, client_request_id: str) -> str | None:
        """Return the project id durably claimed for a client request id."""

        with self._lock:
            index = self._read_idempotency_index_unlocked()
            key_digest = hashlib.sha256(client_request_id.encode("utf-8")).hexdigest()
            entry = index["entries"].get(key_digest)
            return entry["project_id"] if entry is not None else None

    def create_from_staging(
        self,
        effective_request: dict[str, Any],
        client_request_id: str,
        *,
        request_digest: str,
        input_receipt: dict[str, Any],
        source_snapshot: dict[str, Any],
        staged_source: Path,
        replacement_snapshot: dict[str, Any] | None,
        staged_replacement: Path | None,
        replacement_snapshots: list[dict[str, Any]] | None = None,
        staged_replacements: list[Path] | None = None,
        project_id: str | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> tuple[str, Path, dict[str, Any], bool]:
        """Commit staged inputs and a durable idempotency claim as one create step.

        The claim is fsynced before a project is materialized.  A crash can
        therefore leave a claim that returns an availability error, but it can
        never make a retry create a second project for the same request id.
        """

        if not isinstance(client_request_id, str):
            raise TypeError("client_request_id must be a string")
        if not client_request_id.strip():
            raise ValueError("client_request_id must be nonempty")
        if not isinstance(request_digest, str) or SHA256_RE.fullmatch(request_digest) is None:
            raise ValueError("invalid request digest")
        if project_id is not None and (
            not isinstance(project_id, str)
            or PROJECT_ID_RE.fullmatch(project_id) is None
        ):
            raise ValueError("invalid caller-supplied project id")
        if extra_state is not None:
            reserved_keys = {
                "schema", "version", "id", "status", "created_at", "updated_at",
                "effective_request", "client_request_id", "input_receipt",
            }
            if not isinstance(extra_state, dict) or set(extra_state) & reserved_keys:
                raise ValueError("extra state must not override core project fields")
        receipt_digest = self._validated_receipt_digest(
            input_receipt,
            client_request_id=client_request_id,
        )
        if receipt_digest != request_digest:
            raise ValueError("request digest does not match input receipt")
        if generation_request_sha256(effective_request) != input_receipt.get(
            "generation_request_sha256"
        ):
            raise ValueError("effective request does not match input receipt")
        try:
            staged_source_resolved = staged_source.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError("staged source is missing") from exc
        if self.root not in staged_source_resolved.parents or not staged_source_resolved.is_file():
            raise ValueError("staged source must be a regular file inside the store")
        source_bytes, source_digest = self._file_receipt(staged_source_resolved)
        if (
            source_bytes != input_receipt["source"]["bytes"]
            or source_digest != input_receipt["source"]["sha256"]
        ):
            raise ValueError("staged source does not match input receipt")
        replacement_receipt = input_receipt.get("replacement_image")
        staged_replacement_resolved: Path | None = None
        if replacement_receipt is None:
            if staged_replacement is not None or replacement_snapshot is not None:
                raise ValueError("unexpected staged replacement image")
        else:
            if staged_replacement is None or replacement_snapshot is None:
                raise ValueError("staged replacement image is missing")
            try:
                staged_replacement_resolved = staged_replacement.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise ValueError("staged replacement image is missing") from exc
            if (
                self.root not in staged_replacement_resolved.parents
                or not staged_replacement_resolved.is_file()
            ):
                raise ValueError(
                    "staged replacement image must be a regular file inside the store"
                )
            replacement_bytes, replacement_digest = self._file_receipt(
                staged_replacement_resolved
            )
            if (
                replacement_bytes != replacement_receipt["bytes"]
                or replacement_digest != replacement_receipt["sha256"]
            ):
                raise ValueError("staged replacement image does not match input receipt")

        receipts = replacement_receipts(input_receipt)
        snapshots = replacement_snapshots if replacement_snapshots is not None else (
            [replacement_snapshot] if replacement_snapshot is not None else [])
        paths = staged_replacements if staged_replacements is not None else (
            [staged_replacement] if staged_replacement is not None else [])
        if len(paths) != len(receipts) or len(snapshots) != len(receipts):
            raise ValueError("replacement input count does not match receipt")
        if (paths[0] if paths else None) != staged_replacement or (snapshots[0] if snapshots else None) != replacement_snapshot:
            raise ValueError("replacement projection does not match image list")
        resolved_replacements = []
        for path, expected in zip(paths, receipts):
            resolved = path.resolve(strict=True)
            if self.root not in resolved.parents or not resolved.is_file():
                raise ValueError("staged replacement must be a file inside the store")
            size, digest = self._file_receipt(resolved)
            if size != expected["bytes"] or digest != expected["sha256"]:
                raise ValueError("staged replacement does not match receipt")
            resolved_replacements.append(resolved)

        with self._lock:
            index = self._read_idempotency_index_unlocked()
            key_digest = hashlib.sha256(client_request_id.encode("utf-8")).hexdigest()
            existing = index["entries"].get(key_digest)
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise ClientRequestIdConflict(client_request_id)
                project, state = self._read_claimed_project_unlocked(
                    project_id=existing["project_id"],
                    client_request_id=client_request_id,
                    request_digest=request_digest,
                )
                return existing["project_id"], project, state, False

            if project_id is None:
                project_id = uuid.uuid4().hex
                while self.path(project_id).exists():
                    project_id = uuid.uuid4().hex
            elif self.path(project_id).exists():
                raise ValueError("caller-supplied project id already exists")
            index["entries"][key_digest] = {
                "project_id": project_id,
                "request_digest": request_digest,
            }
            self._write_idempotency_index_unlocked(index)

            project = self.path(project_id)
            project.mkdir(mode=0o700)
            os.chmod(project, 0o700)
            for name in ("inputs", "work", "output", "logs"):
                directory = project / name
                directory.mkdir(mode=0o700)
                os.chmod(directory, 0o700)
            self._fsync_directory(project)
            self._fsync_directory(self.root)

            input_dir = project / "inputs"
            source = input_dir / "source.mp4"
            os.replace(staged_source_resolved, source)
            os.chmod(source, 0o600)
            image_paths = replacement_input_paths(len(receipts))
            for staged_path, relative in zip(resolved_replacements, image_paths):
                destination = project / relative
                os.replace(staged_path, destination)
                os.chmod(destination, 0o600)
            self._fsync_directory(input_dir)

            now = utcnow()
            replacement_creation = (
                {
                    **replacement_snapshot,
                    "preview_url": (
                        f"/api/conversations/{project_id}/creation-input/replacement-image"
                    ),
                }
                if replacement_snapshot is not None
                else None
            )
            replacement_creations = [
                {**snapshot, "preview_url": f"/api/conversations/{project_id}/creation-input/replacement-images/{index}"}
                for index, snapshot in enumerate(snapshots, 1)
            ]
            state = {
                "schema": PROJECT_STATE_SCHEMA,
                "version": PROJECT_STATE_VERSION,
                "id": project_id,
                "title": f"视频改编 {project_id[:8]}",
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "duration_s": 0,
                "segment_count": 0,
                "has_source": True,
                "has_video": False,
                "effective_request": effective_request,
                "client_request_id": client_request_id,
                "input_receipt": input_receipt,
                "source_path": "inputs/source.mp4",
                "source_bytes": input_receipt["source"]["bytes"],
                "replacement_image_path": (
                    "inputs/replacement-image" if replacement_snapshot is not None else None
                ),
                "replacement_image_paths": image_paths,
                "project_progress": {
                    "status": "queued",
                    "percent": 0,
                    "stage": "queued",
                },
                "generation": {
                    "status": "queued",
                    "stage": "upstream",
                    "error": None,
                },
                "creation_input": {
                    "version": 1,
                    "source": source_snapshot,
                    "replacement_image": replacement_creation,
                    "replacement_images": replacement_creations,
                },
                "error": None,
            }
            if extra_state:
                state.update(extra_state)
            self.write(project_id, state)
            return project_id, project, state, True

    def create(self, effective_request: dict[str, Any], client_request_id: str) -> tuple[str, Path, dict[str, Any]]:
        if not isinstance(client_request_id, str):
            raise TypeError("client_request_id must be a string")
        if not client_request_id.strip():
            raise ValueError("client_request_id must be nonempty")
        project_id = uuid.uuid4().hex
        project = self.root / project_id
        project.mkdir(mode=0o700)
        os.chmod(project, 0o700)
        for name in ("inputs", "work", "output", "logs"):
            directory = project / name
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
        now = utcnow()
        state = {
            "schema": PROJECT_STATE_SCHEMA,
            "version": PROJECT_STATE_VERSION,
            "id": project_id,
            "title": f"视频改编 {project_id[:8]}",
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "duration_s": 0,
            "segment_count": 0,
            "has_source": False,
            "has_video": False,
            "effective_request": effective_request,
            "client_request_id": client_request_id,
            "project_progress": {"status": "queued", "percent": 0, "stage": "queued"},
            "generation": {"status": "queued", "stage": "upstream", "error": None},
            "creation_input": {"version": 1, "source": None, "replacement_image": None},
            "error": None,
        }
        self.write(project_id, state)
        return project_id, project, state

    def path(self, project_id: str) -> Path:
        if not isinstance(project_id, str) or PROJECT_ID_RE.fullmatch(project_id) is None:
            raise FileNotFoundError(project_id)
        return self.root / project_id

    def resolve_project_file(self, project_id: str, path: Path) -> Path:
        project = self.path(project_id).resolve(strict=True)
        candidate = path.resolve(strict=True)
        if project not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(path)
        return candidate

    @staticmethod
    def _validate_state(project_id: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ProjectStateError("state must be a JSON object")
        if (
            value.get("schema") != PROJECT_STATE_SCHEMA
            or type(value.get("version")) is not int
            or value.get("version") != PROJECT_STATE_VERSION
        ):
            raise ProjectStateError("unsupported project state schema")
        if value.get("id") != project_id:
            raise ProjectStateError("state id does not match its project directory")
        status = value.get("status")
        if status not in PUBLIC_STATUSES:
            raise ProjectStateError("invalid top-level project status")
        progress = value.get("project_progress")
        if not isinstance(progress, dict) or progress.get("status") not in PUBLIC_STATUSES:
            raise ProjectStateError("invalid project progress status")
        percent = progress.get("percent")
        if isinstance(percent, bool) or not isinstance(percent, int) or not 0 <= percent <= 100:
            raise ProjectStateError("invalid project progress percent")
        return value

    def _read_unlocked(self, project_id: str) -> dict[str, Any]:
        target = self.path(project_id) / "state.json"
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectStateError("state.json is not valid JSON") from exc
        return self._validate_state(project_id, value)

    def read(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            return self._read_unlocked(project_id)

    @staticmethod
    def _normalized_changes(changes: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(changes)
        progress = normalized.get("project_progress")
        if (
            "status" not in normalized
            and isinstance(progress, dict)
            and progress.get("status") in PUBLIC_STATUSES
        ):
            normalized["status"] = progress["status"]
        return normalized

    def write(self, project_id: str, value: dict[str, Any]) -> None:
        validated = self._validate_state(project_id, value)
        payload = json.dumps(validated, ensure_ascii=False, indent=2).encode("utf-8")
        with self._lock:
            target = self.path(project_id) / "state.json"
            temporary = target.with_name(".state.json.tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, target)
                directory_descriptor = os.open(
                    target.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except BaseException:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise

    def update(self, project_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked(project_id)
            state.update(self._normalized_changes(changes))
            state["updated_at"] = utcnow()
            self.write(project_id, state)
            return state

    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            for project in self.root.iterdir():
                if not project.is_dir() or PROJECT_ID_RE.fullmatch(project.name) is None:
                    continue
                try:
                    rows.append(self._read_unlocked(project.name))
                except (FileNotFoundError, OSError, ProjectStateError):
                    continue
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return rows

    def mark_failed(self, project_id: str, *, code: str, message: str, stage: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked(project_id)
            progress = state.get("project_progress") or {}
            raw_percent = progress.get("percent")
            percent = raw_percent if isinstance(raw_percent, int) and not isinstance(raw_percent, bool) else 0
            percent = min(99, max(0, percent))
            failure_stage = stage or code
            updated = self.update(
                project_id,
                status="failed",
                project_progress={"status": "failed", "percent": percent, "stage": failure_stage},
                generation={"status": "failed", "stage": failure_stage, "error": code},
                error={"code": code, "message": message},
            )
        self.write_boundary_snapshot(project_id)
        return updated

    def write_boundary_snapshot(self, project_id: str) -> Path | None:
        """把失败现场 state 原样固化到 logs/boundary-snapshots/。

        修法背景（2026-09-30）：harvest_node_fixtures 过去只在项目当前
        status=="failed" 时采 boundary 快照，项目续跑成功后失败现场永久
        丢失（案例 dfe64795）。这里在每次置 failed 后把刚落盘的 state
        逐字节复制为 <UTC时间戳>-<error_code>.json——多次失败各自保留、
        不覆盖，后续续跑也不会改写。快照只供离线 fixtures 采集；写入失败
        仅记 stderr，不阻断业务（与 research log 同纪律）。
        """
        project = self.root / project_id
        state_path = project / "state.json"
        try:
            state_bytes = state_path.read_bytes()
            state = json.loads(state_bytes)
            error = state.get("error")
            code = (
                error.get("code")
                if isinstance(error, dict)
                else error if isinstance(error, str) else None
            )
            code = str(code or "unknown")
            safe_code = "".join(
                character if character.isalnum() or character in "_-" else "-"
                for character in code
            )[:80]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination_dir = project / "logs" / "boundary-snapshots"
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{stamp}-{safe_code}.json"
            destination.write_bytes(state_bytes)
            return destination
        except (OSError, ValueError) as exc:
            print(
                f"boundary snapshot write failed for {project_id}: {exc}",
                file=sys.stderr,
            )
            return None

    @staticmethod
    def _submit_accepted_evidence(project: Path, run_identity: str) -> bool:
        """True when the research log proves this submission was accepted."""
        submit_root = project / "logs" / f"{run_identity}-submit"
        for output in submit_root.glob("*/attempt-*/output.json"):
            try:
                logged = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            record = (
                logged.get("provider_record") if isinstance(logged, dict) else None
            )
            if (
                isinstance(record, dict)
                and record.get("outcome") == "accepted"
                and isinstance(record.get("task_id"), str)
                and record["task_id"]
            ):
                return True
        return False

    def _downgrade_unproven_submissions(self, project_id: str) -> list[dict[str, Any]]:
        """Downgrade submission_started receipts without accepted-submit log
        evidence to rejected, so a restart cannot lock the resume gate."""
        project = self.root / project_id
        state = self._read_unlocked(project_id)
        ledger = state.get("seedance_submission_receipts")
        if not isinstance(ledger, dict):
            return []
        segments = ledger.get("segments")
        if not isinstance(segments, dict):
            return []
        downgraded: list[dict[str, Any]] = []
        changed = False
        for records in segments.values():
            if not isinstance(records, list):
                continue
            for receipt in records:
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("status") != "submission_started"
                ):
                    continue
                run_identity = receipt.get("run_identity")
                if not isinstance(run_identity, str) or not run_identity:
                    continue
                if self._submit_accepted_evidence(project, run_identity):
                    continue
                receipt["status"] = "rejected"
                changed = True
                downgraded.append({
                    "segment_index": receipt.get("segment_index"),
                    "run_identity": run_identity,
                })
        if changed:
            self.update(project_id, seedance_submission_receipts=ledger)
        return downgraded

    def fail_incomplete_after_restart(self) -> list[dict[str, Any]]:
        restarted: list[dict[str, Any]] = []
        with self._lock:
            for project in self.root.iterdir():
                if not project.is_dir() or PROJECT_ID_RE.fullmatch(project.name) is None:
                    continue
                state_path = project / "state.json"
                raw: Any = None
                try:
                    raw = json.loads(state_path.read_text(encoding="utf-8"))
                    state = self._validate_state(project.name, raw)
                except (FileNotFoundError, OSError, json.JSONDecodeError, ProjectStateError):
                    continue
                if state["status"] in {"queued", "running"}:
                    downgraded = self._downgrade_unproven_submissions(project.name)
                    self.mark_failed(
                        project.name,
                        code="process_restarted",
                        message="后端重启，研究任务未恢复",
                    )
                    restarted.append({
                        "project_id": project.name,
                        "downgraded_receipts": downgraded,
                    })
        return restarted
