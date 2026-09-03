from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
PUBLIC_STATUSES = frozenset({"queued", "running", "succeeded", "failed"})
LEGACY_STATUS_ALIASES = {"processing": "running", "done": "succeeded"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStateError(RuntimeError):
    """Persisted project state is structurally unsafe to expose or update."""


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

    def create(self, effective_request: dict[str, Any], client_request_id: str) -> tuple[str, Path, dict[str, Any]]:
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
        progress_status = progress["status"]
        if status != progress_status:
            raise ProjectStateError("top-level and progress statuses disagree")
        if progress_status == "queued" and percent != 0:
            raise ProjectStateError("queued progress must be zero")
        if progress_status == "running" and not 1 <= percent <= 99:
            raise ProjectStateError("running progress must be between 1 and 99")
        if progress_status == "succeeded" and percent != 100:
            raise ProjectStateError("succeeded progress must be 100")
        if progress_status == "failed" and percent > 99:
            raise ProjectStateError("failed progress must be below 100")
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
        status = normalized.get("status")
        if status in LEGACY_STATUS_ALIASES:
            normalized["status"] = LEGACY_STATUS_ALIASES[status]
        progress = normalized.get("project_progress")
        if isinstance(progress, dict) and progress.get("status") in PUBLIC_STATUSES:
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

    def _failed_recovery_state(self, project_id: str, previous: Any, *, code: str, message: str) -> dict[str, Any]:
        old = previous if isinstance(previous, dict) else {}
        now = utcnow()
        raw_percent = (old.get("project_progress") or {}).get("percent") if isinstance(old.get("project_progress"), dict) else 0
        percent = raw_percent if isinstance(raw_percent, int) and not isinstance(raw_percent, bool) else 0
        percent = min(99, max(0, percent))
        source = self.path(project_id) / "inputs" / "source.mp4"
        replacement = self.path(project_id) / "inputs" / "replacement-image"
        has_source = source.is_file()
        # A corrupt state cannot prove whether the fixed source file came from
        # an upload or URL.  Preserve no guessed provenance.
        source_snapshot = {"mode": "link", "reference_url": None}
        return {
            "id": project_id,
            "title": old.get("title") if isinstance(old.get("title"), str) else f"视频改编 {project_id[:8]}",
            "status": "failed",
            "created_at": old.get("created_at") if isinstance(old.get("created_at"), str) else now,
            "updated_at": now,
            "duration_s": old.get("duration_s") if isinstance(old.get("duration_s"), (int, float)) else 0,
            "segment_count": old.get("segment_count") if isinstance(old.get("segment_count"), int) else 0,
            "has_source": has_source,
            "has_video": False,
            "effective_request": old.get("effective_request") if isinstance(old.get("effective_request"), dict) else {},
            "client_request_id": old.get("client_request_id") if isinstance(old.get("client_request_id"), str) else "",
            "source_path": "inputs/source.mp4" if has_source else None,
            "replacement_image_path": "inputs/replacement-image" if replacement.is_file() else None,
            "project_progress": {"status": "failed", "percent": percent, "stage": code},
            "generation": {"status": "failed", "stage": code, "error": code},
            "creation_input": {"version": 1, "source": source_snapshot, "replacement_image": None},
            "error": {"code": code, "message": message},
        }

    def mark_failed(self, project_id: str, *, code: str, message: str, stage: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked(project_id)
            progress = state.get("project_progress") or {}
            raw_percent = progress.get("percent")
            percent = raw_percent if isinstance(raw_percent, int) and not isinstance(raw_percent, bool) else 0
            percent = min(99, max(0, percent))
            failure_stage = stage or code
            return self.update(
                project_id,
                status="failed",
                project_progress={"status": "failed", "percent": percent, "stage": failure_stage},
                generation={"status": "failed", "stage": failure_stage, "error": code},
                error={"code": code, "message": message},
            )

    def fail_incomplete_after_restart(self) -> None:
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
                    repaired = self._failed_recovery_state(
                        project.name,
                        raw,
                        code="invalid_state",
                        message="项目状态损坏，已停止自动执行",
                    )
                    self.write(project.name, repaired)
                    continue
                if state["status"] in {"queued", "running"}:
                    self.mark_failed(
                        project.name,
                        code="process_restarted",
                        message="后端重启，研究任务未恢复",
                    )
