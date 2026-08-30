import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Callable, NamedTuple

from app import dialogue_review, generation_config as generation_config_contract

ALLOWED_EXT = {".mp4", ".mov", ".webm"}
_CHUNK = 1024 * 1024
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
# files 白名单：segments/<正整数N>/work/(keyframes|postprocessed)/<纯文件名>
_SEG_FILE_RE = re.compile(r"^([1-9]\d*)/work/(keyframes|postprocessed)/([^/]+)$")
_META_LOCKS: dict[str, threading.Lock] = {}
_META_LOCKS_GUARD = threading.Lock()
PROCESS_GENERATION = uuid.uuid4().hex
_PIPELINE_RETRY_ATTEMPT_CAP = 31


class UploadError(ValueError):
    """上传校验失败（HTTP 层转 422）。"""


class VideoProbe(NamedTuple):
    duration_s: float
    width: int
    height: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_title(filename: str) -> str:
    """原文件名仅作展示：去路径、去控制字符、限 80 字。"""
    base = _CONTROL_RE.sub("", filename.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    stem = Path(base).stem.strip()
    return stem[:80] or "untitled"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        raise OSError(f"cannot open durable directory: {path.name}") from None
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_meta(cdir: Path, meta: dict) -> None:
    payload = json.dumps(meta, ensure_ascii=False, indent=2)
    fd, temporary = tempfile.mkstemp(prefix=".meta-", suffix=".json", dir=cdir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cdir / "meta.json")
        _fsync_directory(cdir)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_receipt(path: Path, receipt: dict) -> None:
    payload = json.dumps(receipt, ensure_ascii=False, indent=2)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _meta_lock(cdir: Path):
    key = str(cdir.resolve())
    with _META_LOCKS_GUARD:
        lock = _META_LOCKS.setdefault(key, threading.Lock())
    with lock:
        yield


def new_conversation(data_dir: Path, note: str, orig_name: str, client_request_id: str = "",
                     voice_mode: str = "keep", target_language: str = "",
                     generation_config: dict | None = None,
                     dialogue_review_policy: str = dialogue_review.AUTO_CONTINUE,
                     dialogue_mode: str = "auto") -> dict:
    cid = uuid.uuid4().hex
    cdir = data_dir / cid
    data_dir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir()
    _fsync_directory(data_dir)
    (cdir / "work").mkdir()
    _fsync_directory(cdir)
    now = _now()
    meta = {
        "schema_version": 2,
        "id": cid,
        "title": note or sanitize_title(orig_name),
        "note": note,
        "status": "queued",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "keyframes": [],
        "prompt": None,
        "voice_mode": voice_mode,
        "duration_s": None,
        "fit_required": None,
        "dialogue_mode": dialogue_mode,
        "dialogue_review_policy": dialogue_review_policy,
        "generation": None,
    }
    if client_request_id:
        meta["client_request_id"] = client_request_id
    if target_language:
        meta["target_language"] = target_language
    if generation_config is not None:
        generation_config_receipt = generation_config_contract.receipt(
            generation_config
        )
        meta["generation_config"] = dict(generation_config)
        meta["generation_config_sha256"] = generation_config_receipt[
            "generation_config_sha256"
        ]
        receipt_path = cdir / "work" / "generation-config.json"
        _write_receipt(receipt_path, generation_config_receipt)
    _write_meta(cdir, meta)
    return meta


def update_meta(data_dir: Path, cid: str, **changes) -> dict | None:
    """合并写字段并刷新 updated_at；cid 非法或不存在返回 None。"""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = _load_meta_unlocked(data_dir, cid)
        if meta is None:
            return None
        meta.update(changes)
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def mutate_meta(
    data_dir: Path, cid: str, mutator: Callable[[dict], None]
) -> dict | None:
    """Atomically read-modify-write one conversation in the current process.

    ``mutator`` must be synchronous and must not call another storage mutation.
    If it raises, no bytes are written.  This matches the repository's existing
    process-local meta lock model while preventing callers from replacing a
    nested field using a stale snapshot.
    """
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = _load_meta_unlocked(data_dir, cid)
        if meta is None:
            return None
        mutator(meta)
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def _frozen_input_snapshot(cdir: Path, *, include_prompts: bool) -> dict[str, str]:
    patterns = [
        "prepared_input.json",
        "long_video_plan.json",
        "work/**/h3_frames/**/*",
    ]
    if include_prompts:
        patterns.append("work/**/prompt.txt")
    paths = {
        path
        for pattern in patterns
        for path in cdir.glob(pattern)
        if path.is_file()
    }
    return {
        path.relative_to(cdir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _input_owner(cdir: Path, kind: str, request_id: str | None = None) -> dict:
    owner = {
        "kind": kind,
        "process_generation": PROCESS_GENERATION,
        "frozen_input_snapshot": _frozen_input_snapshot(
            cdir, include_prompts=kind == "submit"
        ),
    }
    if request_id is not None:
        owner["request_id"] = request_id
    return owner


def _has_frozen_input(cdir: Path, meta: dict) -> bool:
    return bool(
        meta.get("generation") is not None
        or meta.get("prepared_input_receipt")
        or meta.get("long_video_plan_receipt")
        or (cdir / "prepared_input.json").exists()
        or (cdir / "long_video_plan.json").exists()
        or any(cdir.glob("work/**/h3_frames"))
    )


def _stale_owner_has_new_frozen_input(cdir: Path, meta: dict, owner: object) -> bool:
    if meta.get("generation") is not None:
        return True
    if not isinstance(owner, dict):
        return _has_frozen_input(cdir, meta)
    snapshot = owner.get("frozen_input_snapshot")
    if not isinstance(snapshot, dict):
        return _has_frozen_input(cdir, meta)
    return snapshot != _frozen_input_snapshot(
        cdir, include_prompts=owner.get("kind") == "submit"
    )


def claim_pipeline_input(data_dir: Path, cid: str) -> dict | None:
    """Atomically claim an unfrozen conversation for input preparation."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None:
            return None
        current = meta.get("_input_owner")
        if current:
            if (
                isinstance(current, dict)
                and current.get("process_generation") == PROCESS_GENERATION
            ) or _stale_owner_has_new_frozen_input(cdir, meta, current):
                return None
        elif _has_frozen_input(cdir, meta):
            return None
        meta.update(
            status="processing", error=None,
            _input_owner=_input_owner(cdir, "pipeline"),
        )
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def _ready_queued_pipeline_input(cdir: Path, meta: dict) -> bool:
    """Recognize a fully persisted upload that is safe to enqueue again."""
    duration = meta.get("duration_s")
    width = meta.get("source_width")
    height = meta.get("source_height")
    sources = [
        path
        for path in cdir.glob("source.*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXT
    ]
    retry = meta.get("_pipeline_retry")
    retry_due = True
    if isinstance(retry, dict):
        not_before = retry.get("not_before")
        if isinstance(not_before, str):
            try:
                retry_due = datetime.fromisoformat(not_before) <= datetime.now(
                    timezone.utc
                )
            except (TypeError, ValueError):
                # A malformed private hint must not strand an otherwise valid
                # durable task. Claiming it rewrites the state on the next run.
                retry_due = True
    return bool(
        meta.get("status") == "queued"
        and not meta.get("_input_owner")
        and retry_due
        and not _has_frozen_input(cdir, meta)
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and isfinite(duration)
        and duration > 0
        and isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
        and isinstance(meta.get("dialogue_mode"), str)
        and len(sources) == 1
    )


def claim_ready_queued_pipeline_input(
    data_dir: Path, cid: str,
) -> dict | None:
    """Atomically claim one durable queued upload after a lost enqueue."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None or not _ready_queued_pipeline_input(cdir, meta):
            return None
        meta.update(
            status="processing",
            error=None,
            _input_owner=_input_owner(cdir, "pipeline"),
        )
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def claim_ready_queued_pipeline_inputs(
    data_dir: Path,
) -> list[tuple[str, dict]]:
    """Claim every complete queued upload left without a background task."""
    claimed = []
    for listed in list_conversations(data_dir):
        cid = listed.get("id")
        if not isinstance(cid, str) or isinstance(
            listed.get("_pipeline_retry"), dict
        ):
            continue
        meta = claim_ready_queued_pipeline_input(data_dir, cid)
        if meta is not None:
            claimed.append((cid, meta["_input_owner"]))
    return claimed


def claim_next_ready_queued_pipeline_input(
    data_dir: Path,
) -> tuple[str, dict] | None:
    """Claim at most one due durable item for a capacity-bounded dispatcher."""
    candidates = [
        listed
        for listed in list_conversations(data_dir)
        if isinstance(listed.get("_pipeline_retry"), dict)
    ]
    candidates.sort(
        key=lambda item: (
            str(item["_pipeline_retry"].get("not_before") or ""),
            str(item.get("created_at") or ""),
        )
    )
    for listed in candidates:
        cid = listed.get("id")
        if not isinstance(cid, str):
            continue
        meta = claim_ready_queued_pipeline_input(data_dir, cid)
        if meta is not None:
            return cid, meta["_input_owner"]
    return None


def prepare_incomplete_queued_pipeline_retry(
    data_dir: Path, cid: str,
) -> dict | None:
    """Reuse an unclaimed pre-enqueue CID and clear only its partial source."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if (
            meta is None
            or meta.get("status") != "queued"
            or meta.get("_input_owner")
            or _has_frozen_input(cdir, meta)
            or _ready_queued_pipeline_input(cdir, meta)
        ):
            return None
        for path in cdir.glob("source.*"):
            if path.is_file():
                path.unlink()
        meta.update(
            error=None,
            duration_s=None,
            source_width=None,
            source_height=None,
        )
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def fail_pipeline_input(
    data_dir: Path,
    cid: str,
    owner: object,
    *,
    error: str,
) -> dict | None:
    """Close a queued/current pipeline claim without overwriting another owner."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None:
            return None
        current = meta.get("_input_owner")
        if owner is not None:
            eligible = current == owner
        else:
            eligible = (
                meta.get("status") == "queued" and not current
            ) or (
                isinstance(current, dict)
                and current.get("kind") == "pipeline"
                and current.get("process_generation") == PROCESS_GENERATION
            )
        if not eligible:
            return None
        meta.update(status="failed", error=error, _input_owner=None)
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def requeue_pipeline_input(
    data_dir: Path,
    cid: str,
    owner: object,
    *,
    retry_delay_s: float,
    reason: str,
) -> dict | None:
    """CAS-release one pipeline lease back to the durable queue."""
    if (
        not _ID_RE.match(cid)
        or not isinstance(owner, dict)
        or owner.get("kind") != "pipeline"
        or not isinstance(retry_delay_s, (int, float))
        or isinstance(retry_delay_s, bool)
        or not isfinite(retry_delay_s)
        or retry_delay_s < 0
        or not isinstance(reason, str)
        or not reason
    ):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if (
            meta is None
            or meta.get("status") != "processing"
            or meta.get("_input_owner") != owner
        ):
            return None
        previous = meta.get("_pipeline_retry")
        previous_attempt = (
            previous.get("attempt", 0) if isinstance(previous, dict) else 0
        )
        if not isinstance(previous_attempt, int) or isinstance(
            previous_attempt, bool
        ) or previous_attempt < 0:
            previous_attempt = 0
        attempt = min(previous_attempt + 1, _PIPELINE_RETRY_ATTEMPT_CAP)
        not_before = datetime.now(timezone.utc) + timedelta(
            seconds=float(retry_delay_s)
        )
        meta.update(
            status="queued",
            error=None,
            _input_owner=None,
            _pipeline_retry={
                "attempt": attempt,
                "not_before": not_before.isoformat(),
                "reason": reason,
            },
        )
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def claim_stale_pipeline_inputs(data_dir: Path) -> list[tuple[str, dict]]:
    """Move recoverable pipeline leases from an older boot to this process."""
    claimed = []
    for listed in list_conversations(data_dir):
        cid = listed.get("id")
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            continue
        cdir = data_dir / cid
        with _meta_lock(cdir):
            meta = load_meta(data_dir, cid)
            if meta is None or meta.get("status") != "processing":
                continue
            current = meta.get("_input_owner")
            is_pipeline = current == "pipeline" or (
                isinstance(current, dict) and current.get("kind") == "pipeline"
            )
            if not is_pipeline:
                continue
            if (
                isinstance(current, dict)
                and current.get("process_generation") == PROCESS_GENERATION
            ):
                continue
            if _stale_owner_has_new_frozen_input(cdir, meta, current):
                continue
            owner = _input_owner(cdir, "pipeline")
            meta["_input_owner"] = owner
            meta["updated_at"] = _now()
            _write_meta(cdir, meta)
            claimed.append((cid, owner))
    return claimed


def claim_stale_input_reconciliations(data_dir: Path) -> list[tuple[str, dict]]:
    """Claim stale leases whose frozen bytes changed for local-only reconciliation."""
    claimed = []
    for listed in list_conversations(data_dir):
        cid = listed.get("id")
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            continue
        cdir = data_dir / cid
        with _meta_lock(cdir):
            meta = load_meta(data_dir, cid)
            if meta is None or meta.get("generation") is not None:
                continue
            current = meta.get("_input_owner")
            if current == "pipeline":
                kind, request_id = "pipeline", None
            elif isinstance(current, str) and current.startswith("submit:"):
                kind, request_id = "submit", current.removeprefix("submit:")
            elif isinstance(current, dict):
                kind, request_id = current.get("kind"), current.get("request_id")
            else:
                continue
            if kind not in {"pipeline", "submit"}:
                continue
            if (
                isinstance(current, dict)
                and current.get("process_generation") == PROCESS_GENERATION
            ):
                continue
            if not _stale_owner_has_new_frozen_input(cdir, meta, current):
                continue
            owner = (
                {**current, "process_generation": PROCESS_GENERATION}
                if isinstance(current, dict)
                else {
                    "kind": kind,
                    "process_generation": PROCESS_GENERATION,
                    "frozen_input_snapshot": None,
                    **({"request_id": request_id} if request_id is not None else {}),
                }
            )
            meta["_input_owner"] = owner
            meta["updated_at"] = _now()
            _write_meta(cdir, meta)
            claimed.append((cid, owner))
    return claimed


def load_pipeline_claim(data_dir: Path, cid: str, owner: object) -> dict | None:
    """Load an exact, currently owned pipeline claim."""
    if (
        not _ID_RE.match(cid)
        or not isinstance(owner, dict)
        or owner.get("kind") != "pipeline"
    ):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None or meta.get("_input_owner") != owner:
            return None
        return meta


def load_submission_claim(data_dir: Path, cid: str, owner: object) -> dict | None:
    """Load an exact, currently owned submission claim."""
    if (
        not _ID_RE.match(cid)
        or not isinstance(owner, dict)
        or owner.get("kind") != "submit"
    ):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None or meta.get("_input_owner") != owner:
            return None
        return meta


def claim_submission_input(data_dir: Path, cid: str, request_id: str) -> dict | None:
    """Atomically exclude pipeline work before a first submission freezes files."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None:
            return None
        current = meta.get("_input_owner")
        if current:
            if (
                isinstance(current, dict)
                and current.get("process_generation") == PROCESS_GENERATION
            ) or _stale_owner_has_new_frozen_input(cdir, meta, current):
                return None
        if meta.get("generation") is not None or meta.get("status") != "done":
            return None
        owner = _input_owner(cdir, "submit", request_id)
        meta["_input_owner"] = owner
        meta["error"] = None
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def finish_input_claim(
    data_dir: Path, cid: str, owner: object, **changes,
) -> dict | None:
    """Commit or release a claim only when its current owner still matches."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = load_meta(data_dir, cid)
        if meta is None or meta.get("_input_owner") != owner:
            return None
        meta.update(changes)
        if meta.get("_dialogue_review_continuation") == "running":
            meta.pop("_dialogue_review_continuation", None)
        meta["_input_owner"] = None
        if (
            isinstance(owner, dict)
            and owner.get("kind") == "pipeline"
            and changes.get("status") in {"done", "failed"}
        ):
            meta.pop("_pipeline_retry", None)
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def record_dialogue_analysis(
    data_dir: Path,
    cid: str,
    owner: object,
    *,
    policy: str,
    outcome: str,
    machine_lines: list[dict],
) -> dict | None:
    """Freeze ASR output or durably release the pipeline into review waiting."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = _load_meta_unlocked(data_dir, cid)
        if meta is None or meta.get("_input_owner") != owner:
            return None
        if meta.get("dialogue_review") is not None:
            return None
        review = dialogue_review.analysis_state(policy, outcome, machine_lines)
        meta["dialogue_review"] = review
        if review["status"] == "waiting":
            meta["_input_owner"] = None
            meta["status"] = "processing"
            meta["error"] = None
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def commit_dialogue_review(
    data_dir: Path,
    cid: str,
    *,
    request_id: str,
    expected_revision: int,
    expected_sha256: str,
    lines: list[dict],
) -> tuple[dict | None, bool]:
    """CAS-freeze one reviewed draft. Returns ``(meta, idempotent_replay)``."""
    if not _ID_RE.match(cid):
        return None, False
    cdir = data_dir / cid
    payload_sha256 = dialogue_review.commit_payload_sha256(
        expected_revision=expected_revision,
        expected_sha256=expected_sha256,
        lines=lines,
    )
    with _meta_lock(cdir):
        meta = _load_meta_unlocked(data_dir, cid)
        if meta is None:
            return None, False
        review = meta.get("dialogue_review")
        if not isinstance(review, dict):
            raise dialogue_review.DialogueReviewError("dialogue_review_unavailable")
        if review.get("status") == "frozen":
            if (
                review.get("_commit_request_id") == request_id
                and review.get("_commit_payload_sha256") == payload_sha256
            ):
                return meta, True
            raise dialogue_review.DialogueReviewError("dialogue_review_read_only")
        if review.get("status") != "waiting" or meta.get("_input_owner"):
            raise dialogue_review.DialogueReviewError("dialogue_review_not_waiting")
        if (
            review.get("revision") != expected_revision
            or review.get("sha256") != expected_sha256
        ):
            raise dialogue_review.DialogueReviewError("dialogue_review_conflict")
        normalized = dialogue_review.canonical_lines(lines)
        digest = dialogue_review.lines_sha256(normalized)
        machine_lines = dialogue_review.canonical_lines(
            review.get("machine_lines", [])
        )
        review.update(
            status="frozen",
            revision=expected_revision + 1,
            lines=normalized,
            sha256=digest,
            frozen_by="user",
            _commit_request_id=request_id,
            _commit_payload_sha256=payload_sha256,
        )
        if not normalized:
            meta["dialogue_mode"] = "none"
        elif normalized == machine_lines:
            meta["dialogue_mode"] = "auto"
        else:
            meta["dialogue_mode"] = "edit"
        meta["voice_lines"] = normalized
        meta["dialogue_review"] = review
        meta["_dialogue_review_continuation"] = "queued"
        meta["status"] = "processing"
        meta["error"] = None
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta, False


def claim_queued_dialogue_review_continuation(
    data_dir: Path, cid: str,
) -> dict | None:
    """Claim a committed review continuation exactly once in this boot."""
    if not _ID_RE.match(cid):
        return None
    cdir = data_dir / cid
    with _meta_lock(cdir):
        meta = _load_meta_unlocked(data_dir, cid)
        if (
            meta is None
            or meta.get("status") != "processing"
            or meta.get("_input_owner")
            or meta.get("_dialogue_review_continuation") != "queued"
            or not isinstance(meta.get("dialogue_review"), dict)
            or meta["dialogue_review"].get("status") != "frozen"
        ):
            return None
        owner = _input_owner(cdir, "pipeline")
        meta["_input_owner"] = owner
        meta["_dialogue_review_continuation"] = "running"
        meta["updated_at"] = _now()
        _write_meta(cdir, meta)
        return meta


def claim_queued_dialogue_review_continuations(
    data_dir: Path,
) -> list[tuple[str, dict]]:
    claimed = []
    for listed in list_conversations(data_dir):
        cid = listed.get("id")
        if not isinstance(cid, str):
            continue
        meta = claim_queued_dialogue_review_continuation(data_dir, cid)
        if meta is not None:
            claimed.append((cid, meta["_input_owner"]))
    return claimed


def load_meta(data_dir: Path, cid: str) -> dict | None:
    if not _ID_RE.match(cid):
        return None
    return _load_meta_unlocked(data_dir, cid)


def _load_meta_unlocked(data_dir: Path, cid: str) -> dict | None:
    p = data_dir / cid / "meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def list_conversations(data_dir: Path) -> list[dict]:
    if not data_dir.is_dir():
        return []
    metas = []
    for cdir in data_dir.iterdir():
        if cdir.is_dir() and _ID_RE.match(cdir.name) and (cdir / "meta.json").is_file():
            metas.append(json.loads((cdir / "meta.json").read_text()))
    metas.sort(key=lambda m: m["created_at"], reverse=True)
    return metas


def remove_conversation(data_dir: Path, cid: str) -> None:
    if _ID_RE.match(cid):
        shutil.rmtree(data_dir / cid, ignore_errors=True)


async def save_upload(cdir: Path, upload, max_bytes: int) -> Path:
    """流式落盘为 source.<ext>，超限即删并报错；不读进内存。"""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(f"unsupported extension: {ext or '(none)'}")
    dest = cdir / f"source{ext}"
    written = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await upload.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise UploadError(f"file exceeds {max_bytes} bytes")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest


def probe_video(path: Path) -> VideoProbe:
    """探测首个视频流的视觉时长与尺寸；容器/音频时长不参与。"""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", (
                    "stream=width,height,duration,duration_ts,time_base,"
                    "avg_frame_rate,r_frame_rate"
                ),
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise UploadError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        payload = json.loads(r.stdout)
        stream = payload["streams"][0]
        width, height = stream["width"], stream["height"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise UploadError("cannot parse video duration or dimensions") from e
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise UploadError("cannot parse video dimensions")
    duration: float | None = None
    try:
        raw_duration = stream.get("duration")
        if isinstance(raw_duration, bool):
            raise TypeError
        candidate = float(raw_duration)
        if isfinite(candidate) and candidate > 0:
            duration = candidate
    except (TypeError, ValueError):
        pass
    if duration is None:
        try:
            raw_duration_ts = stream.get("duration_ts")
            if isinstance(raw_duration_ts, bool):
                raise TypeError
            duration_ts = float(raw_duration_ts)
            numerator, denominator = str(stream.get("time_base")).split("/", 1)
            time_base = float(numerator) / float(denominator)
            candidate = duration_ts * time_base
            if isfinite(candidate) and candidate > 0:
                duration = candidate
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if duration is None:
        duration = _probe_packet_duration(path, stream)
    if duration is None:
        raise UploadError("cannot parse video duration")
    return VideoProbe(duration, width, height)


def _positive_rate(value: object) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        rate = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if isfinite(rate) and rate > 0 else None


def _packet_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _probe_packets(path: Path, selector: str) -> list[dict]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", selector,
                "-show_packets", "-show_entries",
                "packet=pts_time,dts_time,duration_time,side_data_list",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise UploadError(f"ffprobe failed: {exc}") from exc
    if result.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        packets = json.loads(result.stdout).get("packets")
    except (AttributeError, TypeError, json.JSONDecodeError):
        packets = None
    if not isinstance(packets, list):
        raise UploadError("cannot parse video packet timeline")
    return packets


def probe_stream_start_time(path: Path, selector: str) -> float:
    """Return a stream's presentation start; correct packet fallback codec priming."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", selector,
                "-show_entries", "stream=start_time,sample_rate", "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise UploadError(f"ffprobe failed: {exc}") from exc
    if result.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        streams = json.loads(result.stdout).get("streams")
    except (AttributeError, TypeError, json.JSONDecodeError):
        streams = None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise UploadError("cannot parse stream timeline")
    stream = streams[0]
    start_time = _packet_number(stream.get("start_time"))
    if start_time is not None:
        return start_time

    starts: list[tuple[float, dict]] = []
    for packet in _probe_packets(path, selector):
        if not isinstance(packet, dict):
            continue
        start = _packet_number(packet.get("pts_time"))
        if start is None:
            start = _packet_number(packet.get("dts_time"))
        if start is not None:
            starts.append((start, packet))
    if not starts:
        raise UploadError("cannot parse stream packet timeline")
    start, first_packet = min(starts, key=lambda item: item[0])
    if selector.startswith("a:"):
        sample_rate = _packet_number(stream.get("sample_rate"))
        side_data = first_packet.get("side_data_list")
        if isinstance(side_data, list):
            for item in side_data:
                if not isinstance(item, dict) or item.get("side_data_type") != "Skip Samples":
                    continue
                skip_samples = _packet_number(item.get("skip_samples"))
                if (
                    sample_rate is None
                    or sample_rate <= 0
                    or skip_samples is None
                    or skip_samples < 0
                ):
                    raise UploadError("cannot parse audio priming timeline")
                start += skip_samples / sample_rate
                break
    return start


def _probe_packet_duration(path: Path, stream: dict) -> float | None:
    packets = _probe_packets(path, "v:0")
    starts: list[float] = []
    explicit_ends: list[float] = []
    starts_with_duration: set[float] = set()
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        start = _packet_number(packet.get("pts_time"))
        if start is None:
            start = _packet_number(packet.get("dts_time"))
        if start is None:
            continue
        starts.append(start)
        packet_duration = _packet_number(packet.get("duration_time"))
        if packet_duration is not None and packet_duration > 0:
            explicit_ends.append(start + packet_duration)
            starts_with_duration.add(start)
    starts = sorted(set(starts))
    if not starts:
        return None
    inferred_step = None
    if len(starts) > 1:
        positive_steps = [b - a for a, b in zip(starts, starts[1:]) if b > a]
        if positive_steps:
            inferred_step = positive_steps[-1]
    if inferred_step is None:
        rate = _positive_rate(stream.get("avg_frame_rate")) or _positive_rate(
            stream.get("r_frame_rate")
        )
        inferred_step = 1 / rate if rate else None
    ends = explicit_ends
    if (
        starts[-1] not in starts_with_duration
        and inferred_step is not None
        and inferred_step > 0
    ):
        ends.append(starts[-1] + inferred_step)
    if not ends:
        return None
    candidate = max(ends) - starts[0]
    return candidate if isfinite(candidate) and candidate > 0 else None


def probe_audio(path: Path) -> bool:
    """ffprobe 探测是否有音轨（-select_streams a，同 video-maker 的 probe_audio 逻辑）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise UploadError(f"ffprobe failed: {e}") from e
    if r.returncode != 0:
        raise UploadError("unreadable video file")
    try:
        return bool(json.loads(r.stdout or "{}").get("streams"))
    except ValueError as e:
        raise UploadError("cannot parse audio probe result") from e


def resolve_file(data_dir: Path, cid: str, name: str) -> Path | None:
    """白名单映射 + resolved-path 防穿越；越界/不存在返回 None。"""
    if not _ID_RE.match(cid):
        return None
    cdir = (data_dir / cid).resolve()
    if name == "source.mp4":
        # 源视频扩展名不定（ALLOWED_EXT 内），取唯一 source.*；缺文件走下方 is_file 判空
        cand = next(iter(sorted(cdir.glob("source.*"))), cdir / "source.mp4")
    elif name == "preview.mp4":
        cand = cdir / "preview.mp4"
    elif name == "generated.mp4":
        cand = cdir / "generated.mp4"
    elif name == "contact_sheet.jpg":
        cand = cdir / "work" / "contact_sheet.jpg"
    elif name.startswith("keyframes/"):
        fn = name[len("keyframes/"):]
        if not fn or fn != Path(fn).name:
            return None
        cand = cdir / "work" / "keyframes" / fn
    elif name.startswith("postprocessed/"):
        fn = name[len("postprocessed/"):]
        if not fn or fn != Path(fn).name:
            return None
        cand = cdir / "work" / "postprocessed" / fn
    elif name.startswith("segments/"):
        m = _SEG_FILE_RE.match(name[len("segments/"):])
        if not m or m.group(3) != Path(m.group(3)).name:
            return None
        cand = cdir / "work" / "segments" / m.group(1) / "work" / m.group(2) / m.group(3)
    else:
        return None
    resolved = cand.resolve()
    if not resolved.is_relative_to(cdir) or not resolved.is_file():
        return None
    return resolved
