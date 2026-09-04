"""Crash-safe append-only credit ledger for the public API.

One credit is an integer unit.  The public contract fixes 100 credits to CNY 1.
Balances are projections of immutable event files; no mutable balance is trusted.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CREDITS_PER_CNY = 100
JOB_PRICE_CREDITS = 1_000
PRICE_VERSION = "credits-fixed-1000-v1"
_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
_JOB_RE = re.compile(r"^vg_([0-9a-f]{32})$")
_ADJUSTMENT_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class CreditError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _root(data_dir: Path) -> Path:
    return data_dir / ".public-api"


def _events_dir(data_dir: Path, owner_id: str) -> Path:
    return _root(data_dir) / "credit-events" / _digest(owner_id)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def ledger_lock(data_dir: Path) -> Iterator[None]:
    root = _root(data_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    lock_path = root / "credits.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CreditError("credit_ledger_unavailable")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _event_path(data_dir: Path, owner_id: str, event_id: str) -> Path:
    return _events_dir(data_dir, owner_id) / f"{_digest(event_id)}.json"


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
    ).encode("utf-8")


def _create_event(data_dir: Path, event: dict[str, Any]) -> bool:
    owner_id = event["owner_id"]
    directory = _events_dir(data_dir, owner_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    destination = _event_path(data_dir, owner_id, event["event_id"])
    payload = _canonical(event)
    if destination.exists():
        try:
            if destination.read_bytes() == payload:
                return False
        except OSError:
            pass
        raise CreditError("credit_event_conflict")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".credit-event-", suffix=".json", dir=directory
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() == payload:
                return False
            raise CreditError("credit_event_conflict") from None
        _fsync_directory(directory)
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


def _job_is_published(data_dir: Path, owner_id: str, job_id: str) -> bool:
    match = _JOB_RE.fullmatch(job_id)
    if match is None:
        return False
    try:
        meta = json.loads(
            (data_dir / match.group(1) / "meta.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    public = meta.get("_public_api") if isinstance(meta, dict) else None
    return bool(
        isinstance(public, dict)
        and public.get("version") == 1
        and public.get("owner_id") == owner_id
        and public.get("job_id") == job_id
    )


def _load_events(data_dir: Path, owner_id: str) -> list[dict[str, Any]]:
    if _OWNER_RE.fullmatch(owner_id) is None:
        raise CreditError("invalid_owner")
    directory = _events_dir(data_dir, owner_id)
    if not directory.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        paths = sorted(directory.glob("*.json"))
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("version") != 1
                or value.get("owner_id") != owner_id
                or not isinstance(value.get("event_id"), str)
                or path.name != f"{_digest(value['event_id'])}.json"
                or not isinstance(value.get("credits"), int)
                or isinstance(value.get("credits"), bool)
                or value["credits"] <= 0
            ):
                raise CreditError("credit_ledger_corrupt")
            events.append(value)
    except CreditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CreditError("credit_ledger_corrupt") from None
    return events


def _balance_unlocked(data_dir: Path, owner_id: str) -> dict[str, int]:
    available = reserved = spent = 0
    job_events: dict[str, dict[str, int]] = {}
    for event in _load_events(data_dir, owner_id):
        kind = event.get("type")
        credits = event["credits"]
        if kind == "adjustment":
            direction = event.get("direction")
            if direction == "credit":
                available += credits
            elif direction == "debit":
                available -= credits
            else:
                raise CreditError("credit_ledger_corrupt")
            continue
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or _JOB_RE.fullmatch(job_id) is None:
            raise CreditError("credit_ledger_corrupt")
        events = job_events.setdefault(job_id, {})
        if kind in events or kind not in {"reserve", "capture", "release"}:
            raise CreditError("credit_ledger_corrupt")
        events[kind] = credits
    for job_id, events in job_events.items():
        if "reserve" not in events or ({"capture", "release"} <= set(events)):
            raise CreditError("credit_ledger_corrupt")
        terminal = "capture" if "capture" in events else (
            "release" if "release" in events else None
        )
        if terminal is not None and events[terminal] != events["reserve"]:
            raise CreditError("credit_ledger_corrupt")
        if terminal is None and not _job_is_published(data_dir, owner_id, job_id):
            # A reserve can durably land immediately before a process crash and
            # before the conversation directory is atomically published. Such
            # an orphan remains auditable but never reduces spendable credits.
            continue
        # A complete reserve + terminal pair remains authoritative after the
        # corresponding project is intentionally removed from retained data.
        available -= events["reserve"]
        reserved += events["reserve"]
        if terminal == "capture":
            reserved -= events["capture"]
            spent += events["capture"]
        elif terminal == "release":
            reserved -= events["release"]
            available += events["release"]
    if available < 0 or reserved < 0:
        raise CreditError("credit_ledger_corrupt")
    return {"available": available, "reserved": reserved, "spent": spent}


def balance(data_dir: Path, owner_id: str) -> dict[str, int]:
    with ledger_lock(data_dir):
        return _balance_unlocked(data_dir, owner_id)


def reserve(data_dir: Path, owner_id: str, job_id: str) -> bool:
    event_id = f"job:{job_id}:reserve"
    with ledger_lock(data_dir):
        destination = _event_path(data_dir, owner_id, event_id)
        if destination.exists():
            return False
        current = _balance_unlocked(data_dir, owner_id)
        if current["available"] < JOB_PRICE_CREDITS:
            raise CreditError("insufficient_credits")
        return _create_event(
            data_dir,
            {
                "version": 1,
                "event_id": event_id,
                "owner_id": owner_id,
                "type": "reserve",
                "credits": JOB_PRICE_CREDITS,
                "job_id": job_id,
                "created_at": _now(),
            },
        )


def settle(data_dir: Path, owner_id: str, job_id: str, *, succeeded: bool) -> bool:
    terminal = "capture" if succeeded else "release"
    opposite = "release" if succeeded else "capture"
    event_id = f"job:{job_id}:{terminal}"
    with ledger_lock(data_dir):
        if _event_path(data_dir, owner_id, f"job:{job_id}:{opposite}").exists():
            raise CreditError("credit_settlement_conflict")
        if not _event_path(data_dir, owner_id, f"job:{job_id}:reserve").exists():
            raise CreditError("credit_reservation_missing")
        if _event_path(data_dir, owner_id, event_id).exists():
            return False
        return _create_event(
            data_dir,
            {
                "version": 1,
                "event_id": event_id,
                "owner_id": owner_id,
                "type": terminal,
                "credits": JOB_PRICE_CREDITS,
                "job_id": job_id,
                "created_at": _now(),
            },
        )


def adjust(
    data_dir: Path,
    owner_id: str,
    credits_delta: int,
    *,
    reason: str,
    idempotency_key: str,
) -> bool:
    if _OWNER_RE.fullmatch(owner_id) is None:
        raise CreditError("invalid_owner")
    if (
        not isinstance(credits_delta, int)
        or isinstance(credits_delta, bool)
        or credits_delta == 0
    ):
        raise CreditError("invalid_credit_adjustment")
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 200:
        raise CreditError("invalid_adjustment_reason")
    if _ADJUSTMENT_KEY_RE.fullmatch(idempotency_key) is None:
        raise CreditError("invalid_idempotency_key")
    event_id = f"admin:{owner_id}:{idempotency_key}"
    event = {
        "version": 1,
        "event_id": event_id,
        "owner_id": owner_id,
        "type": "adjustment",
        "direction": "credit" if credits_delta > 0 else "debit",
        "credits": abs(credits_delta),
        "reason": reason.strip(),
        "created_at": _now(),
    }
    with ledger_lock(data_dir):
        destination = _event_path(data_dir, owner_id, event_id)
        if destination.exists():
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise CreditError("credit_ledger_corrupt") from None
            comparable = dict(existing)
            comparable["created_at"] = event["created_at"]
            if comparable == event:
                return False
            raise CreditError("credit_event_conflict")
        current = _balance_unlocked(data_dir, owner_id)
        if credits_delta < 0 and current["available"] < -credits_delta:
            raise CreditError("insufficient_available_credits")
        return _create_event(data_dir, event)


def recent_events(
    data_dir: Path, owner_id: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise CreditError("invalid_limit")
    with ledger_lock(data_dir):
        events = _load_events(data_dir, owner_id)
    events.sort(key=lambda item: (str(item.get("created_at", "")), item["event_id"]), reverse=True)
    return events[:limit]
