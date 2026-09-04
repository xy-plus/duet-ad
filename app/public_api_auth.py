"""Independent API-key authentication for the server-to-server public API."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tempfile
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_KEY_RE = re.compile(
    r"^duet_live_([A-Za-z0-9][A-Za-z0-9_-]{7,31})\.([A-Za-z0-9_-]{32,128})$"
)
_AUTHORIZATION_RE = re.compile(r"(?i:Bearer)[ \t]+([^ \t\r\n]+)")
_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,31}$")
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1


class PublicAuthError(ValueError):
    def __init__(self, code: str, status_code: int = 401) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class Principal:
    owner_id: str
    key_id: str


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _derive(secret: str, salt: bytes) -> str:
    return hashlib.scrypt(
        secret.encode("ascii"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
    ).hex()


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise PublicAuthError("api_key_registry_insecure", 503)
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except FileNotFoundError:
        raise PublicAuthError("api_key_registry_unavailable", 503) from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise PublicAuthError("api_key_registry_invalid", 503) from None
    if not isinstance(value, dict) or value.get("version") != 1:
        raise PublicAuthError("api_key_registry_invalid", 503)
    clients = value.get("clients")
    if not isinstance(clients, list):
        raise PublicAuthError("api_key_registry_invalid", 503)
    seen: set[str] = set()
    for client in clients:
        if not isinstance(client, dict):
            raise PublicAuthError("api_key_registry_invalid", 503)
        key_id = client.get("key_id")
        owner_id = client.get("owner_id")
        salt_hex = client.get("salt")
        digest = client.get("secret_scrypt")
        if (
            not isinstance(key_id, str)
            or _KEY_ID_RE.fullmatch(key_id) is None
            or key_id in seen
            or not isinstance(owner_id, str)
            or _OWNER_RE.fullmatch(owner_id) is None
            or client.get("state") not in {"active", "revoked"}
            or not isinstance(salt_hex, str)
            or not isinstance(digest, str)
        ):
            raise PublicAuthError("api_key_registry_invalid", 503)
        try:
            if len(bytes.fromhex(salt_hex)) != 16 or len(bytes.fromhex(digest)) != 32:
                raise ValueError
        except ValueError:
            raise PublicAuthError("api_key_registry_invalid", 503) from None
        seen.add(key_id)
    return value


def authenticate(path: Path, authorization: str | None) -> Principal:
    if not isinstance(authorization, str):
        raise PublicAuthError("invalid_api_key")
    authorization_match = _AUTHORIZATION_RE.fullmatch(authorization)
    if authorization_match is None:
        raise PublicAuthError("invalid_api_key")
    match = _KEY_RE.fullmatch(authorization_match.group(1))
    if match is None:
        raise PublicAuthError("invalid_api_key")
    key_id, secret = match.groups()
    registry = _load_registry(path)
    matched: dict[str, Any] | None = None
    for candidate in registry["clients"]:
        if isinstance(candidate, dict) and hmac.compare_digest(
            str(candidate.get("key_id", "")), key_id
        ):
            matched = candidate
            break
    if matched is None or matched.get("state") != "active":
        raise PublicAuthError("invalid_api_key")
    owner_id = matched.get("owner_id")
    salt_hex = matched.get("salt")
    digest = matched.get("secret_scrypt")
    if (
        not isinstance(owner_id, str)
        or _OWNER_RE.fullmatch(owner_id) is None
        or not isinstance(salt_hex, str)
        or not isinstance(digest, str)
    ):
        raise PublicAuthError("api_key_registry_invalid", 503)
    try:
        actual = _derive(secret, bytes.fromhex(salt_hex))
    except (ValueError, TypeError):
        raise PublicAuthError("api_key_registry_invalid", 503) from None
    if not hmac.compare_digest(actual, digest):
        raise PublicAuthError("invalid_api_key")
    return Principal(owner_id=owner_id, key_id=key_id)


def validate_registry(path: Path) -> None:
    """Fail closed during application construction when public API is enabled."""
    _load_registry(path)


def _write_registry(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_mode & 0o077:
        raise PublicAuthError("api_key_registry_directory_insecure", 503)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".public-api-clients-", suffix=".json", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _registry_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_mode & 0o077:
        raise PublicAuthError("api_key_registry_directory_insecure", 503)
    lock_path = path.parent / ".public-api-clients.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PublicAuthError("api_key_registry_unavailable", 503)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def create_client_key(path: Path, owner_id: str, *, key_id: str | None = None) -> str:
    if _OWNER_RE.fullmatch(owner_id) is None:
        raise ValueError("owner_id must match [A-Za-z0-9][A-Za-z0-9_-]{2,63}")
    resolved_key_id = key_id or secrets.token_hex(6)
    if _KEY_ID_RE.fullmatch(resolved_key_id) is None:
        raise ValueError("key_id format is invalid")
    with _registry_lock(path):
        try:
            registry = _load_registry(path)
        except PublicAuthError as exc:
            if exc.code != "api_key_registry_unavailable":
                raise
            registry = {"version": 1, "clients": []}
        if any(
            isinstance(item, dict) and item.get("key_id") == resolved_key_id
            for item in registry["clients"]
        ):
            raise ValueError("key_id already exists")
        secret = secrets.token_urlsafe(32)
        salt = secrets.token_bytes(16)
        registry["clients"].append(
            {
                "key_id": resolved_key_id,
                "owner_id": owner_id,
                "state": "active",
                "salt": salt.hex(),
                "secret_scrypt": _derive(secret, salt),
            }
        )
        _write_registry(path, registry)
    return f"duet_live_{resolved_key_id}.{secret}"


def revoke_client_key(path: Path, key_id: str) -> None:
    if _KEY_ID_RE.fullmatch(key_id) is None:
        raise ValueError("key_id format is invalid")
    with _registry_lock(path):
        registry = _load_registry(path)
        found = False
        for item in registry["clients"]:
            if isinstance(item, dict) and item.get("key_id") == key_id:
                item["state"] = "revoked"
                found = True
                break
        if not found:
            raise ValueError("key_id not found")
        _write_registry(path, registry)
