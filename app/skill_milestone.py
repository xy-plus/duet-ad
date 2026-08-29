"""Durably freeze the three research Skills used by one CID.

The first research pipeline entry point publishes one byte-for-byte copy of
each contracted ``SKILL.md`` and a canonical manifest.  All later consumers
must receive bytes from that manifest; this module deliberately has no
ambient/context-local Skill binding and never falls back to the live tree.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MANIFEST_SCHEMA = "duet.skill-milestone"
MANIFEST_VERSION = 2
MANIFEST_RELATIVE_PATH = Path("work/skills/skill_milestone.json")
MILESTONE_ID_PREFIX = "skill-"
SKILL_NAMES = (
    "video-maker",
    "image-postprocess",
    "video-prompt-fusion",
)

_SKILL_SOURCE_RELATIVE_PATHS = {
    name: Path("skills") / name / "SKILL.md" for name in SKILL_NAMES
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MILESTONE_ID_RE = re.compile(r"^skill-[0-9a-f]{64}$")
_UNSET = object()
_LOCK_FILENAME = ".skill_milestone.lock"


class SkillMilestoneError(ValueError):
    """The durable Skill milestone is missing, corrupt, or unsafe."""


@dataclass(frozen=True)
class FrozenSkill:
    name: str
    source_path: str
    frozen_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class FrozenSkillMilestone:
    root: Path
    manifest_path: Path
    milestone_id: str
    git_commit: str | None
    skills: tuple[FrozenSkill, ...]

    def skill(self, name: str) -> FrozenSkill:
        for item in self.skills:
            if item.name == name:
                return item
        raise SkillMilestoneError(f"unknown frozen Skill: {name}")

    def path(self, name: str) -> Path:
        item = self.skill(name)
        path = _safe_project_path(self.root, item.frozen_path)
        _verify_frozen_file(path, item)
        return path

    def read_bytes(self, name: str) -> bytes:
        item = self.skill(name)
        path = self.path(name)
        data = _read_regular(path, error="frozen Skill is unreadable")
        if len(data) != item.size or hashlib.sha256(data).hexdigest() != item.sha256:
            raise SkillMilestoneError("frozen Skill bytes do not match manifest")
        return data

    def public_summary(self) -> dict:
        """Return the minimal immutable summary safe for API/Web detail."""
        return {
            "id": self.milestone_id,
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "skills": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "size": item.size,
                }
                for item in self.skills
            ],
        }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def canonical_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Serialize a generated manifest using the one canonical byte encoding."""
    return _canonical_json_bytes(manifest)


def derive_milestone_id(entries: list[dict]) -> str:
    """Derive a full content identity from the ordered Skill byte records."""
    if not isinstance(entries, list) or len(entries) != len(SKILL_NAMES):
        raise SkillMilestoneError("Skill milestone skill list is invalid")
    identity = []
    for expected_name, entry in zip(SKILL_NAMES, entries):
        if (
            not isinstance(entry, dict)
            or entry.get("name") != expected_name
            or not isinstance(entry.get("sha256"), str)
            or _SHA256_RE.fullmatch(entry["sha256"]) is None
            or isinstance(entry.get("size"), bool)
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
        ):
            raise SkillMilestoneError("Skill milestone skill entry is invalid")
        identity.append(
            {
                "name": expected_name,
                "sha256": entry["sha256"],
                "size": entry["size"],
            }
        )
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    return f"{MILESTONE_ID_PREFIX}{digest}"


def _lexical_absolute(path: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = Path.cwd() / value
    return value


def _check_directory_chain(path: Path, *, create: bool) -> Path:
    """Check/create every component without ever following a symlink."""
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for part in parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise SkillMilestoneError("Skill milestone path is missing") from None
            try:
                current.mkdir()
                info = current.lstat()
            except FileExistsError:
                info = current.lstat()
            except OSError:
                raise SkillMilestoneError("Skill milestone path is invalid") from None
        except OSError:
            raise SkillMilestoneError("Skill milestone path is invalid") from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise SkillMilestoneError("Skill milestone path contains a symlink or file")
    return absolute


def _project_root(root: Path, *, create: bool) -> Path:
    return _check_directory_chain(Path(root), create=create)


def _safe_project_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SkillMilestoneError("Skill milestone path is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SkillMilestoneError("Skill milestone path is invalid")
    resolved_root = _project_root(root, create=False)
    target = resolved_root / candidate
    current = resolved_root
    for part in candidate.parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise SkillMilestoneError("Skill milestone path is invalid") from None
        if stat.S_ISLNK(info.st_mode):
            raise SkillMilestoneError("Skill milestone path contains a symlink")
        if current != target and not stat.S_ISDIR(info.st_mode):
            raise SkillMilestoneError("Skill milestone path contains a file")
    return target


def _read_regular(path: Path, *, error: str) -> bytes:
    try:
        info = path.lstat()
    except OSError:
        raise SkillMilestoneError(error) from None
    if not stat.S_ISREG(info.st_mode):
        raise SkillMilestoneError(error)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise SkillMilestoneError(error) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SkillMilestoneError(error)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    except SkillMilestoneError:
        raise
    except OSError:
        raise SkillMilestoneError(error) from None
    finally:
        os.close(fd)


def _verify_frozen_file(path: Path, item: FrozenSkill) -> None:
    data = _read_regular(path, error="frozen Skill is missing or invalid")
    if len(data) != item.size or hashlib.sha256(data).hexdigest() != item.sha256:
        raise SkillMilestoneError("frozen Skill bytes do not match manifest")


def _git_commit(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _validate_milestone_id(value: object) -> str:
    if not isinstance(value, str) or _MILESTONE_ID_RE.fullmatch(value) is None:
        raise SkillMilestoneError("Skill milestone id is invalid")
    return value


def _source_paths(
    repository_root: Path,
    skill_sources: Mapping[str, Path] | None,
) -> dict[str, Path]:
    if skill_sources is not None and set(skill_sources) != set(SKILL_NAMES):
        raise SkillMilestoneError("Skill source set is invalid")
    result = {}
    for name in SKILL_NAMES:
        path = (
            Path(skill_sources[name])
            if skill_sources is not None
            else repository_root / _SKILL_SOURCE_RELATIVE_PATHS[name]
        )
        result[name] = _lexical_absolute(path)
    return result


def _manifest_value(
    *, milestone_id: str, git_commit: str | None, entries: list[dict]
) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "milestone_id": milestone_id,
        "git_commit": git_commit,
        "skills": entries,
    }


def _parse_manifest(root: Path, manifest_path: Path) -> FrozenSkillMilestone:
    data = _read_regular(
        manifest_path,
        error="Skill milestone manifest is missing or invalid",
    )
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SkillMilestoneError("Skill milestone manifest is invalid") from None
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "version", "milestone_id", "git_commit", "skills",
    }:
        raise SkillMilestoneError("Skill milestone manifest is invalid")
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["version"] != MANIFEST_VERSION
    ):
        raise SkillMilestoneError("Skill milestone manifest is invalid")
    milestone_id = _validate_milestone_id(manifest["milestone_id"])
    git_commit = manifest["git_commit"]
    if git_commit is not None and (
        not isinstance(git_commit, str) or not git_commit
    ):
        raise SkillMilestoneError("Skill milestone git commit is invalid")
    raw_skills = manifest["skills"]
    if not isinstance(raw_skills, list) or len(raw_skills) != len(SKILL_NAMES):
        raise SkillMilestoneError("Skill milestone skill list is invalid")
    skills = []
    for name, raw in zip(SKILL_NAMES, raw_skills):
        if not isinstance(raw, dict) or set(raw) != {
            "name", "source_path", "frozen_path", "sha256", "size",
        }:
            raise SkillMilestoneError("Skill milestone skill entry is invalid")
        expected_source = _SKILL_SOURCE_RELATIVE_PATHS[name].as_posix()
        expected_frozen = (Path("work") / "skills" / name / "SKILL.md").as_posix()
        if (
            raw["name"] != name
            or raw["source_path"] != expected_source
            or raw["frozen_path"] != expected_frozen
            or not isinstance(raw["sha256"], str)
            or _SHA256_RE.fullmatch(raw["sha256"]) is None
            or isinstance(raw["size"], bool)
            or not isinstance(raw["size"], int)
            or raw["size"] <= 0
        ):
            raise SkillMilestoneError("Skill milestone skill entry is invalid")
        item = FrozenSkill(
            name=name,
            source_path=expected_source,
            frozen_path=expected_frozen,
            sha256=raw["sha256"],
            size=raw["size"],
        )
        _verify_frozen_file(_safe_project_path(root, item.frozen_path), item)
        skills.append(item)
    if milestone_id != derive_milestone_id(raw_skills):
        raise SkillMilestoneError("Skill milestone id does not match Skill bytes")
    if data != _canonical_json_bytes(manifest):
        raise SkillMilestoneError("Skill milestone manifest is not canonical")
    return FrozenSkillMilestone(
        root=root,
        manifest_path=manifest_path,
        milestone_id=milestone_id,
        git_commit=git_commit,
        skills=tuple(skills),
    )


def load(root: Path) -> FrozenSkillMilestone:
    """Load and verify the one durable Skill milestone; never create one."""
    project_root = _project_root(Path(root), create=False)
    manifest_path = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.as_posix()
    )
    return _parse_manifest(project_root, manifest_path)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise SkillMilestoneError("Skill milestone directory is invalid") from None
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_fsync(path: Path, data: bytes) -> None:
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError:
        raise SkillMilestoneError("Skill milestone temporary write failed") from None
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise SkillMilestoneError("Skill milestone temporary write failed") from None
    finally:
        os.close(fd)


def _atomic_replace(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
        _fsync_directory(destination.parent)
    except OSError:
        raise SkillMilestoneError("Skill milestone publish failed") from None


def _lock_path(project_root: Path) -> Path:
    skills_root = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.parent.as_posix()
    )
    _check_directory_chain(skills_root, create=True)
    return skills_root / _LOCK_FILENAME


def _locked(project_root: Path):
    lock_path = _lock_path(project_root)
    try:
        fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        raise SkillMilestoneError("Skill milestone lock is unavailable") from None
    return fd


def _entry_data(repository: Path, skill_sources: Mapping[str, Path] | None):
    sources = _source_paths(repository, skill_sources)
    entries = []
    contents: dict[str, bytes] = {}
    for name in SKILL_NAMES:
        data = _read_regular(
            sources[name], error=f"Skill source is missing or invalid: {name}"
        )
        if not data:
            raise SkillMilestoneError(f"Skill source is empty: {name}")
        contents[name] = data
        entries.append({
            "name": name,
            "source_path": _SKILL_SOURCE_RELATIVE_PATHS[name].as_posix(),
            "frozen_path": (Path("work") / "skills" / name / "SKILL.md").as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    return entries, contents


def _has_partial_frozen_tree(project_root: Path) -> bool:
    for name in SKILL_NAMES:
        path = _safe_project_path(
            project_root, (Path("work") / "skills" / name / "SKILL.md").as_posix()
        )
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise SkillMilestoneError("Skill milestone frozen tree is invalid") from None
        return True
    return False


def _create(
    project_root: Path,
    *,
    repository_root: Path,
    milestone_id: str | None,
    git_commit: str | None,
    skill_sources: Mapping[str, Path] | None,
) -> FrozenSkillMilestone:
    entries, contents = _entry_data(repository_root, skill_sources)
    derived_milestone_id = derive_milestone_id(entries)
    if milestone_id is not None and milestone_id != derived_milestone_id:
        raise SkillMilestoneError("explicit Skill milestone id does not match Skill bytes")
    manifest = _manifest_value(
        milestone_id=derived_milestone_id, git_commit=git_commit, entries=entries
    )
    manifest_data = _canonical_json_bytes(manifest)
    manifest_path = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.as_posix()
    )
    skills_root = manifest_path.parent
    _check_directory_chain(skills_root, create=True)
    stage = Path(tempfile.mkdtemp(prefix=".skill-milestone-", dir=str(skills_root)))
    try:
        _check_directory_chain(stage, create=False)
        for name in SKILL_NAMES:
            staged = stage / name / "SKILL.md"
            _check_directory_chain(staged.parent, create=True)
            _write_fsync(staged, contents[name])
            _fsync_directory(staged.parent)
        staged_manifest = stage / MANIFEST_RELATIVE_PATH.name
        _write_fsync(staged_manifest, manifest_data)
        _fsync_directory(stage)
        for name in SKILL_NAMES:
            destination = _safe_project_path(
                project_root, (Path("work") / "skills" / name / "SKILL.md").as_posix()
            )
            _check_directory_chain(destination.parent, create=True)
            _atomic_replace(stage / name / "SKILL.md", destination)
        _atomic_replace(staged_manifest, manifest_path)
        _fsync_directory(skills_root)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return _parse_manifest(project_root, manifest_path)


def freeze(
    root: Path,
    *,
    repository_root: Path | None = None,
    milestone_id: str | None = None,
    git_commit: str | None | object = _UNSET,
    skill_sources: Mapping[str, Path] | None = None,
) -> FrozenSkillMilestone:
    """Create the first immutable milestone or load the existing one.

    The per-CID lock covers the existence check and complete publication.
    A present but invalid manifest, or a partial frozen tree left without a
    manifest, is an error and is never rebuilt from current source bytes.
    """
    project_root = _project_root(Path(root), create=True)
    repository = _project_root(
        Path(repository_root or Path(__file__).resolve().parents[1]), create=False
    )
    if milestone_id is not None:
        _validate_milestone_id(milestone_id)
    lock_fd = _locked(project_root)
    try:
        manifest_path = _safe_project_path(
            project_root, MANIFEST_RELATIVE_PATH.as_posix()
        )
        try:
            manifest_path.lstat()
        except FileNotFoundError:
            if _has_partial_frozen_tree(project_root):
                raise SkillMilestoneError(
                    "Skill milestone manifest is missing beside a frozen tree"
                )
        except OSError:
            raise SkillMilestoneError("Skill milestone manifest is invalid") from None
        else:
            loaded = _parse_manifest(project_root, manifest_path)
            if milestone_id is not None and milestone_id != loaded.milestone_id:
                raise SkillMilestoneError(
                    "explicit Skill milestone id does not match Skill bytes"
                )
            return loaded
        commit = _git_commit(repository) if git_commit is _UNSET else git_commit
        if commit is not None and (not isinstance(commit, str) or not commit):
            raise SkillMilestoneError("Skill milestone git commit is invalid")
        return _create(
            project_root,
            repository_root=repository,
            milestone_id=milestone_id,
            git_commit=commit,
            skill_sources=skill_sources,
        )
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def ensure(*args, **kwargs) -> FrozenSkillMilestone:
    """Named publisher entry point used immediately before first Skill call."""
    return freeze(*args, **kwargs)


def _atomic_write(path: Path, data: bytes) -> None:
    target = _lexical_absolute(path)
    _check_directory_chain(target.parent, create=False)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    os.close(fd)
    temporary = Path(raw_temporary)
    try:
        _write_fsync(temporary, data)
        _atomic_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def install(milestone: FrozenSkillMilestone, name: str, destination: Path) -> Path:
    """Materialize verified frozen bytes into a caller-owned isolated stage."""
    data = milestone.read_bytes(name)
    target = _lexical_absolute(Path(destination))
    _check_directory_chain(target.parent, create=False)
    try:
        info = target.lstat()
    except FileNotFoundError:
        info = None
    except OSError:
        raise SkillMilestoneError("Skill stage destination is invalid") from None
    if info is not None and stat.S_ISLNK(info.st_mode):
        raise SkillMilestoneError("Skill stage destination is a symlink")
    _atomic_write(target, data)
    return target


def manifest_for_sources(
    repository_root: Path | None = None,
    *,
    git_commit: str | None | object = _UNSET,
) -> dict:
    """Build a current hash-bearing manifest without writing a CID."""
    repository = _project_root(
        Path(repository_root or Path(__file__).resolve().parents[1]), create=False
    )
    entries, _contents = _entry_data(repository, None)
    commit = _git_commit(repository) if git_commit is _UNSET else git_commit
    if commit is not None and (not isinstance(commit, str) or not commit):
        raise SkillMilestoneError("Skill milestone git commit is invalid")
    return _manifest_value(
        milestone_id=derive_milestone_id(entries), git_commit=commit, entries=entries
    )


def manifest_template(repository_root: Path | None = None) -> dict:
    """Generate the manifest from current bytes; no hand-maintained template."""
    return manifest_for_sources(repository_root, git_commit=None)
