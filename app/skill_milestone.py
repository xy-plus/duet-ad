"""Freeze the three research Skills used by one project.

The project directory is the authority after the first Skill call starts.  A
manifest and three byte-for-byte copies are written before that call, so a
later source-tree edit cannot silently change an in-flight or resumable CID.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Mapping


MANIFEST_SCHEMA = "duet.skill-milestone"
MANIFEST_VERSION = 1
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
_MILESTONE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_UNSET = object()


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
        path = self.path(name)
        try:
            data = path.read_bytes()
        except OSError:
            raise SkillMilestoneError("frozen Skill is unreadable") from None
        return data

    def public_summary(self) -> dict:
        """Return only non-secret, read-only data suitable for conversation detail."""
        return {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "milestone_id": self.milestone_id,
            "git_commit": self.git_commit,
            "manifest_path": _relative_path(self.root, self.manifest_path),
            "skills": [
                {
                    "name": item.name,
                    "source_path": item.source_path,
                    "frozen_path": item.frozen_path,
                    "sha256": item.sha256,
                    "size": item.size,
                }
                for item in self.skills
            ],
        }


_ACTIVE: contextvars.ContextVar[FrozenSkillMilestone | None] = contextvars.ContextVar(
    "skill_milestone", default=None
)


def current() -> FrozenSkillMilestone | None:
    """Return the milestone bound to the current pipeline context, if any."""
    return _ACTIVE.get()


@contextmanager
def activate(milestone: FrozenSkillMilestone | None) -> Iterator[None]:
    """Bind one milestone to the current thread/context for pipeline calls."""
    token = _ACTIVE.set(milestone)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


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
    """Derive the content identity from the ordered Skill byte records."""
    identity = [
        {
            "name": entry["name"],
            "sha256": entry["sha256"],
            "size": entry["size"],
        }
        for entry in entries
    ]
    digest = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()[:12]
    return f"{MILESTONE_ID_PREFIX}{digest}"


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise SkillMilestoneError("Skill milestone path escapes project") from None


def _safe_project_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SkillMilestoneError("Skill milestone path is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SkillMilestoneError("Skill milestone path is invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise SkillMilestoneError("Skill milestone path escapes project") from None
    return resolved


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
    if path.is_symlink():
        raise SkillMilestoneError("frozen Skill is a symlink")
    data = _read_regular(path, error="frozen Skill is missing or invalid")
    if not data or len(data) != item.size:
        raise SkillMilestoneError("frozen Skill bytes do not match manifest")
    if hashlib.sha256(data).hexdigest() != item.sha256:
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
        result[name] = path.resolve()
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
        expected_frozen = (
            Path("work") / "skills" / name / "SKILL.md"
        ).as_posix()
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
    expected_milestone_id = derive_milestone_id(raw_skills)
    if milestone_id != expected_milestone_id:
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
    """Load and verify the one durable Skill milestone for a project."""
    project_root = Path(root).resolve()
    manifest_path = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.as_posix()
    )
    return _parse_manifest(project_root, manifest_path)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create(
    project_root: Path,
    *,
    repository_root: Path,
    milestone_id: str | None,
    git_commit: str | None,
    skill_sources: Mapping[str, Path] | None,
) -> FrozenSkillMilestone:
    sources = _source_paths(repository_root, skill_sources)
    entries = []
    contents: dict[str, bytes] = {}
    for name in SKILL_NAMES:
        data = _read_regular(
            sources[name],
            error=f"Skill source is missing or invalid: {name}",
        )
        if not data:
            raise SkillMilestoneError(f"Skill source is empty: {name}")
        contents[name] = data
        entries.append({
            "name": name,
            "source_path": _SKILL_SOURCE_RELATIVE_PATHS[name].as_posix(),
            "frozen_path": (
                Path("work") / "skills" / name / "SKILL.md"
            ).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    derived_milestone_id = derive_milestone_id(entries)
    if milestone_id is not None and milestone_id != derived_milestone_id:
        raise SkillMilestoneError(
            "explicit Skill milestone id does not match Skill bytes"
        )
    manifest = _manifest_value(
        milestone_id=derived_milestone_id,
        git_commit=git_commit,
        entries=entries,
    )
    manifest_data = _canonical_json_bytes(manifest)
    manifest_path = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.as_posix()
    )
    skills_root = manifest_path.parent
    skills_root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".skill-milestone-", dir=str(skills_root))
    )
    try:
        for name in SKILL_NAMES:
            staged = stage / name / "SKILL.md"
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(contents[name])
        staged_manifest = stage / MANIFEST_RELATIVE_PATH.name
        staged_manifest.write_bytes(manifest_data)
        for name in SKILL_NAMES:
            destination = skills_root / name / "SKILL.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name / "SKILL.md", destination)
        os.replace(staged_manifest, manifest_path)
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
    """Create the first milestone, or load the existing immutable one.

    ``milestone_id`` is optional evidence supplied by a caller; when present
    it must equal the deterministic ID derived from the three Skill bytes.
    ``git_commit=None`` explicitly records that Git was unavailable.  When
    omitted, the current repository ``HEAD`` is recorded when it can be read.
    """
    project_root = Path(root).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    repository = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    if milestone_id is not None:
        _validate_milestone_id(milestone_id)
    manifest_path = _safe_project_path(
        project_root, MANIFEST_RELATIVE_PATH.as_posix()
    )
    if manifest_path.exists():
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


def ensure(*args, **kwargs) -> FrozenSkillMilestone:
    """Named alias used by pipeline wiring."""
    return freeze(*args, **kwargs)


def install(
    milestone: FrozenSkillMilestone,
    name: str,
    destination: Path,
) -> Path:
    """Materialize one verified frozen Skill into an isolated Skill stage."""
    data = milestone.read_bytes(name)
    target = Path(destination)
    if target.exists() and target.is_symlink():
        raise SkillMilestoneError("Skill stage destination is a symlink")
    _atomic_write(target, data)
    return target


@contextmanager
def bind_module_skill(
    milestone: FrozenSkillMilestone,
    name: str,
    module: ModuleType,
    *,
    attribute: str = "_SKILL",
) -> Iterator[Path]:
    """Temporarily route a legacy module's Skill lookup to frozen bytes.

    ``image_optimization`` predates the common milestone seam and resolves its
    Skill through a module constant.  This narrow binding lets pipeline.py
    route that lookup without changing the image/postprocess module itself.
    """
    if not hasattr(module, attribute):
        raise SkillMilestoneError("Skill module has no configurable Skill path")
    path = milestone.path(name)
    previous = getattr(module, attribute)
    setattr(module, attribute, path)
    try:
        yield path
    finally:
        setattr(module, attribute, previous)


def manifest_for_sources(
    repository_root: Path | None = None,
    *,
    git_commit: str | None | object = _UNSET,
) -> dict:
    """Build a current manifest from source bytes without writing a CID."""
    repository = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    sources = _source_paths(repository, None)
    entries = []
    for name in SKILL_NAMES:
        data = _read_regular(
            sources[name],
            error=f"Skill source is missing or invalid: {name}",
        )
        if not data:
            raise SkillMilestoneError(f"Skill source is empty: {name}")
        entries.append({
            "name": name,
            "source_path": _SKILL_SOURCE_RELATIVE_PATHS[name].as_posix(),
            "frozen_path": (
                Path("work") / "skills" / name / "SKILL.md"
            ).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    commit = _git_commit(repository) if git_commit is _UNSET else git_commit
    if commit is not None and (not isinstance(commit, str) or not commit):
        raise SkillMilestoneError("Skill milestone git commit is invalid")
    return _manifest_value(
        milestone_id=derive_milestone_id(entries),
        git_commit=commit,
        entries=entries,
    )


def manifest_template(repository_root: Path | None = None) -> dict:
    """Backward-compatible name for the generated, hash-bearing manifest."""
    return manifest_for_sources(repository_root, git_commit=None)
