"""Targeted contract tests for the CID Skill freeze seam."""

import hashlib
import json
import base64
import inspect
from pathlib import Path

import pytest

from app import image_optimization, pipeline, postprocess, skill_milestone


NAMES = skill_milestone.SKILL_NAMES


def _repo(root: Path, marker: str = "initial") -> Path:
    repository = root / "repository"
    for name in NAMES:
        path = repository / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{marker}:{name}\n".encode())
    return repository


def test_freeze_uses_full_ordered_digest_and_minimal_public_summary(tmp_path):
    project = tmp_path / "cid"
    repository = _repo(tmp_path)

    frozen = skill_milestone.ensure(
        project, repository_root=repository, git_commit=None,
    )

    assert frozen.milestone_id.startswith("skill-")
    assert len(frozen.milestone_id) == len("skill-") + 64
    assert frozen.public_summary() == {
        "id": frozen.milestone_id,
        "schema": skill_milestone.MANIFEST_SCHEMA,
        "version": skill_milestone.MANIFEST_VERSION,
        "skills": [
            {"name": item.name, "sha256": item.sha256, "size": item.size}
            for item in frozen.skills
        ],
    }
    assert "path" not in json.dumps(frozen.public_summary())

    manifest = json.loads(frozen.manifest_path.read_text(encoding="utf-8"))
    assert manifest["milestone_id"] == frozen.milestone_id
    assert manifest["milestone_id"] == skill_milestone.derive_milestone_id(
        manifest["skills"]
    )


def test_freeze_never_rebuilds_a_missing_manifest_or_consumes_drift(tmp_path):
    project = tmp_path / "cid"
    repository = _repo(tmp_path)
    frozen = skill_milestone.freeze(project, repository_root=repository, git_commit=None)
    original = {name: frozen.read_bytes(name) for name in NAMES}
    frozen.manifest_path.unlink()
    for name in NAMES:
        (repository / "skills" / name / "SKILL.md").write_bytes(
            f"drift:{name}\n".encode()
        )

    with pytest.raises(skill_milestone.SkillMilestoneError, match="missing"):
        skill_milestone.ensure(project, repository_root=repository, git_commit=None)
    assert {
        name: (project / "work" / "skills" / name / "SKILL.md").read_bytes()
        for name in NAMES
    } == original


def test_freeze_rejects_symlinked_ancestor(tmp_path):
    repository = _repo(tmp_path)
    project = tmp_path / "cid"
    project.mkdir()
    (project / "work").symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(skill_milestone.SkillMilestoneError, match="symlink"):
        skill_milestone.ensure(project, repository_root=repository, git_commit=None)


def test_image_skill_has_no_live_default_and_accepts_only_explicit_frozen_bytes():
    assert not hasattr(image_optimization, "_SKILL")
    with pytest.raises(ValueError, match="frozen image verification skill"):
        image_optimization._skill_bytes()


def test_milestone_bytes_are_stable_after_source_update(tmp_path):
    repository = _repo(tmp_path)
    project = tmp_path / "cid"
    frozen = skill_milestone.freeze(project, repository_root=repository, git_commit=None)
    before = [hashlib.sha256(frozen.read_bytes(name)).hexdigest() for name in NAMES]
    for name in NAMES:
        (repository / "skills" / name / "SKILL.md").write_bytes(b"new bytes\n")
    loaded = skill_milestone.load(project)
    assert [hashlib.sha256(loaded.read_bytes(name)).hexdigest() for name in NAMES] == before


def test_visual_codex_consumes_frozen_skill_in_single_use_stage(tmp_path):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    project = tmp_path / "cid"
    work = project / "work"
    keyframes = work / "keyframes"
    keyframes.mkdir(parents=True)
    frozen_frames = tuple(png for _ in range(9))
    (work / "voice_lines.json").write_text("[]", encoding="utf-8")
    skill = b"frozen video-maker skill"
    seen = {}

    class Runner:
        def run_isolated(self, stage, prompt, *, session_dir, writable_paths):
            seen["stage"] = stage
            seen["prompt"] = prompt
            seen["session"] = session_dir
            seen["writable"] = writable_paths
            assert (stage / "SKILL.md").read_bytes() == skill
            assert not (stage / "work" / "voice_lines.json").exists()
            (stage / "work" / "prompt.txt").write_text("visual result", encoding="utf-8")

    names, text = pipeline._run_visual_attempt(
        Runner(), project, "visual prompt", work,
        isolate_dialogue=True,
        frozen_keyframes=frozen_frames,
        skill_bytes=skill,
    )

    assert names == [f"{index:02d}.png" for index in range(1, 10)]
    assert text == "visual result"
    assert seen["stage"].parent == Path("/tmp")
    assert seen["session"] == project
    assert len(seen["writable"]) == 1
    assert (work / "voice_lines.json").read_text(encoding="utf-8") == "[]"
    assert tuple((keyframes / name).read_bytes() for name in names) == frozen_frames


def test_postprocess_verdict_requires_explicit_frozen_skill_bytes():
    signature = inspect.signature(postprocess._v4_verify_bootstrap_packs)
    assert signature.parameters["skill_bytes"].default is inspect.Parameter.empty
    source = inspect.getsource(postprocess._v4_verify_bootstrap_packs)
    assert "skill_bytes=skill_bytes" in source
