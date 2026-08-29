import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import cv2
import numpy as np

from app import h3_project, image_optimization, long_generation, pipeline, skill_milestone, storage
from conftest import make_settings


ROOT = Path(__file__).resolve().parents[1]
SKILL_NAMES = (
    "video-maker",
    "image-postprocess",
    "video-prompt-fusion",
)


def test_first_freeze_copies_all_skill_bytes_and_writes_canonical_manifest(tmp_path):
    project = tmp_path / "cid-001"

    frozen = skill_milestone.freeze(
        project,
        repository_root=ROOT,
        git_commit="c1a6ae169f8ec7d2499974001959f73ef7be658d",
    )

    manifest_path = project / skill_milestone.MANIFEST_RELATIVE_PATH
    assert frozen.manifest_path == manifest_path
    assert manifest_path.read_bytes() == (
        json.dumps(
            {
                "git_commit": "c1a6ae169f8ec7d2499974001959f73ef7be658d",
                "milestone_id": frozen.milestone_id,
                "schema": skill_milestone.MANIFEST_SCHEMA,
                "skills": [
                    {
                        "frozen_path": (
                            f"work/skills/{name}/SKILL.md"
                        ),
                        "name": name,
                        "sha256": hashlib.sha256(
                            (ROOT / "skills" / name / "SKILL.md").read_bytes()
                        ).hexdigest(),
                        "size": (ROOT / "skills" / name / "SKILL.md").stat().st_size,
                        "source_path": f"skills/{name}/SKILL.md",
                    }
                    for name in SKILL_NAMES
                ],
                "version": skill_milestone.MANIFEST_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )

    for name in SKILL_NAMES:
        source = ROOT / "skills" / name / "SKILL.md"
        destination = project / "work" / "skills" / name / "SKILL.md"
        assert destination.read_bytes() == source.read_bytes()


def test_frozen_bytes_are_authoritative_after_live_skill_source_drifts(tmp_path):
    source_root = tmp_path / "repo"
    project = tmp_path / "cid-002"
    for name in SKILL_NAMES:
        path = source_root / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"initial {name}\n".encode())

    frozen = skill_milestone.freeze(
        project,
        repository_root=source_root,
        git_commit=None,
    )
    before = frozen.manifest_path.read_bytes()
    for name in SKILL_NAMES:
        (source_root / "skills" / name / "SKILL.md").write_bytes(
            f"drifted {name}\n".encode()
        )

    loaded = skill_milestone.load(project)

    assert loaded.public_summary() == frozen.public_summary()
    assert loaded.manifest_path.read_bytes() == before
    for name in SKILL_NAMES:
        assert loaded.read_bytes(name) == f"initial {name}\n".encode()


def test_milestone_id_is_derived_from_ordered_skill_content(tmp_path):
    source_root = tmp_path / "repo"
    first_project = tmp_path / "cid-first"
    second_project = tmp_path / "cid-second"
    for name in SKILL_NAMES:
        path = source_root / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"first {name}\n".encode())
    first = skill_milestone.freeze(
        first_project, repository_root=source_root, git_commit=None,
    )
    expected = skill_milestone.derive_milestone_id([
        {
            "name": item.name,
            "sha256": item.sha256,
            "size": item.size,
        }
        for item in first.skills
    ])
    assert first.milestone_id == expected

    (source_root / "skills" / "image-postprocess" / "SKILL.md").write_bytes(
        b"changed image-postprocess\n"
    )
    second = skill_milestone.freeze(
        second_project, repository_root=source_root, git_commit=None,
    )
    assert second.milestone_id != first.milestone_id
    with pytest.raises(skill_milestone.SkillMilestoneError):
        skill_milestone.freeze(
            tmp_path / "cid-invalid",
            repository_root=source_root,
            milestone_id="human-label-that-is-not-content-id",
            git_commit=None,
        )


def test_install_frozen_skill_stage_uses_the_same_bytes_for_three_skill_calls(tmp_path):
    source_root = tmp_path / "repo"
    project = tmp_path / "cid-003"
    for name in SKILL_NAMES:
        path = source_root / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen {name}\n".encode())
    frozen = skill_milestone.freeze(
        project,
        repository_root=source_root,
        git_commit=None,
    )

    observed = []
    for name in SKILL_NAMES:
        stage = tmp_path / "stages" / name
        stage.mkdir(parents=True)
        skill_path = skill_milestone.install(frozen, name, stage / "SKILL.md")
        observed.append(skill_path.read_bytes())

    assert observed == [frozen.read_bytes(name) for name in SKILL_NAMES]
    assert len({hashlib.sha256(value).hexdigest() for value in observed}) == 3


def test_invalid_manifest_is_not_rebuilt_from_drifted_sources(tmp_path):
    project = tmp_path / "cid-004"
    frozen = skill_milestone.freeze(
        project,
        repository_root=ROOT,
        git_commit=None,
    )
    manifest = json.loads(frozen.manifest_path.read_text(encoding="utf-8"))
    manifest["skills"][0]["sha256"] = "0" * 64
    frozen.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(skill_milestone.SkillMilestoneError):
        skill_milestone.load(project)


def test_pipeline_video_and_image_calls_consume_same_frozen_bytes(
    tmp_path, monkeypatch,
):
    project = tmp_path / "cid-005"
    frozen = skill_milestone.freeze(
        project,
        repository_root=ROOT,
        git_commit=None,
    )
    # Simulate a live worktree update after the CID freeze.  Both seams below
    # must still observe the durable copy, not this replacement path.
    drifted_video_skill = tmp_path / "drifted-video-maker.md"
    drifted_video_skill.write_bytes(b"drifted video-maker")
    monkeypatch.setattr(pipeline, "SKILL_MD", drifted_video_skill)

    frame = project / "source-frame.png"
    image = np.full((8, 12, 3), 96, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame.write_bytes(encoded.tobytes())
    seen_video = []

    class VideoRunner:
        def run_isolated_until_output(
            self, cwd, _prompt, *, session_dir, output_path,
            max_output_bytes, validate_output,
        ):
            seen_video.append((cwd / "SKILL.md").read_bytes())
            output = b'{"people":[],"entities":[],"scenes":[]}\n'
            output_path.write_bytes(output)
            return validate_output(output)

    pipeline._generate_project_element_index(
        VideoRunner(),
        project,
        {1: [frame]},
        milestone=frozen,
    )
    assert seen_video == [frozen.read_bytes("video-maker")]

    seen_image = []

    def generate_project_prompts(*_args, **_kwargs):
        seen_image.append(_kwargs["skill_bytes"])
        return {"version": 4}, {}

    monkeypatch.setattr(
        image_optimization,
        "generate_project_prompts",
        generate_project_prompts,
    )
    settings = SimpleNamespace(
        seedream_edit_mode="independent_parallel",
        retry_count=0,
        retry_interval_s=0,
    )
    pipeline._generate_image_optimization_project(
        settings,
        object(),
        [],
        session_dir=project,
        step="test image postprocess",
        element_index_path=project / "element_index.json",
        milestone=frozen,
    )
    assert seen_image == [frozen.read_bytes("image-postprocess")]


def _fusion_input_for_milestone(root: Path) -> bytes:
    frames = []
    frame_prompts = []
    frame_dir = root / "work" / "segments" / "1" / "new_keyframes"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for order in range(1, 10):
        frame_path = frame_dir / f"{order:02d}.png"
        frame_data = f"frame-{order}".encode()
        frame_path.write_bytes(frame_data)
        frames.append({
            "order": order,
            "path": f"work/segments/1/new_keyframes/{order:02d}.png",
            "sha256": hashlib.sha256(frame_data).hexdigest(),
            "segment_time_s": float(order - 1),
            "source_scene_id": "SCENE_01",
            "transition": {
                "type": "start" if order == 1 else "continuous",
                "at_segment_s": 0.0 if order == 1 else None,
            },
        })
        text = f"source frame prompt {order}"
        frame_prompts.append({
            "order": order,
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    old_text = "source video prompt"
    old_prompt = {
        "text": old_text,
        "sha256": hashlib.sha256(old_text.encode()).hexdigest(),
    }
    lines_json = "[]"
    return pipeline._canonical_json_bytes({
        "schema": long_generation.PROMPT_FUSION_INPUT_SCHEMA,
        "version": long_generation.PROMPT_FUSION_VERSION,
        "segments": [{
            "index": 1,
            "new_keyframes": frames,
            "old_video_prompt": old_prompt,
            "image_optimization_prompt": frame_prompts,
            "audio_content": {
                "lines_json": lines_json,
                "lines_sha256": hashlib.sha256(lines_json.encode()).hexdigest(),
                "voice_references": [],
                "music_policy": "forbid",
            },
        }],
    })


def test_prompt_fusion_call_consumes_cid_frozen_bytes_after_source_drift(
    tmp_path, monkeypatch,
):
    settings = make_settings(tmp_path)
    created = storage.new_conversation(
        settings.data_dir, "fusion-frozen", "source.mp4",
    )
    cid = created["id"]
    root = settings.data_dir / cid
    frozen = skill_milestone.freeze(root, repository_root=ROOT, git_commit=None)
    input_data = _fusion_input_for_milestone(root)
    acceptance_sha256 = "a" * 64
    storage.update_meta(
        settings.data_dir,
        cid,
        _image_user_acceptance={"version": 1, "sha256": acceptance_sha256},
    )
    assert pipeline.queue_prompt_fusion(
        settings,
        cid,
        input_data=input_data,
        image_acceptance_sha256=acceptance_sha256,
    ) == "queued"

    drifted_fusion_skill = tmp_path / "drifted-video-prompt-fusion.md"
    drifted_fusion_skill.write_bytes(b"drifted video-prompt-fusion")
    monkeypatch.setattr(pipeline, "PROMPT_FUSION_SKILL_MD", drifted_fusion_skill)
    seen = []

    class FusionRunner:
        def run_isolated(self, *_args, **_kwargs):
            raise AssertionError("bounded isolated output path should be used")

        def run_isolated_until_output(
            self, cwd, _prompt, *, session_dir, output_path,
            max_output_bytes, validate_output,
        ):
            seen.append((cwd / "SKILL.md").read_bytes())
            input_sha256 = hashlib.sha256(
                (cwd / "work" / h3_project.SKILL_INPUT_FILENAME).read_bytes()
            ).hexdigest()
            output = pipeline._canonical_json_bytes({
                "schema": long_generation.PROMPT_FUSION_OUTPUT_SCHEMA,
                "version": long_generation.PROMPT_FUSION_VERSION,
                "input_sha256": input_sha256,
                "segments": [{"index": 1, "visual": ["fused visual"]}],
            })
            output_path.write_bytes(output)
            return validate_output(output)

    assert pipeline.produce_prompt_fusion(settings, cid, FusionRunner()) == "done"
    assert seen == [frozen.read_bytes("video-prompt-fusion")]
    assert (
        root / "work" / pipeline.PROMPT_FUSION_FROZEN_SKILL_FILENAME
    ).read_bytes() == frozen.read_bytes("video-prompt-fusion")
