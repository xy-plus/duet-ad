"""Tests for the explicit, receipt-bound image-phase continuation operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import image_phase_resume, pipeline, storage
from conftest import make_settings


CID = "a" * 32


@pytest.fixture(autouse=True)
def _fixed_current_segment_plan(monkeypatch):
    monkeypatch.setattr(
        pipeline.scene_planner,
        "plan_segments",
        lambda _duration, _scenes, _dialogue: [{
            "index": 1,
            "start_s": 0.0,
            "end_s": 2.0,
            "chain_id": "chain-001",
            "join_mode": "hard_cut",
            "scene_indices": [1],
        }],
    )


def _png() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _candidate(tmp_path: Path, *, status: str = "failed") -> tuple[object, Path]:
    settings = make_settings(tmp_path)
    cdir = settings.data_dir / CID
    work = cdir / "work"
    work.mkdir(parents=True)
    (cdir / "source.mp4").write_bytes(b"source")
    storage._write_meta(cdir, {
        "schema_version": 2,
        "id": CID,
        "status": status,
        "error": image_phase_resume.IMAGE_FAILURE,
        "duration_s": 2.0,
        "dialogue_mode": "none",
        "voice_mode": "keep",
        "voice_lines": [],
        "source_width": 8,
        "source_height": 8,
        "_input_owner": None,
        "generation": None,
        "keyframes": [],
        "prompt": None,
    })
    _json(work / "manifest.json", {
        "duration_seconds": 2.0,
        "frames": [{"index": 1, "file": "01.png", "time_seconds": 0.0}],
    })
    _json(work / "scenes.json", {
        "duration_s": 2.0,
        "scenes": [{"index": 1, "start_s": 0.0, "end_s": 2.0, "frames": []}],
        "effective_scenes": [{
            "index": 1, "start_s": 0.0, "end_s": 2.0,
            "frames": [{"decode_frame_index": 0}],
        }],
        "diagnostics": [],
        "segments": [{
            "index": 1, "start_s": 0.0, "end_s": 2.0,
            "chain_id": "chain-001", "join_mode": "hard_cut",
        }],
    })
    _json(work / "element_index.json", {"people": {}, "entities": {}, "scenes": {}})
    segment = work / "segments" / "1" / "work"
    (segment / "keyframes").mkdir(parents=True)
    (segment / "anchors").mkdir()
    (work / "segments" / "1" / "source.mp4").write_bytes(b"segment")
    _json(segment / "manifest.json", {"duration_seconds": 2.0})
    (segment / "visual_prompt.txt").write_text("visual", encoding="utf-8")
    (segment / "prompt.txt").write_text("prompt", encoding="utf-8")
    _json(segment / "voice_lines.json", [])
    data = _png()
    entries = []
    for order in range(1, 10):
        name = f"{order:02d}.png"
        (segment / "keyframes" / name).write_bytes(data)
        entries.append({
            "order": order,
            "path": f"keyframes/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_scene_id": "SCENE_01",
            "source_scene_start_s": 0.0,
            "source_time_s": round((order - 1) * 0.2, 6),
            "repeated": order != 1,
        })
    _json(segment / "keyframe_sampling.json", {
        "schema": "duet.backend-keyframe-sampling",
        "version": 1,
        "selection_method": "scene-anchor-capacity-hamilton-v1",
        "keyframes": entries,
    })
    (segment / "anchors" / "first.png").write_bytes(data)
    (segment / "anchors" / "last.png").write_bytes(data)
    return settings, cdir


def test_dry_run_requires_terminal_image_failure_and_complete_artifacts(tmp_path):
    settings, _ = _candidate(tmp_path)
    manifest = image_phase_resume.inspect(settings, CID)
    assert manifest["schema"] == image_phase_resume.SCHEMA
    assert manifest["cid"] == CID
    assert manifest["segments"] == [{
        "index": 1,
        "start_s": 0.0,
        "end_s": 2.0,
        "chain_id": "chain-001",
        "join_mode": "hard_cut",
    }]
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_missing_boundary_fact_is_rejected_before_execution(tmp_path):
    settings, cdir = _candidate(tmp_path)
    (cdir / "work" / "scenes.json").unlink()
    with pytest.raises(image_phase_resume.ResumeRejected, match="scenes"):
        image_phase_resume.inspect(settings, CID)


def test_manifest_drift_is_rejected_before_codex(tmp_path, monkeypatch):
    settings, _ = _candidate(tmp_path)
    manifest = image_phase_resume.inspect(settings, CID)
    called = []
    monkeypatch.setattr(
        pipeline, "_generate_segmented_image_prompts",
        lambda *args, **kwargs: called.append(True),
    )
    manifest["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(image_phase_resume.ResumeRejected, match="manifest"):
        image_phase_resume.execute(settings, CID, manifest, runner=object())
    assert called == []


def test_execute_reuses_index_and_enters_existing_image_phase(tmp_path, monkeypatch):
    settings, _ = _candidate(tmp_path)
    manifest = image_phase_resume.inspect(settings, CID)
    seen = {}

    def generate(*args, **kwargs):
        seen["element_index_path"] = kwargs["element_index_path"]
        return {"version": 4}, {1: "edited"}

    monkeypatch.setattr(pipeline, "_generate_segmented_image_prompts", generate)
    monkeypatch.setattr(
        pipeline, "_recover_long_plan",
        lambda *args: {
            "status": "done",
            "error": None,
            "segments": [{
                "index": 1, "start_s": 0.0, "end_s": 2.0,
                "chain_id": "chain-001", "join_mode": "hard_cut",
            }],
            "long_video_plan_receipt": "long_video_plan.json",
            "fit_required": False,
            "fit_profiles": {"9:16": {"fit_required": False}},
            "aspect_ratio": "9:16", "resolution": "480p", "fit_mode": "none",
        },
    )
    monkeypatch.setattr(
        pipeline, "_freeze_image_optimization",
        lambda *args, **kwargs: ({"_image_continuity": {"version": 4}}, {"_image_optimization": {"version": 4}}),
    )
    result = image_phase_resume.execute(settings, CID, manifest, runner=object())
    assert seen["element_index_path"].name == "element_index.json"
    assert result["status"] == "done"
    assert storage.load_meta(settings.data_dir, CID)["status"] == "done"


def test_nonterminal_candidate_is_rejected(tmp_path):
    settings, _ = _candidate(tmp_path, status="processing")
    with pytest.raises(image_phase_resume.ResumeRejected, match="terminal"):
        image_phase_resume.inspect(settings, CID)


def test_image_failure_keeps_terminal_meta_and_removes_partial_plan(
    tmp_path, monkeypatch,
):
    settings, cdir = _candidate(tmp_path)
    manifest = image_phase_resume.inspect(settings, CID)
    before = (cdir / "meta.json").read_bytes()
    monkeypatch.setattr(
        pipeline,
        "_generate_segmented_image_prompts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(image_phase_resume.ResumeExecutionError, match="boom"):
        image_phase_resume.execute(settings, CID, manifest, runner=object())
    assert (cdir / "meta.json").read_bytes() == before
    assert not (cdir / "long_video_plan.json").exists()


def test_artifact_drift_during_image_generation_is_not_committed(
    tmp_path, monkeypatch,
):
    settings, cdir = _candidate(tmp_path)
    manifest = image_phase_resume.inspect(settings, CID)
    before = (cdir / "meta.json").read_bytes()

    def drift(*args, **kwargs):
        (cdir / "work/segments/1/work/keyframes/01.png").write_bytes(_png() + b"drift")
        return {"version": 4}, {1: "edited"}

    monkeypatch.setattr(pipeline, "_generate_segmented_image_prompts", drift)
    with pytest.raises(image_phase_resume.ResumeExecutionError, match="drifted"):
        image_phase_resume.execute(settings, CID, manifest, runner=object())
    assert (cdir / "meta.json").read_bytes() == before
    assert not (cdir / "long_video_plan.json").exists()


def test_diagnostic_runner_preserves_successful_phase_protocol(tmp_path):
    stage = tmp_path / "duet-image-segment-2-test"
    work = stage / "work"
    work.mkdir(parents=True)
    request = b'{"phase":"segment_frames"}\n'
    output = b'{"frames":{}}\n'
    (work / "request.json").write_bytes(request)

    class Inner:
        def run_isolated_until_output(self, *args, **kwargs):
            kwargs["output_path"].write_bytes(output)
            return {"frames": {}}

    destination = tmp_path / "diagnostics"
    runner = image_phase_resume._DiagnosticRunner(Inner(), destination)
    result = runner.run_isolated_until_output(
        stage,
        "prompt",
        session_dir=tmp_path,
        output_path=work / "segment_frames.json",
        max_output_bytes=1024,
        validate_output=lambda raw: json.loads(raw),
    )
    assert result == {"frames": {}}
    assert (destination / "segment-0002.request.json").read_bytes() == request
    assert (destination / "segment-0002.output.json").read_bytes() == output
    assert not (destination / "segment-0002.error.txt").exists()


def test_diagnostic_runner_preserves_invalid_output_and_error(tmp_path):
    stage = tmp_path / "duet-image-global-test"
    work = stage / "work"
    work.mkdir(parents=True)
    (work / "request.json").write_text('{"phase":"global_plan"}\n')

    class Inner:
        def run_isolated_until_output(self, *args, **kwargs):
            kwargs["output_path"].write_bytes(b'{"unexpected":true}\n')
            raise RuntimeError("invalid phase shape")

    destination = tmp_path / "diagnostics"
    runner = image_phase_resume._DiagnosticRunner(Inner(), destination)
    with pytest.raises(RuntimeError, match="invalid phase shape"):
        runner.run_isolated_until_output(
            stage,
            "prompt",
            session_dir=tmp_path,
            output_path=work / "global_plan.json",
            max_output_bytes=1024,
            validate_output=lambda raw: json.loads(raw),
        )
    assert (destination / "global-plan.output.json").read_bytes() == (
        b'{"unexpected":true}\n'
    )
    assert (destination / "global-plan.error.txt").read_text() == (
        "RuntimeError: invalid phase shape\n"
    )
