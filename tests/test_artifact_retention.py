from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import artifact_retention
from app.artifact_retention import prune_after_success, prune_project
from app.config import Settings, get_settings
from app.pipeline import Pipeline
from app.research_log import ResearchLog
from app.store import ProjectStore


def _settings(path: Path, **changes) -> Settings:
    values = {
        "access_token": "test-token",
        "multiref_mode": "direct",
        "data_dir": path,
        "public_base_url": "https://retention.test",
        "enable_mediakit": False,
        "provider_poll_interval_s": 0,
    }
    values.update(changes)
    return Settings(**values)


def _succeeded_state(project_id: str) -> dict:
    return {
        "schema": "duet.project-state",
        "version": 1,
        "id": project_id,
        "status": "succeeded",
        "project_progress": {"status": "succeeded", "percent": 100, "stage": "completed"},
        "source_path": "inputs/source.mp4",
        "output_path": "output/generated.mp4",
        "h3_prepared_artifacts": {
            "version": 1,
            "segments": {
                "1": {
                    "source_video": {"path": "work/segments/0001/source.mp4"},
                    "muted_video": {"path": "work/segments/0001/source-muted.mp4"},
                    "prompt": {"path": "work/segments/0001/h3_prompt.txt"},
                },
            },
        },
        "picsart_submission_receipts": {"version": 1, "segments": {}},
    }


def _build_project(tmp_path: Path, status: str = "succeeded") -> tuple[Path, dict]:
    project = tmp_path / "projects" / ("a" * 32)
    attempt = project / "logs" / "segment-frames-4fps" / "segment-0001" / "attempt-1"
    (attempt / "input-media").mkdir(parents=True)
    (attempt / "output-media").mkdir(parents=True)
    (attempt / "input-media" / "001-source.mp4").write_bytes(b"source-copy" * 100)
    (attempt / "input-media" / "002-frame.png").write_bytes(b"frame" * 100)
    (attempt / "output-media" / "001-frame.png").write_bytes(b"frame" * 100)
    (attempt / "input.json").write_text("{}", encoding="utf-8")
    (attempt / "output.json").write_text('{"frames": 9}', encoding="utf-8")
    (project / "logs").mkdir(parents=True, exist_ok=True)
    (project / "logs" / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    for name in ("source-4fps", "keyframes", "analysis-keyframes"):
        directory = project / "work" / "segments" / "0001" / name
        directory.mkdir(parents=True)
        (directory / "frame-001.png").write_bytes(b"png" * 100)
    (project / "work" / "source-4fps").mkdir(parents=True)
    (project / "work" / "source-4fps" / "frame-001.png").write_bytes(b"png" * 100)
    (project / "work" / "source-4fps" / "manifest.json").write_text("{}", encoding="utf-8")
    segment = project / "work" / "segments" / "0001"
    (segment / "source.mp4").write_bytes(b"segment-source")
    (segment / "source-muted.mp4").write_bytes(b"muted")
    (segment / "h3_prompt.txt").write_text("prompt", encoding="utf-8")
    (segment / "object-detect-response-target-deadbeef.json").write_text(
        '{"fingerprint": {}, "fal_response": {}}', encoding="utf-8",
    )
    (project / "inputs").mkdir(parents=True, exist_ok=True)
    (project / "inputs" / "source.mp4").write_bytes(b"original-source")
    (project / "output").mkdir(parents=True, exist_ok=True)
    (project / "output" / "generated.mp4").write_bytes(b"final-video")
    state = _succeeded_state(project.name)
    if status != "succeeded":
        state["status"] = status
        state["project_progress"] = {
            "status": status, "percent": 40, "stage": "h3-max-r2v",
        }
    (project / "state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8",
    )
    return project, state


def test_settings_default_retention_is_full(tmp_path: Path) -> None:
    assert _settings(tmp_path).artifact_retention == "full"


def test_settings_reject_unknown_retention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ARTIFACT_RETENTION"):
        _settings(tmp_path, artifact_retention="minimal")


def test_get_settings_reads_retention_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("ARTIFACT_RETENTION", raising=False)
    assert get_settings().artifact_retention == "full"
    monkeypatch.setenv("ARTIFACT_RETENTION", "lean")
    assert get_settings().artifact_retention == "lean"
    monkeypatch.setenv("ARTIFACT_RETENTION", "bogus")
    with pytest.raises(ValueError, match="ARTIFACT_RETENTION"):
        get_settings()


def test_lean_prune_removes_media_copies_and_frame_dirs(tmp_path: Path) -> None:
    project, state = _build_project(tmp_path)
    log = ResearchLog(project)
    summary = prune_project(project, state=state, log=log)
    assert summary["removed_files"] > 0
    assert summary["removed_bytes"] > 0
    assert not summary["errors"]
    assert not summary["skipped_referenced"]
    assert not (project / "logs" / "segment-frames-4fps" / "segment-0001"
                / "attempt-1" / "input-media").exists()
    assert not (project / "logs" / "segment-frames-4fps" / "segment-0001"
                / "attempt-1" / "output-media").exists()
    for name in ("source-4fps", "keyframes", "analysis-keyframes"):
        assert not (project / "work" / "segments" / "0001" / name).exists()
    assert not (project / "work" / "source-4fps").exists()
    kept = [
        project / "state.json",
        project / "inputs" / "source.mp4",
        project / "output" / "generated.mp4",
        project / "logs" / "trace.jsonl",
        project / "logs" / "segment-frames-4fps" / "segment-0001"
        / "attempt-1" / "input.json",
        project / "logs" / "segment-frames-4fps" / "segment-0001"
        / "attempt-1" / "output.json",
        project / "work" / "segments" / "0001" / "source.mp4",
        project / "work" / "segments" / "0001" / "source-muted.mp4",
        project / "work" / "segments" / "0001" / "h3_prompt.txt",
        project / "work" / "segments" / "0001"
        / "object-detect-response-target-deadbeef.json",
    ]
    for path in kept:
        assert path.is_file(), f"must keep {path}"
    events = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]
    prune_events = [row for row in events if row.get("event") == "artifact_prune_lean"]
    assert len(prune_events) == 1
    assert prune_events[0]["removed_files"] == summary["removed_files"]


def test_prune_is_idempotent(tmp_path: Path) -> None:
    project, state = _build_project(tmp_path)
    first = prune_project(project, state=state)
    assert first["removed_files"] > 0
    second = prune_project(project, state=state)
    assert second["removed_files"] == 0
    assert second["removed_bytes"] == 0
    assert not second["errors"]


def test_prune_refuses_failed_project(tmp_path: Path) -> None:
    project, state = _build_project(tmp_path, status="failed")
    with pytest.raises(ValueError, match="succeeded"):
        prune_project(project, state=state)
    assert (project / "work" / "source-4fps" / "frame-001.png").is_file()
    assert (project / "logs" / "segment-frames-4fps" / "segment-0001"
            / "attempt-1" / "input-media" / "001-source.mp4").is_file()


def test_prune_skips_directories_referenced_by_state(tmp_path: Path) -> None:
    project, state = _build_project(tmp_path)
    state["h3_prepared_artifacts"]["segments"]["1"]["evidence"] = {
        "path": "work/segments/0001/keyframes/frame-001.png",
    }
    summary = prune_project(project, state=state)
    assert "work/segments/0001/keyframes" in summary["skipped_referenced"]
    assert (project / "work" / "segments" / "0001" / "keyframes"
            / "frame-001.png").is_file()
    assert not (project / "work" / "segments" / "0001" / "source-4fps").exists()


def test_prune_after_success_contains_failures(tmp_path: Path) -> None:
    project, _state = _build_project(tmp_path, status="failed")
    log = ResearchLog(project)
    result = prune_after_success(project, state={"status": "failed"}, log=log)
    assert result["pruned"] is False
    assert "error" in result
    assert (project / "work" / "source-4fps" / "frame-001.png").is_file()


def _pipeline(tmp_path: Path, retention: str) -> tuple[Pipeline, ProjectStore]:
    store = ProjectStore(tmp_path / "projects")
    pipeline = Pipeline(
        _settings(tmp_path, artifact_retention=retention),
        store,
        lambda _project_id, _path: "https://assets.invalid/value",
        deepseek=object(),
        mediakit=object(),
        picsart=object(),
        seedream=object(),
        autodl=object(),
    )
    return pipeline, store


def test_pipeline_hook_is_inert_under_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        artifact_retention,
        "prune_after_success",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    pipeline, store = _pipeline(tmp_path, "full")
    project_id, project, _state = store.create(
        {"effective": True}, "retention-full-hook",
    )
    pipeline._prune_artifacts_after_success(project, ResearchLog(project))
    assert calls == []


def test_pipeline_hook_prunes_under_lean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _fake_prune(project_dir, *, state, log=None):
        calls.append({"project_dir": project_dir, "state": state})
        return {"pruned": True, "removed_files": 0, "removed_bytes": 0}

    monkeypatch.setattr(artifact_retention, "prune_after_success", _fake_prune)
    pipeline, store = _pipeline(tmp_path, "lean")
    project_id, project, _state = store.create(
        {"effective": True}, "retention-lean-hook",
    )
    store.update(
        project_id,
        status="succeeded",
        project_progress={"status": "succeeded", "percent": 100, "stage": "completed"},
    )
    pipeline._prune_artifacts_after_success(project, ResearchLog(project))
    assert len(calls) == 1
    assert calls[0]["project_dir"] == project
    assert calls[0]["state"]["status"] == "succeeded"
