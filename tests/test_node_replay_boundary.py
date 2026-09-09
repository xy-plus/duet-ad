"""Boundary replays: historical failure states that must no longer break a run.

The dfe64795 project failed with ``dialogue TTS requires a voice reference``:
its Object Replace output deliberately left every person voiceless while four
translated dialogue lines waited for TTS.  The current pipeline materializes a
deterministic catalog fallback voice instead — both on fresh runs and when a
pre-fallback project resumes from its frozen sheet checkpoint.  These tests
replay that exact upstream state (harvested from the project's state
snapshot, no mocks).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.object_replace_schema import OBJECT_REPLACE_OUTPUT_VERSION
from app.pipeline import Pipeline
from app.research_log import ResearchLog

from node_replay_support import BOUNDARY_PROJECT, FIXTURES_ROOT


def _state_snapshot() -> dict:
    path = (
        FIXTURES_ROOT / "boundary" / BOUNDARY_PROJECT / "state-snapshot.json"
    )
    if not path.is_file():
        pytest.fail(
            "boundary state snapshot missing; run "
            "scripts/harvest_node_fixtures.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def test_boundary_snapshot_is_the_tts_no_voice_failure() -> None:
    snapshot = _state_snapshot()
    assert snapshot["error"]["code"] == "pre_picsart_resume_failed"
    people = snapshot["enhanced_element_index"]["people"]
    assert people, "snapshot lost the people records"
    assert all(
        not (record.get("voice") or {}).get("voice_id")
        for record in people.values()
    ), "snapshot no longer represents the voiceless boundary"
    assert len(snapshot["translated_voice_lines"]) == 4
    assert snapshot["voice_references"] == []


def test_voiceless_object_replace_falls_back_to_default_voice(
    pipeline_instance: Pipeline, tmp_path: Path,
) -> None:
    snapshot = _state_snapshot()
    enhanced = snapshot["enhanced_element_index"]
    translated = snapshot["translated_voice_lines"]
    target_language = snapshot["effective_request"]["dialogue"]["target_language"]

    # The real voiceless model decision, mechanically re-keyed to patch shape.
    patch = {
        "people": {
            person_id: {
                "voice_id": (record.get("voice") or {}).get("voice_id"),
            }
            for person_id, record in enhanced["people"].items()
        }
    }
    fallback_voice_id = pipeline_instance.voice_library.default_voice(
        target_language
    )["voice_id"]
    project = tmp_path / "project"
    (
        metadata_by_id, references, paths, fallback_used,
    ) = pipeline_instance._materialize_voice_references(
        project, patch, fallback_voice_id=fallback_voice_id,
    )
    assert fallback_used, "voiceless patch must trigger the catalog fallback"
    assert len(paths) == 1 and paths[0].is_file()
    assert references[0]["voice_id"] == fallback_voice_id
    assert references[0]["asset_path"] == "work/voice-references/audio-01.wav"

    # Replay _run_dialogue_audio's voice resolution over the real lines:
    # every line must resolve to a voice instead of raising
    # "dialogue TTS requires a voice reference".
    valid_voices = {path.resolve() for path in paths}
    default_voice = paths[0].resolve()
    people = enhanced["people"]
    dialogue_voices = enhanced["dialogue_voices"]
    assert len(dialogue_voices) == len(translated)
    for line, speaker in zip(translated, dialogue_voices):
        assert line["text"].strip()
        record = people.get(speaker) if isinstance(speaker, str) else None
        voice = record.get("voice") if isinstance(record, dict) else None
        asset_path = voice.get("asset_path") if isinstance(voice, dict) else None
        resolved = None
        if isinstance(asset_path, str):
            candidate = (project / asset_path).resolve()
            if candidate in valid_voices:
                resolved = candidate
        if resolved is None:
            resolved = default_voice
        assert resolved is not None and resolved.is_file()


def test_pre_picsart_resume_backfills_voice_for_the_boundary_snapshot(
    pipeline_instance: Pipeline,
) -> None:
    """Resume a project frozen before the fallback existed: empty
    voice_references plus the snapshot's four waiting dialogue lines must
    materialize Audio 1 instead of failing at the TTS barrier again."""
    snapshot = _state_snapshot()
    pipeline = pipeline_instance
    store = pipeline.store
    project_id, project, _state = store.create(
        snapshot["effective_request"], "boundary-resume",
    )
    segment_dir = project / "work" / "segments" / "0001"
    segment_dir.mkdir(parents=True)
    source = segment_dir / "source.mp4"
    source.write_bytes(b"boundary-source")
    prepared_segments = [{
        "index": 1,
        "start_s": 0.0,
        "end_s": 5.0,
        "duration_s": 5.0,
        "join_mode": "hard_cut",
        "visual_prompt": "boundary visual",
        "dialogue": [],
        "source": source,
        "status": "running",
    }]
    store.update(
        project_id,
        translated_voice_lines=snapshot["translated_voice_lines"],
        voice_references=snapshot["voice_references"],
        object_replace={
            "version": OBJECT_REPLACE_OUTPUT_VERSION,
            "input_sha256": "a" * 64,
        },
    )
    checkpoint = {
        "replacement_sheet": {"upstream_bindings": None},
        "enhanced_element_index": snapshot["enhanced_element_index"],
        "voice_reference_paths": [],
    }
    log = ResearchLog(project)

    pipeline._backfill_resume_voice_fallback(
        project_id,
        project,
        snapshot["effective_request"],
        prepared_segments,
        None,
        checkpoint,
        log,
    )

    fallback_voice_id = pipeline.voice_library.default_voice(
        snapshot["effective_request"]["dialogue"]["target_language"]
    )["voice_id"]
    assert [path.name for path in checkpoint["voice_reference_paths"]] == [
        "audio-01.wav"
    ]
    assert checkpoint["voice_reference_paths"][0].is_file()
    state = store.read(project_id)
    assert [reference["voice_id"] for reference in state["voice_references"]] == [
        fallback_voice_id
    ]
    fallback_metadata = state["object_replace"]["voice_reference_fallback"]
    assert fallback_metadata["voice_id"] == fallback_voice_id
    assert fallback_metadata["reason"].startswith("resume:")
    bindings = state["replacement_sheet"]["upstream_bindings"]
    assert bindings["voice_references_sha256"] == Pipeline._json_sha256(
        state["voice_references"]
    )
    # The backfilled state reloads through the unchanged frozen Audio 1..N
    # validation, so the resumed run reaches TTS with a real voice.
    assert Pipeline._stored_voice_references(
        project, store.read(project_id),
    ) == [checkpoint["voice_reference_paths"][0].resolve()]
    events = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    fallback_events = [
        event
        for event in events
        if event["event"] == "voice_reference_default_fallback"
    ]
    assert len(fallback_events) == 1
    assert fallback_events[0]["voice_id"] == fallback_voice_id

    # No waiting dialogue -> no backfill (the second call is also a no-op
    # because the voice reference is now in place).
    pipeline._backfill_resume_voice_fallback(
        project_id,
        project,
        snapshot["effective_request"],
        prepared_segments,
        None,
        checkpoint,
        log,
    )
    assert len(store.read(project_id)["voice_references"]) == 1
