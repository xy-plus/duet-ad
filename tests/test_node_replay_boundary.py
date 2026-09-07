"""Boundary replays: historical failure states that must no longer break a run.

The dfe64795 project failed with ``dialogue TTS requires a voice reference``:
its Object Replace output deliberately left every person voiceless while four
translated dialogue lines waited for TTS.  The current pipeline materializes a
deterministic catalog fallback voice instead.  These tests replay that exact
upstream state (harvested from the project's state snapshot, no mocks).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline import Pipeline

from node_replay_support import BOUNDARY_PROJECT, FIXTURES_ROOT


def _state_snapshot() -> dict:
    path = (
        FIXTURES_ROOT / "boundary" / BOUNDARY_PROJECT / "state-snapshot.json"
    )
    if not path.is_file():
        pytest.skip(
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
