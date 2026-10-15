"""Boundary replays: historical failure states that must no longer break a run.

The dfe64795 project failed with ``dialogue TTS requires a voice reference``:
its Object Replace output deliberately left every person voiceless while four
translated dialogue lines waited for TTS.  The voice-file requirement is gone
entirely now — Object Replace binds each line to a Doubao ``voice_type`` from
the language-filtered catalog subset, and an unbound (``null``) line falls
back to the language default at synthesis time.  These tests replay that
exact upstream state (harvested from the project's state snapshot, no mocks).
"""

from __future__ import annotations

import json

import pytest

from app.doubao_voices import is_calibrated_voice
from app.pipeline import Pipeline

pytestmark = pytest.mark.node_replay


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
    # fixture 记录的是失败时刻的历史错误码（2026-09-07 当时代码写
    # pre_picsart_resume_failed）；现行代码在同一收口改名写
    # pre_submit_resume_failed。两者都是同一条失败语义，随 fixtures 来源
    # 年代接受其一。
    assert snapshot["error"]["code"] in {
        "pre_picsart_resume_failed",
        "pre_submit_resume_failed",
    }
    people = snapshot["enhanced_element_index"]["people"]
    assert people, "snapshot lost the people records"
    assert all(
        not (record.get("voice") or {}).get("voice_id")
        for record in people.values()
    ), "snapshot no longer represents the voiceless boundary"
    assert len(snapshot["translated_voice_lines"]) == 4
    assert snapshot["voice_references"] == []


def test_voiceless_dialogue_lines_resolve_to_a_catalog_voice(
    pipeline_instance: Pipeline,
) -> None:
    """Replay _run_dialogue_audio's voice resolution over the real lines:
    every unbound line falls back to the dialogue language's default
    Doubao voice_type instead of failing closed."""
    snapshot = _state_snapshot()
    catalog = pipeline_instance.voice_catalog
    translated = snapshot["translated_voice_lines"]
    target_language = snapshot["effective_request"]["dialogue"]["target_language"]

    dialogue_voices = snapshot["enhanced_element_index"]["dialogue_voices"]
    assert len(dialogue_voices) == len(translated)
    default_voice_type = catalog.default_voice_type(target_language)
    for line, binding in zip(translated, dialogue_voices):
        assert line["text"].strip()
        assert binding is None  # the historical voiceless decision
        # 与 app/pipeline.py 消费侧同一口径：目录外/未标定绑定回落默认音色。
        voice_type = (
            binding
            if catalog.has(binding) and is_calibrated_voice(binding)
            else default_voice_type
        )
        assert catalog.has(voice_type)
        assert is_calibrated_voice(voice_type)


def test_voiceless_frozen_state_needs_no_voice_backfill(
    pipeline_instance: Pipeline,
) -> None:
    """v5 起 voice_references 恒为空：快照的空列表直接通过续跑校验，
    没有任何音色物化或回填步骤。"""
    store = pipeline_instance.store
    project_id, project, _state = store.create(
        _state_snapshot()["effective_request"], "boundary-resume",
    )
    snapshot = _state_snapshot()
    store.update(project_id, voice_references=snapshot["voice_references"])

    assert Pipeline._stored_voice_references(
        project, store.read(project_id),
    ) == []
