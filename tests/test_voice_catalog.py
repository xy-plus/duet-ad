from __future__ import annotations

import hashlib

import pytest

from app.voice_catalog import (
    VOICE_CATALOG_SIZE,
    VOICE_SELECTION_OUTPUT_NAME,
    load_voice_library,
    normalize_voice_selection,
    voice_selection_schema,
)


def test_voice_library_is_exact_bounded_and_compact() -> None:
    library = load_voice_library()

    assert len(library.voices) == VOICE_CATALOG_SIZE == 30
    assert len(library.files) == 30
    for voice in library.voices:
        path = library.files[voice["voice_id"]]
        assert 2.0 <= voice["duration_s"] <= 5.0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == voice["sha256"]
        assert voice["features"]["personality"] == "not_reported"

    compact = library.model_lines()
    lines = compact.splitlines()
    assert len(lines) == 30
    assert len(compact) < 3500
    assert max(map(len, lines)) <= 160
    assert "{" not in compact and "[" not in compact
    assert "not_reported" not in compact
    assert all(line.count(" - ") >= 3 for line in lines)
    assert all(voice["file"] in line for voice, line in zip(library.voices, lines))


def test_voice_selection_schema_binds_input_and_catalog() -> None:
    payload = {"voice_catalog_lines": "voice-a - voices/voice-a.wav - English - male"}
    schema, input_sha256 = voice_selection_schema(payload, ["voice-a", "voice-b"])

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema"]["enum"] == [VOICE_SELECTION_OUTPUT_NAME]
    assert schema["properties"]["voice_id"]["enum"] == ["voice-a", "voice-b"]
    value = {
        "schema": VOICE_SELECTION_OUTPUT_NAME,
        "version": 1,
        "input_sha256": input_sha256,
        "voice_id": "voice-b",
    }
    assert normalize_voice_selection(
        value, input_sha256=input_sha256, voice_ids={"voice-a", "voice-b"},
    ) == "voice-b"

    with pytest.raises(ValueError, match="not bound"):
        normalize_voice_selection(
            {**value, "voice_id": "voice-c"},
            input_sha256=input_sha256,
            voice_ids={"voice-a", "voice-b"},
        )
