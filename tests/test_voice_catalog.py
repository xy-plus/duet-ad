from __future__ import annotations

import copy
import hashlib
import json
import shutil
import wave

import pytest

from app.voice_catalog import (
    VOICE_CATALOG_NAME,
    VOICE_CATALOG_VERSION,
    VOICE_SELECTION_OUTPUT_NAME,
    load_voice_library,
    normalize_voice_selection,
    voice_selection_schema,
)


def test_voice_library_assets_are_usable_and_model_lines_are_transportable() -> None:
    library = load_voice_library()

    assert library.voices
    assert len(library.files) == len(library.voices)
    for voice in library.voices:
        path = library.files[voice["voice_id"]]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == voice["sha256"]
        with wave.open(str(path), "rb") as handle:
            duration_s = handle.getnframes() / handle.getframerate()
        assert 2.0 <= duration_s <= 5.0

    lines = library.model_lines().splitlines()
    assert len(lines) == len(library.voices)
    assert all(line.count(" - ") >= 3 for line in lines)
    assert all(voice["file"] in line for voice, line in zip(library.voices, lines))


def test_voice_library_does_not_review_catalog_size_or_feature_labels(tmp_path) -> None:
    source_library = load_voice_library()
    source_voice = source_library.voices[0]
    source_path = source_library.files[source_voice["voice_id"]]
    target_path = tmp_path / "voice.wav"
    shutil.copyfile(source_path, target_path)

    voice = copy.deepcopy(source_voice)
    voice.update({
        "voice_id": "custom-voice",
        "file": "voice.wav",
        "duration_s": 99.0,
        "source": {"publisher_labels": "passed through without review"},
        "extra_metadata": "allowed",
    })
    voice["features"] = {
        "age": 12,
        "personality": "confident",
        "custom_label": "publisher supplied",
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema": VOICE_CATALOG_NAME,
        "version": VOICE_CATALOG_VERSION,
        "voices": [voice],
        "extra_metadata": "allowed",
    }), encoding="utf-8")

    library = load_voice_library(catalog_path)

    assert len(library.voices) == 1
    assert library.voices[0]["features"]["age"] == 12
    assert library.voices[0]["features"]["personality"] == "confident"
    assert "unspecified - unspecified" in library.model_lines()


def test_voice_library_keeps_actual_provider_duration_boundary(tmp_path) -> None:
    audio_path = tmp_path / "voice.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "schema": VOICE_CATALOG_NAME,
        "version": VOICE_CATALOG_VERSION,
        "voices": [{
            "voice_id": "too-short",
            "file": "voice.wav",
            "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "duration_s": 3.0,
            "media_type": "audio/wav",
            "source": {},
            "features": {},
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="between 2 and 5 seconds"):
        load_voice_library(catalog_path)


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
