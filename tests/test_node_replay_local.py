"""Replay local (deterministic, offline) pipeline nodes with real inputs.

Every test loads a harvested ``input.json`` + ``input-media/`` that the node
actually consumed in a historical project, calls the same ``app.media`` /
``app.face_mask`` / ``app.replacement_sheet`` function the pipeline node
wraps, and asserts the output passes the node's own validation and the gate
its downstream consumer applies.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from app import face_mask, media
from app.pipeline import Pipeline
from app.replacement_sheet import (
    materialize_replacement_sheet_inputs,
    validate_replacement_sheet_spec,
)

from node_replay_support import (
    FIXTURES_ROOT,
    REQUIRED_NODES,
    Fixture,
    discover,
    media_by_suffix,
    node_fixtures,
    sha256_file,
    sibling_fixture,
)


def _audio_duration_s(path: Path) -> float:
    """ffprobe container duration for audio-only files (media.probe needs video)."""
    result = media.run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    return float(json.loads(result["stdout"])["format"]["duration"])


def _params(node: str) -> list[Fixture]:
    return node_fixtures(node)


def _ids(fixtures: list[Fixture]) -> list[str]:
    return [fixture.test_id for fixture in fixtures]


def test_fixture_coverage_complete() -> None:
    """Every required node has at least one harvested real input.

    Fails closed: a missing/stale fixture directory must block the E2E gate,
    not silently skip.  Run ``scripts/harvest_node_fixtures.py`` to fix.
    """
    if not FIXTURES_ROOT.is_dir() or not (FIXTURES_ROOT / "manifest.json").is_file():
        pytest.fail(
            "node replay fixtures are missing; run "
            "`/home/xy/duet-ad1/.venv/bin/python scripts/harvest_node_fixtures.py`"
        )
    covered = {fixture.node for fixture in discover()}
    missing = [node for node in REQUIRED_NODES if node not in covered]
    assert not missing, f"nodes without any harvested fixture: {missing}"


# ---------------------------------------------------------------- probe


@pytest.mark.parametrize("fixture", _params("probe"), ids=_ids(_params("probe")))
def test_probe(fixture: Fixture) -> None:
    source = media_by_suffix(fixture, "source.mp4")
    info = media.probe(source)
    assert info["duration_s"] > 0
    assert info["width"] > 0 and info["height"] > 0
    assert isinstance(info["has_audio"], bool)
    expected = fixture.expected_output
    if isinstance(expected, dict) and expected.get("duration_s"):
        # ffprobe is deterministic for the same file build.
        assert math.isclose(
            info["duration_s"], expected["duration_s"], abs_tol=0.05
        )


# ------------------------------------------------------- scene detection


@pytest.mark.parametrize(
    "fixture", _params("scene-detection"), ids=_ids(_params("scene-detection"))
)
def test_scene_detection(fixture: Fixture) -> None:
    source = media_by_suffix(fixture, "source.mp4")
    duration_s = media.probe(source)["duration_s"]
    scenes = media.detect_scenes(source, duration_s)
    assert scenes, "scene inventory is empty"
    # Downstream gate: media.plan_segments requires populated frame lists.
    assert all(scene.get("frames") for scene in scenes)


# ------------------------------------------------------ segment planning


@pytest.mark.parametrize(
    "fixture", _params("segment-planning"), ids=_ids(_params("segment-planning"))
)
def test_segment_planning(fixture: Fixture) -> None:
    payload = fixture.input
    segments = media.plan_segments(
        payload["scenes"], payload["duration_s"], payload.get("dialogue") or [],
    )
    assert segments, "segment plan is empty"
    # Downstream gate: the pipeline validates join modes before keyframes.
    Pipeline._validated_join_modes(segments)
    expected = fixture.expected_output
    if isinstance(expected, list) and expected:
        # plan_segments is a pure function of its input.
        assert segments == expected


# ---------------------------------------------------------- segment cut


@pytest.mark.parametrize(
    "fixture", _params("segment-cut"), ids=_ids(_params("segment-cut"))
)
def test_segment_cut(fixture: Fixture, tmp_path: Path) -> None:
    payload = fixture.input
    source = media_by_suffix(fixture, "source.mp4")
    destination = tmp_path / "segment.mp4"
    media.cut_segment(
        source, float(payload["start_s"]), float(payload["duration_s"]),
        destination,
    )
    info = media.probe(destination)
    assert math.isclose(
        info["duration_s"], float(payload["duration_s"]), abs_tol=1.0
    )


# ------------------------------------------------------- frame extraction


@pytest.mark.parametrize(
    "fixture",
    _params("source-frames-4fps") + _params("segment-frames-4fps"),
    ids=_ids(_params("source-frames-4fps") + _params("segment-frames-4fps")),
)
def test_extract_frames_4fps(fixture: Fixture, tmp_path: Path) -> None:
    payload = fixture.input
    source = fixture.media[0]
    assert source.suffix == ".mp4"
    destination = tmp_path / "frames"
    manifest = media.extract_frames_4fps(source, destination)
    assert manifest["duration_seconds"] > 0
    assert manifest["frame_count"] > 0
    frames = manifest["frames"]
    assert frames, "frame manifest is empty"
    for record in frames:
        assert (destination / record["file"]).is_file()


# ------------------------------------------------------------ keyframes


@pytest.mark.parametrize(
    "fixture",
    _params("keyframe-selection-exact-nine"),
    ids=_ids(_params("keyframe-selection-exact-nine")),
)
def test_keyframe_selection_exact_nine(fixture: Fixture) -> None:
    payload = fixture.input
    selection = media.select_segment_keyframes(
        payload["scenes"], payload["segment"],
    )
    # Downstream gate: media.materialize_keyframes validates this shape.
    assert len(selection) == 9
    for order, item in enumerate(selection, 1):
        assert item["order"] == order
        decode_index = item["decode_frame_index"]
        assert isinstance(decode_index, int) and decode_index >= 0
    expected = fixture.expected_output
    if isinstance(expected, list) and expected:
        assert selection == expected


@pytest.mark.parametrize(
    "fixture",
    _params("keyframes-exact-nine"),
    ids=_ids(_params("keyframes-exact-nine")),
)
def test_keyframes_exact_nine(fixture: Fixture, tmp_path: Path) -> None:
    source = media_by_suffix(fixture, "source.mp4")
    destination = tmp_path / "keyframes"
    destination.mkdir()
    paths = media.materialize_keyframes(
        source, fixture.input["selection"], destination,
    )
    assert len(paths) == 9
    for path in paths:
        assert path.is_file() and path.suffix == ".png"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


# ---------------------------------------------------------------- audio


@pytest.mark.parametrize(
    "fixture", _params("extract-audio"), ids=_ids(_params("extract-audio"))
)
def test_extract_audio(fixture: Fixture, tmp_path: Path) -> None:
    source = media_by_suffix(fixture, "source.mp4")
    destination = tmp_path / "source.wav"
    media.extract_audio(source, destination)
    assert destination.is_file()
    assert _audio_duration_s(destination) > 0


@pytest.mark.parametrize(
    "fixture", _params("whisper-asr"), ids=_ids(_params("whisper-asr"))
)
def test_whisper_asr(fixture: Fixture, tmp_path: Path) -> None:
    payload = fixture.input
    audio = media_by_suffix(fixture, "source.wav")
    cli = Path(payload["cli"])
    model = Path(payload["model"])
    if not cli.is_file() or not model.is_file():
        pytest.skip(f"whisper.cpp assets missing: {cli} / {model}")
    lines, record = media.transcribe(audio, tmp_path / "voice_lines.json", cli, model)
    assert isinstance(lines, list)
    for line in lines:
        assert isinstance(line["text"], str)
        assert 0 <= float(line["start_s"]) <= float(line["end_s"])
    # The pipeline accepts empty ASR even under vocal signal
    # (empty_asr_accepted), so only the shape is gated here.
    # Downstream gate: the spoken-dialogue filter must accept the lines.
    yamnet = sibling_fixture(fixture, "yamnet")
    if yamnet is not None and isinstance(yamnet.expected_output, dict):
        media.filter_spoken_lines(lines, yamnet.expected_output)
    assert isinstance(record, dict)


@pytest.mark.parametrize("fixture", _params("yamnet"), ids=_ids(_params("yamnet")))
def test_yamnet(fixture: Fixture) -> None:
    audio = media_by_suffix(fixture, "source.wav")
    from app.config import ROOT

    model = ROOT / "models" / "yamnet.tflite"
    if not model.is_file():
        pytest.skip(f"YAMNet model missing: {model}")
    analysis = media.yamnet_analyze(audio, model)
    assert isinstance(analysis["windows"], list)
    for window in analysis["windows"]:
        assert {"start_ms", "end_ms", "sung", "spoken", "music"} <= set(window)
    assert isinstance(analysis["has_bgm"], bool)
    # Downstream gate: the filter must accept this analysis mapping.
    whisper = sibling_fixture(fixture, "whisper-asr")
    if whisper is not None and isinstance(whisper.expected_output, list):
        media.filter_spoken_lines(whisper.expected_output, analysis)


@pytest.mark.parametrize(
    "fixture",
    _params("spoken-dialogue-filter"),
    ids=_ids(_params("spoken-dialogue-filter")),
)
def test_spoken_dialogue_filter(fixture: Fixture) -> None:
    payload = fixture.input
    spoken, decisions = media.filter_spoken_lines(
        payload["asr_lines"], payload["yamnet"],
    )
    assert len(decisions) == len(payload["asr_lines"])
    for line in spoken:
        assert isinstance(line["text"], str) and line["text"].strip()
        assert 0 <= float(line["start_s"]) < float(line["end_s"])
    expected = fixture.expected_output
    if isinstance(expected, list) and len(expected) == 2:
        # (spoken, decisions) is deterministic for the same inputs.
        assert [spoken, decisions] == expected


# -------------------------------------------------- reference video prep


@pytest.mark.parametrize(
    "fixture",
    _params("mute-reference-video"),
    ids=_ids(_params("mute-reference-video")),
)
def test_mute_reference_video(fixture: Fixture, tmp_path: Path) -> None:
    source = fixture.media[0]
    destination = tmp_path / "muted.mp4"
    media.mute(source, destination)
    info = media.probe(destination)
    assert info["has_audio"] is False
    assert info["duration_s"] > 0


@pytest.mark.parametrize(
    "fixture",
    _params("mask-reference-faces"),
    ids=_ids(_params("mask-reference-faces")),
)
def test_mask_reference_faces(fixture: Fixture, tmp_path: Path) -> None:
    source = fixture.media[0]
    destination = tmp_path / "masked.mp4"
    result = face_mask.mask_faces(source, destination)
    assert destination.is_file()
    source_info = media.probe(source)
    masked_info = media.probe(destination)
    assert masked_info["width"] == source_info["width"]
    assert masked_info["height"] == source_info["height"]
    assert isinstance(result, dict)


# -------------------------------------------------------- dialogue track


@pytest.mark.parametrize(
    "fixture",
    _params("assemble-dialogue-track"),
    ids=_ids(_params("assemble-dialogue-track")),
)
def test_assemble_dialogue_track(fixture: Fixture, tmp_path: Path) -> None:
    payload = fixture.input
    lines = [
        {
            "audio": media_by_suffix(fixture, entry["audio"]),
            "start_s": entry["start_s"],
            "end_s": entry["end_s"],
        }
        for entry in payload["lines"]
    ]
    destination = tmp_path / "full-track.wav"
    media.assemble_dialogue_track(
        lines, float(payload["total_duration_s"]), destination,
    )
    assert destination.is_file()
    duration_s = _audio_duration_s(destination)
    assert math.isclose(
        duration_s, float(payload["total_duration_s"]), abs_tol=0.75
    )


@pytest.mark.parametrize(
    "fixture",
    _params("replace-audio-track"),
    ids=_ids(_params("replace-audio-track")),
)
def test_replace_audio_track(fixture: Fixture, tmp_path: Path) -> None:
    video = media_by_suffix(fixture, "generated.mp4")
    audio = media_by_suffix(fixture, "full-track.wav")
    destination = tmp_path / "generated.mp4"
    media.replace_audio_track(video, audio, destination)
    info = media.probe(destination)
    assert info["has_audio"] is True
    assert math.isclose(
        info["duration_s"], media.probe(video)["duration_s"], abs_tol=0.5
    )


# ---------------------------------------------------------------- stitch


@pytest.mark.parametrize("fixture", _params("stitch"), ids=_ids(_params("stitch")))
def test_stitch(fixture: Fixture, tmp_path: Path) -> None:
    payload = fixture.input
    segments = fixture.media
    assert len(segments) == len(payload["segments"])
    destination = tmp_path / "generated.mp4"
    result = media.concat(
        segments,
        destination,
        target_durations=payload["target_durations"],
        join_modes=payload["join_modes"],
    )
    assert destination.is_file()
    info = media.probe(destination)
    assert math.isclose(
        info["duration_s"], result["duration_s"], abs_tol=0.5
    )


# ------------------------------------------------------ replacement sheet


def _user_image_present(fixture: Fixture) -> bool:
    seedream = sibling_fixture(fixture, "seedream-reference-submit")
    if seedream is not None:
        receipts = seedream.input.get("input_receipts") or []
        return len(receipts) > 1
    expected = fixture.expected_output
    if isinstance(expected, dict):
        return "Image 2" in str(expected.get("prompt") or "")
    return False


@pytest.mark.parametrize(
    "fixture",
    _params("replacement-sheet-inputs"),
    ids=_ids(_params("replacement-sheet-inputs")),
)
def test_replacement_sheet_inputs(fixture: Fixture, tmp_path: Path) -> None:
    spec = fixture.input["spec"]
    # The node's own schema gate.
    validate_replacement_sheet_spec(spec, allow_empty=False)
    directory = tmp_path / "sheet"
    result = materialize_replacement_sheet_inputs(
        directory, spec, user_image_present=_user_image_present(fixture),
    )
    assert result["spec_sha256"] == spec["spec_sha256"]
    canvas = Path(result["canvas_path"])
    assert canvas.is_file()
    assert canvas.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["prompt"].strip()
    # Downstream gate: the Seedream submit binding must reproduce byte-exact.
    seedream = sibling_fixture(fixture, "seedream-reference-submit")
    if seedream is not None:
        receipts = seedream.input["input_receipts"]
        assert sha256_file(canvas) == receipts[0]["sha256"]
        assert (
            hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest()
            == seedream.input["prompt_sha256"]
        )


@pytest.mark.parametrize(
    "fixture",
    _params("replacement-sheet-user-reference"),
    ids=_ids(_params("replacement-sheet-user-reference")),
)
def test_replacement_sheet_user_reference(
    fixture: Fixture, tmp_path: Path,
) -> None:
    source = fixture.media[0]
    destination = tmp_path / "user-reference.png"
    media.provider_reference_png(source, destination)
    payload = destination.read_bytes()
    # The pipeline's own gate on the materialized reference.
    assert payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 9
    # Downstream gate: the Seedream submit binding expects this exact bytes.
    seedream = sibling_fixture(fixture, "seedream-reference-submit")
    if seedream is not None:
        receipts = seedream.input["input_receipts"]
        if len(receipts) > 1:
            assert sha256_file(destination) == receipts[1]["sha256"]


# -------------------------------------------------------- dialogue-localize


@pytest.mark.parametrize(
    "project",
    sorted({f.project for f in _params("dialogue-localize")}),
)
def test_dialogue_localize(project: str) -> None:
    fixtures = [
        fixture
        for fixture in _params("dialogue-localize")
        if fixture.project == project
    ]
    fixtures.sort(key=lambda fixture: int(fixture.input["segment"]["index"]))
    segments = [fixture.input["segment"] for fixture in fixtures]
    for fixture in fixtures:
        payload = fixture.input
        localized = media.localize_dialogue(
            payload["translated_lines"], payload["segment"], segments=segments,
        )
        duration = (
            float(payload["segment"]["end_s"])
            - float(payload["segment"]["start_s"])
        )
        for line in localized:
            assert isinstance(line["text"], str) and line["text"]
            assert 0 <= line["start_s"] < line["end_s"] <= duration + 0.01
        expected = fixture.expected_output
        if isinstance(expected, list):
            # localize_dialogue is a pure function of its input.
            assert localized == expected
