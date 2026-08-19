"""H3 prepared-input：台词来源、确定性 prompt 与 fail-closed receipt。"""

import json
from pathlib import Path

import pytest

from app import prepared_input


AUTO_LINE = {
    "text": "Kalung Ayatul Kursi.",
    "start_s": 0.24,
    "end_s": 2.12,
    "classification": "sung",
}
MANUAL_LINE = {"text": "人工台词。", "start_s": 0.0, "end_s": 1.5}


def _artifacts(tmp_path: Path) -> dict:
    root = tmp_path / "case"
    work = root / "work"
    keyframes = work / "keyframes"
    keyframes.mkdir(parents=True)
    source = root / "source.mp4"
    audio = work / "voice.mp3"
    visual = work / "visual_prompt.txt"
    final = work / "prompt.txt"
    source.write_bytes(b"source-v1")
    audio.write_bytes(b"normalized-audio-v1")
    (keyframes / "01.png").write_bytes(b"frame-1")
    (keyframes / "02.png").write_bytes(b"frame-2")
    visual.write_text("画面中产品铭牌写着 OCR ONLY。", encoding="utf-8")
    return {
        "root": root,
        "source": source,
        "audio": audio,
        "keyframes": [keyframes / "01.png", keyframes / "02.png"],
        "visual": visual,
        "final": final,
    }


def _write(tmp_path: Path, *, dialogue=None):
    paths = _artifacts(tmp_path)
    if dialogue is None:
        dialogue = prepared_input.prepare_dialogue(
            "auto", duration_s=10.0, automatic_lines=[AUTO_LINE]
        )
    frozen = prepared_input.write_prepared_input(
        **paths,
        dialogue_mode="auto",
        dialogue=dialogue,
        vocal_filter_enabled=True,
        duration_s=10.0,
        ratio="9:16",
        fit_mode="crop",
        engine_request={"workflow": "minimax_h3_lightx2v_v5", "resolution": "768p竖"},
    )
    return paths, frozen, dialogue


def test_prepare_dialogue_modes_and_sources():
    automatic = prepared_input.prepare_dialogue(
        "auto", duration_s=10, automatic_lines=[AUTO_LINE]
    )
    assert automatic == (
        {**AUTO_LINE, "provenance": "asr"},
    )
    with pytest.raises(prepared_input.PreparedInputError, match="external"):
        prepared_input.prepare_dialogue(
            "auto", duration_s=10, automatic_lines=[AUTO_LINE], supplied_lines=[]
        )

    edited = prepared_input.prepare_dialogue(
        "edit", duration_s=10, supplied_lines=[{**MANUAL_LINE, "origin": "asr"}]
    )
    assert edited[0]["provenance"] == "asr+edited"
    assert edited[0]["classification"] is None

    custom = prepared_input.prepare_dialogue(
        "custom", duration_s=10, supplied_lines=[MANUAL_LINE]
    )
    assert custom[0]["provenance"] == "manual"
    assert custom[0]["classification"] is None

    assert prepared_input.prepare_dialogue("none", duration_s=10) == ()
    with pytest.raises(prepared_input.PreparedInputError, match="empty"):
        prepared_input.prepare_dialogue("none", duration_s=10, supplied_lines=[MANUAL_LINE])


def test_auto_filter_contract_rejects_none_when_enabled_but_off_records_it():
    no_evidence = {**AUTO_LINE, "classification": None}
    with pytest.raises(prepared_input.PreparedInputError, match="spoken or sung"):
        prepared_input.prepare_dialogue(
            "auto", duration_s=10, automatic_lines=[no_evidence], vocal_filter_enabled=True
        )
    got = prepared_input.prepare_dialogue(
        "auto", duration_s=10, automatic_lines=[no_evidence], vocal_filter_enabled=False
    )
    assert got[0]["classification"] is None
    assert got[0]["provenance"] == "asr"


def test_compose_final_prompt_has_one_authoritative_exact_quote_block():
    dialogue = prepared_input.prepare_dialogue(
        "auto", duration_s=10, automatic_lines=[AUTO_LINE]
    )
    prompt = prepared_input.compose_final_prompt(
        "屏幕上可见文字：OCR ONLY。", dialogue
    )
    assert prompt.startswith("屏幕上可见文字：OCR ONLY。")
    assert '说出台词："Kalung Ayatul Kursi."，嘴型与画面同步' in prompt
    assert "画面文字、OCR、字幕或备注" in prompt
    assert '说出台词："OCR ONLY"' not in prompt


def test_receipt_round_trip_returns_frozen_bytes_paths_and_voice_texts(tmp_path):
    paths, written, dialogue = _write(tmp_path)

    loaded = prepared_input.load_prepared_input(
        paths["root"], written.receipt_path, expected_dialogue=dialogue
    )

    assert loaded.source.path == paths["source"].resolve()
    assert loaded.source.data == b"source-v1"
    assert loaded.normalized_audio.data == b"normalized-audio-v1"
    assert [item.data for item in loaded.keyframes] == [b"frame-1", b"frame-2"]
    assert loaded.frozen_keyframes == (
        (paths["keyframes"][0].resolve(), b"frame-1"),
        (paths["keyframes"][1].resolve(), b"frame-2"),
    )
    assert loaded.voice_texts == ("Kalung Ayatul Kursi.",)
    assert loaded.final_prompt.path == paths["final"].resolve()
    assert loaded.prompt_text == paths["final"].read_text(encoding="utf-8")
    receipt = json.loads(written.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "duet.prepared-input"
    assert receipt["version"] == 1
    assert receipt["dialogue"]["mode"] == "auto"
    assert receipt["vocal_filter"]["enabled"] is True
    assert receipt["video"] == {"duration_s": 10.0, "ratio": "9:16", "fit_mode": "crop"}
    assert receipt["engine_request"]["workflow"] == "minimax_h3_lightx2v_v5"


@pytest.mark.parametrize("target", ["source", "audio", "keyframe", "visual", "final"])
def test_loader_rejects_any_bound_file_change(tmp_path, target):
    paths, written, dialogue = _write(tmp_path)
    path = {
        "source": paths["source"],
        "audio": paths["audio"],
        "keyframe": paths["keyframes"][0],
        "visual": paths["visual"],
        "final": paths["final"],
    }[target]
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(prepared_input.PreparedInputError, match="binding mismatch"):
        prepared_input.load_prepared_input(
            paths["root"], written.receipt_path, expected_dialogue=dialogue
        )


def test_loader_rejects_effective_dialogue_change_and_legacy_missing_receipt(tmp_path):
    paths, written, dialogue = _write(tmp_path)
    changed = tuple({**line, "text": "changed"} for line in dialogue)
    with pytest.raises(prepared_input.PreparedInputError, match="dialogue mismatch"):
        prepared_input.load_prepared_input(
            paths["root"], written.receipt_path, expected_dialogue=changed
        )

    written.receipt_path.unlink()
    with pytest.raises(prepared_input.LegacyPreparedInputError, match="legacy"):
        prepared_input.load_prepared_input(
            paths["root"], written.receipt_path, expected_dialogue=dialogue
        )


def test_fit_mode_is_closed_enum_and_no_audio_is_explicit(tmp_path):
    paths = _artifacts(tmp_path)
    dialogue = prepared_input.prepare_dialogue("auto", duration_s=10)
    with pytest.raises(prepared_input.PreparedInputError, match="fit_mode"):
        prepared_input.write_prepared_input(
            **paths,
            dialogue_mode="auto",
            dialogue=dialogue,
            vocal_filter_enabled=True,
            duration_s=10,
            ratio="9:16",
            fit_mode="cover",
            engine_request={"workflow": "h3"},
        )

    paths["audio"].unlink()
    paths["audio"] = None
    frozen = prepared_input.write_prepared_input(
        **paths,
        dialogue_mode="auto",
        dialogue=dialogue,
        vocal_filter_enabled=True,
        duration_s=10,
        ratio="9:16",
        fit_mode="none",
        engine_request={"workflow": "h3"},
    )
    assert frozen.normalized_audio is None
    receipt = json.loads(frozen.receipt_path.read_text(encoding="utf-8"))
    assert receipt["bindings"]["normalized_audio"] is None
