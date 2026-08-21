from pathlib import Path

import pytest

from app import asr, pipeline
from app.config import Settings, get_settings


def test_whisper_full_json_becomes_multilingual_timed_lines():
    payload = {
        "transcription": [
            {
                "text": " ¿Tu perro tiene nudos y demasiado pelo suelto? ",
                "offsets": {"from": 0, "to": 3460},
            },
            {
                "text": "Este peine de doble diente es la solución.",
                "offsets": {"from": 3840, "to": 6670},
            },
            {
                "text": "Desenreda suavemente y sin tirones.",
                "offsets": {"from": 7080, "to": 9340},
            },
        ]
    }

    assert asr._lines_from_json(payload, 10.0) == [
        {
            "text": "¿Tu perro tiene nudos y demasiado pelo suelto?",
            "start_s": 0.0,
            "end_s": 3.46,
        },
        {
            "text": "Este peine de doble diente es la solución.",
            "start_s": 3.84,
            "end_s": 6.67,
        },
        {
            "text": "Desenreda suavemente y sin tirones.",
            "start_s": 7.08,
            "end_s": 9.34,
        },
    ]


def test_keep_mode_uses_configured_local_asr_not_codex(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-small.bin"
    cli.write_bytes(b"binary")
    model.write_bytes(b"model")
    work = tmp_path / "work"
    work.mkdir()
    (work / "voice.mp3").write_bytes(b"audio")
    settings = Settings(access_token="test", asr_cli=cli, asr_model=model)
    expected = [{"text": "Hola mundo.", "start_s": 0.2, "end_s": 1.1}]
    calls = []

    def fake_transcribe(audio: Path, **kwargs):
        calls.append((audio, kwargs))
        return expected

    class NoCodex:
        def run_voice(self, *_args, **_kwargs):
            pytest.fail("keep mode must not use Codex when local ASR is configured")

    monkeypatch.setattr(asr, "transcribe", fake_transcribe)
    assert pipeline._transcribe_voice_attempt(
        settings, NoCodex(), work, "unused", 2.0, "keep"
    ) == expected
    assert calls[0][0] == work / "voice.mp3"
    assert calls[0][1]["model"] == model


def test_half_configured_local_asr_fails_closed(tmp_path):
    settings = Settings(access_token="test", asr_cli=tmp_path / "whisper-cli")
    with pytest.raises(pipeline.PipelineError, match="configuration incomplete"):
        pipeline._transcribe_voice_attempt(
            settings, object(), tmp_path, "unused", 2.0, "keep"
        )


def test_local_asr_retries_transient_timeout(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-small.bin"
    cli.write_bytes(b"binary")
    model.write_bytes(b"model")
    work = tmp_path / "work"
    work.mkdir()
    (work / "voice.mp3").write_bytes(b"audio")
    settings = Settings(
        access_token="test",
        asr_cli=cli,
        asr_model=model,
        retry_interval_s=0,
    )
    expected = [{"text": "Hola.", "start_s": 0.2, "end_s": 1.1}]
    calls = 0

    def flaky_transcribe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise asr.ASRError("asr_timeout")
        return expected

    monkeypatch.setattr(asr, "transcribe", flaky_transcribe)
    lines, unrecognized = pipeline._transcribe_voice_with_retry(
        settings,
        object(),
        work,
        "unused",
        2.0,
        "keep",
        has_vocal=False,
    )

    assert lines == expected
    assert unrecognized is False
    assert calls == 3


def test_production_config_defaults_to_pinned_multilingual_small(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.delenv("ASR_CLI", raising=False)
    monkeypatch.delenv("ASR_MODEL", raising=False)
    settings = get_settings()
    assert settings.asr_cli.name == "whisper-cli"
    assert settings.asr_model == Path(
        "/home/xy/.local/share/duet-asr/ggml-small.bin"
    )
    assert settings.asr_timeout_s == 180
    assert settings.asr_threads == 4
