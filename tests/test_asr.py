import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
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


def test_whisper_drops_non_speech_labels_and_standalone_fillers_before_output():
    texts = [
        "*tepuk tangan*",
        "[applause]",
        "(music)",
        "<laughter>",
        "♪ instrumental ♪",
        "嗯……",
        "um, uh",
        "hmmm",
        "嗯，我知道了。",
        "Oh no!",
        "1, 2, 3",
    ]
    payload = {
        "transcription": [
            {
                "text": text,
                "offsets": {"from": index * 1000, "to": (index + 1) * 1000},
            }
            for index, text in enumerate(texts)
        ]
    }

    assert asr._lines_from_json(payload, float(len(texts))) == [
        {"text": "嗯，我知道了。", "start_s": 8.0, "end_s": 9.0},
        {"text": "Oh no!", "start_s": 9.0, "end_s": 10.0},
        {"text": "1, 2, 3", "start_s": 10.0, "end_s": 11.0},
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
    with pytest.raises(
        pipeline.PipelineError,
        match="local ASR is required for dialogue transcription",
    ):
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


def test_local_asr_ignores_non_utf8_process_diagnostics(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-small.bin"
    audio = tmp_path / "voice.mp3"
    cli.write_bytes(b"binary")
    model.write_bytes(b"model")
    audio.write_bytes(b"audio")

    def fake_run(argv, **kwargs):
        assert "text" not in kwargs
        if argv[0] == "ffmpeg":
            return subprocess.CompletedProcess(argv, 0, b"", b"\xff")
        output = Path(argv[argv.index("-of") + 1]).with_suffix(".json")
        output.write_text(json.dumps({"transcription": []}), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, b"", b"\xff\xfe")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    assert asr.transcribe(
        audio,
        cli=cli,
        model=model,
        duration_s=2.0,
        timeout_s=600,
        threads=4,
        process_budget=asr.ASRProcessBudget(4),
    ) == []


def test_local_asr_filters_non_dialogue_before_returning_voice_lines(
    tmp_path, monkeypatch
):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-small.bin"
    work = tmp_path / "work"
    work.mkdir()
    audio = work / "voice.mp3"
    cli.write_bytes(b"binary")
    model.write_bytes(b"model")
    audio.write_bytes(b"audio")

    def fake_run(argv, **_kwargs):
        if argv[0] != "ffmpeg":
            output = Path(argv[argv.index("-of") + 1]).with_suffix(".json")
            output.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {
                                "text": "*tepuk tangan*",
                                "offsets": {"from": 0, "to": 1000},
                            },
                            {
                                "text": " um ",
                                "offsets": {"from": 1000, "to": 1500},
                            },
                            {
                                "text": "Produk ini mudah digunakan.",
                                "offsets": {"from": 1500, "to": 3000},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    settings = Settings(access_token="test", asr_cli=cli, asr_model=model)
    assert pipeline._transcribe_voice_attempt(
        settings,
        object(),
        work,
        "unused",
        3.0,
        "keep",
    ) == [
        {
            "text": "Produk ini mudah digunakan.",
            "start_s": 1.5,
            "end_s": 3.0,
        }
    ]


def test_local_asr_repairs_truncated_utf8_token_in_json(tmp_path, monkeypatch):
    cli = tmp_path / "whisper-cli"
    model = tmp_path / "ggml-small.bin"
    audio = tmp_path / "voice.mp3"
    cli.write_bytes(b"binary")
    model.write_bytes(b"model")
    audio.write_bytes(b"audio")

    def fake_run(argv, **_kwargs):
        if argv[0] != "ffmpeg":
            output = Path(argv[argv.index("-of") + 1]).with_suffix(".json")
            output.write_bytes(
                b'{"transcription":[{"text":"Hola\xe0\xb6","offsets":{"from":0,"to":1000}}]}'
            )
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(asr.subprocess, "run", fake_run)
    assert asr.transcribe(
        audio,
        cli=cli,
        model=model,
        duration_s=2.0,
        timeout_s=600,
        threads=4,
        process_budget=asr.ASRProcessBudget(4),
    ) == [{"text": "Hola", "start_s": 0.0, "end_s": 1.0}]


def test_production_config_defaults_to_pinned_multilingual_small(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.delenv("ASR_CLI", raising=False)
    monkeypatch.delenv("ASR_MODEL", raising=False)
    settings = get_settings()
    assert settings.asr_cli.name == "whisper-cli"
    assert settings.asr_model == Path(
        "/home/xy/.local/share/duet-asr/ggml-small.bin"
    )
    assert settings.asr_timeout_s == 600
    assert settings.asr_threads == 4
    assert settings.asr_process_budget.threads == 4
    assert (
        Settings(access_token="second").asr_process_budget
        is settings.asr_process_budget
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _real_asr_fixture(tmp_path: Path, monkeypatch):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"active": 0, "max_active": 0, "calls": 0}),
        encoding="utf-8",
    )
    fail_once = tmp_path / "fail-once"
    _write_executable(
        binary_dir / "ffmpeg",
        """#!/usr/bin/python3
import pathlib
import sys
pathlib.Path(sys.argv[-1]).write_bytes(b"wav")
""",
    )
    cli = binary_dir / "whisper-cli"
    _write_executable(
        cli,
        """#!/usr/bin/python3
import fcntl
import json
import os
import pathlib
import sys
import time

state_path = pathlib.Path(os.environ["DUET_ASR_TEST_STATE"])
with state_path.open("r+", encoding="utf-8") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    state = json.load(stream)
    state["active"] += 1
    state["max_active"] = max(state["max_active"], state["active"])
    stream.seek(0)
    stream.truncate()
    json.dump(state, stream)
    stream.flush()
    fcntl.flock(stream, fcntl.LOCK_UN)

time.sleep(0.15)
failure_marker = os.environ.get("DUET_ASR_TEST_FAIL_ONCE")
failed = False
if failure_marker:
    marker = pathlib.Path(failure_marker)
    try:
        marker.open("x").close()
        failed = True
    except FileExistsError:
        pass
if not failed:
    output = pathlib.Path(sys.argv[sys.argv.index("-of") + 1]).with_suffix(".json")
    output.write_text('{"transcription": []}', encoding="utf-8")

with state_path.open("r+", encoding="utf-8") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    state = json.load(stream)
    state["active"] -= 1
    state["calls"] += 1
    stream.seek(0)
    stream.truncate()
    json.dump(state, stream)
    stream.flush()
    fcntl.flock(stream, fcntl.LOCK_UN)
sys.exit(7 if failed else 0)
""",
    )
    model = tmp_path / "ggml-small.bin"
    model.write_bytes(b"model")
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("PATH", f"{binary_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DUET_ASR_TEST_STATE", str(state))
    return cli, model, audio, state, fail_once


def test_process_budget_serializes_real_asr_subprocesses(tmp_path, monkeypatch):
    cli, model, audio, state_path, _fail_once = _real_asr_fixture(
        tmp_path, monkeypatch
    )
    budget = asr.ASRProcessBudget(4)

    def run_one(_index: int):
        return asr.transcribe(
            audio,
            cli=cli,
            model=model,
            duration_s=2.0,
            timeout_s=10,
            threads=4,
            process_budget=budget,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(run_one, range(3))) == [[], [], []]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"active": 0, "max_active": 1, "calls": 3}


def test_process_budget_releases_after_real_asr_failure(tmp_path, monkeypatch):
    cli, model, audio, state_path, fail_once = _real_asr_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv("DUET_ASR_TEST_FAIL_ONCE", str(fail_once))
    budget = asr.ASRProcessBudget(4)
    kwargs = {
        "cli": cli,
        "model": model,
        "duration_s": 2.0,
        "timeout_s": 10,
        "threads": 4,
        "process_budget": budget,
    }

    with pytest.raises(asr.ASRError, match="asr_failed"):
        asr.transcribe(audio, **kwargs)
    assert asr.transcribe(audio, **kwargs) == []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == {"active": 0, "max_active": 1, "calls": 2}


def test_process_budget_rejects_thread_count_drift():
    budget = asr.ASRProcessBudget(4)
    with pytest.raises(asr.ASRError, match="asr_thread_budget_mismatch"):
        with budget.claim(2):
            pytest.fail("mismatched thread count must not acquire the budget")


def test_process_budget_releases_after_cancellation():
    budget = asr.ASRProcessBudget(4)
    with pytest.raises(KeyboardInterrupt):
        with budget.claim(4):
            raise KeyboardInterrupt
    with budget.claim(4):
        pass
