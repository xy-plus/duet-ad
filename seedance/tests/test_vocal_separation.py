from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from app import vocal_separation


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source-audio")
    cli = tmp_path / "audio-separator"
    cli.write_bytes(b"#!/bin/sh\n")
    model = tmp_path / "models" / "UVR-MDX-NET-Inst_HQ_3.onnx"
    model.parent.mkdir()
    model.write_bytes(b"onnx-model")
    return source, cli, model


def _fake_cli_run(*names: str, returncode: int = 0, stderr: str = ""):
    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        if returncode == 0:
            for name in names:
                (output_dir / name).write_bytes(f"payload:{name}".encode())
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    return fake


def _run_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *names: str,
) -> tuple[list[list[str]], Path, Path, dict[str, Any]]:
    source, cli, model = _setup(tmp_path)
    calls: list[list[str]] = []

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return _fake_cli_run(*names)(argv, **kwargs)

    monkeypatch.setattr(vocal_separation.subprocess, "run", fake)
    accompaniment_dest = tmp_path / "work" / "accompaniment.wav"
    vocals_dest = tmp_path / "work" / "vocals.wav"
    result = vocal_separation.separate_vocals(
        source, accompaniment_dest, vocals_dest,
        cli_path=cli, model_path=model,
    )
    return calls, accompaniment_dest, vocals_dest, result


def test_separate_vocals_requires_configured_cli(tmp_path: Path) -> None:
    source, _cli, model = _setup(tmp_path)

    with pytest.raises(
        vocal_separation.VocalSeparationError, match="VOCAL_SEPARATION_CLI"
    ):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=None, model_path=model,
        )


def test_separate_vocals_requires_existing_cli_file(tmp_path: Path) -> None:
    source, _cli, model = _setup(tmp_path)

    with pytest.raises(vocal_separation.VocalSeparationError, match="CLI"):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=tmp_path / "missing-cli", model_path=model,
        )


def test_separate_vocals_requires_configured_model_path(tmp_path: Path) -> None:
    source, cli, _model = _setup(tmp_path)

    with pytest.raises(
        vocal_separation.VocalSeparationError,
        match="VOCAL_SEPARATION_MODEL_PATH",
    ):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=cli, model_path=None,
        )


def test_separate_vocals_requires_existing_model_file(tmp_path: Path) -> None:
    source, cli, _model = _setup(tmp_path)

    with pytest.raises(vocal_separation.VocalSeparationError, match="model"):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=cli, model_path=tmp_path / "missing-model.onnx",
        )


def test_separate_vocals_requires_existing_source(tmp_path: Path) -> None:
    _source, cli, model = _setup(tmp_path)

    with pytest.raises(vocal_separation.VocalSeparationError, match="source"):
        vocal_separation.separate_vocals(
            tmp_path / "missing-source.wav",
            tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=cli, model_path=model,
        )


def test_separate_vocals_invokes_cli_and_moves_stems_to_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, accompaniment_dest, vocals_dest, result = _run_happy_path(
        tmp_path, monkeypatch,
        "source_(Instrumental)_UVR-MDX-NET-Inst_HQ_3.wav",
        "source_(Vocals)_UVR-MDX-NET-Inst_HQ_3.wav",
    )

    (argv,) = calls
    assert argv[0].endswith("audio-separator")
    assert argv[1].endswith("source.wav")
    assert argv[argv.index("-m") + 1] == "UVR-MDX-NET-Inst_HQ_3.onnx"
    assert argv[argv.index("--model_file_dir") + 1].endswith("models")
    assert argv[argv.index("--output_format") + 1] == "WAV"
    assert result["accompaniment"] == accompaniment_dest
    assert result["vocals"] == vocals_dest
    assert result["model"] == "UVR-MDX-NET-Inst_HQ_3.onnx"
    assert accompaniment_dest.read_bytes() == (
        b"payload:source_(Instrumental)_UVR-MDX-NET-Inst_HQ_3.wav"
    )
    assert vocals_dest.read_bytes() == (
        b"payload:source_(Vocals)_UVR-MDX-NET-Inst_HQ_3.wav"
    )


def test_separate_vocals_cli_failure_reports_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cli, model = _setup(tmp_path)
    monkeypatch.setattr(
        vocal_separation.subprocess, "run",
        _fake_cli_run(returncode=1, stderr="onnxruntime boom"),
    )

    with pytest.raises(
        vocal_separation.VocalSeparationError, match="onnxruntime boom"
    ):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=cli, model_path=model,
        )

    assert not (tmp_path / "accompaniment.wav").exists()
    assert not (tmp_path / "vocals.wav").exists()


def test_separate_vocals_cli_timeout_propagates_for_node_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, cli, model = _setup(tmp_path)

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(vocal_separation.subprocess, "run", fake)

    with pytest.raises(subprocess.TimeoutExpired):
        vocal_separation.separate_vocals(
            source, tmp_path / "accompaniment.wav", tmp_path / "vocals.wav",
            cli_path=cli, model_path=model,
        )


def test_separate_vocals_rejects_unexpected_output_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(vocal_separation.VocalSeparationError, match="outputs"):
        _run_happy_path(tmp_path, monkeypatch, "source_(Vocals)_model.wav")

    assert not (tmp_path / "work" / "accompaniment.wav").exists()
    assert not (tmp_path / "work" / "vocals.wav").exists()


def test_separate_vocals_rejects_ambiguous_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(vocal_separation.VocalSeparationError, match="vocals"):
        _run_happy_path(tmp_path, monkeypatch, "stem-a.wav", "stem-b.wav")
