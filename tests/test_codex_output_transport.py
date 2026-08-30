import json
import tempfile
from pathlib import Path

import pytest

from app import codex_runner
from app.codex_runner import CodexError, CodexRunner


def test_clean_json_transport_is_byte_for_byte_unchanged() -> None:
    raw = b'  {"message":"ok","items":[1,2]}\n'

    assert codex_runner._extract_codex_json_output(raw) == raw


def test_codex_tokens_used_tail_is_the_only_accepted_trailing_telemetry() -> None:
    raw = b'{"message":"ok"}\ntokens used\n28,563\n'

    assert codex_runner._extract_codex_json_output(raw) == b'{"message":"ok"}'


@pytest.mark.parametrize(
    "raw",
    [
        b'final answer follows\n{"message":"ok"}',
        b'{"message":"first"}\n{"message":"second"}',
        b'{"message":"ok"}\nThis result is ready for publication.',
    ],
    ids=["leading-business-text", "multiple-json-values", "trailing-business-text"],
)
def test_codex_json_transport_rejects_ambiguous_or_business_text(raw: bytes) -> None:
    with pytest.raises(ValueError, match="Codex final output"):
        codex_runner._extract_codex_json_output(raw)


def test_final_output_wins_and_is_atomically_published_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(
        codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
    )
    observed: dict[str, object] = {}
    original_replace = codex_runner.os.replace

    def inspect_replace(source, destination, *args, **kwargs) -> None:
        destination_name = Path(destination).name
        if destination_name == "result.json":
            source_path = Path(source)
            observed["source"] = source_path
            observed["bytes"] = source_path.read_bytes()
        original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(codex_runner.os, "replace", inspect_replace)
    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-transport-", dir="/tmp",
    ) as raw_stage:
        stage = Path(raw_stage).resolve(strict=True)
        work = stage / "work"
        work.mkdir()
        output = work / "result.json"
        runner = CodexRunner(timeout_s=3, concurrency=1)
        monkeypatch.setattr(
            runner,
            "build_argv",
            lambda _workdir, _prompt: [
                "/usr/bin/bash",
                "-c",
                f"printf '{{\"source\":\"declared\"}}' > '{output}'; "
                f"printf '{{\"source\":\"final\"}}\\ntokens used\\n28,563\\n' > "
                f"'{work / '.codex-final-output.json'}'",
            ],
        )

        value = runner.run_isolated_until_output(
            stage,
            "prompt",
            session_dir=session,
            output_path=output,
            max_output_bytes=1024,
            validate_output=lambda raw: json.loads(raw.decode("utf-8")),
        )

        assert value == {"source": "final"}
        assert output.read_bytes() == b'{"source":"final"}'
        assert observed["bytes"] == b'{"source":"final"}'
        assert Path(observed["source"]).parent == work
        assert not list(work.glob(".result.json-transport-*"))


def test_rejected_final_output_keeps_structured_error_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(
        codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
    )
    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-rejected-", dir="/tmp",
    ) as raw_stage:
        stage = Path(raw_stage).resolve(strict=True)
        work = stage / "work"
        work.mkdir()
        output = work / "result.json"
        runner = CodexRunner(timeout_s=3, concurrency=1)
        monkeypatch.setattr(
            runner,
            "build_argv",
            lambda _workdir, _prompt: [
                "/usr/bin/bash",
                "-c",
                f"printf 'business prefix\\n{{\"ok\":true}}' > "
                f"'{work / '.codex-final-output.json'}'",
            ],
        )

        with pytest.raises(CodexError, match="without publishing valid output"):
            runner.run_isolated_until_output(
                stage,
                "prompt",
                session_dir=session,
                output_path=output,
                max_output_bytes=1024,
                validate_output=lambda raw: json.loads(raw.decode("utf-8")),
            )

        trace = json.loads(
            (session / "work" / "errors" / f"{stage.name}.json").read_text(
                encoding="utf-8"
            )
        )
        assert trace["call_path"] == [
            "pipeline", "codex", stage.name, "result.json",
        ]
        assert trace["error"]["type"] == "CodexError"
        assert trace["error"]["traceback"]
