import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from app import codex_runner
from app.codex_runner import CodexError, CodexOutputValidationError, CodexRunner


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
            output_schema={"type": "object"},
        )

        assert value == {"source": "final"}
        assert output.read_bytes() == b'{"source":"final"}'
        assert observed["bytes"] == b'{"source":"final"}'
        assert Path(observed["source"]).parent == work
        assert not list(work.glob(".result.json-transport-*"))


def test_structured_sandbox_mounts_inputs_readonly_with_writable_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(
        codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
    )
    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-permissions-", dir="/tmp",
    ) as raw_stage:
        stage = Path(raw_stage).resolve(strict=True)
        work = stage / "work"
        work.mkdir()
        skill = stage / "SKILL.md"
        descriptor = work / "input.json"
        final_output = work / ".codex-final-output.json"
        skill.write_text("skill", encoding="utf-8")
        descriptor.write_text("input", encoding="utf-8")
        final_output.touch(mode=0o600)

        argv = codex_runner._isolated_outer_argv(
            stage,
            session,
            ["codex", "exec", "-o", str(final_output)],
            writable_paths=(final_output,),
            structured_stage=True,
        )

        assert any(
            argv[index:index + 3] == ["--bind", str(stage), str(stage)]
            for index in range(len(argv) - 2)
        )
        for frozen in (skill, descriptor):
            assert any(
                argv[index:index + 3] == ["--ro-bind", str(frozen), str(frozen)]
                for index in range(len(argv) - 2)
            )
        assert not any(
            argv[index:index + 3]
            == ["--ro-bind", str(final_output), str(final_output)]
            for index in range(len(argv) - 2)
        )


def test_structured_runner_allows_arbitrary_nested_mountpoints_but_consumes_only_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real double-bwrap probe; no model or provider is invoked."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    session = tmp_path / "conversation"
    session.mkdir()
    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-nested-sandbox-", dir="/tmp",
    ) as raw_stage:
        stage = Path(raw_stage).resolve(strict=True)
        work = stage / "work"
        work.mkdir()
        skill = stage / "SKILL.md"
        descriptor = work / "input.json"
        fake_codex = stage / "codex"
        business_output = work / "result.json"
        evidence = work / "direct-write-evidence.json"
        skill.write_text("frozen-skill", encoding="utf-8")
        descriptor.write_text("frozen-input", encoding="utf-8")
        fake_codex.write_text(
            "#!/usr/bin/bash\n"
            "exec /usr/bin/bwrap --bind / / "
            f"--dir '{stage / '.git'}' "
            f"--dir '{stage / '.agents'}' "
            f"--dir '{stage / '.arbitrary-tool-state'}' "
            f"--chdir '{stage}' /usr/bin/bash -c '"
            "test -d .git && test -d .agents && "
            "test -d .arbitrary-tool-state && "
            "! printf tampered > SKILL.md 2>/dev/null && "
            "! rm -f SKILL.md 2>/dev/null && "
            "! printf tampered > work/input.json 2>/dev/null && "
            "! printf tampered > .codex-output-schema.json 2>/dev/null && "
            "printf \"{\\\"source\\\":\\\"business\\\"}\" > work/result.json && "
            "printf \"{\\\"source\\\":\\\"business\\\"}\" "
            "> work/direct-write-evidence.json && "
            "printf \"{\\\"source\\\":\\\"final\\\"}\" "
            "> work/.codex-final-output.json'\n",
            encoding="utf-8",
        )
        fake_codex.chmod(0o700)
        monkeypatch.setenv("PATH", f"{stage}:/usr/bin:/bin")
        runner = CodexRunner(timeout_s=5, concurrency=1)

        value = runner.run_isolated_until_output(
            stage,
            "structured prompt",
            session_dir=session,
            output_path=business_output,
            max_output_bytes=1024,
            validate_output=lambda raw: json.loads(raw.decode("utf-8")),
            output_schema={
                "type": "object",
                "properties": {"source": {"type": "string"}},
                "required": ["source"],
                "additionalProperties": False,
            },
        )

        assert value == {"source": "final"}
        assert json.loads(business_output.read_text(encoding="utf-8")) == value
        assert json.loads(evidence.read_text(encoding="utf-8")) == {
            "source": "business",
        }
        assert skill.read_text(encoding="utf-8") == "frozen-skill"
        assert descriptor.read_text(encoding="utf-8") == "frozen-input"
        assert json.loads(
            (stage / ".codex-output-schema.json").read_text(encoding="utf-8")
        )["type"] == "object"
        for hidden in (".git", ".agents", ".arbitrary-tool-state"):
            assert (stage / hidden).is_dir()


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
                f"printf 'business prefix\\n{{\"ok\":true}}\\ntokens used\\n59,854\\n' > "
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
                output_schema={"type": "object"},
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
        diagnostic = trace["details"]["codex_final_output"]
        expected = b'business prefix\n{"ok":true}\ntokens used\n59,854\n'
        assert diagnostic["reason"] == "transport_invalid"
        assert diagnostic["returncode"] == 0
        assert diagnostic["size_bytes"] == len(expected)
        assert diagnostic["max_bytes"] == 1024
        assert diagnostic["sha256"] == hashlib.sha256(expected).hexdigest()
        assert diagnostic["telemetry_suffix_matched"] is True
        assert "business" not in diagnostic["head_redacted"]
        assert "59,854" not in diagnostic["tail_redacted"]


@pytest.mark.parametrize(
    ("payload", "max_output_bytes", "reason"),
    [
        (b'{"unexpected":true}', 1024, "schema_invalid"),
        (b'{"ok":"' + b"sensitive-model-output" * 8 + b'"}', 32, "oversize"),
    ],
    ids=["schema-invalid", "oversize"],
)
def test_rejected_final_output_records_exact_bounded_diagnostic_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    max_output_bytes: int,
    reason: str,
) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(
        codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap"),
    )

    def validate_ok(raw: bytes) -> dict[str, object]:
        value = json.loads(raw.decode("utf-8"))
        if set(value) != {"ok"}:
            raise CodexOutputValidationError(
                "field_set_invalid", "/people/0/key",
                message="sensitive validator context",
            )
        return value

    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-diagnostic-", dir="/tmp",
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
                "/usr/bin/bash", "-c", 'printf "%s" "$1" > "$2"', "write-final",
                payload.decode("ascii"), str(work / ".codex-final-output.json"),
            ],
        )

        with pytest.raises(CodexError, match="without publishing valid output"):
            runner.run_isolated_until_output(
                stage,
                "prompt",
                session_dir=session,
                output_path=output,
                max_output_bytes=max_output_bytes,
                validate_output=validate_ok,
                output_schema={"type": "object"},
            )

        trace = json.loads(
            (session / "work" / "errors" / f"{stage.name}.json").read_text(
                encoding="utf-8"
            )
        )
        diagnostic = trace["details"]["codex_final_output"]
        assert diagnostic["reason"] == reason
        assert diagnostic["returncode"] == 0
        assert diagnostic["size_bytes"] == len(payload)
        assert diagnostic["max_bytes"] == max_output_bytes
        assert diagnostic["sha256"] == hashlib.sha256(payload).hexdigest()
        assert diagnostic["telemetry_suffix_matched"] is False
        if reason == "schema_invalid":
            assert diagnostic["validator_error_type"] == (
                "CodexOutputValidationError"
            )
            assert diagnostic["validator_error"] == {
                "reason": "field_set_invalid",
                "field_path": "/people/0/key",
            }
        serialized_diagnostic = json.dumps(diagnostic, ensure_ascii=False)
        assert "sensitive-model-output" not in serialized_diagnostic
        assert "unexpected" not in serialized_diagnostic
        assert "sensitive validator context" not in serialized_diagnostic
