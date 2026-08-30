import hashlib
import json
import shutil
import subprocess
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
            output_schema={"type": "object"},
        )

        assert value == {"source": "final"}
        assert output.read_bytes() == b'{"source":"final"}'
        assert observed["bytes"] == b'{"source":"final"}'
        assert Path(observed["source"]).parent == work
        assert not list(work.glob(".result.json-transport-*"))


def test_structured_sandbox_mounts_stage_readonly_and_only_final_output_writable(
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
        business_output = work / "result.json"
        final_output = work / ".codex-final-output.json"
        final_output.touch(mode=0o600)

        argv = codex_runner._isolated_outer_argv(
            stage,
            session,
            ["codex", "exec", "-o", str(final_output)],
            writable_paths=(final_output,),
            readonly_stage=True,
        )

        assert ["--ro-bind", str(stage), str(stage)] == argv[
            argv.index("--ro-bind"):argv.index("--ro-bind") + 3
        ]
        final_bind = argv.index(str(final_output))
        assert argv[final_bind - 1] == "--bind"
        assert argv[final_bind + 1] == str(final_output)
        assert str(business_output) not in argv


def test_structured_sandbox_supports_nested_git_mountpoint_without_writable_stage(
    tmp_path: Path,
) -> None:
    """Codex's inner fs sandbox must not need to mkdir under the read-only stage."""
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
        business_output = work / "result.json"
        final_output = work / ".codex-final-output.json"
        final_output.touch(mode=0o600)
        inner = [
            str(codex_runner._resolve_bwrap()),
            "--bind", "/", "/",
            "--dir", str(stage / ".git"),
            "--chdir", str(stage),
            "/usr/bin/bash", "-c",
            "test -d .git && "
            "! /usr/bin/touch .git/forbidden 2>/dev/null && "
            "! /usr/bin/touch work/result.json 2>/dev/null && "
            "printf '{\"ok\":true}' > work/.codex-final-output.json",
        ]
        argv = codex_runner._isolated_outer_argv(
            stage,
            session,
            inner,
            writable_paths=(final_output,),
            readonly_stage=True,
        )

        proc = subprocess.run(argv, capture_output=True, text=True, timeout=20)

        assert proc.returncode == 0, proc.stderr
        assert (stage / ".git").is_dir()
        assert not list((stage / ".git").iterdir())
        assert not business_output.exists()
        assert final_output.read_bytes() == b'{"ok":true}'


def test_structured_sandbox_rejects_staged_git_content(tmp_path: Path) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    with tempfile.TemporaryDirectory(
        prefix="duet-final-output-git-content-", dir="/tmp",
    ) as raw_stage:
        stage = Path(raw_stage).resolve(strict=True)
        work = stage / "work"
        work.mkdir()
        final_output = work / ".codex-final-output.json"
        final_output.touch(mode=0o600)
        git_mountpoint = stage / ".git"
        git_mountpoint.mkdir()
        (git_mountpoint / "config").write_text("untrusted", encoding="utf-8")

        with pytest.raises(CodexError, match="git mountpoint"):
            codex_runner._isolated_outer_argv(
                stage,
                session,
                ["codex", "exec", "-o", str(final_output)],
                writable_paths=(final_output,),
                readonly_stage=True,
            )


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
            raise ValueError("unexpected schema")
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
        serialized_diagnostic = json.dumps(diagnostic, ensure_ascii=False)
        assert "sensitive-model-output" not in serialized_diagnostic
        assert "unexpected" not in serialized_diagnostic
