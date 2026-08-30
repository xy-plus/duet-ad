import json
import logging
import tempfile
from pathlib import Path

import pytest

from app import codex_runner, error_trace
from app.codex_runner import CodexError, CodexRunner


def _raise_inner_error() -> None:
    raise ValueError("inner failure")


def _nested_error() -> BaseException:
    try:
        _raise_inner_error()
    except ValueError as inner:
        try:
            raise RuntimeError("outer failure") from inner
        except RuntimeError as outer:
            return outer


def _context_error() -> BaseException:
    try:
        _raise_inner_error()
    except ValueError:
        try:
            raise RuntimeError("context failure")
        except RuntimeError as outer:
            return outer


def test_exception_tree_preserves_cause_order_and_each_traceback() -> None:
    tree = error_trace.exception_tree(_nested_error())

    assert tree["type"] == "RuntimeError"
    assert tree["message"] == "outer failure"
    assert tree["traceback"]
    assert tree["traceback"][-1]["function"] == "_nested_error"
    assert tree["traceback"][-1]["line"] > 0

    cause = tree["cause"]
    assert cause["type"] == "ValueError"
    assert cause["message"] == "inner failure"
    assert cause["traceback"]
    assert cause["traceback"][-1]["function"] == "_raise_inner_error"
    assert cause["traceback"][-1]["file"] == __file__


def test_exception_tree_terminates_a_cyclic_cause_chain() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    tree = error_trace.exception_tree(first)

    assert tree["cause"]["cause"] == {"type": "RuntimeError", "cycle": True}


def test_exception_tree_distinguishes_implicit_context_from_explicit_cause() -> None:
    tree = error_trace.exception_tree(_context_error())

    assert "cause" not in tree
    assert tree["context"]["type"] == "ValueError"
    assert tree["context"]["message"] == "inner failure"


def test_record_expands_exception_group_members(tmp_path: Path) -> None:
    group = ExceptionGroup(
        "parallel failures",
        [ValueError("first child"), RuntimeError("second child")],
    )

    payload = error_trace.record(
        tmp_path / "exception-group.json",
        call_path=["pipeline", "parallel"],
        error=group,
    )
    tree = payload["error"]

    assert tree["type"] == "ExceptionGroup"
    assert tree["message"] == "parallel failures (2 sub-exceptions)"
    assert [node["type"] for node in tree["exceptions"]] == [
        "ValueError",
        "RuntimeError",
    ]
    assert [node["message"] for node in tree["exceptions"]] == [
        "first child",
        "second child",
    ]
    assert json.loads((tmp_path / "exception-group.json").read_text()) == payload


class _JsonResponse:
    status_code = 429
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": "req-123",
        "Authorization": "Bearer header-secret",
        "Set-Cookie": "session=header-secret",
    }
    text = '{"message":"provider rejected request","api_key":"body-secret"}'

    def json(self):
        return {
            "message": "provider rejected request",
            "api_key": "body-secret",
            "nested": {
                "accessToken": "nested-secret",
                "items": [{"password": "list-secret", "safe": "visible"}],
            },
        }


class _TextResponse:
    status_code = 502
    headers = {"Trace-ID": "trace-456"}
    text = "opaque upstream failure; token=body-secret; incident=inc-789"

    def json(self):
        raise ValueError("not JSON")


class _LongTextResponse:
    status_code = 503
    headers = {}
    text = "x" * 40_000

    def json(self):
        raise ValueError("not JSON")


def test_provider_response_redacts_body_secrets_and_allowlists_headers() -> None:
    result = error_trace.provider_response(
        _JsonResponse(), secrets=("header-secret",)
    )

    assert result == {
        "http_status": 429,
        "headers": {
            "Content-Type": "application/json",
            "X-Request-ID": "req-123",
        },
        "body": {
            "message": "provider rejected request",
            "api_key": "[REDACTED]",
            "nested": {
                "accessToken": "[REDACTED]",
                "items": [{"password": "[REDACTED]", "safe": "visible"}],
            },
        },
        "body_raw": (
            '{"message":"provider rejected request","api_key":"[REDACTED]"}'
        ),
        "body_truncated": False,
    }
    assert "secret" not in json.dumps(result)


def test_provider_response_preserves_non_json_body_while_redacting_credentials() -> None:
    result = error_trace.provider_response(_TextResponse())

    assert result == {
        "http_status": 502,
        "headers": {"Trace-ID": "trace-456"},
        "body": (
            "opaque upstream failure; token=[REDACTED]; incident=inc-789"
        ),
        "body_raw": (
            "opaque upstream failure; token=[REDACTED]; incident=inc-789"
        ),
        "body_truncated": False,
    }


def test_provider_response_marks_truncated_body() -> None:
    result = error_trace.provider_response(_LongTextResponse())

    assert result["body_truncated"] is True
    assert len(result["body"]) == 32 * 1024
    assert result["body_raw"] == result["body"]


def test_provider_response_redacts_allowlisted_header_values() -> None:
    response = _JsonResponse()
    response.headers = {"X-Request-ID": "header-private-token"}

    result = error_trace.provider_response(
        response, secrets=("header-private-token",)
    )

    assert result["headers"] == {"X-Request-ID": "[REDACTED]"}


def test_record_publishes_complete_json_with_same_directory_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diagnostics" / "error.json"
    observed: dict[str, object] = {}
    original_replace = error_trace.os.replace

    def inspect_then_replace(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        observed["source"] = source_path
        observed["destination"] = destination_path
        observed["staged"] = json.loads(source_path.read_text(encoding="utf-8"))
        original_replace(source, destination)

    monkeypatch.setattr(error_trace.os, "replace", inspect_then_replace)

    payload = error_trace.record(
        path,
        call_path=["pipeline", "provider"],
        reason={"code": "upstream_failed", "authorization": "Bearer secret"},
    )

    source = observed["source"]
    assert isinstance(source, Path)
    assert source.parent == path.parent
    assert observed["destination"] == path
    assert observed["staged"] == payload
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["schema"] == "duet.error-call-tree"
    assert payload["version"] == 1
    assert payload["call_path"] == ["pipeline", "provider"]
    assert payload["error"]["authorization"] == "[REDACTED]"
    assert not list(path.parent.glob(f".{path.name}-*"))


def test_record_replace_failure_preserves_old_file_and_removes_staging_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "error.json"
    path.write_text("old durable record", encoding="utf-8")
    logger = logging.getLogger("tests.error_trace.persist_failure")

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(error_trace.os, "replace", fail_replace)

    with caplog.at_level(logging.ERROR, logger=logger.name):
        payload = error_trace.record(
            path,
            call_path=["pipeline"],
            reason="new record",
            logger=logger,
        )

    assert payload["error"] == "new record"
    assert path.read_text(encoding="utf-8") == "old durable record"
    assert not list(tmp_path.glob(f".{path.name}-*"))
    assert len(caplog.records) == 2
    assert caplog.records[0].getMessage().startswith(
        "pipeline_error_record_persist_failed "
    )
    assert "simulated replace failure" in caplog.records[0].getMessage()
    assert caplog.records[1].getMessage().startswith("pipeline_call_failed ")


def test_record_emits_structured_error_log_with_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("tests.error_trace")
    error = _nested_error()

    with caplog.at_level(logging.ERROR, logger=logger.name):
        payload = error_trace.record(
            tmp_path / "error.json",
            call_path=["pipeline", "provider"],
            error=error,
            logger=logger,
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert record.getMessage().startswith("pipeline_call_failed ")
    logged_payload = json.loads(record.getMessage().removeprefix("pipeline_call_failed "))
    assert logged_payload == payload
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert record.exc_info[1] is error


def test_record_never_throws_when_exception_stringification_is_broken(
    tmp_path: Path,
) -> None:
    class BadStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("broken exception string")

    payload = error_trace.record(
        tmp_path / "error.json",
        call_path=["pipeline", "internal"],
        error=BadStringError(),
    )

    assert payload["error"]["type"] == "BadStringError"
    assert payload["error"]["message"] == "[unprintable BadStringError]"
    assert json.loads((tmp_path / "error.json").read_text()) == payload


def test_record_redacts_plain_provider_echo_of_environment_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "plain-provider-credential"
    monkeypatch.setenv("THIRD_PARTY_API_KEY", secret)
    logger = logging.getLogger("tests.error_trace.environment")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        error_trace.record(
            tmp_path / "error.json",
            call_path=["provider", secret],
            reason={"message": f"provider echoed {secret}"},
            logger=logger,
        )

    raw = (tmp_path / "error.json").read_text(encoding="utf-8")
    assert secret not in raw
    assert secret not in caplog.text


def test_isolated_process_failure_writes_call_tree_and_sanitized_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "conversation"
    session.mkdir()
    monkeypatch.setattr(
        codex_runner, "_resolve_bwrap", lambda: Path("/usr/bin/bwrap")
    )

    with tempfile.TemporaryDirectory(
        prefix="duet-error-trace-process-", dir="/tmp"
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
                "echo 'process failed; token=process-secret; incident=inc-42' >&2; exit 7",
            ],
        )

        with pytest.raises(CodexError, match="codex exit 7"):
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
        "pipeline",
        "codex",
        stage.name,
        "result.json",
    ]
    assert trace["error"]["type"] == "CodexError"
    assert "incident=inc-42" in trace["error"]["message"]
    assert "process-secret" not in json.dumps(trace)
    assert trace["error"]["traceback"]
