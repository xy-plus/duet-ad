import json
import logging
from pathlib import Path

import pytest

from app import error_trace


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


class _JsonResponse:
    status_code = 429
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": "req-123",
        "Authorization": "Bearer header-secret",
        "Set-Cookie": "session=header-secret",
    }

    def json(self):
        return {
            "message": "provider rejected request",
            "api_key": "body-secret",
            "nested": {
                "accessToken": "nested-secret",
                "items": [{"password": "list-secret", "safe": "visible"}],
            },
        }


def test_provider_response_redacts_body_secrets_and_allowlists_headers() -> None:
    result = error_trace.provider_response(_JsonResponse())

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
    }
    assert "secret" not in json.dumps(result)


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "error.json"
    path.write_text("old durable record", encoding="utf-8")

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(error_trace.os, "replace", fail_replace)

    payload = error_trace.record(path, call_path=["pipeline"], reason="new record")

    assert payload["error"] == "new record"
    assert path.read_text(encoding="utf-8") == "old durable record"
    assert not list(tmp_path.glob(f".{path.name}-*"))


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
