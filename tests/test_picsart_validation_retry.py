import json

import pytest

from app.pipeline import Pipeline
from app.providers import PicsartClient, ProviderFailure
from test_pipeline import _FlowPicsart, _run_h3_with_picsart


VALIDATION_MESSAGE = (
    "ValidationError: [Property: 'body.reference_image_urls.1', "
    "Message: The aspect ratio of the image should be between 0.4 and 2.5.]"
)


def _failure_record(status_code, message=VALIDATION_MESSAGE):
    body = (
        {"status": "error", "reason": f"{PicsartClient.WORKFLOW}_failed", "message": message}
        if status_code == 422 else
        {"response": {"status": "FAILED", "error": {"message": message}}}
    )
    return {"polls": [{"response": {"status_code": status_code, "body": body}}]}


@pytest.mark.parametrize("status_code", [422, 200])
def test_terminal_request_validation_preserves_receipts_without_new_post(tmp_path, monkeypatch, status_code):
    class InvalidRequestPicsart(_FlowPicsart):
        def poll(self, task_id):
            raise ProviderFailure(
                {"task_id": task_id, **_failure_record(status_code)}, [],
                RuntimeError(VALIDATION_MESSAGE),
            )

    picsart = InvalidRequestPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)
    state = store.read(project_id)
    assert state["status"] == "failed"
    # Two source segments make exactly one accepted submission apiece.
    assert len(picsart.submits) == 2
    ledger = state["picsart_submission_receipts"]["segments"]
    for records in ledger.values():
        assert len(records) == 1
        assert records[0]["run_identity"] == "picsart"
        assert records[0]["status"] == "provider_failed"
        assert records[0]["task_id"]
    events = [json.loads(line) for line in (project / "logs" / "trace.jsonl").read_text().splitlines()]
    assert not any(event["event"] == "picsart_terminal_failure_resubmit" for event in events)
    assert sum(event["event"] == "picsart_request_validation_failed" for event in events) == 2
    for segment in ("segment-0001", "segment-0002"):
        output = json.loads((project / "logs" / "picsart-poll" / segment / "attempt-1" / "output.json").read_text())
        assert output["provider_record"]["polls"][0]["response"]["status_code"] == status_code


@pytest.mark.parametrize("message", [
    "ValidationError: Failed to download audio file: All validated IPs unreachable",
    "ValidationError: [Property: 'body.reference_image_urls.1', Message: Failed to download image file]",
    "temporary provider failure",
    "content moderation rejected",
])
def test_other_terminal_failures_keep_existing_retry_policy(message):
    record = _failure_record(422, message)
    assert Pipeline._poll_record_is_terminal_failure(record)
    assert not Pipeline._poll_failure_is_request_validation(record)


@pytest.mark.parametrize("status_code", [400, 403, 429, 500, 503])
def test_nonterminal_http_status_is_not_request_validation(status_code):
    record = _failure_record(422)
    record["polls"][0]["response"]["status_code"] = status_code
    assert not Pipeline._poll_failure_is_request_validation(record)
