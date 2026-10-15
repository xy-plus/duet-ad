import json

from app.providers import ProviderFailure
from test_pipeline import _FlowSeedance, _run_h3_with_seedance


BAD_REQUEST_MESSAGE = (
    "InvalidParameter.TaskTypeConstraint: the output ratio follows the "
    "first-frame image"
)


def test_terminal_bad_request_preserves_receipts_without_new_post(
    tmp_path, monkeypatch,
):
    """bad_request 分桶（参数/策略确定性拒绝）终局不重投，回执留证。

    重投上限 3、content_rejected/reference_fetch_failed 不重投、unknown 挡
    resume 的流级覆盖见 tests/test_pipeline.py 与 test_resume_r2v_*。
    """

    class BadRequestSeedance(_FlowSeedance):
        def poll(self, task_id):
            raise ProviderFailure(
                {
                    "task_id": task_id,
                    "polls": [{"response": {
                        "status_code": 200,
                        "body": {
                            "status": "failed",
                            "error": {
                                "code": "InvalidParameter.TaskTypeConstraint",
                                "message": BAD_REQUEST_MESSAGE,
                            },
                        },
                    }}],
                },
                [],
                RuntimeError(BAD_REQUEST_MESSAGE),
            )

    seedance = BadRequestSeedance()
    store, project_id, project = _run_h3_with_seedance(tmp_path, monkeypatch, seedance)
    state = store.read(project_id)
    assert state["status"] == "failed"
    # Two source segments make exactly one accepted submission apiece.
    assert len(seedance.submits) == 2
    ledger = state["seedance_submission_receipts"]["segments"]
    for records in ledger.values():
        assert len(records) == 1
        assert records[0]["run_identity"] == "seedance"
        assert records[0]["status"] == "provider_failed"
        assert records[0]["task_id"]
    events = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl").read_text().splitlines()
    ]
    assert not any(
        event["event"] == "seedance_terminal_failure_resubmit" for event in events
    )
    no_resubmit = [
        event for event in events
        if event["event"] == "seedance_terminal_failure_no_resubmit"
    ]
    assert len(no_resubmit) == 2
    assert all(event["bucket"] == "bad_request" for event in no_resubmit)
    for segment in ("segment-0001", "segment-0002"):
        output = json.loads(
            (
                project / "logs" / "seedance-poll" / segment
                / "attempt-1" / "output.json"
            ).read_text()
        )
        assert (
            output["provider_record"]["polls"][0]["response"]["status_code"]
            == 200
        )
