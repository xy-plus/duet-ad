import json

import pytest

from app.pipeline import Pipeline
from app.providers import PicsartClient, ProviderFailure
from test_pipeline import _FetchFailPicsart, _FlowPicsart, _run_h3_with_picsart


VALIDATION_MESSAGE = (
    "ValidationError: [Property: 'body.reference_image_urls.1', "
    "Message: The aspect ratio of the image should be between 0.4 and 2.5.]"
)


def _failure_record(status_code, message=VALIDATION_MESSAGE):
    body = (
        {"response": {"status": "FAILED", "error": {"message": message}}}
        if status_code == 200 else
        # 真实的 422/503 终止信封：顶层 status/reason（见
        # Pipeline._poll_record_is_terminal_failure 的注释）。
        {"status": "error", "reason": f"{PicsartClient.WORKFLOW}_failed", "message": message}
    )
    return {"polls": [{"response": {"status_code": status_code, "body": body}}]}


def test_terminal_request_validation_is_resubmitted_and_recovers(tmp_path, monkeypatch):
    """首次 validation 失败 → 自动重提（新 run identity、同一冻结请求），
    上限内成功；回执链 provider_failed → accepted，请求哈希不变。审核器
    是概率性判定（实证 f0997e14/d7eb0406），审核不过供应商不扣费。"""
    picsart = _FetchFailPicsart(fail_once_for=2, message=VALIDATION_MESSAGE)
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "succeeded"
    # 每段：首次提交（审核 422）+ 一次自动重提成功。
    assert len(picsart.submits) == 4
    ledger = state["picsart_submission_receipts"]["segments"]
    for key in ("1", "2"):
        assert [r["status"] for r in ledger[key]] == ["provider_failed", "accepted"]
        assert ledger[key][0]["request_sha256"] == ledger[key][1]["request_sha256"]
        assert ledger[key][1]["run_identity"].endswith("-resubmit-01")
    assert (project / "logs" / "picsart-resubmit-01-submit").is_dir()
    assert (project / "logs" / "picsart-resubmit-01-poll").is_dir()
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("picsart_request_validation_failed") == 2
    assert trace.count("picsart_terminal_failure_resubmit") == 2
    assert trace.count('"reason": "request_validation"') == 2
    assert "picsart_auto_resubmit_exhausted" not in trace


@pytest.mark.parametrize("status_code", [422, 200, 503])
def test_terminal_request_validation_auto_resubmit_exhausts_then_fails_closed(
    tmp_path, monkeypatch, status_code,
):
    """validation 失败与其他终态失败共用 PICSART_AUTO_RESUBMIT_MAX_ATTEMPTS
    上限：首次提交 + 3 次重提仍失败 → fail-closed；每次失败都记
    picsart_request_validation_failed，回执账本逐次落账。"""
    class InvalidRequestPicsart(_FlowPicsart):
        def poll(self, task_id, *, api_key_role="primary"):
            raise ProviderFailure(
                {"task_id": task_id, **_failure_record(status_code)}, [],
                RuntimeError(VALIDATION_MESSAGE),
            )

    picsart = InvalidRequestPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)
    state = store.read(project_id)
    assert state["status"] == "failed"
    # 每段：首次提交 + 自动重提 3 次（上限），第 4 次终态失败 fail-closed。
    assert len(picsart.submits) == 8
    ledger = state["picsart_submission_receipts"]["segments"]
    for records in ledger.values():
        assert [r["status"] for r in records] == ["provider_failed"] * 4
        assert [r["run_identity"] for r in records] == [
            "picsart",
            "picsart-resubmit-01",
            "picsart-resubmit-02",
            "picsart-resubmit-03",
        ]
    events = [
        json.loads(line)
        for line in (project / "logs" / "trace.jsonl").read_text().splitlines()
    ]
    assert sum(
        event["event"] == "picsart_request_validation_failed" for event in events
    ) == 8
    resubmits = [
        event for event in events
        if event["event"] == "picsart_terminal_failure_resubmit"
    ]
    assert len(resubmits) == 6
    assert all(event["reason"] == "request_validation" for event in resubmits)
    assert sum(
        event["event"] == "picsart_auto_resubmit_exhausted" for event in events
    ) == 2
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


@pytest.mark.parametrize("status_code", [400, 403, 429])
def test_nonterminal_http_status_is_not_request_validation(status_code):
    record = _failure_record(422)
    record["polls"][0]["response"]["status_code"] = status_code
    assert not Pipeline._poll_failure_is_request_validation(record)


def test_500_with_error_envelope_and_validation_message_is_request_validation():
    """错误体优先于状态码的必然推论：500 + 终态错误体 + 审核样式 message
    会进 request_validation 子分类（审核器概率性判定，重提可能通过，语义
    与 422 一致）。原"500 非 request_validation"断言随终态识别扩大失效。"""
    record = _failure_record(422)
    record["polls"][0]["response"]["status_code"] = 500
    assert Pipeline._poll_record_is_terminal_failure(record)
    assert Pipeline._poll_failure_is_request_validation(record)


# ------------------------------------------------------------------
# 2026-09-29 项目 0eafd5f5 事故修复：
# 5xx + 终态错误体 → 终态失败分类（错误体优先于状态码）；
# 连续传输级失败封顶 PICSART_POLL_TRANSPORT_FAILURE_LIMIT → fail-closed 不重提。

from app.pipeline import (  # noqa: E402
    PICSART_POLL_TRANSPORT_FAILURE_LIMIT,
    PicsartPollTransportFailureLimitError,
)


FILE_COPY_FAILED_BODY = {
    "status": "error",
    "reason": "file_copy_failed",
    "message": "timeout of 40000ms exceeded undefined",
}


def _record_with(status_code, body):
    return {"polls": [{"response": {"status_code": status_code, "body": body}}]}


@pytest.mark.parametrize("status_code", [500, 502, 504])
def test_5xx_with_terminal_error_body_is_terminal_failure(status_code):
    record = _record_with(status_code, FILE_COPY_FAILED_BODY)
    assert Pipeline._poll_record_is_terminal_failure(record)
    # 与 422 审核判定路径互不干扰：终态 5xx 不是 request validation
    assert not Pipeline._poll_failure_is_request_validation(record)
    assert Pipeline._poll_failure_provider_reason(record) == "file_copy_failed"


@pytest.mark.parametrize("body", [
    "Bad Gateway",                          # 非 JSON body
    {"status": "ok"},                       # JSON 但无终态语义
    {"message": "upstream timeout"},        # 无 status 字段
    {"status": "processing"},               # 非 error 状态
])
def test_5xx_without_error_envelope_stays_transient(body):
    assert not Pipeline._poll_record_is_terminal_failure(
        _record_with(502, body)
    )


def test_422_moderation_envelope_regression():
    """422 审核用的既有判定路径不受 5xx 终态识别影响。"""
    record = _failure_record(422)
    assert Pipeline._poll_record_is_terminal_failure(record)
    assert Pipeline._poll_failure_is_request_validation(record)
    # 裸 503（无信封）仍是传输错误，不是终态
    assert not Pipeline._poll_record_is_terminal_failure(
        _record_with(503, "Service Unavailable")
    )


class _FileCopyFailPicsart(_FlowPicsart):
    """poll 永远以 502 + file_copy_failed 终态错误体失败。"""

    def poll(self, task_id, *, api_key_role="primary"):
        raise ProviderFailure(
            {"task_id": task_id, **_record_with(502, FILE_COPY_FAILED_BODY)},
            [],
            RuntimeError("HTTP 502 file_copy_failed"),
        )


def test_5xx_terminal_body_is_resubmitted_with_provider_reason(
    tmp_path, monkeypatch,
):
    """5xx + 终态错误体进 R4 自动重提分类（与 request_validation 同等地位），
    reason 记录供应商原文 file_copy_failed，上限后 fail-closed。"""
    monkeypatch.setattr(Pipeline, "_retry_sleep", lambda self, _seconds: None)
    picsart = _FileCopyFailPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "failed"
    # 每段：首次提交 + 自动重提 3 次（上限）。
    assert len(picsart.submits) == 8
    ledger = state["picsart_submission_receipts"]["segments"]
    for records in ledger.values():
        assert [r["status"] for r in records] == ["provider_failed"] * 4
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace.splitlines()]
    resubmits = [
        event for event in events
        if event["event"] == "picsart_terminal_failure_resubmit"
    ]
    assert len(resubmits) == 6
    assert all(event["reason"] == "file_copy_failed" for event in resubmits)
    assert trace.count("picsart_auto_resubmit_exhausted") == 2


class _TransportFailPicsart(_FlowPicsart):
    """poll 永远以 502 + 非 JSON body（无终态语义）失败。"""

    def poll(self, task_id, *, api_key_role="primary"):
        raise ProviderFailure(
            {"task_id": task_id, **_record_with(502, "Bad Gateway")},
            [],
            ConnectionError("502 Bad Gateway"),
        )


def test_consecutive_transport_failures_fail_closed_without_resubmit(
    tmp_path, monkeypatch,
):
    """连续传输级失败到上限：按供应商状态未知 fail-closed——不重提付费任务、
    回执保持 accepted、错误清晰、项目 failed 等人工/续跑。"""
    monkeypatch.setattr(Pipeline, "_retry_sleep", lambda self, _seconds: None)
    picsart = _TransportFailPicsart()
    store, project_id, project = _run_h3_with_picsart(tmp_path, monkeypatch, picsart)

    state = store.read(project_id)
    assert state["status"] == "failed"
    # 不重提：每段只有首次提交。
    assert len(picsart.submits) == 2
    # 回执保持 accepted（无终态证据不标 provider_failed）。
    ledger = state["picsart_submission_receipts"]["segments"]
    for records in ledger.values():
        assert [r["status"] for r in records] == ["accepted"]
    # poll 恰好在第 PICSART_POLL_TRANSPORT_FAILURE_LIMIT 次失败停止重试。
    for task_id, count in picsart.poll_counts.items():
        assert count == PICSART_POLL_TRANSPORT_FAILURE_LIMIT
    trace = (project / "logs" / "trace.jsonl").read_text(encoding="utf-8")
    assert trace.count("picsart_poll_transport_failure_limit") == 2
    assert "picsart_terminal_failure_resubmit" not in trace
    # 错误文本清晰：连续 N 次传输级失败、供应商状态未知、fail-closed。
    assert "传输级失败" in state["generation"]["error"]
    assert "供应商状态未知" in state["generation"]["error"]
