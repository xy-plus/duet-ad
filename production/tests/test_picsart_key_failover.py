"""Picsart 402 余额不足 token failover 单测（mock transport，零真实调用）。

判定依据：官方 creative-apis-error-codes（402 Payment Required = 预付 credits
用尽或订阅上限未开透支）与 quotas-and-limits（body {"message":"Payment
Required"}）。402 在 SAFE_REJECT_STATUS_CODES 内 = 同步拒绝、未建任务、未扣费，
同一次 submit 调用内用 backup 重发无双扣风险。任务账户隔离：task 由哪把 key
创建，poll/result/download 必须用同一把（回执 api_key_role 为唯一来源）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.providers import PicsartClient, ProviderFailure

PRIMARY_KEY = "picsart-primary-secret"
BACKUP_KEY = "picsart-backup-secret"


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://failover.test",
        "picsart_api_key": PRIMARY_KEY,
        "picsart_api_base": "https://picsart.test",
        "picsart_api_key_backup": BACKUP_KEY,
        "request_timeout_s": 1.0,
        "provider_poll_interval_s": 0.0,
        "provider_poll_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def _client(tmp_path: Path, handler, **overrides: object) -> PicsartClient:
    return PicsartClient(
        _settings(tmp_path, **overrides),
        transport=httpx.MockTransport(handler),
    )


def _submit_kwargs() -> dict[str, Any]:
    return {
        "prompt": "p",
        "video_url": "https://assets.test/v.mp4",
        "image_urls": ["https://assets.test/i.png"],
        "audio_urls": [],
        "duration": 5,
        "resolution": "768p",
        "aspect_ratio": "16:9",
    }


def _role_of(request: httpx.Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {PRIMARY_KEY}":
        return "primary"
    if auth == f"Bearer {BACKUP_KEY}":
        return "backup"
    return "other"


class TestSubmitFailover:
    def test_primary_402_fails_over_to_backup_in_same_call(
        self, tmp_path: Path,
    ) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            role = _role_of(request)
            seen.append(role)
            if role == "primary":
                return httpx.Response(402, json={"message": "Payment Required"})
            return httpx.Response(200, json={"response": {"id": "task-1"}})

        result = _client(tmp_path, handler).submit(**_submit_kwargs())
        assert result.value == "task-1"
        assert seen == ["primary", "backup"]  # 同一次调用内重发
        assert result.record["outcome"] == "accepted"
        assert result.record["api_key_role"] == "backup"
        assert result.record["key_failover"] == {"from": "primary", "status": 402}

    def test_sticky_backup_after_primary_exhausted(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            role = _role_of(request)
            seen.append(role)
            if role == "primary":
                return httpx.Response(402, json={"message": "Payment Required"})
            return httpx.Response(200, json={"response": {"id": "task-x"}})

        client = _client(tmp_path, handler)
        client.submit(**_submit_kwargs())
        assert client._primary_exhausted is True
        seen.clear()
        client.submit(**_submit_kwargs())
        assert seen == ["backup"]  # 进程内粘性：不再试 primary

    def test_no_backup_configured_402_behaves_as_before(
        self, tmp_path: Path,
    ) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(_role_of(request))
            return httpx.Response(402, json={"message": "Payment Required"})

        client = _client(tmp_path, handler, picsart_api_key_backup="")
        with pytest.raises(ProviderFailure) as caught:
            client.submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "rejected"
        assert seen == ["primary"]  # 未配 backup：不重发，行为与现状一致

    def test_401_never_fails_over(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(_role_of(request))
            return httpx.Response(401, json={"message": "invalid token"})

        client = _client(tmp_path, handler)
        with pytest.raises(ProviderFailure) as caught:
            client.submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "rejected"
        assert seen == ["primary"]  # 401=key 无效属配置错误，不 failover
        assert client._primary_exhausted is False

    def test_backup_also_402_keeps_rejected_semantics(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(_role_of(request))
            return httpx.Response(402, json={"message": "Payment Required"})

        client = _client(tmp_path, handler)
        with pytest.raises(ProviderFailure) as caught:
            client.submit(**_submit_kwargs())
        assert caught.value.record["outcome"] == "rejected"
        assert seen == ["primary", "backup"]  # backup 也 402：不循环重试
        assert caught.value.record["api_key_role"] == "backup"


class TestPollAccountIsolation:
    def test_poll_uses_receipt_role_key(self, tmp_path: Path) -> None:
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((_role_of(request), request.url.path.rsplit("/", 2)[-2]))
            return httpx.Response(200, json={
                "response": {
                    "status": "COMPLETED",
                    "result": {"video": {"url": "https://cdn.test/v.mp4"}},
                },
            })

        client = _client(tmp_path, handler)
        client.poll("task-primary", api_key_role="primary")
        client.poll("task-backup", api_key_role="backup")
        assert seen == [
            ("primary", "task-primary"),
            ("backup", "task-backup"),
        ]

    def test_backup_role_without_backup_key_is_clear_error(
        self, tmp_path: Path,
    ) -> None:
        client = _client(tmp_path, lambda r: httpx.Response(200),
                         picsart_api_key_backup="")
        with pytest.raises(ValueError, match="PICSART_API_KEY_BACKUP"):
            client.poll("task-1", api_key_role="backup")

    def test_download_uses_receipt_role_key(self, tmp_path: Path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(_role_of(request))
            return httpx.Response(200, content=b"video-bytes")

        client = _client(tmp_path, handler)
        out = tmp_path / "out.mp4"
        client.download("https://cdn.test/v.mp4", out, api_key_role="backup")
        assert seen == ["backup"]
        assert out.read_bytes() == b"video-bytes"


class TestReceiptRoleValidation:
    def test_legacy_receipt_without_role_defaults_primary(self) -> None:
        from app.pipeline import Pipeline

        manifest = {"version": 7, "manifest": True}
        ledger = {
            "version": 2,
            "segments": {"1": [{
                "segment_index": 1, "run_identity": "picsart",
                "request_sha256": "a" * 64,
                "preparation_sha256": Pipeline._json_sha256(manifest),
                "status": "accepted", "task_id": "task-1",
            }]},
        }
        state = {
            "picsart_submission_receipts": ledger,
            "h3_prepared_artifacts": manifest,
        }
        assert Pipeline._state_submission_api_key_role(state, 1) == "primary"

    def test_backup_role_read_back_from_receipt(self) -> None:
        from app.pipeline import Pipeline

        manifest = {"version": 7, "manifest": True}
        ledger = {
            "version": 2,
            "segments": {"1": [{
                "segment_index": 1, "run_identity": "picsart",
                "request_sha256": "a" * 64,
                "preparation_sha256": Pipeline._json_sha256(manifest),
                "status": "accepted", "task_id": "task-1",
                "api_key_role": "backup",
            }]},
        }
        state = {
            "picsart_submission_receipts": ledger,
            "h3_prepared_artifacts": manifest,
        }
        assert Pipeline._state_submission_api_key_role(state, 1) == "backup"
