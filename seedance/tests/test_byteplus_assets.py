from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import httpx
import pytest

from app.byteplus_assets import (
    BytePlusArkAssetClient,
    BytePlusArkAssetError,
    sign_ark_request,
)
from app.config import Settings


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "access_token": "test-access",
        "data_dir": tmp_path,
        "public_base_url": "https://assets.test",
        "byteplus_access_key_id": "AKIDtest",
        "byteplus_secret_access_key": "SKtest",
        "byteplus_ark_api_base": "https://ark.test",
        "byteplus_region": "ap-southeast-1",
        "byteplus_project_name": "N0017",
        "request_timeout_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def _client(tmp_path: Path, handler, **overrides: object) -> BytePlusArkAssetClient:
    return BytePlusArkAssetClient(
        _settings(tmp_path, **overrides), httpx.MockTransport(handler)
    )


def test_sign_ark_request_is_deterministic_and_matches_reference_layout() -> None:
    headers = sign_ark_request(
        "CreateAsset",
        '{"GroupId":"g1","ProjectName":"N0017"}',
        access_key_id="AKIDexample",
        secret_access_key="SKexample",
        host="ark.ap-southeast-1.byteplusapi.com",
        region="ap-southeast-1",
        iso_date="20260929T101112Z",
    )
    again = sign_ark_request(
        "CreateAsset",
        '{"GroupId":"g1","ProjectName":"N0017"}',
        access_key_id="AKIDexample",
        secret_access_key="SKexample",
        host="ark.ap-southeast-1.byteplusapi.com",
        region="ap-southeast-1",
        iso_date="20260929T101112Z",
    )
    assert headers == again
    assert headers["Content-Type"] == "application/json"
    assert headers["Host"] == "ark.ap-southeast-1.byteplusapi.com"
    assert headers["X-Date"] == "20260929T101112Z"
    assert headers["X-Content-Sha256"] == hashlib.sha256(
        b'{"GroupId":"g1","ProjectName":"N0017"}'
    ).hexdigest()

    # 独立重算参考链（远端 signArkHeaders 的逐行对照，见模块 docstring）。
    body = '{"GroupId":"g1","ProjectName":"N0017"}'
    payload_hash = hashlib.sha256(body.encode()).hexdigest()
    canonical_request = "\n".join([
        "POST",
        "/",
        "Action=CreateAsset&Version=2024-01-01",
        "content-type:application/json",
        "host:ark.ap-southeast-1.byteplusapi.com",
        f"x-content-sha256:{payload_hash}",
        "x-date:20260929T101112Z",
        "",
        "content-type;host;x-content-sha256;x-date",
        payload_hash,
    ])
    scope = "20260929/ap-southeast-1/ark/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        "20260929T101112Z",
        scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    def hm(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()
    k_signing = hm(hm(hm(hm(b"SKexample", "20260929"), "ap-southeast-1"), "ark"), "request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    assert headers["Authorization"] == (
        "HMAC-SHA256 Credential=AKIDexample/20260929/ap-southeast-1/ark/request, "
        f"SignedHeaders=content-type;host;x-content-sha256;x-date, Signature={signature}"
    )


def test_create_asset_group_sends_exact_query_and_body(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/"
        assert request.url.params["Action"] == "CreateAssetGroup"
        assert request.url.params["Version"] == "2024-01-01"
        assert json.loads(request.content) == {
            "GroupType": "AIGC",
            "Name": "project-1",
            "ProjectName": "N0017",
        }
        # 签名头结构（值本身由确定性用例覆盖）。
        assert request.headers["authorization"].startswith(
            "HMAC-SHA256 Credential=AKIDtest/"
        )
        assert "/ap-southeast-1/ark/request" in request.headers["authorization"]
        assert request.headers["x-content-sha256"] == hashlib.sha256(
            request.content
        ).hexdigest()
        return httpx.Response(200, json={"Result": {"Id": "group-1"}})

    client = _client(tmp_path, handler)
    assert client.create_asset_group(name="project-1") == "group-1"


def test_create_asset_sends_group_url_name_type_and_project(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["Action"] == "CreateAsset"
        assert json.loads(request.content) == {
            "GroupId": "group-1",
            "URL": "https://assets.test/intake/file.png",
            "Name": "asset-name",
            "AssetType": "Image",
            "ProjectName": "N0017",
        }
        return httpx.Response(200, json={"Result": {"Id": "asset-1"}})

    client = _client(tmp_path, handler)
    asset_id = client.create_asset(
        group_id="group-1",
        url="https://assets.test/intake/file.png",
        name="asset-name",
        asset_type="Image",
    )
    assert asset_id == "asset-1"


@pytest.mark.parametrize("asset_type", ["image", "Gif", "", "Image "])
def test_create_asset_rejects_invalid_asset_type(
    tmp_path: Path, asset_type: str
) -> None:
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(200, json={"Result": {"Id": "unexpected"}})

    client = _client(tmp_path, handler)
    with pytest.raises(ValueError, match="Image, Video or Audio"):
        client.create_asset(
            group_id="group-1",
            url="https://assets.test/intake/file.png",
            name="asset-name",
            asset_type=asset_type,
        )
    assert network_calls == 0


@pytest.mark.parametrize(
    "url", ["http://insecure.test/file.png", "asset://x", "", "not-a-url"]
)
def test_create_asset_rejects_non_https_intake_url(tmp_path: Path, url: str) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(500))
    with pytest.raises(ValueError, match="HTTPS"):
        client.create_asset(
            group_id="group-1", url=url, name="asset-name", asset_type="Image"
        )


def test_get_asset_interprets_active_pending_failed(tmp_path: Path) -> None:
    statuses = iter([
        {"Id": "asset-1", "AssetType": "Video", "Status": "Pending"},
        {"Id": "asset-1", "AssetType": "Video", "Status": "Processing"},
        {"Id": "asset-1", "AssetType": "Video", "Status": "Active"},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["Action"] == "GetAsset"
        assert json.loads(request.content) == {
            "Id": "asset-1",
            "ProjectName": "N0017",
        }
        return httpx.Response(200, json={"Result": next(statuses)})

    client = _client(tmp_path, handler)
    assert client.get_asset(asset_id="asset-1", expected_asset_type="Video").status == "Pending"
    assert client.get_asset(asset_id="asset-1", expected_asset_type="Video").status == "Pending"
    assert client.get_asset(asset_id="asset-1", expected_asset_type="Video").status == "Active"


def test_get_asset_failed_carries_error_code_and_message(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "Result": {
                    "Id": "asset-1",
                    "AssetType": "Image",
                    "Status": "Failed",
                    "Error": {
                        "Code": "SensitiveContentDetected",
                        "Message": "moderation blocked",
                    },
                }
            },
        )

    client = _client(tmp_path, handler)
    status = client.get_asset(asset_id="asset-1", expected_asset_type="Image")
    assert status.status == "Failed"
    assert status.error_code == "SensitiveContentDetected"
    assert status.error_message == "moderation blocked"


def test_get_asset_rejects_identity_mismatch(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "Result": {"Id": "other-asset", "AssetType": "Image", "Status": "Active"}
            },
        )

    client = _client(tmp_path, handler)
    with pytest.raises(BytePlusArkAssetError, match="身份不一致"):
        client.get_asset(asset_id="asset-1", expected_asset_type="Image")


def test_get_asset_rejects_unknown_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"Result": {"Id": "asset-1", "AssetType": "Image", "Status": "Weird"}},
        )

    client = _client(tmp_path, handler)
    with pytest.raises(BytePlusArkAssetError, match="未知状态"):
        client.get_asset(asset_id="asset-1", expected_asset_type="Image")


def test_wait_asset_active_polls_until_terminal(tmp_path: Path) -> None:
    statuses = iter(["Pending", "Pending", "Active"])
    sleeps: list[float] = []

    client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json={
                "Result": {
                    "Id": "asset-1",
                    "AssetType": "Audio",
                    "Status": next(statuses),
                }
            },
        ),
    )
    status = client.wait_asset_active(
        asset_id="asset-1",
        expected_asset_type="Audio",
        interval_s=5.0,
        timeout_s=60.0,
        sleep=sleeps.append,
    )
    assert status.status == "Active"
    assert sleeps == [5.0, 5.0]


def test_wait_asset_active_returns_failed_without_retry(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json={
                "Result": {
                    "Id": "asset-1",
                    "AssetType": "Video",
                    "Status": "Failed",
                    "Error": {"Code": "X", "Message": "download failed"},
                }
            },
        ),
    )
    status = client.wait_asset_active(
        asset_id="asset-1",
        expected_asset_type="Video",
        interval_s=5.0,
        timeout_s=60.0,
        sleep=lambda _s: None,
    )
    assert status.status == "Failed"
    assert status.error_message == "download failed"


def test_wait_asset_active_times_out(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            json={
                "Result": {"Id": "asset-1", "AssetType": "Video", "Status": "Pending"}
            },
        ),
    )
    with pytest.raises(TimeoutError, match="asset-1"):
        client.wait_asset_active(
            asset_id="asset-1",
            expected_asset_type="Video",
            interval_s=0.0,
            timeout_s=0.0,
            sleep=lambda _s: None,
        )


def test_delete_asset_sends_id(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["Action"] == "DeleteAsset"
        assert json.loads(request.content) == {
            "Id": "asset-1",
            "ProjectName": "N0017",
        }
        return httpx.Response(200, json={"Result": {}})

    client = _client(tmp_path, handler)
    client.delete_asset(asset_id="asset-1", expected_asset_type="Image")


def test_response_metadata_error_raises_with_code_and_body(tmp_path: Path) -> None:
    body = {
        "ResponseMetadata": {
            "Error": {"Code": "InvalidParameter", "Message": "bad group"},
        }
    }
    client = _client(
        tmp_path, lambda request: httpx.Response(400, json=body)
    )
    with pytest.raises(BytePlusArkAssetError, match="bad group") as caught:
        client.create_asset_group(name="project-1")
    assert caught.value.http_status == 400
    assert caught.value.provider_code == "InvalidParameter"
    assert caught.value.provider_body == body


def test_result_error_raises_for_non_get_asset_actions(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200, json={"Result": {"Error": {"Code": "Throttling", "Message": "slow down"}}}
        ),
    )
    with pytest.raises(BytePlusArkAssetError, match="slow down"):
        client.create_asset(
            group_id="group-1",
            url="https://assets.test/intake/file.png",
            name="asset-name",
            asset_type="Image",
        )


def test_missing_credentials_fail_before_network(tmp_path: Path) -> None:
    network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(200, json={"Result": {"Id": "unexpected"}})

    client = BytePlusArkAssetClient(
        _settings(
            tmp_path,
            byteplus_access_key_id=None,
            byteplus_secret_access_key=None,
        ),
        httpx.MockTransport(handler),
    )
    with pytest.raises(BytePlusArkAssetError, match="凭证未配置"):
        client.create_asset_group(name="project-1")
    assert network_calls == 0


def test_get_asset_retries_503_then_succeeds(tmp_path: Path) -> None:
    outcomes = iter([
        httpx.Response(503, json={"ResponseMetadata": {}}),
        httpx.Response(503, json={"ResponseMetadata": {}}),
        httpx.Response(
            200,
            json={"Result": {"Id": "asset-1", "AssetType": "Video", "Status": "Active"}},
        ),
    ])
    sleeps: list[float] = []
    client = BytePlusArkAssetClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: next(outcomes)),
        sleep=sleeps.append,
    )
    status = client.get_asset(asset_id="asset-1", expected_asset_type="Video")
    assert status.status == "Active"
    assert sleeps == [2.0, 5.0]


def test_get_asset_retries_transport_error_then_succeeds(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(
            200,
            json={"Result": {"Id": "asset-1", "AssetType": "Image", "Status": "Active"}},
        )

    sleeps: list[float] = []
    client = BytePlusArkAssetClient(
        _settings(tmp_path), httpx.MockTransport(handler), sleep=sleeps.append
    )
    assert (
        client.get_asset(asset_id="asset-1", expected_asset_type="Image").status
        == "Active"
    )
    assert calls == 2
    assert sleeps == [2.0]


def test_get_asset_retry_exhaustion_raises(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"ResponseMetadata": {}})

    client = BytePlusArkAssetClient(
        _settings(tmp_path),
        httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with pytest.raises(BytePlusArkAssetError):
        client.get_asset(asset_id="asset-1", expected_asset_type="Video")
    assert calls == 3


def test_delete_asset_retries_429_then_succeeds(tmp_path: Path) -> None:
    outcomes = iter([
        httpx.Response(429, json={"ResponseMetadata": {}}),
        httpx.Response(200, json={"Result": {}}),
    ])
    client = BytePlusArkAssetClient(
        _settings(tmp_path),
        httpx.MockTransport(lambda request: next(outcomes)),
        sleep=lambda _s: None,
    )
    client.delete_asset(asset_id="asset-1", expected_asset_type="Image")


def test_create_asset_does_not_retry_on_503(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"ResponseMetadata": {}})

    client = BytePlusArkAssetClient(
        _settings(tmp_path),
        httpx.MockTransport(handler),
        sleep=lambda _s: pytest.fail("CreateAsset must not retry"),
    )
    with pytest.raises(BytePlusArkAssetError):
        client.create_asset(
            group_id="group-1",
            url="https://assets.test/intake/file.png",
            name="asset-name",
            asset_type="Image",
        )
    assert calls == 1


def test_oversized_response_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.byteplus_assets as byteplus_assets

    monkeypatch.setattr(byteplus_assets, "_ARK_MAX_RESPONSE_BYTES", 16)
    client = _client(
        tmp_path,
        lambda request: httpx.Response(
            200,
            content=b"x" * 64,
            headers={"Content-Type": "application/json"},
        ),
    )
    with pytest.raises(BytePlusArkAssetError, match="大小限制"):
        client.create_asset_group(name="project-1")
