"""BytePlus AIGC 素材白名单控制面客户端。

CreateAssetGroup / CreateAsset / GetAsset / DeleteAsset 四个 Action，火山 V4
HMAC 签名只用 stdlib（hashlib/hmac），不引入 volcengine SDK。

字段级契约来源（事实，2026-09-29 ssh 只读核对远端 duet-video-v2
server/src/services/arkAssetClient.service.ts）：
- 端点：POST {byteplus_ark_api_base}/?Action=<Action>&Version=2024-01-01，
  service=ark，region 可配（国际站 ap-southeast-1）。
- 签名头集合固定为 content-type;host;x-content-sha256;x-date；X-Date 格式
  YYYYMMDDTHHMMSSZ；X-Content-Sha256 是 body 的 sha256 hex；kSigning 派生链
  HMAC(SK, date) → HMAC(.., region) → HMAC(.., "ark") → HMAC(.., "request")。
- intl 的**每个** Action body 都在签名前注入 ProjectName（withProjectName
  应用于 callArk 内全部 Action，不是只有 Create*）。
- 控制面不存在 DeleteAssetGroup/ListAssetGroups：ArkControlAction 联合类型
  只有 CreateAssetGroup|CreateAsset|GetAsset|DeleteAsset|ListEndpoints|
  GetApiKey。空 group 残留由平台侧管理，客户端不提供删除。
- GetAsset 的 Failed 是 Action 结果（Result.Status=Failed + Result.Error），
  不是 RPC 错误信封；Pending/Processing 都归一为"未就绪"。
- 错误信封：ResponseMetadata.Error（含 Code/Message），部分 Action 业务错误
  在 Result.Error。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from app.config import Settings

_ARK_SERVICE = "ark"
_ARK_VERSION = "2024-01-01"
_ARK_CONTENT_TYPE = "application/json"
_ARK_SIGNED_HEADERS = "content-type;host;x-content-sha256;x-date"
# 响应体上限（事实：远端 arkAssetClient.service.ts MAX_RESPONSE_BYTES）。
_ARK_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
# 读/删类 Action 的瞬时故障有界重试（事实：远端 OPERATION_SPECS——
# GetAsset=SAFE_READ、DeleteAsset=LOOKUP_GUARDED_DELETE 可重试 429/5xx/
# 传输错误；CreateAssetGroup/CreateAsset=UNSAFE_CREATE 一律不重试，
# 因为客户端无法证明 POST 未送达）。退避序列沿用 providers.RETRY_BACKOFF_S。
_ARK_RETRY_BACKOFF_S = (2.0, 5.0)
_ARK_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_ARK_RETRYABLE_ACTIONS = frozenset({"GetAsset", "DeleteAsset"})
# GetAsset 轮询间隔（CreateAsset 后供应商异步拉取+审核，必须 Active 才能提交；
# 事实：远端 arkAssetClient.service.ts createAsset 注释）。
_ARK_TERMINAL_STATUSES = frozenset({"Active", "Failed"})
_ARK_PENDING_STATUSES = frozenset({"Pending", "Processing"})
# CreateAsset AssetType 只允许这三个值（事实：远端 requiredAssetType）。
ASSET_TYPES = frozenset({"Image", "Video", "Audio"})


class BytePlusArkAssetError(RuntimeError):
    """控制面调用失败；携带供应商结构化错误供分桶与排查。"""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        provider_code: str | None = None,
        provider_body: Any = None,
    ) -> None:
        self.http_status = http_status
        self.provider_code = provider_code
        self.provider_body = provider_body
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ArkAssetStatus:
    status: str  # "Active" | "Pending" | "Failed"
    error_code: str | None = None
    error_message: str | None = None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def sign_ark_request(
    action: str,
    body: str,
    *,
    access_key_id: str,
    secret_access_key: str,
    host: str,
    region: str,
    iso_date: str,
) -> dict[str, str]:
    """火山 V4 HMAC 签名头（逐字段对齐远端 signArkHeaders，见模块 docstring）。"""
    short_date = iso_date[:8]
    query = f"Action={action}&Version={_ARK_VERSION}"  # Action < Version 已有序
    payload_hash = _sha256_hex(body.encode("utf-8"))
    canonical_request = "\n".join([
        "POST",
        "/",
        query,
        f"content-type:{_ARK_CONTENT_TYPE}",
        f"host:{host}",
        f"x-content-sha256:{payload_hash}",
        f"x-date:{iso_date}",
        "",
        _ARK_SIGNED_HEADERS,
        payload_hash,
    ])
    scope = f"{short_date}/{region}/{_ARK_SERVICE}/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        iso_date,
        scope,
        _sha256_hex(canonical_request.encode("utf-8")),
    ])
    k_date = _hmac_sha256(secret_access_key.encode("utf-8"), short_date)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, _ARK_SERVICE)
    k_signing = _hmac_sha256(k_service, "request")
    signature = hmac.new(
        k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        "Content-Type": _ARK_CONTENT_TYPE,
        "Host": host,
        "X-Date": iso_date,
        "X-Content-Sha256": payload_hash,
        "Authorization": (
            f"HMAC-SHA256 Credential={access_key_id}/{scope}, "
            f"SignedHeaders={_ARK_SIGNED_HEADERS}, Signature={signature}"
        ),
    }


def _now_iso_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class BytePlusArkAssetClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self._sleep = sleep
        parsed = urlsplit(settings.byteplus_ark_api_base)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("BYTEPLUS_ARK_API_BASE must be an HTTPS base URL")
        self._host = parsed.hostname

    def _credentials(self) -> tuple[str, str]:
        access_key_id = self.settings.byteplus_access_key_id
        secret_access_key = self.settings.byteplus_secret_access_key
        if not access_key_id or not secret_access_key:
            raise BytePlusArkAssetError(
                "BytePlus 白名单凭证未配置（BYTEPLUS_ACCESS_KEY_ID/"
                "BYTEPLUS_SECRET_ACCESS_KEY）",
                http_status=401,
            )
        return access_key_id, secret_access_key

    def _http_client_kwargs(self) -> dict[str, Any]:
        """与 SeedanceClient 同一隔离模式：不信任环境代理，显式走配置代理。"""
        client_kwargs: dict[str, Any] = {
            "timeout": self.settings.request_timeout_s,
            "transport": self.transport,
            "trust_env": False,
        }
        if self.settings.seedance_proxy is not None:
            client_kwargs["proxy"] = self.settings.seedance_proxy
        return client_kwargs

    def _call(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        access_key_id, secret_access_key = self._credentials()
        # ProjectName 注入发生在签名之前，参与 payload hash（事实：远端
        # callArk 先 withProjectName 再 signArkHeaders）。
        body_str = json.dumps(
            {**body, "ProjectName": self.settings.byteplus_project_name},
            separators=(",", ":"),
        )
        headers = sign_ark_request(
            action,
            body_str,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            host=self._host,
            region=self.settings.byteplus_region,
            iso_date=_now_iso_date(),
        )
        url = (
            f"{self.settings.byteplus_ark_api_base}"
            f"/?Action={action}&Version={_ARK_VERSION}"
        )
        # GetAsset/DeleteAsset 对 429/5xx/传输错误做有界重试（幂等读/删；
        # 事实见 _ARK_RETRYABLE_ACTIONS 注释）；Create* 撞任何失败都直接
        # 抛——无法证明 POST 未送达时不重投，重放（resume）时才重新加白，
        # 代价仅是可能多一个孤儿 asset（免费，平台 LRU 500 兜底）。
        backoff: tuple[float, ...] = (
            _ARK_RETRY_BACKOFF_S if action in _ARK_RETRYABLE_ACTIONS else ()
        )
        attempt = 0
        while True:
            try:
                with httpx.Client(**self._http_client_kwargs()) as client:
                    response = client.post(url, headers=headers, content=body_str)
            except httpx.HTTPError as exc:
                if attempt < len(backoff):
                    self._sleep(backoff[attempt])
                    attempt += 1
                    continue
                raise BytePlusArkAssetError(
                    f"资产服务网络请求失败: {exc}"
                ) from exc
            if (
                response.status_code in _ARK_RETRYABLE_STATUSES
                and attempt < len(backoff)
            ):
                self._sleep(backoff[attempt])
                attempt += 1
                continue
            break
        declared_length = response.headers.get("content-length")
        if (
            declared_length is not None
            and declared_length.isdigit()
            and int(declared_length) > _ARK_MAX_RESPONSE_BYTES
        ):
            raise BytePlusArkAssetError(
                "资产服务响应超过大小限制",
                http_status=response.status_code,
            )
        if len(response.content) > _ARK_MAX_RESPONSE_BYTES:
            raise BytePlusArkAssetError(
                "资产服务响应超过大小限制",
                http_status=response.status_code,
            )
        try:
            payload = json.loads(response.content)
        except ValueError as exc:
            raise BytePlusArkAssetError(
                "资产服务返回格式无效",
                http_status=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise BytePlusArkAssetError(
                "资产服务返回格式无效",
                http_status=response.status_code,
            )
        metadata_error = payload.get("ResponseMetadata")
        error = (
            metadata_error.get("Error")
            if isinstance(metadata_error, dict)
            else None
        ) or payload.get("Error")
        if isinstance(error, dict) and (error.get("Code") or error.get("Message")):
            raise BytePlusArkAssetError(
                str(error.get("Message") or "资产服务请求失败"),
                http_status=response.status_code,
                provider_code=(
                    str(error["Code"]) if error.get("Code") is not None else None
                ),
                provider_body=payload,
            )
        if not response.is_success:
            raise BytePlusArkAssetError(
                "资产服务请求失败",
                http_status=response.status_code,
                provider_body=payload,
            )
        result_error = (payload.get("Result") or {}).get("Error")
        # GetAsset 用 Result.Status=Failed + Result.Error 报告入库失败，
        # 那是 Action 结果而非 RPC 错误（事实：远端 callArk 的 resultError
        # 特例），由 get_asset 自己解释。
        if action != "GetAsset" and isinstance(result_error, dict) and (
            result_error.get("Code") or result_error.get("Message")
        ):
            raise BytePlusArkAssetError(
                str(result_error.get("Message") or "资产服务返回失败响应"),
                http_status=response.status_code,
                provider_code=(
                    str(result_error["Code"])
                    if result_error.get("Code") is not None
                    else None
                ),
                provider_body=payload,
            )
        return payload

    @staticmethod
    def _result_id(payload: dict[str, Any]) -> str:
        result = payload.get("Result")
        asset_id = result.get("Id") if isinstance(result, dict) else None
        if not isinstance(asset_id, str) or not asset_id:
            raise BytePlusArkAssetError(
                "资产服务返回结果缺少资产ID", provider_body=payload
            )
        return asset_id

    def create_asset_group(self, *, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("asset group name must be a non-empty string")
        return self._result_id(
            self._call(
                "CreateAssetGroup", {"GroupType": "AIGC", "Name": name.strip()}
            )
        )

    def create_asset(
        self,
        *,
        group_id: str,
        url: str,
        name: str,
        asset_type: str,
    ) -> str:
        if asset_type not in ASSET_TYPES:
            raise ValueError("asset_type must be Image, Video or Audio")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("asset URL must be a public HTTPS intake URL")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("asset name must be a non-empty string")
        return self._result_id(
            self._call(
                "CreateAsset",
                {
                    "GroupId": group_id.strip(),
                    "URL": url,
                    "Name": name.strip(),
                    "AssetType": asset_type,
                },
            )
        )

    def get_asset(self, *, asset_id: str, expected_asset_type: str) -> ArkAssetStatus:
        if expected_asset_type not in ASSET_TYPES:
            raise ValueError("expected_asset_type must be Image, Video or Audio")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        payload = self._call("GetAsset", {"Id": asset_id.strip()})
        result = payload.get("Result")
        if not isinstance(result, dict):
            raise BytePlusArkAssetError(
                "资产服务返回格式无效", provider_body=payload
            )
        raw_status = result.get("Status")
        if raw_status not in _ARK_TERMINAL_STATUSES | _ARK_PENDING_STATUSES:
            raise BytePlusArkAssetError(
                "资产服务返回未知状态",
                provider_code="provider_protocol_error",
                provider_body=payload,
            )
        # 素材身份不一致说明请求/响应错配，按协议错误处理（事实：远端
        # assertGetAssetProtocol 同样拒绝）。
        if result.get("Id") != asset_id.strip() or (
            result.get("AssetType") != expected_asset_type
        ):
            raise BytePlusArkAssetError(
                "资产服务返回的素材身份不一致",
                provider_code="asset_identity_mismatch",
                provider_body=payload,
            )
        if raw_status == "Active":
            return ArkAssetStatus(status="Active")
        if raw_status == "Failed":
            error = result.get("Error")
            error = error if isinstance(error, dict) else {}
            return ArkAssetStatus(
                status="Failed",
                error_code=(
                    str(error["Code"]) if error.get("Code") is not None else None
                ),
                error_message=(
                    str(error["Message"])
                    if error.get("Message") is not None
                    else None
                ),
            )
        return ArkAssetStatus(status="Pending")

    def wait_asset_active(
        self,
        *,
        asset_id: str,
        expected_asset_type: str,
        interval_s: float,
        timeout_s: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ArkAssetStatus:
        """轮询 GetAsset 到 Active|Failed；超时抛 TimeoutError。

        CreateAsset 是异步拉取+审核，必须 Active 才能提交生成（事实：远端
        createAsset docstring）；Failed 原样返回给调用方分类。
        """
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.get_asset(
                asset_id=asset_id, expected_asset_type=expected_asset_type
            )
            if status.status in _ARK_TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"asset {asset_id} stayed pending past {timeout_s}s"
                )
            sleep(interval_s)

    def delete_asset(self, *, asset_id: str, expected_asset_type: str) -> None:
        if expected_asset_type not in ASSET_TYPES:
            raise ValueError("expected_asset_type must be Image, Video or Audio")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")
        self._call("DeleteAsset", {"Id": asset_id.strip()})
