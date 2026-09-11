from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import os
import re
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from app.reference_proxy import ReferenceTransport


ResolverResult = Sequence[str | ipaddress.IPv4Address | ipaddress.IPv6Address]
Resolver = Callable[[str, int], ResolverResult | Awaitable[ResolverResult]]
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_PORTS = {"http": 80, "https": 443}
TIKWM_API = "https://www.tikwm.com/api/"


def is_tiktok_page_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (parsed.scheme in DEFAULT_PORTS
            and (host == "tiktok.com" or host.endswith(".tiktok.com"))
            and re.search(r"/video/[0-9]+(?:/|$)", parsed.path) is not None)


async def _tiktok_media_url(url: str, client: httpx.AsyncClient, resolver: Resolver) -> str:
    """Restore duet-ad1's TikWM GET/code=0/data.play protocol, without retries.

    The returned URL remains untrusted and goes through the normal pinned
    downloader. The caller retains the original page as input provenance.
    """
    if not re.search(r"/video/[0-9]+(?:/|$)", urlsplit(url).path):
        raise ReferenceDownloadError("tiktok_resolution_failed", "无法解析 TikTok 视频编号", status_code=422)
    target = await _approved_target(str(httpx.URL(TIKWM_API, params={"url": url})), resolver)
    try:
        async with client.stream("GET", target.pinned_url,
                headers={"Host": target.host_header, "Accept": "application/json", "Accept-Encoding": "identity"},
                extensions={"sni_hostname": target.sni_hostname}) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    raise ValueError("TikWM response too large")
            payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("TikWM response is not an object")
        data = payload.get("data")
        play = data.get("play") if isinstance(data, dict) else None
        if payload.get("code") != 0 or not isinstance(play, str) or not play.strip():
            raise ValueError("TikWM response lacks media URL")
        return play.strip()
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise ReferenceDownloadError("tiktok_resolution_failed", "TikTok 视频解析失败，请稍后重试或上传视频文件", status_code=422) from exc


async def _resolve_tiktok_with_retries(url: str, client: httpx.AsyncClient, resolver: Resolver) -> str:
    for attempt in range(3):
        try:
            return await _tiktok_media_url(url, client, resolver)
        except ReferenceDownloadError as exc:
            if exc.code != "tiktok_resolution_failed" or attempt == 2:
                raise
            await asyncio.sleep(10)
    raise AssertionError("unreachable")


class ReferenceDownloadError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ApprovedTarget:
    original_url: str
    pinned_url: str
    host_header: str
    sni_hostname: str


async def resolve_addresses(hostname: str, port: int) -> ResolverResult:
    """Resolve A/AAAA records without allowing the HTTP client to resolve again."""

    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    return [record[4][0] for record in records if record[0] in {socket.AF_INET, socket.AF_INET6}]


def _invalid(message: str = "视频链接无效") -> ReferenceDownloadError:
    return ReferenceDownloadError("invalid_reference_url", message, status_code=400)


async def _approved_target(url: str, resolver: Resolver) -> ApprovedTarget:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise _invalid() from exc

    if scheme not in DEFAULT_PORTS or not hostname:
        raise _invalid()
    if parsed.username is not None or parsed.password is not None:
        raise _invalid("视频链接不能包含用户凭据")
    if parsed.netloc.rsplit("@", 1)[-1].endswith(":"):
        raise _invalid("视频链接端口无效")
    expected_port = DEFAULT_PORTS[scheme]
    if port is not None and port != expected_port:
        raise _invalid("视频链接端口不受支持")
    port = expected_port

    hostname = hostname.rstrip(".")
    if not hostname or "%" in hostname:
        raise _invalid()
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise _invalid() from exc

    try:
        numeric_host = ipaddress.ip_address(ascii_hostname)
        raw_addresses: ResolverResult = [numeric_host]
    except ValueError:
        try:
            result = resolver(ascii_hostname, port)
            raw_addresses = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            raise _invalid("视频链接无法解析") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        for raw_address in raw_addresses:
            address = raw_address if isinstance(raw_address, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else ipaddress.ip_address(raw_address)
            if address not in addresses:
                addresses.append(address)
    except (TypeError, ValueError) as exc:
        raise _invalid("视频链接无法解析") from exc
    if not addresses:
        raise _invalid("视频链接无法解析")
    # Pin only a safe candidate; an unusable sibling DNS answer need not
    # invalidate the public addresses returned for the same hostname.
    addresses = [address for address in addresses if address.is_global and not address.is_multicast]
    if not addresses:
        raise ReferenceDownloadError(
            "reference_url_forbidden",
            "视频链接不能指向内部或非公网网络",
            status_code=400,
        )

    selected = addresses[0]
    pinned_host = f"[{selected.compressed}]" if selected.version == 6 else selected.compressed
    original_host = f"[{ascii_hostname}]" if numeric_host_or_none(ascii_hostname, version=6) else ascii_hostname
    explicit_port = parsed.port is not None
    host_header = f"{original_host}:{port}" if explicit_port else original_host
    path = parsed.path or "/"
    pinned_url = urlunsplit((scheme, f"{pinned_host}:{port}", path, parsed.query, ""))
    return ApprovedTarget(
        original_url=urlunsplit((scheme, parsed.netloc, path, parsed.query, "")),
        pinned_url=pinned_url,
        host_header=host_header,
        sni_hostname=ascii_hostname,
    )


def numeric_host_or_none(hostname: str, *, version: int) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return None
    return address if address.version == version else None


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    value = raw.strip()
    if not value or not value.isascii() or not value.isdecimal() or len(value) > 20:
        raise ReferenceDownloadError(
            "reference_download_failed",
            "视频链接返回了无效的 Content-Length",
            status_code=502,
        )
    return int(value)


async def download_reference_url(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    timeout_s: float,
    resolver: Resolver = resolve_addresses,
    transport: httpx.AsyncBaseTransport | None = None,
    max_redirects: int = 5,
) -> int:
    """Download a public URL to ``destination`` without a DNS-rebinding window.

    Every redirect is resolved and approved independently.  The actual TCP URL
    contains the approved numeric address while Host and TLS SNI retain the
    original hostname.  A completed body is atomically published from a
    same-directory temporary file.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(max_redirects, bool) or not isinstance(max_redirects, int) or max_redirects < 0:
        raise ValueError("max_redirects must be a non-negative integer")

    current_url = url if isinstance(url, str) else ""
    if (
        not current_url
        or current_url != current_url.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in current_url)
    ):
        raise _invalid()
    redirects = 0
    try:
        async with (
            asyncio.timeout(timeout_s),
            httpx.AsyncClient(
                transport=transport if transport is not None else ReferenceTransport(),
                trust_env=False,
                follow_redirects=False,
                timeout=timeout_s,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client,
        ):
            if is_tiktok_page_url(current_url):
                # Validate the user URL before sending it to the fixed parser.
                await _approved_target(current_url, resolver)
                current_url = await _resolve_tiktok_with_retries(current_url, client, resolver)
            while True:
                target = await _approved_target(current_url, resolver)
                async with client.stream(
                    "GET",
                    target.pinned_url,
                    headers={"Host": target.host_header, "Accept": "*/*", "Accept-Encoding": "identity"},
                    extensions={"sni_hostname": target.sni_hostname},
                ) as response:
                    if response.status_code in REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise ReferenceDownloadError(
                                "reference_download_failed",
                                "视频链接重定向缺少目标地址",
                                status_code=502,
                            )
                        if redirects >= max_redirects:
                            raise ReferenceDownloadError(
                                "reference_redirect_limit",
                                "视频链接重定向次数过多",
                                status_code=400,
                            )
                        redirects += 1
                        current_url = urljoin(target.original_url, location)
                        continue
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise ReferenceDownloadError(
                            "reference_download_failed",
                            f"视频链接返回 HTTP {response.status_code}",
                            status_code=502,
                        ) from exc

                    declared_length = _content_length(response)
                    if declared_length is not None and declared_length > max_bytes:
                        raise ReferenceDownloadError(
                            "reference_too_large",
                            "链接视频超过大小限制",
                            status_code=413,
                        )
                    content_encoding = response.headers.get("content-encoding", "identity").strip().lower()
                    if content_encoding not in {"", "identity"}:
                        raise ReferenceDownloadError(
                            "reference_content_encoding",
                            "视频链接返回了不受支持的压缩编码",
                            status_code=502,
                        )

                    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{destination.name}.",
                        suffix=".tmp",
                        dir=destination.parent,
                    )
                    temporary = Path(temporary_name)
                    total = 0
                    try:
                        with os.fdopen(descriptor, "wb", closefd=True) as handle:
                            async for chunk in response.aiter_bytes():
                                if not chunk:
                                    continue
                                total += len(chunk)
                                if total > max_bytes:
                                    raise ReferenceDownloadError(
                                        "reference_too_large",
                                        "链接视频超过大小限制",
                                        status_code=413,
                                    )
                                handle.write(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.chmod(temporary, 0o600)
                        os.replace(temporary, destination)
                    except BaseException:
                        try:
                            temporary.unlink()
                        except FileNotFoundError:
                            pass
                        raise
                    return total
    except ReferenceDownloadError:
        raise
    except TimeoutError as exc:
        raise ReferenceDownloadError(
            "reference_download_timeout",
            "下载链接视频超时",
            status_code=504,
        ) from exc
    except httpx.TimeoutException as exc:
        raise ReferenceDownloadError(
            "reference_download_timeout",
            "下载链接视频超时",
            status_code=504,
        ) from exc
    except httpx.HTTPError as exc:
        raise ReferenceDownloadError(
            "reference_download_failed",
            "无法下载链接视频",
            status_code=502,
        ) from exc
    except OSError as exc:
        raise ReferenceDownloadError(
            "reference_download_failed",
            "无法保存链接视频",
            status_code=500,
        ) from exc
