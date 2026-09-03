"""URL 参考视频下载：http(s) 直链与 TikTok 视频页，带 SSRF 防护。

移植 TrendScout tools/lib/media.py 的最小必要逻辑（无缓存）：DNS pinning
（解析一次后固定 IP 直连，防 DNS rebinding）、每次跳转独立重校验、
Content-Length 预检 + 流式写盘上限 + 整体 deadline。proxy 为空即直连；
非空时域名解析改问代理对面的 DoH（本机 DNS 可能被污染），校验与拨号仍共用同一 IP。
"""
import http.client
import ipaddress
import logging
import re
import socket
import ssl
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from app.retry import RetryPolicy, run_with_retry
from app.storage import ALLOWED_EXT

_CHUNK = 64 * 1024
_MAX_REDIRECTS = 5
_TIKWM_API = "https://www.tikwm.com/api/"
_DOH_URL = "https://1.1.1.1/dns-query"

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """下载或 URL 校验失败（HTTP 层转 422）。"""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code


def _retryable_http_status(status: int) -> bool:
    return status in {408, 425, 429} or 500 <= status <= 599


def _public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified)


def _local_resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as e:
        raise DownloadError(f"DNS resolution failed: {host}", retryable=True) from e
    return [info[4][0] for info in infos if len(info) >= 5 and info[4]]


def _doh_resolve(host: str, *, proxy: str | None, timeout: int, transport=None) -> list[str]:
    """经代理向 DoH 解析器要 A 记录。查不到就报错，绝不回落到会撒谎的本机解析。"""
    options: dict = {"timeout": timeout, "trust_env": False}
    if transport is not None:
        options["transport"] = transport
    elif proxy:
        options["proxy"] = proxy
    try:
        with httpx.Client(**options) as client:
            response = client.get(
                _DOH_URL, params={"name": host, "type": "A"},
                headers={"accept": "application/dns-json"},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as e:
        raise DownloadError(
            f"DoH resolution failed: {host} ({str(e)[:120]})",
            retryable=_retryable_http_status(e.response.status_code),
        ) from e
    except (httpx.HTTPError, ValueError) as e:
        raise DownloadError(
            f"DoH resolution failed: {host} ({str(e)[:120]})", retryable=True
        ) from e
    answers = payload.get("Answer") if isinstance(payload, dict) else None
    addresses = [
        str(item["data"]) for item in (answers or [])
        if isinstance(item, dict) and item.get("type") == 1 and item.get("data")
    ]
    if not addresses:
        raise DownloadError(f"DoH resolution found no A record: {host}", retryable=True)
    return addresses


def _resolver_for(proxy: str | None, timeout: int):
    """有代理就问代理对面的 DoH（本机 DNS 对 CDN 域名可能被污染，钉到死 IP 隧道必炸）；
    没配代理才用本机解析。校验与拨号共用同一次解析结果，保住"校验的=真连的"。"""
    if not proxy:
        return _local_resolve
    return lambda host: _doh_resolve(host, proxy=proxy, timeout=timeout)


def _validate_public_url(url: str, resolve):
    """仅 http(s)；解析所得全部 IP 必须公网。返回 (parsed, host, port, 钉死的 IP)。"""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DownloadError("only http(s) URLs are supported")
    host = parsed.hostname.strip("[]").lower()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = [str(ipaddress.ip_address(host))]  # 字面 IP 无需解析，当场判死
    except ValueError:
        addresses = list(resolve(host))
    if not addresses or any(not _public_ip(a) for a in addresses):
        raise DownloadError("URL resolves to a local or private address, refused")
    return parsed, host, port, addresses[0]


def _read_proxy_headers(sock) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise DownloadError(
                "proxy closed connection before tunnel established", retryable=True
            )
        data += chunk
        if len(data) > 64 * 1024:
            raise DownloadError("proxy response headers abnormally long")
    return data


def _proxy_tunnel(proxy: str, address: str, port: int, timeout: int):
    """经 HTTP CONNECT 代理开到字面 IP 的隧道；失败即报错，绝不回落直连。"""
    parsed = urlparse(proxy)
    if parsed.scheme != "http" or not parsed.hostname:
        raise DownloadError(f"proxy must be http://host:port, got {proxy!r}")
    target = f"[{address}]:{port}" if ":" in address else f"{address}:{port}"
    try:
        sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout)
    except OSError as e:
        raise DownloadError(f"proxy unreachable: {proxy} ({e})", retryable=True) from e
    try:
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii"))
        head, _, extra = _read_proxy_headers(sock).partition(b"\r\n\r\n")
        if extra:
            # 隧道里永远由我方先开口，代理抢跑的字节无处安放：宁可炸
            raise DownloadError("proxy sent data before tunnel established")
        status_line = head.split(b"\r\n", 1)[0]
        if status_line.split(b" ")[1:2] != [b"200"]:
            raise DownloadError(
                f"proxy refused tunnel to {target}: {status_line.decode('latin-1', 'replace')}",
                retryable=True,
            )
    except OSError as e:
        sock.close()
        raise DownloadError(
            f"proxy tunnel failed: {proxy} -> {target} ({e})", retryable=True
        ) from e
    except Exception:
        sock.close()
        raise
    return sock


def _pinned_socket(address: str, port: int, *, proxy: str | None, timeout: int):
    """连到刚校验过的公网字面 IP；走不走代理都是同一个对端。"""
    if not proxy:
        return socket.create_connection((address, port), timeout)
    return _proxy_tunnel(proxy, address, port, timeout)


def _open_pinned(parsed, host: str, port: int, address: str, *, timeout: int, proxy: str | None):
    """向已校验的 (host, address) 发 GET；host 保留原主机名供 Host 头与 TLS SNI/证书校验。"""

    def dial(connection):
        return _pinned_socket(address, connection.port, proxy=proxy, timeout=connection.timeout)

    class PinnedHTTP(http.client.HTTPConnection):
        def connect(self):
            self.sock = dial(self)

    class PinnedHTTPS(http.client.HTTPSConnection):
        def connect(self):
            self.sock = self._context.wrap_socket(dial(self), server_hostname=self.host)

    path = urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    if parsed.scheme == "https":
        connection = PinnedHTTPS(host, port=port, timeout=timeout, context=ssl.create_default_context())
    else:
        connection = PinnedHTTP(host, port=port, timeout=timeout)
    connection.request("GET", path, headers={"Accept": "video/*,*/*", "User-Agent": "duet-ad1/1.0"})
    return connection, connection.getresponse()


def download_public_video(
    url: str,
    dest: Path,
    *,
    proxy: str | None = None,
    max_bytes: int,
    timeout: int,
) -> None:
    """钉 DNS 流式下载；每次跳转独立重校验；超限/超时/空文件即报错。

    连接/读取阶段的 OSError、ValueError（非法端口）、http.client.HTTPException
    统一归一为 DownloadError，保证 HTTP 层只面对这一种失败。
    """
    current = url
    temporary = dest.with_suffix(dest.suffix + ".part")
    deadline = time.monotonic() + timeout
    temporary.unlink(missing_ok=True)
    resolve = _resolver_for(proxy, timeout)
    try:
        for _ in range(_MAX_REDIRECTS):
            validated = _validate_public_url(current, resolve)
            connection, response = _open_pinned(*validated, timeout=timeout, proxy=proxy)
            try:
                status = int(getattr(response, "status", 0) or 0)
                location = response.getheader("location")
                if 300 <= status < 400 and location:
                    current = urljoin(current, location)
                    continue
                if status < 200 or status >= 300:
                    raise DownloadError(
                        f"download failed: HTTP {status}",
                        retryable=_retryable_http_status(status),
                    )
                expected = response.getheader("content-length")
                if expected and expected.isdigit() and int(expected) > max_bytes:
                    raise DownloadError(
                        f"file exceeds {max_bytes} bytes",
                        code="source_too_large",
                    )
                total = 0
                with temporary.open("wb") as handle:
                    while True:
                        if time.monotonic() > deadline:
                            raise DownloadError("download timed out", retryable=True)
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise DownloadError(
                                f"file exceeds {max_bytes} bytes",
                                code="source_too_large",
                            )
                        handle.write(chunk)
                if total == 0:
                    raise DownloadError("downloaded file is empty", retryable=True)
                temporary.replace(dest)
                return
            finally:
                connection.close()
        raise DownloadError("too many redirects")
    except Exception as e:
        temporary.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        if isinstance(e, DownloadError):
            raise
        if isinstance(e, (OSError, http.client.HTTPException)):
            retryable = not isinstance(e, ssl.SSLCertVerificationError)
            raise DownloadError(f"download failed: {e}", retryable=retryable) from e
        if isinstance(e, ValueError):
            raise DownloadError(f"download failed: {e}") from e
        raise


def is_tiktok_page_url(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and (host == "tiktok.com" or host.endswith(".tiktok.com"))


def tiktok_video_id(url: str) -> str:
    match = re.search(r"/video/([A-Za-z0-9_-]+)", url or "")
    if not match:
        raise DownloadError("cannot parse TikTok video id from URL")
    return match.group(1)


def tiktok_video_facts(
    url: str,
    proxy: str | None = None,
    *,
    timeout: int = 120,
    api_transport=None,
) -> dict:
    """问固定的 TikWM API 拿 {video_id, play}；重试由 fetch_reference 统一控制。"""
    video_id = tiktok_video_id(url)
    options: dict = {"timeout": timeout, "follow_redirects": False, "trust_env": False}
    if api_transport is not None:
        options["transport"] = api_transport
    elif proxy:
        options["proxy"] = proxy
    with httpx.Client(**options) as client:
        try:
            response = client.get(_TIKWM_API, params={"url": url})
        except httpx.HTTPError as e:
            raise DownloadError(
                f"tikwm request failed: {str(e)[:120]}", retryable=True
            ) from e
        if response.status_code < 200 or response.status_code >= 300:
            raise DownloadError(
                f"tikwm request failed: HTTP {response.status_code}",
                retryable=_retryable_http_status(response.status_code),
            )
        try:
            payload = response.json()
        except ValueError as e:
            raise DownloadError("tikwm response is not JSON", retryable=True) from e
        if not isinstance(payload, dict):
            raise DownloadError("tikwm response is not an object", retryable=True)
        code = payload.get("code")
        if code == 0:
            data = payload.get("data") or {}
            play = data.get("play")
            if not isinstance(play, str) or not play.strip():
                raise DownloadError("tikwm response lacks media URL", retryable=True)
            return {"video_id": video_id, "play": play.strip()}
        raise DownloadError(f"tikwm returned code={code}", retryable=code == -1)


def download_tiktok_video(url: str, dest: Path, *, proxy: str | None, max_bytes: int, timeout: int) -> None:
    """TikWM 解析出 play 直链，再经钉 DNS 下载器拉取（无缓存）。"""
    facts = tiktok_video_facts(url, proxy, timeout=timeout)
    download_public_video(facts["play"], dest, proxy=proxy, max_bytes=max_bytes, timeout=timeout)


def _dest_for(url: str, cdir: Path) -> Path:
    """落盘为 source.<ext>（pipeline 契约）：ext 取 URL path 后缀（白名单内），否则默认 .mp4。"""
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in ALLOWED_EXT:
        ext = ".mp4"
    return cdir / f"source{ext}"


def fetch_reference(url: str, cdir: Path, settings) -> Path:
    """按 URL 自动分流 TikTok/直链，下载到 cdir/source.<ext> 并返回路径。"""
    url = url.strip()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    dest = _dest_for(url, cdir)

    def attempt() -> Path:
        if is_tiktok_page_url(url):
            download_tiktok_video(
                url,
                dest,
                proxy=settings.tiktok_proxy or None,
                max_bytes=max_bytes,
                timeout=settings.download_timeout_s,
            )
        else:
            download_public_video(
                url,
                dest,
                max_bytes=max_bytes,
                timeout=settings.download_timeout_s,
            )
        return dest

    policy = RetryPolicy(settings.retry_count, settings.retry_interval_s)

    def report(retry_number: int, _exc: Exception) -> None:
        log.warning(
            "reference download failed; retry %d/%d in %.1fs",
            retry_number,
            policy.retries,
            policy.interval_s,
        )

    return run_with_retry(
        attempt,
        policy=policy,
        is_retryable=lambda exc: isinstance(exc, DownloadError) and exc.retryable,
        on_retry=report,
    )
