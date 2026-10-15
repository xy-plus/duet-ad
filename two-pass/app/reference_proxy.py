"""HTTP proxy support for already-approved, IP-pinned reference requests.

httpcore's CONNECT implementation uses the URL IP for TLS too. Its public
network-backend hook preserves CONNECT while supplying the approved TLS name.
"""
import os

import httpcore
import httpx


def configured_proxy(scheme: str) -> str | None:
    name = "HTTPS_PROXY" if scheme == "https" else "HTTP_PROXY"
    value = os.environ.get(name.lower(), os.environ.get(name, "")).strip()
    if not value:
        return None
    # Only HTTP CONNECT proxies are supported: their transport has no TLS
    # handshake of its own that could be confused with the target handshake.
    if httpx.URL(value).scheme != "http":
        raise ValueError("Reference downloads require an HTTP proxy URL")
    return value


class _TargetTLSStream(httpcore.AsyncNetworkStream):
    def __init__(self, stream, hostname):
        self.stream = stream
        self.hostname = hostname

    async def read(self, max_bytes, timeout=None):
        return await self.stream.read(max_bytes, timeout)

    async def write(self, buffer, timeout=None):
        return await self.stream.write(buffer, timeout)

    async def aclose(self):
        await self.stream.aclose()

    def get_extra_info(self, info):
        return self.stream.get_extra_info(info)

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return await self.stream.start_tls(ssl_context, self.hostname, timeout)


class _TargetTLSBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, hostname):
        self.backend = httpcore.AnyIOBackend()
        self.hostname = hostname

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        stream = await self.backend.connect_tcp(host, port, timeout, local_address, socket_options)
        return _TargetTLSStream(stream, self.hostname)


class _ClosingStream(httpx.AsyncByteStream):
    def __init__(self, stream, transport):
        self.stream, self.transport = stream, transport

    async def __aiter__(self):
        async for chunk in self.stream:
            yield chunk

    async def aclose(self):
        try:
            await self.stream.aclose()
        finally:
            await self.transport.aclose()


class ReferenceTransport(httpx.AsyncHTTPTransport):
    def __init__(self):
        super().__init__(trust_env=False, limits=httpx.Limits(max_keepalive_connections=0))

    async def handle_async_request(self, request):
        proxy_url = configured_proxy(request.url.scheme)
        if proxy_url is None:
            return await super().handle_async_request(request)
        proxy = httpx.Proxy(proxy_url)
        transport = httpx.AsyncHTTPTransport(trust_env=False)
        # Reuse HTTPX's request/response and exception adapter; no global patch.
        transport._pool = httpcore.AsyncHTTPProxy(
            proxy_url=str(proxy.url), proxy_auth=proxy.raw_auth,
            ssl_context=httpx.create_ssl_context(trust_env=False),
            network_backend=_TargetTLSBackend(request.extensions["sni_hostname"]),
            max_keepalive_connections=0,
        )
        try:
            response = await transport.handle_async_request(request)
        except BaseException:
            await transport.aclose()
            raise
        response.stream = _ClosingStream(response.stream, transport)
        return response
