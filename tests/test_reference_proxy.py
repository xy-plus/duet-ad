import asyncio
import ssl
import subprocess

import httpx
import pytest

from app.reference_proxy import ReferenceTransport, configured_proxy


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_proxy_selection_and_unrelated_environment(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "http://unrelated.example:9999")
    assert configured_proxy("https") is None
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    assert configured_proxy("http") == "http://127.0.0.1:7897"
    assert configured_proxy("https") is None
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7898")
    assert configured_proxy("https") == "http://127.0.0.1:7898"
    monkeypatch.setenv("https_proxy", "")
    assert configured_proxy("https") is None


@pytest.mark.parametrize("proxy", ["https://127.0.0.1:7897", "socks5://127.0.0.1:7897"])
def test_unsupported_proxy_scheme_is_explicit(monkeypatch, proxy):
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    with pytest.raises(ValueError, match="HTTP proxy"):
        configured_proxy("https")


def test_direct_pool_has_no_keepalive():
    transport = ReferenceTransport()
    assert transport._pool._max_keepalive_connections == 0
    asyncio.run(transport.aclose())


@pytest.fixture
def certificate(tmp_path):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["/usr/bin/openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1", "-subj", "/CN=media.example",
                    "-addext", "subjectAltName=DNS:media.example"], check=True, capture_output=True)
    return cert, key


@pytest.mark.parametrize("trusted,hostname,redirect_private,target_ip", [
    (True, "media.example", False, "93.184.216.34"),
    (False, "media.example", False, "93.184.216.34"),
    (True, "wrong.example", False, "93.184.216.34"),
    (True, "media.example", True, "93.184.216.34"),
    (True, "media.example", False, "2606:4700:4700::1111"),
])
def test_real_connect_uses_ip_but_tls_verifies_domain(monkeypatch, certificate, trusted, hostname, redirect_private, target_ip, tmp_path):
    cert, key = certificate
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert, key)
    observed = []
    server_context.set_servername_callback(lambda _sock, name, _ctx: observed.append(("sni", name)))
    original_context = ssl.create_default_context
    if trusted:
        monkeypatch.setattr("app.reference_proxy.httpx.create_ssl_context",
                            lambda **_kwargs: original_context(cafile=str(cert)))

    async def scenario():
        async def connection(reader, writer):
            try:
                connect = await reader.readuntil(b"\r\n\r\n")
                observed.append(("connect", connect.split(b"\r\n", 1)[0]))
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                await writer.start_tls(server_context)
                request = await reader.readuntil(b"\r\n\r\n")
                observed.append(("request", request))
                if redirect_private:
                    writer.write(b"HTTP/1.1 302 Found\r\nLocation: https://127.0.0.1/secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                else:
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nvideo")
                await writer.drain()
            except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(connection, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
        try:
            if redirect_private:
                from app.safe_download import ReferenceDownloadError, download_reference_url
                async def resolver(_hostname, _port):
                    return ["127.0.0.1", target_ip]
                with pytest.raises(ReferenceDownloadError) as caught:
                    await download_reference_url("https://media.example/video", tmp_path / "video",
                                                 resolver=resolver, max_bytes=100, timeout_s=3)
                assert caught.value.code == "reference_url_forbidden"
            else:
              async with httpx.AsyncClient(transport=ReferenceTransport(), trust_env=False, timeout=3) as client:
                authority = f"[{target_ip}]" if ":" in target_ip else target_ip
                request = client.get(f"https://{authority}/video", headers={"Host": "media.example"},
                                     extensions={"sni_hostname": hostname})
                if trusted and hostname == "media.example":
                    response = await request
                    assert response.status_code == 200 and response.content == b"video"
                else:
                    with pytest.raises(httpx.ConnectError):
                        await request
        finally:
            server.close()
            await server.wait_closed()
        # httpcore currently emits unbracketed IPv6 CONNECT authority. This
        # accepting local proxy records that existing library limitation.
        assert ("connect", f"CONNECT {target_ip}:443 HTTP/1.1".encode()) in observed
        assert len([item for item in observed if item[0] == "connect"]) == 1
        assert ("sni", hostname) in observed
        if trusted and hostname == "media.example":
            assert b"Host: media.example\r\n" in dict(observed)["request"]

    asyncio.run(scenario())


def test_private_target_never_reaches_configured_proxy(monkeypatch, tmp_path):
    from app.safe_download import ReferenceDownloadError, download_reference_url
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("private target reached transport")

    async def resolver(_hostname, _port):
        return ["127.0.0.1", "169.254.169.254"]

    monkeypatch.setattr(ReferenceTransport, "handle_async_request", forbidden)
    with pytest.raises(ReferenceDownloadError) as caught:
        asyncio.run(download_reference_url("https://private.example/video", tmp_path / "video",
                    max_bytes=100, timeout_s=1, resolver=resolver))
    assert caught.value.code == "reference_url_forbidden"


def test_proxy_tls_context_ignores_certificate_environment(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent-diagnostic-certificate-file")
    context = httpx.create_ssl_context(trust_env=False)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
