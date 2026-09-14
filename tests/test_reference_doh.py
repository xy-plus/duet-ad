import asyncio

import httpx
import pytest

from app import safe_download


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_doh_uses_fixed_bootstrap_and_a_protocol():
    def handler(request):
        assert request.url.host == "1.1.1.1"
        assert request.url.path == "/dns-query"
        assert dict(request.url.params) == {"name": "cdn.example", "type": "A"}
        assert request.headers["accept"] == "application/dns-json"
        return httpx.Response(200, json={"Answer": [{"type": 5, "data": "alias.example"}, {"type": 1, "data": "93.184.216.34"}]})
    result = asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert result == ["93.184.216.34"]


def test_endpoint_candidates_in_priority_order():
    assert safe_download.DOH_ENDPOINTS == (
        "https://1.1.1.1/dns-query",
        "https://8.8.8.8/resolve",
        "https://1.0.0.1/dns-query",
        "https://cloudflare-dns.com/dns-query",
        "https://dns.google/resolve",
    )


def test_connect_error_falls_through_to_next_candidate():
    seen = []
    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "1.1.1.1":
            raise httpx.ConnectError("SSL_ERROR_SYSCALL", request=request)
        return httpx.Response(200, json={"Answer": [{"type": 1, "data": "93.184.216.34"}]})
    result = asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert result == ["93.184.216.34"]
    assert seen == ["1.1.1.1", "8.8.8.8"]


def test_non_200_response_falls_through_to_next_candidate():
    seen = []
    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "1.1.1.1":
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"Answer": [{"type": 1, "data": "93.184.216.34"}]})
    result = asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert result == ["93.184.216.34"]
    assert seen == ["1.1.1.1", "8.8.8.8"]


def test_empty_answers_fall_through_to_next_candidate():
    seen = []
    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "1.1.1.1":
            return httpx.Response(200, json={"Answer": [{"type": 5, "data": "alias.example"}]})
        return httpx.Response(200, json={"Answer": [{"type": 1, "data": "93.184.216.34"}]})
    result = asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert result == ["93.184.216.34"]
    assert seen == ["1.1.1.1", "8.8.8.8"]


def test_all_endpoints_failing_raises_after_trying_every_candidate():
    seen = []
    def handler(request):
        seen.append(request.url.host)
        raise httpx.ConnectError("SSL_ERROR_SYSCALL", request=request)
    with pytest.raises(ValueError, match="dns.google"):
        asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert seen == ["1.1.1.1", "8.8.8.8", "1.0.0.1", "cloudflare-dns.com", "dns.google"]


def test_dns_google_candidate_uses_resolve_path():
    def handler(request):
        if request.url.host != "dns.google":
            raise httpx.ConnectError("SSL_ERROR_SYSCALL", request=request)
        assert request.url.path == "/resolve"
        assert dict(request.url.params) == {"name": "cdn.example", "type": "A"}
        assert request.headers["accept"] == "application/dns-json"
        return httpx.Response(200, json={"Answer": [{"type": 1, "data": "93.184.216.34"}]})
    result = asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897", transport=httpx.MockTransport(handler)))
    assert result == ["93.184.216.34"]


@pytest.mark.parametrize("status,payload", [(503, {}), (302, {}), (200, {}), (200, []), (200, {"Answer": [{"type": 28, "data": "::1"}]})])
def test_invalid_doh_responses_fail(status, payload):
    with pytest.raises((ValueError, httpx.HTTPError)):
        asyncio.run(safe_download._doh_addresses("cdn.example", "http://127.0.0.1:7897",
                    transport=httpx.MockTransport(lambda _: httpx.Response(status, json=payload))))


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_default_resolver_uses_selected_proxy_and_public_filter(monkeypatch, scheme):
    monkeypatch.setenv(scheme.upper() + "_PROXY", "http://127.0.0.1:7897")
    seen = []
    async def doh(hostname, proxy):
        seen.append((hostname, proxy))
        return ["127.0.0.1", "93.184.216.34"]
    monkeypatch.setattr(safe_download, "_doh_addresses", doh)
    result = asyncio.run(safe_download._approved_target(scheme + "://cdn.example/video", safe_download.resolve_addresses))
    assert "93.184.216.34" in result.pinned_url
    assert result.sni_hostname == "cdn.example"
    assert seen == [("cdn.example", "http://127.0.0.1:7897")]


def test_injected_resolver_is_not_overridden(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    async def forbidden(*args):
        raise AssertionError("DoH replaced an injected resolver")
    monkeypatch.setattr(safe_download, "_doh_addresses", forbidden)
    result = asyncio.run(safe_download._approved_target("https://cdn.example/video", lambda *_: ["8.8.8.8"]))
    assert "8.8.8.8" in result.pinned_url


def test_no_proxy_uses_local_dns(monkeypatch):
    async def scenario():
        async def local(*args, **kwargs):
            return [(2, 1, 6, "", ("93.184.216.34", 443))]
        monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", local)
        return await safe_download._approved_target("https://cdn.example/video", safe_download.resolve_addresses)
    assert "93.184.216.34" in asyncio.run(scenario()).pinned_url


@pytest.mark.parametrize("answers", [None, ["127.0.0.1", "169.254.169.254"]])
def test_doh_failure_or_private_answers_never_fall_back(monkeypatch, answers):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    async def doh(*args):
        if answers is None:
            raise ValueError("failed")
        return answers
    monkeypatch.setattr(safe_download, "_doh_addresses", doh)
    with pytest.raises(safe_download.ReferenceDownloadError):
        asyncio.run(safe_download._approved_target("https://cdn.example/video", safe_download.resolve_addresses))


def test_every_redirect_uses_doh_then_pins(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    looked_up, requested = [], []
    async def doh(hostname, proxy):
        looked_up.append(hostname)
        return ["93.184.216.34"] if hostname == "first.example" else ["8.8.8.8"]
    def handler(request):
        requested.append(request.url.host)
        if len(requested) == 1:
            return httpx.Response(302, headers={"Location": "https://second.example/video"})
        return httpx.Response(200, content=b"video")
    monkeypatch.setattr(safe_download, "_doh_addresses", doh)
    assert asyncio.run(safe_download.download_reference_url("https://first.example/video", tmp_path / "video",
                       max_bytes=100, timeout_s=2, transport=httpx.MockTransport(handler))) == 5
    assert looked_up == ["first.example", "second.example"]
    assert requested == ["93.184.216.34", "8.8.8.8"]
