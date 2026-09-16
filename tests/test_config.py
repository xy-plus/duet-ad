from __future__ import annotations

import pytest

from app.config import Settings, get_settings


def _settings(*, picsart_proxy: str | None) -> Settings:
    return Settings(
        access_token="test",
        public_base_url="https://settings.test",
        picsart_proxy=picsart_proxy,
    )


def test_settings_defaults_picsart_proxy_to_direct() -> None:
    assert _settings(picsart_proxy=None).picsart_proxy is None


@pytest.mark.parametrize(
    "proxy",
    ["http://127.0.0.1:7897", "https://proxy.example:443", "socks5://127.0.0.1:1080"],
)
def test_settings_accepts_picsart_proxy_schemes(proxy: str) -> None:
    assert _settings(picsart_proxy=proxy).picsart_proxy == proxy


@pytest.mark.parametrize("proxy", ["", "  ", "ftp://proxy.example", "not a url"])
def test_settings_rejects_invalid_picsart_proxy(proxy: str) -> None:
    with pytest.raises(ValueError, match="PICSART_PROXY"):
        _settings(picsart_proxy=proxy)


def test_get_settings_reads_picsart_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://settings.test")
    monkeypatch.setenv("PICSART_PROXY", "  http://127.0.0.1:7897  ")

    assert get_settings().picsart_proxy == "http://127.0.0.1:7897"
