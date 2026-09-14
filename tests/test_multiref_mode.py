"""multiref_mode 开关：显式设置只允许 composed|direct，缺省默认 direct（spec §1）。"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings

_BASE = {
    "access_token": "test-secret",
    "public_base_url": "https://multiref.test",
}


def test_multiref_mode_accepts_composed() -> None:
    settings = Settings(**_BASE, multiref_mode="composed")
    assert settings.multiref_mode == "composed"


def test_multiref_mode_accepts_direct() -> None:
    settings = Settings(**_BASE, multiref_mode="direct")
    assert settings.multiref_mode == "direct"


def test_multiref_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="MULTIREF_MODE"):
        Settings(**_BASE, multiref_mode="legacy")


def test_multiref_mode_defaults_to_direct() -> None:
    settings = Settings(**_BASE)
    assert settings.multiref_mode == "direct"


def test_get_settings_defaults_multiref_mode_to_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://multiref.test")
    monkeypatch.delenv("MULTIREF_MODE", raising=False)
    assert get_settings().multiref_mode == "direct"


def test_get_settings_reads_multiref_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://multiref.test")
    monkeypatch.setenv("MULTIREF_MODE", "direct")
    assert get_settings().multiref_mode == "direct"


def test_get_settings_rejects_invalid_multiref_mode_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://multiref.test")
    monkeypatch.setenv("MULTIREF_MODE", "bogus")
    with pytest.raises(ValueError, match="MULTIREF_MODE"):
        get_settings()
