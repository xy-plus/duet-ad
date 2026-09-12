"""multiref_mode 开关：只允许 composed|direct，显式配置、无缺省（spec §1）。"""

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


def test_multiref_mode_has_no_default() -> None:
    with pytest.raises(TypeError):
        Settings(**_BASE)


def test_get_settings_requires_multiref_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://multiref.test")
    monkeypatch.delenv("MULTIREF_MODE", raising=False)
    with pytest.raises(RuntimeError, match="MULTIREF_MODE"):
        get_settings()


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
