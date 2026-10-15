from __future__ import annotations

import pytest

from app.config import Settings, get_settings


def _settings(
    *,
    public_base_url: str = "https://settings.test",
    picsart_proxy: str | None = None,
    object_mask_block_opacity: float = 1.0,
    object_mask_block_blur: int = 31,
    object_mask_enabled: bool = True,
    object_mask_person_block_opacity: float = 1.0,
    ark_audio_api_key: str | None = None,
    ark_audio_model: str | None = None,
    ark_audio_api_base: str | None = None,
    vocal_separation_model_path: object = None,
    vocal_separation_cli: object = None,
) -> Settings:
    return Settings(
        access_token="test",
        public_base_url=public_base_url,
        picsart_proxy=picsart_proxy,
        object_mask_block_opacity=object_mask_block_opacity,
        object_mask_block_blur=object_mask_block_blur,
        object_mask_enabled=object_mask_enabled,
        object_mask_person_block_opacity=object_mask_person_block_opacity,
        ark_audio_api_key=ark_audio_api_key,
        ark_audio_model=ark_audio_model,
        ark_audio_api_base=ark_audio_api_base,
        vocal_separation_model_path=vocal_separation_model_path,
        vocal_separation_cli=vocal_separation_cli,
    )


def test_settings_defaults_picsart_proxy_to_direct() -> None:
    assert _settings(picsart_proxy=None).picsart_proxy is None


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:3252",
        "https://127.0.0.1:3252",
        "http://localhost:3252",
        "http://[::1]:3252",
    ],
)
def test_settings_accepts_loopback_public_base_url(base_url: str) -> None:
    assert _settings(public_base_url=base_url).public_base_url == base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "http://settings.test",
        "http://192.168.1.10:3252",
        "http://10.0.0.8",
        "http://127.0.0.1.evil.test",
        "http://app.localhost:3252",
        "http://[2001:db8::1]",
    ],
)
def test_settings_rejects_non_loopback_http_public_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="PUBLIC_BASE_URL"):
        _settings(public_base_url=base_url)


def test_settings_keeps_public_https_public_base_url() -> None:
    assert (
        _settings(public_base_url="https://duet.example.com").public_base_url
        == "https://duet.example.com"
    )


@pytest.mark.parametrize("base_url", ["https://localhost", "https://127.0.0.2"])
def test_settings_accepts_loopback_https_public_base_url(base_url: str) -> None:
    # loopback 放行后 https loopback 同样合法；公网 https 规则不变。
    assert _settings(public_base_url=base_url).public_base_url == base_url


def test_settings_defaults_to_deepseek_flash() -> None:
    assert _settings().deepseek_model == "deepseek-flash"


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


def test_settings_defaults_object_mask_block_opacity_to_solid() -> None:
    assert _settings().object_mask_block_opacity == 1.0


def test_settings_defaults_object_mask_block_blur() -> None:
    assert _settings().object_mask_block_blur == 31


@pytest.mark.parametrize("blur", [1, 3, 31])
def test_settings_accepts_positive_odd_object_mask_block_blur(blur: int) -> None:
    assert _settings(object_mask_block_blur=blur).object_mask_block_blur == blur


@pytest.mark.parametrize("blur", [0, -1, 2, 32, True])
def test_settings_rejects_invalid_object_mask_block_blur(blur: int) -> None:
    with pytest.raises(ValueError, match="OBJECT_MASK_BLOCK_BLUR"):
        _settings(object_mask_block_blur=blur)


@pytest.mark.parametrize("opacity", [0.05, 0.9, 0.95, 1, 1.0])
def test_settings_accepts_object_mask_block_opacity(opacity: float) -> None:
    assert _settings(object_mask_block_opacity=opacity).object_mask_block_opacity == float(
        opacity
    )


@pytest.mark.parametrize("opacity", [0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_settings_rejects_invalid_object_mask_block_opacity(opacity: float) -> None:
    with pytest.raises(ValueError, match="OBJECT_MASK_BLOCK_OPACITY"):
        _settings(object_mask_block_opacity=opacity)


def test_get_settings_reads_object_mask_block_opacity_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("OBJECT_MASK_BLOCK_OPACITY", "0.95")

    assert get_settings().object_mask_block_opacity == pytest.approx(0.95)


def test_get_settings_reads_object_mask_block_blur_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("OBJECT_MASK_BLOCK_BLUR", "15")

    assert get_settings().object_mask_block_blur == 15


def test_settings_defaults_object_mask_enabled() -> None:
    assert _settings().object_mask_enabled is True


@pytest.mark.parametrize("enabled", [True, False])
def test_settings_accepts_object_mask_enabled(enabled: bool) -> None:
    assert _settings(object_mask_enabled=enabled).object_mask_enabled is enabled


@pytest.mark.parametrize("enabled", [0, 1, "true", "false", None])
def test_settings_rejects_non_bool_object_mask_enabled(enabled: object) -> None:
    with pytest.raises(ValueError, match="OBJECT_MASK_ENABLED"):
        _settings(object_mask_enabled=enabled)


def test_get_settings_defaults_object_mask_enabled_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")

    assert get_settings().object_mask_enabled is True


def test_get_settings_reads_object_mask_enabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("OBJECT_MASK_ENABLED", "false")

    assert get_settings().object_mask_enabled is False


def test_settings_defaults_person_block_opacity_to_solid() -> None:
    assert _settings().object_mask_person_block_opacity == 1.0


@pytest.mark.parametrize("opacity", [0.05, 0.99, 1.0])
def test_settings_accepts_person_block_opacity(opacity: float) -> None:
    assert _settings(
        object_mask_person_block_opacity=opacity
    ).object_mask_person_block_opacity == float(opacity)


@pytest.mark.parametrize("opacity", [0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_settings_rejects_invalid_person_block_opacity(opacity: float) -> None:
    with pytest.raises(ValueError, match="OBJECT_MASK_PERSON_BLOCK_OPACITY"):
        _settings(object_mask_person_block_opacity=opacity)


def test_get_settings_reads_person_block_opacity_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("OBJECT_MASK_PERSON_BLOCK_OPACITY", "0.98")

    assert get_settings().object_mask_person_block_opacity == pytest.approx(0.98)


def test_settings_defaults_ark_audio_to_unset() -> None:
    settings = _settings()
    assert settings.ark_audio_api_key is None
    assert settings.ark_audio_model is None
    assert settings.ark_audio_api_base is None


def test_settings_defaults_vocal_separation_model_path_to_unset() -> None:
    assert _settings().vocal_separation_model_path is None


def test_settings_defaults_vocal_separation_cli_to_unset() -> None:
    assert _settings().vocal_separation_cli is None


def test_settings_accepts_vocal_separation_cli(tmp_path) -> None:
    settings = _settings(vocal_separation_cli=tmp_path / "bin" / "audio-separator")
    assert settings.vocal_separation_cli == tmp_path / "bin" / "audio-separator"


@pytest.mark.parametrize("path", ["bin/audio-separator", 42])
def test_settings_rejects_non_path_vocal_separation_cli(path: object) -> None:
    with pytest.raises(ValueError, match="VOCAL_SEPARATION_CLI"):
        _settings(vocal_separation_cli=path)


def test_settings_accepts_ark_audio_configuration(tmp_path) -> None:
    settings = _settings(
        ark_audio_api_key="  ark-audio-secret  ",
        ark_audio_model="doubao-seed-2-0-lite-260428",
        ark_audio_api_base="https://ark.cn-beijing.volces.com/api/v3/",
        vocal_separation_model_path=tmp_path / "model.onnx",
    )
    assert settings.ark_audio_api_key == "ark-audio-secret"
    assert settings.ark_audio_model == "doubao-seed-2-0-lite-260428"
    assert settings.ark_audio_api_base == "https://ark.cn-beijing.volces.com/api/v3"
    assert settings.vocal_separation_model_path == tmp_path / "model.onnx"


@pytest.mark.parametrize("key", ["", "   ", "sk-密钥", "with space"])
def test_settings_rejects_invalid_ark_audio_api_key(key: str) -> None:
    with pytest.raises(ValueError, match="ARK_AUDIO_API_KEY"):
        _settings(ark_audio_api_key=key)


@pytest.mark.parametrize("model", ["", "  ", "has space", "bad/model"])
def test_settings_rejects_invalid_ark_audio_model(model: str) -> None:
    with pytest.raises(ValueError, match="ARK_AUDIO_MODEL"):
        _settings(ark_audio_model=model)


@pytest.mark.parametrize("base", ["", "  ", "http://ark.example", "not a url"])
def test_settings_rejects_invalid_ark_audio_api_base(base: str) -> None:
    with pytest.raises(ValueError, match="ARK_AUDIO_API_BASE"):
        _settings(ark_audio_api_base=base)


@pytest.mark.parametrize("path", ["models/model.onnx", 42])
def test_settings_rejects_non_path_vocal_separation_model_path(path: object) -> None:
    with pytest.raises(ValueError, match="VOCAL_SEPARATION_MODEL_PATH"):
        _settings(vocal_separation_model_path=path)


def test_get_settings_reads_ark_audio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ARK_AUDIO_API_KEY", "ark-audio-secret")
    monkeypatch.setenv("ARK_AUDIO_MODEL", "doubao-seed-2-0-lite-260428")
    monkeypatch.setenv("ARK_AUDIO_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
    monkeypatch.setenv(
        "VOCAL_SEPARATION_MODEL_PATH", "/home/xy/duet-ad1-h3-max-r2v/models/mdx.onnx"
    )
    monkeypatch.setenv(
        "VOCAL_SEPARATION_CLI",
        "/home/xy/duet-ad1-h3-max-r2v/.venv-separation/bin/audio-separator",
    )

    settings = get_settings()
    assert settings.ark_audio_api_key == "ark-audio-secret"
    assert settings.ark_audio_model == "doubao-seed-2-0-lite-260428"
    assert settings.ark_audio_api_base == "https://ark.cn-beijing.volces.com/api/v3"
    assert str(settings.vocal_separation_model_path).endswith("models/mdx.onnx")
    assert str(settings.vocal_separation_cli).endswith("bin/audio-separator")


def test_get_settings_defaults_ark_audio_env_to_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACCESS_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    for name in (
        "ARK_AUDIO_API_KEY",
        "ARK_AUDIO_MODEL",
        "ARK_AUDIO_API_BASE",
        "VOCAL_SEPARATION_MODEL_PATH",
        "VOCAL_SEPARATION_CLI",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()
    assert settings.ark_audio_api_key is None
    assert settings.ark_audio_model is None
    assert settings.ark_audio_api_base is None
    assert settings.vocal_separation_model_path is None
    assert settings.vocal_separation_cli is None
