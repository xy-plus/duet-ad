from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    access_token: str
    data_dir: Path = ROOT / "data"
    public_base_url: str = ""
    picsart_api_key: str = ""
    picsart_api_base: str = "https://api.picsart.com"
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
    mediakit_api_key: str = ""
    mediakit_api_base: str = "https://mediakit.cn-beijing.volces.com/api/v1"
    enable_mediakit: bool = True
    mediakit_concurrency: int = 1
    asr_cli: Path = Path("/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")
    asr_model: Path = Path("/home/xy/.local/share/duet-asr/ggml-small.bin")
    yamnet_model: Path = ROOT / "models" / "yamnet.tflite"
    request_timeout_s: float = 180.0
    provider_poll_interval_s: float = 3.0
    provider_poll_timeout_s: float = 1800.0
    max_upload_mb: int = 500
    retry_delay_s: float = 1.0
    store_lock_enabled: bool = True

    def __post_init__(self) -> None:
        token = self.access_token.strip() if isinstance(self.access_token, str) else ""
        if not token:
            raise ValueError("ACCESS_TOKEN must be a non-empty value")
        if token != self.access_token:
            object.__setattr__(self, "access_token", token)
        base = self.public_base_url.strip() if isinstance(self.public_base_url, str) else ""
        try:
            parsed = urlsplit(base)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("PUBLIC_BASE_URL must be a public HTTPS origin") from exc
        hostname = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or port is not None and not 1 <= port <= 65535
            or hostname == "localhost"
            or hostname.endswith(".localhost")
        ):
            raise ValueError("PUBLIC_BASE_URL must be a public HTTPS origin")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError("PUBLIC_BASE_URL must be a public HTTPS origin")
        object.__setattr__(self, "public_base_url", base.rstrip("/"))
        if isinstance(self.max_upload_mb, bool) or self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be a positive integer")
        if (
            isinstance(self.mediakit_concurrency, bool)
            or self.mediakit_concurrency < 1
        ):
            raise ValueError("MEDIAKIT_CONCURRENCY must be a positive integer")


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    access_token = os.environ.get("ACCESS_TOKEN", "").strip()
    if not access_token:
        raise RuntimeError("ACCESS_TOKEN must be set to a non-empty value")
    return Settings(
        access_token=access_token,
        data_dir=Path(os.environ.get("DATA_DIR", str(ROOT / "data"))).resolve(),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", ""),
        picsart_api_key=os.environ.get("PICSART_API_KEY", ""),
        picsart_api_base=os.environ.get("PICSART_API_BASE", "https://api.picsart.com").rstrip("/"),
        # duet-ad1 stores its DeepSeek credential as ANTHROPIC_AUTH_TOKEN in
        # the protected credential file.  Accept the canonical name first,
        # then the source deployment's name so the independent app can reuse
        # that file without copying the secret into a second location.
        deepseek_api_key=(
            os.environ.get("DEEPSEEK_API_KEY", "").strip()
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
        ),
        deepseek_api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp"),
        mediakit_api_key=os.environ.get("VOLC_MEDIAKIT_API_KEY", ""),
        mediakit_api_base=os.environ.get("MEDIAKIT_API_BASE", "https://mediakit.cn-beijing.volces.com/api/v1").rstrip("/"),
        enable_mediakit=_bool("ENABLE_MEDIAKIT", True),
        mediakit_concurrency=int(os.environ.get("MEDIAKIT_CONCURRENCY", "1")),
        asr_cli=Path(os.environ.get("ASR_CLI", "/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")),
        asr_model=Path(os.environ.get("ASR_MODEL", "/home/xy/.local/share/duet-asr/ggml-small.bin")),
        yamnet_model=Path(os.environ.get("YAMNET_MODEL_PATH", str(ROOT / "models" / "yamnet.tflite"))),
        request_timeout_s=float(os.environ.get("REQUEST_TIMEOUT_S", "180")),
        provider_poll_interval_s=float(os.environ.get("PROVIDER_POLL_INTERVAL_S", "3")),
        provider_poll_timeout_s=float(os.environ.get("PROVIDER_POLL_TIMEOUT_S", "1800")),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        retry_delay_s=float(os.environ.get("RETRY_DELAY_S", "1")),
        # Provider asset grants live in process memory.  Production settings
        # therefore cannot disable the single-worker store lock; tests may do
        # so only by constructing Settings explicitly.
        store_lock_enabled=True,
    )
