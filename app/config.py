from __future__ import annotations

import ipaddress
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
FUSION_AGENTS = frozenset({"codex", "deepseek"})
SEEDREAM_PRO_MODEL = "doubao-seedream-5-0-pro-260628"
SEEDREAM_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class Settings:
    access_token: str
    data_dir: Path = ROOT / "data"
    public_base_url: str = ""
    picsart_api_key: str = ""
    picsart_api_base: str = "https://api.picsart.com"
    autodl_api_key: str = field(default="", repr=False)
    autodl_api_base: str = "https://autodl.art"
    seedream_api_key: str = field(default="", repr=False)
    seedream_api_base: str = SEEDREAM_API_BASE
    seedream_model: str = SEEDREAM_PRO_MODEL
    seedream_timeout_s: float = 300.0
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
    fusion_agent: str = "deepseek"
    codex_timeout_s: float = 1800.0
    codex_concurrency: int = 10
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
        seedream_key = (
            self.seedream_api_key.strip()
            if isinstance(self.seedream_api_key, str)
            else ""
        )
        if seedream_key and (
            not seedream_key.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in seedream_key)
        ):
            raise ValueError("ARK_API_KEY must be header-safe ASCII")
        object.__setattr__(self, "seedream_api_key", seedream_key)
        seedream_base = (
            self.seedream_api_base.strip().rstrip("/")
            if isinstance(self.seedream_api_base, str)
            else ""
        )
        if (
            not seedream_base.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in seedream_base)
        ):
            raise ValueError("SEEDREAM_API_BASE must be an HTTPS base URL")
        try:
            seedream_url = urlsplit(seedream_base)
            seedream_port = seedream_url.port
        except ValueError as exc:
            raise ValueError("SEEDREAM_API_BASE must be an HTTPS base URL") from exc
        if (
            seedream_url.scheme != "https"
            or not seedream_url.netloc
            or not seedream_url.hostname
            or seedream_url.username is not None
            or seedream_url.password is not None
            or seedream_url.query
            or seedream_url.fragment
            or seedream_port is not None and not 1 <= seedream_port <= 65535
        ):
            raise ValueError("SEEDREAM_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "seedream_api_base", seedream_base)
        seedream_model = (
            self.seedream_model.strip()
            if isinstance(self.seedream_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(seedream_model):
            raise ValueError("SEEDREAM_MODEL must be a valid model identifier")
        object.__setattr__(self, "seedream_model", seedream_model)
        if (
            isinstance(self.seedream_timeout_s, bool)
            or not isinstance(self.seedream_timeout_s, (int, float))
            or not math.isfinite(float(self.seedream_timeout_s))
            or self.seedream_timeout_s <= 0
        ):
            raise ValueError("SEEDREAM_TIMEOUT_S must be a positive finite number")
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
        if self.fusion_agent not in FUSION_AGENTS:
            raise ValueError("FUSION_AGENT must be codex or deepseek")
        if (
            isinstance(self.codex_timeout_s, bool)
            or self.codex_timeout_s <= 0
        ):
            raise ValueError("CODEX_TIMEOUT_S must be positive")
        if (
            isinstance(self.codex_concurrency, bool)
            or self.codex_concurrency < 1
        ):
            raise ValueError("CODEX_CONCURRENCY must be a positive integer")


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    access_token = os.environ.get("ACCESS_TOKEN", "").strip()
    if not access_token:
        raise RuntimeError("ACCESS_TOKEN must be set to a non-empty value")
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set to a non-empty value")
    return Settings(
        access_token=access_token,
        data_dir=Path(os.environ.get("DATA_DIR", str(ROOT / "data"))).resolve(),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", ""),
        picsart_api_key=os.environ.get("PICSART_API_KEY", ""),
        picsart_api_base=os.environ.get("PICSART_API_BASE", "https://api.picsart.com").rstrip("/"),
        autodl_api_key=(
            os.environ.get("AUTODL_ART_TOKEN")
            or os.environ.get("AUTODL_TOKEN", "")
        ).strip(),
        autodl_api_base=os.environ.get("AUTODL_API_BASE", "https://autodl.art").rstrip("/"),
        seedream_api_key=os.environ.get("ARK_API_KEY", "").strip(),
        seedream_api_base=os.environ.get(
            "SEEDREAM_API_BASE", SEEDREAM_API_BASE
        ).strip(),
        seedream_model=os.environ.get(
            "SEEDREAM_MODEL", SEEDREAM_PRO_MODEL
        ).strip(),
        seedream_timeout_s=float(os.environ.get("SEEDREAM_TIMEOUT_S", "300")),
        deepseek_api_key=deepseek_api_key,
        deepseek_api_base=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/"),
        deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash-vision-exp"),
        fusion_agent=os.environ.get("FUSION_AGENT", "deepseek").strip().lower(),
        codex_timeout_s=float(os.environ.get("CODEX_TIMEOUT_S", "1800")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
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
