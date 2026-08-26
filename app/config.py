import math
import os
from dataclasses import dataclass, field
from pathlib import Path

SEEDREAM_PRO_MODEL = "doubao-seedream-5-0-pro-260628"
SEEDREAM_MODELS = frozenset({
    SEEDREAM_PRO_MODEL,
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
})
SEEDREAM_EDIT_MODES = frozenset({"anchor_consistency", "independent_parallel"})
SEEDREAM_PROMPT_TEMPLATES = frozenset({"light", "balanced", "strong"})


@dataclass(frozen=True)
class Settings:
    access_token: str
    max_upload_mb: int = 500
    enable_h3_submit: bool = False
    autodl_art_token: str = field(default="", repr=False)
    h3_request_timeout_s: float = 30.0
    h3_poll_timeout_s: float = 1500.0
    h3_download_timeout_s: float = 180.0
    h3_poll_interval_s: float = 3.0
    enable_mediakit_erase: bool = False
    mediakit_api_key: str = field(default="", repr=False)
    # MediaKit 后处理逐帧并行提交的进程级并发上限（单进程内跨会话全局）
    mediakit_concurrency: int = 4
    mediakit_timeout_s: float = 180.0
    seedream_model: str = SEEDREAM_PRO_MODEL
    seedream_edit_mode: str = "anchor_consistency"
    seedream_prompt_template: str = "balanced"
    seedream_concurrency: int = 4
    seedream_timeout_s: float = 300.0
    data_dir: Path = Path("data")
    codex_timeout_s: int = 1800
    codex_concurrency: int = 10
    retry_count: int = 2
    retry_interval_s: float = 15.0
    asr_cli: Path | None = None
    asr_model: Path | None = None
    asr_timeout_s: int = 600
    asr_threads: int = 4
    # queued 状态会话数上限（不计 processing/done/failed），超过即 429
    max_queued: int = 100
    # TikTok 解析/下载走的 HTTP 代理（空 = 直连）；URL 下载大小上限复用 max_upload_mb
    tiktok_proxy: str = ""
    download_timeout_s: int = 120
    # 直建 Settings（测试）默认不跑流水线；get_settings（生产）默认开
    enable_pipeline: bool = False

    def __post_init__(self) -> None:
        if self.seedream_model not in SEEDREAM_MODELS:
            raise ValueError("seedream_model is not supported")
        if self.seedream_edit_mode not in SEEDREAM_EDIT_MODES:
            raise ValueError("seedream_edit_mode is not supported")
        if self.seedream_prompt_template not in SEEDREAM_PROMPT_TEMPLATES:
            raise ValueError("seedream_prompt_template is not supported")
        if (
            isinstance(self.seedream_concurrency, bool)
            or not isinstance(self.seedream_concurrency, int)
            or self.seedream_concurrency < 1
        ):
            raise ValueError("seedream_concurrency must be a positive integer")
        if (
            isinstance(self.seedream_timeout_s, bool)
            or not isinstance(self.seedream_timeout_s, (int, float))
            or not math.isfinite(float(self.seedream_timeout_s))
            or self.seedream_timeout_s <= 0
        ):
            raise ValueError("seedream_timeout_s must be a positive finite number")
        if (
            isinstance(self.mediakit_concurrency, bool)
            or not isinstance(self.mediakit_concurrency, int)
            or self.mediakit_concurrency < 1
        ):
            raise ValueError("mediakit_concurrency must be a positive integer")
        if (
            isinstance(self.mediakit_timeout_s, bool)
            or not isinstance(self.mediakit_timeout_s, (int, float))
            or not math.isfinite(float(self.mediakit_timeout_s))
            or self.mediakit_timeout_s <= 0
        ):
            raise ValueError("mediakit_timeout_s must be a positive finite number")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or self.retry_count < 0
        ):
            raise ValueError("retry_count must be a non-negative integer")
        if (
            isinstance(self.retry_interval_s, bool)
            or not isinstance(self.retry_interval_s, (int, float))
            or not math.isfinite(float(self.retry_interval_s))
            or self.retry_interval_s < 0
        ):
            raise ValueError("retry_interval_s must be a non-negative finite number")


def get_settings() -> Settings:
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ACCESS_TOKEN environment variable is required")
    return Settings(
        access_token=token,
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        enable_h3_submit=os.environ.get("ENABLE_H3_SUBMIT", "").lower() in ("1", "true", "yes"),
        autodl_art_token=os.environ.get("AUTODL_ART_TOKEN", "").strip(),
        h3_request_timeout_s=float(os.environ.get("H3_REQUEST_TIMEOUT_S", "30")),
        h3_poll_timeout_s=float(os.environ.get("H3_POLL_TIMEOUT_S", "1500")),
        h3_download_timeout_s=float(os.environ.get("H3_DOWNLOAD_TIMEOUT_S", "180")),
        h3_poll_interval_s=float(os.environ.get("H3_POLL_INTERVAL_S", "3")),
        enable_mediakit_erase=os.environ.get("ENABLE_MEDIAKIT_ERASE", "").lower() in ("1", "true", "yes"),
        mediakit_api_key=os.environ.get("VOLC_MEDIAKIT_API_KEY", "").strip(),
        mediakit_concurrency=max(1, int(os.environ.get("MEDIAKIT_CONCURRENCY", "4"))),
        mediakit_timeout_s=float(os.environ.get("MEDIAKIT_TIMEOUT_S", "180")),
        seedream_model=os.environ.get(
            "SEEDREAM_MODEL", SEEDREAM_PRO_MODEL
        ).strip(),
        seedream_edit_mode=os.environ.get(
            "SEEDREAM_EDIT_MODE", "anchor_consistency"
        ).strip(),
        seedream_prompt_template=os.environ.get(
            "SEEDREAM_PROMPT_TEMPLATE", "balanced"
        ).strip(),
        seedream_concurrency=int(os.environ.get("SEEDREAM_CONCURRENCY", "4")),
        seedream_timeout_s=float(os.environ.get("SEEDREAM_TIMEOUT_S", "300")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        codex_timeout_s=int(os.environ.get("CODEX_TIMEOUT_S", "1800")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
        retry_count=int(os.environ.get("AUTO_RETRY_COUNT", "2")),
        retry_interval_s=float(os.environ.get("AUTO_RETRY_INTERVAL_S", "15")),
        asr_cli=Path(os.environ.get(
            "ASR_CLI",
            "/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli",
        )),
        asr_model=Path(os.environ.get(
            "ASR_MODEL", "/home/xy/.local/share/duet-asr/ggml-small.bin"
        )),
        asr_timeout_s=int(os.environ.get("ASR_TIMEOUT_S", "600")),
        asr_threads=max(1, int(os.environ.get("ASR_THREADS", "4"))),
        max_queued=int(os.environ.get("MAX_QUEUED", "100")),
        tiktok_proxy=os.environ.get("TIKTOK_PROXY", ""),
        download_timeout_s=int(os.environ.get("DOWNLOAD_TIMEOUT_S", "120")),
        enable_pipeline=os.environ.get("ENABLE_PIPELINE", "1").lower() in ("1", "true", "yes"),
    )
