import math
import os
from dataclasses import dataclass, field
from pathlib import Path


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
    enable_seedream_edit: bool = False
    seedream_model: str = "doubao-seedream-5-0-pro-260628"
    # Seedream 后处理逐帧并行提交的进程级并发上限（asyncio 信号量，单进程内跨会话全局；≤0 钳制为 1）
    seedream_concurrency: int = 10
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
        enable_seedream_edit=os.environ.get("ENABLE_SEEDREAM_EDIT", "").lower() in ("1", "true", "yes"),
        seedream_model=os.environ.get("SEEDREAM_MODEL", "doubao-seedream-5-0-pro-260628"),
        seedream_concurrency=max(1, int(os.environ.get("SEEDREAM_CONCURRENCY", "10"))),
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
