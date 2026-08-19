import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    access_token: str
    max_upload_mb: int = 500
    max_duration_s: int = 15
    enable_h3_submit: bool = False
    minimax_api_key: str = field(default="", repr=False)
    autodl_art_token: str = field(default="", repr=False)
    h3_request_timeout_s: float = 30.0
    h3_upload_timeout_s: float = 60.0
    h3_ir_poll_timeout_s: float = 900.0
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
    # queued 状态会话数上限（不计 processing/done/failed），超过即 429
    max_queued: int = 100
    # TikTok 解析/下载走的 HTTP 代理（空 = 直连）；URL 下载大小上限复用 max_upload_mb
    tiktok_proxy: str = ""
    download_timeout_s: int = 120
    # 直建 Settings（测试）默认不跑流水线；get_settings（生产）默认开
    enable_pipeline: bool = False


def get_settings() -> Settings:
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ACCESS_TOKEN environment variable is required")
    return Settings(
        access_token=token,
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        max_duration_s=int(os.environ.get("MAX_DURATION_S", "15")),
        enable_h3_submit=os.environ.get("ENABLE_H3_SUBMIT", "").lower() in ("1", "true", "yes"),
        minimax_api_key=os.environ.get("MINIMAX_API_KEY", "").strip(),
        autodl_art_token=os.environ.get("AUTODL_ART_TOKEN", "").strip(),
        h3_request_timeout_s=float(os.environ.get("H3_REQUEST_TIMEOUT_S", "30")),
        h3_upload_timeout_s=float(os.environ.get("H3_UPLOAD_TIMEOUT_S", "60")),
        h3_ir_poll_timeout_s=float(os.environ.get("H3_IR_POLL_TIMEOUT_S", "900")),
        h3_poll_timeout_s=float(os.environ.get("H3_POLL_TIMEOUT_S", "1500")),
        h3_download_timeout_s=float(os.environ.get("H3_DOWNLOAD_TIMEOUT_S", "180")),
        h3_poll_interval_s=float(os.environ.get("H3_POLL_INTERVAL_S", "3")),
        enable_seedream_edit=os.environ.get("ENABLE_SEEDREAM_EDIT", "").lower() in ("1", "true", "yes"),
        seedream_model=os.environ.get("SEEDREAM_MODEL", "doubao-seedream-5-0-pro-260628"),
        seedream_concurrency=max(1, int(os.environ.get("SEEDREAM_CONCURRENCY", "10"))),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        codex_timeout_s=int(os.environ.get("CODEX_TIMEOUT_S", "1800")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
        max_queued=int(os.environ.get("MAX_QUEUED", "100")),
        tiktok_proxy=os.environ.get("TIKTOK_PROXY", ""),
        download_timeout_s=int(os.environ.get("DOWNLOAD_TIMEOUT_S", "120")),
        enable_pipeline=os.environ.get("ENABLE_PIPELINE", "1").lower() in ("1", "true", "yes"),
    )
