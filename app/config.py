import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    access_token: str
    max_upload_mb: int = 500
    max_duration_s: int = 300
    enable_seedance_submit: bool = False
    data_dir: Path = Path("data")
    codex_timeout_s: int = 600
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
        max_duration_s=int(os.environ.get("MAX_DURATION_S", "300")),
        enable_seedance_submit=os.environ.get("ENABLE_SEEDANCE_SUBMIT", "").lower() in ("1", "true", "yes"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        codex_timeout_s=int(os.environ.get("CODEX_TIMEOUT_S", "600")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
        max_queued=int(os.environ.get("MAX_QUEUED", "100")),
        tiktok_proxy=os.environ.get("TIKTOK_PROXY", ""),
        download_timeout_s=int(os.environ.get("DOWNLOAD_TIMEOUT_S", "120")),
        enable_pipeline=os.environ.get("ENABLE_PIPELINE", "1").lower() in ("1", "true", "yes"),
    )
