import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    access_token: str
    host: str = "0.0.0.0"
    port: int = 3211
    max_upload_mb: int = 500
    max_duration_s: int = 300
    codex_timeout_s: int = 600
    codex_concurrency: int = 1
    enable_seedance_submit: bool = False
    data_dir: Path = Path("data")


def get_settings() -> Settings:
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ACCESS_TOKEN environment variable is required")
    return Settings(
        access_token=token,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "3211")),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        max_duration_s=int(os.environ.get("MAX_DURATION_S", "300")),
        codex_timeout_s=int(os.environ.get("CODEX_TIMEOUT_S", "600")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "1")),
        enable_seedance_submit=os.environ.get("ENABLE_SEEDANCE_SUBMIT", "").lower() in ("1", "true", "yes"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
    )
