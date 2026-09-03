import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.asr import ASRProcessBudget, process_budget

SEEDREAM_PRO_MODEL = "doubao-seedream-5-0-pro-260628"
SEEDREAM_MODELS = frozenset({
    SEEDREAM_PRO_MODEL,
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
})
SEEDREAM_EDIT_MODES = frozenset({"anchor_consistency", "independent_parallel"})


@dataclass(frozen=True)
class Settings:
    access_token: str
    max_upload_mb: int = 500
    enable_h3_submit: bool = False
    autodl_art_token: str = field(default="", repr=False)
    minimax_api_key: str = field(default="", repr=False)
    h3_request_timeout_s: float = 30.0
    h3_poll_timeout_s: float = 1500.0
    h3_download_timeout_s: float = 180.0
    h3_poll_interval_s: float = 3.0
    h3_gateway_storage_root: Path | None = None
    h3_controlled_storage_retry_attempt_sha256: str = ""
    h3_controlled_storage_retry_evidence_sha256: str = ""
    enable_mediakit_erase: bool = False
    mediakit_api_key: str = field(default="", repr=False)
    # MediaKit 后处理逐帧并行提交的进程级并发上限（单进程内跨会话全局）
    mediakit_concurrency: int = 4
    # 直建 Settings 默认关闭节流，生产 get_settings 默认启用一秒提交间隔。
    mediakit_submit_interval_s: float = 0.0
    mediakit_timeout_s: float = 180.0
    seedream_model: str = SEEDREAM_PRO_MODEL
    seedream_edit_mode: str = "independent_parallel"
    seedream_concurrency: int = 4
    seedream_timeout_s: float = 300.0
    data_dir: Path = Path("data")
    codex_timeout_s: int = 1800
    codex_concurrency: int = 10
    deepseek_credential_file: Path = Path("/home/xy/.config/claude/deepseek.env")
    retry_count: int = 2
    retry_interval_s: float = 15.0
    asr_cli: Path | None = None
    asr_model: Path | None = None
    asr_timeout_s: int = 600
    asr_threads: int = 4
    asr_process_budget: ASRProcessBudget = field(
        init=False, repr=False, compare=False
    )
    # queued 状态会话数上限（不计 processing/done/failed），超过即 429
    max_queued: int = 100
    # TikTok 解析/下载走的 HTTP 代理（空 = 直连）；URL 下载大小上限复用 max_upload_mb
    tiktok_proxy: str = ""
    download_timeout_s: int = 120
    # 直建 Settings（测试）默认不跑流水线；get_settings（生产）默认开
    enable_pipeline: bool = False
    # 极简创建是独立的发布合同；默认关闭，待完整后端链就绪后一次启用。
    enable_minimal_creation: bool = False
    # 第三方 API 与内部 UI 使用完全独立的鉴权域。默认关闭，避免升级后意外暴露。
    public_api_enabled: bool = False
    public_api_clients_file: Path = Path(
        "/home/xy/.config/duet-ad1/public-api-clients.json"
    )

    def __post_init__(self) -> None:
        credential_file = Path(self.deepseek_credential_file)
        if not credential_file.is_absolute():
            raise ValueError("deepseek_credential_file must be absolute")
        object.__setattr__(self, "deepseek_credential_file", credential_file)
        clients_file = Path(self.public_api_clients_file)
        if not clients_file.is_absolute():
            raise ValueError("public_api_clients_file must be absolute")
        object.__setattr__(self, "public_api_clients_file", clients_file)
        if self.public_api_enabled and not Path(self.data_dir).is_absolute():
            raise ValueError("data_dir must be absolute when public API is enabled")
        if self.h3_gateway_storage_root is not None:
            root = Path(self.h3_gateway_storage_root)
            if not root.is_absolute():
                raise ValueError("h3_gateway_storage_root must be absolute")
            object.__setattr__(self, "h3_gateway_storage_root", root)
        for value in (
            self.h3_controlled_storage_retry_attempt_sha256,
            self.h3_controlled_storage_retry_evidence_sha256,
        ):
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("controlled storage retry hashes must be sha256")
        if self.seedream_model not in SEEDREAM_MODELS:
            raise ValueError("seedream_model is not supported")
        if self.seedream_edit_mode not in SEEDREAM_EDIT_MODES:
            raise ValueError("seedream_edit_mode is not supported")
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
            isinstance(self.mediakit_submit_interval_s, bool)
            or not isinstance(self.mediakit_submit_interval_s, (int, float))
            or not math.isfinite(float(self.mediakit_submit_interval_s))
            or self.mediakit_submit_interval_s < 0
        ):
            raise ValueError(
                "mediakit_submit_interval_s must be a non-negative finite number"
            )
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
        if (
            isinstance(self.asr_threads, bool)
            or not isinstance(self.asr_threads, int)
            or self.asr_threads < 1
        ):
            raise ValueError("asr_threads must be a positive integer")
        object.__setattr__(
            self, "asr_process_budget", process_budget(self.asr_threads)
        )

    def minimal_creation_settings_ready(self) -> bool:
        """Return whether configured, non-secret v1 dependencies are usable."""
        return bool(
            self.enable_minimal_creation
            and self.enable_pipeline
            and self.enable_h3_submit
            and self.autodl_art_token.strip()
            and self.minimax_api_key.strip()
            and self.enable_mediakit_erase
            and self.mediakit_api_key.strip()
            and self.asr_cli is not None
            and self.asr_cli.is_file()
            and os.access(self.asr_cli, os.X_OK)
            and self.asr_model is not None
            and self.asr_model.is_file()
        )


def get_settings() -> Settings:
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        raise RuntimeError("ACCESS_TOKEN environment variable is required")
    return Settings(
        access_token=token,
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        enable_h3_submit=os.environ.get("ENABLE_H3_SUBMIT", "").lower() in ("1", "true", "yes"),
        autodl_art_token=os.environ.get("AUTODL_ART_TOKEN", "").strip(),
        minimax_api_key=os.environ.get("MINIMAX_API_KEY", "").strip(),
        h3_request_timeout_s=float(os.environ.get("H3_REQUEST_TIMEOUT_S", "30")),
        h3_poll_timeout_s=float(os.environ.get("H3_POLL_TIMEOUT_S", "1500")),
        h3_download_timeout_s=float(os.environ.get("H3_DOWNLOAD_TIMEOUT_S", "180")),
        h3_poll_interval_s=float(os.environ.get("H3_POLL_INTERVAL_S", "3")),
        h3_gateway_storage_root=Path(os.environ.get(
            "H3_GATEWAY_STORAGE_ROOT", "/data/duet/storage/duet-ad1-h3-inputs"
        )),
        h3_controlled_storage_retry_attempt_sha256=os.environ.get(
            "H3_CONTROLLED_STORAGE_RETRY_ATTEMPT_SHA256", ""
        ).strip(),
        h3_controlled_storage_retry_evidence_sha256=os.environ.get(
            "H3_CONTROLLED_STORAGE_RETRY_EVIDENCE_SHA256", ""
        ).strip(),
        enable_mediakit_erase=os.environ.get("ENABLE_MEDIAKIT_ERASE", "").lower() in ("1", "true", "yes"),
        mediakit_api_key=os.environ.get("VOLC_MEDIAKIT_API_KEY", "").strip(),
        mediakit_concurrency=max(1, int(os.environ.get("MEDIAKIT_CONCURRENCY", "4"))),
        mediakit_submit_interval_s=float(os.environ.get(
            "MEDIAKIT_SUBMIT_INTERVAL_S", "1.0"
        )),
        mediakit_timeout_s=float(os.environ.get("MEDIAKIT_TIMEOUT_S", "180")),
        seedream_model=os.environ.get(
            "SEEDREAM_MODEL", SEEDREAM_PRO_MODEL
        ).strip(),
        seedream_edit_mode=os.environ.get(
            "SEEDREAM_EDIT_MODE", "independent_parallel"
        ).strip(),
        seedream_concurrency=int(os.environ.get("SEEDREAM_CONCURRENCY", "4")),
        seedream_timeout_s=float(os.environ.get("SEEDREAM_TIMEOUT_S", "300")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        codex_timeout_s=int(os.environ.get("CODEX_TIMEOUT_S", "1800")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
        deepseek_credential_file=Path(os.environ.get(
            "DEEPSEEK_CREDENTIAL_FILE", "/home/xy/.config/claude/deepseek.env"
        )),
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
        enable_minimal_creation=os.environ.get(
            "ENABLE_MINIMAL_CREATION", ""
        ).lower() in ("1", "true", "yes"),
        public_api_enabled=os.environ.get(
            "PUBLIC_API_ENABLED", ""
        ).lower() in ("1", "true", "yes"),
        public_api_clients_file=Path(os.environ.get(
            "PUBLIC_API_CLIENTS_FILE",
            "/home/xy/.config/duet-ad1/public-api-clients.json",
        )),
    )
