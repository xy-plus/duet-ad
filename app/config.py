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

# ==== 物品检测 provider（实验分支阶段 4；独立段落，便于 7 家分支合并） ====
# 已知 7 家视频理解 provider；注册表与实现在 app/object_detect.py，
# 两家清单必须保持一致（object_detect.py 断言）。
OBJECT_DETECT_PROVIDER_NAMES = (
    "doubao",
    "qwen",
    "glm",
    "hunyuan",
    "kimi",
    "step",
    "gemini",
)
DOUBAO_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
# 模型 ID 以火山方舟控制台为准，改动时同步 DOUBAO_MODEL 环境变量。
DOUBAO_GROUNDING_MODEL = "doubao-seed-2-1-pro-260628"
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 模型 ID 以阿里云百炼控制台为准，改动时同步 QWEN_MODEL 环境变量。
QWEN_GROUNDING_MODEL = "qwen3-vl-plus"
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"
GLM_GROUNDING_MODEL = "glm-4.5v"
HUNYUAN_API_BASE = "https://api.hunyuan.cloud.tencent.com/v1"
HUNYUAN_VISION_MODEL = "hunyuan-large-vision"


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
    mediakit_video_erase_mode: str = "pro"
    asr_cli: Path = Path("/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")
    asr_model: Path = Path("/home/xy/.local/share/duet-asr/ggml-small.bin")
    yamnet_model: Path = ROOT / "models" / "yamnet.tflite"
    request_timeout_s: float = 180.0
    provider_poll_interval_s: float = 3.0
    provider_poll_timeout_s: float = 1800.0
    picsart_task_hang_limit_s: float = 21600.0
    max_upload_mb: int = 500
    store_lock_enabled: bool = True
    public_api_enabled: bool = False
    public_api_clients_file: Path | None = None
    # ==== 物品检测 provider 字段（阶段 4 独立段落） ====
    # 有序启用列表；空 = 未配置 = 物品灰块节点不触发。
    object_detect_providers: tuple[str, ...] = ()
    # 缺省 None 表示未配置（该 provider 不启用）。
    doubao_api_key: str | None = field(default=None, repr=False)
    doubao_api_base: str = DOUBAO_API_BASE
    doubao_model: str = DOUBAO_GROUNDING_MODEL
    qwen_api_key: str | None = field(default=None, repr=False)
    qwen_api_base: str = QWEN_API_BASE
    qwen_model: str = QWEN_GROUNDING_MODEL
    glm_api_key: str | None = field(default=None, repr=False)
    glm_api_base: str = GLM_API_BASE
    glm_model: str = GLM_GROUNDING_MODEL
    hunyuan_api_key: str | None = field(default=None, repr=False)
    hunyuan_api_base: str = HUNYUAN_API_BASE
    hunyuan_model: str = HUNYUAN_VISION_MODEL

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
        if self.mediakit_video_erase_mode not in {"pro", "standard"}:
            raise ValueError("MEDIAKIT_VIDEO_ERASE_MODE must be pro or standard")
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
        if isinstance(self.public_api_enabled, bool) is not True:
            raise ValueError("PUBLIC_API_ENABLED must be a boolean")
        if (
            isinstance(self.picsart_task_hang_limit_s, bool)
            or not isinstance(self.picsart_task_hang_limit_s, (int, float))
            or not math.isfinite(float(self.picsart_task_hang_limit_s))
            or self.picsart_task_hang_limit_s <= 0
        ):
            raise ValueError("PICSART_TASK_HANG_LIMIT_S must be a positive finite number")
        clients_file = self.public_api_clients_file
        if clients_file is None:
            clients_file = self.data_dir / ".public-api" / "public-api-clients.json"
        if not isinstance(clients_file, Path):
            raise ValueError("PUBLIC_API_CLIENTS_FILE must be a filesystem path")
        object.__setattr__(self, "public_api_clients_file", clients_file)
        # ==== 物品检测 provider 校验（阶段 4 独立段落） ====
        if not isinstance(self.object_detect_providers, tuple) or any(
            not isinstance(name, str) for name in self.object_detect_providers
        ):
            raise ValueError("OBJECT_DETECT_PROVIDERS must be a tuple of names")
        unknown = [
            name
            for name in self.object_detect_providers
            if name not in OBJECT_DETECT_PROVIDER_NAMES
        ]
        if unknown:
            raise ValueError(
                "OBJECT_DETECT_PROVIDERS has unknown provider names: "
                + ", ".join(unknown)
            )
        doubao_key = self.doubao_api_key
        if doubao_key is not None:
            if not isinstance(doubao_key, str) or not doubao_key.strip():
                raise ValueError("DOUBAO_API_KEY must be a non-empty value or unset")
            doubao_key = doubao_key.strip()
            if (
                not doubao_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in doubao_key)
            ):
                raise ValueError("DOUBAO_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "doubao_api_key", doubao_key)
        doubao_base = (
            self.doubao_api_base.strip().rstrip("/")
            if isinstance(self.doubao_api_base, str)
            else ""
        )
        try:
            doubao_url = urlsplit(doubao_base)
        except ValueError as exc:
            raise ValueError("DOUBAO_API_BASE must be an HTTPS base URL") from exc
        if doubao_url.scheme != "https" or not doubao_url.netloc:
            raise ValueError("DOUBAO_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "doubao_api_base", doubao_base)
        doubao_model = (
            self.doubao_model.strip()
            if isinstance(self.doubao_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(doubao_model):
            raise ValueError("DOUBAO_MODEL must be a valid model identifier")
        object.__setattr__(self, "doubao_model", doubao_model)
        qwen_key = self.qwen_api_key
        if qwen_key is not None:
            if not isinstance(qwen_key, str) or not qwen_key.strip():
                raise ValueError("QWEN_API_KEY must be a non-empty value or unset")
            qwen_key = qwen_key.strip()
            if (
                not qwen_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in qwen_key)
            ):
                raise ValueError("QWEN_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "qwen_api_key", qwen_key)
        qwen_base = (
            self.qwen_api_base.strip().rstrip("/")
            if isinstance(self.qwen_api_base, str)
            else ""
        )
        try:
            qwen_url = urlsplit(qwen_base)
        except ValueError as exc:
            raise ValueError("QWEN_API_BASE must be an HTTPS base URL") from exc
        if qwen_url.scheme != "https" or not qwen_url.netloc:
            raise ValueError("QWEN_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "qwen_api_base", qwen_base)
        qwen_model = (
            self.qwen_model.strip()
            if isinstance(self.qwen_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(qwen_model):
            raise ValueError("QWEN_MODEL must be a valid model identifier")
        object.__setattr__(self, "qwen_model", qwen_model)
        glm_key = self.glm_api_key
        if glm_key is not None:
            if not isinstance(glm_key, str) or not glm_key.strip():
                raise ValueError("GLM_API_KEY must be a non-empty value or unset")
            glm_key = glm_key.strip()
            if (
                not glm_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in glm_key)
            ):
                raise ValueError("GLM_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "glm_api_key", glm_key)
        glm_base = (
            self.glm_api_base.strip().rstrip("/")
            if isinstance(self.glm_api_base, str)
            else ""
        )
        try:
            glm_url = urlsplit(glm_base)
        except ValueError as exc:
            raise ValueError("GLM_API_BASE must be an HTTPS base URL") from exc
        if glm_url.scheme != "https" or not glm_url.netloc:
            raise ValueError("GLM_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "glm_api_base", glm_base)
        glm_model = (
            self.glm_model.strip()
            if isinstance(self.glm_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(glm_model):
            raise ValueError("GLM_MODEL must be a valid model identifier")
        object.__setattr__(self, "glm_model", glm_model)
        hunyuan_key = self.hunyuan_api_key
        if hunyuan_key is not None:
            if not isinstance(hunyuan_key, str) or not hunyuan_key.strip():
                raise ValueError("HUNYUAN_API_KEY must be a non-empty value or unset")
            hunyuan_key = hunyuan_key.strip()
            if (
                not hunyuan_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in hunyuan_key)
            ):
                raise ValueError("HUNYUAN_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "hunyuan_api_key", hunyuan_key)
        hunyuan_base = (
            self.hunyuan_api_base.strip().rstrip("/")
            if isinstance(self.hunyuan_api_base, str)
            else ""
        )
        try:
            hunyuan_url = urlsplit(hunyuan_base)
        except ValueError as exc:
            raise ValueError("HUNYUAN_API_BASE must be an HTTPS base URL") from exc
        if hunyuan_url.scheme != "https" or not hunyuan_url.netloc:
            raise ValueError("HUNYUAN_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "hunyuan_api_base", hunyuan_base)
        hunyuan_model = (
            self.hunyuan_model.strip()
            if isinstance(self.hunyuan_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(hunyuan_model):
            raise ValueError("HUNYUAN_MODEL must be a valid model identifier")
        object.__setattr__(self, "hunyuan_model", hunyuan_model)


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
        mediakit_video_erase_mode=os.environ.get(
            "MEDIAKIT_VIDEO_ERASE_MODE", "pro"
        ).strip().lower(),
        asr_cli=Path(os.environ.get("ASR_CLI", "/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")),
        asr_model=Path(os.environ.get("ASR_MODEL", "/home/xy/.local/share/duet-asr/ggml-small.bin")),
        yamnet_model=Path(os.environ.get("YAMNET_MODEL_PATH", str(ROOT / "models" / "yamnet.tflite"))),
        request_timeout_s=float(os.environ.get("REQUEST_TIMEOUT_S", "180")),
        provider_poll_interval_s=float(os.environ.get("PROVIDER_POLL_INTERVAL_S", "3")),
        provider_poll_timeout_s=float(os.environ.get("PROVIDER_POLL_TIMEOUT_S", "1800")),
        picsart_task_hang_limit_s=float(os.environ.get("PICSART_TASK_HANG_LIMIT_S", "21600")),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "500")),
        public_api_enabled=_bool("PUBLIC_API_ENABLED", False),
        public_api_clients_file=(
            Path(clients_file).expanduser() if (clients_file := os.environ.get("PUBLIC_API_CLIENTS_FILE", "").strip()) else None
        ),
        # Provider asset grants live in process memory.  Production settings
        # therefore cannot disable the single-worker store lock; tests may do
        # so only by constructing Settings explicitly.
        store_lock_enabled=True,
        # ==== 物品检测 provider（阶段 4 独立段落） ====
        object_detect_providers=tuple(
            name.strip().lower()
            for name in os.environ.get("OBJECT_DETECT_PROVIDERS", "").split(",")
            if name.strip()
        ),
        doubao_api_key=os.environ.get("DOUBAO_API_KEY", "").strip() or None,
        doubao_api_base=os.environ.get("DOUBAO_API_BASE", DOUBAO_API_BASE).strip(),
        doubao_model=os.environ.get("DOUBAO_MODEL", DOUBAO_GROUNDING_MODEL).strip(),
        qwen_api_key=os.environ.get("QWEN_API_KEY", "").strip() or None,
        qwen_api_base=os.environ.get("QWEN_API_BASE", QWEN_API_BASE).strip(),
        qwen_model=os.environ.get("QWEN_MODEL", QWEN_GROUNDING_MODEL).strip(),
        glm_api_key=os.environ.get("GLM_API_KEY", "").strip() or None,
        glm_api_base=os.environ.get("GLM_API_BASE", GLM_API_BASE).strip(),
        glm_model=os.environ.get("GLM_MODEL", GLM_GROUNDING_MODEL).strip(),
        hunyuan_api_key=os.environ.get("HUNYUAN_API_KEY", "").strip() or None,
        hunyuan_api_base=os.environ.get("HUNYUAN_API_BASE", HUNYUAN_API_BASE).strip(),
        hunyuan_model=os.environ.get("HUNYUAN_MODEL", HUNYUAN_VISION_MODEL).strip(),
    )
