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
MULTIREF_MODES = frozenset({"composed", "direct"})

# 已知 7 家视频理解 provider；注册表与实现在 app/object_detect.py，
# 两处清单必须保持一致（object_detect.py 断言）。
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
KIMI_API_BASE = "https://api.moonshot.cn/v1"
# k2.5/k3 不支持视频输入；模型 ID 以 Moonshot 控制台为准。
KIMI_GROUNDING_MODEL = "kimi-k2.6"
STEP_API_BASE = "https://api.stepfun.com/v1"
# 模型 ID 以阶跃星辰控制台为准（Step-1.5V 起支持视频理解），改动时同步
# STEP_MODEL 环境变量。
STEP_GROUNDING_MODEL = "step-3.7-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
GEMINI_MODEL = "gemini-2.5-flash"
# (名称, 默认 API base, 默认模型) 驱动 Settings 校验与 get_settings 的环境
# 变量读取（<NAME>_API_KEY / <NAME>_API_BASE / <NAME>_MODEL）。
_OBJECT_DETECT_PROVIDER_SPECS: tuple[tuple[str, str, str], ...] = (
    ("doubao", DOUBAO_API_BASE, DOUBAO_GROUNDING_MODEL),
    ("qwen", QWEN_API_BASE, QWEN_GROUNDING_MODEL),
    ("glm", GLM_API_BASE, GLM_GROUNDING_MODEL),
    ("hunyuan", HUNYUAN_API_BASE, HUNYUAN_VISION_MODEL),
    ("kimi", KIMI_API_BASE, KIMI_GROUNDING_MODEL),
    ("step", STEP_API_BASE, STEP_GROUNDING_MODEL),
    ("gemini", GEMINI_API_BASE, GEMINI_MODEL),
)
OBJECT_DETECT_PROVIDER_NAMES = tuple(
    name for name, _base, _model in _OBJECT_DETECT_PROVIDER_SPECS
)
# GEMINI_PROXY 只允许 http/https（socks5/socks5h 需要 socksio，未引入依赖），
# Scheme 白名单见 __post_init__。
_GEMINI_PROXY_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class Settings:
    access_token: str
    # 多参考图模式开关：只有 composed | direct 两个值，必须显式配置、无缺省。
    # composed：在"参考图构造与冻结"环节把多张用户图合成单一 provider
    # reference（app/product_views.py），下游与单图完全同构；direct：逐图直传，
    # reference_image_ids 贯穿 video-maker / object-replace / 图版 / bindings。
    # 两模式的分叉点在参考图构造与冻结环节，H3 提交处共用"按 h3_reference_images
    # 顺序传图"。不存在第三条旧单图链路：上传 1 张图就是所选模式的 1 图情形。
    # 本开关不进前端、不进公开 API 参数与响应。
    multiref_mode: str
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
    # 有序启用列表；空 = 未配置 = 物品灰块节点不触发。名称必须是已知 7 家
    # 之一（OBJECT_DETECT_PROVIDER_NAMES），名称与选择一律不出现在公开 API。
    object_detect_providers: tuple[str, ...] = ()
    # 每家的 <name>_api_key 缺省 None 表示未配置（该 provider 不启用）；
    # 默认值来自 _OBJECT_DETECT_PROVIDER_SPECS。
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
    kimi_api_key: str | None = field(default=None, repr=False)
    kimi_api_base: str = KIMI_API_BASE
    kimi_model: str = KIMI_GROUNDING_MODEL
    step_api_key: str | None = field(default=None, repr=False)
    step_api_base: str = STEP_API_BASE
    step_model: str = STEP_GROUNDING_MODEL
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_api_base: str = GEMINI_API_BASE
    gemini_model: str = GEMINI_MODEL
    # 国内直连 generativelanguage.googleapis.com 不通时需代理，
    # 本机一般为 http://127.0.0.1:7897；None = 直连。
    gemini_proxy: str | None = None

    def __post_init__(self) -> None:
        token = self.access_token.strip() if isinstance(self.access_token, str) else ""
        if not token:
            raise ValueError("ACCESS_TOKEN must be a non-empty value")
        if token != self.access_token:
            object.__setattr__(self, "access_token", token)
        if self.multiref_mode not in MULTIREF_MODES:
            raise ValueError("MULTIREF_MODE must be composed or direct")
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
        for name in OBJECT_DETECT_PROVIDER_NAMES:
            env_prefix = name.upper()
            key = getattr(self, f"{name}_api_key")
            if key is not None:
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        f"{env_prefix}_API_KEY must be a non-empty value or unset"
                    )
                key = key.strip()
                if (
                    not key.isascii()
                    or any(not 0x21 <= ord(character) <= 0x7E for character in key)
                ):
                    raise ValueError(f"{env_prefix}_API_KEY must be header-safe ASCII")
                object.__setattr__(self, f"{name}_api_key", key)
            base = getattr(self, f"{name}_api_base")
            base = base.strip().rstrip("/") if isinstance(base, str) else ""
            try:
                base_url = urlsplit(base)
            except ValueError as exc:
                raise ValueError(
                    f"{env_prefix}_API_BASE must be an HTTPS base URL"
                ) from exc
            if base_url.scheme != "https" or not base_url.netloc:
                raise ValueError(f"{env_prefix}_API_BASE must be an HTTPS base URL")
            object.__setattr__(self, f"{name}_api_base", base)
            model = getattr(self, f"{name}_model")
            model = model.strip() if isinstance(model, str) else ""
            if not _MODEL_ID_RE.fullmatch(model):
                raise ValueError(f"{env_prefix}_MODEL must be a valid model identifier")
            object.__setattr__(self, f"{name}_model", model)
        gemini_proxy = self.gemini_proxy
        if gemini_proxy is not None:
            if not isinstance(gemini_proxy, str) or not gemini_proxy.strip():
                raise ValueError("GEMINI_PROXY must be a non-empty value or unset")
            gemini_proxy = gemini_proxy.strip()
            try:
                proxy_url = urlsplit(gemini_proxy)
            except ValueError as exc:
                raise ValueError(
                    "GEMINI_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                ) from exc
            if (
                proxy_url.scheme not in _GEMINI_PROXY_SCHEMES
                or not proxy_url.netloc
            ):
                raise ValueError(
                    "GEMINI_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                )
            object.__setattr__(self, "gemini_proxy", gemini_proxy)


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
    multiref_mode = os.environ.get("MULTIREF_MODE", "").strip().lower()
    if not multiref_mode:
        raise RuntimeError("MULTIREF_MODE must be set to composed or direct")
    provider_fields: dict[str, str | None] = {}
    for name, default_base, default_model in _OBJECT_DETECT_PROVIDER_SPECS:
        env_prefix = name.upper()
        provider_fields[f"{name}_api_key"] = (
            os.environ.get(f"{env_prefix}_API_KEY", "").strip() or None
        )
        provider_fields[f"{name}_api_base"] = os.environ.get(
            f"{env_prefix}_API_BASE", default_base
        ).strip()
        provider_fields[f"{name}_model"] = os.environ.get(
            f"{env_prefix}_MODEL", default_model
        ).strip()
    return Settings(
        access_token=access_token,
        multiref_mode=multiref_mode,
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
        object_detect_providers=tuple(
            name.strip().lower()
            for name in os.environ.get("OBJECT_DETECT_PROVIDERS", "").split(",")
            if name.strip()
        ),
        **provider_fields,
        gemini_proxy=os.environ.get("GEMINI_PROXY", "").strip() or None,
    )
