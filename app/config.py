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
ARTIFACT_RETENTION_MODES = frozenset({"full", "lean"})

# 物品检测只走 SAM 3.1 video-rle（实现见 app/object_detect.py）：provider 链路
# 顺序 wavespeed → sandbase → fal（2026-09-19：fal 账号被官方 Admin lock、
# sandbase worker 连续 generation_timeout，两家当前均不可用，故新接入的
# wavespeed 排第一位）。
# WaveSpeed SAM3 video-rle：端点均为 <base>/api/v3/...（上传 ticket
# /api/v3/media/uploads、提交 /api/v3/<model>、轮询
# /api/v3/predictions/{id}/result），model 是含路径段的完整模型 ID。
# 凭证环境变量名沿用 WaveSpeed 官方 CLI/SDK 约定 WAVESPEED_API_KEY。
WAVESPEED_SAM3_API_BASE = "https://api.wavespeed.ai"
WAVESPEED_SAM3_MODEL = "wavespeed-ai/sam3-video-rle"
# fal 队列 base 含端点路径，model 为端点末段，提交 URL = <base>/<model>。
# 凭证环境变量名沿用 fal 平台统一约定 FAL_KEY（不是 FAL_SAM3_API_KEY）。
# 端点从 sam-3 升级到 sam-3-1（schema 一致，仅多一个可选 max_num_objects，
# 默认 16）；仍可用 FAL_SAM3_API_BASE 配回旧端点。
FAL_SAM3_API_BASE = "https://queue.fal.run/fal-ai/sam-3-1"
FAL_SAM3_MODEL = "video-rle"
# sandbase.ai SAM 3.1 video-rle（2026-09-19 起替代 fal 成为物品/人脸灰块检测
# 主 provider：fal 账号被官方 Admin lock，storage upload 403；fal 配置保留作
# 后备）。端点均为 <base>/v1/...（上传 /v1/upload、提交 /v1/run、轮询
# /v1/run/{id}），model 是含路径段的完整模型 ID。凭证环境变量名沿用
# sandbase 官方 CLI 约定 SANDBASE_API_KEY。
SANDBASE_SAM3_API_BASE = "https://api.sandbase.ai"
SANDBASE_SAM3_MODEL = "meta/sam-3-1/video-rle"
# sandbase 模型 ID 含 "/" 路径段（meta/sam-3-1/video-rle），校验比
# _MODEL_ID_RE 多允许一个字符。
_MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
# fal 代理只允许 http/https（socks5/socks5h 需要 socksio，未引入依赖），
# Scheme 白名单见 __post_init__。
_PROXY_SCHEMES = frozenset({"http", "https"})
# Picsart 直连可能受 DNS 污染；允许与 httpx 兼容的 SOCKS5 代理。
_PICSART_PROXY_SCHEMES = _PROXY_SCHEMES | frozenset({"socks5"})


@dataclass(frozen=True)
class Settings:
    access_token: str
    # 多参考图模式开关：显式设置只接受 composed | direct，缺省默认 direct。
    # composed：在"参考图构造与冻结"环节把多张用户图合成单一 provider
    # reference（app/product_views.py），下游与单图完全同构；direct：逐图直传，
    # reference_image_ids 贯穿 video-maker / object-replace / 图版 / bindings。
    # 两模式的分叉点在参考图构造与冻结环节，H3 提交处共用"按 h3_reference_images
    # 顺序传图"。不存在第三条旧单图链路：上传 1 张图就是所选模式的 1 图情形。
    # 本开关不进前端、不进公开 API 参数与响应。
    multiref_mode: str = "direct"
    # 落盘保留级别（裁剪实现见 app/artifact_retention.py）：full = 测试期
    # 全量落盘（缺省，不设环境变量时维持现状）；lean = 生产模式，成功项目
    # 完成后幂等清掉节点媒体副本与抽帧/代理图，指纹缓存、回执、state、
    # 成片全部保留，失败项目不裁。
    artifact_retention: str = "full"
    data_dir: Path = ROOT / "data"
    public_base_url: str = ""
    picsart_api_key: str = ""
    picsart_api_base: str = "https://api.picsart.com"
    # 直连 Picsart 受 DNS 污染影响时可设代理；None = 直连。
    picsart_proxy: str | None = None
    # 豆包语音合成 2.0（火山引擎 openspeech V3，实现见 app/doubao_tts.py）：
    # 新版控制台一把 API Key 走 X-Api-Key 鉴权，无 appid/cluster 配置；
    # 未配置时首个 TTS 调用直接报错，不回落其他火山产品凭证。
    volc_voice_api_key: str = field(default="", repr=False)
    seedream_api_key: str = field(default="", repr=False)
    seedream_api_base: str = SEEDREAM_API_BASE
    seedream_model: str = SEEDREAM_PRO_MODEL
    seedream_timeout_s: float = 300.0
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    fusion_agent: str = "deepseek"
    codex_timeout_s: float = 1800.0
    codex_concurrency: int = 10
    mediakit_api_key: str = ""
    mediakit_api_base: str = "https://mediakit.cn-beijing.volces.com/api/v1"
    enable_mediakit: bool = True
    mediakit_video_erase_mode: str = "pro"
    asr_cli: Path = Path("/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")
    asr_model: Path = Path("/home/xy/.local/share/duet-asr/ggml-small.bin")
    yamnet_model: Path = ROOT / "models" / "yamnet.tflite"
    request_timeout_s: float = 180.0
    provider_poll_interval_s: float = 3.0
    provider_poll_timeout_s: float = 1800.0
    # 429 限流长退避的累计等待上限（实现见 app/providers.py
    # RATE_LIMIT_BACKOFF_S 段头注释；仅检测类 provider 启用，其余 provider
    # 429 语义不变）。缺省 900s；合法值为正有限数，加载期校验。不进前端、
    # 不进公开 API。
    provider_rate_limit_budget_s: float = 900.0
    picsart_task_hang_limit_s: float = 21600.0
    max_upload_mb: int = 500
    store_lock_enabled: bool = True
    public_api_enabled: bool = False
    public_api_clients_file: Path | None = None
    # WaveSpeed SAM-3 物品检测 provider（链路第一位）；wavespeed_sam3_api_key
    # 缺省 None = 未配置（WAVESPEED_API_KEY 未设），此时回退 sandbase → fal
    # （都未配置则物品灰块节点不触发）。默认值来自上方
    # WAVESPEED_SAM3_API_BASE / WAVESPEED_SAM3_MODEL。该模型 schema 只有
    # video/prompt/point_prompts/box_prompts/apply_mask，没有置信度阈值参数，
    # 故无 threshold 配置项。直连可用性未实测，暂不引 proxy 配置项（与
    # sandbase 同处理）。
    wavespeed_sam3_api_key: str | None = field(default=None, repr=False)
    wavespeed_sam3_api_base: str = WAVESPEED_SAM3_API_BASE
    wavespeed_sam3_model: str = WAVESPEED_SAM3_MODEL
    # sandbase SAM-3 物品检测 provider（链路第二位）；sandbase_sam3_api_key
    # 缺省 None = 未配置（SANDBASE_API_KEY 未设），此时回退 fal（fal key 也未
    # 配置则物品灰块节点不触发）。默认值来自上方 SANDBASE_SAM3_API_BASE /
    # SANDBASE_SAM3_MODEL。国内直连 api.sandbase.ai 通畅（2026-09-19 实测），
    # 故没有 proxy 配置项。
    sandbase_sam3_api_key: str | None = field(default=None, repr=False)
    sandbase_sam3_api_base: str = SANDBASE_SAM3_API_BASE
    sandbase_sam3_model: str = SANDBASE_SAM3_MODEL
    # SAM 3 文本 prompt 检不出时官方建议降到 0.2-0.3（默认 0.5 偏高）；
    # 与 FAL_SAM3_THRESHOLD 语义相同但独立配置，合法区间 (0, 1]，加载期校验。
    sandbase_sam3_threshold: float = 0.3
    # fal SAM-3 物品检测（后备 provider）；fal_sam3_api_key 缺省 None = 未配置（FAL_KEY 未设，
    # 物品灰块节点不触发）。默认值来自上方 FAL_SAM3_API_BASE / FAL_SAM3_MODEL。
    fal_sam3_api_key: str | None = field(default=None, repr=False)
    fal_sam3_api_base: str = FAL_SAM3_API_BASE
    fal_sam3_model: str = FAL_SAM3_MODEL
    # 国内直连 fal.run 不通时需代理，本机一般为 http://127.0.0.1:7897；
    # None = 直连。
    fal_proxy: str | None = None
    # SAM 3 文本 prompt 检不出时官方建议降到 0.2-0.3（默认 0.5 偏高），
    # 本实例固定用 0.2；合法区间 (0, 1]，加载期校验。
    fal_sam3_threshold: float = 0.2
    # 物品灰块节点总开关：false 时 mask-replaceable-objects 完全不触发，
    # 不发起任何检测调用（无上传、无重试、无等待），行为与
    # object_mask_degraded 降级路径完全一致（参考视频保持 source-muted、
    # 提示词按未遮盖处理），trace 记 object_mask_disabled 以区分配置关闭
    # 与失败降级。缺省 true 维持现状；检测后端不可用期间（2026-09-19 fal
    # 账号被锁、sandbase worker 连续 generation_timeout，wavespeed 已接入
    # 并当日完成输出标定、尚未在真实项目启用）在 .env 显式写 false。本开关
    # 不进前端、不进公开 API。
    object_mask_enabled: bool = True
    # 灰块覆盖不透明度；1.0 保持当前纯实心灰块，低于 1.0 仅供 A/B 测试。
    object_mask_block_opacity: float = 1.0
    # 人物组（person/hair）灰块不透明度，默认 1.0 纯实心纯黑（0,0,0）填充、
    # 不做模糊混合：实测 98% 不透明度的 2% 残影就足够 H3 认出源脸（身份
    # 泄漏），人物块必须全不透明；物品组沿用 object_mask_block_opacity
    # （99%+模糊，保动作锚点）。合法区间 (0, 1]，加载期校验。
    object_mask_person_block_opacity: float = 1.0
    # 泄漏残影的高斯模糊核，仅在 opacity<1 时生效；必须为正奇数像素。
    object_mask_block_blur: int = 31

    def __post_init__(self) -> None:
        token = self.access_token.strip() if isinstance(self.access_token, str) else ""
        if not token:
            raise ValueError("ACCESS_TOKEN must be a non-empty value")
        if token != self.access_token:
            object.__setattr__(self, "access_token", token)
        if self.multiref_mode not in MULTIREF_MODES:
            raise ValueError("MULTIREF_MODE must be composed or direct")
        if self.artifact_retention not in ARTIFACT_RETENTION_MODES:
            raise ValueError("ARTIFACT_RETENTION must be full or lean")
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
        voice_key = (
            self.volc_voice_api_key.strip()
            if isinstance(self.volc_voice_api_key, str)
            else ""
        )
        if voice_key and (
            not voice_key.isascii()
            or any(not 0x21 <= ord(character) <= 0x7E for character in voice_key)
        ):
            raise ValueError("VOLC_VOICE_API_KEY must be header-safe ASCII")
        object.__setattr__(self, "volc_voice_api_key", voice_key)
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
        if (
            isinstance(self.provider_rate_limit_budget_s, bool)
            or not isinstance(self.provider_rate_limit_budget_s, (int, float))
            or not math.isfinite(float(self.provider_rate_limit_budget_s))
            or self.provider_rate_limit_budget_s <= 0
        ):
            raise ValueError(
                "PROVIDER_RATE_LIMIT_BUDGET_S must be a positive finite number"
            )
        clients_file = self.public_api_clients_file
        if clients_file is None:
            clients_file = self.data_dir / ".public-api" / "public-api-clients.json"
        if not isinstance(clients_file, Path):
            raise ValueError("PUBLIC_API_CLIENTS_FILE must be a filesystem path")
        object.__setattr__(self, "public_api_clients_file", clients_file)
        wavespeed_key = self.wavespeed_sam3_api_key
        if wavespeed_key is not None:
            if not isinstance(wavespeed_key, str) or not wavespeed_key.strip():
                raise ValueError("WAVESPEED_API_KEY must be a non-empty value or unset")
            wavespeed_key = wavespeed_key.strip()
            if (
                not wavespeed_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in wavespeed_key)
            ):
                raise ValueError("WAVESPEED_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "wavespeed_sam3_api_key", wavespeed_key)
        wavespeed_base = (
            self.wavespeed_sam3_api_base.strip().rstrip("/")
            if isinstance(self.wavespeed_sam3_api_base, str)
            else ""
        )
        try:
            wavespeed_base_url = urlsplit(wavespeed_base)
        except ValueError as exc:
            raise ValueError(
                "WAVESPEED_SAM3_API_BASE must be an HTTPS base URL"
            ) from exc
        if wavespeed_base_url.scheme != "https" or not wavespeed_base_url.netloc:
            raise ValueError("WAVESPEED_SAM3_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "wavespeed_sam3_api_base", wavespeed_base)
        wavespeed_model = (
            self.wavespeed_sam3_model.strip()
            if isinstance(self.wavespeed_sam3_model, str)
            else ""
        )
        if not _MODEL_PATH_RE.fullmatch(wavespeed_model):
            raise ValueError("WAVESPEED_SAM3_MODEL must be a valid model identifier")
        object.__setattr__(self, "wavespeed_sam3_model", wavespeed_model)
        sandbase_key = self.sandbase_sam3_api_key
        if sandbase_key is not None:
            if not isinstance(sandbase_key, str) or not sandbase_key.strip():
                raise ValueError("SANDBASE_API_KEY must be a non-empty value or unset")
            sandbase_key = sandbase_key.strip()
            if (
                not sandbase_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in sandbase_key)
            ):
                raise ValueError("SANDBASE_API_KEY must be header-safe ASCII")
            object.__setattr__(self, "sandbase_sam3_api_key", sandbase_key)
        sandbase_base = (
            self.sandbase_sam3_api_base.strip().rstrip("/")
            if isinstance(self.sandbase_sam3_api_base, str)
            else ""
        )
        try:
            sandbase_base_url = urlsplit(sandbase_base)
        except ValueError as exc:
            raise ValueError(
                "SANDBASE_SAM3_API_BASE must be an HTTPS base URL"
            ) from exc
        if sandbase_base_url.scheme != "https" or not sandbase_base_url.netloc:
            raise ValueError("SANDBASE_SAM3_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "sandbase_sam3_api_base", sandbase_base)
        sandbase_model = (
            self.sandbase_sam3_model.strip()
            if isinstance(self.sandbase_sam3_model, str)
            else ""
        )
        if not _MODEL_PATH_RE.fullmatch(sandbase_model):
            raise ValueError("SANDBASE_SAM3_MODEL must be a valid model identifier")
        object.__setattr__(self, "sandbase_sam3_model", sandbase_model)
        sandbase_threshold = self.sandbase_sam3_threshold
        if (
            isinstance(sandbase_threshold, bool)
            or not isinstance(sandbase_threshold, (int, float))
            or not math.isfinite(float(sandbase_threshold))
            or not 0.0 < float(sandbase_threshold) <= 1.0
        ):
            raise ValueError("SANDBASE_SAM3_THRESHOLD must be a number in (0, 1]")
        object.__setattr__(self, "sandbase_sam3_threshold", float(sandbase_threshold))
        fal_key = self.fal_sam3_api_key
        if fal_key is not None:
            if not isinstance(fal_key, str) or not fal_key.strip():
                raise ValueError("FAL_KEY must be a non-empty value or unset")
            fal_key = fal_key.strip()
            if (
                not fal_key.isascii()
                or any(not 0x21 <= ord(character) <= 0x7E for character in fal_key)
            ):
                raise ValueError("FAL_KEY must be header-safe ASCII")
            object.__setattr__(self, "fal_sam3_api_key", fal_key)
        fal_base = (
            self.fal_sam3_api_base.strip().rstrip("/")
            if isinstance(self.fal_sam3_api_base, str)
            else ""
        )
        try:
            fal_base_url = urlsplit(fal_base)
        except ValueError as exc:
            raise ValueError(
                "FAL_SAM3_API_BASE must be an HTTPS base URL"
            ) from exc
        if fal_base_url.scheme != "https" or not fal_base_url.netloc:
            raise ValueError("FAL_SAM3_API_BASE must be an HTTPS base URL")
        object.__setattr__(self, "fal_sam3_api_base", fal_base)
        fal_model = (
            self.fal_sam3_model.strip()
            if isinstance(self.fal_sam3_model, str)
            else ""
        )
        if not _MODEL_ID_RE.fullmatch(fal_model):
            raise ValueError("FAL_SAM3_MODEL must be a valid model identifier")
        object.__setattr__(self, "fal_sam3_model", fal_model)
        proxy_value = self.fal_proxy
        if proxy_value is not None:
            if not isinstance(proxy_value, str) or not proxy_value.strip():
                raise ValueError("FAL_PROXY must be a non-empty value or unset")
            proxy_value = proxy_value.strip()
            try:
                proxy_url = urlsplit(proxy_value)
            except ValueError as exc:
                raise ValueError(
                    "FAL_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                ) from exc
            if (
                proxy_url.scheme not in _PROXY_SCHEMES
                or not proxy_url.netloc
            ):
                raise ValueError(
                    "FAL_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                )
            object.__setattr__(self, "fal_proxy", proxy_value)
        picsart_proxy_value = self.picsart_proxy
        if picsart_proxy_value is not None:
            if (
                not isinstance(picsart_proxy_value, str)
                or not picsart_proxy_value.strip()
            ):
                raise ValueError("PICSART_PROXY must be a non-empty value or unset")
            picsart_proxy_value = picsart_proxy_value.strip()
            try:
                picsart_proxy_url = urlsplit(picsart_proxy_value)
            except ValueError as exc:
                raise ValueError(
                    "PICSART_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                ) from exc
            if (
                picsart_proxy_url.scheme not in _PICSART_PROXY_SCHEMES
                or not picsart_proxy_url.netloc
            ):
                raise ValueError(
                    "PICSART_PROXY must be a proxy URL, e.g. http://127.0.0.1:7897"
                )
            object.__setattr__(self, "picsart_proxy", picsart_proxy_value)
        threshold = self.fal_sam3_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) <= 1.0
        ):
            raise ValueError("FAL_SAM3_THRESHOLD must be a number in (0, 1]")
        object.__setattr__(self, "fal_sam3_threshold", float(threshold))
        if isinstance(self.object_mask_enabled, bool) is not True:
            raise ValueError("OBJECT_MASK_ENABLED must be a boolean")
        object_mask_block_opacity = self.object_mask_block_opacity
        if (
            isinstance(object_mask_block_opacity, bool)
            or not isinstance(object_mask_block_opacity, (int, float))
            or not math.isfinite(float(object_mask_block_opacity))
            or not 0.0 < float(object_mask_block_opacity) <= 1.0
        ):
            raise ValueError(
                "OBJECT_MASK_BLOCK_OPACITY must be a number in (0, 1]"
            )
        object.__setattr__(
            self, "object_mask_block_opacity", float(object_mask_block_opacity)
        )
        person_block_opacity = self.object_mask_person_block_opacity
        if (
            isinstance(person_block_opacity, bool)
            or not isinstance(person_block_opacity, (int, float))
            or not math.isfinite(float(person_block_opacity))
            or not 0.0 < float(person_block_opacity) <= 1.0
        ):
            raise ValueError(
                "OBJECT_MASK_PERSON_BLOCK_OPACITY must be a number in (0, 1]"
            )
        object.__setattr__(
            self, "object_mask_person_block_opacity", float(person_block_opacity)
        )
        object_mask_block_blur = self.object_mask_block_blur
        if (
            isinstance(object_mask_block_blur, bool)
            or not isinstance(object_mask_block_blur, int)
            or object_mask_block_blur < 1
            or object_mask_block_blur % 2 != 1
        ):
            raise ValueError("OBJECT_MASK_BLOCK_BLUR must be a positive odd integer")


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
    # 缺省默认 direct；显式设置的合法性由 Settings.__post_init__ 校验。
    multiref_mode = os.environ.get("MULTIREF_MODE", "").strip().lower() or "direct"
    # 缺省默认 full（测试期现状）；生产启动脚本显式写 lean。
    artifact_retention = (
        os.environ.get("ARTIFACT_RETENTION", "").strip().lower() or "full"
    )
    try:
        sandbase_sam3_threshold = float(
            os.environ.get("SANDBASE_SAM3_THRESHOLD", "0.3").strip()
        )
    except ValueError as exc:
        raise ValueError("SANDBASE_SAM3_THRESHOLD must be a number in (0, 1]") from exc
    try:
        fal_sam3_threshold = float(
            os.environ.get("FAL_SAM3_THRESHOLD", "0.2").strip()
        )
    except ValueError as exc:
        raise ValueError("FAL_SAM3_THRESHOLD must be a number in (0, 1]") from exc
    try:
        object_mask_block_opacity = float(
            os.environ.get("OBJECT_MASK_BLOCK_OPACITY", "1.0").strip()
        )
    except ValueError as exc:
        raise ValueError(
            "OBJECT_MASK_BLOCK_OPACITY must be a number in (0, 1]"
        ) from exc
    try:
        object_mask_block_blur = int(
            os.environ.get("OBJECT_MASK_BLOCK_BLUR", "31").strip()
        )
    except ValueError as exc:
        raise ValueError("OBJECT_MASK_BLOCK_BLUR must be a positive odd integer") from exc
    try:
        object_mask_person_block_opacity = float(
            os.environ.get("OBJECT_MASK_PERSON_BLOCK_OPACITY", "1.0").strip()
        )
    except ValueError as exc:
        raise ValueError(
            "OBJECT_MASK_PERSON_BLOCK_OPACITY must be a number in (0, 1]"
        ) from exc
    return Settings(
        access_token=access_token,
        multiref_mode=multiref_mode,
        artifact_retention=artifact_retention,
        data_dir=Path(os.environ.get("DATA_DIR", str(ROOT / "data"))).resolve(),
        public_base_url=os.environ.get("PUBLIC_BASE_URL", ""),
        picsart_api_key=os.environ.get("PICSART_API_KEY", ""),
        picsart_api_base=os.environ.get("PICSART_API_BASE", "https://api.picsart.com").rstrip("/"),
        picsart_proxy=os.environ.get("PICSART_PROXY", "").strip() or None,
        volc_voice_api_key=os.environ.get("VOLC_VOICE_API_KEY", "").strip(),
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
        deepseek_model=os.environ.get(
            "DEEPSEEK_MODEL", "deepseek-flash"
        ),
        fusion_agent=os.environ.get("FUSION_AGENT", "deepseek").strip().lower(),
        codex_timeout_s=float(os.environ.get("CODEX_TIMEOUT_S", "1800")),
        codex_concurrency=int(os.environ.get("CODEX_CONCURRENCY", "10")),
        mediakit_api_key=os.environ.get("VOLC_MEDIAKIT_API_KEY", ""),
        mediakit_api_base=os.environ.get("MEDIAKIT_API_BASE", "https://mediakit.cn-beijing.volces.com/api/v1").rstrip("/"),
        enable_mediakit=_bool("ENABLE_MEDIAKIT", True),
        mediakit_video_erase_mode=os.environ.get(
            "MEDIAKIT_VIDEO_ERASE_MODE", "pro"
        ).strip().lower(),
        asr_cli=Path(os.environ.get("ASR_CLI", "/home/xy/.local/share/duet-asr/whisper.cpp-1.9.2-src/build/bin/whisper-cli")),
        asr_model=Path(os.environ.get("ASR_MODEL", "/home/xy/.local/share/duet-asr/ggml-small.bin")),
        yamnet_model=Path(os.environ.get("YAMNET_MODEL_PATH", str(ROOT / "models" / "yamnet.tflite"))),
        request_timeout_s=float(os.environ.get("REQUEST_TIMEOUT_S", "180")),
        provider_poll_interval_s=float(os.environ.get("PROVIDER_POLL_INTERVAL_S", "3")),
        provider_poll_timeout_s=float(os.environ.get("PROVIDER_POLL_TIMEOUT_S", "1800")),
        provider_rate_limit_budget_s=float(
            os.environ.get("PROVIDER_RATE_LIMIT_BUDGET_S", "900")
        ),
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
        wavespeed_sam3_api_key=os.environ.get("WAVESPEED_API_KEY", "").strip() or None,
        wavespeed_sam3_api_base=os.environ.get(
            "WAVESPEED_SAM3_API_BASE", WAVESPEED_SAM3_API_BASE
        ).strip(),
        wavespeed_sam3_model=os.environ.get(
            "WAVESPEED_SAM3_MODEL", WAVESPEED_SAM3_MODEL
        ).strip(),
        sandbase_sam3_api_key=os.environ.get("SANDBASE_API_KEY", "").strip() or None,
        sandbase_sam3_api_base=os.environ.get(
            "SANDBASE_SAM3_API_BASE", SANDBASE_SAM3_API_BASE
        ).strip(),
        sandbase_sam3_model=os.environ.get(
            "SANDBASE_SAM3_MODEL", SANDBASE_SAM3_MODEL
        ).strip(),
        sandbase_sam3_threshold=sandbase_sam3_threshold,
        fal_sam3_api_key=os.environ.get("FAL_KEY", "").strip() or None,
        fal_sam3_api_base=os.environ.get(
            "FAL_SAM3_API_BASE", FAL_SAM3_API_BASE
        ).strip(),
        fal_sam3_model=os.environ.get(
            "FAL_SAM3_MODEL", FAL_SAM3_MODEL
        ).strip(),
        fal_proxy=os.environ.get("FAL_PROXY", "").strip() or None,
        fal_sam3_threshold=fal_sam3_threshold,
        object_mask_enabled=_bool("OBJECT_MASK_ENABLED", True),
        object_mask_block_opacity=object_mask_block_opacity,
        object_mask_block_blur=object_mask_block_blur,
        object_mask_person_block_opacity=object_mask_person_block_opacity,
    )
