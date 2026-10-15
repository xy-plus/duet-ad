"""Shared fixtures for the real-input node replay suite.

Loads provider credentials from the same env files the deploy script sources
(values are never printed), registers the ``node_replay`` marker and gates
it behind ``NODE_REPLAY_LIVE`` (default OFF: the replay suite — CPU-heavy
local media replays, live DeepSeek contract tests and the paid Seedream
submit — only runs when explicitly enabled with ``NODE_REPLAY_LIVE=1``, as
the E2E 前置门禁 does).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.providers import DeepSeekClient  # noqa: E402
from app.seedream import SeedreamClient  # noqa: E402
from app.store import ProjectStore  # noqa: E402

# Same credential sources as deploy/run-backend-3212.sh.
_ENV_FILES = [
    Path("/home/xy/.config/duet-ad1/service.env"),
    Path("/home/xy/.config/claude/deepseek.env"),
    Path("/home/xy/duet-ad1-h3-max-r2v/.env"),
]


def _load_env_file(path: Path) -> None:
    """Apply `export KEY=value` lines without overriding the environment."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


for _env_file in _ENV_FILES:
    _load_env_file(_env_file)


def _live_enabled() -> bool:
    return os.environ.get("NODE_REPLAY_LIVE", "0").lower() in {
        "1", "true", "yes", "on",
    }


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "node_replay: harvested-input node replay (offline media replays,"
        " live DeepSeek contract tests, paid Seedream submit);"
        " enable with NODE_REPLAY_LIVE=1",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    if _live_enabled():
        return
    skip = pytest.mark.skip(
        reason="node replay suite skipped by default; enable with"
        " NODE_REPLAY_LIVE=1 (E2E 前置门禁)"
    )
    for item in items:
        if "node_replay" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _dialogue_delivery_test_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    """离线测试给目录内全部音色注入语速标定值，使标定门禁（选音色、默认
    音色、提示词精选、delivery 规划）不挡住与配音无关的用例；生产标定表
    仍是 app.doubao_voices.VOICE_BASELINE_RATES，缺口报错用例自行覆盖该表。"""
    from app.doubao_voices import (
        VOICE_BASELINE_RATES,
        cached_doubao_voice_catalog,
    )

    for voice_type in cached_doubao_voice_catalog().voice_types():
        if voice_type not in VOICE_BASELINE_RATES:
            monkeypatch.setitem(VOICE_BASELINE_RATES, voice_type, 12.0)


@pytest.fixture(autouse=True)
def _ark_audio_client_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """离线套件把 pipeline 惰性构造的方舟音频理解客户端换成本地 stub（付费
    推理不进离线套件，也不需要 ARK_AUDIO_* 配置）。要测真实客户端行为：
    providers 层直接构造 app.providers.ArkAudioClient，或把
    app.pipeline.ArkAudioClient 恢复为 app.providers.ArkAudioClient。"""
    import json as _json

    from app.providers import ProviderResult

    class _StubArkAudioClient:
        def __init__(self, settings: object, transport: object = None) -> None:
            pass

        def analyze(self, audio_path: Path, prompt: str) -> object:
            return ProviderResult(
                value=_json.dumps({
                    "transcript": "原音转写",
                    "speech_rate_desc": "中等，约5字每秒",
                    "intonation_emotion": "热情上扬",
                    "pause_pattern": "句尾停顿",
                    "overall_tone": "带货激情",
                    "gender_age": "青年女性",
                }, ensure_ascii=False),
                record={"outcome": "stubbed", "phase": "ark-audio-analysis"},
                evidence_media=[audio_path],
            )

    monkeypatch.setattr("app.pipeline.ArkAudioClient", _StubArkAudioClient)


@pytest.fixture(scope="session")
def replay_settings() -> Settings:
    return Settings(
        access_token="node-replay",
        multiref_mode="composed",
        public_base_url="https://node-replay.test",
        enable_mediakit=False,
        provider_poll_interval_s=2.0,
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        seedream_api_key=os.environ.get("ARK_API_KEY", ""),
        store_lock_enabled=False,
    )


@pytest.fixture(scope="session")
def deepseek_client(replay_settings: Settings) -> DeepSeekClient:
    if not replay_settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY is not set (deepseek.env missing)")
    return DeepSeekClient(replay_settings)


@pytest.fixture(scope="session")
def seedream_client(replay_settings: Settings) -> SeedreamClient:
    if not replay_settings.seedream_api_key:
        pytest.skip("ARK_API_KEY is not set (seedream credential missing)")
    return SeedreamClient(replay_settings)


@pytest.fixture(scope="session")
def pipeline_instance(
    replay_settings: Settings, tmp_path_factory: pytest.TempPathFactory,
) -> Pipeline:
    """A real Pipeline (real voice catalog, real clients) on a scratch store."""
    root = tmp_path_factory.mktemp("node-replay-store")
    store = ProjectStore(root, acquire_process_lock=False)
    return Pipeline(
        replay_settings,
        store,
        lambda _project_id, _path: "https://node-replay.test/asset",
    )
