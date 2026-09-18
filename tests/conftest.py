"""Shared fixtures for the real-input node replay suite.

Loads provider credentials from the same env files the deploy script sources
(values are never printed), registers the ``live_provider`` marker and gates
it behind ``NODE_REPLAY_LIVE`` (default on; set to 0 to skip paid calls).
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
    return os.environ.get("NODE_REPLAY_LIVE", "1").lower() not in {
        "0", "false", "no", "off",
    }


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_provider: real paid provider call (Seedream image);"
        " disable with NODE_REPLAY_LIVE=0",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    if _live_enabled():
        return
    skip = pytest.mark.skip(
        reason="live paid provider calls disabled via NODE_REPLAY_LIVE=0"
    )
    for item in items:
        if "live_provider" in item.keywords:
            item.add_marker(skip)


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
    """A real Pipeline (real voice library, real clients) on a scratch store."""
    root = tmp_path_factory.mktemp("node-replay-store")
    store = ProjectStore(root, acquire_process_lock=False)
    return Pipeline(
        replay_settings,
        store,
        lambda _project_id, _path: "https://node-replay.test/asset",
    )
