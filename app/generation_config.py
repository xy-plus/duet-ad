import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


DEFAULTS = {
    "optimize_image": True,
    "remove_subtitle": False,
    "remove_watermark": False,
}
FIELDS = frozenset(DEFAULTS)
RECEIPT_FILENAME = "generation-config.json"


class GenerationConfigError(ValueError):
    pass


def parse_form(value: str, *, provided: bool) -> dict[str, bool]:
    if not provided:
        return dict(DEFAULTS)
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        raise GenerationConfigError("invalid_generation_config") from None
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise GenerationConfigError("invalid_generation_config")
    if any(not isinstance(raw[key], bool) for key in FIELDS):
        raise GenerationConfigError("invalid_generation_config")
    return {key: raw[key] for key in DEFAULTS}


def sha256(config: Mapping[str, bool]) -> str:
    payload = json.dumps(
        dict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def receipt(config: Mapping[str, bool]) -> dict:
    frozen = {key: config[key] for key in DEFAULTS}
    return {
        "version": 1,
        "generation_config": frozen,
        "generation_config_sha256": sha256(frozen),
        "postprocess_options": {
            "optimize_image": frozen["optimize_image"],
            "remove_subtitle": frozen["remove_subtitle"],
            "remove_brand": frozen["remove_watermark"],
        },
    }


def resolve(cdir: Path, meta: Mapping) -> dict[str, bool] | None:
    """Return the frozen config, with legacy records mapped to old defaults."""
    raw = meta.get("generation_config")
    digest = meta.get("generation_config_sha256")
    if raw is None and digest is None:
        if (cdir / "work" / RECEIPT_FILENAME).exists():
            return None
        return dict(DEFAULTS)
    if (
        not isinstance(raw, dict)
        or set(raw) != FIELDS
        or any(not isinstance(raw[key], bool) for key in FIELDS)
        or digest != sha256(raw)
    ):
        return None
    expected = receipt(raw)
    try:
        stored = json.loads(
            (cdir / "work" / RECEIPT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return dict(raw) if stored == expected else None


def is_frozen(cdir: Path, meta: Mapping) -> bool:
    return (
        "generation_config" in meta
        or "generation_config_sha256" in meta
        or (cdir / "work" / RECEIPT_FILENAME).exists()
    )


def postprocess_options(config: Mapping[str, bool]) -> dict[str, bool]:
    return receipt(config)["postprocess_options"]


def capability() -> dict:
    return {
        "supported": True,
        "create_field": "generation_config",
        "encoding": "multipart_json",
        "fields": {key: "boolean" for key in DEFAULTS},
        "defaults": dict(DEFAULTS),
    }
