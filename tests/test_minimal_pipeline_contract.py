from pathlib import Path

import pytest

from app import pipeline


def _profiles() -> dict[str, dict[str, object]]:
    return {
        "16:9": {
            "fit_required": False,
            "default_fit_mode": "none",
        },
        "9:16": {
            "fit_required": True,
            "default_fit_mode": "crop",
        },
    }


def _minimal_meta() -> dict:
    return {
        "effective_request": {
            "version": 1,
            "output": {
                "aspect_ratio": "9:16",
                "resolution": "768p",
                "fit_mode": "auto",
            },
        },
    }


def test_v1_output_overrides_short_and_long_automatic_recommendations(
    monkeypatch,
):
    profiles = _profiles()
    automatic = (profiles, "16:9", "480p", "none")
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults",
        lambda _paths, _meta: automatic,
    )
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults_from_bytes",
        lambda _frames, _meta: automatic,
    )

    short_result = pipeline._generation_defaults_for_request(
        [Path("/immutable/short-frame.png")],
        _minimal_meta(),
    )
    long_result = pipeline._generation_defaults_from_bytes_for_request(
        [b"immutable-long-frame"],
        _minimal_meta(),
    )

    expected = (profiles, "9:16", "768p", "crop")
    assert short_result == expected
    assert long_result == expected


def test_legacy_short_and_long_inputs_keep_automatic_recommendations(monkeypatch):
    profiles = _profiles()
    automatic = (profiles, "16:9", "480p", "none")
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults",
        lambda _paths, _meta: automatic,
    )
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults_from_bytes",
        lambda _frames, _meta: automatic,
    )
    legacy_meta = {"source_width": 1920, "source_height": 1080}

    assert pipeline._generation_defaults_for_request(
        [Path("/immutable/legacy-short-frame.png")],
        legacy_meta,
    ) == automatic
    assert pipeline._generation_defaults_from_bytes_for_request(
        [b"immutable-legacy-long-frame"],
        legacy_meta,
    ) == automatic


@pytest.mark.parametrize(
    "effective_request",
    [
        {"version": True, "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
        }},
        {"version": 1},
        {"version": 1, "output": None},
        {"version": 1, "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
        }},
        {"version": 1, "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "cover",
        }},
        {"version": 1, "output": {
            "aspect_ratio": "9:16",
            "resolution": "768p",
            "fit_mode": "auto",
            "unexpected": True,
        }},
        {"version": 1, "output": {
            "aspect_ratio": "1:1",
            "resolution": "768p",
            "fit_mode": "auto",
        }},
        {"version": 1, "output": {
            "aspect_ratio": "9:16",
            "resolution": "1080p",
            "fit_mode": "auto",
        }},
    ],
)
@pytest.mark.parametrize(
    ("helper", "payload"),
    [
        (pipeline._generation_defaults_for_request, [Path("/unused/frame.png")]),
        (pipeline._generation_defaults_from_bytes_for_request, [b"unused-frame"]),
    ],
)
def test_invalid_frozen_v1_output_fails_closed(
    monkeypatch,
    helper,
    payload,
    effective_request,
):
    automatic = (_profiles(), "16:9", "480p", "none")
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults",
        lambda _paths, _meta: automatic,
    )
    monkeypatch.setattr(
        pipeline,
        "_generation_defaults_from_bytes",
        lambda _frames, _meta: automatic,
    )

    with pytest.raises(pipeline.PipelineError, match="minimal output config is invalid"):
        helper(payload, {"effective_request": effective_request})
