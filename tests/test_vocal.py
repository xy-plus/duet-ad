from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app import media, vocal


def _window(
    start_ms: int,
    end_ms: int,
    *,
    sung: float = 0.0,
    spoken: float = 0.0,
    music: float = 0.0,
) -> vocal.VocalWindow:
    return vocal.VocalWindow(start_ms, end_ms, sung, spoken, music)


def _scores(values: dict[int, float]) -> list[float]:
    scores = [0.0] * 521
    for index, value in values.items():
        scores[index] = value
    return scores


def test_frozen_yamnet_blob_has_calibrated_size_and_digest():
    model = Path("/home/xy/duet-ad1-h3-max-r2v/models/yamnet.tflite")

    assert model.stat().st_size == vocal.YAMNET_SIZE_BYTES == 4_126_810
    assert hashlib.sha256(model.read_bytes()).hexdigest() == vocal.YAMNET_SHA256
    assert vocal._verify_model(model) is None


def test_group_scores_uses_frozen_audioset_indices():
    result = vocal.group_scores(
        _scores(
            {
                0: 0.11,
                5: 0.42,
                12: 0.31,
                24: 0.17,
                32: 0.53,
                132: 0.29,
                261: 0.37,
                265: 0.31,
            }
        )
    )

    assert result == vocal.WindowScores(sung=0.53, spoken=0.42, music=0.37)


def test_background_music_score_and_ratio_thresholds_are_inclusive():
    assert vocal.detect_background_music(
        [_window(0, 100, music=0.1), _window(100, 200, music=0.0)]
    )
    assert not vocal.detect_background_music(
        [
            _window(0, 100, music=0.1),
            _window(100, 200, music=0.099),
            _window(200, 300, music=0.0),
        ]
    )


def test_per_asr_line_classification_uses_overlap_duration():
    sung = [
        _window(0, 500, sung=0.08, spoken=0.02),
        _window(500, 1000, sung=0.01, spoken=0.02),
    ]
    spoken = [
        _window(0, 200, spoken=0.2),
        _window(200, 1000, spoken=0.1),
    ]

    assert vocal.classify_segment(0, 1000, sung) == "sung"
    assert vocal.classify_segment(0, 1000, spoken) == "spoken"
    assert vocal.classify_segment(
        0, 1000, [_window(0, 1000, spoken=0.199)]
    ) is None


def test_asr_interval_without_acoustic_coverage_is_unclassified():
    assert vocal.classify_segment(
        2000,
        3000,
        [_window(0, 1000, spoken=0.9)],
    ) is None


def test_filter_continues_when_asr_interval_has_no_acoustic_coverage():
    analysis = vocal.VocalAnalysis(
        windows=[_window(0, 1000, spoken=0.9)],
        has_bgm=False,
    )
    lines = [
        {"text": "covered", "start_s": 0.0, "end_s": 1.0},
        {"text": "outside", "start_s": 2.0, "end_s": 3.0},
    ]

    effective, decisions = media.filter_spoken_lines(lines, analysis)

    assert effective == [{"text": "covered", "start_s": 0.0, "end_s": 1.0}]
    assert decisions[1]["classification"] is None
    assert decisions[1]["kept"] is False
    assert decisions[1]["drop_reason"] == "no_vocal_evidence"


def test_filter_spoken_lines_keeps_provenance_but_excludes_sung_and_sentinels():
    analysis = vocal.VocalAnalysis(
        windows=[
            _window(0, 1000, spoken=0.4),
            _window(1000, 2000, sung=0.08, spoken=0.01),
            _window(2000, 3000, spoken=0.4),
        ],
        has_bgm=False,
    )
    lines = [
        {"text": "spoken", "start_s": 0.0, "end_s": 1.0},
        {"text": "singing", "start_s": 1.0, "end_s": 2.0},
        {"text": "[no speech]", "start_s": 2.0, "end_s": 3.0},
    ]

    effective, decisions = media.filter_spoken_lines(lines, analysis)

    assert effective == [{"text": "spoken", "start_s": 0.0, "end_s": 1.0}]
    assert [item["classification"] for item in decisions] == [
        "spoken",
        "sung",
        None,
    ]
    assert decisions[1]["kept"] is True
    assert decisions[2]["drop_reason"] == "unrecognized_sentinel"


@pytest.mark.parametrize(
    "text",
    ["无法识别", "[inaudible]", "No speech.", "blank_audio", "（music）"],
)
def test_asr_non_dialogue_sentinels_are_filtered(text: str):
    assert media.is_unrecognized_text(text)
