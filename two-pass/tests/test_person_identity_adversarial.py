"""Independent, model-free guards for cross-track identity association."""
import itertools
import math

import numpy as np

from app.person_identity import group_observations


def track(label, start, vector, *, scene=None, x=0):
    feature = np.asarray(vector, dtype=float)
    feature = feature / np.linalg.norm(feature)
    return [dict(
        observation_id=f"{label}-{i}", source_time_s=start + i * .25,
        scene_id=scene or label, bbox=[x, 0., 80., 80.], score=.95,
        kind="source", segment_index=None, frame_order=i + 1,
        feature=feature,
    ) for i in range(3)]


def assignment(result):
    return {o["observation_id"]: o["canonical_id"] for o in result["observations"]}


def test_four_identical_nonoverlapping_tracks_are_not_competing_people():
    rows = sum((track(str(i), i * 3, (1, 0, 0)) for i in range(4)), [])
    result = group_observations(rows)
    assert len(result["persons"]) == 1
    assert len(result["persons"][0]["intervals"]) == 4
    assert len(result["persons"][0]["observation_ids"]) == 12


def test_weak_bridge_does_not_merge_incompatible_endpoints():
    rows = (track("a", 0, (1, 0)) + track("b", 3, (1, 1))
            + track("c", 6, (0, 1)))
    result = assignment(group_observations(rows))
    assert result["a-0"] != result["c-0"]


def test_cooccurrence_cannot_be_bypassed_through_a_third_track():
    rows = (track("a", 0, (1, 0), scene="same", x=0)
            + track("b", 3, (1, 0))
            + track("c", 0, (1, 0), scene="same", x=300))
    result = assignment(group_observations(rows))
    assert result["a-0"] != result["c-0"]


def test_below_threshold_outside_candidate_still_blocks_close_margin():
    rows = (track("a", 0, (1, 0, 0))
            + track("b", 3, (.38, math.sqrt(1 - .38**2), 0))
            + track("c", 6, (.35, 0, math.sqrt(1 - .35**2))))
    assert len(group_observations(rows)["persons"]) == 3


def test_two_distinct_identity_groups_are_not_collapsed():
    second = (.34, math.sqrt(1 - .34**2))
    rows = (track("a1", 0, (1, 0)) + track("b1", 3, second)
            + track("a2", 6, (1, 0)) + track("b2", 9, second))
    result = assignment(group_observations(rows))
    assert result["a1-0"] == result["a2-0"]
    assert result["b1-0"] == result["b2-0"]
    assert result["a1-0"] != result["b1-0"]


def test_similar_cooccurring_people_stay_separate_after_reappearance():
    second = (.8, .6)
    rows = (track("a1", 0, (1, 0), scene="same", x=0)
            + track("b1", 0, second, scene="same", x=300)
            + track("a2", 6, (1, 0)) + track("b2", 9, second))
    result = assignment(group_observations(rows))
    # Conservative refusal to join reappearances is allowed; cross-person
    # association is not, even when all cosine scores pass the threshold.
    assert {result["a1-0"], result["a2-0"]}.isdisjoint(
        {result["b1-0"], result["b2-0"]})


def test_track_input_permutations_preserve_assignments_and_intervals():
    tracks = [track(str(i), i * 3, (1, 0, 0)) for i in range(4)]
    expected = group_observations(sum(tracks, []))
    for order in itertools.permutations(tracks):
        actual = group_observations(sum((list(reversed(t)) for t in order), []))
        assert assignment(actual) == assignment(expected)
        assert actual["persons"] == expected["persons"]
