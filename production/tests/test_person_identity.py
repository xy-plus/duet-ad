import itertools

import numpy as np
import pytest

from app.person_identity import analyze_person_identity, group_observations


def group(rows):
    # Add a third distinct observation to every supported two-frame fixture.
    # Singletons and two-frame rejection tests call the pure function directly.
    additions = []
    for row in rows:
        if row["kind"] == "source" and row["source_time_s"] % 1 == .25:
            additions.append({**row, "observation_id": row["observation_id"]+"-third",
                              "source_time_s": row["source_time_s"]+.25})
    return group_observations(rows+additions)


def observation(name, time, scene=1, vector=(1., 0., 0.), x=0., score=.9, kind="source", order=1):
    feature = np.array(vector)
    return dict(observation_id=name, source_time_s=time, scene_id=scene,
                bbox=[x, 0., 80., 80.], score=score, kind=kind,
                segment_index=scene, frame_order=order, feature=feature/np.linalg.norm(feature))


def test_same_person_two_scenes_and_gap_not_filled():
    result = group([observation("a",0), observation("b",.25),
                                 observation("c",4,2,(.8,.6,0)), observation("d",4.25,2,(.8,.6,0))])
    assert len(result["persons"]) == 1
    assert result["persons"][0]["intervals"] == [{"start_s":0,"end_s":.5},{"start_s":4,"end_s":4.5}]


def test_cooccurring_people_never_merge_even_identical_features():
    rows = [observation("a",0), observation("b",.25),
            observation("c",0,x=300), observation("d",.25,x=300)]
    result = group(rows)
    assert len(result["persons"]) == 2
    assert result["observations"][0]["canonical_id"] != result["observations"][2]["canonical_id"]


def test_different_face_vectors_do_not_merge_across_scenes():
    rows = [observation("a",0), observation("b",.25),
            observation("c",4,2,(.27,.96,0)), observation("d",4.25,2,(.27,.96,0))]
    assert len(group(rows)["persons"]) == 2


def test_low_quality_singleton_and_occluded_faces_remain_unresolved():
    rows = [observation("a",0,score=.74), observation("b",.25,score=.74), observation("c",3,2)]
    result = group(rows)
    assert not result["persons"]
    assert all(o["canonical_id"] is None for o in result["observations"])


def test_exact_keyframe_requires_box_and_feature_not_time_only():
    rows = [observation("a",0), observation("b",.25),
            observation("c",.1,kind="keyframe"), observation("d",.1,x=300,kind="keyframe")]
    result = group(rows)
    mapping = {o["observation_id"]:o["canonical_id"] for o in result["observations"]}
    assert mapping["c"] == mapping["a"]
    assert mapping["d"] is None


def test_two_keyframe_faces_cannot_both_map_to_one_person():
    rows = [observation("a",0), observation("b",.25),
            observation("c",.1,kind="keyframe"), observation("d",.1,x=10,kind="keyframe")]
    result = group(rows)
    assert sum(o["canonical_id"] is not None for o in result["observations"] if o["kind"] == "keyframe") == 1


def test_ambiguous_cross_scene_templates_not_arbitrarily_merged():
    rows = [observation("a",0), observation("b",.25)]
    rows += [observation("c",4,2,(.8,.6,0)), observation("d",4.25,2,(.8,.6,0))]
    rows += [observation("e",4,2,(.8,0,.6),x=300), observation("f",4.25,2,(.8,0,.6),x=300)]
    result = group(rows)
    assert len(result["persons"]) == 3


def test_missing_model_explicit_skip(monkeypatch, tmp_path):
    monkeypatch.setenv("PERSON_IDENTITY_MODEL_PATH", str(tmp_path / "missing.onnx"))
    result = analyze_person_identity([])
    assert result["status"] == "skipped"
    assert result["reason"] == "person_identity_model_unavailable_or_invalid"
    assert not result["persons"]


def test_two_frame_false_detection_is_not_confirmed_person():
    result = group_observations([observation("a",.75,score=.754), observation("b",1.,score=.771)])
    assert result["persons"] == []
    assert len(result["unresolved"]) == 2


def test_runner_below_identity_threshold_still_blocks_ambiguous_merge():
    rows = [observation("a",0), observation("b",.25)]
    rows += [observation("c",4,2,(.38,np.sqrt(1-.38**2),0)),
             observation("d",4.25,2,(.38,np.sqrt(1-.38**2),0))]
    rows += [observation("e",4,2,(.35,0,np.sqrt(1-.35**2)),x=300),
             observation("f",4.25,2,(.35,0,np.sqrt(1-.35**2)),x=300)]
    assert len(group(rows)["persons"]) == 3


def test_intervals_stop_at_scene_cut_even_for_adjacent_samples():
    rows = [observation(str(i), i*.25, 1 if i < 3 else 2) for i in range(6)]
    result = group_observations(rows)
    assert len(result["persons"]) == 1
    assert result["persons"][0]["intervals"] == [{"start_s":0.,"end_s":.5},{"start_s":.75,"end_s":1.25}]


def track_rows(index, vector=(1., 0., 0.), *, scene=None):
    return [observation(f"track-{index}-{frame}", index*4+frame*.25,
                        scene=index if scene is None else scene, vector=vector)
            for frame in range(3)]


@pytest.mark.parametrize("count", [3, 6])
def test_many_identical_noncooccurring_tracks_are_one_identity(count):
    rows = [row for index in range(count) for row in track_rows(index)]
    result = group_observations(rows)
    assert len(result["persons"]) == 1
    assert len(result["persons"][0]["observation_ids"]) == count*3
    assert len(result["persons"][0]["intervals"]) == count


def test_six_consistent_multiview_tracks_do_not_compete_with_each_other():
    rows = [row for index, angle in enumerate([0., .12, .23, .34, .45, .56])
            for row in track_rows(index, (np.cos(angle), np.sin(angle), 0.))]
    assert len(group_observations(rows)["persons"]) == 1


def test_same_scene_reappearance_keeps_one_identity_and_disjoint_intervals():
    rows = [row for index in range(3) for row in track_rows(index, scene=1)]
    result = group_observations(rows)
    assert len(result["persons"]) == 1
    assert len(result["persons"][0]["intervals"]) == 3


def test_weak_bridge_does_not_merge_incompatible_endpoints():
    # A-B and B-C exceed .363; A-C is negative. Do not infer transitivity.
    rows = [row for index, angle in enumerate([0., .9, 1.8])
            for row in track_rows(index, (np.cos(angle), np.sin(angle), 0.))]
    assert len(group_observations(rows)["persons"]) == 3


def test_two_independent_identity_groups_both_merge():
    rows = [row for index in range(6)
            for row in track_rows(index, (1.,0.,0.) if index < 3 else (0.,1.,0.))]
    result = group_observations(rows)
    assert len(result["persons"]) == 2
    assert sorted(len(person["observation_ids"]) for person in result["persons"]) == [9,9]


def test_low_similarity_outsider_does_not_block_clear_group_match():
    rows = track_rows(0)+track_rows(1, (.9,np.sqrt(1-.9**2),0.))
    rows += track_rows(2, (.35,0.,np.sqrt(1-.35**2)))
    assert len(group_observations(rows)["persons"]) == 2


def test_identity_groups_are_invariant_to_input_track_order():
    tracks = [track_rows(index) for index in range(3)]
    expected = group_observations([row for track in tracks for row in track])["persons"]
    for permutation in itertools.permutations(tracks):
        rows = [row for track in permutation for row in reversed(track)]
        assert group_observations(rows)["persons"] == expected
