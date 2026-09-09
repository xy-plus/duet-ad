"""CPU face evidence before visual agents; never infer identity from clothing.

Intervals describe sampled face observations, not continuous body visibility.
Embeddings stay in memory. Unresolved observations are not different-person facts.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import face_mask

SCHEMA_VERSION = "1.0"
MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/face_recognition_sface_2021dec.onnx"
PARAMETERS = {"sample_gap_s": .251, "track_gap_s": .501,
              "local_cosine": .20, "identity_cosine": .363,
              "ambiguity_margin": .05, "quality_score": .75,
              "minimum_face_px": 32, "minimum_template_frames": 3}


def model_fingerprints() -> dict[str, Any]:
    model = Path(os.environ.get("PERSON_IDENTITY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    return {"algorithm": "yunet-sface-compatible-track-groups-v2", "opencv": cv2.__version__,
            "recognizer_sha256": _sha(model), "detector_sha256": _sha(face_mask.MODEL_PATH),
            "detection_threshold": face_mask.SCORE_THRESHOLD, "parameters": dict(PARAMETERS)}


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    except OSError:
        return None


identity_configuration_fingerprint = model_fingerprints


def _cos(first, second) -> float:
    return float(np.dot(first, second))


def _near(first: dict, second: dict) -> bool:
    a, b = first["bbox"], second["bbox"]
    distance = np.hypot(a[0]+a[2]/2-b[0]-b[2]/2, a[1]+a[3]/2-b[1]-b[3]/2)
    scale = max(a[2], a[3], b[2], b[3])
    return distance <= scale * .8 and min(a[2]*a[3], b[2]*b[3]) / max(a[2]*a[3], b[2]*b[3]) >= .35


def _quality(obs: dict) -> bool:
    return obs["score"] >= PARAMETERS["quality_score"] and min(obs["bbox"][2:]) >= PARAMETERS["minimum_face_px"] and obs.get("feature") is not None


def _template(track: list[dict]) -> list[dict]:
    # Distinct source timestamps prevent repeated keyframes fabricating support.
    unique = {}
    for obs in sorted(track, key=lambda o: (-o["score"], o["observation_id"])):
        if _quality(obs):
            unique.setdefault(obs["source_time_s"], obs)
    samples = list(unique.values())[:7]
    if len(samples) < PARAMETERS["minimum_template_frames"]:
        return []
    # Select a medoid and its two closest high-quality views. A single poor
    # pose does not veto a stable track, and a single lucky pair cannot merge it.
    similarities = np.array([[_cos(a["feature"], b["feature"]) for b in samples] for a in samples])
    medoid = int(np.argmax(np.median(similarities, axis=1)))
    order = np.argsort(-similarities[medoid], kind="stable")
    return [samples[int(i)] for i in order[:3]]


def _match(first: list[dict], second: list[dict]) -> float:
    a, b = _template(first), _template(second)
    if not a or not b:
        return -1.
    return float(np.median([_cos(x["feature"], y["feature"]) for x in a for y in b]))


def _conflict(first: list[dict], second: list[dict]) -> bool:
    return any(a["source_time_s"] == b["source_time_s"] for a in first for b in second)


def _merge_compatible_tracks(tracks: list[list[dict]]) -> list[list[dict]]:
    """Merge whole compatible candidate groups, never a chain of weak links.

    Every original track retains its multi-frame template for pairwise checks.
    Multiple matching fragments can be one identity, so only candidates outside
    the group compete for the ambiguity margin. Conflicting/incomplete groups
    keep their local tracks; this is not evidence of distinct people.
    """
    supported = [track for track in tracks if _template(track)]
    count = len(supported)
    scores = np.full((count, count), -1.)
    neighbours: list[set[int]] = [set() for _ in supported]
    for i in range(count):
        for j in range(i+1, count):
            scores[i, j] = scores[j, i] = _match(supported[i], supported[j])
            # Include visually plausible co-occurring rivals in the candidate
            # group so cannot-link vetoes indirect merges through a third track.
            if scores[i, j] >= PARAMETERS["identity_cosine"]:
                neighbours[i].add(j)
                neighbours[j].add(i)
    remaining = set(range(count))
    merged = []
    while remaining:
        component, pending = set(), [min(remaining)]
        while pending:
            index = pending.pop()
            if index not in component:
                component.add(index)
                pending.extend(sorted(neighbours[index]-component))
        remaining.difference_update(component)
        indices = sorted(component)
        compatible = len(indices) > 1 and all(
            scores[i, j] >= PARAMETERS["identity_cosine"]
            and not _conflict(supported[i], supported[j])
            for position, i in enumerate(indices) for j in indices[position+1:]
        )
        if compatible:
            outsiders = set(range(count))-component
            # Do not hide a runner-up just below the absolute identity cutoff.
            # Example: .38 inside versus .35 outside remains ambiguous at .05.
            compatible = all(
                max(scores[i, j] for j in indices if j != i)-scores[i, rival]
                >= PARAMETERS["ambiguity_margin"]
                for i in indices for rival in outsiders
            )
        if compatible:
            merged.append([obs for i in indices for obs in supported[i]])
        else:
            merged.extend(supported[i] for i in indices)
    return merged


def group_observations(observations: list[dict]) -> dict[str, list]:
    """Pure association rules; inputs contain normalized in-memory features."""
    tracks: list[list[dict]] = []
    source = [o for o in observations if o.get("kind") != "keyframe"]
    keyframes = [o for o in observations if o.get("kind") == "keyframe"]
    # Without dense source evidence sparse keyframes remain conservative tracks.
    primary = source if source else keyframes
    for obs in sorted(primary, key=lambda o: (o["source_time_s"], o["observation_id"])):
        if not _quality(obs):
            continue
        candidates = []
        for index, track in enumerate(tracks):
            previous = track[-1]
            gap = obs["source_time_s"] - previous["source_time_s"]
            if (obs["scene_id"] == previous["scene_id"] and 0 < gap <= PARAMETERS["track_gap_s"]
                    and _near(obs, previous)):
                similarity = _cos(obs["feature"], previous["feature"])
                candidates.append((similarity, index))
        candidates.sort(reverse=True)
        if candidates and candidates[0][0] >= PARAMETERS["local_cosine"] and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= PARAMETERS["ambiguity_margin"]):
            tracks[candidates[0][1]].append(obs)
        else:
            tracks.append([obs])

    supported = _merge_compatible_tracks(tracks)
    if source:
        # Independently detected exact keyframes map by feature AND nearby box,
        # never by frame order/time alone. A frame permits one face per identity.
        assigned_frames: set[tuple] = set()
        for obs in sorted(keyframes, key=lambda o: o["observation_id"]):
            if not _quality(obs):
                continue
            choices = []
            for index, track in enumerate(supported):
                frame_key = (obs.get("segment_index"), obs["frame_order"], index)
                if frame_key in assigned_frames:
                    continue
                nearby = [o for o in track if o["scene_id"] == obs["scene_id"]
                          and abs(o["source_time_s"]-obs["source_time_s"]) <= PARAMETERS["track_gap_s"]
                          and _near(obs, o)]
                if nearby:
                    score = float(np.median([_cos(obs["feature"], o["feature"]) for o in nearby]))
                    choices.append((score, index))
            choices.sort(reverse=True)
            if choices and choices[0][0] >= PARAMETERS["identity_cosine"] and (len(choices) == 1 or choices[0][0]-choices[1][0] >= PARAMETERS["ambiguity_margin"]):
                index = choices[0][1]
                supported[index].append(obs)
                assigned_frames.add((obs.get("segment_index"), obs["frame_order"], index))

    persons, assignments = [], {}
    for number, track in enumerate(sorted(supported, key=lambda t: min(o["source_time_s"] for o in t)), 1):
        canonical = f"person-cv-{number:03}"
        intervals = []
        previous_scene = None
        for time, scene in sorted({(o["source_time_s"], o["scene_id"]) for o in track}, key=lambda v: v[0]):
            if intervals and scene == previous_scene and time-intervals[-1]["end_s"] <= PARAMETERS["sample_gap_s"]:
                intervals[-1]["end_s"] = time
            else:
                intervals.append({"start_s": time, "end_s": time})
            previous_scene = scene
        ids = sorted(o["observation_id"] for o in track)
        assignments.update({oid: canonical for oid in ids})
        persons.append({"canonical_id": canonical, "intervals": intervals, "observation_ids": ids})
    public, unresolved = [], []
    for obs in observations:
        row = {k: v for k, v in obs.items() if k not in {"feature", "path"}}
        row["canonical_id"] = assignments.get(obs["observation_id"])
        public.append(row)
        if row["canonical_id"] is None:
            unresolved.append({"observation_id": row["observation_id"], "reason": "insufficient_or_ambiguous_face_evidence"})
    return {"persons": persons, "observations": public, "unresolved": unresolved}


def analyze_person_identity(frames: list[dict], *, source: dict | None = None) -> dict[str, Any]:
    """frames: absolute path, source_time_s, scene_id, segment_index, frame_order,
    kind ('source' for dense tracking; 'keyframe' for exact agent images).
    No network access; missing/bad model returns an explicit skipped artifact.
    """
    result = {"schema_version": SCHEMA_VERSION, "source": source or {},
              "model_fingerprints": model_fingerprints(), "status": "skipped",
              "persons": [], "observations": [], "unresolved": []}
    model = Path(os.environ.get("PERSON_IDENTITY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model.is_absolute() or result["model_fingerprints"]["recognizer_sha256"] != MODEL_SHA256 or not face_mask.MODEL_PATH.is_file():
        return {**result, "reason": "person_identity_model_unavailable_or_invalid"}
    try:
        detector = cv2.FaceDetectorYN.create(str(face_mask.MODEL_PATH), "", (320,320), face_mask.SCORE_THRESHOLD,
                                           face_mask.NMS_THRESHOLD, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        recognizer = cv2.FaceRecognizerSF.create(str(model), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        observations = []
        for frame_number, frame in enumerate(frames):
            path = Path(frame["path"])
            if not path.is_absolute() or frame.get("scene_id") is None:
                raise ValueError("absolute frame paths and source scene IDs are required")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("person identity frame unreadable")
            detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = detector.detect(img)
            for face_number, face in enumerate([] if faces is None else faces):
                feature = recognizer.feature(recognizer.alignCrop(img, face)).reshape(-1)
                norm = float(np.linalg.norm(feature))
                obs = {k: frame.get(k) for k in ("segment_index", "frame_order", "source_time_s", "scene_id", "kind")}
                obs.update(observation_id=f"face-observation-{frame_number:05}-{face_number:02}",
                           bbox=[float(v) for v in face[:4]], score=float(face[14]),
                           feature=feature/norm if norm > 0 and np.isfinite(feature).all() else None)
                observations.append(obs)
        return {**result, **group_observations(observations), "status": "complete", "reason": None}
    except (cv2.error, ValueError, OSError) as exc:
        return {**result, "reason": "person_identity_analysis_failed", "error_type": type(exc).__name__}
