"""Durable, versioned dialogue-review contract for newly created projects."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable

from app import voice

AUTO_CONTINUE = "auto_continue"
REVIEW_REQUIRED = "review_required"
POLICIES = frozenset({AUTO_CONTINUE, REVIEW_REQUIRED})
OUTCOMES = frozenset({"recognized", "no_audio", "no_vocal", "vocal_unrecognized"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DialogueReviewError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def capability() -> dict:
    return {
        "supported": True,
        "create_field": "dialogue_review_policy",
        "policies": [AUTO_CONTINUE, REVIEW_REQUIRED],
        "default": AUTO_CONTINUE,
        "commit_path": "/api/conversations/{id}/dialogue-review/commit",
    }


def parse_create_policy(value: str, *, provided: bool) -> str:
    policy = value.strip() if provided else AUTO_CONTINUE
    if policy not in POLICIES:
        raise DialogueReviewError("invalid_dialogue_review_policy")
    return policy


def canonical_lines(value: Iterable[dict]) -> list[dict]:
    return [
        {
            "text": line["text"],
            "start_s": float(line["start_s"]),
            "end_s": float(line["end_s"]),
        }
        for line in value
    ]


def effective_machine_lines(provenance: object) -> list[dict]:
    if not isinstance(provenance, list):
        return []
    return canonical_lines(
        line
        for line in provenance
        if isinstance(line, dict)
        and line.get("kept") is True
        and line.get("classification") == "spoken"
        and not voice.is_unrecognized_text(line.get("text"))
    )


def lines_sha256(lines: Iterable[dict]) -> str:
    payload = json.dumps(
        canonical_lines(lines),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def commit_payload_sha256(
    *, expected_revision: int, expected_sha256: str, lines: Iterable[dict]
) -> str:
    payload = json.dumps(
        {
            "expected_revision": expected_revision,
            "expected_sha256": expected_sha256,
            "lines": canonical_lines(lines),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analysis_state(policy: str, outcome: str, machine_lines: Iterable[dict]) -> dict:
    if policy not in POLICIES or outcome not in OUTCOMES:
        raise DialogueReviewError("invalid_dialogue_review_state")
    lines = canonical_lines(machine_lines)
    digest = lines_sha256(lines)
    waiting = policy == REVIEW_REQUIRED
    return {
        "version": 1,
        "policy": policy,
        "status": "waiting" if waiting else "frozen",
        "outcome": outcome,
        "revision": 1,
        "machine_lines": lines,
        "machine_sha256": digest,
        "lines": lines,
        "sha256": digest,
        "frozen_by": None if waiting else "automatic",
    }


def public_state(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    required = {
        "version", "policy", "status", "outcome", "revision",
        "machine_lines", "machine_sha256", "lines", "sha256", "frozen_by",
    }
    if not required.issubset(value):
        return None
    try:
        machine_lines = canonical_lines(value["machine_lines"])
        lines = canonical_lines(value["lines"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        value["version"] != 1
        or value["policy"] not in POLICIES
        or value["status"] not in {"waiting", "frozen"}
        or value["outcome"] not in OUTCOMES
        or value["frozen_by"] not in {None, "automatic", "user"}
        or not isinstance(value["revision"], int)
        or isinstance(value["revision"], bool)
        or value["revision"] < 1
        or not isinstance(value["machine_sha256"], str)
        or not _SHA256_RE.fullmatch(value["machine_sha256"])
        or not isinstance(value["sha256"], str)
        or not _SHA256_RE.fullmatch(value["sha256"])
        or lines_sha256(machine_lines) != value["machine_sha256"]
        or lines_sha256(lines) != value["sha256"]
        or (
            value["status"] == "waiting"
            and (
                value["policy"] != REVIEW_REQUIRED
                or value["revision"] != 1
                or value["frozen_by"] is not None
            )
        )
        or (
            value["status"] == "frozen"
            and (
                (value["frozen_by"] == "automatic" and (
                    value["policy"] != AUTO_CONTINUE or value["revision"] != 1
                ))
                or (value["frozen_by"] == "user" and (
                    value["policy"] != REVIEW_REQUIRED or value["revision"] != 2
                ))
                or value["frozen_by"] not in {"automatic", "user"}
            )
        )
    ):
        return None
    return {
        "version": 1,
        "policy": value["policy"],
        "status": value["status"],
        "outcome": value["outcome"],
        "revision": value["revision"],
        "machine_lines": machine_lines,
        "machine_sha256": value["machine_sha256"],
        "lines": lines,
        "sha256": value["sha256"],
        "frozen_by": value["frozen_by"],
        "editable": value["status"] == "waiting",
    }
