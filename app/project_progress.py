"""Pure project-level progress projection for conversation APIs."""

from __future__ import annotations

from collections.abc import Mapping


PROGRESS_FLOOR_FIELD = "_project_progress_floor"


def aggregate_project_progress(
    meta: Mapping[str, object], *, has_video: bool
) -> dict[str, int | str]:
    """Return the complete public progress object without mutating ``meta``.

    ``has_video`` is the caller's authoritative, readability-checked final
    artifact fact.  Internal receipts only estimate work completed; they can
    never independently publish success or 100 percent.
    """
    if has_video is True:
        return {"percent": 100, "status": "succeeded"}

    percent = _derived_percent(meta)
    if _is_minimal_v1(meta):
        percent = max(percent, _progress_floor(meta.get(PROGRESS_FLOOR_FIELD)))
    percent = min(max(percent, 0), 99)

    generation = _as_mapping(meta.get("generation"))
    output_missing = (
        generation is not None and generation.get("status") == "succeeded"
    )
    if _has_failed(meta) or output_missing:
        return {
            "percent": max(percent, 99 if output_missing else 0),
            "status": "failed",
        }
    if percent == 0:
        return {"percent": 0, "status": "queued"}
    return {"percent": percent, "status": "running"}


def _derived_percent(meta: Mapping[str, object]) -> int:
    """Coarsely project the existing 439c4b5 receipts onto one percentage."""
    percent = 0
    analysis_status = meta.get("status")
    if analysis_status == "processing" or analysis_status == "failed":
        percent = 10
    elif analysis_status == "done":
        percent = 35

    postprocess = _as_mapping(meta.get("postprocess"))
    if postprocess is not None:
        postprocess_status = postprocess.get("status")
        if postprocess_status == "done":
            percent = max(percent, 60)
        else:
            percent = max(percent, 45)
            percent = max(
                percent,
                _completed_item_progress(
                    postprocess.get("segments"),
                    completed_status="done",
                    start=45,
                    span=14,
                ),
            )

    fusion = _as_mapping(meta.get("_prompt_fusion"))
    if fusion is not None:
        fusion_status = fusion.get("status")
        if fusion_status == "done":
            fusion_progress = 75
        elif fusion_status == "running" or fusion_status == "failed":
            fusion_progress = 70
        else:
            fusion_progress = 65
        percent = max(percent, fusion_progress)

    generation = _as_mapping(meta.get("generation"))
    if generation is not None:
        generation_status = generation.get("status")
        percent = max(percent, 80 if generation_status == "queued" else 85)
        percent = max(
            percent,
            _completed_item_progress(
                generation.get("segments"),
                completed_status="succeeded",
                start=85,
                span=13,
            ),
        )
        if generation.get("stage") == "stitch" or generation_status == "succeeded":
            percent = max(percent, 99)

    return percent


def _completed_item_progress(
    value: object, *, completed_status: str, start: int, span: int
) -> int:
    if not isinstance(value, list) or not value:
        return start
    completed = sum(
        1
        for item in value
        if isinstance(item, Mapping) and item.get("status") == completed_status
    )
    return start + (span * completed // len(value))


def _has_failed(meta: Mapping[str, object]) -> bool:
    if meta.get("status") == "failed":
        return True
    for field in ("postprocess", "_prompt_fusion"):
        state = _as_mapping(meta.get(field))
        if state is not None and state.get("status") == "failed":
            return True
    generation = _as_mapping(meta.get("generation"))
    return generation is not None and generation.get("status") in (
        "failed",
        "submission_unknown",
        "resume_required",
    )


def _is_minimal_v1(meta: Mapping[str, object]) -> bool:
    effective_request = _as_mapping(meta.get("effective_request"))
    input_receipt = _as_mapping(meta.get("input_receipt"))
    if effective_request is None or input_receipt is None:
        return False
    request_version = effective_request.get("version")
    receipt_version = input_receipt.get("version")
    return bool(
        isinstance(request_version, int)
        and not isinstance(request_version, bool)
        and request_version == 1
        and isinstance(receipt_version, int)
        and not isinstance(receipt_version, bool)
        and receipt_version == 1
    )


def _progress_floor(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(max(value, 0), 99)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None
