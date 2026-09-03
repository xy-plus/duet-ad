"""Shared cardinality contract for ordered per-segment keyframes."""

from __future__ import annotations


KEYFRAMES_PER_SEGMENT = 3
LEGACY_KEYFRAMES_PER_SEGMENT = 9
SAMPLING_RECEIPT_VERSION = 3
LEGACY_SAMPLING_RECEIPT_VERSION = 2
SUPPORTED_KEYFRAME_COUNTS = frozenset({
    KEYFRAMES_PER_SEGMENT,
    LEGACY_KEYFRAMES_PER_SEGMENT,
})

FRAME_NAMES = tuple(
    f"{order:02d}.png" for order in range(1, KEYFRAMES_PER_SEGMENT + 1)
)
LEGACY_FRAME_NAMES = tuple(
    f"{order:02d}.png" for order in range(1, LEGACY_KEYFRAMES_PER_SEGMENT + 1)
)


def frame_names(count: int) -> tuple[str, ...]:
    """Return the only supported ordered filename contract."""
    if isinstance(count, bool) or count not in SUPPORTED_KEYFRAME_COUNTS:
        raise ValueError("unsupported per-segment keyframe count")
    return tuple(f"{order:02d}.png" for order in range(1, count + 1))
