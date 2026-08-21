"""Small explicit retry primitive for synchronous, idempotent operations."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, TypeVar


_T = TypeVar("_T")


@dataclass(frozen=True)
class RetryPolicy:
    """Additional retries and fixed delay between attempts."""

    retries: int = 2
    interval_s: float = 15.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.retries, bool)
            or not isinstance(self.retries, int)
            or self.retries < 0
        ):
            raise ValueError("retries must be a non-negative integer")
        if (
            isinstance(self.interval_s, bool)
            or not isinstance(self.interval_s, (int, float))
            or not math.isfinite(float(self.interval_s))
            or self.interval_s < 0
        ):
            raise ValueError("interval_s must be a non-negative finite number")

    @property
    def max_attempts(self) -> int:
        return self.retries + 1


def run_with_retry(
    operation: Callable[[], _T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    on_retry: Callable[[int, Exception], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> _T:
    """Run ``operation`` and retry only exceptions approved by the caller.

    ``on_retry`` receives the one-based retry number and the triggering error.
    The final exception is re-raised unchanged.  ``BaseException`` subclasses
    such as cancellation and process interruption are never intercepted.
    """

    pause = sleep or time.sleep
    for attempt in range(policy.max_attempts):
        try:
            return operation()
        except Exception as exc:
            if attempt >= policy.retries or not is_retryable(exc):
                raise
            retry_number = attempt + 1
            if on_retry is not None:
                on_retry(retry_number, exc)
            if policy.interval_s > 0:
                pause(float(policy.interval_s))
    raise AssertionError("retry loop exhausted without a result or exception")
