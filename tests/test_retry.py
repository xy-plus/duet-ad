import pytest

from app.config import Settings
from app.retry import RetryPolicy, run_with_retry


class TemporaryError(RuntimeError):
    pass


def test_application_retry_defaults_are_two_retries_at_fifteen_seconds():
    settings = Settings(access_token="test")
    assert settings.retry_count == 2
    assert settings.retry_interval_s == 15.0


def test_retries_twice_with_fixed_delay_then_returns():
    calls = 0
    waits = []

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TemporaryError(f"temporary-{calls}")
        return "ok"

    result = run_with_retry(
        operation,
        policy=RetryPolicy(),
        is_retryable=lambda exc: isinstance(exc, TemporaryError),
        sleep=waits.append,
    )

    assert result == "ok"
    assert calls == 3
    assert waits == [15.0, 15.0]


def test_non_retryable_error_runs_once_and_preserves_exception():
    error = ValueError("permanent")
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ValueError) as caught:
        run_with_retry(
            operation,
            policy=RetryPolicy(),
            is_retryable=lambda _exc: False,
            sleep=lambda _seconds: pytest.fail("must not wait"),
        )

    assert caught.value is error
    assert calls == 1


def test_retry_exhaustion_preserves_last_exception():
    errors = [TemporaryError(str(index)) for index in range(3)]
    calls = 0

    def operation():
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    with pytest.raises(TemporaryError) as caught:
        run_with_retry(
            operation,
            policy=RetryPolicy(interval_s=0),
            is_retryable=lambda exc: isinstance(exc, TemporaryError),
        )

    assert caught.value is errors[-1]
    assert calls == 3


@pytest.mark.parametrize(
    "policy",
    [
        lambda: RetryPolicy(retries=-1),
        lambda: RetryPolicy(retries=True),
        lambda: RetryPolicy(interval_s=-1),
        lambda: RetryPolicy(interval_s=float("nan")),
    ],
)
def test_invalid_policy_fails_before_running(policy):
    with pytest.raises(ValueError):
        policy()
