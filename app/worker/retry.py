"""Retry policy for transient failures of the LLM API."""

import random
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import anthropic

# 429 rate limited, 503 unavailable, 529 Anthropic "overloaded" (its flavour of 503).
RETRYABLE_STATUS_CODES = frozenset({429, 503, 529})


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    # Network failures and timeouts (APITimeoutError subclasses APIConnectionError).
    return isinstance(exc, anthropic.APIConnectionError)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-provided delay from `retry-after-ms` / `retry-after` (seconds or HTTP date)."""
    if not isinstance(exc, anthropic.APIStatusError):
        return None
    headers = exc.response.headers
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(float(ms) / 1000, 0.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        return max((parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def backoff_seconds(attempt: int, base: float, cap: float, rand: Callable[[], float] = random.random) -> float:
    """Exponential backoff with "equal jitter": half deterministic, half random.

    attempt=0 -> [base/2, base], attempt=1 -> [base, 2*base], ... capped at `cap`.
    """
    delay = min(cap, base * 2.0**attempt)
    return delay / 2 + rand() * delay / 2


def retry_countdown(
    exc: BaseException,
    attempt: int,
    base: float,
    cap: float,
    hard_cap: float,
    rand: Callable[[], float] = random.random,
) -> float:
    """Never retry sooner than the server asked; never beyond `hard_cap` (broker visibility constraint)."""
    delay = max(backoff_seconds(attempt, base, cap, rand), retry_after_seconds(exc) or 0.0)
    return min(delay, hard_cap)


def describe(exc: BaseException) -> str:
    status = f" (HTTP {exc.status_code})" if isinstance(exc, anthropic.APIStatusError) else ""
    return f"{type(exc).__name__}{status}: {exc}"[:2000]
