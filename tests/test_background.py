"""Tests for the background pipeline's retry policy, Redis task state, and Redis rate limiter."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import anthropic
import fakeredis
import httpx
import pytest

from app.models.enums import InboundStatus
from app.models.inbound import InboundEmail
from app.schemas.tasks import TaskStage, TaskState
from app.services.rate_limit import RedisReplyRateLimiter
from app.services.task_state import TaskStateStore, build_processing_status
from app.worker.retry import backoff_seconds, is_retryable, retry_after_seconds, retry_countdown

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(code: int, headers: dict[str, str] | None = None) -> anthropic.APIStatusError:
    response = httpx.Response(code, headers=headers or {}, request=_REQUEST)
    return anthropic.APIStatusError(f"HTTP {code}", response=response, body=None)


# --- Retry policy --------------------------------------------------------------


@pytest.mark.parametrize(("code", "expected"), [(429, True), (503, True), (529, True), (500, False), (400, False)])
def test_retryable_status_codes(code: int, expected: bool) -> None:
    assert is_retryable(_status_error(code)) is expected


def test_connection_errors_are_retryable_but_others_are_not() -> None:
    assert is_retryable(anthropic.APIConnectionError(request=_REQUEST))
    assert is_retryable(anthropic.APITimeoutError(request=_REQUEST))
    assert not is_retryable(ValueError("bug"))


def test_backoff_grows_exponentially_and_is_capped() -> None:
    def low() -> float:
        return 0.0

    def high() -> float:
        return 1.0

    assert [backoff_seconds(a, base=2, cap=300, rand=high) for a in range(4)] == [2, 4, 8, 16]
    assert [backoff_seconds(a, base=2, cap=300, rand=low) for a in range(4)] == [1, 2, 4, 8]
    assert backoff_seconds(20, base=2, cap=300, rand=high) == 300


def test_retry_after_header_is_honoured_up_to_hard_cap() -> None:
    def high() -> float:
        return 1.0

    assert retry_after_seconds(_status_error(429, {"retry-after": "30"})) == 30
    assert retry_after_seconds(_status_error(429, {"retry-after-ms": "1500"})) == 1.5
    assert retry_after_seconds(_status_error(429)) is None

    exc = _status_error(429, {"retry-after": "30"})
    assert retry_countdown(exc, attempt=0, base=2, cap=300, hard_cap=1800, rand=high) == 30
    assert retry_countdown(exc, attempt=5, base=2, cap=300, hard_cap=1800, rand=high) == 64
    assert retry_countdown(_status_error(429, {"retry-after": "7200"}), 0, 2, 300, 1800, rand=high) == 1800


# --- Redis task state ------------------------------------------------------------


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


async def test_task_state_lifecycle(redis: fakeredis.FakeAsyncRedis) -> None:
    store = TaskStateStore(redis, ttl_seconds=60)
    assert await store.get("e1") is None

    await store.mark_pending("e1")
    record = await store.get("e1")
    assert record is not None and record.state is TaskState.PENDING and record.stage is TaskStage.QUEUED

    await store.mark_running("e1", attempt=0)
    await store.set_stage("e1", TaskStage.DRAFTING)
    await store.mark_retrying("e1", attempts=1, error="RateLimitError (HTTP 429)", retry_in_seconds=4)
    record = await store.get("e1")
    assert record is not None
    assert (record.state, record.stage, record.attempts) == (TaskState.RETRYING, TaskStage.DRAFTING, 1)
    assert record.error and record.next_retry_at is not None

    await store.mark_running("e1", attempt=1)
    record = await store.get("e1")
    assert record is not None and record.state is TaskState.RETRYING and record.next_retry_at is None

    await store.mark_success("e1", attempts=2)
    record = await store.get("e1")
    assert record is not None
    assert (record.state, record.stage, record.attempts, record.error) == (TaskState.SUCCESS, TaskStage.DONE, 2, None)
    assert 0 < await redis.ttl(TaskStateStore.key("e1")) <= 60


def _row(status: InboundStatus, response: str | None = None) -> InboundEmail:
    now = datetime.now(UTC).replace(tzinfo=None)
    return InboundEmail(
        id="e1", user_email="me@x", gmail_message_id="g1", thread_id="t", sender="a@b", subject="s", body="b",
        auto_send=False, status=status.value, response=response, needs_review=False, docs_used=None,
        created_at=now, updated_at=now,
    )


def test_processing_status_falls_back_to_database_when_redis_record_missing() -> None:
    status = build_processing_status(_row(InboundStatus.AWAITING_REVIEW, response="Hello"), None)
    assert status.state is TaskState.SUCCESS
    assert status.draft is not None and status.draft.response == "Hello" and status.draft.docs_used == []

    assert build_processing_status(_row(InboundStatus.QUEUED), None).state is TaskState.PENDING
    assert build_processing_status(_row(InboundStatus.FAILED), None).state is TaskState.FAILED
    assert build_processing_status(_row(InboundStatus.QUEUED), None).draft is None


# --- Redis rate limiter ----------------------------------------------------------


async def test_rate_limiter_is_per_user_and_sender_and_idempotent(redis: fakeredis.FakeAsyncRedis) -> None:
    limiter = RedisReplyRateLimiter(redis, limit=2, window_seconds=3600)
    assert await limiter.allow("me", "a@x", "e1", now=0)
    assert await limiter.allow("me", "a@x", "e1", now=1)  # retry of the same email: no extra slot
    assert await limiter.allow("me", "a@x", "e2", now=2)
    assert not await limiter.allow("me", "a@x", "e3", now=3)
    assert await limiter.allow("other", "a@x", "e4", now=3)
    assert await limiter.allow("me", "a@x", "e5", now=3601)  # window slid past e1
