"""Explicit per-email task state in Redis: PENDING -> (RETRYING)* -> SUCCESS | FAILED.

Stored as a hash `email-task:{email_id}` with a TTL. PostgreSQL remains the source of truth for the
email and its draft; Redis holds the execution state that the polling endpoint reports.
"""

from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis

from app.models.enums import InboundStatus
from app.models.inbound import InboundEmail
from app.schemas.tasks import DraftOut, EmailProcessingStatus, TaskStage, TaskState, TaskStatusRecord


def _now() -> datetime:
    return datetime.now(UTC)


class TaskStateStore:
    KEY_PREFIX = "email-task:"

    def __init__(self, redis: "Redis", ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def key(cls, email_id: str) -> str:
        return f"{cls.KEY_PREFIX}{email_id}"

    async def _write(self, email_id: str, **fields: str) -> None:
        fields["updated_at"] = _now().isoformat()
        key = self.key(email_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=fields)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def mark_pending(self, email_id: str) -> None:
        await self._write(
            email_id, state=TaskState.PENDING, stage=TaskStage.QUEUED, attempts="0", error="", next_retry_at=""
        )

    async def mark_running(self, email_id: str, attempt: int) -> None:
        """Called when an attempt starts. A retried attempt stays RETRYING until it finishes."""
        await self._write(
            email_id,
            state=TaskState.RETRYING if attempt > 0 else TaskState.PENDING,
            stage=TaskStage.FILTERING,
            attempts=str(attempt + 1),
            next_retry_at="",
        )

    async def set_stage(self, email_id: str, stage: TaskStage) -> None:
        await self._write(email_id, stage=stage)

    async def mark_retrying(self, email_id: str, attempts: int, error: str, retry_in_seconds: float) -> None:
        await self._write(
            email_id,
            state=TaskState.RETRYING,
            attempts=str(attempts),
            error=error,
            next_retry_at=(_now() + timedelta(seconds=retry_in_seconds)).isoformat(),
        )

    async def mark_success(self, email_id: str, attempts: int) -> None:
        await self._write(
            email_id, state=TaskState.SUCCESS, stage=TaskStage.DONE, attempts=str(attempts), error="", next_retry_at=""
        )

    async def mark_failed(self, email_id: str, attempts: int, error: str) -> None:
        await self._write(email_id, state=TaskState.FAILED, attempts=str(attempts), error=error, next_retry_at="")

    async def get(self, email_id: str) -> TaskStatusRecord | None:
        raw: dict[str, str] = await self._redis.hgetall(self.key(email_id))  # type: ignore[misc]
        if "state" not in raw:
            return None
        return TaskStatusRecord(
            state=TaskState(raw["state"]),
            stage=TaskStage(raw.get("stage") or TaskStage.QUEUED),
            attempts=int(raw.get("attempts") or 0),
            error=raw.get("error") or None,
            next_retry_at=datetime.fromisoformat(raw["next_retry_at"]) if raw.get("next_retry_at") else None,
            updated_at=datetime.fromisoformat(raw["updated_at"]),
        )


def _state_from_outcome(outcome: InboundStatus) -> TaskState:
    """Fallback when the Redis record has expired or was never written."""
    if outcome is InboundStatus.FAILED:
        return TaskState.FAILED
    if outcome in (InboundStatus.QUEUED, InboundStatus.DRAFTED):
        return TaskState.PENDING
    return TaskState.SUCCESS


def build_processing_status(row: InboundEmail, record: TaskStatusRecord | None) -> EmailProcessingStatus:
    outcome = InboundStatus(row.status)
    draft = DraftOut.model_validate(row) if row.response is not None else None
    if record is None:
        return EmailProcessingStatus(
            email_id=row.id,
            state=_state_from_outcome(outcome),
            stage=None,
            attempts=0,
            error=row.error,
            next_retry_at=None,
            updated_at=row.updated_at,
            outcome=outcome,
            draft=draft,
        )
    return EmailProcessingStatus(
        email_id=row.id,
        # The database is authoritative for a terminal failure (e.g. one recorded outside the worker).
        state=TaskState.FAILED if outcome is InboundStatus.FAILED else record.state,
        stage=record.stage,
        attempts=record.attempts,
        error=record.error or row.error,
        next_retry_at=record.next_retry_at,
        updated_at=record.updated_at,
        outcome=outcome,
        draft=draft,
    )
