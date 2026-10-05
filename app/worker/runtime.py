"""Per-process async runtime for Celery tasks.

Celery tasks are synchronous functions, while the services are async. Each worker process (and
thread, if a thread pool is used) gets one long-lived event loop plus the clients bound to it,
created lazily after fork, so DB/Redis/HTTP connection pools are reused across tasks.
"""

import asyncio
import contextlib
import logging
import os
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from celery.signals import worker_process_shutdown
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.db.session import Database
from app.services.rate_limit import RedisReplyRateLimiter
from app.services.task_state import TaskStateStore

T = TypeVar("T")
logger = logging.getLogger(__name__)


class WorkerRuntime:
    def __init__(self, settings: Settings) -> None:
        self.pid = os.getpid()
        self.settings = settings
        self.loop = asyncio.new_event_loop()
        self.database = Database(settings)
        self.redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        # Celery owns the retry policy (see app.worker.retry); the SDK must not retry on its own.
        self.ai_client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=0,
            timeout=settings.llm_request_timeout_seconds,
        )
        self.task_states = TaskStateStore(self.redis, settings.task_state_ttl_seconds)
        self.rate_limiter = RedisReplyRateLimiter(self.redis, settings.reply_limit_per_sender_per_hour)

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        task = self.loop.create_task(coro)
        try:
            return self.loop.run_until_complete(task)
        except BaseException:
            # e.g. SoftTimeLimitExceeded raised by a signal mid-await: don't leave an orphaned
            # coroutine on the loop to be resumed by the next task.
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    self.loop.run_until_complete(task)
            raise

    def close(self) -> None:
        async def _aclose() -> None:
            await self.ai_client.close()
            await self.redis.aclose()
            await self.database.dispose()

        try:
            self.loop.run_until_complete(_aclose())
        finally:
            self.loop.close()


_local = threading.local()


def get_runtime() -> WorkerRuntime:
    runtime: WorkerRuntime | None = getattr(_local, "runtime", None)
    if runtime is None or runtime.pid != os.getpid():
        runtime = WorkerRuntime(get_settings())
        _local.runtime = runtime
    return runtime


@worker_process_shutdown.connect
def _close_runtime(**_: Any) -> None:
    runtime: WorkerRuntime | None = getattr(_local, "runtime", None)
    if runtime is not None and runtime.pid == os.getpid():
        try:
            runtime.close()
        except Exception:
            logger.exception("Error while closing worker runtime")
