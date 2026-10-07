"""Web-side entry point for queueing email processing."""

import asyncio
import contextlib
import logging

from app.services.task_state import TaskStateStore
from app.worker.tasks import embed_pending_chunks_task, process_incoming_email_task

logger = logging.getLogger(__name__)


class EmailTaskDispatcher:
    def __init__(self, task_states: TaskStateStore) -> None:
        self._task_states = task_states

    async def enqueue(self, email_id: str) -> None:
        # State first, so a poll right after ingestion already sees PENDING.
        await self._task_states.mark_pending(email_id)
        # The email id doubles as the Celery task id: one task per email, easy to correlate in logs.
        # apply_async does blocking broker I/O, so keep it off the event loop.
        try:
            await asyncio.to_thread(process_incoming_email_task.apply_async, args=(email_id,), task_id=email_id)
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._task_states.mark_failed(email_id, 0, f"Could not queue for processing: {exc}")
            raise


async def enqueue_knowledge_embedding(user_email: str) -> bool:
    """Queue embedding of the user's pending chunks. False if the broker could not be reached."""
    try:
        await asyncio.to_thread(embed_pending_chunks_task.apply_async, args=(user_email,))
    except Exception:
        logger.exception("Could not queue knowledge embedding for %s", user_email)
        return False
    return True
