"""Per-user Gmail pollers running in the web process.

The poller only *ingests*: it stores each new unread email and queues
`process_incoming_email_task` for it. All classification, retrieval and drafting happens in the
Celery worker. Deduplication is enforced by the database, so a restarted poller never re-queues.

The running/auto-mode flags are still per process, so run a single web process.
Moving the poller to Celery beat would remove that constraint.
"""

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.gmail import GmailClient
from app.services.inbound import InboundEmailService
from app.worker.dispatch import EmailTaskDispatcher

logger = logging.getLogger(__name__)


@dataclass
class UserAgentState:
    running: bool = False
    auto_mode: bool = False
    task: asyncio.Task[None] | None = None


@dataclass(frozen=True)
class AgentStatus:
    running: bool
    auto_mode: bool


class AgentManager:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        dispatcher: EmailTaskDispatcher,
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._dispatcher = dispatcher
        self._states: dict[str, UserAgentState] = {}

    def status(self, user_email: str) -> AgentStatus:
        state = self._states.get(user_email) or UserAgentState()
        return AgentStatus(running=state.running, auto_mode=state.auto_mode)

    async def start(self, user_email: str, token: Mapping[str, Any], auto_mode: bool) -> None:
        # Restarting replaces the existing loop instead of spawning a duplicate.
        await self.stop(user_email)
        state = self._states.setdefault(user_email, UserAgentState())
        state.auto_mode = auto_mode
        state.running = True
        state.task = asyncio.create_task(self._run(user_email, dict(token)), name=f"inbox-poller:{user_email}")

    async def stop(self, user_email: str) -> None:
        state = self._states.get(user_email)
        if state is None:
            return
        state.running = False
        task, state.task = state.task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def shutdown(self) -> None:
        await asyncio.gather(*(self.stop(email) for email in list(self._states)))

    async def _run(self, user_email: str, token: dict[str, Any]) -> None:
        state = self._states[user_email]
        gmail: GmailClient | None = None
        logger.info("Inbox poller started for %s (auto_mode=%s)", user_email, state.auto_mode)

        while state.running:
            try:
                if gmail is None:
                    gmail = await GmailClient.from_token(token, self._settings)
                await self._poll_once(user_email, state, gmail)
                await asyncio.sleep(self._settings.agent_poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Inbox poller error for %s", user_email)
                await asyncio.sleep(self._settings.agent_error_backoff_seconds)

        logger.info("Inbox poller stopped for %s", user_email)

    async def _poll_once(self, user_email: str, state: UserAgentState, gmail: GmailClient) -> None:
        message_ids = await gmail.list_unread_ids(self._settings.gmail_unread_query)
        if not message_ids:
            return

        async with self._sessionmaker() as session:
            known = await InboundEmailService(session).known_message_ids(user_email, message_ids)

        for message_id in message_ids:
            if not state.running:
                return
            if message_id in known:
                continue
            message = await gmail.get_message(message_id)

            async with self._sessionmaker() as session:
                inbound = InboundEmailService(session)
                email_id = await inbound.ingest(user_email, message, auto_send=state.auto_mode)
                if email_id is None:
                    continue  # ingested concurrently by another poller
                try:
                    await self._dispatcher.enqueue(email_id)
                except Exception as exc:
                    logger.exception("Could not queue email %s", email_id)
                    await inbound.mark_failed(email_id, f"Could not queue for processing: {exc}")
