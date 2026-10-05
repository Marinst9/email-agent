"""Sending (or discarding) drafted replies. Used by the worker (auto mode) and the review dashboard."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import InboundEmail
from app.models.enums import EmailStatus, InboundStatus
from app.schemas.email import EmailLogCreate
from app.services.email_log import EmailLogService
from app.services.gmail import GmailClient
from app.services.inbound import InboundEmailService

logger = logging.getLogger(__name__)


def to_log_entry(row: InboundEmail, status: EmailStatus, response: str) -> EmailLogCreate:
    return EmailLogCreate(
        user_email=row.user_email,
        sender=row.sender,
        subject=row.subject,
        response=response,
        source=row.source or "",
        status=status,
        category=row.category or "",
        confidence=row.confidence,
        reasoning=row.reasoning,
        docs_used=row.docs_used or [],
        priority=row.priority,
        sentiment=row.sentiment,
    )


class EmailDeliveryService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._inbound = InboundEmailService(session)
        self._log = EmailLogService(session)

    async def approve(self, gmail: GmailClient, user_email: str, email_id: str, custom_response: str | None) -> bool:
        row = await self._inbound.claim(
            email_id, expected=InboundStatus.AWAITING_REVIEW, new=InboundStatus.SENT, user_email=user_email
        )
        if row is None:
            return False
        try:
            await self._send(gmail, row, custom_response or row.response or "")
        except Exception:
            await self._inbound.set_status(row, InboundStatus.AWAITING_REVIEW)
            raise
        return True

    async def reject(self, gmail: GmailClient, user_email: str, email_id: str) -> bool:
        row = await self._inbound.claim(
            email_id, expected=InboundStatus.AWAITING_REVIEW, new=InboundStatus.REJECTED, user_email=user_email
        )
        if row is None:
            return False
        await gmail.mark_as_read(row.gmail_message_id)
        await self._log.record(to_log_entry(row, EmailStatus.REJECTED, row.response or ""))
        return True

    async def auto_send(self, gmail: GmailClient, row: InboundEmail) -> InboundStatus:
        """Send a drafted reply without review. On a send failure the draft falls back to human review."""
        claimed = await self._inbound.claim(row.id, expected=InboundStatus.DRAFTED, new=InboundStatus.SENT)
        if claimed is None:
            return InboundStatus(row.status)
        try:
            await self._send(gmail, claimed, claimed.response or "")
        except Exception as exc:
            logger.exception("Auto-send failed for %s; moving to review", row.id)
            await self._inbound.set_status(claimed, InboundStatus.AWAITING_REVIEW, error=f"Auto-send failed: {exc}")
            return InboundStatus.AWAITING_REVIEW
        return InboundStatus.SENT

    async def _send(self, gmail: GmailClient, row: InboundEmail, response_text: str) -> None:
        # Only a failure of the send itself propagates (and un-claims the email). Once the reply is out,
        # bookkeeping failures must not make the email look unsent and invite a duplicate.
        await gmail.send_reply(row.sender, row.subject, response_text)
        try:
            row.response = response_text
            await self._session.commit()
            await gmail.mark_as_read(row.gmail_message_id)
            await self._log.record(to_log_entry(row, EmailStatus.SENT, response_text))
            await self._log.remember_thread(
                row.user_email, row.thread_id, row.sender, row.subject, row.body, response_text
            )
        except Exception:
            logger.exception("Reply to %s was sent but post-send bookkeeping failed", row.id)
