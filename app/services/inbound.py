"""Persistence for ingested emails and their drafts (the durable review queue)."""

from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.models import InboundEmail
from app.models.enums import InboundStatus
from app.schemas.email import GmailMessage


class InboundEmailService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def known_message_ids(self, user_email: str, message_ids: Sequence[str]) -> set[str]:
        if not message_ids:
            return set()
        result = await self._session.scalars(
            select(InboundEmail.gmail_message_id).where(
                InboundEmail.user_email == user_email, InboundEmail.gmail_message_id.in_(message_ids)
            )
        )
        return set(result)

    async def ingest(self, user_email: str, message: GmailMessage, auto_send: bool) -> str | None:
        """Insert the email; returns its new id, or None if it was already ingested."""
        stmt = (
            pg_insert(InboundEmail)
            .values(
                user_email=user_email,
                gmail_message_id=message.id,
                thread_id=message.thread_id,
                sender=message.sender,
                subject=message.subject,
                body=message.body,
                message_id_header=message.message_id_header,
                references=message.references,
                auto_send=auto_send,
                status=InboundStatus.QUEUED.value,
            )
            .on_conflict_do_nothing(constraint="uq_inbound_email_user_message")
            .returning(InboundEmail.id)
        )
        email_id = await self._session.scalar(stmt)
        await self._session.commit()
        return email_id

    async def get(self, email_id: str) -> InboundEmail | None:
        return await self._session.get(InboundEmail, email_id)

    async def get_for_user(self, email_id: str, user_email: str) -> InboundEmail | None:
        return await self._session.scalar(
            select(InboundEmail).where(InboundEmail.id == email_id, InboundEmail.user_email == user_email)
        )

    async def list_by_status(
        self, user_email: str, statuses: Sequence[InboundStatus], *, newest_first: bool = False, limit: int | None = None
    ) -> list[InboundEmail]:
        order = InboundEmail.updated_at.desc() if newest_first else InboundEmail.created_at.asc()
        stmt = (
            select(InboundEmail)
            .where(InboundEmail.user_email == user_email, InboundEmail.status.in_([s.value for s in statuses]))
            .order_by(order)
            .limit(limit)
        )
        return list(await self._session.scalars(stmt))

    async def claim(
        self, email_id: str, *, expected: InboundStatus, new: InboundStatus, user_email: str | None = None
    ) -> InboundEmail | None:
        """Atomically move an email from `expected` to `new` status.

        Returns the row only for the caller that won the transition. This guards against double
        sends (double-clicked approve, a redelivered task racing an approval, ...).
        """
        stmt = (
            update(InboundEmail)
            .where(InboundEmail.id == email_id, InboundEmail.status == expected.value)
            .values(status=new.value, updated_at=utcnow())
            .returning(InboundEmail)
        )
        if user_email is not None:
            stmt = stmt.where(InboundEmail.user_email == user_email)
        row = (await self._session.scalars(stmt)).one_or_none()
        await self._session.commit()
        return row

    async def set_status(self, row: InboundEmail, status: InboundStatus, error: str | None = None) -> None:
        row.status = status.value
        if error is not None:
            row.error = error
        await self._session.commit()

    async def mark_failed(self, email_id: str, error: str) -> None:
        """Mark as failed unless it already reached a final or reviewable state."""
        await self._session.execute(
            update(InboundEmail)
            .where(
                InboundEmail.id == email_id,
                InboundEmail.status.in_([InboundStatus.QUEUED.value, InboundStatus.DRAFTED.value]),
            )
            .values(status=InboundStatus.FAILED.value, error=error[:2000], updated_at=utcnow())
        )
        await self._session.commit()
