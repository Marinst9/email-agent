"""Audit log of processed emails, per-thread conversation memory, and AI feedback."""

import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AIFeedback, EmailLog, KnowledgeDocument, ThreadMemory
from app.models.enums import TEMPLATE_SOURCE_PREFIX, EmailSource, EmailStatus, FeedbackValue
from app.schemas.email import EmailLogCreate, EmailLogRead, FeedbackForm, ThreadTurn
from app.schemas.stats import PlatformSummary, UserStats

THREAD_HISTORY_LIMIT = 5
THREAD_MEMORY_MAX_CHARS = 500


class EmailLogService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(self, entry: EmailLogCreate) -> int:
        log = EmailLog(
            **entry.model_dump(mode="json", exclude={"docs_used"}),
            docs_used=json.dumps(entry.docs_used) if entry.docs_used else None,
        )
        self._session.add(log)
        await self._session.commit()
        return log.id

    async def history(self, user_email: str) -> list[EmailLogRead]:
        result = await self._session.scalars(
            select(EmailLog).where(EmailLog.user_email == user_email).order_by(EmailLog.timestamp.desc())
        )
        return [EmailLogRead.model_validate(log) for log in result]

    async def remember_thread(
        self, user_email: str, thread_id: str, sender: str, subject: str, body: str, response: str
    ) -> None:
        self._session.add(
            ThreadMemory(
                user_email=user_email,
                thread_id=thread_id,
                sender=sender,
                subject=subject,
                body=body[:THREAD_MEMORY_MAX_CHARS],
                response=response[:THREAD_MEMORY_MAX_CHARS],
            )
        )
        await self._session.commit()

    async def thread_history(self, user_email: str, thread_id: str) -> list[ThreadTurn]:
        """Most recent turns first."""
        result = await self._session.scalars(
            select(ThreadMemory)
            .where(ThreadMemory.user_email == user_email, ThreadMemory.thread_id == thread_id)
            .order_by(ThreadMemory.timestamp.desc())
            .limit(THREAD_HISTORY_LIMIT)
        )
        return [ThreadTurn.model_validate(m) for m in result]

    async def submit_feedback(self, user_email: str, log_id: int, form: FeedbackForm) -> bool:
        """Returns False when the log entry does not exist or belongs to another user."""
        log = await self._session.scalar(
            select(EmailLog).where(EmailLog.id == log_id, EmailLog.user_email == user_email)
        )
        if log is None:
            return False
        self._session.add(
            AIFeedback(
                user_email=user_email,
                email_log_id=log_id,
                feedback=form.feedback.value,
                original_response=log.response,
                corrected_response=form.corrected_response,
            )
        )
        await self._session.commit()
        return True


class AnalyticsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def user_stats(self, user_email: str) -> UserStats:
        mine = EmailLog.user_email == user_email
        totals = (
            await self._session.execute(
                select(
                    func.count(),
                    func.count().filter(EmailLog.status == EmailStatus.SENT.value),
                    func.count().filter(EmailLog.status == EmailStatus.REJECTED.value),
                    func.count().filter(EmailLog.status == EmailStatus.IGNORED.value),
                    func.count().filter(EmailLog.source == EmailSource.MULTI_AGENT.value),
                    func.count().filter(EmailLog.source.contains(TEMPLATE_SOURCE_PREFIX)),
                    func.avg(EmailLog.confidence).filter(EmailLog.confidence != 0),
                ).where(mine)
            )
        ).one()
        total, sent, rejected, ignored, ai_generated, from_template, avg_confidence = totals

        category = func.coalesce(func.nullif(EmailLog.category, ""), "UNKNOWN")
        categories = {
            str(name): int(count)
            for name, count in await self._session.execute(
                select(category, func.count()).where(mine).group_by(category)
            )
        }

        feedback_counts = {
            str(value): int(count)
            for value, count in await self._session.execute(
                select(AIFeedback.feedback, func.count())
                .where(AIFeedback.user_email == user_email)
                .group_by(AIFeedback.feedback)
            )
        }

        return UserStats(
            total=total,
            isprateni=sent,
            odbieni=rejected,
            ignorirani=ignored,
            ai_gen=ai_generated,
            shablon=from_template,
            avg_confidence=round(float(avg_confidence), 2) if avg_confidence is not None else 0.0,
            categories=categories,
            positive_feedback=feedback_counts.get(FeedbackValue.POSITIVE.value, 0),
            negative_feedback=feedback_counts.get(FeedbackValue.NEGATIVE.value, 0),
        )

    async def platform_summary(self) -> PlatformSummary:
        async def count(model: type[EmailLog] | type[KnowledgeDocument] | type[AIFeedback]) -> int:
            return int(await self._session.scalar(select(func.count()).select_from(model)) or 0)

        return PlatformSummary(
            total_logs=await count(EmailLog),
            total_docs=await count(KnowledgeDocument),
            feedbacks=await count(AIFeedback),
        )
