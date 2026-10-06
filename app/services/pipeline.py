"""The email processing pipeline executed by the Celery worker for one ingested email.

QUEUED --filter/rules/LLM--> DRAFTED --deliver--> SENT | AWAITING_REVIEW (always for forwards)
   \\--> IGNORED (automated sender, rate limited, or AI decided to ignore)

Re-running is safe: each run resumes from the persisted status, and a transient LLM failure
leaves the email in QUEUED so the retry starts the drafting step again.
"""

import logging

from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import InboundEmail
from app.models.enums import TEMPLATE_SOURCE_PREFIX, DraftAction, EmailSource, EmailStatus, InboundStatus
from app.schemas.agent import DraftResult
from app.schemas.email import EmailLogCreate, GmailMessage
from app.schemas.tasks import TaskStage
from app.services.ai_agents import EmailOrchestrator, StageCallback
from app.services.delivery import EmailDeliveryService
from app.services.email_log import EmailLogService
from app.services.gmail import GmailClient
from app.services.inbound import InboundEmailService
from app.services.knowledge import KnowledgeService
from app.services.rate_limit import RedisReplyRateLimiter
from app.services.rules import BlockedSenderService, ReplyTemplateService, find_matching_template
from app.services.users import UserService

logger = logging.getLogger(__name__)


class EmailNotFoundError(Exception):
    pass


class GmailNotAuthorizedError(Exception):
    pass


class EmailPipeline:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        ai_client: AsyncAnthropic,
        rate_limiter: RedisReplyRateLimiter,
        on_stage: StageCallback,
    ) -> None:
        self._settings = settings
        self._rate_limiter = rate_limiter
        self._on_stage = on_stage
        self._inbound = InboundEmailService(session)
        self._log = EmailLogService(session)
        self._blocked = BlockedSenderService(session)
        self._templates = ReplyTemplateService(session)
        self._users = UserService(session)
        self._delivery = EmailDeliveryService(session)
        self._orchestrator = EmailOrchestrator(ai_client, settings.anthropic_model, KnowledgeService(session))
        self._gmail: GmailClient | None = None

    async def run(self, email_id: str) -> InboundStatus:
        row = await self._inbound.get(email_id)
        if row is None:
            raise EmailNotFoundError(email_id)

        status = InboundStatus(row.status)
        if status is InboundStatus.QUEUED:
            status = await self._draft(row)
        if status is InboundStatus.DRAFTED:
            status = await self._dispatch(row)
        return status

    async def _gmail_for(self, user_email: str) -> GmailClient:
        if self._gmail is None:
            user = await self._users.get_by_email(user_email)
            if user is None or not user.google_token or "access_token" not in user.google_token:
                raise GmailNotAuthorizedError(user_email)
            self._gmail = await GmailClient.from_token(user.google_token, self._settings)
        return self._gmail

    async def _draft(self, row: InboundEmail) -> InboundStatus:
        await self._on_stage(TaskStage.FILTERING)

        if await self._blocked.is_blocked(row.user_email, row.sender):
            return await self._ignore(row, source=EmailSource.AUTOMATED, category="AUTO", reason="Blocked sender")

        if not await self._rate_limiter.allow(row.user_email, row.sender, row.id):
            return await self._ignore(row, reason="Reply rate limit reached for this sender", log=False)

        message = GmailMessage(
            id=row.gmail_message_id, thread_id=row.thread_id, sender=row.sender, subject=row.subject, body=row.body
        )
        # Transient LLM errors propagate from here; nothing has been written yet, so a retry starts clean.
        # Classification runs before template matching so a template reply to an urgent email or a
        # complaint goes through the same review rules as an AI draft.
        classification = await self._orchestrator.classify(message, on_stage=self._on_stage)

        templates = await self._templates.list_for_user(row.user_email)
        template = find_matching_template(row.subject, row.body, templates)
        if template is not None and classification.category != "SPAM":
            reasoning = f"Шаблонот '{template.name}' се совпаѓа со содржината на мејлот."
            draft = DraftResult(
                raw=template.response,
                action=DraftAction.REPLY,
                response_text=template.response,
                confidence=1.0,
                docs_used=[],
                reasoning=reasoning,
            )
            review = self._orchestrator.review(draft, classification)
            row.response = template.response
            row.action = DraftAction.REPLY.value
            row.source = f"{TEMPLATE_SOURCE_PREFIX}: {template.name}"
            row.category = classification.category
            row.priority = classification.priority
            row.sentiment = classification.sentiment
            row.confidence = draft.confidence
            row.reasoning = reasoning
            row.docs_used = []
            row.needs_review = review.needs_review
            row.review_reason = review.reason
        else:
            history = await self._log.thread_history(row.user_email, row.thread_id)
            result = await self._orchestrator.process(
                message, row.user_email, history, on_stage=self._on_stage, classification=classification
            )

            if result.action is DraftAction.IGNORE:
                return await self._ignore(
                    row, source=EmailSource.AI, category=result.classification.category, reason=result.reasoning
                )

            row.response = result.draft.response_text if result.draft else ""
            row.action = result.action.value
            row.forward_to = result.draft.forward_to if result.draft else None
            row.source = EmailSource.MULTI_AGENT.value
            row.category = result.classification.category
            row.priority = result.classification.priority
            row.sentiment = result.classification.sentiment
            row.confidence = result.confidence
            row.reasoning = result.reasoning
            row.docs_used = [d.content[:100] for d in result.retrieved_docs]
            row.needs_review = result.review.needs_review
            row.review_reason = result.review.reason

        await self._inbound.set_status(row, InboundStatus.DRAFTED)
        return InboundStatus.DRAFTED

    async def _dispatch(self, row: InboundEmail) -> InboundStatus:
        # Only plain replies may go out without a human; forwards always wait for review.
        if row.auto_send and not row.needs_review and row.action == DraftAction.REPLY.value:
            await self._on_stage(TaskStage.DELIVERING)
            return await self._delivery.auto_send(await self._gmail_for(row.user_email), row)
        await self._inbound.set_status(row, InboundStatus.AWAITING_REVIEW)
        return InboundStatus.AWAITING_REVIEW

    async def _ignore(
        self,
        row: InboundEmail,
        *,
        reason: str,
        source: EmailSource | None = None,
        category: str = "",
        log: bool = True,
    ) -> InboundStatus:
        await (await self._gmail_for(row.user_email)).mark_as_read(row.gmail_message_id)
        if log:
            await self._log.record(
                EmailLogCreate(
                    user_email=row.user_email,
                    sender=row.sender,
                    subject=row.subject,
                    response="",
                    source=source.value if source else "",
                    status=EmailStatus.IGNORED,
                    category=category,
                )
            )
        row.reasoning = reason
        row.category = category or row.category
        await self._inbound.set_status(row, InboundStatus.IGNORED)
        return InboundStatus.IGNORED
