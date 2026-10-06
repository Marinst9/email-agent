"""Per-user reply templates and blocked-sender rules."""

from collections.abc import Sequence

from sqlalchemy import delete, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BlockedSender, UserTemplate
from app.schemas.rules import ReplyTemplateCreate, ReplyTemplateRead

DEFAULT_BLOCKED_WORDS: tuple[str, ...] = (
    "noreply", "no-reply", "donotreply", "newsletter", "notifications",
    "notification", "mailer", "automated", "bounce", "railway",
)

# Keywords are matched as substrings of subject + body, so keep them specific. Generic words
# ("help", "request", "urgent") matched far too many emails that needed a real answer.
DEFAULT_TEMPLATES: tuple[ReplyTemplateCreate, ...] = (
    ReplyTemplateCreate(
        name="Консултации",
        keywords="консултации,consultation,meeting,состанок",
        response="Здраво, благодарам за пораката. Слободен/на сум за консултации во следните термини: понеделник и среда 10-12ч.",
    ),
    ReplyTemplateCreate(
        name="Потврда за прием",
        keywords="апликација,application",
        response="Здраво, Ви потврдувам дека Вашето барање е примено. Ќе Ви одговориме во рок од 2 работни дена.",
    ),
)


def find_matching_template(
    subject: str, body: str, templates: Sequence[ReplyTemplateRead]
) -> ReplyTemplateRead | None:
    text = f"{subject} {body}".lower()
    for template in templates:
        if any(keyword.lower() in text for keyword in template.keywords):
            return template
    return None


def is_blocked_sender(sender: str, blocked_words: Sequence[str]) -> bool:
    sender_lower = sender.lower()
    return any(word in sender_lower for word in blocked_words)


class ReplyTemplateService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_user(self, user_email: str) -> list[ReplyTemplateRead]:
        result = await self._session.scalars(
            select(UserTemplate).where(UserTemplate.user_email == user_email).order_by(UserTemplate.id)
        )
        return [ReplyTemplateRead.model_validate(t) for t in result]

    async def create(self, user_email: str, data: ReplyTemplateCreate) -> None:
        self._session.add(UserTemplate(user_email=user_email, **data.model_dump()))
        await self._session.commit()

    async def delete(self, user_email: str, template_id: int) -> None:
        # Scoped to the owner: users can only delete their own templates.
        await self._session.execute(
            delete(UserTemplate).where(UserTemplate.id == template_id, UserTemplate.user_email == user_email)
        )
        await self._session.commit()

    async def seed_defaults(self, user_email: str) -> None:
        if await self._session.scalar(select(exists().where(UserTemplate.user_email == user_email))):
            return
        self._session.add_all(UserTemplate(user_email=user_email, **t.model_dump()) for t in DEFAULT_TEMPLATES)
        await self._session.commit()


class BlockedSenderService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_words(self, user_email: str) -> list[str]:
        result = await self._session.scalars(
            select(BlockedSender.word).where(BlockedSender.user_email == user_email).order_by(BlockedSender.id)
        )
        return [word for word in result if word]

    async def add(self, user_email: str, word: str) -> None:
        already = await self._session.scalar(
            select(exists().where(BlockedSender.user_email == user_email, BlockedSender.word == word))
        )
        if not already:
            self._session.add(BlockedSender(user_email=user_email, word=word))
            await self._session.commit()

    async def remove(self, user_email: str, word: str) -> None:
        await self._session.execute(
            delete(BlockedSender).where(BlockedSender.user_email == user_email, BlockedSender.word == word)
        )
        await self._session.commit()

    async def is_blocked(self, user_email: str, sender: str) -> bool:
        return is_blocked_sender(sender, await self.list_words(user_email))

    async def seed_defaults(self, user_email: str) -> None:
        if await self._session.scalar(select(exists().where(BlockedSender.user_email == user_email))):
            return
        self._session.add_all(BlockedSender(user_email=user_email, word=w) for w in DEFAULT_BLOCKED_WORDS)
        await self._session.commit()
