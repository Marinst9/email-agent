"""Multi-agent pipeline: classify -> retrieve (RAG) -> draft -> review."""

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from pydantic import ValidationError

from app.schemas.agent import (
    Classification,
    DraftAction,
    DraftResult,
    OrchestrationResult,
    RetrievedDoc,
    ReviewDecision,
)
from app.schemas.email import GmailMessage, ThreadTurn
from app.schemas.tasks import TaskStage

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """Категоризирај го мејлот. Врати JSON во овој формат:
{
  "category": "INQUIRY/COMPLAINT/URGENT_HUMAN/SPAM",
  "priority": "HIGH/MEDIUM/LOW",
  "language": "mk/en/other",
  "sentiment": "positive/neutral/negative"
}
Врати САМО JSON, ништо друго."""

DRAFT_PROMPT_TEMPLATE = """Ти си AI агент за мејлови. Одговарај на {language} јазик.
Тонот прилагоди го според sentiment: {sentiment}.
Ако имаш информации од документи, користи ги за попрецизен одговор.
ПРАВИЛА:
- Автоматска нотификација -> АКЦИЈА: ИГНОРИРАЈ
- Реална личност -> АКЦИЈА: ОДГОВОР
- Треба препраќање -> АКЦИЈА: ПРЕПРАЌАЊЕ
Формат:
АКЦИЈА: ОДГОВОР или ПРЕПРАЌАЊЕ или ИГНОРИРАЈ
АКО ПРЕПРАЌАЊЕ ДО: (email или НИКОЈ)
ПОРАКА: (текст на одговорот)"""

MESSAGE_MARKER = "ПОРАКА:"
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class DocumentSearcher(Protocol):
    async def search(self, user_email: str, query: str, limit: int = 3) -> list[str]: ...


StageCallback = Callable[[TaskStage], Awaitable[None]]


async def _no_progress(stage: TaskStage) -> None:
    return None


def _first_text(content: Sequence[object]) -> str:
    return next((block.text for block in content if isinstance(block, TextBlock)), "")


def parse_classification(raw: str) -> Classification:
    try:
        return Classification.model_validate_json(_JSON_FENCE.sub("", raw.strip()))
    except ValidationError:
        logger.warning("Could not parse classification, using defaults: %r", raw[:200])
        return Classification()


def parse_draft_action(text: str) -> DraftAction:
    if "АКЦИЈА: ОДГОВОР" in text:
        return DraftAction.REPLY
    if "АКЦИЈА: ПРЕПРАЌАЊЕ" in text:
        return DraftAction.FORWARD
    return DraftAction.IGNORE


def draft_confidence(retrieved_docs: Sequence[RetrievedDoc], has_history: bool) -> float:
    confidence = 0.75 + len(retrieved_docs) * 0.05 + (0.05 if has_history else 0.0)
    return min(round(confidence, 2), 0.99)


class ClassificationAgent:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def execute(self, email: GmailMessage) -> Classification:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=150,
            system=CLASSIFICATION_PROMPT,
            messages=[
                {"role": "user", "content": f"Од: {email.sender}\nНаслов: {email.subject}\nСодржина: {email.body[:300]}"}
            ],
        )
        return parse_classification(_first_text(response.content))


class RetrievalAgent:
    def __init__(self, searcher: DocumentSearcher) -> None:
        self._searcher = searcher

    async def execute(self, user_email: str, query: str) -> list[RetrievedDoc]:
        docs = await self._searcher.search(user_email, query, limit=3)
        return [
            RetrievedDoc(content=doc, index=i, similarity=round(0.95 - i * 0.08, 2)) for i, doc in enumerate(docs)
        ]


class DraftAgent:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def execute(
        self,
        email: GmailMessage,
        classification: Classification,
        retrieved_docs: Sequence[RetrievedDoc],
        thread_history: Sequence[ThreadTurn],
    ) -> DraftResult:
        rag_context = ""
        if retrieved_docs:
            rag_context = "\nРЕЛЕВАНТНИ ИНФОРМАЦИИ:\n" + "".join(f"- {d.content}\n" for d in retrieved_docs)

        history_text = ""
        if thread_history:
            # History arrives newest-first; present it chronologically.
            history_text = "\nПРЕТХОДНИ ПОРАКИ:\n" + "".join(
                f"- Примено: {turn.body[:100]}\n- Одговорено: {turn.response[:100]}\n\n"
                for turn in reversed(thread_history)
            )

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=DRAFT_PROMPT_TEMPLATE.format(language=classification.language, sentiment=classification.sentiment),
            messages=[
                {
                    "role": "user",
                    "content": f"{rag_context}{history_text}Од: {email.sender}\nНаслов: {email.subject}\nСодржина: {email.body}",
                }
            ],
        )
        text = _first_text(response.content)

        return DraftResult(
            raw=text,
            action=parse_draft_action(text),
            response_text=text.split(MESSAGE_MARKER)[-1].strip() if MESSAGE_MARKER in text else "",
            confidence=draft_confidence(retrieved_docs, bool(thread_history)),
            docs_used=[d.content[:100] for d in retrieved_docs],
            reasoning=(
                f"Одговорот е генериран врз основа на {len(retrieved_docs)} документи "
                f"и категоријата {classification.category}."
            ),
        )


class ReviewAgent:
    """Rule-based gate deciding whether a draft may be sent without human review."""

    def execute(self, draft: DraftResult, classification: Classification) -> ReviewDecision:
        reason = ""
        if classification.category in ("COMPLAINT", "URGENT_HUMAN"):
            reason = f"Категорија: {classification.category} бара човечка интервенција"
        elif classification.priority == "HIGH":
            reason = "Висок приоритет — препорачана човечка проверка"
        elif draft.confidence < 0.80:
            reason = f"Низок confidence ({draft.confidence}) — препорачана проверка"
        needs_review = bool(reason)
        return ReviewDecision(needs_review=needs_review, reason=reason, auto_send=not needs_review)


class EmailOrchestrator:
    def __init__(self, client: AsyncAnthropic, model: str, searcher: DocumentSearcher) -> None:
        self._classifier = ClassificationAgent(client, model)
        self._retriever = RetrievalAgent(searcher)
        self._drafter = DraftAgent(client, model)
        self._reviewer = ReviewAgent()

    async def process(
        self,
        email: GmailMessage,
        user_email: str,
        thread_history: Sequence[ThreadTurn],
        on_stage: StageCallback = _no_progress,
    ) -> OrchestrationResult:
        logger.info("Orchestrator: processing email from %s", email.sender)

        await on_stage(TaskStage.CLASSIFYING)
        classification = await self._classifier.execute(email)
        logger.debug("Classification: %s", classification)

        if classification.category == "SPAM":
            return OrchestrationResult(
                action=DraftAction.IGNORE,
                classification=classification,
                retrieved_docs=[],
                draft=None,
                review=ReviewDecision(needs_review=False, auto_send=False),
                confidence=0.99,
                reasoning="Мејлот е класифициран како SPAM",
            )

        await on_stage(TaskStage.RETRIEVING)
        retrieved_docs = await self._retriever.execute(user_email, f"{email.subject} {email.body[:200]}")
        await on_stage(TaskStage.DRAFTING)
        draft = await self._drafter.execute(email, classification, retrieved_docs, thread_history)
        await on_stage(TaskStage.REVIEWING)
        review = self._reviewer.execute(draft, classification)
        logger.info(
            "Orchestrator: action=%s confidence=%s needs_review=%s",
            draft.action, draft.confidence, review.needs_review,
        )

        return OrchestrationResult(
            action=draft.action,
            classification=classification,
            retrieved_docs=retrieved_docs,
            draft=draft,
            review=review,
            confidence=draft.confidence,
            reasoning=draft.reasoning,
        )
