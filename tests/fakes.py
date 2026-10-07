"""In-memory fakes for the pipeline's collaborators: Gmail API, Anthropic, DB-backed services, Redis limiter."""

import base64
import email
from collections.abc import Sequence
from datetime import UTC, datetime
from email import policy
from email.message import EmailMessage
from types import SimpleNamespace
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock, Usage

from app.core.config import Settings
from app.models.enums import InboundStatus
from app.models.inbound import InboundEmail
from app.schemas.agent import Classification, RetrievedDoc
from app.schemas.email import EmailLogCreate, ThreadTurn
from app.schemas.rules import ReplyTemplateRead
from app.services.ai_agents import CLASSIFICATION_PROMPT
from app.services.delivery import EmailDeliveryService
from app.services.gmail import GmailClient
from app.services.pipeline import EmailPipeline

# --- Gmail API (googleapiclient-shaped: service.users().messages().send(...).execute()) ---------


class _Call:
    def __init__(self, result: Any = None) -> None:
        self._result = result

    def execute(self) -> Any:
        return self._result


class _Messages:
    def __init__(self, service: "FakeGmailService") -> None:
        self._service = service

    def send(self, userId: str, body: dict[str, Any]) -> _Call:  # noqa: N803  (Google API argument name)
        self._service.sent.append(body)
        return _Call({"id": f"sent-{len(self._service.sent)}"})

    def modify(self, userId: str, id: str, body: dict[str, Any]) -> _Call:  # noqa: N803
        self._service.modified.append((id, body))
        return _Call({})

    def get(self, userId: str, id: str, format: str) -> _Call:  # noqa: N803
        return _Call(self._service.stored[id])


class _Labels:
    def __init__(self, service: "FakeGmailService") -> None:
        self._service = service

    def list(self, userId: str) -> _Call:  # noqa: N803
        return _Call({"labels": list(self._service.label_list)})

    def create(self, userId: str, body: dict[str, Any]) -> _Call:  # noqa: N803
        label = {"id": f"Label_{len(self._service.label_list) + 1}", "name": body["name"]}
        self._service.label_list.append(label)
        return _Call(label)


class FakeGmailService:
    def __init__(self, stored: dict[str, dict[str, Any]] | None = None) -> None:
        self.stored = stored or {}
        self.sent: list[dict[str, Any]] = []
        self.modified: list[tuple[str, dict[str, Any]]] = []
        self.label_list: list[dict[str, str]] = [{"id": "INBOX", "name": "INBOX"}, {"id": "UNREAD", "name": "UNREAD"}]

    def users(self) -> "FakeGmailService":
        return self

    def messages(self) -> _Messages:
        return _Messages(self)

    def labels(self) -> _Labels:
        return _Labels(self)

    def sent_mime(self, index: int = -1) -> EmailMessage:
        raw = base64.urlsafe_b64decode(self.sent[index]["raw"])
        message = email.message_from_bytes(raw, policy=policy.default)
        assert isinstance(message, EmailMessage)
        return message


def gmail_client(service: FakeGmailService) -> GmailClient:
    return GmailClient(service)


def mime_text(message: EmailMessage) -> str:
    content = message.get_content()
    assert isinstance(content, str)
    return content


# --- Anthropic --------------------------------------------------------------------------------


class FakeAnthropicMessages:
    """Answers the classification prompt with `classification`, everything else with `draft_text`."""

    def __init__(self, classification: Classification, draft_text: str) -> None:
        self.classification = classification
        self.draft_text = draft_text
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        is_classification = kwargs.get("system") == CLASSIFICATION_PROMPT
        text = self.classification.model_dump_json() if is_classification else self.draft_text
        return Message.model_construct(
            id=f"msg_{len(self.calls)}",
            type="message",
            role="assistant",
            model=kwargs["model"],
            content=[TextBlock(type="text", text=text)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=100, output_tokens=20),
        )


def fake_anthropic(
    classification: Classification, draft_text: str = ""
) -> tuple[AsyncAnthropic, FakeAnthropicMessages]:
    messages = FakeAnthropicMessages(classification, draft_text)
    return cast(AsyncAnthropic, SimpleNamespace(messages=messages)), messages


# --- DB-backed services -----------------------------------------------------------------------


class FakeSession:
    async def commit(self) -> None:
        return None


class FakeInbound:
    def __init__(self, *rows: InboundEmail) -> None:
        self.rows = {row.id: row for row in rows}

    async def get(self, email_id: str) -> InboundEmail | None:
        return self.rows.get(email_id)

    async def get_for_user(self, email_id: str, user_email: str) -> InboundEmail | None:
        row = self.rows.get(email_id)
        return row if row is not None and row.user_email == user_email else None

    async def claim(
        self, email_id: str, *, expected: InboundStatus, new: InboundStatus, user_email: str | None = None
    ) -> InboundEmail | None:
        row = self.rows.get(email_id)
        if row is None or row.status != expected.value or (user_email is not None and row.user_email != user_email):
            return None
        row.status = new.value
        return row

    async def set_status(self, row: InboundEmail, status: InboundStatus, error: str | None = None) -> None:
        row.status = status.value
        if error is not None:
            row.error = error


class FakeLog:
    def __init__(self) -> None:
        self.records: list[EmailLogCreate] = []
        self.threads: list[tuple[str, str]] = []

    async def record(self, entry: EmailLogCreate) -> int:
        self.records.append(entry)
        return len(self.records)

    async def remember_thread(
        self, user_email: str, thread_id: str, sender: str, subject: str, body: str, response: str
    ) -> None:
        self.threads.append((thread_id, response))

    async def thread_history(self, user_email: str, thread_id: str) -> list[ThreadTurn]:
        return []


class FakeBlocked:
    def __init__(self, blocked: bool = False) -> None:
        self.blocked = blocked

    async def is_blocked(self, user_email: str, sender: str) -> bool:
        return self.blocked


class FakeTemplates:
    def __init__(self, templates: Sequence[ReplyTemplateRead] = ()) -> None:
        self.templates = list(templates)

    async def list_for_user(self, user_email: str) -> list[ReplyTemplateRead]:
        return self.templates


class FakeRateLimiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    async def allow(self, user_email: str, sender: str, email_id: str, now: float | None = None) -> bool:
        return self.allowed


class FakeKnowledge:
    def __init__(self, docs: Sequence[RetrievedDoc] = ()) -> None:
        self.docs = list(docs)
        self.queries: list[tuple[str, int]] = []

    async def search(self, user_email: str, query: str, limit: int = 3) -> list[RetrievedDoc]:
        self.queries.append((query, limit))
        return self.docs[:limit]


# --- Factories --------------------------------------------------------------------------------


def make_row(**overrides: Any) -> InboundEmail:
    now = datetime.now(UTC).replace(tzinfo=None)
    values: dict[str, Any] = {
        "id": "e1",
        "user_email": "me@firma.mk",
        "gmail_message_id": "g1",
        "thread_id": "t1",
        "sender": "Ana <ana@klient.mk>",
        "subject": "Прашање за понуда",
        "body": "Здраво, ве молам испратете ми понуда.",
        "message_id_header": "<CAB123@mail.gmail.com>",
        "references": "<first@klient.mk>",
        "auto_send": True,
        "status": InboundStatus.QUEUED.value,
        "needs_review": False,
        "action": "ОДГОВОР",
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return InboundEmail(**values)


def make_delivery(inbound: FakeInbound, log: FakeLog) -> EmailDeliveryService:
    delivery = EmailDeliveryService(cast(Any, FakeSession()))
    delivery._inbound = cast(Any, inbound)
    delivery._log = cast(Any, log)
    return delivery


class PipelineHarness:
    def __init__(
        self,
        row: InboundEmail,
        classification: Classification | None = None,
        draft_text: str = "",
        templates: Sequence[ReplyTemplateRead] = (),
        blocked: bool = False,
        rate_allowed: bool = True,
    ) -> None:
        self.row = row
        self.gmail_service = FakeGmailService()
        self.inbound = FakeInbound(row)
        self.log = FakeLog()
        self.ai_client, self.ai = fake_anthropic(classification or Classification(priority="LOW"), draft_text)
        settings = cast(
            Settings,
            SimpleNamespace(
                anthropic_model="claude-sonnet-4-6",
                gmail_ignored_label="AI-Ignored",
                retrieval_candidates=20,
                retrieval_top_k=3,
                rerank_enabled=False,
                rerank_model="claude-haiku-4-5",
                rerank_candidates=20,
            ),
        )

        async def no_stage(stage: object) -> None:
            return None

        pipeline = EmailPipeline(
            cast(Any, FakeSession()),
            settings,
            self.ai_client,
            cast(Any, FakeRateLimiter(rate_allowed)),
            on_stage=no_stage,
        )
        pipeline._inbound = cast(Any, self.inbound)
        pipeline._log = cast(Any, self.log)
        pipeline._blocked = cast(Any, FakeBlocked(blocked))
        pipeline._templates = cast(Any, FakeTemplates(templates))
        pipeline._delivery = make_delivery(self.inbound, self.log)
        pipeline._orchestrator._retriever._searcher = FakeKnowledge()
        pipeline._gmail = gmail_client(self.gmail_service)
        self.pipeline = pipeline

    async def run(self) -> InboundStatus:
        return await self.pipeline.run(self.row.id)
