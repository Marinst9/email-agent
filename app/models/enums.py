"""Domain enums. Values are persisted as plain strings and compared in templates, so they must not change."""

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"


class EmailStatus(StrEnum):
    SENT = "испратено"
    REJECTED = "одбиено"
    IGNORED = "игнориран"


class InboundStatus(StrEnum):
    """Business lifecycle of an ingested email (persisted in PostgreSQL)."""

    QUEUED = "queued"  # ingested, waiting for the worker
    DRAFTED = "drafted"  # draft persisted, delivery decision pending
    AWAITING_REVIEW = "awaiting_review"
    SENT = "sent"
    REJECTED = "rejected"
    IGNORED = "ignored"
    FAILED = "failed"


TERMINAL_INBOUND_STATUSES = frozenset(
    {InboundStatus.SENT, InboundStatus.REJECTED, InboundStatus.IGNORED, InboundStatus.FAILED}
)


class EmailSource(StrEnum):
    MULTI_AGENT = "Multi-Agent AI"
    AI = "AI"
    AUTOMATED = "Автоматски"


TEMPLATE_SOURCE_PREFIX = "Шаблон"


class FeedbackValue(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
