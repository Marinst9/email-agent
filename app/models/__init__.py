"""ORM models. Importing this package registers every table on `Base.metadata` (used by Alembic)."""

from app.models.email import AIFeedback, EmailLog, ThreadMemory
from app.models.inbound import InboundEmail
from app.models.knowledge import KnowledgeDocument
from app.models.rules import BlockedSender, UserTemplate
from app.models.user import User

__all__ = [
    "AIFeedback",
    "BlockedSender",
    "EmailLog",
    "InboundEmail",
    "KnowledgeDocument",
    "ThreadMemory",
    "User",
    "UserTemplate",
]
