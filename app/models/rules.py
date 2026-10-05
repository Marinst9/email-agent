from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UserTemplate(Base):
    __tablename__ = "user_template"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String(100), index=True)
    name: Mapped[str | None] = mapped_column(String(100))
    keywords: Mapped[str | None] = mapped_column(String(500))  # comma-separated
    response: Mapped[str | None] = mapped_column(Text)


class BlockedSender(Base):
    __tablename__ = "blocked_sender"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String(100), index=True)
    word: Mapped[str | None] = mapped_column(String(100))
