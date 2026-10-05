from datetime import datetime
from typing import Any

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow
from app.models.enums import UserRole


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    name: Mapped[str | None] = mapped_column(String(100))
    picture: Mapped[str | None] = mapped_column(String(500))
    role: Mapped[str] = mapped_column(String(20), default=UserRole.EMPLOYEE.value)
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime | None] = mapped_column(default=utcnow)
    last_login: Mapped[datetime | None]
    # Google OAuth token (access + refresh). Kept server-side instead of in the session cookie.
    google_token: Mapped[dict[str, Any] | None] = mapped_column(JSON)
