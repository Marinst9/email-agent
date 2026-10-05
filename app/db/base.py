from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC timestamp — the existing schema uses `timestamp without time zone`."""
    return datetime.now(UTC).replace(tzinfo=None)
