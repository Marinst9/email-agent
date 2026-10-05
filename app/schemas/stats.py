from typing import Literal

from pydantic import BaseModel


class UserStats(BaseModel):
    # Field names match the variables used by templates/stats.html.
    total: int
    isprateni: int
    odbieni: int
    ignorirani: int
    ai_gen: int
    shablon: int
    avg_confidence: float
    categories: dict[str, int]
    positive_feedback: int
    negative_feedback: int


class PlatformSummary(BaseModel):
    total_logs: int
    total_docs: int
    feedbacks: int


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    redis: bool
