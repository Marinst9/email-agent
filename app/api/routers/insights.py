"""History, statistics and AI feedback."""

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from app.api.deps import AnalyticsServiceDep, CurrentUserDep, EmailLogServiceDep
from app.core.templating import templates
from app.schemas.email import FeedbackAck, FeedbackForm

router = APIRouter(tags=["insights"])


@router.get("/history", name="history", response_class=HTMLResponse)
async def history(request: Request, user: CurrentUserDep, logs: EmailLogServiceDep) -> HTMLResponse:
    return templates.TemplateResponse(request, "history.html", {"logs": await logs.history(user.email)})


@router.get("/stats", name="stats", response_class=HTMLResponse)
async def stats(request: Request, user: CurrentUserDep, analytics: AnalyticsServiceDep) -> HTMLResponse:
    user_stats = await analytics.user_stats(user.email)
    return templates.TemplateResponse(request, "stats.html", user_stats.model_dump())


@router.post("/feedback/{log_id}", name="feedback", response_model=FeedbackAck)
async def submit_feedback(
    log_id: int,
    form: Annotated[FeedbackForm, Form()],
    user: CurrentUserDep,
    logs: EmailLogServiceDep,
) -> FeedbackAck:
    if not await logs.submit_feedback(user.email, log_id, form):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Email log not found")
    return FeedbackAck()
