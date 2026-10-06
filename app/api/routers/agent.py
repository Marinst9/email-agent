"""Inbox poller lifecycle and the human-in-the-loop review dashboard."""

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.deps import (
    AgentManagerDep,
    CurrentUserDep,
    DeliveryServiceDep,
    GmailClientDep,
    InboundServiceDep,
    redirect_to,
)
from app.core.errors import LoginRequiredError
from app.core.templating import templates
from app.models.enums import InboundStatus
from app.schemas.agent import AgentStateView, ApproveForm, PendingEmail, ProcessedEmail, StartAgentForm
from app.schemas.user import UserRead

router = APIRouter(tags=["agent"])

PROCESSED_HISTORY_LIMIT = 50


@router.post("/start", name="start_agent")
async def start_agent(
    request: Request,
    form: Annotated[StartAgentForm, Form()],
    user: CurrentUserDep,
    agents: AgentManagerDep,
) -> RedirectResponse:
    if not user.google_token:
        raise LoginRequiredError
    await agents.start(user.email, user.google_token, auto_mode=form.mode == "auto")
    return redirect_to(request, "dashboard")


@router.post("/stop", name="stop_agent")
async def stop_agent(request: Request, user: CurrentUserDep, agents: AgentManagerDep) -> RedirectResponse:
    await agents.stop(user.email)
    return redirect_to(request, "index")


@router.get("/dashboard", name="dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request, user: CurrentUserDep, agents: AgentManagerDep, inbound: InboundServiceDep
) -> HTMLResponse:
    agent_status = agents.status(user.email)
    pending = await inbound.list_by_status(user.email, [InboundStatus.AWAITING_REVIEW])
    processed = await inbound.list_by_status(
        user.email, [InboundStatus.SENT, InboundStatus.REJECTED], newest_first=True, limit=PROCESSED_HISTORY_LIMIT
    )
    state = AgentStateView(
        running=agent_status.running,
        auto_mode=agent_status.auto_mode,
        pending=[PendingEmail.model_validate(row) for row in pending],
        # The template reverses this list to show newest first.
        processed=[ProcessedEmail.model_validate(row) for row in reversed(processed)],
    )
    return templates.TemplateResponse(
        request, "dashboard.html", {"state": state, "user": UserRead.model_validate(user)}
    )


@router.post("/approve/{email_id}", name="approve_email")
async def approve_email(
    request: Request,
    email_id: str,
    form: Annotated[ApproveForm, Form()],
    user: CurrentUserDep,
    gmail: GmailClientDep,
    delivery: DeliveryServiceDep,
) -> RedirectResponse:
    await delivery.approve(gmail, user.email, email_id, form.custom_response, form.forward_to)
    return redirect_to(request, "dashboard")


@router.post("/reject/{email_id}", name="reject_email")
async def reject_email(
    request: Request,
    email_id: str,
    user: CurrentUserDep,
    gmail: GmailClientDep,
    delivery: DeliveryServiceDep,
) -> RedirectResponse:
    await delivery.reject(gmail, user.email, email_id)
    return redirect_to(request, "dashboard")
