"""JSON API for polling background email processing."""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUserDep, InboundServiceDep, TaskStateStoreDep
from app.schemas.tasks import EmailProcessingStatus
from app.services.task_state import build_processing_status

router = APIRouter(prefix="/api/v1/emails", tags=["emails"])


@router.get(
    "/{email_id}/status",
    name="email_processing_status",
    response_model=EmailProcessingStatus,
    responses={404: {"description": "Unknown email, or it belongs to another user"}},
)
async def get_email_status(
    email_id: str,
    user: CurrentUserDep,
    inbound: InboundServiceDep,
    task_states: TaskStateStoreDep,
) -> EmailProcessingStatus:
    """Task state (PENDING / RETRYING / SUCCESS / FAILED), progress stage, and the generated draft once available."""
    row = await inbound.get_for_user(email_id, user.email)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Email not found")
    return build_processing_status(row, await task_states.get(email_id))
