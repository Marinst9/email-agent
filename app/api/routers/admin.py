from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.deps import AdminUserDep, AnalyticsServiceDep, UserServiceDep, redirect_to
from app.core.templating import templates
from app.schemas.user import RoleUpdateForm, UserRead

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", name="admin", response_class=HTMLResponse)
async def admin_panel(
    request: Request, _admin: AdminUserDep, users: UserServiceDep, analytics: AnalyticsServiceDep
) -> HTMLResponse:
    all_users = [UserRead.model_validate(u) for u in await users.list_all()]
    summary = await analytics.platform_summary()
    return templates.TemplateResponse(request, "admin.html", {"users": all_users, **summary.model_dump()})


@router.post("/role/{user_id}", name="change_role")
async def change_role(
    request: Request,
    user_id: int,
    form: Annotated[RoleUpdateForm, Form()],
    _admin: AdminUserDep,
    users: UserServiceDep,
) -> RedirectResponse:
    await users.set_role(user_id, form.role)
    return redirect_to(request, "admin")


@router.post("/toggle/{user_id}", name="toggle_user")
async def toggle_user(request: Request, user_id: int, _admin: AdminUserDep, users: UserServiceDep) -> RedirectResponse:
    await users.toggle_active(user_id)
    return redirect_to(request, "admin")
