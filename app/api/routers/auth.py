from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.api.deps import (
    SESSION_USER_KEY,
    AgentManagerDep,
    BlockedServiceDep,
    GoogleOAuthDep,
    OptionalUserDep,
    SettingsDep,
    TemplateServiceDep,
    UserServiceDep,
    redirect_to,
)
from app.core.templating import templates
from app.schemas.user import GoogleUserInfo, UserRead

router = APIRouter(tags=["auth"])


@router.get("/", name="index", response_class=HTMLResponse)
async def index(request: Request, user: OptionalUserDep) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"user": UserRead.model_validate(user) if user else None})


@router.get("/login", name="login")
async def login(request: Request, google: GoogleOAuthDep, settings: SettingsDep) -> RedirectResponse:
    redirect_uri = settings.google_redirect_uri or str(request.url_for("callback"))
    response: RedirectResponse = await google.authorize_redirect(
        request, redirect_uri, access_type="offline", prompt="consent"
    )
    return response


@router.get("/callback", name="callback")
async def callback(
    request: Request,
    google: GoogleOAuthDep,
    users: UserServiceDep,
    blocked: BlockedServiceDep,
    reply_templates: TemplateServiceDep,
) -> Response:
    token: dict[str, Any] = await google.authorize_access_token(request)
    info = GoogleUserInfo.model_validate(token["userinfo"])

    user = await users.register_login(info, token)
    await blocked.seed_defaults(user.email)
    await reply_templates.seed_defaults(user.email)

    request.session.clear()
    request.session[SESSION_USER_KEY] = user.email
    return redirect_to(request, "index")


@router.get("/logout", name="logout")
async def logout(request: Request, user: OptionalUserDep, agents: AgentManagerDep) -> RedirectResponse:
    if user is not None:
        await agents.stop(user.email)
    request.session.clear()
    return redirect_to(request, "index")
