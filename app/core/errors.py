from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.core.templating import templates

API_PREFIX = "/api/"


class LoginRequiredError(Exception):
    """Raised by auth dependencies when there is no signed-in user."""


class PermissionDeniedError(Exception):
    """Raised when the signed-in user lacks the required role or is deactivated."""


def _is_api(request: Request) -> bool:
    return request.url.path.startswith(API_PREFIX)


async def _login_required_handler(request: Request, exc: Exception) -> Response:
    if _is_api(request):
        return JSONResponse({"detail": "Not authenticated"}, status_code=status.HTTP_401_UNAUTHORIZED)
    return RedirectResponse(request.app.url_path_for("login"), status_code=status.HTTP_303_SEE_OTHER)


async def _permission_denied_handler(request: Request, exc: Exception) -> Response:
    if _is_api(request):
        return JSONResponse({"detail": "Forbidden"}, status_code=status.HTTP_403_FORBIDDEN)
    return templates.TemplateResponse(request, "unauthorized.html", status_code=status.HTTP_403_FORBIDDEN)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(LoginRequiredError, _login_required_handler)
    app.add_exception_handler(PermissionDeniedError, _permission_denied_handler)
