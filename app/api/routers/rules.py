"""Reply templates and blocked-sender rules."""

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.deps import BlockedServiceDep, CurrentUserDep, TemplateServiceDep, redirect_to
from app.core.templating import templates
from app.schemas.rules import BlockedWordForm, ReplyTemplateCreate

router = APIRouter(tags=["rules"])


@router.get("/templates", name="manage_templates", response_class=HTMLResponse)
async def list_templates(request: Request, user: CurrentUserDep, service: TemplateServiceDep) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "templates.html", {"templates": await service.list_for_user(user.email)}
    )


@router.post("/templates", name="create_template")
async def create_template(
    request: Request,
    form: Annotated[ReplyTemplateCreate, Form()],
    user: CurrentUserDep,
    service: TemplateServiceDep,
) -> RedirectResponse:
    await service.create(user.email, form)
    return redirect_to(request, "manage_templates")


@router.post("/templates/delete/{template_id}", name="delete_template")
async def delete_template(
    request: Request, template_id: int, user: CurrentUserDep, service: TemplateServiceDep
) -> RedirectResponse:
    await service.delete(user.email, template_id)
    return redirect_to(request, "manage_templates")


@router.get("/blocked", name="manage_blocked", response_class=HTMLResponse)
async def list_blocked(request: Request, user: CurrentUserDep, service: BlockedServiceDep) -> HTMLResponse:
    return templates.TemplateResponse(request, "blocked.html", {"blocked": await service.list_words(user.email)})


@router.post("/blocked", name="add_blocked")
async def add_blocked(
    request: Request,
    form: Annotated[BlockedWordForm, Form()],
    user: CurrentUserDep,
    service: BlockedServiceDep,
) -> RedirectResponse:
    await service.add(user.email, form.word)
    return redirect_to(request, "manage_blocked")


@router.post("/blocked/delete/{word}", name="delete_blocked")
async def delete_blocked(
    request: Request, word: str, user: CurrentUserDep, service: BlockedServiceDep
) -> RedirectResponse:
    await service.remove(user.email, word)
    return redirect_to(request, "manage_blocked")
