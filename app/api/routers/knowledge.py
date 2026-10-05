from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.deps import CurrentUserDep, KnowledgeServiceDep, SettingsDep, redirect_to
from app.core.templating import templates
from app.services.knowledge import extract_text

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("", name="knowledge", response_class=HTMLResponse)
async def list_documents(request: Request, user: CurrentUserDep, knowledge: KnowledgeServiceDep) -> HTMLResponse:
    docs = await knowledge.list_filenames(user.email)
    return templates.TemplateResponse(request, "knowledge.html", {"docs": docs, "message": None})


@router.post("", name="upload_document", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    user: CurrentUserDep,
    knowledge: KnowledgeServiceDep,
    settings: SettingsDep,
) -> HTMLResponse:
    # Only the base name is kept; the file itself is never written to disk.
    filename = PurePath(file.filename or "").name
    data = await file.read(settings.max_upload_bytes + 1)

    status_code = status.HTTP_200_OK
    if not filename:
        message = None
    elif len(data) > settings.max_upload_bytes:
        message = f"❌ Документот е преголем (макс. {settings.max_upload_bytes // (1024 * 1024)} MB)"
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    else:
        chunks = await knowledge.add_document(user.email, filename, await extract_text(filename, data))
        message = f"✅ Документот '{filename}' е додаден ({chunks} парчиња)"

    docs = await knowledge.list_filenames(user.email)
    return templates.TemplateResponse(
        request, "knowledge.html", {"docs": docs, "message": message}, status_code=status_code
    )


@router.post("/delete/{filename:path}", name="delete_document")
async def delete_document(
    request: Request, filename: str, user: CurrentUserDep, knowledge: KnowledgeServiceDep
) -> RedirectResponse:
    await knowledge.delete_document(user.email, filename)
    return redirect_to(request, "knowledge")
