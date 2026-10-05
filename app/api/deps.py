"""FastAPI dependency providers: configuration, DB sessions, services, and authentication."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, cast

from authlib.integrations.starlette_client import OAuth, StarletteOAuth2App
from fastapi import Depends, Request
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.core.config import Settings, get_settings
from app.core.errors import LoginRequiredError, PermissionDeniedError
from app.db.session import Database
from app.models import User
from app.models.enums import UserRole
from app.services.agent_runner import AgentManager
from app.services.delivery import EmailDeliveryService
from app.services.email_log import AnalyticsService, EmailLogService
from app.services.gmail import GmailClient
from app.services.inbound import InboundEmailService
from app.services.knowledge import KnowledgeService
from app.services.rules import BlockedSenderService, ReplyTemplateService
from app.services.task_state import TaskStateStore
from app.services.users import UserService

SESSION_USER_KEY = "user_email"

SettingsDep = Annotated[Settings, Depends(get_settings)]

# --- Application-scoped resources (created in the lifespan, stored on app.state) ---


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_agent_manager(request: Request) -> AgentManager:
    return cast(AgentManager, request.app.state.agent_manager)


def get_redis(request: Request) -> Redis:
    return cast(Redis, request.app.state.redis)


def get_task_state_store(request: Request) -> TaskStateStore:
    return cast(TaskStateStore, request.app.state.task_states)


def get_google_oauth(request: Request) -> StarletteOAuth2App:
    return cast(StarletteOAuth2App, cast(OAuth, request.app.state.oauth).google)


AgentManagerDep = Annotated[AgentManager, Depends(get_agent_manager)]
GoogleOAuthDep = Annotated[StarletteOAuth2App, Depends(get_google_oauth)]
TaskStateStoreDep = Annotated[TaskStateStore, Depends(get_task_state_store)]

# --- Request-scoped DB session and services --------------------------------------


async def get_db_session(database: Annotated[Database, Depends(get_database)]) -> AsyncIterator[AsyncSession]:
    async with database.sessionmaker() as session:
        yield session


DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_user_service(session: DbSessionDep) -> UserService:
    return UserService(session)


def get_template_service(session: DbSessionDep) -> ReplyTemplateService:
    return ReplyTemplateService(session)


def get_blocked_service(session: DbSessionDep) -> BlockedSenderService:
    return BlockedSenderService(session)


def get_email_log_service(session: DbSessionDep) -> EmailLogService:
    return EmailLogService(session)


def get_analytics_service(session: DbSessionDep) -> AnalyticsService:
    return AnalyticsService(session)


def get_knowledge_service(session: DbSessionDep) -> KnowledgeService:
    return KnowledgeService(session)


def get_inbound_service(session: DbSessionDep) -> InboundEmailService:
    return InboundEmailService(session)


def get_delivery_service(session: DbSessionDep) -> EmailDeliveryService:
    return EmailDeliveryService(session)


UserServiceDep = Annotated[UserService, Depends(get_user_service)]
TemplateServiceDep = Annotated[ReplyTemplateService, Depends(get_template_service)]
BlockedServiceDep = Annotated[BlockedSenderService, Depends(get_blocked_service)]
EmailLogServiceDep = Annotated[EmailLogService, Depends(get_email_log_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
InboundServiceDep = Annotated[InboundEmailService, Depends(get_inbound_service)]
DeliveryServiceDep = Annotated[EmailDeliveryService, Depends(get_delivery_service)]

# --- Authentication ----------------------------------------------------------------


async def get_optional_user(request: Request, users: UserServiceDep) -> User | None:
    email = request.session.get(SESSION_USER_KEY)
    if not isinstance(email, str):
        return None
    return await users.get_by_email(email)


async def get_current_user(user: Annotated[User | None, Depends(get_optional_user)]) -> User:
    if user is None:
        raise LoginRequiredError
    if not user.active:
        raise PermissionDeniedError
    return user


OptionalUserDep = Annotated[User | None, Depends(get_optional_user)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]


def require_roles(*roles: UserRole) -> Callable[[User], Awaitable[User]]:
    allowed = {r.value for r in roles}

    async def dependency(user: CurrentUserDep) -> User:
        if user.role not in allowed:
            raise PermissionDeniedError
        return user

    return dependency


AdminUserDep = Annotated[User, Depends(require_roles(UserRole.ADMIN))]


async def get_gmail_client(user: CurrentUserDep, settings: SettingsDep) -> GmailClient:
    if not user.google_token or "access_token" not in user.google_token:
        raise LoginRequiredError
    return await GmailClient.from_token(user.google_token, settings)


GmailClientDep = Annotated[GmailClient, Depends(get_gmail_client)]

# --- Helpers -----------------------------------------------------------------------


def redirect_to(request: Request, route_name: str, **path_params: str | int) -> RedirectResponse:
    # 303 so the browser follows a POST with a GET (FastAPI's default 307 would re-POST).
    return RedirectResponse(
        request.app.url_path_for(route_name, **{k: str(v) for k, v in path_params.items()}),
        status_code=status.HTTP_303_SEE_OTHER,
    )
