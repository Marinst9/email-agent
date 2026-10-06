"""Application factory. Run with: uvicorn app.main:app"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis
from starlette.middleware.sessions import SessionMiddleware

from app.api.routers import admin, agent, auth, emails_api, health, insights, knowledge, rules
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.oauth import build_oauth
from app.db.session import Database
from app.services.agent_runner import AgentManager
from app.services.task_state import TaskStateStore
from app.worker.dispatch import EmailTaskDispatcher


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(settings)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        task_states = TaskStateStore(redis, settings.task_state_ttl_seconds)
        agent_manager = AgentManager(settings, database.sessionmaker, EmailTaskDispatcher(task_states))

        app.state.database = database
        app.state.redis = redis
        app.state.task_states = task_states
        app.state.oauth = build_oauth(settings)
        app.state.agent_manager = agent_manager
        try:
            yield
        finally:
            await agent_manager.shutdown()
            await redis.aclose()
            await database.dispose()

    is_dev = settings.environment == "development"
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs" if is_dev else None,
        redoc_url=None,
        openapi_url="/openapi.json" if is_dev else None,
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key.get_secret_value(),
        session_cookie=settings.session_cookie_name,
        max_age=settings.session_max_age_seconds,
        same_site="lax",
        https_only=settings.session_https_only,
    )
    register_exception_handlers(app)

    for module in (health, auth, agent, rules, insights, knowledge, admin, emails_api):
        app.include_router(module.router)

    return app


app = create_app()
