import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from redis.asyncio import Redis

from app.api.deps import get_database, get_redis
from app.db.session import Database
from app.schemas.stats import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/healthz", name="health", response_model=HealthStatus)
async def health(
    response: Response,
    database: Annotated[Database, Depends(get_database)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> HealthStatus:
    db_ok = await database.ping()
    redis_ok = False
    with contextlib.suppress(Exception):
        redis_ok = bool(await redis.ping())  # type: ignore[misc]
    healthy = db_ok and redis_ok
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(status="ok" if healthy else "degraded", database=db_ok, redis=redis_ok)
