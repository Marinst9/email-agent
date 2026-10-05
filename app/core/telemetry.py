import logging
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("telemetry")


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking HTTP request latency, token counts, and estimated LLM costs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        response: Response = await call_next(request)

        latency_ms = (time.time() - start_time) * 1000

        # Extract custom telemetry headers if set by downstream endpoints/services
        prompt_tokens = int(response.headers.get("X-Prompt-Tokens", "0"))
        completion_tokens = int(response.headers.get("X-Completion-Tokens", "0"))

        # Estimated cost at Claude Sonnet 4.6 rates ($3/1M input, $15/1M output tokens)
        cost_usd = (prompt_tokens * 0.000003) + (completion_tokens * 0.000015)

        logger.info(
            f"path={request.url.path} "
            f"status={response.status_code} "
            f"latency_ms={latency_ms:.2f}ms "
            f"prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} "
            f"cost_usd=${cost_usd:.6f}"
        )

        return response