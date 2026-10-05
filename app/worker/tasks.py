import logging
from functools import partial

from celery import Task

from app.models.enums import InboundStatus
from app.services.inbound import InboundEmailService
from app.services.pipeline import EmailNotFoundError, EmailPipeline
from app.worker.celery_app import MAX_RETRY_COUNTDOWN_SECONDS, celery_app
from app.worker.retry import describe, is_retryable, retry_countdown
from app.worker.runtime import WorkerRuntime, get_runtime

logger = logging.getLogger(__name__)


async def _execute(runtime: WorkerRuntime, email_id: str, attempt: int) -> InboundStatus:
    await runtime.task_states.mark_running(email_id, attempt)
    async with runtime.database.sessionmaker() as session:
        pipeline = EmailPipeline(
            session,
            runtime.settings,
            runtime.ai_client,
            runtime.rate_limiter,
            on_stage=partial(runtime.task_states.set_stage, email_id),
        )
        return await pipeline.run(email_id)


async def _fail(runtime: WorkerRuntime, email_id: str, attempts: int, error: str) -> None:
    await runtime.task_states.mark_failed(email_id, attempts, error)
    async with runtime.database.sessionmaker() as session:
        await InboundEmailService(session).mark_failed(email_id, error)


@celery_app.task(bind=True, name="emails.process_incoming_email")
def process_incoming_email_task(self: Task, email_id: str) -> str:
    """Filter, classify, retrieve context, draft and (optionally) send the reply for one ingested email.

    Retries with exponential backoff on transient LLM API failures (429/503/529, network errors).
    """
    runtime = get_runtime()
    settings = runtime.settings
    attempt: int = self.request.retries
    attempts = attempt + 1

    try:
        outcome = runtime.run(_execute(runtime, email_id, attempt))
    except EmailNotFoundError:
        logger.error("Email %s not found; dropping task", email_id)
        runtime.run(runtime.task_states.mark_failed(email_id, attempts, "Email not found"))
        return "not_found"
    except Exception as exc:
        if is_retryable(exc) and attempt < settings.llm_max_retries:
            countdown = retry_countdown(
                exc,
                attempt,
                base=settings.llm_retry_base_seconds,
                cap=settings.llm_retry_max_seconds,
                hard_cap=MAX_RETRY_COUNTDOWN_SECONDS,
            )
            logger.warning(
                "Transient failure for %s (attempt %d/%d), retrying in %.1fs: %s",
                email_id, attempts, settings.llm_max_retries + 1, countdown, describe(exc),
            )
            runtime.run(runtime.task_states.mark_retrying(email_id, attempts, describe(exc), countdown))
            raise self.retry(exc=exc, countdown=countdown, max_retries=settings.llm_max_retries) from exc

        logger.exception("Processing failed for %s after %d attempt(s)", email_id, attempts)
        runtime.run(_fail(runtime, email_id, attempts, describe(exc)))
        raise

    runtime.run(runtime.task_states.mark_success(email_id, attempts))
    logger.info("Processed %s -> %s", email_id, outcome)
    return outcome.value
