import logging
from functools import partial

from celery import Task

from app.models.enums import InboundStatus
from app.services.inbound import InboundEmailService
from app.services.knowledge import KnowledgeService
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
            embedder=runtime.embedder,
        )
        return await pipeline.run(email_id)


async def _fail(runtime: WorkerRuntime, email_id: str, attempts: int, error: str) -> None:
    await runtime.task_states.mark_failed(email_id, attempts, error)
    async with runtime.database.sessionmaker() as session:
        await InboundEmailService(session).mark_failed(email_id, error)


@celery_app.task(bind=True, name="emails.process_incoming_email")
def process_incoming_email_task(self: Task[[str], str], email_id: str) -> str:
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
                email_id,
                attempts,
                settings.llm_max_retries + 1,
                countdown,
                describe(exc),
            )
            runtime.run(runtime.task_states.mark_retrying(email_id, attempts, describe(exc), countdown))
            raise self.retry(exc=exc, countdown=countdown, max_retries=settings.llm_max_retries) from exc

        logger.exception("Processing failed for %s after %d attempt(s)", email_id, attempts)
        runtime.run(_fail(runtime, email_id, attempts, describe(exc)))
        raise

    runtime.run(runtime.task_states.mark_success(email_id, attempts))
    logger.info("Processed %s -> %s", email_id, outcome)
    return outcome.value


# One run embeds at most this many chunks and re-queues itself, so a large PDF never hits the task time limit.
EMBED_CHUNKS_PER_TASK = 256
EMBED_MAX_RETRIES = 3


async def _embed_pending(runtime: WorkerRuntime, user_email: str | None) -> int:
    async with runtime.database.sessionmaker() as session:
        return await KnowledgeService(session).embed_pending(
            runtime.embedder, user_email, max_chunks=EMBED_CHUNKS_PER_TASK
        )


@celery_app.task(bind=True, name="knowledge.embed_pending")
def embed_pending_chunks_task(self: Task[[str | None], int], user_email: str | None = None) -> int:
    """Embed pending knowledge-base chunks (of one user, or of everyone when `user_email` is None).

    Idempotent: only chunks still marked pending are touched, and each batch is committed on its own,
    so a retry or a duplicate task resumes where the previous one stopped.
    """
    runtime = get_runtime()
    try:
        embedded = runtime.run(_embed_pending(runtime, user_email))
    except Exception as exc:
        if self.request.retries < EMBED_MAX_RETRIES:
            logger.warning("Embedding failed for %s, retrying: %s", user_email or "all users", describe(exc))
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1), max_retries=EMBED_MAX_RETRIES) from exc
        logger.exception("Embedding failed for %s; chunks stay pending", user_email or "all users")
        raise
    if embedded >= EMBED_CHUNKS_PER_TASK:
        self.apply_async(args=(user_email,))
    logger.info("Embedded %d chunks for %s", embedded, user_email or "all users")
    return embedded
