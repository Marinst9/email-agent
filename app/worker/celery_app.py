"""Celery application. Start a worker with:

    celery -A app.worker.celery_app worker --loglevel=INFO -Q emails
"""

from celery import Celery

from app.core.config import get_settings

EMAIL_QUEUE = "emails"

# Redis redelivers a task that is not acked within this window. Retries are scheduled with a
# countdown (held unacked by the worker), so every countdown must stay well below it.
BROKER_VISIBILITY_TIMEOUT_SECONDS = 3600
MAX_RETRY_COUNTDOWN_SECONDS = BROKER_VISIBILITY_TIMEOUT_SECONDS // 2


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery("email_agent", broker=settings.redis_url, include=["app.worker.tasks"])
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_default_queue=EMAIL_QUEUE,
        # Task state is tracked explicitly per email (see TaskStateStore), not via a Celery result backend.
        task_ignore_result=True,
        # At-least-once execution: ack after completion and requeue if the worker process dies.
        # The pipeline is idempotent (status checkpoints + atomic claims) to make this safe.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=settings.task_soft_time_limit_seconds,
        task_time_limit=settings.task_time_limit_seconds,
        broker_transport_options={"visibility_timeout": BROKER_VISIBILITY_TIMEOUT_SECONDS},
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
    )
    return app


celery_app = create_celery_app()
