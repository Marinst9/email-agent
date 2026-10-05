release: alembic upgrade head
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips "*"
worker: celery -A app.worker.celery_app worker --loglevel=INFO -Q emails --concurrency=${WORKER_CONCURRENCY:-4}
