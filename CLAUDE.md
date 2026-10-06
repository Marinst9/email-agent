# Email Agent — project rules

- Stack: FastAPI, SQLAlchemy 2.0 async, Alembic, Celery + Redis, PostgreSQL, Anthropic SDK.
- Web app lives in `app/`. `main.py` is a standalone CLI.
- Every schema change needs an Alembic migration in `migrations/versions/`.
- The pipeline must stay idempotent: status checkpoints + atomic `claim()` in `InboundEmailService`. Never introduce a path that can send an email twice.
- Prompts and user-facing text may be Macedonian; code, comments and docs are English.
- Before finishing any task run: `ruff check .`, `mypy app`, `pytest`. All must pass.
- Do not use real API keys in tests; use fakes and fakeredis.
- Work on a feature branch, small focused commits.
