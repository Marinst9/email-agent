"""Test-wide setup.

Some modules build `Settings` at import time (the Celery app). Force dummy values here, before any
test module imports them, so tests never need, or accidentally pick up, real credentials.
"""

import os

os.environ.update(
    {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379/15",
        "SECRET_KEY": "test-secret-key",
        "GOOGLE_CLIENT_ID": "test-client-id",
        "GOOGLE_CLIENT_SECRET": "test-client-secret",
        "ANTHROPIC_API_KEY": "test-not-a-real-key",
    }
)
