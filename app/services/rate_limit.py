"""Reply rate limiting shared by every worker process (Redis sliding window)."""

import time

from redis.asyncio import Redis

# Atomic sliding-window check. Idempotent per member (email id): a retried task that was
# already admitted is admitted again without consuming another slot.
_SLIDING_WINDOW_LUA = """
local key, now, window, limit, member = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
if redis.call('ZSCORE', key, member) then
  return 1
end
if redis.call('ZCARD', key) >= limit then
  return 0
end
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window))
return 1
"""


class RedisReplyRateLimiter:
    """At most `limit` replies per (user, sender) per `window_seconds`."""

    def __init__(self, redis: "Redis", limit: int, window_seconds: float = 3600) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_seconds
        self._script = redis.register_script(_SLIDING_WINDOW_LUA)

    async def allow(self, user_email: str, sender: str, email_id: str, now: float | None = None) -> bool:
        key = f"reply-rate:{user_email}:{sender.lower()}"
        now = time.time() if now is None else now
        result = await self._script(keys=[key], args=[now, self._window, self._limit, email_id])
        return bool(int(result))
