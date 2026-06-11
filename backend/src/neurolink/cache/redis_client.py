"""Optional Redis client for cross-process state sharing.

Disabled by default (NEUROLINK_REDIS_ENABLED=false).
When enabled, hub.snapshot() is pushed to Redis on every frame.
"""
from __future__ import annotations

import json

import structlog

log = structlog.get_logger(__name__)

_REDIS_KEY = "neurolink:state"
_TTL_SEC = 10


async def push_state(state_dict: dict) -> None:
    """Push current NeurolinkState dict to Redis.

    No-op if Redis is disabled or unreachable.
    """
    from neurolink.config import get_settings
    settings = get_settings()
    if not settings.redis_enabled:
        return
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.setex(_REDIS_KEY, _TTL_SEC, json.dumps(state_dict))
        await r.aclose()
    except Exception as exc:
        log.warning("redis_push_error", error=str(exc))


async def get_state() -> dict | None:
    """Get NeurolinkState dict from Redis.

    Returns None if Redis is disabled or key not found.
    """
    from neurolink.config import get_settings
    settings = get_settings()
    if not settings.redis_enabled:
        return None
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        raw = await r.get(_REDIS_KEY)
        await r.aclose()
        if raw:
            return json.loads(raw)
    except Exception as exc:
        log.warning("redis_get_error", error=str(exc))
    return None
