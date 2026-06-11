"""Health check endpoint.

GET /health — returns adapter status, hub frame count, Redis ping, DB reachability.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter

from neurolink.dependencies import ServiceDep
from neurolink.models.eeg import HealthResponse

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    service: ServiceDep,
) -> HealthResponse:
    """Return service health status."""
    from neurolink.config import get_settings
    from neurolink.hub import get_hub

    settings = get_settings()
    hub = get_hub()
    state = hub.get_state()

    # Redis check
    redis_status = "disabled"
    if settings.redis_enabled:
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url)
            await r.ping()
            await r.aclose()
            redis_status = "connected"
        except Exception as exc:
            log.warning("health_redis_error", error=str(exc))
            redis_status = "error"

    # DB check
    db_status = "error"
    try:
        from neurolink.db.engine import get_engine
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        log.warning("health_db_error", error=str(exc))
        db_status = "error"

    overall = "ok" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall,
        adapter_type=service.adapter_type if service.is_connected else settings.adapter_type,
        adapter_connected=service.is_connected,
        hub_frame_count=state.frame_count,
        redis=redis_status,
        db=db_status,
    )
