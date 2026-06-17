"""Health check and diagnostics endpoints.

GET /health      — adapter status, hub frame count, Redis ping, DB reachability.
GET /ready       — lightweight Kubernetes-style readiness probe.
GET /hub/stats   — lightweight hub diagnostic counters (frames, settling, SSE
                    client count, baseline phase).  Safe to poll at any cadence.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neurolink.dependencies import ServiceDep
from neurolink.models.eeg import HealthResponse

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


# ── Response model ────────────────────────────────────────────────────────────


class HubStatsResponse(BaseModel):
    """Lightweight hub diagnostic counters returned by GET /hub/stats.

    All values are best-effort snapshots; they may be slightly stale under
    concurrent writes.  Intended for developer dashboards and health probes,
    not for session logic.

    Attributes
    ----------
    frames_processed:
        Total hub.update() calls since construction or last reset().
    settling_events_emitted:
        Total hub.emit_settling() calls since construction or last reset().
        A high ratio of settling / (settling + frames_processed) indicates
        the acquisition guard is holding many frames (noisy environment,
        poor electrode contact, or excessive motion).
    sse_client_count:
        Number of currently registered SSE queues (one per connected browser
        tab / mobile client).
    frame_count:
        NeurolinkState.frame_count of the current state snapshot.  Increments
        with every accepted frame; does NOT count settling-held frames.
    baseline_phase:
        Current baseline recorder phase string: 'warmup' | 'recording' |
        'complete' | None (before first device connect).
    """

    frames_processed: int
    settling_events_emitted: int
    sse_client_count: int
    frame_count: int
    baseline_phase: str | None


# ── Endpoints ────────────────────────────────────────────────────────────────


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


@router.get("/ready")
async def ready() -> JSONResponse:
    """Kubernetes-style readiness probe.

    Returns 200 when the service is ready to accept traffic, 503 otherwise.
    'Ready' means the database is reachable; Redis is optional.
    """
    try:
        from neurolink.db.engine import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return JSONResponse(status_code=200, content={"status": "ready"})
    except Exception as exc:
        log.warning("ready_db_error", error=str(exc))
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(exc)})


@router.get("/hub/stats", response_model=HubStatsResponse)
async def hub_stats() -> HubStatsResponse:
    """Return lightweight hub diagnostic counters.

    Calls hub.get_stats() on the process-global singleton.  No service
    dependency needed — safe to call before any device has connected.

    Useful for:
    - Verifying the pump is running (frames_processed increments at 4 Hz)
    - Diagnosing Stage 0 hold-rate (settling_events_emitted / frames_processed)
    - Counting live SSE clients (sse_client_count)
    - Checking baseline progress (baseline_phase)

    Example response
    ----------------
    {
      "frames_processed": 1440,
      "settling_events_emitted": 12,
      "sse_client_count": 2,
      "frame_count": 1440,
      "baseline_phase": "recording"
    }
    """
    from neurolink.hub import get_hub

    stats = get_hub().get_stats()
    return HubStatsResponse(**stats)
