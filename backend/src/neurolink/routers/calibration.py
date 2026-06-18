"""Calibration endpoints.

POST /api/v1/calibration/start  -- start 90-second baseline alpha capture.
GET  /api/v1/calibration/progress -- lightweight progress poll for non-SSE clients.

Note: legacy routes at /api/v1/neurolink/calibrate and
/api/v1/neurolink/baseline remain accessible via the neurolink router
for backward compatibility.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from neurolink.dependencies import ServiceDep
from neurolink.models.eeg import BaselineProgressResponse, CalibrateResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("/start", response_model=CalibrateResponse)
async def start_calibration(service: ServiceDep) -> CalibrateResponse:
    """Start a 90-second personal alpha baseline calibration session.

    Returns immediately with {"status": "started"}.
    Calibration runs in the background; hub.baseline_alpha is updated on completion.
    """
    return await service.start_calibration()


@router.get("/progress", response_model=BaselineProgressResponse)
async def get_calibration_progress(service: ServiceDep) -> BaselineProgressResponse:
    """Return current calibration progress for polling clients.

    Lightweight alternative to the SSE stream for clients that cannot
    consume server-sent events (e.g. React Native fetch pollers, CLI
    health checks, embedded HTTP clients).

    Response
    --------
    ::

        {
          "phase":       "idle" | "warmup" | "baseline" | "complete",
          "elapsed_s":   12.34,
          "remaining_s": 77.66,
          "total_s":     90.0
        }

    The endpoint is always available -- callers do **not** need to call
    ``POST /calibration/start`` first.  When no session is running ``phase``
    is ``"idle"`` and both timing fields are ``0.0``.

    Recommended poll interval: 1-2 s.  Clients that consume the SSE
    stream already receive ``baseline_phase`` on every ``NeurolinkState``
    event and should use that instead.
    """
    return service.get_baseline_progress()
