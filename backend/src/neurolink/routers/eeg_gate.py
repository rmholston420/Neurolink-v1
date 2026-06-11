"""EEG gate middleware and router.

Blocks requests that require an active EEG session.
Ported from Rigpa-v2 eeg_gate_middleware.py + eeg_gate_router.py.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from neurolink.dependencies import get_eeg_hub

router = APIRouter(prefix="/api/v1/gate", tags=["eeg_gate"])

# Paths that require an active EEG session
_GATED_PATHS = frozenset({
    "/api/v1/neurolink/stream",
    "/api/v1/neurolink/bands",
    "/api/v1/neurolink/ea1",
})


class EEGGateMiddleware(BaseHTTPMiddleware):
    """Middleware that blocks gated endpoints when hub has no data."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[type-arg]
        """Check if the path is gated and whether EEG data is available."""
        if request.url.path in _GATED_PATHS:
            hub = get_eeg_hub()
            state = hub.get_state()
            if not state.connected and state.frame_count == 0:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "EEG adapter not connected", "code": "NOT_CONNECTED"},
                )
        return await call_next(request)


@router.get("/status")
async def gate_status(
    hub=Depends(get_eeg_hub),  # type: ignore[type-arg]
) -> dict:
    """Return EEG gate status."""
    state = hub.get_state()
    return {
        "connected": state.connected,
        "frame_count": state.frame_count,
        "gate_open": state.connected or state.frame_count > 0,
    }
