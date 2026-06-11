"""EEG gate router — session gate status and middleware.

GET /api/v1/gate/status — returns whether an active EEG session is running.
The EEGGateMiddleware is registered in main.py.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from neurolink.dependencies import ServiceDep

router = APIRouter(prefix="/gate", tags=["eeg_gate"])


class GateStatusResponse(BaseModel):
    """EEG session gate status."""

    active: bool
    source: str
    frame_count: int


@router.get("/status", response_model=GateStatusResponse)
async def get_gate_status(service: ServiceDep) -> GateStatusResponse:
    """Return current EEG session gate status."""
    state = await service.get_current_state()
    return GateStatusResponse(
        active=service.is_connected,
        source=state.source,
        frame_count=state.frame_count,
    )
