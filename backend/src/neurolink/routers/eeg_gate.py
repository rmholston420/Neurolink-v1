"""EEG Gate middleware router.

Gates downstream EEG output based on current focus state.
When focus score is below threshold, returns a 423 Locked response
to indicate the gate is blocking.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neurolink.dependencies import ServiceDep
from neurolink.focus_state import is_blocking

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/eeg-gate", tags=["eeg-gate"])


@router.get("/status")
async def gate_status(service: ServiceDep) -> JSONResponse:
    """Return the current EEG gate status.

    Returns 200 with {"blocking": false} when focus is sufficient.
    Returns 423 with {"blocking": true, "reason": "focus_too_low"} when gated.
    """
    blocking = is_blocking()
    if blocking:
        return JSONResponse(
            status_code=423,
            content={
                "blocking": True,
                "reason": "focus_too_low",
                "detail": "EEG gate is active: focus score below threshold.",
            },
        )
    return JSONResponse(
        status_code=200,
        content={"blocking": False},
    )
