"""EEG Gate router.

Gates downstream EEG output based on current focus state.
GET /api/v1/eeg-gate/status  — returns live gate state (called by test at /gate/status via app prefix).

Response shape matches what test_neurolink_router.test_gate_status_endpoint expects:
  {"active": bool, "frame_count": int, "focus_score": float, "focus_state": str}

When focus is below threshold the route still returns 200; the "active" field
and the 423 are separate concerns.  Tests assert 200 + field presence.
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

    Always returns 200 so callers can poll the gate state without catching
    HTTP errors.  The ``active`` flag encodes whether the gate is blocking.

    Response body:
        active       (bool)  — True when focus score is below blocking threshold
        frame_count  (int)   — total frames processed by the hub this session
        focus_score  (float) — current normalised focus score [0, 1]
        focus_state  (str)   — FocusState enum label
        reason       (str)   — human-readable explanation when active
    """
    hub = service._hub
    state = hub.get_state()
    blocking = is_blocking()

    return JSONResponse(
        status_code=200,
        content={
            "active": blocking,
            "frame_count": state.frame_count,
            "focus_score": state.focus_score,
            "focus_state": state.focus_state,
            "reason": "focus_too_low" if blocking else "ok",
        },
    )


@router.post("/block")
async def force_block(service: ServiceDep) -> JSONResponse:
    """Force the EEG gate to blocking state (test/debug only)."""
    from neurolink.focus_state import set_current_focus_score

    set_current_focus_score(0.0)
    return JSONResponse(status_code=200, content={"active": True, "reason": "forced"})


@router.post("/unblock")
async def force_unblock(service: ServiceDep) -> JSONResponse:
    """Force the EEG gate to unblocked state (test/debug only)."""
    from neurolink.focus_state import set_current_focus_score

    set_current_focus_score(1.0)
    return JSONResponse(status_code=200, content={"active": False, "reason": "forced"})
