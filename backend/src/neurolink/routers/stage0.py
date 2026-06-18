"""Stage 0 REST endpoints.

GET  /api/v1/stage0/status
    Full Stage0Status snapshot (impedance, IMU, environment).

POST /api/v1/stage0/impedance
    Push per-channel impedance readings (kΩ) from the client / amplifier.

POST /api/v1/stage0/environment/ack
    Acknowledge one or all environment-checklist steps.

GET  /api/v1/stage0/environment/ready
    Poll-friendly readiness gate (200 OK when ready, 202 Accepted when not).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stage0", tags=["stage0"])


# ── Request / Response models ──────────────────────────────────────────────


class ImpedanceUpdateRequest(BaseModel):
    """Per-channel impedance values in kΩ.

    Supply only the channels your amplifier reports.
    Example: {"TP9": 15.3, "AF7": 250.0, "AF8": 12.1}
    """

    readings: dict[str, float] = Field(
        description="Channel-label to kΩ mapping",
        examples=[{"TP9": 15.3, "AF7": 180.0, "AF8": 12.1, "TP10": 9.4}],
    )


class AcknowledgeRequest(BaseModel):
    """Acknowledge one or all environment-checklist steps.

    Set step_id to a specific prompt id (e.g. 'phone_distance') or
    'all' to acknowledge every step at once.
    """

    step_id: str = Field(
        description="Prompt id to acknowledge, or 'all'",
        examples=["phone_distance", "all"],
    )


class Stage0StatusResponse(BaseModel):
    acquisition_ready: bool
    impedance: dict
    imu: dict
    environment: dict


class ReadinessResponse(BaseModel):
    ready: bool
    reason: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────


def _get_guard(request: Request):
    """Pull the Stage0Guard singleton from app.state (set in lifespan)."""
    guard = getattr(request.app.state, "stage0_guard", None)
    if guard is None:
        raise HTTPException(status_code=503, detail="Stage0Guard not initialised")
    return guard


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.get("/status", response_model=Stage0StatusResponse)
async def get_status(request: Request) -> Stage0StatusResponse:
    """Return the full Stage 0 status snapshot."""
    guard = _get_guard(request)
    data = guard.status_dict()
    return Stage0StatusResponse(**data)


@router.post("/impedance")
async def update_impedance(
    request: Request,
    body: ImpedanceUpdateRequest,
) -> dict:
    """Push per-channel impedance readings (kΩ) from the client.

    Clients that have access to raw impedance values from their amplifier
    (e.g. OpenBCI, g.tec) should POST here at session start and periodically
    during the session.  Muse devices use the `poor_contact` boolean path
    (automatic, via EEGSample) and do not need to call this endpoint.
    """
    guard = _get_guard(request)
    guard.impedance.update_from_kohm(body.readings)
    summary = guard.impedance.summary_dict()
    log.info(
        "stage0_impedance_updated",
        bad_channels=summary["bad_channels"],
        all_ok=summary["all_channels_ok"],
    )
    return {"ok": True, "summary": summary}


@router.post("/environment/ack")
async def acknowledge_step(
    request: Request,
    body: AcknowledgeRequest,
) -> dict:
    """Acknowledge one or all environment-checklist steps.

    Returns the updated environment status dict.
    """
    guard = _get_guard(request)
    if body.step_id == "all":
        guard.environment.acknowledge_all()
        log.info("stage0_environment_all_acked")
    else:
        ok = guard.environment.acknowledge(body.step_id)
        if not ok:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown step_id '{body.step_id}'. "
                f"Valid ids: {[p['id'] for p in guard.environment._all_prompt_ids]}",
            )
        log.info("stage0_environment_step_acked", step_id=body.step_id)
    return {"ok": True, "environment": guard.environment.status_dict()}


@router.get("/environment/ready")
async def environment_ready(request: Request) -> ReadinessResponse:
    """Poll-friendly readiness check.

    Returns HTTP 200 with ready=True when stabilisation is complete and all
    steps are acknowledged.  Returns HTTP 202 (Accepted / not yet ready)
    otherwise so clients can differentiate without parsing the body.
    """
    from fastapi.responses import JSONResponse

    guard = _get_guard(request)
    env = guard.environment
    ready = env.is_ready

    reasons = []
    if not env.stabilise_complete:
        remaining = round(env.stabilise_remaining_s, 1)
        reasons.append(f"stabilisation countdown: {remaining}s remaining")
    if not env.all_steps_acked:
        unacked = [p["id"] for p in guard.environment.status_dict()["prompts"] if not p["acked"]]
        reasons.append(f"unacknowledged steps: {unacked}")

    body = ReadinessResponse(
        ready=ready,
        reason="; ".join(reasons) if reasons else "all conditions met",
    )
    status_code = 200 if ready else 202
    return JSONResponse(content=body.model_dump(), status_code=status_code)
