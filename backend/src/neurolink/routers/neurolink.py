"""Neurolink REST + SSE endpoints.

All business logic is delegated to NeuroLinkService.
No hub access in this module.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from neurolink.db.repository import SessionLogRepository
from neurolink.dependencies import get_db_session, get_eeg_hub, get_neurolink_service
from neurolink.exceptions import AdapterNotConnectedError, CalibrationBusyError
from neurolink.models.eeg import (
    BandPowerResponse,
    ConnectRequest,
    ConnectResponse,
    DisconnectResponse,
    EA1Result,
    NeurolinkState,
    SessionSummary,
)
from neurolink.service import NeuroLinkService

router = APIRouter(prefix="/api/v1/neurolink", tags=["neurolink"])


@router.post("/connect", response_model=ConnectResponse)
async def connect(
    request: ConnectRequest,
    service: NeuroLinkService = Depends(get_neurolink_service),
    db: object = Depends(get_db_session),
) -> ConnectResponse:
    """Connect to the EEG adapter."""
    from sqlalchemy.ext.asyncio import AsyncSession
    repo = SessionLogRepository(db)  # type: ignore[arg-type]
    try:
        return await service.connect(request, session_repo=repo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/disconnect", response_model=DisconnectResponse)
async def disconnect(
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> DisconnectResponse:
    """Disconnect the active EEG adapter."""
    return await service.disconnect()


@router.get("/state", response_model=NeurolinkState)
async def get_state(
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> NeurolinkState:
    """Return the current Neurolink state."""
    return await service.get_current_state()


@router.get("/bands", response_model=BandPowerResponse)
async def get_bands(
    channel: str = Query(default="all", description="Channel name or 'all'"),
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> BandPowerResponse:
    """Return band powers for a channel."""
    return await service.get_band_powers(channel=channel)


@router.get("/ea1", response_model=EA1Result)
async def get_ea1(
    service: NeuroLinkService = Depends(get_neurolink_service),
) -> EA1Result:
    """Return the latest EA-1 eligibility result."""
    return await service.get_ea1()


@router.get("/sessions")
async def list_sessions(
    db: object = Depends(get_db_session),
) -> list[dict]:
    """Return recent session log entries."""
    from sqlalchemy.ext.asyncio import AsyncSession
    repo = SessionLogRepository(db)  # type: ignore[arg-type]
    rows = await repo.list_recent(limit=20)
    result = []
    for row in rows:
        result.append({
            "id": row.id,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            "device_model": row.device_model,
            "adapter_type": row.adapter_type,
            "frame_count": row.frame_count,
            "final_ea1_eligible": row.final_ea1_eligible,
        })
    return result


async def _sse_generator(
    hub,  # type: ignore[type-arg]
) -> AsyncGenerator[dict, None]:
    """SSE generator: subscribe to hub SSE queue and fan out state frames."""
    q = hub.register_sse_queue()
    try:
        while True:
            try:
                state = await asyncio.wait_for(q.get(), timeout=30.0)
                yield {"data": state.model_dump_json()}
            except asyncio.TimeoutError:
                # Send a keepalive comment
                yield {"comment": "keepalive"}
    finally:
        hub.deregister_sse_queue(q)


@router.get("/stream")
async def stream_sse(
    hub=Depends(get_eeg_hub),  # type: ignore[type-arg]
) -> EventSourceResponse:
    """SSE stream of NeurolinkState frames at ~4 Hz."""
    return EventSourceResponse(_sse_generator(hub))
