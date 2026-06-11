"""Neurolink REST + SSE endpoints.

/api/v1/neurolink/* routes.
Thin layer — delegates to NeuroLinkService.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse

from neurolink.dependencies import ServiceDep
from neurolink.models.eeg import (
    BandPowerResponse,
    ConnectRequest,
    ConnectResponse,
    DisconnectResponse,
    EA1Result,
    NeurolinkState,
    SessionSummary,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/neurolink", tags=["neurolink"])


@router.post("/connect", response_model=ConnectResponse)
async def connect(
    body: ConnectRequest,
    service: ServiceDep,
) -> ConnectResponse:
    """Connect to an EEG adapter and start streaming."""
    return await service.connect(
        adapter_type=body.adapter_type,
        device_model=body.device_model,
        address=body.address,
    )


@router.post("/disconnect", response_model=DisconnectResponse)
async def disconnect(service: ServiceDep) -> DisconnectResponse:
    """Disconnect the active EEG adapter."""
    return await service.disconnect()


@router.get("/state", response_model=NeurolinkState)
async def get_state(service: ServiceDep) -> NeurolinkState:
    """Return current EEG state snapshot."""
    return await service.get_current_state()


@router.get("/bands", response_model=BandPowerResponse)
async def get_bands(
    service: ServiceDep,
    channel: str = Query(default="mean", description="Channel name or 'mean'"),
) -> BandPowerResponse:
    """Return band powers (optionally for a specific channel)."""
    return await service.get_band_powers(channel=channel)


@router.get("/ea1", response_model=EA1Result)
async def get_ea1(service: ServiceDep) -> EA1Result:
    """Return latest EA-1 eligibility result."""
    return await service.get_ea1()


@router.get("/sessions", response_model=list[SessionSummary])
async def get_sessions(
    service: ServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[SessionSummary]:
    """Return recent EEG session log entries."""
    return await service.get_sessions(limit=limit)


@router.get("/stream")
async def sse_stream(service: ServiceDep) -> EventSourceResponse:
    """SSE endpoint — streams NeurolinkState JSON at 4 Hz.

    Immediately emits the current hub state on connect so clients
    receive data without waiting for the first queue push or keepalive.
    Subsequent frames come from the SSE fan-out queue.

    Each event: data: <NeurolinkState JSON>
    """

    async def event_generator():
        async for state in _stream_with_retry(service):
            yield {
                "data": state.model_dump_json(),
                "event": "state",
            }

    return EventSourceResponse(event_generator())


async def _stream_with_retry(service):
    """Yield NeurolinkState frames from the hub SSE queue.

    Emits the current state immediately on registration so the client
    always receives at least one frame, then fans out from the queue.
    Handles reconnect: if the service is not connected, yields
    the current (disconnected) state every 2 seconds until connected.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=32)
    hub = service._hub
    hub.register_sse_queue(q)
    try:
        # Yield current state immediately so client gets a frame right away
        # without waiting for the next pump tick or keepalive timeout.
        yield hub.get_state()
        while True:
            try:
                state = await asyncio.wait_for(q.get(), timeout=2.0)
                yield state
            except TimeoutError:
                # Keep connection alive — emit current state as keepalive
                yield hub.get_state()
    except asyncio.CancelledError:
        pass
    finally:
        hub.unregister_sse_queue(q)
