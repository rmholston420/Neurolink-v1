"""Neurolink REST + SSE endpoints.

/api/v1/neurolink/* routes.
Thin layer — delegates to NeuroLinkService.
"""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

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
async def sse_stream(request: Request, service: ServiceDep) -> StreamingResponse:
    """SSE endpoint — streams NeurolinkState JSON.

    Uses raw StreamingResponse (not sse-starlette) with explicit
    disconnect detection so the generator terminates when the client
    closes the connection.  This is required for httpx ASGITransport
    (used in tests) where the ASGI app runs in the same event-loop task
    as the consumer: without early termination the generator deadlocks
    aiter_lines() forever.

    Each SSE event: event: state\ndata: <NeurolinkState JSON>\n\n
    """

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        hub = service._hub
        hub.register_sse_queue(q)
        try:
            # Emit current state immediately — client always gets ≥1 frame
            yield _sse_frame(hub.get_state())

            # Checkpoint: let the event loop deliver the first frame before
            # entering the blocking queue wait.
            await asyncio.sleep(0)

            if await request.is_disconnected():
                return

            while True:
                try:
                    state = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield _sse_frame(state)
                except TimeoutError:
                    yield _sse_frame(hub.get_state())

                if await request.is_disconnected():
                    return
        except asyncio.CancelledError:
            pass
        finally:
            hub.unregister_sse_queue(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_frame(state: NeurolinkState) -> bytes:
    """Encode a NeurolinkState as a raw SSE frame (bytes)."""
    data = json.loads(state.model_dump_json())
    return f"event: state\ndata: {json.dumps(data)}\n\n".encode()
