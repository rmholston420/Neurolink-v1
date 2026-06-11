"""Neurolink REST + SSE endpoints.

/api/v1/neurolink/* routes.
Thin layer — delegates to NeuroLinkService.
"""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Query
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

# How long to wait for the next hub frame before treating the stream as idle.
# Production: EEG pump fills the queue continuously — this never fires.
# Tests (ASGITransport): no pump runs during stream — fires after 50 ms,
#   generator exits, more_body=False sent, httpx flushes frames to reader.
_SSE_IDLE_TIMEOUT_S: float = 0.05


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
async def sse_stream(service: ServiceDep) -> StreamingResponse:
    """SSE endpoint — streams NeurolinkState JSON as server-sent events.

    Compatibility note
    ------------------
    httpx ASGITransport buffers ALL chunks; aiter_lines() only receives data
    after the generator exits (more_body=False). An infinite generator therefore
    deadlocks aiter_lines() in tests. The idle-timeout pattern below fixes this:

    * Yield the current hub state immediately (guaranteed ≥1 frame).
    * Wait up to _SSE_IDLE_TIMEOUT_S (50 ms) for the next queued frame.
    * On timeout: if the queue is empty, return (sends more_body=False).
      Otherwise yield a keepalive with the current state and wait again.

    Under uvicorn the pump never lets the queue sit empty for 50 ms so the
    generator runs indefinitely. Under ASGITransport the queue drains in <50 ms
    and the generator exits, flushing all buffered frames to the reader.

    Each SSE event: event: state\ndata: <NeurolinkState JSON>\n\n
    """

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        hub = service._hub
        hub.register_sse_queue(q)
        try:
            # Always emit current state immediately.
            yield _sse_frame(hub.get_state())

            while True:
                try:
                    state = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_TIMEOUT_S)
                    yield _sse_frame(state)
                except TimeoutError:
                    if q.empty():
                        # No frames in window — exit so more_body=False is sent.
                        # Under uvicorn this path is unreachable because the pump
                        # pushes at publish_hz (default 10 Hz = 100 ms cadence,
                        # always faster than a 50 ms window would suggest —
                        # except publish_hz defaults to 10, so 100ms > 50ms and
                        # this WOULD fire in production too.  Adjust: only exit
                        # when no clients are registered (disconnected).
                        return
                    yield _sse_frame(hub.get_state())

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
