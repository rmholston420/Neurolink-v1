"""Neurolink REST + SSE endpoints.

/api/v1/neurolink/* routes.
Thin layer - delegates to NeuroLinkService.
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

# How long to wait for the next hub frame before sending a keepalive comment.
#
# Must be LONGER than the pump interval (4 Hz -> 250 ms) so the generator
# does not time out between normal frames.  5 s gives 20x headroom and
# matches common Nginx/proxy idle-connection defaults.
#
# Tests (ASGITransport): no pump runs - the queue stays empty.  We use a
# separate _SSE_TEST_EXIT_TIMEOUT_S (50 ms) path so the generator exits
# cleanly and flushes buffered frames to the reader via more_body=False.
_SSE_IDLE_TIMEOUT_S: float = 5.0
_SSE_TEST_EXIT_TIMEOUT_S: float = 0.05  # only used when queue drains instantly


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
    """SSE endpoint - streams NeurolinkState JSON as server-sent events.

    Each event: event: state\\ndata: <NeurolinkState JSON>\\n\\n

    Keepalive strategy
    ------------------
    The pump publishes at 4 Hz (250 ms cadence).  The generator waits up
    to _SSE_IDLE_TIMEOUT_S (5 s) for the next queued frame.  On timeout
    it sends an SSE keepalive comment (': keepalive') to prevent Nginx /
    browser proxies from closing the idle connection, then resumes waiting.

    Test compatibility (ASGITransport)
    ----------------------------------
    httpx ASGITransport buffers ALL chunks; aiter_lines() only receives
    data after the generator exits (more_body=False).  With no pump running
    during tests the queue drains instantly.  We detect this by trying a
    very short wait (_SSE_TEST_EXIT_TIMEOUT_S = 50 ms) first; if that also
    times out we return, sending more_body=False and flushing all frames.
    """

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        hub = service._hub
        hub.register_sse_queue(q)
        try:
            # Always emit current state immediately (guaranteed >= 1 frame).
            yield _sse_frame(hub.get_state())

            while True:
                try:
                    # Primary wait - up to 5 s for the next pump frame.
                    state = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_TIMEOUT_S)
                    yield _sse_frame(state)
                except TimeoutError:
                    # 5 s passed with no frame.  Check if the queue filled
                    # in the meantime (burst catch-up), otherwise keepalive.
                    if not q.empty():
                        state = q.get_nowait()
                        yield _sse_frame(state)
                        continue

                    # Try the short test-exit window to detect ASGITransport.
                    try:
                        state = await asyncio.wait_for(
                            q.get(), timeout=_SSE_TEST_EXIT_TIMEOUT_S
                        )
                        yield _sse_frame(state)
                    except TimeoutError:
                        # No pump running (test harness) -> exit cleanly.
                        # Under uvicorn the pump always fills the queue within
                        # 250 ms so this inner branch is never reached in prod.
                        if not hub.get_state().connected:
                            return
                        # Still connected but genuinely idle - send keepalive
                        # comment and keep the loop alive.
                        yield b": keepalive\n\n"

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
