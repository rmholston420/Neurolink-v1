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

    Design notes
    ------------
    * Uses raw StreamingResponse (not sse-starlette) for full control over
      flushing behaviour.
    * Does NOT call request.is_disconnected(): that helper blocks on the ASGI
      receive() callable until an http.disconnect event arrives, which creates
      a deadlock under httpx ASGITransport (used in tests) because the client
      cannot close while the generator is blocked.
    * Cleanup on client disconnect is handled via asyncio.CancelledError:
      when the httpx reader task closes the response, it cancels the ASGI
      background task, which propagates CancelledError into this generator.
      The finally block then unregisters the SSE queue.
    * asyncio.sleep(0) after the first yield gives the httpx reader one event-
      loop tick to consume the frame before we block on q.get().

    Each event: event: state\ndata: <NeurolinkState JSON>\n\n
    """

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        hub = service._hub
        hub.register_sse_queue(q)
        try:
            # Always emit the current state immediately so the client
            # receives at least one frame even if the pump is idle.
            yield _sse_frame(hub.get_state())

            # Yield control so the httpx reader task can consume the
            # frame above before we block on the queue.
            await asyncio.sleep(0)

            while True:
                try:
                    state = await asyncio.wait_for(q.get(), timeout=2.0)
                    yield _sse_frame(state)
                except TimeoutError:
                    # Keepalive: re-emit current state so the connection
                    # stays alive through proxies that close idle streams.
                    yield _sse_frame(hub.get_state())

                # One tick after each frame so the reader can drain
                # before we wait on the next queue item.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Client disconnected — httpx cancelled the ASGI task.
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
