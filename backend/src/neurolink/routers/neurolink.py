"""Neurolink REST + SSE endpoints.

/api/v1/neurolink/* routes.
Thin layer - delegates to NeuroLinkService.

SSE event types
---------------
The generator encodes three distinct queue item types into SSE frames:

  NeurolinkState (object)
    event: state
    data:  <NeurolinkState model_dump_json>

  baseline_complete sentinel  {"event": "baseline_complete"}
    event: baseline_complete
    data:  {}

  settling sentinel  {"event": "settling", "reason": <code>}
    event: settling
    data:  {"reason": "impedance_unstable" | "motion_settling"
                      | "env_not_ready"    | "settling"}

  Any other dict (future sentinels, forwards-compatible)
    event: unknown
    data:  <raw dict JSON>

Keepalive (no hub item)
    : keepalive   (SSE comment line, not an event)
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
    BaselineProgressResponse,
    CalibrateResponse,
    ConnectRequest,
    ConnectResponse,
    DisconnectResponse,
    EA1Result,
    NeurolinkState,
    SessionSummary,
)

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/neurolink", tags=["neurolink"])

_SSE_IDLE_TIMEOUT_S: float = 5.0
_SSE_TEST_EXIT_TIMEOUT_S: float = 0.05


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


@router.post("/calibrate", response_model=CalibrateResponse)
async def calibrate(service: ServiceDep) -> CalibrateResponse:
    """Start a 90-second alpha baseline calibration session (legacy route).

    Kept for backward compatibility alongside POST /api/v1/calibration/start.
    Both routes delegate to the same service method.
    """
    return await service.start_calibration()


@router.get("/baseline", response_model=BaselineProgressResponse)
async def get_baseline_progress(service: ServiceDep) -> BaselineProgressResponse:
    """Return current calibration/baseline progress for polling clients.

    Lightweight alternative to the SSE stream.  Always available — callers
    do not need to call POST /calibrate first.  Returns phase='idle' when
    no session is running.
    """
    return service.get_baseline_progress()


@router.get("/stream")
async def sse_stream(service: ServiceDep) -> StreamingResponse:
    """SSE endpoint — streams EEG events as server-sent events."""

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        hub = service._hub
        hub.register_sse_queue(q)
        try:
            # Emit current state immediately so tests always receive≥ 1 frame.
            yield _encode_sse_item(hub.get_state())

            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_TIMEOUT_S)
                    yield _encode_sse_item(item)
                except TimeoutError:
                    if not q.empty():
                        yield _encode_sse_item(q.get_nowait())
                        continue
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=_SSE_TEST_EXIT_TIMEOUT_S)
                        yield _encode_sse_item(item)
                    except TimeoutError:
                        if not hub.get_state().connected:
                            return
                        yield b": keepalive\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            hub.unregister_sse_queue(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _encode_sse_item(item: NeurolinkState | dict) -> bytes:
    if isinstance(item, NeurolinkState):
        data = json.loads(item.model_dump_json())
        return f"event: state\ndata: {json.dumps(data)}\n\n".encode()
    if isinstance(item, dict):
        event_type = item.get("event", "unknown")
        if event_type == "baseline_complete":
            return b"event: baseline_complete\ndata: {}\n\n"
        if event_type == "settling":
            reason = item.get("reason", "settling")
            payload = json.dumps({"reason": reason})
            return f"event: settling\ndata: {payload}\n\n".encode()
        log.warning("sse_unknown_sentinel", item=item)
        return f"event: unknown\ndata: {json.dumps(item)}\n\n".encode()
    log.error("sse_unrecognised_item_type", item_type=type(item).__name__)
    return b""
