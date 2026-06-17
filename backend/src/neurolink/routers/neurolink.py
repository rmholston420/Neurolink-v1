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
    """SSE endpoint - streams EEG events as server-sent events.

    Three event types are multiplexed on the same stream; see module
    docstring for the full event-type table.

    Keepalive strategy
    ------------------
    The pump publishes at 4 Hz (250 ms cadence).  The generator waits up
    to _SSE_IDLE_TIMEOUT_S (5 s) for the next queued item.  On timeout
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
            # Emit current state immediately (unconditionally) so tests
            # always receive at least one frame even when no pump is running
            # and hub.connected is False.
            yield _encode_sse_item(hub.get_state())

            while True:
                try:
                    # Primary wait — up to 5 s for the next hub item.
                    item = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_TIMEOUT_S)
                    yield _encode_sse_item(item)
                except TimeoutError:
                    # 5 s passed with no item.  Drain any burst that arrived
                    # while we were not watching, then keepalive.
                    if not q.empty():
                        yield _encode_sse_item(q.get_nowait())
                        continue

                    # Try the short test-exit window to detect ASGITransport.
                    try:
                        item = await asyncio.wait_for(
                            q.get(), timeout=_SSE_TEST_EXIT_TIMEOUT_S
                        )
                        yield _encode_sse_item(item)
                    except TimeoutError:
                        # No pump running (test harness) -> exit cleanly.
                        # Under uvicorn the pump always fills the queue within
                        # 250 ms so this inner branch is never reached in prod.
                        if not hub.get_state().connected:
                            return
                        # Still connected but genuinely idle — send keepalive
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


# ── SSE serialisation helpers ───────────────────────────────────────────────

def _encode_sse_item(item: NeurolinkState | dict) -> bytes:
    """Encode any hub queue item as a raw SSE frame (bytes).

    Dispatches on item type:

      NeurolinkState
        -> event: state
           data: <model_dump_json>

      {"event": "baseline_complete"}
        -> event: baseline_complete
           data: {}

      {"event": "settling", "reason": <code>}
        -> event: settling
           data: {"reason": <code>}

      Any other dict (forwards-compatible)
        -> event: unknown
           data: <raw JSON>

    Returns bytes ready to write directly into the SSE stream.
    """
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

        # Forwards-compatible fallback: unknown sentinel shape.
        log.warning("sse_unknown_sentinel", item=item)
        return f"event: unknown\ndata: {json.dumps(item)}\n\n".encode()

    # Should never happen — log and drop.
    log.error("sse_unrecognised_item_type", item_type=type(item).__name__)
    return b""
