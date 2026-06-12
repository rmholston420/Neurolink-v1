"""Integration test for the SSE /stream endpoint.

AC13: SSE stream must emit at least one named 'state' event.

Note on ASGITransport + lifespan
---------------------------------
httpx ASGITransport does NOT call the FastAPI lifespan, so the mock
adapter and EEG pump are NOT auto-started.  We must explicitly POST
/connect to initialise the service.

The router emits ``hub.get_state()`` immediately on stream open
(guaranteed ≥1 frame) before entering the queue-wait loop, so we
receive at least one frame even without a running pump.

ASGITransport also buffers ALL chunks and only delivers them once the
generator exits (more_body=False).  The router has a built-in
test-exit path: after the primary 5s timeout and a 50ms short timeout
both expire with no new frame AND connected==False, it returns —
which flushes all buffered frames to the client.
"""

from __future__ import annotations

import asyncio
import json


async def test_sse_stream_emits_at_least_one_frame(app):
    """SSE /stream must emit at least one 'event: state' frame."""
    import httpx
    from httpx import ASGITransport

    received_frames: list[dict] = []

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        # Explicitly connect: ASGITransport skips lifespan startup.
        connect_r = await c.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert connect_r.status_code == 200, (
            f"Expected 200 from /connect, got {connect_r.status_code}: {connect_r.text}"
        )

        # Give the pump at least one tick (4 Hz → 250 ms cadence)
        await asyncio.sleep(0.35)

        async def collect_sse() -> None:
            async with c.stream("GET", "/api/v1/neurolink/stream") as response:
                assert response.status_code == 200
                pending_event: str | None = None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        pending_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw = line.split(":", 1)[1].strip()
                        if pending_event in ("state", None):
                            try:
                                received_frames.append(json.loads(raw))
                            except json.JSONDecodeError:
                                pass
                        pending_event = None
                        if received_frames:
                            return

        try:
            await asyncio.wait_for(collect_sse(), timeout=10.0)
        except (TimeoutError, asyncio.TimeoutError):
            pass

    assert len(received_frames) >= 1, (
        "SSE stream emitted no frames. The router should emit hub.get_state() "
        "immediately on stream open."
    )
    frame = received_frames[0]
    assert "connected" in frame
    assert "bands" in frame
    assert "frame_count" in frame
