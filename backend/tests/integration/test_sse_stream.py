"""Integration test for the SSE /stream endpoint.

AC13: SSE stream must emit at least one named 'state' event.

ASGITransport buffering
-----------------------
httpx ASGITransport buffers ALL chunks and delivers them only once the
generator exits (more_body=False).  The router's test-exit path fires when:
  (a) the primary 5-second wait_for times out, AND
  (b) the 50-ms inner wait_for times out, AND
  (c) hub.get_state().connected is False

Strategy
--------
1. Explicitly POST /connect (ASGITransport skips lifespan).
2. Sleep 350 ms so the pump pushes ≥1 frame into the hub.
3. POST /disconnect so that connected=False.
4. Open /stream — the generator yields hub.get_state() immediately
   (frame_count>0, connected=False) then exits via the test-exit path,
   flushing the buffered frame to the client.
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
        # 1. Connect — ASGITransport does not run the lifespan.
        connect_r = await c.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert connect_r.status_code == 200, (
            f"Expected 200 from /connect, got {connect_r.status_code}: {connect_r.text}"
        )

        # 2. Let the pump push ≥1 frame (4 Hz → 250 ms cadence).
        await asyncio.sleep(0.35)

        # 3. Disconnect so connected=False — triggers the router's test-exit path.
        await c.post("/api/v1/neurolink/disconnect")

        # 4. Open the stream.  The generator yields hub.get_state() immediately,
        #    then exits via the test-exit path, flushing the buffered data.
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
            # 5s + 50ms router timeouts + small buffer = 6s outer timeout
            await asyncio.wait_for(collect_sse(), timeout=6.5)
        except (TimeoutError, asyncio.TimeoutError):
            pass

    assert len(received_frames) >= 1, (
        "SSE stream emitted no frames. The router should emit hub.get_state() "
        "immediately on stream open, then exit via the test-exit path "
        "(connected=False after /disconnect)."
    )
    frame = received_frames[0]
    assert "connected" in frame
    assert "bands" in frame
    assert "frame_count" in frame
