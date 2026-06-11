"""Integration test for the SSE /stream endpoint.

AC13: SSE stream must emit at least one named 'state' event.

Strategy: connect via POST /connect (starts mock adapter + pump), then
open /stream and wait for the first frame.  The event_generator yields
the current hub state immediately on connect, so this should be
instantaneous.
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
        # Start mock adapter so the pump begins pushing frames into this app's hub
        connect_r = await c.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert connect_r.status_code == 200

        # Give the pump one tick to push at least one frame
        await asyncio.sleep(0.1)

        async def collect_sse() -> None:
            async with c.stream("GET", "/api/v1/neurolink/stream") as response:
                assert response.status_code == 200
                pending_event = None
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
            await asyncio.wait_for(collect_sse(), timeout=5.0)
        except TimeoutError:
            pass

    assert len(received_frames) >= 1, (
        "SSE stream emitted no frames. The event_generator should yield the "
        "current hub state immediately on connect."
    )
    frame = received_frames[0]
    assert "connected" in frame
    assert "bands" in frame
    assert "frame_count" in frame
