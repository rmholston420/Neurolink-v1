"""Integration test for the SSE /stream endpoint.

AC13: SSE stream must emit at least one named 'state' event.

Strategy: the app fixture lifespan auto-connects the mock adapter and
starts the EEG pump.  Open /stream immediately and wait for the first
frame.  The event_generator yields the current hub state on the first
tick (~250 ms at 4 Hz), so a 5-second timeout is generous.

Do NOT call POST /connect here — the lifespan already connected; a
second connect call would return 409 and abort the test.
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
        # Give the pump one tick to push at least one frame (4 Hz → 250 ms)
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
            await asyncio.wait_for(collect_sse(), timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            pass

    assert len(received_frames) >= 1, (
        "SSE stream emitted no frames within 5 seconds. "
        "Check that EEGPump is running and hub.get_state() is non-empty."
    )
    frame = received_frames[0]
    assert "connected" in frame
    assert "bands" in frame
    assert "frame_count" in frame
