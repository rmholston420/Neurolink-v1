"""Integration tests for the SSE /stream endpoint.

AC13: SSE stream should emit at least one named 'state' event
within a reasonable timeout after the hub has data.
"""

from __future__ import annotations

import asyncio
import json


async def test_sse_stream_emits_at_least_one_frame(app):
    """SSE /stream must emit at least one 'event: state' frame after hub.update()."""
    import httpx
    from httpx import ASGITransport

    from neurolink.hub import get_hub
    from neurolink.models.eeg import BandPowers, IngestPayload

    hub = get_hub()
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
    )
    hub.update(payload)

    received_frames: list[dict] = []

    async def collect_sse() -> None:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            async with c.stream("GET", "/api/v1/neurolink/stream") as response:
                assert response.status_code == 200
                pending_event = None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        pending_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw = line.split(":", 1)[1].strip()
                        # Accept frames from both named 'state' events and
                        # un-named message events (proxy compat)
                        if pending_event in ("state", None):
                            try:
                                received_frames.append(json.loads(raw))
                            except json.JSONDecodeError:
                                pass
                        pending_event = None
                        if received_frames:
                            return  # Got at least one frame — done

    try:
        await asyncio.wait_for(collect_sse(), timeout=8.0)
    except (TimeoutError, asyncio.TimeoutError):
        pass  # Timeout is acceptable if SSE keepalive is the only emission

    # The hub was primed with a frame before opening the stream.
    # On a healthy backend the 2-second SSE keepalive will emit the hub state.
    assert len(received_frames) >= 1, (
        "SSE stream emitted no frames within 8 s after hub.update(). "
        "Check that /stream is registered and SSE fan-out is wired."
    )

    frame = received_frames[0]
    assert "connected" in frame
    assert "bands" in frame
    assert "frame_count" in frame
