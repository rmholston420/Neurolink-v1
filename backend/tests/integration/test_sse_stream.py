"""Integration test for SSE stream."""
from __future__ import annotations

import asyncio
import json
import pytest


async def test_sse_stream_emits_at_least_one_frame(app):
    """AC4: SSE stream emits at least one data frame within 5 seconds."""
    from httpx import AsyncClient, ASGITransport
    from neurolink.hub import get_hub
    from neurolink.models.eeg import BandPowers, IngestPayload

    hub = get_hub()

    async def inject_frames():
        await asyncio.sleep(0.2)
        for _ in range(5):
            payload = IngestPayload(
                source="mock",
                bands=BandPowers(alpha=0.30, theta=0.18, beta=0.12, delta=0.20, gamma=0.05),
                timestamp=1000.0,
            )
            hub.update(payload)
            await asyncio.sleep(0.1)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        timeout=10.0,
    ) as ac:
        # Start frame injection concurrently
        inject_task = asyncio.create_task(inject_frames())
        try:
            received = False
            async with ac.stream("GET", "/api/v1/neurolink/stream") as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if payload_str:
                            data = json.loads(payload_str)
                            assert "connected" in data
                            received = True
                            break
                    if received:
                        break
            assert received, "No SSE data frame received within timeout"
        finally:
            inject_task.cancel()
            try:
                await inject_task
            except asyncio.CancelledError:
                pass
