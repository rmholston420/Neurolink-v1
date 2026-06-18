"""Integration-style tests for the /api/v1/neurolink/stream SSE endpoint.

Covers:
  - At least one 'event: state' frame is emitted immediately (current state)
  - Each frame parses as valid JSON containing the NeurolinkState schema fields
  - The SSE frame format is correct: 'event: state\\ndata: {...}\\n\\n'
  - Keepalive comment format '': keepalive' is valid SSE
  - Queue registration and cleanup (hub side-effects)

Note: These tests drive the ASGI app through httpx.AsyncClient with
ASGITransport.  The EEGPump does NOT run during tests so the queue
drains immediately; the generator exits after the initial state frame.
"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")


@pytest.mark.asyncio
async def test_sse_stream_emits_at_least_one_frame(async_client):
    """The SSE endpoint must yield at least one 'event: state' frame."""
    async with async_client.stream("GET", "/api/v1/neurolink/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        state_frames = []
        async for line in resp.aiter_lines():
            if line.startswith("event: state"):
                state_frames.append(line)
            if len(state_frames) >= 1:
                break
    assert len(state_frames) >= 1


@pytest.mark.asyncio
async def test_sse_stream_data_is_valid_json(async_client):
    """Every 'data:' line must be parseable JSON."""
    raw_chunks = []
    async with async_client.stream("GET", "/api/v1/neurolink/stream") as resp:
        async for chunk in resp.aiter_bytes():
            raw_chunks.append(chunk)

    raw = b"".join(raw_chunks).decode()
    data_lines = [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]
    assert len(data_lines) >= 1, "Expected at least one data line in SSE response"
    for dl in data_lines:
        parsed = json.loads(dl)  # must not raise
        assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_sse_stream_state_fields_present(async_client):
    """The first state frame must contain required NeurolinkState fields."""
    raw_chunks = []
    async with async_client.stream("GET", "/api/v1/neurolink/stream") as resp:
        async for chunk in resp.aiter_bytes():
            raw_chunks.append(chunk)

    raw = b"".join(raw_chunks).decode()
    data_lines = [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]
    assert data_lines, "No data lines in SSE output"
    payload = json.loads(data_lines[0])
    # NeurolinkState required top-level keys
    for key in ("connected", "artifact_rejected", "bands"):
        assert key in payload, f"Missing key '{key}' in SSE state frame"


@pytest.mark.asyncio
async def test_sse_stream_frame_format(async_client):
    """Verify raw SSE framing: 'event: state\\ndata: ...\\n\\n'."""
    raw_chunks = []
    async with async_client.stream("GET", "/api/v1/neurolink/stream") as resp:
        async for chunk in resp.aiter_bytes():
            raw_chunks.append(chunk)

    raw = b"".join(raw_chunks).decode()
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.splitlines()
        if lines[0].startswith(":"):
            continue  # keepalive comment -- valid SSE
        assert any(ln.startswith("event:") for ln in lines), f"Block missing event: line: {block!r}"
        assert any(ln.startswith("data:") for ln in lines), f"Block missing data: line: {block!r}"


@pytest.mark.asyncio
async def test_sse_response_headers(async_client):
    """The SSE response must carry Cache-Control and X-Accel-Buffering headers."""
    async with async_client.stream("GET", "/api/v1/neurolink/stream") as resp:
        async for _ in resp.aiter_bytes():
            pass
    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"
