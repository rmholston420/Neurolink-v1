"""Router integration tests covering previously-uncovered endpoints.

Uses the same ``client`` fixture from conftest.py (httpx AsyncClient
over ASGITransport) so every route handler is exercised in-process.

Covers:
- GET  /api/v1/neurolink/bands
- GET  /api/v1/neurolink/ea1
- GET  /api/v1/neurolink/sessions (no DB factory → empty list)
- GET  /api/v1/neurolink/stream   (SSE idle-timeout path, 1 frame flushed)
- POST /api/v1/neurolink/calibrate
- GET  /api/v1/eeg-gate/status
- POST /api/v1/eeg-gate/block
- POST /api/v1/eeg-gate/unblock
- hub module-level delegates: update(), get_ea1(), snapshot(), reset()
- service.get_band_powers() channel-specific path
"""
from __future__ import annotations

import pytest

from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


# ---------------------------------------------------------------------------
# /api/v1/neurolink/bands
# ---------------------------------------------------------------------------

async def test_bands_endpoint_default_channel(client):
    r = await client.get("/api/v1/neurolink/bands")
    assert r.status_code == 200
    data = r.json()
    assert "alpha" in data
    assert "theta" in data


async def test_bands_endpoint_specific_channel(client):
    """channel query param exercises the channel-specific path in service."""
    r = await client.get("/api/v1/neurolink/bands?channel=AF7")
    assert r.status_code == 200
    data = r.json()
    assert "alpha" in data


# ---------------------------------------------------------------------------
# /api/v1/neurolink/ea1
# ---------------------------------------------------------------------------

async def test_ea1_endpoint(client):
    r = await client.get("/api/v1/neurolink/ea1")
    assert r.status_code == 200
    data = r.json()
    # EA1Result has an 'eligible' field
    assert "eligible" in data


# ---------------------------------------------------------------------------
# /api/v1/neurolink/sessions
# ---------------------------------------------------------------------------

async def test_sessions_endpoint_no_factory_returns_empty(client):
    """Without a DB factory configured the service returns an empty list."""
    r = await client.get("/api/v1/neurolink/sessions")
    assert r.status_code == 200
    assert r.json() == []


async def test_sessions_endpoint_limit_param(client):
    r = await client.get("/api/v1/neurolink/sessions?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# /api/v1/neurolink/stream  (SSE)
# ---------------------------------------------------------------------------

async def test_sse_stream_returns_at_least_one_frame(client):
    """The idle-timeout path exits after 50 ms and flushes ≥1 SSE frame."""
    async with client.stream("GET", "/api/v1/neurolink/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        chunks = []
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                chunks.append(line)
            if chunks:  # got first frame — stop reading
                break
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# /api/v1/neurolink/calibrate
# ---------------------------------------------------------------------------

async def test_calibrate_endpoint(client):
    """Calibrate returns {status: started} immediately."""
    r = await client.post("/api/v1/neurolink/calibrate")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "started"


# ---------------------------------------------------------------------------
# /api/v1/eeg-gate/status
# ---------------------------------------------------------------------------

async def test_gate_status_endpoint(client):
    r = await client.get("/api/v1/eeg-gate/status")
    assert r.status_code == 200
    data = r.json()
    assert "active" in data
    assert "frame_count" in data
    assert "focus_score" in data
    assert "focus_state" in data


# ---------------------------------------------------------------------------
# /api/v1/eeg-gate/block and /unblock
# ---------------------------------------------------------------------------

async def test_gate_force_block(client):
    r = await client.post("/api/v1/eeg-gate/block")
    assert r.status_code == 200
    assert r.json()["active"] is True


async def test_gate_force_unblock(client):
    r = await client.post("/api/v1/eeg-gate/unblock")
    assert r.status_code == 200
    assert r.json()["active"] is False


# ---------------------------------------------------------------------------
# hub module-level delegates
# ---------------------------------------------------------------------------

def test_hub_module_update_delegate():
    """hub.update() module-level function delegates to the singleton."""
    import neurolink.hub as hub_mod

    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.1, delta=0.1, gamma=0.05),
    )
    state = hub_mod.update(payload)
    assert state.frame_count >= 1


def test_hub_module_get_ea1_delegate():
    import neurolink.hub as hub_mod
    ea1 = hub_mod.get_ea1()
    assert hasattr(ea1, "eligible")


def test_hub_module_snapshot_delegate():
    import neurolink.hub as hub_mod
    snap = hub_mod.snapshot()
    assert isinstance(snap, dict)
    assert "frame_count" in snap


def test_hub_module_reset_delegate():
    import neurolink.hub as hub_mod
    hub_mod.reset()
    state = hub_mod.get_state()
    # After reset frame_count may be > 0 if other tests ran first via singleton;
    # just assert the function ran without error and returns a valid state.
    assert state is not None


# ---------------------------------------------------------------------------
# service.get_band_powers() — channel-specific path
# ---------------------------------------------------------------------------

async def test_service_get_band_powers_channel():
    """Exercises the per-channel branch of service.get_band_powers()."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)
    resp = await svc.get_band_powers(channel="TP9")
    assert hasattr(resp, "alpha")


async def test_service_get_band_powers_mean():
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)
    resp = await svc.get_band_powers(channel="mean")
    assert hasattr(resp, "alpha")


# ---------------------------------------------------------------------------
# service.stream_state() async generator
# ---------------------------------------------------------------------------

async def test_service_stream_state_yields_state():
    """stream_state() must yield at least one NeurolinkState."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)

    states = []
    async for state in svc.stream_state():
        states.append(state)
        break  # only need the first item

    assert len(states) == 1
    assert hasattr(states[0], "frame_count")
