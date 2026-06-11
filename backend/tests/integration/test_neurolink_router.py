"""Integration tests for /api/v1/neurolink/* endpoints."""

from __future__ import annotations


async def test_state_endpoint_returns_neurolink_state(client):
    """GET /state should return a valid NeurolinkState schema."""
    resp = await client.get("/api/v1/neurolink/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "connected" in data
    assert "bands" in data
    assert "region" in data
    assert "alchemical_stage" in data
    assert "focus_state" in data
    assert "fatigue_score" in data


async def test_connect_mock_returns_ok(client):
    """POST /connect with mock adapter should return ok."""
    # Disconnect first since lifespan auto-connects mock
    await client.post("/api/v1/neurolink/disconnect")
    resp = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["source"] == "mock"


async def test_disconnect_returns_ok(client):
    resp = await client.post("/api/v1/neurolink/disconnect")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


async def test_bands_endpoint(client):
    resp = await client.get("/api/v1/neurolink/bands")
    assert resp.status_code == 200
    data = resp.json()
    assert "channel" in data


async def test_ea1_endpoint(client):
    resp = await client.get("/api/v1/neurolink/ea1")
    assert resp.status_code == 200
    data = resp.json()
    assert "eligible" in data
    assert "score" in data


async def test_sessions_endpoint(client):
    resp = await client.get("/api/v1/neurolink/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_gate_status_endpoint(client):
    resp = await client.get("/api/v1/gate/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "active" in data
    assert "frame_count" in data
