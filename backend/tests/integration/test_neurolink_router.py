"""Integration tests for /api/v1/neurolink/* endpoints.

AC coverage:
  AC2  — POST /connect returns ok + source
  AC3  — POST /disconnect returns ok
  AC4  — GET /state returns NeurolinkState schema
  AC12 — GET /sessions returns list
  AC13 — GET /eeg-gate/status returns active + frame_count
"""

from __future__ import annotations


async def test_state_endpoint_returns_neurolink_state(client):
    """AC4: GET /state should return a valid NeurolinkState schema."""
    resp = await client.get("/api/v1/neurolink/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "connected" in data
    assert "bands" in data
    assert "region" in data
    assert "alchemical_stage" in data
    assert "focus_state" in data
    assert "fatigue_score" in data
    assert "frame_count" in data


async def test_connect_mock_returns_ok(client):
    """AC2: POST /connect with mock adapter should return ok."""
    # Disconnect first since lifespan may auto-connect mock
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
    """AC3: POST /disconnect should return ok."""
    resp = await client.post("/api/v1/neurolink/disconnect")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


async def test_connect_already_connected_returns_409(client):
    """AC2: Connecting when already connected should return 409."""
    # Connect once
    await client.post("/api/v1/neurolink/disconnect")
    resp1 = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert resp1.status_code == 200

    # Connect again without disconnecting — should get 409
    resp2 = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert resp2.status_code == 409


async def test_bands_endpoint(client):
    """GET /bands should return band power fields."""
    resp = await client.get("/api/v1/neurolink/bands")
    assert resp.status_code == 200
    data = resp.json()
    assert "channel" in data
    assert "alpha" in data
    assert "theta" in data


async def test_ea1_endpoint(client):
    """GET /ea1 should return EA-1 eligibility fields."""
    resp = await client.get("/api/v1/neurolink/ea1")
    assert resp.status_code == 200
    data = resp.json()
    assert "eligible" in data
    assert "score" in data


async def test_sessions_endpoint_returns_list(client):
    """AC12: GET /sessions should return a list."""
    resp = await client.get("/api/v1/neurolink/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)


async def test_gate_status_endpoint(client):
    """AC13: GET /eeg-gate/status should return active + frame_count."""
    resp = await client.get("/api/v1/eeg-gate/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "active" in data
    assert "frame_count" in data
    assert isinstance(data["active"], bool)
    assert isinstance(data["frame_count"], int)
