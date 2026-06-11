"""Integration tests for /api/v1/neurolink/* endpoints."""
from __future__ import annotations

import pytest


async def test_connect_mock_returns_ok(client):
    """AC2: POST /api/v1/neurolink/connect returns ok=true for mock."""
    response = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "muse_s_gen1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["source"] == "mock"


async def test_state_endpoint_returns_neurolink_state(client):
    """AC3: GET /api/v1/neurolink/state returns NeurolinkState."""
    response = await client.get("/api/v1/neurolink/state")
    assert response.status_code == 200
    data = response.json()
    # Check required fields from NeurolinkState
    assert "connected" in data
    assert "region" in data
    assert "bands" in data
    assert "ea1" in data
    assert "focus_state" in data
    assert "fatigue_score" in data


async def test_bands_endpoint(client):
    response = await client.get("/api/v1/neurolink/bands")
    assert response.status_code == 200
    data = response.json()
    assert "channel" in data


async def test_ea1_endpoint(client):
    response = await client.get("/api/v1/neurolink/ea1")
    assert response.status_code == 200
    data = response.json()
    assert "eligible" in data
    assert "score" in data


async def test_disconnect_returns_ok(client):
    response = await client.post("/api/v1/neurolink/disconnect")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


async def test_sessions_endpoint(client):
    response = await client.get("/api/v1/neurolink/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
