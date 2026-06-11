"""Integration tests for GET /health."""
from __future__ import annotations

import pytest


async def test_health_endpoint_ok_mock_mode(client):
    """Health endpoint should return 200 and status ok in mock mode."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "adapter_connected" in data
    assert "hub_frame_count" in data
    assert "db" in data
    assert "redis" in data


async def test_health_has_adapter_type(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "adapter_type" in data
