"""Integration tests for /health endpoint."""
from __future__ import annotations

import pytest


async def test_health_endpoint_ok_mock_mode(client):
    """AC1: /health returns 200 in mock mode."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "adapter_type" in data
    assert "hub_frame_count" in data
    assert "redis" in data
    assert "db" in data


async def test_health_adapter_type_is_mock(client):
    response = await client.get("/health")
    data = response.json()
    assert data["adapter_type"] == "mock"
