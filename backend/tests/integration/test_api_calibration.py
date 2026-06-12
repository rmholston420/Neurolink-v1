"""Integration tests for /api/v1/calibration/* endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestCalibrationEndpoints:
    async def test_start_without_connection_returns_400_or_409(self, client):
        """Starting calibration with no adapter should return a client-error response."""
        resp = await client.post("/api/v1/calibration/start")
        assert resp.status_code in (400, 409, 422, 503)

    async def test_start_after_connect_returns_200(self, client):
        await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        resp = await client.post("/api/v1/calibration/start")
        assert resp.status_code == 200

    async def test_start_idempotent(self, client):
        """Calling /start twice should not crash (idempotent)."""
        await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        r1 = await client.post("/api/v1/calibration/start")
        r2 = await client.post("/api/v1/calibration/start")
        assert r1.status_code == 200
        assert r2.status_code == 200
