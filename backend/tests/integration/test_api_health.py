"""Integration tests for /health and /ready endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestHealthEndpoints:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_body_ok(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data.get("status") == "ok"

    async def test_ready_returns_200_or_503(self, client):
        """Ready may return 503 when Redis is absent — that's fine in unit env."""
        resp = await client.get("/ready")
        assert resp.status_code in (200, 503)
