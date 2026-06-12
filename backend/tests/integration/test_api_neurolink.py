"""Integration tests for /api/v1/neurolink/* endpoints."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestConnect:
    async def test_connect_mock_returns_200(self, client):
        resp = await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert resp.status_code == 200

    async def test_connect_response_ok_true(self, client):
        resp = await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert resp.json()["ok"] is True

    async def test_connect_already_connected_returns_409(self, client):
        await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        resp = await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        assert resp.status_code == 409


@pytest.mark.asyncio
class TestDisconnect:
    async def test_disconnect_after_connect_returns_200(self, client):
        await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        resp = await client.post("/api/v1/neurolink/disconnect")
        assert resp.status_code == 200

    async def test_disconnect_response_ok_true(self, client):
        await client.post(
            "/api/v1/neurolink/connect",
            json={"adapter_type": "mock", "device_model": "mock"},
        )
        resp = await client.post("/api/v1/neurolink/disconnect")
        assert resp.json()["ok"] is True


@pytest.mark.asyncio
class TestStateEndpoint:
    async def test_state_returns_200(self, client):
        resp = await client.get("/api/v1/neurolink/state")
        assert resp.status_code == 200

    async def test_state_has_connected_field(self, client):
        resp = await client.get("/api/v1/neurolink/state")
        data = resp.json()
        assert "connected" in data

    async def test_state_connected_false_before_connect(self, client):
        resp = await client.get("/api/v1/neurolink/state")
        assert resp.json()["connected"] is False


@pytest.mark.asyncio
class TestBandsEndpoint:
    async def test_bands_returns_200(self, client):
        resp = await client.get("/api/v1/neurolink/bands")
        assert resp.status_code == 200

    async def test_bands_has_alpha(self, client):
        resp = await client.get("/api/v1/neurolink/bands")
        data = resp.json()
        assert "alpha" in data


@pytest.mark.asyncio
class TestEA1Endpoint:
    async def test_ea1_returns_200(self, client):
        resp = await client.get("/api/v1/neurolink/ea1")
        assert resp.status_code == 200

    async def test_ea1_has_eligible(self, client):
        resp = await client.get("/api/v1/neurolink/ea1")
        data = resp.json()
        assert "eligible" in data


@pytest.mark.asyncio
class TestSessionsEndpoint:
    async def test_sessions_returns_200(self, client):
        resp = await client.get("/api/v1/neurolink/sessions")
        assert resp.status_code == 200

    async def test_sessions_returns_list(self, client):
        resp = await client.get("/api/v1/neurolink/sessions")
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
class TestSSEStream:
    async def test_stream_returns_first_event(self, client):
        """SSE /stream should return at least one event: state frame."""
        async with client.stream("GET", "/api/v1/neurolink/stream") as response:
            assert response.status_code == 200
            lines = []
            async for line in response.aiter_lines():
                lines.append(line)
                if any(l.startswith("data:") for l in lines):
                    break
        assert any(l.startswith("data:") for l in lines)

    async def test_stream_content_type_is_sse(self, client):
        async with client.stream("GET", "/api/v1/neurolink/stream") as response:
            ct = response.headers.get("content-type", "")
            assert "text/event-stream" in ct
