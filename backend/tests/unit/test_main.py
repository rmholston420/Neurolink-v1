"""Unit tests for main.py: create_app, routes, lifespan."""

from __future__ import annotations


def test_create_app_returns_fastapi_app(app):
    from fastapi import FastAPI

    assert isinstance(app, FastAPI.__class__) or hasattr(app, "routes")


async def test_health_endpoint_via_app(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "adapter_type" in data


async def test_state_endpoint_via_app(client):
    r = await client.get("/api/v1/neurolink/state")
    assert r.status_code == 200
    data = r.json()
    assert "connected" in data
    assert "frame_count" in data


async def test_unknown_route_returns_404(client):
    r = await client.get("/api/v1/does_not_exist")
    assert r.status_code == 404


async def test_connect_endpoint_via_app(client):
    r = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


async def test_disconnect_endpoint_via_app(client):
    await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    r = await client.post("/api/v1/neurolink/disconnect")
    assert r.status_code == 200
    assert r.json()["ok"] is True
