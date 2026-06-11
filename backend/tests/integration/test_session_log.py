"""Integration tests for session log."""
from __future__ import annotations

import pytest


async def test_session_log_created_on_connect(client):
    """AC12: A session log entry is created after connect."""
    # Connect
    resp = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "muse_s_gen1"},
    )
    assert resp.status_code == 200

    # Check sessions list
    sessions_resp = await client.get("/api/v1/neurolink/sessions")
    assert sessions_resp.status_code == 200
    sessions = sessions_resp.json()
    assert len(sessions) >= 1
    assert sessions[0]["adapter_type"] == "mock"
    assert sessions[0]["device_model"] == "muse_s_gen1"


async def test_calibrate_starts_and_returns_started(client):
    """Calibration endpoint should return status=started."""
    resp = await client.post("/api/v1/neurolink/calibrate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
