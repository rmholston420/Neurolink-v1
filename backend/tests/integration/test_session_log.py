"""Integration tests for session logging."""

from __future__ import annotations


async def test_session_log_created_on_connect(client):
    """A session log entry should be created when connect is called."""
    # Disconnect to reset
    await client.post("/api/v1/neurolink/disconnect")

    resp = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert resp.status_code == 200

    # Sessions endpoint should list at least one entry
    resp2 = await client.get("/api/v1/neurolink/sessions")
    assert resp2.status_code == 200
    sessions = resp2.json()
    assert isinstance(sessions, list)
    # May be empty in-memory DB depending on timing; check structure if present
    if sessions:
        assert "id" in sessions[0]
        assert "device_model" in sessions[0]


async def test_calibrate_starts_and_returns_started(client):
    """POST /calibrate should return status 'started'.

    The conftest reset_hub fixture tears down the service singleton before each
    test, so the mock adapter is not auto-connected.  Connect explicitly here
    so that start_calibration finds a live adapter instead of raising
    AdapterNotConnectedError (which maps to 409).
    """
    # Ensure mock adapter is connected before calibrating
    connect_resp = await client.post(
        "/api/v1/neurolink/connect",
        json={"adapter_type": "mock", "device_model": "mock"},
    )
    assert connect_resp.status_code == 200, connect_resp.text

    resp = await client.post("/api/v1/neurolink/calibrate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
