"""Integration tests for GET /api/v1/neurolink/baseline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neurolink.calibration import TOTAL_DURATION_SEC


@pytest.mark.asyncio
class TestBaselineProgressEndpoint:
    """Tests for the lightweight baseline-progress polling endpoint."""

    async def test_idle_before_calibration_started(self, client):
        """Before any calibration is started, phase must be 'idle'."""
        resp = await client.get("/api/v1/neurolink/baseline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "idle"
        assert body["elapsed_s"] == 0.0
        assert body["remaining_s"] == 0.0
        assert body["total_s"] == TOTAL_DURATION_SEC

    async def test_warmup_phase_in_progress(self, client):
        """Simulate an in-progress warmup by injecting a mock CalibrationSession."""
        import neurolink.dependencies as deps

        service = deps.get_service()

        mock_sess = MagicMock()
        mock_sess.phase = "warmup"
        mock_sess.elapsed = 10.0
        service._calibration_session = mock_sess
        service._calibration_task = MagicMock(done=lambda: False)

        resp = await client.get("/api/v1/neurolink/baseline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "warmup"
        assert body["elapsed_s"] == 10.0
        assert body["remaining_s"] == round(TOTAL_DURATION_SEC - 10.0, 2)
        assert body["total_s"] == TOTAL_DURATION_SEC

    async def test_baseline_phase_in_progress(self, client):
        """Simulate the active baseline-capture window (30-90 s)."""
        import neurolink.dependencies as deps

        service = deps.get_service()

        mock_sess = MagicMock()
        mock_sess.phase = "baseline"
        mock_sess.elapsed = 45.0
        service._calibration_session = mock_sess
        service._calibration_task = MagicMock(done=lambda: False)

        resp = await client.get("/api/v1/neurolink/baseline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "baseline"
        assert body["elapsed_s"] == 45.0
        assert body["remaining_s"] == round(TOTAL_DURATION_SEC - 45.0, 2)

    async def test_complete_phase_when_task_done(self, client):
        """When the calibration task is done, phase must be 'complete'."""
        import neurolink.dependencies as deps

        service = deps.get_service()

        mock_sess = MagicMock()
        mock_sess.phase = "complete"
        mock_sess.elapsed = TOTAL_DURATION_SEC
        service._calibration_session = mock_sess
        service._calibration_task = MagicMock(done=lambda: True)

        resp = await client.get("/api/v1/neurolink/baseline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "complete"
        assert body["remaining_s"] == 0.0
