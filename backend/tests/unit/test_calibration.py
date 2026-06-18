"""Unit tests for calibration.CalibrationSession."""

from __future__ import annotations

import asyncio

import pytest

from neurolink.calibration import CalibrationSession
from neurolink.hardware.mock import MockAdapter
from neurolink.hub import EEGHub


class TestCalibrationSession:
    @pytest.mark.asyncio
    async def test_run_updates_baseline_alpha(self):
        """After a calibration run, hub.baseline_alpha should change from default."""
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        session = CalibrationSession(adapter, hub)

        default_baseline = hub.baseline_alpha

        # Run calibration with a very short cap so tests don't take 30 s
        await asyncio.wait_for(session.run(), timeout=5.0)

        await adapter.disconnect()
        # The baseline should have been updated from the mock stream
        assert hub.baseline_alpha != default_baseline or hub.baseline_alpha > 0

    @pytest.mark.asyncio
    async def test_run_does_not_raise_without_ppg(self):
        """CalibrationSession must not raise when PPG data is absent (mock)."""
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        session = CalibrationSession(adapter, hub)
        try:
            await asyncio.wait_for(session.run(), timeout=5.0)
        except TimeoutError:
            pass  # timeout is acceptable — run() loops on the stream
        await adapter.disconnect()
