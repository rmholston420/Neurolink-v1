"""Unit tests for calibration.py."""
from __future__ import annotations

import asyncio
import pytest

from neurolink.calibration import CalibrationSession
from neurolink.hub import EEGHub
from neurolink.hardware.mock import MockAdapter


async def test_calibration_sets_baseline_alpha():
    """Calibration should set hub.baseline_alpha to a positive float."""
    import neurolink.calibration as cal_module
    # Override duration for test speed
    original = cal_module._CALIBRATION_DURATION_SEC
    cal_module._CALIBRATION_DURATION_SEC = 1.5  # 1.5 seconds for test
    cal_module._MIN_FRAMES = 2
    try:
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        session = CalibrationSession(adapter=adapter, hub=hub)
        baseline = await session.run()
        assert baseline is not None
        assert isinstance(baseline, float)
        assert baseline > 0.0
        assert hub.baseline_alpha == baseline
    finally:
        cal_module._CALIBRATION_DURATION_SEC = original
        cal_module._MIN_FRAMES = 10
        await adapter.disconnect()


async def test_calibration_returns_none_for_no_data():
    """Returns None when adapter gives no EEG buffer."""
    from unittest.mock import AsyncMock, MagicMock
    from neurolink.hardware.base import EEGSample
    import time

    hub = EEGHub()
    adapter = MagicMock()
    adapter.read_sample = AsyncMock(
        return_value=EEGSample(channels=[0.0] * 5, eeg_buffer=None)
    )

    import neurolink.calibration as cal_module
    original_dur = cal_module._CALIBRATION_DURATION_SEC
    original_min = cal_module._MIN_FRAMES
    cal_module._CALIBRATION_DURATION_SEC = 0.2
    cal_module._MIN_FRAMES = 100  # impossible to meet
    try:
        session = CalibrationSession(adapter=adapter, hub=hub)
        result = await session.run()
        assert result is None
    finally:
        cal_module._CALIBRATION_DURATION_SEC = original_dur
        cal_module._MIN_FRAMES = original_min
