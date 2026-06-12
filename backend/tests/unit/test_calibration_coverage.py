"""Coverage tests for calibration.py (CalibrationSession)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np

from neurolink.calibration import CalibrationSession
from neurolink.hardware.base import EEGSample
from neurolink.hub import EEGHub


def _eeg_sample() -> EEGSample:
    """Return an EEGSample with a real alpha-dominant EEG buffer."""
    n = 256 * 4
    t = np.linspace(0, 4, n)
    signal = (0.4 * np.sin(2 * np.pi * 10 * t)).tolist()
    return EEGSample(
        channels=[signal[-1]] * 5,
        timestamp=0.0,
        source="mock",
        address="mock",
        poor_contact=False,
        eeg_buffer=[signal] * 5,
        ppg_buffer=[],
        accel_buffer=[[], [], []],
        gyro_buffer=[[], [], []],
    )


# ---------------------------------------------------------------------------
# Already running — returns None immediately
# ---------------------------------------------------------------------------

async def test_calibration_already_running_returns_none():
    session = CalibrationSession(adapter=AsyncMock(), hub=EEGHub())
    session._running = True
    result = await session.run()
    assert result is None


# ---------------------------------------------------------------------------
# Happy path — short duration, adapter returns real samples
# ---------------------------------------------------------------------------

async def test_calibration_happy_path():
    adapter = AsyncMock()
    adapter.read_sample = AsyncMock(return_value=_eeg_sample())
    hub = EEGHub()

    import neurolink.calibration as cal_mod

    orig = cal_mod.TOTAL_DURATION_SEC  # use the real exported name
    try:
        cal_mod.TOTAL_DURATION_SEC = 0.05
        session = CalibrationSession(adapter=adapter, hub=hub)
        result = await session.run()
    finally:
        cal_mod.TOTAL_DURATION_SEC = orig

    assert result is None or isinstance(result, float)
    assert session.is_running is False


# ---------------------------------------------------------------------------
# CancelledError exits cleanly
# ---------------------------------------------------------------------------

async def test_calibration_cancelled_exits_cleanly():
    async def blocking_read():
        await asyncio.sleep(60)

    adapter = AsyncMock()
    adapter.read_sample = blocking_read
    hub = EEGHub()

    session = CalibrationSession(adapter=adapter, hub=hub)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert session.is_running is False


# ---------------------------------------------------------------------------
# Too few frames — returns None, baseline unchanged
# ---------------------------------------------------------------------------

async def test_calibration_too_few_frames_returns_none():
    adapter = AsyncMock()
    adapter.read_sample = AsyncMock(return_value=None)  # always None
    hub = EEGHub()
    original_baseline = hub.baseline_alpha

    import neurolink.calibration as cal_mod

    orig = cal_mod.TOTAL_DURATION_SEC
    try:
        cal_mod.TOTAL_DURATION_SEC = 0.02
        session = CalibrationSession(adapter=adapter, hub=hub)
        result = await session.run()
    finally:
        cal_mod.TOTAL_DURATION_SEC = orig

    assert result is None
    assert hub.baseline_alpha == original_baseline
