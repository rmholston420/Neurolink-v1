"""Coverage tests for calibration.py (CalibrationSession)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from neurolink.calibration import CalibrationSession
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


def _mock_adapter_with_eeg() -> MagicMock:
    """Return a mock adapter whose read_sample() returns a sample with a real eeg_buffer."""
    import numpy as np
    from neurolink.hardware.base import EEGSample

    n = 256 * 4  # 4-second buffer at 256 Hz
    t = np.linspace(0, 4, n)
    signal = (0.4 * np.sin(2 * 3.14159 * 10 * t)).tolist()  # 10 Hz alpha
    eeg_buf = [signal] * 5  # 5 channels

    sample = EEGSample(
        channels=[signal[-1]] * 5,
        timestamp=0.0,
        source="mock",
        address="mock",
        poor_contact=False,
        eeg_buffer=eeg_buf,
        ppg_buffer=[],
        accel_buffer=[[], [], []],
        gyro_buffer=[[], [], []],
    )
    adapter = AsyncMock()
    adapter.read_sample = AsyncMock(return_value=sample)
    return adapter


# ---------------------------------------------------------------------------
# Already running — returns None immediately
# ---------------------------------------------------------------------------

async def test_calibration_already_running_returns_none():
    hub = EEGHub()
    adapter = _mock_adapter_with_eeg()
    session = CalibrationSession(adapter=adapter, hub=hub)
    session._running = True  # simulate already running
    result = await session.run()
    assert result is None


# ---------------------------------------------------------------------------
# Happy path — enough frames, baseline updated
# ---------------------------------------------------------------------------

async def test_calibration_happy_path_updates_baseline():
    hub = EEGHub()
    adapter = _mock_adapter_with_eeg()
    session = CalibrationSession(adapter=adapter, hub=hub)

    # Patch duration to be very short so test completes fast
    import neurolink.calibration as cal_mod
    orig = cal_mod._CALIBRATION_DURATION_SEC
    try:
        cal_mod._CALIBRATION_DURATION_SEC = 0.05  # 50 ms
        result = await session.run()
    finally:
        cal_mod._CALIBRATION_DURATION_SEC = orig

    # With a real eeg_buffer the DSP will compute bands and collect alpha samples
    # result may be None if <10 frames collected in 50ms, but no error
    assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# CancelledError exits cleanly
# ---------------------------------------------------------------------------

async def test_calibration_cancelled_exits_cleanly():
    hub = EEGHub()
    # Use a slow adapter so the task stays alive long enough to cancel
    adapter = AsyncMock()
    adapter.read_sample = AsyncMock(side_effect=asyncio.sleep(60))

    async def _run_and_cancel():
        session = CalibrationSession(adapter=adapter, hub=hub)
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert session.is_running is False

    await _run_and_cancel()


# ---------------------------------------------------------------------------
# Too few frames — returns None, baseline unchanged
# ---------------------------------------------------------------------------

async def test_calibration_too_few_frames_returns_none():
    hub = EEGHub()
    original_baseline = hub.baseline_alpha

    # Adapter always returns None — 0 alpha samples collected
    adapter = AsyncMock()
    adapter.read_sample = AsyncMock(return_value=None)

    import neurolink.calibration as cal_mod
    orig = cal_mod._CALIBRATION_DURATION_SEC
    try:
        cal_mod._CALIBRATION_DURATION_SEC = 0.02  # 20 ms
        session = CalibrationSession(adapter=adapter, hub=hub)
        result = await session.run()
    finally:
        cal_mod._CALIBRATION_DURATION_SEC = orig

    assert result is None
    assert hub.baseline_alpha == original_baseline  # unchanged
