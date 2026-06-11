"""Unit tests for calibration.py."""
from __future__ import annotations

import asyncio
import pytest

from neurolink.calibration import CalibrationSession
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


async def _pump_hub(hub: EEGHub, n: int = 40) -> None:
    """Inject synthetic frames into the hub."""
    for _ in range(n):
        payload = IngestPayload(
            source="mock",
            bands=BandPowers(alpha=0.30, theta=0.18, beta=0.12, delta=0.20, gamma=0.05),
            timestamp=1000.0,
        )
        hub.update(payload)


async def test_calibration_not_running_initially():
    hub = EEGHub()
    cal = CalibrationSession(hub)
    assert cal.is_running is False


async def test_calibrate_sets_baseline():
    """After calibration with enough frames, hub.baseline_alpha is set."""
    hub = EEGHub()
    await _pump_hub(hub, n=40)
    cal = CalibrationSession(hub)

    # Shorten calibration for test by patching _CALIBRATION_SEC
    import neurolink.calibration as cal_module
    original = cal_module._CALIBRATION_SEC
    cal_module._CALIBRATION_SEC = 1.0  # 1 second for test speed
    try:
        await cal.start()
        await asyncio.wait_for(cal.wait_for_completion(), timeout=5.0)
    finally:
        cal_module._CALIBRATION_SEC = original

    assert hub.baseline_alpha is not None
    assert isinstance(hub.baseline_alpha, float)
    assert hub.baseline_alpha > 0.0
