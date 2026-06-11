"""Unit tests for EEGPump — start, stop, tick with a mock adapter."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hardware.base import EEGSample
from neurolink.hub import EEGHub


class _ConstantAdapter:
    """Minimal adapter that always returns a fixed EEGSample."""

    is_connected = True
    source = "mock"
    address = ""

    def _make_sample(self) -> EEGSample:
        fs = 256
        t = np.linspace(0, 1, fs, endpoint=False).astype(np.float32)
        ch = np.sin(2 * np.pi * 10 * t)
        eeg = [ch.tolist() for _ in range(5)]
        return EEGSample(
            source="mock",
            address="",
            timestamp=0.0,
            eeg_buffer=eeg,
            ppg_buffer=[],
            accel_buffer=[[], [], []],
            gyro_buffer=[[], [], []],
            poor_contact=False,
            extra={},
        )

    async def read_sample(self) -> EEGSample:
        return self._make_sample()


class _NullAdapter:
    """Adapter that always returns None (no data available)."""

    is_connected = True
    source = "mock"
    address = ""

    async def read_sample(self) -> None:
        return None


async def test_pump_start_stop():
    hub = EEGHub()
    adapter = _ConstantAdapter()
    pump = EEGPump(adapter, hub, publish_hz=100.0)
    await pump.start()
    assert pump._running is True
    await asyncio.sleep(0.05)
    await pump.stop()
    assert pump._running is False
    assert hub.get_state().frame_count >= 1


async def test_pump_null_adapter_does_not_crash():
    hub = EEGHub()
    adapter = _NullAdapter()
    pump = EEGPump(adapter, hub, publish_hz=100.0)
    await pump.start()
    await asyncio.sleep(0.05)
    await pump.stop()
    # No frames — null adapter returns None every tick
    assert hub.get_state().frame_count == 0


async def test_pump_stop_before_start_is_safe():
    hub = EEGHub()
    pump = EEGPump(_NullAdapter(), hub)
    await pump.stop()  # should not raise


async def test_pump_multiple_stop_calls_are_safe():
    hub = EEGHub()
    pump = EEGPump(_ConstantAdapter(), hub, publish_hz=100.0)
    await pump.start()
    await pump.stop()
    await pump.stop()  # second stop should be a no-op
