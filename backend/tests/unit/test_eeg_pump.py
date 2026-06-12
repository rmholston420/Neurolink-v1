"""Unit tests for EEGPump — start/stop lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hardware.mock import MockAdapter
from neurolink.hub import EEGHub


class TestEEGPump:
    @pytest.mark.asyncio
    async def test_start_stop_no_exception(self):
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        pump = EEGPump(adapter, hub, publish_hz=4)
        await pump.start()
        await asyncio.sleep(0.3)  # let it tick a few frames
        await pump.stop()
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_pump_increments_frame_count(self):
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        pump = EEGPump(adapter, hub, publish_hz=10)
        await pump.start()
        await asyncio.sleep(0.4)
        await pump.stop()
        await adapter.disconnect()
        assert hub.get_state().frame_count >= 1

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        hub = EEGHub()
        adapter = MockAdapter()
        await adapter.connect()
        pump = EEGPump(adapter, hub, publish_hz=4)
        await pump.start()
        await pump.stop()
        await pump.stop()  # second stop should not raise
        await adapter.disconnect()
