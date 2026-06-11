"""Coverage tests for EEGPump."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hub import EEGHub


def _hub() -> EEGHub:
    return EEGHub()


def _pump(hub=None, adapter=None, hz=4.0) -> EEGPump:
    return EEGPump(adapter=adapter or AsyncMock(), hub=hub or _hub(), publish_hz=hz)


# ---------------------------------------------------------------------------
# __init__ field inspection
# ---------------------------------------------------------------------------

def test_pump_init_fields():
    hub = _hub()
    adapter = AsyncMock()
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=4.0)
    assert pump._hub is hub
    assert pump._adapter is adapter
    assert pump._running is False
    assert pump._publish_hz == 4.0


# ---------------------------------------------------------------------------
# stop() while task is None (not started)
# ---------------------------------------------------------------------------

async def test_stop_while_not_started_is_noop():
    pump = _pump()
    await pump.stop()  # _task is None, must not raise
    assert pump._running is False


# ---------------------------------------------------------------------------
# _tick() with None sample (continue branch)
# ---------------------------------------------------------------------------

async def test_tick_none_sample_is_noop():
    """_tick() returns immediately when adapter.read_sample() returns None."""
    hub = _hub()
    adapter = AsyncMock()
    adapter.read_sample.return_value = None
    pump = EEGPump(adapter=adapter, hub=hub)
    await pump._tick()  # must not raise, hub should not be updated
    assert hub.get_state().frame_count == 0


# ---------------------------------------------------------------------------
# _tick() with a real MockAdapter sample
# ---------------------------------------------------------------------------

async def test_tick_real_sample_updates_hub():
    """_tick() with a real EEGSample updates the hub frame_count."""
    from neurolink.hardware.mock import MockAdapter

    hub = _hub()
    adapter = MockAdapter()
    await adapter.connect()
    pump = EEGPump(adapter=adapter, hub=hub)
    await pump._tick()
    assert hub.get_state().frame_count == 1
    await adapter.disconnect()


# ---------------------------------------------------------------------------
# start() creates background task; stop() cancels it
# ---------------------------------------------------------------------------

async def test_start_and_stop():
    from neurolink.hardware.mock import MockAdapter

    hub = _hub()
    adapter = MockAdapter()
    await adapter.connect()
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=20.0)  # fast for test
    await pump.start()
    assert pump._running is True
    assert pump._task is not None
    await asyncio.sleep(0.1)  # let at least one tick fire
    await pump.stop()
    assert pump._running is False
    assert pump._task is None


# ---------------------------------------------------------------------------
# _pump_loop watchdog branch (last_frame_ts set, then time passes)
# ---------------------------------------------------------------------------

async def test_pump_loop_tick_error_is_logged_not_raised():
    """Errors inside _tick are caught by _pump_loop and logged, not re-raised."""
    hub = _hub()
    adapter = AsyncMock()
    call_count = 0

    async def flaky_read():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient hardware error")
        # Second call: stop the pump cleanly
        pump._running = False
        return None

    adapter.read_sample = flaky_read
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=100.0)
    await pump.start()
    # Wait for both ticks to complete
    for _ in range(20):
        await asyncio.sleep(0.02)
        if call_count >= 2:
            break
    await pump.stop()
    assert call_count >= 2  # loop continued after the error
