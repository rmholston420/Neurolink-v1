"""Coverage tests for EEGPump."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload


def _hub() -> EEGHub:
    return EEGHub()


# ---------------------------------------------------------------------------
# __init__ inspection
# ---------------------------------------------------------------------------

def test_pump_init_fields():
    hub = _hub()
    pump = EEGPump(hub=hub, adapter=MagicMock(), device_model="mock")
    assert pump._hub is hub
    assert pump._running is False


# ---------------------------------------------------------------------------
# stop() while not running
# ---------------------------------------------------------------------------

def test_stop_while_not_running_is_noop():
    pump = EEGPump(hub=_hub(), adapter=MagicMock(), device_model="mock")
    pump.stop()  # must not raise
    assert pump._running is False


# ---------------------------------------------------------------------------
# run() — adapter returns None (continue branch)
# ---------------------------------------------------------------------------

async def test_run_none_sample_continues():
    """When adapter.read_sample() returns None the pump continues the loop."""
    hub = _hub()
    adapter = AsyncMock()
    call_count = 0

    async def read_sample():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None  # triggers continue branch
        # Second call: stop the pump then raise CancelledError to exit
        pump.stop()
        raise asyncio.CancelledError

    adapter.read_sample = read_sample
    pump = EEGPump(hub=hub, adapter=adapter, device_model="mock")

    with pytest.raises(asyncio.CancelledError):
        await pump.run()

    assert call_count == 2


# ---------------------------------------------------------------------------
# run() — CancelledError exits cleanly
# ---------------------------------------------------------------------------

async def test_run_cancelled_error_exits():
    adapter = AsyncMock()
    adapter.read_sample.side_effect = asyncio.CancelledError
    pump = EEGPump(hub=_hub(), adapter=adapter, device_model="mock")
    with pytest.raises(asyncio.CancelledError):
        await pump.run()


# ---------------------------------------------------------------------------
# run() — one real frame processed via MockAdapter
# ---------------------------------------------------------------------------

async def test_run_processes_one_frame():
    """Full pump cycle with MockAdapter: one frame reaches the hub."""
    from neurolink.hardware.mock import MockAdapter

    hub = _hub()
    adapter = MockAdapter()
    await adapter.connect()
    pump = EEGPump(hub=hub, adapter=adapter, device_model="mock")

    # Run pump inside a task; cancel after first hub update
    task = asyncio.create_task(pump.run())
    # Wait until hub receives at least one frame
    for _ in range(40):  # max ~10 s
        await asyncio.sleep(0.05)
        if hub.get_state().frame_count >= 1:
            break
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert hub.get_state().frame_count >= 1


# ---------------------------------------------------------------------------
# run() — unexpected exception is logged and re-raised
# ---------------------------------------------------------------------------

async def test_run_unexpected_exception_reraises():
    adapter = AsyncMock()
    adapter.read_sample.side_effect = RuntimeError("hardware exploded")
    pump = EEGPump(hub=_hub(), adapter=adapter, device_model="mock")
    with pytest.raises(RuntimeError, match="hardware exploded"):
        await pump.run()


# ---------------------------------------------------------------------------
# stop() while running
# ---------------------------------------------------------------------------

async def test_stop_while_running_terminates_loop():
    adapter = AsyncMock()
    frames = 0

    async def read_sample():
        nonlocal frames
        frames += 1
        if frames >= 2:
            pump.stop()
            raise asyncio.CancelledError
        return None  # keep looping

    adapter.read_sample = read_sample
    pump = EEGPump(hub=_hub(), adapter=adapter, device_model="mock")
    with pytest.raises(asyncio.CancelledError):
        await pump.run()
    assert pump._running is False
