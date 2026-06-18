"""Unit tests for BLEBridge supervisor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from neurolink.ble_bridge import BLEBridge


def _make_adapter(connected: bool = True):
    adapter = MagicMock()
    adapter.is_connected = connected
    adapter.connect = AsyncMock()
    adapter.disconnect = AsyncMock()
    adapter._address = "AA:BB:CC:DD:EE:FF"
    return adapter


async def test_start_creates_task():
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)

    # Patch _supervisor so it exits immediately
    async def _noop():
        pass

    bridge._supervisor = _noop
    await bridge.start()
    assert bridge._task is not None
    # Clean up
    bridge._running = False
    await asyncio.sleep(0)


async def test_stop_cancels_task_and_disconnects():
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)

    async def _block():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass

    bridge._supervisor = _block
    await bridge.start()
    await bridge.stop()
    adapter.disconnect.assert_awaited_once()


async def test_stop_no_disconnect_when_not_connected():
    adapter = _make_adapter(connected=False)
    bridge = BLEBridge(adapter)

    async def _block():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            pass

    bridge._supervisor = _block
    await bridge.start()
    await bridge.stop()
    adapter.disconnect.assert_not_awaited()


async def test_wait_for_link_drop_exits_on_event():
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)
    bridge.link_dropped.set()
    # Should return immediately since link_dropped is set
    await bridge._wait_for_link_drop()


async def test_wait_for_link_drop_exits_when_not_running():
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)
    bridge._running = False
    # Should return immediately since _running is False
    await bridge._wait_for_link_drop()


async def test_wait_for_link_drop_exits_when_disconnected():
    adapter = _make_adapter(connected=False)
    bridge = BLEBridge(adapter)
    # Should return immediately since adapter is not connected
    await bridge._wait_for_link_drop()


async def test_bridge_link_dropped_attribute():
    adapter = _make_adapter()
    bridge = BLEBridge(adapter)
    assert isinstance(bridge.link_dropped, asyncio.Event)
    assert not bridge.link_dropped.is_set()
