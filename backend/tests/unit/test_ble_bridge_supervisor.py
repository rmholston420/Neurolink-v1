"""Coverage tests for BLEBridge._supervisor() loop body.

test_ble_bridge.py covers start/stop/wait_for_link_drop but leaves the
entire _supervisor while-loop body uncovered (~15 statements).
This file targets those specific branches.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from neurolink.ble_bridge import BLEBridge


def _make_adapter(connected: bool = True):
    adapter = MagicMock()
    adapter.is_connected = connected
    adapter.connect = AsyncMock()
    adapter.disconnect = AsyncMock()
    adapter._address = "AA:BB:CC:DD:EE:FF"
    return adapter


# ---------------------------------------------------------------------------
# Supervisor: adapter already connected → skips connect(), waits, then stops
# ---------------------------------------------------------------------------


async def test_supervisor_already_connected_stop_breaks_loop():
    """_supervisor sees is_connected=True, calls _wait_for_link_drop,
    then _running=False causes the outer while to exit cleanly.
    """
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)

    async def _instant_wait():
        bridge._running = False  # signal stop after first wait

    bridge._wait_for_link_drop = _instant_wait
    bridge._running = True

    await bridge._supervisor()

    # connect() should NOT have been called (was already connected)
    adapter.connect.assert_not_awaited()


# ---------------------------------------------------------------------------
# Supervisor: adapter NOT connected → connect() + link_dropped.clear() called
# ---------------------------------------------------------------------------


async def test_supervisor_not_connected_calls_connect():
    """When not connected, _supervisor calls connect() and clears link_dropped."""
    adapter = _make_adapter(connected=False)
    bridge = BLEBridge(adapter)

    call_count = 0

    async def _instant_wait():
        nonlocal call_count
        call_count += 1
        # After first wait, flip to connected=True and stop
        adapter.is_connected = True
        bridge._running = False

    bridge._wait_for_link_drop = _instant_wait
    bridge._running = True

    await bridge._supervisor()

    adapter.connect.assert_awaited_once()
    assert not bridge.link_dropped.is_set()


# ---------------------------------------------------------------------------
# Supervisor: link dropped while running → log warning, disconnect, sleep
# ---------------------------------------------------------------------------


async def test_supervisor_reconnect_on_link_drop():
    """After _wait_for_link_drop returns with _running still True,
    supervisor logs a warning, disconnects, sleeps, then loops again.
    Adapter is connected for the disconnect call.
    """
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)

    iteration = 0

    async def _wait_then_drop():
        nonlocal iteration
        iteration += 1
        if iteration >= 2:
            # Stop on the second iteration to prevent an infinite loop
            bridge._running = False

    bridge._wait_for_link_drop = _wait_then_drop
    bridge._running = True

    with patch("neurolink.ble_bridge.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        await bridge._supervisor()

    # disconnect + reconnect-wait sleep should have been called
    adapter.disconnect.assert_awaited()
    mock_sleep.assert_awaited()


# ---------------------------------------------------------------------------
# Supervisor: exception in loop body → error logged, sleep, continue
# ---------------------------------------------------------------------------


async def test_supervisor_exception_logs_error_and_continues():
    """An exception inside the try block is caught, logged, and the loop
    sleeps before retrying. We stop after the first error iteration.
    """
    adapter = _make_adapter(connected=False)
    bridge = BLEBridge(adapter)

    # Raise on first connect, then stop
    connect_calls = 0

    async def _failing_connect():
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise OSError("BLE radio off")
        # second call: allow success but then stop
        adapter.is_connected = True
        bridge._running = False

    adapter.connect = _failing_connect

    async def _instant_wait():
        bridge._running = False

    bridge._wait_for_link_drop = _instant_wait
    bridge._running = True

    with patch("neurolink.ble_bridge.log") as mock_log:
        with patch("neurolink.ble_bridge.asyncio.sleep", new=AsyncMock()):
            await bridge._supervisor()

    mock_log.error.assert_called()


# ---------------------------------------------------------------------------
# Supervisor: link dropped but adapter no longer connected → skips disconnect
# ---------------------------------------------------------------------------


async def test_supervisor_link_drop_adapter_not_connected_skips_disconnect():
    """If link is dropped but adapter.is_connected is already False,
    the disconnect() call inside the reconnect path is skipped.
    """
    adapter = _make_adapter(connected=True)
    bridge = BLEBridge(adapter)

    iteration = 0

    async def _wait_then_drop():
        nonlocal iteration
        iteration += 1
        # Simulate adapter already losing connection
        adapter.is_connected = False
        if iteration >= 2:
            bridge._running = False

    bridge._wait_for_link_drop = _wait_then_drop
    bridge._running = True

    with patch("neurolink.ble_bridge.asyncio.sleep", new=AsyncMock()):
        await bridge._supervisor()

    # disconnect should NOT have been called (is_connected was False)
    adapter.disconnect.assert_not_awaited()
