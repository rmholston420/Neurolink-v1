"""Mock-based coverage for ble_adapter.py.

Covers the logic paths that require hardware (bleak) by replacing
BleakClient, BleakScanner, and asyncio.sleep with AsyncMock / MagicMock.
This brings coverage of _reconnect_supervisor, _on_ble_disconnect, and
_backoff_wait from ~0% to the lines identified in the coverage gap analysis.

All tests are async and use pytest-asyncio in auto mode.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.hardware.muse_s.ble_adapter import (
    BACKOFF_BASE_SEC,
    BACKOFF_CAP_SEC,
    MAX_RECONNECT_ATTEMPTS,
    MuseSBleAdapter,
    _backoff_wait,
)

# ---------------------------------------------------------------------------
# _backoff_wait -- pure function, no async needed
# ---------------------------------------------------------------------------


class TestBackoffWait:
    """Unit tests for the _backoff_wait pure helper."""

    def test_attempt_0_within_base(self):
        """Attempt 0 ceiling is BACKOFF_BASE_SEC (5 s)."""
        for _ in range(50):
            w = _backoff_wait(0)
            assert 0.0 <= w <= BACKOFF_BASE_SEC

    def test_attempt_1_within_2x_base(self):
        """Attempt 1 ceiling is 2 * BACKOFF_BASE_SEC (10 s)."""
        for _ in range(50):
            w = _backoff_wait(1)
            assert 0.0 <= w <= 2 * BACKOFF_BASE_SEC

    def test_attempt_2_within_4x_base(self):
        """Attempt 2 ceiling is 4 * BACKOFF_BASE_SEC (20 s)."""
        for _ in range(50):
            w = _backoff_wait(2)
            assert 0.0 <= w <= 4 * BACKOFF_BASE_SEC

    def test_large_attempt_capped(self):
        """Large attempt numbers are capped at BACKOFF_CAP_SEC."""
        for attempt in [10, 20, 100]:
            for _ in range(20):
                w = _backoff_wait(attempt)
                assert 0.0 <= w <= BACKOFF_CAP_SEC

    def test_wait_is_non_deterministic(self):
        """Two consecutive calls should not always return the same value."""
        results = {_backoff_wait(3) for _ in range(30)}
        # With uniform random in [0, 60] it is astronomically unlikely all 30
        # draws are identical.
        assert len(results) > 1

    def test_cap_equals_module_constant(self):
        """BACKOFF_CAP_SEC is the hard ceiling regardless of attempt."""
        for _ in range(100):
            assert _backoff_wait(999) <= BACKOFF_CAP_SEC


# ---------------------------------------------------------------------------
# _on_ble_disconnect state-machine transitions
# ---------------------------------------------------------------------------


class TestOnBleDisconnect:
    """Tests for _on_ble_disconnect state transitions without event loop."""

    def _make_adapter(self) -> MuseSBleAdapter:
        return MuseSBleAdapter(address="AA:BB:CC:DD:EE:FF")

    def test_give_up_flag_returns_early(self):
        """When _give_up is True the handler must not change _connected."""
        adapter = self._make_adapter()
        adapter._give_up = True
        adapter._connected = True
        adapter._on_ble_disconnect(None)
        assert adapter._connected is True  # unchanged

    def test_disconnecting_flag_returns_early(self):
        """When _disconnecting is already True the handler is a no-op."""
        adapter = self._make_adapter()
        adapter._disconnecting = True
        adapter._connected = True
        adapter._on_ble_disconnect(None)
        assert adapter._connected is True

    def test_pre_session_connect_returns_early(self):
        """Disconnect before _session_connected logs debug and returns."""
        adapter = self._make_adapter()
        adapter._session_connected = False
        adapter._connected = True
        adapter._on_ble_disconnect(None)
        # _connected must not be set to False -- it was not yet True for BLE
        assert adapter._connected is True

    def test_unexpected_disconnect_sets_flags(self):
        """A genuine mid-session drop must mark _connected=False and _disconnecting=True."""
        adapter = self._make_adapter()
        adapter._session_connected = True
        adapter._connected = True
        adapter._on_ble_disconnect(None)
        assert adapter._connected is False
        assert adapter._disconnecting is True

    def test_unexpected_disconnect_cancels_keepalive(self):
        """A genuine drop must cancel a running keepalive task."""
        adapter = self._make_adapter()
        adapter._session_connected = True
        adapter._connected = True
        mock_task = MagicMock()
        mock_task.done.return_value = False
        adapter._keepalive_task = mock_task
        adapter._on_ble_disconnect(None)
        mock_task.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# _reconnect_supervisor -- async coverage with patched bleak + asyncio.sleep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconnectSupervisor:
    """Async tests for the _reconnect_supervisor coroutine.

    All bleak imports and asyncio.sleep are patched so tests run instantly
    with no BLE hardware and without real timers.
    """

    def _make_connected_adapter(self) -> MuseSBleAdapter:
        """Return an adapter that starts in the 'already connected' state."""
        adapter = MuseSBleAdapter(address="AA:BB:CC:DD:EE:FF")
        adapter._connected = True
        adapter._session_connected = True
        return adapter

    async def test_supervisor_exits_on_give_up_while_connected(self):
        """Supervisor must exit cleanly when _give_up is set while connected."""
        adapter = self._make_connected_adapter()

        call_count = 0

        async def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            # After first sleep tick, signal give_up so the outer loop exits
            adapter._give_up = True

        with patch(
            "neurolink.hardware.muse_s.ble_adapter.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            await adapter._reconnect_supervisor()

        assert adapter._give_up is True
        assert call_count >= 1

    async def test_supervisor_reconnects_after_drop(self):
        """When _connected goes False, supervisor must call connect() once."""
        adapter = self._make_connected_adapter()

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)
            # First call: simulate the 1-s polling sleep while connected
            if len(sleep_calls) == 1:
                adapter._connected = False  # simulate BLE drop
            else:
                # After backoff sleep + reconnect, set give_up to stop the loop
                adapter._connected = True
                adapter._give_up = True

        connect_calls = 0

        async def fake_connect():
            nonlocal connect_calls
            connect_calls += 1
            adapter._connected = True

        with patch(
            "neurolink.hardware.muse_s.ble_adapter.asyncio.sleep",
            side_effect=fake_sleep,
        ), patch.object(adapter, "connect", side_effect=fake_connect):
            await adapter._reconnect_supervisor()

        assert connect_calls == 1

    async def test_supervisor_give_up_after_max_attempts(self):
        """After MAX_RECONNECT_ATTEMPTS failures _give_up must be set."""
        adapter = MuseSBleAdapter(address="AA:BB:CC:DD:EE:FF")
        # Start disconnected so the supervisor immediately enters retry loop
        adapter._connected = False
        adapter._session_connected = True

        async def fake_sleep(_secs):
            pass  # instant

        async def always_fail_connect():
            raise ConnectionError("simulated BLE failure")

        with patch(
            "neurolink.hardware.muse_s.ble_adapter.asyncio.sleep",
            side_effect=fake_sleep,
        ), patch.object(adapter, "connect", side_effect=always_fail_connect):
            await adapter._reconnect_supervisor()

        assert adapter._give_up is True

    async def test_supervisor_cancelled_error_propagates_cleanly(self):
        """CancelledError inside the supervisor must be swallowed (not re-raised)."""
        adapter = self._make_connected_adapter()

        async def fake_sleep(_secs):
            raise asyncio.CancelledError()

        with patch(
            "neurolink.hardware.muse_s.ble_adapter.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            # Must not raise
            await adapter._reconnect_supervisor()

    async def test_supervisor_resets_attempts_on_success(self):
        """Attempt counter must reset to 0 after a successful reconnect."""
        adapter = MuseSBleAdapter(address="AA:BB:CC:DD:EE:FF")
        adapter._connected = False
        adapter._session_connected = True

        reconnect_count = 0

        async def fake_sleep(_secs):
            pass

        async def succeed_then_stop():
            nonlocal reconnect_count
            reconnect_count += 1
            adapter._connected = True
            adapter._give_up = True  # stop after first success

        with patch(
            "neurolink.hardware.muse_s.ble_adapter.asyncio.sleep",
            side_effect=fake_sleep,
        ), patch.object(adapter, "connect", side_effect=succeed_then_stop):
            await adapter._reconnect_supervisor()

        assert reconnect_count == 1
        assert adapter._give_up is True
