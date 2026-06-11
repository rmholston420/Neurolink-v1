"""BLE supervisor loop for Muse S Gen 1.

Wraps MuseSBleAdapter with automatic reconnect on link drop.
Ported from Rigpa-v2 ble_bridge.py.

BLE protocol constants (CMD_DATA, KEEPALIVE_SEC, RECONNECT_WAIT_SEC)
are defined in hardware/muse_s/ble_adapter.py and must not be overridden.
"""

from __future__ import annotations

import asyncio

import structlog

from neurolink.hardware.muse_s.ble_adapter import RECONNECT_WAIT_SEC

log = structlog.get_logger(__name__)


class BLEBridge:
    """Supervisor that keeps a MuseSBleAdapter connected.

    Reconnects automatically after link drop (up to RECONNECT_WAIT_SEC wait).
    """

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self.link_dropped: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Start the supervisor loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._supervisor())

    async def stop(self) -> None:
        """Stop the supervisor loop and disconnect."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._adapter.is_connected:
            await self._adapter.disconnect()

    async def _supervisor(self) -> None:
        """Supervisor loop: connect, watch for drops, reconnect."""
        while self._running:
            try:
                if not self._adapter.is_connected:
                    log.info(
                        "ble_bridge_connecting", address=getattr(self._adapter, "_address", "")
                    )
                    await self._adapter.connect()
                    self.link_dropped.clear()
                    log.info("ble_bridge_connected")

                # Wait until link drops or stop is requested
                await self._wait_for_link_drop()

                if not self._running:
                    break

                log.warning("ble_bridge_link_dropped_reconnecting", wait_sec=RECONNECT_WAIT_SEC)
                if self._adapter.is_connected:
                    await self._adapter.disconnect()
                await asyncio.sleep(RECONNECT_WAIT_SEC)

            except Exception as exc:
                log.error("ble_bridge_error", error=str(exc))
                await asyncio.sleep(RECONNECT_WAIT_SEC)

    async def _wait_for_link_drop(self) -> None:
        """Block until link_dropped event is set or adapter disconnects."""
        while self._running:
            if self.link_dropped.is_set():
                return
            if not self._adapter.is_connected:
                return
            await asyncio.sleep(1.0)
