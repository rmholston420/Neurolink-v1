"""BLE supervisor loop for Muse S Gen 1/Gen 2.

Ported from Rigpa-v2 ble_bridge.py.
Manages reconnect logic; protocol constants verbatim.
"""
from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger(__name__)

# Protocol constants — DO NOT MODIFY
RECONNECT_WAIT: float = 20.0


class BLESupervisor:
    """Supervisor that restarts the BLE adapter after a link drop.

    Runs as a background asyncio Task. On link drop, waits RECONNECT_WAIT
    seconds then reconnects.
    """

    def __init__(self, adapter, on_sample=None) -> None:  # type: ignore[type-arg]
        """
        Args:
            adapter: MuseSBleAdapter instance
            on_sample: optional async callable(EEGSample) called for each frame
        """
        self._adapter = adapter
        self._on_sample = on_sample
        self._running = False
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self.link_dropped: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Start the supervisor loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the supervisor loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """Main supervisor loop: connect, stream, reconnect on drop."""
        while self._running:
            try:
                await self._adapter.connect()
                self.link_dropped.clear()
                log.info("ble_supervisor_streaming")
                async for sample in self._adapter.stream():
                    if not self._running:
                        return
                    if self.link_dropped.is_set():
                        break
                    if self._on_sample is not None:
                        await self._on_sample(sample)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("ble_supervisor_error", error=str(exc))

            if not self._running:
                return

            log.info("ble_supervisor_reconnect_wait", wait_sec=RECONNECT_WAIT)
            try:
                await asyncio.sleep(RECONNECT_WAIT)
            except asyncio.CancelledError:
                return
