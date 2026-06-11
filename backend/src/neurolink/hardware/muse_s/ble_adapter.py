"""Muse S Gen 1 direct BLE adapter via bleak.

Ported from Rigpa-v2 ble_bridge.py.
All BLE protocol constants are FIXED and must not be modified.
See Section 14 of neurolink-app-spec.md for authoritative values.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter

if TYPE_CHECKING:
    from bleak import BleakClient, BleakGATTCharacteristic

log = structlog.get_logger(__name__)

# ── FIXED BLE PROTOCOL CONSTANTS ────────────────────────────────────────────────────
MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"

CHAR_EEG_TP9 = "273e0003-4c4d-454d-96be-a864010b7d2c"
CHAR_EEG_AF7 = "273e0004-4c4d-454d-96be-a864010b7d2c"
CHAR_EEG_AF8 = "273e0005-4c4d-454d-96be-a864010b7d2c"
CHAR_EEG_TP10 = "273e0006-4c4d-454d-96be-a864010b7d2c"
CHAR_EEG_RIGHTAUX = "273e0007-4c4d-454d-96be-a864010b7d2c"
CHAR_CONTROL = "273e0001-4c4d-454d-96be-a864010b7d2c"
CHAR_TELEMETRY = "273e000b-4c4d-454d-96be-a864010b7d2c"
CHAR_ACCEL = "273e000a-4c4d-454d-96be-a864010b7d2c"
CHAR_GYRO = "273e0009-4c4d-454d-96be-a864010b7d2c"
CHAR_PPG_AMBIENT = "273e000f-4c4d-454d-96be-a864010b7d2c"
CHAR_PPG_IR = "273e0010-4c4d-454d-96be-a864010b7d2c"
CHAR_PPG_RED = "273e0011-4c4d-454d-96be-a864010b7d2c"

CMD_PRESET_20 = b"\x02\x31\x30\x0a"
CMD_DATA = b"\x02\x64\x0a"
CMD_STOP = b"\x02\x68\x0a"
CMD_KEEPALIVE = b"\x02\x6b\x0a"

CMD_DATA_DELAY_SEC: float = 0.250
KEEPALIVE_SEC: float = 30.0
RECONNECT_WAIT_SEC: float = 20.0  # spec-mandated wait before BLE reconnect attempt
MAX_RECONNECT_ATTEMPTS: int = 10  # give up after this many consecutive failures

_EEG_CHARS = [CHAR_EEG_TP9, CHAR_EEG_AF7, CHAR_EEG_AF8, CHAR_EEG_TP10, CHAR_EEG_RIGHTAUX]

_RING_SECS: float = 4.0
_EEG_FS: float = 256.0
_N_EEG: int = int(_EEG_FS * _RING_SECS)


class MuseSBleAdapter(HardwareAdapter):
    """Direct BLE adapter for Muse S Gen 1 using bleak.

    Lazy-imports bleak so mock mode never loads BLE drivers.

    Reconnect supervisor (Task 8.x):
        After a successful connect(), a _supervisor_task runs in the background.
        If bleak raises a disconnect event or read_sample returns None while
        is_connected is True, the supervisor waits RECONNECT_WAIT_SEC then
        calls connect() again.  It gives up after MAX_RECONNECT_ATTEMPTS.
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self._client: BleakClient | None = None
        self._connected: bool = False
        self._eeg_rings: list[deque] = [deque(maxlen=_N_EEG) for _ in range(5)]
        self._ppg_ring: deque = deque(maxlen=1920)
        self._accel_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._gyro_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._poor_contact: bool = False
        self._keepalive_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        # Set by supervisor when it decides to give up permanently
        self._give_up: bool = False

    # ───────────────────────────────────────────────────────────────────────────
    # Public HardwareAdapter interface
    # ───────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the Muse S BLE device and start EEG streaming.

        Idempotent: if already connected, returns immediately.
        Called both from service.connect() and from the reconnect supervisor.
        """
        if self._connected:
            return

        from bleak import BleakClient
        from neurolink.dsp.decoders import decode_eeg

        self._client = BleakClient(
            self._address,
            timeout=15.0,
            disconnected_callback=self._on_ble_disconnect,
        )
        await self._client.connect()
        self._connected = True
        log.info("muse_ble_connected", address=self._address)

        # Subscribe EEG channels
        for i, char_uuid in enumerate(_EEG_CHARS):
            ch_idx = i

            def make_eeg_handler(idx: int):
                def handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
                    samples = decode_eeg(bytes(data))
                    self._eeg_rings[idx].extend(samples)

                return handler

            await self._client.start_notify(char_uuid, make_eeg_handler(ch_idx))

        # PPG
        def ppg_handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
            from neurolink.dsp.decoders import decode_ppg

            samples = decode_ppg(bytes(data))
            self._ppg_ring.extend(samples)

        await self._client.start_notify(CHAR_PPG_IR, ppg_handler)

        # IMU — accelerometer
        def accel_handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
            from neurolink.dsp.decoders import decode_imu

            accel_flat, _ = decode_imu(bytes(data))
            for j in range(0, len(accel_flat), 3):
                if j + 2 < len(accel_flat):
                    self._accel_ring[0].append(accel_flat[j])
                    self._accel_ring[1].append(accel_flat[j + 1])
                    self._accel_ring[2].append(accel_flat[j + 2])

        # IMU — gyroscope
        def gyro_handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
            from neurolink.dsp.decoders import decode_imu

            _, gyro_flat = decode_imu(bytes(data))
            for j in range(0, len(gyro_flat), 3):
                if j + 2 < len(gyro_flat):
                    self._gyro_ring[0].append(gyro_flat[j])
                    self._gyro_ring[1].append(gyro_flat[j + 1])
                    self._gyro_ring[2].append(gyro_flat[j + 2])

        await self._client.start_notify(CHAR_ACCEL, accel_handler)
        await self._client.start_notify(CHAR_GYRO, gyro_handler)

        # Send firmware preset + start-streaming commands
        await self._client.write_gatt_char(CHAR_CONTROL, CMD_PRESET_20, response=True)
        await asyncio.sleep(0.1)
        await self._client.write_gatt_char(CHAR_CONTROL, CMD_DATA, response=True)
        await asyncio.sleep(CMD_DATA_DELAY_SEC)
        await self._client.write_gatt_char(CHAR_CONTROL, CMD_DATA, response=True)

        # Start keepalive loop
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="muse_ble_keepalive"
        )

        # Start reconnect supervisor (only once; re-connect re-uses the same task)
        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = asyncio.create_task(
                self._reconnect_supervisor(), name="muse_ble_supervisor"
            )

    async def disconnect(self) -> None:
        """Stop streaming and disconnect BLE.  Cancels supervisor so it does not
        attempt to reconnect after an intentional disconnect."""
        # Signal supervisor to stop
        self._give_up = True

        if self._supervisor_task and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass
            self._supervisor_task = None

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._client and self._connected:
            try:
                await self._client.write_gatt_char(CHAR_CONTROL, CMD_STOP, response=True)
                await self._client.disconnect()
            except Exception as exc:
                log.warning("muse_ble_disconnect_error", error=str(exc))

        self._connected = False
        log.info("muse_ble_disconnected", address=self._address)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "muse_ble"

    async def read_sample(self) -> EEGSample | None:
        """Snapshot current ring buffer state as an EEGSample."""
        if not self._connected:
            return None

        eeg_buf = [list(ring) for ring in self._eeg_rings]
        ppg_buf = list(self._ppg_ring)
        accel_buf = [list(ring) for ring in self._accel_ring]
        gyro_buf = [list(ring) for ring in self._gyro_ring]

        channels = [buf[-1] if buf else 0.0 for buf in eeg_buf]

        return EEGSample(
            channels=channels,
            timestamp=time.time(),
            source="muse_ble",
            address=self._address,
            poor_contact=self._poor_contact,
            eeg_buffer=eeg_buf,
            ppg_buffer=ppg_buf if ppg_buf else None,
            accel_buffer=accel_buf if any(accel_buf) else None,
            gyro_buffer=gyro_buf if any(gyro_buf) else None,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # Internal tasks
    # ───────────────────────────────────────────────────────────────────────────

    def _on_ble_disconnect(self, _client) -> None:
        """Bleak disconnected callback — fires on unexpected BLE drop.

        Sets _connected = False so the supervisor loop detects the drop
        and schedules a reconnect after RECONNECT_WAIT_SEC.
        """
        if not self._give_up:
            log.warning("muse_ble_unexpected_disconnect", address=self._address)
            self._connected = False

    async def _keepalive_loop(self) -> None:
        """Send periodic keepalive commands to prevent firmware timeout."""
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_SEC)
                if self._connected and self._client:
                    try:
                        await self._client.write_gatt_char(
                            CHAR_CONTROL, CMD_KEEPALIVE, response=True
                        )
                        log.debug("muse_ble_keepalive_sent", address=self._address)
                    except Exception as exc:
                        log.warning("muse_ble_keepalive_error", error=str(exc))
        except asyncio.CancelledError:
            pass

    async def _reconnect_supervisor(self) -> None:
        """Background supervisor task that watches for BLE drops and reconnects.

        Algorithm:
          1. Wait for _connected to go False (poll every 1 s).
          2. If _give_up is True, exit cleanly (intentional disconnect).
          3. Wait RECONNECT_WAIT_SEC (20 s) as specified in the spec.
          4. Attempt connect(); on success reset the attempt counter and loop.
          5. On failure, increment counter; give up after MAX_RECONNECT_ATTEMPTS.
        """
        try:
            while not self._give_up:
                # Poll until a drop is detected
                while self._connected and not self._give_up:
                    await asyncio.sleep(1.0)

                if self._give_up:
                    break

                log.info(
                    "muse_ble_supervisor_drop_detected",
                    address=self._address,
                    wait_sec=RECONNECT_WAIT_SEC,
                )

                attempts = 0
                while not self._connected and not self._give_up:
                    await asyncio.sleep(RECONNECT_WAIT_SEC)

                    if self._give_up:
                        break

                    attempts += 1
                    log.info(
                        "muse_ble_reconnect_attempt",
                        address=self._address,
                        attempt=attempts,
                        max=MAX_RECONNECT_ATTEMPTS,
                    )

                    try:
                        # Cancel stale keepalive task before re-connecting
                        if self._keepalive_task and not self._keepalive_task.done():
                            self._keepalive_task.cancel()
                            try:
                                await self._keepalive_task
                            except asyncio.CancelledError:
                                pass
                            self._keepalive_task = None

                        await self.connect()
                        log.info(
                            "muse_ble_reconnect_success",
                            address=self._address,
                            attempt=attempts,
                        )
                        break  # success — go back to watching for next drop

                    except Exception as exc:
                        log.warning(
                            "muse_ble_reconnect_failed",
                            address=self._address,
                            attempt=attempts,
                            error=str(exc),
                        )
                        if attempts >= MAX_RECONNECT_ATTEMPTS:
                            log.error(
                                "muse_ble_supervisor_giving_up",
                                address=self._address,
                                attempts=attempts,
                            )
                            self._give_up = True
                            break

        except asyncio.CancelledError:
            pass
        finally:
            log.debug("muse_ble_supervisor_exited", address=self._address)
