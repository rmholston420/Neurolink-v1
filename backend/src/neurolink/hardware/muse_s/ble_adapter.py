"""Muse S Gen 1 direct BLE adapter via bleak.

Ported from Rigpa-v2 ble_bridge.py.
All BLE protocol constants are FIXED and must not be modified.
See Section 14 of neurolink-app-spec.md for authoritative values.

Reconnect supervisor — exponential backoff
------------------------------------------
The reconnect supervisor uses a truncated binary exponential backoff with
full jitter ("full jitter" strategy from the AWS Architecture Blog):

    wait = random.uniform(0, min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * 2^attempt))

This avoids thundering-herd if multiple instances are running (unlikely
for a single-headband setup, but good practice) and spreads retry load
over the BlueZ stack evenly.

Parameters:
    BACKOFF_BASE_SEC   First-attempt base (seconds)
    BACKOFF_CAP_SEC    Maximum wait per attempt (seconds)
    MAX_RECONNECT_ATTEMPTS  Give up after this many consecutive failures
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from typing import TYPE_CHECKING

import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter

if TYPE_CHECKING:
    from bleak import BleakClient, BleakGATTCharacteristic

log = structlog.get_logger(__name__)

# -- FIXED BLE PROTOCOL CONSTANTS ------------------------------------------------
MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"

CHAR_EEG_TP9 = "273e0003-4c4d-454d-96be-f03bac821358"
CHAR_EEG_AF7 = "273e0004-4c4d-454d-96be-f03bac821358"
CHAR_EEG_AF8 = "273e0005-4c4d-454d-96be-f03bac821358"
CHAR_EEG_TP10 = "273e0006-4c4d-454d-96be-f03bac821358"
CHAR_EEG_RIGHTAUX = "273e0007-4c4d-454d-96be-f03bac821358"
CHAR_CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
CHAR_TELEMETRY = "273e000b-4c4d-454d-96be-f03bac821358"
CHAR_ACCEL = "273e000a-4c4d-454d-96be-f03bac821358"
CHAR_GYRO = "273e0009-4c4d-454d-96be-f03bac821358"
CHAR_PPG_AMBIENT = "273e000f-4c4d-454d-96be-f03bac821358"
CHAR_PPG_IR = "273e0010-4c4d-454d-96be-f03bac821358"
CHAR_PPG_RED = "273e0011-4c4d-454d-96be-f03bac821358"

CMD_PRESET_20 = b"\x02\x31\x30\x0a"
CMD_DATA = b"\x02\x64\x0a"
CMD_STOP = b"\x02\x68\x0a"
CMD_KEEPALIVE = b"\x02\x6b\x0a"

CMD_DATA_DELAY_SEC: float = 0.250
KEEPALIVE_SEC: float = 5.0

# --- Exponential backoff parameters -------------------------------------------
# Wait between reconnect attempts uses truncated binary exponential backoff
# with full jitter: wait = uniform(0, min(BACKOFF_CAP_SEC, BASE * 2^attempt))
# Attempt 0: 0-5 s, attempt 1: 0-10 s, attempt 2: 0-20 s, attempt 3+: 0-60 s
BACKOFF_BASE_SEC: float = 5.0
BACKOFF_CAP_SEC: float = 60.0
MAX_RECONNECT_ATTEMPTS: int = 10

# Legacy alias so any code reading RECONNECT_WAIT_SEC still compiles
RECONNECT_WAIT_SEC: float = BACKOFF_BASE_SEC

BLE_CONNECT_TIMEOUT: float = 45.0
POST_CONNECT_SETTLE_SEC: float = 0.5

_WRITE_RETRIES: int = 3
_WRITE_RETRY_DELAY_SEC: float = 0.3
PRE_SCAN_SEC: float = 10.0

_EEG_CHARS = [CHAR_EEG_TP9, CHAR_EEG_AF7, CHAR_EEG_AF8, CHAR_EEG_TP10, CHAR_EEG_RIGHTAUX]

_RING_SECS: float = 4.0
_EEG_FS: float = 256.0
_N_EEG: int = int(_EEG_FS * _RING_SECS)


def _backoff_wait(attempt: int) -> float:
    """Return a jittered wait in seconds for the given attempt index (0-based).

    Uses truncated binary exponential backoff with full jitter:
        wait = uniform(0, min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * 2^attempt))

    This avoids deterministic retry storms (all adapters retry simultaneously)
    and keeps early retries fast while preventing runaway polling.

    random.uniform is intentional here: this is timing jitter, not a
    cryptographic operation.  S311 is suppressed accordingly.
    """
    ceiling = min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2**attempt))
    return random.uniform(0.0, ceiling)  # noqa: S311


class MuseSBleAdapter(HardwareAdapter):
    """Direct BLE adapter for Muse S Gen 1 using bleak.

    Lazy-imports bleak so mock mode never loads BLE drivers.

    Reconnect supervisor uses exponential backoff with full jitter.
    See module docstring for algorithm details.

    _session_connected flag semantics
    ----------------------------------
    See original docstring — unchanged.
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self._client: BleakClient | None = None
        self._connected: bool = False
        self._session_connected: bool = False
        self._eeg_rings: list[deque] = [deque(maxlen=_N_EEG) for _ in range(5)]
        self._ppg_ring: deque = deque(maxlen=1920)
        self._accel_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._gyro_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._poor_contact: bool = False
        self._keepalive_task: asyncio.Task | None = None
        self._supervisor_task: asyncio.Task | None = None
        self._give_up: bool = False
        self._disconnecting: bool = False

    # ---------------------------------------------------------------------------
    # Public HardwareAdapter interface
    # ---------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Muse S BLE device and start EEG streaming.

        Idempotent: if already connected, returns immediately.

        Raises:
            ConnectionError: if the BLE connection attempt times out or is
                rejected by BlueZ.
        """
        if self._connected:
            return

        self._disconnecting = False

        from bleak import BleakClient

        from neurolink.dsp.decoders import decode_eeg

        found, rssi = await self._prescan_until_seen()
        log.info(
            "muse_ble_device_found",
            address=self._address,
            name=found.name,
            rssi=rssi,
        )

        self._client = BleakClient(
            found,
            timeout=BLE_CONNECT_TIMEOUT,
            disconnected_callback=self._on_ble_disconnect,
        )
        try:
            await self._client.connect()
        except TimeoutError as exc:
            raise ConnectionError(
                f"BLE connect timed out after {BLE_CONNECT_TIMEOUT:.0f} s "
                f"({self._address}, rssi={rssi}). "
                "Move the headband closer to the host and retry."
            ) from exc

        self._connected = True
        self._session_connected = True
        log.info("muse_ble_connected", address=self._address, rssi=rssi)

        _ = self._client.services
        log.debug("muse_ble_services_discovered", address=self._address)

        await asyncio.sleep(POST_CONNECT_SETTLE_SEC)

        for i, char_uuid in enumerate(_EEG_CHARS):
            ch_idx = i

            def make_eeg_handler(idx: int):
                def handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
                    samples = decode_eeg(bytes(data))
                    self._eeg_rings[idx].extend(samples)

                return handler

            await self._client.start_notify(char_uuid, make_eeg_handler(ch_idx))

        def ppg_handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
            from neurolink.dsp.decoders import decode_ppg

            samples = decode_ppg(bytes(data))
            self._ppg_ring.extend(samples)

        await self._client.start_notify(CHAR_PPG_IR, ppg_handler)

        def accel_handler(_sender: BleakGATTCharacteristic, data: bytearray) -> None:
            from neurolink.dsp.decoders import decode_imu

            accel_flat, _ = decode_imu(bytes(data))
            for j in range(0, len(accel_flat), 3):
                if j + 2 < len(accel_flat):
                    self._accel_ring[0].append(accel_flat[j])
                    self._accel_ring[1].append(accel_flat[j + 1])
                    self._accel_ring[2].append(accel_flat[j + 2])

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

        await self._write_control(CMD_PRESET_20)
        await asyncio.sleep(0.1)
        await self._write_control(CMD_DATA)
        await asyncio.sleep(CMD_DATA_DELAY_SEC)
        await self._write_control(CMD_DATA)

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="muse_ble_keepalive"
        )

        if self._supervisor_task is None or self._supervisor_task.done():
            self._supervisor_task = asyncio.create_task(
                self._reconnect_supervisor(), name="muse_ble_supervisor"
            )

    async def disconnect(self) -> None:
        """Stop streaming and disconnect BLE."""
        self._give_up = True
        self._session_connected = False

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
                await self._write_control(CMD_STOP)
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

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    async def _prescan_until_seen(self):
        from bleak import BleakScanner

        seen_event: asyncio.Event = asyncio.Event()
        found_device = None
        found_rssi: int | None = None

        target_addr = self._address.upper()

        def _on_detection(device, advertisement_data) -> None:
            nonlocal found_device, found_rssi
            if device.address.upper() == target_addr:
                found_device = device
                found_rssi = advertisement_data.rssi
                seen_event.set()

        log.info(
            "muse_ble_prescanning",
            address=self._address,
            timeout=PRE_SCAN_SEC,
            method="detection_callback",
        )

        scanner = BleakScanner(
            detection_callback=_on_detection,
            service_uuids=[MUSE_SERVICE_UUID],
        )

        async with scanner:
            try:
                await asyncio.wait_for(seen_event.wait(), timeout=PRE_SCAN_SEC)
            except TimeoutError as err:
                raise ConnectionError(
                    f"Muse S not seen after {PRE_SCAN_SEC:.0f} s scan "
                    f"({self._address}). Is the headband powered on and in range?"
                ) from err

        return found_device, found_rssi

    async def _write_control(self, cmd: bytes) -> None:
        if self._client is None:
            return
        last_exc: Exception | None = None
        for attempt in range(1, _WRITE_RETRIES + 1):
            try:
                await self._client.write_gatt_char(CHAR_CONTROL, cmd, response=False)
                return
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "muse_ble_write_control_retry",
                    attempt=attempt,
                    max=_WRITE_RETRIES,
                    cmd=cmd.hex(),
                    error=str(exc),
                )
                if attempt < _WRITE_RETRIES:
                    await asyncio.sleep(_WRITE_RETRY_DELAY_SEC)
        log.error(
            "muse_ble_write_control_failed",
            cmd=cmd.hex(),
            error=str(last_exc),
        )
        raise last_exc  # type: ignore[misc]

    # ---------------------------------------------------------------------------
    # Internal tasks
    # ---------------------------------------------------------------------------

    def _on_ble_disconnect(self, _client) -> None:
        """Bleak disconnected callback — see class docstring for state semantics."""
        if self._give_up:
            return
        if self._disconnecting:
            return

        if not self._session_connected:
            log.debug(
                "muse_ble_disconnect_during_handshake",
                address=self._address,
            )
            return

        self._disconnecting = True
        log.warning("muse_ble_unexpected_disconnect", address=self._address)
        self._connected = False
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()

    async def _keepalive_loop(self) -> None:
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_SEC)
                if (
                    self._connected
                    and self._client is not None
                    and self._client.is_connected
                ):
                    try:
                        await self._write_control(CMD_KEEPALIVE)
                        log.debug("muse_ble_keepalive_sent", address=self._address)
                    except Exception as exc:
                        log.warning("muse_ble_keepalive_error", error=str(exc))
        except asyncio.CancelledError:
            pass

    async def _reconnect_supervisor(self) -> None:
        """Background supervisor: watches for BLE drops and reconnects.

        Uses truncated binary exponential backoff with full jitter so
        repeated failures back off gracefully rather than hammering BlueZ
        at a fixed 20-second rate.

        Control flow note
        -----------------
        The _give_up guard is intentionally checked AFTER self.connect() on
        the success path.  Checking it before connect() would cause the test
        (and real code) to skip the connection attempt when _give_up is set
        concurrently during the backoff sleep -- which is not the desired
        behaviour; we want to make at least one attempt once the sleep
        completes unless _give_up was already True before the sleep started.
        """
        try:
            while not self._give_up:
                # Poll while connected.
                while self._connected and not self._give_up:
                    await asyncio.sleep(1.0)

                if self._give_up:
                    break

                log.info(
                    "muse_ble_supervisor_drop_detected",
                    address=self._address,
                )

                attempts = 0
                while not self._connected and not self._give_up:
                    wait = _backoff_wait(attempts)
                    log.info(
                        "muse_ble_reconnect_wait",
                        address=self._address,
                        attempt=attempts + 1,
                        max=MAX_RECONNECT_ATTEMPTS,
                        wait_sec=round(wait, 1),
                    )
                    await asyncio.sleep(wait)

                    # Note: do NOT check _give_up here before connect().
                    # If _give_up was set during the sleep we still attempt
                    # connect() once -- the outer while-condition handles
                    # the loop exit after connect() returns or raises.

                    attempts += 1
                    log.info(
                        "muse_ble_reconnect_attempt",
                        address=self._address,
                        attempt=attempts,
                        max=MAX_RECONNECT_ATTEMPTS,
                    )

                    try:
                        if self._keepalive_task and not self._keepalive_task.done():
                            self._keepalive_task.cancel()
                            try:
                                await self._keepalive_task
                            except asyncio.CancelledError:
                                pass

                        await self.connect()
                        log.info(
                            "muse_ble_reconnect_ok",
                            address=self._address,
                            attempt=attempts,
                        )
                        attempts = 0  # reset on success
                    except Exception as exc:
                        log.warning(
                            "muse_ble_reconnect_failed",
                            address=self._address,
                            attempt=attempts,
                            error=str(exc),
                        )
                        if attempts >= MAX_RECONNECT_ATTEMPTS:
                            log.error(
                                "muse_ble_giving_up",
                                address=self._address,
                                attempts=attempts,
                            )
                            self._give_up = True
                            break
        except asyncio.CancelledError:
            pass
