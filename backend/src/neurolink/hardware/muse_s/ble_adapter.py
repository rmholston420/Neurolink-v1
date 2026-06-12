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

CHAR_EEG_TP9      = "273e0003-4c4d-454d-96be-f03bac821358"
CHAR_EEG_AF7      = "273e0004-4c4d-454d-96be-f03bac821358"
CHAR_EEG_AF8      = "273e0005-4c4d-454d-96be-f03bac821358"
CHAR_EEG_TP10     = "273e0006-4c4d-454d-96be-f03bac821358"
CHAR_EEG_RIGHTAUX = "273e0007-4c4d-454d-96be-f03bac821358"
CHAR_CONTROL      = "273e0001-4c4d-454d-96be-f03bac821358"
CHAR_TELEMETRY    = "273e000b-4c4d-454d-96be-f03bac821358"
CHAR_ACCEL        = "273e000a-4c4d-454d-96be-f03bac821358"
CHAR_GYRO         = "273e0009-4c4d-454d-96be-f03bac821358"
CHAR_PPG_AMBIENT  = "273e000f-4c4d-454d-96be-f03bac821358"
CHAR_PPG_IR       = "273e0010-4c4d-454d-96be-f03bac821358"
CHAR_PPG_RED      = "273e0011-4c4d-454d-96be-f03bac821358"

CMD_PRESET_20 = b"\x02\x31\x30\x0a"
CMD_DATA      = b"\x02\x64\x0a"
CMD_STOP      = b"\x02\x68\x0a"
CMD_KEEPALIVE = b"\x02\x6b\x0a"

CMD_DATA_DELAY_SEC: float   = 0.250
# Reduced from 30.0 s — the Muse S BLE supervision window is 2000 ms (0x00c8).
# A 30 s interval left the link silent long enough for the Muse to time out
# (~28 s, confirmed via btmon: Disconnect Complete reason 0x08).
# 5 s is safely within the supervision window while avoiding unnecessary
# GATT traffic on the 15 ms connection interval.
KEEPALIVE_SEC: float        = 5.0
RECONNECT_WAIT_SEC: float   = 20.0   # spec-mandated wait before BLE reconnect attempt
MAX_RECONNECT_ATTEMPTS: int = 10     # give up after this many consecutive failures

# Raised from 30 s — GATT discovery at weak RSSI (-92 to -94 dBm) needs extra
# headroom before BlueZ gives up on the ATT Bearer negotiation.
BLE_CONNECT_TIMEOUT: float = 45.0

# Settle delay after connect() before issuing any GATT writes.  The Muse S
# link layer needs ~400–500 ms to finish the LE feature exchange (seen in
# btmon: LE Read Remote Used Features completes at t+~150 ms, ATT MTU exchange
# at t+~270 ms).  Writing before this window closes causes 0x3e / 0x08 drops.
POST_CONNECT_SETTLE_SEC: float = 0.5

# How many times to retry a control-char write before giving up.
_WRITE_RETRIES: int = 3
_WRITE_RETRY_DELAY_SEC: float = 0.3

# Maximum time to wait for the Muse to appear during the pre-scan phase.
# The live detection_callback scanner typically fires within 100–300 ms of
# the first ADV_IND; this timeout is only reached if the headband is off or
# out of range.
PRE_SCAN_SEC: float = 10.0

_EEG_CHARS = [CHAR_EEG_TP9, CHAR_EEG_AF7, CHAR_EEG_AF8, CHAR_EEG_TP10, CHAR_EEG_RIGHTAUX]

_RING_SECS: float = 4.0
_EEG_FS: float    = 256.0
_N_EEG: int       = int(_EEG_FS * _RING_SECS)


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
        # Gate flag: ensures _on_ble_disconnect executes its side-effects only
        # once per physical disconnect event. bleak fires the disconnected_callback
        # repeatedly on weak-RSSI links (once per BLE supervision timeout packet),
        # which would otherwise spam logs and trigger the reconnect supervisor
        # multiple times per drop. Cleared at the top of connect() so each new
        # session starts clean.
        self._disconnecting: bool = False

    # ───────────────────────────────────────────────────────────────────────────
    # Public HardwareAdapter interface
    # ───────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to the Muse S BLE device and start EEG streaming.

        Idempotent: if already connected, returns immediately.
        Called both from service.connect() and from the reconnect supervisor.

        Uses a live detection_callback scanner filtered on MUSE_SERVICE_UUID
        to mirror the official Muse app behaviour: connect fires on the first
        matching ADV_IND packet (~100–300 ms) rather than waiting a fixed
        pre-scan window.
        """
        if self._connected:
            return

        # Reset disconnect gate for this new session so _on_ble_disconnect
        # will fire correctly if this connection subsequently drops.
        self._disconnecting = False

        from bleak import BleakClient

        from neurolink.dsp.decoders import decode_eeg

        # ── Pre-scan: live UUID-filtered scanner, connect on first ADV_IND ─────
        found, rssi = await self._prescan_until_seen()
        log.info(
            "muse_ble_device_found",
            address=self._address,
            name=found.name,
            rssi=rssi,
        )

        # ── Connect ────────────────────────────────────────────────────────────
        self._client = BleakClient(
            found,  # pass the BLEDevice object directly — avoids a second
                    # address-lookup inside bleak and reuses the warm cache entry
            timeout=BLE_CONNECT_TIMEOUT,
            disconnected_callback=self._on_ble_disconnect,
        )
        await self._client.connect()
        self._connected = True
        log.info("muse_ble_connected", address=self._address, rssi=rssi)

        # ── Settle: wait for LE feature exchange + ATT MTU negotiation ─────────
        # The Muse S drops connections (0x3e / 0x08) if GATT writes arrive
        # before the link-layer handshake completes.  POST_CONNECT_SETTLE_SEC
        # gives it the required window.
        await asyncio.sleep(POST_CONNECT_SETTLE_SEC)

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

        # Send firmware preset + start-streaming commands (with retry)
        await self._write_control(CMD_PRESET_20)
        await asyncio.sleep(0.1)
        await self._write_control(CMD_DATA)
        await asyncio.sleep(CMD_DATA_DELAY_SEC)
        await self._write_control(CMD_DATA)

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

        eeg_buf   = [list(ring) for ring in self._eeg_rings]
        ppg_buf   = list(self._ppg_ring)
        accel_buf = [list(ring) for ring in self._accel_ring]
        gyro_buf  = [list(ring) for ring in self._gyro_ring]

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
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────────────

    async def _prescan_until_seen(self):
        """Run a live BLE scan filtered on MUSE_SERVICE_UUID and wait for
        the target address to appear in an advertisement.

        Mirrors the official Muse app behaviour: instead of polling
        find_device_by_address() for a fixed duration, we start a
        BleakScanner with a detection_callback so we connect the instant
        the first matching ADV_IND / SCAN_RSP arrives (typically 100–300 ms).

        Returns (BLEDevice, rssi: int | None) so the caller can pass the
        BLEDevice object directly to BleakClient, reusing the warm cache
        entry rather than triggering a second address lookup inside bleak.

        Raises ConnectionError if the device is not seen within PRE_SCAN_SEC.
        """
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
            except asyncio.TimeoutError:
                raise ConnectionError(
                    f"Muse S not seen after {PRE_SCAN_SEC:.0f} s scan "
                    f"({self._address}). Is the headband powered on and in range?"
                )

        return found_device, found_rssi

    async def _write_control(self, cmd: bytes) -> None:
        """Write a command to CHAR_CONTROL with automatic retry.

        The Muse S occasionally drops ATT write responses during the initial
        link-layer handshake (observed as 0x3e / 0x08 disconnects in btmon).
        Retrying with a short back-off recovers in the vast majority of cases.
        """
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

    # ───────────────────────────────────────────────────────────────────────────
    # Internal tasks
    # ───────────────────────────────────────────────────────────────────────────

    def _on_ble_disconnect(self, _client) -> None:
        """Bleak disconnected callback — fires on unexpected BLE drop.

        Sets _connected = False so the supervisor loop detects the drop
        and schedules a reconnect after RECONNECT_WAIT_SEC.  Also cancels
        the keepalive task immediately so it does not attempt writes on a
        dead handle while the supervisor waits.

        Guard: on weak-RSSI links bleak fires this callback repeatedly (once
        per BLE supervision timeout packet — typically 5–20 times per physical
        disconnect).  The _disconnecting flag ensures the side-effects below
        execute exactly once per physical disconnect.  The flag is cleared at
        the top of connect() so each new session starts fresh.
        """
        if self._give_up:
            return

        # Atomically claim the first callback; all subsequent ones are no-ops.
        if self._disconnecting:
            return
        self._disconnecting = True

        log.warning("muse_ble_unexpected_disconnect", address=self._address)
        self._connected = False
        # Cancel keepalive immediately — writing on a dead handle causes
        # bleak to raise and can mask the real disconnect in logs.
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()

    async def _keepalive_loop(self) -> None:
        """Send periodic keepalive commands to prevent firmware timeout."""
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_SEC)
                if self._connected and self._client:
                    try:
                        await self._write_control(CMD_KEEPALIVE)
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
