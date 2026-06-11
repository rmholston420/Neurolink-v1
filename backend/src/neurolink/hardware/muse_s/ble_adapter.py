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

# ── FIXED BLE PROTOCOL CONSTANTS ──────────────────────────────────────────────
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
RECONNECT_WAIT_SEC: float = 20.0

_EEG_CHARS = [CHAR_EEG_TP9, CHAR_EEG_AF7, CHAR_EEG_AF8, CHAR_EEG_TP10, CHAR_EEG_RIGHTAUX]

_RING_SECS: float = 4.0
_EEG_FS: float = 256.0
_N_EEG: int = int(_EEG_FS * _RING_SECS)


class MuseSBleAdapter(HardwareAdapter):
    """Direct BLE adapter for Muse S Gen 1 using bleak.

    Lazy-imports bleak so mock mode never loads BLE drivers.
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self._client: BleakClient | None = None
        self._connected: bool = False
        self._sample_queue: deque[EEGSample] = deque(maxlen=16)
        self._eeg_rings: list[deque] = [deque(maxlen=_N_EEG) for _ in range(5)]
        self._ppg_ring: deque = deque(maxlen=1920)
        self._accel_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._gyro_ring: list[deque] = [deque(maxlen=208) for _ in range(3)]
        self._poor_contact: bool = False
        self._keepalive_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to the Muse S BLE device and start EEG streaming."""
        from bleak import BleakClient, BleakGATTCharacteristic  # noqa: PLC0415
        from neurolink.dsp.decoders import decode_eeg

        self._client = BleakClient(self._address, timeout=15.0)
        await self._client.connect()
        self._connected = True
        log.info("muse_ble_connected", address=self._address)

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

        await self._client.write_gatt_char(CHAR_CONTROL, CMD_PRESET_20, response=True)
        await asyncio.sleep(0.1)
        await self._client.write_gatt_char(CHAR_CONTROL, CMD_DATA, response=True)
        await asyncio.sleep(CMD_DATA_DELAY_SEC)
        await self._client.write_gatt_char(CHAR_CONTROL, CMD_DATA, response=True)

        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def disconnect(self) -> None:
        """Stop streaming and disconnect BLE."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        if self._client and self._connected:
            try:
                await self._client.write_gatt_char(CHAR_CONTROL, CMD_STOP, response=True)
                await self._client.disconnect()
            except Exception as exc:
                log.warning("muse_ble_disconnect_error", error=str(exc))
        self._connected = False
        log.info("muse_ble_disconnected", address=self._address)

    async def _keepalive_loop(self) -> None:
        """Send periodic keepalive commands to prevent firmware timeout."""
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
