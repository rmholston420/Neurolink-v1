"""Muse S Gen 1 direct BLE adapter.

Ported from Rigpa-v2 ble_bridge.py + Rigpa-v3 hardware/muse_s/ble_adapter.py.
All protocol constants are firmware-level. DO NOT MODIFY.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

import numpy as np
import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.dsp.decoders import decode_eeg, decode_ppg, decode_imu

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# BLE GATT UUIDs — DO NOT MODIFY (firmware constants)
# ──────────────────────────────────────────────────────────────────────
_UUID_CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
_UUID_EEG_TP9 = "273e0003-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AF7 = "273e0004-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AF8 = "273e0005-4c4d-454d-96be-f03bac821358"
_UUID_EEG_TP10 = "273e0006-4c4d-454d-96be-f03bac821358"
_UUID_EEG_AUX = "273e0007-4c4d-454d-96be-f03bac821358"
_UUID_PPG1 = "273e000f-4c4d-454d-96be-f03bac821358"
_UUID_PPG2 = "273e0010-4c4d-454d-96be-f03bac821358"
_UUID_PPG3 = "273e0011-4c4d-454d-96be-f03bac821358"
_UUID_ACC = "273e000a-4c4d-454d-96be-f03bac821358"
_UUID_GYRO = "273e0009-4c4d-454d-96be-f03bac821358"

# ──────────────────────────────────────────────────────────────────────
# Control commands — DO NOT MODIFY (firmware constants)
# ──────────────────────────────────────────────────────────────────────
_CMD_HALT = bytes([0x02, 0x68, 0x0A])           # "h"
_CMD_PRESET = bytes([0x05, 0x70, 0x35, 0x30, 0x0A])  # "p50"
_CMD_START = bytes([0x02, 0x73, 0x0A])           # "s"
_CMD_DATA = bytes([0x02, 0x64, 0x0A])            # "d"

_DATA_DOUBLE_SEND_GAP: float = 0.25   # seconds between two CMD_DATA sends
KEEPALIVE_SEC: float = 30.0           # re-arm interval
RECONNECT_WAIT: float = 20.0          # wait after link drop before reconnect
PUBLISH_HZ: float = 4.0              # hub publish cadence

_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]
_EEG_UUIDS = [_UUID_EEG_TP9, _UUID_EEG_AF7, _UUID_EEG_AF8, _UUID_EEG_TP10, _UUID_EEG_AUX]


class MuseSBleAdapter(HardwareAdapter):
    """Direct BLE adapter for Muse S Gen 1/Gen 2 via bleak GATT.

    Supports BLE arming sequence, keepalive, and reconnect supervisor.
    Hardware imports are lazy (bleak only imported on connect).
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self._connected = False
        self._client = None  # type: ignore[assignment]
        self._sample_queue: asyncio.Queue[EEGSample] = asyncio.Queue(maxsize=64)
        self._eeg_bufs: dict[str, list[float]] = {ch: [] for ch in _EEG_CHANNELS}
        self._ppg_buf: list[float] = []
        self._accel_buf: list[float] = []
        self._gyro_buf: list[float] = []
        self._keepalive_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def connect(self) -> None:
        """Connect to Muse S via BLE and arm the stream."""
        await self._connect_with_retry()

    async def _connect_with_retry(self, max_attempts: int = 3) -> None:
        """Connect with retry on failure."""
        from bleak import BleakClient  # lazy import

        for attempt in range(1, max_attempts + 1):
            try:
                log.info("ble_connecting", address=self._address, attempt=attempt)
                self._client = BleakClient(self._address)
                await self._client.connect()
                await self._arm_stream()
                self._connected = True
                log.info("ble_connected", address=self._address)
                return
            except Exception as exc:
                log.warning(
                    "ble_connect_failed",
                    address=self._address,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < max_attempts:
                    await asyncio.sleep(2.0)
        raise RuntimeError(f"Failed to connect to Muse S at {self._address}")

    async def _arm_stream(self) -> None:
        """Send BLE arming sequence.

        Arming sequence: halt -> 50ms -> preset[p50] -> 50ms -> start -> 50ms
        -> data[1/2] -> 250ms -> data[2/2]
        """
        client = self._client
        await client.write_gatt_char(_UUID_CONTROL, _CMD_HALT, response=True)
        await asyncio.sleep(0.05)
        await client.write_gatt_char(_UUID_CONTROL, _CMD_PRESET, response=True)
        await asyncio.sleep(0.05)
        await client.write_gatt_char(_UUID_CONTROL, _CMD_START, response=True)
        await asyncio.sleep(0.05)
        await self._ctrl_data_double()

    async def _ctrl_data_double(self) -> None:
        """Send CMD_DATA twice with 250ms gap (critical Muse S Gen 2 protocol)."""
        client = self._client
        await client.write_gatt_char(_UUID_CONTROL, _CMD_DATA, response=True)
        await asyncio.sleep(_DATA_DOUBLE_SEND_GAP)
        await client.write_gatt_char(_UUID_CONTROL, _CMD_DATA, response=True)

    async def _keepalive_task_fn(self) -> None:
        """Re-arm stream every KEEPALIVE_SEC to prevent Muse S idle timeout."""
        try:
            while self._connected:
                await asyncio.sleep(KEEPALIVE_SEC)
                if self._connected and self._client is not None:
                    log.debug("ble_keepalive", address=self._address)
                    await self._ctrl_data_double()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("ble_keepalive_error", error=str(exc))

    async def disconnect(self) -> None:
        """Disconnect BLE client."""
        self._connected = False
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:
                log.warning("ble_disconnect_error", error=str(exc))

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "muse_ble"

    def _eeg_callback(self, channel: str, sender: object, data: bytearray) -> None:  # noqa: ARG002
        """Notification callback for EEG GATT characteristic."""
        samples = decode_eeg(bytes(data))
        self._eeg_bufs[channel].extend(samples)

    def _ppg_callback(self, sender: object, data: bytearray) -> None:  # noqa: ARG002
        """Notification callback for PPG GATT characteristic."""
        samples = decode_ppg(bytes(data))
        self._ppg_buf.extend(samples)

    def _accel_callback(self, sender: object, data: bytearray) -> None:  # noqa: ARG002
        """Notification callback for accelerometer GATT characteristic."""
        flat, _ = decode_imu(bytes(data))
        self._accel_buf.extend(flat)

    def _gyro_callback(self, sender: object, data: bytearray) -> None:  # noqa: ARG002
        """Notification callback for gyroscope GATT characteristic."""
        _, flat = decode_imu(bytes(data))
        self._gyro_buf.extend(flat)

    async def stream(self) -> AsyncGenerator[EEGSample, None]:  # type: ignore[override]
        """Subscribe to GATT notifications and yield EEGSamples at PUBLISH_HZ."""
        client = self._client
        interval = 1.0 / PUBLISH_HZ

        # Subscribe to EEG channels
        for ch, uuid in zip(_EEG_CHANNELS, _EEG_UUIDS):
            ch_name = ch

            def make_eeg_cb(channel: str):
                def cb(sender: object, data: bytearray) -> None:
                    self._eeg_callback(channel, sender, data)
                return cb

            await client.start_notify(uuid, make_eeg_cb(ch_name))

        # Subscribe to PPG
        await client.start_notify(_UUID_PPG1, self._ppg_callback)

        # Subscribe to IMU
        await client.start_notify(_UUID_ACC, self._accel_callback)
        await client.start_notify(_UUID_GYRO, self._gyro_callback)

        # Start keepalive
        self._keepalive_task = asyncio.create_task(self._keepalive_task_fn())

        # Publish frames at PUBLISH_HZ
        while self._connected:
            await asyncio.sleep(interval)
            sample = EEGSample(
                timestamp=time.time(),
                eeg={ch: list(self._eeg_bufs[ch]) for ch in _EEG_CHANNELS},
                ppg=list(self._ppg_buf),
                accel=list(self._accel_buf),
                gyro=list(self._gyro_buf),
                source="muse_ble",
                address=self._address,
                poor_contact=False,
                contact_quality=None,
            )
            # Rotate buffers (keep last N samples)
            _max_eeg = int(256 * 4)
            _max_ppg = int(64 * 30)
            _max_imu = int(52 * 4) * 3
            for ch in _EEG_CHANNELS:
                self._eeg_bufs[ch] = self._eeg_bufs[ch][-_max_eeg:]
            self._ppg_buf = self._ppg_buf[-_max_ppg:]
            self._accel_buf = self._accel_buf[-_max_imu:]
            self._gyro_buf = self._gyro_buf[-_max_imu:]
            yield sample
