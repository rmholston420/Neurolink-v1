"""Muse S Athena (Gen 2) BLE adapter via OpenMuse LSL outlet.

Ported from Rigpa-v3 hardware/muse_athena/ble_adapter.py.
Requires OpenMuse subprocess running externally (not managed here).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_RING_SECS: float = 4.0
_N_EEG: int = int(_EEG_FS * _RING_SECS)
_N_PPG: int = int(64.0 * 30.0)


class AthenaBlueAdapter(HardwareAdapter):
    """Consumes Muse S Athena data via OpenMuse LSL outlets.

    Supports EEG, PPG, IMU, and fNIRS.
    Lazy-imports pylsl.
    """

    def __init__(self) -> None:
        self._eeg_inlet = None
        self._fnirs_inlet = None
        self._connected: bool = False
        self._eeg_rings: list[deque] = [deque(maxlen=_N_EEG) for _ in range(5)]
        self._ppg_ring: deque = deque(maxlen=_N_PPG)
        self._fnirs_decoder = FNIRSDecoder()
        self._latest_fnirs: dict[str, float] = {}

    async def connect(self) -> None:
        """Resolve OpenMuse LSL EEG and fNIRS streams."""
        import pylsl  # lazy import

        loop = asyncio.get_event_loop()

        # EEG stream
        eeg_streams = await loop.run_in_executor(
            None, lambda: pylsl.resolve_stream("type", "EEG", 1, 5.0)
        )
        if not eeg_streams:
            raise RuntimeError("No EEG LSL stream found for Athena. Is OpenMuse running?")
        self._eeg_inlet = pylsl.StreamInlet(eeg_streams[0])

        # Optional fNIRS stream (don't fail if absent)
        try:
            fnirs_streams = await loop.run_in_executor(
                None, lambda: pylsl.resolve_stream("type", "NIRS", 1, 2.0)
            )
            if fnirs_streams:
                self._fnirs_inlet = pylsl.StreamInlet(fnirs_streams[0])
                log.info("athena_fnirs_stream_found")
        except Exception:
            log.info("athena_fnirs_stream_not_found")

        self._connected = True
        log.info("athena_ble_connected")

    async def disconnect(self) -> None:
        """Close all LSL inlets."""
        for inlet in (self._eeg_inlet, self._fnirs_inlet):
            if inlet:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
        self._eeg_inlet = None
        self._fnirs_inlet = None
        self._connected = False
        log.info("athena_ble_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "athena_ble"

    async def read_sample(self) -> EEGSample | None:
        """Pull latest EEG and fNIRS samples."""
        if not self._connected or not self._eeg_inlet:
            return None

        loop = asyncio.get_event_loop()

        # Pull EEG
        try:
            chunk, _ = await loop.run_in_executor(
                None, lambda: self._eeg_inlet.pull_chunk(timeout=0.01, max_samples=64)
            )
            for sample in chunk:
                for ch, val in enumerate(sample[:5]):
                    if ch < len(self._eeg_rings):
                        self._eeg_rings[ch].append(float(val))
        except Exception as exc:
            log.warning("athena_eeg_read_error", error=str(exc))

        # Pull fNIRS if available
        if self._fnirs_inlet:
            try:
                fnirs_sample, _ = await loop.run_in_executor(
                    None, lambda: self._fnirs_inlet.pull_sample(timeout=0.001)
                )
                if fnirs_sample:
                    self._latest_fnirs = self._fnirs_decoder.decode(fnirs_sample)
            except Exception:
                pass

        eeg_buf = [list(ring) for ring in self._eeg_rings]
        channels = [buf[-1] if buf else 0.0 for buf in eeg_buf]

        sample = EEGSample(
            channels=channels,
            timestamp=time.time(),
            source="athena_ble",
            address="",
            poor_contact=False,
            eeg_buffer=eeg_buf,
        )
        if self._latest_fnirs:
            sample.extra.update(self._latest_fnirs)
        return sample
