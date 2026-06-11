"""Muse S Gen 1 LSL adapter (pylsl consumer).

Ported from Rigpa-v3 hardware/muse_s/lsl_adapter.py.
Requires muselsl running externally:
  muselsl stream --address <mac>
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_RING_SECS: float = 4.0
_N_EEG: int = int(_EEG_FS * _RING_SECS)
_N_PPG: int = int(64.0 * 30.0)
_STREAM_NAME_EEG: str = "Muse"
_RESOLVE_TIMEOUT_SEC: float = 5.0


class MuseSLslAdapter(HardwareAdapter):
    """Consumes Muse S EEG via pylsl (muselsl LSL outlet).

    Lazy-imports pylsl so mock mode never loads LSL library.
    """

    def __init__(self) -> None:
        self._inlet = None
        self._connected: bool = False
        self._eeg_rings: list[deque] = [deque(maxlen=_N_EEG) for _ in range(5)]
        self._ppg_ring: deque = deque(maxlen=_N_PPG)

    async def connect(self) -> None:
        """Resolve LSL stream and create inlet."""
        import pylsl  # lazy import

        loop = asyncio.get_event_loop()
        streams = await loop.run_in_executor(
            None,
            lambda: pylsl.resolve_stream("type", "EEG", 1, _RESOLVE_TIMEOUT_SEC),
        )
        if not streams:
            raise RuntimeError("No LSL EEG stream found. Is muselsl stream running?")
        self._inlet = pylsl.StreamInlet(streams[0])
        self._connected = True
        log.info("muse_lsl_connected")

    async def disconnect(self) -> None:
        """Close LSL inlet."""
        if self._inlet:
            self._inlet.close_stream()
            self._inlet = None
        self._connected = False
        log.info("muse_lsl_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "muse_lsl"

    async def read_sample(self) -> EEGSample | None:
        """Pull available samples from LSL inlet into ring buffers."""
        if not self._connected or not self._inlet:
            return None

        loop = asyncio.get_event_loop()
        try:
            chunk, _ = await loop.run_in_executor(
                None, lambda: self._inlet.pull_chunk(timeout=0.01, max_samples=64)
            )
        except Exception as exc:
            log.warning("muse_lsl_read_error", error=str(exc))
            return None

        for sample in chunk:
            # sample = [TP9, AF7, AF8, TP10, AUX (optional)]
            for ch, val in enumerate(sample[:5]):
                if ch < len(self._eeg_rings):
                    self._eeg_rings[ch].append(float(val))

        eeg_buf = [list(ring) for ring in self._eeg_rings]
        channels = [buf[-1] if buf else 0.0 for buf in eeg_buf]

        return EEGSample(
            channels=channels,
            timestamp=time.time(),
            source="muse_lsl",
            address="",
            poor_contact=False,
            eeg_buffer=eeg_buf,
        )
