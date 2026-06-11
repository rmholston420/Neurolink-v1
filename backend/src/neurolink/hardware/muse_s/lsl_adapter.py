"""Muse S Gen 1 LSL consumer adapter.

Ported from Rigpa-v3 hardware/muse_s/lsl_adapter.py.
Requires muselsl running externally.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

import numpy as np
import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter

log = structlog.get_logger(__name__)

_STREAM_TYPE = "EEG"
_RESOLVE_TIMEOUT: float = 5.0
_PUBLISH_HZ: float = 4.0
_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]


class MuseSLslAdapter(HardwareAdapter):
    """LSL-based EEG adapter for Muse S Gen 1.

    Consumes from an LSL outlet created by muselsl.
    pylsl is lazily imported to allow mock mode without hardware.
    """

    def __init__(self) -> None:
        self._connected = False
        self._inlet = None  # type: ignore[assignment]
        self._eeg_buf: list[list[float]] = []

    async def connect(self) -> None:
        """Resolve and open an LSL EEG stream."""
        import pylsl  # lazy import

        log.info("lsl_resolving", stream_type=_STREAM_TYPE)
        streams = await asyncio.to_thread(
            pylsl.resolve_stream, "type", _STREAM_TYPE, timeout=_RESOLVE_TIMEOUT
        )
        if not streams:
            raise RuntimeError("No LSL EEG stream found. Is muselsl running?")
        self._inlet = pylsl.StreamInlet(streams[0])
        self._connected = True
        log.info("lsl_connected")

    async def disconnect(self) -> None:
        """Close the LSL inlet."""
        self._connected = False
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "muse_lsl"

    async def stream(self) -> AsyncGenerator[EEGSample, None]:  # type: ignore[override]
        """Pull LSL samples and yield EEGSamples at PUBLISH_HZ."""
        interval = 1.0 / _PUBLISH_HZ
        while self._connected:
            await asyncio.sleep(interval)
            if self._inlet is None:
                continue
            try:
                chunk, timestamps = await asyncio.to_thread(
                    self._inlet.pull_chunk, timeout=0.0
                )
                if chunk:
                    # chunk is list of [TP9, AF7, AF8, TP10, AUX] rows
                    eeg: dict[str, list[float]] = {ch: [] for ch in _EEG_CHANNELS}
                    for row in chunk:
                        for i, ch in enumerate(_EEG_CHANNELS):
                            if i < len(row):
                                eeg[ch].append(float(row[i]))
                    yield EEGSample(
                        timestamp=time.time(),
                        eeg=eeg,
                        ppg=None,
                        accel=None,
                        gyro=None,
                        source="muse_lsl",
                        address="",
                    )
            except Exception as exc:
                log.warning("lsl_pull_error", error=str(exc))
