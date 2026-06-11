"""Muse S Athena (Gen 2) OpenMuse LSL consumer adapter.

Ported from Rigpa-v3 hardware/muse_athena/ble_adapter.py.
Requires OpenMuse process running externally.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder

log = structlog.get_logger(__name__)

_EEG_STREAM_TYPE = "EEG"
_FNIRS_STREAM_TYPE = "NIRS"
_RESOLVE_TIMEOUT: float = 5.0
_PUBLISH_HZ: float = 4.0
_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]


class AthenaBlueAdapter(HardwareAdapter):
    """LSL-based adapter for Muse S Athena via OpenMuse.

    Consumes EEG + fNIRS streams from OpenMuse LSL outlets.
    pylsl is lazily imported.
    """

    def __init__(self) -> None:
        self._connected = False
        self._eeg_inlet = None  # type: ignore[assignment]
        self._fnirs_inlet = None  # type: ignore[assignment]
        self._fnirs_decoder = FNIRSDecoder()

    async def connect(self) -> None:
        """Resolve EEG and optional fNIRS LSL streams."""
        import pylsl  # lazy import

        log.info("athena_lsl_resolving")
        streams = await asyncio.to_thread(
            pylsl.resolve_stream, "type", _EEG_STREAM_TYPE, timeout=_RESOLVE_TIMEOUT
        )
        if not streams:
            raise RuntimeError("No LSL EEG stream found. Is OpenMuse running?")
        self._eeg_inlet = pylsl.StreamInlet(streams[0])
        # Try to find fNIRS stream (optional)
        try:
            fnirs_streams = await asyncio.to_thread(
                pylsl.resolve_stream, "type", _FNIRS_STREAM_TYPE, timeout=1.0
            )
            if fnirs_streams:
                self._fnirs_inlet = pylsl.StreamInlet(fnirs_streams[0])
                log.info("athena_fnirs_connected")
        except Exception:
            log.info("athena_fnirs_not_found")
        self._connected = True
        log.info("athena_connected")

    async def disconnect(self) -> None:
        """Close LSL inlets."""
        self._connected = False
        for inlet in (self._eeg_inlet, self._fnirs_inlet):
            if inlet is not None:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
        self._eeg_inlet = None
        self._fnirs_inlet = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def source_name(self) -> str:
        return "athena_ble"

    async def stream(self) -> AsyncGenerator[EEGSample, None]:  # type: ignore[override]
        """Yield EEGSamples with optional fNIRS data at PUBLISH_HZ."""
        interval = 1.0 / _PUBLISH_HZ
        while self._connected:
            await asyncio.sleep(interval)
            if self._eeg_inlet is None:
                continue
            try:
                chunk, _ = await asyncio.to_thread(
                    self._eeg_inlet.pull_chunk, timeout=0.0
                )
                eeg: dict[str, list[float]] = {ch: [] for ch in _EEG_CHANNELS}
                if chunk:
                    for row in chunk:
                        for i, ch in enumerate(_EEG_CHANNELS):
                            if i < len(row):
                                eeg[ch].append(float(row[i]))

                fnirs_raw: list[float] | None = None
                if self._fnirs_inlet is not None:
                    try:
                        fsample, _ = await asyncio.to_thread(
                            self._fnirs_inlet.pull_sample, timeout=0.0
                        )
                        if fsample:
                            fnirs_raw = [float(v) for v in fsample]
                    except Exception:
                        pass

                yield EEGSample(
                    timestamp=time.time(),
                    eeg=eeg,
                    source="athena_ble",
                    address="",
                    fnirs=fnirs_raw,
                )
            except Exception as exc:
                log.warning("athena_lsl_error", error=str(exc))
