"""EEG Pump — background asyncio task that reads from adapter at 4 Hz.

Ported from Rigpa-v2 eeg_pump.py + Rigpa-v3 eeg_pump.py.
Builds IngestPayload from EEGSample and calls hub.update().
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import numpy as np
import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.models.eeg import (
    BandPowers,
    BreathingPayload,
    IMUPayload,
    IngestPayload,
    PPGPayload,
)

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_IMU_FS: float = 52.0
_WATCHDOG_SEC: float = 5.0  # warn if no frames in this many seconds

_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10", "AUX"]


class EEGPump:
    """Reads EEGSamples from an adapter, enriches them, and pushes to hub.

    Runs as a single asyncio background task.
    """

    def __init__(self, adapter: HardwareAdapter, hub) -> None:  # type: ignore[type-arg]
        """
        Args:
            adapter: connected HardwareAdapter instance
            hub: EEGHub instance
        """
        self._adapter = adapter
        self._hub = hub
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._running = False
        self._last_frame_ts: float = 0.0
        # Rolling EEG ring buffer (5 channels x 4s @ 256 Hz)
        _eeg_buf_size = int(_EEG_FS * 4)
        self._eeg_bufs: dict[str, deque[float]] = {
            ch: deque(maxlen=_eeg_buf_size) for ch in _EEG_CHANNELS
        }
        # Rolling PPG ring buffer
        _ppg_buf_size = int(_PPG_FS * 30)
        self._ppg_buf: deque[float] = deque(maxlen=_ppg_buf_size)
        # Rolling IMU buffers
        _imu_buf_size = int(_IMU_FS * 4)
        self._accel_buf: deque[float] = deque(maxlen=_imu_buf_size * 3)
        self._gyro_buf: deque[float] = deque(maxlen=_imu_buf_size * 3)

    async def start(self) -> None:
        """Start the pump as a background asyncio task."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("eeg_pump_started", source=self._adapter.source_name)

    async def stop(self) -> None:
        """Stop the pump task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("eeg_pump_stopped")

    async def _run(self) -> None:
        """Main pump loop: read samples, build payload, update hub."""
        watchdog_task = asyncio.create_task(self._watchdog())
        try:
            async for sample in self._adapter.stream():
                if not self._running:
                    break
                self._ingest(sample)
                payload = self._build_payload(sample)
                self._hub.update(payload)
                self._last_frame_ts = time.time()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("eeg_pump_error", error=str(exc), exc_info=True)
        finally:
            watchdog_task.cancel()

    async def _watchdog(self) -> None:
        """Warn if no frames received within _WATCHDOG_SEC."""
        await asyncio.sleep(_WATCHDOG_SEC)
        while self._running:
            if self._last_frame_ts > 0:
                age = time.time() - self._last_frame_ts
                if age > _WATCHDOG_SEC:
                    log.warning("eeg_pump_watchdog", age_sec=age)
            await asyncio.sleep(_WATCHDOG_SEC)

    def _ingest(self, sample: EEGSample) -> None:
        """Append sample data to ring buffers."""
        if isinstance(sample.eeg, dict):
            for ch in _EEG_CHANNELS:
                samples = sample.eeg.get(ch, [])
                self._eeg_bufs[ch].extend(samples)
        if sample.ppg:
            self._ppg_buf.extend(sample.ppg)
        if sample.accel:
            self._accel_buf.extend(sample.accel)
        if sample.gyro:
            self._gyro_buf.extend(sample.gyro)

    def _build_payload(self, sample: EEGSample) -> IngestPayload:
        """Build an IngestPayload from ring buffers + current sample."""
        # Build numpy EEG array
        eeg_arrays = np.array(
            [list(self._eeg_bufs[ch]) for ch in _EEG_CHANNELS], dtype=np.float32
        )

        # Band powers
        bands = self._compute_bands(eeg_arrays)

        # Derived EEG
        faa: float | None = None
        fmt: float | None = None
        if eeg_arrays.shape[1] >= 2:
            from neurolink.dsp.derived_eeg import derived_eeg
            derived = derived_eeg(eeg_arrays)
            faa = derived.get("faa")
            fmt = derived.get("fmt")

        # Poor contact heuristic
        poor_contact = bands.delta > 0.50

        # PPG
        ppg_payload: PPGPayload | None = None
        ppg_arr = np.array(list(self._ppg_buf), dtype=np.float32)
        if len(ppg_arr) >= int(_PPG_FS * 5):
            from neurolink.dsp.ppg import compute_ppg
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        # Breathing
        breathing_payload: BreathingPayload | None = None
        if ppg_payload and ppg_payload.ibi_ms:
            accel_arr = np.array(list(self._accel_buf), dtype=np.float32)
            # Extract Z-axis (every 3rd value starting at 2)
            accel_z = accel_arr[2::3] if len(accel_arr) >= 3 else np.array([])
            from neurolink.dsp.breathing import compute_breathing
            breathing_payload = compute_breathing(
                ibis_ms=ppg_payload.ibi_ms,
                accel_z=accel_z if len(accel_z) > 0 else None,
            )

        # IMU
        imu_payload: IMUPayload | None = None
        if len(self._accel_buf) >= 9:  # at least 3 samples * 3 axes
            accel_arr = np.array(list(self._accel_buf), dtype=np.float32)
            gyro_arr = np.array(list(self._gyro_buf), dtype=np.float32)
            # Reshape: (3, N)
            n_accel = (len(accel_arr) // 3) * 3
            if n_accel > 0:
                accel_m = accel_arr[:n_accel].reshape(-1, 3).T
                gyro_m = gyro_arr[:n_accel].reshape(-1, 3).T if len(gyro_arr) >= n_accel else None
                from neurolink.dsp.imu import head_orientation
                imu_payload = head_orientation(accel_m, gyro_m)

        # fNIRS (Athena)
        fnirs_oxy: float | None = None
        fnirs_deoxy: float | None = None
        if sample.fnirs:
            from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder
            decoded = FNIRSDecoder().decode(sample.fnirs)
            fnirs_oxy = decoded.get("fnirs_oxy")
            fnirs_deoxy = decoded.get("fnirs_deoxy")

        return IngestPayload(
            timestamp=sample.timestamp,
            source=sample.source,
            address=sample.address,
            bands=bands,
            poor_contact=poor_contact,
            contact_quality=sample.contact_quality,
            faa=faa,
            fmt=fmt,
            ppg=ppg_payload,
            breathing=breathing_payload,
            imu=imu_payload,
            fnirs_oxy=fnirs_oxy,
            fnirs_deoxy=fnirs_deoxy,
        )

    def _compute_bands(self, eeg_arrays: np.ndarray) -> BandPowers:
        """Compute band powers from EEG ring buffer."""
        if eeg_arrays.shape[1] < 2:
            return BandPowers()
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        raw = compute_band_powers_from_buffer(eeg_arrays, fs=_EEG_FS)
        return BandPowers(
            delta=raw.get("delta", 0.0),
            theta=raw.get("theta", 0.0),
            alpha=raw.get("alpha", 0.0),
            beta=raw.get("beta", 0.0),
            gamma=raw.get("gamma", 0.0),
        )
