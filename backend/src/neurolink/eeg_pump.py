"""EEGPump — background asyncio task that reads from the adapter at 4 Hz.

Builds IngestPayload from EEGSample and calls hub.update().
Ported from Rigpa-v2 eeg_pump.py + Rigpa-v3 eeg_pump.py.

Stage 0 integration
-------------------
When a Stage0Guard is supplied:
  1. gate_sample() is called on every raw EEGSample to annotate motion flags.
  2. acquisition_ready is checked before hub.update(); if not ready the frame
     is dropped silently (except in mock mode where we always forward so that
     tests and demos keep producing state updates).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog

from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.models.eeg import BandPowers, BreathingPayload, IMUPayload, IngestPayload

if TYPE_CHECKING:
    from neurolink.stage0 import Stage0Guard

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_ACCEL_FS: float = 52.0
_WATCHDOG_SEC: float = 10.0
# Number of raw samples to forward per SSE frame for spectrogram rendering.
# At 256 Hz EEG and 4 Hz publish rate the buffer holds ~64 samples/channel.
_EEG_SAMPLES_WINDOW: int = 64


class EEGPump:
    """Background asyncio task that drives the EEG processing pipeline.

    At each tick (1/publish_hz seconds):
    1. Read EEGSample from adapter
    2. [Stage 0] Apply IMU motion gate (flag_segment)
    3. [Stage 0] Check acquisition_ready gate
    4. Compute band powers from EEG buffer
    5. Compute derived EEG (FAA, FMt)
    6. Compute PPG HRV if buffer available
    7. Compute breathing rate if IMU available
    8. Compute head orientation
    9. Build IngestPayload (including raw eeg_samples window)
    10. Call hub.update()
    """

    def __init__(
        self,
        adapter: HardwareAdapter,
        hub,
        publish_hz: float = 4.0,
        stage0_guard: "Stage0Guard | None" = None,
    ) -> None:
        self._adapter = adapter
        self._hub = hub
        self._publish_hz = publish_hz
        self._stage0 = stage0_guard
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_frame_ts: float = 0.0

    async def start(self) -> None:
        """Start the pump as a background asyncio task."""
        self._running = True
        self._task = asyncio.create_task(self._pump_loop())
        log.info("eeg_pump_started", publish_hz=self._publish_hz)

    async def stop(self) -> None:
        """Stop the pump and cancel the background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("eeg_pump_stopped")

    async def _pump_loop(self) -> None:
        """Main pump loop. Runs at publish_hz frequency."""
        interval = 1.0 / self._publish_hz
        while self._running:
            tick_start = time.monotonic()
            try:
                await self._tick()
            except Exception as exc:
                log.error("eeg_pump_tick_error", error=str(exc), exc_info=True)
            if self._last_frame_ts > 0 and (time.time() - self._last_frame_ts) > _WATCHDOG_SEC:
                log.warning("eeg_pump_no_frames", since_sec=_WATCHDOG_SEC)
            elapsed = time.monotonic() - tick_start
            sleep_time = max(0.0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _tick(self) -> None:
        """Single pump tick: read sample, apply Stage-0 gates, process, update hub."""
        sample = await self._adapter.read_sample()
        if sample is None:
            return

        # ── Stage 0 — IMU motion gate (mutates sample.extra in-place) ────────
        if self._stage0 is not None:
            sample = self._stage0.gate_sample(sample)

        # ── Stage 0 — impedance update from poor_contact flag ──────────────
        if self._stage0 is not None:
            self._stage0.impedance.update_from_sample(
                poor_contact=sample.poor_contact,
                channels=sample.channels,
            )

        # ── Stage 0 — acquisition readiness gate ──────────────────────────
        # In mock mode we always forward frames so tests & demos work without
        # going through the setup wizard.
        if (
            self._stage0 is not None
            and not self._stage0.acquisition_ready
            and sample.source != "mock"
        ):
            log.debug(
                "stage0_frame_held",
                impedance_ok=self._stage0.impedance.all_channels_ok,
                env_ready=self._stage0.environment.is_ready,
                motion_flagged=sample.extra.get("motion_flagged", False),
            )
            return

        self._last_frame_ts = time.time()
        self._hub.set_latest_sample(sample)

        payload = await self._build_payload(sample)
        self._hub.update(payload)

    async def _build_payload(self, sample: EEGSample) -> IngestPayload:
        """Build an IngestPayload from a raw EEGSample."""
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        from neurolink.dsp.breathing import compute_breathing
        from neurolink.dsp.derived_eeg import derived_eeg
        from neurolink.dsp.imu import head_orientation
        from neurolink.dsp.ppg import compute_ppg

        # ── Band powers ──────────────────────────────────────────────────────
        bands_dict: dict[str, float] = {}
        eeg_arr: np.ndarray | None = None
        if sample.eeg_buffer:
            _min_len = min(len(b) for b in sample.eeg_buffer)
            if _min_len >= 2:
                eeg_arr = np.array([b[:_min_len] for b in sample.eeg_buffer], dtype=np.float32)
                bands_dict = compute_band_powers_from_buffer(eeg_arr, fs=_EEG_FS)

        bands = BandPowers(
            alpha=bands_dict.get("alpha", 0.0),
            theta=bands_dict.get("theta", 0.0),
            beta=bands_dict.get("beta", 0.0),
            delta=bands_dict.get("delta", 0.0),
            gamma=bands_dict.get("gamma", 0.0),
        )

        # ── Raw EEG sample window for spectrogram / topo ──────────────────────
        eeg_samples: list[list[float]] = []
        if eeg_arr is not None and eeg_arr.ndim == 2:
            n_samples = eeg_arr.shape[1]
            start = max(0, n_samples - _EEG_SAMPLES_WINDOW)
            eeg_samples = eeg_arr[:, start:].tolist()

        # ── Derived EEG (FAA, FMt) ─────────────────────────────────────────
        faa: float | None = None
        fmt: float | None = None
        derived: dict = {}
        if sample.eeg_buffer:
            _min2 = min(len(b) for b in sample.eeg_buffer)
            if _min2 >= 2:
                eeg_arr2 = np.array([b[:_min2] for b in sample.eeg_buffer], dtype=np.float32)
                derived = derived_eeg(eeg_arr2, fs=_EEG_FS)
            faa = derived.get("faa")
            fmt = derived.get("fmt")

        # ── PPG HRV ─────────────────────────────────────────────────────────
        ppg_payload = None
        if sample.ppg_buffer:
            ppg_arr = np.array(sample.ppg_buffer, dtype=np.float32)
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        # ── Breathing ─────────────────────────────────────────────────────────
        breathing_payload: BreathingPayload | None = None
        accel_z: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            accel_z = np.array(sample.accel_buffer[2], dtype=np.float32)

        ibis: list[float] = ppg_payload.ibi_ms if ppg_payload else []
        breathing_payload = compute_breathing(ibis, accel_z=accel_z)

        # ── IMU head orientation ───────────────────────────────────────────────
        imu_payload: IMUPayload | None = None
        if sample.accel_buffer and sample.gyro_buffer:
            accel_arr = np.array(sample.accel_buffer, dtype=np.float32)
            gyro_arr = np.array(sample.gyro_buffer, dtype=np.float32)
            if accel_arr.shape[1] > 0:
                imu_payload = head_orientation(accel_arr, gyro_arr)

        # ── fNIRS (Athena) ───────────────────────────────────────────────────
        fnirs_oxy: float | None = sample.extra.get("fnirs_oxy")
        fnirs_deoxy: float | None = sample.extra.get("fnirs_deoxy")

        return IngestPayload(
            source=sample.source,
            address=sample.address,
            timestamp=sample.timestamp,
            bands=bands,
            poor_contact=sample.poor_contact,
            faa=faa,
            fmt=fmt,
            ppg=ppg_payload,
            breathing=breathing_payload,
            imu=imu_payload,
            fnirs_oxy=fnirs_oxy,
            fnirs_deoxy=fnirs_deoxy,
            eeg_samples=eeg_samples,
        )
