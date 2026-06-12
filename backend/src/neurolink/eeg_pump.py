"""EEGPump — background asyncio task that reads from the adapter at 4 Hz.

Builds IngestPayload from EEGSample and calls hub.update().

Stage 0 integration
-------------------
When a Stage0Guard is supplied:
  1. gate_sample() annotates motion flags on every raw EEGSample.
  2. acquisition_ready is checked before hub.update(); frames dropped
     when not ready (except mock source).

Stage 1 integration
-------------------
After eeg_arr is assembled, apply_online_filters() runs the zero-phase
FIR chain (HP + notch(es) + LP) on the buffer.  The filtered array
flows into all downstream DSP.

Stage 2 integration
-------------------
After Stage 1 filtering:
  1. detector.update(eeg_arr) updates EMA variance/PSD stats.
  2. bad = detector.get_bad_channels() returns names of bad channels.
  3. eeg_arr = interpolate_bad_channels(eeg_arr, bad) replaces bad
     channels with spherical-spline estimates from good neighbours.
  4. bad_channels list is carried through IngestPayload to hub /
     NeurolinkState / SSE stream so the UI can show a per-channel
     quality indicator.

Stage 3 integration
-------------------
After Stage 2 interpolation:
  1. gate.evaluate(eeg_arr, accel_arr) runs three independent passes:
       a. Amplitude threshold (±100 µV default) — EEG channels only
       b. IMU motion gate (0.15 g RMS default) — from accel_buffer
       c. Kurtosis burst detection (excess kurtosis > 5.0 default)
  2. If decision.reject is True:
       - Band powers, derived EEG (FAA/FMt) are zeroed / skipped
       - artifact_rejected=True and artifact_reasons are set
       - PPG, breathing, IMU orientation still computed (unaffected)
  3. decision is carried through IngestPayload → NeurolinkState →
     SSE stream so the frontend can show a per-frame quality indicator.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog

from neurolink.dsp.artifact_gate import ArtifactGate
from neurolink.dsp.bad_channels import BadChannelDetector
from neurolink.dsp.online_filter import FilterChainRegistry, get_registry
from neurolink.dsp.spherical_spline import interpolate_bad_channels
from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.models.eeg import BandPowers, BreathingPayload, IMUPayload, IngestPayload

if TYPE_CHECKING:
    from neurolink.stage0 import Stage0Guard

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_ACCEL_FS: float = 52.0
_WATCHDOG_SEC: float = 10.0
_EEG_SAMPLES_WINDOW: int = 64


class EEGPump:
    """Background asyncio task that drives the EEG processing pipeline.

    Pipeline per tick
    -----------------
    1.  Read EEGSample from adapter
    2.  [Stage 0] IMU motion gate
    3.  [Stage 0] Impedance update
    4.  [Stage 0] Acquisition readiness gate
    5.  [Stage 1] Zero-phase FIR filter chain
    6.  [Stage 2] Bad channel detection (EMA update)
    7.  [Stage 2] Spherical spline interpolation of bad channels
    8.  [Stage 3] Epoch-level artifact gate (amplitude / IMU / kurtosis)
    9.  Band powers from interpolated+filtered buffer (clean frames only)
    10. Derived EEG (FAA, FMt)  (clean frames only)
    11. PPG HRV
    12. Breathing
    13. IMU head orientation
    14. Build IngestPayload (bad_channels, artifact_rejected, artifact_reasons)
    15. hub.update()
    """

    def __init__(
        self,
        adapter: HardwareAdapter,
        hub,
        publish_hz: float = 4.0,
        stage0_guard: "Stage0Guard | None" = None,
        stage1_registry: FilterChainRegistry | None = None,
        bad_channel_detector: BadChannelDetector | None = None,
        artifact_gate: ArtifactGate | None = None,
    ) -> None:
        self._adapter = adapter
        self._hub = hub
        self._publish_hz = publish_hz
        self._stage0 = stage0_guard
        self._stage1: FilterChainRegistry = stage1_registry or get_registry()
        self._stage2: BadChannelDetector = bad_channel_detector or BadChannelDetector()
        self._stage3: ArtifactGate = artifact_gate or ArtifactGate()
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_frame_ts: float = 0.0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._pump_loop())
        log.info("eeg_pump_started", publish_hz=self._publish_hz)

    async def stop(self) -> None:
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
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _tick(self) -> None:
        sample = await self._adapter.read_sample()
        if sample is None:
            return

        # ── Stage 0 ───────────────────────────────────────────────────────────────────────
        if self._stage0 is not None:
            sample = self._stage0.gate_sample(sample)
        if self._stage0 is not None:
            self._stage0.impedance.update_from_sample(
                poor_contact=sample.poor_contact,
                channels=sample.channels,
            )
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
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        from neurolink.dsp.breathing import compute_breathing
        from neurolink.dsp.derived_eeg import derived_eeg
        from neurolink.dsp.imu import head_orientation
        from neurolink.dsp.ppg import compute_ppg

        # ── Assemble raw EEG array ───────────────────────────────────────
        eeg_arr: np.ndarray | None = None
        if sample.eeg_buffer:
            _min_len = min(len(b) for b in sample.eeg_buffer)
            if _min_len >= 2:
                eeg_arr = np.array(
                    [b[:_min_len] for b in sample.eeg_buffer], dtype=np.float32
                )

        # ── Stage 1 — zero-phase FIR filter chain ──────────────────────
        if eeg_arr is not None:
            eeg_arr = self._stage1.apply(eeg_arr)

        # ── Stage 2 — bad channel detection & interpolation ─────────────
        bad_channels: list[str] = []
        if eeg_arr is not None:
            self._stage2.update(eeg_arr)
            bad_channels = self._stage2.get_bad_channels()
            if bad_channels:
                eeg_arr = interpolate_bad_channels(eeg_arr, bad_channels)
                log.debug(
                    "stage2_interpolated",
                    bad=bad_channels,
                    n_bad=len(bad_channels),
                )

        # ── Stage 3 — epoch-level artifact gate ────────────────────────
        artifact_rejected: bool = False
        artifact_reasons: list[str] = []
        if eeg_arr is not None:
            # Build accel array for IMU gate (same buffer used by IMU orientation)
            accel_arr: np.ndarray | None = None
            if sample.accel_buffer and len(sample.accel_buffer) >= 3:
                try:
                    accel_arr = np.array(sample.accel_buffer, dtype=np.float32)
                except Exception:
                    accel_arr = None

            decision = self._stage3.evaluate(eeg_arr, accel_arr)
            if decision.reject:
                artifact_rejected = True
                artifact_reasons = decision.reasons
                log.debug(
                    "stage3_frame_rejected",
                    reasons=artifact_reasons,
                    n_bad_ch=len(bad_channels),
                )

        # ── Band powers (clean frames only) ─────────────────────────────
        bands_dict: dict[str, float] = {}
        if eeg_arr is not None and not artifact_rejected:
            bands_dict = compute_band_powers_from_buffer(eeg_arr, fs=_EEG_FS)

        bands = BandPowers(
            alpha=bands_dict.get("alpha", 0.0),
            theta=bands_dict.get("theta", 0.0),
            beta=bands_dict.get("beta", 0.0),
            delta=bands_dict.get("delta", 0.0),
            gamma=bands_dict.get("gamma", 0.0),
        )

        # ── Raw EEG sample window (filtered + interpolated) ───────────────
        eeg_samples: list[list[float]] = []
        if eeg_arr is not None and eeg_arr.ndim == 2:
            n_samples = eeg_arr.shape[1]
            start = max(0, n_samples - _EEG_SAMPLES_WINDOW)
            eeg_samples = eeg_arr[:, start:].tolist()

        # ── Derived EEG (FAA, FMt) — clean frames only ──────────────────
        faa: float | None = None
        fmt: float | None = None
        if eeg_arr is not None and eeg_arr.shape[1] >= 2 and not artifact_rejected:
            from neurolink.dsp.derived_eeg import derived_eeg as _derived
            derived = _derived(eeg_arr, fs=_EEG_FS)
            faa = derived.get("faa")
            fmt = derived.get("fmt")

        # ── PPG HRV ───────────────────────────────────────────────────
        ppg_payload = None
        if sample.ppg_buffer:
            ppg_arr = np.array(sample.ppg_buffer, dtype=np.float32)
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        # ── Breathing ──────────────────────────────────────────────────
        breathing_payload: BreathingPayload | None = None
        accel_z: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            accel_z = np.array(sample.accel_buffer[2], dtype=np.float32)
        ibis: list[float] = ppg_payload.ibi_ms if ppg_payload else []
        breathing_payload = compute_breathing(ibis, accel_z=accel_z)

        # ── IMU head orientation ───────────────────────────────────────
        imu_payload: IMUPayload | None = None
        if sample.accel_buffer and sample.gyro_buffer:
            accel_arr_imu = np.array(sample.accel_buffer, dtype=np.float32)
            gyro_arr = np.array(sample.gyro_buffer, dtype=np.float32)
            if accel_arr_imu.shape[1] > 0:
                imu_payload = head_orientation(accel_arr_imu, gyro_arr)

        # ── fNIRS (Athena) ──────────────────────────────────────────
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
            bad_channels=bad_channels,
            artifact_rejected=artifact_rejected,
            artifact_reasons=artifact_reasons,
        )
