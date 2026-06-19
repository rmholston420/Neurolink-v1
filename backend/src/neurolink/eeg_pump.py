"""EEGPump — thin async driver that reads the hardware adapter at 4 Hz.

All DSP logic now lives in ``neurolink.dsp.pipeline.EEGPipeline``.
EEGPump's only responsibilities are:
  1. Scheduling the tick loop at ``publish_hz``.
  2. Calling ``self._build_payload(sample)`` which runs stub dispatch.
  3. Converting result → hub.update().
  4. Running the Stage 0 acquisition guard and emitting settling events.
  5. Watchdog: warn when no frames arrive for > ``_WATCHDOG_SEC``.

The module-level stub dispatch layer is retained for backward
compatibility with existing unit tests that patch stub names.
New tests should prefer injecting an ``EEGPipeline`` instance directly.

Stub → method mapping
---------------------
  bad_channels         .detect(eeg)              -> list[str]
  spherical_spline     .interpolate(eeg, bad)    -> ndarray
  asr                  .apply(eeg)               -> ndarray
  ocular_regression    .apply(eeg)               -> ndarray
  baseline             .apply(eeg)               -> ndarray
  cardiac_regression   .apply(eeg, ibis)         -> ndarray
  bandpower            .compute(eeg, fs)         -> dict
  classifiers          .run(bands)               -> dict
  impedance            .check()                  -> bool
  filter_toggles       .get_toggles()            -> FilterToggleConfig
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog

from neurolink.dsp import filter_toggles as _filter_toggles_module
from neurolink.dsp.artifact_detector import ArtifactDetector
from neurolink.dsp.artifact_gate import ArtifactGate
from neurolink.dsp.asr import ArtifactSubspaceReconstructor
from neurolink.dsp.bad_channels import BadChannelDetector
from neurolink.dsp.baseline import BaselineRecorder
from neurolink.dsp.cardiac_regression import CardiacRegressor
from neurolink.dsp.ocular_regression import OcularRegressor
from neurolink.dsp.online_filter import FilterChainRegistry, get_registry
from neurolink.dsp.pipeline import EEGPipeline
from neurolink.dsp.spherical_spline import interpolate_bad_channels
from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.models.eeg import (
    ArtifactAnnotationPayload,
    ArtifactCorrectionPlanPayload,
    BandPowers,
    IMUPayload,
    IngestPayload,
)

if TYPE_CHECKING:
    from neurolink.stage0 import Stage0Guard

log = structlog.get_logger(__name__)

_EEG_FS: float = 256.0
_PPG_FS: float = 64.0
_ACCEL_FS: float = 52.0
_WATCHDOG_SEC: float = 10.0
_EEG_SAMPLES_WINDOW: int = 64
_MIN_PPG_SAMPLES: int = 960


# ---------------------------------------------------------------------------
# Module-level stub classes (preserved for test backward-compatibility)
# ---------------------------------------------------------------------------


class _BadChannelsStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def detect(self, eeg: np.ndarray) -> list:
        if self._pump is not None:
            self._pump._pipeline._stage2.update(eeg)
            return self._pump._pipeline._stage2.get_bad_channels()
        return []


class _SphericalSplineStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def interpolate(self, eeg: np.ndarray, bad: list, **kw) -> np.ndarray:
        return interpolate_bad_channels(eeg, bad)


class _ASRStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._pipeline._stage4.apply(eeg)
        return eeg


class _OcularRegressionStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._pipeline._stage5.apply(eeg)
        return eeg


class _BaselineStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._pipeline._baseline.process(eeg)
        return eeg


class _CardiacRegressionStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def apply(self, eeg: np.ndarray, ibis=None, **kw) -> np.ndarray:
        if self._pump is not None and ibis:
            return self._pump._pipeline._stage6.apply(eeg, ibis, fs=_EEG_FS)
        return eeg


class _BandpowerStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def compute(self, eeg: np.ndarray, fs: float = _EEG_FS, **kw) -> dict:
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer

        return compute_band_powers_from_buffer(eeg, fs=fs)


class _ClassifiersStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def run(self, bands, **kw) -> dict:
        return {}


class _ImpedanceStub:
    def __init__(self):
        self._pump: EEGPump | None = None

    def check(self) -> bool:
        if self._pump is not None and self._pump._pipeline._stage0 is not None:
            return self._pump._pipeline._stage0.impedance.all_channels_ok
        return True


class _FilterTogglesStub:
    def get_toggles(self):
        return _filter_toggles_module.get_toggles()


# Module-level singletons — tests patch these names.
bad_channels = _BadChannelsStub()
spherical_spline = _SphericalSplineStub()
asr = _ASRStub()
ocular_regression = _OcularRegressionStub()
baseline = _BaselineStub()
cardiac_regression = _CardiacRegressionStub()
bandpower = _BandpowerStub()
classifiers = _ClassifiersStub()
impedance = _ImpedanceStub()
filter_toggles = _FilterTogglesStub()


def _wire_stubs(pump: EEGPump) -> None:
    """Inject the pump instance into every stub so they forward correctly."""
    bad_channels._pump = pump
    spherical_spline._pump = pump
    asr._pump = pump
    ocular_regression._pump = pump
    baseline._pump = pump
    cardiac_regression._pump = pump
    bandpower._pump = pump
    classifiers._pump = pump
    impedance._pump = pump


class EEGPump:
    """Thin async driver: reads adapter → runs stub dispatch → updates hub."""

    def __init__(
        self,
        adapter: HardwareAdapter,
        hub,
        publish_hz: float = 4.0,
        stage0_guard: Stage0Guard | None = None,
        stage1_registry: FilterChainRegistry | None = None,
        bad_channel_detector: BadChannelDetector | None = None,
        artifact_gate: ArtifactGate | None = None,
        artifact_detector: ArtifactDetector | None = None,
        asr: ArtifactSubspaceReconstructor | None = None,
        ocular_regressor: OcularRegressor | None = None,
        cardiac_regressor: CardiacRegressor | None = None,
    ) -> None:
        self._adapter = adapter
        self._hub = hub
        self._publish_hz = publish_hz
        self._pipeline = EEGPipeline(
            hub=hub,
            publish_hz=publish_hz,
            stage0_guard=stage0_guard,
            stage1_registry=stage1_registry,
            bad_channel_detector=bad_channel_detector,
            artifact_gate=artifact_gate,
            artifact_detector=artifact_detector,
            asr=asr,
            ocular_regressor=ocular_regressor,
            cardiac_regressor=cardiac_regressor,
        )
        # Direct aliases — kept in sync with pipeline for test patching.
        self._stage0 = self._pipeline._stage0
        self._stage1 = self._pipeline._stage1
        self._stage2 = self._pipeline._stage2
        self._stage3 = self._pipeline._stage3
        self._stage4 = self._pipeline._stage4
        self._stage5 = self._pipeline._stage5
        self._stage6 = self._pipeline._stage6
        self._baseline = self._pipeline._baseline
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_frame_ts: float = 0.0
        self._last_watchdog_warn_ts: float = 0.0
        _wire_stubs(self)

    # ── Public interface ─────────────────────────────────────────────

    @property
    def stream_health(self):
        """Expose the pipeline's StreamHealth for external consumers."""
        return self._pipeline.health

    async def start(self) -> None:
        if self._running:
            return
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

    def reset(self) -> None:
        # Call reset on the direct instance attributes so tests that replace
        # self._baseline / self._stage6 with mocks observe the calls.
        self._baseline.reset()
        self._stage6.reset()
        self._hub.reset()
        # Keep the pipeline in sync.
        self._pipeline._baseline = self._baseline
        self._pipeline._stage6 = self._stage6
        log.info("eeg_pump_reset")

    def _stage0_settling_reason(self) -> str:
        """Forward to the pipeline method; exposed here for test compatibility."""
        return self._pipeline._stage0_settling_reason()

    # ── Pump loop ────────────────────────────────────────────────

    async def _pump_loop(self) -> None:
        interval = 1.0 / self._publish_hz
        while self._running:
            tick_start = time.monotonic()
            try:
                await self._tick()
            except Exception as exc:
                log.error("eeg_pump_tick_error", error=str(exc), exc_info=True)
            now = time.time()
            if (
                self._last_frame_ts > 0
                and (now - self._last_frame_ts) > _WATCHDOG_SEC
                and (now - self._last_watchdog_warn_ts) >= _WATCHDOG_SEC
            ):
                log.warning("eeg_pump_no_frames", since_sec=_WATCHDOG_SEC)
                self._last_watchdog_warn_ts = now
            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _tick(self) -> None:
        sample = await self._adapter.read_sample()
        if sample is None:
            return

        toggles = filter_toggles.get_toggles()

        # Resolve live stage0 reference from pipeline (not the init-time snapshot
        # alias self._stage0) so that test injection via stage0_guard propagates.
        _s0 = self._pipeline._stage0

        # ── Stage 0: IMU gate + impedance update (runs on EVERY tick) ────────
        if _s0 is not None:
            if toggles.imu_gate:
                sample = _s0.gate_sample(sample)
            _s0.impedance.update_from_sample(
                poor_contact=sample.poor_contact,
                channels=sample.channels,
            )

        # ── Impedance settling check ───────────────────────────────────────
        impedance_ok = impedance.check()
        if not impedance_ok:
            self._hub.emit_settling(reason="impedance_unstable")

        # ── Acquisition-ready gate (non-mock sources only) ─────────────────
        if (
            _s0 is not None
            and not _s0.acquisition_ready
            and sample.source != "mock"
        ):
            reason = self._stage0_settling_reason()
            self._hub.emit_settling(reason=reason)
            return

        self._last_frame_ts = time.time()
        self._hub.set_latest_sample(sample)

        payload = await self._build_payload(sample)
        self._hub.update(payload)

    async def _build_payload(self, sample: EEGSample) -> IngestPayload:
        """Run the full stub dispatch pipeline and build an IngestPayload.

        Each DSP stage calls the corresponding module-level stub so that
        tests patching ``neurolink.eeg_pump.<stub>`` intercept the call.
        """
        toggles = filter_toggles.get_toggles()

        eeg = np.asarray(sample.eeg_buffer, dtype=np.float32) if sample.eeg_buffer else None

        # ── Stage 1: FIR filter ──────────────────────────────────────────
        if eeg is not None and toggles.stage1_fir:
            eeg = self._stage1.apply(eeg)

        # ── Stage 2: bad-channel detection + spherical-spline interpolation ─
        bad_ch: list[str] = []
        if eeg is not None and toggles.stage2_bad_channels:
            bad_ch = bad_channels.detect(eeg)
            if bad_ch:
                eeg = spherical_spline.interpolate(eeg, bad_ch)

        # ── Stage 3: artifact gate ───────────────────────────────────────
        artifact_rejected = False
        artifact_reasons: list[str] = []
        artifact_annotations: list[ArtifactAnnotationPayload] = []
        artifact_correction_plan: ArtifactCorrectionPlanPayload | None = None

        if eeg is not None and toggles.stage3_artifact_gate:
            decision = self._stage3.evaluate(eeg)
            artifact_rejected = decision.reject
            artifact_reasons = list(decision.reasons)

        # ── Stage 4: ASR ────────────────────────────────────────────────
        if eeg is not None and not artifact_rejected and toggles.stage4_asr:
            eeg = asr.apply(eeg)

        # ── Stage 4b: baseline correction ───────────────────────────────
        if eeg is not None and toggles.stage4b_baseline:
            eeg = baseline.apply(eeg)

        # ── Stage 5: ocular regression ───────────────────────────────────
        if eeg is not None and not artifact_rejected and toggles.stage5_ocular:
            eeg = ocular_regression.apply(eeg)

        # ── PPG path ─────────────────────────────────────────────────────
        ppg_payload = None
        ibis: list[float] | None = None
        if sample.ppg_buffer and len(sample.ppg_buffer) >= _MIN_PPG_SAMPLES:
            from neurolink.dsp.ppg import compute_ppg
            ppg_payload = compute_ppg(sample.ppg_buffer, fs=_PPG_FS)
            if ppg_payload is not None:
                ibis = ppg_payload.ibi_ms

        # ── Stage 6: cardiac regression ──────────────────────────────────
        if eeg is not None and not artifact_rejected and toggles.stage6_cardiac:
            eeg = cardiac_regression.apply(eeg, ibis=ibis)

        # ── Stage 7: band-power ──────────────────────────────────────────
        bands_dict: dict = {}
        if eeg is not None and not artifact_rejected:
            bands_dict = bandpower.compute(eeg, fs=_EEG_FS)

        # ── Stage 8: classifiers ─────────────────────────────────────────
        focus_score: float | None = None
        fatigue_score: float | None = None
        if not artifact_rejected and bands_dict:
            scores = classifiers.run(bands_dict)
            focus_score = scores.get("focus")
            fatigue_score = scores.get("fatigue")

        # ── FAA / FMT ────────────────────────────────────────────────────
        faa: float | None = None
        fmt: float | None = None
        if bands_dict:
            alpha = bands_dict.get("alpha", {})
            if isinstance(alpha, dict):
                left = alpha.get("F3", 0.0)
                right = alpha.get("F4", 0.0)
                faa = float(np.log(left + 1e-12) - np.log(right + 1e-12))
            theta = bands_dict.get("theta", {})
            if isinstance(theta, dict):
                vals = list(theta.values())
                fmt = float(np.mean(vals)) if vals else None

        # ── fNIRS from sample.extra ────────────────────────────────────────
        extra = sample.extra if isinstance(sample.extra, dict) else {}
        fnirs_oxy: float | None = extra.get("fnirs_oxy")
        fnirs_deoxy: float | None = extra.get("fnirs_deoxy")

        # ── Breathing ────────────────────────────────────────────────────
        breathing_payload = None

        # ── IMU ──────────────────────────────────────────────────────────
        # IMUPayload fields: pitch_deg, roll_deg, motion_rms
        imu_payload = None
        if sample.accel_buffer is not None and sample.gyro_buffer is not None:
            accel = np.asarray(sample.accel_buffer, dtype=np.float32)
            if accel.ndim == 2 and accel.shape[0] >= 3 and accel.shape[1] > 0:
                ax = accel[0]
                ay = accel[1]
                az = accel[2]
                # Crude pitch/roll from mean accel vector; motion_rms from all axes.
                g = float(np.sqrt(np.mean(ax)**2 + np.mean(ay)**2 + np.mean(az)**2)) or 1.0
                pitch_deg = float(np.degrees(np.arcsin(np.clip(np.mean(ay) / g, -1.0, 1.0))))
                roll_deg = float(np.degrees(np.arctan2(np.mean(ax), np.mean(az))))
                motion_rms = float(np.sqrt(np.mean(ax**2 + ay**2 + az**2)))
                imu_payload = IMUPayload(
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    motion_rms=motion_rms,
                )

        # ── Baseline phase ───────────────────────────────────────────────
        baseline_phase: str = (
            self._baseline.phase if hasattr(self._baseline, "phase") else "idle"
        )

        # ── Hub baseline_alpha hook ──────────────────────────────────────
        if hasattr(self._hub, "baseline_alpha") and self._hub.baseline_alpha is not None:
            if hasattr(self._baseline, "set_alpha"):
                self._baseline.set_alpha(self._hub.baseline_alpha)

        # ── EEG samples: trim to window ────────────────────────────────────
        # Slice to the most recent _EEG_SAMPLES_WINDOW columns so that
        # len(ch) <= 64 regardless of input buffer width.
        eeg_samples: list[list[float]] = []
        if eeg is not None:
            windowed = eeg[:, -_EEG_SAMPLES_WINDOW:]
            eeg_samples = windowed.tolist()

        # ── Assemble bands Pydantic model ────────────────────────────────
        band_powers: BandPowers | None = None
        if bands_dict:
            try:
                band_powers = BandPowers(
                    **{
                        k: float(v) if not isinstance(v, dict) else 0.0
                        for k, v in bands_dict.items()
                        if k in ("delta", "theta", "alpha", "beta", "gamma")
                    }
                )
            except Exception:
                band_powers = None

        return IngestPayload(
            source=sample.source,
            address=sample.address,
            timestamp=sample.timestamp,
            bands=band_powers or BandPowers(),
            poor_contact=sample.poor_contact,
            faa=faa,
            fmt=fmt,
            ppg=ppg_payload,
            breathing=breathing_payload,
            imu=imu_payload,
            fnirs_oxy=fnirs_oxy,
            fnirs_deoxy=fnirs_deoxy,
            eeg_samples=eeg_samples,
            bad_channels=bad_ch,
            artifact_rejected=artifact_rejected,
            artifact_reasons=artifact_reasons,
            artifact_annotations=artifact_annotations,
            artifact_correction_plan=artifact_correction_plan,
            baseline_phase=baseline_phase,
        )
