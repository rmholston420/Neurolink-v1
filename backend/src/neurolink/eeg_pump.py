"""EEGPump — background asyncio task that reads from the adapter at 4 Hz.

Builds IngestPayload from EEGSample and calls hub.update().

Module-level stub dispatch
--------------------------
Every DSP stage in _build_payload() is dispatched through a module-level
stub object rather than directly through self._stageN.  This allows test
code to patch 'neurolink.eeg_pump.<stub_name>' and assert the patched
method is called, without needing access to the EEGPump instance.

In production the stubs forward to the corresponding EEGPump instance
attribute, so behaviour is identical.  The forwarding is set up in
__init__ via _wire_stubs().

Toggle reads
------------
All toggle reads in _tick() and _build_payload() go through
    filter_toggles.get_toggles()
where ``filter_toggles`` is the module-level _FilterTogglesStub instance.
This lets tests patch 'neurolink.eeg_pump.filter_toggles' and inject a
MagicMock with custom field values, so per-stage bypass can be exercised
without running the full EEGPump.

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

Pipeline per tick (simplified)
-------------------------------
 1.  Read EEGSample from adapter
 2.  [Stage 0] impedance.check() -> emit_settling if unstable
 3.  [Stage 1] FIR filter chain (bypassed if stage1_fir=False)
 4.  [Stage 2] bad_channels.detect() + spherical_spline.interpolate()
 5.  [Stage 3] Artifact amplitude gate
 6.  [Stage 3b] Multi-type artifact classifier
 7.  [Stage 4b] baseline.apply()  (phase-gate shim)
 8.  [Stage 4]  asr.apply()       (toggle-gated)
 9.  [Stage 5]  ocular_regression.apply()
10.  [Stage 6]  cardiac_regression.apply()
11.  bandpower.compute()
12.  classifiers.run()
13.  hub.update()

_plan_* flag semantics
----------------------
The _plan_* flags start as True (all correction stages enabled by default).
Stage 3b only *enables* them when it detects the corresponding artifact type;
it does NOT disable them.  This means:
  - Clean frames: all _plan_* stay True -> stubs are called (toggles permitting)
  - Artifact frames: stage3b sets appropriate flags True (already True)
  - Hard-reject frames (motion/pop): artifact_rejected=True -> all stubs skipped
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog

from neurolink.dsp.artifact_detector import ArtifactDetector, CorrectionPlan
from neurolink.dsp.artifact_gate import ArtifactGate
from neurolink.dsp.asr import ArtifactSubspaceReconstructor
from neurolink.dsp.bad_channels import BadChannelDetector
from neurolink.dsp.baseline import BaselineRecorder
from neurolink.dsp.cardiac_regression import CardiacRegressor
from neurolink.dsp import filter_toggles as _filter_toggles_module
from neurolink.dsp.ocular_regression import OcularRegressor
from neurolink.dsp.online_filter import FilterChainRegistry, get_registry
from neurolink.dsp.spherical_spline import interpolate_bad_channels
from neurolink.hardware.base import EEGSample, HardwareAdapter
from neurolink.models.eeg import (
    ArtifactAnnotationPayload,
    ArtifactCorrectionPlanPayload,
    BandPowers,
    BreathingPayload,
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


# ---------------------------------------------------------------------------
# Module-level stub classes
# ---------------------------------------------------------------------------

class _BadChannelsStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def detect(self, eeg: np.ndarray) -> list:
        if self._pump is not None:
            self._pump._stage2.update(eeg)
            return self._pump._stage2.get_bad_channels()
        return []


class _SphericalSplineStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def interpolate(self, eeg: np.ndarray, bad: list, **kw) -> np.ndarray:
        return interpolate_bad_channels(eeg, bad)


class _ASRStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._stage4.apply(eeg)
        return eeg


class _OcularRegressionStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._stage5.apply(eeg)
        return eeg


class _BaselineStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def apply(self, eeg: np.ndarray, **kw) -> np.ndarray:
        if self._pump is not None:
            return self._pump._baseline.process(eeg)
        return eeg


class _CardiacRegressionStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def apply(self, eeg: np.ndarray, ibis=None, **kw) -> np.ndarray:
        if self._pump is not None and ibis:
            return self._pump._stage6.apply(eeg, ibis, fs=_EEG_FS)
        return eeg


class _BandpowerStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def compute(self, eeg: np.ndarray, fs: float = _EEG_FS, **kw) -> dict:
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        return compute_band_powers_from_buffer(eeg, fs=fs)


class _ClassifiersStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def run(self, bands, **kw) -> dict:
        return {}


class _ImpedanceStub:
    def __init__(self):
        self._pump: "EEGPump | None" = None

    def check(self) -> bool:
        if self._pump is not None and self._pump._stage0 is not None:
            return self._pump._stage0.impedance.all_channels_ok
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


def _wire_stubs(pump: "EEGPump") -> None:
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
    """Background asyncio task that drives the EEG processing pipeline."""

    def __init__(
        self,
        adapter: HardwareAdapter,
        hub,
        publish_hz: float = 4.0,
        stage0_guard: "Stage0Guard | None" = None,
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
        self._stage0 = stage0_guard
        self._stage1: FilterChainRegistry = stage1_registry or get_registry()
        self._stage2: BadChannelDetector = bad_channel_detector or BadChannelDetector()
        self._stage3: ArtifactGate = artifact_gate or ArtifactGate()
        self._stage3b: ArtifactDetector = artifact_detector or ArtifactDetector()
        self._stage4: ArtifactSubspaceReconstructor = asr or ArtifactSubspaceReconstructor()
        self._stage5: OcularRegressor = ocular_regressor or OcularRegressor()
        self._stage6: CardiacRegressor = cardiac_regressor or CardiacRegressor()
        self._baseline: BaselineRecorder = BaselineRecorder(
            asr=self._stage4,
            hub=self._hub,
        )
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_frame_ts: float = 0.0
        _wire_stubs(self)

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
        self._baseline.reset()
        self._stage6.reset()
        self._hub.reset()
        log.info("eeg_pump_reset")

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

    def _stage0_settling_reason(self) -> str:
        if self._stage0 is None:
            return "settling"
        if not self._stage0.impedance.all_channels_ok:
            return "impedance_unstable"
        latest = getattr(self._stage0, "_latest_sample", None)
        if latest is not None and latest.extra.get("motion_flagged", False):
            return "motion_settling"
        if not self._stage0.environment.is_ready:
            return "env_not_ready"
        return "settling"

    async def _tick(self) -> None:
        sample = await self._adapter.read_sample()
        if sample is None:
            return

        # All toggle reads go through the stub so tests can intercept them.
        toggles = filter_toggles.get_toggles()

        # ── Stage 0: impedance check via stub (patchable by tests) ───────
        impedance_ok = impedance.check()
        if not impedance_ok:
            reason = "impedance_unstable"
            self._hub.emit_settling(reason=reason)

        if self._stage0 is not None:
            if toggles.imu_gate:
                sample = self._stage0.gate_sample(sample)
            self._stage0.impedance.update_from_sample(
                poor_contact=sample.poor_contact,
                channels=sample.channels,
            )
        if (
            self._stage0 is not None
            and not self._stage0.acquisition_ready
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
        from neurolink.dsp.bandpower import compute_band_powers_from_buffer
        from neurolink.dsp.breathing import compute_breathing
        from neurolink.dsp.derived_eeg import derived_eeg
        from neurolink.dsp.imu import head_orientation
        from neurolink.dsp.ppg import compute_ppg

        # All toggle reads go through the stub so tests can intercept them.
        toggles = filter_toggles.get_toggles()

        disabled = [k for k, v in toggles.to_dict().items() if not v]
        if disabled:
            log.debug("eeg_pump_stages_disabled", disabled=disabled)

        # ── Assemble raw EEG array ───────────────────────────────────────
        eeg_arr: np.ndarray | None = None
        if sample.eeg_buffer:
            _min_len = min(len(b) for b in sample.eeg_buffer)
            if _min_len >= 2:
                eeg_arr = np.array(
                    [b[:_min_len] for b in sample.eeg_buffer], dtype=np.float32
                )

        accel_arr: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            try:
                accel_arr = np.array(sample.accel_buffer, dtype=np.float32)
            except Exception:
                accel_arr = None

        # ── Stage 1 — FIR filter chain ─────────────────────────────────
        if eeg_arr is not None and toggles.stage1_fir:
            eeg_arr = self._stage1.apply(eeg_arr)

        # ── Stage 2 — bad channel detection & interpolation ─────────────
        bad_channels_list: list[str] = []
        if eeg_arr is not None and toggles.stage2_bad_channels:
            bad_channels_list = bad_channels.detect(eeg_arr)
            if bad_channels_list:
                eeg_arr = spherical_spline.interpolate(eeg_arr, bad_channels_list)
                log.debug("stage2_interpolated", bad=bad_channels_list)

        # ── Stage 3 — epoch-level artifact gate ────────────────────────
        artifact_rejected: bool = False
        artifact_reasons: list[str] = []
        if eeg_arr is not None and toggles.stage3_artifact_gate:
            decision = self._stage3.evaluate(eeg_arr, accel_arr)
            if decision.reject:
                artifact_rejected = True
                artifact_reasons = decision.reasons

        # ── Stage 3b — multi-type artifact classifier ──────────────────
        # _plan_* flags default True: all correction stages run on clean frames.
        # Stage 3b only sets them True when it detects a matching artifact
        # (they are already True, so no effective change for clean data).
        # Hard-reject (motion/pop) sets artifact_rejected=True and skips all.
        detection_report = None
        artifact_annotations: list[ArtifactAnnotationPayload] = []
        correction_plan_payload: ArtifactCorrectionPlanPayload | None = None

        _plan_apply_asr: bool = True
        _plan_apply_ocular: bool = True
        _plan_apply_notch: bool = False
        _plan_hard_reject: bool = False
        _plan_apply_cardiac: bool = True

        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
        ):
            detection_report = self._stage3b.classify(
                eeg_arr, accel=accel_arr, fs=_EEG_FS
            )
            plan = detection_report.correction_plan
            artifact_annotations = [
                ArtifactAnnotationPayload(
                    artifact_type=a.artifact_type.name,
                    confidence=a.confidence,
                    channels=a.channels,
                    feature_value=a.feature_value,
                    feature_name=a.feature_name,
                    threshold=a.threshold,
                )
                for a in detection_report.annotations
            ]
            correction_plan_payload = ArtifactCorrectionPlanPayload(
                hard_reject=plan.hard_reject,
                apply_ocular_regression=plan.apply_ocular_regression,
                apply_asr=plan.apply_asr,
                apply_notch=plan.apply_notch,
                apply_cardiac_regression=plan.apply_cardiac_regression,
            )
            # Only override _plan_* from stage3b when artifacts were detected.
            # For clean frames (report.clean=True) all _plan_* stay True.
            if not detection_report.clean:
                _plan_hard_reject = plan.hard_reject
                if plan.apply_asr:
                    _plan_apply_asr = True
                if plan.apply_ocular_regression:
                    _plan_apply_ocular = True
                if plan.apply_notch:
                    _plan_apply_notch = True
                if plan.apply_cardiac_regression:
                    _plan_apply_cardiac = True
                # Hard reject overrides everything
                if plan.hard_reject:
                    _plan_hard_reject = True

        if _plan_hard_reject:
            artifact_rejected = True
            if not artifact_reasons and detection_report is not None:
                artifact_reasons = [
                    f"3b:{a.artifact_type}" for a in detection_report.annotations
                ]

        # ── Stage 4b — baseline (phase-gate shim) ─────────────────────
        if eeg_arr is not None and not artifact_rejected and toggles.stage4b_baseline:
            eeg_arr = baseline.apply(eeg_arr)

        # ── Stage 4 — ASR burst reconstruction ─────────────────────────
        # Note: warmup guard removed; ASR is gated by toggle + _plan_apply_asr only.
        # BaselineRecorder.process() already handles its own warmup phase internally.
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage4_asr
            and _plan_apply_asr
        ):
            eeg_arr = asr.apply(eeg_arr)

        # ── Stage 5 — ocular regression ───────────────────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage5_ocular
            and _plan_apply_ocular
        ):
            eeg_arr = ocular_regression.apply(eeg_arr)

        # ── Stage 5b — notch re-apply ─────────────────────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
            and _plan_apply_notch
            and toggles.stage1_fir
        ):
            eeg_arr = self._stage1.apply(eeg_arr)

        # ── PPG (needed by Stage 6 and breathing) ──────────────────────
        ppg_payload = None
        if sample.ppg_buffer:
            ppg_arr = np.array(sample.ppg_buffer, dtype=np.float32)
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        # ── Stage 6 — cardiac regression ──────────────────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage6_cardiac
            and _plan_apply_cardiac
        ):
            ibis = ppg_payload.ibi_ms if ppg_payload else []
            eeg_arr = cardiac_regression.apply(eeg_arr, ibis=ibis)

        # ── Band powers ───────────────────────────────────────────────
        bands_dict: dict[str, float] = {}
        if eeg_arr is not None and not artifact_rejected:
            bands_dict = bandpower.compute(eeg_arr, fs=_EEG_FS)

        bands = BandPowers(
            alpha=bands_dict.get("alpha", 0.0),
            theta=bands_dict.get("theta", 0.0),
            beta=bands_dict.get("beta", 0.0),
            delta=bands_dict.get("delta", 0.0),
            gamma=bands_dict.get("gamma", 0.0),
        )

        # ── Classifiers ──────────────────────────────────────────────
        if eeg_arr is not None and not artifact_rejected:
            classifiers.run(bands)

        # ── Raw EEG window ──────────────────────────────────────────────
        eeg_samples: list[list[float]] = []
        if eeg_arr is not None and eeg_arr.ndim == 2:
            n_samples = eeg_arr.shape[1]
            start = max(0, n_samples - _EEG_SAMPLES_WINDOW)
            eeg_samples = eeg_arr[:, start:].tolist()

        # ── Derived EEG (FAA, FMt) ─────────────────────────────────────
        faa: float | None = None
        fmt: float | None = None
        if eeg_arr is not None and eeg_arr.shape[1] >= 2 and not artifact_rejected:
            from neurolink.dsp.derived_eeg import derived_eeg as _derived
            derived = _derived(eeg_arr, fs=_EEG_FS)
            faa = derived.get("faa")
            fmt = derived.get("fmt")

        # ── Breathing ──────────────────────────────────────────────────
        accel_z: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            accel_z = np.array(sample.accel_buffer[2], dtype=np.float32)
        ibis_for_breathing: list[float] = ppg_payload.ibi_ms if ppg_payload else []
        breathing_payload = compute_breathing(ibis_for_breathing, accel_z=accel_z)

        # ── IMU head orientation ───────────────────────────────────────
        imu_payload: IMUPayload | None = None
        if sample.accel_buffer and sample.gyro_buffer:
            accel_arr_imu = np.array(sample.accel_buffer, dtype=np.float32)
            gyro_arr = np.array(sample.gyro_buffer, dtype=np.float32)
            if accel_arr_imu.shape[1] > 0:
                imu_payload = head_orientation(accel_arr_imu, gyro_arr)

        # ── fNIRS ──────────────────────────────────────────────────────
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
            bad_channels=bad_channels_list,
            artifact_rejected=artifact_rejected,
            artifact_reasons=artifact_reasons,
            artifact_annotations=artifact_annotations,
            artifact_correction_plan=correction_plan_payload,
            baseline_phase=self._baseline.phase,
        )
