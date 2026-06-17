"""EEGPump — background asyncio task that reads from the adapter at 4 Hz.

Builds IngestPayload from EEGSample and calls hub.update().

Stage 0 integration
-------------------
When a Stage0Guard is supplied:
  1. gate_sample() annotates motion flags on every raw EEGSample.
  2. acquisition_ready is checked before hub.update(); frames dropped
     when not ready (except mock source).
  3. When a frame is held, _stage0_settling_reason() emits a structured
     reason code ('impedance_unstable' | 'motion_settling' | 'env_not_ready'
     | 'settling') and hub.emit_settling(reason) pushes a settling SSE
     event to all connected clients so the frontend can show a contextual
     waiting indicator.

Stage 1 integration
-------------------
After eeg_arr is assembled, apply_online_filters() runs the zero-phase
FIR chain (HP 0.5 Hz + notch(s) + LP 55 Hz) on the buffer.  The
filtered array flows into all downstream DSP.

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
       a. Amplitude threshold (device-aware: dry=75µV, semi=90µV, wet=100µV)
       b. IMU motion gate (0.15 g RMS default) — from accel_buffer
       c. Kurtosis burst detection (excess kurtosis > 5.0 default)
  2. If decision.reject is True:
       - All downstream stages are skipped entirely.
       - artifact_rejected=True and artifact_reasons are set.
       - PPG, breathing, IMU orientation still computed (unaffected).
  3. decision is carried through IngestPayload → NeurolinkState →
     SSE stream so the frontend can show a per-frame quality indicator.

Stage 3b integration — ArtifactDetector (multi-type classifier)
---------------------------------------------------------------
Runs on frames that pass Stage 3 (not coarse-rejected).
  1. detector.classify(eeg_arr, accel_arr, fs) identifies up to 7
     artifact types: BLINK, HORIZONTAL_EOG, EMG, CARDIAC,
     ELECTRODE_POP, LINE_NOISE, MOTION.
  2. Returns a DetectionReport with:
       - annotations  : list of ArtifactAnnotation (type, confidence,
                        channels, feature value, threshold)
       - correction_plan : CorrectionPlan routing flags
  3. CorrectionPlan overrides downstream corrector routing:
       - hard_reject          → skip Stages 4/5 and band powers
                                (same effect as Stage 3 reject)
       - apply_asr            → Stage 4 ASR runs only when True
       - apply_ocular_regression → Stage 5 runs only when True
       - apply_notch          → notch filter re-applied after Stage 5
       - apply_cardiac_regression → Stage 6 runs only when True
  4. When Stage 3b is disabled (toggle off), Stages 4-6 run
     unconditionally on clean frames — prior behaviour preserved.
  5. Annotations and correction plan serialised as
     ArtifactAnnotationPayload / ArtifactCorrectionPlanPayload and
     carried through IngestPayload → NeurolinkState → SSE stream.

Stage 4b integration — BaselineRecorder (impedance stabilisation + ASR gate)
----------------------------------------------------------------------------
A 150-second eyes-closed resting baseline runs at session startup.

  Stage 4b now runs BEFORE Stage 4 (ASR) on every clean tick so that
  the recorder can advance its phase state machine first.  This is the
  correct ordering: BaselineRecorder is the gatekeeper that determines
  whether ASR is allowed to receive the current frame.

  1. Impedance stabilisation (dry electrodes):
     The first 30 seconds are discarded (WARMUP phase).  Dry electrodes
     require 20-40 s to form a stable sweat-film contact; data from
     this window is mechanically unreliable regardless of amplitude.

  2. ASR calibration window:
     Clean frames during the RECORDING phase (seconds 30-150) flow
     through to ASR (Stage 4) because the warmup guard is now lifted.
     ASR calibrates on genuinely rested, stabilised signal.

  When the 150-second window completes, BaselineRecorder calls
  hub.notify_baseline_complete(), which pushes a baseline_complete SSE
  event to all connected clients so the frontend can sound a bell.

  The current phase ("warmup" | "recording" | "complete") is carried
  through IngestPayload.baseline_phase → NeurolinkState.baseline_phase
  → SSE stream on every tick so the frontend can display a progress
  indicator during the baseline window.

Stage 4 integration — Artifact Subspace Reconstruction (ASR)
-------------------------------------------------------------
ASR runs on clean frames only (after Stage 3 passes the frame) AND
only once the baseline is past the WARMUP phase.
  - Guard: self._baseline.phase != "warmup" — prevents ASR from
    ingesting electrode-stabilisation data for covariance fitting.
  - When Stage 3b is active, ASR also runs only when plan.apply_asr
    is True.
  - Prioritised over pure ICA for low-channel-count wearable EEG;
    ICA source separation degrades below ~64 channels.
  - Calibration phase (first calib_sec s, default 30 s) accumulates
    data silently; the frame is returned unchanged during this period.
  - After calibration, burst samples (> burst_sd × calib_rms in the
    whitened subspace) are reconstructed from the calibration
    covariance projection.

Stage 5 integration — Gratton-Coles Ocular Regression
-------------------------------------------------------
Ocular regression runs on clean frames only (after Stage 4).
  - When Stage 3b is active, regression runs only when
    plan.apply_ocular_regression is True.
  - Gratton-Coles temporal regression is used in preference to ICA
    for the same reason as Stage 4.
  - Requires an EOG/AUX reference channel (default: channel 4, Muse
    AUX jack).  Falls back to a pass-through when no AUX is present.
  - OLS slope coefficients are refitted adaptively: EOG variance shift
    >2x or <0.5x rolling mean triggers early recalibration; the fixed
    recalib_frames interval acts as a fallback floor.

Stage 6 integration — PPG-referenced Cardiac Regression (AAS)
--------------------------------------------------------------
Cardiac regression runs on clean frames only (after Stage 5).
  - When Stage 3b is active, runs only when plan.apply_cardiac_regression
    is True AND toggles.stage6_cardiac is True.
  - Uses PPG inter-beat intervals (IBI ms list) from the PPG payload
    to build a trimmed-mean cardiac template via AAS (Adaptive Artifact
    Subtraction) and subtract it from each EEG channel.
  - Gracefully skipped when ppg_payload is None or ibi_ms is empty.

Filter toggles
--------------
Each stage can be disabled at runtime via PUT /api/v1/filters without
restarting the server.  get_toggles() returns an immutable snapshot of
the current FilterToggleConfig; the pump reads it at the top of every
_build_payload() call so changes take effect on the very next tick.
Disabled stages are logged at DEBUG level so the pipeline audit trail
remains complete even when stages are bypassed.

Module-level stubs
------------------
The module exposes a set of module-level stub objects that mirror the
interfaces used by unit tests patching 'neurolink.eeg_pump.<name>'.  Each
stub delegates to the per-pump instance method so production behaviour is
unchanged.  The stubs are:

  bad_channels       .detect(eeg) -> list[str]
  spherical_spline   .interpolate(eeg, bad) -> ndarray
  asr                .apply(eeg) -> ndarray
  ocular_regression  .apply(eeg) -> ndarray
  baseline           .apply(eeg) -> ndarray
  cardiac_regression .apply(eeg, ibis) -> ndarray
  bandpower          .compute(eeg, fs) -> dict
  classifiers        .run(bands) -> dict
  impedance          .check() -> bool
  filter_toggles     .get_toggles() -> FilterToggleConfig

Pipeline per tick
-----------------
 1.  Read EEGSample from adapter
 2.  [Stage 0] IMU motion gate          (skipped if imu_gate=False)
 3.  [Stage 0] Impedance update
 4.  [Stage 0] Acquisition readiness gate → emit_settling(reason) on hold
 5.  [Stage 1] Zero-phase FIR filter chain (HP + notch + LP 55 Hz)
                                         (bypassed if stage1_fir=False)
 6.  [Stage 2] Bad channel detection (EMA update)
 7.  [Stage 2] Spherical spline interpolation of bad channels
                                         (bypassed if stage2_bad_channels=False)
 8.  [Stage 3] Epoch-level artifact gate (amplitude / IMU / kurtosis)
                                         (bypassed if stage3_artifact_gate=False)
 8b. [Stage 3b] Multi-type artifact classifier + correction router
                                         (bypassed if stage3b_artifact_detector=False)
 9.  [Stage 4b] Baseline recording / impedance stabilisation
                                         (bypassed if stage4b_baseline=False)
                                         NOTE: runs BEFORE Stage 4 so the phase
                                         gate is advanced before ASR sees the frame.
 9b. [Stage 4] ASR burst reconstruction  (clean frames; post-warmup only;
                                          plan.apply_asr when 3b active)
                                         (bypassed if stage4_asr=False)
10.  [Stage 5] Gratton-Coles ocular regression
               (clean frames; plan.apply_ocular_regression when 3b active)
                                         (bypassed if stage5_ocular=False)
10b. [Stage 5b] Notch re-apply (plan.apply_notch when 3b active)
11.  [Stage 6] PPG cardiac regression (AAS)
               (clean frames; plan.apply_cardiac_regression + ppg IBI present)
                                         (bypassed if stage6_cardiac=False)
12.  Band powers from corrected+filtered buffer  (clean frames only)
13.  Derived EEG (FAA, FMt)                      (clean frames only)
14.  PPG HRV
15.  Breathing
16.  IMU head orientation
17.  Build IngestPayload  (includes baseline_phase, artifact_annotations,
                           artifact_correction_plan)
18.  hub.update()
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
from neurolink.dsp.filter_toggles import get_toggles
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
# Module-level stub objects
# ---------------------------------------------------------------------------
# These stubs allow test code to patch 'neurolink.eeg_pump.<name>' without
# needing to reach into EEGPump instance attributes.  In production the stubs
# are never called directly; all real work goes through EEGPump._stage* attrs.
# ---------------------------------------------------------------------------

class _BadChannelsStub:
    """Stub for neurolink.eeg_pump.bad_channels — patchable by tests."""
    def detect(self, eeg) -> list:
        return []


class _SphericalSplineStub:
    """Stub for neurolink.eeg_pump.spherical_spline — patchable by tests."""
    def interpolate(self, eeg, bad, **kw):
        return eeg


class _ASRStub:
    """Stub for neurolink.eeg_pump.asr — patchable by tests."""
    def apply(self, eeg, **kw):
        return eeg


class _OcularRegressionStub:
    """Stub for neurolink.eeg_pump.ocular_regression — patchable by tests."""
    def apply(self, eeg, **kw):
        return eeg


class _BaselineStub:
    """Stub for neurolink.eeg_pump.baseline — patchable by tests."""
    def apply(self, eeg, **kw):
        return eeg


class _CardiacRegressionStub:
    """Stub for neurolink.eeg_pump.cardiac_regression — patchable by tests."""
    def apply(self, eeg, ibis=None, **kw):
        return eeg


class _BandpowerStub:
    """Stub for neurolink.eeg_pump.bandpower — patchable by tests."""
    def compute(self, eeg, fs=256.0, **kw) -> dict:
        return {}


class _ClassifiersStub:
    """Stub for neurolink.eeg_pump.classifiers — patchable by tests."""
    def run(self, bands, **kw) -> dict:
        return {}


class _ImpedanceStub:
    """Stub for neurolink.eeg_pump.impedance — patchable by tests."""
    def check(self) -> bool:
        return True


class _FilterTogglesStub:
    """Stub for neurolink.eeg_pump.filter_toggles — patchable by tests."""
    def get_toggles(self):
        return _filter_toggles_module.get_toggles()


# Module-level singleton stubs — tests patch these names.
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


class EEGPump:
    """Background asyncio task that drives the EEG processing pipeline.

    Artifact strategy
    -----------------
    ASR (Stage 4) + temporal filtering (Stage 1) + Gratton-Coles
    regression (Stage 5) + cardiac regression (Stage 6) are prioritised
    over pure ICA.  ICA source separation degrades significantly at the
    4-channel density of Muse-class hardware; the temporal/regression
    stack achieves equivalent or better correction with lower compute
    cost and no minimum-channel requirement.

    Stage 3b (ArtifactDetector) adds multi-type classification between
    the coarse gate and the correctors, enabling precise routing:
    only the correctors relevant to the detected artifact type(s) are
    invoked, rather than running all correctors unconditionally.

    Baseline
    --------
    A BaselineRecorder is created at construction time using the same
    ASR and hub instances that the pump owns.  It runs silently inside
    _build_payload() on every clean frame; no external configuration
    is required.

    Stage ordering (Stages 4b → 4)
    --------------------------------
    BaselineRecorder (Stage 4b) runs BEFORE ASR (Stage 4) so the
    recorder advances its phase state machine first.  Stage 4 is
    additionally guarded by `self._baseline.phase != "warmup"` to
    prevent ASR from calibrating on electrode-stabilisation data.

    Filter toggles
    --------------
    get_toggles() is called once per tick at the top of _build_payload()
    to snapshot the current FilterToggleConfig.  Each stage is wrapped
    in `if toggles.<flag>:` so a PUT /api/v1/filters change takes effect
    on the very next tick with no restart.

    Pipeline per tick
    -----------------
    1.  Read EEGSample from adapter
    2.  [Stage 0] IMU motion gate
    3.  [Stage 0] Impedance update
    4.  [Stage 0] Acquisition readiness gate → emit_settling(reason) on hold
    5.  [Stage 1] Zero-phase FIR filter chain (HP + notch + LP 55 Hz)
    6.  [Stage 2] Bad channel detection (EMA update)
    7.  [Stage 2] Spherical spline interpolation of bad channels
    8.  [Stage 3] Epoch-level artifact gate (amplitude / IMU / kurtosis)
    8b. [Stage 3b] Multi-type artifact classifier + correction router
    9.  [Stage 4b] Baseline recording / impedance stabilisation  ← BEFORE Stage 4
    9b. [Stage 4]  ASR burst reconstruction (post-warmup; plan-gated when 3b active)
    10. [Stage 5] Gratton-Coles ocular regression (plan-gated when 3b active)
    10b.[Stage 5b] Notch re-apply (plan.apply_notch when 3b active)
    11. [Stage 6] PPG cardiac regression / AAS (plan-gated when 3b active)
    12. Band powers from corrected+filtered buffer  (clean frames only)
    13. Derived EEG (FAA, FMt)                      (clean frames only)
    14. PPG HRV
    15. Breathing
    16. IMU head orientation
    17. Build IngestPayload
    18. hub.update()
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
        # Stage 4b: per-session resting baseline (impedance stabilisation +
        # ASR calibration gate).  Constructed here so it shares the same
        # ASR instance as Stage 4.
        # IMPORTANT: BaselineRecorder no longer calls asr.apply() internally.
        # Instead it advances the phase state machine; Stage 4 in the pipeline
        # is guarded by `self._baseline.phase != "warmup"` and runs after 4b.
        self._baseline: BaselineRecorder = BaselineRecorder(
            asr=self._stage4,
            hub=self._hub,
        )
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

    def reset(self) -> None:
        """Reset DSP sub-components to initial state.

        Called by NeuroLinkService.disconnect() so that a subsequent
        reconnect starts a fresh 150 s baseline window.  Resets the
        BaselineRecorder first (restores WARMUP phase and restarts the
        monotonic clock), then resets Stage 6 (CardiacRegressor) so the
        cardiac template is rebuilt fresh, then delegates to hub.reset()
        so all hub-owned state is cleared in the same call.

        Safe to call while the pump loop is still running — both
        BaselineRecorder and EEGHub protect their state with locks.
        """
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
        """Return a structured reason code for the Stage 0 acquisition hold.

        Codes (first match wins):
            'impedance_unstable'  -- electrode contact not yet stable
            'motion_settling'     -- movement detected; waiting for rest
            'env_not_ready'       -- environment sensor not yet ready
            'settling'            -- generic fallback

        The reason is passed to hub.emit_settling(reason) so the frontend
        can display a contextual waiting message (e.g. "Keep still" vs
        "Adjusting headband") instead of a generic spinner.
        """
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

        # ── Stage 0 ───────────────────────────────────────────────────────────
        toggles = get_toggles()

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
            log.debug(
                "stage0_frame_held",
                impedance_ok=self._stage0.impedance.all_channels_ok,
                env_ready=self._stage0.environment.is_ready,
                motion_flagged=sample.extra.get("motion_flagged", False),
                settling_reason=reason,
            )
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

        # Snapshot toggle state once per tick — changes from PUT /api/v1/filters
        # take effect on the next tick; within this tick the view is consistent.
        toggles = get_toggles()

        # Log any disabled stages at debug level so audit trail is complete.
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

        # ── Shared accel array (built once, reused by Stage 3, 3b, IMU) ─
        accel_arr: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            try:
                accel_arr = np.array(sample.accel_buffer, dtype=np.float32)
            except Exception:
                accel_arr = None

        # ── Stage 1 — zero-phase FIR filter chain ──────────────────────
        if eeg_arr is not None and toggles.stage1_fir:
            eeg_arr = self._stage1.apply(eeg_arr)

        # ── Stage 2 — bad channel detection & interpolation ─────────────
        bad_channels_list: list[str] = []
        if eeg_arr is not None and toggles.stage2_bad_channels:
            self._stage2.update(eeg_arr)
            bad_channels_list = self._stage2.get_bad_channels()
            if bad_channels_list:
                eeg_arr = interpolate_bad_channels(eeg_arr, bad_channels_list)
                log.debug(
                    "stage2_interpolated",
                    bad=bad_channels_list,
                    n_bad=len(bad_channels_list),
                )

        # ── Stage 3 — epoch-level artifact gate ────────────────────────
        artifact_rejected: bool = False
        artifact_reasons: list[str] = []
        if eeg_arr is not None and toggles.stage3_artifact_gate:
            decision = self._stage3.evaluate(eeg_arr, accel_arr)
            if decision.reject:
                artifact_rejected = True
                artifact_reasons = decision.reasons
                log.debug(
                    "stage3_frame_rejected",
                    reasons=artifact_reasons,
                    n_bad_ch=len(bad_channels_list),
                )

        # ── Stage 3b — multi-type artifact classifier + router ──────────
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

            _plan_hard_reject = plan.hard_reject
            _plan_apply_asr = plan.apply_asr
            _plan_apply_ocular = plan.apply_ocular_regression
            _plan_apply_notch = plan.apply_notch
            _plan_apply_cardiac = plan.apply_cardiac_regression

            if _plan_hard_reject:
                log.debug(
                    "stage3b_hard_reject",
                    types=detection_report.type_names(),
                    n_annotations=len(artifact_annotations),
                )

        if _plan_hard_reject:
            artifact_rejected = True
            if not artifact_reasons:
                artifact_reasons = [
                    f"3b:{a.artifact_type}" for a in detection_report.annotations
                    if detection_report is not None
                ]

        # ── Stage 4b — Baseline recording (clean frames only) ───────────
        if eeg_arr is not None and not artifact_rejected and toggles.stage4b_baseline:
            eeg_arr = self._baseline.process(eeg_arr)

        # ── Stage 4 — ASR burst reconstruction ─────────────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage4_asr
            and _plan_apply_asr
            and self._baseline.phase != "warmup"
        ):
            eeg_arr = self._stage4.apply(eeg_arr)

        # ── Stage 5 — Gratton-Coles ocular regression ───────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage5_ocular
            and _plan_apply_ocular
        ):
            eeg_arr = self._stage5.apply(eeg_arr)

        # ── Stage 5b — Notch re-apply ───────────────────────────────────
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
            and _plan_apply_notch
            and toggles.stage1_fir
        ):
            eeg_arr = self._stage1.apply(eeg_arr)
            log.debug("stage5b_notch_reapply")

        # ── Stage 6 — PPG-referenced cardiac regression (AAS) ──────────
        ppg_payload = None
        if sample.ppg_buffer:
            ppg_arr = np.array(sample.ppg_buffer, dtype=np.float32)
            from neurolink.dsp.ppg import compute_ppg
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage6_cardiac
            and _plan_apply_cardiac
            and ppg_payload is not None
            and ppg_payload.ibi_ms
        ):
            eeg_arr = self._stage6.apply(eeg_arr, ppg_payload.ibi_ms, fs=_EEG_FS)
            log.debug(
                "stage6_cardiac_regression_applied",
                n_ibis=len(ppg_payload.ibi_ms),
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

        # ── Raw EEG sample window (filtered + corrected) ─────────────────
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
            bad_channels=bad_channels_list,
            artifact_rejected=artifact_rejected,
            artifact_reasons=artifact_reasons,
            artifact_annotations=artifact_annotations,
            artifact_correction_plan=correction_plan_payload,
            baseline_phase=self._baseline.phase,
        )
