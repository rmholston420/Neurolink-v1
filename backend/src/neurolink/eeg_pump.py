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
FIR chain (HP 0.5 Hz + notch(s) + LP 45 Hz) on the buffer.  The
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
       a. Amplitude threshold (±100 µV default) — EEG channels only
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
  4. When Stage 3b is disabled (toggle off), Stages 4-5 run
     unconditionally on clean frames — prior behaviour preserved.
  5. Annotations and correction plan serialised as
     ArtifactAnnotationPayload / ArtifactCorrectionPlanPayload and
     carried through IngestPayload → NeurolinkState → SSE stream.

Stage 4 integration — Artifact Subspace Reconstruction (ASR)
-------------------------------------------------------------
ASR runs on clean frames only (after Stage 3 passes the frame).
  - When Stage 3b is active, ASR runs only when plan.apply_asr is True.
  - Prioritised over pure ICA for low-channel-count wearable EEG;
    ICA source separation degrades below ~64 channels.
  - Calibration phase (first calib_sec s, default 30 s) accumulates
    data silently; the frame is returned unchanged during this period.
  - After calibration, burst samples (> burst_sd × calib_rms in the
    whitened subspace) are reconstructed from the calibration
    covariance projection.

Baseline integration (Stage 4b)
--------------------------------
A 150-second eyes-closed resting baseline runs at session startup,
serving two purposes:

  1. Impedance stabilisation (dry electrodes):
     The first 30 seconds are discarded (WARMUP phase).  Dry electrodes
     require 20-40 s to form a stable sweat-film contact; data from
     this window is mechanically unreliable regardless of amplitude.

  2. ASR calibration window:
     Clean frames during the RECORDING phase (seconds 30-150) are
     forwarded to the ASR instance so that ASR calibrates on genuinely
     rested, stabilised signal rather than the noisier early-session
     data it would otherwise receive.

  When the 150-second window completes, BaselineRecorder calls
  hub.notify_baseline_complete(), which pushes a baseline_complete SSE
  event to all connected clients so the frontend can sound a bell.

  The current phase ("warmup" | "recording" | "complete") is carried
  through IngestPayload.baseline_phase → NeurolinkState.baseline_phase
  → SSE stream on every tick so the frontend can display a progress
  indicator during the baseline window.

Stage 5 integration — Gratton-Coles Ocular Regression
-------------------------------------------------------
Ocular regression runs on clean frames only (after Stage 4).
  - When Stage 3b is active, regression runs only when
    plan.apply_ocular_regression is True.
  - Gratton-Coles temporal regression is used in preference to ICA
    for the same reason as Stage 4.
  - Requires an EOG/AUX reference channel (default: channel 4, Muse
    AUX jack).  Falls back to a pass-through when no AUX is present.
  - OLS slope coefficients are refitted every recalib_frames ticks
    (default 512 ≈ 2 min at 4 Hz) to track slow skin-potential drift.

Filter toggles
--------------
Each stage can be disabled at runtime via PUT /api/v1/filters without
restarting the server.  get_toggles() returns an immutable snapshot of
the current FilterToggleConfig; the pump reads it at the top of every
_build_payload() call so changes take effect on the very next tick.
Disabled stages are logged at DEBUG level so the pipeline audit trail
remains complete even when stages are bypassed.

Pipeline per tick
-----------------
 1.  Read EEGSample from adapter
 2.  [Stage 0] IMU motion gate          (skipped if imu_gate=False)
 3.  [Stage 0] Impedance update
 4.  [Stage 0] Acquisition readiness gate
 5.  [Stage 1] Zero-phase FIR filter chain (HP + notch + LP)
                                         (bypassed if stage1_fir=False)
 6.  [Stage 2] Bad channel detection (EMA update)
 7.  [Stage 2] Spherical spline interpolation of bad channels
                                         (bypassed if stage2_bad_channels=False)
 8.  [Stage 3] Epoch-level artifact gate (amplitude / IMU / kurtosis)
                                         (bypassed if stage3_artifact_gate=False)
 8b. [Stage 3b] Multi-type artifact classifier + correction router
                                         (bypassed if stage3b_artifact_detector=False)
 9.  [Stage 4] ASR burst reconstruction  (clean frames; plan.apply_asr when 3b active)
                                         (bypassed if stage4_asr=False)
 9b. [Stage 4b] Baseline recording / impedance stabilisation
                                         (bypassed if stage4b_baseline=False)
10.  [Stage 5] Gratton-Coles ocular regression
               (clean frames; plan.apply_ocular_regression when 3b active)
                                         (bypassed if stage5_ocular=False)
10b. [Stage 5b] Notch re-apply (plan.apply_notch when 3b active)
11.  Band powers from corrected+filtered buffer  (clean frames only)
12.  Derived EEG (FAA, FMt)                      (clean frames only)
13.  PPG HRV
14.  Breathing
15.  IMU head orientation
16.  Build IngestPayload  (includes baseline_phase, artifact_annotations,
                           artifact_correction_plan)
17.  hub.update()
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


class EEGPump:
    """Background asyncio task that drives the EEG processing pipeline.

    Artifact strategy
    -----------------
    ASR (Stage 4) + temporal filtering (Stage 1) + Gratton-Coles
    regression (Stage 5) are prioritised over pure ICA.  ICA source
    separation degrades significantly at the 4-channel density of
    Muse-class hardware; the temporal/regression stack achieves
    equivalent or better correction with lower compute cost and no
    minimum-channel requirement.

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
    4.  [Stage 0] Acquisition readiness gate
    5.  [Stage 1] Zero-phase FIR filter chain (HP + notch + LP)
    6.  [Stage 2] Bad channel detection (EMA update)
    7.  [Stage 2] Spherical spline interpolation of bad channels
    8.  [Stage 3] Epoch-level artifact gate (amplitude / IMU / kurtosis)
    8b. [Stage 3b] Multi-type artifact classifier + correction router
    9.  [Stage 4] ASR burst reconstruction  (plan-gated when 3b active)
    9b. [Stage 4b] Baseline recording / impedance stabilisation
    10. [Stage 5] Gratton-Coles ocular regression (plan-gated when 3b active)
    10b.[Stage 5b] Notch re-apply (plan.apply_notch when 3b active)
    11. Band powers from corrected+filtered buffer  (clean frames only)
    12. Derived EEG (FAA, FMt)                      (clean frames only)
    13. PPG HRV
    14. Breathing
    15. IMU head orientation
    16. Build IngestPayload
    17. hub.update()
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
        # Stage 4b: per-session resting baseline (impedance stabilisation +
        # ASR calibration window).  Constructed here so it shares the same
        # ASR instance as Stage 4 — BaselineRecorder.process() feeds frames
        # directly to self._stage4.apply() during the RECORDING phase.
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
        bad_channels: list[str] = []
        if eeg_arr is not None and toggles.stage2_bad_channels:
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
        if eeg_arr is not None and toggles.stage3_artifact_gate:
            decision = self._stage3.evaluate(eeg_arr, accel_arr)
            if decision.reject:
                artifact_rejected = True
                artifact_reasons = decision.reasons
                log.debug(
                    "stage3_frame_rejected",
                    reasons=artifact_reasons,
                    n_bad_ch=len(bad_channels),
                )

        # ── Stage 3b — multi-type artifact classifier + router ──────────
        # Only runs on frames that passed the coarse Stage 3 gate.
        # The CorrectionPlan produced here overrides whether Stages 4/5
        # are invoked.  When Stage 3b is disabled, plan defaults to
        # "run all correctors" (apply_asr=True, apply_ocular_regression=True)
        # preserving prior behaviour exactly.
        detection_report = None
        artifact_annotations: list[ArtifactAnnotationPayload] = []
        correction_plan_payload: ArtifactCorrectionPlanPayload | None = None

        # Plan controls downstream corrector routing.
        # Default: run all correctors unconditionally (prior behaviour).
        _plan_apply_asr: bool = True
        _plan_apply_ocular: bool = True
        _plan_apply_notch: bool = False
        _plan_hard_reject: bool = False

        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
        ):
            detection_report = self._stage3b.classify(
                eeg_arr, accel=accel_arr, fs=_EEG_FS
            )
            plan = detection_report.correction_plan

            # Serialise annotations for IngestPayload
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

            # Extract routing flags from plan
            _plan_hard_reject = plan.hard_reject
            _plan_apply_asr = plan.apply_asr
            _plan_apply_ocular = plan.apply_ocular_regression
            _plan_apply_notch = plan.apply_notch

            if _plan_hard_reject:
                log.debug(
                    "stage3b_hard_reject",
                    types=detection_report.type_names(),
                    n_annotations=len(artifact_annotations),
                )

        # Treat Stage 3b hard_reject identically to Stage 3 gate reject
        if _plan_hard_reject:
            artifact_rejected = True
            if not artifact_reasons:
                artifact_reasons = [
                    f"3b:{a.artifact_type}" for a in detection_report.annotations
                    if detection_report is not None
                ]

        # ── Stage 4 — ASR burst reconstruction (clean frames only) ─────
        # When Stage 3b active: only runs if plan.apply_asr is True.
        # When Stage 3b disabled: runs unconditionally on clean frames.
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage4_asr
            and _plan_apply_asr
        ):
            eeg_arr = self._stage4.apply(eeg_arr)

        # ── Stage 4b — Baseline recording (clean frames only) ───────────
        if eeg_arr is not None and not artifact_rejected and toggles.stage4b_baseline:
            eeg_arr = self._baseline.process(eeg_arr)

        # ── Stage 5 — Gratton-Coles ocular regression (clean frames) ───
        # When Stage 3b active: only runs if plan.apply_ocular_regression.
        # When Stage 3b disabled: runs unconditionally on clean frames.
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage5_ocular
            and _plan_apply_ocular
        ):
            eeg_arr = self._stage5.apply(eeg_arr)

        # ── Stage 5b — Notch re-apply (only when Stage 3b requests it) ─
        # Line-noise artifacts that slip through Stage 1 are re-notched here.
        # Only fires when Stage 3b is active AND plan.apply_notch is True.
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
            and _plan_apply_notch
            and toggles.stage1_fir
        ):
            eeg_arr = self._stage1.apply(eeg_arr)
            log.debug("stage5b_notch_reapply")

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
            artifact_annotations=artifact_annotations,
            artifact_correction_plan=correction_plan_payload,
            baseline_phase=self._baseline.phase,
        )
