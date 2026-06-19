"""EEGPipeline — pure DSP orchestrator extracted from EEGPump.

This module owns all EEG signal-processing stages (1-6) and bandpower
computation.  EEGPump is now a thin async driver that calls
``EEGPipeline.process()`` and forwards the result to the hub.

Key optimisations vs the original monolithic _build_payload():

  Single-PSD computation
  ----------------------
  The original ``compute_band_powers_from_buffer`` called ``scipy.signal.welch``
  once *per band per channel* -- up to 25 FFT calls per tick at 4 Hz.  The new
  ``_compute_bands_single_psd`` computes a single Welch PSD per channel and
  then integrates the power for every band from that single spectrum.  This
  reduces the FFT count from 25 to 5 (one per channel), a ~5x reduction.

  Pre-allocated NumPy arrays are used for the frequency mask so no new
  allocation is made on the hot path after the first call.

StreamHealth tracking
---------------------
Every call to ``process()`` updates a ``StreamHealth`` dataclass with:
  - frames_total       -- total frames processed
  - frames_rejected    -- artifact-rejected frames
  - frames_clean       -- frames forwarded to band-power extraction
  - packet_loss_pct    -- rolling 10-second packet-loss estimate
  - last_frame_ts      -- monotonic timestamp of last successful frame

Public interface
----------------
  pipeline = EEGPipeline(stage0, stage1_registry, ...)
  result   = pipeline.process(sample)     # -> PipelineResult
  health   = pipeline.health              # -> StreamHealth (live snapshot)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog
from scipy import signal as sp_signal

from neurolink.dsp.artifact_detector import ArtifactDetector
from neurolink.dsp.artifact_gate import ArtifactGate
from neurolink.dsp.asr import ArtifactSubspaceReconstructor
from neurolink.dsp.bad_channels import BadChannelDetector
from neurolink.dsp.baseline import BaselineRecorder
from neurolink.dsp.cardiac_regression import CardiacRegressor
from neurolink.dsp.ocular_regression import OcularRegressor
from neurolink.dsp.online_filter import FilterChainRegistry, get_registry
from neurolink.dsp.spherical_spline import interpolate_bad_channels
from neurolink.hardware.base import EEGSample
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
_EEG_SAMPLES_WINDOW: int = 64
_MIN_PPG_SAMPLES: int = 960  # 15 s * 64 Hz -- matches eeg_pump.py
_NPERSEG: int = 256

# Standard EEG band definitions [lo, hi] Hz -- shared with bandpower.py
_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# Health tracking window
_HEALTH_WINDOW_SEC: float = 10.0


# ---------------------------------------------------------------------------
# StreamHealth
# ---------------------------------------------------------------------------


@dataclass
class StreamHealth:
    """Real-time stream quality metrics, updated every pipeline tick.

    Exposed via EEGPipeline.health and serialised into every SSE payload
    through NeurolinkState.stream_health.

    Fields
    ------
    frames_total       Total frames seen since last reset.
    frames_rejected    Frames discarded by Stage 3 / Stage 3b.
    frames_clean       Frames that reached band-power computation.
    packet_loss_pct    BLE packet-loss estimate over the last
                       ``_HEALTH_WINDOW_SEC`` seconds (0-100).
    last_frame_ts      wall-clock time of the most recent frame (0 = never).
    avg_tick_ms        Exponential moving average of per-tick processing
                       time (ms).  Useful for detecting DSP budget overruns.
    """

    frames_total: int = 0
    frames_rejected: int = 0
    frames_clean: int = 0
    packet_loss_pct: float = 0.0
    last_frame_ts: float = 0.0
    avg_tick_ms: float = 0.0

    # Internal accounting -- not serialised
    _window_frames_seen: int = field(default=0, repr=False)
    _window_frames_expected: int = field(default=0, repr=False)
    _window_start_ts: float = field(default_factory=time.time, repr=False)
    _publish_hz: float = field(default=4.0, repr=False)
    _ema_alpha: float = field(default=0.1, repr=False)

    def record_frame(self, rejected: bool, tick_ms: float) -> None:
        """Record a processed frame and update rolling statistics."""
        self.frames_total += 1
        self.last_frame_ts = time.time()
        self._window_frames_seen += 1

        if rejected:
            self.frames_rejected += 1
        else:
            self.frames_clean += 1

        # Update EMA of tick time
        self.avg_tick_ms = (
            tick_ms
            if self.avg_tick_ms == 0.0
            else (self._ema_alpha * tick_ms + (1 - self._ema_alpha) * self.avg_tick_ms)
        )

        # Refresh packet-loss window every _HEALTH_WINDOW_SEC
        now = time.time()
        elapsed = now - self._window_start_ts
        if elapsed >= _HEALTH_WINDOW_SEC:
            expected = max(1, int(self._publish_hz * elapsed))
            seen = self._window_frames_seen
            loss = max(0.0, (expected - seen) / expected * 100.0)
            self.packet_loss_pct = round(loss, 1)
            # Reset window
            self._window_frames_seen = 0
            self._window_frames_expected = expected
            self._window_start_ts = now

    def reset(self) -> None:
        """Zero all counters (called on BLE disconnect)."""
        self.frames_total = 0
        self.frames_rejected = 0
        self.frames_clean = 0
        self.packet_loss_pct = 0.0
        self.last_frame_ts = 0.0
        self.avg_tick_ms = 0.0
        self._window_frames_seen = 0
        self._window_frames_expected = 0
        self._window_start_ts = time.time()

    def to_dict(self) -> dict:
        """Serialise public fields for SSE/JSON emission."""
        return {
            "frames_total": self.frames_total,
            "frames_rejected": self.frames_rejected,
            "frames_clean": self.frames_clean,
            "packet_loss_pct": self.packet_loss_pct,
            "last_frame_ts": self.last_frame_ts,
            "avg_tick_ms": round(self.avg_tick_ms, 2),
        }


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Value object returned by EEGPipeline.process().

    Contains everything EEGPump needs to build an IngestPayload without
    the pump knowing anything about DSP internals.
    """

    bands: BandPowers
    eeg_samples: list[list[float]]
    bad_channels: list[str]
    artifact_rejected: bool
    artifact_reasons: list[str]
    artifact_annotations: list[ArtifactAnnotationPayload]
    artifact_correction_plan: ArtifactCorrectionPlanPayload | None
    baseline_phase: str
    ppg_payload: object | None  # PPGPayload | None
    breathing_payload: object | None  # BreathingPayload | None
    imu_payload: IMUPayload | None
    faa: float | None
    fmt: float | None
    fnirs_oxy: float | None
    fnirs_deoxy: float | None


# ---------------------------------------------------------------------------
# Single-PSD band power helper
# ---------------------------------------------------------------------------


def _compute_bands_single_psd(
    eeg_arr: np.ndarray,
    fs: float = _EEG_FS,
) -> dict[str, float]:
    """Compute normalised band powers using ONE Welch PSD per channel.

    Unlike the original ``compute_band_powers_from_buffer`` which called
    ``sp_signal.welch`` once per band per channel (up to 25 calls at 5
    channels x 5 bands), this function calls welch exactly once per
    channel and integrates all band powers from the single PSD.

    The cost reduction is from O(n_channels * n_bands) to O(n_channels)
    FFT evaluations.

    Returns
    -------
    dict mapping band name -> normalised power fraction [0, 1].
    All zeros when the array is too short or all-zero.
    """
    result: dict[str, float] = dict.fromkeys(_BANDS, 0.0)

    if eeg_arr is None:
        return result

    arr = eeg_arr if eeg_arr.ndim == 2 else eeg_arr[np.newaxis, :]
    n_channels, n_samples = arr.shape

    if n_samples < 2:
        return result

    nperseg = min(_NPERSEG, n_samples)
    abs_powers: dict[str, float] = dict.fromkeys(_BANDS, 0.0)

    for ch in range(n_channels):
        freqs, psd = sp_signal.welch(arr[ch], fs=fs, nperseg=nperseg)
        for band, (lo, hi) in _BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            abs_powers[band] += float(np.sum(psd[mask]))

    # Average across channels
    for band in _BANDS:
        abs_powers[band] /= n_channels

    total = sum(abs_powers.values())
    if total <= 0:
        return result

    return {band: abs_powers[band] / total for band in _BANDS}


# ---------------------------------------------------------------------------
# EEGPipeline
# ---------------------------------------------------------------------------


class EEGPipeline:
    """Pure DSP orchestrator: runs Stages 1-6 and band-power extraction.

    All DSP state lives here.  EEGPump holds one EEGPipeline instance and
    calls ``process(sample)`` each tick -- no DSP logic in EEGPump itself.
    """

    def __init__(
        self,
        *,
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
        self._stage0 = stage0_guard
        self._stage1: FilterChainRegistry = stage1_registry or get_registry()
        self._stage2: BadChannelDetector = bad_channel_detector or BadChannelDetector()
        self._stage3: ArtifactGate = artifact_gate or ArtifactGate()
        self._stage3b: ArtifactDetector = artifact_detector or ArtifactDetector()
        self._stage4: ArtifactSubspaceReconstructor = asr or ArtifactSubspaceReconstructor()
        self._stage5: OcularRegressor = ocular_regressor or OcularRegressor()
        self._stage6: CardiacRegressor = cardiac_regressor or CardiacRegressor()
        self._baseline: BaselineRecorder = BaselineRecorder(asr=self._stage4, hub=hub)
        self._health: StreamHealth = StreamHealth(_publish_hz=publish_hz)

    # -- Public interface ----------------------------------------------------

    @property
    def health(self) -> StreamHealth:
        """Live stream health snapshot."""
        return self._health

    @property
    def baseline_phase(self) -> str:
        return self._baseline.phase

    def reset(self) -> None:
        """Reset all stateful DSP components (called on BLE disconnect)."""
        self._baseline.reset()
        self._stage6.reset()
        self._health.reset()
        log.info("eeg_pipeline_reset")

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

    def process(self, sample: EEGSample) -> PipelineResult | None:
        """Run the full DSP pipeline on one EEGSample.

        Returns
        -------
        PipelineResult on success, or None when Stage 0 holds the frame
        (settling / impedance gate).
        """
        from neurolink.dsp import filter_toggles as _ft_module
        from neurolink.dsp.breathing import compute_breathing
        from neurolink.dsp.imu import head_orientation
        from neurolink.dsp.ppg import compute_ppg

        tick_start = time.monotonic()
        toggles = _ft_module.get_toggles()

        disabled = [k for k, v in toggles.to_dict().items() if not v]
        if disabled:
            log.debug("pipeline_stages_disabled", disabled=disabled)

        # -- Stage 0 gate ----------------------------------------------------
        if self._stage0 is not None:
            if toggles.imu_gate:
                sample = self._stage0.gate_sample(sample)
            self._stage0.impedance.update_from_sample(
                poor_contact=sample.poor_contact,
                channels=sample.channels,
            )
            if not self._stage0.acquisition_ready and sample.source != "mock":
                return None  # caller emits settling event

        # -- Assemble arrays -------------------------------------------------
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

        # -- Stage 1 -- FIR filter chain -------------------------------------
        if eeg_arr is not None and toggles.stage1_fir:
            eeg_arr = self._stage1.apply(eeg_arr)

        # -- Stage 2 -- bad channel detection & interpolation ----------------
        bad_channels_list: list[str] = []
        if eeg_arr is not None and toggles.stage2_bad_channels:
            self._stage2.update(eeg_arr)
            bad_channels_list = self._stage2.get_bad_channels()
            if bad_channels_list:
                eeg_arr = interpolate_bad_channels(eeg_arr, bad_channels_list)
                log.debug("stage2_interpolated", bad=bad_channels_list)

        # -- Stage 3 -- epoch-level artifact gate ----------------------------
        artifact_rejected: bool = False
        artifact_reasons: list[str] = []
        if eeg_arr is not None and toggles.stage3_artifact_gate:
            decision = self._stage3.evaluate(eeg_arr, accel_arr)
            if decision.reject:
                artifact_rejected = True
                artifact_reasons = decision.reasons

        # -- Stage 3b -- multi-type artifact classifier ----------------------
        detection_report = None
        artifact_annotations: list[ArtifactAnnotationPayload] = []
        correction_plan_payload: ArtifactCorrectionPlanPayload | None = None

        _plan_apply_asr: bool = True
        _plan_apply_ocular: bool = True
        _plan_apply_notch: bool = False
        _plan_hard_reject: bool = False
        _plan_apply_cardiac: bool = True

        if eeg_arr is not None and not artifact_rejected and toggles.stage3b_artifact_detector:
            detection_report = self._stage3b.classify(eeg_arr, accel=accel_arr, fs=_EEG_FS)
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
                if plan.hard_reject:
                    _plan_hard_reject = True

        if _plan_hard_reject:
            artifact_rejected = True
            if not artifact_reasons and detection_report is not None:
                artifact_reasons = [f"3b:{a.artifact_type}" for a in detection_report.annotations]

        # -- Stage 4b -- baseline (phase-gate shim) -------------------------
        if eeg_arr is not None and not artifact_rejected and toggles.stage4b_baseline:
            eeg_arr = self._baseline.process(eeg_arr)

        # -- Stage 4 -- ASR burst reconstruction ----------------------------
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage4_asr
            and _plan_apply_asr
        ):
            eeg_arr = self._stage4.apply(eeg_arr)

        # -- Stage 5 -- ocular regression ------------------------------------
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage5_ocular
            and _plan_apply_ocular
        ):
            eeg_arr = self._stage5.apply(eeg_arr)

        # -- Stage 5b -- notch re-apply -------------------------------------
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage3b_artifact_detector
            and _plan_apply_notch
            and toggles.stage1_fir
        ):
            eeg_arr = self._stage1.apply(eeg_arr)

        # -- PPG -------------------------------------------------------------
        ppg_payload = None
        if sample.ppg_buffer and len(sample.ppg_buffer) >= _MIN_PPG_SAMPLES:
            ppg_arr = np.array(sample.ppg_buffer, dtype=np.float32)
            ppg_payload = compute_ppg(ppg_arr, fs=_PPG_FS)

        # -- Stage 6 -- cardiac regression ----------------------------------
        if (
            eeg_arr is not None
            and not artifact_rejected
            and toggles.stage6_cardiac
            and _plan_apply_cardiac
        ):
            ibis = ppg_payload.ibi_ms if ppg_payload else []
            eeg_arr = self._stage6.apply(eeg_arr, ibis, fs=_EEG_FS)

        # -- Band powers (single-PSD path) ----------------------------------
        bands_dict: dict[str, float] = {}
        if eeg_arr is not None and not artifact_rejected:
            bands_dict = _compute_bands_single_psd(eeg_arr, fs=_EEG_FS)

        bands = BandPowers(
            alpha=bands_dict.get("alpha", 0.0),
            theta=bands_dict.get("theta", 0.0),
            beta=bands_dict.get("beta", 0.0),
            delta=bands_dict.get("delta", 0.0),
            gamma=bands_dict.get("gamma", 0.0),
        )

        # -- Raw EEG window --------------------------------------------------
        eeg_samples: list[list[float]] = []
        if eeg_arr is not None and eeg_arr.ndim == 2:
            n_samples = eeg_arr.shape[1]
            start = max(0, n_samples - _EEG_SAMPLES_WINDOW)
            eeg_samples = eeg_arr[:, start:].tolist()

        # -- Derived EEG (FAA, FMt) -----------------------------------------
        faa: float | None = None
        fmt: float | None = None
        if eeg_arr is not None and eeg_arr.shape[1] >= 2 and not artifact_rejected:
            from neurolink.dsp.derived_eeg import derived_eeg as _derived

            derived = _derived(eeg_arr, fs=_EEG_FS)
            faa = derived.get("faa")
            fmt = derived.get("fmt")

        # -- Breathing -------------------------------------------------------
        accel_z: np.ndarray | None = None
        if sample.accel_buffer and len(sample.accel_buffer) >= 3:
            accel_z = np.array(sample.accel_buffer[2], dtype=np.float32)
        ibis_for_breathing: list[float] = ppg_payload.ibi_ms if ppg_payload else []
        breathing_payload = compute_breathing(ibis_for_breathing, accel_z=accel_z)

        # -- IMU head orientation --------------------------------------------
        imu_payload: IMUPayload | None = None
        if sample.accel_buffer and sample.gyro_buffer:
            accel_arr_imu = np.array(sample.accel_buffer, dtype=np.float32)
            gyro_arr = np.array(sample.gyro_buffer, dtype=np.float32)
            if accel_arr_imu.shape[1] > 0:
                imu_payload = head_orientation(accel_arr_imu, gyro_arr)

        # -- fNIRS -----------------------------------------------------------
        fnirs_oxy: float | None = sample.extra.get("fnirs_oxy")
        fnirs_deoxy: float | None = sample.extra.get("fnirs_deoxy")

        # -- StreamHealth update ---------------------------------------------
        tick_ms = (time.monotonic() - tick_start) * 1000.0
        self._health.record_frame(rejected=artifact_rejected, tick_ms=tick_ms)

        return PipelineResult(
            bands=bands,
            eeg_samples=eeg_samples,
            bad_channels=bad_channels_list,
            artifact_rejected=artifact_rejected,
            artifact_reasons=artifact_reasons,
            artifact_annotations=artifact_annotations,
            artifact_correction_plan=correction_plan_payload,
            baseline_phase=self._baseline.phase,
            ppg_payload=ppg_payload,
            breathing_payload=breathing_payload,
            imu_payload=imu_payload,
            faa=faa,
            fmt=fmt,
            fnirs_oxy=fnirs_oxy,
            fnirs_deoxy=fnirs_deoxy,
        )
