"""Stage 3b — Multi-type EEG artifact detector and correction router.

This module sits in the pipeline **after** the coarse ArtifactGate
(Stage 3, epoch-level reject/keep) and **before** the correction stages:

    Stage 1 : online_filter       — FIR high-pass / low-pass / notch
    Stage 2 : bad_channels        — bad channel detection + spline interp
    Stage 3 : artifact_gate       — coarse amplitude / IMU / kurtosis gate
    Stage 3b: artifact_detector   ← THIS MODULE
    Stage 4 : asr                 — Artifact Subspace Reconstruction
    Stage 5 : ocular_regression   — Gratton-Coles EOG regression

Motivation
----------
A 2025 systematic review (Sensors, Univ. Naples Federico II) identified
artifact *categorisation* as the most underaddressed step in wearable
EEG pipelines: applying generic removal without first identifying the
artifact *type* risks corrupting genuine neural components that share
frequency or amplitude characteristics with the artifact.

This module provides that categorisation layer.  It classifies up to
7 artifact types per frame, assigns per-type confidence scores, and
returns a CorrectionPlan that tells Stages 4–5 exactly which corrector
to invoke — so ocular artifacts go to ocular_regression, burst
artifacts go to ASR, line noise goes to the notch filter, and pure
motion frames are hard-rejected before any corrector wastes CPU on them.

Artifact types detected
-----------------------
BLINK           — Frontal high-amplitude slow transient (EOG blink)
HORIZONTAL_EOG  — Lateral asymmetry between AF7/AF8 (saccade)
EMG             — High-frequency broadband muscle noise (>30 Hz)
CARDIAC         — ~1.2 Hz pulse visible at temporal channels
ELECTRODE_POP   — Abrupt single-channel step / impedance transient
LINE_NOISE      — Power at the notch-band (50 or 60 Hz ± 2 Hz)
MOTION          — IMU-corroborated movement (requires accel array)

Detection strategy
------------------
Each type uses a dedicated feature extractor that operates in the
frequency domain, time domain, or both.  Spatial priors from Muse's
4-channel montage (TP9=left-temporal, AF7=left-frontal,
AF8=right-frontal, TP10=right-temporal) are used as discriminating
constraints — e.g. true eye blinks are largest at AF7/AF8, while
temporal-only high amplitude is more likely electrode pop or EMG.

All thresholds are sourced from ``neurolink.dsp.artifact_config`` and
can be overridden at runtime via ``DetectorConfig`` / ``set_config()``
without restarting the EEG pump.

Thread safety
-------------
``ArtifactDetector`` holds a single ``threading.Lock`` guarding
``_cfg``.  All detection logic runs outside the lock to avoid blocking
the EEG pump asyncio task.  Stats counters use a separate lock so
``get_stats()`` never blocks detection.

Usage
-----
    from neurolink.dsp.artifact_detector import ArtifactDetector, DetectorConfig

    detector = ArtifactDetector()

    # In the EEG pump tick (after ArtifactGate passes):
    report = detector.classify(eeg_frame, accel=accel_frame, fs=256.0)

    if report.correction_plan.hard_reject:
        continue  # motion or unrecoverable — skip frame entirely

    if report.correction_plan.apply_ocular_regression:
        eeg_frame = ocular_regression.remove(eeg_frame)

    if report.correction_plan.apply_asr:
        eeg_frame = asr.process(eeg_frame)

    if report.correction_plan.apply_notch:
        eeg_frame = online_filter.notch(eeg_frame)

    bands = compute_band_powers(eeg_frame)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np
import structlog
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    ARTIFACT_PK2PK_UV,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Channel layout (Muse S / Muse 2 / compatible 4-channel dry-electrode EEG)
# ---------------------------------------------------------------------------
_CH_TP9 = 0   # left-temporal
_CH_AF7 = 1   # left-frontal
_CH_AF8 = 2   # right-frontal
_CH_TP10 = 3  # right-temporal
_CH_AUX = 4   # auxiliary — excluded from all EEG analysis

_FRONTAL = [_CH_AF7, _CH_AF8]
_TEMPORAL = [_CH_TP9, _CH_TP10]
_ALL_EEG = [_CH_TP9, _CH_AF7, _CH_AF8, _CH_TP10]
_CH_NAMES = {_CH_TP9: "TP9", _CH_AF7: "AF7", _CH_AF8: "AF8", _CH_TP10: "TP10"}


# ---------------------------------------------------------------------------
# Artifact type taxonomy
# ---------------------------------------------------------------------------

class ArtifactType(Enum):
    """Enumeration of detectable artifact categories."""
    BLINK           = auto()   # eye-blink — frontal slow transient
    HORIZONTAL_EOG  = auto()   # lateral saccade — AF7 vs AF8 asymmetry
    EMG             = auto()   # muscle burst — broadband high-frequency
    CARDIAC         = auto()   # cardiac pulse — ~1.2 Hz temporal
    ELECTRODE_POP   = auto()   # electrode pop / impedance transient
    LINE_NOISE      = auto()   # 50 / 60 Hz power-line interference
    MOTION          = auto()   # IMU-corroborated movement artifact


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Tunable thresholds for ArtifactDetector.

    All defaults are sourced from ``neurolink.dsp.artifact_config``
    so all pipeline stages share the same authoritative baseline values.
    Runtime overrides via ``set_config()`` take effect on the next pump
    tick without restarting the process.
    """

    # ── Blink detection ────────────────────────────────────────────────
    blink_frontal_uv: float = ARTIFACT_PK2PK_UV * 0.80  # 80 µV at AF7/AF8
    blink_freq_hz_max: float = 10.0   # blink energy concentrated below 10 Hz
    blink_frontal_ratio: float = 2.0  # frontal pk2pk must be >= 2× temporal

    # ── Horizontal EOG ─────────────────────────────────────────────────
    heog_asymmetry_uv: float = 30.0   # |AF7 mean − AF8 mean| threshold (µV)
    heog_freq_hz_max: float = 4.0     # saccade energy below 4 Hz

    # ── EMG / muscle ───────────────────────────────────────────────────
    emg_hf_ratio: float = 0.30        # fraction of total power above 30 Hz
    emg_freq_low_hz: float = 30.0     # lower bound of EMG band
    emg_freq_high_hz: float = 100.0   # upper bound of EMG band

    # ── Cardiac pulse ──────────────────────────────────────────────────
    cardiac_freq_low_hz: float = 0.8   # heart-rate band start
    cardiac_freq_high_hz: float = 1.8  # heart-rate band end
    cardiac_temporal_uv: float = 15.0  # µV threshold at temporal channels

    # ── Electrode pop ──────────────────────────────────────────────────
    pop_step_uv: float = 60.0          # µV single-sample step change
    pop_isolation_ratio: float = 3.0   # single-channel amplitude / median ratio

    # ── Line noise ─────────────────────────────────────────────────────
    line_freq_hz: float = 60.0         # nominal line frequency (50 or 60 Hz)
    line_band_hz: float = 2.0          # ± bandwidth around line frequency
    line_power_ratio: float = 0.15     # fraction of broadband power in notch-band

    # ── Motion (IMU) ───────────────────────────────────────────────────
    motion_accel_rms_g: float = ARTIFACT_ACCEL_RMS_G  # g RMS threshold

    # ── Global feature switches ────────────────────────────────────────
    enable_blink: bool = True
    enable_heog: bool = True
    enable_emg: bool = True
    enable_cardiac: bool = True
    enable_electrode_pop: bool = True
    enable_line_noise: bool = True
    enable_motion: bool = True

    # ── Minimum samples for frequency-domain features ─────────────────
    min_samples_for_fft: int = 64


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ArtifactAnnotation:
    """One detected artifact instance."""
    artifact_type: ArtifactType
    confidence: float          # 0.0–1.0
    channels: list[str]        # channel names where artifact was detected
    feature_value: float       # the raw feature that triggered detection
    feature_name: str          # human-readable feature label
    threshold: float           # the threshold that was exceeded


@dataclass
class CorrectionPlan:
    """Routing instructions for downstream corrector stages.

    Each flag tells the pump which corrector(s) to apply.  Multiple
    flags can be True simultaneously (e.g. a frame can need both
    ocular regression AND notch filtering).
    """
    hard_reject: bool = False           # skip frame entirely (motion / pop)
    apply_ocular_regression: bool = False  # route to ocular_regression.py
    apply_asr: bool = False             # route to asr.py
    apply_notch: bool = False           # re-apply notch filter
    apply_cardiac_regression: bool = False  # (future) cardiac regression

    def any_correction(self) -> bool:
        """True if any corrector should be applied."""
        return (
            self.apply_ocular_regression
            or self.apply_asr
            or self.apply_notch
            or self.apply_cardiac_regression
        )


@dataclass
class DetectionReport:
    """Full result of one ArtifactDetector.classify() call."""
    annotations: list[ArtifactAnnotation] = field(default_factory=list)
    correction_plan: CorrectionPlan = field(default_factory=CorrectionPlan)
    artifact_types: list[ArtifactType] = field(default_factory=list)
    clean: bool = True  # True only when zero artifacts detected

    def add(self, annotation: ArtifactAnnotation) -> None:
        self.annotations.append(annotation)
        if annotation.artifact_type not in self.artifact_types:
            self.artifact_types.append(annotation.artifact_type)
        self.clean = False

    def type_names(self) -> list[str]:
        return [a.name for a in self.artifact_types]


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ArtifactDetector:
    """Multi-type EEG artifact classifier and correction router.

    This is a *stateless* per-frame classifier (no internal signal
    buffer).  The caller (EEGPump) is responsible for passing a frame
    of sufficient length for frequency-domain analysis
    (``DetectorConfig.min_samples_for_fft``, default 64 samples =
    250 ms at 256 Hz).

    Parameters
    ----------
    config : DetectorConfig | None
        Initial configuration.  If None, defaults are used.
    line_freq_hz : float
        Power-line frequency for this deployment environment.
        Use 60.0 for North America / Japan, 50.0 for Europe / Asia.
        Overrides ``DetectorConfig.line_freq_hz`` at construction.
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        line_freq_hz: float = 60.0,
    ) -> None:
        self._cfg_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        cfg = config or DetectorConfig()
        cfg.line_freq_hz = line_freq_hz
        self._cfg: DetectorConfig = cfg

        # Per-type counters for get_stats()
        self._total_frames: int = 0
        self._type_counts: dict[ArtifactType, int] = {t: 0 for t in ArtifactType}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        eeg: np.ndarray,
        accel: np.ndarray | None = None,
        fs: float = 256.0,
    ) -> DetectionReport:
        """Classify a single EEG frame for artifact types.

        Parameters
        ----------
        eeg : np.ndarray
            Shape (n_channels, n_samples).  Channels 0-3 are TP9, AF7,
            AF8, TP10.  Channel 4 (AUX) is ignored if present.
            Values are in microvolts (µV).
        accel : np.ndarray | None
            Accelerometer data.  Shape (3, n_samples) or (n_samples,).
            Values in *g*.  Required for MOTION detection.
        fs : float
            Sampling rate in Hz.  Must match the EEG frame's sample rate.

        Returns
        -------
        DetectionReport
            Contains all detected annotations, artifact type list, a
            CorrectionPlan routing instruction, and a ``clean`` flag.
        """
        with self._cfg_lock:
            cfg = self._cfg

        report = DetectionReport()

        if eeg is None or eeg.ndim != 2 or eeg.shape[1] < 2:
            with self._stats_lock:
                self._total_frames += 1
            return report

        n_ch = eeg.shape[0]
        valid_eeg_idx = [i for i in _ALL_EEG if i < n_ch]
        if not valid_eeg_idx:
            with self._stats_lock:
                self._total_frames += 1
            return report

        eeg_f64 = eeg[valid_eeg_idx].astype(np.float64)
        n_samples = eeg_f64.shape[1]
        has_fft = n_samples >= cfg.min_samples_for_fft

        # ── Pre-compute shared features ──────────────────────────────
        pk2pk = np.ptp(eeg_f64, axis=1)          # (n_valid_ch,)
        ch_means = np.mean(eeg_f64, axis=1)       # (n_valid_ch,)

        freqs: np.ndarray | None = None
        psd: np.ndarray | None = None
        if has_fft:
            freqs, psd = self._compute_psd(eeg_f64, fs)

        # ── Run detectors ─────────────────────────────────────────────
        if cfg.enable_motion and accel is not None:
            self._detect_motion(accel, cfg, report)

        # If hard-rejected by motion, no further spectral work needed
        if not report.correction_plan.hard_reject:

            if cfg.enable_electrode_pop:
                self._detect_electrode_pop(
                    eeg_f64, pk2pk, valid_eeg_idx, cfg, report
                )

            if has_fft and freqs is not None and psd is not None:
                frontal_idx = [
                    valid_eeg_idx.index(i) for i in _FRONTAL if i in valid_eeg_idx
                ]
                temporal_idx = [
                    valid_eeg_idx.index(i) for i in _TEMPORAL if i in valid_eeg_idx
                ]

                if cfg.enable_blink and frontal_idx:
                    self._detect_blink(
                        pk2pk, ch_means, freqs, psd,
                        frontal_idx, temporal_idx, cfg, report
                    )

                if cfg.enable_heog and frontal_idx:
                    self._detect_horizontal_eog(
                        eeg_f64, ch_means, valid_eeg_idx, cfg, report
                    )

                if cfg.enable_emg:
                    self._detect_emg(
                        freqs, psd, valid_eeg_idx, cfg, report
                    )

                if cfg.enable_cardiac and temporal_idx:
                    self._detect_cardiac(
                        freqs, psd, pk2pk, temporal_idx, valid_eeg_idx,
                        cfg, report
                    )

                if cfg.enable_line_noise:
                    self._detect_line_noise(
                        freqs, psd, valid_eeg_idx, cfg, report
                    )

        # ── Build correction plan ─────────────────────────────────────
        self._build_correction_plan(report)

        # ── Update stats ──────────────────────────────────────────────
        with self._stats_lock:
            self._total_frames += 1
            for art_type in report.artifact_types:
                self._type_counts[art_type] += 1

        if report.artifact_types:
            log.debug(
                "artifact_detector_classify",
                types=report.type_names(),
                n_annotations=len(report.annotations),
                hard_reject=report.correction_plan.hard_reject,
            )

        return report

    def get_stats(self) -> dict:
        """Return running detection counters and per-type rates."""
        with self._stats_lock:
            total = self._total_frames
            counts = dict(self._type_counts)
        rate = lambda c: round(c / total, 4) if total else 0.0
        return {
            "total_frames": total,
            "artifact_types": {
                t.name: {
                    "count": c,
                    "rate": rate(c),
                }
                for t, c in counts.items()
            },
        }

    def reset_stats(self) -> None:
        """Reset all counters. Call at session start."""
        with self._stats_lock:
            self._total_frames = 0
            for t in ArtifactType:
                self._type_counts[t] = 0
        log.info("artifact_detector_stats_reset")

    def get_config(self) -> DetectorConfig:
        import copy
        with self._cfg_lock:
            return copy.copy(self._cfg)

    def set_config(self, config: DetectorConfig) -> None:
        with self._cfg_lock:
            self._cfg = config
        log.info("artifact_detector_config_updated")

    # ------------------------------------------------------------------ #
    # Shared feature helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_psd(
        eeg_f64: np.ndarray,
        fs: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Welch PSD for all channels.

        Returns
        -------
        freqs : (n_freqs,) array
        psd   : (n_channels, n_freqs) array — power in µV²/Hz
        """
        n_samples = eeg_f64.shape[1]
        # nperseg: use at most half the window, minimum 32 samples
        nperseg = max(32, min(256, n_samples // 2))
        freqs, psd = sp_signal.welch(
            eeg_f64,
            fs=fs,
            nperseg=nperseg,
            axis=1,
            scaling="density",
        )
        return freqs, psd

    @staticmethod
    def _band_power(
        freqs: np.ndarray,
        psd: np.ndarray,
        f_low: float,
        f_high: float,
    ) -> np.ndarray:
        """Integrate PSD between f_low and f_high for each channel.

        Returns (n_channels,) array of band power values.
        """
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not mask.any():
            return np.zeros(psd.shape[0])
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return np.sum(psd[:, mask], axis=1) * df

    @staticmethod
    def _total_power(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """Total broadband power per channel (1–100 Hz)."""
        mask = (freqs >= 1.0) & (freqs <= 100.0)
        if not mask.any():
            return np.sum(psd, axis=1)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return np.sum(psd[:, mask], axis=1) * df

    # ------------------------------------------------------------------ #
    # Per-type detectors
    # ------------------------------------------------------------------ #

    def _detect_blink(
        self,
        pk2pk: np.ndarray,
        ch_means: np.ndarray,
        freqs: np.ndarray,
        psd: np.ndarray,
        frontal_idx: list[int],
        temporal_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect eye-blink artifact.

        Criteria (all must pass):
        1. Frontal pk2pk >= blink_frontal_uv
        2. Frontal low-frequency power ratio (0–10 Hz) >= 0.60
        3. Frontal pk2pk >= blink_frontal_ratio × temporal pk2pk
           (ensures the large amplitude is genuinely frontal, not global)
        """
        frontal_pk2pk = pk2pk[frontal_idx]
        max_frontal_pk2pk = float(np.max(frontal_pk2pk))

        # Criterion 1: amplitude threshold at frontal channels
        if max_frontal_pk2pk < cfg.blink_frontal_uv:
            return

        # Criterion 2: spectral — blink energy concentrated below 10 Hz
        frontal_low = float(
            np.mean(self._band_power(freqs, psd[frontal_idx], 0.5, cfg.blink_freq_hz_max))
        )
        frontal_total = float(np.mean(self._total_power(freqs, psd[frontal_idx])))
        if frontal_total <= 0:
            return
        low_freq_ratio = frontal_low / frontal_total
        if low_freq_ratio < 0.50:
            return

        # Criterion 3: topographic — frontal much larger than temporal
        if temporal_idx:
            temporal_pk2pk = pk2pk[temporal_idx]
            median_temporal = float(np.median(temporal_pk2pk))
            if median_temporal > 0:
                ratio = max_frontal_pk2pk / median_temporal
                if ratio < cfg.blink_frontal_ratio:
                    return
            frontal_ch_names = [_CH_NAMES.get(_ALL_EEG[i], f"ch{i}") for i in frontal_idx]
        else:
            frontal_ch_names = [_CH_NAMES.get(_ALL_EEG[i], f"ch{i}") for i in frontal_idx]

        confidence = min(1.0, (max_frontal_pk2pk / cfg.blink_frontal_uv) * 0.5 + low_freq_ratio * 0.5)
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.BLINK,
            confidence=round(confidence, 3),
            channels=frontal_ch_names,
            feature_value=round(max_frontal_pk2pk, 2),
            feature_name="frontal_pk2pk_uv",
            threshold=cfg.blink_frontal_uv,
        ))

    def _detect_horizontal_eog(
        self,
        eeg_f64: np.ndarray,
        ch_means: np.ndarray,
        valid_eeg_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect horizontal eye movement (saccade) artifact.

        Strategy: A saccade produces a slow potential with opposite
        polarity at AF7 vs AF8 (left vs right frontal).  We measure
        the absolute difference in channel means as a proxy for
        lateral asymmetry.  A large asymmetry at low frequency with
        opposite sign indicates a saccade.
        """
        if _CH_AF7 not in valid_eeg_idx or _CH_AF8 not in valid_eeg_idx:
            return

        af7_pos = valid_eeg_idx.index(_CH_AF7)
        af8_pos = valid_eeg_idx.index(_CH_AF8)

        # Low-pass at cfg.heog_freq_hz_max to isolate slow component
        # (simple mean approximation — avoids heavy FIR in hot path)
        af7_mean = float(ch_means[af7_pos])
        af8_mean = float(ch_means[af8_pos])
        asymmetry = abs(af7_mean - af8_mean)

        if asymmetry < cfg.heog_asymmetry_uv:
            return

        # Confirm opposite polarity (classic saccade signature)
        opposite_polarity = (af7_mean > 0) != (af8_mean > 0)

        confidence = min(1.0, asymmetry / (cfg.heog_asymmetry_uv * 2))
        if not opposite_polarity:
            confidence *= 0.70  # lower confidence without polarity confirmation

        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.HORIZONTAL_EOG,
            confidence=round(confidence, 3),
            channels=["AF7", "AF8"],
            feature_value=round(asymmetry, 2),
            feature_name="af7_af8_mean_asymmetry_uv",
            threshold=cfg.heog_asymmetry_uv,
        ))

    def _detect_emg(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        valid_eeg_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect EMG / muscle artifact.

        A high ratio of power in the 30–100 Hz band relative to total
        broadband power (1–100 Hz) indicates broadband muscle noise.
        Evaluated per channel; any channel exceeding the threshold
        triggers a detection.
        """
        hf_power = self._band_power(freqs, psd, cfg.emg_freq_low_hz, cfg.emg_freq_high_hz)
        total = self._total_power(freqs, psd)

        contaminated_channels: list[str] = []
        max_ratio = 0.0

        for i, ch_idx in enumerate(valid_eeg_idx):
            if total[i] <= 0:
                continue
            ratio = float(hf_power[i] / total[i])
            if ratio > cfg.emg_hf_ratio:
                contaminated_channels.append(_CH_NAMES.get(ch_idx, f"ch{ch_idx}"))
                max_ratio = max(max_ratio, ratio)

        if not contaminated_channels:
            return

        confidence = min(1.0, max_ratio / cfg.emg_hf_ratio)
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.EMG,
            confidence=round(confidence, 3),
            channels=contaminated_channels,
            feature_value=round(max_ratio, 4),
            feature_name="hf_power_ratio_30_100hz",
            threshold=cfg.emg_hf_ratio,
        ))

    def _detect_cardiac(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        pk2pk: np.ndarray,
        temporal_idx: list[int],
        valid_eeg_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect cardiac / ballistocardiographic artifact.

        The heartbeat produces a characteristic ~1.2 Hz component most
        prominent at temporal channels (TP9, TP10) due to their
        proximity to temporal arteries.

        Criteria:
        1. Cardiac-band (0.8–1.8 Hz) power at temporal channels is
           a significant fraction of their total low-frequency power
        2. Temporal pk2pk exceeds cardiac_temporal_uv threshold
        """
        cardiac_power = self._band_power(
            freqs, psd, cfg.cardiac_freq_low_hz, cfg.cardiac_freq_high_hz
        )
        low_total = self._band_power(freqs, psd, 0.5, 4.0)  # delta + cardiac band

        contaminated: list[str] = []
        max_ratio = 0.0

        for li, vi in zip(temporal_idx, [_CH_TP9, _CH_TP10]):
            if vi not in valid_eeg_idx:
                continue
            vi_pos = valid_eeg_idx.index(vi)
            if low_total[vi_pos] <= 0:
                continue
            ratio = float(cardiac_power[vi_pos] / low_total[vi_pos])
            amp = float(pk2pk[vi_pos])
            if ratio > 0.35 and amp > cfg.cardiac_temporal_uv:
                contaminated.append(_CH_NAMES[vi])
                max_ratio = max(max_ratio, ratio)

        if not contaminated:
            return

        confidence = min(1.0, max_ratio / 0.5)
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.CARDIAC,
            confidence=round(confidence, 3),
            channels=contaminated,
            feature_value=round(max_ratio, 4),
            feature_name="cardiac_band_power_ratio",
            threshold=0.35,
        ))

    def _detect_electrode_pop(
        self,
        eeg_f64: np.ndarray,
        pk2pk: np.ndarray,
        valid_eeg_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect electrode pop / impedance transient.

        An electrode pop is characterised by:
        1. An abrupt single-sample step change >= pop_step_uv on one channel
        2. The affected channel's pk2pk is >= pop_isolation_ratio × median
           of all other channels (spatially isolated, not global)
        """
        n_samples = eeg_f64.shape[1]
        if n_samples < 3:
            return

        # Per-channel max step (absolute first-difference)
        diff = np.abs(np.diff(eeg_f64, axis=1))
        max_step = np.max(diff, axis=1)  # (n_valid_ch,)

        median_pk2pk = float(np.median(pk2pk)) if len(pk2pk) > 1 else float(pk2pk[0])

        contaminated: list[str] = []
        max_feature = 0.0

        for i, ch_idx in enumerate(valid_eeg_idx):
            if max_step[i] < cfg.pop_step_uv:
                continue
            # Isolation: this channel's pk2pk must be >> others
            if median_pk2pk > 0:
                isolation = float(pk2pk[i] / median_pk2pk)
            else:
                isolation = cfg.pop_isolation_ratio + 1.0  # pass if all zero median
            if isolation >= cfg.pop_isolation_ratio:
                contaminated.append(_CH_NAMES.get(ch_idx, f"ch{ch_idx}"))
                max_feature = max(max_feature, float(max_step[i]))

        if not contaminated:
            return

        confidence = min(1.0, max_feature / (cfg.pop_step_uv * 2))
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.ELECTRODE_POP,
            confidence=round(confidence, 3),
            channels=contaminated,
            feature_value=round(max_feature, 2),
            feature_name="max_single_sample_step_uv",
            threshold=cfg.pop_step_uv,
        ))

    def _detect_line_noise(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        valid_eeg_idx: list[int],
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect power-line interference (50 or 60 Hz).

        Measures the fraction of broadband power (1–100 Hz) concentrated
        in a narrow band around the line frequency.  Because Stage 1
        (online_filter) should already attenuate line noise, a high ratio
        here indicates either filter bypass or extremely strong coupling.
        """
        f_lo = cfg.line_freq_hz - cfg.line_band_hz
        f_hi = cfg.line_freq_hz + cfg.line_band_hz

        notch_power = self._band_power(freqs, psd, f_lo, f_hi)
        total = self._total_power(freqs, psd)

        contaminated: list[str] = []
        max_ratio = 0.0

        for i, ch_idx in enumerate(valid_eeg_idx):
            if total[i] <= 0:
                continue
            ratio = float(notch_power[i] / total[i])
            if ratio > cfg.line_power_ratio:
                contaminated.append(_CH_NAMES.get(ch_idx, f"ch{ch_idx}"))
                max_ratio = max(max_ratio, ratio)

        if not contaminated:
            return

        confidence = min(1.0, max_ratio / (cfg.line_power_ratio * 2))
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.LINE_NOISE,
            confidence=round(confidence, 3),
            channels=contaminated,
            feature_value=round(max_ratio, 4),
            feature_name=f"line_{int(cfg.line_freq_hz)}hz_power_ratio",
            threshold=cfg.line_power_ratio,
        ))

    def _detect_motion(
        self,
        accel: np.ndarray,
        cfg: DetectorConfig,
        report: DetectionReport,
    ) -> None:
        """Detect motion artifact using IMU accelerometer data.

        This is the most reliable motion detector because it uses a
        direct physical measurement rather than inferring motion from
        the EEG signal itself.  A detected MOTION artifact sets
        hard_reject=True in the CorrectionPlan — the frame is
        discarded before any corrector is invoked.
        """
        accel_arr = np.asarray(accel, dtype=np.float64)
        if accel_arr.ndim == 1:
            accel_arr = accel_arr[np.newaxis, :]
        rms = float(np.sqrt(np.mean(accel_arr ** 2)))

        if rms <= cfg.motion_accel_rms_g:
            return

        confidence = min(1.0, rms / (cfg.motion_accel_rms_g * 2))
        report.add(ArtifactAnnotation(
            artifact_type=ArtifactType.MOTION,
            confidence=round(confidence, 3),
            channels=["TP9", "AF7", "AF8", "TP10"],  # motion affects all channels
            feature_value=round(rms, 4),
            feature_name="accel_rms_g",
            threshold=cfg.motion_accel_rms_g,
        ))
        report.correction_plan.hard_reject = True

    # ------------------------------------------------------------------ #
    # Correction plan builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_correction_plan(report: DetectionReport) -> None:
        """Map detected artifact types to corrector routing flags.

        Routing logic (applied in priority order):
        - MOTION          → hard_reject (already set by _detect_motion)
        - ELECTRODE_POP   → hard_reject (spatially isolated, uncorrectable)
        - BLINK / H_EOG   → apply_ocular_regression
        - EMG             → apply_asr (broadband, ASR handles it best)
        - CARDIAC         → apply_cardiac_regression (future) + apply_asr
        - LINE_NOISE      → apply_notch
        """
        if report.correction_plan.hard_reject:
            return  # already decided — no further routing

        types = report.artifact_types

        if ArtifactType.ELECTRODE_POP in types:
            report.correction_plan.hard_reject = True
            return

        if ArtifactType.BLINK in types or ArtifactType.HORIZONTAL_EOG in types:
            report.correction_plan.apply_ocular_regression = True

        if ArtifactType.EMG in types:
            report.correction_plan.apply_asr = True

        if ArtifactType.CARDIAC in types:
            report.correction_plan.apply_cardiac_regression = True
            report.correction_plan.apply_asr = True

        if ArtifactType.LINE_NOISE in types:
            report.correction_plan.apply_notch = True
