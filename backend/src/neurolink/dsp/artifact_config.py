"""Single source of truth for all artifact detection thresholds
and DSP configuration defaults in Neurolink-v1.

All other modules — artifact_gate, asr, ea1_scorer, classifiers,
artifact_detector, ocular_regression — import constants from here
instead of defining their own.  Runtime per-session overrides go
through GateConfig / ASRConfig / DetectorConfig set_config(), which
are seeded from these defaults so the existing override API requires
no changes.

Adaptive tightening example
---------------------------
from neurolink.dsp.artifact_config import ARTIFACT_PK2PK_UV, ARTIFACT_ACCEL_RMS_G
from neurolink.dsp.artifact_gate import GateConfig

# Standard defaults at session start:
gate.set_config(GateConfig())

# Conservative tighten after clean calibration window:
gate.set_config(GateConfig(
    pk2pk_uv=ARTIFACT_PK2PK_UV * 0.75,        # 75 µV
    accel_rms_g=ARTIFACT_ACCEL_RMS_G * 0.80,  # 0.12 g
))
"""

from __future__ import annotations

# ── Stage 3: Amplitude gate ───────────────────────────────────────────────────
ARTIFACT_PK2PK_UV: float = 100.0
"""Peak-to-peak amplitude limit (µV).  Lindsley 1944 / EEGLAB convention;
validated for Muse-class dry-electrode wearables."""

# ── Stage 3 / EA-1: Motion gate ──────────────────────────────────────────────
ARTIFACT_ACCEL_RMS_G: float = 0.15
"""IMU accelerometer RMS threshold (g) above which a frame is motion-
contaminated.  Shared by ArtifactGate (frame-level rejection) and
EA1Scorer (session-level motion criterion) so both layers gate on the
same physical threshold — a session cannot be scored eligible on frames
the gate has already rejected as motion-contaminated."""

# ── Stage 3: Kurtosis burst detection ────────────────────────────────────────
ARTIFACT_KURTOSIS_THRESHOLD: float = 5.0
"""Excess-kurtosis threshold (Fisher convention, scipy.stats default).
Values > 5 indicate EMG burst or electrode-pop contamination."""

# ── Stage 3b: Blink detection ────────────────────────────────────────────────
ARTIFACT_BLINK_FRONTAL_UV: float = 80.0
"""Minimum peak-to-peak amplitude at AF7/AF8 (µV) to classify a blink.
Set to 80 % of ARTIFACT_PK2PK_UV — blinks are slightly sub-threshold
for the coarse amplitude gate but clearly frontal."""

BLINK_FREQ_HZ_MAX: float = 10.0
"""Blink energy is concentrated below this frequency (Hz)."""

BLINK_LOW_FREQ_RATIO_MIN: float = 0.50
"""Minimum fraction of total frontal power that must lie below
BLINK_FREQ_HZ_MAX for a blink classification to proceed.  Prevents
high-amplitude broadband bursts (EMG) from being misclassified as blinks."""

BLINK_FRONTAL_RATIO: float = 2.0
"""Frontal pk2pk must be >= this multiple of temporal pk2pk.
Ensures the high amplitude is genuinely frontal-dominant (blink)
rather than a global artifact."""

# ── Stage 3b: Horizontal EOG (saccade) ───────────────────────────────────────
ARTIFACT_HEOG_ASYMMETRY_UV: float = 30.0
"""Minimum |AF7_mean − AF8_mean| (µV) to flag a horizontal saccade."""

HEOG_FREQ_HZ_MAX: float = 4.0
"""Saccade energy is concentrated below this frequency (Hz)."""

# ── Stage 3b: EMG / muscle noise ─────────────────────────────────────────────
ARTIFACT_EMG_HF_RATIO: float = 0.30
"""Fraction of total broadband power (1–100 Hz) that must lie in the
30–100 Hz band to classify a frame as EMG-contaminated."""

EMG_FREQ_LOW_HZ: float = 30.0
"""Lower bound of the EMG detection band (Hz)."""

EMG_FREQ_HIGH_HZ: float = 100.0
"""Upper bound of the EMG detection band (Hz)."""

# ── Stage 3b: Line noise ─────────────────────────────────────────────────────
ARTIFACT_LINE_FREQ_HZ: float = 60.0
"""Nominal power-line frequency (Hz).  Use 60.0 for North America /
Japan; 50.0 for Europe / Asia.  Passed to ArtifactDetector at
construction via DetectorConfig.line_freq_hz."""

ARTIFACT_LINE_BAND_HZ: float = 2.0
"""Half-bandwidth (Hz) around ARTIFACT_LINE_FREQ_HZ used to measure
notch-band power.  Total window = line_freq ± line_band_hz."""

ARTIFACT_LINE_POWER_RATIO: float = 0.15
"""Fraction of broadband power (1–100 Hz) in the notch-band above
which the frame is classified as line-noise contaminated."""

# ── Stage 3b: Cardiac / ballistocardiographic ────────────────────────────────
CARDIAC_FREQ_LOW_HZ: float = 0.8
"""Lower bound of the cardiac-band (Hz)."""

CARDIAC_FREQ_HIGH_HZ: float = 1.8
"""Upper bound of the cardiac-band (Hz)."""

CARDIAC_TEMPORAL_UV: float = 15.0
"""Minimum pk2pk (µV) at temporal channels required to confirm cardiac
artifact (prevents false positives from very low-amplitude delta
transients in the same frequency band)."""

# ── Stage 3b: Electrode pop ──────────────────────────────────────────────────
ELECTRODE_POP_STEP_UV: float = 60.0
"""Minimum single-sample step change (µV) to flag an electrode pop."""

ELECTRODE_POP_ISOLATION_RATIO: float = 3.0
"""Minimum ratio of the flagged channel's pk2pk to the median of all
other channels.  Ensures the large step is spatially isolated to one
channel (true pop) rather than a global movement artifact."""

# ── Stage 4: ASR parameters ──────────────────────────────────────────────────
ASR_BURST_SD: float = 20.0
"""BurstCriterion (calibration SDs).  EEGLAB default = 20.
Lower values (e.g. 15) are more aggressive; raise to 25 for noisier
environments."""

ASR_CALIB_SEC: float = 30.0
"""Seconds of clean resting EEG required before ASR activates.
The BaselineRecorder feeds ASR during its 120-second RECORDING window
(seconds 30-150 of the baseline), so ASR will be calibrated well before
the baseline completes.  This value is the ASR module's own internal
minimum — raise to 60 s for more stable covariance estimates."""

# ── Session baseline (impedance stabilisation + ASR calibration) ─────────────
BASELINE_TOTAL_SEC: float = 150.0
"""Total eyes-closed resting baseline duration (seconds).
The first BASELINE_DISCARD_SEC are discarded for dry-electrode
impedance stabilisation; the remainder feed ASR calibration.
A bell sounds when this window is complete."""

BASELINE_DISCARD_SEC: float = 30.0
"""Seconds discarded at the start of the baseline for electrode
stabilisation.  Dry electrodes require 20-40 s to form a stable
sweat-film contact; data from this period is unreliable regardless
of amplitude."""

# ── EA-1 scorer thresholds ───────────────────────────────────────────────────
EA1_ALPHA_THRESHOLD: float = 0.30
EA1_THETA_THRESHOLD: float = 0.15
EA1_CONTACT_QUALITY_MIN: float = 0.5

# ── Classifier v0.1 thresholds ───────────────────────────────────────────────
V01_ALPHA_E: float = 0.30
V01_THETA_E: float = 0.15
V01_ALPHA_D: float = 0.22
V01_THETA_D: float = 0.18
V01_ALPHA_C: float = 0.22
V01_BETA_B: float = 0.30
V01_DELTA_F: float = 0.50
V01_GAMMA_G: float = 0.20
V01_MULTIPLICATIO_ALPHA: float = 0.35
V01_MULTIPLICATIO_THETA: float = 0.15
V01_MULTIPLICATIO_FAA: float = -0.05

# ── Classifier v2 thresholds ─────────────────────────────────────────────────
V2_ALPHA_RUBEDO: float = 0.30
V2_THETA_RUBEDO: float = 0.15
V2_BETA_RUBEDO_MAX: float = 0.25
V2_ALPHA_MULTIPLICATIO: float = 0.33
V2_BETA_ALBEDO: float = 0.28
V2_THETA_SOLUTIO: float = 0.25
V2_DELTA_COAGULATIO: float = 0.45
V2_GAMMA_SUBLIMATIO: float = 0.20
V2_BETA_CALCINATIO: float = 0.40
