"""Single source of truth for all artifact detection thresholds
and DSP configuration defaults in Neurolink-v1.
"""

from __future__ import annotations

# ── Stage 3: Amplitude gate ───────────────────────────────────────────────────
ARTIFACT_PK2PK_UV: float = 100.0

# ── Stage 3 / EA-1: Motion gate ──────────────────────────────────────────────
ARTIFACT_ACCEL_RMS_G: float = 0.15

# ── Stage 3: Kurtosis burst detection ────────────────────────────────────────
ARTIFACT_KURTOSIS_THRESHOLD: float = 5.0

# ── Stage 3b: Blink detection ────────────────────────────────────────────────
ARTIFACT_BLINK_FRONTAL_UV: float = 80.0
BLINK_FREQ_HZ_MAX: float = 10.0
BLINK_LOW_FREQ_RATIO_MIN: float = 0.50
BLINK_FRONTAL_RATIO: float = 2.0

# ── Stage 3b: Horizontal EOG (saccade) ───────────────────────────────────────
ARTIFACT_HEOG_ASYMMETRY_UV: float = 30.0
HEOG_FREQ_HZ_MAX: float = 4.0

# ── Stage 3b: EMG / muscle noise ─────────────────────────────────────────────
# Raised from 0.30 to 0.65: broadband (white-noise) background EEG at low
# amplitude has a flat spectrum where HF power ≈ 70 % of total — the old
# 0.30 threshold would false-positive on any Gaussian background signal.
# At 0.65 the threshold passes clean low-amplitude noise but still catches
# true EMG bursts (50 µV RMS broadband noise → HF ratio ≈ 0.70+).
ARTIFACT_EMG_HF_RATIO: float = 0.65

EMG_FREQ_LOW_HZ: float = 30.0
EMG_FREQ_HIGH_HZ: float = 100.0

# ── Stage 3b: Line noise ─────────────────────────────────────────────────────
ARTIFACT_LINE_FREQ_HZ: float = 60.0
ARTIFACT_LINE_BAND_HZ: float = 2.0
ARTIFACT_LINE_POWER_RATIO: float = 0.15

# ── Stage 3b: Cardiac / ballistocardiographic ────────────────────────────────
CARDIAC_FREQ_LOW_HZ: float = 0.8
CARDIAC_FREQ_HIGH_HZ: float = 1.8
CARDIAC_TEMPORAL_UV: float = 15.0

# ── Stage 3b: Electrode pop ──────────────────────────────────────────────────
ELECTRODE_POP_STEP_UV: float = 60.0
ELECTRODE_POP_ISOLATION_RATIO: float = 3.0

# ── Stage 4: ASR parameters ──────────────────────────────────────────────────
ASR_BURST_SD: float = 20.0
ASR_CALIB_SEC: float = 30.0

# ── Session baseline (impedance stabilisation + ASR calibration) ─────────────
BASELINE_TOTAL_SEC: float = 150.0
BASELINE_DISCARD_SEC: float = 30.0

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
