"""Single source of truth for all artifact detection thresholds
and DSP configuration defaults in Neurolink-v1.

All other modules — artifact_gate, asr, ea1_scorer, classifiers —
import constants from here instead of defining their own.
Runtime per-session overrides go through GateConfig / ASRConfig
set_config(), which are seeded from these defaults so the existing
override API requires no changes.

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

# ── Stage 4: ASR parameters ──────────────────────────────────────────────────
ASR_BURST_SD: float = 20.0
"""BurstCriterion (calibration SDs).  EEGLAB default = 20.
Lower values (e.g. 15) are more aggressive; raise to 25 for noisier
environments."""

ASR_CALIB_SEC: float = 30.0
"""Seconds of clean resting EEG required before ASR activates.
Increase to 60 s for more stable covariance estimates."""

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
