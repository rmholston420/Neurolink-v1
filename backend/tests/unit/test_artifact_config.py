"""Contract tests for neurolink.dsp.artifact_config constants.

These tests act as a guard against accidental threshold changes that
would silently shift the pipeline's sensitivity.  Every constant must:
  - Be a positive finite float
  - Fall within a physiologically and algorithmically valid range

If a threshold needs to change intentionally, update both the constant
and the corresponding assertion here with a brief comment explaining why.
"""

from __future__ import annotations

import math

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    # Stage 3 -- amplitude / motion / kurtosis
    ARTIFACT_PK2PK_UV,
    # Stage 4 -- ASR
    ASR_BURST_SD,
    ASR_CALIB_SEC,
    BASELINE_DISCARD_SEC,
    # Baseline recording
    BASELINE_TOTAL_SEC,
    # EA-1 scorer
    EA1_ALPHA_THRESHOLD,
    EA1_CONTACT_QUALITY_MIN,
    EA1_THETA_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_positive_finite(v: float) -> bool:
    return isinstance(v, float) and math.isfinite(v) and v > 0.0


# ---------------------------------------------------------------------------
# Stage 3 thresholds
# ---------------------------------------------------------------------------


def test_pk2pk_uv_positive_finite():
    assert _is_positive_finite(ARTIFACT_PK2PK_UV)


def test_pk2pk_uv_physiological_range():
    """EEGLAB convention: 50-200 uV is reasonable for wearable EEG."""
    assert 20.0 <= ARTIFACT_PK2PK_UV <= 500.0


def test_accel_rms_g_positive_finite():
    assert _is_positive_finite(ARTIFACT_ACCEL_RMS_G)


def test_accel_rms_g_range():
    """0.05-1.0 g covers subtle to vigorous head movement."""
    assert 0.01 <= ARTIFACT_ACCEL_RMS_G <= 2.0


def test_kurtosis_threshold_positive_finite():
    assert _is_positive_finite(ARTIFACT_KURTOSIS_THRESHOLD)


def test_kurtosis_threshold_range():
    """Fisher kurtosis threshold: 3-10 is the typical EMG-burst detection range."""
    assert 1.0 <= ARTIFACT_KURTOSIS_THRESHOLD <= 20.0


# ---------------------------------------------------------------------------
# Stage 4 -- ASR
# ---------------------------------------------------------------------------


def test_asr_burst_sd_positive_finite():
    assert _is_positive_finite(ASR_BURST_SD)


def test_asr_burst_sd_range():
    """EEGLAB recommends 10-25 SD.  Anything outside 5-50 is almost certainly wrong."""
    assert 5.0 <= ASR_BURST_SD <= 50.0


def test_asr_calib_sec_positive_finite():
    assert _is_positive_finite(ASR_CALIB_SEC)


def test_asr_calib_sec_range():
    """10-300 s is a sensible range for calibration windows."""
    assert 10.0 <= ASR_CALIB_SEC <= 300.0


# ---------------------------------------------------------------------------
# Baseline recording
# ---------------------------------------------------------------------------


def test_baseline_total_sec_positive_finite():
    assert _is_positive_finite(BASELINE_TOTAL_SEC)


def test_baseline_discard_sec_positive_finite():
    assert _is_positive_finite(BASELINE_DISCARD_SEC)


def test_baseline_discard_less_than_total():
    """The discard window must be shorter than the total baseline."""
    assert BASELINE_DISCARD_SEC < BASELINE_TOTAL_SEC


def test_baseline_total_range():
    """30 s to 10 min is a sensible total baseline."""
    assert 30.0 <= BASELINE_TOTAL_SEC <= 600.0


# ---------------------------------------------------------------------------
# EA-1 scorer thresholds
# ---------------------------------------------------------------------------


def test_ea1_alpha_threshold_range():
    """Relative band-power thresholds must be in (0, 1)."""
    assert 0.0 < EA1_ALPHA_THRESHOLD < 1.0


def test_ea1_theta_threshold_range():
    assert 0.0 < EA1_THETA_THRESHOLD < 1.0


def test_ea1_contact_quality_min_range():
    """Contact quality is normalised 0-1."""
    assert 0.0 < EA1_CONTACT_QUALITY_MIN <= 1.0


# ---------------------------------------------------------------------------
# Consistency: ASR calibration must fit inside baseline recording window
# ---------------------------------------------------------------------------


def test_asr_calib_fits_in_baseline_recording_window():
    """ASR_CALIB_SEC must be <= BASELINE_TOTAL_SEC - BASELINE_DISCARD_SEC."""
    recording_window = BASELINE_TOTAL_SEC - BASELINE_DISCARD_SEC
    assert ASR_CALIB_SEC <= recording_window, (
        f"ASR_CALIB_SEC ({ASR_CALIB_SEC}s) exceeds available recording window ({recording_window}s)"
    )
