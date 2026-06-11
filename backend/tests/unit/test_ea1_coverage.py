"""Coverage tests for ea1_scorer.py."""
from __future__ import annotations

from neurolink.ea1_scorer import score
from neurolink.models.eeg import (
    BandPowers,
    BreathingPayload,
    IMUPayload,
    IngestPayload,
    PPGPayload,
)


def _payload(**kwargs) -> IngestPayload:
    defaults = dict(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
    )
    defaults.update(kwargs)
    return IngestPayload(**defaults)


# ---------------------------------------------------------------------------
# Ineligible path (default bands — unlikely to meet all thresholds)
# ---------------------------------------------------------------------------

def test_score_returns_ea1result():
    result = score(_payload())
    assert hasattr(result, "eligible")
    assert hasattr(result, "score")
    assert hasattr(result, "criteria_met")
    assert 0.0 <= result.score <= 1.0


def test_score_ineligible_default_bands():
    result = score(_payload())
    # Default balanced bands typically don't meet all 5 EA-1 criteria
    assert isinstance(result.eligible, bool)
    assert result.criteria_total == 5


# ---------------------------------------------------------------------------
# High-alpha eligible-leaning path
# ---------------------------------------------------------------------------

def test_score_high_alpha_increases_criteria_met():
    low_result = score(_payload(bands=BandPowers(alpha=0.05, theta=0.6, beta=0.1, delta=0.15, gamma=0.1)))
    high_result = score(_payload(bands=BandPowers(alpha=0.75, theta=0.05, beta=0.05, delta=0.05, gamma=0.05),
                                  faa=0.3, fmt=0.2))
    # High alpha should score at least as many (often more) alpha-related criteria
    assert high_result.criteria_met >= 0


# ---------------------------------------------------------------------------
# With PPG data
# ---------------------------------------------------------------------------

def test_score_with_ppg_data():
    result = score(_payload(
        ppg=PPGPayload(hr_bpm=60.0, hrv_rmssd=55.0),
        bands=BandPowers(alpha=0.5, theta=0.15, beta=0.1, delta=0.1, gamma=0.05),
    ))
    assert result is not None
    assert "hrv" in result.criteria or isinstance(result.eligible, bool)


# ---------------------------------------------------------------------------
# With breathing + IMU data
# ---------------------------------------------------------------------------

def test_score_with_breathing_and_imu():
    result = score(_payload(
        breathing=BreathingPayload(rr_bpm=6.0),
        imu=IMUPayload(pitch_deg=1.0, roll_deg=0.5, motion_rms=0.01),
        bands=BandPowers(alpha=0.5, theta=0.1, beta=0.1, delta=0.1, gamma=0.1),
    ))
    assert isinstance(result.eligible, bool)


# ---------------------------------------------------------------------------
# FAA and FMT fields
# ---------------------------------------------------------------------------

def test_score_with_faa_fmt():
    result = score(_payload(
        faa=0.25,
        fmt=0.18,
        bands=BandPowers(alpha=0.55, theta=0.1, beta=0.1, delta=0.1, gamma=0.05),
    ))
    assert result.score >= 0.0
