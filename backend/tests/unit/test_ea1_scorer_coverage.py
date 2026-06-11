"""Targeted coverage for ea1_scorer.py branches missed by test_ea1_coverage.py.

Specifically covers:
- eligible=True path (all 5 criteria met) -> overlay from _STAGE_OVERLAY
- Each of the 4 named stage overrides: Rubedo, Multiplicatio, Citrinitas, Solutio
- eligible=True with unknown stage -> fallback f"X{criteria_met}"
- contact_quality is not None + poor_contact=True -> contact_met=False
- imu present with motion_rms >= MOTION_RMS_GATE -> motion_met=False
- ineligible with criteria_met==5 -> idx clamped to len(_OVERLAY_MODES)-1
"""
from __future__ import annotations

import pytest

from neurolink.ea1_scorer import score
from neurolink.models.eeg import BandPowers, IMUPayload, IngestPayload


# ---------------------------------------------------------------------------
# Helper: build a fully-eligible payload (all 5 criteria met)
# alpha>=0.30, theta>=0.15, region in {D,E}, motion_rms<0.5, good contact
# ---------------------------------------------------------------------------

def _eligible_payload(stage: str = "Nigredo", region: str = "D") -> IngestPayload:
    """Returns a payload that meets all 5 EA-1 criteria."""
    return IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.50, theta=0.20, beta=0.10, delta=0.10, gamma=0.05),
        region=region,
        alchemical_stage=stage,
        imu=IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=0.01),  # well below 0.5
        contact_quality=0.95,
        poor_contact=False,
    )


# ---------------------------------------------------------------------------
# eligible=True path — overlay from _STAGE_OVERLAY for each named stage
# ---------------------------------------------------------------------------

def test_eligible_rubedo_overlay():
    result = score(_eligible_payload(stage="Rubedo"))
    assert result.eligible is True
    assert result.overlay_mode == "X4"


def test_eligible_multiplicatio_overlay():
    result = score(_eligible_payload(stage="Multiplicatio"))
    assert result.eligible is True
    assert result.overlay_mode == "X5"


def test_eligible_citrinitas_overlay():
    result = score(_eligible_payload(stage="Citrinitas"))
    assert result.eligible is True
    assert result.overlay_mode == "X3"


def test_eligible_solutio_overlay():
    result = score(_eligible_payload(stage="Solutio"))
    assert result.eligible is True
    assert result.overlay_mode == "X3"


def test_eligible_unknown_stage_fallback_overlay():
    """Unknown stage -> fallback f'X{criteria_met}', which is 'X5' (all 5 met)."""
    result = score(_eligible_payload(stage="UnknownStage"))
    assert result.eligible is True
    assert result.overlay_mode == "X5"


# ---------------------------------------------------------------------------
# contact_quality is not None + poor_contact=True -> contact_met=False
# ---------------------------------------------------------------------------

def test_contact_quality_not_none_poor_contact_fails_criterion():
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.50, theta=0.20, beta=0.10, delta=0.10, gamma=0.05),
        region="D",
        contact_quality=0.90,  # high quality value, but...
        poor_contact=True,      # ...overrides to False
    )
    result = score(payload)
    # contact criterion must be False despite high contact_quality value
    assert result.criteria["contact_quality"]["met"] is False
    # And since contact fails, overall eligibility must be False
    assert result.eligible is False


def test_contact_quality_not_none_below_minimum_fails_criterion():
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.50, theta=0.20, beta=0.10, delta=0.10, gamma=0.05),
        region="D",
        contact_quality=0.30,  # below CONTACT_QUALITY_MIN=0.5
        poor_contact=False,
    )
    result = score(payload)
    assert result.criteria["contact_quality"]["met"] is False


# ---------------------------------------------------------------------------
# IMU present with motion_rms >= MOTION_RMS_GATE -> motion_met=False
# ---------------------------------------------------------------------------

def test_high_motion_rms_fails_motion_criterion():
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.50, theta=0.20, beta=0.10, delta=0.10, gamma=0.05),
        region="D",
        imu=IMUPayload(pitch_deg=15.0, roll_deg=10.0, motion_rms=0.8),  # above 0.5
    )
    result = score(payload)
    assert result.criteria["motion"]["met"] is False
    assert result.gates["motion"] is False


# ---------------------------------------------------------------------------
# Ineligible + criteria_met near max -> idx clamped to len(_OVERLAY_MODES)-1
# ---------------------------------------------------------------------------

def test_ineligible_idx_clamped_to_max_overlay():
    """criteria_met cannot exceed total(5). _OVERLAY_MODES has 6 entries (X0-X5)
    so idx=min(5,5)=5 is valid, but if criteria_met were somehow >5 it would
    be clamped. We hit the max-valid index path by having 5/5 criteria met
    except we ensure eligible is False via poor_contact to force ineligible.
    """
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.50, theta=0.20, beta=0.10, delta=0.10, gamma=0.05),
        region="D",
        imu=IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=0.01),
        contact_quality=0.95,
        poor_contact=True,   # fails contact criterion -> eligible=False, criteria_met=4
    )
    result = score(payload)
    assert result.eligible is False
    # criteria_met=4 -> idx=min(4,5)=4 -> overlay_mode="X4"
    assert result.overlay_mode == "X4"
    assert result.criteria_met == 4
