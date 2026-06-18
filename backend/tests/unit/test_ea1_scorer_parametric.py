"""Parametric boundary tests for the EA-1 scorer.

Each of the 5 criteria is tested individually at its threshold boundary,
plus combinations for full-eligible and never-eligible payloads.

All tests are pure (no I/O, no async).
"""

from __future__ import annotations

import pytest

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    EA1_ALPHA_THRESHOLD,
    EA1_CONTACT_QUALITY_MIN,
    EA1_THETA_THRESHOLD,
)
from neurolink.ea1_scorer import score as ea1_score
from neurolink.models.eeg import BandPowers, IMUPayload, IngestPayload, SSpaceCoords


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eligible_payload(
    alpha: float = EA1_ALPHA_THRESHOLD,
    theta: float = EA1_THETA_THRESHOLD,
    region: str = "D",
    motion_rms: float = 0.0,
    contact_quality: float = EA1_CONTACT_QUALITY_MIN,
    poor_contact: bool = False,
) -> IngestPayload:
    """Build a payload that meets all EA-1 criteria by default."""
    return IngestPayload(
        source="mock",
        bands=BandPowers(alpha=alpha, theta=theta, beta=0.15, delta=0.2, gamma=0.05),
        region=region,
        imu=IMUPayload(pitch_deg=0.0, roll_deg=0.0, motion_rms=motion_rms),
        contact_quality=contact_quality,
        poor_contact=poor_contact,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full eligibility
# ─────────────────────────────────────────────────────────────────────────────

class TestEA1FullEligibility:
    def test_all_criteria_met_returns_eligible(self):
        result = ea1_score(_eligible_payload())
        assert result.eligible is True

    def test_eligible_score_is_one(self):
        result = ea1_score(_eligible_payload())
        assert result.score == pytest.approx(1.0)

    def test_eligible_criteria_met_is_five(self):
        result = ea1_score(_eligible_payload())
        assert result.criteria_met == 5

    def test_eligible_label_text(self):
        result = ea1_score(_eligible_payload())
        assert result.label == "Eligible"

    def test_eligible_region_e_also_works(self):
        result = ea1_score(_eligible_payload(region="E"))
        assert result.eligible is True


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 1: alpha_power
# ─────────────────────────────────────────────────────────────────────────────

class TestCriterionAlpha:
    def test_alpha_exactly_at_threshold_meets_criterion(self):
        result = ea1_score(_eligible_payload(alpha=EA1_ALPHA_THRESHOLD))
        assert result.criteria["alpha_power"]["met"] is True

    def test_alpha_below_threshold_fails_criterion(self):
        result = ea1_score(_eligible_payload(alpha=EA1_ALPHA_THRESHOLD - 0.001))
        assert result.criteria["alpha_power"]["met"] is False
        assert result.eligible is False

    def test_alpha_above_threshold_meets_criterion(self):
        result = ea1_score(_eligible_payload(alpha=EA1_ALPHA_THRESHOLD + 0.1))
        assert result.criteria["alpha_power"]["met"] is True

    def test_alpha_zero_fails_and_decrements_score(self):
        result = ea1_score(_eligible_payload(alpha=0.0))
        assert result.eligible is False
        assert result.criteria_met < 5


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 2: theta_power
# ─────────────────────────────────────────────────────────────────────────────

class TestCriterionTheta:
    def test_theta_exactly_at_threshold_meets_criterion(self):
        result = ea1_score(_eligible_payload(theta=EA1_THETA_THRESHOLD))
        assert result.criteria["theta_power"]["met"] is True

    def test_theta_below_threshold_fails_criterion(self):
        result = ea1_score(_eligible_payload(theta=EA1_THETA_THRESHOLD - 0.001))
        assert result.criteria["theta_power"]["met"] is False
        assert result.eligible is False

    def test_theta_above_threshold_meets_criterion(self):
        result = ea1_score(_eligible_payload(theta=EA1_THETA_THRESHOLD + 0.1))
        assert result.criteria["theta_power"]["met"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 3: s_space region
# ─────────────────────────────────────────────────────────────────────────────

class TestCriterionSSpace:
    @pytest.mark.parametrize("region", ["D", "E"])
    def test_eligible_regions(self, region):
        result = ea1_score(_eligible_payload(region=region))
        assert result.criteria["s_space"]["met"] is True

    @pytest.mark.parametrize("region", ["A", "B", "C", "F", ""])
    def test_ineligible_regions(self, region):
        result = ea1_score(_eligible_payload(region=region))
        assert result.criteria["s_space"]["met"] is False
        assert result.eligible is False

    def test_s_space_region_stored_on_result(self):
        result = ea1_score(_eligible_payload(region="D"))
        assert result.s_space_region == "D"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 4: motion gating
# ─────────────────────────────────────────────────────────────────────────────

class TestCriterionMotion:
    def test_zero_motion_meets_criterion(self):
        result = ea1_score(_eligible_payload(motion_rms=0.0))
        assert result.criteria["motion"]["met"] is True

    def test_motion_just_below_gate_meets_criterion(self):
        result = ea1_score(_eligible_payload(motion_rms=ARTIFACT_ACCEL_RMS_G - 0.001))
        assert result.criteria["motion"]["met"] is True

    def test_motion_at_gate_fails_criterion(self):
        """Gate is strict < not <=, so exactly at threshold should fail."""
        result = ea1_score(_eligible_payload(motion_rms=ARTIFACT_ACCEL_RMS_G))
        assert result.criteria["motion"]["met"] is False
        assert result.eligible is False

    def test_high_motion_fails(self):
        result = ea1_score(_eligible_payload(motion_rms=1.0))
        assert result.criteria["motion"]["met"] is False

    def test_no_imu_payload_defaults_zero_motion(self):
        """When imu is None motion_rms defaults to 0.0, criterion should pass."""
        payload = _eligible_payload()
        payload.imu = None
        result = ea1_score(payload)
        assert result.criteria["motion"]["met"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 5: contact quality
# ─────────────────────────────────────────────────────────────────────────────

class TestCriterionContact:
    def test_good_contact_meets_criterion(self):
        result = ea1_score(_eligible_payload(contact_quality=EA1_CONTACT_QUALITY_MIN))
        assert result.criteria["contact_quality"]["met"] is True

    def test_contact_quality_below_min_fails(self):
        result = ea1_score(
            _eligible_payload(contact_quality=EA1_CONTACT_QUALITY_MIN - 0.01)
        )
        assert result.criteria["contact_quality"]["met"] is False
        assert result.eligible is False

    def test_poor_contact_flag_overrides_good_quality(self):
        result = ea1_score(
            _eligible_payload(
                contact_quality=1.0,
                poor_contact=True,
            )
        )
        assert result.criteria["contact_quality"]["met"] is False
        assert result.eligible is False

    def test_none_contact_quality_without_poor_contact_passes(self):
        """No contact data + poor_contact=False → criterion passes."""
        payload = _eligible_payload()
        payload.contact_quality = None
        payload.poor_contact = False
        result = ea1_score(payload)
        assert result.criteria["contact_quality"]["met"] is True

    def test_none_contact_quality_with_poor_contact_fails(self):
        payload = _eligible_payload()
        payload.contact_quality = None
        payload.poor_contact = True
        result = ea1_score(payload)
        assert result.criteria["contact_quality"]["met"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Overlay mode
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayMode:
    def test_x0_when_no_criteria_met(self):
        result = ea1_score(IngestPayload(
            source="mock",
            bands=BandPowers(),  # all zeros
            region="A",
            imu=IMUPayload(motion_rms=99.0),
            contact_quality=0.0,
            poor_contact=True,
        ))
        assert result.overlay_mode == "X0"

    def test_overlay_mode_x5_for_fully_eligible(self):
        """With no special alchemical stage, all-criteria-met = X5."""
        payload = _eligible_payload()
        payload.alchemical_stage = ""  # no stage override
        result = ea1_score(payload)
        # eligible, no stage override → f"X{criteria_met}" = "X5"
        assert result.overlay_mode == "X5"

    @pytest.mark.parametrize("stage,expected_mode", [
        ("Rubedo", "X4"),
        ("Multiplicatio", "X5"),
        ("Citrinitas", "X3"),
        ("Solutio", "X3"),
    ])
    def test_alchemical_stage_overrides_mode_when_eligible(self, stage, expected_mode):
        payload = _eligible_payload()
        payload.alchemical_stage = stage
        result = ea1_score(payload)
        assert result.eligible is True
        assert result.overlay_mode == expected_mode


# ─────────────────────────────────────────────────────────────────────────────
# Score arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreArithmetic:
    @pytest.mark.parametrize("n_criteria_met", [0, 1, 2, 3, 4, 5])
    def test_raw_score_equals_fraction(self, n_criteria_met):
        """raw_score = criteria_met / 5"""
        # Build a payload so exactly n_criteria_met criteria are satisfied.
        # Easiest: vary alpha (crit1) and theta (crit2) independently.
        fail_alpha = n_criteria_met < 1
        fail_theta = n_criteria_met < 2
        fail_sspace = n_criteria_met < 3
        fail_motion = n_criteria_met < 4
        fail_contact = n_criteria_met < 5

        payload = IngestPayload(
            source="mock",
            bands=BandPowers(
                alpha=0.0 if fail_alpha else EA1_ALPHA_THRESHOLD,
                theta=0.0 if fail_theta else EA1_THETA_THRESHOLD,
                beta=0.1, delta=0.1, gamma=0.0,
            ),
            region="A" if fail_sspace else "D",
            imu=IMUPayload(
                pitch_deg=0.0, roll_deg=0.0,
                motion_rms=ARTIFACT_ACCEL_RMS_G if fail_motion else 0.0,
            ),
            contact_quality=None if fail_contact else EA1_CONTACT_QUALITY_MIN,
            poor_contact=fail_contact,
        )
        result = ea1_score(payload)
        assert result.criteria_met == n_criteria_met
        assert result.score == pytest.approx(n_criteria_met / 5.0)
        assert result.criteria_total == 5
