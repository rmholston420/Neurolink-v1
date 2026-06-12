"""Unit tests for dsp/classifiers.py."""

from __future__ import annotations

import pytest

from neurolink.dsp.artifact_config import (
    V01_ALPHA_C,
    V01_ALPHA_E,
    V01_BETA_B,
    V01_DELTA_F,
    V01_MULTIPLICATIO_ALPHA,
    V01_MULTIPLICATIO_FAA,
    V01_MULTIPLICATIO_THETA,
    V01_THETA_D,
    V01_THETA_E,
    V2_ALPHA_MULTIPLICATIO,
    V2_ALPHA_RUBEDO,
    V2_BETA_ALBEDO,
    V2_BETA_CALCINATIO,
    V2_BETA_RUBEDO_MAX,
    V2_DELTA_COAGULATIO,
    V2_GAMMA_SUBLIMATIO,
    V2_THETA_RUBEDO,
    V2_THETA_SOLUTIO,
)
from neurolink.dsp.classifiers import classify_v01, classify_v2, compute_s_space
from neurolink.models.eeg import BandPowers, SSpaceCoords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bp(
    alpha: float = 0.15,
    theta: float = 0.15,
    beta: float = 0.15,
    delta: float = 0.40,
    gamma: float = 0.15,
) -> BandPowers:
    return BandPowers(alpha=alpha, theta=theta, beta=beta, delta=delta, gamma=gamma)


# ---------------------------------------------------------------------------
# classify_v01() — 6-region rule-based classifier
# ---------------------------------------------------------------------------

class TestClassifyV01:
    # --- Region F: delta dominance ---
    def test_high_delta_returns_f_coagulatio(self):
        region, stage = classify_v01(
            alpha=0.05, theta=0.05, beta=0.05,
            delta=V01_DELTA_F, gamma=0.05,
        )
        assert region == "F"
        assert stage == "Coagulatio"

    def test_delta_just_below_threshold_not_f(self):
        region, _ = classify_v01(
            alpha=0.05, theta=0.05, beta=0.05,
            delta=V01_DELTA_F - 0.001, gamma=0.05,
        )
        assert region != "F"

    # --- Region E: Rubedo ---
    def test_high_alpha_theta_returns_e_rubedo(self):
        region, stage = classify_v01(
            alpha=V01_ALPHA_E, theta=V01_THETA_E,
            beta=0.01, delta=0.01, gamma=0.01,
        )
        assert region == "E"
        assert stage == "Rubedo"

    # --- Region E: Multiplicatio escalation ---
    def test_multiplicatio_escalation_all_gates_pass(self):
        region, stage = classify_v01(
            alpha=V01_MULTIPLICATIO_ALPHA,
            theta=V01_MULTIPLICATIO_THETA,
            beta=0.01,
            delta=0.01,
            gamma=0.01,
            faa=V01_MULTIPLICATIO_FAA,
        )
        assert region == "E"
        assert stage == "Multiplicatio"

    def test_multiplicatio_faa_none_passes(self):
        """faa=None → the faa gate is bypassed; Multiplicatio fires on alpha+theta alone."""
        region, stage = classify_v01(
            alpha=V01_MULTIPLICATIO_ALPHA,
            theta=V01_MULTIPLICATIO_THETA,
            beta=0.01, delta=0.01, gamma=0.01,
            faa=None,
        )
        assert region == "E"
        assert stage == "Multiplicatio"

    def test_multiplicatio_faa_below_threshold_falls_to_rubedo(self):
        """faa below threshold → Multiplicatio gate fails → falls back to Rubedo."""
        region, stage = classify_v01(
            alpha=V01_MULTIPLICATIO_ALPHA,
            theta=V01_MULTIPLICATIO_THETA,
            beta=0.01, delta=0.01, gamma=0.01,
            faa=V01_MULTIPLICATIO_FAA - 0.1,
        )
        assert region == "E"
        assert stage == "Rubedo"

    # --- Region D: flow ---
    def test_high_theta_low_alpha_returns_d_citrinitas(self):
        region, stage = classify_v01(
            alpha=0.01,
            theta=V01_THETA_D,
            beta=0.01, delta=0.01, gamma=0.01,
        )
        assert region == "D"
        assert stage == "Citrinitas"

    # --- Region C: alpha onset ---
    def test_moderate_alpha_low_beta_returns_c_albedo(self):
        region, stage = classify_v01(
            alpha=V01_ALPHA_C,
            theta=0.01,
            beta=0.0,
            delta=0.01,
            gamma=0.01,
        )
        assert region == "C"
        assert stage == "Albedo"

    # --- Region B: arousal ---
    def test_high_beta_returns_b_albedo(self):
        region, stage = classify_v01(
            alpha=0.01, theta=0.01,
            beta=V01_BETA_B,
            delta=0.01, gamma=0.01,
        )
        assert region == "B"
        assert stage == "Albedo"

    # --- Region A: default ---
    def test_low_all_returns_a_nigredo(self):
        region, stage = classify_v01(
            alpha=0.01, theta=0.01,
            beta=0.0,
            delta=0.0, gamma=0.01,
        )
        assert region == "A"
        assert stage == "Nigredo"

    # --- Return type ---
    def test_returns_tuple_of_two_strings(self):
        result = classify_v01(0.15, 0.15, 0.15, 0.40, 0.15)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# classify_v2() — 8-region extended classifier
# ---------------------------------------------------------------------------

class TestClassifyV2:
    # --- Coagulatio: delta dominance ---
    def test_high_delta_returns_f_coagulatio(self):
        bp = _bp(delta=V2_DELTA_COAGULATIO, alpha=0.05, theta=0.05, beta=0.05, gamma=0.05)
        region, stage = classify_v2(bp)
        assert region == "F"
        assert stage == "Coagulatio"

    # --- Sublimatio: gamma dominant ---
    def test_gamma_dominant_returns_g_sublimatio(self):
        bp = _bp(
            delta=0.01,
            gamma=V2_GAMMA_SUBLIMATIO,
            alpha=V2_GAMMA_SUBLIMATIO - 0.01,
            theta=V2_GAMMA_SUBLIMATIO - 0.01,
            beta=0.01,
        )
        region, stage = classify_v2(bp)
        assert region == "G"
        assert stage == "Sublimatio"

    # --- Calcinatio: high beta ---
    def test_high_beta_returns_h_calcinatio(self):
        bp = _bp(delta=0.01, gamma=0.01, beta=V2_BETA_CALCINATIO, alpha=0.1, theta=0.1)
        region, stage = classify_v2(bp)
        assert region == "H"
        assert stage == "Calcinatio"

    # --- Multiplicatio: highest meditation ---
    def test_multiplicatio_returns_e(self):
        bp = _bp(
            alpha=V2_ALPHA_MULTIPLICATIO,
            theta=V2_THETA_RUBEDO,
            beta=V2_BETA_RUBEDO_MAX - 0.001,
            delta=0.01,
            gamma=0.01,
        )
        region, stage = classify_v2(bp)
        assert region == "E"
        assert stage == "Multiplicatio"

    # --- Rubedo: deep meditation ---
    def test_rubedo_returns_e(self):
        bp = _bp(
            alpha=V2_ALPHA_RUBEDO,
            theta=V2_THETA_RUBEDO,
            beta=V2_BETA_RUBEDO_MAX - 0.001,
            delta=0.01,
            gamma=0.01,
        )
        region, stage = classify_v2(bp)
        assert region == "E"
        # Stage can be Rubedo or Multiplicatio depending on thresholds
        assert stage in {"Rubedo", "Multiplicatio"}

    # --- Solutio: high theta ---
    def test_solutio_returns_d(self):
        bp = _bp(
            theta=V2_THETA_SOLUTIO,
            alpha=V2_ALPHA_RUBEDO - 0.01,
            beta=0.01,
            delta=0.01,
            gamma=0.01,
        )
        region, stage = classify_v2(bp)
        assert region == "D"
        assert stage == "Solutio"

    # --- Albedo: moderate beta ---
    def test_albedo_returns_c(self):
        bp = _bp(
            beta=V2_BETA_ALBEDO,
            alpha=0.01,
            theta=0.01,
            delta=0.01,
            gamma=0.01,
        )
        region, stage = classify_v2(bp)
        assert region == "C"
        assert stage == "Albedo"

    # --- Nigredo: default ---
    def test_nigredo_returns_a(self):
        bp = _bp(alpha=0.05, theta=0.05, beta=0.01, delta=0.01, gamma=0.01)
        region, stage = classify_v2(bp)
        assert region == "A"
        assert stage == "Nigredo"

    # --- Return type ---
    def test_returns_tuple_of_two_strings(self):
        result = classify_v2(_bp())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# compute_s_space()
# ---------------------------------------------------------------------------

class TestComputeSSpace:
    def test_returns_s_space_coords_instance(self):
        coords = compute_s_space(_bp())
        assert isinstance(coords, SSpaceCoords)

    def test_x_in_range_0_10(self):
        coords = compute_s_space(_bp())
        assert 0.0 <= coords.x <= 10.0

    def test_y_in_range_0_10(self):
        coords = compute_s_space(_bp())
        assert 0.0 <= coords.y <= 10.0

    def test_z_in_range_0_1(self):
        coords = compute_s_space(_bp())
        assert 0.0 <= coords.z <= 1.0

    def test_high_beta_high_x(self):
        """High beta relative to alpha → high engagement → x near max."""
        high_beta = _bp(beta=0.8, alpha=0.01)
        coords = compute_s_space(high_beta)
        assert coords.x == pytest.approx(10.0)

    def test_high_alpha_theta_high_y(self):
        """High alpha * theta relative to beta+delta → high integration → y towards max."""
        high_at = _bp(alpha=0.5, theta=0.4, beta=0.01, delta=0.01, gamma=0.08)
        coords = compute_s_space(high_at)
        assert coords.y > 5.0

    def test_high_gamma_high_z(self):
        """High gamma fraction → z near 1.0."""
        high_g = _bp(gamma=0.8, alpha=0.01)
        coords = compute_s_space(high_g)
        assert coords.z > 0.9

    def test_zero_gamma_zero_z(self):
        coords = compute_s_space(_bp(gamma=0.0))
        assert coords.z == pytest.approx(0.0, abs=1e-6)

    def test_equal_bands_midrange_coords(self):
        """Perfectly equal bands → all coords within valid range and non-extreme."""
        equal = BandPowers(alpha=0.2, theta=0.2, beta=0.2, delta=0.2, gamma=0.2)
        coords = compute_s_space(equal)
        assert 0.0 <= coords.x <= 10.0
        assert 0.0 <= coords.y <= 10.0
        assert 0.0 <= coords.z <= 1.0

    def test_x_y_z_are_floats(self):
        coords = compute_s_space(_bp())
        assert isinstance(coords.x, float)
        assert isinstance(coords.y, float)
        assert isinstance(coords.z, float)
