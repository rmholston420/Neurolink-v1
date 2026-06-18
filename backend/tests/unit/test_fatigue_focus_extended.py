"""Extended unit tests for FatigueDetector and focus_state utilities.

Covers boundary conditions, rolling window eviction, legacy alias
resolution, and the module-level is_blocking() gate.
"""

from __future__ import annotations

import pytest

from neurolink.fatigue import FatigueDetector, _THETA_DOMINANCE_THRESHOLD, _WINDOW_SIZE
from neurolink.focus_state import (
    FocusState,
    classify_focus,
    compute_focus_score,
    is_blocking,
    set_current_focus_score,
    _HIGH_FOCUS_THRESHOLD,
    _LOW_FOCUS_THRESHOLD,
    _MODERATE_FOCUS_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# FatigueDetector
# ─────────────────────────────────────────────────────────────────────────────

class TestFatigueDetector:
    def test_initial_score_is_zero(self):
        fd = FatigueDetector()
        assert fd.score == 0.0

    def test_sample_count_starts_empty(self):
        fd = FatigueDetector()
        assert fd.sample_count == 0

    def test_single_update_returns_zero_score(self):
        """< 2 samples -> score 0.0"""
        fd = FatigueDetector()
        score = fd.update(theta=0.3, alpha=0.2)
        assert score == 0.0

    def test_two_updates_returns_positive_score(self):
        fd = FatigueDetector()
        fd.update(theta=0.3, alpha=0.2)
        score = fd.update(theta=0.3, alpha=0.2)
        assert score > 0.0

    def test_high_theta_low_alpha_saturates_at_one(self):
        """Ratio well above threshold -> score caps at 1.0"""
        fd = FatigueDetector()
        for _ in range(10):
            fd.update(theta=0.9, alpha=0.01)
        assert fd.score == pytest.approx(1.0)

    def test_zero_alpha_uses_epsilon_guard(self):
        """Alpha = 0 must not raise ZeroDivisionError"""
        fd = FatigueDetector()
        for _ in range(5):
            fd.update(theta=0.5, alpha=0.0)
        assert 0.0 <= fd.score <= 1.0

    def test_zero_theta_zero_alpha_yields_low_score(self):
        fd = FatigueDetector()
        for _ in range(5):
            fd.update(theta=0.0, alpha=0.0)
        # ratio = 0 / epsilon ~= 0 -> score ~= 0
        assert fd.score < 0.01

    def test_reset_clears_samples(self):
        fd = FatigueDetector()
        for _ in range(10):
            fd.update(theta=0.5, alpha=0.2)
        fd.reset()
        assert fd.sample_count == 0
        assert fd.score == 0.0

    def test_rolling_window_evicts_old_samples(self):
        """Window size is respected; old high-fatigue samples age out."""
        fd = FatigueDetector(window=5)
        # Fill with high-fatigue samples
        for _ in range(5):
            fd.update(theta=0.9, alpha=0.01)
        high_score = fd.score
        # Replace with low-fatigue samples
        for _ in range(5):
            fd.update(theta=0.0, alpha=0.9)
        assert fd.score < high_score

    def test_window_does_not_exceed_maxlen(self):
        fd = FatigueDetector(window=10)
        for _i in range(50):  # B007: renamed i -> _i
            fd.update(theta=0.3, alpha=0.3)
        assert fd.sample_count == 10

    def test_score_property_matches_update_return(self):
        fd = FatigueDetector()
        fd.update(theta=0.3, alpha=0.2)
        returned = fd.update(theta=0.3, alpha=0.2)
        assert returned == fd.score

    def test_score_normalised_in_range(self):
        fd = FatigueDetector()
        for _ in range(30):  # F841: removed unused `score =`
            fd.update(theta=0.4, alpha=0.3)
        assert 0.0 <= fd.score <= 1.0

    @pytest.mark.parametrize("theta,alpha", [
        (0.1, 0.5),  # low fatigue
        (0.3, 0.3),  # neutral
        (0.6, 0.2),  # moderate fatigue
        (0.9, 0.05), # high fatigue
    ])
    def test_parametric_score_bounds(self, theta, alpha):
        fd = FatigueDetector()
        for _ in range(10):
            fd.update(theta=theta, alpha=alpha)
        assert 0.0 <= fd.score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_focus_score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFocusScore:
    def test_returns_float_in_unit_interval(self):
        score = compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=0.3)
        assert 0.0 <= score <= 1.0

    def test_high_alpha_high_beta_yields_high_score(self):
        score = compute_focus_score(alpha=0.9, beta=0.5, baseline_alpha=0.3)
        assert score > 0.7

    def test_zero_alpha_zero_beta_yields_low_score(self):
        score = compute_focus_score(alpha=0.0, beta=0.0, baseline_alpha=0.3)
        assert score < 0.1

    def test_zero_baseline_falls_back_gracefully(self):
        """baseline_alpha <= 0 must not raise and must use fallback 0.3"""
        score = compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=0.0)
        assert 0.0 <= score <= 1.0

    def test_negative_baseline_falls_back_gracefully(self):
        score = compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=-0.5)
        assert 0.0 <= score <= 1.0

    def test_alpha_capped_at_1_5x_baseline(self):
        """alpha much larger than 1.5x baseline should not exceed 1.0"""
        score = compute_focus_score(alpha=10.0, beta=1.0, baseline_alpha=0.3)
        assert score <= 1.0

    def test_legacy_bands_alpha_overrides_alpha(self):
        score_new = compute_focus_score(alpha=0.5, beta=0.2, baseline_alpha=0.3)
        score_legacy = compute_focus_score(
            alpha=0.0, beta=0.2, baseline_alpha=0.3, bands_alpha=0.5
        )
        assert score_new == pytest.approx(score_legacy)

    def test_legacy_bands_beta_overrides_beta(self):
        score_new = compute_focus_score(alpha=0.4, beta=0.3, baseline_alpha=0.3)
        score_legacy = compute_focus_score(
            alpha=0.4, beta=0.0, baseline_alpha=0.3, bands_beta=0.3
        )
        assert score_new == pytest.approx(score_legacy)

    def test_beta_capped_at_0_5(self):
        """beta > 0.5 should not exceed the component's maximum contribution"""
        score_capped = compute_focus_score(alpha=0.3, beta=0.5, baseline_alpha=0.3)
        score_excess = compute_focus_score(alpha=0.3, beta=0.99, baseline_alpha=0.3)
        assert score_capped == pytest.approx(score_excess)

    @pytest.mark.parametrize("alpha,beta,baseline", [
        (0.1, 0.1, 0.3),
        (0.3, 0.3, 0.3),
        (0.6, 0.4, 0.3),
        (0.9, 0.5, 0.6),
    ])
    def test_parametric_bounds(self, alpha, beta, baseline):
        score = compute_focus_score(alpha=alpha, beta=beta, baseline_alpha=baseline)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# classify_focus
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyFocus:
    def test_high_focus_at_threshold(self):
        assert classify_focus(_HIGH_FOCUS_THRESHOLD) == FocusState.HIGH_FOCUS

    def test_high_focus_above_threshold(self):
        assert classify_focus(1.0) == FocusState.HIGH_FOCUS

    def test_moderate_focus(self):
        assert classify_focus(_MODERATE_FOCUS_THRESHOLD) == FocusState.MODERATE_FOCUS

    def test_low_focus(self):
        assert classify_focus(_LOW_FOCUS_THRESHOLD) == FocusState.LOW_FOCUS

    def test_distracted_below_low_threshold(self):
        assert classify_focus(_LOW_FOCUS_THRESHOLD - 0.01) == FocusState.DISTRACTED

    def test_distracted_at_zero(self):
        assert classify_focus(0.0) == FocusState.DISTRACTED

    @pytest.mark.parametrize("score,expected", [
        (0.00, FocusState.DISTRACTED),
        (0.24, FocusState.DISTRACTED),
        (0.25, FocusState.LOW_FOCUS),
        (0.49, FocusState.LOW_FOCUS),
        (0.50, FocusState.MODERATE_FOCUS),
        (0.74, FocusState.MODERATE_FOCUS),
        (0.75, FocusState.HIGH_FOCUS),
        (1.00, FocusState.HIGH_FOCUS),
    ])
    def test_boundary_transitions(self, score, expected):
        assert classify_focus(score) == expected


# ─────────────────────────────────────────────────────────────────────────────
# set_current_focus_score / is_blocking
# ─────────────────────────────────────────────────────────────────────────────

class TestFocusGate:
    def setup_method(self):
        """Reset to a neutral state before each test."""
        set_current_focus_score(0.5)

    def test_blocking_below_low_threshold(self):
        set_current_focus_score(0.0)
        assert is_blocking() is True

    def test_not_blocking_above_threshold(self):
        set_current_focus_score(0.5)
        assert is_blocking() is False

    def test_not_blocking_exactly_at_threshold(self):
        set_current_focus_score(_LOW_FOCUS_THRESHOLD)
        assert is_blocking() is False

    def test_blocking_just_below_threshold(self):
        set_current_focus_score(_LOW_FOCUS_THRESHOLD - 0.001)
        assert is_blocking() is True

    def test_set_score_persists(self):
        set_current_focus_score(0.99)
        assert is_blocking() is False
        set_current_focus_score(0.01)
        assert is_blocking() is True
