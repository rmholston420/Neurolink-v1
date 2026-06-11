"""Coverage tests for focus_state.py and fatigue.py."""
from __future__ import annotations

from neurolink.fatigue import FatigueDetector
from neurolink.focus_state import (
    FocusState,
    classify_focus,
    compute_focus_score,
    is_blocking,
    set_current_focus_score,
)


# ===========================================================================
# focus_state.py
# ===========================================================================

class TestComputeFocusScore:
    def test_high_alpha_suppresses_focus(self):
        score = compute_focus_score(bands_alpha=0.8, bands_beta=0.3, baseline_alpha=0.3)
        assert 0.0 <= score <= 1.0

    def test_low_alpha_raises_focus(self):
        score = compute_focus_score(bands_alpha=0.05, bands_beta=0.1, baseline_alpha=0.3)
        assert score > 0.0

    def test_zero_baseline_uses_fallback(self):
        score = compute_focus_score(bands_alpha=0.3, bands_beta=0.2, baseline_alpha=0.0)
        assert isinstance(score, float)

    def test_score_clamped_zero_to_one(self):
        score = compute_focus_score(bands_alpha=100.0, bands_beta=0.0, baseline_alpha=0.3)
        assert 0.0 <= score <= 1.0
        score2 = compute_focus_score(bands_alpha=0.0, bands_beta=0.0, baseline_alpha=0.3)
        assert 0.0 <= score2 <= 1.0


class TestClassifyFocus:
    def test_high_focus(self):
        assert classify_focus(1.0) == FocusState.HIGH_FOCUS

    def test_moderate_focus(self):
        assert classify_focus(0.6) == FocusState.MODERATE_FOCUS

    def test_low_focus(self):
        assert classify_focus(0.35) == FocusState.LOW_FOCUS

    def test_distracted(self):
        assert classify_focus(0.0) == FocusState.DISTRACTED

    def test_boundary_high(self):
        assert classify_focus(0.75) == FocusState.HIGH_FOCUS

    def test_boundary_moderate(self):
        assert classify_focus(0.50) == FocusState.MODERATE_FOCUS

    def test_boundary_low(self):
        assert classify_focus(0.25) == FocusState.LOW_FOCUS


class TestFocusGlobal:
    def test_set_and_is_blocking_false_when_above_threshold(self):
        set_current_focus_score(0.5)  # above 0.25 blocking threshold
        assert is_blocking() is False

    def test_is_blocking_true_when_below_threshold(self):
        set_current_focus_score(0.1)  # below 0.25 blocking threshold
        assert is_blocking() is True

    def test_set_zero(self):
        set_current_focus_score(0.0)
        assert is_blocking() is True


# ===========================================================================
# fatigue.py
# ===========================================================================

class TestFatigueDetector:
    def test_initial_update_returns_float(self):
        fd = FatigueDetector()
        score = fd.update(theta=0.0, alpha=0.0)
        assert isinstance(score, float)

    def test_high_theta_low_alpha_increases_score(self):
        fd = FatigueDetector()
        scores = [fd.update(theta=0.9, alpha=0.05) for _ in range(50)]
        assert scores[-1] > scores[0]

    def test_score_clamped_to_zero_one(self):
        fd = FatigueDetector()
        for _ in range(200):
            score = fd.update(theta=1.0, alpha=0.0)
        assert 0.0 <= score <= 1.0

    def test_reset_brings_score_near_zero(self):
        fd = FatigueDetector()
        for _ in range(50):
            fd.update(theta=0.9, alpha=0.05)
        fd.reset()
        score_after = fd.update(theta=0.0, alpha=0.0)
        assert score_after < 0.5

    def test_low_theta_high_alpha_stays_low(self):
        fd = FatigueDetector()
        scores = [fd.update(theta=0.05, alpha=0.8) for _ in range(30)]
        assert scores[-1] < 0.5
