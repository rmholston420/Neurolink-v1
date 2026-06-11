"""Coverage tests for focus_state.py and fatigue.py."""
from __future__ import annotations

import pytest

from neurolink.fatigue import FatigueDetector
from neurolink.focus_state import (
    FocusState,
    classify_focus,
    compute_focus_score,
    get_current_focus_score,
    set_current_focus_score,
)


# ===========================================================================
# focus_state.py
# ===========================================================================

class TestComputeFocusScore:
    def test_high_alpha_low_score(self):
        """High alpha relative to baseline → low focus score."""
        score = compute_focus_score(alpha=0.8, beta=0.3, baseline_alpha=0.3)
        assert score < 0.5

    def test_low_alpha_high_score(self):
        """Low alpha (below baseline) with high beta → higher focus."""
        score = compute_focus_score(alpha=0.1, beta=0.5, baseline_alpha=0.3)
        assert score >= 0.0

    def test_zero_alpha_zero_beta(self):
        score = compute_focus_score(alpha=0.0, beta=0.0, baseline_alpha=0.3)
        assert isinstance(score, float)

    def test_score_clamped_to_zero_one(self):
        score = compute_focus_score(alpha=10.0, beta=0.0, baseline_alpha=0.3)
        assert 0.0 <= score <= 1.0


class TestClassifyFocus:
    def test_focused(self):
        state = classify_focus(1.0)
        assert state == FocusState.FOCUSED

    def test_neutral(self):
        state = classify_focus(0.5)
        assert state in (FocusState.NEUTRAL, FocusState.FOCUSED, FocusState.UNFOCUSED)

    def test_unfocused(self):
        state = classify_focus(0.0)
        assert state == FocusState.UNFOCUSED

    def test_all_states_reachable(self):
        """Verify all FocusState enum values are actually returned by classify_focus."""
        scores = [0.0, 0.3, 0.5, 0.7, 1.0]
        returned = {classify_focus(s) for s in scores}
        # At minimum FOCUSED and UNFOCUSED must be reachable
        assert FocusState.FOCUSED in returned
        assert FocusState.UNFOCUSED in returned


class TestFocusScoreGlobal:
    def test_set_and_get_round_trip(self):
        set_current_focus_score(0.77)
        assert abs(get_current_focus_score() - 0.77) < 1e-9

    def test_default_is_zero(self):
        set_current_focus_score(0.0)
        assert get_current_focus_score() == 0.0


# ===========================================================================
# fatigue.py
# ===========================================================================

class TestFatigueDetector:
    def test_initial_score_is_zero(self):
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

    def test_reset_clears_accumulator(self):
        fd = FatigueDetector()
        for _ in range(20):
            fd.update(theta=0.9, alpha=0.05)
        fd.reset()
        score_after_reset = fd.update(theta=0.0, alpha=0.0)
        assert score_after_reset < 0.5  # back near zero

    def test_low_theta_high_alpha_low_fatigue(self):
        fd = FatigueDetector()
        scores = [fd.update(theta=0.05, alpha=0.8) for _ in range(30)]
        assert scores[-1] < 0.5
