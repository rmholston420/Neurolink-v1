"""Unit tests for focus and fatigue state logic."""

from __future__ import annotations

import pytest

from neurolink.fatigue import FatigueDetector
from neurolink.focus_state import classify_focus, compute_focus_score


class TestComputeFocusScore:
    def test_high_beta_low_theta_gives_high_focus(self):
        # signature: compute_focus_score(bands_alpha, bands_beta, baseline_alpha)
        score = compute_focus_score(0.3, 0.6, 0.3)
        assert score > 0.5

    def test_high_theta_low_beta_gives_low_focus(self):
        score = compute_focus_score(0.1, 0.05, 0.3)
        assert score < 0.5

    def test_returns_float(self):
        score = compute_focus_score(0.3, 0.3, 0.3)
        assert isinstance(score, float)

    def test_score_in_unit_interval(self):
        for alpha in [0.0, 0.2, 0.5, 1.0]:
            for beta in [0.0, 0.2, 0.5, 1.0]:
                score = compute_focus_score(alpha, beta, 0.3)
                assert 0.0 <= score <= 1.0, (
                    f"score={score} out of [0,1] for alpha={alpha} beta={beta}"
                )


class TestClassifyFocus:
    def test_high_score_returns_focused(self):
        state = classify_focus(0.85)
        assert hasattr(state, "value")

    def test_returns_has_value_attribute(self):
        state = classify_focus(0.5)
        assert hasattr(state, "value")


class TestFatigueDetector:
    def test_initial_score_is_zero(self):
        fd = FatigueDetector()
        assert isinstance(fd, FatigueDetector)

    def test_update_returns_float_in_unit_interval(self):
        fd = FatigueDetector()
        for _ in range(10):
            score = fd.update(theta=0.4, alpha=0.3)
            assert 0.0 <= score <= 1.0, f"fatigue score {score} out of [0,1]"

    def test_high_theta_low_alpha_increases_fatigue(self):
        fd = FatigueDetector()
        scores_fatigued = [fd.update(theta=0.7, alpha=0.05) for _ in range(20)]
        fd2 = FatigueDetector()
        scores_alert = [fd2.update(theta=0.05, alpha=0.6) for _ in range(20)]
        assert scores_fatigued[-1] > scores_alert[-1]

    def test_reset_clears_accumulator(self):
        fd = FatigueDetector()
        for _ in range(10):
            fd.update(theta=0.8, alpha=0.05)
        fd.reset()
        score_after_reset = fd.update(theta=0.8, alpha=0.05)
        assert score_after_reset < 0.5
