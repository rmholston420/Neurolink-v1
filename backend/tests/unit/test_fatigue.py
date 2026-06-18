"""Unit tests for fatigue.FatigueDetector."""

from __future__ import annotations

from neurolink.fatigue import FatigueDetector


class TestFatigueDetector:
    def test_initial_score_zero_or_float(self):
        fd = FatigueDetector()
        score = fd.update(theta=0.0, alpha=0.0)
        assert isinstance(score, float)

    def test_score_increases_with_high_theta(self):
        """Sustained high theta (drowsiness marker) should accumulate fatigue."""
        fd = FatigueDetector()
        scores = [fd.update(theta=0.9, alpha=0.05) for _ in range(50)]
        assert scores[-1] >= scores[0]

    def test_score_bounded(self):
        fd = FatigueDetector()
        for _ in range(200):
            score = fd.update(theta=1.0, alpha=0.0)
        assert 0.0 <= score <= 1.0

    def test_reset_clears_accumulation(self):
        fd = FatigueDetector()
        for _ in range(50):
            fd.update(theta=0.9, alpha=0.05)
        score_before = fd.update(theta=0.9, alpha=0.05)
        fd.reset()
        score_after = fd.update(theta=0.0, alpha=0.0)
        assert score_after <= score_before

    def test_low_theta_high_alpha_stays_low(self):
        """Alert, focused state should not accumulate fatigue rapidly."""
        fd = FatigueDetector()
        scores = [fd.update(theta=0.05, alpha=0.6) for _ in range(30)]
        # After 30 frames, fatigue should be modest (< 0.5)
        assert scores[-1] < 0.5
