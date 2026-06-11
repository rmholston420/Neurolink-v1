"""Unit tests for fatigue.py."""

from __future__ import annotations

from neurolink.fatigue import FatigueDetector


def test_fatigue_detector_zero_when_empty():
    fd = FatigueDetector()
    assert fd.score == 0.0


def test_fatigue_detector_zero_after_one_sample():
    fd = FatigueDetector()
    fd.update(theta=0.2, alpha=0.3)
    assert fd.score == 0.0  # needs at least 2 samples


def test_fatigue_detector_high_when_theta_dominates():
    """Fatigue score should be > 0.8 after 30 samples with theta/alpha = 4.0."""
    fd = FatigueDetector(window=30)
    for _ in range(30):
        fd.update(theta=0.4, alpha=0.1)  # ratio = 4.0
    assert fd.score > 0.8, f"Expected score > 0.8, got {fd.score}"


def test_fatigue_detector_low_when_alpha_dominates():
    fd = FatigueDetector(window=30)
    for _ in range(30):
        fd.update(theta=0.05, alpha=0.40)  # low ratio
    assert fd.score < 0.3


def test_fatigue_detector_reset():
    fd = FatigueDetector()
    for _ in range(5):
        fd.update(0.4, 0.1)
    fd.reset()
    assert fd.sample_count == 0
    assert fd.score == 0.0


def test_fatigue_detector_window_limits():
    """Window should limit to last N samples."""
    fd = FatigueDetector(window=5)
    for _ in range(3):
        fd.update(0.4, 0.1)
    for _ in range(10):
        fd.update(0.05, 0.4)  # low ratio overwhelms
    # Score should reflect recent low-ratio samples
    assert fd.score < 0.5
