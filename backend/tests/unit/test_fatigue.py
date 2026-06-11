"""Unit tests for fatigue.py."""
from __future__ import annotations

import pytest

from neurolink.fatigue import FatigueDetector


def test_fatigue_detector_zero_when_empty():
    fd = FatigueDetector()
    assert fd.score() == 0.0


def test_fatigue_detector_returns_float():
    fd = FatigueDetector()
    val = fd.update(0.20, 0.30)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_fatigue_detector_high_when_theta_dominates():
    """After 30 samples with theta/alpha = 4.0, score should be > 0.8."""
    fd = FatigueDetector(window=30)
    for _ in range(30):
        fd.update(theta=0.40, alpha=0.10)
    s = fd.score()
    assert s > 0.8, f"Expected > 0.8, got {s}"


def test_fatigue_detector_low_when_alpha_dominates():
    fd = FatigueDetector(window=30)
    for _ in range(30):
        fd.update(theta=0.05, alpha=0.40)
    s = fd.score()
    assert s < 0.3, f"Expected < 0.3, got {s}"


def test_fatigue_detector_reset_clears_state():
    fd = FatigueDetector()
    fd.update(0.40, 0.10)
    fd.reset()
    assert fd.score() == 0.0
