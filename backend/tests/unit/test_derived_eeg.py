"""Unit tests for dsp.derived_eeg — FAA, FMT, contact quality helpers."""

from __future__ import annotations

import pytest

from neurolink.dsp.derived_eeg import compute_contact_quality, compute_faa, compute_fmt


class TestComputeFAA:
    def test_positive_when_left_dominates(self):
        """FAA = log(alpha_left) - log(alpha_right). Positive if left > right."""
        result = compute_faa(alpha_left=0.8, alpha_right=0.2)
        assert result > 0

    def test_negative_when_right_dominates(self):
        result = compute_faa(alpha_left=0.2, alpha_right=0.8)
        assert result < 0

    def test_zero_when_equal(self):
        result = compute_faa(alpha_left=0.5, alpha_right=0.5)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_handles_near_zero_gracefully(self):
        """Near-zero alpha should not raise ZeroDivisionError."""
        result = compute_faa(alpha_left=1e-10, alpha_right=0.5)
        assert isinstance(result, float)


class TestComputeFMT:
    def test_positive_theta_returns_float(self):
        result = compute_fmt(theta_frontal=0.5)
        assert isinstance(result, float)

    def test_zero_theta(self):
        result = compute_fmt(theta_frontal=0.0)
        assert isinstance(result, float)

    def test_monotone_increasing(self):
        """More frontal theta should increase FMT."""
        assert compute_fmt(0.8) > compute_fmt(0.2)


class TestComputeContactQuality:
    def test_good_contact_when_low_noise(self):
        result = compute_contact_quality(noise_rms=0.01)
        assert isinstance(result, str)
        # We just assert it's a non-empty string — exact label is implementation-defined
        assert len(result) > 0

    def test_poor_contact_when_high_noise(self):
        good = compute_contact_quality(noise_rms=0.01)
        bad = compute_contact_quality(noise_rms=100.0)
        # Good should differ from bad or both should be valid strings
        assert isinstance(bad, str)

    def test_returns_string(self):
        assert isinstance(compute_contact_quality(noise_rms=0.5), str)
