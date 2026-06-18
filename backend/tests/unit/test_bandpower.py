"""Unit tests for dsp.bandpower — band-power computation."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bandpower import compute_band_powers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, duration_s: float = 1.0, fs: int = 256) -> list[float]:
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t)).tolist()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestComputeBandPowers:
    def test_returns_five_bands(self, eeg_buffer_256hz):
        result = compute_band_powers(eeg_buffer_256hz, fs=256)
        assert hasattr(result, "alpha")
        assert hasattr(result, "theta")
        assert hasattr(result, "beta")
        assert hasattr(result, "delta")
        assert hasattr(result, "gamma")

    def test_values_are_non_negative(self, eeg_buffer_256hz):
        result = compute_band_powers(eeg_buffer_256hz, fs=256)
        for band in (result.alpha, result.theta, result.beta, result.delta, result.gamma):
            assert band >= 0.0

    def test_alpha_dominant_when_10hz_sine(self):
        """A 10 Hz sine should produce dominant alpha power."""
        channel = _sine(10.0)
        result = compute_band_powers([channel], fs=256)
        assert result.alpha > result.theta
        assert result.alpha > result.beta
        assert result.alpha > result.delta

    def test_theta_dominant_when_6hz_sine(self):
        """A 6 Hz sine should produce dominant theta power."""
        channel = _sine(6.0)
        result = compute_band_powers([channel], fs=256)
        assert result.theta > result.beta
        assert result.theta > result.delta

    def test_beta_dominant_when_20hz_sine(self):
        """A 20 Hz sine should produce dominant beta power."""
        channel = _sine(20.0)
        result = compute_band_powers([channel], fs=256)
        assert result.beta >= result.alpha

    def test_multi_channel_mean(self):
        """Multi-channel input is averaged."""
        ch_alpha = _sine(10.0)
        ch_theta = _sine(6.0)
        result = compute_band_powers([ch_alpha, ch_theta], fs=256)
        assert result.alpha > 0
        assert result.theta > 0

    def test_reproducible_with_same_seed(self, eeg_buffer_256hz):
        r1 = compute_band_powers(eeg_buffer_256hz, fs=256)
        r2 = compute_band_powers(eeg_buffer_256hz, fs=256)
        assert r1.alpha == pytest.approx(r2.alpha)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestBandPowerEdgeCases:
    def test_single_channel(self):
        channel = _sine(10.0)
        result = compute_band_powers([channel], fs=256)
        assert result.alpha > 0

    def test_all_zeros_returns_zeros(self):
        channel = [0.0] * 256
        result = compute_band_powers([channel], fs=256)
        for val in (result.alpha, result.theta, result.beta, result.delta, result.gamma):
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_pure_noise_sums_to_positive(self):
        rng = np.random.default_rng(0)
        channel = rng.standard_normal(512).tolist()
        result = compute_band_powers([channel], fs=256)
        total = result.alpha + result.theta + result.beta + result.delta + result.gamma
        assert total > 0
