"""Unit tests for DSP band power computation."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.bandpower import compute_band_powers_from_buffer


def _sine(freq_hz: float, fs: float = 256.0, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


class TestComputeBandPowers:
    def test_returns_five_bands(self):
        buf = np.stack([_sine(10)] * 4)  # 4 ch alpha
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_alpha_dominant_for_10hz(self):
        buf = np.stack([_sine(10)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert result["alpha"] == max(result.values()), f"Expected alpha dominant, got {result}"

    def test_theta_dominant_for_6hz(self):
        buf = np.stack([_sine(6)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert result["theta"] == max(result.values()), f"Expected theta dominant, got {result}"

    def test_beta_dominant_for_20hz(self):
        buf = np.stack([_sine(20)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert result["beta"] == max(result.values()), f"Expected beta dominant, got {result}"

    def test_values_sum_to_one_approx(self):
        buf = np.stack([_sine(10) + _sine(20)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.05, f"Band powers sum to {total}, expected ~1.0"

    def test_all_values_in_unit_interval(self):
        buf = np.stack([_sine(10)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        for band, val in result.items():
            assert 0.0 <= val <= 1.0, f"{band}={val} out of [0,1]"

    def test_handles_short_buffer_gracefully(self):
        buf = np.ones((4, 10), dtype=np.float32)
        # Should not raise — may return zeros or partial values
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert isinstance(result, dict)

    def test_delta_dominant_for_2hz(self):
        buf = np.stack([_sine(2)] * 4)
        result = compute_band_powers_from_buffer(buf, fs=256.0)
        assert result["delta"] == max(result.values()), f"Expected delta dominant, got {result}"
