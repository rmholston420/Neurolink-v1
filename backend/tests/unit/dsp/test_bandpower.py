"""Unit tests for dsp/bandpower.py."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bandpower import (
    BandPowers,
    bandpower,
    compute_band_powers,
    compute_band_powers_from_buffer,
    make_buffers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
N = 1024  # 4 seconds at 256 Hz


def _sine(freq: float, n: int = N, fs: float = FS, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _noise(n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


# ---------------------------------------------------------------------------
# bandpower() — core helper
# ---------------------------------------------------------------------------

class TestBandpower:
    def test_none_returns_zero(self):
        assert bandpower(None, 8.0, 13.0) == 0.0

    def test_empty_returns_zero(self):
        assert bandpower(np.array([]), 8.0, 13.0) == 0.0

    def test_single_sample_returns_zero(self):
        assert bandpower(np.array([1.0]), 8.0, 13.0) == 0.0

    def test_returns_float(self):
        sig = _sine(10.0)
        result = bandpower(sig, 8.0, 13.0)
        assert isinstance(result, float)

    def test_nonnegative(self):
        sig = _sine(10.0)
        assert bandpower(sig, 8.0, 13.0) >= 0.0

    def test_alpha_sine_dominant_in_alpha_band(self):
        """10 Hz sine should have most power in alpha (8-13 Hz)."""
        sig = _sine(10.0)
        alpha_p = bandpower(sig, 8.0, 13.0)
        theta_p = bandpower(sig, 4.0, 8.0)
        beta_p = bandpower(sig, 13.0, 30.0)
        assert alpha_p > theta_p
        assert alpha_p > beta_p

    def test_theta_sine_dominant_in_theta_band(self):
        sig = _sine(6.0)
        theta_p = bandpower(sig, 4.0, 8.0)
        alpha_p = bandpower(sig, 8.0, 13.0)
        assert theta_p > alpha_p

    def test_power_scales_with_amplitude(self):
        lo, hi = 8.0, 13.0
        p1 = bandpower(_sine(10.0, amp=1.0), lo, hi)
        p2 = bandpower(_sine(10.0, amp=2.0), lo, hi)
        assert p2 > p1

    def test_zero_signal_returns_zero(self):
        assert bandpower(np.zeros(N), 8.0, 13.0) == 0.0

    def test_custom_fs(self):
        # Should not raise; just verify it returns a float
        result = bandpower(_sine(10.0, fs=512.0, n=2048), 8.0, 13.0, fs=512.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# BandPowers dataclass
# ---------------------------------------------------------------------------

class TestBandPowersDataclass:
    def test_defaults_all_zero(self):
        bp = BandPowers()
        assert bp.delta == 0.0
        assert bp.theta == 0.0
        assert bp.alpha == 0.0
        assert bp.beta == 0.0
        assert bp.gamma == 0.0

    def test_fields_assignable(self):
        bp = BandPowers(delta=0.1, theta=0.2, alpha=0.3, beta=0.25, gamma=0.15)
        assert bp.alpha == 0.3


# ---------------------------------------------------------------------------
# compute_band_powers()
# ---------------------------------------------------------------------------

class TestComputeBandPowers:
    def test_none_returns_zero_band_powers(self):
        bp = compute_band_powers(None)
        assert bp == BandPowers()

    def test_empty_list_returns_zero(self):
        bp = compute_band_powers([])
        assert bp == BandPowers()

    def test_all_zero_signal_returns_zero(self):
        zero_ch = [np.zeros(N).tolist()]
        bp = compute_band_powers(zero_ch)
        assert bp == BandPowers()

    def test_returns_band_powers_instance(self):
        bp = compute_band_powers([_sine(10.0).tolist()])
        assert isinstance(bp, BandPowers)

    def test_normalised_to_one(self):
        """Sum of all bands must equal 1.0 for any non-zero signal."""
        channels = [_noise(seed=i).tolist() for i in range(4)]
        bp = compute_band_powers(channels)
        total = bp.delta + bp.theta + bp.alpha + bp.beta + bp.gamma
        assert abs(total - 1.0) < 1e-5

    def test_alpha_dominant_for_10hz_sine(self):
        channels = [_sine(10.0).tolist()]
        bp = compute_band_powers(channels)
        assert bp.alpha > bp.theta
        assert bp.alpha > bp.beta

    def test_theta_dominant_for_6hz_sine(self):
        channels = [_sine(6.0).tolist()]
        bp = compute_band_powers(channels)
        assert bp.theta > bp.alpha

    def test_1d_input_treated_as_single_channel(self):
        arr = _sine(10.0)
        bp = compute_band_powers(arr)
        assert isinstance(bp, BandPowers)
        total = bp.delta + bp.theta + bp.alpha + bp.beta + bp.gamma
        assert abs(total - 1.0) < 1e-5

    def test_multi_channel_averaging(self):
        """Two-channel input must return valid normalised powers."""
        channels = [_sine(10.0).tolist(), _sine(20.0).tolist()]
        bp = compute_band_powers(channels)
        total = bp.delta + bp.theta + bp.alpha + bp.beta + bp.gamma
        assert abs(total - 1.0) < 1e-5

    def test_single_sample_returns_zero(self):
        bp = compute_band_powers([[1.0]])
        assert bp == BandPowers()


# ---------------------------------------------------------------------------
# compute_band_powers_from_buffer()
# ---------------------------------------------------------------------------

class TestComputeBandPowersFromBuffer:
    def test_none_returns_zero_dict(self):
        result = compute_band_powers_from_buffer(None)
        assert result == {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}

    def test_returns_dict_with_all_bands(self):
        eeg = np.stack([_noise(seed=i) for i in range(5)])  # (5, N)
        result = compute_band_powers_from_buffer(eeg)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_normalised_to_one(self):
        eeg = np.stack([_noise(seed=i) for i in range(5)])
        result = compute_band_powers_from_buffer(eeg)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_all_zero_returns_zero_dict(self):
        eeg = np.zeros((5, N))
        result = compute_band_powers_from_buffer(eeg)
        assert all(v == 0.0 for v in result.values())

    def test_1d_input_treated_as_single_channel(self):
        sig = _sine(10.0)
        result = compute_band_powers_from_buffer(sig)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_alpha_dominant_for_10hz(self):
        eeg = np.stack([_sine(10.0)] * 5)
        result = compute_band_powers_from_buffer(eeg)
        assert result["alpha"] > result["theta"]
        assert result["alpha"] > result["beta"]

    def test_single_sample_returns_zero_dict(self):
        eeg = np.ones((5, 1))
        result = compute_band_powers_from_buffer(eeg)
        assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# make_buffers()
# ---------------------------------------------------------------------------

class TestMakeBuffers:
    def test_returns_dict(self):
        buffers = make_buffers()
        assert isinstance(buffers, dict)

    def test_has_expected_keys(self):
        buffers = make_buffers()
        assert set(buffers.keys()) == {"eeg", "ppg", "accel", "gyro"}

    def test_eeg_shape(self):
        buffers = make_buffers()
        assert buffers["eeg"].shape == (5, 1024)

    def test_ppg_1d(self):
        buffers = make_buffers()
        assert buffers["ppg"].ndim == 1

    def test_accel_gyro_1d(self):
        buffers = make_buffers()
        assert buffers["accel"].ndim == 1
        assert buffers["gyro"].ndim == 1

    def test_accel_gyro_same_length(self):
        buffers = make_buffers()
        assert buffers["accel"].shape == buffers["gyro"].shape

    def test_all_zeros(self):
        buffers = make_buffers()
        for arr in buffers.values():
            assert np.all(arr == 0.0)

    def test_dtype_float32(self):
        buffers = make_buffers()
        for arr in buffers.values():
            assert arr.dtype == np.float32

    def test_returns_new_arrays_each_call(self):
        b1 = make_buffers()
        b2 = make_buffers()
        b1["eeg"][0, 0] = 999.0
        assert b2["eeg"][0, 0] == 0.0  # independent allocation
