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

FS = 256.0
N = 512


# ---------------------------------------------------------------------------
# bandpower()
# ---------------------------------------------------------------------------

class TestBandpower:
    def test_none_returns_zero(self):
        assert bandpower(None, 8.0, 13.0) == 0.0

    def test_empty_returns_zero(self):
        assert bandpower(np.array([]), 8.0, 13.0) == 0.0

    def test_single_sample_returns_zero(self):
        assert bandpower(np.array([1.0]), 8.0, 13.0) == 0.0

    def test_returns_float(self):
        sig = np.random.default_rng(0).standard_normal(N).astype(np.float32)
        result = bandpower(sig, 8.0, 13.0)
        assert isinstance(result, float)

    def test_nonnegative(self):
        sig = np.random.default_rng(1).standard_normal(N).astype(np.float32)
        assert bandpower(sig, 8.0, 13.0) >= 0.0

    def test_sine_at_10hz_has_alpha_power(self):
        """10 Hz tone → dominant alpha band power."""
        t = np.arange(N) / FS
        sig = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        alpha = bandpower(sig, 8.0, 13.0, fs=FS)
        theta = bandpower(sig, 4.0, 8.0, fs=FS)
        beta = bandpower(sig, 13.0, 30.0, fs=FS)
        assert alpha > theta
        assert alpha > beta

    def test_sine_at_5hz_has_theta_power(self):
        t = np.arange(N) / FS
        sig = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)
        theta = bandpower(sig, 4.0, 8.0, fs=FS)
        alpha = bandpower(sig, 8.0, 13.0, fs=FS)
        assert theta > alpha

    def test_all_zeros_returns_zero(self):
        assert bandpower(np.zeros(N), 8.0, 13.0) == 0.0

    def test_custom_fs_accepted(self):
        sig = np.random.default_rng(2).standard_normal(N).astype(np.float32)
        result = bandpower(sig, 4.0, 8.0, fs=128.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_band_powers()
# ---------------------------------------------------------------------------

class TestComputeBandPowers:
    def test_none_returns_zero_bandpowers(self):
        result = compute_band_powers(None)
        assert result == BandPowers()

    def test_returns_bandpowers_instance(self):
        ch = np.random.default_rng(3).standard_normal((4, N)).astype(np.float32)
        result = compute_band_powers(ch)
        assert isinstance(result, BandPowers)

    def test_single_sample_returns_zero(self):
        result = compute_band_powers(np.zeros((4, 1)))
        assert result == BandPowers()

    def test_normalised_sum_approx_one(self):
        """Normalised powers must sum to 1 for non-zero signal."""
        rng = np.random.default_rng(4)
        ch = rng.standard_normal((4, N)).astype(np.float32)
        bp = compute_band_powers(ch)
        total = bp.delta + bp.theta + bp.alpha + bp.beta + bp.gamma
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_all_powers_nonnegative(self):
        ch = np.random.default_rng(5).standard_normal((4, N)).astype(np.float32)
        bp = compute_band_powers(ch)
        for attr in ("delta", "theta", "alpha", "beta", "gamma"):
            assert getattr(bp, attr) >= 0.0

    def test_all_zeros_returns_zero_bandpowers(self):
        result = compute_band_powers(np.zeros((4, N)))
        assert result == BandPowers()

    def test_1d_input_treated_as_single_channel(self):
        sig = np.random.default_rng(6).standard_normal(N).astype(np.float32)
        result = compute_band_powers(sig)
        assert isinstance(result, BandPowers)

    def test_alpha_dominant_in_10hz_signal(self):
        t = np.arange(N) / FS
        ch = np.tile(np.sin(2 * np.pi * 10.0 * t), (4, 1)).astype(np.float32)
        bp = compute_band_powers(ch, fs=FS)
        assert bp.alpha > bp.theta
        assert bp.alpha > bp.beta


# ---------------------------------------------------------------------------
# compute_band_powers_from_buffer()
# ---------------------------------------------------------------------------

class TestComputeBandPowersFromBuffer:
    def test_none_returns_zero_dict(self):
        result = compute_band_powers_from_buffer(None)
        assert all(v == 0.0 for v in result.values())

    def test_returns_dict_with_band_keys(self):
        eeg = np.random.default_rng(7).standard_normal((5, N)).astype(np.float32)
        result = compute_band_powers_from_buffer(eeg)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_single_sample_returns_zeros(self):
        result = compute_band_powers_from_buffer(np.zeros((5, 1)))
        assert all(v == 0.0 for v in result.values())

    def test_normalised_sum_approx_one(self):
        eeg = np.random.default_rng(8).standard_normal((5, N)).astype(np.float32)
        result = compute_band_powers_from_buffer(eeg)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_all_values_nonnegative(self):
        eeg = np.random.default_rng(9).standard_normal((5, N)).astype(np.float32)
        for v in compute_band_powers_from_buffer(eeg).values():
            assert v >= 0.0

    def test_1d_input_treated_as_single_channel(self):
        sig = np.random.default_rng(10).standard_normal(N).astype(np.float32)
        result = compute_band_powers_from_buffer(sig)
        assert isinstance(result, dict)

    def test_all_zeros_returns_zero_dict(self):
        result = compute_band_powers_from_buffer(np.zeros((5, N)))
        assert all(v == 0.0 for v in result.values())


# ---------------------------------------------------------------------------
# make_buffers()
# ---------------------------------------------------------------------------

class TestMakeBuffers:
    def test_returns_dict(self):
        assert isinstance(make_buffers(), dict)

    def test_has_expected_keys(self):
        assert set(make_buffers().keys()) == {"eeg", "ppg", "accel", "gyro"}

    def test_eeg_buffer_shape(self):
        """5 channels × 4 s × 256 Hz = 1024 samples."""
        b = make_buffers()
        assert b["eeg"].shape == (5, 1024)

    def test_ppg_buffer_1d(self):
        b = make_buffers()
        assert b["ppg"].ndim == 1
        assert b["ppg"].shape[0] > 0

    def test_accel_gyro_same_shape(self):
        b = make_buffers()
        assert b["accel"].shape == b["gyro"].shape

    def test_all_zeros_initially(self):
        b = make_buffers()
        for key, arr in b.items():
            assert np.all(arr == 0.0), f"{key} buffer is not all zeros"

    def test_dtype_float32(self):
        b = make_buffers()
        for key, arr in b.items():
            assert arr.dtype == np.float32, f"{key} buffer dtype is {arr.dtype}"
