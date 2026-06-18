"""Unit tests for dsp/bandpower.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.bandpower import (
    bandpower,
    compute_band_powers_from_buffer,
    make_buffers,
)

FS = 256.0


# ---------------------------------------------------------------------------
# make_buffers() factory
# ---------------------------------------------------------------------------


class TestMakeBuffers:
    def test_returns_dict(self):
        b = make_buffers()
        assert isinstance(b, dict)

    def test_eeg_buffer_shape(self):
        """5 channels x 4 s x 256 Hz = 1024 samples."""
        b = make_buffers()
        assert b["eeg"].shape == (5, 1024)

    def test_ppg_buffer_present(self):
        b = make_buffers()
        assert "ppg" in b
        assert isinstance(b["ppg"], np.ndarray)

    def test_accel_buffer_present(self):
        b = make_buffers()
        assert "accel" in b
        assert isinstance(b["accel"], np.ndarray)

    def test_gyro_buffer_present(self):
        b = make_buffers()
        assert "gyro" in b
        assert isinstance(b["gyro"], np.ndarray)


# ---------------------------------------------------------------------------
# bandpower() -- single-band scalar
# ---------------------------------------------------------------------------


class TestBandpower:
    def test_returns_float(self):
        sig = np.random.randn(512).astype(np.float32)
        result = bandpower(sig, 8.0, 13.0)
        assert isinstance(result, float)

    def test_alpha_signal_has_high_alpha_power(self):
        """Pure 10 Hz sine should have dominant alpha power."""
        t = np.linspace(0, 4, 1024)
        sig = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        alpha = bandpower(sig, 8.0, 13.0)
        beta = bandpower(sig, 13.0, 30.0)
        assert alpha > beta

    def test_none_signal_returns_zero(self):
        assert bandpower(None, 8.0, 13.0) == 0.0

    def test_too_short_signal_returns_zero(self):
        tiny = np.zeros(2, dtype=np.float32)
        assert bandpower(tiny, 8.0, 13.0) == 0.0

    def test_nonnegative_output(self):
        sig = np.random.randn(512).astype(np.float32)
        assert bandpower(sig, 1.0, 50.0) >= 0.0


# ---------------------------------------------------------------------------
# compute_band_powers_from_buffer() -- multi-channel aggregation
# ---------------------------------------------------------------------------


class TestComputeBandPowers:
    def test_returns_dict_with_five_bands(self):
        eeg = np.random.randn(5, 1024).astype(np.float32)
        result = compute_band_powers_from_buffer(eeg)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_values_sum_to_one_for_active_signal(self):
        """Normalised power fractions must sum to 1.0."""
        t = np.linspace(0, 4, 1024)
        alpha = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        eeg = np.tile(alpha, (5, 1))
        result = compute_band_powers_from_buffer(eeg)
        assert abs(sum(result.values()) - 1.0) < 1e-5

    def test_zero_signal_returns_all_zeros(self):
        eeg = np.zeros((5, 1024), dtype=np.float32)
        result = compute_band_powers_from_buffer(eeg)
        assert all(v == 0.0 for v in result.values())

    def test_none_input_returns_zeros(self):
        result = compute_band_powers_from_buffer(None)
        assert all(v == 0.0 for v in result.values())

    def test_1d_input_is_handled(self):
        sig = np.random.randn(1024).astype(np.float32)
        result = compute_band_powers_from_buffer(sig)
        assert isinstance(result, dict)

    def test_too_short_returns_zeros(self):
        tiny = np.zeros((5, 1), dtype=np.float32)
        result = compute_band_powers_from_buffer(tiny)
        assert all(v == 0.0 for v in result.values())

    def test_alpha_dominant_when_alpha_signal(self):
        t = np.linspace(0, 4, 1024)
        alpha_sig = (np.sin(2 * np.pi * 10.0 * t) * 100).astype(np.float32)
        eeg = np.tile(alpha_sig, (5, 1))
        result = compute_band_powers_from_buffer(eeg)
        assert result["alpha"] == max(result.values())
