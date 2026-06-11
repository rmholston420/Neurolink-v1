"""Unit tests for dsp/bandpower.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.bandpower import bandpower, compute_band_powers_from_buffer, make_buffers


def test_bandpower_returns_zero_for_short_signal():
    """bandpower() returns 0.0 for signals shorter than 2 samples."""
    result = bandpower(np.array([1.0]), lo=8.0, hi=13.0, fs=256.0)
    assert result == 0.0


def test_bandpower_returns_zero_for_empty_signal():
    result = bandpower(np.array([]), lo=8.0, hi=13.0, fs=256.0)
    assert result == 0.0


def test_bandpower_alpha_peak_at_10hz():
    """A pure 10 Hz sine wave should have dominant power in alpha band."""
    fs = 256.0
    t = np.linspace(0, 4, int(fs * 4))
    sig = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz (alpha)
    alpha = bandpower(sig, 8.0, 13.0, fs)
    theta = bandpower(sig, 4.0, 8.0, fs)
    beta = bandpower(sig, 13.0, 30.0, fs)
    assert alpha > theta
    assert alpha > beta
    assert alpha > 0


def test_bandpower_theta_peak():
    """A 6 Hz sine should have most power in theta band."""
    fs = 256.0
    t = np.linspace(0, 4, int(fs * 4))
    sig = np.sin(2 * np.pi * 6.0 * t)
    theta = bandpower(sig, 4.0, 8.0, fs)
    alpha = bandpower(sig, 8.0, 13.0, fs)
    assert theta > alpha


def test_compute_band_powers_from_buffer_returns_all_bands():
    """Should return normalised fractions for all 5 bands."""
    fs = 256.0
    n = int(fs * 4)
    eeg = np.random.randn(5, n).astype(np.float32)
    result = compute_band_powers_from_buffer(eeg, fs=fs)
    assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01  # should sum to ~1


def test_compute_band_powers_short_buffer():
    """Short buffer (< 2 samples) should return all zeros."""
    eeg = np.zeros((5, 1), dtype=np.float32)
    result = compute_band_powers_from_buffer(eeg)
    assert all(v == 0.0 for v in result.values())


def test_make_buffers_shapes():
    """make_buffers() should return correctly sized numpy arrays."""
    bufs = make_buffers()
    assert bufs["eeg"].shape == (5, 1024)
    assert bufs["ppg"].shape == (1920,)
    assert bufs["accel"].shape[0] == 624  # 52 * 4 * 3
    assert bufs["gyro"].shape[0] == 624
