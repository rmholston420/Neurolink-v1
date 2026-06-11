"""Unit tests for dsp/bandpower.py."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bandpower import (
    bandpower,
    compute_band_powers_from_buffer,
    make_buffers,
)


def test_bandpower_returns_zero_for_short_signal():
    sig = np.array([1.0])
    result = bandpower(sig, 8.0, 13.0, 256.0)
    assert result == 0.0


def test_bandpower_returns_zero_for_empty_signal():
    sig = np.array([])
    result = bandpower(sig, 8.0, 13.0, 256.0)
    assert result == 0.0


def test_bandpower_alpha_peak_at_10hz():
    fs = 256.0
    t = np.linspace(0, 4.0, int(fs * 4), endpoint=False)
    sig = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz alpha

    alpha = bandpower(sig, 8.0, 13.0, fs)
    theta = bandpower(sig, 4.0, 8.0, fs)
    beta = bandpower(sig, 13.0, 30.0, fs)

    assert alpha > theta, f"alpha={alpha} should > theta={theta}"
    assert alpha > beta, f"alpha={alpha} should > beta={beta}"
    assert alpha > 0.0


def test_bandpower_theta_peak_at_6hz():
    fs = 256.0
    t = np.linspace(0, 4.0, int(fs * 4), endpoint=False)
    sig = np.sin(2 * np.pi * 6.0 * t)

    theta = bandpower(sig, 4.0, 8.0, fs)
    alpha = bandpower(sig, 8.0, 13.0, fs)
    assert theta > alpha


def test_make_buffers_shapes():
    bufs = make_buffers()
    assert "eeg" in bufs
    assert "ppg" in bufs
    assert "accel" in bufs
    assert "gyro" in bufs
    assert bufs["eeg"].shape == (5, 1024)  # 5ch x 4s@256Hz
    assert bufs["ppg"].shape == (1920,)    # 30s@64Hz


def test_compute_band_powers_normalised():
    fs = 256.0
    t = np.linspace(0, 4.0, int(fs * 4), endpoint=False)
    ch = np.sin(2 * np.pi * 10.0 * t)
    eeg = np.stack([ch] * 5)
    result = compute_band_powers_from_buffer(eeg, fs=fs)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01, f"bands should sum ~1.0, got {total}"


def test_compute_band_powers_short_returns_zeros():
    eeg = np.zeros((5, 1))
    result = compute_band_powers_from_buffer(eeg)
    assert all(v == 0.0 for v in result.values())
