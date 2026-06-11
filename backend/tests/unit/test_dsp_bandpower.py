"""Unit tests for dsp.bandpower — pure signal processing functions."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.bandpower import bandpower, compute_band_powers_from_buffer, make_buffers


def test_bandpower_returns_float_for_synthetic_signal():
    fs = 256.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    result = bandpower(sig, lo=8.0, hi=13.0, fs=fs)
    assert isinstance(result, float)
    assert result > 0.0


def test_bandpower_empty_returns_zero():
    assert bandpower(np.array([]), 8.0, 13.0) == 0.0


def test_bandpower_single_sample_returns_zero():
    assert bandpower(np.array([1.0]), 8.0, 13.0) == 0.0


def test_compute_band_powers_sums_to_one():
    fs = 256.0
    t = np.linspace(0, 1, int(fs), endpoint=False)
    eeg = np.tile(np.sin(2 * np.pi * 10 * t), (5, 1)).astype(np.float32)
    result = compute_band_powers_from_buffer(eeg, fs=fs)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01
    assert result["alpha"] > 0.5


def test_compute_band_powers_empty_eeg_returns_zeros():
    result = compute_band_powers_from_buffer(None)
    assert all(v == 0.0 for v in result.values())


def test_make_buffers_correct_shapes():
    bufs = make_buffers()
    assert bufs["eeg"].shape == (5, 1024)
    assert bufs["ppg"].ndim == 1
    assert bufs["accel"].ndim == 1
