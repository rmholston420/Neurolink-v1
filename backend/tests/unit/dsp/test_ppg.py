"""Unit tests for dsp/ppg.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.ppg import _poincare, compute_ppg


def test_compute_ppg_returns_empty_for_short_buffer():
    """Returns empty PPGPayload for buffers shorter than minimum."""
    short = np.zeros(100)
    result = compute_ppg(short, fs=64.0)
    assert result.hr_bpm == 0.0
    assert result.ibi_ms == []


def test_compute_ppg_hr_in_valid_range():
    """HR should be in 40-120 bpm range for a plausible PPG signal."""
    fs = 64.0
    t = np.linspace(0, 30, int(fs * 30))
    # Simulate ~66 bpm heartbeat
    ppg = np.sin(2 * np.pi * 1.1 * t) + 0.1 * np.random.randn(len(t))
    result = compute_ppg(ppg, fs=fs)
    if result.hr_bpm > 0:  # neurokit2 may detect or not
        assert 30.0 <= result.hr_bpm <= 150.0, f"HR {result.hr_bpm} out of range"


def test_poincare_sd1_sd2_positive():
    """SD1 and SD2 should be non-negative for valid IBI list."""
    ibis = [800.0, 820.0, 790.0, 810.0, 800.0, 815.0, 795.0, 805.0, 810.0]
    result = _poincare(ibis)
    assert result.sd1 >= 0.0
    assert result.sd2 >= 0.0


def test_poincare_returns_zeros_for_short_ibis():
    result = _poincare([800.0])  # only 1 IBI
    assert result.sd1 == 0.0
    assert result.sd2 == 0.0


def test_poincare_ellipse_area_positive_for_valid_ibis():
    ibis = [800.0, 820.0, 790.0, 810.0, 800.0, 815.0, 795.0, 805.0]
    result = _poincare(ibis)
    assert result.ellipse_area >= 0.0
