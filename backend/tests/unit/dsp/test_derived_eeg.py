"""Unit tests for dsp/derived_eeg.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.derived_eeg import derived_eeg


def test_derived_eeg_returns_none_for_short_buffer():
    """Returns None for FAA/FMt when buffer is too short."""
    eeg = np.zeros((5, 10))
    result = derived_eeg(eeg)
    assert result["faa"] is None
    assert result["fmt"] is None


def test_derived_eeg_faa_sign_convention():
    """FAA should be positive when AF8 alpha > AF7 alpha."""
    fs = 256.0
    n = int(fs * 2)  # 2 seconds
    t = np.linspace(0, 2, n)

    eeg = np.zeros((5, n))
    # AF7 (index 1): weaker alpha
    eeg[1] = 0.5 * np.sin(2 * np.pi * 10.0 * t)
    # AF8 (index 2): stronger alpha
    eeg[2] = 2.0 * np.sin(2 * np.pi * 10.0 * t)
    # AUX (index 4): theta for FMt
    eeg[4] = 1.0 * np.sin(2 * np.pi * 6.0 * t)

    result = derived_eeg(eeg, fs=fs)
    assert result["faa"] is not None
    assert result["faa"] > 0, f"Expected positive FAA, got {result['faa']}"


def test_derived_eeg_faa_negative_when_af7_dominant():
    """FAA should be negative when AF7 alpha > AF8 alpha."""
    fs = 256.0
    n = int(fs * 2)
    t = np.linspace(0, 2, n)
    eeg = np.zeros((5, n))
    eeg[1] = 3.0 * np.sin(2 * np.pi * 10.0 * t)  # AF7 strong
    eeg[2] = 0.5 * np.sin(2 * np.pi * 10.0 * t)  # AF8 weak
    result = derived_eeg(eeg, fs=fs)
    assert result["faa"] is not None
    assert result["faa"] < 0


def test_derived_eeg_fmt_positive():
    """FMt should be positive for signal with theta in AUX."""
    fs = 256.0
    n = int(fs * 2)
    t = np.linspace(0, 2, n)
    eeg = np.zeros((5, n))
    eeg[4] = 2.0 * np.sin(2 * np.pi * 6.0 * t)
    eeg[1] = 1.0 * np.sin(2 * np.pi * 10.0 * t)
    eeg[2] = 1.0 * np.sin(2 * np.pi * 10.0 * t)
    result = derived_eeg(eeg, fs=fs)
    assert result["fmt"] is not None
    assert result["fmt"] > 0
