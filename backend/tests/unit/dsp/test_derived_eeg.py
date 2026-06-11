"""Unit tests for dsp/derived_eeg.py."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.derived_eeg import derived_eeg

FS = 256.0


def make_alpha_sig(freq: float, amp: float, fs: float = FS) -> np.ndarray:
    t = np.linspace(0, 4.0, int(fs * 4), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)


def test_derived_eeg_faa_sign_convention():
    """FAA = ln(alpha_AF8) - ln(alpha_AF7). When AF8 > AF7, FAA > 0."""
    tp9 = make_alpha_sig(10.0, 20.0)
    af7 = make_alpha_sig(10.0, 10.0)  # smaller
    af8 = make_alpha_sig(10.0, 30.0)  # larger
    tp10 = make_alpha_sig(10.0, 20.0)
    aux = make_alpha_sig(6.0, 15.0)

    eeg = np.stack([tp9, af7, af8, tp10, aux])
    result = derived_eeg(eeg, fs=FS)

    assert result["faa"] is not None
    assert result["faa"] > 0.0, f"FAA should be positive when AF8 > AF7, got {result['faa']}"


def test_derived_eeg_faa_negative_when_af7_dominant():
    tp9 = make_alpha_sig(10.0, 20.0)
    af7 = make_alpha_sig(10.0, 40.0)  # larger
    af8 = make_alpha_sig(10.0, 10.0)  # smaller
    tp10 = make_alpha_sig(10.0, 20.0)
    aux = make_alpha_sig(6.0, 15.0)

    eeg = np.stack([tp9, af7, af8, tp10, aux])
    result = derived_eeg(eeg, fs=FS)

    assert result["faa"] is not None
    assert result["faa"] < 0.0


def test_derived_eeg_fmt_is_theta():
    tp9 = make_alpha_sig(10.0, 20.0)
    af7 = make_alpha_sig(10.0, 20.0)
    af8 = make_alpha_sig(10.0, 20.0)
    tp10 = make_alpha_sig(10.0, 20.0)
    aux = make_alpha_sig(6.0, 50.0)  # strong theta on AUX

    eeg = np.stack([tp9, af7, af8, tp10, aux])
    result = derived_eeg(eeg, fs=FS)

    assert result["fmt"] is not None
    assert result["fmt"] > 0.0


def test_derived_eeg_short_buffer_returns_none():
    eeg = np.zeros((5, 1))
    result = derived_eeg(eeg)
    assert result["faa"] is None
    assert result["fmt"] is None
