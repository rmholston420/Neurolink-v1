"""Unit tests for dsp/ppg.py."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ppg import compute_ppg, _poincare
from neurolink.models.eeg import PPGPayload, PoincareIndices

PPG_FS = 64.0


def test_compute_ppg_returns_empty_for_short_buffer():
    short = np.zeros(10)
    result = compute_ppg(short, fs=PPG_FS)
    assert isinstance(result, PPGPayload)
    assert result.hr_bpm == 0.0


def test_poincare_returns_empty_for_single_ibi():
    result = _poincare([800.0])
    assert isinstance(result, PoincareIndices)
    assert result.sd1 == 0.0
    assert result.sd2 == 0.0


def test_poincare_sd1_sd2_positive():
    ibis = [800.0, 810.0, 790.0, 820.0, 780.0, 800.0, 810.0]
    result = _poincare(ibis)
    assert result.sd1 >= 0.0
    assert result.sd2 >= 0.0
    assert result.ellipse_area >= 0.0


def test_compute_ppg_returns_empty_payload_without_neurokit2():
    """Should return empty PPGPayload without raising when buffer is short."""
    sig = np.random.randn(10).astype(np.float32)
    result = compute_ppg(sig, fs=PPG_FS)
    assert result.hr_bpm == 0.0  # short buffer -> empty
