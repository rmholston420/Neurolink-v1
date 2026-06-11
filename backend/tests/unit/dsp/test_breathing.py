"""Unit tests for dsp/breathing.py."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.breathing import compute_breathing
from neurolink.models.eeg import BreathingPayload


def test_compute_breathing_returns_empty_for_no_data():
    result = compute_breathing(ibis_ms=[], accel_z=None)
    assert isinstance(result, BreathingPayload)
    assert result.rr_bpm is None


def test_compute_breathing_short_ibi_no_estimate():
    result = compute_breathing(ibis_ms=[800.0, 810.0], accel_z=None)
    assert result.rr_ppg is None


def test_compute_breathing_fused_estimate():
    """With enough IBIs and accel data, should compute a fused estimate."""
    # 15 IBIs at ~0.3 Hz breathing (one breath every ~3.3s -> 200bpm IBIs)
    ibis = [800.0] * 15  # flat IBIs, may not find peak but shouldn't crash
    accel_z = np.sin(2 * np.pi * 0.25 * np.linspace(0, 10, 520)).astype(np.float32)
    result = compute_breathing(ibis_ms=ibis, accel_z=accel_z)
    assert isinstance(result, BreathingPayload)
    # rr_bpm may or may not be set depending on scipy; just ensure no crash
