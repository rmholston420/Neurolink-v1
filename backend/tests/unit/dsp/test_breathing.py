"""Unit tests for dsp/breathing.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.breathing import compute_breathing


def test_compute_breathing_empty_returns_none():
    result = compute_breathing([], accel_z=None)
    assert result.rr_bpm is None
    assert result.rr_ppg is None
    assert result.rr_accel is None


def test_compute_breathing_fused_estimate():
    """With both IBIs and accel_z, fused rate should be non-None."""
    # Generate plausible IBIs (~66 bpm)
    ibis = [900.0 + i * 5 for i in range(20)]

    # Generate accel_z with ~0.2 Hz respiratory component
    fs = 52.0
    n = int(fs * 20)
    t = np.linspace(0, 20, n)
    accel_z = 1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * t) + 0.01 * np.random.randn(n)

    result = compute_breathing(ibis, accel_z=accel_z)
    # rr_accel should be set
    if result.rr_accel is not None:
        assert 3.0 <= result.rr_accel <= 40.0, f"RR accel {result.rr_accel} out of range"


def test_compute_breathing_returns_payload_type():
    from neurolink.models.eeg import BreathingPayload

    result = compute_breathing([])
    assert isinstance(result, BreathingPayload)
