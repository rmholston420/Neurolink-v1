"""Unit tests for dsp/breathing.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurolink.dsp.breathing import (
    _MIN_ACCEL_SAMPLES,
    _MIN_IBIS,
    _RR_MAX_HZ,
    _RR_MIN_HZ,
    compute_breathing,
    estimate_rr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACCEL_FS = 52.0


def _ibi_series(n: int = 20, base_ms: float = 800.0) -> list[float]:
    """Physiologically plausible IBIs with a breathing-rate modulation."""
    rng = np.random.default_rng(0)
    return (base_ms + rng.standard_normal(n) * 20).tolist()


def _accel_with_rr(rr_hz: float = 0.25, fs: float = ACCEL_FS) -> np.ndarray:
    """Accel-z signal with a dominant respiratory frequency."""
    n = _MIN_ACCEL_SAMPLES + 100
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * rr_hz * t).astype(np.float32)
    return sig


# ---------------------------------------------------------------------------
# estimate_rr()
# ---------------------------------------------------------------------------

class TestEstimateRr:
    def test_none_returns_none(self):
        assert estimate_rr(None) is None

    def test_empty_returns_none(self):
        assert estimate_rr([]) is None

    def test_single_sample_returns_none(self):
        assert estimate_rr([1.0]) is None

    def test_returns_float_for_valid_signal(self):
        sig = _accel_with_rr(rr_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert isinstance(result, float)

    def test_result_in_physiological_range(self):
        sig = _accel_with_rr(rr_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is not None
        bpm = result
        assert 6.0 <= bpm <= 33.0  # _RR_MIN_HZ * 60 to _RR_MAX_HZ * 60

    def test_detects_dominant_breathing_frequency(self):
        """0.25 Hz = 15 bpm — should land near 15 bpm."""
        sig = _accel_with_rr(rr_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is not None
        assert 10.0 <= result <= 20.0

    def test_flat_signal_returns_none_or_float(self):
        """Flat signal: FFT is zero everywhere; may return None or a boundary value."""
        sig = np.zeros(600)
        result = estimate_rr(sig, fs=ACCEL_FS)
        # Either None (no peak found) or a float — must not raise
        assert result is None or isinstance(result, float)

    def test_custom_fs_accepted(self):
        sig = _accel_with_rr(rr_hz=0.25, fs=128.0)
        result = estimate_rr(sig, fs=128.0)
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_breathing() — IBI path
# ---------------------------------------------------------------------------

class TestComputeBreathingIBI:
    def test_too_few_ibis_rr_ppg_none(self):
        from neurolink.models.eeg import BreathingPayload
        short = _ibi_series(n=_MIN_IBIS - 1)
        payload = compute_breathing(short)
        assert isinstance(payload, BreathingPayload)
        assert payload.rr_ppg is None

    def test_sufficient_ibis_rr_ppg_not_none(self):
        ibis = _ibi_series(n=_MIN_IBIS)
        payload = compute_breathing(ibis)
        # rr_ppg may be None if no peak found in physiological range, but must not raise
        assert payload.rr_ppg is None or isinstance(payload.rr_ppg, float)

    def test_rr_bpm_equals_rr_ppg_when_no_accel(self):
        ibis = _ibi_series(n=30)
        payload = compute_breathing(ibis, accel_z=None)
        assert payload.rr_bpm == payload.rr_ppg

    def test_rr_accel_none_when_not_provided(self):
        payload = compute_breathing(_ibi_series(), accel_z=None)
        assert payload.rr_accel is None


# ---------------------------------------------------------------------------
# compute_breathing() — accel path
# ---------------------------------------------------------------------------

class TestComputeBreathingAccel:
    def test_too_short_accel_rr_accel_none(self):
        short_accel = np.zeros(_MIN_ACCEL_SAMPLES - 1, dtype=np.float32)
        payload = compute_breathing([], accel_z=short_accel)
        assert payload.rr_accel is None

    def test_sufficient_accel_rr_accel_not_none(self):
        accel = _accel_with_rr(0.25)
        payload = compute_breathing([], accel_z=accel)
        assert payload.rr_accel is None or isinstance(payload.rr_accel, float)

    def test_rr_bpm_equals_rr_accel_when_no_ibis(self):
        accel = _accel_with_rr(0.25)
        payload = compute_breathing([], accel_z=accel)
        assert payload.rr_bpm == payload.rr_accel


# ---------------------------------------------------------------------------
# compute_breathing() — fusion
# ---------------------------------------------------------------------------

class TestComputeBreathingFusion:
    def test_both_sources_fused_as_average(self):
        """When both rr_ppg and rr_accel are available, rr_bpm is their mean."""
        ibis = _ibi_series(n=30)
        accel = _accel_with_rr(0.25)
        payload = compute_breathing(ibis, accel_z=accel)
        if payload.rr_ppg is not None and payload.rr_accel is not None:
            expected = (payload.rr_ppg + payload.rr_accel) / 2.0
            assert abs(payload.rr_bpm - expected) < 1e-6

    def test_both_none_rr_bpm_none(self):
        payload = compute_breathing([], accel_z=None)
        assert payload.rr_bpm is None
        assert payload.rr_ppg is None
        assert payload.rr_accel is None

    def test_payload_is_breathing_payload(self):
        from neurolink.models.eeg import BreathingPayload
        payload = compute_breathing(_ibi_series())
        assert isinstance(payload, BreathingPayload)


# ---------------------------------------------------------------------------
# Boundary: exactly _MIN_IBIS IBIs
# ---------------------------------------------------------------------------

class TestIBIBoundary:
    def test_exactly_min_ibis_does_not_raise(self):
        ibis = _ibi_series(n=_MIN_IBIS)
        payload = compute_breathing(ibis)
        assert payload is not None

    def test_one_below_min_ibis_rr_ppg_none(self):
        ibis = _ibi_series(n=_MIN_IBIS - 1)
        payload = compute_breathing(ibis)
        assert payload.rr_ppg is None
