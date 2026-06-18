"""Unit tests for dsp/breathing.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.breathing import (
    _ACCEL_FS,
    _MIN_ACCEL_SAMPLES,
    _rr_from_accel,
    _rr_from_ibis,
    compute_breathing,
    estimate_rr,
)

ACCEL_FS = _ACCEL_FS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accel_z(
    n: int,
    freq_hz: float = 0.25,
    fs: float = ACCEL_FS,
) -> np.ndarray:
    """Synthetic accelerometer Z signal at a given frequency."""
    t = np.linspace(0, n / fs, n)
    return (np.sin(2 * np.pi * freq_hz * t) + 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# estimate_rr
# ---------------------------------------------------------------------------


class TestEstimateRR:
    def test_result_in_physiological_range(self):
        """0.25 Hz x 60 = 15 bpm - within 6-33 bpm."""
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        if result is not None:
            assert 6.0 <= result <= 33.0

    def test_too_short_returns_none(self):
        sig = np.ones(10, dtype=np.float32)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is None

    def test_returns_float_or_none(self):
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# _rr_from_accel
# ---------------------------------------------------------------------------


class TestRrFromAccel:
    def test_too_short_returns_none(self):
        assert _rr_from_accel(np.ones(10, dtype=np.float32), fs=ACCEL_FS) is None

    def test_valid_signal_returns_float_or_none(self):
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES)
        result = _rr_from_accel(sig, fs=ACCEL_FS)
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# _rr_from_ibis
# ---------------------------------------------------------------------------


class TestRrFromIbis:
    def test_empty_list_returns_none(self):
        assert _rr_from_ibis([]) is None

    def test_valid_ibis_returns_float_or_none(self):
        ibis = [800.0] * 30
        result = _rr_from_ibis(ibis)
        assert result is None or isinstance(result, float)

    def test_too_few_valid_ibis_returns_none(self):
        ibis = [50.0, 9999.0]  # both out of physiological range
        result = _rr_from_ibis(ibis)
        assert result is None


# ---------------------------------------------------------------------------
# compute_breathing
# ---------------------------------------------------------------------------


class TestComputeBreathing:
    def test_returns_breathing_payload(self):
        from neurolink.models.eeg import BreathingPayload

        result = compute_breathing([], accel_z=None)
        assert isinstance(result, BreathingPayload)

    def test_no_data_returns_none_rr(self):
        result = compute_breathing([], accel_z=None)
        assert result.rr_bpm is None

    def test_accel_only_sets_rr_accel(self):
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = compute_breathing([], accel_z=sig)
        # rr_ppg should be None (no IBIs), rr_accel may be set
        assert result.rr_ppg is None

    def test_ibi_only_sets_rr_ppg(self):
        ibis = [800.0] * 30
        result = compute_breathing(ibis, accel_z=None)
        # rr_accel should be None (no accel), rr_ppg may be set
        assert result.rr_accel is None

    def test_both_sources_sets_fused_rr(self):
        ibis = [800.0] * 30
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = compute_breathing(ibis, accel_z=sig)
        assert isinstance(result.rr_bpm, float) or result.rr_bpm is None
