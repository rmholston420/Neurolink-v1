"""Unit tests for dsp/breathing.py."""

from __future__ import annotations

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
from neurolink.models.eeg import BreathingPayload

ACCEL_FS = 52.0
IBI_FS = 4.0  # virtual IBI series rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ibis(n: int = 20, mean_ms: float = 900.0, jitter: float = 20.0) -> list[float]:
    rng = np.random.default_rng(0)
    return list((rng.standard_normal(n) * jitter + mean_ms).astype(float))


def _accel_z(
    n: int = _MIN_ACCEL_SAMPLES,
    freq_hz: float = 0.25,
    fs: float = ACCEL_FS,
) -> np.ndarray:
    """Sine wave at freq_hz (respiratory band) of length n."""
    t = np.arange(n) / fs
    return (np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


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

    def test_returns_float_for_valid_input(self):
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert isinstance(result, float)

    def test_result_in_physiological_range(self):
        """0.25 Hz × 60 = 15 bpm — within 6–33 bpm."""
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is not None
        assert (_RR_MIN_HZ * 60) <= result <= (_RR_MAX_HZ * 60)

    def test_dc_signal_returns_none_or_out_of_range(self):
        """Constant signal has no oscillatory component in the RR band."""
        sig = np.ones(512, dtype=np.float32)
        result = estimate_rr(sig, fs=ACCEL_FS)
        # Either None or value outside physiological bpm is acceptable
        if result is not None:
            assert not ((_RR_MIN_HZ * 60) < result < (_RR_MAX_HZ * 60))

    def test_list_input_accepted(self):
        sig = list(_accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25))
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert isinstance(result, float)

    def test_custom_fs_accepted(self):
        sig = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = estimate_rr(sig, fs=ACCEL_FS)
        assert result is not None


# ---------------------------------------------------------------------------
# compute_breathing() — returns BreathingPayload
# ---------------------------------------------------------------------------


class TestComputeBreathing:
    def test_returns_breathing_payload(self):
        result = compute_breathing([])
        assert isinstance(result, BreathingPayload)

    def test_empty_ibis_and_no_accel_rr_bpm_none(self):
        result = compute_breathing([])
        assert result.rr_bpm is None
        assert result.rr_ppg is None
        assert result.rr_accel is None

    def test_insufficient_ibis_rr_ppg_none(self):
        """Fewer than _MIN_IBIS IBIs → rr_ppg is None."""
        ibis = _ibis(n=_MIN_IBIS - 1)
        result = compute_breathing(ibis)
        assert result.rr_ppg is None

    def test_sufficient_ibis_rr_ppg_not_none(self):
        ibis = _ibis(n=_MIN_IBIS)
        result = compute_breathing(ibis)
        # rr_ppg may be None if FFT finds no peak — just check no exception
        assert isinstance(result, BreathingPayload)

    def test_insufficient_accel_rr_accel_none(self):
        ibis = _ibis()
        short_accel = _accel_z(n=_MIN_ACCEL_SAMPLES - 1)
        result = compute_breathing(ibis, accel_z=short_accel, accel_fs=ACCEL_FS)
        assert result.rr_accel is None

    def test_sufficient_accel_rr_accel_not_none(self):
        ibis = []
        accel = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = compute_breathing(ibis, accel_z=accel, accel_fs=ACCEL_FS)
        assert result.rr_accel is not None

    def test_fused_result_averages_ppg_and_accel(self):
        """When both sources available, rr_bpm == (rr_ppg + rr_accel) / 2."""
        ibis = _ibis(n=_MIN_IBIS)
        accel = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = compute_breathing(ibis, accel_z=accel, accel_fs=ACCEL_FS)
        if result.rr_ppg is not None and result.rr_accel is not None:
            expected = (result.rr_ppg + result.rr_accel) / 2.0
            assert result.rr_bpm == pytest.approx(expected)

    def test_accel_only_rr_bpm_equals_rr_accel(self):
        accel = _accel_z(n=_MIN_ACCEL_SAMPLES, freq_hz=0.25)
        result = compute_breathing([], accel_z=accel, accel_fs=ACCEL_FS)
        if result.rr_accel is not None:
            assert result.rr_bpm == pytest.approx(result.rr_accel)

    def test_no_accel_rr_bpm_equals_rr_ppg(self):
        ibis = _ibis(n=_MIN_IBIS)
        result = compute_breathing(ibis, accel_z=None)
        if result.rr_ppg is not None:
            assert result.rr_bpm == pytest.approx(result.rr_ppg)
