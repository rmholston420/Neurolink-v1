"""Unit tests for dsp/ppg.py -- PPG/HRV computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurolink.dsp.ppg import (
    _MIN_SAMPLES,
    HRVResult,
    PoincareMetrics,
    _poincare,
    compute_hrv,
    compute_ppg,
)
from neurolink.models.eeg import PPGPayload

FS = 64.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rr(n: int = 30, mean_ms: float = 900.0, jitter: float = 30.0) -> list[float]:
    rng = np.random.default_rng(0)
    return list((rng.standard_normal(n) * jitter + mean_ms).astype(float))


# ---------------------------------------------------------------------------
# compute_ppg() -- guard conditions (neurokit2 not required for guards)
# ---------------------------------------------------------------------------


class TestComputePpgGuards:
    def test_none_returns_empty_payload(self):
        result = compute_ppg(None)
        assert result == PPGPayload(hr_bpm=0.0, hrv_rmssd=0.0, ibi_ms=[])

    def test_too_short_returns_empty_payload(self):
        short = np.random.randn(int(_MIN_SAMPLES) - 1).astype(np.float32)
        result = compute_ppg(short)
        assert result == PPGPayload(hr_bpm=0.0, hrv_rmssd=0.0, ibi_ms=[])

    def test_returns_ppg_payload_instance(self):
        short = np.random.randn(10).astype(np.float32)
        result = compute_ppg(short)
        assert isinstance(result, PPGPayload)


# ---------------------------------------------------------------------------
# compute_hrv() -- pure IBI math, no neurokit2
# ---------------------------------------------------------------------------


class TestComputeHrv:
    def test_empty_list_returns_none(self):
        assert compute_hrv([]) is None

    def test_all_invalid_ibis_returns_none(self):
        """IBIs outside 300-2000 ms -> no valid intervals."""
        assert compute_hrv([100.0, 2500.0]) is None

    def test_returns_hrv_result_for_valid_ibis(self):
        result = compute_hrv(_rr())
        assert isinstance(result, HRVResult)

    def test_hr_bpm_in_physiological_range(self):
        result = compute_hrv(_rr(mean_ms=900.0))
        assert result is not None
        assert 30.0 <= result.hr_bpm <= 200.0

    def test_hr_bpm_known_value(self):
        """mean IBI = 1000 ms -> HR = 60 bpm exactly."""
        result = compute_hrv([1000.0] * 20)
        assert result is not None
        assert result.hr_bpm == pytest.approx(60.0, abs=0.01)

    def test_hrv_rmssd_nonnegative(self):
        result = compute_hrv(_rr())
        assert result is not None
        assert result.hrv_rmssd >= 0.0

    def test_constant_ibis_rmssd_zero(self):
        """No beat-to-beat variation -> RMSSD = 0."""
        result = compute_hrv([1000.0] * 10)
        assert result is not None
        assert result.hrv_rmssd == pytest.approx(0.0, abs=1e-6)
