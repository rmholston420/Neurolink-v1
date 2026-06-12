"""Unit tests for dsp/ppg.py — PPG/HRV computation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurolink.dsp.ppg import (
    HRVResult,
    PoincareMetrics,
    _MIN_SAMPLES,
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
# compute_ppg() — guard conditions (neurokit2 not required for guards)
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
# compute_hrv() — pure IBI math, no neurokit2
# ---------------------------------------------------------------------------

class TestComputeHrv:
    def test_empty_list_returns_none(self):
        assert compute_hrv([]) is None

    def test_all_invalid_ibis_returns_none(self):
        """IBIs outside 300–2000 ms → no valid intervals."""
        assert compute_hrv([100.0, 2500.0]) is None

    def test_returns_hrv_result_for_valid_ibis(self):
        result = compute_hrv(_rr())
        assert isinstance(result, HRVResult)

    def test_hr_bpm_in_physiological_range(self):
        result = compute_hrv(_rr(mean_ms=900.0))
        assert result is not None
        assert 30.0 <= result.hr_bpm <= 200.0

    def test_hr_bpm_known_value(self):
        """mean IBI = 1000 ms → HR = 60 bpm exactly."""
        result = compute_hrv([1000.0] * 20)
        assert result is not None
        assert result.hr_bpm == pytest.approx(60.0, abs=0.01)

    def test_hrv_rmssd_nonnegative(self):
        result = compute_hrv(_rr())
        assert result is not None
        assert result.hrv_rmssd >= 0.0

    def test_constant_ibis_rmssd_zero(self):
        """No beat-to-beat variation → RMSSD = 0."""
        result = compute_hrv([1000.0] * 10)
        assert result is not None
        assert result.hrv_rmssd == pytest.approx(0.0, abs=1e-9)

    def test_single_valid_ibi_rmssd_zero(self):
        result = compute_hrv([1000.0])
        assert result is not None
        assert result.hrv_rmssd == pytest.approx(0.0)

    def test_very_fast_hr_above_200bpm_returns_none(self):
        """IBI = 200 ms → HR = 300 bpm > valid max."""
        result = compute_hrv([200.0] * 10)
        assert result is None

    def test_very_slow_hr_below_30bpm_returns_none(self):
        """IBI = 2500 ms is filtered as out-of-range (> 2000 ms)."""
        result = compute_hrv([2500.0] * 10)
        assert result is None

    def test_filters_out_of_range_ibis(self):
        """Mixed valid/invalid — invalid IBIs discarded, valid ones used."""
        ibis = [1000.0] * 10 + [100.0, 3000.0]  # two out-of-range
        result = compute_hrv(ibis)
        assert result is not None
        assert result.hr_bpm == pytest.approx(60.0, abs=0.01)


# ---------------------------------------------------------------------------
# _poincare() — internal helper
# ---------------------------------------------------------------------------

class TestPoincare:
    def test_single_ibi_returns_zero_metrics(self):
        result = _poincare([1000.0])
        assert result == PoincareMetrics()

    def test_empty_returns_zero_metrics(self):
        result = _poincare([])
        assert result == PoincareMetrics()

    def test_returns_poincare_metrics_instance(self):
        result = _poincare(_rr())
        assert isinstance(result, PoincareMetrics)

    def test_sd1_sd2_nonnegative(self):
        result = _poincare(_rr())
        assert result.sd1 >= 0.0
        assert result.sd2 >= 0.0

    def test_ellipse_area_nonnegative(self):
        result = _poincare(_rr())
        assert result.ellipse_area >= 0.0

    def test_constant_ibis_sd1_zero(self):
        """No beat-to-beat variation → SD1 = 0."""
        result = _poincare([1000.0] * 20)
        assert result.sd1 == pytest.approx(0.0, abs=1e-9)

    def test_ellipse_area_equals_pi_sd1_sd2(self):
        result = _poincare(_rr())
        expected = math.pi * result.sd1 * result.sd2
        assert result.ellipse_area == pytest.approx(expected)


# ---------------------------------------------------------------------------
# derived_eeg — FAA, FMt, contact quality (smoke tests via derived_eeg.py)
# ---------------------------------------------------------------------------

class TestDerivedEEG:
    """Quick sanity checks for compute_faa, compute_fmt, compute_contact_quality."""

    def test_compute_faa_returns_float(self):
        from neurolink.dsp.derived_eeg import compute_faa
        assert isinstance(compute_faa(1.0, 1.0), float)

    def test_compute_faa_equal_powers_near_zero(self):
        from neurolink.dsp.derived_eeg import compute_faa
        assert compute_faa(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_compute_faa_left_dominant_positive(self):
        from neurolink.dsp.derived_eeg import compute_faa
        assert compute_faa(2.0, 1.0) > 0.0

    def test_compute_faa_right_dominant_negative(self):
        from neurolink.dsp.derived_eeg import compute_faa
        assert compute_faa(1.0, 2.0) < 0.0

    def test_compute_faa_zero_inputs_no_error(self):
        """log(0) is clamped to epsilon — must not raise."""
        from neurolink.dsp.derived_eeg import compute_faa
        result = compute_faa(0.0, 0.0)
        assert isinstance(result, float)

    def test_compute_fmt_returns_float(self):
        from neurolink.dsp.derived_eeg import compute_fmt
        assert isinstance(compute_fmt(0.5), float)

    def test_compute_fmt_identity(self):
        from neurolink.dsp.derived_eeg import compute_fmt
        assert compute_fmt(3.14) == pytest.approx(3.14)

    def test_contact_quality_good(self):
        from neurolink.dsp.derived_eeg import compute_contact_quality
        assert compute_contact_quality(0.05) == "good"

    def test_contact_quality_fair(self):
        from neurolink.dsp.derived_eeg import compute_contact_quality
        assert compute_contact_quality(1.0) == "fair"

    def test_contact_quality_poor(self):
        from neurolink.dsp.derived_eeg import compute_contact_quality
        assert compute_contact_quality(50.0) == "poor"

    def test_derived_eeg_none_returns_none_values(self):
        from neurolink.dsp.derived_eeg import derived_eeg
        result = derived_eeg(None)
        assert result["faa"] is None
        assert result["fmt"] is None

    def test_derived_eeg_too_short_returns_none_values(self):
        from neurolink.dsp.derived_eeg import derived_eeg
        eeg = np.zeros((5, 10), dtype=np.float32)
        result = derived_eeg(eeg)
        assert result["faa"] is None
        assert result["fmt"] is None

    def test_derived_eeg_too_few_channels_returns_none_values(self):
        from neurolink.dsp.derived_eeg import derived_eeg
        eeg = np.zeros((4, 512), dtype=np.float32)  # needs 5 channels
        result = derived_eeg(eeg)
        assert result["faa"] is None
        assert result["fmt"] is None

    def test_derived_eeg_returns_float_values(self):
        from neurolink.dsp.derived_eeg import derived_eeg
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((5, 512)).astype(np.float32) * 5.0
        result = derived_eeg(eeg)
        assert isinstance(result["faa"], float)
        assert isinstance(result["fmt"], float)
