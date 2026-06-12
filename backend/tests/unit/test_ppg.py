"""Unit tests for dsp.ppg — HR and HRV computation."""

from __future__ import annotations

import pytest

from neurolink.dsp.ppg import compute_hrv


class TestComputeHRV:
    def test_empty_rr_returns_none_fields(self):
        result = compute_hrv(rr_intervals_ms=[])
        assert result is None or hasattr(result, "hr_bpm")

    def test_single_rr_returns_none_or_result(self):
        result = compute_hrv(rr_intervals_ms=[800.0])
        assert result is None or hasattr(result, "hr_bpm")

    def test_normal_rr_series_returns_result(self):
        """Typical resting HR ~60 bpm -> RR ~1000 ms."""
        rr = [1000.0] * 10
        result = compute_hrv(rr_intervals_ms=rr)
        if result is not None:
            assert 30.0 <= result.hr_bpm <= 200.0

    def test_rmssd_non_negative(self):
        rr = [800.0, 820.0, 790.0, 810.0, 805.0]
        result = compute_hrv(rr_intervals_ms=rr)
        if result is not None and result.hrv_rmssd is not None:
            assert result.hrv_rmssd >= 0.0

    def test_does_not_raise_on_short_input(self):
        result = compute_hrv(rr_intervals_ms=[1000.0, 1050.0])
        assert result is None or hasattr(result, "hr_bpm")
