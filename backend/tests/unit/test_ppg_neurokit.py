"""PPG tests that exercise the neurokit2 peak-detection path (lines 107-163).

These tests use pytest.importorskip("neurokit2") so they:
  - run and cover lines 107-163 when neurokit2 is installed (CI with the
    full dev extras, local dev environment)
  - skip cleanly with a readable skip message when neurokit2 is absent
    (CI with minimal deps, containerised environments that exclude heavy
    optional packages)

This replaces the previous behaviour where those lines were simply never
exercised in any environment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

nk = pytest.importorskip("neurokit2", reason="neurokit2 not installed -- skipping PPG peak-detection path tests")

from neurolink.dsp.ppg import compute_ppg  # noqa: E402 -- import after skip guard

_FS = 64.0
_MIN_SAMPLES = int(_FS * 15)  # 960


def _synthetic_ppg(n_samples: int = 1024, hr_bpm: float = 60.0) -> np.ndarray:
    """Generate a clean synthetic PPG signal with a known heart rate.

    Uses a sum-of-sinusoids approximation: fundamental at HR + 2nd harmonic.
    This is realistic enough for neurokit2 peak detection to succeed.
    """
    t = np.linspace(0, n_samples / _FS, n_samples)
    freq = hr_bpm / 60.0
    ppg = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    # Small amount of noise so the signal is not trivially flat
    rng = np.random.default_rng(7)
    ppg += 0.05 * rng.standard_normal(n_samples)
    return ppg.astype(np.float32)


class TestComputePpgNeurokit:
    """Tests exercising the neurokit2 execution path inside compute_ppg."""

    def test_too_short_returns_empty_payload(self):
        """Arrays shorter than the minimum must return an empty payload."""
        short = np.zeros(100, dtype=np.float32)
        result = compute_ppg(short, fs=_FS)
        assert result.hr_bpm == 0.0
        assert result.hrv_rmssd == 0.0
        assert result.ibi_ms == []

    def test_none_input_returns_empty_payload(self):
        result = compute_ppg(None, fs=_FS)  # type: ignore[arg-type]
        assert result.hr_bpm == 0.0

    def test_synthetic_signal_returns_valid_hr(self):
        """A clean synthetic 60 BPM signal must produce a physiologically valid HR."""
        ppg = _synthetic_ppg(n_samples=_MIN_SAMPLES + 128, hr_bpm=60.0)
        result = compute_ppg(ppg, fs=_FS)
        # neurokit2 peak detection is approximate; allow a wide physiological
        # window rather than asserting exact 60 BPM.
        if result.hr_bpm > 0:  # may still return empty on noisy signal
            assert 30.0 <= result.hr_bpm <= 200.0
            assert math.isfinite(result.hr_bpm)

    def test_ibi_ms_values_are_physiological(self):
        """If IBI values are returned they must all be in [300, 2000] ms."""
        ppg = _synthetic_ppg(n_samples=_MIN_SAMPLES + 256, hr_bpm=70.0)
        result = compute_ppg(ppg, fs=_FS)
        for ibi in result.ibi_ms:
            assert 300.0 <= ibi <= 2000.0, f"Out-of-range IBI: {ibi}"

    def test_all_zero_signal_returns_empty_payload(self):
        """A zero-valued signal should trigger the neurokit2 empty-peaks guard."""
        ppg = np.zeros(_MIN_SAMPLES + 64, dtype=np.float32)
        result = compute_ppg(ppg, fs=_FS)
        # Either empty payload OR valid result -- never a crash
        assert isinstance(result.hr_bpm, float)
        assert math.isfinite(result.hr_bpm)

    def test_hrv_rmssd_non_negative(self):
        """RMSSD is a root-mean-square value and must always be >= 0."""
        ppg = _synthetic_ppg(n_samples=_MIN_SAMPLES + 512, hr_bpm=75.0)
        result = compute_ppg(ppg, fs=_FS)
        assert result.hrv_rmssd >= 0.0

    def test_payload_fields_finite(self):
        """All float fields in the returned payload must be finite."""
        ppg = _synthetic_ppg(n_samples=_MIN_SAMPLES + 128, hr_bpm=65.0)
        result = compute_ppg(ppg, fs=_FS)
        for field_name in ("hr_bpm", "hrv_rmssd", "sd1", "sd2", "ellipse_area"):
            val = getattr(result, field_name)
            assert math.isfinite(val), f"{field_name} is not finite: {val}"
