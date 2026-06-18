"""Unit tests for dsp/derived_eeg.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurolink.dsp.derived_eeg import (
    _MIN_SAMPLES,
    compute_contact_quality,
    compute_faa,
    compute_fmt,
    derived_eeg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
N = 512  # > _MIN_SAMPLES=256
N_CH = 5  # Muse 5-channel layout


def _eeg(n_ch: int = N_CH, n_samples: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_ch, n_samples)).astype(np.float32)


def _alpha_sine_ch(n_samples: int = N, ch: int = 0, n_ch: int = N_CH) -> np.ndarray:
    """EEG with a strong 10 Hz alpha tone on `ch`; noise elsewhere."""
    rng = np.random.default_rng(1)
    arr = (rng.standard_normal((n_ch, n_samples)) * 0.01).astype(np.float32)
    t = np.arange(n_samples) / FS
    arr[ch] = (np.sin(2 * np.pi * 10.0 * t) * 5.0).astype(np.float32)
    return arr


# ---------------------------------------------------------------------------
# derived_eeg()
# ---------------------------------------------------------------------------


class TestDerivedEeg:
    def test_none_returns_none_dict(self):
        result = derived_eeg(None)
        assert result == {"faa": None, "fmt": None}

    def test_1d_returns_none_dict(self):
        result = derived_eeg(np.zeros(N))
        assert result == {"faa": None, "fmt": None}

    def test_too_few_samples_returns_none_dict(self):
        short = np.zeros((N_CH, _MIN_SAMPLES - 1))
        result = derived_eeg(short)
        assert result == {"faa": None, "fmt": None}

    def test_too_few_channels_returns_none_dict(self):
        arr = np.zeros((4, N))  # only 4 channels, need 5
        result = derived_eeg(arr)
        assert result == {"faa": None, "fmt": None}

    def test_returns_dict_with_faa_and_fmt_keys(self):
        result = derived_eeg(_eeg())
        assert set(result.keys()) == {"faa", "fmt"}

    def test_faa_is_float_or_none(self):
        result = derived_eeg(_eeg())
        assert result["faa"] is None or isinstance(result["faa"], float)

    def test_fmt_is_float_or_none(self):
        result = derived_eeg(_eeg())
        assert result["fmt"] is None or isinstance(result["fmt"], float)

    def test_all_zero_eeg_faa_is_zero(self):
        """Zero EEG → both alpha powers are 0; FAA falls to the else branch (0.0)."""
        arr = np.zeros((N_CH, N), dtype=np.float32)
        result = derived_eeg(arr)
        assert result["faa"] == 0.0

    def test_all_zero_eeg_fmt_is_zero(self):
        arr = np.zeros((N_CH, N), dtype=np.float32)
        result = derived_eeg(arr)
        assert result["fmt"] == 0.0

    def test_exactly_min_samples_does_not_raise(self):
        arr = np.zeros((N_CH, _MIN_SAMPLES), dtype=np.float32)
        result = derived_eeg(arr)
        assert isinstance(result, dict)

    def test_faa_positive_when_af8_alpha_dominant(self):
        """AF8 (ch2) strong alpha, AF7 (ch1) noise → FAA should be positive."""
        arr = _alpha_sine_ch(ch=2)  # AF8 loud
        result = derived_eeg(arr)
        # faa = log(alpha_af8) - log(alpha_af7); af8 > af7 → positive
        assert result["faa"] is not None
        assert result["faa"] > 0.0

    def test_faa_negative_when_af7_alpha_dominant(self):
        """AF7 (ch1) strong alpha, AF8 (ch2) noise → FAA should be negative."""
        arr = _alpha_sine_ch(ch=1)  # AF7 loud
        result = derived_eeg(arr)
        assert result["faa"] is not None
        assert result["faa"] < 0.0

    def test_fmt_nonnegative(self):
        result = derived_eeg(_eeg())
        if result["fmt"] is not None:
            assert result["fmt"] >= 0.0


# ---------------------------------------------------------------------------
# compute_faa() — functional helper
# ---------------------------------------------------------------------------


class TestComputeFaa:
    def test_equal_powers_returns_zero(self):
        assert compute_faa(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_left_dominant_returns_positive(self):
        assert compute_faa(2.0, 1.0) > 0.0

    def test_right_dominant_returns_negative(self):
        assert compute_faa(1.0, 2.0) < 0.0

    def test_returns_float(self):
        assert isinstance(compute_faa(1.0, 1.0), float)

    def test_zero_left_clamps_to_epsilon(self):
        """log(0) avoided by epsilon clamp; must not raise."""
        result = compute_faa(0.0, 1.0)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_zero_right_clamps_to_epsilon(self):
        result = compute_faa(1.0, 0.0)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_both_zero_returns_zero(self):
        """Both zero → log(eps) - log(eps) = 0."""
        result = compute_faa(0.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        """log(2) - log(1) = log(2) ≈ 0.6931."""
        result = compute_faa(2.0, 1.0)
        assert result == pytest.approx(math.log(2.0), rel=1e-6)

    def test_large_asymmetry(self):
        result = compute_faa(100.0, 1.0)
        assert result == pytest.approx(math.log(100.0), rel=1e-6)


# ---------------------------------------------------------------------------
# compute_fmt() — functional helper
# ---------------------------------------------------------------------------


class TestComputeFmt:
    def test_returns_float(self):
        assert isinstance(compute_fmt(1.5), float)

    def test_returns_same_value(self):
        assert compute_fmt(3.7) == pytest.approx(3.7, rel=1e-9)

    def test_zero_returns_zero(self):
        assert compute_fmt(0.0) == 0.0

    def test_negative_passes_through(self):
        assert compute_fmt(-1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# compute_contact_quality()
# ---------------------------------------------------------------------------


class TestComputeContactQuality:
    def test_zero_rms_is_good(self):
        assert compute_contact_quality(0.0) == "good"

    def test_below_0_1_is_good(self):
        assert compute_contact_quality(0.09) == "good"

    def test_exactly_0_1_is_fair(self):
        assert compute_contact_quality(0.1) == "fair"

    def test_between_0_1_and_10_is_fair(self):
        assert compute_contact_quality(5.0) == "fair"

    def test_just_below_10_is_fair(self):
        assert compute_contact_quality(9.99) == "fair"

    def test_exactly_10_is_poor(self):
        assert compute_contact_quality(10.0) == "poor"

    def test_above_10_is_poor(self):
        assert compute_contact_quality(100.0) == "poor"

    def test_returns_string(self):
        for v in [0.0, 1.0, 50.0]:
            assert isinstance(compute_contact_quality(v), str)

    def test_exhaustive_enum(self):
        values = {compute_contact_quality(v) for v in [0.0, 5.0, 50.0]}
        assert values == {"good", "fair", "poor"}
