"""Unit tests for dsp/spherical_spline.py (Stage 2 Perrin interpolator)."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.spherical_spline import (
    _EEG_CHANNELS,
    interpolate_bad_channels,
)

N_SAMPLES = 256
N_CH = 5


def _make_eeg(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N_CH, N_SAMPLES)) * 10.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestGoodPath:
    def test_no_bad_channels_returns_unchanged(self):
        eeg = _make_eeg()
        out = interpolate_bad_channels(eeg, [])
        # Should be the exact same object (no copy)
        assert out is eeg

    def test_single_bad_channel_interpolated(self):
        eeg = _make_eeg()
        # Zero out AF8 to simulate a dead channel
        eeg[2] = 0.0
        out = interpolate_bad_channels(eeg, ["AF8"])
        assert out is not eeg  # copy was made
        # Interpolated channel should not be all zeros
        assert not np.allclose(out[2], 0.0)
        # Good channels must be untouched
        np.testing.assert_array_equal(out[0], eeg[0])  # TP9
        np.testing.assert_array_equal(out[1], eeg[1])  # AF7
        np.testing.assert_array_equal(out[3], eeg[3])  # TP10

    def test_interpolated_values_are_finite(self):
        eeg = _make_eeg(1)
        eeg[0] = 0.0  # TP9 bad
        out = interpolate_bad_channels(eeg, ["TP9"])
        assert np.all(np.isfinite(out[0]))

    def test_two_bad_channels(self):
        eeg = _make_eeg(2)
        eeg[1] = 0.0  # AF7
        eeg[3] = 0.0  # TP10
        out = interpolate_bad_channels(eeg, ["AF7", "TP10"])
        # Both should be non-zero after interpolation
        assert not np.allclose(out[1], 0.0)
        assert not np.allclose(out[3], 0.0)


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------

class TestFallbacks:
    def test_three_bad_channels_fallback_mean(self):
        """Only 1 good EEG channel → spline impossible → fill with mean."""
        eeg = _make_eeg(3)
        eeg[0] = 0.0
        eeg[1] = 0.0
        eeg[2] = 0.0  # Only TP10 (index 3) is good
        out = interpolate_bad_channels(eeg, ["TP9", "AF7", "AF8"])
        # Bad channels filled with mean of TP10
        expected = eeg[3].copy()
        np.testing.assert_array_almost_equal(out[0], expected, decimal=4)

    def test_all_four_eeg_bad_fills_zeros(self):
        """No good channels at all → fill with zeros."""
        eeg = _make_eeg(4)
        out = interpolate_bad_channels(eeg, _EEG_CHANNELS)
        for idx in range(4):
            np.testing.assert_array_equal(out[idx], np.zeros(N_SAMPLES, dtype=np.float32))


# ---------------------------------------------------------------------------
# AUX passthrough
# ---------------------------------------------------------------------------

class TestAUXPassthrough:
    def test_aux_never_interpolated(self):
        """AUX (index 4) should always pass through unchanged."""
        eeg = _make_eeg(5)
        aux_original = eeg[4].copy()
        out = interpolate_bad_channels(eeg, ["AUX"])
        # interpolate_bad_channels ignores AUX in bad list → returns eeg unchanged
        assert out is eeg
        np.testing.assert_array_equal(out[4], aux_original)

    def test_aux_unchanged_when_eeg_channels_interpolated(self):
        eeg = _make_eeg(6)
        aux_original = eeg[4].copy()
        eeg[0] = 0.0
        out = interpolate_bad_channels(eeg, ["TP9"])
        np.testing.assert_array_equal(out[4], aux_original)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

class TestShapePreservation:
    def test_output_shape_matches_input(self):
        eeg = _make_eeg()
        eeg[1] = 0.0
        out = interpolate_bad_channels(eeg, ["AF7"])
        assert out.shape == eeg.shape

    def test_output_dtype_is_float32(self):
        eeg = _make_eeg()
        eeg[2] = 0.0
        out = interpolate_bad_channels(eeg, ["AF8"])
        assert out.dtype == np.float32
