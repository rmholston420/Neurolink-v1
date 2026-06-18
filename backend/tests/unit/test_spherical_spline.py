"""Unit tests for neurolink.dsp.spherical_spline.

Real public API: module-level function (no class):
  interpolate_bad_channels(eeg, bad_channels) -> np.ndarray

Muse montage: TP9=ch0, AF7=ch1, AF8=ch2, TP10=ch3.
AUX (ch4) is never interpolated.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.spherical_spline import interpolate_bad_channels


N_SAMPLES = 128
N_CH_EEG = 4
N_CH_WITH_AUX = 5


@pytest.fixture
def clean_eeg() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 10.0, (N_CH_EEG, N_SAMPLES))


@pytest.fixture
def eeg_with_aux() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0, 10.0, (N_CH_WITH_AUX, N_SAMPLES))


@pytest.fixture
def flat_tp9_eeg() -> np.ndarray:
    """EEG where TP9 (ch0) is flat-line (bad channel)."""
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10.0, (N_CH_EEG, N_SAMPLES))
    eeg[0] = 0.0  # TP9 flat
    return eeg


class TestNoOpCases:
    def test_empty_bad_channels_returns_unchanged(self, clean_eeg):
        out = interpolate_bad_channels(clean_eeg, [])
        # Docstring: returns eeg unchanged (no copy) when bad_channels is empty
        assert out is clean_eeg

    def test_empty_bad_channels_shape_preserved(self, clean_eeg):
        out = interpolate_bad_channels(clean_eeg, [])
        assert out.shape == clean_eeg.shape


class TestSingleBadChannel:
    def test_output_shape_preserved(self, flat_tp9_eeg):
        out = interpolate_bad_channels(flat_tp9_eeg, ["TP9"])
        assert out.shape == flat_tp9_eeg.shape

    def test_bad_channel_interpolated_nonzero(self, flat_tp9_eeg):
        """After interpolation, the previously flat TP9 should have variance."""
        out = interpolate_bad_channels(flat_tp9_eeg, ["TP9"])
        assert np.var(out[0]) > 0.0

    def test_good_channels_unchanged(self, flat_tp9_eeg):
        """Channels not in bad_channels list should be untouched."""
        out = interpolate_bad_channels(flat_tp9_eeg, ["TP9"])
        np.testing.assert_array_equal(out[1], flat_tp9_eeg[1])  # AF7
        np.testing.assert_array_equal(out[2], flat_tp9_eeg[2])  # AF8
        np.testing.assert_array_equal(out[3], flat_tp9_eeg[3])  # TP10

    def test_af7_bad_channel(self, clean_eeg):
        rng = np.random.default_rng(2)
        eeg = clean_eeg.copy()
        eeg[1] = 0.0  # AF7 flat
        out = interpolate_bad_channels(eeg, ["AF7"])
        assert out.shape == eeg.shape
        assert np.var(out[1]) > 0.0


class TestFallbackCases:
    def test_three_bad_channels_fills_with_mean(self):
        """Only 1 good channel remains -> fallback: fill bad channels with mean."""
        rng = np.random.default_rng(3)
        eeg = rng.normal(0, 10.0, (N_CH_EEG, N_SAMPLES))
        out = interpolate_bad_channels(eeg, ["TP9", "AF7", "AF8"])
        assert out.shape == eeg.shape
        # After fallback, bad channels should not be all-zero
        assert not np.allclose(out[0], 0.0)

    def test_all_bad_channels_no_crash(self):
        """All 4 EEG channels bad -> fallback handles without exception."""
        rng = np.random.default_rng(4)
        eeg = rng.normal(0, 10.0, (N_CH_EEG, N_SAMPLES))
        try:
            out = interpolate_bad_channels(eeg, ["TP9", "AF7", "AF8", "TP10"])
            assert out.shape == eeg.shape
        except Exception as exc:
            pytest.fail(f"all-bad fallback raised unexpectedly: {exc}")


class TestAUXHandling:
    def test_aux_channel_not_interpolated(self, eeg_with_aux):
        """AUX (ch4) is never part of the spline."""
        original_aux = eeg_with_aux[4].copy()
        out = interpolate_bad_channels(eeg_with_aux, ["TP9"])
        np.testing.assert_array_equal(out[4], original_aux)

    def test_5ch_array_shape_preserved(self, eeg_with_aux):
        out = interpolate_bad_channels(eeg_with_aux, [])
        assert out.shape == eeg_with_aux.shape


class TestEdgeCases:
    def test_unknown_channel_name_no_crash(self, clean_eeg):
        """Channel name not in Muse montage -> ignored, array returned."""
        out = interpolate_bad_channels(clean_eeg, ["FPZ"])
        assert out.shape == clean_eeg.shape

    def test_duplicate_bad_channel_names_no_crash(self, clean_eeg):
        eeg = clean_eeg.copy()
        eeg[0] = 0.0
        out = interpolate_bad_channels(eeg, ["TP9", "TP9"])
        assert out.shape == eeg.shape
