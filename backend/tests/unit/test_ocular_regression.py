"""Unit tests for neurolink.dsp.ocular_regression.

Public API confirmed from source:
  OcularRegressor(config)
  .apply(eeg) -> np.ndarray   # method is apply(), not remove()
  .reset() / .get_stats() / .get_config() / .set_config()

Graceful degradation: eog_channel_idx >= n_channels -> passthrough unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegressor, OcularRegressionConfig


FS = 256.0
N_CH_EEG = 4     # TP9, AF7, AF8, TP10
N_CH_WITH_AUX = 5
N_SAMPLES = 256


@pytest.fixture
def regressor_no_aux() -> OcularRegressor:
    """EOG channel index 4, only 4-ch array provided -> graceful degradation."""
    cfg = OcularRegressionConfig(eog_channel_idx=4)
    return OcularRegressor(config=cfg)


@pytest.fixture
def regressor_with_aux() -> OcularRegressor:
    """EOG channel is ch4 (AUX), present in 5-ch array."""
    cfg = OcularRegressionConfig(eog_channel_idx=4)
    return OcularRegressor(config=cfg)


@pytest.fixture
def regressor_disabled() -> OcularRegressor:
    return OcularRegressor(config=OcularRegressionConfig(enable=False))


@pytest.fixture
def eeg_4ch() -> np.ndarray:
    return np.random.default_rng(42).normal(0, 5.0, (N_CH_EEG, N_SAMPLES))


@pytest.fixture
def eeg_5ch_blink() -> np.ndarray:
    """5-channel array where ch4 = synthetic blink EOG reference."""
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 5.0, (N_CH_WITH_AUX, N_SAMPLES))
    t = np.linspace(0, 1, N_SAMPLES)
    blink = 120.0 * np.sin(2 * np.pi * 1.0 * t)
    eeg[4] = blink
    eeg[1] += 0.8 * blink  # AF7
    eeg[2] += 0.7 * blink  # AF8
    return eeg


class TestConstruction:
    def test_default_construction(self):
        r = OcularRegressor()
        assert r is not None

    def test_custom_config(self):
        cfg = OcularRegressionConfig(recalib_frames=50)
        r = OcularRegressor(config=cfg)
        assert r is not None


class TestDisabledPassthrough:
    def test_disabled_returns_array_unchanged(self, regressor_disabled, eeg_4ch):
        out = regressor_disabled.apply(eeg_4ch)
        np.testing.assert_array_equal(out, eeg_4ch)


class TestGracefulDegradation:
    def test_missing_eog_channel_passthrough(self, regressor_no_aux, eeg_4ch):
        """EOG channel 4 beyond 4-ch array -> returns input unchanged."""
        out = regressor_no_aux.apply(eeg_4ch)
        np.testing.assert_array_equal(out, eeg_4ch)

    def test_output_shape_preserved_no_aux(self, regressor_no_aux, eeg_4ch):
        out = regressor_no_aux.apply(eeg_4ch)
        assert out.shape == eeg_4ch.shape


class TestWithEOGReference:
    def test_output_same_shape_as_input(self, regressor_with_aux, eeg_5ch_blink):
        out = regressor_with_aux.apply(eeg_5ch_blink)
        assert out.shape == eeg_5ch_blink.shape

    def test_output_is_finite(self, regressor_with_aux, eeg_5ch_blink):
        out = regressor_with_aux.apply(eeg_5ch_blink)
        assert np.isfinite(out).all()

    def test_frontal_amplitude_reduced_after_warmup(self, regressor_with_aux):
        """After enough frames the regressor should reduce blink amplitude."""
        rng = np.random.default_rng(1)
        t = np.linspace(0, 1, N_SAMPLES)
        blink = 120.0 * np.sin(2 * np.pi * 1.0 * t)

        # Build a fresh contaminated frame for measurement
        eeg_test = rng.normal(0, 5.0, (N_CH_WITH_AUX, N_SAMPLES))
        eeg_test[4] = blink
        eeg_test[1] += 0.8 * blink
        original_ptp = np.ptp(eeg_test[1])

        # Warm up with 30 frames
        for _ in range(30):
            frame = rng.normal(0, 5.0, (N_CH_WITH_AUX, N_SAMPLES))
            frame[4] = blink
            frame[1] += 0.8 * blink
            regressor_with_aux.apply(frame)

        out = regressor_with_aux.apply(eeg_test)
        corrected_ptp = np.ptp(out[1])
        assert corrected_ptp < original_ptp * 1.1


class TestStats:
    def test_get_stats_returns_dict(self, regressor_with_aux, eeg_5ch_blink):
        regressor_with_aux.apply(eeg_5ch_blink)
        stats = regressor_with_aux.get_stats()
        assert isinstance(stats, dict)


class TestReset:
    def test_reset_does_not_raise(self, regressor_with_aux):
        regressor_with_aux.reset()

    def test_after_reset_still_functional(self, regressor_with_aux, eeg_5ch_blink):
        regressor_with_aux.reset()
        out = regressor_with_aux.apply(eeg_5ch_blink)
        assert out.shape == eeg_5ch_blink.shape


class TestEdgeCases:
    def test_single_sample_no_crash(self, regressor_with_aux):
        eeg = np.zeros((N_CH_WITH_AUX, 1))
        out = regressor_with_aux.apply(eeg)
        assert out.shape == eeg.shape

    def test_config_roundtrip(self, regressor_with_aux):
        cfg = regressor_with_aux.get_config()
        regressor_with_aux.set_config(cfg)
        assert regressor_with_aux.get_config().enable == cfg.enable
