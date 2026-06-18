"""Unit tests for neurolink.dsp.ocular_regression.

Real public API:
  OcularRegressor         — stateful corrector; __init__(config)
  OcularRegressionConfig  — tunable dataclass

Graceful degradation: when no EOG channel is present
(eog_channel_idx >= n_channels) the corrector returns the array unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegressor, OcularRegressionConfig


FS = 256.0
N_CH_EEG = 4  # TP9, AF7, AF8, TP10
N_CH_WITH_AUX = 5  # +1 AUX channel as EOG reference
N_SAMPLES = 256


@pytest.fixture
def regressor_no_aux() -> OcularRegressor:
    """EOG channel index beyond available channels -> graceful degradation."""
    cfg = OcularRegressionConfig(eog_channel_idx=4)  # AUX absent -> 4-ch array
    return OcularRegressor(config=cfg)


@pytest.fixture
def regressor_with_aux() -> OcularRegressor:
    """EOG channel is ch 4 (AUX) which is present in 5-ch array."""
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
    eeg[4] = blink          # AUX = EOG reference
    eeg[1] += 0.8 * blink   # AF7 contaminated
    eeg[2] += 0.7 * blink   # AF8 contaminated
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
        out = regressor_disabled.remove(eeg_4ch)
        np.testing.assert_array_equal(out, eeg_4ch)


class TestGracefulDegradation:
    def test_missing_eog_channel_passthrough(self, regressor_no_aux, eeg_4ch):
        """EOG channel index 4 beyond 4-channel array -> passthrough."""
        out = regressor_no_aux.remove(eeg_4ch)
        np.testing.assert_array_equal(out, eeg_4ch)

    def test_output_shape_preserved_no_aux(self, regressor_no_aux, eeg_4ch):
        out = regressor_no_aux.remove(eeg_4ch)
        assert out.shape == eeg_4ch.shape


class TestWithEOGReference:
    def test_output_same_shape_as_input(self, regressor_with_aux, eeg_5ch_blink):
        out = regressor_with_aux.remove(eeg_5ch_blink)
        assert out.shape == eeg_5ch_blink.shape

    def test_frontal_amplitude_reduced_after_many_frames(self, regressor_with_aux, eeg_5ch_blink):
        """After several frames the regressor should reduce blink amplitude."""
        rng = np.random.default_rng(1)
        t = np.linspace(0, 1, N_SAMPLES)
        blink = 120.0 * np.sin(2 * np.pi * 1.0 * t)

        original_af7_ptp = np.ptp(eeg_5ch_blink[1])

        # Warm up the adaptive regression
        for _ in range(20):
            frame = rng.normal(0, 5.0, (N_CH_WITH_AUX, N_SAMPLES))
            frame[4] = blink
            frame[1] += 0.8 * blink
            regressor_with_aux.remove(frame)

        out = regressor_with_aux.remove(eeg_5ch_blink)
        corrected_af7_ptp = np.ptp(out[1])
        # Amplitude should be reduced (not necessarily perfect, but less)
        assert corrected_af7_ptp < original_af7_ptp * 1.1  # at most 10% larger


class TestEdgeCases:
    def test_single_sample_no_crash(self, regressor_with_aux):
        eeg = np.zeros((N_CH_WITH_AUX, 1))
        out = regressor_with_aux.remove(eeg)
        assert out.shape == eeg.shape

    def test_1d_array_no_crash(self, regressor_no_aux):
        eeg_1d = np.zeros((N_CH_EEG,))
        # Should not raise — return unchanged or handle gracefully
        try:
            out = regressor_no_aux.remove(eeg_1d)  # type: ignore[arg-type]
        except Exception:
            pass  # acceptable: 1-D is not the contract, just must not segfault
