"""Unit tests for dsp.cardiac_regression.CardiacRegression."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.cardiac_regression import CardiacRegression


N_EEG = 4
FS = 256


@pytest.fixture()
def regressor() -> CardiacRegression:
    return CardiacRegression(n_eeg_channels=N_EEG, fs=FS)


def _eeg() -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.normal(0, 5e-6, size=(N_EEG,))


def _ppg_sample(amp: float = 1.0) -> float:
    return float(np.sin(2 * np.pi * 1.2 * np.arange(1) / FS)[0]) * amp


class TestConstruction:
    def test_instantiation(self):
        reg = CardiacRegression(n_eeg_channels=N_EEG, fs=FS)
        assert reg is not None

    def test_n_channels_stored(self, regressor):
        assert regressor.n_eeg_channels == N_EEG

    def test_fs_stored(self, regressor):
        assert regressor.fs == FS


class TestProcessOutput:
    def test_output_shape(self, regressor):
        out = regressor.process(_eeg(), _ppg_sample())
        assert out.shape == (N_EEG,)

    def test_output_dtype(self, regressor):
        out = regressor.process(_eeg(), _ppg_sample())
        assert np.issubdtype(out.dtype, np.floating)

    def test_zero_ppg_no_correction(self, regressor):
        eeg = _eeg()
        out = regressor.process(eeg, 0.0)
        np.testing.assert_array_almost_equal(out, eeg)


class TestCalibration:
    def test_calibrate_accepts_data(self, regressor):
        eeg_cal = np.random.default_rng(0).normal(0, 5e-6, size=(FS * 60, N_EEG))
        ppg_cal = np.sin(2 * np.pi * 1.2 * np.arange(FS * 60) / FS)
        regressor.calibrate(eeg_cal, ppg_cal)

    def test_after_calibration_output_shape(self, regressor):
        eeg_cal = np.random.default_rng(0).normal(0, 5e-6, size=(FS * 60, N_EEG))
        ppg_cal = np.sin(2 * np.pi * 1.2 * np.arange(FS * 60) / FS)
        regressor.calibrate(eeg_cal, ppg_cal)
        out = regressor.process(_eeg(), _ppg_sample())
        assert out.shape == (N_EEG,)


class TestReset:
    def test_reset_does_not_raise(self, regressor):
        regressor.reset()

    def test_output_valid_after_reset(self, regressor):
        regressor.reset()
        out = regressor.process(_eeg(), _ppg_sample())
        assert out.shape == (N_EEG,)
        assert not np.any(np.isnan(out))
