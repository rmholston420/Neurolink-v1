"""Unit tests for dsp.ocular_regression.OcularRegression."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegression


N_EEG = 4
N_EOG = 2
FS = 256


@pytest.fixture()
def regressor() -> OcularRegression:
    return OcularRegression(n_eeg_channels=N_EEG, n_eog_channels=N_EOG, fs=FS)


def _eeg(amp: float = 5e-6) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, amp, size=(N_EEG,))


def _eog(amp: float = 100e-6) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0, amp, size=(N_EOG,))


class TestConstruction:
    def test_instantiation(self):
        reg = OcularRegression(n_eeg_channels=N_EEG, n_eog_channels=N_EOG, fs=FS)
        assert reg is not None

    def test_n_eeg_stored(self, regressor):
        assert regressor.n_eeg_channels == N_EEG

    def test_n_eog_stored(self, regressor):
        assert regressor.n_eog_channels == N_EOG


class TestProcessOutput:
    def test_output_shape(self, regressor):
        out = regressor.process(_eeg(), _eog())
        assert out.shape == (N_EEG,)

    def test_output_dtype_float(self, regressor):
        out = regressor.process(_eeg(), _eog())
        assert np.issubdtype(out.dtype, np.floating)

    def test_zero_eog_output_equals_input(self, regressor):
        """With zero EOG signal no correction should be applied."""
        eeg = _eeg()
        eog = np.zeros(N_EOG)
        out = regressor.process(eeg, eog)
        # No correction when EOG is zero — output should equal input
        np.testing.assert_array_almost_equal(out, eeg)


class TestCalibration:
    def test_calibrate_does_not_raise(self, regressor):
        eeg_cal = np.random.default_rng(0).normal(0, 5e-6, size=(FS * 30, N_EEG))
        eog_cal = np.random.default_rng(1).normal(0, 100e-6, size=(FS * 30, N_EOG))
        regressor.calibrate(eeg_cal, eog_cal)

    def test_after_calibration_output_shape_unchanged(self, regressor):
        eeg_cal = np.random.default_rng(0).normal(0, 5e-6, size=(FS * 30, N_EEG))
        eog_cal = np.random.default_rng(1).normal(0, 100e-6, size=(FS * 30, N_EOG))
        regressor.calibrate(eeg_cal, eog_cal)
        out = regressor.process(_eeg(), _eog())
        assert out.shape == (N_EEG,)


class TestReset:
    def test_reset_does_not_raise(self, regressor):
        regressor.reset()

    def test_after_reset_output_shape_preserved(self, regressor):
        regressor.reset()
        out = regressor.process(_eeg(), _eog())
        assert out.shape == (N_EEG,)
