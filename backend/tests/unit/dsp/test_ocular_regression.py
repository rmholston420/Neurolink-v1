"""Unit tests for dsp/ocular_regression.py."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegressor

FS = 256.0
N_CH = 5


def _eeg(seed: int = 0, n_samples: int = 32) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_CH, n_samples)).astype(np.float32)


def _warm_up(
    reg: OcularRegressor,
    n_frames: int = 40,  # 40 x 32 = 1280 samples > calib_window=1024
    n_ch: int = N_CH,
) -> None:
    for _ in range(n_frames):
        reg.apply(_eeg())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestOcularRegressorInit:
    def test_default_init(self):
        reg = OcularRegressor()
        assert reg is not None

    def test_custom_fs(self):
        reg = OcularRegressor(fs=128.0)
        assert reg.fs == pytest.approx(128.0)


# ---------------------------------------------------------------------------
# apply() before warmup
# ---------------------------------------------------------------------------


class TestOcularRegressorBeforeWarmup:
    def test_output_shape_preserved(self):
        reg = OcularRegressor()
        eeg = _eeg()
        result = reg.apply(eeg)
        assert result.shape == eeg.shape

    def test_output_dtype_preserved(self):
        reg = OcularRegressor()
        eeg = _eeg().astype(np.float32)
        result = reg.apply(eeg)
        assert result.dtype == np.float32

    def test_no_exception_on_first_call(self):
        reg = OcularRegressor()
        reg.apply(_eeg())  # should not raise


# ---------------------------------------------------------------------------
# apply() after warmup
# ---------------------------------------------------------------------------


class TestOcularRegressorAfterWarmup:
    def test_output_shape_after_warmup(self):
        reg = OcularRegressor()
        _warm_up(reg)
        result = reg.apply(_eeg())
        assert result.shape == (N_CH, 32)

    def test_output_dtype_after_warmup(self):
        reg = OcularRegressor()
        _warm_up(reg)
        result = reg.apply(_eeg().astype(np.float32))
        assert result.dtype == np.float32

    def test_output_finite_after_warmup(self):
        reg = OcularRegressor()
        _warm_up(reg)
        result = reg.apply(_eeg())
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestOcularRegressorReset:
    def test_reset_clears_buffer(self):
        reg = OcularRegressor()
        _warm_up(reg)
        reg.reset()
        # After reset, regression should not be active
        eeg = _eeg(seed=77)
        result = reg.apply(eeg)
        np.testing.assert_array_equal(result, eeg)


# ---------------------------------------------------------------------------
# None / bad input
# ---------------------------------------------------------------------------


class TestOcularRegressorBadInput:
    def test_none_input_returns_none(self):
        reg = OcularRegressor()
        result = reg.apply(None)
        assert result is None

    def test_1d_input_returns_unchanged(self):
        reg = OcularRegressor()
        eeg_1d = np.zeros(32, dtype=np.float32)
        result = reg.apply(eeg_1d)
        np.testing.assert_array_equal(result, eeg_1d)
