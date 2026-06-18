"""Unit tests for dsp.baseline.BaselineNormalizer."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.baseline import BaselineNormalizer


N_CH = 4
FS = 256
WINDOW_S = 5.0


@pytest.fixture()
def normalizer() -> BaselineNormalizer:
    return BaselineNormalizer(n_channels=N_CH, fs=FS, window_s=WINDOW_S)


def _frame(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(offset, 1e-6, size=(N_CH,))


class TestConstruction:
    def test_instantiation(self):
        bn = BaselineNormalizer(n_channels=N_CH, fs=FS, window_s=WINDOW_S)
        assert bn is not None

    def test_window_stored(self, normalizer):
        assert normalizer.window_s == WINDOW_S

    def test_channel_count(self, normalizer):
        assert normalizer.n_channels == N_CH


class TestNormalization:
    def test_output_shape(self, normalizer):
        out = normalizer.update(_frame())
        assert out.shape == (N_CH,)

    def test_output_dtype(self, normalizer):
        out = normalizer.update(_frame())
        assert np.issubdtype(out.dtype, np.floating)

    def test_after_warmup_mean_near_zero(self, normalizer):
        """After a full window of identical data the normalized mean should be ≈0."""
        n_samples = int(FS * WINDOW_S)
        data = np.ones((N_CH,)) * 50e-6
        for _ in range(n_samples):
            out = normalizer.update(data)
        assert np.allclose(np.abs(out), 0.0, atol=1e-5)

    def test_zero_input_returns_finite(self, normalizer):
        out = normalizer.update(np.zeros(N_CH))
        assert np.all(np.isfinite(out))


class TestReset:
    def test_reset_does_not_raise(self, normalizer):
        for _ in range(10):
            normalizer.update(_frame())
        normalizer.reset()

    def test_after_reset_output_valid(self, normalizer):
        for _ in range(int(FS * WINDOW_S)):
            normalizer.update(_frame())
        normalizer.reset()
        out = normalizer.update(_frame())
        assert out.shape == (N_CH,)
        assert np.all(np.isfinite(out))


class TestFixedMode:
    def test_fixed_baseline_mode(self):
        bn = BaselineNormalizer(n_channels=N_CH, fs=FS, window_s=WINDOW_S, mode="fixed")
        ref = np.ones(N_CH) * 10e-6
        bn.set_reference(ref)
        out = bn.update(ref * 2)
        # In fixed mode the baseline is subtracted — output should be ≈ ref
        np.testing.assert_allclose(out, ref, atol=1e-8)
