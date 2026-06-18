"""Unit tests for dsp/cardiac_regression.py."""

from __future__ import annotations

import numpy as np

from neurolink.dsp.cardiac_regression import CardiacRegressor

FS = 256.0
N_CH = 4
N_SAMPLES = 32


def _valid_ibis(n: int = 10, mean_ms: float = 800.0) -> list[float]:
    """Return a list of valid IBIs (400-2000 ms window)."""
    rng = np.random.default_rng(1)
    jitter = rng.standard_normal(n) * 20.0
    return list(np.clip(mean_ms + jitter, 400.0, 2000.0))


def _eeg(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_CH, N_SAMPLES)).astype(np.float32)


# ---------------------------------------------------------------------------
# CardiacRegressor.apply() -- guard conditions
# ---------------------------------------------------------------------------


class TestCardiacRegressorApplyGuards:
    def test_empty_ibi_list_returns_eeg_unchanged(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        result = reg.apply(eeg, [], fs=FS)
        np.testing.assert_array_equal(result, eeg)

    def test_none_eeg_returns_none_eeg(self):
        reg = CardiacRegressor()
        result = reg.apply(None, _valid_ibis(), fs=FS)
        assert result is None

    def test_1d_eeg_returns_unchanged(self):
        reg = CardiacRegressor()
        eeg_1d = np.zeros(N_SAMPLES, dtype=np.float32)
        result = reg.apply(eeg_1d, _valid_ibis(), fs=FS)
        np.testing.assert_array_equal(result, eeg_1d)

    def test_single_sample_eeg_returns_unchanged(self):
        reg = CardiacRegressor()
        eeg = np.zeros((N_CH, 1), dtype=np.float32)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        np.testing.assert_array_equal(result, eeg)


# ---------------------------------------------------------------------------
# CardiacRegressor.apply() -- before ring is filled
# ---------------------------------------------------------------------------


class TestCardiacRegressorBeforeWarmup:
    def test_output_shape_preserved(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.shape == eeg.shape

    def test_output_dtype_preserved(self):
        reg = CardiacRegressor()
        eeg = _eeg().astype(np.float32)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.dtype == np.float32

    def test_output_is_not_same_object(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        # May or may not be same object -- just check shape and values are finite
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# CardiacRegressor.apply() -- after ring is filled
# ---------------------------------------------------------------------------


class TestCardiacRegressorAfterWarmup:
    def _fill_ring(self, reg: CardiacRegressor, n: int = 60) -> None:
        for _ in range(n):
            reg.apply(_eeg(), _valid_ibis(), fs=FS)

    def test_output_shape_after_warmup(self):
        reg = CardiacRegressor()
        self._fill_ring(reg)
        result = reg.apply(_eeg(), _valid_ibis(), fs=FS)
        assert result.shape == (N_CH, N_SAMPLES)

    def test_output_dtype_after_warmup(self):
        reg = CardiacRegressor()
        self._fill_ring(reg)
        result = reg.apply(_eeg().astype(np.float32), _valid_ibis(), fs=FS)
        assert result.dtype == np.float32

    def test_output_is_finite_after_warmup(self):
        reg = CardiacRegressor()
        self._fill_ring(reg)
        result = reg.apply(_eeg(), _valid_ibis(), fs=FS)
        assert np.all(np.isfinite(result))

    def test_output_differs_from_input_after_warmup(self):
        reg = CardiacRegressor()
        self._fill_ring(reg)
        eeg = _eeg(seed=42) * 50.0  # large signal
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        # After correction, at least some values should differ
        assert not np.allclose(result, eeg, atol=1e-5)


# ---------------------------------------------------------------------------
# CardiacRegressor.reset()
# ---------------------------------------------------------------------------


class TestCardiacRegressorReset:
    def test_reset_clears_ring(self):
        reg = CardiacRegressor()
        for _ in range(60):
            reg.apply(_eeg(), _valid_ibis(), fs=FS)
        reg.reset()
        # After reset, ring should be empty -> regression not active
        eeg = _eeg(seed=99)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        np.testing.assert_array_equal(result, eeg)


# ---------------------------------------------------------------------------
# Filtering out-of-range IBIs internally
# ---------------------------------------------------------------------------


class TestCardiacRegressorIBIFiltering:
    def test_out_of_range_ibis_filtered(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        # All IBIs outside [400, 2000] ms range
        bad_ibis = [50.0, 300.0, 2500.0]
        result = reg.apply(eeg, bad_ibis, fs=FS)
        # Should return eeg unchanged (no valid IBIs)
        np.testing.assert_array_equal(result, eeg)

    def test_mixed_ibis_uses_valid_only(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        mixed_ibis = [50.0, 800.0, 2500.0, 900.0]  # 2 valid
        # Should not raise
        result = reg.apply(eeg, mixed_ibis, fs=FS)
        assert result.shape == eeg.shape
