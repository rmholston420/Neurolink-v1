"""Unit tests for neurolink.dsp.cardiac_regression.

Public API confirmed from source:
  CardiacRegressor(config)
  .apply(eeg, ibi_ms, fs=256.0) -> np.ndarray   # method is apply(), not remove()
  .reset() / .get_config() / .set_config()
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.cardiac_regression import CardiacRegressor, CardiacRegressionConfig


FS = 256.0
N_CH = 4
N_SAMPLES = 256


@pytest.fixture
def corrector() -> CardiacRegressor:
    return CardiacRegressor()


@pytest.fixture
def corrector_disabled() -> CardiacRegressor:
    return CardiacRegressor(config=CardiacRegressionConfig(enable=False))


@pytest.fixture
def clean_eeg() -> np.ndarray:
    return np.random.default_rng(42).normal(0, 5.0, (N_CH, N_SAMPLES))


@pytest.fixture
def ibi_sequence() -> list[float]:
    """Physiologically valid IBIs: ~75 bpm = 800 ms interval."""
    return [800.0] * 12


class TestConstruction:
    def test_default_construction(self):
        c = CardiacRegressor()
        assert c is not None

    def test_custom_config(self):
        cfg = CardiacRegressionConfig(half_win_ms=300.0, template_beats=6)
        c = CardiacRegressor(config=cfg)
        assert c is not None


class TestDisabledPassthrough:
    def test_disabled_returns_eeg_unchanged(self, corrector_disabled, clean_eeg):
        out = corrector_disabled.apply(clean_eeg, ibi_ms=[], fs=FS)
        np.testing.assert_array_equal(out, clean_eeg)

    def test_disabled_with_valid_ibi_still_passthrough(self, corrector_disabled, clean_eeg, ibi_sequence):
        out = corrector_disabled.apply(clean_eeg, ibi_ms=ibi_sequence, fs=FS)
        np.testing.assert_array_equal(out, clean_eeg)


class TestGracefulDegradation:
    def test_empty_ibi_returns_eeg_unchanged(self, corrector, clean_eeg):
        out = corrector.apply(clean_eeg, ibi_ms=[], fs=FS)
        np.testing.assert_array_equal(out, clean_eeg)

    def test_out_of_range_ibi_returns_unchanged(self, corrector, clean_eeg):
        bad_ibi = [100.0, 3000.0]  # below min_ibi_ms=400 and above max_ibi_ms=2000
        out = corrector.apply(clean_eeg, ibi_ms=bad_ibi, fs=FS)
        np.testing.assert_array_equal(out, clean_eeg)

    def test_insufficient_beats_returns_unchanged(self, corrector, clean_eeg):
        """Only 1 IBI -- fewer than template_beats=8 -> passthrough."""
        out = corrector.apply(clean_eeg, ibi_ms=[800.0], fs=FS)
        np.testing.assert_array_equal(out, clean_eeg)


class TestOutputShape:
    def test_output_same_shape_as_input(self, corrector, clean_eeg, ibi_sequence):
        out = corrector.apply(clean_eeg, ibi_ms=ibi_sequence, fs=FS)
        assert out.shape == clean_eeg.shape

    def test_output_is_numeric(self, corrector, clean_eeg, ibi_sequence):
        out = corrector.apply(clean_eeg, ibi_ms=ibi_sequence, fs=FS)
        assert np.isfinite(out).all()


class TestReset:
    def test_reset_does_not_raise(self, corrector):
        corrector.reset()

    def test_after_reset_still_functional(self, corrector, clean_eeg):
        corrector.reset()
        out = corrector.apply(clean_eeg, ibi_ms=[], fs=FS)
        assert out.shape == clean_eeg.shape


class TestEdgeCases:
    def test_single_channel_no_crash(self, corrector):
        eeg = np.random.default_rng(5).normal(0, 5.0, (1, N_SAMPLES))
        out = corrector.apply(eeg, ibi_ms=[], fs=FS)
        assert out.shape == eeg.shape

    def test_config_roundtrip(self, corrector):
        cfg = corrector.get_config()
        corrector.set_config(cfg)
        assert corrector.get_config().enable == cfg.enable
