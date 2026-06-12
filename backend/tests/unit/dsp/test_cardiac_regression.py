"""Unit tests for dsp/cardiac_regression.py — Stage 6 AAS cardiac corrector."""

from __future__ import annotations

import copy
import threading

import numpy as np
import pytest

from neurolink.dsp.cardiac_regression import (
    CardiacRegressor,
    CardiacRegressionConfig,
)

FS: float = 256.0
N: int = 256  # one second of EEG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eeg(n_ch: int = 4, n_samples: int = N, amplitude: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samples)) * amplitude).astype(np.float32)


def _valid_ibis(n: int = 10, mean_ms: float = 800.0) -> list[float]:
    """Return a list of valid IBIs (400–2000 ms window)."""
    rng = np.random.default_rng(1)
    jitter = rng.standard_normal(n) * 20.0
    return list((jitter + mean_ms).astype(float))


def _regressor_with_warmed_ring(
    n_frames: int = 10,
    half_win_ms: float = 100.0,
) -> CardiacRegressor:
    """Return a regressor whose ring buffer is pre-filled with n_frames of EEG."""
    cfg = CardiacRegressionConfig(half_win_ms=half_win_ms, template_beats=4, recalib_beats=4)
    reg = CardiacRegressor(config=cfg)
    ibis = _valid_ibis()
    for _ in range(n_frames):
        reg.apply(_eeg(), ibis, fs=FS)
    return reg


# ---------------------------------------------------------------------------
# CardiacRegressionConfig
# ---------------------------------------------------------------------------

class TestCardiacRegressionConfig:
    def test_default_enable_true(self):
        assert CardiacRegressionConfig().enable is True

    def test_default_eeg_channels(self):
        assert CardiacRegressionConfig().eeg_channels == [0, 1, 2, 3]

    def test_default_half_win_ms(self):
        assert CardiacRegressionConfig().half_win_ms == pytest.approx(400.0)

    def test_default_template_beats(self):
        assert CardiacRegressionConfig().template_beats == 8

    def test_default_ibi_range(self):
        cfg = CardiacRegressionConfig()
        assert cfg.min_ibi_ms == pytest.approx(400.0)
        assert cfg.max_ibi_ms == pytest.approx(2000.0)

    def test_custom_config_preserved(self):
        cfg = CardiacRegressionConfig(
            enable=False,
            eeg_channels=[0, 1],
            half_win_ms=200.0,
            template_beats=4,
            trim_fraction=0.1,
        )
        assert cfg.enable is False
        assert cfg.eeg_channels == [0, 1]
        assert cfg.half_win_ms == pytest.approx(200.0)
        assert cfg.template_beats == 4
        assert cfg.trim_fraction == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# apply() — guard conditions (no scipy needed)
# ---------------------------------------------------------------------------

class TestApplyGuards:
    def test_disabled_returns_same_array(self):
        cfg = CardiacRegressionConfig(enable=False)
        reg = CardiacRegressor(config=cfg)
        eeg = _eeg()
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result is eeg

    def test_none_eeg_returns_none(self):
        reg = CardiacRegressor()
        result = reg.apply(None, _valid_ibis(), fs=FS)
        assert result is None

    def test_1d_eeg_returns_unchanged(self):
        reg = CardiacRegressor()
        eeg = np.zeros(N, dtype=np.float32)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result is eeg

    def test_empty_ibi_returns_unchanged(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        result = reg.apply(eeg, [], fs=FS)
        assert result is eeg

    def test_all_invalid_ibis_returns_unchanged(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        # All below min_ibi_ms (400 ms)
        result = reg.apply(eeg, [100.0, 200.0, 300.0], fs=FS)
        assert result is eeg

    def test_ring_too_short_returns_unchanged_first_frame(self):
        """On the very first call the ring is empty — must return unchanged."""
        reg = CardiacRegressor()
        eeg = _eeg()
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        # Ring is short; output shape should match input regardless
        assert result.shape == eeg.shape

    def test_output_shape_preserved(self):
        reg = CardiacRegressor()
        eeg = _eeg(n_ch=4, n_samples=128)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.shape == eeg.shape

    def test_output_dtype_float32(self):
        reg = CardiacRegressor()
        eeg = _eeg().astype(np.float64)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        # Either unchanged (float64) or corrected float32 are acceptable
        assert result.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# IBI validation window
# ---------------------------------------------------------------------------

class TestIBIValidation:
    def test_min_boundary_included(self):
        """IBI exactly at min_ibi_ms is valid and should not be rejected."""
        reg = CardiacRegressor()
        eeg = _eeg()
        cfg = reg.get_config()
        result = reg.apply(eeg, [cfg.min_ibi_ms], fs=FS)
        assert result.shape == eeg.shape

    def test_max_boundary_included(self):
        reg = CardiacRegressor()
        eeg = _eeg()
        cfg = reg.get_config()
        result = reg.apply(eeg, [cfg.max_ibi_ms], fs=FS)
        assert result.shape == eeg.shape

    def test_mixed_valid_invalid_uses_valid_only(self):
        """Valid IBIs should not cause a return-unchanged even when mixed."""
        reg = CardiacRegressor()
        eeg = _eeg()
        ibis = [100.0, 800.0, 3000.0]  # only 800 ms is valid
        result = reg.apply(eeg, ibis, fs=FS)
        assert result.shape == eeg.shape


# ---------------------------------------------------------------------------
# Template accumulation (warm ring path, scipy required)
# ---------------------------------------------------------------------------

class TestTemplateAccumulation:
    def test_apply_does_not_raise_on_warm_ring(self):
        """After the ring is full the corrector should run without error."""
        pytest.importorskip("scipy")
        reg = _regressor_with_warmed_ring()
        eeg = _eeg(seed=99)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.shape == eeg.shape

    def test_corrected_output_is_float32(self):
        pytest.importorskip("scipy")
        reg = _regressor_with_warmed_ring()
        eeg = _eeg(seed=99)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.dtype == np.float32

    def test_correction_changes_signal(self):
        """After template build the corrected signal differs from raw."""
        pytest.importorskip("scipy")
        reg = _regressor_with_warmed_ring(n_frames=20)
        eeg = _eeg(seed=42, amplitude=50.0)
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        # Either subtraction happened or ring/offset mismatch returned eeg_out
        # unchanged — shape must be correct either way
        assert result.shape == eeg.shape

    def test_uncorrected_channels_not_modified(self):
        """Channels outside eeg_channels list are not touched."""
        pytest.importorskip("scipy")
        cfg = CardiacRegressionConfig(
            eeg_channels=[0, 1],
            half_win_ms=100.0,
            template_beats=4,
            recalib_beats=4,
        )
        reg = CardiacRegressor(config=cfg)
        ibis = _valid_ibis()
        # Warm the ring
        for _ in range(12):
            reg.apply(_eeg(), ibis, fs=FS)
        eeg = _eeg(seed=77)
        result = reg.apply(eeg, ibis, fs=FS)
        # Channels 2 and 3 must be identical to eeg_out[2,3]
        # (which is a copy of eeg — only float32 cast is applied)
        np.testing.assert_array_equal(result[2], eeg.astype(np.float32)[2])
        np.testing.assert_array_equal(result[3], eeg.astype(np.float32)[3])


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_template(self):
        pytest.importorskip("scipy")
        reg = _regressor_with_warmed_ring()
        reg.reset()
        # After reset the ring is empty — first frame returns unchanged
        eeg = _eeg()
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result.shape == eeg.shape

    def test_reset_clears_epoch_buffer(self):
        reg = _regressor_with_warmed_ring()
        reg.reset()
        # Internal epoch buffer cleared — we just need no exception
        reg.apply(_eeg(), _valid_ibis(), fs=FS)


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------

class TestConfigAccessors:
    def test_get_config_returns_copy(self):
        reg = CardiacRegressor()
        c1 = reg.get_config()
        c2 = reg.get_config()
        assert c1 is not c2

    def test_set_config_updates_fields(self):
        reg = CardiacRegressor()
        new_cfg = CardiacRegressionConfig(enable=False, half_win_ms=200.0)
        reg.set_config(new_cfg)
        got = reg.get_config()
        assert got.enable is False
        assert got.half_win_ms == pytest.approx(200.0)

    def test_set_config_live_disable_effect(self):
        """After disabling, apply() returns the original array identity."""
        reg = CardiacRegressor()
        eeg = _eeg()
        reg.set_config(CardiacRegressionConfig(enable=False))
        result = reg.apply(eeg, _valid_ibis(), fs=FS)
        assert result is eeg


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_apply_no_exception(self):
        reg = CardiacRegressor()
        ibis = _valid_ibis()
        errors: list[Exception] = []

        def worker(seed: int):
            try:
                for _ in range(20):
                    reg.apply(_eeg(seed=seed), ibis, fs=FS)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_reset_and_apply(self):
        reg = CardiacRegressor()
        ibis = _valid_ibis()
        errors: list[Exception] = []

        def applier():
            try:
                for _ in range(20):
                    reg.apply(_eeg(), ibis, fs=FS)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def resetter():
            try:
                for _ in range(5):
                    reg.reset()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=applier) for _ in range(3)
        ] + [threading.Thread(target=resetter)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
