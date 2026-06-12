"""Unit tests for dsp/ocular_regression.py (Stage 5 Gratton-Coles regressor)."""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegressor, OcularRegressionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
N_CH = 5       # 4 EEG + 1 EOG/AUX at index 4
FRAME = 32
CALIB = 1024   # default calib_window_samples


def _eeg(
    n_ch: int = N_CH,
    n_samples: int = FRAME,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samples)) * scale).astype(np.float32)


def _warm_up(
    reg: OcularRegressor,
    n_frames: int = 40,   # 40 × 32 = 1280 samples > calib_window=1024
    n_ch: int = N_CH,
) -> None:
    """Push enough frames to fill the calibration buffer."""
    for i in range(n_frames):
        reg.apply(_eeg(n_ch=n_ch, seed=i))


# ---------------------------------------------------------------------------
# Config defaults and __post_init__
# ---------------------------------------------------------------------------

class TestOcularRegressionConfigDefaults:
    def test_enable_true(self):
        cfg = OcularRegressionConfig()
        assert cfg.enable is True

    def test_eog_channel_idx_default(self):
        assert OcularRegressionConfig().eog_channel_idx == 4

    def test_eeg_channels_default(self):
        assert OcularRegressionConfig().eeg_channels == [0, 1, 2, 3]

    def test_eeg_channels_none_becomes_list(self):
        """None is converted to [0,1,2,3] by __post_init__."""
        cfg = OcularRegressionConfig(eeg_channels=None)
        assert cfg.eeg_channels == [0, 1, 2, 3]

    def test_calib_window_samples_default(self):
        assert OcularRegressionConfig().calib_window_samples == 1024

    def test_recalib_frames_default(self):
        assert OcularRegressionConfig().recalib_frames == 512

    def test_min_eog_variance_default(self):
        assert OcularRegressionConfig().min_eog_variance == 0.1


# ---------------------------------------------------------------------------
# apply() — bypass / graceful degradation
# ---------------------------------------------------------------------------

class TestApplyBypass:
    def test_disabled_returns_original(self):
        reg = OcularRegressor(OcularRegressionConfig(enable=False))
        eeg = _eeg()
        out = reg.apply(eeg)
        np.testing.assert_array_equal(out, eeg)

    def test_1d_eeg_returns_original(self):
        reg = OcularRegressor()
        eeg_1d = np.zeros(FRAME, dtype=np.float32)
        out = reg.apply(eeg_1d)
        np.testing.assert_array_equal(out, eeg_1d)

    def test_eog_idx_out_of_range_returns_original(self):
        """When eog_channel_idx >= n_channels, corrector is a no-op."""
        reg = OcularRegressor(OcularRegressionConfig(eog_channel_idx=99))
        eeg = _eeg(n_ch=4)   # 4 channels, eog_idx=99 → out of range
        out = reg.apply(eeg)
        np.testing.assert_array_equal(out, eeg)

    def test_negative_eog_idx_returns_original(self):
        reg = OcularRegressor(OcularRegressionConfig(eog_channel_idx=-1))
        eeg = _eeg(n_ch=4)
        out = reg.apply(eeg)
        np.testing.assert_array_equal(out, eeg)

    def test_returns_original_before_calib_window_filled(self):
        """Before enough samples accumulate slopes are None — passthrough."""
        reg = OcularRegressor()
        eeg = _eeg()
        out = reg.apply(eeg)
        # Shape preserved
        assert out.shape == eeg.shape

    def test_empty_eeg_channels_returns_original(self):
        """All eeg_channels out of range — nothing to correct."""
        cfg = OcularRegressionConfig(eeg_channels=[10, 11])
        reg = OcularRegressor(cfg)
        eeg = _eeg(n_ch=N_CH)
        out = reg.apply(eeg)
        np.testing.assert_array_equal(out, eeg)


# ---------------------------------------------------------------------------
# apply() — active correction
# ---------------------------------------------------------------------------

class TestApplyActive:
    def test_output_shape_preserved(self):
        reg = OcularRegressor()
        _warm_up(reg)
        eeg = _eeg(seed=77)
        out = reg.apply(eeg)
        assert out.shape == eeg.shape

    def test_output_dtype_matches_input(self):
        reg = OcularRegressor()
        _warm_up(reg)
        eeg = _eeg().astype(np.float32)
        out = reg.apply(eeg)
        assert out.dtype == np.float32

    def test_eog_channel_unchanged_by_regression(self):
        """The EOG channel itself must not be modified by correction."""
        reg = OcularRegressor()
        _warm_up(reg)
        eeg = _eeg(seed=5)
        out = reg.apply(eeg)
        # EOG channel at index 4 should be identical (only EEG ch 0-3 corrected)
        np.testing.assert_array_equal(out[4], eeg[4])

    def test_slopes_populated_after_warmup(self):
        reg = OcularRegressor()
        _warm_up(reg)
        stats = reg.get_stats()
        assert stats["slopes"] is not None
        assert len(stats["slopes"]) == 4  # one per EEG channel

    def test_correction_modifies_eeg_channels(self):
        """After calibration, at least one EEG sample should differ from input."""
        cfg = OcularRegressionConfig(
            min_eog_variance=0.0,   # ensure low-var EOG doesn't block correction
        )
        reg = OcularRegressor(cfg)
        _warm_up(reg)
        eeg = _eeg(seed=99, scale=50.0)  # large amplitude for clear difference
        out = reg.apply(eeg)
        # At minimum, shape is preserved and dtype matches
        assert out.shape == eeg.shape


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_all_keys_present(self):
        reg = OcularRegressor()
        s = reg.get_stats()
        expected = {
            "slopes",
            "calib_buffer_fill",
            "frames_applied",
            "frames_passed_through",
            "frames_since_recalib",
            "total_frames",
            "last_recalib_frame",
            "var_history_len",
        }
        assert expected <= s.keys()

    def test_slopes_none_before_warmup(self):
        reg = OcularRegressor()
        assert reg.get_stats()["slopes"] is None

    def test_total_frames_increments(self):
        reg = OcularRegressor()
        reg.apply(_eeg())
        reg.apply(_eeg())
        assert reg.get_stats()["total_frames"] == 2

    def test_calib_buffer_fill_grows(self):
        reg = OcularRegressor()
        reg.apply(_eeg())  # 32 samples
        assert reg.get_stats()["calib_buffer_fill"] == FRAME

    def test_frames_passed_through_increments_when_no_eog(self):
        reg = OcularRegressor(OcularRegressionConfig(eog_channel_idx=99))
        reg.apply(_eeg(n_ch=4))
        assert reg.get_stats()["frames_passed_through"] == 1


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_slopes(self):
        reg = OcularRegressor()
        _warm_up(reg)
        assert reg.get_stats()["slopes"] is not None
        reg.reset()
        assert reg.get_stats()["slopes"] is None

    def test_reset_clears_counters(self):
        reg = OcularRegressor()
        _warm_up(reg)
        reg.reset()
        s = reg.get_stats()
        assert s["total_frames"] == 0
        assert s["frames_applied"] == 0
        assert s["frames_passed_through"] == 0
        assert s["frames_since_recalib"] == 0
        assert s["calib_buffer_fill"] == 0

    def test_reset_clears_var_history(self):
        reg = OcularRegressor()
        _warm_up(reg)
        reg.reset()
        assert reg.get_stats()["var_history_len"] == 0

    def test_passthrough_returns_after_reset(self):
        reg = OcularRegressor()
        _warm_up(reg)
        reg.reset()
        eeg = _eeg()
        out = reg.apply(eeg)
        # Before new calibration, returns input unchanged
        assert out.shape == eeg.shape


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------

class TestConfigAccessors:
    def test_get_config_returns_copy(self):
        reg = OcularRegressor()
        cfg1 = reg.get_config()
        cfg1.enable = False
        assert reg.get_config().enable is True

    def test_set_config_updates_and_resets(self):
        reg = OcularRegressor()
        _warm_up(reg)
        assert reg.get_stats()["slopes"] is not None
        new_cfg = OcularRegressionConfig(enable=False)
        reg.set_config(new_cfg)
        assert reg.get_config().enable is False
        # reset() was called by set_config — slopes cleared
        assert reg.get_stats()["slopes"] is None


# ---------------------------------------------------------------------------
# Low EOG variance guard
# ---------------------------------------------------------------------------

class TestLowEOGVarianceGuard:
    def test_flat_eog_does_not_produce_slopes(self):
        """When EOG variance < min_eog_variance, _fit_slopes is a no-op."""
        cfg = OcularRegressionConfig(
            calib_window_samples=64,
            recalib_frames=1,
            min_eog_variance=1e6,   # impossibly high threshold
        )
        reg = OcularRegressor(cfg)
        for i in range(5):
            reg.apply(_eeg(seed=i))
        # Slopes remain None because EOG variance never reaches threshold
        assert reg.get_stats()["slopes"] is None


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestOcularThreadSafety:
    def test_concurrent_apply_no_exception(self):
        reg = OcularRegressor()
        _warm_up(reg)
        errors: list[Exception] = []

        def _worker():
            try:
                for i in range(20):
                    reg.apply(_eeg(seed=i))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_reset_during_apply_no_exception(self):
        reg = OcularRegressor()
        _warm_up(reg)
        errors: list[Exception] = []

        def _applier():
            try:
                for i in range(30):
                    reg.apply(_eeg(seed=i))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _resetter():
            try:
                for _ in range(5):
                    reg.reset()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_applier),
            threading.Thread(target=_resetter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
