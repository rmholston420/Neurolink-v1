"""Unit tests for Stage 5 — OcularRegressor (ocular_regression.py).

Covers:
  - Disabled pass-through (enable=False)
  - Graceful degradation when EOG channel index is out of bounds
  - Rolling buffer accumulation
  - OLS slope fitting after calib_window_samples
  - Correction applied to EEG channels
  - Recalibration scheduling (frames_since_recalib >= recalib_frames)
  - Low-variance EOG skips slope update
  - reset() clears all state
  - set_config() resets and uses new config
  - get_stats() structure
  - Malformed input (1-D array) returned unchanged
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.ocular_regression import OcularRegressor, OcularRegressionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CH = 5   # 4 EEG + 1 EOG (index 4)
FS = 256.0
CALIB_WIN = 128  # small window for fast tests
SAMPLES_PER_TICK = 32


def _make_cfg(
    enable: bool = True,
    eog_idx: int = 4,
    calib_win: int = CALIB_WIN,
    recalib_frames: int = 1000,
    min_var: float = 0.1,
) -> OcularRegressionConfig:
    return OcularRegressionConfig(
        enable=enable,
        eog_channel_idx=eog_idx,
        eeg_channels=[0, 1, 2, 3],
        calib_window_samples=calib_win,
        recalib_frames=recalib_frames,
        min_eog_variance=min_var,
    )


def _make_reg(**kwargs) -> OcularRegressor:
    return OcularRegressor(_make_cfg(**kwargs))


def _make_frame(
    eog_amplitude: float = 50.0,
    eeg_scale: float = 10.0,
    n_samples: int = SAMPLES_PER_TICK,
    seed: int = 7,
) -> np.ndarray:
    """Return a (5, n_samples) float32 frame with realistic EOG content."""
    rng = np.random.default_rng(seed)
    eeg = rng.standard_normal((4, n_samples)) * eeg_scale
    # EOG: sine wave + noise simulating blink
    t = np.linspace(0, 1, n_samples)
    eog = (np.sin(2 * np.pi * 2 * t) * eog_amplitude + rng.standard_normal(n_samples) * 2.0)
    frame = np.vstack([eeg, eog[np.newaxis, :]]).astype(np.float32)
    return frame


def _feed_calibration(reg: OcularRegressor, n_frames: int = 6) -> None:
    """Feed enough frames to exceed calib_window_samples."""
    for i in range(n_frames):
        reg.apply(_make_frame(seed=i))


# ---------------------------------------------------------------------------
# Tests: disabled mode
# ---------------------------------------------------------------------------


def test_disabled_pass_through():
    reg = _make_reg(enable=False)
    frame = _make_frame()
    out = reg.apply(frame)
    assert out is frame


# ---------------------------------------------------------------------------
# Tests: graceful degradation — no EOG channel
# ---------------------------------------------------------------------------


def test_no_eog_channel_pass_through_negative_idx():
    reg = _make_reg(eog_idx=-1)
    frame = _make_frame()
    out = reg.apply(frame)
    np.testing.assert_array_equal(out, frame)
    assert reg.get_stats()["frames_passed_through"] == 1


def test_no_eog_channel_pass_through_oob_idx():
    """EOG index >= n_channels → pass-through without error."""
    reg = _make_reg(eog_idx=99)
    frame = _make_frame()
    out = reg.apply(frame)
    np.testing.assert_array_equal(out, frame)


# ---------------------------------------------------------------------------
# Tests: calibration accumulation
# ---------------------------------------------------------------------------


def test_pass_through_before_calibration():
    """Before enough data is accumulated slopes are None; frame returned unchanged."""
    reg = _make_reg(calib_win=10000)  # very large window — never fills
    frame = _make_frame()
    out = reg.apply(frame)
    np.testing.assert_array_equal(out, frame)
    assert reg.get_stats()["slopes"] is None


def test_buffer_fills_after_enough_frames():
    reg = _make_reg()
    _feed_calibration(reg)
    stats = reg.get_stats()
    assert stats["calib_buffer_fill"] == CALIB_WIN  # deque capped at maxlen


def test_slopes_fitted_after_calibration():
    reg = _make_reg()
    _feed_calibration(reg)
    stats = reg.get_stats()
    assert stats["slopes"] is not None
    assert len(stats["slopes"]) == 4  # one per EEG channel


# ---------------------------------------------------------------------------
# Tests: correction applied
# ---------------------------------------------------------------------------


def test_correction_changes_eeg_channels():
    """After calibration the corrected frame must differ from the raw input."""
    reg = _make_reg()
    _feed_calibration(reg)
    frame = _make_frame(eog_amplitude=200.0)  # strong EOG for visible effect
    out = reg.apply(frame)
    assert out.shape == frame.shape
    # EEG channels (0-3) should differ; EOG channel (4) should be unchanged.
    eeg_changed = not np.allclose(out[:4], frame[:4], atol=0.0)
    assert eeg_changed


def test_eog_channel_not_modified():
    reg = _make_reg()
    _feed_calibration(reg)
    frame = _make_frame()
    out = reg.apply(frame)
    np.testing.assert_array_equal(out[4], frame[4])


def test_output_dtype_preserved():
    reg = _make_reg()
    _feed_calibration(reg)
    frame_f32 = _make_frame().astype(np.float32)
    out = reg.apply(frame_f32)
    assert out.dtype == np.float32


def test_frames_applied_increments():
    reg = _make_reg()
    _feed_calibration(reg)
    for i in range(3):
        reg.apply(_make_frame(seed=i + 100))
    assert reg.get_stats()["frames_applied"] >= 3


# ---------------------------------------------------------------------------
# Tests: recalibration scheduling
# ---------------------------------------------------------------------------


def test_recalibration_triggers_on_schedule():
    """After recalib_frames ticks a new slope fit should be triggered."""
    reg = _make_reg(recalib_frames=3)
    _feed_calibration(reg)
    initial_slopes = list(reg.get_stats()["slopes"])
    # Feed recalib_frames + 1 more frames with different signal
    for i in range(4):
        reg.apply(_make_frame(eog_amplitude=100.0 + i * 20, seed=i + 200))
    # frames_since_recalib should have reset at least once
    stats = reg.get_stats()
    assert stats["frames_since_recalib"] < 4


# ---------------------------------------------------------------------------
# Tests: low-variance skip
# ---------------------------------------------------------------------------


def test_low_variance_eog_skips_slope_update():
    """A flat EOG channel (zero variance) must not update slopes."""
    reg = _make_reg(min_var=1000.0)  # threshold so high it is never met
    # Feed normally to accumulate buffer
    for i in range(6):
        reg.apply(_make_frame(seed=i))
    # Slopes should remain None because variance never exceeds threshold
    assert reg.get_stats()["slopes"] is None


# ---------------------------------------------------------------------------
# Tests: reset and set_config
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    reg = _make_reg()
    _feed_calibration(reg)
    assert reg.get_stats()["slopes"] is not None
    reg.reset()
    stats = reg.get_stats()
    assert stats["slopes"] is None
    assert stats["calib_buffer_fill"] == 0
    assert stats["frames_applied"] == 0
    assert stats["frames_passed_through"] == 0


def test_set_config_resets_and_uses_new_config():
    reg = _make_reg()
    _feed_calibration(reg)
    new_cfg = _make_cfg(eog_idx=4, calib_win=64)
    reg.set_config(new_cfg)
    stats = reg.get_stats()
    assert stats["slopes"] is None
    assert reg.get_config().calib_window_samples == 64


# ---------------------------------------------------------------------------
# Tests: get_stats structure
# ---------------------------------------------------------------------------


def test_get_stats_keys():
    reg = _make_reg()
    stats = reg.get_stats()
    expected = {
        "slopes",
        "calib_buffer_fill",
        "frames_applied",
        "frames_passed_through",
        "frames_since_recalib",
    }
    assert expected.issubset(stats.keys())


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_1d_input_returned_unchanged():
    reg = _make_reg()
    bad = np.zeros(64, dtype=np.float32)
    out = reg.apply(bad)
    assert out is bad


def test_get_config_returns_copy():
    reg = _make_reg()
    cfg = reg.get_config()
    cfg.min_eog_variance = 9999.0
    assert reg.get_config().min_eog_variance != 9999.0


def test_default_eeg_channels_auto_detected():
    cfg = OcularRegressionConfig()
    assert cfg.eeg_channels == [0, 1, 2, 3]
