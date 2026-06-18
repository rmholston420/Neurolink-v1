"""Unit tests for Stage 4 — ArtifactSubspaceReconstructor (asr.py).

Covers:
  - Disabled pass-through (enable=False)
  - Calibration accumulation and model fitting
  - Burst artifact reconstruction after calibration
  - Zero-burst frame (clean signal, no change)
  - Stats and state reporting
  - reset() clears state back to CALIBRATING
  - set_config() resets calibration
  - Edge cases: 1-D input, too few samples, degenerate covariance fallback
"""

from __future__ import annotations

import numpy as np

from neurolink.dsp.artifact_config import ASR_BURST_SD, ASR_CALIB_SEC
from neurolink.dsp.asr import (
    ArtifactSubspaceReconstructor,
    ASRConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CH = 4
FS = 256.0
CALIB_SEC = 1.0  # short for tests
SAMPLES_PER_TICK = 64  # 250 ms at 256 Hz

# Number of ticks required to complete calibration:
# calib_samples_needed = CALIB_SEC * FS = 256
# ticks_needed = ceil(256 / 64) = 4
_CALIB_TICKS = 4


def _make_cfg(enable: bool = True, burst_sd: float = ASR_BURST_SD) -> ASRConfig:
    return ASRConfig(
        enable=enable,
        fs=FS,
        calib_sec=CALIB_SEC,
        burst_sd=burst_sd,
        eeg_channels=[0, 1, 2, 3],
    )


def _make_asr(enable: bool = True, burst_sd: float = ASR_BURST_SD) -> ArtifactSubspaceReconstructor:
    return ArtifactSubspaceReconstructor(_make_cfg(enable=enable, burst_sd=burst_sd))


def _clean_frame(n_samples: int = SAMPLES_PER_TICK, scale: float = 10.0) -> np.ndarray:
    """Return a (4, n_samples) array of low-amplitude bandlimited noise."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((N_CH, n_samples)) * scale).astype(np.float32)


def _feed_calibration(asr: ArtifactSubspaceReconstructor, n_frames: int = _CALIB_TICKS) -> None:
    """Feed exactly enough frames to complete calibration (CALIB_SEC * FS samples).

    Default n_frames=_CALIB_TICKS ensures calibration completes on the last
    frame and _frames_processed is reset to 0 with no extra post-calib calls.
    """
    for _ in range(n_frames):
        asr.apply(_clean_frame())


# ---------------------------------------------------------------------------
# Tests: disabled mode
# ---------------------------------------------------------------------------


def test_disabled_pass_through():
    asr = _make_asr(enable=False)
    frame = _clean_frame()
    out = asr.apply(frame)
    assert out is frame  # exact same object returned
    assert asr.get_state() == "DISABLED"


# ---------------------------------------------------------------------------
# Tests: calibration phase
# ---------------------------------------------------------------------------


def test_state_starts_calibrating():
    asr = _make_asr()
    assert asr.get_state() == "CALIBRATING"


def test_calibration_returns_unchanged_frame():
    asr = _make_asr()
    frame = _clean_frame()
    out = asr.apply(frame)
    np.testing.assert_array_equal(out, frame)


def test_calibration_accumulates_samples():
    asr = _make_asr()
    stats_before = asr.get_stats()
    assert stats_before["calib_samples_collected"] == 0
    asr.apply(_clean_frame(n_samples=32))
    stats_after = asr.get_stats()
    assert stats_after["calib_samples_collected"] == 32


def test_transitions_to_ready_after_calibration():
    asr = _make_asr()
    _feed_calibration(asr)
    assert asr.get_state() == "READY"


def test_stats_after_calibration():
    asr = _make_asr()
    _feed_calibration(asr)
    stats = asr.get_stats()
    assert stats["state"] == "READY"
    assert stats["calib_rms"] > 0.0


# ---------------------------------------------------------------------------
# Tests: reconstruction phase
# ---------------------------------------------------------------------------


def test_clean_frame_passes_through_unchanged():
    """A clean frame (well within burst_sd) should be returned nearly unchanged."""
    asr = _make_asr(burst_sd=20.0)
    _feed_calibration(asr)
    frame = _clean_frame(scale=5.0)  # small amplitude
    out = asr.apply(frame)
    assert out.shape == frame.shape
    # Clean frames should not have large corrections.
    np.testing.assert_allclose(out, frame, atol=1.0)


def test_burst_frame_is_corrected():
    """Artificially large amplitude spikes should be reconstructed."""
    asr = _make_asr(burst_sd=2.0)  # aggressive threshold for test
    _feed_calibration(asr)
    # Inject a large spike
    burst = _clean_frame(scale=1.0)
    burst[:, 10:15] = 500.0  # saturate 5 samples
    out = asr.apply(burst)
    # Burst samples should have been pulled back toward calibration subspace
    assert out.shape == burst.shape
    stats = asr.get_stats()
    assert stats["frames_corrected"] >= 1
    assert stats["samples_reconstructed"] >= 1


def test_output_dtype_preserved():
    """Output dtype must match the input dtype."""
    asr = _make_asr()
    _feed_calibration(asr)
    frame_f32 = _clean_frame().astype(np.float32)
    out = asr.apply(frame_f32)
    assert out.dtype == np.float32

    frame_f64 = _clean_frame().astype(np.float64)
    out64 = asr.apply(frame_f64)
    assert out64.dtype == np.float64


def test_frames_processed_increments():
    asr = _make_asr()
    _feed_calibration(asr)
    for _ in range(3):
        asr.apply(_clean_frame())
    assert asr.get_stats()["frames_processed"] == 3


# ---------------------------------------------------------------------------
# Tests: reset and set_config
# ---------------------------------------------------------------------------


def test_reset_returns_to_calibrating():
    asr = _make_asr()
    _feed_calibration(asr)
    assert asr.get_state() == "READY"
    asr.reset()
    assert asr.get_state() == "CALIBRATING"
    stats = asr.get_stats()
    assert stats["calib_samples_collected"] == 0
    assert stats["frames_processed"] == 0


def test_set_config_resets_calibration():
    asr = _make_asr()
    _feed_calibration(asr)
    assert asr.get_state() == "READY"
    new_cfg = _make_cfg(burst_sd=15.0)
    asr.set_config(new_cfg)
    assert asr.get_state() == "CALIBRATING"
    assert asr.get_config().burst_sd == 15.0


def test_set_config_disabled_stays_disabled():
    asr = _make_asr()
    disabled_cfg = _make_cfg(enable=False)
    asr.set_config(disabled_cfg)
    assert asr.get_state() == "DISABLED"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_1d_input_returned_unchanged():
    """A 1-D array (malformed frame) must be returned without error."""
    asr = _make_asr()
    bad = np.zeros(64, dtype=np.float32)
    out = asr.apply(bad)
    assert out is bad


def test_single_sample_frame_returns_unchanged():
    """Frames with < 2 samples are returned unchanged."""
    asr = _make_asr()
    small = np.zeros((4, 1), dtype=np.float32)
    out = asr.apply(small)
    np.testing.assert_array_equal(out, small)


def test_all_eeg_channels_out_of_bounds():
    """If eeg_channels are all >= n_channels the frame is returned unchanged."""
    cfg = _make_cfg()
    cfg = ASRConfig(
        enable=True,
        fs=FS,
        calib_sec=CALIB_SEC,
        burst_sd=ASR_BURST_SD,
        eeg_channels=[10, 11, 12],  # indices beyond 4-channel frame
    )
    asr = ArtifactSubspaceReconstructor(cfg)
    frame = _clean_frame()
    out = asr.apply(frame)
    np.testing.assert_array_equal(out, frame)


def test_get_config_returns_copy():
    """get_config() must return a copy, not the internal reference."""
    asr = _make_asr()
    cfg = asr.get_config()
    cfg.burst_sd = 999.0
    assert asr.get_config().burst_sd != 999.0


def test_default_eeg_channels_auto_detected():
    """When eeg_channels is None it should default to [0,1,2,3]."""
    cfg = ASRConfig(enable=True, fs=256.0)
    assert cfg.eeg_channels == [0, 1, 2, 3]


def test_calib_rms_positive_after_calibration():
    asr = _make_asr()
    _feed_calibration(asr)
    stats = asr.get_stats()
    assert stats["calib_rms"] > 0.0


def test_degenerate_single_channel_covariance():
    """Single-channel subspace (edge case for Cholesky) must not crash."""
    cfg = ASRConfig(
        enable=True,
        fs=FS,
        calib_sec=CALIB_SEC,
        burst_sd=ASR_BURST_SD,
        eeg_channels=[0],  # single channel
    )
    asr = ArtifactSubspaceReconstructor(cfg)
    rng = np.random.default_rng(0)
    # Feed exactly enough single-channel data
    for _ in range(_CALIB_TICKS):
        frame = rng.standard_normal((4, SAMPLES_PER_TICK)).astype(np.float32)
        asr.apply(frame)
    assert asr.get_state() == "READY"
    out = asr.apply(_clean_frame())
    assert out.shape == (4, SAMPLES_PER_TICK)


def test_artifact_config_constants_seeded():
    """ASRConfig defaults must match artifact_config module constants."""
    cfg = ASRConfig()
    assert cfg.burst_sd == ASR_BURST_SD
    assert cfg.calib_sec == ASR_CALIB_SEC
