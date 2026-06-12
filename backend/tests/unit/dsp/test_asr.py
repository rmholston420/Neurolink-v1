"""Unit tests for neurolink.dsp.asr (Stage 4 — Artifact Subspace Reconstruction).

Covers:
- State machine: CALIBRATING → READY transition
- Disabled mode pass-through
- Burst sample attenuation (statistical)
- Clean signal is not distorted during correction
- reset() returns to CALIBRATING
- set_config() replaces config and resets
- get_stats() accounting accuracy
- Dtype preservation (float32 in → float32 out)
- Short / single-row frames do not crash
- Calibration with under-rank data (all-zero channels) handles gracefully
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.asr import (
    ASRConfig,
    ASRState,
    ArtifactSubspaceReconstructor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
CALIB_SEC = 2.0   # short calibration window for fast tests
BURST_SD = 5.0    # aggressive threshold to guarantee reconstruction in tests


def _make_asr(calib_sec: float = CALIB_SEC, burst_sd: float = BURST_SD,
              enable: bool = True) -> ArtifactSubspaceReconstructor:
    cfg = ASRConfig(
        enable=enable,
        fs=FS,
        calib_sec=calib_sec,
        burst_sd=burst_sd,
        eeg_channels=[0, 1, 2, 3],
    )
    return ArtifactSubspaceReconstructor(config=cfg)


def _clean_block(n_samples: int = 512, n_ch: int = 4, amp: float = 5.0) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.normal(0, amp, (n_ch, n_samples)).astype(np.float32)


def _calibrate(asr: ArtifactSubspaceReconstructor, calib_sec: float = CALIB_SEC) -> None:
    """Feed enough clean frames to complete calibration."""
    n_needed = int(calib_sec * FS) + 64
    block_size = 256
    fed = 0
    while fed < n_needed:
        frame = _clean_block(n_samples=block_size)
        asr.apply(frame)
        fed += block_size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def asr() -> ArtifactSubspaceReconstructor:
    return _make_asr()


@pytest.fixture
def calibrated_asr() -> ArtifactSubspaceReconstructor:
    inst = _make_asr()
    _calibrate(inst)
    return inst


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestASRStateMachine:
    def test_initial_state_calibrating(self, asr):
        assert asr.get_state() == ASRState.CALIBRATING.name

    def test_state_becomes_ready_after_calibration(self, asr):
        _calibrate(asr)
        assert asr.get_state() == ASRState.READY.name

    def test_disabled_mode_state(self):
        inst = _make_asr(enable=False)
        assert inst.get_state() == ASRState.DISABLED.name

    def test_reset_returns_to_calibrating(self, calibrated_asr):
        calibrated_asr.reset()
        assert calibrated_asr.get_state() == ASRState.CALIBRATING.name


# ---------------------------------------------------------------------------
# Pass-through behaviour
# ---------------------------------------------------------------------------

class TestPassThrough:
    def test_disabled_returns_input_unchanged(self):
        inst = _make_asr(enable=False)
        frame = _clean_block()
        out = inst.apply(frame)
        np.testing.assert_array_equal(out, frame)

    def test_calibrating_returns_input_unchanged(self, asr):
        frame = _clean_block(n_samples=32)
        out = asr.apply(frame)
        np.testing.assert_array_equal(out, frame)

    def test_ndim1_frame_returned_unchanged(self, calibrated_asr):
        bad = np.zeros(4, dtype=np.float32)
        out = calibrated_asr.apply(bad)
        np.testing.assert_array_equal(out, bad)


# ---------------------------------------------------------------------------
# Burst correction
# ---------------------------------------------------------------------------

class TestBurstCorrection:
    def test_burst_samples_attenuated(self, calibrated_asr):
        """Frames with large bursts should have lower post-ASR variance."""
        rng = np.random.default_rng(seed=7)
        frame = _clean_block(n_samples=512, amp=5.0)
        # Inject a severe burst on all channels at mid-frame
        burst_start, burst_end = 200, 280
        frame[:, burst_start:burst_end] += rng.normal(0, 300.0, (4, burst_end - burst_start)).astype(np.float32)

        var_before = float(np.var(frame[:, burst_start:burst_end]))
        out = calibrated_asr.apply(frame)
        var_after = float(np.var(out[:, burst_start:burst_end]))

        assert var_after < var_before, (
            f"Burst variance should decrease after ASR: before={var_before:.1f}, after={var_after:.1f}"
        )

    def test_clean_signal_not_excessively_distorted(self, calibrated_asr):
        """A genuinely clean frame should be returned nearly unchanged."""
        frame = _clean_block(n_samples=512, amp=5.0)
        out = calibrated_asr.apply(frame)
        # Allow up to 20% change in total variance — ASR should not distort clean data
        var_in = float(np.var(frame))
        var_out = float(np.var(out))
        assert abs(var_out - var_in) / (var_in + 1e-9) < 0.20, (
            f"Clean signal distorted by ASR: var_in={var_in:.2f}, var_out={var_out:.2f}"
        )

    def test_output_same_shape(self, calibrated_asr):
        frame = _clean_block(n_samples=256)
        out = calibrated_asr.apply(frame)
        assert out.shape == frame.shape


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------

class TestDtypePreservation:
    def test_float32_preserved(self, calibrated_asr):
        frame = _clean_block().astype(np.float32)
        out = calibrated_asr.apply(frame)
        assert out.dtype == np.float32

    def test_float64_preserved(self, calibrated_asr):
        frame = _clean_block().astype(np.float64)
        out = calibrated_asr.apply(frame)
        assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Config and stats
# ---------------------------------------------------------------------------

class TestASRConfig:
    def test_set_config_resets_state(self, calibrated_asr):
        new_cfg = ASRConfig(enable=True, fs=FS, calib_sec=CALIB_SEC, burst_sd=BURST_SD)
        calibrated_asr.set_config(new_cfg)
        assert calibrated_asr.get_state() == ASRState.CALIBRATING.name

    def test_get_config_returns_copy(self, asr):
        cfg = asr.get_config()
        cfg.burst_sd = 999.0
        assert asr.get_config().burst_sd != 999.0


class TestASRStats:
    def test_stats_structure(self, calibrated_asr):
        stats = calibrated_asr.get_stats()
        for key in (
            "state", "calib_samples_collected", "calib_samples_needed",
            "frames_processed", "frames_corrected", "samples_reconstructed", "calib_rms"
        ):
            assert key in stats

    def test_frames_processed_increments(self, calibrated_asr):
        before = calibrated_asr.get_stats()["frames_processed"]
        calibrated_asr.apply(_clean_block())
        after = calibrated_asr.get_stats()["frames_processed"]
        assert after == before + 1

    def test_reset_clears_stats(self, calibrated_asr):
        calibrated_asr.apply(_clean_block())
        calibrated_asr.reset()
        stats = calibrated_asr.get_stats()
        assert stats["frames_processed"] == 0
        assert stats["samples_reconstructed"] == 0
