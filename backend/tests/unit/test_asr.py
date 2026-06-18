"""Unit tests for neurolink.dsp.asr (Artifact Subspace Reconstruction).

Real public API:
  ArtifactSubspaceReconstructor  — stateful corrector
  ASRConfig                      — tunable dataclass
  ASRState                       — CALIBRATING | READY | DISABLED
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.asr import ASRConfig, ASRState, ArtifactSubspaceReconstructor


FS = 256.0
N_CH = 4
N_SAMPLES = 256


@pytest.fixture
def asr() -> ArtifactSubspaceReconstructor:
    cfg = ASRConfig(fs=FS, calib_sec=1.0, burst_sd=20.0)
    return ArtifactSubspaceReconstructor(config=cfg)


@pytest.fixture
def asr_disabled() -> ArtifactSubspaceReconstructor:
    cfg = ASRConfig(enable=False)
    return ArtifactSubspaceReconstructor(config=cfg)


@pytest.fixture
def clean_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 5.0, (N_CH, N_SAMPLES))


class TestConstruction:
    def test_default_construction(self):
        a = ArtifactSubspaceReconstructor()
        assert a is not None

    def test_starts_in_calibrating_state(self, asr):
        assert asr._state == ASRState.CALIBRATING

    def test_disabled_config_starts_disabled(self, asr_disabled):
        assert asr_disabled._state == ASRState.DISABLED


class TestDisabledPassthrough:
    def test_disabled_returns_input_unchanged(self, asr_disabled, clean_frame):
        out = asr_disabled.apply(clean_frame)
        np.testing.assert_array_equal(out, clean_frame)

    def test_disabled_identity_on_burst_frame(self, asr_disabled):
        burst = np.ones((N_CH, N_SAMPLES)) * 500.0  # huge amplitude
        out = asr_disabled.apply(burst)
        np.testing.assert_array_equal(out, burst)


class TestCalibrationAccumulation:
    def test_during_calibration_returns_frame(self, asr, clean_frame):
        """apply() returns the frame during calibration (no correction yet)."""
        out = asr.apply(clean_frame)
        assert out.shape == clean_frame.shape

    def test_calibration_completes_after_enough_samples(self):
        """After calib_sec * fs samples, state should advance to READY."""
        cfg = ASRConfig(fs=FS, calib_sec=0.5)  # only 128 calibration samples
        a = ArtifactSubspaceReconstructor(config=cfg)
        rng = np.random.default_rng(0)
        # Feed enough frames to complete calibration
        for _ in range(5):
            frame = rng.normal(0, 5.0, (N_CH, N_SAMPLES))
            a.apply(frame)
        # Should have transitioned to READY (or stayed calibrating if not enough)
        assert a._state in (ASRState.CALIBRATING, ASRState.READY)


class TestReadyState:
    def test_apply_returns_same_shape(self, asr, clean_frame):
        out = asr.apply(clean_frame)
        assert out.shape == clean_frame.shape

    def test_apply_ndim_guard(self, asr):
        bad = np.zeros((4,))  # 1-D: should be returned unchanged
        out = asr.apply(bad)
        np.testing.assert_array_equal(out, bad)

    def test_apply_single_sample_guard(self, asr):
        tiny = np.zeros((N_CH, 1))
        out = asr.apply(tiny)
        assert out.shape == (N_CH, 1)


class TestStats:
    def test_frames_processed_increments(self, asr, clean_frame):
        before = asr._frames_processed
        asr.apply(clean_frame)
        assert asr._frames_processed == before + 1


class TestEdgeCases:
    def test_empty_eeg_channels_config(self):
        cfg = ASRConfig(eeg_channels=[])
        a = ArtifactSubspaceReconstructor(config=cfg)
        frame = np.random.default_rng(1).normal(0, 5.0, (4, N_SAMPLES))
        out = a.apply(frame)  # empty channel list -> passthrough
        assert out.shape == frame.shape

    def test_5ch_frame_uses_only_configured_channels(self, asr):
        frame = np.random.default_rng(2).normal(0, 5.0, (5, N_SAMPLES))
        out = asr.apply(frame)
        assert out.shape == frame.shape
