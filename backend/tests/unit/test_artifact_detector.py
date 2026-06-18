"""Dedicated unit tests for dsp.artifact_detector.ArtifactDetector."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.artifact_detector import ArtifactDetector
from neurolink.dsp.artifact_config import ArtifactConfig


N_CH = 4
FS = 256


@pytest.fixture()
def config() -> ArtifactConfig:
    return ArtifactConfig()


@pytest.fixture()
def detector(config: ArtifactConfig) -> ArtifactDetector:
    return ArtifactDetector(n_channels=N_CH, fs=FS, config=config)


def _clean_frame(n_ch: int = N_CH) -> np.ndarray:
    """Return a single sample of small-amplitude EEG (shape n_ch,)."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 5e-6, size=(n_ch,))


def _artifact_frame(amplitude: float = 500e-6, n_ch: int = N_CH) -> np.ndarray:
    return np.full((n_ch,), amplitude)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self, config):
        det = ArtifactDetector(n_channels=N_CH, fs=FS, config=config)
        assert det is not None

    def test_channel_count_stored(self, detector):
        assert detector.n_channels == N_CH

    def test_fs_stored(self, detector):
        assert detector.fs == FS


# ---------------------------------------------------------------------------
# Clean signal — no artifact flagged
# ---------------------------------------------------------------------------

class TestCleanSignal:
    def test_clean_frame_not_flagged(self, detector):
        frame = _clean_frame()
        result = detector.update(frame)
        assert not result.artifact

    def test_clean_stream_all_clear(self, detector):
        for _ in range(FS):  # 1 second of clean data
            frame = _clean_frame()
            result = detector.update(frame)
        assert not result.artifact


# ---------------------------------------------------------------------------
# Amplitude threshold
# ---------------------------------------------------------------------------

class TestAmplitudeThreshold:
    def test_large_amplitude_flagged(self, detector):
        frame = _artifact_frame(amplitude=1000e-6)
        result = detector.update(frame)
        assert result.artifact

    def test_borderline_below_threshold_not_flagged(self, config):
        """Signal just under the threshold should not be flagged."""
        det = ArtifactDetector(n_channels=N_CH, fs=FS, config=config)
        # warm up
        for _ in range(10):
            det.update(_clean_frame())
        safe_amp = (config.amplitude_threshold_uv * 0.8) * 1e-6
        result = det.update(np.full((N_CH,), safe_amp))
        assert not result.artifact


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self, detector):
        # contaminate
        for _ in range(20):
            detector.update(_artifact_frame())
        detector.reset()
        # after reset clean data should not be flagged
        result = detector.update(_clean_frame())
        assert not result.artifact


# ---------------------------------------------------------------------------
# Per-channel mask
# ---------------------------------------------------------------------------

class TestPerChannelMask:
    def test_single_bad_channel_identified(self, config):
        det = ArtifactDetector(n_channels=N_CH, fs=FS, config=config)
        frame = _clean_frame()
        frame[2] = 2000e-6  # only channel 2 is bad
        result = det.update(frame)
        assert result.artifact
        if hasattr(result, 'channel_mask'):
            assert result.channel_mask[2]


# ---------------------------------------------------------------------------
# Update returns a result object
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_update_returns_object_with_artifact_attr(self, detector):
        result = detector.update(_clean_frame())
        assert hasattr(result, 'artifact')

    def test_artifact_attr_is_bool(self, detector):
        result = detector.update(_clean_frame())
        assert isinstance(result.artifact, bool)
