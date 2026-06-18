"""Unit tests for neurolink.dsp.bad_channels.

Public API confirmed from source:
  BadChannelDetector(config=None)   # no n_channels arg
  .update(eeg)                      # called each pump tick
  .get_bad_channels() -> list[str]  # returns names like ['TP9', 'AF7']
  .get_stats() -> list[ChannelStats]
  .set_manual_bad(channel, bad)
  .reset()
  .get_config() / .set_config()

CHANNEL_NAMES = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']
DetectorConfig fields: var_threshold, psd_ratio_threshold, ema_alpha, fs, nperseg
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bad_channels import BadChannelDetector, ChannelStats, DetectorConfig


FS = 256.0
N_CH = 5   # TP9, AF7, AF8, TP10, AUX
N_SAMPLES = 256

# Channel indices
_TP9, _AF7, _AF8, _TP10, _AUX = 0, 1, 2, 3, 4


@pytest.fixture
def detector() -> BadChannelDetector:
    cfg = DetectorConfig(fs=FS, ema_alpha=1.0)  # alpha=1.0: no smoothing, instant response
    return BadChannelDetector(config=cfg)


@pytest.fixture
def clean_eeg() -> np.ndarray:
    """All channels active with normal variance."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 10.0, (N_CH, N_SAMPLES))  # 100 uV^2 variance >> 0.01 threshold


@pytest.fixture
def flat_tp9_eeg() -> np.ndarray:
    """TP9 (ch0) is flat-line: variance = 0."""
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10.0, (N_CH, N_SAMPLES))
    eeg[0] = 0.0
    return eeg


class TestConstruction:
    def test_instantiation(self):
        d = BadChannelDetector()
        assert d is not None

    def test_instantiation_with_config(self):
        cfg = DetectorConfig(var_threshold=0.05)
        d = BadChannelDetector(config=cfg)
        assert d.get_config().var_threshold == pytest.approx(0.05)

    def test_channel_count(self):
        """Detector always tracks 5 channels (TP9, AF7, AF8, TP10, AUX)."""
        d = BadChannelDetector()
        stats = d.get_stats()
        assert len(stats) == 5


class TestDetection:
    def test_all_good_channels_returns_empty_list(self, detector, clean_eeg):
        detector.update(clean_eeg)
        bad = detector.get_bad_channels()
        assert bad == []

    def test_flat_channel_detected(self, detector, flat_tp9_eeg):
        """TP9 flat-line -> variance 0 -> detected as bad."""
        # Feed several frames so EMA converges (alpha=1.0 so 1 frame is enough)
        detector.update(flat_tp9_eeg)
        bad = detector.get_bad_channels()
        assert "TP9" in bad

    def test_good_channels_not_flagged(self, detector, flat_tp9_eeg):
        detector.update(flat_tp9_eeg)
        bad = detector.get_bad_channels()
        # Only TP9 should be bad
        assert "AF7" not in bad
        assert "AF8" not in bad
        assert "TP10" not in bad

    def test_high_amplitude_channel_detected(self, detector):
        """Channel with very high PSD ratio relative to median -> noisy/bad."""
        rng = np.random.default_rng(5)
        eeg = rng.normal(0, 10.0, (N_CH, N_SAMPLES))
        # Inject extreme noise on AF8 (ch2)
        eeg[2] = rng.normal(0, 1000.0, N_SAMPLES)
        detector.update(eeg)
        bad = detector.get_bad_channels()
        assert "AF8" in bad


class TestMaskReturnType:
    def test_get_bad_channels_returns_list(self, detector, clean_eeg):
        detector.update(clean_eeg)
        result = detector.get_bad_channels()
        assert isinstance(result, list)

    def test_bad_channel_names_are_strings(self, detector, flat_tp9_eeg):
        detector.update(flat_tp9_eeg)
        bad = detector.get_bad_channels()
        for name in bad:
            assert isinstance(name, str)

    def test_bad_channel_names_in_montage(self, detector, flat_tp9_eeg):
        detector.update(flat_tp9_eeg)
        valid = {"TP9", "AF7", "AF8", "TP10", "AUX"}
        for name in detector.get_bad_channels():
            assert name in valid


class TestManualBad:
    def test_set_manual_bad_marks_channel(self, detector, clean_eeg):
        detector.update(clean_eeg)
        detector.set_manual_bad("AF7", True)
        assert "AF7" in detector.get_bad_channels()

    def test_clear_manual_bad(self, detector, clean_eeg):
        detector.update(clean_eeg)
        detector.set_manual_bad("AF7", True)
        detector.set_manual_bad("AF7", False)
        assert "AF7" not in detector.get_bad_channels()


class TestReset:
    def test_reset_does_not_raise(self, detector, flat_tp9_eeg):
        detector.update(flat_tp9_eeg)
        detector.reset()

    def test_after_reset_detects_correctly(self, detector, flat_tp9_eeg, clean_eeg):
        detector.update(flat_tp9_eeg)
        detector.reset()
        detector.update(clean_eeg)
        # After reset + good data, no bad channels expected
        bad = detector.get_bad_channels()
        assert "TP9" not in bad


class TestStats:
    def test_get_stats_returns_list_of_channel_stats(self, detector, clean_eeg):
        detector.update(clean_eeg)
        stats = detector.get_stats()
        assert isinstance(stats, list)
        assert all(isinstance(s, ChannelStats) for s in stats)

    def test_stats_channel_names(self, detector, clean_eeg):
        detector.update(clean_eeg)
        names = [s.name for s in detector.get_stats()]
        assert "TP9" in names
        assert "AF7" in names
