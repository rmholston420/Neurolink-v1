"""Unit tests for dsp/bad_channels.py (Stage 2 bad-channel detector)."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bad_channels import (
    BadChannelDetector,
    ChannelStats,
    DetectorConfig,
)

FS = 256.0
N_SAMPLES = 256  # 1-second buffer at 256 Hz
N_CH = 5  # TP9, AF7, AF8, TP10, AUX


def _make_clean_eeg(n_ch: int = N_CH, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Return a (n_ch, n_samples) array of 10 µV white noise — all channels clean."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((n_ch, n_samples)) * 10.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Flat-line detection
# ---------------------------------------------------------------------------

class TestFlatLineDetection:
    def test_flat_channel_detected_after_warmup(self):
        det = BadChannelDetector(DetectorConfig(var_threshold=0.01, ema_alpha=1.0))
        eeg = _make_clean_eeg()
        # Zero out channel 1 (AF7) — variance = 0
        eeg[1] = 0.0
        det.update(eeg)
        bad = det.get_bad_channels()
        assert "AF7" in bad

    def test_clean_channels_not_flagged(self):
        det = BadChannelDetector(DetectorConfig(var_threshold=0.01, ema_alpha=1.0))
        eeg = _make_clean_eeg()
        det.update(eeg)
        bad = det.get_bad_channels()
        # No channel should be flat (10 µV noise variance >> 0.01)
        assert not bad

    def test_flat_line_clears_after_recovery(self):
        det = BadChannelDetector(DetectorConfig(var_threshold=0.01, ema_alpha=1.0))
        eeg_flat = _make_clean_eeg()
        eeg_flat[2] = 0.0  # AF8 flat
        det.update(eeg_flat)
        assert "AF8" in det.get_bad_channels()

        eeg_ok = _make_clean_eeg()
        det.update(eeg_ok)  # alpha=1 → instant update
        assert "AF8" not in det.get_bad_channels()


# ---------------------------------------------------------------------------
# Noisy / high-PSD detection
# ---------------------------------------------------------------------------

class TestNoisyDetection:
    def test_high_noise_channel_flagged(self):
        det = BadChannelDetector(
            DetectorConfig(psd_ratio_threshold=3.0, ema_alpha=1.0)
        )
        rng = np.random.default_rng(0)
        eeg = (rng.standard_normal((N_CH, N_SAMPLES)) * 1.0).astype(np.float32)
        # Inject 100× higher-amplitude noise on TP9 (index 0)
        eeg[0] = (rng.standard_normal(N_SAMPLES) * 100.0).astype(np.float32)
        det.update(eeg)
        bad = det.get_bad_channels()
        assert "TP9" in bad

    def test_aux_never_flagged_as_noisy(self):
        """AUX (index 4) should never appear in noisy bad list."""
        det = BadChannelDetector(
            DetectorConfig(psd_ratio_threshold=1.0, ema_alpha=1.0)
        )
        rng = np.random.default_rng(1)
        eeg = (rng.standard_normal((N_CH, N_SAMPLES)) * 100.0).astype(np.float32)
        det.update(eeg)
        bad = det.get_bad_channels()
        assert "AUX" not in bad or (
            any(det.get_stats()[4].flat_line for _ in [1])  # only flat-line possible
        )


# ---------------------------------------------------------------------------
# Manual override
# ---------------------------------------------------------------------------

class TestManualOverride:
    def test_manual_bad_flag_takes_priority(self):
        det = BadChannelDetector()
        eeg = _make_clean_eeg()
        for _ in range(5):
            det.update(eeg)
        assert det.get_bad_channels() == []

        det.set_manual_bad("TP10", True)
        assert "TP10" in det.get_bad_channels()

    def test_manual_unflag_restores_channel(self):
        det = BadChannelDetector()
        det.set_manual_bad("AF7", True)
        assert "AF7" in det.get_bad_channels()
        det.set_manual_bad("AF7", False)
        assert "AF7" not in det.get_bad_channels()

    def test_unknown_channel_raises(self):
        det = BadChannelDetector()
        with pytest.raises(ValueError, match="Unknown channel"):
            det.set_manual_bad("CZ", True)


# ---------------------------------------------------------------------------
# reset() and config API
# ---------------------------------------------------------------------------

class TestResetAndConfig:
    def test_reset_clears_all_stats(self):
        det = BadChannelDetector(DetectorConfig(var_threshold=0.01, ema_alpha=1.0))
        eeg = _make_clean_eeg()
        eeg[0] = 0.0  # make TP9 flat
        det.update(eeg)
        assert "TP9" in det.get_bad_channels()
        det.reset()
        assert det.get_bad_channels() == []

    def test_get_stats_returns_all_channels(self):
        det = BadChannelDetector()
        stats = det.get_stats()
        assert len(stats) == N_CH
        assert all(isinstance(s, ChannelStats) for s in stats)

    def test_set_config_updates_thresholds(self):
        det = BadChannelDetector()
        new_cfg = DetectorConfig(var_threshold=99.0, psd_ratio_threshold=2.0)
        det.set_config(new_cfg)
        cfg = det.get_config()
        assert cfg.var_threshold == 99.0
        assert cfg.psd_ratio_threshold == 2.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_buffer_skipped(self):
        """update() with n_samples < 2 should not raise or mutate state."""
        det = BadChannelDetector()
        eeg = np.zeros((N_CH, 1), dtype=np.float32)
        det.update(eeg)  # should silently skip
        assert det.get_bad_channels() == []

    def test_none_input_skipped(self):
        det = BadChannelDetector()
        det.update(None)  # type: ignore[arg-type]
        assert det.get_bad_channels() == []

    def test_fewer_than_five_channels(self):
        """Devices that stream only 4 EEG channels (no AUX) should work."""
        det = BadChannelDetector(DetectorConfig(var_threshold=0.01, ema_alpha=1.0))
        eeg = _make_clean_eeg(n_ch=4)  # no AUX
        eeg[1] = 0.0  # AF7 flat
        det.update(eeg)
        assert "AF7" in det.get_bad_channels()
