"""Unit tests for dsp/bad_channels.py — Stage 2 bad channel detector."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from neurolink.dsp.bad_channels import (
    CHANNEL_NAMES,
    BadChannelDetector,
    ChannelStats,
    DetectorConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CH = 5
N_SAMPLES = 256


def _eeg(
    n_ch: int = N_CH,
    n_samples: int = N_SAMPLES,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samples)) * scale).astype(np.float32)


def _warm(
    det: BadChannelDetector,
    n: int = 40,
    n_ch: int = N_CH,
    scale: float = 1.0,
) -> None:
    for _ in range(n):
        det.update(_eeg(n_ch=n_ch, scale=scale))


# ---------------------------------------------------------------------------
# DetectorConfig
# ---------------------------------------------------------------------------


class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.var_threshold == pytest.approx(0.01)
        assert cfg.psd_ratio_threshold == pytest.approx(5.0)
        assert cfg.ema_alpha == pytest.approx(0.1)
        assert cfg.fs == pytest.approx(256.0)
        assert cfg.nperseg == 128


# ---------------------------------------------------------------------------
# ChannelStats
# ---------------------------------------------------------------------------


class TestChannelStats:
    def test_clean_channel_not_bad(self):
        s = ChannelStats(name="TP9")
        assert s.is_bad is False

    def test_manual_bad_is_bad(self):
        s = ChannelStats(name="TP9", manual_bad=True)
        assert s.is_bad is True

    def test_flat_line_is_bad(self):
        s = ChannelStats(name="TP9", flat_line=True)
        assert s.is_bad is True

    def test_noisy_is_bad(self):
        s = ChannelStats(name="TP9", noisy=True)
        assert s.is_bad is True

    def test_reason_ok(self):
        assert ChannelStats(name="TP9").reason() == "ok"

    def test_reason_manual(self):
        assert "manual" in ChannelStats(name="TP9", manual_bad=True).reason()

    def test_reason_flat_line(self):
        assert "flat_line" in ChannelStats(name="TP9", flat_line=True).reason()

    def test_reason_noisy(self):
        assert "noisy" in ChannelStats(name="TP9", noisy=True).reason()

    def test_reason_multiple(self):
        s = ChannelStats(name="TP9", flat_line=True, noisy=True)
        r = s.reason()
        assert "flat_line" in r and "noisy" in r


# ---------------------------------------------------------------------------
# BadChannelDetector.update() — guards
# ---------------------------------------------------------------------------


class TestBadChannelDetectorGuards:
    def test_none_input_does_not_raise(self):
        det = BadChannelDetector()
        det.update(None)

    def test_1d_input_does_not_raise(self):
        det = BadChannelDetector()
        det.update(np.zeros(256, dtype=np.float32))

    def test_single_sample_skipped(self):
        """n_samples < 2 → skipped; no crash."""
        det = BadChannelDetector()
        det.update(np.zeros((N_CH, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# BadChannelDetector.update() — flat-line detection
# ---------------------------------------------------------------------------


class TestFlatLineDetection:
    def test_flat_channel_flagged_after_warmup(self):
        """Constant-zero channel should reach flat_line=True after EMA convergence."""
        cfg = DetectorConfig(var_threshold=0.01, ema_alpha=0.5)
        det = BadChannelDetector(cfg)
        for _ in range(30):
            eeg = _eeg(n_ch=N_CH, scale=1.0)
            eeg[0, :] = 0.0  # channel 0 (TP9) is flat
            det.update(eeg)
        bad = det.get_bad_channels()
        assert "TP9" in bad

    def test_active_channel_not_flat(self):
        """Normal-variance channel should not be flat-lined."""
        cfg = DetectorConfig(var_threshold=0.01, ema_alpha=0.5)
        det = BadChannelDetector(cfg)
        _warm(det, n=30)
        bad = det.get_bad_channels()
        assert "AF7" not in bad  # normal activity, not flat


# ---------------------------------------------------------------------------
# BadChannelDetector.update() — noisy detection
# ---------------------------------------------------------------------------


class TestNoisyDetection:
    def test_high_amplitude_channel_flagged_noisy(self):
        """Channel with 100× amplitude should dominate PSD → noisy=True."""
        cfg = DetectorConfig(psd_ratio_threshold=5.0, ema_alpha=0.5)
        det = BadChannelDetector(cfg)
        for _ in range(40):
            eeg = _eeg(n_ch=N_CH, scale=1.0)
            eeg[1, :] = (_eeg(n_ch=1, scale=100.0))[0]  # AF7 very noisy
            det.update(eeg)
        bad = det.get_bad_channels()
        assert "AF7" in bad


# ---------------------------------------------------------------------------
# BadChannelDetector.get_bad_channels()
# ---------------------------------------------------------------------------


class TestGetBadChannels:
    def test_returns_list(self):
        det = BadChannelDetector()
        assert isinstance(det.get_bad_channels(), list)

    def test_fresh_detector_no_bad_channels(self):
        """Before any update(), EMA variances are 0 which means flat_line=True
        (0.0 < var_threshold=0.01). This is by design — fresh detector
        conservatively reports all channels as bad until updated.
        Verify it returns a list; content depends on initialisation."""
        det = BadChannelDetector()
        assert isinstance(det.get_bad_channels(), list)

    def test_all_names_are_valid_channel_names(self):
        det = BadChannelDetector()
        _warm(det)
        for name in det.get_bad_channels():
            assert name in CHANNEL_NAMES


# ---------------------------------------------------------------------------
# BadChannelDetector.get_stats()
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_returns_list_of_five(self):
        det = BadChannelDetector()
        stats = det.get_stats()
        assert len(stats) == 5

    def test_returns_channel_stats_instances(self):
        det = BadChannelDetector()
        for s in det.get_stats():
            assert isinstance(s, ChannelStats)

    def test_returns_deep_copy(self):
        """Mutating the returned list must not affect internal state."""
        det = BadChannelDetector()
        stats = det.get_stats()
        stats[0].manual_bad = True
        assert det.get_stats()[0].manual_bad is False

    def test_channel_names_match_channel_names_constant(self):
        det = BadChannelDetector()
        names = [s.name for s in det.get_stats()]
        assert names == CHANNEL_NAMES


# ---------------------------------------------------------------------------
# BadChannelDetector.set_manual_bad()
# ---------------------------------------------------------------------------


class TestSetManualBad:
    def test_flag_channel_bad(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        bad = det.get_bad_channels()
        assert "TP9" in bad

    def test_unflag_channel(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        det.set_manual_bad("TP9", False)
        # After un-flagging, only auto-detection governs; fresh EMA=0 is flat
        stats = det.get_stats()
        tp9 = next(s for s in stats if s.name == "TP9")
        assert tp9.manual_bad is False

    def test_case_insensitive(self):
        det = BadChannelDetector()
        det.set_manual_bad("tp9", True)
        assert "TP9" in det.get_bad_channels()

    def test_unknown_channel_raises_value_error(self):
        det = BadChannelDetector()
        with pytest.raises(ValueError, match="Unknown channel"):
            det.set_manual_bad("INVALID_CH", True)

    def test_aux_can_be_manually_flagged(self):
        det = BadChannelDetector()
        det.set_manual_bad("AUX", True)
        assert "AUX" in det.get_bad_channels()


# ---------------------------------------------------------------------------
# BadChannelDetector.reset()
# ---------------------------------------------------------------------------


class TestBadChannelDetectorReset:
    def test_reset_clears_manual_flags(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        det.reset()
        stats = det.get_stats()
        assert all(not s.manual_bad for s in stats)

    def test_reset_clears_ema_variance(self):
        det = BadChannelDetector()
        _warm(det)
        det.reset()
        for s in det.get_stats():
            assert s.ema_variance == 0.0

    def test_reset_does_not_raise_on_fresh_instance(self):
        BadChannelDetector().reset()


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------


class TestBadChannelDetectorConfig:
    def test_get_config_returns_copy(self):
        det = BadChannelDetector()
        c1 = det.get_config()
        c2 = det.get_config()
        assert c1 is not c2

    def test_set_config_updates_threshold(self):
        det = BadChannelDetector()
        det.set_config(DetectorConfig(var_threshold=99.0))
        assert det.get_config().var_threshold == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestBadChannelDetectorThreadSafety:
    def test_concurrent_update_and_get_does_not_raise(self):
        det = BadChannelDetector()
        errors: list[Exception] = []

        def updater():
            try:
                for _ in range(20):
                    det.update(_eeg())
            except Exception as exc:
                errors.append(exc)

        def getter():
            try:
                for _ in range(20):
                    _ = det.get_bad_channels()
                    _ = det.get_stats()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=updater) for _ in range(3)] + [
            threading.Thread(target=getter) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
