"""Unit tests for dsp/bad_channels.py (Stage 2 bad channel detector)."""

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

N_CH = 4    # EEG-only (no AUX) for most tests
N_ALL = 5   # EEG + AUX
FS = 256.0
NSAMP = 256  # 1 second of samples — enough for Welch


def _flat(n_ch: int = N_CH, n_samples: int = NSAMP) -> np.ndarray:
    """All-zero (flat-line) EEG."""
    return np.zeros((n_ch, n_samples), dtype=np.float32)


def _normal(n_ch: int = N_CH, n_samples: int = NSAMP, seed: int = 0, scale: float = 10.0) -> np.ndarray:
    """Gaussian noise — healthy variance."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samples)) * scale).astype(np.float32)


def _noisy(n_ch: int = N_CH, n_samples: int = NSAMP, bad_ch: int = 0, multiplier: float = 20.0) -> np.ndarray:
    """One channel has 20x amplitude (noisy)."""
    arr = _normal(n_ch=n_ch, n_samples=n_samples)
    arr[bad_ch] *= multiplier
    return arr


def _pump(detector: BadChannelDetector, arr: np.ndarray, n: int = 25) -> None:
    """Pump n identical frames to let EMA converge."""
    for _ in range(n):
        detector.update(arr)


# ---------------------------------------------------------------------------
# DetectorConfig defaults
# ---------------------------------------------------------------------------

class TestDetectorConfigDefaults:
    def test_var_threshold(self):
        assert DetectorConfig().var_threshold == 0.01

    def test_psd_ratio_threshold(self):
        assert DetectorConfig().psd_ratio_threshold == 5.0

    def test_ema_alpha(self):
        assert DetectorConfig().ema_alpha == 0.1

    def test_fs(self):
        assert DetectorConfig().fs == 256.0

    def test_nperseg(self):
        assert DetectorConfig().nperseg == 128


# ---------------------------------------------------------------------------
# ChannelStats
# ---------------------------------------------------------------------------

class TestChannelStats:
    def test_is_bad_false_by_default(self):
        s = ChannelStats(name="TP9")
        assert s.is_bad is False

    def test_is_bad_flat_line(self):
        s = ChannelStats(name="TP9", flat_line=True)
        assert s.is_bad is True

    def test_is_bad_noisy(self):
        s = ChannelStats(name="TP9", noisy=True)
        assert s.is_bad is True

    def test_is_bad_manual(self):
        s = ChannelStats(name="TP9", manual_bad=True)
        assert s.is_bad is True

    def test_reason_ok(self):
        assert ChannelStats(name="TP9").reason() == "ok"

    def test_reason_flat_line(self):
        assert ChannelStats(name="TP9", flat_line=True).reason() == "flat_line"

    def test_reason_noisy(self):
        assert ChannelStats(name="TP9", noisy=True).reason() == "noisy"

    def test_reason_manual(self):
        assert ChannelStats(name="TP9", manual_bad=True).reason() == "manual"

    def test_reason_combined(self):
        s = ChannelStats(name="TP9", flat_line=True, manual_bad=True)
        reasons = s.reason().split(",")
        assert "manual" in reasons
        assert "flat_line" in reasons


# ---------------------------------------------------------------------------
# CHANNEL_NAMES constant
# ---------------------------------------------------------------------------

class TestChannelNames:
    def test_length_five(self):
        assert len(CHANNEL_NAMES) == 5

    def test_names(self):
        assert CHANNEL_NAMES == ["TP9", "AF7", "AF8", "TP10", "AUX"]


# ---------------------------------------------------------------------------
# BadChannelDetector — construction
# ---------------------------------------------------------------------------

class TestBadChannelDetectorInit:
    def test_default_no_bad_channels(self):
        det = BadChannelDetector()
        assert det.get_bad_channels() == []

    def test_stats_length_five(self):
        det = BadChannelDetector()
        assert len(det.get_stats()) == 5

    def test_stats_names_match_channel_names(self):
        det = BadChannelDetector()
        assert [s.name for s in det.get_stats()] == CHANNEL_NAMES


# ---------------------------------------------------------------------------
# update() — input guards
# ---------------------------------------------------------------------------

class TestUpdateInputGuards:
    def test_none_input_no_exception(self):
        det = BadChannelDetector()
        det.update(None)  # type: ignore[arg-type]

    def test_1d_input_no_exception(self):
        det = BadChannelDetector()
        det.update(np.zeros(256, dtype=np.float32))  # type: ignore[arg-type]

    def test_single_sample_no_exception(self):
        det = BadChannelDetector()
        det.update(np.zeros((4, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# Flat-line detection
# ---------------------------------------------------------------------------

class TestFlatLineDetection:
    def test_flat_channel_detected_after_convergence(self):
        det = BadChannelDetector()
        _pump(det, _flat(n_ch=N_CH))
        bad = det.get_bad_channels()
        # All 4 channels are flat — all should be detected
        assert set(bad) == {"TP9", "AF7", "AF8", "TP10"}

    def test_normal_channel_not_flat(self):
        det = BadChannelDetector()
        _pump(det, _normal())
        stats = det.get_stats()
        for s in stats[:N_CH]:
            assert s.flat_line is False, f"{s.name} incorrectly flagged flat"

    def test_single_flat_channel_detected(self):
        """Only one channel flat among healthy neighbours."""
        cfg = DetectorConfig(ema_alpha=0.5)  # faster convergence
        det = BadChannelDetector(cfg)
        arr = _normal(n_ch=N_CH)
        arr[2] = 0.0  # AF8 flat
        _pump(det, arr, n=30)
        bad = det.get_bad_channels()
        assert "AF8" in bad


# ---------------------------------------------------------------------------
# Noisy detection
# ---------------------------------------------------------------------------

class TestNoisyDetection:
    def test_noisy_channel_detected(self):
        cfg = DetectorConfig(psd_ratio_threshold=3.0, ema_alpha=0.5)
        det = BadChannelDetector(cfg)
        _pump(det, _noisy(bad_ch=0, multiplier=30.0), n=30)
        assert "TP9" in det.get_bad_channels()

    def test_clean_channels_not_noisy(self):
        det = BadChannelDetector()
        _pump(det, _normal())
        for s in det.get_stats()[:N_CH]:
            assert s.noisy is False

    def test_aux_channel_not_noisy_flagged(self):
        """AUX (index 4) is excluded from PSD ratio comparison."""
        arr = np.zeros((N_ALL, NSAMP), dtype=np.float32)
        arr[:N_CH] = _normal(n_ch=N_CH)  # healthy EEG
        arr[4] = _normal(n_ch=1, scale=1000.0)[0]   # very noisy AUX
        det = BadChannelDetector()
        _pump(det, arr)
        aux_stat = det.get_stats()[4]
        assert aux_stat.noisy is False  # AUX never gets noisy flag


# ---------------------------------------------------------------------------
# Manual override
# ---------------------------------------------------------------------------

class TestManualOverride:
    def test_manual_flag_marks_channel_bad(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        assert "TP9" in det.get_bad_channels()

    def test_manual_unflag_clears_channel(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        det.set_manual_bad("TP9", False)
        # Without other flags, should be clean
        assert "TP9" not in det.get_bad_channels()

    def test_manual_flag_case_insensitive(self):
        det = BadChannelDetector()
        det.set_manual_bad("tp9", True)
        assert "TP9" in det.get_bad_channels()

    def test_manual_flag_aux(self):
        det = BadChannelDetector()
        det.set_manual_bad("AUX", True)
        assert "AUX" in det.get_bad_channels()

    def test_unknown_channel_raises(self):
        det = BadChannelDetector()
        with pytest.raises(ValueError, match="Unknown channel"):
            det.set_manual_bad("FAKE", True)

    def test_manual_flag_persists_after_update(self):
        det = BadChannelDetector()
        det.set_manual_bad("AF7", True)
        _pump(det, _normal())  # healthy EEG pumped
        assert "AF7" in det.get_bad_channels()


# ---------------------------------------------------------------------------
# get_stats() returns deep copy
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_list_of_channel_stats(self):
        det = BadChannelDetector()
        stats = det.get_stats()
        assert all(isinstance(s, ChannelStats) for s in stats)

    def test_mutation_does_not_affect_internal_state(self):
        det = BadChannelDetector()
        stats = det.get_stats()
        stats[0].manual_bad = True
        assert det.get_bad_channels() == []  # internal state unchanged


# ---------------------------------------------------------------------------
# get_bad_channels()
# ---------------------------------------------------------------------------

class TestGetBadChannels:
    def test_returns_list_of_strings(self):
        det = BadChannelDetector()
        assert isinstance(det.get_bad_channels(), list)

    def test_empty_initially(self):
        det = BadChannelDetector()
        assert det.get_bad_channels() == []


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all_flags(self):
        det = BadChannelDetector()
        _pump(det, _flat())
        assert det.get_bad_channels() != []
        det.reset()
        assert det.get_bad_channels() == []

    def test_reset_clears_manual_flags(self):
        det = BadChannelDetector()
        det.set_manual_bad("TP9", True)
        det.reset()
        assert det.get_bad_channels() == []

    def test_reset_clears_ema_values(self):
        det = BadChannelDetector()
        _pump(det, _normal())
        det.reset()
        for s in det.get_stats():
            assert s.ema_variance == 0.0
            assert s.ema_mean_psd == 0.0


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------

class TestConfigAccessors:
    def test_get_config_returns_copy(self):
        det = BadChannelDetector()
        cfg = det.get_config()
        cfg.var_threshold = 999.0
        assert det.get_config().var_threshold == 0.01  # unchanged

    def test_set_config_updates(self):
        det = BadChannelDetector()
        new_cfg = DetectorConfig(var_threshold=5.0)
        det.set_config(new_cfg)
        assert det.get_config().var_threshold == 5.0


# ---------------------------------------------------------------------------
# EMA convergence behaviour
# ---------------------------------------------------------------------------

class TestEMAConvergence:
    def test_ema_variance_increases_with_noisy_data(self):
        det = BadChannelDetector()
        before = det.get_stats()[0].ema_variance
        _pump(det, _normal(scale=100.0), n=10)
        after = det.get_stats()[0].ema_variance
        assert after > before

    def test_ema_variance_stays_low_for_flat_data(self):
        det = BadChannelDetector()
        _pump(det, _flat(), n=50)
        for s in det.get_stats()[:N_CH]:
            assert s.ema_variance < DetectorConfig().var_threshold


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_updates_no_exception(self):
        det = BadChannelDetector()
        errors: list[Exception] = []

        def _worker():
            try:
                for _ in range(20):
                    det.update(_normal())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_read_write_no_exception(self):
        det = BadChannelDetector()
        errors: list[Exception] = []

        def _updater():
            try:
                for _ in range(20):
                    det.update(_normal())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _reader():
            try:
                for _ in range(20):
                    det.get_bad_channels()
                    det.get_stats()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _flagger():
            try:
                for _ in range(5):
                    det.set_manual_bad("TP9", True)
                    det.set_manual_bad("TP9", False)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_updater),
            threading.Thread(target=_reader),
            threading.Thread(target=_flagger),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
