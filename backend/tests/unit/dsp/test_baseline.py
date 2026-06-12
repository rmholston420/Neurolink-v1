"""Unit tests for dsp/baseline.py — BaselineRecorder + ASR covariance cache."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from neurolink.dsp.baseline import (
    BaselinePhase,
    BaselineRecorder,
    _COV_CACHE_TTL_SEC,
    _cov_cache,
    _cov_cache_lock,
    load_asr_covariance,
    save_asr_covariance,
)
from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC, BASELINE_TOTAL_SEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder(start_offset: float = 0.0) -> tuple[BaselineRecorder, MagicMock]:
    """Return a (recorder, hub_mock) pair with _start_ts shifted by offset."""
    asr_mock = MagicMock()
    hub_mock = MagicMock()
    rec = BaselineRecorder(asr=asr_mock, hub=hub_mock)
    rec._start_ts -= start_offset  # simulate time already elapsed
    return rec, hub_mock


def _eeg(n_ch: int = 4, n_samples: int = 32) -> np.ndarray:
    return np.random.default_rng(0).standard_normal((n_ch, n_samples)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixture: clear covariance cache between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cov_cache():
    with _cov_cache_lock:
        _cov_cache.clear()
    yield
    with _cov_cache_lock:
        _cov_cache.clear()


# ---------------------------------------------------------------------------
# BaselinePhase enum
# ---------------------------------------------------------------------------

class TestBaselinePhaseEnum:
    def test_values(self):
        assert BaselinePhase.WARMUP.value == "warmup"
        assert BaselinePhase.RECORDING.value == "recording"
        assert BaselinePhase.COMPLETE.value == "complete"

    def test_str_enum(self):
        # BaselinePhase extends str — value must equal the string itself
        assert str(BaselinePhase.WARMUP) == "warmup" or BaselinePhase.WARMUP == "warmup"


# ---------------------------------------------------------------------------
# BaselineRecorder — initial state
# ---------------------------------------------------------------------------

class TestBaselineRecorderInit:
    def test_initial_phase_is_warmup(self):
        rec, _ = _make_recorder()
        assert rec.phase == "warmup"

    def test_is_complete_false_at_start(self):
        rec, _ = _make_recorder()
        assert rec.is_complete is False

    def test_process_returns_eeg_unchanged_in_warmup(self):
        rec, _ = _make_recorder()
        eeg = _eeg()
        out = rec.process(eeg)
        np.testing.assert_array_equal(out, eeg)


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_warmup_to_recording_after_discard_period(self):
        # Simulate that BASELINE_DISCARD_SEC have elapsed
        rec, _ = _make_recorder(start_offset=BASELINE_DISCARD_SEC + 1.0)
        rec.process(_eeg())
        assert rec.phase == "recording"

    def test_still_warmup_before_discard_period(self):
        rec, _ = _make_recorder(start_offset=BASELINE_DISCARD_SEC - 1.0)
        rec.process(_eeg())
        assert rec.phase == "warmup"

    def test_recording_to_complete_after_total_sec(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        assert rec.phase == "complete"
        assert rec.is_complete is True

    def test_complete_is_terminal(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 5.0)
        rec.process(_eeg())
        assert rec.phase == "complete"
        rec.process(_eeg())  # second call
        assert rec.phase == "complete"

    def test_is_complete_true_after_total(self):
        rec, _ = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        assert rec.is_complete is True


# ---------------------------------------------------------------------------
# process() — return value contract
# ---------------------------------------------------------------------------

class TestProcessReturnValue:
    def test_returns_eeg_in_warmup(self):
        rec, _ = _make_recorder()
        eeg = _eeg()
        assert rec.process(eeg) is eeg

    def test_returns_eeg_in_recording(self):
        rec, _ = _make_recorder(start_offset=BASELINE_DISCARD_SEC + 1.0)
        rec.process(_eeg())  # trigger transition
        eeg = _eeg(seed=5)
        assert rec.process(eeg) is eeg

    def test_returns_eeg_in_complete(self):
        rec, _ = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        eeg = _eeg(seed=7)
        assert rec.process(eeg) is eeg


# ---------------------------------------------------------------------------
# Bell notification — fires exactly once
# ---------------------------------------------------------------------------

class TestBellNotification:
    def test_bell_fires_once_on_complete(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        hub.notify_baseline_complete.assert_called_once()

    def test_bell_does_not_fire_in_warmup(self):
        rec, hub = _make_recorder()
        rec.process(_eeg())
        hub.notify_baseline_complete.assert_not_called()

    def test_bell_does_not_fire_in_recording(self):
        rec, hub = _make_recorder(start_offset=BASELINE_DISCARD_SEC + 1.0)
        rec.process(_eeg())
        hub.notify_baseline_complete.assert_not_called()

    def test_bell_fires_exactly_once_across_multiple_complete_calls(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        for _ in range(5):
            rec.process(_eeg())
        hub.notify_baseline_complete.assert_called_once()

    def test_bell_hub_exception_does_not_propagate(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        hub.notify_baseline_complete.side_effect = RuntimeError("hub down")
        # Must not raise
        rec.process(_eeg())


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_to_warmup(self):
        rec, _ = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        assert rec.is_complete
        rec.reset()
        assert rec.phase == "warmup"
        assert rec.is_complete is False

    def test_reset_allows_bell_to_fire_again(self):
        rec, hub = _make_recorder(start_offset=BASELINE_TOTAL_SEC + 1.0)
        rec.process(_eeg())
        hub.notify_baseline_complete.assert_called_once()
        rec.reset()
        # Advance time past BASELINE_TOTAL_SEC again
        rec._start_ts -= BASELINE_TOTAL_SEC + 1.0
        rec.process(_eeg())
        assert hub.notify_baseline_complete.call_count == 2

    def test_reset_resets_bell_fired_flag(self):
        rec, _ = _make_recorder()
        rec._bell_fired = True
        rec.reset()
        assert rec._bell_fired is False


# ---------------------------------------------------------------------------
# ASR covariance cache — save + load
# ---------------------------------------------------------------------------

class TestASRCovarianceCache:
    def test_save_and_load_returns_same_array(self):
        cov = np.eye(4, dtype=np.float64)
        save_asr_covariance("AA:BB", "user1", cov)
        loaded = load_asr_covariance("AA:BB", "user1")
        np.testing.assert_array_equal(loaded, cov)

    def test_cache_miss_returns_none(self):
        assert load_asr_covariance("00:11", "unknown") is None

    def test_different_keys_are_independent(self):
        cov_a = np.eye(4) * 1.0
        cov_b = np.eye(4) * 2.0
        save_asr_covariance("AA:BB", "user1", cov_a)
        save_asr_covariance("CC:DD", "user2", cov_b)
        np.testing.assert_array_equal(load_asr_covariance("AA:BB", "user1"), cov_a)
        np.testing.assert_array_equal(load_asr_covariance("CC:DD", "user2"), cov_b)

    def test_same_key_overwritten_by_second_save(self):
        cov1 = np.eye(4) * 1.0
        cov2 = np.eye(4) * 3.0
        save_asr_covariance("AA:BB", "user1", cov1)
        save_asr_covariance("AA:BB", "user1", cov2)
        np.testing.assert_array_equal(load_asr_covariance("AA:BB", "user1"), cov2)

    def test_expired_entry_returns_none(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        # Manually backdate the timestamp to force expiry
        key = "AA:BB::user1"
        with _cov_cache_lock:
            _cov_cache[key]["ts"] -= _COV_CACHE_TTL_SEC + 1.0
        assert load_asr_covariance("AA:BB", "user1") is None

    def test_expired_entry_is_evicted(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        key = "AA:BB::user1"
        with _cov_cache_lock:
            _cov_cache[key]["ts"] -= _COV_CACHE_TTL_SEC + 1.0
        load_asr_covariance("AA:BB", "user1")  # triggers eviction
        with _cov_cache_lock:
            assert key not in _cov_cache

    def test_ttl_boundary_not_expired(self):
        """Entry saved exactly at TTL boundary should still be valid."""
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        key = "AA:BB::user1"
        with _cov_cache_lock:
            # Backdate to exactly TTL — age == TTL is NOT expired (> check)
            _cov_cache[key]["ts"] -= _COV_CACHE_TTL_SEC
        result = load_asr_covariance("AA:BB", "user1")
        assert result is not None

    def test_non_array_covariance_round_trips(self):
        """Cache is type-agnostic — any pickleable object should survive."""
        obj = {"matrix": [[1, 0], [0, 1]], "meta": "test"}
        save_asr_covariance("AA:BB", "user1", obj)
        loaded = load_asr_covariance("AA:BB", "user1")
        assert loaded == obj


# ---------------------------------------------------------------------------
# Covariance cache — thread safety
# ---------------------------------------------------------------------------

class TestCovarianceCacheThreadSafety:
    def test_concurrent_save_load_no_exception(self):
        errors: list[Exception] = []
        cov = np.eye(4)

        def _writer():
            try:
                for i in range(20):
                    save_asr_covariance("XX:YY", f"user{i % 3}", cov)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _reader():
            try:
                for i in range(20):
                    load_asr_covariance("XX:YY", f"user{i % 3}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_writer),
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
