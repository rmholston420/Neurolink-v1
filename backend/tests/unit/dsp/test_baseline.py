"""Unit tests for dsp/baseline.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC, BASELINE_TOTAL_SEC
from neurolink.dsp.baseline import (
    BaselinePhase,
    BaselineRecorder,
    _COV_CACHE_TTL_SEC,
    _cov_cache,
    load_asr_covariance,
    save_asr_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder(
    discard_elapsed: float = 0.0,
    total_elapsed: float = 0.0,
) -> tuple[BaselineRecorder, MagicMock, MagicMock]:
    """Return (recorder, mock_asr, mock_hub) with time offset already applied."""
    mock_asr = MagicMock()
    mock_hub = MagicMock()
    recorder = BaselineRecorder(asr=mock_asr, hub=mock_hub)
    # Back-date start_ts so that `elapsed` == requested value on the next process() call
    if discard_elapsed or total_elapsed:
        elapsed = max(discard_elapsed, total_elapsed)
        recorder._start_ts = time.monotonic() - elapsed
    return recorder, mock_asr, mock_hub


def _eeg(n_ch: int = 5, n_samples: int = 64) -> np.ndarray:
    return np.random.randn(n_ch, n_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# BaselinePhase enum
# ---------------------------------------------------------------------------

class TestBaselinePhase:
    def test_warmup_value(self):
        assert BaselinePhase.WARMUP.value == "warmup"

    def test_recording_value(self):
        assert BaselinePhase.RECORDING.value == "recording"

    def test_complete_value(self):
        assert BaselinePhase.COMPLETE.value == "complete"


# ---------------------------------------------------------------------------
# BaselineRecorder — initial state
# ---------------------------------------------------------------------------

class TestBaselineRecorderInit:
    def test_initial_phase_is_warmup(self):
        recorder, _, _ = _make_recorder()
        assert recorder.phase == "warmup"

    def test_initial_is_complete_false(self):
        recorder, _, _ = _make_recorder()
        assert recorder.is_complete is False

    def test_phase_property_returns_string(self):
        recorder, _, _ = _make_recorder()
        assert isinstance(recorder.phase, str)


# ---------------------------------------------------------------------------
# BaselineRecorder.process() — state transitions
# ---------------------------------------------------------------------------

class TestBaselineRecorderProcess:
    def test_process_returns_eeg_unchanged_warmup(self):
        recorder, _, _ = _make_recorder()
        eeg = _eeg()
        result = recorder.process(eeg)
        assert result is eeg  # same object returned

    def test_process_returns_eeg_unchanged_recording(self):
        recorder, _, _ = _make_recorder(discard_elapsed=BASELINE_DISCARD_SEC + 1)
        recorder.process(_eeg())  # trigger WARMUP→RECORDING
        eeg = _eeg()
        result = recorder.process(eeg)
        assert result is eeg

    def test_warmup_to_recording_transition(self):
        recorder, _, _ = _make_recorder(discard_elapsed=BASELINE_DISCARD_SEC + 1)
        recorder.process(_eeg())
        assert recorder.phase == "recording"

    def test_warmup_not_left_early(self):
        recorder, _, _ = _make_recorder(discard_elapsed=BASELINE_DISCARD_SEC - 1)
        recorder.process(_eeg())
        assert recorder.phase == "warmup"

    def test_recording_to_complete_transition(self):
        recorder, _, _ = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())  # force WARMUP→RECORDING first
        recorder.process(_eeg())  # now RECORDING→COMPLETE
        assert recorder.phase == "complete"

    def test_complete_is_terminal(self):
        recorder, _, _ = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        recorder.process(_eeg())   # extra ticks
        assert recorder.phase == "complete"

    def test_is_complete_true_after_transition(self):
        recorder, _, _ = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        assert recorder.is_complete is True

    # --- Bell / hub notification ---
    def test_bell_fired_exactly_once(self):
        recorder, _, mock_hub = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        recorder.process(_eeg())
        recorder.process(_eeg())
        mock_hub.notify_baseline_complete.assert_called_once()

    def test_bell_not_fired_during_warmup(self):
        recorder, _, mock_hub = _make_recorder()
        for _ in range(5):
            recorder.process(_eeg())
        mock_hub.notify_baseline_complete.assert_not_called()

    def test_bell_not_fired_during_recording(self):
        recorder, _, mock_hub = _make_recorder(discard_elapsed=BASELINE_DISCARD_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        mock_hub.notify_baseline_complete.assert_not_called()

    def test_hub_exception_does_not_propagate(self):
        recorder, _, mock_hub = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        mock_hub.notify_baseline_complete.side_effect = RuntimeError("hub dead")
        recorder.process(_eeg())  # WARMUP→RECORDING
        # Must not raise
        recorder.process(_eeg())  # RECORDING→COMPLETE + bell


# ---------------------------------------------------------------------------
# BaselineRecorder.reset()
# ---------------------------------------------------------------------------

class TestBaselineRecorderReset:
    def test_reset_returns_to_warmup(self):
        recorder, _, _ = _make_recorder(discard_elapsed=BASELINE_DISCARD_SEC + 1)
        recorder.process(_eeg())
        assert recorder.phase == "recording"
        recorder.reset()
        assert recorder.phase == "warmup"

    def test_reset_clears_is_complete(self):
        recorder, _, _ = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        recorder.reset()
        assert recorder.is_complete is False

    def test_bell_can_fire_again_after_reset(self):
        recorder, _, mock_hub = _make_recorder(total_elapsed=BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())
        recorder.process(_eeg())
        assert mock_hub.notify_baseline_complete.call_count == 1

        recorder.reset()
        # Back-date again after reset
        recorder._start_ts = time.monotonic() - (BASELINE_TOTAL_SEC + 1)
        recorder.process(_eeg())  # WARMUP→RECORDING
        recorder.process(_eeg())  # RECORDING→COMPLETE + second bell
        assert mock_hub.notify_baseline_complete.call_count == 2


# ---------------------------------------------------------------------------
# ASR Covariance Cache
# ---------------------------------------------------------------------------

class TestAsrCovarianceCache:
    def setup_method(self):
        # Clear module-level cache before each test
        from neurolink.dsp.baseline import _cov_cache
        _cov_cache.clear()

    def test_save_and_load_returns_same_object(self):
        cov = np.eye(4, dtype=np.float32)
        save_asr_covariance("AA:BB", "user1", cov)
        result = load_asr_covariance("AA:BB", "user1")
        assert np.array_equal(result, cov)

    def test_miss_returns_none(self):
        assert load_asr_covariance("00:00", "nobody") is None

    def test_different_device_returns_none(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        assert load_asr_covariance("CC:DD", "user1") is None

    def test_different_user_returns_none(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        assert load_asr_covariance("AA:BB", "user2") is None

    def test_overwrite_updates_value(self):
        cov1 = np.eye(4)
        cov2 = np.ones((4, 4))
        save_asr_covariance("AA:BB", "user1", cov1)
        save_asr_covariance("AA:BB", "user1", cov2)
        result = load_asr_covariance("AA:BB", "user1")
        assert np.array_equal(result, cov2)

    def test_expired_entry_returns_none(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        # Back-date the entry's timestamp beyond TTL
        from neurolink.dsp.baseline import _cov_cache
        key = "AA:BB::user1"
        _cov_cache[key]["ts"] -= (_COV_CACHE_TTL_SEC + 1)
        result = load_asr_covariance("AA:BB", "user1")
        assert result is None

    def test_expired_entry_evicted_from_cache(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        from neurolink.dsp.baseline import _cov_cache
        key = "AA:BB::user1"
        _cov_cache[key]["ts"] -= (_COV_CACHE_TTL_SEC + 1)
        load_asr_covariance("AA:BB", "user1")  # triggers eviction
        assert key not in _cov_cache

    def test_fresh_entry_not_expired(self):
        cov = np.eye(4)
        save_asr_covariance("AA:BB", "user1", cov)
        result = load_asr_covariance("AA:BB", "user1")
        assert result is not None

    def test_non_array_covariance_stored_and_retrieved(self):
        """Cache is type-agnostic; any object can be stored."""
        save_asr_covariance("AA:BB", "user1", {"foo": 42})
        result = load_asr_covariance("AA:BB", "user1")
        assert result == {"foo": 42}
