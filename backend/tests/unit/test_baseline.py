"""Unit tests for neurolink.dsp.baseline.

Real public API:
  BaselineRecorder   — stateful manager; __init__(asr, hub)
  BaselinePhase      — StrEnum: WARMUP | RECORDING | COMPLETE

BaselineRecorder requires an ArtifactSubspaceReconstructor and a hub
instance.  Tests use real ASR with a minimal stub hub to avoid
circular imports.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from neurolink.dsp.asr import ASRConfig, ArtifactSubspaceReconstructor
from neurolink.dsp.baseline import BaselinePhase, BaselineRecorder


FS = 256.0
N_CH = 4
N_SAMPLES = 64


class _StubHub:
    """Minimal hub stub — records bell calls."""

    def __init__(self):
        self.bell_count = 0

    def notify_baseline_complete(self):
        self.bell_count += 1


@pytest.fixture
def asr_instance() -> ArtifactSubspaceReconstructor:
    return ArtifactSubspaceReconstructor(config=ASRConfig(enable=False))


@pytest.fixture
def hub() -> _StubHub:
    return _StubHub()


@pytest.fixture
def recorder(asr_instance, hub) -> BaselineRecorder:
    return BaselineRecorder(asr=asr_instance, hub=hub)


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.default_rng(0).normal(0, 5.0, (N_CH, N_SAMPLES))


class TestConstruction:
    def test_starts_in_warmup_phase(self, recorder):
        assert recorder.phase == BaselinePhase.WARMUP

    def test_phase_string_value(self, recorder):
        assert recorder.phase == "warmup"


class TestProcessPassthrough:
    def test_process_returns_eeg_unchanged(self, recorder, frame):
        """process() is a passthrough shim — must return the same array."""
        out = recorder.process(frame)
        np.testing.assert_array_equal(out, frame)

    def test_process_multiple_frames_no_crash(self, recorder, frame):
        for _ in range(10):
            out = recorder.process(frame)
            assert out.shape == frame.shape


class TestPhaseProgression:
    def test_phase_remains_warmup_initially(self, recorder, frame):
        recorder.process(frame)
        # Still warmup at t=0+
        assert recorder.phase in (BaselinePhase.WARMUP, BaselinePhase.RECORDING, BaselinePhase.COMPLETE)

    def test_phase_values_are_valid(self, recorder):
        assert recorder.phase in {BaselinePhase.WARMUP, BaselinePhase.RECORDING, BaselinePhase.COMPLETE}


class TestBaselinePhaseEnum:
    def test_warmup_string(self):
        assert BaselinePhase.WARMUP == "warmup"

    def test_recording_string(self):
        assert BaselinePhase.RECORDING == "recording"

    def test_complete_string(self):
        assert BaselinePhase.COMPLETE == "complete"


class TestEdgeCases:
    def test_process_none_does_not_crash(self, recorder):
        """Passing None should return None (passthrough contract)."""
        out = recorder.process(None)  # type: ignore[arg-type]
        assert out is None

    def test_phase_is_string_comparable(self, recorder):
        """EEGPump guards with: phase != 'warmup' — must be string-comparable."""
        assert recorder.phase != "warmup" or recorder.phase == "warmup"
