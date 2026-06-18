"""Integration: hub.reset() via disconnect re-arms the BaselineRecorder.

Regression guard for the bug where service.disconnect() called
hub.reset() directly, leaving BaselineRecorder in COMPLETE phase so
that a subsequent reconnect silently skipped the 150 s window.

Fix: EEGPump.reset() calls self._baseline.reset() then self._hub.reset().
service.disconnect() now calls pump.reset() instead of hub.reset().
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from neurolink.dsp.asr import ArtifactSubspaceReconstructor
from neurolink.dsp.baseline import BaselinePhase, BaselineRecorder
from neurolink.eeg_pump import EEGPump
from neurolink.hardware.base import HardwareAdapter
from neurolink.hub import EEGHub

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hub() -> EEGHub:
    return EEGHub()


def _make_recorder(hub: EEGHub) -> BaselineRecorder:
    asr = ArtifactSubspaceReconstructor()
    return BaselineRecorder(asr=asr, hub=hub)


def _make_pump(hub: EEGHub) -> EEGPump:
    adapter = MagicMock(spec=HardwareAdapter)
    adapter.source_name = "mock"
    return EEGPump(adapter, hub)


# ---------------------------------------------------------------------------
# BaselineRecorder.reset() unit tests
# ---------------------------------------------------------------------------


class TestBaselineRecorderReset:
    def test_reset_restores_warmup_phase(self):
        """reset() moves a COMPLETE recorder back to WARMUP."""
        hub = _make_hub()
        rec = _make_recorder(hub)
        rec._phase = BaselinePhase.COMPLETE
        rec._bell_fired = True

        rec.reset()

        assert rec.phase == "warmup"
        assert rec._bell_fired is False

    def test_reset_resets_start_ts(self):
        """reset() refreshes _start_ts so elapsed restarts from ~0."""
        hub = _make_hub()
        rec = _make_recorder(hub)
        rec._start_ts = time.monotonic() - 200  # simulate old session

        rec.reset()

        elapsed_after = time.monotonic() - rec._start_ts
        assert elapsed_after < 1.0

    def test_reset_from_recording_phase(self):
        """reset() also works when phase is RECORDING (mid-baseline)."""
        hub = _make_hub()
        rec = _make_recorder(hub)
        rec._phase = BaselinePhase.RECORDING

        rec.reset()

        assert rec.phase == "warmup"


# ---------------------------------------------------------------------------
# EEGPump.reset() integration tests
# ---------------------------------------------------------------------------


class TestEEGPumpReset:
    def test_pump_reset_calls_baseline_reset(self):
        """EEGPump.reset() drives BaselineRecorder back to WARMUP."""
        hub = _make_hub()
        pump = _make_pump(hub)

        pump._baseline._phase = BaselinePhase.COMPLETE
        pump._baseline._bell_fired = True

        pump.reset()

        assert pump._baseline.phase == "warmup"
        assert pump._baseline._bell_fired is False

    def test_pump_reset_also_resets_hub(self):
        """EEGPump.reset() clears hub state (frame_count back to 0)."""
        hub = _make_hub()
        pump = _make_pump(hub)

        # Simulate hub having accumulated frames
        with hub._lock:
            hub._state.frame_count = 42

        pump.reset()

        assert hub.get_state().frame_count == 0

    def test_reconnect_sees_fresh_window(self):
        """After pump.reset(), elapsed is near zero and phase is warmup."""
        hub = _make_hub()
        pump = _make_pump(hub)

        pump._baseline._phase = BaselinePhase.COMPLETE
        pump._baseline._start_ts = time.monotonic() - 300

        pump.reset()

        elapsed_after = time.monotonic() - pump._baseline._start_ts
        assert elapsed_after < 1.0
        assert pump._baseline.phase == "warmup"

    def test_pump_reset_is_idempotent(self):
        """Calling pump.reset() twice leaves the recorder in WARMUP."""
        hub = _make_hub()
        pump = _make_pump(hub)

        pump._baseline._phase = BaselinePhase.COMPLETE
        pump.reset()
        pump.reset()

        assert pump._baseline.phase == "warmup"
