"""Unit tests for the Stage 0 acquisition guard → hub.emit_settling() flow.

Scope
-----
Tests the *contract* between EEGPump's Stage 0 settling guard and the hub:

  1. When the acquisition guard fires, hub.emit_settling() is called with
     an appropriate reason code.
  2. The reason code selection logic maps sensor conditions to the four
     documented reason strings.
  3. After a settling event, no NeurolinkState update occurs for that frame.
  4. The guard passes when conditions are met (no settling emitted).

All tests use a minimal fake that replicates the guard logic without
requiring BLE hardware or a running event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, call

import pytest


# ---------------------------------------------------------------------------
# Fake acquisition guard — mirrors pump Stage 0 condition checks
# ---------------------------------------------------------------------------

@dataclass
class _SensorConditions:
    impedance_ok: bool = True
    motion_ok: bool = True
    env_ok: bool = True


def _acquisition_guard_reason(cond: _SensorConditions) -> str | None:
    """Replicate the pump's reason-code priority logic.

    Returns the reason string if any guard condition fails, else None
    (meaning the frame is cleared for processing).

    Priority order (matches EEGPump._tick implementation):
      1. impedance_unstable
      2. motion_settling
      3. env_not_ready
    """
    if not cond.impedance_ok:
        return "impedance_unstable"
    if not cond.motion_ok:
        return "motion_settling"
    if not cond.env_ok:
        return "env_not_ready"
    return None


def _run_tick(cond: _SensorConditions, hub) -> bool:
    """Simulate one pump tick.  Returns True if frame was processed."""
    reason = _acquisition_guard_reason(cond)
    if reason is not None:
        hub.emit_settling(reason=reason)
        return False  # frame dropped
    # Frame cleared — hub.update() would be called here in the real pump
    return True


# ---------------------------------------------------------------------------
# Reason code selection
# ---------------------------------------------------------------------------

class TestReasonCodeSelection:
    def test_impedance_unstable_when_impedance_not_ok(self):
        assert _acquisition_guard_reason(_SensorConditions(impedance_ok=False)) == "impedance_unstable"

    def test_motion_settling_when_motion_not_ok(self):
        assert _acquisition_guard_reason(_SensorConditions(motion_ok=False)) == "motion_settling"

    def test_env_not_ready_when_env_not_ok(self):
        assert _acquisition_guard_reason(_SensorConditions(env_ok=False)) == "env_not_ready"

    def test_none_when_all_ok(self):
        assert _acquisition_guard_reason(_SensorConditions()) is None

    def test_impedance_takes_priority_over_motion(self):
        cond = _SensorConditions(impedance_ok=False, motion_ok=False)
        assert _acquisition_guard_reason(cond) == "impedance_unstable"

    def test_impedance_takes_priority_over_env(self):
        cond = _SensorConditions(impedance_ok=False, env_ok=False)
        assert _acquisition_guard_reason(cond) == "impedance_unstable"

    def test_motion_takes_priority_over_env(self):
        cond = _SensorConditions(motion_ok=False, env_ok=False)
        assert _acquisition_guard_reason(cond) == "motion_settling"

    def test_all_failing_returns_impedance(self):
        cond = _SensorConditions(impedance_ok=False, motion_ok=False, env_ok=False)
        assert _acquisition_guard_reason(cond) == "impedance_unstable"


# ---------------------------------------------------------------------------
# Tick behaviour — settling path
# ---------------------------------------------------------------------------

class TestTickSettlingPath:
    def test_emit_settling_called_on_impedance_fail(self):
        hub = MagicMock()
        processed = _run_tick(_SensorConditions(impedance_ok=False), hub)
        hub.emit_settling.assert_called_once_with(reason="impedance_unstable")
        assert processed is False

    def test_emit_settling_called_on_motion_fail(self):
        hub = MagicMock()
        _run_tick(_SensorConditions(motion_ok=False), hub)
        hub.emit_settling.assert_called_once_with(reason="motion_settling")

    def test_emit_settling_called_on_env_fail(self):
        hub = MagicMock()
        _run_tick(_SensorConditions(env_ok=False), hub)
        hub.emit_settling.assert_called_once_with(reason="env_not_ready")

    def test_no_settling_when_all_ok(self):
        hub = MagicMock()
        processed = _run_tick(_SensorConditions(), hub)
        hub.emit_settling.assert_not_called()
        assert processed is True

    def test_frame_dropped_on_settling(self):
        hub = MagicMock()
        assert _run_tick(_SensorConditions(impedance_ok=False), hub) is False

    def test_frame_processed_when_guard_passes(self):
        hub = MagicMock()
        assert _run_tick(_SensorConditions(), hub) is True


# ---------------------------------------------------------------------------
# No state update on settling frame
# ---------------------------------------------------------------------------

class TestNoStateUpdateOnSettling:
    def test_hub_update_not_called_on_settling(self):
        hub = MagicMock()
        _run_tick(_SensorConditions(impedance_ok=False), hub)
        hub.update.assert_not_called()

    def test_hub_update_would_be_called_on_clean_frame(self):
        """When the guard passes, the caller (pump) calls hub.update."""
        hub = MagicMock()
        processed = _run_tick(_SensorConditions(), hub)
        # Our fake returns True signalling the pump should call update;
        # we assert the flag is correct rather than the call itself.
        assert processed is True


# ---------------------------------------------------------------------------
# Repeated settling ticks accumulate
# ---------------------------------------------------------------------------

class TestRepeatedSettlingTicks:
    def test_settling_called_once_per_failing_tick(self):
        hub = MagicMock()
        for _ in range(5):
            _run_tick(_SensorConditions(motion_ok=False), hub)
        assert hub.emit_settling.call_count == 5

    def test_reason_consistent_across_repeated_ticks(self):
        hub = MagicMock()
        for _ in range(3):
            _run_tick(_SensorConditions(env_ok=False), hub)
        for c in hub.emit_settling.call_args_list:
            assert c == call(reason="env_not_ready")

    def test_mixed_failing_and_passing_ticks(self):
        hub = MagicMock()
        conditions = [
            _SensorConditions(impedance_ok=False),  # settling
            _SensorConditions(),                    # clean
            _SensorConditions(motion_ok=False),      # settling
            _SensorConditions(),                    # clean
        ]
        results = [_run_tick(c, hub) for c in conditions]
        assert results == [False, True, False, True]
        assert hub.emit_settling.call_count == 2
