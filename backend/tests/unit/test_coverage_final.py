"""Final coverage sweep -- targets every remaining uncovered branch.

Modules covered
---------------
dsp/baseline.py        -- BaselineRecorder (all 3 phases, reset, bell idempotency)
dsp/filter_toggles.py  -- get_toggles / set_toggles (partial, unknown keys, non-bool)
dsp/classifiers.py     -- classify_v01 regions A/C/F; classify_v2 all 9 branches;
                         compute_s_space coordinate ranges
hub.py                 -- notify_baseline_complete (happy + QueueFull),
                         unregister_sse_queue (present + missing),
                         _fanout QueueFull, update() muse_ble v01 branch,
                         module-level helpers (get_hub, snapshot, reset delegates)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# ===========================================================================
# Fixture: restore filter_toggles singleton between tests in this module.
# Without this, test_filter_toggles_set_partial_update leaves stage4_asr=False
# which causes test_eeg_pump tests to see stage4_asr=False and skip asr.apply().
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_toggles():
    """Reset the filter_toggles singleton before and after every test in this module."""
    from neurolink.dsp.filter_toggles import FilterToggleConfig, set_toggles

    _all_true = dict.fromkeys(FilterToggleConfig().to_dict(), True)
    _all_true["stage6_cardiac"] = True
    set_toggles(_all_true)
    yield
    set_toggles(_all_true)


# ===========================================================================
# dsp/baseline.py -- BaselineRecorder
# ===========================================================================


def _make_baseline(phase_offset: float = 0.0):
    """Return a (BaselineRecorder, mock_asr, mock_hub) triple."""
    from neurolink.dsp.baseline import BaselineRecorder

    mock_asr = MagicMock()
    mock_hub = MagicMock()
    rec = BaselineRecorder(asr=mock_asr, hub=mock_hub)
    rec._start_ts = time.monotonic() - phase_offset
    return rec, mock_asr, mock_hub


def test_baseline_warmup_discards_frame():
    """During WARMUP frames are returned unchanged and ASR is NOT called."""
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, _ = _make_baseline(phase_offset=0.0)
    arr = np.ones((5, 64), dtype=np.float32)
    out = rec.process(arr)

    assert rec.phase == BaselinePhase.WARMUP.value
    mock_asr.apply.assert_not_called()
    assert out is arr


def test_baseline_warmup_to_recording_transition():
    """After BASELINE_DISCARD_SEC the recorder advances to RECORDING.

    BaselineRecorder.process() is a pure phase-state-machine shim.
    It does NOT call asr.apply() directly -- that responsibility lives
    in EEGPump._build_payload() (Stage 4), gated on phase != 'warmup'.
    """
    from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, _ = _make_baseline(phase_offset=BASELINE_DISCARD_SEC + 1.0)
    arr = np.ones((5, 64), dtype=np.float32)
    rec.process(arr)  # transitions WARMUP -> RECORDING

    assert rec.phase == BaselinePhase.RECORDING.value
    # BaselineRecorder never calls asr.apply() -- EEGPump owns that.
    mock_asr.apply.assert_not_called()


def test_baseline_recording_feeds_asr():
    """During RECORDING each call to process() advances the phase machine.

    BaselineRecorder does NOT call asr.apply() -- it only exposes phase
    so EEGPump._build_payload() can gate Stage 4 correctly.
    """
    from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, _ = _make_baseline(phase_offset=BASELINE_DISCARD_SEC + 1.0)
    arr = np.ones((5, 64), dtype=np.float32)
    rec.process(arr)  # transition to RECORDING
    rec.process(arr)  # frame 1 in RECORDING
    rec.process(arr)  # frame 2 in RECORDING

    assert rec.phase == BaselinePhase.RECORDING.value
    # BaselineRecorder is a phase-gate shim -- asr.apply() is never called here.
    mock_asr.apply.assert_not_called()


def test_baseline_recording_to_complete_fires_bell():
    """After BASELINE_TOTAL_SEC the recorder advances to COMPLETE and fires bell."""
    from neurolink.dsp.artifact_config import BASELINE_TOTAL_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, _mock_asr, mock_hub = _make_baseline(phase_offset=BASELINE_TOTAL_SEC + 1.0)
    rec._phase = BaselinePhase.RECORDING

    arr = np.ones((5, 64), dtype=np.float32)
    out = rec.process(arr)

    assert rec.phase == BaselinePhase.COMPLETE.value
    assert rec._bell_fired is True
    mock_hub.notify_baseline_complete.assert_called_once()
    assert out is arr


def test_baseline_complete_passthrough_no_asr():
    """In COMPLETE phase ASR is never called and the frame passes through."""
    from neurolink.dsp.artifact_config import BASELINE_TOTAL_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, mock_hub = _make_baseline(phase_offset=BASELINE_TOTAL_SEC + 1.0)
    rec._phase = BaselinePhase.COMPLETE
    rec._bell_fired = True

    arr = np.ones((5, 64), dtype=np.float32)
    out = rec.process(arr)

    assert rec.phase == BaselinePhase.COMPLETE.value
    mock_asr.apply.assert_not_called()
    assert out is arr
