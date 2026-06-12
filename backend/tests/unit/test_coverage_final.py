"""Final coverage sweep — targets every remaining uncovered branch.

Modules covered
---------------
dsp/baseline.py        — BaselineRecorder (all 3 phases, reset, bell idempotency)
dsp/filter_toggles.py  — get_toggles / set_toggles (partial, unknown keys, non-bool)
dsp/classifiers.py     — classify_v01 regions A/C/F; classify_v2 all 9 branches;
                         compute_s_space coordinate ranges
hub.py                 — notify_baseline_complete (happy + QueueFull),
                         unregister_sse_queue (present + missing),
                         _fanout QueueFull, update() muse_ble v01 branch,
                         module-level helpers (get_hub, snapshot, reset delegates)
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ===========================================================================
# dsp/baseline.py — BaselineRecorder
# ===========================================================================

def _make_baseline(phase_offset: float = 0.0):
    """Return a (BaselineRecorder, mock_asr, mock_hub) triple."""
    from neurolink.dsp.baseline import BaselineRecorder

    mock_asr = MagicMock()
    mock_hub = MagicMock()
    rec = BaselineRecorder(asr=mock_asr, hub=mock_hub)
    # Rewind start time so tests can force phase transitions instantly
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
    assert out is arr  # same object returned


def test_baseline_warmup_to_recording_transition():
    """After BASELINE_DISCARD_SEC the recorder advances to RECORDING."""
    from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, _ = _make_baseline(phase_offset=BASELINE_DISCARD_SEC + 1.0)
    arr = np.ones((5, 64), dtype=np.float32)
    rec.process(arr)

    assert rec.phase == BaselinePhase.RECORDING.value
    # First call after transition: still in RECORDING, not yet COMPLETE → ASR fed
    rec.process(arr)
    mock_asr.apply.assert_called()


def test_baseline_recording_feeds_asr():
    """During RECORDING each frame is forwarded to asr.apply()."""
    from neurolink.dsp.artifact_config import BASELINE_DISCARD_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, _ = _make_baseline(phase_offset=BASELINE_DISCARD_SEC + 1.0)
    arr = np.ones((5, 64), dtype=np.float32)
    rec.process(arr)           # transition to RECORDING
    rec.process(arr)           # first RECORDING call
    rec.process(arr)           # second RECORDING call

    assert rec.phase == BaselinePhase.RECORDING.value
    assert mock_asr.apply.call_count == 2


def test_baseline_recording_to_complete_fires_bell():
    """After BASELINE_TOTAL_SEC the recorder advances to COMPLETE and fires bell."""
    from neurolink.dsp.artifact_config import BASELINE_TOTAL_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, mock_asr, mock_hub = _make_baseline(phase_offset=BASELINE_TOTAL_SEC + 1.0)
    # Force directly into RECORDING so the next process() sees TOTAL elapsed
    from neurolink.dsp.baseline import BaselinePhase as BP
    rec._phase = BP.RECORDING

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
    rec._bell_fired = True  # bell already fired

    arr = np.ones((5, 64), dtype=np.float32)
    out = rec.process(arr)

    mock_asr.apply.assert_not_called()
    mock_hub.notify_baseline_complete.assert_not_called()
    assert out is arr


def test_baseline_bell_idempotent():
    """_fire_bell() is a no-op if already fired (_bell_fired guard)."""
    from neurolink.dsp.artifact_config import BASELINE_TOTAL_SEC

    rec, _, mock_hub = _make_baseline(phase_offset=BASELINE_TOTAL_SEC + 1.0)
    rec._bell_fired = True  # pre-set

    rec._fire_bell(elapsed=200.0)

    mock_hub.notify_baseline_complete.assert_not_called()


def test_baseline_reset_returns_to_warmup():
    """reset() returns the recorder to WARMUP and clears bell flag."""
    from neurolink.dsp.artifact_config import BASELINE_TOTAL_SEC
    from neurolink.dsp.baseline import BaselinePhase

    rec, _, _ = _make_baseline(phase_offset=BASELINE_TOTAL_SEC + 1.0)
    rec._phase = BaselinePhase.COMPLETE
    rec._bell_fired = True

    rec.reset()

    assert rec.phase == BaselinePhase.WARMUP.value
    assert rec._bell_fired is False
    assert rec.is_complete is False


def test_baseline_is_complete_false_during_warmup():
    rec, _, _ = _make_baseline()
    assert rec.is_complete is False


# ===========================================================================
# dsp/filter_toggles.py
# ===========================================================================

def test_filter_toggles_get_returns_copy():
    """get_toggles() returns a copy; mutating it does not alter global state."""
    from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles, set_toggles

    # Reset to known state
    set_toggles({"stage1_fir": True})
    snap1 = get_toggles()
    snap1.stage1_fir = False  # mutate the copy
    snap2 = get_toggles()
    assert snap2.stage1_fir is True  # global unchanged


def test_filter_toggles_set_partial_update():
    """set_toggles() accepts a partial dict and merges correctly."""
    from neurolink.dsp.filter_toggles import get_toggles, set_toggles

    set_toggles({"stage1_fir": True, "stage4_asr": True})  # reset both
    result = set_toggles({"stage4_asr": False})

    assert result.stage4_asr is False
    assert get_toggles().stage4_asr is False
    # Other fields untouched
    assert get_toggles().stage1_fir is True


def test_filter_toggles_unknown_keys_ignored():
    """Unknown keys in the update dict are silently ignored."""
    from neurolink.dsp.filter_toggles import get_toggles, set_toggles

    original = get_toggles().to_dict()
    result = set_toggles({"nonexistent_key": False, "another_bogus": True})

    for k, v in original.items():
        assert getattr(result, k) == v


def test_filter_toggles_non_bool_values_ignored():
    """Non-bool values for valid keys are silently ignored."""
    from neurolink.dsp.filter_toggles import get_toggles, set_toggles

    set_toggles({"stage3_artifact_gate": True})  # ensure it's True
    set_toggles({"stage3_artifact_gate": "yes"})  # type: ignore[arg-type]  — ignored
    assert get_toggles().stage3_artifact_gate is True


def test_filter_toggles_to_dict_roundtrip():
    """to_dict() includes all 8 stage fields."""
    from neurolink.dsp.filter_toggles import FilterToggleConfig

    cfg = FilterToggleConfig()
    d = cfg.to_dict()
    expected_keys = {
        "stage1_fir", "stage2_bad_channels", "stage3_artifact_gate",
        "stage3b_artifact_detector", "stage4_asr", "stage4b_baseline",
        "stage5_ocular", "imu_gate",
    }
    assert expected_keys == set(d.keys())
    assert all(isinstance(v, bool) for v in d.values())


# ===========================================================================
# dsp/classifiers.py — classify_v01 remaining regions
# ===========================================================================

def test_v01_region_f_delta_dominance():
    """High delta → Region F / Coagulatio (deep sleep branch)."""
    from neurolink.dsp.classifiers import classify_v01

    region, stage = classify_v01(alpha=0.05, theta=0.05, beta=0.05,
                                  delta=0.60, gamma=0.02)
    assert region == "F"
    assert stage == "Coagulatio"


def test_v01_region_c_alpha_onset():
    """Moderate alpha, beta below threshold → Region C / Albedo."""
    from neurolink.dsp.classifiers import classify_v01
    from neurolink.dsp.artifact_config import V01_ALPHA_C, V01_BETA_B

    region, stage = classify_v01(
        alpha=V01_ALPHA_C + 0.01,
        theta=0.05,
        beta=V01_BETA_B - 0.05,
        delta=0.05,
        gamma=0.02,
    )
    assert region == "C"
    assert stage == "Albedo"


def test_v01_region_a_default():
    """All thresholds unmet → Region A / Nigredo (default)."""
    from neurolink.dsp.classifiers import classify_v01

    region, stage = classify_v01(
        alpha=0.05, theta=0.05, beta=0.05, delta=0.05, gamma=0.02
    )
    assert region == "A"
    assert stage == "Nigredo"


def test_v01_multiplicatio_faa_none_allowed():
    """faa=None means the FAA gate is skipped → Multiplicatio still reached."""
    from neurolink.dsp.classifiers import classify_v01
    from neurolink.dsp.artifact_config import (
        V01_ALPHA_E, V01_THETA_E,
        V01_MULTIPLICATIO_ALPHA, V01_MULTIPLICATIO_THETA,
    )

    region, stage = classify_v01(
        alpha=V01_MULTIPLICATIO_ALPHA + 0.01,
        theta=V01_MULTIPLICATIO_THETA + 0.01,
        beta=0.05,
        delta=0.05,
        gamma=0.02,
        faa=None,
    )
    assert region == "E"
    assert stage == "Multiplicatio"


# ===========================================================================
# dsp/classifiers.py — classify_v2 full branch sweep
# ===========================================================================

def test_v2_coagulatio():
    """Heavy delta → Coagulatio (Region F)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import V2_DELTA_COAGULATIO
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(delta=V2_DELTA_COAGULATIO + 0.01, alpha=0.05,
                   theta=0.05, beta=0.05, gamma=0.02)
    )
    assert region == "F"
    assert stage == "Coagulatio"


def test_v2_sublimatio():
    """High gamma dominant → Sublimatio (Region G)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import V2_GAMMA_SUBLIMATIO
    from neurolink.models.eeg import BandPowers

    g = V2_GAMMA_SUBLIMATIO + 0.01
    region, stage = classify_v2(
        BandPowers(gamma=g, alpha=g - 0.05, theta=g - 0.05,
                   beta=0.05, delta=0.05)
    )
    assert region == "G"
    assert stage == "Sublimatio"


def test_v2_calcinatio():
    """Very high beta → Calcinatio (Region H)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import V2_BETA_CALCINATIO
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(beta=V2_BETA_CALCINATIO + 0.01, alpha=0.05,
                   theta=0.05, delta=0.05, gamma=0.02)
    )
    assert region == "H"
    assert stage == "Calcinatio"


def test_v2_multiplicatio():
    """Very high alpha + theta, low beta → Multiplicatio (Region E)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import (
        V2_ALPHA_MULTIPLICATIO, V2_THETA_RUBEDO, V2_BETA_RUBEDO_MAX,
    )
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(
            alpha=V2_ALPHA_MULTIPLICATIO + 0.01,
            theta=V2_THETA_RUBEDO + 0.01,
            beta=V2_BETA_RUBEDO_MAX - 0.01,
            delta=0.02,
            gamma=0.02,
        )
    )
    assert region == "E"
    assert stage == "Multiplicatio"


def test_v2_rubedo():
    """High alpha + theta (below Multiplicatio threshold), low beta → Rubedo."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import (
        V2_ALPHA_RUBEDO, V2_ALPHA_MULTIPLICATIO,
        V2_THETA_RUBEDO, V2_BETA_RUBEDO_MAX,
    )
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(
            alpha=V2_ALPHA_RUBEDO + 0.01,
            theta=V2_THETA_RUBEDO + 0.01,
            beta=V2_BETA_RUBEDO_MAX - 0.01,
            delta=0.02,
            gamma=0.02,
        )
    )
    assert region == "E"
    assert stage == "Rubedo"


def test_v2_solutio():
    """High theta, alpha below Rubedo threshold → Solutio (Region D)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import V2_THETA_SOLUTIO, V2_ALPHA_RUBEDO
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(
            theta=V2_THETA_SOLUTIO + 0.01,
            alpha=V2_ALPHA_RUBEDO - 0.05,
            beta=0.05,
            delta=0.02,
            gamma=0.02,
        )
    )
    assert region == "D"
    assert stage == "Solutio"


def test_v2_albedo():
    """Moderate beta dominance → Albedo (Region C)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.dsp.artifact_config import V2_BETA_ALBEDO
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(beta=V2_BETA_ALBEDO + 0.01, alpha=0.05,
                   theta=0.05, delta=0.05, gamma=0.02)
    )
    assert region == "C"
    assert stage == "Albedo"


def test_v2_nigredo_default():
    """All thresholds unmet → Nigredo (Region A)."""
    from neurolink.dsp.classifiers import classify_v2
    from neurolink.models.eeg import BandPowers

    region, stage = classify_v2(
        BandPowers(alpha=0.05, theta=0.05, beta=0.05, delta=0.05, gamma=0.02)
    )
    assert region == "A"
    assert stage == "Nigredo"


# ===========================================================================
# dsp/classifiers.py — compute_s_space
# ===========================================================================

def test_compute_s_space_ranges():
    """x, y, z are all within their defined ranges [0,10], [0,10], [0,1]."""
    from neurolink.dsp.classifiers import compute_s_space
    from neurolink.models.eeg import BandPowers

    bands = BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.1, gamma=0.05)
    coords = compute_s_space(bands)

    assert 0.0 <= coords.x <= 10.0
    assert 0.0 <= coords.y <= 10.0
    assert 0.0 <= coords.z <= 1.0


def test_compute_s_space_zero_alpha_does_not_divide_by_zero():
    """Zero alpha uses the 1e-6 epsilon guard without raising."""
    from neurolink.dsp.classifiers import compute_s_space
    from neurolink.models.eeg import BandPowers

    coords = compute_s_space(
        BandPowers(alpha=0.0, theta=0.0, beta=0.3, delta=0.1, gamma=0.02)
    )
    assert coords.x == pytest.approx(10.0)  # clamped at max
    assert coords.y == pytest.approx(0.0)   # alpha*theta = 0


def test_compute_s_space_high_gamma_clamps_z():
    """Very high gamma clamps z to 1.0."""
    from neurolink.dsp.classifiers import compute_s_space
    from neurolink.models.eeg import BandPowers

    coords = compute_s_space(
        BandPowers(alpha=0.0, theta=0.0, beta=0.0, delta=0.0, gamma=1.0)
    )
    assert coords.z == pytest.approx(1.0)


# ===========================================================================
# hub.py — notify_baseline_complete
# ===========================================================================

def test_hub_notify_baseline_complete_delivers_sentinel():
    """Registered SSE queues receive the baseline_complete sentinel."""
    from neurolink.hub import EEGHub, _BASELINE_COMPLETE_EVENT

    hub = EEGHub()
    q = asyncio.Queue(maxsize=10)
    hub.register_sse_queue(q)

    hub.notify_baseline_complete()

    assert not q.empty()
    item = q.get_nowait()
    assert item == _BASELINE_COMPLETE_EVENT


def test_hub_notify_baseline_complete_queue_full_suppressed():
    """A full SSE queue must not raise — QueueFull is swallowed."""
    from neurolink.hub import EEGHub

    hub = EEGHub()
    q = asyncio.Queue(maxsize=1)
    q.put_nowait({"dummy": True})  # fill it up
    hub.register_sse_queue(q)

    hub.notify_baseline_complete()   # must not raise


def test_hub_notify_baseline_complete_no_queues():
    """notify_baseline_complete() with no subscribers must not raise."""
    from neurolink.hub import EEGHub

    hub = EEGHub()
    hub.notify_baseline_complete()   # must not raise


# ===========================================================================
# hub.py — unregister_sse_queue
# ===========================================================================

def test_hub_unregister_present_queue():
    """Unregistering a present queue removes it from the fan-out list."""
    from neurolink.hub import EEGHub

    hub = EEGHub()
    q = asyncio.Queue()
    hub.register_sse_queue(q)
    hub.unregister_sse_queue(q)

    # After removal no items should arrive in q
    hub.notify_baseline_complete()
    assert q.empty()


def test_hub_unregister_missing_queue_no_raise():
    """Unregistering a queue that was never registered is a silent no-op."""
    from neurolink.hub import EEGHub

    hub = EEGHub()
    q = asyncio.Queue()
    hub.unregister_sse_queue(q)   # must not raise


# ===========================================================================
# hub.py — _fanout QueueFull suppressed
# ===========================================================================

def test_hub_fanout_queue_full_suppressed():
    """A full SSE queue during _fanout must not raise."""
    from neurolink.hub import EEGHub
    from neurolink.models.eeg import NeurolinkState

    hub = EEGHub()
    q = asyncio.Queue(maxsize=1)
    q.put_nowait(NeurolinkState())   # fill it
    hub.register_sse_queue(q)

    hub._fanout(NeurolinkState())    # must not raise


# ===========================================================================
# hub.py — update() muse_ble branch exercises classify_v01
# ===========================================================================

def test_hub_update_muse_ble_runs_v01():
    """update() with source='muse_ble' must produce a non-default v01 region
    when the bands clearly land in Region E (Rubedo)."""
    from neurolink.dsp.artifact_config import V01_ALPHA_E, V01_THETA_E
    from neurolink.hub import EEGHub
    from neurolink.models.eeg import BandPowers, IngestPayload

    hub = EEGHub()
    payload = IngestPayload(
        source="muse_ble",
        bands=BandPowers(
            alpha=V01_ALPHA_E + 0.01,
            theta=V01_THETA_E + 0.01,
            beta=0.05,
            delta=0.05,
            gamma=0.02,
        ),
    )
    state = hub.update(payload)
    assert state.region_v01 == "E"
    assert state.alchemical_stage_v01 == "Rubedo"


def test_hub_update_non_muse_source_skips_v01():
    """update() with source != 'muse_ble' leaves v01 at default Region A."""
    from neurolink.hub import EEGHub
    from neurolink.models.eeg import BandPowers, IngestPayload

    hub = EEGHub()
    payload = IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.1, gamma=0.05),
    )
    state = hub.update(payload)
    assert state.region_v01 == "A"
    assert state.alchemical_stage_v01 == "Nigredo"


# ===========================================================================
# hub.py — module-level delegate helpers
# ===========================================================================

def test_hub_module_level_snapshot_returns_dict():
    """Module-level snapshot() returns a plain dict."""
    import neurolink.hub as hub_module

    result = hub_module.snapshot()
    assert isinstance(result, dict)
    assert "frame_count" in result


def test_hub_module_level_reset_clears_frame_count():
    """Module-level reset() sets frame_count back to 0."""
    import neurolink.hub as hub_module
    from neurolink.models.eeg import BandPowers, IngestPayload

    hub_module.update(IngestPayload(
        source="mock",
        bands=BandPowers(alpha=0.1, theta=0.1, beta=0.1, delta=0.1, gamma=0.05),
    ))
    hub_module.reset()
    assert hub_module.get_state().frame_count == 0


def test_hub_module_level_get_hub_returns_singleton():
    """get_hub() returns the same singleton on repeated calls."""
    from neurolink.hub import get_hub

    h1 = get_hub()
    h2 = get_hub()
    assert h1 is h2


def test_hub_set_and_get_latest_sample():
    """set_latest_sample / get_latest round-trip."""
    from neurolink.hub import EEGHub
    from neurolink.hardware.base import EEGSample

    hub = EEGHub()
    assert hub.get_latest() is None

    sample = EEGSample(
        channels=[0.0] * 5,
        timestamp=1.0,
        source="mock",
        address="AA:BB",
        poor_contact=False,
        eeg_buffer=[],
        ppg_buffer=[],
        accel_buffer=[],
        gyro_buffer=[],
    )
    hub.set_latest_sample(sample)
    assert hub.get_latest() is sample
