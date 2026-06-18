"""Unit tests for the Stage 0 hardware-setup prevention subsystem.

Covers:
  - ImpedanceGuard (impedance.py)
  - IMUGate (imu_gate.py)
  - EnvironmentChecklist (environment.py)
  - Stage0Guard facade (__init__.py)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurolink.stage0 import EnvironmentChecklist, IMUGate, ImpedanceGuard, Stage0Guard
from neurolink.stage0.impedance import ImpedanceLevel, ImpedanceChannelStatus, MUSE_CHANNELS
from neurolink.stage0.environment import ENVIRONMENT_PROMPTS, _DRY_STABILISE_SEC, _SEMI_WET_STABILISE_SEC
from neurolink.hardware.base import EEGSample


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sample(
    accel: list[list[float]] | None = None,
    poor_contact: bool = False,
) -> EEGSample:
    """Return a minimal EEGSample with controllable accel_buffer and extra{}."""
    sample = MagicMock(spec=EEGSample)
    sample.accel_buffer = accel
    sample.poor_contact = poor_contact
    sample.extra = {}
    return sample


def _still_accel(n: int = 10) -> list[list[float]]:
    """3×N accel buffer representing near-zero motion (gravity on z only)."""
    return [
        [0.0] * n,          # ax
        [0.0] * n,          # ay
        [1.0] * n,          # az  (gravity)
    ]


def _motion_accel(n: int = 10, magnitude: float = 0.5) -> list[list[float]]:
    """3×N accel buffer with clear motion signal."""
    return [
        [magnitude] * n,
        [magnitude] * n,
        [1.0 + magnitude] * n,
    ]


# ═════════════════════════════════════════════════════════════════════════════
# ImpedanceGuard
# ═════════════════════════════════════════════════════════════════════════════

class TestImpedanceGuardInit:
    def test_default_electrode_type_is_dry(self):
        g = ImpedanceGuard()
        assert g.electrode_type == "dry"

    def test_dry_threshold_is_200(self):
        g = ImpedanceGuard(electrode_type="dry")
        assert g.threshold_kohm == 200.0

    def test_semi_wet_threshold_is_20(self):
        g = ImpedanceGuard(electrode_type="semi_wet")
        assert g.threshold_kohm == 20.0

    def test_wet_threshold_is_5(self):
        g = ImpedanceGuard(electrode_type="wet")
        assert g.threshold_kohm == 5.0

    def test_unknown_electrode_type_defaults_to_200(self):
        g = ImpedanceGuard(electrode_type="gel")
        assert g.threshold_kohm == 200.0

    def test_initial_all_channels_ok(self):
        g = ImpedanceGuard()
        assert g.all_channels_ok is True

    def test_initial_bad_channels_empty(self):
        g = ImpedanceGuard()
        assert g.bad_channels == []

    def test_channels_labelled_with_muse_names(self):
        g = ImpedanceGuard(n_channels=5)
        status = g.summary_dict()
        labels = [ch["label"] for ch in status["channels"]]
        assert labels[:5] == MUSE_CHANNELS

    def test_extra_channels_labelled_chN(self):
        g = ImpedanceGuard(n_channels=7)
        status = g.summary_dict()
        assert status["channels"][5]["label"] == "CH5"
        assert status["channels"][6]["label"] == "CH6"


class TestImpedanceGuardUpdateFromSample:
    def test_poor_contact_true_sets_high_level(self):
        g = ImpedanceGuard()
        g.update_from_sample(poor_contact=True)
        assert g.all_channels_ok is False

    def test_poor_contact_true_populates_bad_channels(self):
        g = ImpedanceGuard()
        g.update_from_sample(poor_contact=True)
        bad = g.bad_channels
        # AUX is skipped, so 4 non-AUX channels should be bad
        assert "AUX" not in bad
        assert len(bad) == 4

    def test_poor_contact_false_from_unknown_sets_ok(self):
        g = ImpedanceGuard()
        g.update_from_sample(poor_contact=False)
        for label in ["TP9", "AF7", "AF8", "TP10"]:
            ch = g.channel_status(label)
            assert ch.level == ImpedanceLevel.OK

    def test_poor_contact_false_after_high_stays_high(self):
        """Boolean flag going False does NOT clear a HIGH set by a previous True."""
        g = ImpedanceGuard()
        g.update_from_sample(poor_contact=True)
        g.update_from_sample(poor_contact=False)
        # HIGH is sticky; only a good kohm reading clears it
        assert g.all_channels_ok is False

    def test_aux_channel_skipped_by_update_from_sample(self):
        g = ImpedanceGuard()
        g.update_from_sample(poor_contact=True)
        aux = g.channel_status("AUX")
        assert aux.level == ImpedanceLevel.UNKNOWN


class TestImpedanceGuardUpdateFromKohm:
    def test_above_threshold_sets_high(self):
        g = ImpedanceGuard(electrode_type="dry")  # threshold 200
        g.update_from_kohm({"TP9": 250.0})
        assert g.channel_status("TP9").level == ImpedanceLevel.HIGH
        assert g.all_channels_ok is False

    def test_below_threshold_sets_ok(self):
        g = ImpedanceGuard(electrode_type="dry")
        g.update_from_kohm({"TP9": 150.0})
        assert g.channel_status("TP9").level == ImpedanceLevel.OK
        assert g.all_channels_ok is True

    def test_exact_threshold_is_ok(self):
        g = ImpedanceGuard(electrode_type="dry")
        g.update_from_kohm({"TP9": 200.0})
        assert g.channel_status("TP9").level == ImpedanceLevel.OK

    def test_kohm_stored_on_channel(self):
        g = ImpedanceGuard()
        g.update_from_kohm({"AF7": 45.2})
        assert g.channel_status("AF7").kohm == pytest.approx(45.2)

    def test_unknown_label_ignored(self):
        g = ImpedanceGuard()
        g.update_from_kohm({"UNKNOWN_CH": 999.0})
        assert g.all_channels_ok is True

    def test_multiple_channels_updated(self):
        g = ImpedanceGuard(electrode_type="dry")
        g.update_from_kohm({"TP9": 10.0, "AF7": 300.0, "AF8": 5.0})
        assert g.channel_status("TP9").level == ImpedanceLevel.OK
        assert g.channel_status("AF7").level == ImpedanceLevel.HIGH
        assert g.channel_status("AF8").level == ImpedanceLevel.OK
        assert "AF7" in g.bad_channels


class TestImpedanceGuardSummaryDict:
    def test_summary_dict_keys(self):
        g = ImpedanceGuard()
        d = g.summary_dict()
        assert "electrode_type" in d
        assert "threshold_kohm" in d
        assert "all_channels_ok" in d
        assert "bad_channels" in d
        assert "channels" in d

    def test_channel_to_dict_keys(self):
        g = ImpedanceGuard()
        ch = g.channel_status("TP9")
        d = ch.to_dict()
        for key in ("label", "kohm", "poor_contact", "level", "threshold_kohm", "last_updated"):
            assert key in d

    def test_channel_status_returns_none_for_unknown_label(self):
        g = ImpedanceGuard()
        assert g.channel_status("DOES_NOT_EXIST") is None


# ═════════════════════════════════════════════════════════════════════════════
# IMUGate
# ═════════════════════════════════════════════════════════════════════════════

class TestIMUGateInit:
    def test_initial_not_flagged(self):
        gate = IMUGate()
        assert gate.is_flagged is False

    def test_initial_last_rms_zero(self):
        gate = IMUGate()
        assert gate.last_rms == 0.0

    def test_threshold_stored(self):
        gate = IMUGate(threshold_g=0.3)
        assert gate.threshold_g == pytest.approx(0.3)


class TestIMUGateFlagSegment:
    def test_none_accel_buffer_sets_motion_flagged_false(self):
        gate = IMUGate()
        sample = _make_sample(accel=None)
        out = gate.flag_segment(sample)
        assert out.extra["motion_flagged"] is False
        assert out.extra["motion_rms"] == 0.0

    def test_short_accel_buffer_no_flag(self):
        """Buffer with fewer than 3 rows is treated as no accel data."""
        gate = IMUGate()
        sample = _make_sample(accel=[[0.1, 0.2]])  # only 1 row
        out = gate.flag_segment(sample)
        assert out.extra["motion_flagged"] is False

    def test_still_sample_below_threshold_not_flagged(self):
        gate = IMUGate(threshold_g=0.15)
        sample = _make_sample(accel=_still_accel())
        out = gate.flag_segment(sample)
        assert out.extra["motion_flagged"] is False
        assert gate.is_flagged is False

    def test_motion_sample_above_threshold_flagged(self):
        gate = IMUGate(threshold_g=0.15)
        sample = _make_sample(accel=_motion_accel(magnitude=1.0))
        out = gate.flag_segment(sample)
        assert out.extra["motion_flagged"] is True
        assert gate.is_flagged is True

    def test_rms_stored_in_extra(self):
        gate = IMUGate(threshold_g=0.15)
        sample = _make_sample(accel=_motion_accel(magnitude=0.5))
        out = gate.flag_segment(sample)
        assert out.extra["motion_rms"] > 0.0

    def test_motion_ts_set_on_valid_sample(self):
        gate = IMUGate()
        sample = _make_sample(accel=_still_accel())
        before = time.time()
        gate.flag_segment(sample)
        assert sample.extra["motion_ts"] >= before

    def test_returns_same_sample_object(self):
        gate = IMUGate()
        sample = _make_sample(accel=_still_accel())
        out = gate.flag_segment(sample)
        assert out is sample

    def test_window_rms_drops_after_still_samples(self):
        """After many still samples the window RMS should fall below threshold."""
        gate = IMUGate(threshold_g=0.15, window_samples=5)
        # First: inject motion
        for _ in range(5):
            gate.flag_segment(_make_sample(accel=_motion_accel(magnitude=1.0)))
        assert gate.is_flagged is True
        # Then: inject still samples to flush the window
        for _ in range(10):
            gate.flag_segment(_make_sample(accel=_still_accel()))
        assert gate.is_flagged is False


class TestIMUGateComputeRms:
    def test_zero_accel_gives_zero_rms(self):
        rms = IMUGate._compute_rms([[0.0] * 10, [0.0] * 10, [0.0] * 10])
        assert rms == pytest.approx(0.0)

    def test_gravity_only_z_gives_near_zero_after_demean(self):
        """Constant z=1 g (gravity) demeaned to 0, so RMS ~= 0."""
        rms = IMUGate._compute_rms([[0.0] * 10, [0.0] * 10, [1.0] * 10])
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_xy_contributes_to_rms(self):
        rms = IMUGate._compute_rms([[0.5] * 10, [0.5] * 10, [1.0] * 10])
        assert rms > 0.0


class TestIMUGateStatusDict:
    def test_status_dict_keys(self):
        gate = IMUGate()
        d = gate.status_dict()
        for key in ("flagged", "motion_rms_g", "threshold_g", "last_ts"):
            assert key in d

    def test_status_dict_reflects_flag_state(self):
        gate = IMUGate(threshold_g=0.15)
        gate.flag_segment(_make_sample(accel=_motion_accel(magnitude=1.0)))
        d = gate.status_dict()
        assert d["flagged"] is True
        assert d["motion_rms_g"] > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# EnvironmentChecklist
# ═════════════════════════════════════════════════════════════════════════════

class TestEnvironmentChecklistInit:
    def test_dry_stabilise_duration(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        assert ec._stabilise_duration == _DRY_STABILISE_SEC

    def test_semi_wet_stabilise_duration(self):
        ec = EnvironmentChecklist(electrode_type="semi_wet")
        assert ec._stabilise_duration == _SEMI_WET_STABILISE_SEC

    def test_initial_acked_empty(self):
        ec = EnvironmentChecklist()
        assert ec.all_steps_acked is False

    def test_is_ready_false_initially(self):
        ec = EnvironmentChecklist()
        assert ec.is_ready is False

    def test_stabilise_remaining_near_full_on_init(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        assert ec.stabilise_remaining_s == pytest.approx(_DRY_STABILISE_SEC, abs=1.0)


class TestEnvironmentChecklistAcknowledge:
    def test_valid_step_returns_true(self):
        ec = EnvironmentChecklist()
        assert ec.acknowledge("emi_distance") is True

    def test_invalid_step_returns_false(self):
        ec = EnvironmentChecklist()
        assert ec.acknowledge("nonexistent_step") is False

    def test_acknowledge_all_sets_all_acked(self):
        ec = EnvironmentChecklist()
        ec.acknowledge_all()
        assert ec.all_steps_acked is True

    def test_partial_ack_not_all_acked(self):
        ec = EnvironmentChecklist()
        ec.acknowledge("emi_distance")
        assert ec.all_steps_acked is False

    def test_all_prompts_can_be_acked_individually(self):
        ec = EnvironmentChecklist()
        for p in ENVIRONMENT_PROMPTS:
            ec.acknowledge(p["id"])
        assert ec.all_steps_acked is True


class TestEnvironmentChecklistStabilise:
    def test_stabilise_complete_when_time_elapsed(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        # Backdate start time so countdown has expired
        ec._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        assert ec.stabilise_complete is True
        assert ec.stabilise_remaining_s == 0.0

    def test_stabilise_remaining_floored_at_zero(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        ec._start_ts = time.time() - 9999.0
        assert ec.stabilise_remaining_s == 0.0

    def test_stabilise_not_complete_initially(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        assert ec.stabilise_complete is False


class TestEnvironmentChecklistIsReady:
    def test_ready_when_acked_and_stabilised(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        ec.acknowledge_all()
        ec._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        assert ec.is_ready is True

    def test_not_ready_when_acked_but_not_stabilised(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        ec.acknowledge_all()
        assert ec.is_ready is False

    def test_not_ready_when_stabilised_but_not_acked(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        ec._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        assert ec.is_ready is False


class TestEnvironmentChecklistReset:
    def test_reset_clears_acked(self):
        ec = EnvironmentChecklist()
        ec.acknowledge_all()
        ec.reset()
        assert ec.all_steps_acked is False

    def test_reset_restarts_countdown(self):
        ec = EnvironmentChecklist(electrode_type="dry")
        ec._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        assert ec.stabilise_complete is True
        ec.reset()
        assert ec.stabilise_complete is False


class TestEnvironmentChecklistStatusDict:
    def test_status_dict_keys(self):
        ec = EnvironmentChecklist()
        d = ec.status_dict()
        for key in (
            "is_ready",
            "stabilise_remaining_s",
            "stabilise_complete",
            "all_steps_acked",
            "acked_steps",
            "prompts",
        ):
            assert key in d

    def test_prompts_include_acked_field(self):
        ec = EnvironmentChecklist()
        ec.acknowledge("emi_distance")
        d = ec.status_dict()
        acked_map = {p["id"]: p["acked"] for p in d["prompts"]}
        assert acked_map["emi_distance"] is True
        assert acked_map["phone_distance"] is False

    def test_acked_steps_sorted(self):
        ec = EnvironmentChecklist()
        ec.acknowledge_all()
        d = ec.status_dict()
        assert d["acked_steps"] == sorted(d["acked_steps"])


# ═════════════════════════════════════════════════════════════════════════════
# Stage0Guard facade
# ═════════════════════════════════════════════════════════════════════════════

class TestStage0GuardInit:
    def test_creates_sub_components(self):
        g = Stage0Guard()
        assert isinstance(g.impedance, ImpedanceGuard)
        assert isinstance(g.imu, IMUGate)
        assert isinstance(g.environment, EnvironmentChecklist)

    def test_electrode_type_propagated_to_impedance(self):
        g = Stage0Guard(electrode_type="semi_wet")
        assert g.impedance.threshold_kohm == pytest.approx(20.0)


class TestStage0GuardGateSample:
    def test_gate_sample_returns_none_for_none(self):
        g = Stage0Guard()
        assert g.gate_sample(None) is None

    def test_gate_sample_returns_sample_with_motion_extra(self):
        g = Stage0Guard()
        sample = _make_sample(accel=_still_accel())
        out = g.gate_sample(sample)
        assert out is sample
        assert "motion_flagged" in out.extra

    def test_gate_sample_with_motion_sets_flag(self):
        g = Stage0Guard()
        sample = _make_sample(accel=_motion_accel(magnitude=1.0))
        g.gate_sample(sample)
        assert sample.extra["motion_flagged"] is True


class TestStage0GuardAcquisitionReady:
    def test_not_ready_initially(self):
        g = Stage0Guard()
        # Environment countdown not elapsed, steps not acked
        assert g.acquisition_ready is False

    def test_ready_when_impedance_ok_and_environment_ready(self):
        g = Stage0Guard(electrode_type="dry")
        g.environment.acknowledge_all()
        g.environment._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        # Impedance defaults to all_channels_ok = True
        assert g.acquisition_ready is True

    def test_not_ready_when_impedance_bad(self):
        g = Stage0Guard(electrode_type="dry")
        g.environment.acknowledge_all()
        g.environment._start_ts = time.time() - (_DRY_STABILISE_SEC + 1.0)
        g.impedance.update_from_sample(poor_contact=True)
        assert g.acquisition_ready is False


class TestStage0GuardStatusDict:
    def test_status_dict_top_level_keys(self):
        g = Stage0Guard()
        d = g.status_dict()
        for key in ("acquisition_ready", "impedance", "imu", "environment"):
            assert key in d

    def test_status_dict_acquisition_ready_matches_property(self):
        g = Stage0Guard()
        d = g.status_dict()
        assert d["acquisition_ready"] == g.acquisition_ready
