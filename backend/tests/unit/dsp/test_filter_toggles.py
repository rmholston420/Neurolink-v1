"""Unit tests for dsp/filter_toggles.py — runtime pipeline stage toggles."""

from __future__ import annotations

import threading

import pytest

import neurolink.dsp.filter_toggles as ft
from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles, set_toggles


# ---------------------------------------------------------------------------
# Fixtures: restore the singleton between tests so tests do not bleed state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_toggles():
    """Reset the module-level singleton before and after each test.

    Resets both the 8 public fields (via to_dict()) AND stage6_cardiac
    so that no test can leave the singleton in a dirty state.
    """
    _all_true = {k: True for k in FilterToggleConfig().to_dict()}
    _all_true["stage6_cardiac"] = True
    set_toggles(_all_true)
    yield
    set_toggles(_all_true)


# ---------------------------------------------------------------------------
# FilterToggleConfig dataclass
# ---------------------------------------------------------------------------

class TestFilterToggleConfig:
    def test_all_defaults_true(self):
        cfg = FilterToggleConfig()
        for name, val in cfg.to_dict().items():
            assert val is True, f"{name} default should be True"

    def test_to_dict_returns_dict(self):
        assert isinstance(FilterToggleConfig().to_dict(), dict)

    def test_to_dict_keys_match_fields(self):
        """to_dict() returns the 8 public stage keys (stage6_cardiac excluded)."""
        cfg = FilterToggleConfig()
        d = cfg.to_dict()
        expected = {
            "stage1_fir",
            "stage2_bad_channels",
            "stage3_artifact_gate",
            "stage3b_artifact_detector",
            "stage4_asr",
            "stage4b_baseline",
            "stage5_ocular",
            "imu_gate",
        }
        assert set(d.keys()) == expected

    def test_stage6_cardiac_field_exists(self):
        """Stage 6 cardiac toggle must be present as a dataclass field."""
        assert hasattr(FilterToggleConfig(), "stage6_cardiac")

    def test_stage6_cardiac_default_true(self):
        assert FilterToggleConfig().stage6_cardiac is True

    def test_all_stage_fields_present(self):
        """All 8 public fields appear in to_dict(); stage6_cardiac on the object."""
        cfg = FilterToggleConfig()
        expected_public = {
            "stage1_fir",
            "stage2_bad_channels",
            "stage3_artifact_gate",
            "stage3b_artifact_detector",
            "stage4_asr",
            "stage4b_baseline",
            "stage5_ocular",
            "imu_gate",
        }
        assert expected_public.issubset(set(cfg.to_dict().keys()))
        # stage6_cardiac is a dataclass field, just not in to_dict()
        assert hasattr(cfg, "stage6_cardiac")

    def test_individual_field_override(self):
        cfg = FilterToggleConfig(stage6_cardiac=False)
        assert cfg.stage6_cardiac is False
        assert cfg.stage1_fir is True  # others unchanged


# ---------------------------------------------------------------------------
# get_toggles() — snapshot isolation
# ---------------------------------------------------------------------------

class TestGetToggles:
    def test_returns_filter_toggle_config(self):
        assert isinstance(get_toggles(), FilterToggleConfig)

    def test_returns_copy_not_singleton(self):
        a = get_toggles()
        b = get_toggles()
        assert a is not b

    def test_mutating_snapshot_does_not_affect_singleton(self):
        snap = get_toggles()
        snap.stage6_cardiac = False
        # Singleton should still report True
        assert get_toggles().stage6_cardiac is True

    def test_defaults_all_true_on_fresh_singleton(self):
        for name, val in get_toggles().to_dict().items():
            assert val is True, f"{name} should default True"


# ---------------------------------------------------------------------------
# set_toggles() — merge semantics
# ---------------------------------------------------------------------------

class TestSetToggles:
    def test_returns_filter_toggle_config(self):
        result = set_toggles({"stage6_cardiac": False})
        assert isinstance(result, FilterToggleConfig)

    def test_single_key_update(self):
        set_toggles({"stage6_cardiac": False})
        assert get_toggles().stage6_cardiac is False

    def test_partial_update_preserves_other_fields(self):
        set_toggles({"stage6_cardiac": False})
        cfg = get_toggles()
        assert cfg.stage1_fir is True
        assert cfg.stage5_ocular is True

    def test_unknown_key_silently_ignored(self):
        """Unknown keys must not raise and must not corrupt existing state."""
        before = get_toggles().to_dict()
        set_toggles({"nonexistent_stage": False})
        after = get_toggles().to_dict()
        assert before == after

    def test_non_bool_value_silently_ignored(self):
        """Only bool values are accepted; other types are silently dropped."""
        set_toggles({"stage6_cardiac": "off"})  # type: ignore[arg-type]
        assert get_toggles().stage6_cardiac is True

    def test_empty_dict_is_noop(self):
        before = get_toggles().to_dict()
        set_toggles({})
        after = get_toggles().to_dict()
        assert before == after

    def test_multiple_keys_updated_atomically(self):
        set_toggles({"stage4_asr": False, "stage5_ocular": False})
        cfg = get_toggles()
        assert cfg.stage4_asr is False
        assert cfg.stage5_ocular is False

    def test_re_enable_after_disable(self):
        set_toggles({"stage6_cardiac": False})
        assert get_toggles().stage6_cardiac is False
        set_toggles({"stage6_cardiac": True})
        assert get_toggles().stage6_cardiac is True

    def test_all_stages_can_be_disabled(self):
        # Include stage6_cardiac explicitly since it is not in to_dict()
        all_false = {k: False for k in FilterToggleConfig().to_dict()}
        all_false["stage6_cardiac"] = False
        set_toggles(all_false)
        for name, val in get_toggles().to_dict().items():
            assert val is False, f"{name} should be False"
        assert get_toggles().stage6_cardiac is False


# ---------------------------------------------------------------------------
# stage6_cardiac wiring — integration with cardiac_regression
# ---------------------------------------------------------------------------

class TestStage6CardiacWiring:
    """Verify the toggle bool is correctly forwarded to the stage."""

    def test_stage6_cardiac_true_by_default(self):
        toggles = get_toggles()
        assert toggles.stage6_cardiac is True

    def test_disable_stage6_cardiac(self):
        set_toggles({"stage6_cardiac": False})
        toggles = get_toggles()
        assert toggles.stage6_cardiac is False

    def test_toggle_bool_gates_cardiac_regressor(self):
        """Simulate the EEGPump Stage 6 guard: only call regressor when toggle is True."""
        import numpy as np
        from neurolink.dsp.cardiac_regression import CardiacRegressor

        reg = CardiacRegressor()
        eeg = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)
        ibis = [800.0] * 5

        # With toggle ON: regressor.apply() is called
        set_toggles({"stage6_cardiac": True})
        toggles = get_toggles()
        if toggles.stage6_cardiac:
            out = reg.apply(eeg, ibis)
        else:
            out = eeg
        assert out.shape == eeg.shape

        # With toggle OFF: regressor.apply() is skipped; raw eeg returned
        set_toggles({"stage6_cardiac": False})
        toggles = get_toggles()
        if toggles.stage6_cardiac:
            out = reg.apply(eeg, ibis)
        else:
            out = eeg
        assert out is eeg  # exact identity: not even a copy


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestFilterTogglesThreadSafety:
    def test_concurrent_get_and_set_no_exception(self):
        errors: list[Exception] = []

        def getter():
            try:
                for _ in range(50):
                    get_toggles()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def setter():
            try:
                for _ in range(20):
                    set_toggles({"stage6_cardiac": False})
                    set_toggles({"stage6_cardiac": True})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=getter) for _ in range(3)]
            + [threading.Thread(target=setter) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_snapshot_isolation_under_concurrent_writes(self):
        """A snapshot taken before a set_toggles must not be mutated by it."""
        snap = get_toggles()
        snap_stage6 = snap.stage6_cardiac

        set_toggles({"stage6_cardiac": not snap_stage6})
        # snap is a copy; its value must be unchanged
        assert snap.stage6_cardiac == snap_stage6
