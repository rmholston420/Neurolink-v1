"""Unit tests for dsp/filter_toggles.py."""

from __future__ import annotations

import threading

import pytest


# ---------------------------------------------------------------------------
# Helpers — always re-import the module so each test class starts fresh.
# The module-level singleton persists across tests in the same process;
# we reset it before each test using set_toggles to all-True.
# ---------------------------------------------------------------------------

def _reset():
    from neurolink.dsp.filter_toggles import FilterToggleConfig, set_toggles
    defaults = FilterToggleConfig().to_dict()
    set_toggles(defaults)


# ---------------------------------------------------------------------------
# FilterToggleConfig — dataclass
# ---------------------------------------------------------------------------

class TestFilterToggleConfig:
    def test_all_defaults_true(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig
        cfg = FilterToggleConfig()
        for k, v in cfg.to_dict().items():
            assert v is True, f"{k} should default to True"

    def test_stage6_cardiac_field_exists(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig
        assert hasattr(FilterToggleConfig(), "stage6_cardiac")

    def test_to_dict_returns_all_stage_keys(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig
        d = FilterToggleConfig().to_dict()
        expected = {
            "stage1_fir",
            "stage2_bad_channels",
            "stage3_artifact_gate",
            "stage3b_artifact_detector",
            "stage4_asr",
            "stage4b_baseline",
            "stage5_ocular",
            "stage6_cardiac",
            "imu_gate",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values_are_bool(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig
        for v in FilterToggleConfig().to_dict().values():
            assert isinstance(v, bool)

    def test_custom_construction(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig
        cfg = FilterToggleConfig(stage1_fir=False, stage6_cardiac=False)
        assert cfg.stage1_fir is False
        assert cfg.stage6_cardiac is False
        assert cfg.stage2_bad_channels is True  # others unchanged


# ---------------------------------------------------------------------------
# get_toggles()
# ---------------------------------------------------------------------------

class TestGetToggles:
    def setup_method(self):
        _reset()

    def test_returns_filter_toggle_config_instance(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles
        assert isinstance(get_toggles(), FilterToggleConfig)

    def test_default_all_true(self):
        from neurolink.dsp.filter_toggles import get_toggles
        cfg = get_toggles()
        for v in cfg.to_dict().values():
            assert v is True

    def test_returns_copy_not_singleton(self):
        from neurolink.dsp.filter_toggles import get_toggles
        cfg1 = get_toggles()
        cfg2 = get_toggles()
        assert cfg1 is not cfg2

    def test_mutating_returned_copy_does_not_affect_singleton(self):
        from neurolink.dsp.filter_toggles import get_toggles
        cfg = get_toggles()
        cfg.stage1_fir = False
        # singleton still True
        assert get_toggles().stage1_fir is True


# ---------------------------------------------------------------------------
# set_toggles()
# ---------------------------------------------------------------------------

class TestSetToggles:
    def setup_method(self):
        _reset()

    def test_set_single_key(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage1_fir": False})
        assert get_toggles().stage1_fir is False

    def test_set_stage6_cardiac_false(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage6_cardiac": False})
        assert get_toggles().stage6_cardiac is False

    def test_other_keys_unchanged_after_partial_update(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage4_asr": False})
        cfg = get_toggles()
        assert cfg.stage4_asr is False
        assert cfg.stage1_fir is True
        assert cfg.stage6_cardiac is True

    def test_returns_new_config(self):
        from neurolink.dsp.filter_toggles import FilterToggleConfig, set_toggles
        result = set_toggles({"stage1_fir": False})
        assert isinstance(result, FilterToggleConfig)
        assert result.stage1_fir is False

    def test_unknown_key_silently_ignored(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"nonexistent_key": False})
        cfg = get_toggles()
        for v in cfg.to_dict().values():
            assert v is True

    def test_non_bool_value_silently_ignored(self):
        """Non-bool values are rejected; field stays True."""
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage1_fir": "off"})  # type: ignore[arg-type]
        assert get_toggles().stage1_fir is True

    def test_set_multiple_keys(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage1_fir": False, "stage6_cardiac": False, "imu_gate": False})
        cfg = get_toggles()
        assert cfg.stage1_fir is False
        assert cfg.stage6_cardiac is False
        assert cfg.imu_gate is False

    def test_re_enable_after_disable(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({"stage1_fir": False})
        set_toggles({"stage1_fir": True})
        assert get_toggles().stage1_fir is True

    def test_empty_dict_is_noop(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        set_toggles({})
        for v in get_toggles().to_dict().values():
            assert v is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestFilterTogglesThreadSafety:
    def setup_method(self):
        _reset()

    def test_concurrent_get_and_set_does_not_raise(self):
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        errors: list[Exception] = []

        def getter():
            try:
                for _ in range(50):
                    _ = get_toggles()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def setter():
            try:
                for i in range(50):
                    set_toggles({"stage1_fir": bool(i % 2)})
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
        _reset()  # leave clean

    def test_singleton_remains_consistent_after_concurrent_writes(self):
        """After concurrent writes all values must still be valid bools."""
        from neurolink.dsp.filter_toggles import get_toggles, set_toggles
        errors: list[Exception] = []

        def flipper(key: str):
            try:
                for i in range(30):
                    set_toggles({key: bool(i % 2)})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        keys = ["stage1_fir", "stage4_asr", "stage6_cardiac", "imu_gate"]
        threads = [threading.Thread(target=flipper, args=(k,)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for v in get_toggles().to_dict().values():
            assert isinstance(v, bool)
        _reset()
