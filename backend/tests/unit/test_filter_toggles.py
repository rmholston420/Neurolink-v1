"""Unit tests for neurolink.dsp.filter_toggles.

Real public API (module-level functions + dataclass — no FilterToggles class):
  FilterToggleConfig  — dataclass with one bool per pipeline stage
  get_toggles()       -> FilterToggleConfig  (copy of singleton)
  set_toggles(dict)   -> FilterToggleConfig  (merge + return new state)
"""
from __future__ import annotations

import pytest

from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles, set_toggles


@pytest.fixture(autouse=True)
def reset_toggles():
    """Reset the module singleton to all-True defaults before each test."""
    set_toggles({field: True for field in vars(FilterToggleConfig()).keys()})
    yield
    set_toggles({field: True for field in vars(FilterToggleConfig()).keys()})


class TestDefaultState:
    def test_all_stages_enabled_by_default(self):
        cfg = get_toggles()
        assert cfg.stage1_fir is True
        assert cfg.stage2_bad_channels is True
        assert cfg.stage3_artifact_gate is True
        assert cfg.stage3b_artifact_detector is True
        assert cfg.stage4_asr is True
        assert cfg.stage4b_baseline is True
        assert cfg.stage5_ocular is True
        assert cfg.imu_gate is True

    def test_to_dict_returns_8_public_keys(self):
        d = get_toggles().to_dict()
        # stage6_cardiac is intentionally excluded from to_dict()
        assert "stage6_cardiac" not in d
        assert len(d) == 8

    def test_to_dict_all_true(self):
        d = get_toggles().to_dict()
        assert all(v is True for v in d.values())


class TestSetToggles:
    def test_disable_single_stage(self):
        set_toggles({"stage1_fir": False})
        assert get_toggles().stage1_fir is False

    def test_enable_after_disable(self):
        set_toggles({"stage4_asr": False})
        assert get_toggles().stage4_asr is False
        set_toggles({"stage4_asr": True})
        assert get_toggles().stage4_asr is True

    def test_unknown_key_silently_ignored(self):
        cfg_before = get_toggles()
        set_toggles({"nonexistent_stage": False})
        cfg_after = get_toggles()
        assert cfg_before.stage1_fir == cfg_after.stage1_fir

    def test_non_bool_value_ignored(self):
        set_toggles({"stage1_fir": "yes"})  # type: ignore[dict-item]
        assert isinstance(get_toggles().stage1_fir, bool)

    def test_multiple_keys_in_one_call(self):
        set_toggles({"stage1_fir": False, "stage5_ocular": False})
        cfg = get_toggles()
        assert cfg.stage1_fir is False
        assert cfg.stage5_ocular is False

    def test_set_toggles_returns_new_config(self):
        result = set_toggles({"stage2_bad_channels": False})
        assert isinstance(result, FilterToggleConfig)
        assert result.stage2_bad_channels is False

    def test_stage6_cardiac_settable_via_set_toggles(self):
        set_toggles({"stage6_cardiac": False})
        # stage6_cardiac is internal — not in to_dict(), but attribute exists
        assert get_toggles().stage6_cardiac is False


class TestGetTogglesCopy:
    def test_get_toggles_returns_copy(self):
        c1 = get_toggles()
        c1.stage1_fir = False
        # Mutating the copy should not affect the singleton
        assert get_toggles().stage1_fir is True


class TestFilterToggleConfigDataclass:
    def test_construction_with_kwargs(self):
        cfg = FilterToggleConfig(stage1_fir=False, stage4_asr=False)
        assert cfg.stage1_fir is False
        assert cfg.stage4_asr is False
        assert cfg.stage2_bad_channels is True  # default

    def test_to_dict_excludes_internal(self):
        cfg = FilterToggleConfig()
        d = cfg.to_dict()
        assert "stage6_cardiac" not in d
