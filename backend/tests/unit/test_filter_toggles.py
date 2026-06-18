"""Unit tests for dsp.filter_toggles.FilterToggles."""
from __future__ import annotations

import pytest

from neurolink.dsp.filter_toggles import FilterToggles


@pytest.fixture()
def toggles() -> FilterToggles:
    return FilterToggles()


class TestDefaults:
    def test_instantiation(self):
        ft = FilterToggles()
        assert ft is not None

    def test_all_filters_enabled_by_default(self, toggles):
        """All standard filters should be on at startup."""
        for name in toggles.filter_names:
            assert toggles.is_enabled(name), f"{name} should be enabled by default"


class TestEnable:
    def test_disable_then_enable(self, toggles):
        name = toggles.filter_names[0]
        toggles.disable(name)
        assert not toggles.is_enabled(name)
        toggles.enable(name)
        assert toggles.is_enabled(name)

    def test_enable_unknown_filter_raises(self, toggles):
        with pytest.raises((KeyError, ValueError)):
            toggles.enable("nonexistent_filter")


class TestDisable:
    def test_disable_reduces_active_count(self, toggles):
        initial = sum(1 for n in toggles.filter_names if toggles.is_enabled(n))
        name = toggles.filter_names[0]
        toggles.disable(name)
        after = sum(1 for n in toggles.filter_names if toggles.is_enabled(n))
        assert after == initial - 1

    def test_double_disable_idempotent(self, toggles):
        name = toggles.filter_names[0]
        toggles.disable(name)
        toggles.disable(name)  # second call must not raise
        assert not toggles.is_enabled(name)


class TestReset:
    def test_reset_restores_all_defaults(self, toggles):
        for name in toggles.filter_names:
            toggles.disable(name)
        toggles.reset()
        for name in toggles.filter_names:
            assert toggles.is_enabled(name)


class TestToggle:
    def test_toggle_flips_state(self, toggles):
        name = toggles.filter_names[0]
        initial = toggles.is_enabled(name)
        toggles.toggle(name)
        assert toggles.is_enabled(name) != initial
        toggles.toggle(name)
        assert toggles.is_enabled(name) == initial


class TestAsDict:
    def test_as_dict_returns_all_filters(self, toggles):
        d = toggles.as_dict()
        assert isinstance(d, dict)
        for name in toggles.filter_names:
            assert name in d

    def test_as_dict_values_are_bool(self, toggles):
        d = toggles.as_dict()
        for v in d.values():
            assert isinstance(v, bool)
