"""Unit tests for dsp/filter_toggles.py — including stage6_cardiac."""

from __future__ import annotations

import threading

import pytest

from neurolink.dsp.filter_toggles import FilterToggleConfig, get_toggles, set_toggles


# ---------------------------------------------------------------------------
# Helpers: always reset to defaults before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_toggles():
    """Ensure the module singleton is fully enabled before and after each test."""
    set_toggles(
        {
            "stage1_fir": True,
            "stage2_bad_channels": True,
            "stage3_artifact_gate": True,
            "stage3b_artifact_detector": True,
            "stage4_asr": True,
            "stage4b_baseline": True,
            "stage5_ocular": True,
            "stage6_cardiac": True,
            "imu_gate": True,
        }
    )
    yield
    set_toggles(
        {
            "stage1_fir": True,
            "stage2_bad_channels": True,
            "stage3_artifact_gate": True,
            "stage3b_artifact_detector": True,
            "stage4_asr": True,
            "stage4b_baseline": True,
            "stage5_ocular": True,
            "stage6_cardiac": True,
            "imu_gate": True,
        }
    )


# ---------------------------------------------------------------------------
# FilterToggleConfig defaults
# ---------------------------------------------------------------------------

class TestFilterToggleConfigDefaults:
    def test_all_stages_enabled_by_default(self):
        cfg = FilterToggleConfig()
        assert cfg.stage1_fir is True
        assert cfg.stage2_bad_channels is True
        assert cfg.stage3_artifact_gate is True
        assert cfg.stage3b_artifact_detector is True
        assert cfg.stage4_asr is True
        assert cfg.stage4b_baseline is True
        assert cfg.stage5_ocular is True
        assert cfg.stage6_cardiac is True
        assert cfg.imu_gate is True

    def test_stage6_cardiac_field_present(self):
        """stage6_cardiac must exist and default True — the new Stage 6 wire."""
        cfg = FilterToggleConfig()
        assert hasattr(cfg, "stage6_cardiac")
        assert cfg.stage6_cardiac is True

    def test_to_dict_includes_all_stages(self):
        d = FilterToggleConfig().to_dict()
        expected_keys = {
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
        assert expected_keys <= d.keys()

    def test_to_dict_values_are_bool(self):
        d = FilterToggleConfig().to_dict()
        for k, v in d.items():
            assert isinstance(v, bool), f"{k} should be bool, got {type(v)}"


# ---------------------------------------------------------------------------
# get_toggles returns copy, not reference
# ---------------------------------------------------------------------------

class TestGetToggles:
    def test_returns_filtertoggleconfig(self):
        t = get_toggles()
        assert isinstance(t, FilterToggleConfig)

    def test_returns_copy_not_singleton(self):
        t1 = get_toggles()
        t1.stage6_cardiac = False
        t2 = get_toggles()
        assert t2.stage6_cardiac is True  # singleton unchanged

    def test_all_true_initially(self):
        t = get_toggles()
        assert t.stage6_cardiac is True
        assert t.stage5_ocular is True


# ---------------------------------------------------------------------------
# set_toggles — partial update semantics
# ---------------------------------------------------------------------------

class TestSetToggles:
    def test_set_stage6_cardiac_false(self):
        result = set_toggles({"stage6_cardiac": False})
        assert result.stage6_cardiac is False
        assert get_toggles().stage6_cardiac is False

    def test_re_enable_stage6_cardiac(self):
        set_toggles({"stage6_cardiac": False})
        set_toggles({"stage6_cardiac": True})
        assert get_toggles().stage6_cardiac is True

    def test_partial_update_preserves_other_stages(self):
        set_toggles({"stage6_cardiac": False})
        t = get_toggles()
        assert t.stage1_fir is True
        assert t.stage5_ocular is True
        assert t.stage6_cardiac is False

    def test_disable_multiple_stages_at_once(self):
        result = set_toggles({"stage5_ocular": False, "stage6_cardiac": False})
        assert result.stage5_ocular is False
        assert result.stage6_cardiac is False
        assert result.stage4_asr is True

    def test_unknown_keys_silently_ignored(self):
        result = set_toggles({"stage99_imaginary": False})
        # All existing stages should still be enabled
        assert result.stage6_cardiac is True

    def test_non_bool_value_silently_ignored(self):
        """set_toggles only accepts booleans; non-bool values are dropped."""
        set_toggles({"stage6_cardiac": "false"})  # type: ignore[dict-item]
        assert get_toggles().stage6_cardiac is True  # unchanged

    def test_empty_dict_is_no_op(self):
        result = set_toggles({})
        assert result.stage6_cardiac is True

    def test_set_returns_filtertoggleconfig(self):
        result = set_toggles({"stage6_cardiac": False})
        assert isinstance(result, FilterToggleConfig)


# ---------------------------------------------------------------------------
# Stage 6 cardiac — pipeline wiring contract
# ---------------------------------------------------------------------------

class TestStage6CardiacWiring:
    """Verify the toggle can gate the cardiac corrector consistently."""

    def test_stage6_in_to_dict_round_trips(self):
        cfg = FilterToggleConfig(stage6_cardiac=False)
        d = cfg.to_dict()
        assert d["stage6_cardiac"] is False
        restored = FilterToggleConfig(**d)
        assert restored.stage6_cardiac is False

    def test_disable_stage6_via_set_and_read_back(self):
        set_toggles({"stage6_cardiac": False})
        t = get_toggles()
        assert t.stage6_cardiac is False

    def test_stage6_isolated_from_stage5(self):
        """Disabling stage6 must not affect stage5 and vice-versa."""
        set_toggles({"stage6_cardiac": False})
        assert get_toggles().stage5_ocular is True
        set_toggles({"stage5_ocular": False})
        assert get_toggles().stage6_cardiac is False  # still off
        assert get_toggles().stage5_ocular is False


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestToggleThreadSafety:
    def test_concurrent_reads_no_exception(self):
        errors: list[Exception] = []

        def _reader():
            try:
                for _ in range(50):
                    get_toggles()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_reads_and_writes_no_exception(self):
        errors: list[Exception] = []

        def _writer():
            try:
                for i in range(20):
                    set_toggles({"stage6_cardiac": i % 2 == 0})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _reader():
            try:
                for _ in range(50):
                    t = get_toggles()
                    assert isinstance(t.stage6_cardiac, bool)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=_writer),
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
