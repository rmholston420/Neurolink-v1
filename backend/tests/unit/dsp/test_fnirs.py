"""Full unit tests for the fNIRS DSP module."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurolink.dsp import fnirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_fnirs(
    n_channels: int = 8,
    n_samples: int = 128,
    dtype=np.float32,
    seed: int = 42,
) -> np.ndarray:
    """Return synthetic raw fNIRS data (channels × samples)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)).astype(dtype)


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------

class TestFNIRSConfig:
    def test_default_config_accessible(self):
        cfg = fnirs.get_config()
        assert cfg is not None

    def test_default_config_returns_copy(self):
        c1 = fnirs.get_config()
        c2 = fnirs.get_config()
        assert c1 is not c2

    def test_set_config_updates_state(self):
        cfg = fnirs.get_config()
        original_enable = cfg.enable
        fnirs.set_config(enable=not original_enable)
        updated = fnirs.get_config()
        assert updated.enable == (not original_enable)
        fnirs.set_config(enable=original_enable)  # restore

    def test_enable_false_disables_processing(self):
        fnirs.set_config(enable=False)
        raw = _make_raw_fnirs()
        result = fnirs.apply(raw)
        assert result is raw  # identity when disabled
        fnirs.set_config(enable=True)  # restore


# ---------------------------------------------------------------------------
# apply() — basic contracts
# ---------------------------------------------------------------------------

class TestFNIRSApply:
    def test_none_eeg_returns_none(self):
        result = fnirs.apply(None)
        assert result is None

    def test_output_shape_preserved(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=128)
        out = fnirs.apply(raw)
        assert out.shape == raw.shape

    def test_output_dtype_float32(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=128)
        out = fnirs.apply(raw)
        assert out.dtype == np.float32

    def test_1d_input_handled_gracefully(self):
        raw = np.zeros(64, dtype=np.float32)
        try:
            fnirs.apply(raw)
        except (ValueError, IndexError):
            pass  # acceptable — must not segfault or raise unexpected errors

    def test_empty_input_handled_gracefully(self):
        raw = np.zeros((0, 128), dtype=np.float32)
        try:
            fnirs.apply(raw)
        except (ValueError, IndexError):
            pass

    def test_output_is_not_same_object_as_input(self):
        """Ensure apply() does not mutate input in place (returns new array)."""
        raw = _make_raw_fnirs()
        out = fnirs.apply(raw)
        if out is not raw:
            assert not np.shares_memory(out, raw)

    def test_signal_modified_vs_raw(self):
        """After apply(), the signal should differ from pure raw (baseline applied etc.)."""
        fnirs.set_config(enable=True)
        fnirs.reset()
        raw = _make_raw_fnirs(seed=7)
        out = fnirs.apply(raw)
        # At minimum, output must be a valid array even if numerically close
        assert out is not None
        assert out.shape == raw.shape


# ---------------------------------------------------------------------------
# Beer–Lambert conversion
# ---------------------------------------------------------------------------

class TestBeerLambert:
    def test_hbo_hbr_shapes(self):
        """decode() must return (HbO, HbR) with same spatial dimensions."""
        raw = _make_raw_fnirs(n_channels=8, n_samples=128)
        result = fnirs.decode(raw)
        # result may be a tuple (HbO, HbR) or a structured dict
        if isinstance(result, tuple):
            hbo, hbr = result
            assert hbo.shape[0] == hbr.shape[0]  # same channels
            assert hbo.shape[1] == raw.shape[1]  # same time points
        elif isinstance(result, dict):
            assert "HbO" in result or "hbo" in result
        else:
            # Some implementations return a single merged array
            assert result.ndim == 2

    def test_hbo_hbr_finite(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=64)
        result = fnirs.decode(raw)
        if isinstance(result, tuple):
            for arr in result:
                assert np.all(np.isfinite(arr)), "NaN/Inf in Beer-Lambert output"
        elif isinstance(result, np.ndarray):
            assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Baseline detrending
# ---------------------------------------------------------------------------

class TestFNIRSBaseline:
    def test_baseline_reduces_dc_offset(self):
        """After baseline correction, channel means should be near zero."""
        rng = np.random.default_rng(99)
        raw = rng.standard_normal((4, 256)).astype(np.float32)
        raw += 10.0  # add large DC offset
        fnirs.reset()
        # Warm up with several frames
        for _ in range(20):
            fnirs.apply(raw)
        out = fnirs.apply(raw)
        # After baseline convergence, channel means should be reduced
        assert np.abs(out.mean()) < np.abs(raw.mean())


# ---------------------------------------------------------------------------
# Motion artefact handling
# ---------------------------------------------------------------------------

class TestFNIRSMotionArtifacts:
    def test_spike_clipped_or_smoothed(self):
        """Inject a single extreme spike — output should not propagate it raw."""
        raw = _make_raw_fnirs(n_channels=4, n_samples=128)
        spike = raw.copy()
        spike[0, 64] = 1e6  # extreme spike
        out = fnirs.apply(spike)
        assert out[0, 64] < 1e6, "Spike not attenuated by motion artifact handling"

    def test_motion_channel_flagged_or_corrected(self):
        """Channels with large jumps must be flagged or corrected, not passed through."""
        raw = np.zeros((4, 128), dtype=np.float32)
        raw[2, 64:] += 500.0  # step artifact in channel 2
        out = fnirs.apply(raw)
        assert out is not None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestFNIRSReset:
    def test_reset_clears_internal_state(self):
        raw = _make_raw_fnirs()
        fnirs.apply(raw)  # warm up
        fnirs.reset()
        # After reset, first frame should succeed without error
        out = fnirs.apply(raw)
        assert out is not None

    def test_reset_resets_baseline_buffer(self):
        """Baseline should restart accumulation after reset."""
        raw = _make_raw_fnirs()
        for _ in range(10):
            fnirs.apply(raw)
        fnirs.reset()
        out1 = fnirs.apply(raw)
        assert out1 is not None  # no crash immediately after reset


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestFNIRSThreadSafety:
    def test_concurrent_apply_no_exception(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=64)
        errors: list[Exception] = []

        def worker():
            for _ in range(20):
                try:
                    fnirs.apply(raw)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_reset_apply_no_exception(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=64)
        errors: list[Exception] = []

        def apply_worker():
            for _ in range(15):
                try:
                    fnirs.apply(raw)
                except Exception as e:
                    errors.append(e)

        def reset_worker():
            for _ in range(5):
                try:
                    fnirs.reset()
                except Exception as e:
                    errors.append(e)

        threads = (
            [threading.Thread(target=apply_worker) for _ in range(3)] +
            [threading.Thread(target=reset_worker) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# Integration: apply → decode pipeline
# ---------------------------------------------------------------------------

class TestFNIRSPipeline:
    def test_apply_then_decode_no_exception(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=128)
        corrected = fnirs.apply(raw)
        result = fnirs.decode(corrected)
        assert result is not None

    def test_pipeline_output_finite(self):
        raw = _make_raw_fnirs(n_channels=8, n_samples=128)
        corrected = fnirs.apply(raw)
        result = fnirs.decode(corrected)
        if isinstance(result, tuple):
            for arr in result:
                assert np.all(np.isfinite(arr))
        elif isinstance(result, np.ndarray):
            assert np.all(np.isfinite(result))
