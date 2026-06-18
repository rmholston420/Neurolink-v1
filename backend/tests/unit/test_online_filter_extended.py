"""Targeted tests for dsp/online_filter.py missing lines.

Missing lines (coverage report):
  64-74   get_default_line_freq: unusual env-var value -> log.warning
          invalid (non-numeric) env-var value -> log.warning + fallback 60 Hz
  83-84   ImportError fallback: artifact_config unavailable -> return 60.0
  121     OnlineFilterChain.apply(): 1-D input reshaped to (1, n_samples)
  253-256 FilterChainRegistry.set_config(): replaces config and rebuilds chain
  260-261 FilterChainRegistry.get_config(): returns current config
  289     get_registry(): module-level singleton initialised on first call

Line 210 (filtfilt exception handler) carries `# pragma: no cover` in source
and is intentionally excluded.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest

from neurolink.dsp.online_filter import (
    FilterChainRegistry,
    FilterConfig,
    OnlineFilterChain,
    apply_online_filters,
    get_default_line_freq,
    get_registry,
)


# ─────────────────────────────────────────────────────────────────────────────
class TestGetDefaultLineFreq:
    """Lines 56-84: full coverage of get_default_line_freq()."""

    def test_no_env_var_returns_artifact_config_value(self):
        """No env var -> fall through to artifact_config.ARTIFACT_LINE_FREQ_HZ."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEUROLINK_LINE_FREQ_HZ", None)
            freq = get_default_line_freq()
        assert freq in (50.0, 60.0)  # whatever artifact_config says

    def test_env_var_50_returns_50(self):
        with patch.dict(os.environ, {"NEUROLINK_LINE_FREQ_HZ": "50"}):
            assert get_default_line_freq() == 50.0

    def test_env_var_60_returns_60(self):
        with patch.dict(os.environ, {"NEUROLINK_LINE_FREQ_HZ": "60"}):
            assert get_default_line_freq() == 60.0

    def test_unusual_numeric_env_var_logs_warning_and_returns_value(self):
        """Lines 64-71: unusual value (not 50 or 60) -> log.warning + value returned."""
        with patch("neurolink.dsp.online_filter.log") as mock_log, \
             patch.dict(os.environ, {"NEUROLINK_LINE_FREQ_HZ": "45"}):
            freq = get_default_line_freq()
        assert freq == 45.0
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0][0]
        assert "unusual" in call_args

    def test_invalid_non_numeric_env_var_logs_warning_and_falls_back(self):
        """Lines 72-74: non-numeric value -> log.warning + fallback via artifact_config."""
        with patch("neurolink.dsp.online_filter.log") as mock_log, \
             patch.dict(os.environ, {"NEUROLINK_LINE_FREQ_HZ": "not_a_number"}):
            freq = get_default_line_freq()
        # Falls through to artifact_config or hard-coded 60.0
        assert freq in (50.0, 60.0)
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args[0][0]
        assert "invalid" in call_args

    def test_import_error_fallback_returns_60(self):
        """Lines 83-84: ImportError from artifact_config -> hard fallback 60.0."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEUROLINK_LINE_FREQ_HZ", None)
            # Temporarily shadow the artifact_config import inside the function
            with patch.dict(sys.modules, {"neurolink.dsp.artifact_config": None}):
                freq = get_default_line_freq()
        assert freq == 60.0


# ─────────────────────────────────────────────────────────────────────────────
class TestOnlineFilterChainApply:
    """Line 121: 1-D input is reshaped to (1, n_samples) before filtering."""

    def _make_chain(self) -> OnlineFilterChain:
        """Build a chain with a short filter order so min_samples is small."""
        cfg = FilterConfig(
            hz_highpass=0.5,
            hz_notch_freqs=[60.0],
            hz_lowpass=55.0,
            fs=256.0,
            filter_order=10,  # min_samples = 3*11+1 = 34
        )
        return OnlineFilterChain(cfg)

    def test_1d_input_is_accepted_and_returns_2d(self):
        """Line 121: eeg.ndim == 1 -> reshaped to (1, n) before filtering."""
        chain = self._make_chain()
        n = chain._min_samples + 10  # ensure buffer is long enough
        eeg_1d = np.random.randn(n).astype(np.float32)
        out = chain.apply(eeg_1d)
        # After reshape the chain processes it as (1, n); result is (1, n) float32
        assert out.ndim == 2
        assert out.shape[0] == 1
        assert out.dtype == np.float32

    def test_2d_input_short_buffer_returned_unchanged(self):
        """Buffer shorter than min_samples -> raw array returned (log.debug path)."""
        chain = self._make_chain()
        short = np.random.randn(4, 5).astype(np.float32)
        out = chain.apply(short)
        # Must be the same object (no copy)
        assert out is short

    def test_2d_input_sufficient_buffer_filtered(self):
        """Normal (channels, samples) path produces float32 output."""
        chain = self._make_chain()
        n = chain._min_samples + 50
        eeg = np.random.randn(4, n).astype(np.float32)
        out = chain.apply(eeg)
        assert out.shape == eeg.shape
        assert out.dtype == np.float32

    def test_empty_kernel_list_returns_input(self):
        """Chain with no kernels (all disabled) returns input as float32."""
        cfg = FilterConfig(
            hz_highpass=None,
            hz_notch_freqs=[],
            hz_lowpass=None,
            fs=256.0,
            filter_order=10,
        )
        chain = OnlineFilterChain(cfg)
        eeg = np.random.randn(4, 100).astype(np.float32)
        out = chain.apply(eeg)
        assert out.dtype == np.float32
        assert out.shape == eeg.shape


# ─────────────────────────────────────────────────────────────────────────────
class TestFilterChainRegistry:
    """Lines 253-261: set_config() and get_config() on FilterChainRegistry."""

    def test_set_config_replaces_chain(self):
        """Lines 253-256: set_config() rebuilds the internal chain."""
        registry = FilterChainRegistry()
        original_chain = registry._chain

        new_cfg = FilterConfig(hz_lowpass=45.0)
        registry.set_config(new_cfg)

        assert registry._chain is not original_chain
        assert registry._config.hz_lowpass == 45.0

    def test_get_config_returns_active_config(self):
        """Lines 260-261: get_config() returns the config set via set_config()."""
        registry = FilterChainRegistry()
        new_cfg = FilterConfig(hz_highpass=1.0, hz_notch_freqs=[50.0, 100.0])
        registry.set_config(new_cfg)
        got = registry.get_config()
        assert got.hz_highpass == 1.0
        assert got.hz_notch_freqs == [50.0, 100.0]

    def test_apply_uses_active_chain(self):
        """registry.apply() routes through the currently active chain."""
        registry = FilterChainRegistry()
        cfg = FilterConfig(filter_order=10)
        registry.set_config(cfg)
        n = registry._chain._min_samples + 20
        eeg = np.random.randn(4, n).astype(np.float32)
        out = registry.apply(eeg)
        assert out.shape == eeg.shape
        assert out.dtype == np.float32

    def test_thread_safety_concurrent_set_config(self):
        """Concurrent set_config() calls must not corrupt internal state."""
        registry = FilterChainRegistry()
        errors: list[Exception] = []

        def _worker(lp: float) -> None:
            try:
                registry.set_config(FilterConfig(hz_lowpass=lp, filter_order=10))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(45.0 + i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Registry should still be usable
        assert registry._chain is not None


# ─────────────────────────────────────────────────────────────────────────────
class TestGetRegistry:
    """Line 289: get_registry() singleton initialisation."""

    def test_get_registry_returns_filter_chain_registry(self):
        """Line 289: module-level singleton is a FilterChainRegistry."""
        reg = get_registry()
        assert isinstance(reg, FilterChainRegistry)

    def test_get_registry_returns_same_instance(self):
        """Subsequent calls return the identical singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_apply_online_filters_convenience_wrapper(self):
        """apply_online_filters() delegates to the singleton registry."""
        n = get_registry()._chain._min_samples + 20
        eeg = np.random.randn(4, n).astype(np.float32)
        out = apply_online_filters(eeg)
        assert out.shape == eeg.shape
        assert out.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
class TestFilterConfigHelpers:
    """FilterConfig helpers not yet covered."""

    def test_with_line_freq_50_sets_notch_freqs(self):
        cfg = FilterConfig()
        new = cfg.with_line_freq(50.0)
        assert new.hz_notch_freqs == [50.0, 100.0]
        assert new.hz_highpass == cfg.hz_highpass
        assert new.hz_lowpass == cfg.hz_lowpass

    def test_with_line_freq_60_sets_notch_freqs(self):
        cfg = FilterConfig()
        new = cfg.with_line_freq(60.0)
        assert new.hz_notch_freqs == [60.0, 120.0]

    def test_ensure_even_order_pads_odd(self):
        cfg = FilterConfig(filter_order=127)
        assert cfg._ensure_even_order() == 128

    def test_ensure_even_order_leaves_even(self):
        cfg = FilterConfig(filter_order=128)
        assert cfg._ensure_even_order() == 128

    def test_out_of_range_highpass_produces_no_hp_kernel(self):
        """hp_norm >= 1.0 (e.g. hz_highpass=200 at fs=256) -> kernel skipped."""
        cfg = FilterConfig(
            hz_highpass=200.0,  # 200/128 = 1.5625 > 1 -> skipped
            hz_notch_freqs=[],
            hz_lowpass=None,
            fs=256.0,
            filter_order=10,
        )
        chain = OnlineFilterChain(cfg)
        assert not any("HP" in lbl for lbl in chain._labels)

    def test_out_of_range_lowpass_produces_no_lp_kernel(self):
        """lp_norm >= 1.0 -> kernel skipped."""
        cfg = FilterConfig(
            hz_highpass=None,
            hz_notch_freqs=[],
            hz_lowpass=200.0,  # > nyq -> skipped
            fs=256.0,
            filter_order=10,
        )
        chain = OnlineFilterChain(cfg)
        assert not any("LP" in lbl for lbl in chain._labels)
