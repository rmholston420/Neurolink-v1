"""Unit tests for dsp/fnirs.py."""

from __future__ import annotations

import numpy as np

import neurolink.dsp.fnirs as fnirs


def _raw(
    n_channels: int = 4,
    n_samples: int = 256,
    seed: int = 42,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return synthetic raw fNIRS data (channels x samples)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, n_samples)).astype(dtype)


# ---------------------------------------------------------------------------
# Module reset between tests
# ---------------------------------------------------------------------------


def setup_function():
    fnirs.reset()


# ---------------------------------------------------------------------------
# apply() -- preprocessing pipeline
# ---------------------------------------------------------------------------


class TestApply:
    def setup_method(self):
        fnirs.reset()

    def test_returns_ndarray(self):
        result = fnirs.apply(_raw())
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self):
        raw = _raw(n_channels=4, n_samples=256)
        result = fnirs.apply(raw)
        assert result.shape == raw.shape

    def test_none_input_returns_none(self):
        assert fnirs.apply(None) is None

    def test_output_is_float32(self):
        result = fnirs.apply(_raw())
        assert result.dtype == np.float32

    def test_output_values_are_finite(self):
        result = fnirs.apply(_raw(n_samples=512))
        assert np.all(np.isfinite(result))

    def test_disabled_returns_raw(self):
        fnirs.set_config(enable=False)
        raw = _raw()
        result = fnirs.apply(raw)
        assert result is raw
        fnirs.set_config(enable=True)

    def test_1d_input_returns_unchanged(self):
        """1D arrays are not valid fNIRS frames -- returned as-is."""
        arr = np.ones(64, dtype=np.float32)
        result = fnirs.apply(arr)
        assert result is arr

    def test_empty_channel_dim_returns_unchanged(self):
        """Zero-channel array should be returned unchanged."""
        arr = np.zeros((0, 64), dtype=np.float32)
        result = fnirs.apply(arr)
        assert result is arr


# ---------------------------------------------------------------------------
# Beer-Lambert conversion
# ---------------------------------------------------------------------------


class TestDecode:
    def setup_method(self):
        fnirs.reset()

    def test_returns_tuple_of_two_arrays(self):
        raw = _raw(n_channels=4, n_samples=512)
        result = fnirs.decode(raw)
        assert isinstance(result, tuple)
        hbo, _hbr = result  # RUF059: prefix unused hbr with _
        assert isinstance(hbo, np.ndarray)

    def test_hbo_hbr_shape(self):
        """n_pairs = n_channels // 2."""
        raw = _raw(n_channels=4, n_samples=256)
        hbo, hbr = fnirs.decode(raw)
        assert hbo.shape == (2, 256)
        assert hbr.shape == (2, 256)

    def test_none_input_returns_none(self):
        assert fnirs.decode(None) is None

    def test_negative_raw_handled(self):
        raw = -np.abs(_raw(n_samples=256))
        result = fnirs.decode(raw)
        assert isinstance(result, tuple)

    def test_too_few_channels_returns_empty_arrays(self):
        """Fewer than min_channels (default 2) returns empty (0, N) arrays."""
        raw = _raw(n_channels=1, n_samples=64)
        result = fnirs.decode(raw)
        assert isinstance(result, tuple)
        hbo, _hbr = result  # RUF059: prefix unused hbr with _
        assert hbo.shape[0] == 0


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_get_config_returns_fnirs_config(self):
        cfg = fnirs.get_config()
        assert hasattr(cfg, "enable")
        assert hasattr(cfg, "baseline_alpha")
        assert hasattr(cfg, "spike_threshold")

    def test_set_config_updates_field(self):
        fnirs.set_config(baseline_alpha=0.05)
        assert fnirs.get_config().baseline_alpha == 0.05
        fnirs.set_config(baseline_alpha=0.01)  # restore

    def test_set_config_ignores_unknown_keys(self):
        fnirs.set_config(nonexistent_key=99)  # must not raise


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_baseline(self):
        fnirs.apply(_raw())  # warm up baseline
        fnirs.reset()
        # After reset, first apply should behave as fresh init
        result = fnirs.apply(_raw())
        assert result is not None
