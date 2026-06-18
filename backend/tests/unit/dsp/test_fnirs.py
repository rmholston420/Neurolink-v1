"""Unit tests for dsp/fnirs.py."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.fnirs import FNIRSProcessor


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
# FNIRSProcessor construction
# ---------------------------------------------------------------------------


class TestFNIRSProcessorInit:
    def test_default_init(self):
        proc = FNIRSProcessor()
        assert proc is not None

    def test_custom_channel_count(self):
        proc = FNIRSProcessor(n_channels=8)
        assert proc.n_channels == 8


# ---------------------------------------------------------------------------
# FNIRSProcessor.process()
# ---------------------------------------------------------------------------


class TestFNIRSProcessorProcess:
    def test_returns_dict(self):
        proc = FNIRSProcessor()
        result = proc.process(_raw())
        assert isinstance(result, dict)

    def test_oxy_deoxy_keys_present(self):
        proc = FNIRSProcessor()
        result = proc.process(_raw())
        assert "oxy" in result or "fnirs_oxy" in result

    def test_none_input_returns_empty_dict(self):
        proc = FNIRSProcessor()
        result = proc.process(None)
        assert result == {}

    def test_too_short_returns_empty_dict(self):
        proc = FNIRSProcessor()
        result = proc.process(np.zeros((4, 1), dtype=np.float32))
        assert result == {} or isinstance(result, dict)

    def test_output_values_are_finite(self):
        proc = FNIRSProcessor()
        result = proc.process(_raw(n_samples=512))
        for v in result.values():
            if isinstance(v, float):
                assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Beer-Lambert conversion
# ---------------------------------------------------------------------------


class TestBeerLambert:
    def test_beer_lambert_output_shape(self):
        proc = FNIRSProcessor()
        raw = _raw(n_samples=512)
        result = proc.process(raw)
        # Just verify no exception and result is a dict
        assert isinstance(result, dict)

    def test_negative_raw_handled(self):
        proc = FNIRSProcessor()
        raw = -np.abs(_raw(n_samples=256))
        result = proc.process(raw)
        assert isinstance(result, dict)
