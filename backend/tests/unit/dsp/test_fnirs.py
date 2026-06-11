"""Unit tests for hardware/muse_athena/fnirs.py."""

from __future__ import annotations

from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder


def test_fnirs_decoder_averages_oxy_channels():
    """Even-indexed values should be averaged as oxy."""
    decoder = FNIRSDecoder()
    # [oxy0, deoxy0, oxy1, deoxy1, oxy2, deoxy2]
    raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = decoder.decode(raw)
    expected_oxy = (1.0 + 3.0 + 5.0) / 3
    assert abs(result["fnirs_oxy"] - expected_oxy) < 1e-9


def test_fnirs_decoder_averages_deoxy_channels():
    """Odd-indexed values should be averaged as deoxy."""
    decoder = FNIRSDecoder()
    raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = decoder.decode(raw)
    expected_deoxy = (2.0 + 4.0 + 6.0) / 3
    assert abs(result["fnirs_deoxy"] - expected_deoxy) < 1e-9


def test_fnirs_decoder_empty_returns_zeros():
    decoder = FNIRSDecoder()
    result = decoder.decode([])
    assert result["fnirs_oxy"] == 0.0
    assert result["fnirs_deoxy"] == 0.0


def test_fnirs_decoder_single_channel():
    decoder = FNIRSDecoder()
    result = decoder.decode([10.0])
    assert result["fnirs_oxy"] == 10.0
    assert result["fnirs_deoxy"] == 0.0  # no odd-indexed channels
