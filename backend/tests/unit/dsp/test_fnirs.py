"""Unit tests for hardware/muse_athena/fnirs.py."""
from __future__ import annotations

import pytest

from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder


def test_fnirs_decoder_averages_oxy_channels():
    """Even indices (0, 2, ...) are oxy channels."""
    decoder = FNIRSDecoder()
    # [oxy0=10, deoxy0=20, oxy1=30, deoxy1=40] -> oxy avg = 20.0
    result = decoder.decode([10.0, 20.0, 30.0, 40.0])
    assert result["fnirs_oxy"] == 20.0


def test_fnirs_decoder_averages_deoxy_channels():
    """Odd indices (1, 3, ...) are deoxy channels."""
    decoder = FNIRSDecoder()
    result = decoder.decode([10.0, 20.0, 30.0, 40.0])
    assert result["fnirs_deoxy"] == 30.0


def test_fnirs_decoder_single_pair():
    decoder = FNIRSDecoder()
    result = decoder.decode([5.0, 15.0])
    assert result["fnirs_oxy"] == 5.0
    assert result["fnirs_deoxy"] == 15.0


def test_fnirs_decoder_empty_returns_zeros():
    decoder = FNIRSDecoder()
    result = decoder.decode([])
    assert result["fnirs_oxy"] == 0.0
    assert result["fnirs_deoxy"] == 0.0


def test_fnirs_decoder_six_channels():
    decoder = FNIRSDecoder()
    # [o0, d0, o1, d1, o2, d2] = [10, 20, 12, 22, 14, 24]
    result = decoder.decode([10.0, 20.0, 12.0, 22.0, 14.0, 24.0])
    assert abs(result["fnirs_oxy"] - 12.0) < 0.01
    assert abs(result["fnirs_deoxy"] - 22.0) < 0.01
