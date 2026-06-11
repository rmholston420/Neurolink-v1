"""Unit tests for hardware/muse_athena/fnirs.py FNIRSDecoder.

The .coverage report shows 12 covered lines but the empty-raw guard branch
and the edge case of an odd-length sample are missing.
"""
from __future__ import annotations

import pytest

from neurolink.hardware.muse_athena.fnirs import FNIRSDecoder


@pytest.fixture
def decoder():
    return FNIRSDecoder()


# ---------------------------------------------------------------------------
# Empty raw → returns zero dict (the uncovered guard branch)
# ---------------------------------------------------------------------------

def test_decode_empty_raw_returns_zeros(decoder):
    result = decoder.decode([])
    assert result == {"fnirs_oxy": 0.0, "fnirs_deoxy": 0.0}


# ---------------------------------------------------------------------------
# Normal even-length sample
# ---------------------------------------------------------------------------

def test_decode_even_sample(decoder):
    # [oxy0, deoxy0, oxy1, deoxy1]
    result = decoder.decode([1.0, 0.5, 3.0, 1.5])
    assert result["fnirs_oxy"] == pytest.approx(2.0)   # (1.0+3.0)/2
    assert result["fnirs_deoxy"] == pytest.approx(1.0)  # (0.5+1.5)/2


def test_decode_single_pair(decoder):
    result = decoder.decode([2.0, 0.8])
    assert result["fnirs_oxy"] == pytest.approx(2.0)
    assert result["fnirs_deoxy"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Odd-length sample → deoxy list is shorter (edge case)
# ---------------------------------------------------------------------------

def test_decode_odd_length_sample(decoder):
    """Odd-length raw: oxy gets one more element than deoxy.
    raw = [oxy0, deoxy0, oxy1]  -> oxy=[raw[0], raw[2]], deoxy=[raw[1]]
    """
    result = decoder.decode([1.0, 0.5, 3.0])
    assert result["fnirs_oxy"] == pytest.approx(2.0)   # (1.0+3.0)/2
    assert result["fnirs_deoxy"] == pytest.approx(0.5)  # 0.5/1


# ---------------------------------------------------------------------------
# Single element (only oxy, no deoxy) → fnirs_deoxy=0.0 (empty deoxy list)
# ---------------------------------------------------------------------------

def test_decode_single_element_no_deoxy(decoder):
    """raw=[x] -> oxy=[x], deoxy=[] -> fnirs_deoxy uses 'if deoxy else 0.0' branch."""
    result = decoder.decode([5.0])
    assert result["fnirs_oxy"] == pytest.approx(5.0)
    assert result["fnirs_deoxy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# All zeros
# ---------------------------------------------------------------------------

def test_decode_all_zeros(decoder):
    result = decoder.decode([0.0, 0.0, 0.0, 0.0])
    assert result["fnirs_oxy"] == pytest.approx(0.0)
    assert result["fnirs_deoxy"] == pytest.approx(0.0)
