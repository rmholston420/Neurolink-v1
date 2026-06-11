"""Unit tests for utils/timing.py.

utils/timing.py has 0 covered lines in the current coverage report.
All four functions are pure / deterministic wrappers around time.monotonic*
and are straightforward to cover exhaustively.
"""
from __future__ import annotations

import time

import pytest

from neurolink.utils.timing import elapsed_ms, mono_ns, mono_sec, rate_limiter


# ---------------------------------------------------------------------------
# mono_ns
# ---------------------------------------------------------------------------

def test_mono_ns_returns_int():
    result = mono_ns()
    assert isinstance(result, int)


def test_mono_ns_is_monotonic():
    t1 = mono_ns()
    t2 = mono_ns()
    assert t2 >= t1


def test_mono_ns_is_positive():
    assert mono_ns() > 0


# ---------------------------------------------------------------------------
# mono_sec
# ---------------------------------------------------------------------------

def test_mono_sec_returns_float():
    result = mono_sec()
    assert isinstance(result, float)


def test_mono_sec_is_monotonic():
    t1 = mono_sec()
    t2 = mono_sec()
    assert t2 >= t1


def test_mono_sec_is_positive():
    assert mono_sec() > 0.0


# ---------------------------------------------------------------------------
# elapsed_ms
# ---------------------------------------------------------------------------

def test_elapsed_ms_returns_float():
    start = mono_ns()
    result = elapsed_ms(start)
    assert isinstance(result, float)


def test_elapsed_ms_is_non_negative():
    start = mono_ns()
    result = elapsed_ms(start)
    assert result >= 0.0


def test_elapsed_ms_increases_with_time():
    start = mono_ns()
    time.sleep(0.01)  # 10 ms
    result = elapsed_ms(start)
    assert result >= 5.0  # generous lower bound


def test_elapsed_ms_scale_is_milliseconds():
    """elapsed_ms for a 10ms sleep should be ~10ms, not ~10000ms (seconds)."""
    start = mono_ns()
    time.sleep(0.01)
    result = elapsed_ms(start)
    # Should be in the 5-500ms range, not 0.01 (seconds) or 10000 (microseconds)
    assert 5.0 <= result <= 500.0


# ---------------------------------------------------------------------------
# rate_limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_returns_callable():
    limiter = rate_limiter(1.0)
    assert callable(limiter)


def test_rate_limiter_first_call_returns_true():
    """First call always returns True because last=0.0 and now >= 0.0 + interval."""
    limiter = rate_limiter(0.1)
    assert limiter() is True


def test_rate_limiter_immediate_second_call_returns_false():
    """Immediate second call within the interval must return False."""
    limiter = rate_limiter(10.0)  # 10 second interval
    assert limiter() is True   # first call
    assert limiter() is False  # second call, well within interval


def test_rate_limiter_returns_true_after_interval():
    """After the interval elapses, the limiter returns True again."""
    limiter = rate_limiter(0.02)  # 20 ms interval
    assert limiter() is True
    assert limiter() is False
    time.sleep(0.03)  # wait >20ms
    assert limiter() is True


def test_rate_limiter_independent_instances():
    """Two rate_limiter instances are independent."""
    a = rate_limiter(10.0)
    b = rate_limiter(10.0)
    assert a() is True
    assert b() is True
    assert a() is False
    assert b() is False
