"""Unit tests for utils.timing."""

from __future__ import annotations

import time

from neurolink.utils.timing import elapsed_ms, mono_ns, mono_sec, rate_limiter


def test_mono_ns_returns_int():
    result = mono_ns()
    assert isinstance(result, int)
    assert result > 0


def test_mono_ns_increases():
    t1 = mono_ns()
    time.sleep(0.001)
    t2 = mono_ns()
    assert t2 > t1


def test_mono_sec_returns_float():
    result = mono_sec()
    assert isinstance(result, float)
    assert result > 0


def test_mono_sec_increases():
    t1 = mono_sec()
    time.sleep(0.001)
    t2 = mono_sec()
    assert t2 > t1


def test_elapsed_ms_positive():
    start = mono_ns()
    time.sleep(0.005)
    ms = elapsed_ms(start)
    assert ms >= 1.0


def test_elapsed_ms_near_zero_for_immediate_call():
    start = mono_ns()
    ms = elapsed_ms(start)
    assert ms < 100.0  # should be well under 100ms


def test_rate_limiter_allows_first_call():
    limiter = rate_limiter(1.0)
    assert limiter() is True


def test_rate_limiter_blocks_second_immediate_call():
    limiter = rate_limiter(1.0)
    limiter()  # first call
    assert limiter() is False


def test_rate_limiter_allows_after_interval():
    limiter = rate_limiter(0.01)
    limiter()  # first call
    time.sleep(0.02)
    assert limiter() is True


def test_rate_limiter_independent_instances():
    a = rate_limiter(1.0)
    b = rate_limiter(1.0)
    assert a() is True
    assert b() is True  # independent state
