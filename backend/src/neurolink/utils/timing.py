"""Monotonic clock helpers used across the Neurolink pipeline.

Ported from Rigpa-v3 utils/timing.py.
All helpers are pure / no-side-effects except mono_ns() which reads the
system clock.
"""

from __future__ import annotations

import time


def mono_ns() -> int:
    """Return current monotonic time in nanoseconds.

    Suitable for high-resolution frame timing without wall-clock drift.

    Returns:
        int: Monotonic time in nanoseconds.
    """
    return time.monotonic_ns()


def mono_sec() -> float:
    """Return current monotonic time in seconds (float).

    Returns:
        float: Monotonic time in seconds.
    """
    return time.monotonic()


def elapsed_ms(start_ns: int) -> float:
    """Return elapsed milliseconds since a previous mono_ns() reading.

    Args:
        start_ns: Monotonic start time in nanoseconds (from mono_ns()).

    Returns:
        float: Elapsed time in milliseconds.
    """
    return (time.monotonic_ns() - start_ns) / 1_000_000.0


def rate_limiter(interval_sec: float):
    """Return a zero-argument callable that returns True at most once per
    interval_sec, and False otherwise.  Useful for log throttling.

    Args:
        interval_sec: Minimum seconds between True returns.

    Returns:
        Callable[[], bool]
    """
    last: list[float] = [0.0]

    def _check() -> bool:
        now = time.monotonic()
        if now - last[0] >= interval_sec:
            last[0] = now
            return True
        return False

    return _check
