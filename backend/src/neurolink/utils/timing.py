"""Monotonic clock utilities for Neurolink."""
from __future__ import annotations

import time


def now_ms() -> float:
    """Return current time as Unix timestamp in milliseconds."""
    return time.time() * 1000.0


def monotonic_ms() -> float:
    """Return monotonic clock in milliseconds."""
    return time.monotonic() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Return elapsed milliseconds since start_ms (monotonic)."""
    return monotonic_ms() - start_ms
