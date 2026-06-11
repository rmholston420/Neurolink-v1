"""Monotonic clock helpers."""
import time


def now_utc_ts() -> float:
    """Return current UTC time as Unix timestamp."""
    return time.time()


def monotonic_ms() -> float:
    """Return monotonic time in milliseconds."""
    return time.monotonic() * 1000.0
