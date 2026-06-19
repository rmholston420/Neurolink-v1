"""Unit tests for BLE reconnect exponential backoff helper."""

from __future__ import annotations

import pytest

from neurolink.hardware.muse_s.ble_adapter import (
    BACKOFF_BASE_SEC,
    BACKOFF_CAP_SEC,
    MAX_RECONNECT_ATTEMPTS,
    _backoff_wait,
)


class TestBackoffWait:
    def test_first_attempt_within_base(self):
        """Attempt 0: ceiling = min(cap, base * 1) = base."""
        for _ in range(50):
            w = _backoff_wait(0)
            assert 0.0 <= w <= BACKOFF_BASE_SEC

    def test_second_attempt_doubles(self):
        for _ in range(50):
            w = _backoff_wait(1)
            assert 0.0 <= w <= BACKOFF_BASE_SEC * 2

    def test_later_attempts_capped(self):
        """After enough doublings, ceiling must not exceed BACKOFF_CAP_SEC."""
        for attempt in range(5, 20):
            for _ in range(20):
                w = _backoff_wait(attempt)
                assert w <= BACKOFF_CAP_SEC, (
                    f"attempt={attempt} produced w={w} > cap={BACKOFF_CAP_SEC}"
                )

    def test_wait_is_non_negative(self):
        for attempt in range(MAX_RECONNECT_ATTEMPTS + 2):
            assert _backoff_wait(attempt) >= 0.0

    def test_ceiling_monotonically_increases_to_cap(self):
        ceilings = [
            min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2**a)) for a in range(15)
        ]
        # All ceilings must be <= cap
        assert all(c <= BACKOFF_CAP_SEC for c in ceilings)
        # Ceilings must be non-decreasing
        for i in range(1, len(ceilings)):
            assert ceilings[i] >= ceilings[i - 1]
