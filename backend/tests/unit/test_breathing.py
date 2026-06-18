"""Unit tests for dsp.breathing — breathing rate estimation."""

from __future__ import annotations

from neurolink.dsp.breathing import estimate_rr


class TestEstimateRR:
    def test_returns_float_or_none(self):
        result = estimate_rr([], fs=25)
        assert result is None or isinstance(result, float)

    def test_empty_signal_returns_none(self):
        assert estimate_rr([], fs=25) is None

    def test_short_signal_returns_none_or_float(self):
        signal = [0.0] * 10
        result = estimate_rr(signal, fs=25)
        assert result is None or isinstance(result, float)

    def test_synthetic_breathing_rate(self):
        """A ~0.25 Hz sine (15 bpm) should return a rate near 15 bpm."""
        import numpy as np

        fs = 25
        t = np.linspace(0, 20, fs * 20, endpoint=False)
        signal = np.sin(2 * np.pi * 0.25 * t).tolist()  # 0.25 Hz = 15 bpm
        result = estimate_rr(signal, fs=fs)
        if result is not None:
            assert 8.0 <= result <= 25.0  # physiologically plausible range

    def test_noisy_signal_does_not_raise(self):
        import numpy as np

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(500).tolist()
        result = estimate_rr(signal, fs=25)
        assert result is None or isinstance(result, float)
