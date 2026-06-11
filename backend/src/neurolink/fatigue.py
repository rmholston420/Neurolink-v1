"""Fatigue detector using rolling theta/alpha ratio.

Ported from Rigpa-v3 fatigue.py.
"""
from __future__ import annotations

from collections import deque

_WINDOW_SIZE: int = 30  # rolling window in frames
_EPS: float = 1e-6     # prevent division by zero


class FatigueDetector:
    """Computes a rolling fatigue score as theta/alpha ratio.

    Score approaches 1.0 as theta dominates; approaches 0 as alpha dominates.
    Normalised by sigmoid to [0, 1] range.
    """

    def __init__(self, window: int = _WINDOW_SIZE) -> None:
        self._window = window
        self._theta_buf: deque[float] = deque(maxlen=window)
        self._alpha_buf: deque[float] = deque(maxlen=window)

    def update(self, theta: float, alpha: float) -> float:
        """Update rolling window and return current fatigue score.

        Args:
            theta: current theta band power fraction
            alpha: current alpha band power fraction

        Returns:
            Fatigue score in [0, 1]. 0 = not fatigued; 1 = highly fatigued.
        """
        self._theta_buf.append(theta)
        self._alpha_buf.append(alpha)
        return self.score()

    def score(self) -> float:
        """Return current fatigue score without updating buffers."""
        if not self._theta_buf:
            return 0.0
        import numpy as np
        mean_theta = float(np.mean(list(self._theta_buf)))
        mean_alpha = float(np.mean(list(self._alpha_buf)))
        ratio = mean_theta / (mean_alpha + _EPS)
        # Normalise: ratio of ~1 -> fatigued; ratio ~0 -> not fatigued
        # Use sigmoid-like normalisation: score = ratio / (1 + ratio)
        return float(ratio / (1.0 + ratio))

    def reset(self) -> None:
        """Clear rolling buffers."""
        self._theta_buf.clear()
        self._alpha_buf.clear()
