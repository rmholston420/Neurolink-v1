"""Fatigue detector: rolling 30-sample theta/alpha ratio.

Ported from Rigpa-v3 fatigue.py.
Thread-safe via internal deque (no additional locking needed for single writer).
"""

from __future__ import annotations

from collections import deque

_WINDOW_SIZE: int = 30
_THETA_DOMINANCE_THRESHOLD: float = 1.5  # theta/alpha ratio above which fatigue is high


class FatigueDetector:
    """Rolling window theta/alpha ratio fatigue tracker.

    A rising theta/alpha ratio is associated with fatigue and drowsiness.
    """

    def __init__(self, window: int = _WINDOW_SIZE) -> None:
        self._window = window
        self._ratios: deque[float] = deque(maxlen=window)

    def update(self, theta: float, alpha: float) -> float:
        """Update the rolling window with a new theta/alpha ratio.

        Args:
            theta: Theta band power fraction
            alpha: Alpha band power fraction

        Returns:
            Current normalised fatigue score in [0, 1].
            0.0 if fewer than 2 samples have been recorded.
        """
        ratio = theta / (alpha + 1e-6)
        self._ratios.append(ratio)
        return self._compute_score()

    def _compute_score(self) -> float:
        """Compute normalised fatigue score from current rolling window."""
        if len(self._ratios) < 2:
            return 0.0
        mean_ratio = sum(self._ratios) / len(self._ratios)
        # Normalise: 0 = no fatigue (ratio=0), 1 = high fatigue (ratio >= threshold)
        return float(min(1.0, mean_ratio / _THETA_DOMINANCE_THRESHOLD))

    def reset(self) -> None:
        """Clear the rolling window."""
        self._ratios.clear()

    @property
    def score(self) -> float:
        """Return current fatigue score without updating."""
        return self._compute_score()

    @property
    def sample_count(self) -> int:
        """Return number of samples in current window."""
        return len(self._ratios)
