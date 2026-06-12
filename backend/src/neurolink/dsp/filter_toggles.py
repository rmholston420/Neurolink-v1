"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True — behaviour is unchanged on first deploy.

The module-level singleton is accessed by:
  from neurolink.dsp.filter_toggles import get_toggles, set_toggles

Thread safety
-------------
get_toggles() returns an immutable snapshot (dataclass copy) so the
EEGPump hot-path never races with a concurrent PUT /api/v1/filters.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field


@dataclass
class FilterToggleConfig:
    """One bool per pipeline stage.  True = stage runs; False = bypassed."""

    # Stage 1 — zero-phase FIR chain (HP 0.5 Hz + notch + LP 45 Hz)
    stage1_fir: bool = True

    # Stage 2 — bad channel detection + spherical-spline interpolation
    stage2_bad_channels: bool = True

    # Stage 3 — epoch-level artifact gate (amplitude / IMU RMS / kurtosis)
    stage3_artifact_gate: bool = True

    # Stage 4 — Artifact Subspace Reconstruction (ASR burst repair)
    stage4_asr: bool = True

    # Stage 4b — session baseline recorder (impedance stabilisation + ASR cal)
    stage4b_baseline: bool = True

    # Stage 5 — Gratton-Coles ocular regression
    stage5_ocular: bool = True

    # IMU motion gate (Stage 0 / Stage 3 motion criterion)
    imu_gate: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_toggles = FilterToggleConfig()


def get_toggles() -> FilterToggleConfig:
    """Return a shallow copy of the current toggle config (thread-safe)."""
    with _lock:
        # dataclass is mutable; return a copy so callers cannot mutate state
        return FilterToggleConfig(**_toggles.to_dict())


def set_toggles(updates: dict[str, bool]) -> FilterToggleConfig:
    """Merge *updates* into the live config and return the new state.

    Only keys present in FilterToggleConfig are accepted; unknown keys
    are silently ignored so partial PUT bodies are safe.
    """
    global _toggles
    valid_keys = FilterToggleConfig().to_dict().keys()
    with _lock:
        current = _toggles.to_dict()
        for k, v in updates.items():
            if k in valid_keys and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
