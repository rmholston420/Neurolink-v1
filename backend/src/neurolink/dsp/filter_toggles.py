"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True.

The module-level singleton is accessed by:
  from neurolink.dsp.filter_toggles import get_toggles, set_toggles
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

# Keys excluded from the public to_dict() / set_toggles() API.
# stage6_cardiac is internal to eeg_pump and not exposed via PUT /api/v1/filters.
_EXCLUDED_KEYS: frozenset[str] = frozenset({"stage6_cardiac"})


@dataclass
class FilterToggleConfig:
    """One bool per pipeline stage.  True = stage runs; False = bypassed."""

    stage1_fir: bool = True
    stage2_bad_channels: bool = True
    stage3_artifact_gate: bool = True
    stage3b_artifact_detector: bool = True
    stage4_asr: bool = True
    stage4b_baseline: bool = True
    stage5_ocular: bool = True
    stage6_cardiac: bool = True
    imu_gate: bool = True

    def to_dict(self) -> dict[str, bool]:
        """Return public toggle keys (excludes internal-only keys)."""
        d = asdict(self)
        for k in _EXCLUDED_KEYS:
            d.pop(k, None)
        return d


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_toggles = FilterToggleConfig()


def get_toggles() -> FilterToggleConfig:
    """Return a shallow copy of the current toggle config (thread-safe)."""
    with _lock:
        return FilterToggleConfig(**asdict(_toggles))


def set_toggles(updates: dict[str, bool]) -> FilterToggleConfig:
    """Merge *updates* into the live config and return the new state.

    Only public keys (those returned by to_dict()) are accepted;
    internal keys (stage6_cardiac) and unknown keys are silently ignored.
    """
    global _toggles
    valid_keys = FilterToggleConfig().to_dict().keys()
    with _lock:
        current = asdict(_toggles)
        for k, v in updates.items():
            if k in valid_keys and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
