"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True.

Public API
----------
  get_toggles()  -> FilterToggleConfig  (copy of current singleton)
  set_toggles()  -> FilterToggleConfig  (merge updates, return new state)

to_dict() returns all 9 dataclass fields, including stage6_cardiac.
This is the complete set used by the reset_toggles test fixture and by
any REST endpoint that serialises the full toggle state.
Unknown keys and non-bool values passed to set_toggles() are silently ignored.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass
class FilterToggleConfig:
    """One bool per pipeline stage. True = stage runs; False = bypassed."""

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
        """Return all toggle fields as a plain dict."""
        return asdict(self)


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

    Accepts all dataclass field names.  Unknown keys and non-bool values
    are silently ignored.
    """
    global _toggles
    valid_keys = {f for f in asdict(FilterToggleConfig())}
    with _lock:
        current = asdict(_toggles)
        for k, v in updates.items():
            if k in valid_keys and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
