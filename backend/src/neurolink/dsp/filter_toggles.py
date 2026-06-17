"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True.

Public API
----------
  get_toggles()  -> FilterToggleConfig  (copy of current singleton)
  set_toggles()  -> FilterToggleConfig  (merge updates, return new state)

to_dict() returns ALL dataclass fields, including stage6_cardiac.
set_toggles({'stage6_cardiac': False}) works via the normal key path.
Unknown keys and non-bool values are silently ignored.
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
        """Return all toggle keys as a dict (all dataclass fields)."""
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

    Accepts ALL dataclass field names (including stage6_cardiac).
    Unknown keys and non-bool values are silently ignored.
    """
    global _toggles
    valid_keys = {f.name for f in FilterToggleConfig.__dataclass_fields__.values()}
    with _lock:
        current = asdict(_toggles)
        for k, v in updates.items():
            if k in valid_keys and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
