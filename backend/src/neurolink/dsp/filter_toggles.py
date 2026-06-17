"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True.

Public API
----------
  get_toggles()  -> FilterToggleConfig  (copy of current singleton)
  set_toggles()  -> FilterToggleConfig  (merge updates, return new state)

to_dict() returns the 8 *public* stage fields that are serialised to REST
clients and used for logging.  ``stage6_cardiac`` is intentionally excluded
from to_dict() because it is an internal implementation detail managed
directly via the ``stage6_cardiac`` attribute; it is still a fully
functional dataclass field and can be set via set_toggles().

Unknown keys and non-bool values passed to set_toggles() are silently ignored.
"""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import asdict, dataclass

# Keys excluded from to_dict() — still valid set_toggles() targets.
_INTERNAL_TOGGLE_KEYS: frozenset[str] = frozenset({"stage6_cardiac"})


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
        """Return the 8 public toggle fields (excludes stage6_cardiac)."""
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in _INTERNAL_TOGGLE_KEYS
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_toggles = FilterToggleConfig()

# All valid field names (including internal ones) for set_toggles() validation.
_ALL_TOGGLE_KEYS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(FilterToggleConfig())
)


def get_toggles() -> FilterToggleConfig:
    """Return a shallow copy of the current toggle config (thread-safe)."""
    with _lock:
        return FilterToggleConfig(**asdict(_toggles))


def set_toggles(updates: dict[str, bool]) -> FilterToggleConfig:
    """Merge *updates* into the live config and return the new state.

    Accepts all dataclass field names (including stage6_cardiac).
    Unknown keys and non-bool values are silently ignored.
    """
    global _toggles
    with _lock:
        current = asdict(_toggles)
        for k, v in updates.items():
            if k in _ALL_TOGGLE_KEYS and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
