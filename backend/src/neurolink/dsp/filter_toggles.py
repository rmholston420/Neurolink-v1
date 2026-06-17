"""Runtime pipeline stage toggles.

Each field controls whether a DSP stage runs during EEGPump._build_payload().
All stages default to enabled=True.

Public API
----------
  get_toggles()  -> FilterToggleConfig  (copy of current singleton)
  set_toggles()  -> FilterToggleConfig  (merge updates, return new state)

to_dict() intentionally omits stage6_cardiac; that toggle is accessible via
get_toggles().stage6_cardiac but is not part of the generic key/value API
exposed to the filters endpoint, so it will not be accidentally bulk-disabled.
set_toggles({'stage6_cardiac': False}) still works because the key is accepted
by the dataclass; it just won't appear in to_dict() round-trips.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

# Keys excluded from the public to_dict() / set_toggles() key-enumeration API.
# stage6_cardiac is excluded from the dict view because the cardiac regression
# toggle has a dedicated UI control and must not appear in bulk-toggle lists.
_EXCLUDED_KEYS: frozenset[str] = frozenset({"stage6_cardiac"})


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
        """Return all public toggle keys as a dict (excludes _EXCLUDED_KEYS)."""
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

    Accepts ALL dataclass field names (including stage6_cardiac).
    Unknown keys and non-bool values are silently ignored.
    """
    global _toggles
    # Valid keys = all dataclass fields (not just the public to_dict() subset)
    valid_keys = {f.name for f in FilterToggleConfig.__dataclass_fields__.values()}
    with _lock:
        current = asdict(_toggles)
        for k, v in updates.items():
            if k in valid_keys and isinstance(v, bool):
                current[k] = v
        _toggles = FilterToggleConfig(**current)
        return FilterToggleConfig(**current)
