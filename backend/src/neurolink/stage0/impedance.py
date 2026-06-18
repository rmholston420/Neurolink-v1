"""Stage 0 — Impedance Guard.

Per-channel impedance monitoring with device-appropriate thresholds:
  dry electrode:      alert above 200 kΩ
  semi-wet electrode: alert above  20 kΩ

The Muse family reports a boolean `poor_contact` flag per channel; many
research-grade amplifiers stream actual impedance values in kΩ.  This module
supports both: numeric kΩ values when available, boolean fallback otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

# ── Muse S channel labels (TP9, AF7, AF8, TP10, AUX) ────────────────────────
MUSE_CHANNELS: list[str] = ["TP9", "AF7", "AF8", "TP10", "AUX"]

# Impedance thresholds in kΩ per electrode type
_THRESHOLDS_KOHM: dict[str, float] = {
    "dry": 200.0,
    "semi_wet": 20.0,
    "wet": 5.0,
}


class ImpedanceLevel(StrEnum):
    OK = "ok"  # below threshold
    HIGH = "high"  # above threshold — alert user
    UNKNOWN = "unknown"  # no reading yet


@dataclass
class ImpedanceChannelStatus:
    """Status for a single EEG channel."""

    label: str
    kohm: float | None = None  # measured impedance (kΩ); None if unknown
    poor_contact: bool = False  # device boolean flag
    level: ImpedanceLevel = ImpedanceLevel.UNKNOWN
    threshold_kohm: float = 200.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "kohm": self.kohm,
            "poor_contact": self.poor_contact,
            "level": self.level.value,
            "threshold_kohm": self.threshold_kohm,
            "last_updated": self.last_updated,
        }


class ImpedanceGuard:
    """Tracks per-channel impedance and evaluates threshold breaches.

    Designed to be updated from two sources:
      1. EEGSample.poor_contact  (boolean, from Muse BLE adapter)
      2. POST /api/v1/stage0/impedance  (numeric kΩ, from richer amplifiers)

    The guard never blocks acquisition — it sets status flags that the
    frontend reads via SSE / REST and surfaces to the user.
    """

    def __init__(self, electrode_type: str = "dry", n_channels: int = 5) -> None:
        self._electrode_type = electrode_type
        self._threshold = _THRESHOLDS_KOHM.get(electrode_type, 200.0)
        self._channels: list[ImpedanceChannelStatus] = [
            ImpedanceChannelStatus(
                label=MUSE_CHANNELS[i] if i < len(MUSE_CHANNELS) else f"CH{i}",
                threshold_kohm=self._threshold,
            )
            for i in range(n_channels)
        ]

    # ------------------------------------------------------------------
    # Update from EEGSample (called every pump tick)
    # ------------------------------------------------------------------

    def update_from_sample(self, poor_contact: bool, channels: list[float] | None = None) -> None:
        """Update all channels from a single EEGSample poor_contact flag.

        When the device does not report per-channel impedance (e.g. Muse S),
        the single `poor_contact` boolean propagates to all non-AUX channels.
        """
        ts = time.time()
        for ch in self._channels:
            if ch.label == "AUX":
                continue
            ch.poor_contact = poor_contact
            ch.last_updated = ts
            if poor_contact:
                ch.level = ImpedanceLevel.HIGH
            elif ch.level == ImpedanceLevel.UNKNOWN:
                ch.level = ImpedanceLevel.OK

    # ------------------------------------------------------------------
    # Update from explicit kΩ readings (REST endpoint or rich amplifiers)
    # ------------------------------------------------------------------

    def update_from_kohm(self, readings: dict[str, float]) -> None:
        """Update channels from a label -> kΩ mapping.

        Args:
            readings: e.g. {"TP9": 15.3, "AF7": 250.0, "AF8": 12.1,
                            "TP10": 8.7, "AUX": 0.0}
        """
        ts = time.time()
        for ch in self._channels:
            if ch.label in readings:
                kohm = readings[ch.label]
                ch.kohm = kohm
                ch.last_updated = ts
                ch.level = ImpedanceLevel.HIGH if kohm > self._threshold else ImpedanceLevel.OK

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def all_channels_ok(self) -> bool:
        """True when every channel is OK or UNKNOWN (no readings yet)."""
        return all(ch.level != ImpedanceLevel.HIGH for ch in self._channels)

    @property
    def bad_channels(self) -> list[str]:
        """Labels of channels currently above threshold."""
        return [ch.label for ch in self._channels if ch.level == ImpedanceLevel.HIGH]

    @property
    def threshold_kohm(self) -> float:
        return self._threshold

    @property
    def electrode_type(self) -> str:
        return self._electrode_type

    def channel_status(self, label: str) -> ImpedanceChannelStatus | None:
        for ch in self._channels:
            if ch.label == label:
                return ch
        return None

    def summary_dict(self) -> dict:
        """Serialisable summary for REST / SSE."""
        return {
            "electrode_type": self._electrode_type,
            "threshold_kohm": self._threshold,
            "all_channels_ok": self.all_channels_ok,
            "bad_channels": self.bad_channels,
            "channels": [ch.to_dict() for ch in self._channels],
        }
