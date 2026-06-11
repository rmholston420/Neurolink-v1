"""Hardware adapter abstract base class and supporting types.

All concrete adapters inherit from HardwareAdapter and implement:
    connect(), disconnect(), read_sample(), is_connected
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeviceModel(str, Enum):
    """Supported Muse device models."""
    MUSE_S_GEN1 = "muse_s_gen1"
    MUSE_S_ATHENA = "muse_s_athena"
    MOCK = "mock"


@dataclass
class EEGSample:
    """A single EEG sample snapshot from any adapter."""
    # Core EEG — one value per channel (TP9, AF7, AF8, TP10, AUX)
    channels: list[float] = field(default_factory=lambda: [0.0] * 5)
    timestamp: float = field(default_factory=time.time)
    source: str = "mock"
    address: str = ""
    poor_contact: bool = False
    # Raw multi-sample ring data (optional; filled by adapters)
    eeg_buffer: list[list[float]] | None = None      # (5, N) as nested list
    ppg_buffer: list[float] | None = None             # (N,)
    accel_buffer: list[list[float]] | None = None     # (3, N)
    gyro_buffer: list[list[float]] | None = None      # (3, N)
    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)


class HardwareAdapter(ABC):
    """Abstract base class for all Neurolink hardware adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the EEG device."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the EEG device and release resources."""
        ...

    @abstractmethod
    async def read_sample(self) -> EEGSample | None:
        """Read the latest EEG sample from the device buffer.

        Returns None if no new sample is available.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter is currently connected."""
        ...

    @property
    def source_name(self) -> str:
        """Human-readable source identifier for IngestPayload.source."""
        return "unknown"
