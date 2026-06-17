"""Hardware adapter abstract base class and supporting types.

All concrete adapters inherit from HardwareAdapter and implement:
    connect(), disconnect(), read_sample(), is_connected
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DeviceModel(StrEnum):
    MUSE_S_GEN1 = "muse_s_gen1"
    MUSE_S_ATHENA = "muse_s_athena"
    MOCK = "mock"


@dataclass
class EEGSample:
    """A single EEG sample snapshot from any adapter."""

    channels: list[float] = field(default_factory=lambda: [0.0] * 5)
    timestamp: float = field(default_factory=time.time)
    source: str = "mock"
    address: str = ""
    poor_contact: bool = False
    eeg_buffer: list[list[float]] | None = None
    ppg_buffer: list[float] | None = None
    accel_buffer: list[list[float]] | None = None
    gyro_buffer: list[list[float]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def eeg(self) -> list[float]:
        """Alias for channels (backward compatibility with tests)."""
        return self.channels


class HardwareAdapter(ABC):
    """Abstract base class for all Neurolink hardware adapters."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def read_sample(self) -> EEGSample | None:
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @property
    def source_name(self) -> str:
        return "unknown"
