"""Hardware adapter abstract base class.

Ported from Rigpa-v3 hardware/base.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator

import numpy as np


class DeviceModel(str, Enum):
    """Supported EEG device models."""

    MUSE_S_GEN1 = "muse_s_gen1"
    MUSE_S_ATHENA = "muse_s_athena"
    MOCK = "mock"


class EEGSample:
    """A single EEG sample frame from any hardware adapter."""

    __slots__ = (
        "timestamp",
        "eeg",       # dict[channel_name, list[float]] or np.ndarray
        "ppg",       # list[float] | None
        "accel",     # list[float] | None
        "gyro",      # list[float] | None
        "source",    # str
        "address",   # str
        "poor_contact",   # bool
        "contact_quality",  # float | None
        "fnirs",     # list[float] | None (Athena only)
    )

    def __init__(
        self,
        timestamp: float,
        eeg: dict[str, list[float]] | np.ndarray | None = None,
        ppg: list[float] | None = None,
        accel: list[float] | None = None,
        gyro: list[float] | None = None,
        source: str = "unknown",
        address: str = "",
        poor_contact: bool = False,
        contact_quality: float | None = None,
        fnirs: list[float] | None = None,
    ) -> None:
        self.timestamp = timestamp
        self.eeg = eeg or {}
        self.ppg = ppg
        self.accel = accel
        self.gyro = gyro
        self.source = source
        self.address = address
        self.poor_contact = poor_contact
        self.contact_quality = contact_quality
        self.fnirs = fnirs


class HardwareAdapter(ABC):
    """Abstract base class for all EEG hardware adapters."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the hardware device."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the hardware device."""
        ...

    @abstractmethod
    async def stream(self) -> AsyncGenerator[EEGSample, None]:
        """Yield EEGSample frames as they arrive from the hardware."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter is currently connected."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source identifier string for this adapter."""
        ...
