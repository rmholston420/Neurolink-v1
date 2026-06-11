"""Custom exception classes for Neurolink."""

from __future__ import annotations


class NeurolinkError(Exception):
    """Base exception for all Neurolink errors."""


class AdapterNotConnectedError(NeurolinkError):
    """Raised when an operation requires an active EEG adapter but none is connected."""


class AdapterAlreadyConnectedError(NeurolinkError):
    """Raised when connect() is called on an already-connected adapter."""


class CalibrationError(NeurolinkError):
    """Raised when calibration fails."""


class BLEConnectionError(NeurolinkError):
    """Raised when BLE connection or command fails."""


class InvalidAdapterTypeError(NeurolinkError):
    """Raised for unknown adapter_type in adapter factory."""
