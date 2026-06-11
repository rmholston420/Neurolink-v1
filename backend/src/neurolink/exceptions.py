"""Custom exception classes for Neurolink."""
from __future__ import annotations


class NeurolinkError(Exception):
    """Base Neurolink exception."""


class AdapterNotConnectedError(NeurolinkError):
    """Raised when an operation requires an active EEG adapter."""


class CalibrationBusyError(NeurolinkError):
    """Raised when a calibration is already in progress."""


class BLETimeoutError(NeurolinkError):
    """Raised when BLE scan/connect times out."""


class UnknownDeviceError(NeurolinkError):
    """Raised when an unrecognised device model is requested."""


class NoEEGDataError(NeurolinkError):
    """Raised when no EEG data is available yet."""
