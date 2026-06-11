"""Unit tests for custom Neurolink exception classes."""

from __future__ import annotations

import pytest

from neurolink.exceptions import (
    AdapterAlreadyConnectedError,
    AdapterNotConnectedError,
    BLEConnectionError,
    CalibrationError,
    InvalidAdapterTypeError,
    NeurolinkError,
)


def test_neurolink_error_is_exception():
    with pytest.raises(NeurolinkError):
        raise NeurolinkError("base error")


def test_adapter_not_connected_error():
    with pytest.raises(AdapterNotConnectedError):
        raise AdapterNotConnectedError("not connected")


def test_adapter_not_connected_is_neurolink_error():
    assert issubclass(AdapterNotConnectedError, NeurolinkError)


def test_adapter_already_connected_error():
    with pytest.raises(AdapterAlreadyConnectedError):
        raise AdapterAlreadyConnectedError("already connected")


def test_adapter_already_connected_is_neurolink_error():
    assert issubclass(AdapterAlreadyConnectedError, NeurolinkError)


def test_calibration_error():
    with pytest.raises(CalibrationError):
        raise CalibrationError("calibration failed")


def test_calibration_error_is_neurolink_error():
    assert issubclass(CalibrationError, NeurolinkError)


def test_ble_connection_error():
    with pytest.raises(BLEConnectionError):
        raise BLEConnectionError("ble failed")


def test_ble_connection_error_is_neurolink_error():
    assert issubclass(BLEConnectionError, NeurolinkError)


def test_invalid_adapter_type_error():
    with pytest.raises(InvalidAdapterTypeError):
        raise InvalidAdapterTypeError("unknown adapter")


def test_invalid_adapter_type_is_neurolink_error():
    assert issubclass(InvalidAdapterTypeError, NeurolinkError)


def test_exception_message_preserved():
    msg = "something went wrong"
    err = NeurolinkError(msg)
    assert str(err) == msg


def test_exceptions_catchable_as_base():
    for exc_class in [
        AdapterNotConnectedError,
        AdapterAlreadyConnectedError,
        CalibrationError,
        BLEConnectionError,
        InvalidAdapterTypeError,
    ]:
        with pytest.raises(NeurolinkError):
            raise exc_class("test")
