"""Unit tests for custom exception hierarchy."""

from __future__ import annotations

import pytest

from neurolink.exceptions import (
    AdapterAlreadyConnectedError,
    AdapterNotConnectedError,
    AdapterNotFoundError,
    NeurolinkError,
)


class TestExceptionHierarchy:
    def test_base_is_exception(self):
        exc = NeurolinkError("base")
        assert isinstance(exc, Exception)

    def test_not_connected_is_neurolink_error(self):
        exc = AdapterNotConnectedError("msg")
        assert isinstance(exc, NeurolinkError)

    def test_already_connected_is_neurolink_error(self):
        exc = AdapterAlreadyConnectedError("msg")
        assert isinstance(exc, NeurolinkError)

    def test_not_found_is_neurolink_error(self):
        exc = AdapterNotFoundError("msg")
        assert isinstance(exc, NeurolinkError)

    def test_raise_and_catch_not_connected(self):
        with pytest.raises(AdapterNotConnectedError, match="not connected"):
            raise AdapterNotConnectedError("not connected")

    def test_raise_and_catch_already_connected(self):
        with pytest.raises(AdapterAlreadyConnectedError):
            raise AdapterAlreadyConnectedError("already")
