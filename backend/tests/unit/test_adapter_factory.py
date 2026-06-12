"""Unit tests for adapter_factory.py."""

from __future__ import annotations

import os

import pytest

from neurolink.adapter_factory import create_adapter
from neurolink.hardware.mock import MockAdapter


def test_create_adapter_mock():
    adapter = create_adapter(adapter_type="mock")
    assert isinstance(adapter, MockAdapter)


def test_create_adapter_mock_default():
    """Default adapter_type from settings should be mock in test env."""
    import neurolink.config as config_module

    prev_type = os.environ.get("NEUROLINK_ADAPTER_TYPE")
    prev_settings = config_module._settings
    try:
        os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
        config_module._settings = None
        adapter = create_adapter()
        assert isinstance(adapter, MockAdapter)
    finally:
        if prev_type is None:
            os.environ.pop("NEUROLINK_ADAPTER_TYPE", None)
        else:
            os.environ["NEUROLINK_ADAPTER_TYPE"] = prev_type
        config_module._settings = prev_settings


def test_create_adapter_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown adapter_type"):
        create_adapter(adapter_type="unknown_xyz")


def test_create_adapter_ble_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown device_model"):
        create_adapter(
            adapter_type="ble", device_model="unknown_model", address="AA:BB:CC:DD:EE:FF"
        )
