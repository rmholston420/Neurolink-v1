"""Unit tests for adapter_factory.py."""
from __future__ import annotations

import os
import pytest


def test_factory_returns_mock_adapter():
    os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
    import neurolink.config as cfg
    cfg._settings = None
    from neurolink.adapter_factory import create_adapter
    from neurolink.hardware.mock import MockAdapter
    adapter = create_adapter(adapter_type="mock")
    assert isinstance(adapter, MockAdapter)


def test_factory_mock_is_source_mock():
    from neurolink.adapter_factory import create_adapter
    adapter = create_adapter(adapter_type="mock")
    assert adapter.source_name == "mock"


def test_factory_ble_without_address_raises():
    import neurolink.config as cfg
    cfg._settings = None
    os.environ["NEUROLINK_ADAPTER_TYPE"] = "ble"
    os.environ["NEUROLINK_MUSE_BLE_ADDRESS"] = ""
    cfg._settings = None
    from neurolink.adapter_factory import create_adapter
    with pytest.raises(ValueError, match="NEUROLINK_MUSE_BLE_ADDRESS"):
        create_adapter(adapter_type="ble", device_model="muse_s_gen1", address="")
    os.environ["NEUROLINK_ADAPTER_TYPE"] = "mock"
    cfg._settings = None


def test_factory_unknown_adapter_type_raises():
    from neurolink.adapter_factory import create_adapter
    with pytest.raises(ValueError, match="Unknown adapter_type"):
        create_adapter(adapter_type="nope")


def test_factory_unknown_device_model_raises():
    from neurolink.adapter_factory import create_adapter
    with pytest.raises(ValueError, match="Unknown device_model"):
        create_adapter(adapter_type="ble", device_model="gadget", address="11:22:33:44:55:66")
