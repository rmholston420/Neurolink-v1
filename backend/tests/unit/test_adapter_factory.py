"""Unit tests for adapter_factory.create_adapter."""

from __future__ import annotations

import pytest

from neurolink.adapter_factory import create_adapter
from neurolink.exceptions import AdapterNotFoundError
from neurolink.hardware.mock import MockAdapter


class TestCreateAdapter:
    def test_mock_adapter_returned(self):
        adapter = create_adapter(adapter_type="mock", device_model="mock")
        assert isinstance(adapter, MockAdapter)

    def test_unknown_type_raises(self):
        with pytest.raises((AdapterNotFoundError, ValueError, KeyError)):
            create_adapter(adapter_type="nonexistent_xyz", device_model="mock")

    def test_mock_source_name(self):
        adapter = create_adapter(adapter_type="mock", device_model="mock")
        assert isinstance(adapter.source_name, str)
        assert len(adapter.source_name) > 0
