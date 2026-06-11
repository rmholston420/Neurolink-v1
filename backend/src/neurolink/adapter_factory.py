"""Adapter factory: creates the correct HardwareAdapter based on env/settings.

Ported from Rigpa-v3 adapter_factory.py.
All hardware imports are lazy.
"""
from __future__ import annotations

import structlog

from neurolink.config import get_settings
from neurolink.hardware.base import HardwareAdapter

log = structlog.get_logger(__name__)


def create_adapter(
    adapter_type: str | None = None,
    device_model: str | None = None,
    address: str | None = None,
) -> HardwareAdapter:
    """Create and return the appropriate HardwareAdapter.

    Args:
        adapter_type: 'mock' | 'ble' | 'lsl'. Defaults to settings.
        device_model: 'muse_s_gen1' | 'muse_s_athena'. Defaults to settings.
        address: BLE MAC address. Defaults to settings.

    Returns:
        Concrete HardwareAdapter instance.

    Raises:
        ValueError: if adapter_type or device_model is unrecognised.
    """
    settings = get_settings()
    _adapter_type = adapter_type or settings.adapter_type
    _device_model = device_model or settings.device_model
    _address = address or settings.muse_ble_address

    log.info(
        "adapter_factory",
        adapter_type=_adapter_type,
        device_model=_device_model,
        address=_address,
    )

    if _adapter_type == "mock":
        from neurolink.hardware.mock import MockAdapter  # lazy
        return MockAdapter(publish_hz=settings.publish_hz)

    if _adapter_type == "ble":
        if _device_model == "muse_s_gen1":
            from neurolink.hardware.muse_s.ble_adapter import MuseSBleAdapter  # lazy
            if not _address:
                raise ValueError(
                    "NEUROLINK_MUSE_BLE_ADDRESS is required for ble adapter mode"
                )
            return MuseSBleAdapter(address=_address)
        elif _device_model == "muse_s_athena":
            from neurolink.hardware.muse_athena.ble_adapter import AthenaBlueAdapter  # lazy
            return AthenaBlueAdapter()
        else:
            raise ValueError(f"Unknown device_model: {_device_model}")

    if _adapter_type == "lsl":
        if _device_model == "muse_s_gen1":
            from neurolink.hardware.muse_s.lsl_adapter import MuseSLslAdapter  # lazy
            return MuseSLslAdapter()
        elif _device_model == "muse_s_athena":
            from neurolink.hardware.muse_athena.ble_adapter import AthenaBlueAdapter  # lazy
            return AthenaBlueAdapter()
        else:
            raise ValueError(f"Unknown device_model: {_device_model}")

    raise ValueError(f"Unknown adapter_type: {_adapter_type}")
