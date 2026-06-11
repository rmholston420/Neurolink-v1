"""Adapter factory: creates the correct HardwareAdapter from settings.

Ported from Rigpa-v3 adapter_factory.py.
"""

from __future__ import annotations

from neurolink.config import get_settings
from neurolink.hardware.base import HardwareAdapter


def create_adapter(
    adapter_type: str | None = None,
    device_model: str | None = None,
    address: str | None = None,
) -> HardwareAdapter:
    """Create and return the appropriate HardwareAdapter.

    Args:
        adapter_type: "mock" | "ble" | "lsl" (defaults to settings.adapter_type)
        device_model: "muse_s_gen1" | "muse_s_athena" (defaults to settings.device_model)
        address: BLE MAC address (required for BLE mode)

    Returns:
        Configured HardwareAdapter instance.

    Raises:
        ValueError: for unknown adapter_type or device_model.
    """
    settings = get_settings()
    _type = (adapter_type or settings.adapter_type).lower()
    _model = (device_model or settings.device_model).lower()
    _address = address or settings.muse_ble_address

    if _type == "mock":
        from neurolink.hardware.mock import MockAdapter

        return MockAdapter()

    if _type == "ble":
        if _model == "muse_s_gen1":
            from neurolink.hardware.muse_s.ble_adapter import MuseSBleAdapter

            return MuseSBleAdapter(address=_address)
        elif _model == "muse_s_athena":
            # Athena BLE is managed via OpenMuse LSL
            from neurolink.hardware.muse_athena.ble_adapter import AthenaBlueAdapter

            return AthenaBlueAdapter()
        else:
            raise ValueError(f"Unknown device_model for BLE: {_model!r}")

    if _type == "lsl":
        if _model == "muse_s_athena":
            from neurolink.hardware.muse_athena.ble_adapter import AthenaBlueAdapter

            return AthenaBlueAdapter()
        else:
            from neurolink.hardware.muse_s.lsl_adapter import MuseSLslAdapter

            return MuseSLslAdapter()

    raise ValueError(f"Unknown adapter_type: {_type!r}")
