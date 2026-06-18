"""
routers/ble.py - Path B: backend BLE scan + connection endpoints.

Exposes four terminal-free endpoints so the frontend ConnectionPanel
(Backend BLE tab) can discover and connect to a Muse S headband without
the user ever opening a shell.

Endpoints
---------
GET  /ble/scan
    Run a bleak BLE discovery scan (default 5 s) and return every
    device whose name starts with "Muse" or matches the Muse InterAxon
    service UUID 0xfe8d.  Returns {devices: [{address, name, rssi}],
    scan_duration_sec}.

POST /ble/connect
    Body: {address: str, device_model: str}  (device_model optional,
    defaults to "muse_s_gen1").
    Calls adapter_factory.create_adapter() with adapter_type="ble",
    wraps it in a BLEBridge supervisor, and starts the bridge.
    If a bridge is already running, stops it first.
    Returns ConnectResponse.

POST /ble/disconnect
    Stops the running BLEBridge (if any) and disconnects the adapter.
    Returns {ok, message}.

GET  /ble/status
    Returns {running: bool, connected: bool, address: str|null,
    device_name: str|null}.

Registration
------------
Add to main.py::

    from neurolink.routers.ble import router as ble_router
    app.include_router(ble_router, prefix="/api/v1/neurolink")

    # Also add to the lifespan context if you want the bridge
    # automatically torn down on shutdown (see ble_router.bridge_state).
"""

from __future__ import annotations

import asyncio

try:
    from bleak import BleakScanner

    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = structlog.get_logger(__name__)

router = APIRouter(tags=["ble"])

# -- Muse device filter ----------------------------------------------------------
MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"
MUSE_NAME_PREFIX = "Muse"
DEFAULT_SCAN_SEC = 5.0


def _is_muse_adv(device, advertisement_data) -> bool:
    """Return True if the BLE advertisement looks like a Muse headband.

    Uses AdvertisementData (the current Bleak API) rather than the
    deprecated BLEDevice.metadata attribute.
    """
    name = (getattr(device, "name", "") or "").strip()
    service_uuids = getattr(advertisement_data, "service_uuids", []) or []
    return (
        name.startswith(MUSE_NAME_PREFIX)
        or MUSE_SERVICE_UUID in [u.lower() for u in service_uuids]
    )


# -- Bridge state (module-level singleton) ---------------------------------------
class _BridgeState:
    from neurolink.ble_bridge import BLEBridge as _BLEBridge
    from neurolink.hardware.base import HardwareAdapter as _HardwareAdapter

    bridge: _BLEBridge | None = None
    adapter: _HardwareAdapter | None = None
    address: str | None = None
    device_model: str = "muse_s_gen1"


bridge_state = _BridgeState()


async def _stop_existing_bridge() -> None:
    if bridge_state.bridge is not None:
        try:
            await bridge_state.bridge.stop()
        except Exception as exc:
            log.warning("ble_router_stop_bridge_error", error=str(exc))
        bridge_state.bridge = None
        bridge_state.adapter = None


# -- Pydantic schemas ------------------------------------------------------------
class BLEDeviceOut(BaseModel):
    address: str
    name: str | None = None
    rssi: int | None = None


class BLEScanResponse(BaseModel):
    devices: list[BLEDeviceOut]
    scan_duration_sec: float


class BLEConnectRequest(BaseModel):
    address: str
    device_model: str = "muse_s_gen1"


class BLEConnectResponse(BaseModel):
    ok: bool
    source: str = ""
    message: str


class BLEStatusResponse(BaseModel):
    running: bool
    connected: bool
    address: str | None = None
    device_model: str = ""


# -- Endpoints -------------------------------------------------------------------


@router.get("/ble/scan", response_model=BLEScanResponse)
async def ble_scan(duration: float = DEFAULT_SCAN_SEC) -> BLEScanResponse:
    """
    Discover nearby BLE devices and return those that look like Muse headbands.

    Uses the detection_callback form of BleakScanner so AdvertisementData
    (including rssi and service_uuids) is available without accessing the
    deprecated BLEDevice.metadata / BLEDevice.rssi attributes.
    """
    if not BLEAK_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="bleak is not installed on this server. Install with: pip install bleak",
        )

    duration = max(2.0, min(duration, 30.0))
    log.info("ble_scan_start", duration=duration)

    # Collect (device, advertisement_data) pairs via detection_callback so we
    # have access to AdvertisementData.rssi and .service_uuids without touching
    # the deprecated BLEDevice.metadata / BLEDevice.rssi properties.
    seen: dict[str, tuple] = {}  # address -> (BLEDevice, AdvertisementData)

    def _on_detection(device, advertisement_data) -> None:
        seen[device.address.upper()] = (device, advertisement_data)

    try:
        scanner = BleakScanner(detection_callback=_on_detection)
        async with scanner:
            await asyncio.sleep(duration)
    except Exception as exc:
        log.error("ble_scan_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"BLE scan failed: {exc}") from exc

    all_devices = list(seen.values())

    muse_devices = [
        BLEDeviceOut(
            address=dev.address,
            name=dev.name or None,
            rssi=adv.rssi,
        )
        for dev, adv in all_devices
        if _is_muse_adv(dev, adv)
    ]

    # If no Muse-named device found, return ALL discovered devices so the
    # user can manually identify their headband by address.
    if not muse_devices:
        muse_devices = [
            BLEDeviceOut(
                address=dev.address,
                name=dev.name or None,
                rssi=adv.rssi,
            )
            for dev, adv in all_devices
        ]

    log.info("ble_scan_done", total=len(all_devices), muse=len(muse_devices))
    return BLEScanResponse(devices=muse_devices, scan_duration_sec=duration)


@router.post("/ble/connect", response_model=BLEConnectResponse)
async def ble_connect(req: BLEConnectRequest) -> BLEConnectResponse:
    """
    Start a BLEBridge for the given device address.

    Stops any existing bridge first.  The bridge supervisor handles
    automatic reconnection on link drop.
    """
    if not BLEAK_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="bleak is not installed. Install with: pip install bleak",
        )

    await _stop_existing_bridge()

    try:
        from neurolink.adapter_factory import create_adapter
        from neurolink.ble_bridge import BLEBridge
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Adapter factory not available: {exc}",
        ) from exc

    try:
        adapter = create_adapter(
            adapter_type="ble",
            device_model=req.device_model,
            address=req.address,
        )
        bridge = BLEBridge(adapter)
        await bridge.start()

        bridge_state.bridge = bridge
        bridge_state.adapter = adapter
        bridge_state.address = req.address
        bridge_state.device_model = req.device_model

        await asyncio.sleep(0.5)

        connected = getattr(adapter, "is_connected", False)
        log.info("ble_connect_ok", address=req.address, model=req.device_model, connected=connected)

        return BLEConnectResponse(
            ok=True,
            source=f"ble:{req.device_model}",
            message=f"Bridge started for {req.address}. {'Connected.' if connected else 'Connecting...'}",
        )

    except Exception as exc:
        log.error("ble_connect_error", address=req.address, error=str(exc))
        await _stop_existing_bridge()
        return BLEConnectResponse(
            ok=False,
            source="",
            message=f"Failed to start bridge: {exc}",
        )


@router.post("/ble/disconnect", response_model=BLEConnectResponse)
async def ble_disconnect() -> BLEConnectResponse:
    """Stop the running BLEBridge and disconnect the adapter."""
    if bridge_state.bridge is None:
        return BLEConnectResponse(ok=True, source="", message="No bridge was running.")

    addr = bridge_state.address
    await _stop_existing_bridge()
    log.info("ble_disconnect_ok", address=addr)
    return BLEConnectResponse(ok=True, source="", message=f"Disconnected from {addr}.")


@router.get("/ble/status", response_model=BLEStatusResponse)
async def ble_status() -> BLEStatusResponse:
    """Return current BLEBridge and adapter connection state."""
    running = bridge_state.bridge is not None
    connected = (
        getattr(bridge_state.adapter, "is_connected", False)
        if bridge_state.adapter is not None
        else False
    )
    return BLEStatusResponse(
        running=running,
        connected=connected,
        address=bridge_state.address,
        device_model=bridge_state.device_model,
    )
