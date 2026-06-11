"""Neurolink routers package.

Exports all FastAPI routers so they can be imported with:
    from neurolink.routers import neurolink_router, health_router, ...
"""

from __future__ import annotations

from neurolink.routers.calibration import router as calibration_router
from neurolink.routers.eeg_gate import router as eeg_gate_router
from neurolink.routers.health import router as health_router
from neurolink.routers.neurolink import router as neurolink_router

__all__ = [
    "calibration_router",
    "eeg_gate_router",
    "health_router",
    "neurolink_router",
]
