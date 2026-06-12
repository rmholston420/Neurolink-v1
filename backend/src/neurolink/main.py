"""Neurolink FastAPI application factory and lifespan.

Entry point: uvicorn neurolink.main:app
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neurolink.config import get_settings
from neurolink.exceptions import (
    AdapterAlreadyConnectedError,
    AdapterNotConnectedError,
    NeurolinkError,
)
from neurolink.logging_config import configure_logging

log = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown logic."""
    settings = get_settings()
    configure_logging(log_json=settings.log_json, log_level=settings.log_level)
    log.info(
        "neurolink_starting",
        adapter_type=settings.adapter_type,
        device_model=settings.device_model,
    )

    # Create data directory for SQLite
    if settings.db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)), exist_ok=True)

    # Initialize database
    from neurolink.db.engine import create_tables, dispose_engine, get_session_factory

    await create_tables()
    log.info("neurolink_db_initialized", db_path=settings.db_path)

    # ── Stage 0 Guard ────────────────────────────────────────────────────
    # Electrode type defaults to 'dry' (Muse S); override via settings or
    # a future POST /api/v1/stage0/configure endpoint.
    from neurolink.stage0 import Stage0Guard

    electrode_type = getattr(settings, "electrode_type", "dry")
    stage0_guard = Stage0Guard(electrode_type=electrode_type)
    app.state.stage0_guard = stage0_guard
    log.info("stage0_guard_initialised", electrode_type=electrode_type)

    # Inject DB session factory into service
    from neurolink.dependencies import get_neurolink_service

    service = get_neurolink_service()
    service.set_db_session_factory(get_session_factory())

    # Optional: auto-connect if adapter_type is set to mock.
    # In mock mode acknowledge all environment steps so the demo/test
    # pipeline flows without the setup wizard.
    if settings.adapter_type == "mock":
        stage0_guard.environment.acknowledge_all()
        try:
            await service.connect(
                adapter_type="mock",
                device_model="mock",
            )
            log.info("neurolink_mock_auto_connected")
        except Exception as exc:
            log.warning("neurolink_mock_auto_connect_failed", error=str(exc))

    yield

    # Shutdown — stop any running BLE bridge (Path B)
    log.info("neurolink_shutting_down")
    try:
        from neurolink.routers.ble import bridge_state

        if bridge_state.bridge is not None:
            await bridge_state.bridge.stop()
            log.info("neurolink_ble_bridge_stopped_on_shutdown")
    except Exception:
        pass
    try:
        await service.disconnect()
    except Exception:
        pass
    await dispose_engine()
    log.info("neurolink_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Neurolink",
        description="EEG-based meditation and contemplative practice API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(AdapterNotConnectedError)
    async def adapter_not_connected_handler(
        request: Request, exc: AdapterNotConnectedError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(AdapterAlreadyConnectedError)
    async def adapter_already_connected_handler(
        request: Request, exc: AdapterAlreadyConnectedError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NeurolinkError)
    async def neurolink_error_handler(request: Request, exc: NeurolinkError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # Include routers
    from neurolink.routers.ble import router as ble_router
    from neurolink.routers.calibration import router as calibration_router
    from neurolink.routers.eeg_gate import router as eeg_gate_router
    from neurolink.routers.health import router as health_router
    from neurolink.routers.neurolink import router as neurolink_router
    from neurolink.routers.stage0 import router as stage0_router

    app.include_router(health_router)
    app.include_router(neurolink_router, prefix="/api/v1")
    app.include_router(calibration_router, prefix="/api/v1")
    app.include_router(eeg_gate_router, prefix="/api/v1")
    app.include_router(ble_router, prefix="/api/v1/neurolink")
    app.include_router(stage0_router, prefix="/api/v1")

    log.info("neurolink_app_created")
    return app


# Module-level app instance for uvicorn
app = create_app()
