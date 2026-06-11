"""Neurolink FastAPI application factory and lifespan."""
from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neurolink.config import get_settings
from neurolink.db.engine import create_tables, dispose_engine
from neurolink.exceptions import (
    AdapterNotConnectedError,
    BLETimeoutError,
    CalibrationBusyError,
    NeurolinkError,
    NoEEGDataError,
    UnknownDeviceError,
)
from neurolink.models.eeg import HealthResponse
from neurolink.routers import calibration, eeg_gate, neurolink as neurolink_router

log = structlog.get_logger(__name__)


def _configure_logging(json_logs: bool) -> None:
    import logging
    import sys

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    _configure_logging(settings.log_json)
    log.info("neurolink_starting", adapter_type=settings.adapter_type)
    await create_tables()
    yield
    await dispose_engine()
    log.info("neurolink_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Neurolink",
        version="1.0.0",
        description="EEG-based meditation and contemplative practice app",
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
    async def not_connected_handler(
        request: Request, exc: AdapterNotConnectedError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "code": "NOT_CONNECTED"},
        )

    @app.exception_handler(CalibrationBusyError)
    async def calibration_busy_handler(
        request: Request, exc: CalibrationBusyError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "code": "CALIBRATION_BUSY"},
        )

    @app.exception_handler(BLETimeoutError)
    async def ble_timeout_handler(
        request: Request, exc: BLETimeoutError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"detail": str(exc), "code": "BLE_TIMEOUT"},
        )

    @app.exception_handler(UnknownDeviceError)
    async def unknown_device_handler(
        request: Request, exc: UnknownDeviceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "code": "UNKNOWN_DEVICE"},
        )

    @app.exception_handler(NoEEGDataError)
    async def no_data_handler(
        request: Request, exc: NoEEGDataError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=202,
            content={"detail": str(exc), "code": "NO_DATA"},
        )

    # Routers
    app.include_router(neurolink_router.router)
    app.include_router(calibration.router)
    app.include_router(eeg_gate.router)

    # Health endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Report adapter status, hub frame count, Redis and DB reachability."""
        from neurolink.hub import get_hub
        from neurolink.dependencies import get_neurolink_service

        hub = get_hub()
        state = hub.get_state()

        # Redis check
        redis_status = "disabled"
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(settings.redis_url, socket_connect_timeout=1)
            await r.ping()
            await r.aclose()
            redis_status = "connected"
        except Exception:
            redis_status = "error"

        # DB check
        db_status = "error"
        try:
            from neurolink.db.engine import get_engine
            engine = get_engine()
            async with engine.connect() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception:
            db_status = "error"

        # Adapter status
        _service = get_neurolink_service(hub=hub)
        adapter_connected = _service.adapter_connected
        adapter_type_str = _service.adapter_type

        return HealthResponse(
            status="ok" if db_status == "connected" else "degraded",
            adapter_type=adapter_type_str,
            adapter_connected=adapter_connected,
            hub_frame_count=state.frame_count,
            redis=redis_status,
            db=db_status,
        )

    return app


# Default app instance
app = create_app()
