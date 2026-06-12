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

    if settings.db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)), exist_ok=True)

    from neurolink.db.engine import create_tables, dispose_engine, get_session_factory
    await create_tables()
    log.info("neurolink_db_initialized", db_path=settings.db_path)

    # ── Stage 0 Guard ────────────────────────────────────────────────────
    from neurolink.stage0 import Stage0Guard
    electrode_type = getattr(settings, "electrode_type", "dry")
    stage0_guard = Stage0Guard(electrode_type=electrode_type)
    app.state.stage0_guard = stage0_guard
    log.info("stage0_guard_initialised", electrode_type=electrode_type)

    # ── Stage 1 — Online FIR filter chain ──────────────────────────────
    from neurolink.dsp.online_filter import get_registry as get_filter_registry
    region = getattr(settings, "region", "EU").upper()
    line_freq = 60.0 if region in {"US", "CA", "MX", "JP"} else 50.0
    filter_registry = get_filter_registry()
    filter_registry.pre_warm(line_freq=line_freq, fs=256.0)
    app.state.filter_registry = filter_registry
    log.info("stage1_filter_chain_prewarmed", region=region, line_freq=line_freq)

    # ── Stage 2 — Bad channel detector ─────────────────────────────────
    from neurolink.dsp.bad_channels import BadChannelDetector
    bad_channel_detector = BadChannelDetector()
    app.state.bad_channel_detector = bad_channel_detector
    log.info("stage2_bad_channel_detector_initialised")

    # ── Stage 3 — Artifact gate ─────────────────────────────────────────
    from neurolink.dsp.artifact_gate import ArtifactGate
    artifact_gate = ArtifactGate()
    app.state.artifact_gate = artifact_gate
    log.info("stage3_artifact_gate_initialised")

    # ── Stage 3b — Multi-type artifact detector ─────────────────────────
    from neurolink.dsp.artifact_detector import ArtifactDetector
    artifact_detector = ArtifactDetector(line_freq_hz=line_freq)
    app.state.artifact_detector = artifact_detector
    log.info("stage3b_artifact_detector_initialised", line_freq_hz=line_freq)

    # Inject DB session factory into service
    from neurolink.dependencies import get_neurolink_service
    service = get_neurolink_service()
    service.set_db_session_factory(get_session_factory())

    # Mock auto-connect + Stage 0 bypass
    if settings.adapter_type == "mock":
        stage0_guard.environment.acknowledge_all()
        try:
            await service.connect(adapter_type="mock", device_model="mock")
            log.info("neurolink_mock_auto_connected")
        except Exception as exc:
            log.warning("neurolink_mock_auto_connect_failed", error=str(exc))

    yield

    # Shutdown
    log.info("neurolink_shutting_down")
    try:
        from neurolink.routers.ble import bridge_state
        if bridge_state.bridge is not None:
            await bridge_state.bridge.stop()
    except Exception:
        pass
    try:
        await service.disconnect()
    except Exception:
        pass
    await dispose_engine()
    log.info("neurolink_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Neurolink",
        description="EEG-based meditation and contemplative practice API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    from neurolink.routers.ble import router as ble_router
    from neurolink.routers.calibration import router as calibration_router
    from neurolink.routers.eeg_gate import router as eeg_gate_router
    from neurolink.routers.filters import router as filters_router
    from neurolink.routers.health import router as health_router
    from neurolink.routers.neurolink import router as neurolink_router
    from neurolink.routers.stage0 import router as stage0_router
    from neurolink.routers.stage1 import router as stage1_router
    from neurolink.routers.stage2 import router as stage2_router
    from neurolink.routers.stage3 import router as stage3_router
    from neurolink.routers.stage3b import router as stage3b_router

    app.include_router(health_router)
    app.include_router(neurolink_router, prefix="/api/v1")
    app.include_router(calibration_router, prefix="/api/v1")
    app.include_router(eeg_gate_router, prefix="/api/v1")
    app.include_router(filters_router, prefix="/api/v1")
    app.include_router(ble_router, prefix="/api/v1/neurolink")
    app.include_router(stage0_router, prefix="/api/v1")
    app.include_router(stage1_router, prefix="/api/v1")
    app.include_router(stage2_router, prefix="/api/v1")
    app.include_router(stage3_router, prefix="/api/v1")
    app.include_router(stage3b_router, prefix="/api/v1")

    log.info("neurolink_app_created")
    return app


app = create_app()
