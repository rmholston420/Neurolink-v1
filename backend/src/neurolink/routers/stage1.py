"""Stage 1 — Online filter REST endpoints.

Mounted at /api/v1/stage1 by main.py.

Endpoints
---------
GET  /config          Return the currently active FilterConfig as JSON.
POST /config          Replace the active config (rebuilds filter chain).
POST /config/reset    Restore defaults for the detected / supplied region.
GET  /diagnostics     Per-channel Welch PSD snapshot from the latest
                      EEGSample so the front-end can verify notch depth.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import structlog
from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel, Field

from neurolink.dsp.online_filter import FilterConfig, FilterChainRegistry, get_registry

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/stage1", tags=["Stage1"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FilterConfigSchema(BaseModel):
    hz_highpass: float | None = Field(0.5, description="High-pass cut-off Hz; null to skip")
    hz_notch_freqs: list[float] = Field(
        [50.0, 100.0], description="Notch centre frequencies Hz"
    )
    hz_lowpass: float | None = Field(45.0, description="Low-pass cut-off Hz; null to skip")
    notch_bw_hz: float = Field(2.0, description="Full bandwidth of each notch Hz")
    fs: float = Field(256.0, description="Sampling rate Hz")
    filter_order: int = Field(128, ge=4, le=512, description="FIR filter order (even)")


class FilterDiagnosticsSchema(BaseModel):
    fs: float
    freqs_hz: list[float]
    psd_by_channel: dict[str, list[float]]
    active_config: FilterConfigSchema
    note: str


# ---------------------------------------------------------------------------
# Dependency: registry
# ---------------------------------------------------------------------------

def _get_registry() -> FilterChainRegistry:
    return get_registry()


RegistryDep = Annotated[FilterChainRegistry, Depends(_get_registry)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_schema(cfg: FilterConfig) -> FilterConfigSchema:
    return FilterConfigSchema(
        hz_highpass=cfg.hz_highpass,
        hz_notch_freqs=cfg.hz_notch_freqs,
        hz_lowpass=cfg.hz_lowpass,
        notch_bw_hz=cfg.notch_bw_hz,
        fs=cfg.fs,
        filter_order=cfg.filter_order,
    )


def _schema_to_config(schema: FilterConfigSchema) -> FilterConfig:
    return FilterConfig(
        hz_highpass=schema.hz_highpass,
        hz_notch_freqs=schema.hz_notch_freqs,
        hz_lowpass=schema.hz_lowpass,
        notch_bw_hz=schema.notch_bw_hz,
        fs=schema.fs,
        filter_order=schema.filter_order,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/config", response_model=FilterConfigSchema)
async def get_filter_config(registry: RegistryDep) -> FilterConfigSchema:
    """Return the currently active filter configuration."""
    return _config_to_schema(registry.get_config())


@router.post("/config", response_model=FilterConfigSchema)
async def set_filter_config(
    registry: RegistryDep,
    body: Annotated[FilterConfigSchema, Body()],
) -> FilterConfigSchema:
    """Replace the active filter configuration.

    The new FIR kernels are built synchronously before the response is
    returned.  The next EEGPump tick will use the updated chain.
    """
    cfg = _schema_to_config(body)
    registry.set_config(cfg)
    log.info("stage1_config_set_via_api", config=body.model_dump())
    return _config_to_schema(registry.get_config())


@router.post("/config/reset", response_model=FilterConfigSchema)
async def reset_filter_config(
    registry: RegistryDep,
    line_freq: float = 50.0,
    fs: float = 256.0,
) -> FilterConfigSchema:
    """Restore default filter settings for the given region.

    Args:
        line_freq: 50 (EU/Asia/default) or 60 (Americas)
        fs:        EEG sampling rate
    """
    registry.pre_warm(line_freq=line_freq, fs=fs)
    log.info("stage1_config_reset", line_freq=line_freq, fs=fs)
    return _config_to_schema(registry.get_config())


@router.get("/diagnostics", response_model=FilterDiagnosticsSchema)
async def get_filter_diagnostics(
    request: Request,
    registry: RegistryDep,
) -> FilterDiagnosticsSchema:
    """Return a per-channel Welch PSD snapshot.

    Reads the latest EEGSample from hub.latest_sample (if available)
    and computes PSD so the front-end can verify that the notch filter
    is attenuating line noise.  Falls back to a zero-signal response
    when no sample is available yet.
    """
    from scipy import signal as sp_signal  # local to avoid startup overhead

    cfg = registry.get_config()
    fs = cfg.fs
    nperseg = 256

    psd_by_channel: dict[str, list[float]] = {}
    freqs_out: list[float] = []
    note = ""

    try:
        from neurolink.dependencies import get_neurolink_service

        service = get_neurolink_service()
        hub = service.get_hub()
        sample = hub.get_latest_sample() if hub else None

        if sample and sample.eeg_buffer:
            _min_len = min(len(b) for b in sample.eeg_buffer)
            if _min_len >= nperseg:
                eeg_arr = np.array(
                    [b[:_min_len] for b in sample.eeg_buffer], dtype=np.float64
                )
                # Apply current filter chain before computing PSD
                eeg_filt = registry.apply(eeg_arr.astype(np.float32)).astype(np.float64)
                ch_names = ["TP9", "AF7", "AF8", "TP10", "AUX"]
                for idx, ch in enumerate(ch_names):
                    if idx < eeg_filt.shape[0]:
                        freqs, psd = sp_signal.welch(
                            eeg_filt[idx], fs=fs, nperseg=min(nperseg, _min_len)
                        )
                        psd_by_channel[ch] = [round(float(p), 6) for p in psd]
                freqs_out = [round(float(f), 3) for f in freqs]
                note = "PSD computed from latest filtered EEGSample"
            else:
                note = f"Buffer too short ({_min_len} samples); need >= {nperseg}"
        else:
            note = "No EEGSample available yet"
    except Exception as exc:
        log.warning("stage1_diagnostics_error", error=str(exc))
        note = f"Diagnostics unavailable: {exc}"

    if not psd_by_channel:
        # Return a zero PSD spanning 0-Nyquist so the schema is always valid
        freqs_out = [round(f, 3) for f in np.linspace(0, fs / 2, nperseg // 2 + 1).tolist()]
        for ch in ["TP9", "AF7", "AF8", "TP10", "AUX"]:
            psd_by_channel[ch] = [0.0] * len(freqs_out)

    return FilterDiagnosticsSchema(
        fs=fs,
        freqs_hz=freqs_out,
        psd_by_channel=psd_by_channel,
        active_config=_config_to_schema(cfg),
        note=note,
    )
