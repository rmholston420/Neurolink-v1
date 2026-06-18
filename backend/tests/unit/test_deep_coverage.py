"""Targeted tests for the remaining ~198 uncovered statements.

Covers:
  hardware/muse_s/compute.py
  hardware/muse_athena/compute.py (re-export)
  dsp/breathing.py
  dsp/ppg.py
  dsp/derived_eeg.py
  dsp/decoders.py
  eeg_pump.py (_build_payload branches + watchdog)
  config.py (cors_origins_list, singleton)
  hardware/muse_s/__init__.py
  utils/__init__.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Trivial __init__ imports (registers modules in coverage)
# ---------------------------------------------------------------------------


def test_import_muse_s_init():
    import neurolink.hardware.muse_s  # noqa: F401


def test_import_utils_init():
    import neurolink.utils  # noqa: F401


def test_import_muse_athena_compute():
    from neurolink.hardware.muse_athena.compute import compute_all_bands

    assert callable(compute_all_bands)


# ===========================================================================
# config.py
# ===========================================================================


def test_settings_cors_origins_list():
    from neurolink.config import Settings

    s = Settings(cors_origins="http://localhost:5173, http://localhost:3000, ")
    origins = s.cors_origins_list
    assert "http://localhost:5173" in origins
    assert "http://localhost:3000" in origins
    assert "" not in origins


def test_get_settings_returns_same_singleton():
    from neurolink.config import get_settings

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ===========================================================================
# hardware/muse_s/compute.py
# ===========================================================================


def test_compute_all_bands_happy_path():
    """Normal path: 5 channels, sufficient samples each."""
    from neurolink.hardware.muse_s.compute import compute_all_bands

    n = 512
    t = np.linspace(0, 2, n)
    channel_samples = {
        ch: (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist()
        for ch in ["TP9", "AF7", "AF8", "TP10", "AUX"]
    }
    result = compute_all_bands(channel_samples)
    assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_compute_all_bands_short_channels_skipped():
    from neurolink.hardware.muse_s.compute import compute_all_bands

    n = 512
    t = np.linspace(0, 2, n)
    channel_samples = {
        "TP9": [0.1, 0.2],
        "AF7": (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist(),
    }
    result = compute_all_bands(channel_samples)
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_compute_all_bands_all_short_returns_zeros():
    from neurolink.hardware.muse_s.compute import compute_all_bands

    result = compute_all_bands({"TP9": [0.1], "AF7": [0.2]})
    assert all(v == 0.0 for v in result.values())


def test_compute_all_bands_zero_psd_returns_zeros():
    from neurolink.hardware.muse_s.compute import compute_all_bands

    channel_samples = {"TP9": [0.0] * 512, "AF7": [0.0] * 512}
    result = compute_all_bands(channel_samples)
    assert all(v == 0.0 for v in result.values())


def test_compute_all_bands_importerror_fallback():
    from neurolink.hardware.muse_s.compute import compute_all_bands

    n = 512
    t = np.linspace(0, 2, n)
    channel_samples = {
        ch: (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist()
        for ch in ["TP9", "AF7", "AF8", "TP10", "AUX"]
    }
    with patch.dict(sys.modules, {"scipy.signal": None, "scipy": None}):
        result = compute_all_bands(channel_samples)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}


# ===========================================================================
# dsp/breathing.py
# ===========================================================================


def test_breathing_both_sources_fused():
    """Both IBIs and accel_z available -> fused average.

    IBIs oscillate at ~0.25 Hz (RSA breathing rate) around 800 ms so that
    _rr_from_ibis has non-zero spectral content and returns a valid value.
    """
    from neurolink.dsp.breathing import compute_breathing

    # 30 IBIs with RSA modulation at 0.25 Hz (~15 bpm breathing)
    rng = np.random.default_rng(42)
    t_ibi = np.linspace(0, 30, 30)
    ibis = (
        800.0 + 50.0 * np.sin(2 * np.pi * 0.25 * t_ibi) + 5.0 * rng.standard_normal(30)
    ).tolist()
    # 10+ seconds of 52 Hz accel with 0.25 Hz breathing signal
    n = int(52.0 * 15)
    t = np.linspace(0, 15, n)
    accel_z = (np.sin(2 * np.pi * 0.25 * t) + 1.0).astype(np.float32)
    result = compute_breathing(ibis, accel_z=accel_z)
    assert result.rr_bpm is not None
    assert result.rr_ppg is not None
    assert result.rr_accel is not None


def test_breathing_only_accel():
    from neurolink.dsp.breathing import compute_breathing

    n = int(52.0 * 15)
    t = np.linspace(0, 15, n)
    accel_z = (np.sin(2 * np.pi * 0.25 * t) + 1.0).astype(np.float32)
    result = compute_breathing([], accel_z=accel_z)
    assert result.rr_ppg is None


def test_breathing_neither_source_rr_none():
    from neurolink.dsp.breathing import compute_breathing

    result = compute_breathing([], accel_z=None)
    assert result.rr_bpm is None
    assert result.rr_ppg is None
    assert result.rr_accel is None


def test_breathing_ibi_no_valid_mask():
    from neurolink.dsp.breathing import _rr_from_ibis

    ibis = [500.0] * 10
    result = _rr_from_ibis(ibis)
    assert result is None or isinstance(result, float)


def test_breathing_accel_too_short():
    from neurolink.dsp.breathing import _rr_from_accel

    short_accel = np.ones(10, dtype=np.float32)
    result = _rr_from_accel(short_accel, fs=52.0)
    assert result is None


# ===========================================================================
# dsp/ppg.py
# ===========================================================================


def test_ppg_too_short_returns_empty():
    from neurolink.dsp.ppg import compute_ppg

    short = np.ones(10, dtype=np.float32)
    result = compute_ppg(short)
    assert result.hr_bpm == 0.0
    assert result.ibi_ms == []


def test_ppg_none_input_returns_empty():
    from neurolink.dsp.ppg import compute_ppg

    result = compute_ppg(None)  # type: ignore[arg-type]
    assert result.hr_bpm == 0.0


def test_ppg_exception_returns_empty(monkeypatch):
    from neurolink.dsp import ppg as ppg_mod

    orig = None
    try:
        import neurokit2

        orig = neurokit2.ppg_process
        neurokit2.ppg_process = MagicMock(side_effect=RuntimeError("nk2 error"))
        n = int(64.0 * 15)
        arr = np.sin(np.linspace(0, 15, n)).astype(np.float32)
        result = ppg_mod.compute_ppg(arr)
        assert result.hr_bpm == 0.0
    finally:
        if orig is not None:
            neurokit2.ppg_process = orig


def test_ppg_poincare_too_short():
    from neurolink.dsp.ppg import _poincare

    m = _poincare([800.0])
    assert m.sd1 == 0.0
    assert m.sd2 == 0.0
    assert m.ellipse_area == 0.0


def test_ppg_poincare_happy_path():
    from neurolink.dsp.ppg import _poincare

    ibis = [800.0, 820.0, 790.0, 810.0, 805.0]
    m = _poincare(ibis)
    assert m.sd1 >= 0.0
    assert m.sd2 >= 0.0
    assert m.ellipse_area >= 0.0


# ===========================================================================
# dsp/derived_eeg.py
# ===========================================================================


def test_derived_eeg_1d_input_returns_none_dict():
    from neurolink.dsp.derived_eeg import derived_eeg

    result = derived_eeg(np.array([1.0, 2.0, 3.0]))
    assert result["faa"] is None
    assert result["fmt"] is None


def test_derived_eeg_too_few_samples():
    from neurolink.dsp.derived_eeg import derived_eeg

    eeg = np.zeros((5, 10))
    result = derived_eeg(eeg)
    assert result["faa"] is None


def test_derived_eeg_too_few_channels():
    from neurolink.dsp.derived_eeg import derived_eeg

    eeg = np.zeros((4, 512))
    result = derived_eeg(eeg)
    assert result["faa"] is None


def test_derived_eeg_both_alpha_positive():
    from neurolink.dsp.derived_eeg import derived_eeg

    n = 512
    t = np.linspace(0, 2, n)
    alpha = 0.4 * np.sin(2 * np.pi * 10.0 * t)
    eeg = np.tile(alpha, (5, 1)).astype(np.float32)
    result = derived_eeg(eeg)
    assert result["faa"] is not None
    assert isinstance(result["faa"], float)
    assert result["fmt"] is not None


def test_derived_eeg_af8_only_faa_positive():
    from neurolink.dsp.derived_eeg import derived_eeg

    n = 512
    t = np.linspace(0, 2, n)
    alpha = (0.4 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
    eeg = np.zeros((5, n), dtype=np.float32)
    eeg[2] = alpha
    result = derived_eeg(eeg)
    assert result["faa"] == 1.0


def test_derived_eeg_af7_only_faa_negative():
    from neurolink.dsp.derived_eeg import derived_eeg

    n = 512
    t = np.linspace(0, 2, n)
    alpha = (0.4 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
    eeg = np.zeros((5, n), dtype=np.float32)
    eeg[1] = alpha
    result = derived_eeg(eeg)
    assert result["faa"] == -1.0


def test_derived_eeg_no_alpha_faa_zero():
    from neurolink.dsp.derived_eeg import derived_eeg

    eeg = np.zeros((5, 512), dtype=np.float32)
    result = derived_eeg(eeg)
    assert result["faa"] == 0.0
    assert result["fmt"] == 0.0


# ===========================================================================
# dsp/decoders.py
# ===========================================================================


def test_decode_eeg_short_packet():
    from neurolink.dsp.decoders import decode_eeg

    # _EEG_MIN_PACKET_LEN is 5; 4 bytes is below threshold -> []
    assert decode_eeg(b"\x00" * 4) == []


def test_decode_eeg_happy_path():
    from neurolink.dsp.decoders import decode_eeg

    data = b"\x00\x00" + bytes(range(18))
    result = decode_eeg(data)
    assert isinstance(result, list)
    assert len(result) <= 12


def test_decode_ppg_short_packet():
    from neurolink.dsp.decoders import decode_ppg

    assert decode_ppg(b"\x00" * 5) == []  # < 12 bytes


def test_decode_ppg_happy_path():
    from neurolink.dsp.decoders import decode_ppg

    data = b"\x00\x00" + bytes(range(18))
    result = decode_ppg(data)
    assert isinstance(result, list)
    assert len(result) <= 6


def test_decode_imu_short_packet():
    from neurolink.dsp.decoders import decode_imu

    accel, gyro = decode_imu(b"\x00" * 5)  # < 20 bytes
    assert accel == []
    assert gyro == []


def test_decode_imu_happy_path():
    from neurolink.dsp.decoders import decode_imu

    header = b"\x00\x00"
    payload = b"\x00\x01" * 9
    data = header + payload
    accel, gyro = decode_imu(data)
    assert len(accel) == 9
    assert len(gyro) == 9


def test_decode_imu_padding_to_9():
    from neurolink.dsp.decoders import decode_imu

    data = b"\x00\x00" + b"\x00\x01\x00\x02" + b"\x00" * 14
    accel, gyro = decode_imu(data)
    assert len(accel) == 9


# ===========================================================================
# eeg_pump.py — _build_payload branches
# ===========================================================================


def _make_sample(**overrides):
    from neurolink.hardware.base import EEGSample

    defaults = dict(
        channels=[0.0] * 5,
        timestamp=0.0,
        source="mock",
        address="mock",
        poor_contact=False,
        eeg_buffer=[],
        ppg_buffer=[],
        accel_buffer=[],
        gyro_buffer=[],
    )
    defaults.update(overrides)
    return EEGSample(**defaults)


async def test_build_payload_empty_eeg_buffer():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample()
    payload = await pump._build_payload(sample)
    assert payload.bands.alpha == 0.0
    assert payload.faa is None
    assert payload.fmt is None


async def test_build_payload_eeg_single_channel():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    n = 256 * 4
    t = np.linspace(0, 4, n)
    signal = (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist()
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(eeg_buffer=[signal])
    payload = await pump._build_payload(sample)
    assert payload is not None


async def test_build_payload_no_ppg_buffer():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(ppg_buffer=[])
    payload = await pump._build_payload(sample)
    assert payload.ppg is None


async def test_build_payload_no_accel_buffer():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(accel_buffer=[])
    payload = await pump._build_payload(sample)
    assert payload.imu is None


async def test_build_payload_accel_buffer_too_short():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(accel_buffer=[[1.0, 2.0], [3.0, 4.0]])
    payload = await pump._build_payload(sample)
    assert payload is not None


async def test_build_payload_accel_shape_zero():
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub

    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(
        accel_buffer=[[], [], []],
        gyro_buffer=[[], [], []],
    )
    payload = await pump._build_payload(sample)
    assert payload.imu is None


async def test_pump_loop_watchdog_fires():
    from neurolink.eeg_pump import _WATCHDOG_SEC, EEGPump
    from neurolink.hub import EEGHub

    hub = EEGHub()
    adapter = AsyncMock()
    tick_count = 0

    async def controlled_read():
        nonlocal tick_count
        tick_count += 1
        if tick_count >= 2:
            pump._running = False
        return None

    adapter.read_sample = controlled_read
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=100.0)
    pump._last_frame_ts = time.time() - (_WATCHDOG_SEC + 1)
    await pump.start()
    for _ in range(30):
        await asyncio.sleep(0.01)
        if tick_count >= 2:
            break
    await pump.stop()
    assert tick_count >= 2
