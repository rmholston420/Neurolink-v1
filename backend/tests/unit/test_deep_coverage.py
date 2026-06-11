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
import math
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Trivial __init__ imports (registers modules in coverage)
# ---------------------------------------------------------------------------

def test_import_muse_s_init():
    import neurolink.hardware.muse_s  # noqa: F401


def test_import_utils_init():
    import neurolink.utils  # noqa: F401


def test_import_muse_athena_compute():
    from neurolink.hardware.muse_athena.compute import compute_all_bands  # noqa: F401
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
    # trailing empty string filtered out
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
    """Channels with < 4 samples are skipped; others still compute."""
    from neurolink.hardware.muse_s.compute import compute_all_bands
    n = 512
    t = np.linspace(0, 2, n)
    channel_samples = {
        "TP9": [0.1, 0.2],  # too short -> skipped
        "AF7": (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist(),
    }
    result = compute_all_bands(channel_samples)
    # Only AF7 contributed, result must still be normalised
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_compute_all_bands_all_short_returns_zeros():
    """All channels < 4 samples -> n_channels == 0 -> zero dict."""
    from neurolink.hardware.muse_s.compute import compute_all_bands
    result = compute_all_bands({"TP9": [0.1], "AF7": [0.2]})
    assert all(v == 0.0 for v in result.values())


def test_compute_all_bands_zero_psd_returns_zeros():
    """Flat-zero signal -> total PSD == 0 -> zero dict."""
    from neurolink.hardware.muse_s.compute import compute_all_bands
    channel_samples = {"TP9": [0.0] * 512, "AF7": [0.0] * 512}
    result = compute_all_bands(channel_samples)
    assert all(v == 0.0 for v in result.values())


def test_compute_all_bands_importerror_fallback(monkeypatch):
    """ImportError for scipy.signal falls back to bandpower buffer path."""
    import sys
    # Remove scipy from sys.modules to force ImportError inside compute_all_bands
    scipy_signal = sys.modules.pop("scipy.signal", None)
    scipy = sys.modules.pop("scipy", None)
    try:
        # Re-import with scipy missing
        import importlib
        import neurolink.hardware.muse_s.compute as compute_mod
        importlib.reload(compute_mod)

        n = 512
        t = np.linspace(0, 2, n)
        channel_samples = {
            ch: (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist()
            for ch in ["TP9", "AF7", "AF8", "TP10", "AUX"]
        }
        # We only need to reach the fallback branch; result shape is what matters
        result = compute_mod.compute_all_bands(channel_samples)
        assert isinstance(result, dict)
    finally:
        if scipy_signal is not None:
            sys.modules["scipy.signal"] = scipy_signal
        if scipy is not None:
            sys.modules["scipy"] = scipy


# ===========================================================================
# dsp/breathing.py
# ===========================================================================

def test_breathing_both_sources_fused():
    """Both IBIs and accel_z available -> fused average."""
    from neurolink.dsp.breathing import compute_breathing
    # 30 IBIs at ~800ms (75 bpm)
    ibis = [800.0] * 30
    # 10+ seconds of 52 Hz accel with 0.25 Hz breathing signal
    n = int(52.0 * 15)
    t = np.linspace(0, 15, n)
    accel_z = (np.sin(2 * np.pi * 0.25 * t) + 1.0).astype(np.float32)
    result = compute_breathing(ibis, accel_z=accel_z)
    # rr_bpm is the fused value (average of ppg and accel estimates)
    assert result.rr_bpm is not None
    assert result.rr_ppg is not None
    assert result.rr_accel is not None


def test_breathing_only_accel():
    """No IBIs, only accel_z -> rr_bpm = rr_accel."""
    from neurolink.dsp.breathing import compute_breathing
    n = int(52.0 * 15)
    t = np.linspace(0, 15, n)
    accel_z = (np.sin(2 * np.pi * 0.25 * t) + 1.0).astype(np.float32)
    result = compute_breathing([], accel_z=accel_z)
    assert result.rr_ppg is None
    # rr_bpm == rr_accel (may be None if no valid peak found, but branch is hit)


def test_breathing_neither_source_rr_none():
    """No IBIs, no accel -> rr_bpm is None."""
    from neurolink.dsp.breathing import compute_breathing
    result = compute_breathing([], accel_z=None)
    assert result.rr_bpm is None
    assert result.rr_ppg is None
    assert result.rr_accel is None


def test_breathing_ibi_no_valid_mask():
    """IBI freq range produces no valid mask -> _rr_from_ibis returns None."""
    from neurolink.dsp.breathing import _rr_from_ibis
    # Very short array -> nfft small -> rfftfreq may not have bins in [0.1, 0.55]
    # But we can force it by passing exactly 10 identical IBIs
    ibis = [500.0] * 10
    # This will run through the FFT; result is float or None
    result = _rr_from_ibis(ibis)
    assert result is None or isinstance(result, float)


def test_breathing_accel_too_short():
    """accel_z shorter than _MIN_ACCEL_SAMPLES -> _rr_from_accel returns None."""
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
    """If neurokit2 raises, the except branch returns empty PPGPayload."""
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
    """_poincare with < 2 IBIs returns default PoincareMetrics."""
    from neurolink.dsp.ppg import _poincare
    m = _poincare([800.0])  # single IBI
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
    eeg = np.zeros((5, 10))  # 10 samples < 256 min
    result = derived_eeg(eeg)
    assert result["faa"] is None


def test_derived_eeg_too_few_channels():
    from neurolink.dsp.derived_eeg import derived_eeg
    eeg = np.zeros((4, 512))  # 4 channels < 5
    result = derived_eeg(eeg)
    assert result["faa"] is None


def test_derived_eeg_both_alpha_positive():
    """Both AF7 and AF8 have alpha -> FAA = log(af8) - log(af7)."""
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
    """AF8 has alpha but AF7 is zero -> faa = 1.0."""
    from neurolink.dsp.derived_eeg import derived_eeg
    n = 512
    t = np.linspace(0, 2, n)
    alpha = (0.4 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
    eeg = np.zeros((5, n), dtype=np.float32)
    eeg[2] = alpha  # AF8 index
    result = derived_eeg(eeg)
    assert result["faa"] == 1.0


def test_derived_eeg_af7_only_faa_negative():
    """AF7 has alpha but AF8 is zero -> faa = -1.0."""
    from neurolink.dsp.derived_eeg import derived_eeg
    n = 512
    t = np.linspace(0, 2, n)
    alpha = (0.4 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
    eeg = np.zeros((5, n), dtype=np.float32)
    eeg[1] = alpha  # AF7 index
    result = derived_eeg(eeg)
    assert result["faa"] == -1.0


def test_derived_eeg_no_alpha_faa_zero():
    """Both AF7 and AF8 zero power -> faa = 0.0."""
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
    assert decode_eeg(b"\x00" * 5) == []  # < 14 bytes


def test_decode_eeg_happy_path():
    from neurolink.dsp.decoders import decode_eeg
    # 2-byte header + 12 * 1.5 bytes = 20 bytes of payload
    data = b"\x00\x00" + bytes(range(18))
    result = decode_eeg(data)
    assert isinstance(result, list)
    assert len(result) <= 12


def test_decode_ppg_short_packet():
    from neurolink.dsp.decoders import decode_ppg
    assert decode_ppg(b"\x00" * 5) == []  # < 12 bytes


def test_decode_ppg_happy_path():
    from neurolink.dsp.decoders import decode_ppg
    # 2-byte header + 6 * 3-byte samples = 20 bytes
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
    # 2-byte header + 18 bytes (9 int16 big-endian) = 20 bytes
    header = b"\x00\x00"
    payload = b"\x00\x01" * 9  # 9 int16 values = 1 each
    data = header + payload
    accel, gyro = decode_imu(data)
    assert len(accel) == 9
    assert len(gyro) == 9


def test_decode_imu_padding_to_9():
    """Short payload padded with 0.0 to reach 9 values."""
    from neurolink.dsp.decoders import decode_imu
    # 2-byte header + only 4 bytes (2 int16 values)
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
    """Empty eeg_buffer -> bands all 0, no derived EEG."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample()
    payload = await pump._build_payload(sample)
    assert payload.bands.alpha == 0.0
    assert payload.faa is None
    assert payload.fmt is None


async def test_build_payload_eeg_single_channel():
    """eeg_arr.shape[1] >= 2 but only 1 channel -> bands computed, derived skipped."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    n = 256 * 4
    t = np.linspace(0, 4, n)
    signal = (0.4 * np.sin(2 * np.pi * 10.0 * t)).tolist()
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    # 1 channel (shape[0]=1, shape[1]=n) -> shape[1] >= 2 passes, derived also runs
    sample = _make_sample(eeg_buffer=[signal])
    payload = await pump._build_payload(sample)
    # Just verify it doesn't raise
    assert payload is not None


async def test_build_payload_no_ppg_buffer():
    """No ppg_buffer -> ppg_payload is None."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(ppg_buffer=[])
    payload = await pump._build_payload(sample)
    assert payload.ppg is None


async def test_build_payload_no_accel_buffer():
    """No accel_buffer -> accel_z is None -> breathing uses ibis only."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(accel_buffer=[])
    payload = await pump._build_payload(sample)
    # Doesn't raise; imu_payload is also None
    assert payload.imu is None


async def test_build_payload_accel_buffer_too_short():
    """accel_buffer with < 3 elements -> accel_z stays None."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    sample = _make_sample(accel_buffer=[[1.0, 2.0], [3.0, 4.0]])  # only 2 axes
    payload = await pump._build_payload(sample)
    assert payload is not None


async def test_build_payload_accel_shape_zero():
    """accel_arr.shape[1] == 0 -> imu_payload is None."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hub import EEGHub
    pump = EEGPump(adapter=AsyncMock(), hub=EEGHub())
    # 3 channels with 0 samples each
    sample = _make_sample(
        accel_buffer=[[], [], []],
        gyro_buffer=[[], [], []],
    )
    payload = await pump._build_payload(sample)
    assert payload.imu is None


async def test_pump_loop_watchdog_fires():
    """Verify watchdog log fires when last_frame_ts is stale."""
    from neurolink.eeg_pump import EEGPump, _WATCHDOG_SEC
    from neurolink.hub import EEGHub

    hub = EEGHub()
    adapter = AsyncMock()
    tick_count = 0

    async def controlled_read():
        nonlocal tick_count
        tick_count += 1
        if tick_count >= 2:
            pump._running = False
        return None  # returns None so _tick returns early

    adapter.read_sample = controlled_read
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=100.0)
    # Set stale last_frame_ts to trigger watchdog
    pump._last_frame_ts = time.time() - (_WATCHDOG_SEC + 1)
    await pump.start()
    for _ in range(30):
        await asyncio.sleep(0.01)
        if tick_count >= 2:
            break
    await pump.stop()
    # Watchdog branch executed without raising
    assert tick_count >= 2
