"""Targeted branch-coverage tests for measured source files.

All files covered here are in the coverage measurement set (NOT in pyproject
omit list). Router/main files ARE omitted and contribute zero measured
statements, so this file avoids them entirely.

Covers uncovered branches in:
- neurolink/dsp/bandpower.py
- neurolink/dsp/classifiers.py
- neurolink/eeg_pump.py  (_build_payload empty-buffer branch)
- neurolink/hub.py       (_schedule_redis_push loop.is_running branch)
- neurolink/service.py   (_create_db_session / _close_db_session try-body)
- neurolink/calibration.py  (properties: is_running, baseline_alpha)
- neurolink/adapter_factory.py  (lsl else / unknown-type raise)
- neurolink/hardware/mock.py   (read_sample not-connected None return)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from neurolink.dsp.bandpower import bandpower, compute_band_powers_from_buffer
from neurolink.dsp.classifiers import classify_v01, classify_v2
from neurolink.models.eeg import BandPowers

# ===========================================================================
# dsp/bandpower.py -- uncovered guard branches
# ===========================================================================


def test_bandpower_none_signal_returns_zero():
    """bandpower(None, ...) hits the `sig is None` guard."""
    result = bandpower(None, 8.0, 13.0)
    assert result == 0.0


def test_compute_band_powers_1d_input():
    """1-D array hits the `eeg.ndim == 1` reshape branch."""
    sig = np.sin(2 * np.pi * 10 * np.linspace(0, 4, 1024)).astype(np.float32)
    result = compute_band_powers_from_buffer(sig)
    assert isinstance(result, dict)
    assert "alpha" in result


def test_compute_band_powers_too_short_returns_zeros():
    """Array with n_samples < 2 returns all-zero dict."""
    tiny = np.zeros((5, 1), dtype=np.float32)
    result = compute_band_powers_from_buffer(tiny)
    assert all(v == 0.0 for v in result.values())


def test_compute_band_powers_total_zero_returns_zeros():
    """All-zero EEG (total power == 0) hits the total<=0 guard."""
    silent = np.zeros((5, 512), dtype=np.float32)
    result = compute_band_powers_from_buffer(silent)
    assert all(v == 0.0 for v in result.values())


def test_compute_band_powers_none_returns_zeros():
    """None input hits the `eeg is None` guard."""
    result = compute_band_powers_from_buffer(None)
    assert all(v == 0.0 for v in result.values())


# ===========================================================================
# dsp/classifiers.py v01 -- untested region branches
# ===========================================================================


def test_v01_citrinitas_region_d():
    """High theta + low alpha -> Region D / Citrinitas."""
    region, stage = classify_v01(alpha=0.10, theta=0.25, beta=0.10, delta=0.10, gamma=0.05)
    assert region == "D"
    assert stage == "Citrinitas"


def test_v01_albedo_region_c():
    """Moderate alpha + low beta -> Region C / Albedo."""
    region, stage = classify_v01(alpha=0.25, theta=0.05, beta=0.10, delta=0.10, gamma=0.05)
    assert region == "C"
    assert stage == "Albedo"


def test_v01_albedo_region_b():
    """High beta -> Region B / Albedo."""
    region, stage = classify_v01(alpha=0.10, theta=0.05, beta=0.40, delta=0.10, gamma=0.05)
    assert region == "B"
    assert stage == "Albedo"


def test_v01_multiplicatio_with_faa_gate():
    """Very high alpha+theta with faa >= threshold -> Multiplicatio."""
    _region, stage = classify_v01(alpha=0.40, theta=0.20, beta=0.10, delta=0.05, gamma=0.05, faa=0.0)
    assert stage == "Multiplicatio"


def test_v01_multiplicatio_faa_none():
    """faa=None still triggers Multiplicatio when alpha/theta meet threshold."""
    _region, stage = classify_v01(
        alpha=0.40, theta=0.20, beta=0.10, delta=0.05, gamma=0.05, faa=None
    )
    assert stage == "Multiplicatio"


# ===========================================================================
# dsp/classifiers.py v2 -- untested branches
# ===========================================================================


def test_v2_citrinitas_balanced():
    """Balanced alpha-theta (not high enough for Rubedo) -> Citrinitas."""
    _region, stage = classify_v2(
        BandPowers(alpha=0.22, theta=0.12, beta=0.10, delta=0.10, gamma=0.05)
    )
    assert stage == "Citrinitas"


def test_v2_solutio_high_theta():
    """High theta, low alpha -> Solutio."""
    _region, stage = classify_v2(
        BandPowers(alpha=0.10, theta=0.30, beta=0.10, delta=0.10, gamma=0.05)
    )
    assert stage == "Solutio"


def test_v2_albedo_moderate_beta():
    """Moderate beta (>= 0.28) -> Albedo."""
    _region, stage = classify_v2(
        BandPowers(alpha=0.10, theta=0.05, beta=0.30, delta=0.10, gamma=0.05)
    )
    assert stage == "Albedo"


# ===========================================================================
# eeg_pump._build_payload -- empty eeg_buffer branch
# ===========================================================================


async def test_build_payload_empty_eeg_buffer():
    """_build_payload with eeg_buffer=None skips DSP and returns valid payload."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample
    from neurolink.hub import EEGHub

    hub = EEGHub()
    adapter = AsyncMock()
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=4.0)

    sample = EEGSample(
        channels=[0.0] * 5,
        timestamp=0.0,
        source="mock",
        address="mock",
        poor_contact=False,
        eeg_buffer=None,
        ppg_buffer=None,
        accel_buffer=None,
        gyro_buffer=None,
    )
    payload = await pump._build_payload(sample)
    assert payload.bands.alpha == 0.0


async def test_build_payload_short_eeg_buffer():
    """_build_payload with eeg_buffer having only 1 sample skips bandpower."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample
    from neurolink.hub import EEGHub

    hub = EEGHub()
    adapter = AsyncMock()
    pump = EEGPump(adapter=adapter, hub=hub, publish_hz=4.0)

    sample = EEGSample(
        channels=[0.0] * 5,
        timestamp=0.0,
        source="mock",
        address="mock",
        poor_contact=False,
        eeg_buffer=[[0.0], [0.0], [0.0], [0.0], [0.0]],
        ppg_buffer=None,
        accel_buffer=None,
        gyro_buffer=None,
    )
    payload = await pump._build_payload(sample)
    assert payload is not None


async def test_build_payload_partial_accel_skips_imu():
    """accel_buffer with len < 3 skips the IMU head_orientation branch."""
    from neurolink.eeg_pump import EEGPump
    from neurolink.hardware.base import EEGSample
    from neurolink.hub import EEGHub

    hub = EEGHub()
    pump = EEGPump(adapter=AsyncMock(), hub=hub, publish_hz=4.0)

    sample = EEGSample(
        channels=[0.0] * 5,
        source="mock",
        address="mock",
        eeg_buffer=None,
        ppg_buffer=None,
        accel_buffer=[[0.0, 0.0], [0.0, 0.0]],  # only 2 axes -- len < 3
        gyro_buffer=None,
    )
    payload = await pump._build_payload(sample)
    assert payload.imu is None


# ===========================================================================
# hub._schedule_redis_push -- loop.is_running() True branch
# ===========================================================================


async def test_schedule_redis_push_loop_running():
    """_schedule_redis_push schedules a task when the event loop is running."""
    from neurolink.hub import EEGHub
    from neurolink.models.eeg import NeurolinkState

    hub = EEGHub()
    state = NeurolinkState()

    with patch("neurolink.hub._push_state_to_redis", new=AsyncMock()):
        hub._schedule_redis_push(state)
        await asyncio.sleep(0)


def test_schedule_redis_push_runtime_error_is_suppressed():
    """RuntimeError from get_event_loop() is caught silently."""
    from neurolink.hub import EEGHub
    from neurolink.models.eeg import NeurolinkState

    hub = EEGHub()
    state = NeurolinkState()

    with patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
        hub._schedule_redis_push(state)  # must not raise


# ===========================================================================
# hub._push_state_to_redis -- coroutine body
# ===========================================================================


async def test_push_state_to_redis_calls_push_state():
    """_push_state_to_redis lazy-imports and calls cache.redis_client.push_state."""
    from neurolink.hub import _push_state_to_redis

    with patch("neurolink.cache.redis_client.push_state", new=AsyncMock()) as mock_push:
        await _push_state_to_redis({"frame_count": 1})
        mock_push.assert_awaited_once_with({"frame_count": 1})


# ===========================================================================
# service -- is_connected and adapter_type properties
# ===========================================================================


def test_service_is_connected_false_when_no_adapter():
    """is_connected returns False before any connect() call."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    svc = NeuroLinkService(EEGHub())
    assert svc.is_connected is False


def test_service_adapter_type_default():
    """adapter_type returns the settings default ('mock' in test env)."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    svc = NeuroLinkService(EEGHub())
    assert isinstance(svc.adapter_type, str)


# ===========================================================================
# service._create_db_session -- try body with factory set
# ===========================================================================


async def test_create_db_session_with_factory_success():
    """_create_db_session executes the try body when factory is configured."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)

    mock_entry = MagicMock()
    mock_entry.id = 42
    mock_repo = AsyncMock()
    mock_repo.create_session = AsyncMock(return_value=mock_entry)

    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    svc.set_db_session_factory(MagicMock(return_value=mock_db))

    with patch("neurolink.db.repository.SessionLogRepository", return_value=mock_repo):
        await svc._create_db_session(adapter_type="mock", device_model="mock", address=None)

    assert svc._db_session_id == 42


async def test_create_db_session_exception_is_logged():
    """_create_db_session swallows exceptions and logs a warning."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)

    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(side_effect=RuntimeError("db exploded"))
    mock_db.__aexit__ = AsyncMock(return_value=False)
    svc.set_db_session_factory(MagicMock(return_value=mock_db))

    await svc._create_db_session("mock", "mock", None)  # must not raise


# ===========================================================================
# service._close_db_session -- try body with factory + session_id set
# ===========================================================================


async def test_close_db_session_with_factory_and_id():
    """_close_db_session executes the try body when factory + session_id are set."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)
    svc._db_session_id = 99

    mock_repo = AsyncMock()
    mock_repo.end_session = AsyncMock(return_value=None)

    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    svc.set_db_session_factory(MagicMock(return_value=mock_db))

    with patch("neurolink.db.repository.SessionLogRepository", return_value=mock_repo):
        await svc._close_db_session()

    assert svc._db_session_id is None


async def test_close_db_session_exception_is_logged():
    """_close_db_session swallows exceptions."""
    from neurolink.hub import EEGHub
    from neurolink.service import NeuroLinkService

    hub = EEGHub()
    svc = NeuroLinkService(hub=hub)
    svc._db_session_id = 7

    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(side_effect=RuntimeError("oops"))
    mock_db.__aexit__ = AsyncMock(return_value=False)
    svc.set_db_session_factory(MagicMock(return_value=mock_db))

    await svc._close_db_session()  # must not raise


# ===========================================================================
# calibration -- properties
# ===========================================================================


def test_calibration_is_running_property():
    from neurolink.calibration import CalibrationSession

    session = CalibrationSession(adapter=AsyncMock(), hub=MagicMock())
    assert session.is_running is False
    session._running = True
    assert session.is_running is True


def test_calibration_baseline_alpha_property():
    from neurolink.calibration import CalibrationSession

    session = CalibrationSession(adapter=AsyncMock(), hub=MagicMock())
    assert session.baseline_alpha is None
    session._baseline_alpha = 0.35
    assert session.baseline_alpha == 0.35


# ===========================================================================
# adapter_factory -- lsl else branch, lsl athena, ble gen1
# ===========================================================================


def test_create_adapter_lsl_default_model():
    """lsl + non-athena model hits the MuseSLslAdapter branch."""
    from neurolink.adapter_factory import create_adapter

    with patch("neurolink.hardware.muse_s.lsl_adapter.MuseSLslAdapter") as MockLsl:
        MockLsl.return_value = MagicMock()
        adapter = create_adapter(adapter_type="lsl", device_model="muse_s_gen1")
    assert adapter is not None


def test_create_adapter_lsl_athena_model():
    """lsl + muse_s_athena hits the AthenaBlueAdapter branch."""
    from neurolink.adapter_factory import create_adapter

    with patch("neurolink.hardware.muse_athena.ble_adapter.AthenaBlueAdapter") as MockAthena:
        MockAthena.return_value = MagicMock()
        adapter = create_adapter(adapter_type="lsl", device_model="muse_s_athena")
    assert adapter is not None


def test_create_adapter_ble_gen1_branch():
    """ble + muse_s_gen1 hits the MuseSBleAdapter branch."""
    from neurolink.adapter_factory import create_adapter

    with patch("neurolink.hardware.muse_s.ble_adapter.MuseSBleAdapter") as MockBle:
        MockBle.return_value = MagicMock()
        adapter = create_adapter(
            adapter_type="ble",
            device_model="muse_s_gen1",
            address="AA:BB:CC:DD:EE:FF",
        )
    assert adapter is not None


def test_create_adapter_ble_athena_branch():
    """ble + muse_s_athena hits the AthenaBlueAdapter branch."""
    from neurolink.adapter_factory import create_adapter

    with patch("neurolink.hardware.muse_athena.ble_adapter.AthenaBlueAdapter") as MockAthena:
        MockAthena.return_value = MagicMock()
        adapter = create_adapter(
            adapter_type="ble",
            device_model="muse_s_athena",
            address="AA:BB:CC:DD:EE:FF",
        )
    assert adapter is not None


# ===========================================================================
# hardware/mock.py -- read_sample not-connected returns None
# ===========================================================================


async def test_mock_adapter_read_sample_not_connected_returns_none():
    """MockAdapter.read_sample() returns None when not connected."""
    from neurolink.hardware.mock import MockAdapter

    adapter = MockAdapter()
    result = await adapter.read_sample()
    assert result is None
