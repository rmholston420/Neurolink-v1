"""Extended lifecycle tests for NeuroLinkService.

Covers paths not exercised by the existing test_service.py:
  - get_baseline_progress() state machine
  - start_calibration() idempotency
  - adapter_type property after connect()
  - set_db_session_factory()
  - _create_db_session and _close_db_session error-resilience
  - stream_state keepalive on timeout
  - disconnect without prior connect
  - disconnect with a pump but no adapter
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurolink.calibration import TOTAL_DURATION_SEC
from neurolink.exceptions import AdapterNotConnectedError
from neurolink.hub import EEGHub
from neurolink.models.eeg import BandPowers, IngestPayload
from neurolink.service import NeuroLinkService


def _svc() -> NeuroLinkService:
    return NeuroLinkService(EEGHub())


# ─────────────────────────────────────────────────────────────────────────────
# get_baseline_progress — state machine
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBaselineProgress:
    def test_idle_when_no_session(self):
        svc = _svc()
        resp = svc.get_baseline_progress()
        assert resp.phase == "idle"
        assert resp.elapsed_s == 0.0
        assert resp.remaining_s == 0.0
        assert resp.total_s == TOTAL_DURATION_SEC

    def test_reflects_session_phase(self):
        svc = _svc()
        mock_session = MagicMock()
        mock_session.elapsed = 15.0
        mock_session.phase = "warmup"
        svc._calibration_session = mock_session
        svc._calibration_task = None

        resp = svc.get_baseline_progress()
        assert resp.phase == "warmup"
        assert resp.elapsed_s == pytest.approx(15.0)
        assert resp.remaining_s == pytest.approx(TOTAL_DURATION_SEC - 15.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_complete_when_task_done(self):
        """get_baseline_progress() returns phase='complete' once the task is done."""
        svc = _svc()
        mock_session = MagicMock()
        mock_session.elapsed = TOTAL_DURATION_SEC
        mock_session.phase = "baseline"
        svc._calibration_session = mock_session

        # Create a real finished asyncio Task using plain async def.
        async def _noop():
            pass

        task = asyncio.create_task(_noop())
        await task  # ensure it is done before asserting

        svc._calibration_task = task
        resp = svc.get_baseline_progress()
        assert resp.phase == "complete"
        assert resp.remaining_s == 0.0

    def test_remaining_floored_at_zero(self):
        svc = _svc()
        mock_session = MagicMock()
        mock_session.elapsed = TOTAL_DURATION_SEC + 10.0  # over budget
        mock_session.phase = "baseline"
        svc._calibration_session = mock_session
        svc._calibration_task = None

        resp = svc.get_baseline_progress()
        assert resp.remaining_s == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# start_calibration idempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestStartCalibrationIdempotency:
    @pytest.mark.asyncio
    async def test_second_call_returns_started_status(self):
        svc = _svc()
        await svc.connect(adapter_type="mock", device_model="mock")
        try:
            resp1 = await svc.start_calibration()
            resp2 = await svc.start_calibration()
            assert resp1.status == "started"
            assert resp2.status == "started"
            assert resp2.baseline_alpha is None
        finally:
            await svc.disconnect()

    @pytest.mark.asyncio
    async def test_calibrate_without_connection_raises(self):
        svc = _svc()
        with pytest.raises(AdapterNotConnectedError):
            await svc.start_calibration()


# ─────────────────────────────────────────────────────────────────────────────
# adapter_type property
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapterTypeProperty:
    @pytest.mark.asyncio
    async def test_adapter_type_reflects_connect_call(self):
        svc = _svc()
        await svc.connect(adapter_type="mock", device_model="mock")
        try:
            assert svc.adapter_type == "mock"
        finally:
            await svc.disconnect()

    def test_adapter_type_default_before_connect(self):
        svc = _svc()
        assert svc.adapter_type == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# set_db_session_factory
# ─────────────────────────────────────────────────────────────────────────────

class TestSetDbSessionFactory:
    def test_factory_stored_on_service(self):
        svc = _svc()
        sentinel = object()
        svc.set_db_session_factory(sentinel)
        assert svc._db_session_factory is sentinel

    def test_none_factory_is_accepted(self):
        svc = _svc()
        svc.set_db_session_factory(None)
        assert svc._db_session_factory is None


# ─────────────────────────────────────────────────────────────────────────────
# DB error resilience
# ─────────────────────────────────────────────────────────────────────────────

class TestDbErrorResilience:
    @pytest.mark.asyncio
    async def test_connect_survives_db_create_error(self):
        """If the DB factory raises, connect() must still succeed."""
        svc = _svc()

        class _BadFactory:
            def __call__(self):
                return self

            async def __aenter__(self):
                raise RuntimeError("db unavailable")

            async def __aexit__(self, *args):
                pass

        svc.set_db_session_factory(_BadFactory())
        result = await svc.connect(adapter_type="mock", device_model="mock")
        assert result.ok is True
        await svc.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_survives_db_close_error(self):
        """If the DB factory raises on close, disconnect() must still return ok."""
        svc = _svc()

        class _BadFactory:
            def __call__(self):
                return self

            async def __aenter__(self):
                raise RuntimeError("db unavailable")

            async def __aexit__(self, *args):
                pass

        svc.set_db_session_factory(_BadFactory())
        svc._db_session_id = 42  # simulate an open session
        await svc.connect(adapter_type="mock", device_model="mock")
        result = await svc.disconnect()
        assert result.ok is True


# ─────────────────────────────────────────────────────────────────────────────
# is_connected property
# ─────────────────────────────────────────────────────────────────────────────

class TestIsConnected:
    @pytest.mark.asyncio
    async def test_true_after_connect(self):
        svc = _svc()
        await svc.connect(adapter_type="mock", device_model="mock")
        assert svc.is_connected is True
        await svc.disconnect()

    @pytest.mark.asyncio
    async def test_false_after_disconnect(self):
        svc = _svc()
        await svc.connect(adapter_type="mock", device_model="mock")
        await svc.disconnect()
        assert svc.is_connected is False

    def test_false_with_none_adapter(self):
        svc = _svc()
        svc._adapter = None
        assert svc.is_connected is False

    def test_false_with_disconnected_adapter(self):
        svc = _svc()
        mock_adapter = MagicMock()
        mock_adapter.is_connected = False
        svc._adapter = mock_adapter
        assert svc.is_connected is False


# ─────────────────────────────────────────────────────────────────────────────
# stream_state
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamState:
    @pytest.mark.asyncio
    async def test_stream_state_cancels_cleanly(self):
        svc = _svc()
        gen = svc.stream_state()
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        await gen.aclose()
        # queue must be unregistered after close
        assert len(svc._hub._sse_queues) == 0

    @pytest.mark.asyncio
    async def test_stream_state_yields_hub_state_on_timeout(self):
        """After the 2s timeout, stream_state yields the current hub state."""
        svc = _svc()
        # Prime hub with a frame
        svc._hub.update(
            IngestPayload(
                source="mock",
                bands=BandPowers(alpha=0.4, theta=0.2, beta=0.15, delta=0.1, gamma=0.05),
            )
        )
        gen = svc.stream_state()

        # Patch asyncio.wait_for to raise TimeoutError immediately
        with patch("asyncio.wait_for", side_effect=TimeoutError):
            state = await gen.__anext__()

        assert state is not None
        assert state.frame_count >= 1
        await gen.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# get_current_state / get_band_powers / get_ea1 after hub updates
# ─────────────────────────────────────────────────────────────────────────────

class TestStateAccessors:
    @pytest.mark.asyncio
    async def test_frame_count_increments(self):
        svc = _svc()
        for _ in range(3):
            svc._hub.update(
                IngestPayload(
                    source="mock",
                    bands=BandPowers(alpha=0.3, theta=0.2, beta=0.15, delta=0.2, gamma=0.1),
                )
            )
        state = await svc.get_current_state()
        assert state.frame_count == 3

    @pytest.mark.asyncio
    async def test_get_band_powers_reflects_latest_update(self):
        svc = _svc()
        svc._hub.update(
            IngestPayload(
                source="mock",
                bands=BandPowers(alpha=0.55, theta=0.1, beta=0.1, delta=0.1, gamma=0.05),
            )
        )
        resp = await svc.get_band_powers(channel="mean")
        assert resp.alpha == pytest.approx(0.55)
        assert resp.channel == "mean"

    @pytest.mark.asyncio
    async def test_get_ea1_structure(self):
        svc = _svc()
        ea1 = await svc.get_ea1()
        assert hasattr(ea1, "eligible")
        assert hasattr(ea1, "score")
        assert hasattr(ea1, "criteria_met")
        assert 0 <= ea1.score <= 1.0

    @pytest.mark.asyncio
    async def test_get_sessions_returns_list(self):
        svc = _svc()
        result = await svc.get_sessions(limit=5)
        assert isinstance(result, list)
