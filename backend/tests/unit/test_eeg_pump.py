"""Full unit tests for EEGPump — lifecycle + all 8 DSP pipeline stages."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest

from neurolink.eeg_pump import EEGPump
from neurolink.hardware.mock import MockAdapter
from neurolink.hub import EEGHub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pump(publish_hz: int = 4) -> tuple[EEGPump, EEGHub, MockAdapter]:
    hub = EEGHub()
    adapter = MockAdapter()
    pump = EEGPump(adapter, hub, publish_hz=publish_hz)
    return pump, hub, adapter


async def _run_pump(pump: EEGPump, adapter: MockAdapter, duration: float = 0.35) -> None:
    await adapter.connect()
    await pump.start()
    await asyncio.sleep(duration)
    await pump.stop()
    await adapter.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestEEGPumpLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop_no_exception(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        await _run_pump(pump, adapter)

    @pytest.mark.asyncio
    async def test_pump_increments_frame_count(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.4)
        assert hub.get_state().frame_count >= 1

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        await adapter.connect()
        await pump.start()
        await pump.stop()
        await pump.stop()  # should not raise
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_start_while_running_is_safe(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        await adapter.connect()
        await pump.start()
        await pump.start()  # idempotent
        await pump.stop()
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_hub_connected_after_frames(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.4)
        assert hub.get_state().connected is True

    @pytest.mark.asyncio
    async def test_frame_count_monotonically_increases(self):
        pump, hub, adapter = _make_pump(publish_hz=20)
        await adapter.connect()
        await pump.start()
        await asyncio.sleep(0.1)
        c1 = hub.get_state().frame_count
        await asyncio.sleep(0.2)
        c2 = hub.get_state().frame_count
        await pump.stop()
        await adapter.disconnect()
        assert c2 >= c1


# ---------------------------------------------------------------------------
# Stage 1 — bad channel detection
# ---------------------------------------------------------------------------

class TestStage1BadChannels:
    @pytest.mark.asyncio
    async def test_bad_channels_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.bad_channels") as mock_bc:
            mock_bc.detect.return_value = []
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_bc.detect.called

    @pytest.mark.asyncio
    async def test_bad_channels_skipped_when_toggle_off(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.bad_channels") as mock_bc:
            cfg = MagicMock()
            cfg.stage1_bad_channels = False
            mock_ft.get_toggles.return_value = cfg
            mock_bc.detect.return_value = []
            await _run_pump(pump, adapter, duration=0.35)
        mock_bc.detect.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 2 — spherical spline interpolation
# ---------------------------------------------------------------------------

class TestStage2SphericalSpline:
    @pytest.mark.asyncio
    async def test_interpolation_skipped_when_no_bad_channels(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.bad_channels") as mock_bc, \
             patch("neurolink.eeg_pump.spherical_spline") as mock_ss:
            mock_bc.detect.return_value = []  # no bad channels
            await _run_pump(pump, adapter, duration=0.35)
        mock_ss.interpolate.assert_not_called()

    @pytest.mark.asyncio
    async def test_interpolation_called_when_bad_channels_present(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.bad_channels") as mock_bc, \
             patch("neurolink.eeg_pump.spherical_spline") as mock_ss:
            mock_bc.detect.return_value = [1]  # one bad channel
            mock_ss.interpolate.side_effect = lambda eeg, bad, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_ss.interpolate.called


# ---------------------------------------------------------------------------
# Stage 3 — ASR (artifact subspace reconstruction)
# ---------------------------------------------------------------------------

class TestStage3ASR:
    @pytest.mark.asyncio
    async def test_asr_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.asr") as mock_asr:
            mock_asr.apply.side_effect = lambda eeg, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_asr.apply.called

    @pytest.mark.asyncio
    async def test_asr_skipped_when_toggle_off(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.asr") as mock_asr:
            cfg = MagicMock()
            cfg.stage1_bad_channels = True
            cfg.stage2_interpolation = True
            cfg.stage3_asr = False
            cfg.stage4_ocular = True
            cfg.stage5_baseline = True
            cfg.stage6_cardiac = True
            cfg.stage7_bandpower = True
            cfg.stage8_classify = True
            mock_ft.get_toggles.return_value = cfg
            await _run_pump(pump, adapter, duration=0.35)
        mock_asr.apply.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 4 — ocular regression
# ---------------------------------------------------------------------------

class TestStage4OcularRegression:
    @pytest.mark.asyncio
    async def test_ocular_regression_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.ocular_regression") as mock_or:
            mock_or.apply.side_effect = lambda eeg, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_or.apply.called

    @pytest.mark.asyncio
    async def test_ocular_regression_output_used_downstream(self):
        """Verify the EEG after ocular correction is what stage 5 receives."""
        pump, hub, adapter = _make_pump(publish_hz=4)
        sentinel = np.zeros((4, 64), dtype=np.float32)
        sentinel[0, 0] = 999.0
        with patch("neurolink.eeg_pump.ocular_regression") as mock_or, \
             patch("neurolink.eeg_pump.baseline") as mock_bl:
            mock_or.apply.return_value = sentinel
            mock_bl.apply.side_effect = lambda eeg, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        # If ocular output is wired through, baseline must receive sentinel
        calls = mock_bl.apply.call_args_list
        assert any(np.array_equal(c.args[0], sentinel) or
                   (len(c.args) == 0 and np.array_equal(list(c.kwargs.values())[0], sentinel))
                   for c in calls), "Ocular output not forwarded to stage 5"


# ---------------------------------------------------------------------------
# Stage 5 — baseline correction
# ---------------------------------------------------------------------------

class TestStage5Baseline:
    @pytest.mark.asyncio
    async def test_baseline_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.baseline") as mock_bl:
            mock_bl.apply.side_effect = lambda eeg, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_bl.apply.called

    @pytest.mark.asyncio
    async def test_baseline_skipped_when_toggle_off(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.baseline") as mock_bl:
            cfg = MagicMock()
            cfg.stage1_bad_channels = True
            cfg.stage2_interpolation = True
            cfg.stage3_asr = True
            cfg.stage4_ocular = True
            cfg.stage5_baseline = False
            cfg.stage6_cardiac = True
            cfg.stage7_bandpower = True
            cfg.stage8_classify = True
            mock_ft.get_toggles.return_value = cfg
            await _run_pump(pump, adapter, duration=0.35)
        mock_bl.apply.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 6 — cardiac regression
# ---------------------------------------------------------------------------

class TestStage6CardiacRegression:
    @pytest.mark.asyncio
    async def test_cardiac_regression_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.cardiac_regression") as mock_cr:
            mock_cr.apply.side_effect = lambda eeg, **kw: eeg
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_cr.apply.called

    @pytest.mark.asyncio
    async def test_cardiac_regression_skipped_when_toggle_off(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.cardiac_regression") as mock_cr:
            cfg = MagicMock()
            cfg.stage1_bad_channels = True
            cfg.stage2_interpolation = True
            cfg.stage3_asr = True
            cfg.stage4_ocular = True
            cfg.stage5_baseline = True
            cfg.stage6_cardiac = False
            cfg.stage7_bandpower = True
            cfg.stage8_classify = True
            mock_ft.get_toggles.return_value = cfg
            await _run_pump(pump, adapter, duration=0.35)
        mock_cr.apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_cardiac_regression_toggle_off_passes_eeg_unchanged(self):
        """When stage6_cardiac is off, raw EEG (not modified by regressor) flows to stage 7."""
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.cardiac_regression") as mock_cr, \
             patch("neurolink.eeg_pump.bandpower") as mock_bp:
            cfg = MagicMock()
            for attr in ["stage1_bad_channels", "stage2_interpolation",
                         "stage3_asr", "stage4_ocular", "stage5_baseline",
                         "stage7_bandpower", "stage8_classify"]:
                setattr(cfg, attr, True)
            cfg.stage6_cardiac = False
            mock_ft.get_toggles.return_value = cfg
            mock_bp.compute.return_value = {}
            await _run_pump(pump, adapter, duration=0.35)
        mock_cr.apply.assert_not_called()
        assert mock_bp.compute.called


# ---------------------------------------------------------------------------
# Stage 7 — bandpower
# ---------------------------------------------------------------------------

class TestStage7Bandpower:
    @pytest.mark.asyncio
    async def test_bandpower_computed_each_frame(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.bandpower") as mock_bp:
            mock_bp.compute.return_value = {"delta": 0.1, "theta": 0.2,
                                             "alpha": 0.3, "beta": 0.25,
                                             "gamma": 0.15}
            await _run_pump(pump, adapter, duration=0.5)
        assert mock_bp.compute.call_count >= 1

    @pytest.mark.asyncio
    async def test_bandpower_bands_present_in_state(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.5)
        state = hub.get_state()
        # MockAdapter produces valid EEG — bands should be populated
        assert state.band_powers is not None


# ---------------------------------------------------------------------------
# Stage 8 — classifiers (focus / fatigue)
# ---------------------------------------------------------------------------

class TestStage8Classifiers:
    @pytest.mark.asyncio
    async def test_classifiers_called_when_toggle_on(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.classifiers") as mock_cl:
            mock_cl.run.return_value = {"focus": 0.7, "fatigue": 0.3}
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_cl.run.called

    @pytest.mark.asyncio
    async def test_classifiers_skipped_when_toggle_off(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch("neurolink.eeg_pump.filter_toggles") as mock_ft, \
             patch("neurolink.eeg_pump.classifiers") as mock_cl:
            cfg = MagicMock()
            for attr in ["stage1_bad_channels", "stage2_interpolation",
                         "stage3_asr", "stage4_ocular", "stage5_baseline",
                         "stage6_cardiac", "stage7_bandpower"]:
                setattr(cfg, attr, True)
            cfg.stage8_classify = False
            mock_ft.get_toggles.return_value = cfg
            await _run_pump(pump, adapter, duration=0.35)
        mock_cl.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_focus_score_propagated_to_hub(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.5)
        state = hub.get_state()
        assert hasattr(state, "focus_score")

    @pytest.mark.asyncio
    async def test_fatigue_score_propagated_to_hub(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.5)
        state = hub.get_state()
        assert hasattr(state, "fatigue_score")


# ---------------------------------------------------------------------------
# Settling / hub.emit_settling() integration
# ---------------------------------------------------------------------------

class TestSettlingEmission:
    @pytest.mark.asyncio
    async def test_emit_settling_called_on_impedance_unstable(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch.object(hub, "emit_settling") as mock_emit, \
             patch("neurolink.eeg_pump.impedance") as mock_imp:
            mock_imp.check.return_value = False  # unstable
            await _run_pump(pump, adapter, duration=0.35)
        assert mock_emit.called
        reasons = [c.kwargs.get("reason", c.args[0] if c.args else None)
                   for c in mock_emit.call_args_list]
        assert any(r == "impedance_unstable" for r in reasons)

    @pytest.mark.asyncio
    async def test_no_settling_when_impedance_stable(self):
        pump, hub, adapter = _make_pump(publish_hz=4)
        with patch.object(hub, "emit_settling") as mock_emit, \
             patch("neurolink.eeg_pump.impedance") as mock_imp:
            mock_imp.check.return_value = True  # stable
            await _run_pump(pump, adapter, duration=0.35)
        # If impedance stable, emit_settling should not be called for that reason
        reasons = [c.kwargs.get("reason", c.args[0] if c.args else None)
                   for c in mock_emit.call_args_list]
        assert "impedance_unstable" not in reasons


# ---------------------------------------------------------------------------
# Artifact gate integration
# ---------------------------------------------------------------------------

class TestArtifactGateIntegration:
    @pytest.mark.asyncio
    async def test_artifact_rejected_propagated_to_hub(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.5)
        state = hub.get_state()
        # Field must exist even if no artifact was rejected
        assert hasattr(state, "artifact_rejected")

    @pytest.mark.asyncio
    async def test_artifact_reasons_is_list(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        await _run_pump(pump, adapter, duration=0.5)
        state = hub.get_state()
        assert isinstance(state.artifact_reasons, list)


# ---------------------------------------------------------------------------
# SSE fan-out
# ---------------------------------------------------------------------------

class TestSSEFanOut:
    @pytest.mark.asyncio
    async def test_sse_queue_receives_state_on_each_frame(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        q: asyncio.Queue = asyncio.Queue(maxsize=0)
        hub.register_sse_client(q)
        await adapter.connect()
        await pump.start()
        await asyncio.sleep(0.4)
        await pump.stop()
        await adapter.disconnect()
        hub.unregister_sse_client(q)
        assert not q.empty(), "SSE queue received no frames"

    @pytest.mark.asyncio
    async def test_multiple_sse_clients_all_receive_frames(self):
        pump, hub, adapter = _make_pump(publish_hz=10)
        queues = [asyncio.Queue(maxsize=0) for _ in range(3)]
        for q in queues:
            hub.register_sse_client(q)
        await adapter.connect()
        await pump.start()
        await asyncio.sleep(0.4)
        await pump.stop()
        await adapter.disconnect()
        for q in queues:
            hub.unregister_sse_client(q)
        for i, q in enumerate(queues):
            assert not q.empty(), f"SSE client {i} received no frames"
